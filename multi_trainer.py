"""
PyTorch's policy class used for PPO.
"""
import logging
from abc import ABC
from typing import Dict, List, Optional, Type, Union

import gym
import numpy as np
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.episode import MultiAgentEpisode
from ray.rllib.evaluation.postprocessing import Postprocessing, compute_advantages
from ray.rllib.models import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import explained_variance, sequence_mask
from ray.rllib.utils.typing import AgentID, TensorType

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)


class InvalidActionSpace(Exception):
    """Raised when the action space is invalid"""

    pass


def compute_gae_for_sample_batch(
    policy: Policy,
    sample_batch: SampleBatch,
    other_agent_batches: Optional[Dict[AgentID, SampleBatch]] = None,
    episode: Optional[MultiAgentEpisode] = None,
) -> SampleBatch:
    """Adds GAE (generalized advantage estimations) to a trajectory.
    The trajectory contains only data from one episode and from one agent.
    - If  `config.batch_mode=truncate_episodes` (default), sample_batch may
    contain a truncated (at-the-end) episode, in case the
    `config.rollout_fragment_length` was reached by the sampler.
    - If `config.batch_mode=complete_episodes`, sample_batch will contain
    exactly one episode (no matter how long).
    New columns can be added to sample_batch and existing ones may be altered.
    Args:
        policy (Policy): The Policy used to generate the trajectory
            (`sample_batch`)
        sample_batch (SampleBatch): The SampleBatch to postprocess.
        other_agent_batches (Optional[Dict[PolicyID, SampleBatch]]): Optional
            dict of AgentIDs mapping to other agents' trajectory data (from the
            same episode). NOTE: The other agents use the same policy.
        episode (Optional[MultiAgentEpisode]): Optional multi-agent episode
            object in which the agents operated.
    Returns:
        SampleBatch: The postprocessed, modified SampleBatch (or a new one).
    """

    # the trajectory view API will pass populate the info dict with a np.zeros((n,))
    # array in the first call, in that case the dtype will be float32 and we
    # have to ignore it. For regular calls, we extract the rewards from the info
    # dict into the samplebatch_infos_rewards dict, which now holds the rewards
    # for all agents as dict.
    samplebatch_infos_rewards = {"0": sample_batch[SampleBatch.INFOS]}
    if not sample_batch[SampleBatch.INFOS].dtype == "float32":
        samplebatch_infos = SampleBatch.concat_samples(
            [
                SampleBatch({k: [v] for k, v in s.items()})
                for s in sample_batch[SampleBatch.INFOS]
            ]
        )
        samplebatch_infos_rewards = SampleBatch.concat_samples(
            [
                SampleBatch({str(k): [v] for k, v in s.items()})
                for s in samplebatch_infos["rewards"]
            ]
        )

    if not isinstance(policy.action_space, gym.spaces.tuple.Tuple):
        raise InvalidActionSpace("Expect tuple action space")

    # samplebatches for each agent
    batches = []
    action_index = 0
    for key, action_space in zip(samplebatch_infos_rewards.keys(), policy.action_space):
        i = int(key)
        sample_batch_agent = sample_batch.copy()
        sample_batch_agent[SampleBatch.REWARDS] = samplebatch_infos_rewards[key]
        if isinstance(action_space, gym.spaces.box.Box):
            assert len(action_space.shape) == 1
            a_w = action_space.shape[0]
        elif isinstance(action_space, gym.spaces.discrete.Discrete):
            a_w = 1
        else:
            raise InvalidActionSpace(
                "Expect gym.spaces.box or gym.spaces.discrete action space"
            )

        sample_batch_agent[SampleBatch.ACTIONS] = sample_batch[SampleBatch.ACTIONS][
            :, action_index : (action_index + a_w)
        ]
        sample_batch_agent[SampleBatch.VF_PREDS] = sample_batch[SampleBatch.VF_PREDS][
            :, i
        ]
        action_index += a_w
        # Trajectory is actually complete -> last r=0.0.
        if sample_batch[SampleBatch.DONES][-1]:
            last_r = 0.0
        # Trajectory has been truncated -> last r=VF estimate of last obs.
        else:
            # Input dict is provided to us automatically via the Model's
            # requirements. It's a single-timestep (last one in trajectory)
            # input_dict.
            # Create an input dict according to the Model's requirements.
            input_dict = sample_batch.get_single_step_input_dict(
                policy.model.view_requirements, index="last"
            )
            all_values = policy._value(**input_dict)
            last_r = all_values[i].item()

        # Adds the policy logits, VF preds, and advantages to the batch,
        # using GAE ("generalized advantage estimation") or not.
        batches.append(
            compute_advantages(
                sample_batch_agent,
                last_r,
                policy.config["gamma"],
                policy.config["lambda"],
                use_gae=policy.config["use_gae"],
                use_critic=policy.config.get("use_critic", True),
            )
        )

    # Now take original samplebatch and overwrite following elements as a concatenation of these
    for k in [
        SampleBatch.REWARDS,
        SampleBatch.VF_PREDS,
        Postprocessing.ADVANTAGES,
        Postprocessing.VALUE_TARGETS,
    ]:
        sample_batch[k] = np.stack([b[k] for b in batches], axis=-1)

    return sample_batch


def ppo_surrogate_loss(
    policy: Policy,
    model: ModelV2,
    dist_class: Type[ActionDistribution],
    train_batch: SampleBatch,
) -> Union[TensorType, List[TensorType]]:
    """Constructs the loss for Proximal Policy Objective.
    Args:
        policy (Policy): The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        dist_class (Type[ActionDistribution]): The action distr. class.
        train_batch (SampleBatch): The training data.
    Returns:
        Union[TensorType, List[TensorType]]: A single loss tensor or a list
            of loss tensors.
    """
    logits, state = model(train_batch)
    curr_action_dist = dist_class(logits, model)

    # RNN case: Mask away 0-padded chunks at end of time axis.
    if state:
        B = len(train_batch[SampleBatch.SEQ_LENS])
        max_seq_len = logits.shape[0] // B
        mask = sequence_mask(
            train_batch[SampleBatch.SEQ_LENS],
            max_seq_len,
            time_major=model.is_time_major(),
        )
        mask = torch.reshape(mask, [-1])
        num_valid = torch.sum(mask)

        def reduce_mean_valid(t):
            return torch.sum(t[mask]) / num_valid

    # non-RNN case: No masking.
    else:
        mask = None
        reduce_mean_valid = torch.mean

    prev_action_dist = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS], model)
    logps = curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])

    curr_entropies = curr_action_dist.entropy()
    use_kl = policy.config["kl_coeff"] > 0.0

    if use_kl > 0.0:
        action_kl = prev_action_dist.kl(curr_action_dist)
    else:
        action_kl = torch.tensor(0.0, device=logps.device)

    loss_data = []
    for i in range(len(train_batch[SampleBatch.VF_PREDS][0])):
        logp_ratio = torch.exp(logps[:, i] - train_batch[SampleBatch.ACTION_LOGP][:, i])

        eps = policy.config["clip_param"]
        surrogate = torch.clamp(logp_ratio, 1 - eps, 1 + eps)
        surrogate_loss = torch.min(
            train_batch[Postprocessing.ADVANTAGES][..., i] * logp_ratio,
            train_batch[Postprocessing.ADVANTAGES][..., i] * surrogate,
        )

        # Compute a value function loss.
        if policy.config["use_critic"]:
            value_fn_out = model.value_function()[..., i]
            vf_loss = torch.pow(
                value_fn_out - train_batch[Postprocessing.VALUE_TARGETS][..., i], 2.0
            )
            vf_loss_clipped = torch.clamp(vf_loss, 0, policy.config["vf_clip_param"])
        # Ignore the value function.
        else:
            vf_loss_clipped = 0.0

        total_loss = (
            -surrogate_loss
            + policy.config["vf_loss_coeff"] * vf_loss_clipped
            - policy.entropy_coeff * curr_entropies[:, i]
        )

        # Add mean_kl_loss (already processed through `reduce_mean_valid`),
        # if necessary.
        if use_kl:
            total_loss += policy.kl_coeff * action_kl[:, i]

        total_loss = reduce_mean_valid(total_loss)
        mean_policy_loss = reduce_mean_valid(-surrogate_loss)
        mean_vf_loss = (
            reduce_mean_valid(vf_loss_clipped) if policy.config["use_critic"] else 0.0
        )
        mean_entropy = reduce_mean_valid(curr_entropies[:, i])
        mean_kl = reduce_mean_valid(action_kl[:, i]) if use_kl else torch.tensor([0.0])

        # Store stats in policy for stats_fn.
        loss_data.append(
            {
                "total_loss": total_loss,
                "mean_policy_loss": mean_policy_loss,
                "mean_vf_loss": mean_vf_loss,
                "mean_entropy": mean_entropy,
                "mean_kl": mean_kl,
            }
        )

    model.tower_stats["total_loss"] = torch.sum(
        torch.stack([o["total_loss"] for o in loss_data])
    )
    model.tower_stats["mean_policy_loss"] = torch.mean(
        torch.stack([o["mean_policy_loss"] for o in loss_data])
    )
    model.tower_stats["mean_vf_loss"] = torch.mean(
        torch.stack([o["mean_vf_loss"] for o in loss_data])
    )
    model.tower_stats["vf_explained_var"] = explained_variance(
        train_batch[Postprocessing.VALUE_TARGETS], policy.model.value_function()
    )
    model.tower_stats["mean_entropy"] = torch.mean(
        torch.stack([o["mean_entropy"] for o in loss_data])
    )
    model.tower_stats["mean_kl_loss"] = torch.mean(
        torch.stack([o["mean_kl"] for o in loss_data])
    )

    return torch.sum(torch.stack([o["total_loss"] for o in loss_data]))


class MultiPPOTorchPolicy(PPOTorchPolicy, ABC):
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)

    @override(PPOTorchPolicy)
    def loss(self, model, dist_class, train_batch):
        return ppo_surrogate_loss(self, model, dist_class, train_batch)

    @override(PPOTorchPolicy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        return compute_gae_for_sample_batch(
            self, sample_batch, other_agent_batches, episode
        )

    @override(PPOTorchPolicy)
    def _value(self, **input_dict):
        """This is exactly the as in PPOTorchPolicy,
        but that one calls .item() on self.model.value_function()[0],
        which will not work for us since our value function returns
        multiple values. Instead, we call .item() in
        compute_gae_for_sample_batch above.
        """

        # When doing GAE, we need the value function estimate on the
        # observation.
        if self.config["use_gae"]:
            # Input dict is provided to us automatically via the Model's
            # requirements. It's a single-timestep (last one in trajectory)
            # input_dict.
            input_dict = self._lazy_tensor_dict(input_dict)
            model_out, _ = self.model(input_dict)
            # [0] = remove the batch dim.
            return self.model.value_function()[0]
        # When not doing GAE, we do not require the value function's output.
        else:
            return 0.0


class MultiPPOTrainer(PPOTrainer, ABC):
    @override(PPOTrainer)
    def get_default_policy_class(self, config):
        return MultiPPOTorchPolicy
