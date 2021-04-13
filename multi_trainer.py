"""
PyTorch policy class used for PPO.
"""
import gym
import logging
import numpy as np
from typing import Dict, List, Optional, Type, Union

import ray
from ray.rllib.agents.ppo.ppo_tf_policy import setup_config
from ray.rllib.agents.ppo.ppo_torch_policy import kl_and_loss_stats, \
    vf_preds_fetches, setup_mixins, KLCoeffMixin, ValueNetworkMixin
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.evaluation.episode import MultiAgentEpisode
from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import EntropyCoeffSchedule, \
    LearningRateSchedule
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import apply_grad_clipping, \
    convert_to_torch_tensor, explained_variance, sequence_mask
from ray.rllib.utils.typing import TensorType, TrainerConfigDict, AgentID

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)

class InvalidActionSpace(Exception):
    """Raised when the action space is invalid"""
    pass


def compute_gae_for_sample_batch(
        policy: Policy,
        sample_batch: SampleBatch,
        other_agent_batches: Optional[Dict[AgentID, SampleBatch]] = None,
        episode: Optional[MultiAgentEpisode] = None) -> SampleBatch:
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
    samplebatch_infos_rewards = {'0': sample_batch[SampleBatch.INFOS]}
    if not sample_batch[SampleBatch.INFOS].dtype == "float32":
        samplebatch_infos = SampleBatch.concat_samples([
            SampleBatch({k: [v] for k, v in s.items()})
            for s in sample_batch[SampleBatch.INFOS]
        ])
        samplebatch_infos_rewards = SampleBatch.concat_samples([
            SampleBatch({str(k): [v] for k, v in s.items()})
            for s in samplebatch_infos["rewards"]
        ])

    if not isinstance(policy.action_space, gym.spaces.tuple.Tuple):
        raise InvalidActionSpace("Expect tuple action space")

    # samplebatches for each agents
    batches = []
    for key, action_space in zip(samplebatch_infos_rewards.keys(), policy.action_space):
        i = int(key)
        sample_batch_agent = sample_batch.copy()
        sample_batch_agent[SampleBatch.REWARDS] = (samplebatch_infos_rewards[key])
        if isinstance(action_space, gym.spaces.box.Box):
            assert len(action_space.shape) == 1
            a_w = action_space.shape[0]
        elif isinstance(action_space, gym.spaces.discrete.Discrete):
            a_w = 1
        else:
            raise InvalidActionSpace("Expect gym.spaces.box or gym.spaces.discrete action space")

        sample_batch_agent[SampleBatch.ACTIONS] = sample_batch[SampleBatch.ACTIONS][:, a_w * i : a_w * (i + 1)]
        sample_batch_agent[SampleBatch.VF_PREDS] = sample_batch[SampleBatch.VF_PREDS][:, i]

        # Trajectory is actually complete -> last r=0.0.
        if sample_batch[SampleBatch.DONES][-1]:
            last_r = 0.0
        # Trajectory has been truncated -> last r=VF estimate of last obs.
        else:
            # Input dict is provided to us automatically via the Model's
            # requirements. It's a single-timestep (last one in trajectory)
            # input_dict.
            # Create an input dict according to the Model's requirements.
            input_dict = policy.model.get_input_dict(
                sample_batch, index="last")
            all_values = policy._value(**input_dict, seq_lens=input_dict.seq_lens)
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
                use_critic=policy.config.get("use_critic", True)
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
        policy: Policy, model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
    """Constructs the loss for Proximal Policy Objective.
    Args:
        policy (Policy): The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        dist_class (Type[ActionDistribution]: The action distr. class.
        train_batch (SampleBatch): The training data.
    Returns:
        Union[TensorType, List[TensorType]]: A single loss tensor or a list
            of loss tensors.
    """
    logits, state = model.from_batch(train_batch, is_training=True)
    curr_action_dist = dist_class(logits, model)

    # RNN case: Mask away 0-padded chunks at end of time axis.
    if state:
        B = len(train_batch["seq_lens"])
        max_seq_len = logits.shape[0] // B
        mask = sequence_mask(
            train_batch["seq_lens"],
            max_seq_len,
            time_major=model.is_time_major())
        mask = torch.reshape(mask, [-1])
        num_valid = torch.sum(mask)

        def reduce_mean_valid(t):
            return torch.sum(t[mask]) / num_valid

    # non-RNN case: No masking.
    else:
        mask = None
        reduce_mean_valid = torch.mean

    loss_data = []

    curr_action_dist = dist_class(logits, model)
    prev_action_dist = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS],
                                  model)
    logps = curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
    entropies = curr_action_dist.entropy()

    action_kl = prev_action_dist.kl(curr_action_dist)
    mean_kl = reduce_mean_valid(torch.sum(action_kl, axis=1))

    for i in range(len(train_batch[SampleBatch.VF_PREDS][0])):
        logp_ratio = torch.exp(
            logps[:, i] -
            train_batch[SampleBatch.ACTION_LOGP][:, i])

        mean_entropy = reduce_mean_valid(entropies[:, i])

        surrogate_loss = torch.min(
            train_batch[Postprocessing.ADVANTAGES][..., i] * logp_ratio,
            train_batch[Postprocessing.ADVANTAGES][..., i] * torch.clamp(
                logp_ratio, 1 - policy.config["clip_param"],
                1 + policy.config["clip_param"]))
        mean_policy_loss = reduce_mean_valid(-surrogate_loss)

        if policy.config["use_gae"]:
            prev_value_fn_out = train_batch[SampleBatch.VF_PREDS][..., i]
            value_fn_out = model.value_function()[..., i]
            vf_loss1 = torch.pow(
                value_fn_out - train_batch[Postprocessing.VALUE_TARGETS][..., i], 2.0)
            vf_clipped = prev_value_fn_out + torch.clamp(
                value_fn_out - prev_value_fn_out, -policy.config["vf_clip_param"],
                policy.config["vf_clip_param"])
            vf_loss2 = torch.pow(
                vf_clipped - train_batch[Postprocessing.VALUE_TARGETS][..., i], 2.0)
            vf_loss = torch.max(vf_loss1, vf_loss2)
            mean_vf_loss = reduce_mean_valid(vf_loss)
            total_loss = reduce_mean_valid(
                -surrogate_loss + policy.kl_coeff * action_kl[:, i] +
                policy.config["vf_loss_coeff"] * vf_loss -
                policy.entropy_coeff * entropies[:, i])
        else:
            mean_vf_loss = 0.0
            total_loss = reduce_mean_valid(-surrogate_loss +
                                           policy.kl_coeff * action_kl[:, i] -
                                           policy.entropy_coeff * entropies[:, i])

        # Store stats in policy for stats_fn.
        loss_data.append(
            {
                "total_loss": total_loss,
                "mean_policy_loss": mean_policy_loss,
                "mean_vf_loss": mean_vf_loss,
                "mean_entropy": mean_entropy,
            }
        )

    policy._total_loss = (torch.sum(torch.stack([o["total_loss"] for o in loss_data])),)
    policy._mean_policy_loss = torch.mean(
        torch.stack([o["mean_policy_loss"] for o in loss_data])
    )
    policy._mean_vf_loss = torch.mean(
        torch.stack([o["mean_vf_loss"] for o in loss_data])
    )
    policy._mean_entropy = torch.mean(
        torch.stack([o["mean_entropy"] for o in loss_data])
    )
    policy._vf_explained_var = explained_variance(
        train_batch[Postprocessing.VALUE_TARGETS],
        policy.model.value_function())
    policy._mean_kl = mean_kl

    return policy._total_loss


class ValueNetworkMixin:
    """This is exactly the same mixin class as in ppo_torch_policy,
    but that one calls .item() on self.model.value_function()[0],
    which will not work for us since our value function returns
    multiple values. Instead, we call .item() in
    compute_gae_for_sample_batch above.
    """

    def __init__(self, obs_space, action_space, config):
        if config["use_gae"]:

            def value(**input_dict):
                input_dict = SampleBatch(input_dict)
                input_dict = self._lazy_tensor_dict(input_dict)
                model_out, _ = self.model(input_dict)
                # [0] = remove the batch dim.
                return self.model.value_function()[0]

        else:

            def value(*args, **kwargs):
                return 0.0

        self._value = value


def setup_mixins_override(policy: Policy, obs_space: gym.spaces.Space,
                          action_space: gym.spaces.Space,
                          config: TrainerConfigDict) -> None:
    """Have to initialize the custom ValueNetworkMixin
    """
    setup_mixins(policy, obs_space, action_space, config)
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)


# Build a child class of `TorchPolicy`, given the custom functions defined
# above.
MultiPPOTorchPolicy = build_policy_class(
    name="MultiPPOTorchPolicy",
    framework="torch",
    get_default_config=lambda: ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG,
    loss_fn=ppo_surrogate_loss,
    stats_fn=kl_and_loss_stats,
    extra_action_out_fn=vf_preds_fetches,
    postprocess_fn=compute_gae_for_sample_batch,
    extra_grad_process_fn=apply_grad_clipping,
    before_init=setup_config,
    before_loss_init=setup_mixins_override,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin
    ],
)

def get_policy_class(config):
    return MultiPPOTorchPolicy

MultiPPOTrainer = build_trainer(
    name="MultiPPO",
    default_config=ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG,
    validate_config=ray.rllib.agents.ppo.ppo.validate_config,
    default_policy=MultiPPOTorchPolicy,
    get_policy_class=get_policy_class,
    execution_plan=ray.rllib.agents.ppo.ppo.execution_plan
)
