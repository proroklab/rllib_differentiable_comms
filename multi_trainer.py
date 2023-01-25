"""
PyTorch's policy class used for PPO.
"""
#  Copyright (c) 2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import logging
from abc import ABC
from typing import Dict, Optional, Union, List, Type

import gym
import numpy as np
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.algorithms.ppo import PPOTorchPolicy
from ray.rllib.algorithms.ppo.ppo_tf_policy import validate_config
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.postprocessing import Postprocessing, compute_advantages
from ray.rllib.models import ModelV2, ActionDistribution
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch, concat_samples
from ray.rllib.policy.torch_mixins import (
    LearningRateSchedule,
    KLCoeffMixin,
    EntropyCoeffSchedule,
)
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import (
    warn_if_infinite_kl_divergence,
    explained_variance,
    sequence_mask,
)
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
    episode: Optional[Episode] = None,
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
    n_agents = len(policy.action_space)

    if sample_batch[SampleBatch.INFOS].dtype == "float32":
        # The trajectory view API will pass populate the info dict with a np.zeros((ROLLOUT_SIZE,))
        # array in the first call, in that case the dtype will be float32, and we
        # ignore it by assignining it to all agents
        samplebatch_infos_rewards = concat_samples(
            [
                SampleBatch(
                    {
                        str(i): sample_batch[SampleBatch.REWARDS].copy()
                        for i in range(n_agents)
                    }
                )
            ]
        )

    else:
        #  For regular calls, we extract the rewards from the info
        #  dict into the samplebatch_infos_rewards dict, which now holds the rewards
        #  for all agents as dict.

        # sample_batch[SampleBatch.INFOS] = list of len ROLLOUT_SIZE of which every element is
        # {'rewards': {0: -0.077463925, 1: -0.0029145998, 2: -0.08233316}} if there are 3 agents

        samplebatch_infos_rewards = concat_samples(
            [
                SampleBatch({str(k): [np.float32(v)] for k, v in s["rewards"].items()})
                for s in sample_batch[SampleBatch.INFOS]
                # s = {'rewards': {0: -0.077463925, 1: -0.0029145998, 2: -0.08233316}} if there are 3 agents
            ]
        )

        # samplebatch_infos_rewards = SampleBatch(ROLLOUT_SIZE: ['0', '1', '2']) if there are 3 agents
        # (i.e. it has ROLLOUT_SIZE entries with keys '0','1','2')

    if not isinstance(policy.action_space, gym.spaces.tuple.Tuple):
        raise InvalidActionSpace("Expect tuple action space")

    keys_to_overwirte = [
        SampleBatch.REWARDS,
        SampleBatch.VF_PREDS,
        Postprocessing.ADVANTAGES,
        Postprocessing.VALUE_TARGETS,
    ]

    original_batch = sample_batch.copy()

    # We prepare the sample batch to contain the agent batches
    for k in keys_to_overwirte:
        sample_batch[k] = np.zeros((len(original_batch), n_agents), dtype=np.float32)

    if original_batch[SampleBatch.DONES][-1]:
        all_values = None
    else:
        input_dict = original_batch.get_single_step_input_dict(
            policy.model.view_requirements, index="last"
        )
        all_values = policy._value(**input_dict)

    # Create the sample_batch for each agent
    for key in samplebatch_infos_rewards.keys():
        agent_index = int(key)
        sample_batch_agent = original_batch.copy()
        sample_batch_agent[SampleBatch.REWARDS] = samplebatch_infos_rewards[key]
        sample_batch_agent[SampleBatch.VF_PREDS] = original_batch[SampleBatch.VF_PREDS][
            :, agent_index
        ]

        if all_values is None:
            last_r = 0.0
        # Trajectory has been truncated -> last r=VF estimate of last obs.
        else:
            last_r = (
                all_values[agent_index].item()
                if policy.config["use_gae"]
                else all_values
            )

        # Adds the policy logits, VF preds, and advantages to the batch,
        # using GAE ("generalized advantage estimation") or not.
        sample_batch_agent = compute_advantages(
            sample_batch_agent,
            last_r,
            policy.config["gamma"],
            policy.config["lambda"],
            use_gae=policy.config["use_gae"],
            use_critic=policy.config.get("use_critic", True),
        )

        for k in keys_to_overwirte:
            sample_batch[k][:, agent_index] = sample_batch_agent[k]

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
    # logits has shape (BATCH, num_agents * num_outputs_per_agent)
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
    # train_batch[SampleBatch.ACTIONS] has shape (BATCH, num_agents * action_size)
    logp_ratio = torch.exp(
        curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
        - train_batch[SampleBatch.ACTION_LOGP]
    )

    use_kl = policy.config["kl_coeff"] > 0.0
    if use_kl:
        action_kl = prev_action_dist.kl(curr_action_dist)
    else:
        action_kl = torch.tensor(0.0, device=logp_ratio.device)

    curr_entropies = curr_action_dist.entropy()

    # Compute a value function loss.
    if policy.config["use_critic"]:
        value_fn_out = model.value_function()
    else:
        value_fn_out = torch.tensor(0.0, device=logp_ratio.device)

    loss_data = []
    n_agents = len(policy.action_space)
    for i in range(n_agents):

        surrogate_loss = torch.min(
            train_batch[Postprocessing.ADVANTAGES][..., i] * logp_ratio[..., i],
            train_batch[Postprocessing.ADVANTAGES][..., i]
            * torch.clamp(
                logp_ratio[..., i],
                1 - policy.config["clip_param"],
                1 + policy.config["clip_param"],
            ),
        )

        # Compute a value function loss.
        if policy.config["use_critic"]:
            agent_value_fn_out = value_fn_out[..., i]
            vf_loss = torch.pow(
                agent_value_fn_out - train_batch[Postprocessing.VALUE_TARGETS][..., i],
                2.0,
            )
            vf_loss_clipped = torch.clamp(vf_loss, 0, policy.config["vf_clip_param"])
            mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
        # Ignore the value function.
        else:
            agent_value_fn_out = torch.tensor(0.0).to(surrogate_loss.device)
            vf_loss_clipped = mean_vf_loss = torch.tensor(0.0).to(surrogate_loss.device)

        total_loss = (
            -surrogate_loss
            + policy.config["vf_loss_coeff"] * vf_loss_clipped
            - policy.entropy_coeff * curr_entropies[..., i]
        )

        # Add mean_kl_loss if necessary.
        if use_kl:
            mean_kl_loss = reduce_mean_valid(action_kl[..., i])
            total_loss += policy.kl_coeff * mean_kl_loss
            # TODO smorad: should we do anything besides warn? Could discard KL term
            # for this update
            warn_if_infinite_kl_divergence(policy, mean_kl_loss)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        total_loss = reduce_mean_valid(total_loss)
        mean_policy_loss = reduce_mean_valid(-surrogate_loss)
        mean_entropy = reduce_mean_valid(curr_entropies[..., i])
        vf_explained_var = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS][..., i], agent_value_fn_out
        )

        # Store stats in policy for stats_fn.
        loss_data.append(
            {
                "total_loss": total_loss,
                "mean_policy_loss": mean_policy_loss,
                "mean_vf_loss": mean_vf_loss,
                "mean_entropy": mean_entropy,
                "mean_kl": mean_kl_loss,
                "vf_explained_var": vf_explained_var,
            }
        )

    aggregation = torch.mean
    total_loss = aggregation(torch.stack([o["total_loss"] for o in loss_data]))

    model.tower_stats["total_loss"] = total_loss
    model.tower_stats["mean_policy_loss"] = aggregation(
        torch.stack([o["mean_policy_loss"] for o in loss_data])
    )
    model.tower_stats["mean_vf_loss"] = aggregation(
        torch.stack([o["mean_vf_loss"] for o in loss_data])
    )
    model.tower_stats["vf_explained_var"] = aggregation(
        torch.stack([o["vf_explained_var"] for o in loss_data])
    )
    model.tower_stats["mean_entropy"] = aggregation(
        torch.stack([o["mean_entropy"] for o in loss_data])
    )
    model.tower_stats["mean_kl_loss"] = aggregation(
        torch.stack([o["mean_kl"] for o in loss_data])
    )

    return total_loss


class MultiAgentValueNetworkMixin:
    """Assigns the `_value()` method to a TorchPolicy.

    This way, Policy can call `_value()` to get the current VF estimate on a
    single(!) observation (as done in `postprocess_trajectory_fn`).
    Note: When doing this, an actual forward pass is being performed.
    This is different from only calling `model.value_function()`, where
    the result of the most recent forward pass is being used to return an
    already calculated tensor.
    """

    def __init__(self, config):
        # When doing GAE, we need the value function estimate on the
        # observation.
        if config["use_gae"]:
            # Input dict is provided to us automatically via the Model's
            # requirements. It's a single-timestep (last one in trajectory)
            # input_dict.
            def value(**input_dict):
                """This is exactly the as in PPOTorchPolicy,
                but that one calls .item() on self.model.value_function()[0],
                which will not work for us since our value function returns
                multiple values. Instead, we call .item() in
                compute_gae_for_sample_batch above.
                """
                input_dict = SampleBatch(input_dict)
                input_dict = self._lazy_tensor_dict(input_dict)
                model_out, _ = self.model(input_dict)
                # [0] = remove the batch dim.
                return self.model.value_function()[0]
                # When not doing GAE, we do not require the value function's output.

        # When not doing GAE, we do not require the value function's output.
        else:

            def value(*args, **kwargs):
                return 0.0

        self._value = value


class MultiPPOTorchPolicy(PPOTorchPolicy, MultiAgentValueNetworkMixin):
    def __init__(self, observation_space, action_space, config):
        config = dict(ray.rllib.algorithms.ppo.ppo.PPOConfig().to_dict(), **config)
        # TODO: Move into Policy API, if needed at all here. Why not move this into
        #  `PPOConfig`?.
        validate_config(config)

        TorchPolicyV2.__init__(
            self,
            observation_space,
            action_space,
            config,
            max_seq_len=config["model"]["max_seq_len"],
        )

        # Only difference from ray code
        MultiAgentValueNetworkMixin.__init__(self, config)
        LearningRateSchedule.__init__(self, config["lr"], config["lr_schedule"])
        EntropyCoeffSchedule.__init__(
            self, config["entropy_coeff"], config["entropy_coeff_schedule"]
        )
        KLCoeffMixin.__init__(self, config)

        # TODO: Don't require users to call this manually.
        self._initialize_loss_from_dummy_batch()

    @override(PPOTorchPolicy)
    def loss(self, model, dist_class, train_batch):
        return ppo_surrogate_loss(self, model, dist_class, train_batch)

    @override(PPOTorchPolicy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        # Do all post-processing always with no_grad().
        # Not using this here will introduce a memory leak
        # in torch (issue #6962).
        # TODO: no_grad still necessary?
        with torch.no_grad():
            return compute_gae_for_sample_batch(
                self, sample_batch, other_agent_batches, episode
            )


class MultiPPOTrainer(PPOTrainer, ABC):
    @override(PPOTrainer)
    def get_default_policy_class(self, config):
        return MultiPPOTorchPolicy
