"""
PyTorch's policy class used for PPO.
"""
#  Copyright (c) 2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import logging
from abc import ABC
from typing import Dict
from typing import List, Optional, Union
from typing import Type

import gym
import numpy as np
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOTorchPolicy
from ray.rllib.algorithms.ppo.ppo_tf_policy import validate_config
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.postprocessing import Postprocessing, compute_advantages
from ray.rllib.execution import synchronous_parallel_sample
from ray.rllib.execution.common import (
    _check_sample_batch_type,
)
from ray.rllib.execution.train_ops import (
    train_one_step,
    multi_gpu_train_one_step,
)
from ray.rllib.models import ModelV2, ActionDistribution
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import (
    SampleBatch,
    DEFAULT_POLICY_ID,
    concat_samples,
)
from ray.rllib.policy.torch_mixins import (
    LearningRateSchedule,
    KLCoeffMixin,
    EntropyCoeffSchedule,
)
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SYNCH_WORKER_WEIGHTS_TIMER,
)
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import (
    apply_grad_clipping,
)
from ray.rllib.utils.torch_utils import (
    warn_if_infinite_kl_divergence,
    explained_variance,
    sequence_mask,
)
from ray.rllib.utils.typing import AgentID, TensorType, ResultDict
from ray.rllib.utils.typing import PolicyID, SampleBatchType

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)


class InvalidActionSpace(Exception):
    """Raised when the action space is invalid"""

    pass


def standardized(array: np.ndarray):
    """Normalize the values in an array.

    Args:
        array (np.ndarray): Array of values to normalize.

    Returns:
        array with zero mean and unit standard deviation.
    """
    return (array - array.mean(axis=0, keepdims=True)) / array.std(
        axis=0, keepdims=True
    ).clip(min=1e-4)


def standardize_fields(samples: SampleBatchType, fields: List[str]) -> SampleBatchType:
    """Standardize fields of the given SampleBatch"""
    _check_sample_batch_type(samples)
    wrapped = False

    if isinstance(samples, SampleBatch):
        samples = samples.as_multi_agent()
        wrapped = True

    for policy_id in samples.policy_batches:
        batch = samples.policy_batches[policy_id]
        for field in fields:
            if field in batch:
                batch[field] = standardized(batch[field])

    if wrapped:
        samples = samples.policy_batches[DEFAULT_POLICY_ID]

    return samples


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
        self.grad_gnorm = 0

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

    @override(PPOTorchPolicy)
    def extra_grad_process(self, local_optimizer, loss):
        grad_gnorm = apply_grad_clipping(self, local_optimizer, loss)
        if "grad_gnorm" in grad_gnorm:
            self.grad_gnorm = grad_gnorm["grad_gnorm"]
        return grad_gnorm

    @override(TorchPolicyV2)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        return convert_to_numpy(
            {
                "cur_kl_coeff": self.kl_coeff,
                "cur_lr": self.cur_lr,
                "total_loss": torch.mean(
                    torch.stack(self.get_tower_stats("total_loss"))
                ),
                "policy_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_policy_loss"))
                ),
                "vf_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_vf_loss"))
                ),
                "vf_explained_var": torch.mean(
                    torch.stack(self.get_tower_stats("vf_explained_var"))
                ),
                "kl": torch.mean(torch.stack(self.get_tower_stats("mean_kl_loss"))),
                "entropy": torch.mean(
                    torch.stack(self.get_tower_stats("mean_entropy"))
                ),
                "entropy_coeff": self.entropy_coeff,
                "grad_gnorm": self.grad_gnorm,
            }
        )


class MultiPPOTrainer(PPOTrainer, ABC):
    @override(PPOTrainer)
    def get_default_policy_class(self, config):
        return MultiPPOTorchPolicy

    @override(PPOTrainer)
    def training_step(self) -> ResultDict:
        # Collect SampleBatches from sample workers until we have a full batch.
        if self._by_agent_steps:
            assert False
            train_batch = synchronous_parallel_sample(
                worker_set=self.workers, max_agent_steps=self.config["train_batch_size"]
            )
        else:
            train_batch = synchronous_parallel_sample(
                worker_set=self.workers, max_env_steps=self.config["train_batch_size"]
            )
        train_batch = train_batch.as_multi_agent()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()

        # Standardize advantage
        train_batch = standardize_fields(train_batch, ["advantages"])
        # Train
        if self.config["simple_optimizer"]:
            assert False
            train_results = train_one_step(self, train_batch)
        else:
            train_results = multi_gpu_train_one_step(self, train_batch)

        global_vars = {
            "timestep": self._counters[NUM_AGENT_STEPS_SAMPLED],
        }

        # Update weights - after learning on the local worker - on all remote
        # workers.
        if self.workers.remote_workers():
            with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
                self.workers.sync_weights(global_vars=global_vars)

        # For each policy: update KL scale and warn about possible issues
        for policy_id, policy_info in train_results.items():
            # Update KL loss with dynamic scaling
            # for each (possibly multiagent) policy we are training
            kl_divergence = policy_info[LEARNER_STATS_KEY].get("kl")
            self.get_policy(policy_id).update_kl(kl_divergence)

            # Warn about excessively high value function loss
            scaled_vf_loss = (
                self.config["vf_loss_coeff"] * policy_info[LEARNER_STATS_KEY]["vf_loss"]
            )
            policy_loss = policy_info[LEARNER_STATS_KEY]["policy_loss"]
            if scaled_vf_loss > 100:
                logger.warning(
                    "The magnitude of your value function loss for policy: {} is "
                    "extremely large ({}) compared to the policy loss ({}). This "
                    "can prevent the policy from learning. Consider scaling down "
                    "the VF loss by reducing vf_loss_coeff, or disabling "
                    "vf_share_layers.".format(policy_id, scaled_vf_loss, policy_loss)
                )
            # Warn about bad clipping configs.
            train_batch.policy_batches[policy_id].set_get_interceptor(None)
            mean_reward = train_batch.policy_batches[policy_id]["rewards"].mean()
            if mean_reward > self.config["vf_clip_param"]:
                self.warned_vf_clip = True
                logger.warning(
                    f"The mean reward returned from the environment is {mean_reward}"
                    f" but the vf_clip_param is set to {self.config['vf_clip_param']}."
                    f" Consider increasing it for policy: {policy_id} to improve"
                    " value function convergence."
                )

        # Update global vars on local worker as well.
        self.workers.local_worker().set_global_vars(global_vars)

        return train_results
