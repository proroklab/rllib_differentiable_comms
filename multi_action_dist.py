import gym
import tree
from ray.rllib.models.torch.torch_action_dist import (
    TorchMultiActionDistribution,
    TorchCategorical,
    TorchBeta,
    TorchDiagGaussian,
    TorchDistributionWrapper,
)
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType, List, Union

torch, nn = try_import_torch()


class InvalidActionSpace(Exception):
    """Raised when the action space is invalid"""

    pass


# Override the TorchBeta class to allow for vectors on
class TorchBetaMulti(TorchBeta):
    def __init__(
        self,
        inputs: List[TensorType],
        model: TorchModelV2,
        low: Union[float, TensorType] = 0.0,
        high: Union[float, TensorType] = 1.0,
    ):
        super().__init__(inputs, model)
        device = self.inputs.device
        self.low = torch.tensor(low).to(device)
        self.high = torch.tensor(high).to(device)

        assert len(self.low.shape) == 1, "Low vector of beta must have only 1 dimension"
        assert (
            len(self.high.shape) == 1
        ), "High vector of beta must have only 1 dimension"
        assert (
            self.low.shape[0] == 1 or self.low.shape[0] == self.inputs.shape[-1] // 2
        ), f"Size of low vector of beta must be either 1 ore match the size of the input, got {self.low.shape[0]} expected {self.inputs.shape[-1]}"
        assert (
            self.high.shape[0] == 1 or self.high.shape[0] == self.inputs.shape[-1] // 2
        ), f"Size of high vector of beta must be either 1 ore match the size of the input, got {self.high.shape[0]} expected {self.inputs.shape[-1]}"


class TorchHomogeneousMultiActionDistribution(TorchMultiActionDistribution):
    @override(TorchMultiActionDistribution)
    def __init__(self, inputs, model, *, child_distributions, input_lens, action_space):
        # Skip calling parent constructor, instead call grandparent constructor because
        # we do not want to compute the self.flat_child_distributions in the super constructor
        super(TorchMultiActionDistribution, self).__init__(inputs, model)

        if not isinstance(inputs, torch.Tensor):
            inputs = torch.from_numpy(inputs)
            if isinstance(model, TorchModelV2):
                inputs = inputs.to(next(model.parameters()).device)

        self.action_space_struct = get_base_struct_from_space(action_space)

        self.input_lens = tree.flatten(input_lens)
        split_inputs = torch.split(inputs, self.input_lens, dim=1)
        self.flat_child_distributions = []
        for agent_action_space, agent_inputs in zip(
            self.action_space_struct, split_inputs
        ):
            if isinstance(agent_action_space, gym.spaces.box.Box):
                assert len(agent_action_space.shape) == 1
                if model.use_beta:
                    self.flat_child_distributions.append(
                        TorchBetaMulti(
                            agent_inputs,
                            model,
                            low=agent_action_space.low,
                            high=agent_action_space.high,
                        )
                    )
                else:
                    self.flat_child_distributions.append(
                        TorchDiagGaussian(agent_inputs, model)
                    )
            elif isinstance(agent_action_space, gym.spaces.discrete.Discrete):
                self.flat_child_distributions.append(
                    TorchCategorical(agent_inputs, model)
                )
            else:
                raise InvalidActionSpace(
                    "Expect gym.spaces.box or gym.spaces.discrete action space for each agent"
                )

    @override(TorchMultiActionDistribution)
    def logp(self, x):
        # x.shape = (BATCH, num_agents)
        logps = []
        assert len(self.flat_child_distributions) == len(self.action_space_struct)
        i = 0
        for agent_distribution, agent_action_space in zip(
            self.flat_child_distributions, self.action_space_struct
        ):
            if isinstance(agent_action_space, gym.spaces.box.Box):
                # print(f"Agent action space shape: {action_space.shape}")
                a_w = agent_action_space.shape[0]
                x_agent = x[:, i : (i + a_w)]
                i += a_w
            elif isinstance(agent_action_space, gym.spaces.discrete.Discrete):
                x_agent = x[:, i].int()
                i += 1
            else:
                raise InvalidActionSpace(
                    "Expect gym.spaces.box or gym.spaces.discrete action space for each agent"
                )
            agent_logps = agent_distribution.logp(x_agent)
            if len(agent_logps.shape) > 1:
                agent_logps = torch.sum(agent_logps, dim=1)

            # agent_logps shape (BATCH_SIZE, 1)
            logps.append(agent_logps)

        # logps shape (BATCH_SIZE, NUM_AGENTS)
        return torch.stack(logps, axis=-1)

    @override(TorchMultiActionDistribution)
    def entropy(self):
        entropies = []
        for d in self.flat_child_distributions:
            agent_entropy = d.entropy()
            if len(agent_entropy.shape) > 1:
                agent_entropy = torch.sum(agent_entropy, dim=1)
            entropies.append(agent_entropy)
        return torch.stack(entropies, axis=-1)

    @override(TorchMultiActionDistribution)
    def sampled_action_logp(self):
        return torch.stack(
            [d.sampled_action_logp() for d in self.flat_child_distributions], axis=-1
        )

    @override(TorchMultiActionDistribution)
    def kl(self, other):
        kls = []
        for d, o in zip(self.flat_child_distributions, other.flat_child_distributions):
            agent_kl = d.kl(o)
            if len(agent_kl.shape) > 1:
                agent_kl = torch.sum(agent_kl, dim=1)
            kls.append(agent_kl)
        return torch.stack(
            kls,
            axis=-1,
        )
