import gym
import numpy as np
import tree
from ray.rllib.models.torch.torch_action_dist import TorchMultiActionDistribution
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class InvalidActionSpace(Exception):
    """Raised when the action space is invalid"""

    pass


class TorchHomogeneousMultiActionDistribution(TorchMultiActionDistribution):
    @override(TorchMultiActionDistribution)
    def logp(self, x):
        logps = []
        for i, (d, action_space) in enumerate(
            zip(self.flat_child_distributions, self.action_space_struct)
        ):
            if isinstance(action_space, gym.spaces.box.Box):
                assert len(action_space.shape) == 1
                a_w = action_space.shape[0]
                x_sel = x[:, a_w * i : a_w * (i + 1)]
            elif isinstance(action_space, gym.spaces.discrete.Discrete):
                x_sel = x[:, i]
            else:
                raise InvalidActionSpace(
                    "Expect gym.spaces.box or gym.spaces.discrete action space"
                )
            logps.append(d.logp(x_sel))

        return torch.stack(logps, axis=1)

    @override(TorchMultiActionDistribution)
    def entropy(self):
        return torch.stack(
            [d.entropy() for d in self.flat_child_distributions], axis=-1
        )

    @override(TorchMultiActionDistribution)
    def sampled_action_logp(self):
        return torch.stack(
            [d.sampled_action_logp() for d in self.flat_child_distributions], axis=-1
        )

    @override(TorchMultiActionDistribution)
    def kl(self, other):
        return torch.stack(
            [
                d.kl(o)
                for d, o in zip(
                    self.flat_child_distributions, other.flat_child_distributions
                )
            ],
            axis=-1,
        )
