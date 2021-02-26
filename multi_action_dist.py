import numpy as np
import tree
from ray.rllib.models.torch.torch_action_dist import TorchMultiActionDistribution
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class TorchHomogeneousMultiActionDistribution(TorchMultiActionDistribution):
    @override(TorchMultiActionDistribution)
    def logp(self, x):
        return torch.stack(
            [d.logp(x[:, i]) for i, d in enumerate(self.flat_child_distributions)],
            axis=-1,
        )

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
