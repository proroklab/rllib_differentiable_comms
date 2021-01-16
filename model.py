from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.models.torch.misc import (
    normc_initializer,
    same_padding,
    SlimConv2d,
    SlimFC,
)
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_torch

import numpy as np
import copy

torch, nn = try_import_torch()

# https://ray.readthedocs.io/en/latest/using-ray-with-pytorch.html

DEFAULT_OPTIONS = {
    "encoder_out_features": 8,
    "shared_nn_out_features_per_agent": 8,
    "value_state_encoder_cnn_out_features": 16,
    "share_observations": True,
}


class Model(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.cfg = copy.deepcopy(DEFAULT_OPTIONS)
        self.cfg.update(model_config["custom_model_config"])

        self.n_agents = len(obs_space.original_space["agents"])
        self.outputs_per_agent = int(num_outputs / self.n_agents)

        obs_shape = obs_space.original_space["agents"][0].shape
        state_shape = obs_space.original_space["state"].shape

        self.action_encoder = nn.Sequential(
            nn.Linear(obs_shape[0], 32),
            nn.ReLU(),
            nn.Linear(32, self.cfg["encoder_out_features"]),
            nn.ReLU(),
        )

        share_n_agents = self.n_agents if self.cfg["share_observations"] else 1
        self.action_shared = nn.Sequential(
            nn.Linear(self.cfg["encoder_out_features"] * share_n_agents, 64),
            nn.ReLU(),
            nn.Linear(
                64, self.cfg["shared_nn_out_features_per_agent"] * share_n_agents
            ),
            nn.ReLU(),
        )

        post_logits = [
            nn.Linear(self.cfg["shared_nn_out_features_per_agent"], 32),
            nn.ReLU(),
            nn.Linear(32, self.outputs_per_agent),
        ]
        nn.init.xavier_uniform_(post_logits[-1].weight)
        nn.init.constant_(post_logits[-1].bias, 0)
        self.action_output = nn.Sequential(*post_logits)

        #############
        self.value_encoder = nn.Sequential(
            nn.Linear(obs_shape[0], 32),
            nn.ReLU(),
            nn.Linear(32, self.cfg["encoder_out_features"]),
            nn.ReLU(),
        )

        self.value_encoder_state = nn.Sequential(
            SlimConv2d(
                2, 8, 3, 2, 1
            ),  # in_channels, out_channels, kernel, stride, padding
            SlimConv2d(
                8, 8, 3, 2, 1
            ),  # in_channels, out_channels, kernel, stride, padding
            SlimConv2d(8, self.cfg["value_state_encoder_cnn_out_features"], 3, 2, 1),
            nn.Flatten(1, -1),
        )

        self.value_shared = nn.Sequential(
            nn.Linear(
                self.cfg["encoder_out_features"] * self.n_agents
                + self.cfg["value_state_encoder_cnn_out_features"],
                64,
            ),
            nn.ReLU(),
            nn.Linear(64, self.cfg["shared_nn_out_features_per_agent"] * self.n_agents),
            nn.ReLU(),
        )

        value_post_logits = [
            nn.Linear(self.cfg["shared_nn_out_features_per_agent"], 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        ]
        nn.init.xavier_uniform_(value_post_logits[-1].weight)
        nn.init.constant_(value_post_logits[-1].bias, 0)
        self.value_output = nn.Sequential(*value_post_logits)

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        batch_size = input_dict["obs"]["state"].shape[0]
        device = input_dict["obs"]["state"].device

        action_feature_map = torch.zeros(
            batch_size, self.n_agents, self.cfg["encoder_out_features"]
        ).to(device)
        value_feature_map = torch.zeros(
            batch_size, self.n_agents, self.cfg["encoder_out_features"]
        ).to(device)
        for i in range(self.n_agents):
            agent_obs = input_dict["obs"]["agents"][i]
            action_feature_map[:, i] = self.action_encoder(agent_obs)
            value_feature_map[:, i] = self.value_encoder(agent_obs)
        value_state_features = self.value_encoder_state(
            input_dict["obs"]["state"].permute(0, 3, 1, 2)
        )

        if self.cfg["share_observations"]:
            # We have a big common shared center NN so that all agents have access to the encoded observations of all agents
            action_shared_features = self.action_shared(
                action_feature_map.view(
                    batch_size, self.n_agents * self.cfg["encoder_out_features"]
                )
            ).view(
                batch_size, self.n_agents, self.cfg["shared_nn_out_features_per_agent"]
            )
        else:
            # Each agent only has access to its own local observation
            action_shared_features = torch.empty(
                batch_size, self.n_agents, self.cfg["shared_nn_out_features_per_agent"]
            ).to(device)
            for i in range(self.n_agents):
                action_shared_features[:, i] = self.action_shared(
                    action_feature_map[:, i]
                )

        value_shared_features = self.value_shared(
            torch.cat(
                [
                    value_feature_map.view(
                        batch_size, self.n_agents * self.cfg["encoder_out_features"]
                    ),
                    value_state_features,
                ],
                dim=1,
            )
        ).view(batch_size, self.n_agents, self.cfg["shared_nn_out_features_per_agent"])

        outputs = torch.empty(batch_size, self.n_agents, self.outputs_per_agent).to(
            device
        )
        values = torch.empty(batch_size, self.n_agents).to(device)

        for i in range(self.n_agents):
            outputs[:, i] = self.action_output(action_shared_features[:, i])
            values[:, i] = self.value_output(value_shared_features[:, i]).squeeze(1)

        self._cur_value = values

        return outputs.view(batch_size, self.n_agents * self.outputs_per_agent), state

    @override(ModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value
