from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import (
    SlimConv2d,
)
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import try_import_torch
from ray.rllib.utils.annotations import override

torch, nn = try_import_torch()


class Model(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        encoder_out_features,
        shared_nn_out_features_per_agent,
        value_state_encoder_cnn_out_features,
        share_observations,
        use_beta,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.encoder_out_features = encoder_out_features
        self.shared_nn_out_features_per_agent = shared_nn_out_features_per_agent
        self.value_state_encoder_cnn_out_features = value_state_encoder_cnn_out_features
        self.share_observations = share_observations
        self.use_beta = use_beta

        self.n_agents = len(obs_space.original_space["agents"])
        self.outputs_per_agent = int(num_outputs / self.n_agents)

        obs_shape = obs_space.original_space["agents"][0].shape
        state_shape = obs_space.original_space["state"].shape

        ###########
        # Action NN

        self.action_encoder = nn.Sequential(
            nn.Linear(obs_shape[0], 32),
            nn.ReLU(),
            nn.Linear(32, self.encoder_out_features),
            nn.ReLU(),
        )

        share_n_agents = self.n_agents if self.share_observations else 1
        self.action_shared = nn.Sequential(
            nn.Linear(self.encoder_out_features * share_n_agents, 64),
            nn.ReLU(),
            nn.Linear(64, self.shared_nn_out_features_per_agent * share_n_agents),
            nn.ReLU(),
        )

        post_logits = [
            nn.Linear(self.shared_nn_out_features_per_agent, 32),
            nn.ReLU(),
            nn.Linear(32, self.outputs_per_agent),
        ]
        nn.init.xavier_uniform_(post_logits[-1].weight)
        nn.init.constant_(post_logits[-1].bias, 0)
        self.action_output = nn.Sequential(*post_logits)

        ###########
        # Value NN

        self.value_encoder = nn.Sequential(
            nn.Linear(obs_shape[0], 32),
            nn.ReLU(),
            nn.Linear(32, self.encoder_out_features),
            nn.ReLU(),
        )

        self.value_encoder_state = nn.Sequential(
            SlimConv2d(
                2, 8, 3, 2, 1
            ),  # in_channels, out_channels, kernel, stride, padding
            SlimConv2d(
                8, 8, 3, 2, 1
            ),  # in_channels, out_channels, kernel, stride, padding
            SlimConv2d(8, self.value_state_encoder_cnn_out_features, 3, 2, 1),
            nn.Flatten(1, -1),
        )

        self.value_shared = nn.Sequential(
            nn.Linear(
                self.encoder_out_features * self.n_agents
                + self.value_state_encoder_cnn_out_features,
                64,
            ),
            nn.ReLU(),
            nn.Linear(64, self.shared_nn_out_features_per_agent * self.n_agents),
            nn.ReLU(),
        )

        value_post_logits = [
            nn.Linear(self.shared_nn_out_features_per_agent, 32),
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
            batch_size, self.n_agents, self.encoder_out_features
        ).to(device)
        value_feature_map = torch.zeros(
            batch_size, self.n_agents, self.encoder_out_features
        ).to(device)
        for i in range(self.n_agents):
            agent_obs = input_dict["obs"]["agents"][i]
            action_feature_map[:, i] = self.action_encoder(agent_obs)
            value_feature_map[:, i] = self.value_encoder(agent_obs)
        value_state_features = self.value_encoder_state(
            input_dict["obs"]["state"].permute(0, 3, 1, 2)
        )

        if self.share_observations:
            # We have a big common shared center NN so that all agents have access to the encoded observations of all agents
            action_shared_features = self.action_shared(
                action_feature_map.view(
                    batch_size, self.n_agents * self.encoder_out_features
                )
            ).view(batch_size, self.n_agents, self.shared_nn_out_features_per_agent)
        else:
            # Each agent only has access to its own local observation
            action_shared_features = torch.empty(
                batch_size, self.n_agents, self.shared_nn_out_features_per_agent
            ).to(device)
            for i in range(self.n_agents):
                action_shared_features[:, i] = self.action_shared(
                    action_feature_map[:, i]
                )

        value_shared_features = self.value_shared(
            torch.cat(
                [
                    value_feature_map.view(
                        batch_size, self.n_agents * self.encoder_out_features
                    ),
                    value_state_features,
                ],
                dim=1,
            )
        ).view(batch_size, self.n_agents, self.shared_nn_out_features_per_agent)

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
