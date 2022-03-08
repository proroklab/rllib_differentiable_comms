import argparse
import ray
from ray import tune

# from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.registry import register_env

from environment import DemoMultiAgentEnv
from model import Model
from ray.rllib.models import ModelCatalog
from multi_trainer import MultiPPOTrainer
from multi_action_dist import TorchHomogeneousMultiActionDistribution


def train(
    share_observations=True, use_beta=True, action_space="discrete", goal_shift=1
):
    ray.init()

    register_env("demo_env", lambda config: DemoMultiAgentEnv(config))
    ModelCatalog.register_custom_model("model", Model)
    ModelCatalog.register_custom_action_dist(
        "hom_multi_action", TorchHomogeneousMultiActionDistribution
    )

    tune.run(
        MultiPPOTrainer,
        checkpoint_freq=1,
        keep_checkpoints_num=1,
        local_dir="/tmp",
        # callbacks=[WandbLoggerCallback(
        #     project="",
        #     api_key_file="",
        #     log_config=True
        # )],
        stop={"training_iteration": 30},
        config={
            "framework": "torch",
            "env": "demo_env",
            "kl_coeff": 0.0,
            "lambda": 0.95,
            "clip_param": 0.2,
            "entropy_coeff": 0.01,
            "train_batch_size": 10000,
            "rollout_fragment_length": 1250,
            "sgd_minibatch_size": 2048,
            "num_sgd_iter": 16,
            "num_gpus": 1,
            "num_workers": 8,
            "num_envs_per_worker": 1,
            "lr": 5e-4,
            "gamma": 0.99,
            "batch_mode": "truncate_episodes",
            "observation_filter": "NoFilter",
            "model": {
                "custom_model": "model",
                "custom_action_dist": "hom_multi_action",
                "custom_model_config": {
                    "encoder_out_features": 8,
                    "shared_nn_out_features_per_agent": 8,
                    "value_state_encoder_cnn_out_features": 16,
                    "share_observations": share_observations,
                    "use_beta": use_beta,
                },
            },
            "env_config": {
                "world_shape": [5, 5],
                "n_agents": 3,
                "max_episode_len": 10,
                "action_space": action_space,
                "goal_shift": goal_shift,
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RLLib multi-agent with shared NN demo."
    )
    parser.add_argument(
        "--action_space",
        default="discrete",
        const="discrete",
        nargs="?",
        choices=["continuous", "discrete"],
        help="Train with continuous or discrete action space",
    )
    parser.add_argument(
        "--disable_sharing",
        action="store_true",
        help="Do not instantiate shared central NN for sharing information",
    )
    parser.add_argument(
        "--disable_beta",
        action="store_true",
        help="Use a gaussian distribution instead of the default beta distribution",
    )
    parser.add_argument(
        "--goal_shift",
        type=int,
        default=1,
        choices=range(0, 2),
        help="Goal shift offset (0 means that each agent moves to its own goal, 1 to its neighbor, etc.)",
    )

    args = parser.parse_args()
    train(
        share_observations=not args.disable_sharing,
        use_beta=not args.disable_beta,
        action_space=args.action_space,
        goal_shift=args.goal_shift,
    )
