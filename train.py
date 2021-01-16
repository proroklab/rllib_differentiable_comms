import time
import numpy as np
import yaml
import ray
from ray import tune
from ray.tune.registry import register_env

from ray.tune.logger import pretty_print, DEFAULT_LOGGERS, TBXLogger
from ray.tune.integration.wandb import WandbLogger

from environment import DemoMultiAgentEnv
from model import Model
from ray.rllib.models import ModelCatalog
from multi_trainer import MultiPPOTrainer
from multi_action_dist import TorchHomogeneousMultiActionDistribution

if __name__ == '__main__':
    #ray.init(local_mode=True)
    ray.init()

    register_env("demo_env", lambda config: DemoMultiAgentEnv(config))
    ModelCatalog.register_custom_model("model", Model)
    ModelCatalog.register_custom_action_dist("hom_multi_action", TorchHomogeneousMultiActionDistribution)

    num_workers = 16
    tune.run(
        MultiPPOTrainer,
        #restore="/home/jb2270/ray_results/PPO/PPO_world_0_2020-04-04_23-01-16c532w9iy/checkpoint_100/checkpoint-100",
        checkpoint_freq=1,
        keep_checkpoints_num=1,
        #local_dir="/tmp",
        loggers=DEFAULT_LOGGERS + (WandbLogger,),
        config={
            "framework": "torch",
            "env": "demo_env",
            "lambda": 0.95,
            "clip_param": 0.2,
            "entropy_coeff": 0.01,
            "train_batch_size": 10000,
            "sgd_minibatch_size": 2048,
            "num_sgd_iter": 16,
            "num_gpus": 1,
            "num_workers": 8,
            "num_envs_per_worker": 1,
            "lr": 5e-4,
            "gamma": 0.99,
            "batch_mode": "truncate_episodes", # complete_episodes, truncate_episodes
            "observation_filter": "NoFilter",
            "model": {
                "custom_model": "model",
                "custom_action_dist": "hom_multi_action",
                "custom_model_config": {
                    "encoder_out_features": 8,
                    "shared_nn_out_features_per_agent": 8,
                    "value_state_encoder_cnn_out_features": 16,
                    "share_observations": False,
                }
            },
            "logger_config": {
                "wandb": {
                    "project": "ray_multi_agent_trajectory",
                    "group": "a",
                    "api_key_file": "./wandb_api_key_file"
                }
            },
            "env_config": {
                'world_shape': [5, 5],
                'n_agents': 3,
                'max_episode_len': 10
            }
        }
    )

