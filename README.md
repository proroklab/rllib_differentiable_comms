# Minimal example for multi-agent RL in RLLib with differentiable communication channel

This is a minimal example to demonstrate how multi-agent reinforcement learning with differentiable communication channels and centralized critics can be realized in RLLib. This example serves as a reference implementation and starting point for making RLLib more compatible with such architectures.

## Environment
The environment is a (grid) world populated with agents that can move in 2D space, either discrete (into one of its four neighboring cells) or continuous (dx and dy). Each agent's state consists of a 2D position and goal, both of which are the local observation for each agent. The agents are ordered, and each agent is only rewarded for moving to the next agent's goal. This can only be achieved with a shared, differentiable communication channel.

## Model
The model is summarized in this visualization:

![overview image](https://raw.githubusercontent.com/janblumenkamp/rllib_multi_agent_demo/master/img/ray_multi_agent_demo_model_env.png "Overview")

## Setup
The most recent Ray version has to be installed from master from commit `8cedd16f4440f5baf8c68d5012896512466c9f6a`. `requirements.txt` assumes Python 3.8 and Linux for Ray, depending on your Python version and operating system you have to modify it as exlained [here](https://docs.ray.io/en/master/installation.html#installing-from-a-specific-commit).

## Results

![overview image](https://raw.githubusercontent.com/janblumenkamp/rllib_multi_agent_demo/master/img/results_rewards.svg "Overview")

| Type | Comm | Command                                                       | Reward | Ep len |
|------|------|---------------------------------------------------------------|--------|--------|
| Cont | yes  | `python train.py --action_space discrete`                     | -0.5   | 2.9    |
| Dis  | yes  | `python train.py --action_space discrete`                     | -3.9   | 4.7    |
| Cont | no   | `python train.py --action_space continuous --disable_sharing` | -14.8  | 8.4    |
| Dis  | no   | `python train.py --action_space continuous --disable_sharing` | -22.2  | 8.9    |

