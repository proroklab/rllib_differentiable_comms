# Minimal example for multi-agent RL in RLLib with differentiable communication channel

This is a minimal example to demonstrate how multi-agent reinforcement learning with differentiable communication channels and centralized critics can be realized in RLLib. This example serves as a starting point for making RLLib more compatible with such architectures.

## Environment
The environment is a grid world populated with agents that can navigate in this grid world. Each agent's state consists of a 2D position and goal, both of which are the local observation for each agent. The agents are ordered, and each agent is only rewarded for moving to the next agent's goal. This can only be achieved with a shared, differentiable communication channel.

## Model
The model is summarized in this visualization:

![overview image](https://raw.githubusercontent.com/janblumenkamp/rllib_multi_agent_demo/master/img/ray_multi_agent_demo_model_env.png "Overview")

