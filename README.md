# Minimal example for multi-agent RL in RLLib with differentiable communication channel

This is a minimal example to demonstrate how multi-agent reinforcement learning with differentiable communication channels and centralized critics can be realized in RLLib. This example serves as a reference implementation and starting point for making RLLib more compatible with such architectures.

## Citation
This project is derived from our paper "The Emergence of Adversarial Communication in Multi-Agent Reinforcement Learning" (code available [here](https://github.com/proroklab/adversarial_comms)). If you use any part of this code in your research, please cite our paper:
```
@article{blumenkamp2020adversarial,
  title={The Emergence of Adversarial Communication in Multi-Agent Reinforcement Learning},
  author={Blumenkamp, Jan and Prorok, Amanda},
  journal={Conference on Robot Learning (CoRL)},
  year={2020}
}
```

## Introduction
While RLLib provides a multi-agent API, this API only supports non-differentiable communication channels, which means that agents can only communicate to each other through their actions and observations. Recently, Graph Neural Networks have gained traction and were demonstrated to be effective to learn decentralized homogeneous control policies for muli-agent systems [[1]](https://arxiv.org/abs/2012.14906), [[2]](https://arxiv.org/abs/2008.02616), [[3]](https://arxiv.org/abs/1912.06095). While it would be possible to learn a policy that utilizes such a GNN in RLLib with a centralized model and an environment that encapsulates all observations and actions at once, it will be hard to learn anything due to the credit assignment problem, since the default OpenAI Gym interface only allows to return a single scalar reward per time step.

This repository uses the `info` dict that is returned from the `step` function to propagate local per-agent rewards into RLLib. The rewards are extracted for each agent and the trajectories are processed and discounted individually. The PPO loss is then computed for each trajectory and eventually summed so that the NN can be optimized for all agents. This procedure is explained and derived in [[2]](https://arxiv.org/abs/2008.02616).

## Environment
The environment is a (grid) world populated with agents that can move in 2D space, either discrete (into one of its four neighboring cells) or continuous (dx and dy). Each agent's state consists of a 2D position and goal, both of which are the local observation for each agent. The agents are ordered, and each agent is only rewarded for moving to the next agent's goal. This can only be achieved with a shared, differentiable communication channel.

## Model
Instead of a GNN, we use a simple centralized feedforward network. Such a model assumes a fixed number of agents and will not work for most more complicated scenarios, but for the purpose of demonstrating the efficiency of the implemented trainer it is sufficient. The model is summarized in this visualization:

![overview image](https://raw.githubusercontent.com/janblumenkamp/rllib_multi_agent_demo/master/img/ray_multi_agent_demo_model_env.png "Overview")

## Setup
```
pip install -r requirements.txt
```

## Results

![overview image](https://raw.githubusercontent.com/janblumenkamp/rllib_multi_agent_demo/master/img/results_rewards.svg "Overview")

| Type | Comm | Command                                                       | Reward | Ep len |
|------|------|---------------------------------------------------------------|--------|--------|
| Cont | yes  | `python train.py --action_space continuous`                   | -0.75  | 3.0    |
| Dis  | yes  | `python train.py --action_space discrete`                     | -3.95  | 4.7    |
| Cont | no   | `python train.py --action_space continuous --disable_sharing` | -15.3  | 8.5    |
| Dis  | no   | `python train.py --action_space discrete --disable_sharing`   | -21.2  | 8.9    |

