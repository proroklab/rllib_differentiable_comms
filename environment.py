import numpy as np
import gym
from gym import spaces
from gym.utils import seeding, EzPickle
from enum import Enum
import copy

DEFAULT_OPTIONS = {
    'world_shape': [5, 5],
    'n_agents': 3,
    'max_episode_len': 5
}

X = 1
Y = 0

class Action(Enum):
    NOP         = 0
    MOVE_RIGHT  = 1
    MOVE_LEFT   = 2
    MOVE_UP     = 3
    MOVE_DOWN   = 4

class Agent():
    def __init__(self, index, world_shape, random_state):
        self.random_state = random_state
        self.index = index
        self.world_shape = world_shape
        self.reset()

    def reset(self):
        self.pose = self.random_state.randint((0, 0), self.world_shape, (2,))
        self.goal = self.random_state.randint((0, 0), self.world_shape, (2,))
        self.reached_goal = False
        return self.goal

    def step(self, action):
        delta_pose = {
            Action.MOVE_RIGHT:  [ 0,  1],
            Action.MOVE_LEFT:   [ 0, -1],
            Action.MOVE_UP:     [-1,  0],
            Action.MOVE_DOWN:   [ 1,  0],
            Action.NOP:         [ 0,  0]
        }[Action(action)]

        is_valid_pose = lambda p: all([p[c] >= 0 and p[c] < self.world_shape[c] for c in [Y, X]])
        desired_pos = self.pose + delta_pose
        if is_valid_pose(desired_pos):
            self.pose = desired_pos

        return np.hstack([self.goal, self.pose])

class DemoMultiAgentEnv(gym.Env, EzPickle):
    def __init__(self, env_config):
        EzPickle.__init__(self)
        self.seed(1)

        self.cfg = copy.deepcopy(DEFAULT_OPTIONS)
        self.cfg.update(env_config)

        self.observation_space = spaces.Dict({
            'agents': spaces.Tuple((
                spaces.Box(low=0, high=max(self.cfg['world_shape']), shape=(4,)),
            )*self.cfg['n_agents']),
            'state': spaces.Box(low=0, high=1, shape=self.cfg['world_shape']+[2]),
        })
        self.action_space = spaces.Tuple((spaces.Discrete(5),)*self.cfg['n_agents'])

        self.agents = [Agent(i, self.cfg['world_shape'], self.random_state) for i in range(self.cfg['n_agents'])]

        self.reset()

    def seed(self, seed=None):
        self.random_state, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.goal_poses = [agent.reset() for agent in self.agents]
        self.timestep = 0
        return self.step([Action.NOP]*self.cfg['n_agents'])[0]

    def step(self, actions):
        self.timestep += 1

        observations = [agent.step(action) for agent, action in zip(self.agents, actions)]

        rewards = {}
        # shift each agent's goal so that the shared NN has to be used to solve the problem
        shift = 1
        shifted_poses = self.goal_poses[shift:]+self.goal_poses[:shift]
        for i, (agent, goal) in enumerate(zip(self.agents, shifted_poses)):
            rewards[i] = -1 if not agent.reached_goal else 0
            if not agent.reached_goal and np.all(agent.pose == goal):
                rewards[i] = 1
                agent.reached_goal = True

        all_reached_goal = all([agent.reached_goal for agent in self.agents])
        max_timestep_reached = self.timestep == self.cfg['max_episode_len']
        done = all_reached_goal or max_timestep_reached

        global_state = np.zeros(self.cfg['world_shape'] + [2], dtype=np.uint8)
        for agent in self.agents:
            global_state[agent.pose[Y], agent.pose[X], 0] = 1
            global_state[agent.goal[Y], agent.goal[X], 1] = 1

        obs = {
            'agents': tuple(observations),
            'state': global_state
        }
        info = {'rewards': rewards}
        all_rewards = sum(rewards.values())

        return obs, all_rewards, done, info

    def render(self, mode='human'):
        top_bot_margin = " " + "-"*self.cfg['world_shape'][Y]*2 + "\n"
        r = top_bot_margin
        for y in range(self.cfg['world_shape'][Y]):
            r += "|"
            for x in range(self.cfg['world_shape'][X]):
                c = ' '
                for i, agent in enumerate(self.agents):
                    if np.all(agent.pose == np.array([y, x])):
                        c = "x" if agent.reached_goal else str(i)
                    if np.all(agent.goal == np.array([y, x])):
                        c = "abcdefghijklmnopqrstuvwxyz"[i]
                r += c + " "
            r += "|\n"
        r += top_bot_margin
        print(r)

