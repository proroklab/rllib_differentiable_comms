import numpy as np
import gym
from gym import spaces
from gym.utils import seeding, EzPickle

X = 1
Y = 0


class BaseAgent:
    def __init__(self, index, world_shape, random_state):
        self.goal = None
        self.pose = None
        self.reached_goal = None
        self.random_state = random_state
        self.index = index
        self.world_shape = world_shape
        self.reset()

    def is_valid_pose(self, p):
        return all([0 <= p[c] < self.world_shape[c] for c in [Y, X]])

    def update_pose(self, delta_p):
        desired_pos = self.pose + delta_p
        if self.is_valid_pose(desired_pos):
            self.pose = desired_pos

    def get_obs(self):
        return np.hstack([self.goal, self.pose])

    def reset(self):
        raise NotImplementedError()

    def step(self, action):
        raise NotImplementedError()


class DiscreteAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self):
        self.pose = self.random_state.randint((0, 0), self.world_shape)
        self.goal = self.random_state.randint((0, 0), self.world_shape)
        self.reached_goal = False
        return 0

    def step(self, action):
        delta_pose = {
            0: [0, 0],
            1: [0, 1],
            2: [0, -1],
            3: [-1, 0],
            4: [1, 0],
        }[action]
        self.update_pose(delta_pose)
        return self.get_obs()


class ContinuousAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self):
        self.pose = self.random_state.uniform((0, 0), self.world_shape)
        self.goal = self.random_state.randint((0, 0), self.world_shape)
        self.reached_goal = False
        return [0, 0]

    def step(self, action):
        action_clipped = np.clip(action, -1, 1)
        self.update_pose(action_clipped)
        return self.get_obs()


class InvalidConfigParameter(Exception):
    """Raised when a configuration parameter is invalid"""

    pass


class DemoMultiAgentEnv(gym.Env, EzPickle):
    def __init__(self, env_config):
        EzPickle.__init__(self)
        self.timestep = None
        self.goal_poses = None
        self.random_state = None
        self.seed(1)

        self.cfg = env_config

        self.observation_space = spaces.Dict(
            {
                "agents": spaces.Tuple(
                    (
                        spaces.Box(
                            low=0.0,
                            high=float(max(self.cfg["world_shape"])),
                            shape=(4,),
                        ),
                    )
                    * self.cfg["n_agents"]
                ),
                "state": spaces.Box(
                    low=0.0, high=1.0, shape=self.cfg["world_shape"] + [2]
                ),
            }
        )
        if self.cfg["action_space"] == "discrete":
            agent_action_space = spaces.Discrete(5)
            agent_class = DiscreteAgent
        elif self.cfg["action_space"] == "continuous":
            agent_action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)
            agent_class = ContinuousAgent
        else:
            raise InvalidConfigParameter("Invalid action_space")
        self.action_space = spaces.Tuple((agent_action_space,) * self.cfg["n_agents"])

        self.agents = [
            agent_class(i, self.cfg["world_shape"], self.random_state)
            for i in range(self.cfg["n_agents"])
        ]

        self.reset()

    def seed(self, seed=None):
        self.random_state, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        reset_actions = [agent.reset() for agent in self.agents]
        self.goal_poses = [agent.goal for agent in self.agents]
        self.timestep = 0
        return self.step(reset_actions)[0]

    def step(self, actions):
        self.timestep += 1

        observations = [
            agent.step(action) for agent, action in zip(self.agents, actions)
        ]

        rewards = {}
        # shift each agent's goal so that the shared NN has to be used to solve the problem
        shifted_poses = (
            self.goal_poses[self.cfg["goal_shift"] :]
            + self.goal_poses[: self.cfg["goal_shift"]]
        )
        for i, (agent, goal) in enumerate(zip(self.agents, shifted_poses)):
            rewards[i] = -1 if not agent.reached_goal else 0
            if not agent.reached_goal and np.linalg.norm(agent.pose - goal) < 1:
                rewards[i] = 1
                agent.reached_goal = True

        all_reached_goal = all([agent.reached_goal for agent in self.agents])
        max_timestep_reached = self.timestep == self.cfg["max_episode_len"]
        done = all_reached_goal or max_timestep_reached

        global_state = np.zeros(self.cfg["world_shape"] + [2], dtype=np.uint8)
        for agent in self.agents:
            global_state[int(agent.pose[Y]), int(agent.pose[X]), 0] = 1
            global_state[int(agent.goal[Y]), int(agent.goal[X]), 1] = 1

        obs = {"agents": tuple(observations), "state": global_state}
        info = {"rewards": rewards}
        all_rewards = sum(rewards.values())

        return obs, all_rewards, done, info

    def render(self, mode="human"):
        top_bot_margin = " " + "-" * self.cfg["world_shape"][Y] * 2 + "\n"
        r = top_bot_margin
        for y in range(self.cfg["world_shape"][Y]):
            r += "|"
            for x in range(self.cfg["world_shape"][X]):
                c = " "
                for i, agent in enumerate(self.agents):
                    if np.all(agent.pose.astype(int) == np.array([y, x])):
                        c = "x" if agent.reached_goal else str(i)
                    if np.all(agent.goal == np.array([y, x])):
                        c = "abcdefghijklmnopqrstuvwxyz"[i]
                r += c + " "
            r += "|\n"
        r += top_bot_margin
        print(r)
