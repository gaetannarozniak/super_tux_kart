import numpy as np
from .sac_torch import AgentSac
import gymnasium as gym
from bbrl.agents import Agent
import torch


class MyWrapper(gym.ActionWrapper):
    def __init__(self, env, option: int):
        super().__init__(env)
        self.option = option

    def action(self, action):
        # We do nothing here
        return action


class ActorSac(Agent):
    def __init__(self, obs_space: gym.Space, action_space: gym.Space):
        super().__init__()

        self.agent = AgentSac(
            observation_space=obs_space,
            action_space=action_space,
        )
        self.obs_space = obs_space
        self.action_space = action_space

    def forward(self, t: int):
        obs = {
            key: self.workspace.get(key, t) for key in self.workspace.variables.keys()
        }
        action = self.agent.choose_action(obs)
        action = torch.LongTensor(np.array(action))
        self.set(
            ("action", t), torch.LongTensor(np.array([self.action_space.sample()]))
        )


class Actor(Agent):
    """Computes probabilities over action"""

    def __init__(self, obs_space: gym.Space, action_space: gym.Space):
        super().__init__()
        self.obs_space = obs_space
        self.action_space = action_space

    def forward(self, t: int):
        self.set(("action", t), torch.LongTensor([self.action_space.sample()]))


class ArgmaxActor(Agent):
    """Actor that computes the action"""

    def forward(self, t: int):
        # Selects the best actions according to the policy
        pass


class SamplingActor(Agent):
    """Samples random actions"""

    def __init__(self, action_space: gym.Space):
        super().__init__()
        self.action_space = action_space

    def forward(self, t: int):
        self.set(("action", t), torch.LongTensor([self.action_space.sample()]))
