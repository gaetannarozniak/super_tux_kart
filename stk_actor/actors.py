from bbrl.workspace import SlicedTemporalTensor
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
        # Retrieve observations from workspace
        obs = {
            key: self.workspace.get(key, t)
            for key in self.workspace.keys()
            if "action" not in key
        }
        action = self.agent.choose_action(obs)
        example = self.workspace.get("env/env_obs/max_steer_angle", t)
        action_tensor = torch.as_tensor(
            action,
            device=self.workspace.get("env/env_obs/max_steer_angle", t).device,
            dtype=torch.float32,  # Use long if discrete actions
        )
        if "action" not in self.workspace.variables:
            self.workspace.variables["action"] = SlicedTemporalTensor()
            self.workspace.set("action", t, action_tensor.unsqueeze(0))
        self.workspace.set("action", t + 1, action_tensor.unsqueeze(0))


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
