import gymnasium as gym
from gymnasium.core import Env, spaces
import numpy as np

class ObsFilterWrapper(gym.ObservationWrapper): 
    def __init__(self, env: Env, obs_filters):#, action_filters):
        super().__init__(env)
        self.obs_filters = obs_filters
        # self.action_filters = action_filters
        obs_space_dict = {k: env.observation_space[k] for k in 
                          env.observation_space.keys() if k in obs_filters}
        # obs_space_dict.update({k: env.observation_space['action'][k] for
                            #    k in env.observation_space['action'].keys() if
                            #    k in action_filters})
        self.observation_space = spaces.Dict(obs_space_dict)

    def observation(self, observation):
        obs_filtered = {k:v for k,v in observation.items() if k in self.obs_filters}
        # action_filtered = {k:v for k,v in observation['action'].items() if k in self.action_filters}
        return obs_filtered # | action_filtered

class RewardCenterTrackWrapper(gym.Wrapper):
    def __init__(self, env: Env, scale=1):
        super().__init__(env)
        self.scale = scale

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        distance_center_loss = self.scale*np.abs(obs['center_path'][2])
        reward -= distance_center_loss
        return obs, reward, done, truncated, info

class FlattenObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        flat_size = np.sum(np.prod(space.shape) for space in env.observation_space.values())
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_size,), dtype=np.float32
        )

    def observation(self, observation):
        obs_flattened = np.concatenate([t.flatten() for t in observation.values()])
        return obs_flattened

class ActionFilterWrapper(gym.ActionWrapper):
    def __init__(self, env: Env, action_filters):
        super().__init__(env)
        self.filters = action_filters
        self.original_action_space = env.action_space
        # self.action_space = spaces.Dict(
        #     {k: v for k, v in env.action_space.spaces.items() if k in self.filters}
        # )
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(len(action_filters),), dtype=np.float32
        )
        print(self.action_space)
    
    def action(self, action):
        action_dict = {k: action[i] for i, k in enumerate(self.filters)}
        full_action = action_dict | {
            k: 0 for k in self.original_action_space.spaces.keys() if k not in self.filters
        }
        return full_action


