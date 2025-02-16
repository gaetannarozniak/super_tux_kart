import torch
import gymnasium as gym
from gymnasium.core import Env, spaces
import numpy as np


class ObsFilterWrapper(gym.ObservationWrapper):
    def __init__(self, env: Env, obs_filters, action_filters):  # , action_filters):
        super().__init__(env)
        self.obs_filters = obs_filters
        self.last_action = {
            action_filters[i]: np.zeros((1,)) for i in range(len(action_filters))
        }
        obs_space_dict = {
            k: env.observation_space[k]
            for k in env.observation_space.keys()
            if k in obs_filters
        } | {
            k: spaces.Box(low=-np.inf, high=np.inf, shape=(1,))
            for k in self.last_action.keys()
        }
        self.observation_space = spaces.Dict(obs_space_dict)
        print("ObsFilter")

    def observation(self, observation):
        obs_filtered = {
            k: v for k, v in observation.items() if k in self.obs_filters
        } | self.last_action
        return obs_filtered


class RewardCenterTrackWrapper(gym.Wrapper):
    def __init__(self, env: Env, scale):
        super().__init__(env)
        self.scale = scale

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        distance_center_loss = self.scale * np.abs(obs["center_path"][2])
        reward += -distance_center_loss
        return obs, reward, done, truncated, info


class RewardSmoothWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, scale=0.1):
        super().__init__(env)
        self.scale = scale
        self.prev_velocity = None

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        current_velocity = obs["velocity"]

        if self.prev_velocity is not None:
            dv = current_velocity - self.prev_velocity
            if obs["distance_down_track"] > 10:
                reward -= self.scale * np.linalg.norm(dv)
        self.prev_velocity = current_velocity
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_velocity = obs["velocity"]
        return obs, info


class RewardStartWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, scale=1):
        super().__init__(env)
        self.scale = scale
        self.step_count = 0

    def reset(self, **kwargs):
        self.step_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.step_count += 1

        reward_start_speed = np.linalg.norm(obs["velocity"])
        reward += reward_start_speed

        if self.step_count < 20:
            track_deviation = np.linalg.norm(obs["center_path"])
            reward -= 2 * track_deviation

        exit_direction = obs["paths_end"][0] - obs["front"]
        steering_alignment = np.dot(obs["velocity"], exit_direction)
        reward += steering_alignment

        if self.step_count < 50:
            if np.linalg.norm(obs["velocity"]) > 50 and track_deviation < 0.5:
                reward += 5

        return obs, reward, done, truncated, info


class RewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, scale=0.1):
        super().__init__(env)
        self.scale = scale

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        speed = np.linalg.norm(obs["velocity"])
        current_path_width = obs["paths_width"][0]
        reward -= (self.scale / current_path_width) * obs["center_path_distance"] ** 2
        reward += self.scale * speed * current_path_width
        if obs["center_path_distance"] > obs["paths_width"][0]:
            reward -= 10.0

        return obs, reward, done, truncated, info


class StuckResetWrapper(gym.Wrapper):
    def __init__(self, env, stuck_time=2, speed_threshold=0.1):
        super().__init__(env)
        self.stuck_time = stuck_time
        self.speed_threshold = speed_threshold
        self.time_stuck = 0

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        velocity = obs.get("velocity")  # Default to 1.0 if speed is not in obs
        speed = np.linalg.norm(velocity)

        if speed < self.speed_threshold:
            self.time_stuck += 1
        else:
            self.time_stuck = 0  # Reset if moving again

        if self.time_stuck >= self.stuck_time * self.env.metadata.get(
            "video.frames_per_second", 30
        ):
            print("Agent stuck, repositioning...")
            obs = self.reset_agent_position()
            self.time_stuck = 0

        return obs, reward, done, truncated, info

    def reset_agent_position(self):
        obs, info = self.env.reset()
        return obs


class FlattenObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        flat_size = np.sum(
            np.prod(space.shape) for space in env.observation_space.values()
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_size,), dtype=np.float32
        )
        print("FlattenObservation")

    def observation(self, observation):
        obs_flattened = np.concatenate([t.flatten() for t in observation.values()])
        return obs_flattened


class ActionTrackerWrapper(gym.Wrapper):
    def __init__(self, env, action_filters):
        super().__init__(env)
        self.filters = action_filters

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.env.last_action = {
            self.filters[i]: np.array([action[i]]) for i in range(len(self.filters))
        }
        return obs, reward, done, truncated, info


class ActionFilterWrapper(gym.ActionWrapper):
    def __init__(self, env: Env, action_filters):
        super().__init__(env)
        self.filters = action_filters
        self.original_action_space = env.action_space
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(len(action_filters),), dtype=np.float32
        )
        print("ActionFilter")

    def action(self, action):
        action = {k: action[i] for i, k in enumerate(self.filters)}
        full_action = action | {
            k: 0
            for k in self.original_action_space.spaces.keys()
            if k not in self.filters
        }
        full_action['fire'] = 1
        return full_action
