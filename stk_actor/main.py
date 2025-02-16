import numpy as np
from gymnasium.core import Env
from pystk2_gymnasium import AgentSpec
import gymnasium as gym
from sac_torch import AgentSac

from my_wrappers import (ObsFilterWrapper, RewardCenterTrackWrapper,
                         FlattenObservationWrapper, ActionFilterWrapper,
                         ActionTrackerWrapper)
from utils import plot_rewards


obs_filters = [
    'distance_down_track',
    'front',
    'max_steer_angle',
    'center_path',
    'paths_start',
    'paths_end',
    'paths_width',
    'paths_distance',
    'velocity',
    'acceleration',
    'steer'
]

action_filters = ['acceleration', 'steer']

if __name__ == "__main__":

    player_name = "smail_gaetan_kart"
    env_name = "supertuxkart/simple-v0"
    n_envs = 1
    n_steps = 300
    n_epochs = 10000
    scale_distance_center = 0.05
    rewards = np.zeros((n_epochs, n_steps))
    load_model = False

    env = gym.make(env_name, render_mode='human',
                    agent=AgentSpec(use_ai=False, name=player_name))
    env = ActionFilterWrapper(env, action_filters)
    env = ObsFilterWrapper(env, obs_filters, action_filters)
    env = ActionTrackerWrapper(env, action_filters)
    env = RewardCenterTrackWrapper(env, scale_distance_center)
    env = FlattenObservationWrapper(env)

    agent = AgentSac(input_dims=env.observation_space.shape, env=env,
                     n_actions=env.action_space.shape[0])
    
    print(f"{env.observation_space = } | {env.action_space = }")

    if load_model:
        try:
            agent.load_models()
        except Exception as e:
            pass

    for e in range(n_epochs):  
        obs = env.reset()[0]   
        for t in range(n_steps):
            action = agent.choose_action(obs)
            new_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.remember(obs, action, reward, new_obs, done)
            agent.learn()
            obs = new_obs
            rewards[e][t] = reward
            if t%50 == 0:
                print(reward, action, obs)
        plot_rewards(rewards, e)
        agent.save_models()

    env.close()
    