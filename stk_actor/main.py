import torch
from gymnasium.core import Env
from pystk2_gymnasium import AgentSpec
import gymnasium as gym
from sac_torch import AgentSac

from my_wrappers import (ObsFilterWrapper, RewardCenterTrackWrapper,
                         FlattenObservationWrapper, ActionFilterWrapper)


obs_filters = ['center_path']#, 'paths_start', 'paths_end', 'paths_width']#, 'front',
                #'paths_distance', 'paths_width']
action_filters = ['acceleration', 'steer']

if __name__ == "__main__":

    player_name = "smail_gaetan_kart"
    env_name = "supertuxkart/simple-v0"
    n_envs = 1
    n_steps = 300
    n_epochs = 20 
    scale_distance_center = 0.2

    env = gym.make(env_name, render_mode='human',
                    agent=AgentSpec(use_ai=False, name=player_name))
    env = ObsFilterWrapper(env, obs_filters)
    env = RewardCenterTrackWrapper(env, scale_distance_center)
    env = FlattenObservationWrapper(env)
    env = ActionFilterWrapper(env, action_filters)

    agent = AgentSac(input_dims=env.observation_space.shape, env=env,
                     n_actions=env.action_space.shape[0])
    
    print(f"{env.observation_space = } | {env.action_space = }")

    # try:
    #     agent.load_models()
    # except Exception as e:
    #     pass

    for e in range(n_epochs):  
        obs = env.reset()[0]   
        for t in range(n_steps):
            action = agent.choose_action(obs)
            new_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.remember(obs, action, reward, new_obs, done)
            agent.learn()
            obs = new_obs
            if t%50 == 0:
                print(reward, action, obs)

    env.close()
    agent.save_models()
    