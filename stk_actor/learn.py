from tqdm import tqdm
import gymnasium as gym
import numpy as np
from pathlib import Path
from pystk2_gymnasium import AgentSpec
from functools import partial
import torch
import inspect
from bbrl.agents.gymnasium import ParallelGymAgent, make_env
from bbrl.agents import Agents, TemporalAgent
from bbrl.workspace import Workspace
from utils import plot_rewards

from .my_wrappers import (
    ObsFilterWrapper,
    RewardCenterTrackWrapper,
    FlattenObservationWrapper,
    ActionFilterWrapper,
    ActionTrackerWrapper,
    StuckResetWrapper,
    RewardWrapper,
    RewardSmoothWrapper,
)


# Note the use of relative imports
from .actors import ActorSac
from .pystk_actor import env_name, get_wrappers, player_name


n_steps = 800
n_epochs = 1000
scale_distance_center = 0.05
rewards = np.zeros((n_epochs, n_steps))
obs_filters = [
    "distance_down_track",
    "front",
    "max_steer_angle",
    "center_path",
    "center_path_distance",
    "paths_start",
    "paths_end",
    "paths_width",
    "paths_distance",
    "velocity",
    "acceleration",
    "steer",
    "sliding_velocity",
    "items_position",
    "items_type",
]
action_filters = ["acceleration", "steer", "drift", "nitro"]


if __name__ == "__main__":
    # Setup the environment
    env = gym.make(
        env_name, render_mode="human", agent=AgentSpec(use_ai=False, name=player_name)
    )
    wrappers = get_wrappers()

    env = ActionFilterWrapper(env, action_filters)
    env = ObsFilterWrapper(env, obs_filters, action_filters)
    env = StuckResetWrapper(env, speed_threshold=0.5, stuck_time=1)
    env = RewardCenterTrackWrapper(env, scale_distance_center)
    env = RewardWrapper(env)
    env = RewardSmoothWrapper(env)

    actor = ActorSac(obs_space=env.observation_space, action_space=env.action_space)
    rewards = np.zeros((n_epochs, n_steps))

    # (2) Learn

    for e in tqdm(range(n_epochs)):
        obs = env.reset()[0]
        episode_reward = 0  # Track total episode reward
        for t in range(n_steps):
            action = actor.agent.choose_action(obs)
            new_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            actor.agent.remember(obs, action, reward, new_obs, done)

            # Perform a learning step and get loss values
            loss_dict = actor.agent.learn()
            obs = new_obs
            rewards[e][t] = reward
            if done:
                obs = env.reset()[0]  # Reset and start next episode
                break

    plot_rewards(rewards, n_epochs - 1)

    # (3) Save the actor sate
    mod_path = Path(inspect.getfile(get_wrappers)).parent
    torch.save(actor.state_dict(), mod_path / "pystk_actor.pth")
