from typing import List, Callable
from bbrl.agents import Agents, Agent
import gymnasium as gym

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

# Imports our Actor class
# IMPORTANT: note the relative import
from .actors import ActorSac

#: The base environment name
env_name = "supertuxkart/simple-v0"

#: Player name
player_name = "smail_gaetan_kart"


def get_actor(
    state, observation_space: gym.spaces.Space, action_space: gym.spaces.Space
) -> Agent:
    actor = ActorSac(observation_space, action_space)
    actor.load_state_dict(state)
    return Agents(actor)


def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
    """Returns a list of additional wrappers to be applied to the base
    environment"""
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
    scale_distance_center = 0.05
    return [
        lambda env: ActionFilterWrapper(env, action_filters),
        lambda env: ObsFilterWrapper(env, obs_filters, action_filters),
    ]
