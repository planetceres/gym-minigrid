import os
import numpy
import gym
from gym import spaces

try:
    import gym_minigrid
    from gym_minigrid.wrappers import *
except:
    pass

def make_env(env_id, seed, rank, log_dir, image_rgb):
    def _thunk():
        env = gym.make(env_id)

        env.seed(seed + rank)

        # set preference for state space in rgb vs compact encodings
        env.image_rgb = image_rgb

        # Maxime: until RL code supports dict observations, squash observations into a flat vector
        if isinstance(env.observation_space, spaces.Dict):
            env = FlatObsWrapper(env)

        return env

    return _thunk
