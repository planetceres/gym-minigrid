import os
import numpy
import gym
from gym import spaces

from processor import DependencyParser

try:
    import gym_minigrid
    from gym_minigrid.wrappers import *
except:
    pass




def make_env(env_id, seed, rank, log_dir, image_rgb):
    def _thunk():
        env = gym.make(env_id)

        env.seed(seed + rank)

        def image_size(function):
            def wrapper(*args, **kwargs):
                return function(*args, **kwargs)
            return wrapper

        @image_size
        def get_size():
            imgSpace = env.observation_space.spaces['image']
            return reduce(operator.mul, imgSpace.shape, 1)

        env.imgSize = image_size(get_size)



        # Maxime: until RL code supports dict observations, squash observations into a flat vector
        if isinstance(env.observation_space, spaces.Dict):
            env = DependencyParser(env)
            if image_rgb:
                env = FlatObsWrapperRGB(env)
            else:
                #TODO: create option for language embedding type
                #env = FlatObsWrapper(env)
                #env = FlatObsWrapperEmbed(env)
                #env = FlatObsWrapperCharEmbed(env)
                env = FlatObsWrapperCESim(env)


        return env

    return _thunk
