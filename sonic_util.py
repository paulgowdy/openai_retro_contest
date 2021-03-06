import gym
import numpy as np

from baselines.common.atari_wrappers import WarpFrame, FrameStack
import gym_remote.client as grc

from retro_contest.local import make

import numpy as np
from collections import deque
import gym
from gym import spaces
import cv2

class myWarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        dim = 64
        self.width = dim
        self.height = dim
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 3), dtype=np.uint8)

    def observation(self, frame):
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        frame = frame[65:130,40:280,:]

        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

        return frame[:, :]

def make_env_multi(game_name, stage, stack=True, scale_rew=True):
    """
    Create an environment with some standard wrappers.
    """
    #env = grc.RemoteEnv('tmp/sock')
    #env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')
    env = make(game=game_name, state=stage)

    env = SonicDiscretizer(env)

    if scale_rew:
        env = RewardScaler(env)

    #env = WarpFrame(env)
    env = myWarpFrame(env)

    if stack:
        env = FrameStack(env, 4)
    return env

def make_env(stack=True, scale_rew=True):
    """
    Create an environment with some standard wrappers.
    """
    #env = grc.RemoteEnv('tmp/sock')
    env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')

    env = SonicDiscretizer(env)

    if scale_rew:
        env = RewardScaler(env)
    env = myWarpFrame(env)
    if stack:
        env = FrameStack(env, 4)
    return env

class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()

class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.
    This is incredibly important and effects performance
    drastically.
    """
    def reward(self, reward):
        return reward * 0.01

class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """
    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs): # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info
