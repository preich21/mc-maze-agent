import gymnasium as gym
import numpy as np


class MinecraftWSClient:
    """
    TODO
    """


class MinecraftEnv(gym.Env):
    """
    TODO
    """
    metadata = {"render_modes": []}

    def __init__(self, uri: str, step_ticks: int = 5):
        super().__init__()

        # TODO: initialize websocket

        self.step_ticks = step_ticks
        self.episode = 0
        self.step_idx = 0

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32 # TODO: adjust this to the actual data
        )

        self.action_space = gym.spaces.Dict({
            "move": gym.spaces.MultiBinary(5), # forward, backward, left, right, jump
            "look": gym.spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32), # TODO: adjust to actual range
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        self.step_idx = 0

        # TODO: send reset request to server

        obs = {}
        info = {}

        return obs, info

    def step(self, action):
        self.step_idx += 1
        move = action["move"].astype(int).tolist()
        look = action["look"].astype(float).tolist()

        # TODO: send action to server and await response

        obs = {}
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info
