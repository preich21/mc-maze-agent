import gymnasium as gym
import numpy as np
import logging

from ws import MinecraftWsBridge

LOGGER = logging.getLogger(__name__)

class MinecraftEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, uri: str = 'ws://127.0.0.1:8081', step_ticks: int = 5):
        super().__init__()

        self._ws = MinecraftWsBridge(uri)

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

        result = self._ws.reset(episode=self.episode, seed=seed, options=options)
        obs = self._parse_obs(result.observation)
        info = result.info
        return obs, info

    def step(self, action):
        self.step_idx += 1
        move = action["move"].astype(int).tolist()
        look = action["look"].astype(float).tolist()

        result = self._ws.step(
            episode=self.episode,
            step=self.step_idx,
            ticks=self.step_ticks,
            move=move,
            look=look,
        )
        obs = self._parse_obs(result.observation)
        reward = 0.0
        terminated = result.raw.get("terminated", False)
        truncated = result.raw.get("truncated", False)
        info = result.info
        return obs, reward, terminated, truncated, info

    def close(self):
        self._ws.close()
        super().close()

    def _parse_obs(self, observation):
        if not isinstance(observation, dict):
            return {}
        obs = {
            "episode": observation.get("episode", self.episode),
            "step": observation.get("step", self.step_idx),
            "tick_start": observation.get("tickStart"),
            "tick_end": observation.get("tickEnd"),
            "position": {
                "x": observation.get("x"),
                "y": observation.get("y"),
                "z": observation.get("z"),
            },
            "rotation": {
                "yaw": observation.get("yaw"),
                "pitch": observation.get("pitch"),
            },
            "standing_on": observation.get("standingOn"),
            "field_of_view": observation.get("fieldOfView", []),
            "died": bool(observation.get("died", False)),
        }
        return obs
