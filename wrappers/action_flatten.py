"""Flatten Dict action space into a single Box for SB3 compatibility."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from types import MinecraftAction


class ActionFlattenWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, look_low: float = -10.0, look_high: float = 10.0):
        super().__init__(env)
        self._look_low = look_low
        self._look_high = look_high
        # move(5 binaries) + look(2 floats)
        low = np.array([0, 0, 0, 0, 0, look_low, look_low], dtype=np.float32)
        high = np.array([1, 1, 1, 1, 1, look_high, look_high], dtype=np.float32)
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def action(self, action):  # noqa: ANN001
        # Expect action as vector len 7
        move = np.array(action[:5] >= 0.5, dtype=np.int8)
        look = np.array(action[5:], dtype=np.float32)
        return MinecraftAction(move=move, look=look)
