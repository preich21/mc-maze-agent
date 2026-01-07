import enum
import logging
from typing import Tuple

import gymnasium as gym

from mc_env.action import MinecraftAction, MinecraftActionVector
from mc_env.observation import MinecraftObservation
from ws import MinecraftWsBridge

LOGGER = logging.getLogger(__name__)

class BlockTypes(enum.IntEnum):
    """Matches Java enum: AIR(0), START_BLOCK(2), BLOCK(1), GOAL_BLOCK(3)."""

    AIR = 0
    BLOCK = 1
    START_BLOCK = 2
    GOAL_BLOCK = 3

SOLID_BLOCKS = {BlockTypes.BLOCK, BlockTypes.START_BLOCK, BlockTypes.GOAL_BLOCK}

class MinecraftEnv(gym.Env[MinecraftObservation, MinecraftActionVector]):
    metadata = {"render_modes": []}

    def __init__(self, uri: str = 'ws://127.0.0.1:8081',
                 step_ticks: int = 2,
                 yaw_delta_max_deg: float = 90.0,
                 pitch_delta_max_deg: float = 70.0):
        super().__init__()

        self._ws = MinecraftWsBridge(uri)

        self.step_ticks = step_ticks
        self.yaw_delta_max_deg = yaw_delta_max_deg
        self.pitch_delta_max_deg = pitch_delta_max_deg
        self.episode = 0
        self.step_idx = 0

        # real obs space is defined in wrapper
        self.observation_space = gym.spaces.Space()
        self.action_space = MinecraftActionVector.get_space(self)

    def reset(self, seed=None, options=None) -> Tuple[MinecraftObservation, dict]:
        super().reset(seed=seed)
        self.episode += 1
        self.step_idx = 0

        result = self._ws.reset(episode=self.episode, seed=seed, options=options)
        obs = MinecraftObservation.from_message(result.observation)
        info = result.info
        return obs, info

    def step(self, action: MinecraftActionVector) -> Tuple[MinecraftObservation, float, bool, bool, dict]:
        self.step_idx += 1

        parsed_action = MinecraftAction.from_vector(action, env=self)

        # Send via WS using existing messages.ActionRequest shape
        result = self._ws.step(parsed_action)

        obs = MinecraftObservation.from_message(result.observation)
        reward = 0.0
        terminated = result.raw.get("terminated", False)
        truncated = result.raw.get("truncated", False)
        info = result.info
        return obs, reward, terminated, truncated, info

    def close(self):
        self._ws.close()
        super().close()
