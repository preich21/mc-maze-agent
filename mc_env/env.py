import enum
import logging
from typing import Tuple

import gymnasium as gym
import numpy as np

from mc_env.action import MinecraftAction
from mc_env.observation import MinecraftObservation
from mc_env.reset import ResetRequest
from ws.bridge import MinecraftWsBridge
from ws.messages import IncomingMessageType

LOGGER = logging.getLogger(__name__)

class BlockTypes(enum.IntEnum):
    """Matches Java enum: AIR(0), START_BLOCK(2), BLOCK(1), GOAL_BLOCK(3)."""

    AIR = 0
    BLOCK = 1
    START_BLOCK = 2
    GOAL_BLOCK = 3

SOLID_BLOCKS = {BlockTypes.BLOCK, BlockTypes.START_BLOCK, BlockTypes.GOAL_BLOCK}

FOV_RAYS = 50 * 50

class MinecraftEnv(gym.Env[MinecraftObservation, np.ndarray]):
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
        self.action_space = MinecraftAction.get_space(self)

    def reset(self, seed=None, options=None) -> Tuple[MinecraftObservation, dict]:
        super().reset(seed=seed)
        self.episode += 1
        self.step_idx = 0


        start_point_nonce = int(self.np_random.integers(0, 2**32, dtype=np.uint32))

        request = ResetRequest(episode=self.episode, start_point_nonce=start_point_nonce, seed=seed, options=options)

        obs = self._ws.send(request, IncomingMessageType.STATE_AFTER_RESET)
        info = {"start_point_nonce": start_point_nonce}
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[MinecraftObservation, float, bool, bool, dict]:
        self.step_idx += 1

        parsed_action = MinecraftAction.from_vector(action, env=self)

        obs = self._ws.send(parsed_action, IncomingMessageType.STATE_AFTER_ACTION)

        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def close(self):
        self._ws.close()
        super().close()
