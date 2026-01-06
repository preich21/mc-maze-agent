import enum
import logging
from dataclasses import dataclass
from typing import Tuple, Mapping, Any, List

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from ws import MinecraftWsBridge

LOGGER = logging.getLogger(__name__)

@dataclass
class MinecraftAction:
    move: NDArray[np.int_]
    look: NDArray[np.floating]

class BlockTypes(enum.IntEnum):
    """Matches Java enum: AIR(0), START_BLOCK(2), BLOCK(1), GOAL_BLOCK(3)."""

    AIR = 0
    BLOCK = 1
    START_BLOCK = 2
    GOAL_BLOCK = 3


SOLID_BLOCKS = {BlockTypes.BLOCK, BlockTypes.START_BLOCK, BlockTypes.GOAL_BLOCK}


@dataclass
class MinecraftObservation:
    episode: int
    step: int
    tickStart: int
    tickEnd: int
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    died: bool
    standingOn: BlockTypes
    fovDistances: List[float]
    fovBlocks: List[BlockTypes]

    @staticmethod
    def from_message(message: Mapping[str, Any]) -> "MinecraftObservation":
        def get_or_throw(key: str) -> Any:
            if key not in message:
                raise ValueError(f"Missing observation field: {key}")
            return message[key]

        if not isinstance(message, dict):
            raise ValueError("message must be a dict")

        # Drop protocol type if present.
        message = {k: v for k, v in message.items() if k != "type"}

        fov_dist = list(get_or_throw("fovDistances"))
        fov_blocks = list(get_or_throw("fovBlocks"))
        if len(fov_dist) != 2500 or len(fov_blocks) != 2500:
            raise ValueError("fovDistances and fovBlocks must have length 2500")

        standing_raw = get_or_throw("standingOn")
        standing = BlockTypes(int(standing_raw)) if isinstance(standing_raw, (int, float)) else BlockTypes[str(standing_raw)]

        return MinecraftObservation(
            episode=int(get_or_throw("episode")),
            step=int(get_or_throw("step")),
            tickStart=int(get_or_throw("tickStart")),
            tickEnd=int(get_or_throw("tickEnd")),
            x=float(get_or_throw("x")),
            y=float(get_or_throw("y")),
            z=float(get_or_throw("z")),
            yaw=float(get_or_throw("yaw")),
            pitch=float(get_or_throw("pitch")),
            died=bool(get_or_throw("died")),
            standingOn=standing,
            fovDistances=[float(v) for v in fov_dist],
            fovBlocks=[BlockTypes(int(v)) for v in fov_blocks],
        )

class MinecraftEnv(gym.Env[MinecraftObservation, MinecraftAction]):
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

    def reset(self, seed=None, options=None) -> Tuple[MinecraftObservation, dict]:
        super().reset(seed=seed)
        self.episode += 1
        self.step_idx = 0

        result = self._ws.reset(episode=self.episode, seed=seed, options=options)
        obs = MinecraftObservation.from_message(result.observation)
        info = result.info
        return obs, info

    def step(self, action: MinecraftAction) -> Tuple[MinecraftObservation, float, bool, bool, dict]:
        self.step_idx += 1
        move = action.move.astype(int).tolist()
        look = action.look.astype(float).tolist()

        result = self._ws.step(
            episode=self.episode,
            step=self.step_idx,
            ticks=self.step_ticks,
            move=move,
            look=look,
        )
        obs = MinecraftObservation.from_message(result.observation)
        reward = 0.0
        terminated = result.raw.get("terminated", False)
        truncated = result.raw.get("truncated", False)
        info = result.info
        return obs, reward, terminated, truncated, info

    def close(self):
        self._ws.close()
        super().close()
