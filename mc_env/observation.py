from dataclasses import dataclass
from typing import List, Any, Dict

import numpy as np

from mc_env.action import MinecraftAction
from ws.messages import IncomingMessage, get_or_throw

@dataclass
class MinecraftObservation(IncomingMessage):
    tick: int
    actionStartedTick: int | None
    lastActions: np.ndarray | None
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    died: bool
    standingOn: int
    fovDistances: List[float]
    fovBlocks: List[int]

    @staticmethod
    def from_message(message: Dict[str, Any]) -> "MinecraftObservation":
        if not isinstance(message, dict):
            raise ValueError("message must be a dict")

        fov_dist = list(get_or_throw(message, "fovDistances"))
        fov_blocks = list(get_or_throw(message, "fovBlocks"))
        from mc_env.env import FOV_RAYS
        if len(fov_dist) != FOV_RAYS or len(fov_blocks) != FOV_RAYS:
            raise ValueError("fovDistances and fovBlocks must have length " + FOV_RAYS)

        standing_raw = get_or_throw(message, "standingOn")
        standing = int(standing_raw)

        action_started_tick = message.get("actionStartedTick")
        if action_started_tick is not None:
            action_started_tick = int(action_started_tick)

        return MinecraftObservation(
            tick=int(get_or_throw(message, "tick")),
            actionStartedTick=action_started_tick,
            lastActions=None,
            x=float(get_or_throw(message, "x")),
            y=float(get_or_throw(message, "y")),
            z=float(get_or_throw(message, "z")),
            yaw=float(get_or_throw(message, "yaw")),
            pitch=float(get_or_throw(message, "pitch")),
            died=bool(get_or_throw(message, "died")),
            standingOn=standing,
            fovDistances=[float(v) for v in fov_dist],
            fovBlocks=[int(v) for v in fov_blocks],
        )