import struct
from dataclasses import dataclass
from typing import List, Any, Dict

import numpy as np

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
    has_ground_below: bool
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
            has_ground_below=bool(get_or_throw(message, "hasGroundBelow")),
            fovDistances=[float(v) for v in fov_dist],
            fovBlocks=[int(v) for v in fov_blocks],
        )

    @staticmethod
    def half_to_float(h: int) -> float:
        """Converts 16-bit float (half) in 32-bit float."""
        s = int((h >> 15) & 0x00000001)  # sign
        e = int((h >> 10) & 0x0000001f)  # exponent
        f = int(h & 0x03ff)  # fraction

        if e == 0:
            if f == 0:
                return float((-1) ** s * 0.0)
            else:
                # subnormal number
                return (-1) ** s * 2 ** (-14) * (f / 1024.0)
        elif e == 31:
            return float('inf') if f == 0 else float('nan')
        else:
            return (-1) ** s * 2 ** (e - 15) * (1 + f / 1024.0)

    @staticmethod
    def from_bytes_message(data: bytes) -> "MinecraftObservation":
        offset = 0
        # Long (8 bytes, big endian)
        tick, = struct.unpack_from(">Q", data, offset)
        offset += 8
        actionStartedTick, = struct.unpack_from(">Q", data, offset)
        offset += 8

        # Int (4 bytes each)
        x, y, z = struct.unpack_from(">iii", data, offset)
        offset += 12

        # Float (4 bytes each)
        yaw, pitch = struct.unpack_from(">ff", data, offset)
        offset += 8

        # Boolean / byte (died, hasGroundBelow, standingOn)
        died = bool(data[offset])
        offset += 1
        has_ground_below = bool(data[offset])
        offset += 1
        standingOn = data[offset]
        offset += 1

        # fovDistances: float16 -> float32
        from mc_env.env import FOV_RAYS
        fovDistances = []
        for _ in range(FOV_RAYS):
            half_val, = struct.unpack_from(">H", data, offset)
            offset += 2
            fovDistances.append(MinecraftObservation.half_to_float(half_val))

        fovBlocks = list(data[offset:offset + FOV_RAYS])
        offset += FOV_RAYS

        return MinecraftObservation(
            tick=tick,
            actionStartedTick=actionStartedTick if actionStartedTick != 0 else None,
            lastActions=None,
            x=x, y=y, z=z,
            yaw=yaw,
            pitch=pitch,
            died=died,
            has_ground_below=has_ground_below,
            standingOn=standingOn,
            fovDistances=fovDistances,
            fovBlocks=fovBlocks
        )
