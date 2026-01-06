"""Typed aliases for environment actions and observations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping

import numpy as np
from numpy.typing import NDArray


@dataclass
class MinecraftAction:
    move: NDArray[np.int_]
    look: NDArray[np.floating]

    @staticmethod
    def from_mapping(data: Mapping[str, Any]) -> "MinecraftAction":
        move = np.asarray(data.get("move", []), dtype=np.int_)
        look = np.asarray(data.get("look", []), dtype=np.float32)
        return MinecraftAction(move=move, look=look)


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
    standingOn: str
    fovDistances: List[float]
    fovBlocks: List[int]

    @staticmethod
    def from_message(message: Mapping[str, Any]) -> "MinecraftObservation":
        def get_or_throw(key: str) -> Any:
            if key not in message:
                raise ValueError(f"Missing observation field: {key}")
            return message[key]

        if not isinstance(message, dict):
            raise ValueError(f"message has not datatype dict")
        fov_dist = list(get_or_throw("fovDistances"))
        fov_blocks = list(get_or_throw("fovBlocks"))
        if len(fov_dist) != 25 or len(fov_blocks) != 25:
            raise ValueError("fovDistances and fovBlocks must have length 25")

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
            standingOn=str(get_or_throw("standingOn")),
            fovDistances=fov_dist,
            fovBlocks=[int(v) for v in fov_blocks],
        )
