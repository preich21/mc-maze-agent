from dataclasses import dataclass
from typing import List, Mapping, Any

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
    standingOn: int
    fovDistances: List[float]
    fovBlocks: List[int]

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
        standing = int(standing_raw)

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
            fovBlocks=[int(v) for v in fov_blocks],
        )