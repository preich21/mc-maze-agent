from dataclasses import dataclass
from typing import List, Mapping, Any, Optional


@dataclass
class MinecraftObservation:
    episode: int
    step: int
    tickStart: int
    tickEnd: int
    x: float
    y: float
    z: float
    dx: float
    dy: float
    dz: float
    yaw: float
    pitch: float
    died: bool
    standingOn: int
    surroundingBlocks: List[int]
    fovDistances: List[float]
    fovBlocks: List[int]
    maze: Optional[List[List[bool]]]
    maze_distance: Optional[float] = None # filled by maze wrapper

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
        from mc_env.env import FOV_RAYS
        if len(fov_dist) != FOV_RAYS or len(fov_blocks) != FOV_RAYS:
            raise ValueError(f"fovDistances and fovBlocks must have length {FOV_RAYS} but is fov_dist={len(fov_dist)} and fov_blocks={len(fov_blocks)}")

        standing_raw = get_or_throw("standingOn")
        standing = int(standing_raw)

        surrounding_blocks_raw = get_or_throw("surroundingBlocks")
        surrounding_blocks = [int(b) for b in surrounding_blocks_raw]

        maze_raw = list(list(message["maze"])) if "maze" in message else None

        return MinecraftObservation(
            episode=int(get_or_throw("episode")),
            step=int(get_or_throw("step")),
            tickStart=int(get_or_throw("tickStart")),
            tickEnd=int(get_or_throw("tickEnd")),
            x=float(get_or_throw("x")),
            y=float(get_or_throw("y")),
            z=float(get_or_throw("z")),
            dx=float(get_or_throw("dx")),
            dy=float(get_or_throw("dy")),
            dz=float(get_or_throw("dz")),
            yaw=float(get_or_throw("yaw")),
            pitch=float(get_or_throw("pitch")),
            died=bool(get_or_throw("died")),
            standingOn=standing,
            surroundingBlocks=surrounding_blocks,
            fovDistances=[float(v) for v in fov_dist],
            fovBlocks=[int(v) for v in fov_blocks],
            maze=[[bool(cell) for cell in row] for row in maze_raw] if maze_raw else None,
        )