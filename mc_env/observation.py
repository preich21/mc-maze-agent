from dataclasses import dataclass
from typing import List, Any, Dict

from mc_env.action import MinecraftAction
from ws.messages import IncomingMessage, get_or_throw

@dataclass
class MinecraftObservation(IncomingMessage):
    tick: int
    actionStartedTick: int | None
    activeActionRequest: MinecraftAction | None
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
            raise ValueError("fovDistances and fovBlocks must have length 2500")

        standing_raw = get_or_throw(message, "standingOn")
        standing = int(standing_raw)

        action_started_tick = message.get("actionStartedTick")
        if action_started_tick is not None:
            action_started_tick = int(action_started_tick)
        active_action_request = message.get("activeActionRequest")
        if active_action_request is not None:
            active_action_request = MinecraftAction(
                moveForward=bool(get_or_throw(active_action_request, "moveForward")),
                moveBackward=bool(get_or_throw(active_action_request, "moveBackward")),
                moveLeft=bool(get_or_throw(active_action_request, "moveLeft")),
                moveRight=bool(get_or_throw(active_action_request, "moveRight")),
                jump=bool(get_or_throw(active_action_request, "jump")),
                yawDelta=float(get_or_throw(active_action_request, "yawDelta")),
                pitchDelta=float(get_or_throw(active_action_request, "pitchDelta")),
            )


        return MinecraftObservation(
            tick=int(get_or_throw(message, "tick")),
            actionStartedTick=action_started_tick,
            activeActionRequest=active_action_request,
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