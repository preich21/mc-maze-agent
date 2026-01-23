import enum
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from mc_env.start_points import StartPoint
from ws.messages import OutgoingMessageType, OutgoingMessage

class StartPointRotation(enum.StrEnum):
    KEEP_POSITION = "KEEP_POSITION"
    ROTATE_90 = "ROTATE_90"
    ROTATE_180 = "ROTATE_180"
    ROTATE_270 = "ROTATE_270"

StartPointRotations = [StartPointRotation.KEEP_POSITION, StartPointRotation.ROTATE_90, StartPointRotation.ROTATE_180, StartPointRotation.ROTATE_270]

@dataclass(slots=True)
class ResetRequest(OutgoingMessage):
    episode: int
    start_point: StartPoint | None
    startPointRotation: StartPointRotation | None
    seed: Optional[int] = None
    options: Optional[Mapping[str, Any]] = None

    def to_message(self) -> Dict[str, Any]:
        message: Dict[str, Any] = {"type": OutgoingMessageType.RESET_REQUEST.value, "episode": self.episode, "startPoint": self.start_point.to_json(), "startPointRotation": self.startPointRotation}
        if self.seed is not None:
            message["seed"] = int(self.seed)
        if self.options:
            message["options"] = dict(self.options)
        return message