from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from ws.messages import OutgoingMessageType, OutgoingMessage


@dataclass(slots=True)
class ResetRequest(OutgoingMessage):
    episode: int
    seed: Optional[int] = None
    options: Optional[Mapping[str, Any]] = None
    mazeGeneration: bool = False
    mazeSize: Optional[int] = None

    def to_message(self) -> Dict[str, Any]:
        message: Dict[str, Any] = {"type": OutgoingMessageType.RESET_REQUEST.value, "episode": self.episode, "mazeGeneration": self.mazeGeneration}
        if self.seed is not None:
            message["seed"] = int(self.seed)
        if self.options:
            message["options"] = dict(self.options)
        if self.mazeSize is not None:
            message["mazeSize"] = self.mazeSize
        return message