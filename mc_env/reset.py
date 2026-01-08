from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from ws.messages import OutgoingMessageType, OutgoingMessage


@dataclass(slots=True)
class ResetRequest(OutgoingMessage):
    episode: int
    start_point_nonce: int
    seed: Optional[int] = None
    options: Optional[Mapping[str, Any]] = None

    def to_message(self) -> Dict[str, Any]:
        message: Dict[str, Any] = {"type": OutgoingMessageType.RESET_REQUEST.value, "episode": self.episode, "startPointNonce": self.start_point_nonce}
        if self.seed is not None:
            message["seed"] = int(self.seed)
        if self.options:
            message["options"] = dict(self.options)
        return message