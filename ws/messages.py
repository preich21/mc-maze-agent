"""Lightweight message models for the Minecraft WebSocket protocol."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence


class OutgoingMessageType(str, Enum):
    RESET_REQUEST = "RESET_REQUEST"
    ACTION_REQUEST = "ACTION_REQUEST"


class IncomingMessageType(str, Enum):
    STATE_AFTER_RESET = "STATE_AFTER_RESET"
    STATE_AFTER_ACTION = "STATE_AFTER_ACTION"
    ERROR = "ERROR"
    HELLO = "hello"
    UNKNOWN = "UNKNOWN"


@dataclass(slots=True)
class ResetRequest:
    episode: int
    seed: Optional[int] = None
    options: Optional[Mapping[str, Any]] = None

    def to_message(self) -> Dict[str, Any]:
        message: Dict[str, Any] = {"type": OutgoingMessageType.RESET_REQUEST.value, "episode": self.episode}
        if self.seed is not None:
            message["seed"] = int(self.seed)
        if self.options:
            message["options"] = dict(self.options)
        return message


@dataclass(slots=True)
class ActionRequest:
    episode: int
    step: int
    apply_for_ticks: int
    move: Sequence[int]
    look: Sequence[float]

    def to_message(self) -> Dict[str, Any]:
        if len(self.move) != 5:
            raise ValueError("move action must contain 5 elements (fwd, back, left, right, jump)")
        if len(self.look) != 2:
            raise ValueError("look action must contain 2 elements (yaw, pitch)")

        move_forward = float(int(self.move[0]) - int(self.move[1]))
        move_sidewards = float(int(self.move[3]) - int(self.move[2]))
        jump = bool(self.move[4])
        yaw_delta = float(self.look[0])
        pitch_delta = float(self.look[1])

        return {
            "type": OutgoingMessageType.ACTION_REQUEST.value,
            "episode": self.episode,
            "step": self.step,
            "applyForTicks": self.apply_for_ticks,
            "moveForward": move_forward,
            "moveSidewards": move_sidewards,
            "jump": jump,
            "yawDelta": yaw_delta,
            "pitchDelta": pitch_delta,
        }


@dataclass(slots=True)
class ObservationResult:
    source: IncomingMessageType
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]
    raw: MutableMapping[str, Any]

    @staticmethod
    def from_message(message: MutableMapping[str, Any]) -> "ObservationResult":
        msg_type = message.get("type")
        try:
            source = IncomingMessageType(msg_type)
        except ValueError:
            source = IncomingMessageType.UNKNOWN

        observation: MutableMapping[str, Any] = dict(message)
        reward = float(message.get("reward", 0.0))
        terminated = bool(message.get("terminated", False))
        truncated = bool(message.get("truncated", False))
        info = message.get("info") or {}

        return ObservationResult(
            source=source,
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=dict(info),
            raw=message,
        )
