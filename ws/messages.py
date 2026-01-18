"""Lightweight message models for the Minecraft WebSocket protocol."""
from __future__ import annotations

from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict

from mc_env.start_points import StartPoint

class OutgoingMessageType(str, Enum):
    RESET_REQUEST = "RESET_REQUEST"
    ACTION_REQUEST = "ACTION_REQUEST"


class IncomingMessageType(str, Enum):
    OBSERVATION = "OBSERVATION"
    ERROR = "ERROR"
    HELLO = "HELLO"
    UNKNOWN = "UNKNOWN"

class OutgoingMessage(ABC):
    @abstractmethod
    def to_message(self) -> Dict[str, Any]:
        pass

class IncomingMessage(ABC):
    @staticmethod
    @abstractmethod
    def from_message(message: Dict[str, Any]) -> IncomingMessage:
        pass

def get_or_throw(message: Dict[str, Any], key: str) -> Any:
    if key not in message:
        raise ValueError(f"Missing observation field: {key}")
    return message[key]

@dataclass
class HelloMessage(IncomingMessage):
    start_points: list[StartPoint]

    @staticmethod
    def from_message(message: Dict[str, Any]) -> "HelloMessage":
        points = get_or_throw(message, "startPoints")
        return HelloMessage(
            start_points=[
                StartPoint(
                    id=str(get_or_throw(point, "id")),
                    weight=float(get_or_throw(point, "weight")),
                    x=int(get_or_throw(point, "x")),
                    y=int(get_or_throw(point, "y")),
                    z=int(get_or_throw(point, "z")),
                    yaw=float(get_or_throw(point, "yaw")),
                    pitch=float(get_or_throw(point, "pitch")),
                )
                for point in points
            ]
        )