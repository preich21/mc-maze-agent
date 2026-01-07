"""Lightweight message models for the Minecraft WebSocket protocol."""
from __future__ import annotations

from abc import abstractmethod, ABC
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

class OutgoingMessage(ABC):
    @abstractmethod
    def to_message(self) -> Dict[str, Any]:
        pass
