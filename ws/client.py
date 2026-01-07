"""Async WebSocket client around the minecraft state extractor mod."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

import websockets
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed
from websockets.protocol import State

from mc_env.action import MinecraftAction
from mc_env.observation import MinecraftObservation
from mc_env.reset import ResetRequest
from .messages import IncomingMessageType, OutgoingMessage

LOGGER = logging.getLogger(__name__)


class WsProtocolError(RuntimeError):
    """Raised when the protocol is violated."""


class WebSocketClient:
    def __init__(self, uri: str, connect_timeout: float = 5.0):
        self._uri = uri
        self._connect_timeout = connect_timeout
        self._conn: Optional[ClientConnection] = None
        self._recv_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()
        self._connect_lock = asyncio.Lock()
        self._last_frame: Optional[Dict[str, Any]] = None

    async def __aenter__(self) -> "WebSocketClient":
        await self.ensure_connected()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        await self.close()

    async def ensure_connected(self) -> None:
        async with self._connect_lock:
            if self._conn and self._conn.state is State.OPEN:
                return
            LOGGER.debug("Connecting to %s", self._uri)
            self._conn = await asyncio.wait_for(websockets.connect(self._uri), timeout=self._connect_timeout)

    async def close(self) -> None:
        if self._conn and self._conn.state is State.OPEN:
            await self._conn.close()
        self._conn = None

    async def send(self, request: OutgoingMessage, response_message_type: IncomingMessageType) -> MinecraftObservation:
        await self.ensure_connected()
        await self._send_json(request.to_message())
        return await self._wait_for_state(response_message_type)

    async def _wait_for_state(self, expected: IncomingMessageType) -> MinecraftObservation | None:
        while True:
            frame = await self._recv_json()
            self._last_frame = frame
            LOGGER.debug("Received frame: %s", frame)
            frame_type = IncomingMessageType(frame.get("type"))

            match frame_type:
                case IncomingMessageType.STATE_AFTER_ACTION | IncomingMessageType.STATE_AFTER_RESET:
                    result = MinecraftObservation.from_message(frame)
                    if frame_type == expected:
                        return result
                case IncomingMessageType.HELLO.value:
                    LOGGER.debug("Received hello frame: %s", frame)
                    continue
                case IncomingMessageType.ERROR.value:
                    raise WsProtocolError(frame.get("message", "Server reported an error"))
                case _:
                    LOGGER.error("Unexpected frame type %s (expected %s)", frame_type, expected)
                    raise WsProtocolError("Received unexpected frame type")

    async def _send_json(self, data: Dict) -> None:
        if not self._conn or self._conn.state is not State.OPEN:
            await self.ensure_connected()
        text = json.dumps(data)
        async with self._send_lock:
            await self._conn.send(text)

    async def _recv_json(self) -> Dict[str, Any]:
        if not self._conn or self._conn.state is not State.OPEN:
            await self.ensure_connected()
        async with self._recv_lock:
            try:
                raw = await self._conn.recv()
            except ConnectionClosed as exc:  # pragma: no cover - network failure path
                raise RuntimeError("WebSocket connection closed") from exc
        return json.loads(raw)

    @property
    def last_frame(self) -> Optional[Dict[str, Any]]:
        return self._last_frame
