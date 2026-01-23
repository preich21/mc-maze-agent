"""Async WebSocket client around the minecraft state extractor mod."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional, Callable

import websockets
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed
from websockets.protocol import State

from mc_env.observation import MinecraftObservation
from .messages import IncomingMessageType, OutgoingMessage, HelloMessage, IncomingMessage

LOGGER = logging.getLogger(__name__)


class WsProtocolError(RuntimeError):
    """Raised when the protocol is violated."""


class WebSocketClient:
    def __init__(self, uri: str, on_hello: Callable[[HelloMessage], None], on_message: Callable[[MinecraftObservation], None], connect_timeout: float = 5.0):
        self._uri = uri
        self.on_hello = on_hello
        self.on_message = on_message
        self._connect_timeout = connect_timeout
        self._conn: Optional[ClientConnection] = None
        self._recv_lock = asyncio.Lock()
        self.rec_msg_task = None
        self._send_lock = asyncio.Lock()
        self._connect_lock = asyncio.Lock()
        self._connected = False

    async def __aenter__(self) -> "WebSocketClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        await self.close()

    async def connect(self) -> None:
        async with self._connect_lock:
            if self._conn and self._conn.state is State.OPEN:
                return
            LOGGER.debug("Connecting to %s", self._uri)
            self._conn = await asyncio.wait_for(websockets.connect(self._uri), timeout=self._connect_timeout)
            self.rec_msg_task = asyncio.create_task(self._process_ws_message_async())
            while not self._connected:
                await asyncio.sleep(0.01)

    async def close(self) -> None:
        if self._conn and self._conn.state is State.OPEN:
            self.rec_msg_task.cancel()
            await self._conn.close()
        self._conn = None

    async def send(self, request: OutgoingMessage):
        if not self._conn or self._conn.state is not State.OPEN or not self._connected:
            raise RuntimeError("WebSocket connection expected open but is closed.")
        text = json.dumps(request.to_message())
        async with self._send_lock:
            await self._conn.send(text)

    async def _process_ws_message_async(self):
        if not self._conn or self._conn.state is not State.OPEN:
            raise RuntimeError("WebSocket connection expected open but is closed.")
        while True:
            async with self._recv_lock:
                try:
                    message = await self._conn.recv()
                except ConnectionClosed as exc:  # pragma: no cover - network failure path
                    raise RuntimeError("WebSocket connection closed") from exc
            if isinstance(message, bytes):
                parsed_message = MinecraftObservation.from_bytes_message(message)
                self.on_message(parsed_message)
                continue
            if not isinstance(message, str):
                raise WsProtocolError("Expected text or binary message from server")
            frame = json.loads(message)
            LOGGER.debug("Received frame: %s", frame)
            frame_type = IncomingMessageType(frame.get("type"))

            match frame_type:
                case IncomingMessageType.OBSERVATION:
                    parsed_message = MinecraftObservation.from_message(frame)
                    self.on_message(parsed_message)
                case IncomingMessageType.HELLO.value:
                    LOGGER.debug("Received hello frame: %s", frame)
                    self.on_hello(HelloMessage.from_message(frame))
                    self._connected = True
                case IncomingMessageType.ERROR.value:
                    raise WsProtocolError(frame.get("message", "Server reported an error"))
                case _:
                    LOGGER.error("Unexpected frame type %s", frame_type)
                    raise WsProtocolError("Received unexpected frame type")
