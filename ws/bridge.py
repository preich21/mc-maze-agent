"""Synchronous bridge around the async WebSocket client."""
from __future__ import annotations

import asyncio
import contextlib
import threading

from .client import WebSocketClient
from .messages import IncomingMessageType, OutgoingMessage


class MinecraftWsBridge:
    """Runs the async WebSocket client in a private event loop."""

    def __init__(self, uri: str, connect_timeout: float = 5.0):
        self._uri = uri
        self._timeout = connect_timeout
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._client = WebSocketClient(uri, connect_timeout)
        self._started = threading.Event()
        self._closing = False
        self._thread.start()
        self._started.wait()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._started.set()
        self._loop.run_forever()

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        if self._loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self._client.close(), self._loop)
            with contextlib.suppress(Exception):
                future.result(timeout=self._timeout)
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=1)

    def _shutdown(self) -> None:
        pass

    def send(self, request: OutgoingMessage, response_message_type: IncomingMessageType):
        future = asyncio.run_coroutine_threadsafe(self._client.send(request, response_message_type), self._loop)
        try:
            return future.result(timeout=self._timeout)
        except Exception:
            last_frame = self._client.last_frame
            if last_frame is not None:
                raise RuntimeError(f"WebSocket call timed out; last frame: {last_frame}")
            raise

    def __del__(self) -> None:  # pragma: no cover
        self.close()
