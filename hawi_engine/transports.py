"""Transport implementations for the Hawi core JSON protocol."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
import uuid
from abc import ABC, abstractmethod
from typing import Any

from .protocol import json_dumps, make_error
from .tlv import TYPE_JSON_FRAME, encode_frame

logger = logging.getLogger(__name__)


class QueuedJsonClient(ABC):
    """Base client with a bounded outbound queue."""

    def __init__(self, *, queue_max: int = 100, client_id: str | None = None) -> None:
        self.id = client_id or uuid.uuid4().hex[:12]
        self.authenticated = False
        self.negotiated_caps: set[str] = set()
        self._outbound: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(
            maxsize=queue_max
        )
        self._writer_task: asyncio.Task | None = None
        self._closed = False
        self._close_after_drain = False

    async def start(self) -> None:
        self._writer_task = asyncio.create_task(self._writer_loop())

    async def send(self, frame: dict[str, Any]) -> None:
        if self._closed:
            return
        try:
            self._outbound.put_nowait(frame)
        except asyncio.QueueFull:
            self._close_after_drain = True
            self._drop_pending_frames()
            with contextlib.suppress(asyncio.QueueFull):
                self._outbound.put_nowait(
                    make_error(
                        "Client outbound queue overflowed; closing connection.",
                        code="client_backpressure",
                    )
                )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_after_drain = True
        with contextlib.suppress(asyncio.QueueFull):
            self._outbound.put_nowait(None)
        if self._writer_task and self._writer_task is not asyncio.current_task():
            await asyncio.gather(self._writer_task, return_exceptions=True)
        await self._close_transport()

    def _drop_pending_frames(self) -> None:
        while True:
            try:
                self._outbound.get_nowait()
            except asyncio.QueueEmpty:
                return

    async def _writer_loop(self) -> None:
        try:
            while True:
                frame = await self._outbound.get()
                if frame is None:
                    break
                await self._write_frame(frame)
                if self._close_after_drain and self._outbound.empty():
                    break
        except Exception:
            logger.exception("Client writer failed")
        finally:
            self._closed = True
            await self._close_transport()

    @abstractmethod
    async def _write_frame(self, frame: dict[str, Any]) -> None:
        """Write one JSON frame to the concrete transport."""

    @abstractmethod
    async def _close_transport(self) -> None:
        """Close the concrete transport."""


class StdIoClient(QueuedJsonClient):
    async def _write_frame(self, frame: dict[str, Any]) -> None:
        body = json_dumps(frame).encode("utf-8")
        sys.stdout.buffer.write(encode_frame(TYPE_JSON_FRAME, body))
        sys.stdout.buffer.flush()

    async def _close_transport(self) -> None:
        return None


class TcpJsonClient(QueuedJsonClient):
    def __init__(
        self,
        writer: asyncio.StreamWriter,
        *,
        queue_max: int,
    ) -> None:
        super().__init__(queue_max=queue_max)
        self._writer = writer

    async def _write_frame(self, frame: dict[str, Any]) -> None:
        body = json_dumps(frame).encode("utf-8")
        self._writer.write(encode_frame(TYPE_JSON_FRAME, body))
        await self._writer.drain()

    async def _close_transport(self) -> None:
        if not self._writer.is_closing():
            self._writer.close()
            with contextlib.suppress(Exception):
                await self._writer.wait_closed()


class WebSocketJsonClient(QueuedJsonClient):
    def __init__(self, websocket: Any, *, queue_max: int) -> None:
        super().__init__(queue_max=queue_max)
        self._websocket = websocket

    async def _write_frame(self, frame: dict[str, Any]) -> None:
        await self._websocket.send(json_dumps(frame))

    async def _close_transport(self) -> None:
        with contextlib.suppress(Exception):
            await self._websocket.close()
