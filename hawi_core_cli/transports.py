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
from .runtime import CoreRuntime

logger = logging.getLogger(__name__)


class QueuedJsonClient(ABC):
    """Base client with a bounded outbound queue."""

    def __init__(self, *, queue_max: int = 100, client_id: str | None = None) -> None:
        self.id = client_id or uuid.uuid4().hex[:12]
        self.authenticated = False
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
        sys.stdout.write(json_dumps(frame) + "\n")
        sys.stdout.flush()

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
        self._writer.write((json_dumps(frame) + "\n").encode("utf-8"))
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


async def run_stdio(runtime: CoreRuntime, *, queue_max: int = 100) -> None:
    """Run the stdio NDJSON transport."""
    client = StdIoClient(queue_max=queue_max, client_id="stdio")
    await client.start()
    await runtime.register_client(client)

    reader = await _stdin_reader()

    async def read_loop() -> None:
        while not runtime.is_shutdown_requested:
            line = await reader.readline()
            if not line:
                break
            await runtime.handle_frame(client, line)

    reader_task = asyncio.create_task(read_loop())
    shutdown_task = asyncio.create_task(runtime.wait_shutdown())
    done, pending = await asyncio.wait(
        {reader_task, shutdown_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    await asyncio.gather(*done, return_exceptions=True)
    await runtime.unregister_client(client)
    await client.close()
    if not runtime.is_shutdown_requested:
        await runtime.stop()


async def run_tcp(
    runtime: CoreRuntime,
    *,
    host: str,
    port: int,
    queue_max: int = 100,
) -> None:
    """Run the TCP NDJSON transport."""
    clients: set[TcpJsonClient] = set()

    async def handle_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        client = TcpJsonClient(writer, queue_max=queue_max)
        clients.add(client)
        await client.start()
        await runtime.register_client(client)
        try:
            while not runtime.is_shutdown_requested:
                line = await reader.readline()
                if not line:
                    break
                await runtime.handle_frame(client, line)
        finally:
            await runtime.unregister_client(client)
            clients.discard(client)
            await client.close()

    server = await asyncio.start_server(handle_client, host, port)
    sockets = ", ".join(str(sock.getsockname()) for sock in (server.sockets or []))
    logger.info("Hawi core TCP transport listening on %s", sockets)
    async with server:
        await runtime.wait_shutdown()
    for client in list(clients):
        await client.close()


async def run_websocket(
    runtime: CoreRuntime,
    *,
    host: str,
    port: int,
    queue_max: int = 100,
) -> None:
    """Run the WebSocket JSON-message transport."""
    try:
        from websockets.asyncio.server import serve
    except ImportError as exc:
        raise RuntimeError(
            "The websocket transport requires websockets>=15.0. "
            "Install project dependencies with `uv sync`."
        ) from exc

    clients: set[WebSocketJsonClient] = set()

    async def handle_client(websocket: Any) -> None:
        client = WebSocketJsonClient(websocket, queue_max=queue_max)
        clients.add(client)
        await client.start()
        await runtime.register_client(client)
        try:
            async for raw in websocket:
                await runtime.handle_frame(client, raw)
                if runtime.is_shutdown_requested:
                    break
        finally:
            await runtime.unregister_client(client)
            clients.discard(client)
            await client.close()

    async with serve(handle_client, host, port) as server:
        sockets = ", ".join(str(sock.getsockname()) for sock in (server.sockets or []))
        logger.info("Hawi core WebSocket transport listening on %s", sockets)
        await runtime.wait_shutdown()

    for client in list(clients):
        await client.close()


async def _stdin_reader() -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    loop = asyncio.get_running_loop()
    try:
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)
    except (AttributeError, NotImplementedError):
        return _ThreadedStdinReader()
    return reader


class _ThreadedStdinReader:
    async def readline(self) -> bytes:
        line = await asyncio.to_thread(sys.stdin.buffer.readline)
        return line
