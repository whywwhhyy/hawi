"""Built-in gateways: stdio, tcp, websocket.

Each gateway is registered with `gateway.GATEWAY_REGISTRY` at import time.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import TYPE_CHECKING

from .gateway import Gateway, register_gateway
from .transports import StdIoClient, TcpJsonClient, WebSocketJsonClient

if TYPE_CHECKING:
    from .runtime import CoreRuntime

logger = logging.getLogger(__name__)


class _ThreadedStdinReader:
    async def readline(self) -> bytes:
        return await asyncio.to_thread(sys.stdin.buffer.readline)


async def _stdin_reader() -> "asyncio.StreamReader | _ThreadedStdinReader":
    if sys.platform == "win32":
        return _ThreadedStdinReader()

    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    loop = asyncio.get_running_loop()
    try:
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)
    except (AttributeError, NotImplementedError, OSError):
        return _ThreadedStdinReader()
    return reader


class StdioGateway(Gateway):
    name = "stdio"

    def register_args(self, parser: argparse.ArgumentParser) -> None:
        # stdio gateway has no extra args — all global flags (--token, --outbound-queue-size, etc.)
        # are already on the top-level parser.
        return

    async def serve(self, runtime: "CoreRuntime", args: argparse.Namespace) -> None:
        client = StdIoClient(queue_max=args.outbound_queue_size, client_id="stdio")
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


class TcpGateway(Gateway):
    name = "tcp"

    def register_args(self, parser: argparse.ArgumentParser) -> None:
        # tcp shares --host / --port with the global parser; no per-gateway args.
        return

    async def serve(self, runtime: "CoreRuntime", args: argparse.Namespace) -> None:
        clients: set[TcpJsonClient] = set()
        port = args.port if args.port is not None else 8765

        async def handle_client(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            client = TcpJsonClient(writer, queue_max=args.outbound_queue_size)
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

        server = await asyncio.start_server(handle_client, args.host, port)
        sockets = ", ".join(str(sock.getsockname()) for sock in (server.sockets or []))
        logger.info("Hawi engine TCP gateway listening on %s", sockets)
        async with server:
            await runtime.wait_shutdown()
        for client in list(clients):
            await client.close()


class WebSocketGateway(Gateway):
    name = "websocket"

    def register_args(self, parser: argparse.ArgumentParser) -> None:
        return

    async def serve(self, runtime: "CoreRuntime", args: argparse.Namespace) -> None:
        try:
            from websockets.asyncio.server import serve
        except ImportError as exc:
            raise RuntimeError(
                "The websocket gateway requires websockets>=15.0. "
                "Install project dependencies with `uv sync`."
            ) from exc

        clients: set[WebSocketJsonClient] = set()
        port = args.port if args.port is not None else 8766

        async def handle_client(websocket) -> None:
            client = WebSocketJsonClient(websocket, queue_max=args.outbound_queue_size)
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

        async with serve(handle_client, args.host, port) as server:
            sockets = ", ".join(str(sock.getsockname()) for sock in (server.sockets or []))
            logger.info("Hawi engine WebSocket gateway listening on %s", sockets)
            await runtime.wait_shutdown()

        for client in list(clients):
            await client.close()


register_gateway(StdioGateway())
register_gateway(TcpGateway())
register_gateway(WebSocketGateway())
