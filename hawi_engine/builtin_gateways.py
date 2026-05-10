"""Built-in gateways: stdio, tcp.

Each gateway is registered with `gateway.GATEWAY_REGISTRY` at import time.
The HTTP gateway (`http`) lives in `hawi_engine.http_gateway`; its
WS-upgrade path replaces the standalone WebSocket gateway removed in Plan 4.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import TYPE_CHECKING

from .gateway import Gateway, register_gateway
from .protocol import make_error
from .tlv import (
    DEFAULT_MAX_FRAME_SIZE,
    TYPE_BINARY_BLOB,
    TYPE_JSON_FRAME,
    FrameTooLargeError,
    UnexpectedEOFError,
    read_frame,
)
from .transports import StdIoClient, TcpJsonClient

if TYPE_CHECKING:
    from .runtime import CoreRuntime

logger = logging.getLogger(__name__)


class _ThreadedStdinReader:
    """asyncio.StreamReader-compatible byte reader using a worker thread.

    Used on platforms where loop.connect_read_pipe(sys.stdin) fails (Windows,
    some sandboxes). We only need the readexactly() shape that tlv.read_frame
    consumes.
    """

    async def readexactly(self, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            chunk = await asyncio.to_thread(sys.stdin.buffer.read, n - len(out))
            if not chunk:
                # EOF; emulate StreamReader.readexactly's IncompleteReadError contract
                raise asyncio.IncompleteReadError(bytes(out), n)
            out.extend(chunk)
        return bytes(out)


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
        max_size = getattr(args, "max_frame_size", DEFAULT_MAX_FRAME_SIZE)

        async def read_loop() -> None:
            while not runtime.is_shutdown_requested:
                try:
                    result = await read_frame(reader, max_size=max_size)
                except FrameTooLargeError as exc:
                    await client.send(make_error(str(exc), code="frame_too_large"))
                    break
                except UnexpectedEOFError:
                    break
                if result is None:
                    break  # clean EOF
                type_byte, value = result
                if type_byte == TYPE_JSON_FRAME:
                    await runtime.handle_frame(client, value)
                elif type_byte == TYPE_BINARY_BLOB:
                    # Reserved for Plan 5 blob fast-path. Plan 3 has no consumer;
                    # log and discard so the stream stays in sync.
                    logger.debug(
                        "Discarded TYPE_BINARY_BLOB frame (%d bytes); blob support not implemented",
                        len(value),
                    )
                else:
                    logger.warning(
                        "Discarded unknown TLV frame type 0x%02x (%d bytes)",
                        type_byte,
                        len(value),
                    )

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
        max_size = getattr(args, "max_frame_size", DEFAULT_MAX_FRAME_SIZE)

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
                    try:
                        result = await read_frame(reader, max_size=max_size)
                    except FrameTooLargeError as exc:
                        await client.send(make_error(str(exc), code="frame_too_large"))
                        break
                    except UnexpectedEOFError:
                        break
                    if result is None:
                        break
                    type_byte, value = result
                    if type_byte == TYPE_JSON_FRAME:
                        await runtime.handle_frame(client, value)
                    elif type_byte == TYPE_BINARY_BLOB:
                        logger.debug(
                            "Discarded TYPE_BINARY_BLOB frame on tcp; blob support not implemented"
                        )
                    else:
                        logger.warning(
                            "Discarded unknown TLV frame type 0x%02x on tcp", type_byte
                        )
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


register_gateway(StdioGateway())
register_gateway(TcpGateway())
