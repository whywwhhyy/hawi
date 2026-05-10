"""HTTP gateway: POST /rpc + GET /events (SSE or WS upgrade) for the Hawi engine.

Per-client state lives on HttpGatewayClient. The aiohttp app (HttpGateway.serve)
binds endpoints onto a shared {client_id: HttpGatewayClient} map.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import contextlib
import logging
import uuid
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from .gateway import Gateway, register_gateway
from .protocol import json_dumps
from .transports import QueuedJsonClient

if TYPE_CHECKING:
    from .runtime import CoreRuntime

logger = logging.getLogger(__name__)

EventSink = Callable[[int, dict[str, Any]], Awaitable[None]]
"""Callable invoked when a new event is dispatched. Receives (seq, frame)."""


async def _wait_first(*events: asyncio.Event) -> None:
    """Return as soon as any of the given asyncio.Events is set."""
    if not events:
        return
    waiters = [asyncio.create_task(e.wait()) for e in events]
    try:
        await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for w in waiters:
            if not w.done():
                w.cancel()
        # Drain cancellations so we don't emit "Task was destroyed" warnings.
        for w in waiters:
            with contextlib.suppress(BaseException):
                await w


class HttpGatewayClient(QueuedJsonClient):
    """Per-logical-client state for the HTTP gateway.

    Inbound: aiohttp POST handlers call `expect_response()` then push the
    request frame through the runtime; the runtime's `client.send(ack)` is
    routed to the matching future and returned as the POST body.

    Outbound: every server-pushed event is assigned a monotonic seq, stored
    in a ring buffer for Last-Event-ID resume, and dispatched to all
    currently-subscribed sinks (SSE or WS connections).
    """

    def __init__(
        self,
        *,
        client_id: str,
        queue_max: int = 100,
        ring_buffer_size: int = 1000,
    ) -> None:
        super().__init__(queue_max=queue_max, client_id=client_id)
        self._ring: collections.deque[tuple[int, dict[str, Any]]] = collections.deque(
            maxlen=ring_buffer_size
        )
        self._next_seq = 1
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._sinks: set[EventSink] = set()
        self._buffer_size = ring_buffer_size
        # Set when the client is being torn down. SSE/WS handlers wait on this
        # alongside their own connection-close events so shutdown is prompt.
        self._closing: asyncio.Event = asyncio.Event()

    @property
    def closing_event(self) -> asyncio.Event:
        """Asyncio event set when the client is being torn down."""
        return self._closing

    async def expect_response(self, request_id: str) -> asyncio.Future[dict[str, Any]]:
        """Allocate a future to receive the response frame for `request_id`."""
        if request_id in self._pending:
            raise ValueError(f"request_id {request_id!r} already pending")
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = fut
        return fut

    def cancel_pending(self, request_id: str) -> None:
        """Drop a pending future without resolving it (e.g., on POST abort)."""
        fut = self._pending.pop(request_id, None)
        if fut and not fut.done():
            fut.cancel()

    def subscribe(self, sink: EventSink) -> EventSink:
        """Register an event sink; returns the same callable as a handle."""
        self._sinks.add(sink)
        return sink

    def unsubscribe(self, handle: EventSink) -> None:
        self._sinks.discard(handle)

    def iter_buffer_since(
        self, last_seq: int
    ) -> Iterator[tuple[int, dict[str, Any]]]:
        """Yield buffered (seq, frame) pairs with seq > last_seq, in order."""
        for seq, frame in self._ring:
            if seq > last_seq:
                yield seq, frame

    def has_gap_since(self, last_seq: int) -> bool:
        """True iff last_seq < oldest_seq_in_buffer (resume request would lose events)."""
        if not self._ring:
            return False
        oldest_seq = self._ring[0][0]
        return last_seq < oldest_seq - 1

    # Override _write_frame: route by request_id, otherwise dispatch as event.
    async def _write_frame(self, frame: dict[str, Any]) -> None:
        request_id = frame.get("id")
        if request_id and request_id in self._pending:
            fut = self._pending.pop(request_id)
            if not fut.done():
                fut.set_result(frame)
            return

        # It's an event — assign seq, buffer, dispatch to sinks.
        seq = self._next_seq
        self._next_seq += 1
        self._ring.append((seq, frame))

        # Snapshot sinks so we don't mutate during iteration.
        for sink in list(self._sinks):
            try:
                await sink(seq, frame)
            except Exception:
                logger.exception("Event sink raised; removing")
                self._sinks.discard(sink)

    async def _close_transport(self) -> None:
        # Cancel any pending POST futures.
        self._closing.set()
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.cancel()
        self._pending.clear()
        self._sinks.clear()


class HttpGateway(Gateway):
    name = "http"

    def __init__(self) -> None:
        # Per-listener state lives in serve(); this constructor stays argument-free
        # so the entry-point loader can instantiate it.
        pass

    def register_args(self, parser: argparse.ArgumentParser) -> None:
        # http shares --host / --port with the global parser.
        parser.add_argument(
            "--http-ring-buffer-size",
            type=int,
            default=1000,
            help="Per-client event ring buffer size (Last-Event-ID resume window). Default 1000.",
        )

    async def serve(self, runtime: "CoreRuntime", args: argparse.Namespace) -> None:
        from aiohttp import web

        clients: dict[str, HttpGatewayClient] = {}
        ring_size = getattr(args, "http_ring_buffer_size", 1000)
        port = args.port if args.port is not None else 8767

        async def _resolve_client(request: "web.Request") -> tuple[HttpGatewayClient, bool]:
            """Return (client, is_new). Mints a new client_id if missing."""
            client_id = (
                request.headers.get("X-Hawi-Client-Id")
                or request.cookies.get("hawi_client_id")
            )
            is_new = False
            if not client_id or client_id not in clients:
                client_id = client_id or uuid.uuid4().hex[:16]
                client = HttpGatewayClient(
                    client_id=client_id,
                    queue_max=getattr(args, "outbound_queue_size", 100),
                    ring_buffer_size=ring_size,
                )
                await client.start()
                await runtime.register_client(client)
                clients[client_id] = client
                is_new = True
            return clients[client_id], is_new

        async def _maybe_synthesize_hello(
            request: "web.Request", client: HttpGatewayClient
        ) -> None:
            """If client isn't authenticated, run a synthetic hello using the bearer token."""
            if client.authenticated:
                return
            auth = request.headers.get("Authorization", "")
            token = auth.removeprefix("Bearer ").strip() or None
            caps_header = request.headers.get("X-Hawi-Client-Caps", "")
            client_caps = [c.strip() for c in caps_header.split(",") if c.strip()]
            payload: dict[str, Any] = {"client_caps": client_caps}
            if token is not None:
                payload["token"] = token
            hello_id = f"http-hello-{uuid.uuid4().hex[:8]}"
            hello_frame = {
                "version": "hawi.core.v1",
                "type": "hello",
                "id": hello_id,
                "payload": payload,
            }
            fut = await client.expect_response(hello_id)
            await runtime.handle_frame(client, json_dumps(hello_frame))
            try:
                response = await asyncio.wait_for(fut, timeout=5.0)
            except asyncio.TimeoutError:
                client.cancel_pending(hello_id)
                raise web.HTTPException(reason="hello timed out")
            if response["type"] == "error":
                raise web.HTTPUnauthorized(
                    reason=response["payload"].get("message", "unauthorized")
                )

        async def post_rpc(request: "web.Request") -> "web.Response":
            try:
                body = await request.json()
            except ValueError:
                raise web.HTTPBadRequest(reason="body must be JSON")
            if not isinstance(body, dict) or "type" not in body:
                raise web.HTTPBadRequest(reason="body must be a protocol envelope")

            client, is_new = await _resolve_client(request)
            await _maybe_synthesize_hello(request, client)

            request_id = body.get("id")
            if not isinstance(request_id, str):
                # Auto-assign one so we can correlate.
                request_id = f"http-{uuid.uuid4().hex[:8]}"
                body = {**body, "id": request_id}

            fut = await client.expect_response(request_id)
            await runtime.handle_frame(client, json_dumps(body))
            try:
                response_frame = await asyncio.wait_for(fut, timeout=30.0)
            except asyncio.TimeoutError:
                client.cancel_pending(request_id)
                raise web.HTTPException(reason="rpc response timed out")

            resp = web.json_response(response_frame)
            if is_new:
                resp.set_cookie(
                    "hawi_client_id", client.id, httponly=True, samesite="Strict", path="/"
                )
                resp.headers["X-Hawi-Client-Id"] = client.id
            return resp

        async def get_events(request: "web.Request") -> "web.StreamResponse":
            # WS upgrade is handled by ws_events
            if request.headers.get("Upgrade", "").lower() == "websocket":
                return await ws_events(request)

            client, is_new = await _resolve_client(request)
            await _maybe_synthesize_hello(request, client)

            last_event_id = request.headers.get("Last-Event-ID")
            try:
                last_seq = int(last_event_id) if last_event_id else 0
            except ValueError:
                last_seq = 0

            response = web.StreamResponse(
                status=200,
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Hawi-Client-Id": client.id,
                },
            )
            if is_new:
                response.set_cookie(
                    "hawi_client_id", client.id, httponly=True, samesite="Strict", path="/"
                )
            await response.prepare(request)

            stream_done: asyncio.Event = asyncio.Event()

            # 1. If client asked to resume but the buffer evicted past Last-Event-ID, emit a gap.
            if last_seq and client.has_gap_since(last_seq):
                oldest = client._ring[0][0] if client._ring else 0
                try:
                    await response.write(
                        f"event: gap\ndata: {{\"oldest_seq\": {oldest}}}\n\n".encode("utf-8")
                    )
                except (ConnectionResetError, asyncio.CancelledError):
                    stream_done.set()

            # 2. Replay buffered events newer than Last-Event-ID.
            for seq, frame in client.iter_buffer_since(last_seq):
                line = f"id: {seq}\ndata: {json_dumps(frame)}\n\n"
                try:
                    await response.write(line.encode("utf-8"))
                except (ConnectionResetError, asyncio.CancelledError):
                    stream_done.set()
                    break

            # 3. Live stream: subscribe a sink that writes to this response.
            async def sink(seq: int, frame: dict) -> None:
                if stream_done.is_set():
                    return
                try:
                    await response.write(
                        f"id: {seq}\ndata: {json_dumps(frame)}\n\n".encode("utf-8")
                    )
                except (ConnectionResetError, asyncio.CancelledError):
                    stream_done.set()

            client.subscribe(sink)
            try:
                # Park until the connection drops or the client is torn down.
                await _wait_first(stream_done, client.closing_event)
            finally:
                client.unsubscribe(sink)

            return response

        async def ws_events(request: "web.Request") -> "web.WebSocketResponse":
            client, _is_new = await _resolve_client(request)
            await _maybe_synthesize_hello(request, client)

            ws = web.WebSocketResponse()
            await ws.prepare(request)

            # Same Last-Event-ID semantics, transmitted via header
            last_seq = 0
            last_id_header = request.headers.get("Last-Event-ID")
            if last_id_header:
                try:
                    last_seq = int(last_id_header)
                except ValueError:
                    last_seq = 0

            stream_done: asyncio.Event = asyncio.Event()

            if last_seq and client.has_gap_since(last_seq):
                oldest = client._ring[0][0] if client._ring else 0
                with contextlib.suppress(ConnectionResetError, RuntimeError):
                    await ws.send_json({"event": "gap", "oldest_seq": oldest})

            for seq, frame in client.iter_buffer_since(last_seq):
                try:
                    await ws.send_json({"seq": seq, "frame": frame})
                except (ConnectionResetError, RuntimeError):
                    stream_done.set()
                    break

            async def sink(seq: int, frame: dict) -> None:
                if ws.closed or stream_done.is_set():
                    stream_done.set()
                    return
                try:
                    await ws.send_json({"seq": seq, "frame": frame})
                except (ConnectionResetError, RuntimeError):
                    stream_done.set()

            client.subscribe(sink)

            async def reader_task() -> None:
                # WS clients on /events are read-only: any inbound message is ignored
                # (commands go through POST /rpc, not WS). We still need to read the
                # socket to detect close.
                try:
                    async for msg in ws:
                        if msg.type == web.WSMsgType.CLOSE:
                            break
                except Exception:
                    pass
                stream_done.set()

            reader = asyncio.create_task(reader_task())
            try:
                await _wait_first(stream_done, client.closing_event)
            finally:
                client.unsubscribe(sink)
                reader.cancel()
                with contextlib.suppress(Exception):
                    await reader
                if not ws.closed:
                    await ws.close()
            return ws

        app = web.Application()
        app.router.add_post("/rpc", post_rpc)
        app.router.add_get("/events", get_events)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, args.host, port)
        await site.start()
        logger.info("Hawi engine HTTP gateway listening on %s:%d", args.host, port)

        try:
            await runtime.wait_shutdown()
        finally:
            # Signal all event sinks (SSE/WS handlers parked on closing_event)
            # to exit before we await runner.cleanup(), which otherwise waits
            # the full aiohttp shutdown_timeout (default 60s) for in-flight
            # responses to finish.
            for client in list(clients.values()):
                client.closing_event.set()
            for client in list(clients.values()):
                await runtime.unregister_client(client)
                await client.close()
            await runner.cleanup()


register_gateway(HttpGateway())
