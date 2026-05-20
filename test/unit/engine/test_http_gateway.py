"""Tests for HTTPGateway components.

Task 2 covers HttpGatewayClient in isolation. Tasks 3-7 add aiohttp
integration tests.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import socket
from contextlib import closing

import pytest

from hawi.engine.http_gateway import HttpGatewayClient


@pytest.fixture
async def client():
    c = HttpGatewayClient(client_id="abc123", queue_max=100, ring_buffer_size=8)
    await c.start()
    yield c
    await c.close()


async def test_send_event_assigns_seq(client):
    await client.send({"type": "run.text_delta", "id": None, "ts": 0, "payload": {"delta": "hi"}})
    await asyncio.sleep(0.01)
    items = list(client.iter_buffer_since(0))
    assert len(items) == 1
    seq, frame = items[0]
    assert seq == 1
    assert frame["payload"]["delta"] == "hi"


async def test_send_response_resolves_pending_request(client):
    fut = await client.expect_response("req-1")
    await client.send({"type": "ack", "id": "req-1", "ts": 0, "payload": {"ok": True}})
    response = await asyncio.wait_for(fut, timeout=1.0)
    assert response["type"] == "ack"
    assert response["payload"]["ok"] is True


async def test_send_response_does_not_buffer(client):
    """Frames matching a pending request route to the future, not the event buffer."""
    fut = await client.expect_response("req-2")
    await client.send({"type": "ack", "id": "req-2", "ts": 0, "payload": {"ok": True}})
    await asyncio.wait_for(fut, timeout=1.0)
    assert list(client.iter_buffer_since(0)) == []


async def test_event_without_pending_request_goes_to_buffer(client):
    """Frames carrying an id that no one is expecting still flow as events."""
    # Some events legitimately have id=None; some control frames have id but no pending future.
    await client.send({"type": "core.ready", "id": None, "ts": 0, "payload": {}})
    await asyncio.sleep(0.01)
    items = list(client.iter_buffer_since(0))
    assert len(items) == 1


async def test_ring_buffer_eviction(client):
    """Pushing more than buffer size evicts the oldest entries; iter_buffer_since reports gap."""
    for i in range(20):
        await client.send({"type": "run.text_delta", "id": None, "ts": 0, "payload": {"i": i}})
    await asyncio.sleep(0.01)
    # Buffer holds the last 8.
    seqs = [seq for seq, _ in client.iter_buffer_since(0)]
    assert seqs == list(range(13, 21))  # seq 1..20, last 8 are 13..20

    # Asking for seq 5 (older than buffer's oldest) reports a gap.
    assert client.has_gap_since(5)
    assert not client.has_gap_since(13)


async def test_subscribe_and_unsubscribe_event_sink(client):
    """An attached sink receives events live; once detached, no more events arrive."""
    queue: asyncio.Queue = asyncio.Queue()

    async def sink(seq: int, frame: dict) -> None:
        await queue.put((seq, frame))

    handle = client.subscribe(sink)
    await client.send({"type": "run.text_delta", "id": None, "ts": 0, "payload": {"delta": "a"}})
    seq, frame = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert seq == 1
    assert frame["payload"]["delta"] == "a"

    client.unsubscribe(handle)
    await client.send({"type": "run.text_delta", "id": None, "ts": 0, "payload": {"delta": "b"}})
    await asyncio.sleep(0.05)
    assert queue.empty()


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
async def http_engine(tmp_path):
    """Start an HttpGateway on a free port wired to a real CoreRuntime; yield (port, runtime)."""
    from hawi.engine.runtime import CoreRuntime
    from hawi.engine.http_gateway import HttpGateway

    runtime = CoreRuntime(
        model_name="dummy/dummy",
        token=None,
        status_interval=60.0,  # silence status broadcasts during tests
    )
    # We cannot actually call runtime.start() without a registered model; for HTTP
    # tests we only need the runtime's frame-routing surface area, not the agent.
    # Use a minimal substitute that exposes the methods HttpGateway calls.
    test_runtime = _MinimalTestRuntime()
    runtime = test_runtime  # use the lightweight stub

    port = _free_port()
    args = argparse.Namespace(
        host="127.0.0.1",
        port=port,
        outbound_queue_size=100,
        http_ring_buffer_size=16,
    )
    gateway = HttpGateway()
    serve_task = asyncio.create_task(gateway.serve(runtime, args))
    # Give aiohttp a moment to bind
    for _ in range(40):
        try:
            r, w = await asyncio.open_connection("127.0.0.1", port)
            w.close()
            await w.wait_closed()
            break
        except OSError:
            await asyncio.sleep(0.05)
    else:
        raise RuntimeError("HTTP gateway did not start")
    try:
        yield port, runtime
    finally:
        runtime.request_shutdown()
        serve_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await serve_task


@pytest.fixture
async def http_engine_with_token():
    """HTTP gateway wired to a runtime stub that requires Bearer auth."""
    from hawi.engine.http_gateway import HttpGateway

    runtime = _MinimalTestRuntime(
        token="secret",
        server_caps={"last_event_id", "tlv_v1"},
    )
    port = _free_port()
    args = argparse.Namespace(
        host="127.0.0.1",
        port=port,
        outbound_queue_size=100,
        http_ring_buffer_size=16,
    )
    gateway = HttpGateway()
    serve_task = asyncio.create_task(gateway.serve(runtime, args))
    for _ in range(40):
        try:
            r, w = await asyncio.open_connection("127.0.0.1", port)
            w.close()
            await w.wait_closed()
            break
        except OSError:
            await asyncio.sleep(0.05)
    else:
        raise RuntimeError("HTTP gateway did not start")
    try:
        yield port, runtime
    finally:
        runtime.request_shutdown()
        serve_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await serve_task


class _MinimalTestRuntime:
    """Minimal CoreRuntime stand-in for HTTP gateway tests.

    Implements the subset HttpGateway calls: register_client, unregister_client,
    handle_frame, wait_shutdown. Authenticates clients on hello (with no token
    requirement) and replies to ping with pong. Exposes emit() to push events.
    """

    def __init__(
        self,
        *,
        token: str | None = None,
        server_caps: set[str] | None = None,
    ) -> None:
        self._shutdown = asyncio.Event()
        self._clients: set = set()
        self._token = token
        self._server_caps = server_caps or set()
        self.hello_payloads: list[dict] = []

    @property
    def is_shutdown_requested(self) -> bool:
        return self._shutdown.is_set()

    async def register_client(self, client) -> None:
        self._clients.add(client)

    async def unregister_client(self, client) -> None:
        self._clients.discard(client)

    async def handle_frame(self, client, raw) -> None:
        from hawi.engine.protocol import make_ack, make_frame, parse_frame

        if isinstance(raw, dict):
            frame_dict = raw
        else:
            frame_dict = None
        if frame_dict is None:
            command = parse_frame(raw)
        else:
            # Build a CoreCommand-like object from the dict.
            from hawi.engine.protocol import CoreCommand
            command = CoreCommand(
                type=frame_dict["type"],
                payload=frame_dict.get("payload") or {},
                id=frame_dict.get("id"),
            )

        if command.type == "hello":
            self.hello_payloads.append(dict(command.payload))
            if self._token is not None and command.payload.get("token") != self._token:
                from hawi.engine.protocol import make_error
                await client.send(
                    make_error(
                        "Invalid authentication token.",
                        request_id=command.id,
                        code="unauthorized",
                    )
                )
                return
            client_caps = set(command.payload.get("client_caps") or [])
            negotiated = sorted(client_caps & self._server_caps)
            client.authenticated = True
            client.negotiated_caps = set(negotiated)
            await client.send(
                make_ack(
                    "hello",
                    request_id=command.id,
                    payload={
                        "authenticated": True,
                        "server_caps": sorted(self._server_caps),
                        "negotiated": negotiated,
                    },
                )
            )
        elif command.type == "ping":
            await client.send(
                make_frame("pong", {"ok": True}, request_id=command.id)
            )
        else:
            from hawi.engine.protocol import make_error
            await client.send(make_error("unhandled in test", request_id=command.id))

    async def wait_shutdown(self) -> None:
        await self._shutdown.wait()

    def request_shutdown(self) -> None:
        self._shutdown.set()

    def emit(self, frame: dict) -> None:
        """Broadcast a frame to all authenticated clients (test helper)."""
        async def _push() -> None:
            for c in list(self._clients):
                if getattr(c, "authenticated", False):
                    await c.send(frame)
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            return
        loop.create_task(_push())


async def test_post_rpc_returns_ack_and_sets_cookie(http_engine):
    port, _ = http_engine
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://127.0.0.1:{port}/rpc",
            json={"version": "hawi.core.v1", "type": "ping", "id": "p1", "payload": {}},
        ) as resp:
            assert resp.status == 200
            body = await resp.json()
            # Runtime returns "pong" for ping commands.
            assert body["type"] == "pong"
            assert body["id"] == "p1"
            assert "hawi_client_id" in resp.cookies


async def test_http_rpc_rejects_missing_or_bad_bearer_token(http_engine_with_token):
    port, runtime = http_engine_with_token
    import aiohttp

    async with aiohttp.ClientSession() as session:
        for headers in ({}, {"Authorization": "Bearer wrong"}):
            async with session.post(
                f"http://127.0.0.1:{port}/rpc",
                json={"version": "hawi.core.v1", "type": "ping", "id": "p1", "payload": {}},
                headers=headers,
            ) as resp:
                assert resp.status == 401
                await resp.read()

    assert runtime.hello_payloads == [
        {"client_caps": []},
        {"client_caps": [], "token": "wrong"},
    ]
    assert all(not client._pending for client in runtime._clients)


async def test_http_rpc_with_bearer_auth_synthesizes_hello_and_negotiates_caps(http_engine_with_token):
    port, runtime = http_engine_with_token
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://127.0.0.1:{port}/rpc",
            json={"version": "hawi.core.v1", "type": "ping", "id": "p1", "payload": {}},
            headers={
                "Authorization": "Bearer secret",
                "X-Hawi-Client-Caps": "tlv_v1,last_event_id,unknown",
            },
        ) as resp:
            assert resp.status == 200
            body = await resp.json()
            assert body["type"] == "pong"
            assert body["id"] == "p1"

    assert runtime.hello_payloads == [
        {
            "client_caps": ["tlv_v1", "last_event_id", "unknown"],
            "token": "secret",
        }
    ]
    client = next(iter(runtime._clients))
    assert client.authenticated is True
    assert client.negotiated_caps == {"tlv_v1", "last_event_id"}


async def test_http_events_rejects_missing_bearer_token(http_engine_with_token):
    port, _ = http_engine_with_token
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://127.0.0.1:{port}/events") as resp:
            assert resp.status == 401
            await resp.read()


async def test_sse_streams_events(http_engine):
    port, runtime = http_engine
    import aiohttp

    async with aiohttp.ClientSession() as session:
        # Open SSE connection
        async with session.get(f"http://127.0.0.1:{port}/events") as resp:
            assert resp.status == 200
            assert resp.headers["Content-Type"] == "text/event-stream"

            # Trigger an event by emitting a debug.info frame from runtime
            client_id = resp.headers["X-Hawi-Client-Id"]
            await asyncio.sleep(0.1)  # let SSE prepare
            runtime.emit({"version": "hawi.core.v1", "type": "debug.info",
                          "id": None, "ts": 0, "payload": {"msg": "hello-sse"}})

            # Read until we see the event
            chunk = await asyncio.wait_for(resp.content.read(1024), timeout=2.0)
            text = chunk.decode("utf-8")
            assert "hello-sse" in text
            assert "id: 1" in text


async def test_ws_upgrade_streams_events(http_engine):
    port, runtime = http_engine
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(f"http://127.0.0.1:{port}/events") as ws:
            await asyncio.sleep(0.1)
            runtime.emit({"version": "hawi.core.v1", "type": "debug.info",
                          "id": None, "ts": 0, "payload": {"msg": "hello-ws"}})
            msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
            data = msg.json()
            assert data["seq"] == 1
            assert data["frame"]["payload"]["msg"] == "hello-ws"


async def test_ws_resume_replays_missed_events(http_engine):
    port, runtime = http_engine
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://127.0.0.1:{port}/rpc",
            json={"version": "hawi.core.v1", "type": "ping", "id": "p1", "payload": {}},
        ) as resp:
            client_id = resp.headers["X-Hawi-Client-Id"]
            await resp.read()

        for i in range(1, 5):
            runtime.emit({"version": "hawi.core.v1", "type": "debug.info",
                          "id": None, "ts": 0, "payload": {"i": i}})
        await asyncio.sleep(0.1)

        async with session.ws_connect(
            f"http://127.0.0.1:{port}/events",
            headers={"Last-Event-ID": "2", "X-Hawi-Client-Id": client_id},
        ) as ws:
            msg3 = await asyncio.wait_for(ws.receive(), timeout=2.0)
            msg4 = await asyncio.wait_for(ws.receive(), timeout=2.0)
            data3 = msg3.json()
            data4 = msg4.json()
            assert data3["seq"] == 3
            assert data3["frame"]["payload"]["i"] == 3
            assert data4["seq"] == 4
            assert data4["frame"]["payload"]["i"] == 4


async def test_ws_emits_gap_when_buffer_evicted(http_engine):
    port, runtime = http_engine
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://127.0.0.1:{port}/rpc",
            json={"version": "hawi.core.v1", "type": "ping", "id": "p1", "payload": {}},
        ) as resp:
            client_id = resp.headers["X-Hawi-Client-Id"]
            await resp.read()

        for i in range(25):
            runtime.emit({"version": "hawi.core.v1", "type": "debug.info",
                          "id": None, "ts": 0, "payload": {"i": i}})
        await asyncio.sleep(0.1)

        async with session.ws_connect(
            f"http://127.0.0.1:{port}/events",
            headers={"Last-Event-ID": "1", "X-Hawi-Client-Id": client_id},
        ) as ws:
            msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
            data = msg.json()
            assert data["event"] == "gap"
            assert "oldest_seq" in data


async def test_sse_resume_replays_missed_events(http_engine):
    port, runtime = http_engine
    import aiohttp

    cookie_jar = aiohttp.CookieJar(unsafe=True)
    async with aiohttp.ClientSession(cookie_jar=cookie_jar) as session:
        # First connect: get client_id and emit two events while connected
        async with session.get(f"http://127.0.0.1:{port}/events") as resp:
            client_id = resp.headers["X-Hawi-Client-Id"]
            await asyncio.sleep(0.1)
            runtime.emit({"version": "hawi.core.v1", "type": "debug.info",
                          "id": None, "ts": 0, "payload": {"i": 1}})
            runtime.emit({"version": "hawi.core.v1", "type": "debug.info",
                          "id": None, "ts": 0, "payload": {"i": 2}})
            chunk = await asyncio.wait_for(resp.content.read(2048), timeout=2.0)
            assert "i\":1" in chunk.decode() and "i\":2" in chunk.decode()
            # Connection closed when the `async with` exits.

        # Emit more events while disconnected.
        runtime.emit({"version": "hawi.core.v1", "type": "debug.info",
                      "id": None, "ts": 0, "payload": {"i": 3}})
        runtime.emit({"version": "hawi.core.v1", "type": "debug.info",
                      "id": None, "ts": 0, "payload": {"i": 4}})

        # Reconnect with Last-Event-ID = 2 (the seq of "i: 2"). Expect 3 and 4 replayed.
        async with session.get(
            f"http://127.0.0.1:{port}/events",
            headers={"Last-Event-ID": "2", "X-Hawi-Client-Id": client_id},
        ) as resp:
            chunk = await asyncio.wait_for(resp.content.read(2048), timeout=2.0)
            text = chunk.decode()
            assert "i\":3" in text
            assert "i\":4" in text
            assert "id: 3" in text
            assert "id: 4" in text


async def test_sse_emits_gap_when_buffer_evicted(http_engine):
    port, runtime = http_engine
    import aiohttp

    # The fixture sets ring_buffer_size=16. Emit 25 events, then ask for Last-Event-ID=1
    # which is older than the ring's oldest entry.
    cookie_jar = aiohttp.CookieJar(unsafe=True)
    async with aiohttp.ClientSession(cookie_jar=cookie_jar) as session:
        # Bootstrap an http client (just hit POST /rpc with a ping).
        async with session.post(
            f"http://127.0.0.1:{port}/rpc",
            json={"version": "hawi.core.v1", "type": "ping", "id": "p1", "payload": {}},
        ) as resp:
            client_id = resp.headers["X-Hawi-Client-Id"]
            await resp.read()

        for i in range(25):
            runtime.emit({"version": "hawi.core.v1", "type": "debug.info",
                          "id": None, "ts": 0, "payload": {"i": i}})

        # Allow emit's create_task to complete
        await asyncio.sleep(0.1)

        async with session.get(
            f"http://127.0.0.1:{port}/events",
            headers={"Last-Event-ID": "1", "X-Hawi-Client-Id": client_id},
        ) as resp:
            chunk = await asyncio.wait_for(resp.content.read(4096), timeout=2.0)
            text = chunk.decode()
            assert "event: gap" in text
            assert "oldest_seq" in text
