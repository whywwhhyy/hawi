from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from hawi_core_cli.protocol import VERSION, make_ack, make_frame
from hawi_core_cli.runtime import CoreRuntime, parse_extra_tool_parameter, parse_extra_tool_parameters
from hawi_core_cli.transports import QueuedJsonClient, run_tcp, run_websocket
from hawi.agent import HawiAgent
from hawi.tool import AgentTool, ToolResult


@dataclass(eq=False)
class FakeClient:
    id: str = "client"
    authenticated: bool = False
    sent: list[dict[str, Any]] = field(default_factory=list)
    closed: bool = False

    async def send(self, frame: dict[str, Any]) -> None:
        self.sent.append(frame)

    async def close(self) -> None:
        self.closed = True


class DummyQueueManager:
    def get_queue_lengths(self) -> dict[str, int]:
        return {"normal": 0, "high_prio": 0, "urgent": 0}


class DummyState:
    name = "IDLE"


class DummyExecutor:
    state = DummyState()
    is_idle = True


class DummyAgent:
    def __init__(self) -> None:
        self.context = self
        self.model_name = ""

    def clear(self) -> None:
        self.cleared = True

    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt

    def set_model(self, model_name: str) -> None:
        self.model_name = model_name


class DummyScheduler:
    state = DummyState()

    def __init__(self) -> None:
        self._executor = DummyExecutor()
        self.agent = DummyAgent()
        self.enqueued: list[tuple[Any, str, dict[str, Any]]] = []

    def get_queue_lengths(self) -> dict[str, int]:
        return {"normal": 0, "high_prio": 0, "urgent": 0}

    def enqueue(self, content: Any, queue: str, metadata: dict[str, Any]) -> str:
        self.enqueued.append((content, queue, metadata))
        return "msg-123"

    async def interrupt(self, reason: str) -> list[str]:
        self.interrupt_reason = reason
        return ["tc-1"]

    def clear_all_queues(self) -> dict[str, int]:
        return {"normal": 1, "high_prio": 0, "urgent": 0}

    def clear_queue(self, queue: str) -> int:
        self.cleared_queue = queue
        return 2


class SimpleTool(AgentTool):
    @property
    def name(self) -> str:
        return "simple_tool"

    @property
    def description(self) -> str:
        return "A simple test tool"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        }

    def run(self, value: str) -> ToolResult:  # type: ignore[override]
        return ToolResult(True, value)


@pytest.mark.asyncio
async def test_register_client_without_token_authenticates_and_sends_ready() -> None:
    runtime = CoreRuntime(model_name="test-model", token=None)
    client = FakeClient()

    await runtime.register_client(client)

    assert client.authenticated is True
    assert client.sent[0]["type"] == "core.ready"


@pytest.mark.asyncio
async def test_start_does_not_queue_duplicate_initial_ready() -> None:
    runtime = CoreRuntime(model_name="test-model", token=None, status_interval=60)
    client = FakeClient()
    wait_forever = asyncio.create_task(asyncio.Event().wait())

    async def build_scheduler(**_: Any) -> tuple[DummyScheduler, asyncio.Task, list[Any]]:
        return DummyScheduler(), wait_forever, []

    async def stop_scheduler(*_: Any) -> None:
        wait_forever.cancel()
        await asyncio.gather(wait_forever, return_exceptions=True)

    runtime._build_scheduler = build_scheduler  # type: ignore[method-assign]
    runtime._stop_scheduler = stop_scheduler  # type: ignore[method-assign]

    await runtime.start()
    await runtime.register_client(client)
    await asyncio.sleep(0)
    await runtime.stop()

    assert [frame["type"] for frame in client.sent] == ["core.ready"]


@pytest.mark.asyncio
async def test_hello_with_token_required() -> None:
    runtime = CoreRuntime(model_name="test-model", token="secret")
    client = FakeClient()
    await runtime.register_client(client)

    await runtime.handle_frame(
        client,
        '{"version":"%s","type":"ping","id":"pre-auth","payload":{}}' % VERSION,
    )
    assert client.sent[-1]["type"] == "error"
    assert client.sent[-1]["payload"]["code"] == "unauthenticated"

    await runtime.handle_frame(
        client,
        '{"version":"%s","type":"hello","id":"bad","payload":{"token":"nope"}}'
        % VERSION,
    )
    assert client.authenticated is False
    assert client.sent[-1]["type"] == "error"
    assert client.sent[-1]["payload"]["code"] == "unauthorized"

    await runtime.handle_frame(
        client,
        '{"version":"%s","type":"hello","id":"ok","payload":{"token":"secret"}}'
        % VERSION,
    )
    assert client.authenticated is True
    assert [frame["type"] for frame in client.sent[-2:]] == ["ack", "core.ready"]


@pytest.mark.asyncio
async def test_enqueue_command_returns_message_id() -> None:
    runtime = CoreRuntime(model_name="test-model", token=None)
    client = FakeClient(authenticated=True)
    scheduler = DummyScheduler()
    runtime._scheduler = scheduler  # type: ignore[assignment]

    await runtime.handle_frame(
        client,
        '{"version":"%s","type":"enqueue","id":"req","payload":{"content":"hi","queue":"high_prio"}}'
        % VERSION,
    )

    assert client.sent[-1]["type"] == "ack"
    assert client.sent[-1]["payload"]["message_id"] == "msg-123"
    assert scheduler.enqueued == [("hi", "high_prio", {"queue_kind": "high_prio"})]


def test_parse_extra_tool_parameter() -> None:
    parameter = parse_extra_tool_parameter(["tool_call_description", "str", "Describe the call"])

    assert parameter.name == "tool_call_description"
    assert parameter.description == "Describe the call"
    assert parameter.schema == {"type": "string"}


def test_parse_extra_tool_parameter_allows_colons_in_description() -> None:
    parameter = parse_extra_tool_parameter(["note", "str", "Reason: use the fast path"])

    assert parameter.name == "note"
    assert parameter.description == "Reason: use the fast path"


def test_parser_accepts_space_separated_extra_tool_parameters() -> None:
    from hawi_core_cli.__main__ import build_parser

    args = build_parser().parse_args([
        "--extra-tool-parameter",
        "note",
        "str",
        "Reason: use the fast path",
        "--extra-tool-parameter",
        "priority",
        "int",
        "Priority from 1 to 5",
    ])

    assert args.extra_tool_parameter == [
        ["note", "str", "Reason: use the fast path"],
        ["priority", "int", "Priority from 1 to 5"],
    ]


def test_parse_extra_tool_parameters_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        parse_extra_tool_parameters([["note", "str", "first"], ["note", "int", "second"]])


def test_runtime_applies_extra_tool_parameters_to_agent() -> None:
    runtime = CoreRuntime(
        model_name="test-model",
        extra_tool_parameters=[
            parse_extra_tool_parameter(["tool_call_description", "str", "Describe the call"]),
            parse_extra_tool_parameter(["priority", "int", "Priority from 1 to 5"]),
        ],
    )
    agent = HawiAgent(model=MagicMock())
    agent.plugins.add_tool(SimpleTool())

    runtime._apply_extra_tool_parameters(agent)
    schema = agent.plugins.get_tool_definitions()[0]["schema"]
    description = agent.plugins.get_tool_definitions()[0]["description"]

    assert schema["properties"]["tool_call_description"]["type"] == "string"
    assert schema["properties"]["tool_call_description"]["description"] == "Describe the call"
    assert schema["properties"]["priority"]["type"] == "integer"
    assert schema["required"] == ["value", "tool_call_description", "priority"]
    assert "Injected framework parameters" in description
    assert "- tool_call_description (string, required): Describe the call" in description
    assert "- priority (integer, required): Priority from 1 to 5" in description


class CapturingQueuedClient(QueuedJsonClient):
    def __init__(self) -> None:
        super().__init__(queue_max=1, client_id="capture")
        self.written: list[dict[str, Any]] = []

    async def _write_frame(self, frame: dict[str, Any]) -> None:
        self.written.append(frame)

    async def _close_transport(self) -> None:
        return None


@pytest.mark.asyncio
async def test_queued_client_overflow_replaces_pending_with_error() -> None:
    client = CapturingQueuedClient()

    await client.send({"type": "first"})
    await client.send({"type": "second"})

    assert client._close_after_drain is True
    assert client._outbound.qsize() == 1
    frame = client._outbound.get_nowait()
    assert frame is not None
    assert frame["type"] == "error"
    assert frame["payload"]["code"] == "client_backpressure"


class FakeTransportRuntime:
    def __init__(self) -> None:
        self._shutdown = asyncio.Event()
        self.clients: set[Any] = set()

    @property
    def is_shutdown_requested(self) -> bool:
        return self._shutdown.is_set()

    async def register_client(self, client: Any) -> None:
        self.clients.add(client)
        client.authenticated = True
        await client.send(make_frame("core.ready", {"model_name": "fake"}))

    async def unregister_client(self, client: Any) -> None:
        self.clients.discard(client)

    async def handle_frame(self, client: Any, raw: str | bytes) -> None:
        data = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        request_id = data.get("id")
        if data["type"] == "ping":
            await client.send(make_frame("pong", {"ok": True}, request_id=request_id))
        elif data["type"] == "shutdown":
            await client.send(make_ack("shutdown", request_id=request_id))
            self._shutdown.set()

    async def wait_shutdown(self) -> None:
        await self._shutdown.wait()


async def _connect_tcp_with_retry(port: int) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    last_exc: Exception | None = None
    for _ in range(50):
        try:
            return await asyncio.open_connection("127.0.0.1", port)
        except OSError as exc:
            last_exc = exc
            await asyncio.sleep(0.02)
    assert last_exc is not None
    raise last_exc


@pytest.mark.asyncio
async def test_tcp_transport_smoke(unused_tcp_port: int) -> None:
    runtime = FakeTransportRuntime()
    server_task = asyncio.create_task(
        run_tcp(runtime, host="127.0.0.1", port=unused_tcp_port, queue_max=10)  # type: ignore[arg-type]
    )
    reader, writer = await _connect_tcp_with_retry(unused_tcp_port)

    try:
        ready = json.loads((await asyncio.wait_for(reader.readline(), timeout=2)).decode())
        assert ready["type"] == "core.ready"

        writer.write(
            b'{"version":"hawi.core.v1","type":"ping","id":"ping-1","payload":{}}\n'
        )
        await writer.drain()
        pong = json.loads((await asyncio.wait_for(reader.readline(), timeout=2)).decode())
        assert pong["type"] == "pong"
        assert pong["id"] == "ping-1"

        writer.write(
            b'{"version":"hawi.core.v1","type":"shutdown","id":"stop-1","payload":{}}\n'
        )
        await writer.drain()
        ack = json.loads((await asyncio.wait_for(reader.readline(), timeout=2)).decode())
        assert ack["type"] == "ack"
        assert ack["id"] == "stop-1"

        await asyncio.wait_for(server_task, timeout=2)
    finally:
        writer.close()
        await writer.wait_closed()
        if not server_task.done():
            server_task.cancel()
            await asyncio.gather(server_task, return_exceptions=True)


async def _connect_websocket_with_retry(port: int) -> Any:
    from websockets.asyncio.client import connect

    last_exc: Exception | None = None
    uri = f"ws://127.0.0.1:{port}"
    for _ in range(50):
        try:
            return await connect(uri)
        except OSError as exc:
            last_exc = exc
            await asyncio.sleep(0.02)
    assert last_exc is not None
    raise last_exc


@pytest.mark.asyncio
async def test_websocket_transport_smoke(unused_tcp_port: int) -> None:
    runtime = FakeTransportRuntime()
    server_task = asyncio.create_task(
        run_websocket(runtime, host="127.0.0.1", port=unused_tcp_port, queue_max=10)  # type: ignore[arg-type]
    )
    websocket = await _connect_websocket_with_retry(unused_tcp_port)

    try:
        ready = json.loads(await asyncio.wait_for(websocket.recv(), timeout=2))
        assert ready["type"] == "core.ready"

        await websocket.send(
            '{"version":"hawi.core.v1","type":"ping","id":"ping-ws","payload":{}}'
        )
        pong = json.loads(await asyncio.wait_for(websocket.recv(), timeout=2))
        assert pong["type"] == "pong"
        assert pong["id"] == "ping-ws"

        await websocket.send(
            '{"version":"hawi.core.v1","type":"shutdown","id":"stop-ws","payload":{}}'
        )
        ack = json.loads(await asyncio.wait_for(websocket.recv(), timeout=2))
        assert ack["type"] == "ack"
        assert ack["id"] == "stop-ws"

        await asyncio.wait_for(server_task, timeout=2)
    finally:
        await websocket.close()
        if not server_task.done():
            server_task.cancel()
            await asyncio.gather(server_task, return_exceptions=True)
