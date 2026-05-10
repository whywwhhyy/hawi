from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

import hawi_engine.builtin_gateways as builtin_gateways
from hawi_engine.protocol import VERSION, make_ack, make_frame
from hawi_engine.runtime import CoreRuntime, parse_extra_tool_parameter, parse_extra_tool_parameters
from hawi_engine.tlv import TYPE_JSON_FRAME, encode_frame, read_frame
from hawi_engine.transports import QueuedJsonClient
from hawi.agent import HawiAgent
from hawi.tool import AgentTool, ToolResult


async def _send_tlv(writer: asyncio.StreamWriter, raw: bytes) -> None:
    writer.write(encode_frame(TYPE_JSON_FRAME, raw))
    await writer.drain()


async def _recv_tlv(reader: asyncio.StreamReader, *, timeout: float = 2) -> dict[str, Any]:
    result = await asyncio.wait_for(read_frame(reader), timeout=timeout)
    assert result is not None, "stream closed"
    type_byte, value = result
    assert type_byte == TYPE_JSON_FRAME, f"expected JSON frame, got 0x{type_byte:02x}"
    return json.loads(value.decode("utf-8"))


@dataclass(eq=False)
class FakeClient:
    id: str = "client"
    authenticated: bool = False
    negotiated_caps: set[str] = field(default_factory=set)
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
        self.loaded_steer: list[Any] | None = None
        self.loaded_runtime: dict[str, Any] | None = None

    def clear(self) -> None:
        self.cleared = True

    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt

    def set_model(self, model_name: str) -> None:
        self.model_name = model_name

    def load_steer(self, data: list[Any]) -> None:
        self.loaded_steer = data

    def load_runtime(self, data: dict[str, Any]) -> None:
        self.loaded_runtime = data


class DummyScheduler:
    state = DummyState()

    def __init__(self) -> None:
        self._executor = DummyExecutor()
        self.agent = DummyAgent()
        self.enqueued: list[tuple[Any, str, dict[str, Any]]] = []

    def get_queue_lengths(self) -> dict[str, int]:
        return {"normal": 0, "high_prio": 0, "urgent": 0}

    def get_queue_messages(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "normal": [
                {
                    "id": "msg-1",
                    "queue": "normal",
                    "content_preview": "queued",
                    "created_at": 123.0,
                    "metadata": {},
                }
            ],
            "high_prio": [],
            "urgent": [],
        }

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


class DummySessionManager:
    current_session_id = "current-session"

    def __init__(self) -> None:
        self.loaded: str | None = None
        self.deleted: list[str] = []
        self.histories = {
            "current-session": [
                {
                    "version": 1,
                    "run_id": "run-current",
                    "role": "user",
                    "content": [{"type": "text", "text": "current"}],
                    "metadata": None,
                }
            ],
            "saved-session": [
                {
                    "version": 1,
                    "run_id": "run-saved",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "saved"}],
                    "metadata": None,
                }
            ],
        }

    def read_message_history(
        self,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return list(self.histories[session_id or self.current_session_id])

    def load_session(self, session_id: str) -> None:
        self.loaded = session_id
        self.current_session_id = session_id

    def new_session(self, name: str | None = None) -> str:
        self.new_session_name = name
        self.current_session_id = "new-session"
        return self.current_session_id

    def delete_session(self, session_id: str) -> None:
        self.deleted.append(session_id)
        self.histories.pop(session_id, None)


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


@pytest.mark.asyncio
async def test_session_history_command_returns_current_history() -> None:
    runtime = CoreRuntime(model_name="test-model", token=None)
    client = FakeClient(authenticated=True)
    runtime._session_manager = DummySessionManager()  # type: ignore[assignment]

    await runtime.handle_frame(
        client,
        '{"version":"%s","type":"session_history","id":"hist","payload":{}}'
        % VERSION,
    )

    payload = client.sent[-1]["payload"]
    assert client.sent[-1]["type"] == "ack"
    assert payload["command"] == "session_history"
    assert payload["session_id"] == "current-session"
    assert payload["message_history"][0]["content"][0]["text"] == "current"


@pytest.mark.asyncio
async def test_session_load_ack_includes_message_history() -> None:
    runtime = CoreRuntime(model_name="test-model", token=None)
    client = FakeClient(authenticated=True)
    sm = DummySessionManager()
    runtime._session_manager = sm  # type: ignore[assignment]

    await runtime.handle_frame(
        client,
        (
            '{"version":"%s","type":"session_load","id":"load",'
            '"payload":{"session_id":"saved-session"}}'
        )
        % VERSION,
    )

    payload = client.sent[-1]["payload"]
    assert sm.loaded == "saved-session"
    assert payload["command"] == "session_load"
    assert payload["session_id"] == "saved-session"
    assert payload["message_history"][0]["content"][0]["text"] == "saved"


@pytest.mark.asyncio
async def test_session_delete_rejects_current_session() -> None:
    runtime = CoreRuntime(model_name="test-model", token=None)
    client = FakeClient(authenticated=True)
    sm = DummySessionManager()
    runtime._session_manager = sm  # type: ignore[assignment]

    await runtime.handle_frame(
        client,
        (
            '{"version":"%s","type":"session_delete","id":"delete",'
            '"payload":{"session_id":"current-session"}}'
        )
        % VERSION,
    )

    payload = client.sent[-1]["payload"]
    assert client.sent[-1]["type"] == "error"
    assert payload["code"] == "invalid_session_delete"
    assert sm.deleted == []


@pytest.mark.asyncio
async def test_session_delete_removes_non_current_session() -> None:
    runtime = CoreRuntime(model_name="test-model", token=None)
    client = FakeClient(authenticated=True)
    sm = DummySessionManager()
    runtime._session_manager = sm  # type: ignore[assignment]

    await runtime.handle_frame(
        client,
        (
            '{"version":"%s","type":"session_delete","id":"delete",'
            '"payload":{"session_id":"saved-session"}}'
        )
        % VERSION,
    )

    payload = client.sent[-1]["payload"]
    assert client.sent[-1]["type"] == "ack"
    assert payload["command"] == "session_delete"
    assert payload["session_id"] == "saved-session"
    assert sm.deleted == ["saved-session"]


@pytest.mark.asyncio
async def test_session_new_resets_live_state_without_materializing_history() -> None:
    runtime = CoreRuntime(model_name="test-model", token=None)
    client = FakeClient(authenticated=True)
    scheduler = DummyScheduler()
    sm = DummySessionManager()
    runtime._scheduler = scheduler  # type: ignore[assignment]
    runtime._session_manager = sm  # type: ignore[assignment]

    await runtime.handle_frame(
        client,
        (
            '{"version":"%s","type":"session_new","id":"new",'
            '"payload":{"name":"fresh"}}'
        )
        % VERSION,
    )

    payload = client.sent[-1]["payload"]
    assert client.sent[-1]["type"] == "ack"
    assert payload["command"] == "session_new"
    assert payload["session_id"] == "new-session"
    assert sm.new_session_name == "fresh"
    assert scheduler.agent.cleared is True
    assert scheduler.agent.loaded_steer == []
    assert scheduler.agent.loaded_runtime["current_tool_calls"] == []


def test_status_payload_includes_queue_messages() -> None:
    runtime = CoreRuntime(model_name="test-model", token=None)
    scheduler = DummyScheduler()
    scheduler.get_queue_messages = lambda: {  # type: ignore[method-assign]
        "normal": [
            {
                "id": "msg-1",
                "queue": "normal",
                "content_preview": "queued",
                "created_at": 123.0,
                "metadata": {},
            }
        ],
        "high_prio": [],
        "urgent": [
            {
                "id": "urgent-1",
                "queue": "urgent",
                "content_preview": "interrupt now",
                "created_at": 123.5,
                "metadata": {},
            }
        ],
    }
    scheduler.agent.get_pending_input_messages = lambda: [  # type: ignore[attr-defined]
        {
            "id": "steer-plain",
            "queue": "normal",
            "content_preview": "pending plain steer",
            "created_at": 124.0,
            "metadata": {},
        },
        {
            "id": "steer-1",
            "queue": "high_prio",
            "content_preview": "pending steer",
            "created_at": 125.0,
            "metadata": {},
        }
    ]
    runtime._scheduler = scheduler  # type: ignore[assignment]

    payload = runtime._status_payload()

    assert payload["queue_messages"]["normal"][0]["content_preview"] == "queued"
    assert payload["queue_messages"]["normal"][1]["content_preview"] == "pending plain steer"
    assert payload["queue_messages"]["high_prio"][0]["content_preview"] == "pending steer"
    assert payload["queue_messages"]["urgent"] == []


def test_parse_extra_tool_parameter() -> None:
    parameter = parse_extra_tool_parameter(["tool_call_purpose", "str", "Describe the call"])

    assert parameter.name == "tool_call_purpose"
    assert parameter.description == "Describe the call"
    assert parameter.schema == {"type": "string"}


def test_parse_extra_tool_parameter_allows_colons_in_description() -> None:
    parameter = parse_extra_tool_parameter(["note", "str", "Reason: use the fast path"])

    assert parameter.name == "note"
    assert parameter.description == "Reason: use the fast path"


def test_parser_accepts_space_separated_extra_tool_parameters() -> None:
    from hawi_engine.__main__ import build_parser

    args = build_parser().parse_args([
        "--model",
        "test/model",
        "--max-context-tokens",
        "64000",
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
    assert args.max_context_tokens == 64_000


def test_parse_extra_tool_parameters_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        parse_extra_tool_parameters([["note", "str", "first"], ["note", "int", "second"]])


def test_runtime_applies_extra_tool_parameters_to_agent() -> None:
    runtime = CoreRuntime(
        model_name="test-model",
        extra_tool_parameters=[
            parse_extra_tool_parameter(["tool_call_purpose", "str", "Describe the call"]),
            parse_extra_tool_parameter(["priority", "int", "Priority from 1 to 5"]),
        ],
    )
    agent = HawiAgent(model=MagicMock())
    agent.plugins.add_tool(SimpleTool())

    runtime._apply_extra_tool_parameters(agent)
    schema = agent.plugins.get_tool_definitions()[0]["schema"]
    description = agent.plugins.get_tool_definitions()[0]["description"]

    assert schema["properties"]["tool_call_purpose"]["type"] == "string"
    assert schema["properties"]["tool_call_purpose"]["description"] == "Describe the call"
    assert schema["properties"]["priority"]["type"] == "integer"
    assert schema["required"] == ["value", "tool_call_purpose", "priority"]
    assert "Injected framework parameters" in description
    assert "- tool_call_purpose (string, required): Describe the call" in description
    assert "- priority (integer, required): Priority from 1 to 5" in description


@pytest.mark.asyncio
async def test_runtime_can_create_plan_plugin() -> None:
    runtime = CoreRuntime(model_name="test-model")

    plugins = await runtime._create_plugins(["plan"], {})

    assert len(plugins) == 1
    assert plugins[0].plugin_id == "plan"
    assert plugins[0].plugin_name == "PlanPlugin"


@pytest.mark.asyncio
async def test_runtime_passes_plan_folding_config() -> None:
    runtime = CoreRuntime(model_name="test-model")

    plugins = await runtime._create_plugins(
        ["plan"],
        {"plan": {"fold_completed_tasks": True}},
    )

    assert len(plugins) == 1
    state = plugins[0].list_plan_items()
    assert isinstance(state.output, dict)
    assert state.output["context_folding_enabled"] is True


@pytest.mark.asyncio
async def test_runtime_can_create_workflow_plugin() -> None:
    runtime = CoreRuntime(model_name="test-model")

    plugins = await runtime._create_plugins(["workflow"], {})

    assert len(plugins) == 1
    assert plugins[0].plugin_id == "workflow"
    assert plugins[0].plugin_name == "WorkflowPlugin"


class CapturingQueuedClient(QueuedJsonClient):
    def __init__(self) -> None:
        super().__init__(queue_max=1, client_id="capture")
        self.written: list[dict[str, Any]] = []

    async def _write_frame(self, frame: dict[str, Any]) -> None:
        self.written.append(frame)

    async def _close_transport(self) -> None:
        return None


@pytest.mark.asyncio
async def test_stdin_reader_uses_threaded_reader_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(builtin_gateways.sys, "platform", "win32")

    reader = await builtin_gateways._stdin_reader()

    assert isinstance(reader, builtin_gateways._ThreadedStdinReader)


@pytest.mark.asyncio
async def test_stdin_reader_falls_back_when_connect_read_pipe_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLoop:
        async def connect_read_pipe(self, *_: Any) -> None:
            raise OSError("pipe unavailable")

    monkeypatch.setattr(builtin_gateways.sys, "platform", "linux")
    monkeypatch.setattr(builtin_gateways.asyncio, "get_running_loop", lambda: FakeLoop())

    reader = await builtin_gateways._stdin_reader()

    assert isinstance(reader, builtin_gateways._ThreadedStdinReader)


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
    args = argparse.Namespace(
        host="127.0.0.1",
        port=unused_tcp_port,
        outbound_queue_size=10,
        max_frame_size=16 * 1024 * 1024,
    )
    server_task = asyncio.create_task(
        builtin_gateways.TcpGateway().serve(runtime, args)  # type: ignore[arg-type]
    )
    reader, writer = await _connect_tcp_with_retry(unused_tcp_port)

    try:
        ready = await _recv_tlv(reader)
        assert ready["type"] == "core.ready"

        await _send_tlv(
            writer,
            b'{"version":"hawi.core.v1","type":"ping","id":"ping-1","payload":{}}',
        )
        pong = await _recv_tlv(reader)
        assert pong["type"] == "pong"
        assert pong["id"] == "ping-1"

        await _send_tlv(
            writer,
            b'{"version":"hawi.core.v1","type":"shutdown","id":"stop-1","payload":{}}',
        )
        ack = await _recv_tlv(reader)
        assert ack["type"] == "ack"
        assert ack["id"] == "stop-1"

        await asyncio.wait_for(server_task, timeout=2)
    finally:
        writer.close()
        await writer.wait_closed()
        if not server_task.done():
            server_task.cancel()
            await asyncio.gather(server_task, return_exceptions=True)


# Plan 4 removed the standalone WebSocketGateway. The HTTP gateway's WS-upgrade
# path now provides the WebSocket carrier; see test_http_gateway.py for
# WS-upgrade integration coverage.
