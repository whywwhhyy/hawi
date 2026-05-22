from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

import hawi.engine.builtin_gateways as builtin_gateways
from hawi.engine.protocol import VERSION, make_ack, make_frame
from hawi.engine.runtime import CoreRuntime, load_model_configs, parse_extra_tool_parameter, parse_extra_tool_parameters
from hawi.engine.tlv import TYPE_JSON_FRAME, encode_frame, read_frame
from hawi.engine.transports import QueuedJsonClient
from hawi.agent import AutoCompactConfig, HawiAgent, AgentRunner
from hawi.agent.context import AgentContext
from hawi.models import model_registry
from hawi.plugin import HookContext
from hawi.session import SessionContextBranchResult, SessionManager
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


class DummyCompactionRecord:
    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": "manual summary",
            "replaced_messages": [],
            "kept_messages": 2,
            "tokens_before": 120,
            "tokens_after": 48,
            "created_at": 123.0,
        }


class DummyExecutor:
    state = DummyState()
    is_idle = True


class DummyAgent:
    def __init__(self) -> None:
        self.context = self
        self.model_name = ""
        self._auto_compact = AutoCompactConfig(
            max_context_tokens=1000,
            compression_budget=200,
        )
        self.loaded_steer: list[Any] | None = None
        self.loaded_runtime: dict[str, Any] | None = None
        self.compact_calls: list[dict[str, Any]] = []

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

    async def acompact(self, **kwargs: Any) -> DummyCompactionRecord:
        self.compact_calls.append(kwargs)
        return DummyCompactionRecord()


class DummyAgentRunner:
    state = DummyState()

    def __init__(self) -> None:
        self._executor = DummyExecutor()
        self.agent = DummyAgent()
        self.enqueued: list[tuple[Any, str, dict[str, Any]]] = []
        self.resumed = False

    @property
    def agent_state(self) -> DummyState:
        return self._executor.state

    @property
    def is_idle(self) -> bool:
        return self._executor.is_idle

    def get_queue_lengths(self) -> dict[str, int]:
        return {"normal": 0, "high_prio": 0, "urgent": 0}

    def has_pending_immediate_work(self) -> bool:
        return False

    def resume(self) -> None:
        self.resumed = True

    def control_snapshot(self) -> dict[str, Any]:
        return {"paused": False, "resumable": False}

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

    def submit_immediate_message(
        self,
        content: Any,
        *,
        intent: str,
        metadata: dict[str, Any],
    ) -> str:
        self.enqueued.append((content, "high_prio", {"intent": intent, **metadata}))
        return "msg-immediate"

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

    def export_markdown(
        self,
        session_id: str | None = None,
        *,
        model: str | None = None,
    ) -> Any:
        sid = session_id or self.current_session_id

        class Export:
            def to_dict(self, *, include_markdown: bool = True) -> dict[str, Any]:
                payload = {
                    "suggested_filename": f"{sid}.md",
                    "reference_dir_name": f"{sid}-ref",
                    "references": [],
                    "session_jsonl_path": f"/sessions/{sid}/message_history.jsonl",
                }
                if include_markdown:
                    payload["markdown"] = f"# {sid}\n\nmodel={model}\n"
                return payload

        return Export()

    def load_session(self, session_id: str) -> None:
        self.loaded = session_id
        self.current_session_id = session_id

    def new_session(self, name: str | None = None) -> str:
        self.new_session_name = name
        self.current_session_id = "new-session"
        return self.current_session_id

    def fork_session(
        self,
        session_id: str | None = None,
        name: str | None = None,
    ) -> str:
        self.forked_from = session_id or self.current_session_id
        self.fork_session_name = name
        self.current_session_id = "forked-session"
        self.histories["forked-session"] = list(self.histories[self.forked_from])
        return self.current_session_id

    def fork_session_after_message(
        self,
        *,
        session_id: str | None = None,
        name: str | None = None,
        after_message_index: int,
    ) -> SessionContextBranchResult:
        self.forked_after_message_index = after_message_index
        forked = self.fork_session(session_id=session_id, name=name)
        self.histories[forked] = self.histories[forked][:1]
        return SessionContextBranchResult(
            session_id=forked,
            source_session_id=self.forked_from,
            message_index=after_message_index,
            target_role="user",
            boundary_index=0,
            popped_user_message={
                "role": "user",
                "content": [{"type": "text", "text": "popped"}],
                "name": None,
                "metadata": None,
            },
        )

    def fork_session_after_message_id(
        self,
        *,
        session_id: str | None = None,
        name: str | None = None,
        context_message_id: str,
    ) -> SessionContextBranchResult:
        self.forked_after_context_message_id = context_message_id
        forked = self.fork_session(session_id=session_id, name=name)
        self.histories[forked] = self.histories[forked][:1]
        return SessionContextBranchResult(
            session_id=forked,
            source_session_id=self.forked_from,
            message_index=3,
            context_message_id=context_message_id,
            target_role="assistant",
            boundary_index=4,
        )

    def rewind_session_after_message(
        self,
        *,
        after_message_index: int,
    ) -> SessionContextBranchResult:
        self.rewound_after_message_index = after_message_index
        self.histories[self.current_session_id] = []
        return SessionContextBranchResult(
            session_id=self.current_session_id,
            source_session_id=self.current_session_id,
            message_index=after_message_index,
            target_role="user",
            boundary_index=0,
            popped_user_message={
                "role": "user",
                "content": [{"type": "text", "text": "rewound"}],
                "name": None,
                "metadata": None,
            },
        )

    def rewind_session_after_message_id(
        self,
        *,
        context_message_id: str,
    ) -> SessionContextBranchResult:
        self.rewound_after_context_message_id = context_message_id
        self.histories[self.current_session_id] = []
        return SessionContextBranchResult(
            session_id=self.current_session_id,
            source_session_id=self.current_session_id,
            message_index=2,
            context_message_id=context_message_id,
            target_role="assistant",
            boundary_index=3,
        )

    def save_now(self) -> None:
        self.saved_now = True

    def delete_session(self, session_id: str) -> None:
        self.deleted.append(session_id)
        self.histories.pop(session_id, None)

    def rename_session(self, session_id: str, name: str) -> None:
        self.renamed_session = (session_id, name)


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

    async def build_runner(**_: Any) -> tuple[DummyAgentRunner, asyncio.Task, list[Any]]:
        return DummyAgentRunner(), wait_forever, []

    async def stop_runner(*_: Any) -> None:
        wait_forever.cancel()
        await asyncio.gather(wait_forever, return_exceptions=True)

    runtime._build_runner = build_runner  # type: ignore[method-assign]
    runtime._stop_runner = stop_runner  # type: ignore[method-assign]

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
    runner = DummyAgentRunner()
    runtime._runner = runner  # type: ignore[assignment]

    await runtime.handle_frame(
        client,
        '{"version":"%s","type":"enqueue","id":"req","payload":{"content":"hi","queue":"high_prio"}}'
        % VERSION,
    )

    assert client.sent[-1]["type"] == "ack"
    assert client.sent[-1]["payload"]["message_id"] == "msg-123"
    assert runner.enqueued == [("hi", "high_prio", {"queue_kind": "high_prio"})]


@pytest.mark.asyncio
async def test_compact_context_command_runs_manual_compaction() -> None:
    runtime = CoreRuntime(model_name="test-model", token=None)
    client = FakeClient(authenticated=True)
    runner = DummyAgentRunner()
    sm = DummySessionManager()
    runtime._runner = runner  # type: ignore[assignment]
    runtime._session_manager = sm  # type: ignore[assignment]

    await runtime.handle_frame(
        client,
        '{"version":"%s","type":"compact_context","id":"compact","payload":{}}'
        % VERSION,
    )

    payload = client.sent[-1]["payload"]
    assert client.sent[-1]["type"] == "ack"
    assert payload["command"] == "compact_context"
    assert payload["status"] == "success"
    assert payload["record"]["summary"] == "manual summary"
    assert payload["session_id"] == "current-session"
    assert payload["message_history"][0]["content"][0]["text"] == "current"
    assert sm.saved_now is True
    assert runner.agent.compact_calls == [
        {"run_id": "manual-compact-compact", "mode": "manual"}
    ]


@pytest.mark.asyncio
async def test_compact_context_command_rejects_busy_runner() -> None:
    runtime = CoreRuntime(model_name="test-model", token=None)
    client = FakeClient(authenticated=True)
    runner = DummyAgentRunner()
    runner._executor.is_idle = False
    runtime._runner = runner  # type: ignore[assignment]
    runtime._session_manager = DummySessionManager()  # type: ignore[assignment]

    await runtime.handle_frame(
        client,
        '{"version":"%s","type":"compact_context","id":"compact","payload":{}}'
        % VERSION,
    )

    assert client.sent[-1]["type"] == "error"
    assert client.sent[-1]["payload"]["code"] == "busy"
    assert runner.agent.compact_calls == []


@pytest.mark.asyncio
async def test_set_auto_compact_command_updates_runtime_threshold() -> None:
    runtime = CoreRuntime(model_name="test-model", token=None)
    client = FakeClient(authenticated=True)
    runner = DummyAgentRunner()
    runtime._runner = runner  # type: ignore[assignment]

    await runtime.handle_frame(
        client,
        (
            '{"version":"%s","type":"set_auto_compact","id":"auto",'
            '"payload":{"trigger_tokens":720,"trigger_ratio":0.72}}'
        )
        % VERSION,
    )

    payload = client.sent[-1]["payload"]
    assert client.sent[-1]["type"] == "ack"
    assert payload["command"] == "set_auto_compact"
    assert payload["auto_compact"]["trigger_tokens"] == 720
    assert payload["auto_compact"]["trigger_ratio"] == 0.72
    assert payload["auto_compact"]["token_limit"] == 720
    assert payload["auto_compact"]["token_limit_ratio"] == 0.72
    assert runner.agent._auto_compact.trigger_tokens == 720


@pytest.mark.asyncio
async def test_set_auto_compact_rejects_threshold_past_context_window() -> None:
    runtime = CoreRuntime(model_name="test-model", token=None)
    client = FakeClient(authenticated=True)
    runner = DummyAgentRunner()
    runtime._runner = runner  # type: ignore[assignment]

    await runtime.handle_frame(
        client,
        (
            '{"version":"%s","type":"set_auto_compact","id":"auto",'
            '"payload":{"trigger_tokens":1001}}'
        )
        % VERSION,
    )

    assert client.sent[-1]["type"] == "error"
    assert "max_context_tokens" in client.sent[-1]["payload"]["message"]


@pytest.mark.asyncio
async def test_refresh_models_command_returns_provider_models(monkeypatch) -> None:
    runtime = CoreRuntime(model_name="test-model", token=None)
    client = FakeClient(authenticated=True)
    calls: list[str] = []

    def refresh_provider(provider: str) -> list[str]:
        calls.append(provider)
        return [f"{provider}/remote-a"]

    monkeypatch.setattr(model_registry, "refresh_provider_models", refresh_provider)
    monkeypatch.setattr(
        model_registry,
        "list_models",
        lambda: ["dynamic/local-a", "dynamic/remote-a"],
    )

    await runtime.handle_frame(
        client,
        '{"version":"%s","type":"refresh_models","id":"refresh","payload":{"provider":"dynamic"}}'
        % VERSION,
    )

    payload = client.sent[-1]["payload"]
    assert client.sent[-1]["type"] == "ack"
    assert payload["command"] == "refresh_models"
    assert payload["provider"] == "dynamic"
    assert payload["models"] == ["dynamic/remote-a"]
    assert payload["all_models"] == ["dynamic/local-a", "dynamic/remote-a"]
    assert calls == ["dynamic"]


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
async def test_session_export_markdown_command_returns_export_payload() -> None:
    runtime = CoreRuntime(model_name="test-model", token=None)
    client = FakeClient(authenticated=True)
    runtime._session_manager = DummySessionManager()  # type: ignore[assignment]

    await runtime.handle_frame(
        client,
        '{"version":"%s","type":"session_export_markdown","id":"export","payload":{}}'
        % VERSION,
    )

    payload = client.sent[-1]["payload"]
    assert payload["command"] == "session_export_markdown"
    assert payload["session_id"] == "current-session"
    assert payload["export"]["suggested_filename"] == "current-session.md"
    assert payload["export"]["markdown"].startswith("# current-session")


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
async def test_session_rename_updates_session_manager() -> None:
    runtime = CoreRuntime(model_name="test-model", token=None)
    client = FakeClient(authenticated=True)
    sm = DummySessionManager()
    runtime._session_manager = sm  # type: ignore[assignment]

    await runtime.handle_frame(
        client,
        (
            '{"version":"%s","type":"session_rename","id":"rename",'
            '"payload":{"session_id":"saved-session","name":"  Renamed  "}}'
        )
        % VERSION,
    )

    payload = client.sent[-1]["payload"]
    assert client.sent[-1]["type"] == "ack"
    assert payload["command"] == "session_rename"
    assert payload["session_id"] == "saved-session"
    assert payload["name"] == "Renamed"
    assert sm.renamed_session == ("saved-session", "Renamed")


@pytest.mark.asyncio
async def test_session_new_resets_live_state_without_materializing_history() -> None:
    runtime = CoreRuntime(model_name="test-model", token=None)
    client = FakeClient(authenticated=True)
    runner = DummyAgentRunner()
    sm = DummySessionManager()
    runtime._runner = runner  # type: ignore[assignment]
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
    assert runner.agent.cleared is True
    assert runner.agent.loaded_steer == []
    assert runner.agent.loaded_runtime["current_tool_calls"] == []


@pytest.mark.asyncio
async def test_session_fork_command_returns_forked_history() -> None:
    runtime = CoreRuntime(model_name="test-model", token=None)
    client = FakeClient(authenticated=True)
    runner = DummyAgentRunner()
    sm = DummySessionManager()
    runtime._runner = runner  # type: ignore[assignment]
    runtime._session_manager = sm  # type: ignore[assignment]

    await runtime.handle_frame(
        client,
        (
            '{"version":"%s","type":"session_fork","id":"fork",'
            '"payload":{"session_id":"saved-session","name":"copy"}}'
        )
        % VERSION,
    )

    payload = client.sent[-1]["payload"]
    assert client.sent[-1]["type"] == "ack"
    assert payload["command"] == "session_fork"
    assert payload["session_id"] == "forked-session"
    assert payload["forked_from_session_id"] == "saved-session"
    assert payload["message_history"][0]["content"][0]["text"] == "saved"
    assert sm.fork_session_name == "copy"


@pytest.mark.asyncio
async def test_session_fork_command_accepts_message_index() -> None:
    runtime = CoreRuntime(model_name="test-model", token=None)
    client = FakeClient(authenticated=True)
    runner = DummyAgentRunner()
    sm = DummySessionManager()
    runtime._runner = runner  # type: ignore[assignment]
    runtime._session_manager = sm  # type: ignore[assignment]

    await runtime.handle_frame(
        client,
        (
            '{"version":"%s","type":"session_fork","id":"fork",'
            '"payload":{"session_id":"saved-session","message_index":2}}'
        )
        % VERSION,
    )

    payload = client.sent[-1]["payload"]
    assert client.sent[-1]["type"] == "ack"
    assert payload["command"] == "session_fork"
    assert payload["session_id"] == "forked-session"
    assert payload["message_index"] == 2
    assert payload["target_role"] == "user"
    assert payload["popped_user_text"] == "popped"
    assert sm.forked_after_message_index == 2
    assert sm.saved_now is True
    assert runner.agent.loaded_steer == []
    assert runner.agent.loaded_runtime["current_tool_calls"] == []


@pytest.mark.asyncio
async def test_session_fork_command_accepts_context_message_id() -> None:
    runtime = CoreRuntime(model_name="test-model", token=None)
    client = FakeClient(authenticated=True)
    runner = DummyAgentRunner()
    sm = DummySessionManager()
    runtime._runner = runner  # type: ignore[assignment]
    runtime._session_manager = sm  # type: ignore[assignment]

    await runtime.handle_frame(
        client,
        (
            '{"version":"%s","type":"session_fork","id":"fork",'
            '"payload":{"session_id":"saved-session","context_message_id":"ctxmsg-a"}}'
        )
        % VERSION,
    )

    payload = client.sent[-1]["payload"]
    assert client.sent[-1]["type"] == "ack"
    assert payload["command"] == "session_fork"
    assert payload["session_id"] == "forked-session"
    assert payload["context_message_id"] == "ctxmsg-a"
    assert payload["message_index"] == 3
    assert payload["target_role"] == "assistant"
    assert sm.forked_after_context_message_id == "ctxmsg-a"
    assert sm.saved_now is True


@pytest.mark.asyncio
async def test_session_rewind_command_returns_popped_user_message() -> None:
    runtime = CoreRuntime(model_name="test-model", token=None)
    client = FakeClient(authenticated=True)
    runner = DummyAgentRunner()
    sm = DummySessionManager()
    runtime._runner = runner  # type: ignore[assignment]
    runtime._session_manager = sm  # type: ignore[assignment]

    await runtime.handle_frame(
        client,
        (
            '{"version":"%s","type":"session_rewind","id":"rewind",'
            '"payload":{"message_index":1}}'
        )
        % VERSION,
    )

    payload = client.sent[-1]["payload"]
    assert client.sent[-1]["type"] == "ack"
    assert payload["command"] == "session_rewind"
    assert payload["session_id"] == "current-session"
    assert payload["message_index"] == 1
    assert payload["popped_user_text"] == "rewound"
    assert payload["message_history"] == []
    assert sm.rewound_after_message_index == 1
    assert sm.saved_now is True


@pytest.mark.asyncio
async def test_session_rewind_command_accepts_context_message_id() -> None:
    runtime = CoreRuntime(model_name="test-model", token=None)
    client = FakeClient(authenticated=True)
    runner = DummyAgentRunner()
    sm = DummySessionManager()
    runtime._runner = runner  # type: ignore[assignment]
    runtime._session_manager = sm  # type: ignore[assignment]

    await runtime.handle_frame(
        client,
        (
            '{"version":"%s","type":"session_rewind","id":"rewind",'
            '"payload":{"context_message_id":"ctxmsg-b"}}'
        )
        % VERSION,
    )

    payload = client.sent[-1]["payload"]
    assert client.sent[-1]["type"] == "ack"
    assert payload["command"] == "session_rewind"
    assert payload["session_id"] == "current-session"
    assert payload["context_message_id"] == "ctxmsg-b"
    assert payload["message_index"] == 2
    assert payload["target_role"] == "assistant"
    assert sm.rewound_after_context_message_id == "ctxmsg-b"
    assert sm.saved_now is True


@pytest.mark.asyncio
async def test_session_switch_after_runner_replace_uses_live_event_bus(tmp_path) -> None:
    runtime = CoreRuntime(model_name="test-model", token=None, status_interval=60)
    client = FakeClient(authenticated=True)
    runtime._loop = asyncio.get_running_loop()

    old_runner = AgentRunner(HawiAgent(model=MagicMock()))
    old_task = asyncio.create_task(asyncio.Event().wait())
    runtime._runner = old_runner
    runtime._runner_task = old_task
    runtime._plugins = []

    sm = SessionManager(root=tmp_path / "sessions")
    sm.attach(old_runner.agent, old_runner, event_bus=old_runner.agent.event_bus)
    runtime._session_manager = sm

    current_session_id = sm.new_session(name="current")
    old_runner.agent.context.add_user_message("current")
    sm.save_now()

    old_runner.agent.context.clear()
    saved_session_id = sm.new_session(name="saved")
    old_runner.agent.context.add_user_message("saved")
    sm.save_now()

    sm.load_session(current_session_id)

    new_runner = AgentRunner(HawiAgent(model=MagicMock()))
    new_task = asyncio.create_task(asyncio.Event().wait())

    async def build_runner(**_: Any) -> tuple[AgentRunner, asyncio.Task, list[Any]]:
        return new_runner, new_task, []

    runtime._build_runner = build_runner  # type: ignore[method-assign]

    try:
        await runtime._replace_runner(
            model_name="test-model",
            selected_plugins=[],
            plugin_configs={},
            preserve_context=old_runner.agent.context.copy(),
        )

        await runtime.handle_frame(
            client,
            (
                '{"version":"%s","type":"session_switch","id":"switch",'
                '"payload":{"session_id":"%s"}}'
            )
            % (VERSION, saved_session_id),
        )

        assert client.sent[-1]["type"] == "ack"
        assert client.sent[-1]["payload"]["command"] == "session_switch"
        assert client.sent[-1]["payload"]["session_id"] == saved_session_id
    finally:
        sm.detach()
        await runtime._stop_runner(runtime._runner, runtime._runner_task, [])
        runtime._runner = None
        runtime._runner_task = None


def test_status_payload_includes_queue_messages() -> None:
    runtime = CoreRuntime(model_name="test-model", token=None)
    runner = DummyAgentRunner()
    runner.get_queue_messages = lambda: {  # type: ignore[method-assign]
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
    runner.agent.get_pending_input_messages = lambda: [  # type: ignore[attr-defined]
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
    runtime._runner = runner  # type: ignore[assignment]

    payload = runtime._status_payload()

    assert payload["queue_messages"]["normal"][0]["content_preview"] == "queued"
    assert payload["queue_messages"]["normal"][1]["content_preview"] == "pending plain steer"
    assert payload["queue_messages"]["high_prio"][0]["content_preview"] == "pending steer"
    assert payload["queue_messages"]["urgent"] == []
    assert payload["auto_compact"]["enabled"] is True
    assert payload["auto_compact"]["max_context_tokens"] == 1000
    assert payload["auto_compact"]["token_limit"] == 800
    assert payload["auto_compact"]["token_limit_ratio"] == 0.8


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
    from hawi.engine.__main__ import build_parser

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
async def test_runtime_passes_explicit_auto_compact_config_from_model_window(monkeypatch: pytest.MonkeyPatch) -> None:
    model = MagicMock()
    model.get_max_context_tokens.return_value = 64_000
    monkeypatch.setattr(model_registry, "create_model", lambda *args, **kwargs: model)
    runtime = CoreRuntime(model_name="test-model", token=None)

    runner, runner_task, _ = await runtime._build_runner(
        model_name="test-model",
        selected_plugins=[],
        plugin_configs={},
        context_to_restore=None,
    )
    try:
        assert runner.agent._auto_compact.max_context_tokens == 64_000
    finally:
        runner.stop()
        runner_task.cancel()
        await asyncio.gather(runner_task, return_exceptions=True)


def test_load_model_configs_can_skip_user_config(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    home_config = home / ".hawi" / "models.yaml"
    workspace_config = workspace / ".hawi" / "models.yaml"
    root_config = workspace / "models.yaml"
    for path in (home_config, workspace_config, root_config):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("providers: {}\n", encoding="utf-8")

    loaded_paths: list[str] = []
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(
        "hawi.engine.runtime.model_registry._auto_load_needed",
        True,
    )
    monkeypatch.setattr(
        "hawi.engine.runtime.model_registry.load_config",
        lambda path, quiet=True: loaded_paths.append(str(path)),
    )

    loaded = load_model_configs(include_user=False)

    assert loaded == [workspace_config, root_config]
    assert loaded_paths == [str(workspace_config), str(root_config)]


def test_load_model_configs_uses_git_root_from_nested_cwd(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    nested = workspace / "src" / "package"
    workspace_config = workspace / ".hawi" / "models.yaml"
    root_config = workspace / "models.yaml"
    nested.mkdir(parents=True)
    (workspace / ".git").mkdir()
    for path in (workspace_config, root_config):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("providers: {}\n", encoding="utf-8")

    loaded_paths: list[str] = []
    monkeypatch.chdir(nested)
    monkeypatch.setattr(
        "hawi.engine.runtime.model_registry._auto_load_needed",
        True,
    )
    monkeypatch.setattr(
        "hawi.engine.runtime.model_registry.load_config",
        lambda path, quiet=True: loaded_paths.append(str(path)),
    )

    loaded = load_model_configs(include_user=False)

    assert loaded == [workspace_config, root_config]
    assert loaded_paths == [str(workspace_config), str(root_config)]


def test_load_model_configs_chains_workspace_then_user_config(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    extra_config = tmp_path / "extra.yaml"
    home_config = home / ".hawi" / "models.yaml"
    workspace_config = workspace / ".hawi" / "models.yaml"
    root_config = workspace / "models.yaml"
    for path in (home_config, workspace_config, root_config, extra_config):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("providers: {}\n", encoding="utf-8")

    loaded_paths: list[str] = []
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(
        "hawi.engine.runtime.model_registry._auto_load_needed",
        True,
    )
    monkeypatch.setattr(
        "hawi.engine.runtime.model_registry.load_config",
        lambda path, quiet=True: loaded_paths.append(str(path)),
    )

    loaded = load_model_configs(extra_paths=[str(extra_config)])

    assert loaded == [workspace_config, root_config, home_config, extra_config]
    assert loaded_paths == [str(path) for path in loaded]
    assert not model_registry._auto_load_needed


@pytest.mark.asyncio
async def test_runtime_can_create_plan_plugin() -> None:
    runtime = CoreRuntime(model_name="test-model")

    plugins = await runtime._create_plugins(["hawi/plan"], {})

    assert len(plugins) == 1
    assert plugins[0].plugin_id == "hawi/plan"
    assert plugins[0].plugin_name == "Plan"


@pytest.mark.asyncio
async def test_runtime_can_create_taskflow_plugin() -> None:
    runtime = CoreRuntime(model_name="test-model")

    plugins = await runtime._create_plugins(["hawi/taskflow"], {})

    assert len(plugins) == 1
    assert plugins[0].plugin_id == "hawi/taskflow"
    assert plugins[0].plugin_name == "Taskflow"


@pytest.mark.asyncio
async def test_runtime_resume_continues_existing_high_prio_work_without_prompt() -> None:
    class PendingRunner(DummyAgentRunner):
        def has_pending_immediate_work(self) -> bool:
            return True

    runtime = CoreRuntime(model_name="test-model")
    runner = PendingRunner()
    runtime._runner = runner  # type: ignore[assignment]
    client = FakeClient(authenticated=True)

    await runtime.handle_command(
        client,
        argparse.Namespace(type="resume", id="cmd-resume", payload={}),
    )

    assert runner.resumed is True
    assert runner.enqueued == []
    assert client.sent[-1]["payload"] == {
        "command": "resume",
        "ok": True,
        "message_id": None,
        "queue": None,
        "resumed_existing_work": True,
        "control": {"paused": False, "resumable": False},
    }


@pytest.mark.asyncio
async def test_runtime_default_resume_prompt_skips_before_conversation_hooks() -> None:
    runtime = CoreRuntime(model_name="test-model")
    runner = DummyAgentRunner()
    runtime._runner = runner  # type: ignore[assignment]
    client = FakeClient(authenticated=True)

    await runtime.handle_command(
        client,
        argparse.Namespace(type="resume", id="cmd-resume", payload={}),
    )

    assert runner.enqueued
    _, queue, metadata = runner.enqueued[-1]
    assert queue == "high_prio"
    assert metadata["intent"] == "resume"
    assert metadata["display_message_type"] == "resume"
    assert metadata["auto_generated"] is True
    assert metadata["skip_before_conversation_hooks"] is True


@pytest.mark.asyncio
async def test_runtime_plugin_action_approves_taskflow_review_and_resumes() -> None:
    from hawi.builtin_plugins.taskflow_plugin import TaskflowPlugin

    runtime = CoreRuntime(model_name="test-model")
    runner = DummyAgentRunner()
    runtime._runner = runner  # type: ignore[assignment]
    plugin = TaskflowPlugin()
    plugin.bind_plugin_identity(plugin_id="hawi/taskflow", plugin_name="Taskflow")
    runtime._plugins = [plugin]
    client = FakeClient(authenticated=True)

    created = plugin.create_taskflow(
        title="Human Flow",
        mode="workflow",
        execution_policy="gated_graph",
        mutable=False,
        start_step_id="review",
        steps=[
            {"id": "review", "title": "Review", "review": {"type": "human"}},
            {"id": "next", "title": "Next"},
        ],
        edges=[{"from": "review", "to": "next", "type": "transitions"}],
    )
    assert created.success is True
    assert plugin.start_taskflow().success is True
    submitted = plugin.submit_taskflow_step(output="ready")
    assert submitted.success is True
    review_id = next(iter(plugin._pending_human_reviews))

    await runtime.handle_command(
        client,
        argparse.Namespace(
            type="plugin_action",
            id="cmd-approve",
            payload={
                "plugin_id": "hawi/taskflow",
                "action": "approve_taskflow_review",
                "arguments": {"review_id": review_id, "feedback": "Approved"},
            },
        ),
    )

    assert client.sent[-1]["type"] == "ack"
    assert client.sent[-1]["payload"]["command"] == "plugin_action"
    assert client.sent[-1]["payload"]["resume_message_id"] == "msg-immediate"
    assert runner.enqueued
    assert runner.enqueued[-1][1] == "high_prio"
    assert "Entering next step" in runner.enqueued[-1][0]


@pytest.mark.asyncio
async def test_runtime_review_action_resolves_blocking_taskflow_review_without_resume_enqueue() -> None:
    from hawi.builtin_plugins.taskflow_plugin import TaskflowPlugin

    runtime = CoreRuntime(model_name="test-model")
    runner = DummyAgentRunner()
    runtime._runner = runner  # type: ignore[assignment]
    plugin = TaskflowPlugin()
    plugin.bind_plugin_identity(plugin_id="hawi/taskflow", plugin_name="Taskflow")
    runtime._plugins = [plugin]
    client = FakeClient(authenticated=True)

    created = plugin.create_taskflow(
        title="Blocking Human Flow",
        mode="workflow",
        execution_policy="gated_graph",
        mutable=False,
        start_step_id="review",
        steps=[
            {"id": "review", "title": "Review", "review": {"type": "human"}},
            {"id": "next", "title": "Next"},
        ],
        edges=[{"from": "review", "to": "next", "type": "transitions"}],
    )
    assert created.success is True
    assert plugin.start_taskflow().success is True

    runtime_tool_ctx = argparse.Namespace(
        context=AgentContext(),
        review=runtime._review_broker,
    )
    submitted = plugin.submit_taskflow_step(output="ready", ctx=runtime_tool_ctx)
    assert submitted.success is True
    assert submitted.output["review_pending"] is True
    assert not plugin._pending_human_reviews

    hook_task = asyncio.create_task(
        plugin.review_submitted_step(
            agent=None,
            tool_name="submit_taskflow_step",
            arguments={},
            result=submitted,
            ctx=HookContext(
                run_id="run-review",
                iteration=1,
                tool_call_id="tc-submit",
                context=runtime_tool_ctx.context,
                review=runtime._review_broker,
            ),
        )
    )
    for _ in range(20):
        if plugin._pending_human_reviews:
            break
        await asyncio.sleep(0.01)
    assert plugin._pending_human_reviews
    review_id = next(iter(plugin._pending_human_reviews))

    await runtime.handle_command(
        client,
        argparse.Namespace(
            type="plugin_action",
            id="cmd-approve",
            payload={
                "plugin_id": "hawi/taskflow",
                "action": "approve_taskflow_review",
                "arguments": {"review_id": review_id, "feedback": "Approved"},
            },
        ),
    )

    hook_result = await asyncio.wait_for(hook_task, timeout=1)
    assert hook_result is not None
    assert "Entering next step" in str(hook_result.message)
    assert client.sent[-1]["type"] == "ack"
    assert client.sent[-1]["payload"]["resume_message_id"] is None
    assert runner.enqueued == []
    assert submitted.output["approved"] is True
    assert plugin.get_taskflow_status().output["run"]["current_step_id"] == "next"


def test_runtime_expands_selected_plugin_dependencies() -> None:
    runtime = CoreRuntime(model_name="test-model", selected_plugins=["hawi/skills"])

    assert runtime._selected_plugins == [
        "hawi/filesystem",
        "hawi/shell",
        "hawi/skills",
    ]


@pytest.mark.asyncio
async def test_runtime_passes_plan_folding_config() -> None:
    runtime = CoreRuntime(model_name="test-model")

    plugins = await runtime._create_plugins(
        ["hawi/plan"],
        {"hawi/plan": {"fold_completed_tasks": True}},
    )

    assert len(plugins) == 1
    state = plugins[0].list_plan_items()
    assert isinstance(state.output, dict)
    assert state.output["context_folding_enabled"] is True


@pytest.mark.asyncio
async def test_runtime_can_create_workflow_plugin() -> None:
    runtime = CoreRuntime(model_name="test-model")

    plugins = await runtime._create_plugins(["hawi/workflow"], {})

    assert len(plugins) == 1
    assert plugins[0].plugin_id == "hawi/workflow"
    assert plugins[0].plugin_name == "Workflow"


@pytest.mark.asyncio
async def test_runtime_can_create_subagent_plugin() -> None:
    runtime = CoreRuntime(model_name="test-model")

    plugins = await runtime._create_plugins(["hawi/subagent"], {})

    assert len(plugins) == 1
    assert plugins[0].plugin_id == "hawi/subagent"
    assert plugins[0].plugin_name == "Subagent"


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


@pytest.mark.asyncio
async def test_tcp_gateway_rejects_oversized_frame_and_closes(unused_tcp_port: int) -> None:
    runtime = FakeTransportRuntime()
    args = argparse.Namespace(
        host="127.0.0.1",
        port=unused_tcp_port,
        outbound_queue_size=10,
        max_frame_size=8,
    )
    server_task = asyncio.create_task(
        builtin_gateways.TcpGateway().serve(runtime, args)  # type: ignore[arg-type]
    )
    reader, writer = await _connect_tcp_with_retry(unused_tcp_port)

    try:
        ready = await _recv_tlv(reader)
        assert ready["type"] == "core.ready"

        writer.write(encode_frame(TYPE_JSON_FRAME, b"x" * 9))
        await writer.drain()

        err = await _recv_tlv(reader)
        assert err["type"] == "error"
        assert err["payload"]["code"] == "frame_too_large"

        result = await asyncio.wait_for(read_frame(reader), timeout=2)
        assert result is None
        for _ in range(20):
            if not runtime.clients:
                break
            await asyncio.sleep(0.01)
        assert runtime.clients == set()
    finally:
        runtime._shutdown.set()
        writer.close()
        await writer.wait_closed()
        if not server_task.done():
            await asyncio.gather(server_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_tcp_gateway_discards_binary_and_unknown_frames_then_continues(unused_tcp_port: int) -> None:
    from hawi.engine.tlv import TYPE_BINARY_BLOB

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

        writer.write(encode_frame(TYPE_BINARY_BLOB, b"opaque"))
        writer.write(encode_frame(0x42, b"future"))
        await writer.drain()

        await _send_tlv(
            writer,
            b'{"version":"hawi.core.v1","type":"ping","id":"ping-after-unknown","payload":{}}',
        )
        pong = await _recv_tlv(reader)
        assert pong["type"] == "pong"
        assert pong["id"] == "ping-after-unknown"
    finally:
        runtime._shutdown.set()
        writer.close()
        await writer.wait_closed()
        if not server_task.done():
            await asyncio.gather(server_task, return_exceptions=True)


# Plan 4 removed the standalone WebSocketGateway. The HTTP gateway's WS-upgrade
# path now provides the WebSocket carrier; see test_http_gateway.py for
# WS-upgrade integration coverage.
