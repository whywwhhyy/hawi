"""Tests for hawi.session: layout, writer, manager.

Component snapshots are tested in their own files (test_queue_manager_snapshot,
test_agent_context_snapshot). This file focuses on the SessionManager glue
and the writer thread's atomicity / backpressure / failure semantics.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest

from hawi.agent import HawiAgent
from hawi.agent.context import AgentContext
from hawi.agent.runner.queue import MessageQueueManager
from hawi.errors import DeniedError, ToolExecutionError
from hawi.events import (
    AgentContextInjectedEvent,
    AgentErrorEvent,
    AgentCompactStartEvent,
    AgentCompactStopEvent,
    AgentInterruptEvent,
    AgentMessageAddedEvent,
    AgentSystemPromptEvent,
    AgentToolRuntimeContextInjectedEvent,
    EventBus,
    ModelErrorEvent,
    ModelRetryEvent,
    AgentRunnerInterruptEvent,
    PluginEvent,
    SessionWriteFailedEvent,
    SubAgentEvent,
)
from hawi.models import Model
from hawi.models.message import DeltaPart, MessageRequest, MessageResponse, TokenUsage
from hawi.plugin import HawiPlugin, HookContext, before_conversation
from hawi.session import SessionLockedError, SessionManager, SessionWriter, WriteJob
from hawi.session import layout
from hawi.session.lock import SessionFileLock, make_lock_metadata
from hawi.utils.lifecycle import (
    EXIT_PRIORITY_NORMAL,
    EXIT_PRIORITY_PLUGIN_TEARDOWN,
    EXIT_PRIORITY_SESSION_FLUSH,
    ExitHandler,
)


# ---------------------------------------------------------------------------
# Fixtures: stubs that mimic the agent / runner surface SessionManager uses
# ---------------------------------------------------------------------------


class _StubAgentRunner:
    def __init__(self) -> None:
        self._queue_manager = MessageQueueManager()

    def control_snapshot(self) -> dict[str, Any]:
        return {"paused": False, "resumable": False}


class _StubPluginManager:
    def __init__(self, plugins: list | None = None) -> None:
        self.plugins = plugins or []


class _StubAgent:
    def __init__(self, plugins: list | None = None) -> None:
        self.context = AgentContext()
        self._plugin_manager = _StubPluginManager(plugins=plugins)
        self.event_bus = EventBus()

    def snapshot_runtime(self) -> dict:
        return {
            "version": 1,
            "active_run_id": None,
            "iteration": 0,
            "current_tool_calls": [],
            "interrupted_tool_call_ids": [],
            "last_unsent_tool_results": [],
            "last_interrupt_reason": None,
        }

    def load_runtime(self, data: dict) -> None:  # pragma: no cover - exercised below
        pass

    def snapshot_steer(self) -> list:
        return []

    def load_steer(self, data: list) -> None:  # pragma: no cover - exercised below
        pass


class _PartialStreamingModel(Model):
    default_steer_merge_mode = "user_message_template"

    def __init__(self) -> None:
        super().__init__()
        self.delta_processed = asyncio.Event()

    @property
    def model_id(self) -> str:
        return "partial-streaming-model"

    def _prepare_request_impl(self, request: MessageRequest) -> dict[str, Any]:
        return {}

    def _parse_response_impl(self, response: dict[str, Any]) -> MessageResponse:
        return MessageResponse(
            id="response",
            content=[{"type": "text", "text": "complete"}],
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=1, output_tokens=1),
        )

    def _invoke_impl(self, request: MessageRequest) -> MessageResponse:
        return self._parse_response_impl({})

    async def _astream_impl(
        self,
        request: MessageRequest,
    ) -> AsyncGenerator[DeltaPart, None]:
        yield {
            "type": "text_delta",
            "index": 0,
            "delta": "",
            "is_start": True,
            "is_end": False,
        }
        yield {
            "type": "text_delta",
            "index": 0,
            "delta": "half answer",
            "is_start": False,
            "is_end": False,
        }
        self.delta_processed.set()
        await asyncio.sleep(60)


class _PromptHookPlugin(HawiPlugin):
    def __init__(self) -> None:
        self.calls = 0

    @before_conversation(system_prompt_variability="hardcoded")
    def inject_system_prompt(self, agent, ctx) -> None:
        self.calls += 1
        system_prompt = list(agent.context.system_prompt or [])
        system_prompt.append({"type": "text", "text": "regenerated prompt"})
        agent.context.system_prompt = system_prompt


@pytest.fixture
def session_root(tmp_path: Path) -> Path:
    return tmp_path / "sessions"


@pytest.fixture
def stub_setup(session_root: Path):
    agent = _StubAgent()
    runner = _StubAgentRunner()
    sm = SessionManager(root=session_root)
    sm.attach(agent, runner, event_bus=agent.event_bus)
    yield sm, agent, runner
    sm.detach()


# ---------------------------------------------------------------------------
# layout.atomic_write_text
# ---------------------------------------------------------------------------


class TestAtomicWrite:
    def test_writes_atomically(self, tmp_path: Path) -> None:
        target = tmp_path / "x.json"
        layout.atomic_write_text(target, '{"k": 1}', fsync=False)
        assert target.read_text() == '{"k": 1}'

    def test_overwrites_existing(self, tmp_path: Path) -> None:
        target = tmp_path / "x.json"
        layout.atomic_write_text(target, "first", fsync=False)
        layout.atomic_write_text(target, "second", fsync=False)
        assert target.read_text() == "second"

    def test_no_tmp_files_left_on_success(self, tmp_path: Path) -> None:
        target = tmp_path / "x.json"
        layout.atomic_write_text(target, "ok", fsync=False)
        leftovers = list(tmp_path.glob("x.json.*.tmp"))
        assert leftovers == []

    def test_tmp_cleaned_up_on_replace_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "x.json"

        def boom(src, dst):
            raise OSError("simulated")

        monkeypatch.setattr("os.replace", boom)
        with pytest.raises(OSError):
            layout.atomic_write_text(target, "ok", fsync=False)
        leftovers = list(tmp_path.glob("x.json.*.tmp"))
        assert leftovers == []
        assert not target.exists()


# ---------------------------------------------------------------------------
# SessionWriter
# ---------------------------------------------------------------------------


class TestSessionWriter:
    def test_writes_each_component_file(self, tmp_path: Path) -> None:
        writer = SessionWriter()
        writer.start()
        try:
            sd = tmp_path / "session-1"
            writer.submit(
                WriteJob(
                    session_dir=sd,
                    snapshots={
                        layout.COMPONENT_CONTEXT: {"version": "1.0", "messages": []},
                        layout.COMPONENT_QUEUES: {"version": 1, "runner": {}},
                        layout.COMPONENT_RUNTIME: {"version": 1, "iteration": 0},
                    },
                    fsync=True,
                )
            )
            assert writer.wait_idle(timeout=3.0)
        finally:
            writer.shutdown()
        assert layout.context_path(sd).exists()
        assert layout.queues_path(sd).exists()
        assert layout.runtime_path(sd).exists()
        assert layout.manifest_path(sd).exists()

    def test_appends_message_history_incrementally(self, tmp_path: Path) -> None:
        writer = SessionWriter()
        writer.start()
        try:
            sd = tmp_path / "session-1"
            writer.submit(
                WriteJob(
                    session_dir=sd,
                    message_history_entries=[
                        {
                            "version": 1,
                            "run_id": "r1",
                            "role": "user",
                            "content": [{"type": "text", "text": "hello"}],
                            "metadata": None,
                        }
                    ],
                    manifest_patch={"session_id": "session-1"},
                )
            )
            writer.submit(
                WriteJob(
                    session_dir=sd,
                    message_history_entries=[
                        {
                            "version": 1,
                            "run_id": "r1",
                            "role": "assistant",
                            "content": [{"type": "text", "text": "hi"}],
                            "metadata": None,
                        }
                    ],
                    manifest_patch={"session_id": "session-1"},
                )
            )
            assert writer.wait_idle(timeout=3.0)
        finally:
            writer.shutdown()

        entries = layout.read_jsonl(layout.message_history_path(sd))
        assert [entry["role"] for entry in entries] == ["user", "assistant"]
        manifest = json.loads(layout.manifest_path(sd).read_text())
        assert layout.COMPONENT_MESSAGE_HISTORY in manifest["components_present"]

    def test_drop_oldest_under_backpressure(self, tmp_path: Path) -> None:
        writer = SessionWriter(max_queue_size=2)
        # Don't start the thread — fill the queue manually so we can test the
        # drop policy without races.
        sd = tmp_path / "s"

        def make_job(label: str, components: tuple[str, ...]) -> WriteJob:
            return WriteJob(
                session_dir=sd,
                snapshots={c: {"label": label} for c in components},
                component_set_key=",".join(sorted(components)),
            )

        writer.submit(make_job("old1", (layout.COMPONENT_CONTEXT,)))
        writer.submit(make_job("old2", (layout.COMPONENT_CONTEXT,)))
        # Queue is now full. Submitting another for the same component-set
        # must drop the oldest matching job.
        writer.submit(make_job("new", (layout.COMPONENT_CONTEXT,)))

        # Drain manually, snapshot the labels left in the queue.
        items = []
        while not writer._queue.empty():
            items.append(writer._queue.get_nowait())
        labels = [
            item.snapshots[layout.COMPONENT_CONTEXT]["label"]
            for item in items
            if isinstance(item, WriteJob)
        ]
        # Oldest "old1" got dropped, "old2" + "new" remain.
        assert "old1" not in labels
        assert "old2" in labels
        assert "new" in labels

    def test_failure_emits_event_and_keeps_thread_alive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bus = EventBus()
        captured: list[SessionWriteFailedEvent] = []

        def collect(ev):
            captured.append(ev)

        bus.subscribe_blocking(collect, ["session.write_failed"])
        writer = SessionWriter(event_bus=bus)
        writer.start()
        try:
            calls = {"n": 0}

            def fail_once(*a, **kw):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise OSError("boom")

            monkeypatch.setattr(layout, "atomic_write_text", fail_once)

            writer.submit(
                WriteJob(
                    session_dir=tmp_path / "s",
                    snapshots={layout.COMPONENT_CONTEXT: {"v": 1}},
                )
            )
            writer.wait_idle(timeout=3.0)
            # Subsequent submission still gets handled.
            writer.submit(
                WriteJob(
                    session_dir=tmp_path / "s",
                    snapshots={layout.COMPONENT_RUNTIME: {"v": 2}},
                )
            )
            writer.wait_idle(timeout=3.0)
        finally:
            writer.shutdown()

        # Allow the bus to dispatch.
        time.sleep(0.05)
        assert any(ev.component == layout.COMPONENT_CONTEXT for ev in captured)


# ---------------------------------------------------------------------------
# SessionManager — high-level
# ---------------------------------------------------------------------------


class TestSessionManager:
    def test_new_session_is_lazy(self, stub_setup) -> None:
        sm, _, _ = stub_setup
        sid = sm.new_session(name="alpha")
        sd = layout.session_dir(sm._root, sid)
        assert sm.current_session_id == sid
        assert not sd.exists()
        assert sm.list_sessions() == []

    def test_new_session_id_includes_time_to_seconds(self, stub_setup) -> None:
        sm, _, _ = stub_setup
        sid = sm.new_session(name="alpha")

        assert re.fullmatch(r"session-\d{8}-\d{6}-[0-9a-f]{6}", sid)

    def test_new_session_accepts_gui_supplied_id(self, stub_setup) -> None:
        sm, _, _ = stub_setup

        sid = sm.new_session(name="alpha", session_id="session-custom-gui")

        assert sid == "session-custom-gui"
        assert sm.current_session_id == "session-custom-gui"

    def test_save_now_skips_empty_session(self, stub_setup) -> None:
        sm, _, _ = stub_setup
        sid = sm.new_session(name="empty")
        sm.save_now()
        assert not layout.session_dir(sm._root, sid).exists()
        assert sm.list_sessions() == []

    def test_save_now_round_trip(self, session_root: Path) -> None:
        agent = _StubAgent()
        runner = _StubAgentRunner()
        sm = SessionManager(root=session_root)
        sm.attach(agent, runner, event_bus=agent.event_bus)
        try:
            agent.context.add_user_message("hello session")
            sid = sm.new_session(name="rt")
            sm.save_now()
            assert layout.context_path(layout.session_dir(session_root, sid)).exists()
        finally:
            sm.detach()

        # Fresh agent picks up the persisted state.
        agent2 = _StubAgent()
        runner2 = _StubAgentRunner()
        sm2 = SessionManager(root=session_root)
        sm2.attach(agent2, runner2, event_bus=agent2.event_bus)
        try:
            sm2.load_session(sid)
            assert len(agent2.context.messages) == 1
            assert (
                agent2.context.messages[0]["content"][0]["text"] == "hello session"
            )
        finally:
            sm2.detach()

    def test_manifest_includes_gui_launch_profile(self, session_root: Path) -> None:
        profile = {
            "version": 1,
            "modelName": "deepseek-chat",
            "systemPrompt": "profile prompt",
            "selectedPlugins": ["hawi/filesystem"],
            "pluginConfigs": {"hawi/filesystem": {"root": "."}},
        }
        agent = _StubAgent()
        runner = _StubAgentRunner()
        sm = SessionManager(
            root=session_root,
            manifest_metadata_provider=lambda: {
                "gui_launch_profile": profile,
                "last_cwd": "/tmp/hawi-workspace",
            },
        )
        sm.attach(agent, runner, event_bus=agent.event_bus)
        try:
            sid = sm.new_session(name="profiled")
            agent.context.add_user_message("hello")
            sm.save_now()

            manifest = json.loads(
                layout.manifest_path(layout.session_dir(session_root, sid)).read_text()
            )
            assert manifest["gui_launch_profile"] == profile
            meta = sm.list_sessions()[0]
            assert meta.gui_launch_profile == profile
            assert meta.last_cwd == "/tmp/hawi-workspace"
        finally:
            sm.detach()

    @pytest.mark.asyncio
    async def test_load_session_keeps_saved_system_prompt_by_default(
        self,
        session_root: Path,
    ) -> None:
        agent = HawiAgent(model=object(), plugins=[_PromptHookPlugin()])
        runner = _StubAgentRunner()
        sm = SessionManager(root=session_root)
        sm.attach(agent, runner, event_bus=agent.event_bus)
        try:
            agent.context.system_prompt = [{"type": "text", "text": "saved prompt"}]
            agent.context.add_user_message("hello")
            sid = sm.new_session(name="saved")
            sm.save_now()
        finally:
            sm.detach()

        plugin2 = _PromptHookPlugin()
        agent2 = HawiAgent(model=object(), plugins=[plugin2])
        runner2 = _StubAgentRunner()
        sm2 = SessionManager(root=session_root)
        sm2.attach(agent2, runner2, event_bus=agent2.event_bus)
        try:
            sm2.load_session(sid)
            await agent2._invoke_session_hook(
                "before_conversation",
                HookContext(run_id="r1", iteration=0),
            )

            assert plugin2.calls == 0
            assert agent2.context.system_prompt == [
                {"type": "text", "text": "saved prompt"}
            ]
        finally:
            sm2.detach()

    @pytest.mark.asyncio
    async def test_load_session_can_regenerate_system_prompt_when_configured(
        self,
        session_root: Path,
    ) -> None:
        agent = HawiAgent(model=object(), plugins=[_PromptHookPlugin()])
        runner = _StubAgentRunner()
        sm = SessionManager(root=session_root)
        sm.attach(agent, runner, event_bus=agent.event_bus)
        try:
            agent.context.system_prompt = [{"type": "text", "text": "saved prompt"}]
            agent.context.add_user_message("hello")
            sid = sm.new_session(name="saved")
            sm.save_now()
        finally:
            sm.detach()

        plugin2 = _PromptHookPlugin()
        agent2 = HawiAgent(model=object(), plugins=[plugin2])
        runner2 = _StubAgentRunner()
        sm2 = SessionManager(
            root=session_root,
            keep_session_system_prompt=False,
        )
        sm2.attach(agent2, runner2, event_bus=agent2.event_bus)
        try:
            sm2.load_session(sid)
            await agent2._invoke_session_hook(
                "before_conversation",
                HookContext(run_id="r1", iteration=0),
            )

            assert plugin2.calls == 1
            assert agent2.context.system_prompt == [
                {"type": "text", "text": "saved prompt"},
                {"type": "text", "text": "regenerated prompt"},
            ]
        finally:
            sm2.detach()

    def test_list_sessions_returns_metas(self, stub_setup) -> None:
        sm, agent, _ = stub_setup
        agent.context.add_user_message("message A")
        a = sm.new_session(name="A")
        sm.save_now()
        agent.context.clear()
        agent.context.add_user_message("message B")
        b = sm.new_session(name="B")
        sm.save_now()
        ids = {m.session_id for m in sm.list_sessions()}
        assert {a, b}.issubset(ids)

    def test_list_sessions_sorts_by_created_at(self, session_root: Path) -> None:
        sm = SessionManager(root=session_root)
        records = [
            ("old-created", "2024-01-01T00:00:00", "2026-01-01T00:00:00"),
            ("new-created", "2025-01-01T00:00:00", "2025-01-01T00:00:00"),
        ]
        for session_id, created_at, updated_at in records:
            sd = layout.session_dir(session_root, session_id)
            layout.ensure_session_layout(sd)
            layout.append_jsonl(
                layout.message_history_path(sd),
                [
                    {
                        "version": 1,
                        "run_id": "r1",
                        "role": "user",
                        "content": [{"type": "text", "text": session_id}],
                        "metadata": None,
                    }
                ],
                fsync=False,
            )
            layout.atomic_write_text(
                layout.manifest_path(sd),
                json.dumps(
                    {
                        "version": 1,
                        "session_id": session_id,
                        "name": session_id,
                        "created_at": created_at,
                        "updated_at": updated_at,
                    }
                ),
                fsync=False,
            )

        assert [m.session_id for m in sm.list_sessions()] == [
            "new-created",
            "old-created",
        ]

    def test_locked_session_rejects_second_loader(self, session_root: Path) -> None:
        agent = _StubAgent()
        runner = _StubAgentRunner()
        sm = SessionManager(root=session_root)
        sm.attach(agent, runner, event_bus=agent.event_bus)
        try:
            agent.context.add_user_message("locked")
            sid = sm.new_session(name="locked")
            sm.save_now()
        finally:
            sm.detach()

        external_lock = SessionFileLock(
            layout.session_lock_path(layout.session_dir(session_root, sid)),
            owner_token="external",
            metadata=make_lock_metadata("external"),
        ).acquire()
        agent2 = _StubAgent()
        runner2 = _StubAgentRunner()
        sm2 = SessionManager(root=session_root)
        sm2.attach(agent2, runner2, event_bus=agent2.event_bus)
        try:
            metas = {m.session_id: m for m in sm2.list_sessions()}
            assert metas[sid].locked is True
            with pytest.raises(SessionLockedError):
                sm2.load_session(sid)
        finally:
            sm2.detach()
            external_lock.release()

    def test_locked_session_can_be_forked_by_second_manager(
        self,
        session_root: Path,
    ) -> None:
        agent = _StubAgent()
        runner = _StubAgentRunner()
        sm = SessionManager(root=session_root)
        sm.attach(agent, runner, event_bus=agent.event_bus)
        try:
            agent.context.add_user_message("fork source")
            sid = sm.new_session(name="source")
            sm.save_now()
        finally:
            sm.detach()

        external_lock = SessionFileLock(
            layout.session_lock_path(layout.session_dir(session_root, sid)),
            owner_token="external",
            metadata=make_lock_metadata("external"),
        ).acquire()
        agent2 = _StubAgent()
        runner2 = _StubAgentRunner()
        sm2 = SessionManager(root=session_root)
        sm2.attach(agent2, runner2, event_bus=agent2.event_bus)
        try:
            forked = sm2.fork_session(sid, name="forked")

            assert forked != sid
            assert sm2.current_session_id == forked
            assert agent2.context.messages[0]["content"][0]["text"] == "fork source"
            fork_manifest = json.loads(
                layout.manifest_path(layout.session_dir(session_root, forked)).read_text()
            )
            assert fork_manifest["forked_from_session_id"] == sid
            assert sm.list_sessions()[0].session_id in {sid, forked}
        finally:
            sm2.detach()
            external_lock.release()

    def test_fork_session_after_user_pops_user_and_prunes_history(
        self,
        stub_setup,
    ) -> None:
        sm, agent, _ = stub_setup
        agent.context.add_user_message("first")
        agent.context.add_assistant_message([{"type": "text", "text": "reply"}])
        agent.context.add_user_message("second")
        agent.context.add_assistant_message([{"type": "text", "text": "later"}])
        sid = sm.new_session(name="source")
        sm.save_now()
        layout.write_jsonl(
            layout.message_history_path(layout.session_dir(sm._root, sid)),
            [
                {
                    "version": 1,
                    "run_id": "r1",
                    "role": message["role"],
                    "content": message["content"],
                    "metadata": message.get("metadata"),
                }
                for message in agent.context.messages
            ],
            fsync=True,
        )

        result = sm.fork_session_after_message(
            session_id=sid,
            after_message_index=2,
        )

        assert result.session_id != sid
        assert result.target_role == "user"
        assert result.boundary_index == 2
        assert result.popped_user_message is not None
        assert result.popped_user_message["content"][0]["text"] == "second"
        assert [m["role"] for m in agent.context.messages] == ["user", "assistant"]
        history = sm.read_message_history(result.session_id)
        assert [entry["role"] for entry in history] == ["user", "assistant"]
        assert [entry["context_message_index"] for entry in history] == [0, 1]

    def test_fork_session_after_context_message_id(
        self,
        stub_setup,
    ) -> None:
        sm, agent, _ = stub_setup
        agent.context.add_user_message("first")
        agent.context.add_assistant_message([{"type": "text", "text": "reply"}])
        target_id = agent.context.add_user_message("second")
        agent.context.add_assistant_message([{"type": "text", "text": "later"}])
        sid = sm.new_session(name="source")
        sm.save_now()
        layout.write_jsonl(
            layout.message_history_path(layout.session_dir(sm._root, sid)),
            [
                {
                    "version": 1,
                    "run_id": "r1",
                    "role": message["role"],
                    "content": message["content"],
                    "metadata": message.get("metadata"),
                    "context_message_id": message.get("context_message_id"),
                }
                for message in agent.context.messages
            ],
            fsync=True,
        )

        result = sm.fork_session_after_message_id(
            session_id=sid,
            context_message_id=target_id,
        )

        assert result.context_message_id == target_id
        assert result.message_index == 2
        assert result.target_role == "user"
        assert result.popped_user_message is not None
        assert result.popped_user_message["context_message_id"] == target_id
        assert [m["role"] for m in agent.context.messages] == ["user", "assistant"]
        history = sm.read_message_history(result.session_id)
        assert [entry["role"] for entry in history] == ["user", "assistant"]
        assert all(entry.get("context_message_id") for entry in history)

    def test_rewind_session_after_assistant_keeps_tool_results(
        self,
        stub_setup,
    ) -> None:
        sm, agent, _ = stub_setup
        agent.context.add_user_message("do it")
        agent.context.add_assistant_message([
            {
                "type": "tool_call",
                "id": "call-1",
                "name": "tool",
                "arguments": {},
            }
        ])
        agent.context.add_tool_result("call-1", "done")
        agent.context.add_user_message("next")
        sid = sm.new_session(name="source")
        sm.save_now()
        layout.write_jsonl(
            layout.message_history_path(layout.session_dir(sm._root, sid)),
            [
                {
                    "version": 1,
                    "run_id": "r1",
                    "role": message["role"],
                    "content": message["content"],
                    "metadata": message.get("metadata"),
                }
                for message in agent.context.messages
            ],
            fsync=True,
        )

        result = sm.rewind_session_after_message(after_message_index=1)

        assert result.session_id == sid
        assert result.target_role == "assistant"
        assert result.boundary_index == 3
        assert [m["role"] for m in agent.context.messages] == [
            "user",
            "assistant",
            "tool",
        ]
        history = sm.read_message_history(sid)
        assert [entry["role"] for entry in history] == [
            "user",
            "assistant",
            "tool",
        ]

    def test_delete_removes_directory(self, stub_setup) -> None:
        sm, agent, _ = stub_setup
        agent.context.add_user_message("delete me")
        sid = sm.new_session()
        sm.save_now()
        sd = layout.session_dir(sm._root, sid)
        assert sd.exists()
        sm.delete_session(sid)
        assert not sd.exists()

    def test_switch_to_does_not_persist_empty_previous(self, stub_setup) -> None:
        sm, agent, _ = stub_setup
        agent.context.add_user_message("target")
        target = sm.new_session(name="target")
        sm.save_now()

        agent.context.clear()
        empty = sm.new_session(name="empty")
        sm.switch_to(target)

        assert not layout.session_dir(sm._root, empty).exists()
        assert layout.session_dir(sm._root, target).exists()

    def test_load_recovers_interrupted_tool_calls(self, session_root: Path) -> None:
        """A session saved mid-tool-call must come back with synthetic
        error results so GUI snapshots taken right after load are
        provider-valid (no orphan tool_calls)."""
        agent = _StubAgent()
        runner = _StubAgentRunner()
        sm = SessionManager(root=session_root)
        sm.attach(agent, runner, event_bus=agent.event_bus)
        try:
            # Build a context that mimics a crash mid-tool-call: assistant
            # message with a tool_call but no tool result.
            agent.context.add_user_message("do a thing")
            agent.context.add_assistant_message([
                {"type": "text", "text": "ok"},
                {
                    "type": "tool_call",
                    "id": "tc-orphan-1",
                    "name": "do_thing",
                    "arguments": {"x": 1},
                },
            ])
            sid = sm.new_session(name="crash")
            sm.save_now()
        finally:
            sm.detach()

        agent2 = _StubAgent()
        runner2 = _StubAgentRunner()
        sm2 = SessionManager(root=session_root)
        sm2.attach(agent2, runner2, event_bus=agent2.event_bus)
        try:
            sm2.load_session(sid)
            tool_results = [
                m for m in agent2.context.messages if m["role"] == "tool"
            ]
            assert len(tool_results) == 1, (
                "load_session should synthesize a tool_result for the orphan"
            )
            result_part = tool_results[0]["content"][0]
            assert result_part["tool_call_id"] == "tc-orphan-1"
            assert result_part["is_error"] is True
        finally:
            sm2.detach()

    def test_event_triggers_checkpoint(self, session_root: Path) -> None:
        agent = _StubAgent()
        runner = _StubAgentRunner()
        sm = SessionManager(root=session_root)
        sm.attach(agent, runner, event_bus=agent.event_bus)
        try:
            sid = sm.new_session()
            from hawi.events import AgentRunStartEvent
            agent.event_bus.publish(
                AgentMessageAddedEvent.create(
                    run_id="r1",
                    role="user",
                    content=[{"type": "text", "text": "visible"}],
                    metadata={"display_message_type": "normal"},
                )
            )
            agent.event_bus.publish(AgentRunStartEvent.create(run_id="r1"))
            # SessionManager subscribes blocking, so capture runs in-line.
            # Wait for the writer thread to flush.
            sm._writer.wait_idle(timeout=2.0)
            assert layout.runtime_path(layout.session_dir(session_root, sid)).exists()
        finally:
            sm.detach()

    def test_tool_result_message_persisted(self, session_root: Path) -> None:
        """A role=tool message_added event must land in message_history.jsonl.

        Regression: prior to this fix, only user/assistant messages emitted
        AgentMessageAddedEvent, so tool results never got persisted and
        re-loading a session with tools showed only text/thinking blocks.
        """
        agent = _StubAgent()
        runner = _StubAgentRunner()
        sm = SessionManager(root=session_root)
        sm.attach(agent, runner, event_bus=agent.event_bus)
        try:
            sid = sm.new_session()
            # First emit a user message so the session is "non-empty" and
            # subsequent checkpoints fire (manager skips empty sessions).
            agent.event_bus.publish(
                AgentMessageAddedEvent.create(
                    run_id="r1",
                    role="user",
                    content=[{"type": "text", "text": "do something"}],
                )
            )
            agent.event_bus.publish(
                AgentMessageAddedEvent.create(
                    run_id="r1",
                    role="tool",
                    content=[
                        {
                            "type": "tool_result",
                            "tool_call_id": "call_abc",
                            "content": [{"type": "text", "text": "tool says hi"}],
                            "is_error": False,
                        }
                    ],
                )
            )
            sm._writer.wait_idle(timeout=2.0)
            entries = sm.read_message_history(sid)
            tool_entries = [e for e in entries if e["role"] == "tool"]
            assert len(tool_entries) == 1, (
                f"expected 1 tool record, got entries={entries}"
            )
            part = tool_entries[0]["content"][0]
            assert part["type"] == "tool_result"
            assert part["tool_call_id"] == "call_abc"
            assert part["is_error"] is False
        finally:
            sm.detach()

    @pytest.mark.asyncio
    async def test_interrupted_stream_persists_partial_assistant_message(
        self,
        session_root: Path,
    ) -> None:
        model = _PartialStreamingModel()
        agent = HawiAgent(model=model, streaming=True)
        sm = SessionManager(root=session_root)
        sm.attach(agent, event_bus=agent.event_bus)
        try:
            sid = sm.new_session()
            task = asyncio.create_task(agent.arun("write a long answer"))
            await asyncio.wait_for(model.delta_processed.wait(), timeout=2.0)

            agent.interrupt("user")
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            sm.save_now()
            entries = sm.read_message_history(sid)
            assistant_entries = [e for e in entries if e["role"] == "assistant"]
            assert len(assistant_entries) == 1
            assert assistant_entries[0]["content"] == [
                {"type": "text", "text": "half answer"}
            ]
            assert assistant_entries[0]["metadata"]["partial"] is True
            assert assistant_entries[0]["metadata"]["interrupt_reason"] == "user"
        finally:
            sm.detach()

    def test_message_added_appends_visible_history_only(self, session_root: Path) -> None:
        agent = _StubAgent()
        runner = _StubAgentRunner()
        sm = SessionManager(root=session_root)
        sm.attach(agent, runner, event_bus=agent.event_bus)
        try:
            sid = sm.new_session()
            agent.event_bus.publish(
                AgentMessageAddedEvent.create(
                    run_id="r1",
                    role="user",
                    content=[{"type": "text", "text": "visible"}],
                    metadata={"display_message_type": "normal"},
                )
            )
            agent.event_bus.publish(
                AgentMessageAddedEvent.create(
                    run_id="r1",
                    role="user",
                    content=[{"type": "text", "text": "hidden"}],
                    metadata={"display": False},
                )
            )
            sm._writer.wait_idle(timeout=2.0)
            entries = sm.read_message_history(sid)
            assert len(entries) == 1
            assert entries[0]["content"][0]["text"] == "visible"

            sd = layout.session_dir(session_root, sid)
            assert layout.context_path(sd).exists()
            manifest = json.loads(layout.manifest_path(sd).read_text())
            assert layout.COMPONENT_MESSAGE_HISTORY in manifest["components_present"]
        finally:
            sm.detach()

    def test_context_compaction_events_append_message_history(
        self,
        session_root: Path,
    ) -> None:
        agent = _StubAgent()
        runner = _StubAgentRunner()
        sm = SessionManager(root=session_root)
        sm.attach(agent, runner, event_bus=agent.event_bus)
        try:
            sid = sm.new_session()
            agent.event_bus.publish(
                AgentCompactStartEvent.create(
                    run_id="r1",
                    mode="auto",
                    keep_last_messages=8,
                    tokens_before=25_000,
                    message_count_before=18,
                )
            )
            agent.event_bus.publish(
                AgentCompactStopEvent.create(
                    run_id="r1",
                    mode="auto",
                    status="success",
                    duration_ms=1234,
                    tokens_before=25_000,
                    tokens_after=8_000,
                    message_count_before=18,
                    message_count_after=7,
                    replaced_message_count=12,
                    kept_message_count=6,
                )
            )
            sm._writer.wait_idle(timeout=2.0)

            entries = sm.read_message_history(sid)
            assert [entry["role"] for entry in entries] == ["event", "event"]
            assert entries[0]["content"][0]["text"] == "Compressing context..."
            assert entries[0]["metadata"]["event_type"] == "agent.compact_start"
            assert entries[1]["content"][0]["text"] == "Context compacted"
            assert entries[1]["metadata"]["event_type"] == "agent.compact_stop"
            assert entries[1]["metadata"]["tokens_after"] == 8_000

            manifest = json.loads(
                layout.manifest_path(layout.session_dir(session_root, sid)).read_text()
            )
            assert layout.COMPONENT_MESSAGE_HISTORY in manifest["components_present"]
        finally:
            sm.detach()

    def test_injection_and_plugin_events_append_replayable_history(
        self,
        session_root: Path,
    ) -> None:
        agent = _StubAgent()
        runner = _StubAgentRunner()
        sm = SessionManager(root=session_root)
        sm.attach(agent, runner, event_bus=agent.event_bus)
        try:
            sid = sm.new_session()
            agent.event_bus.publish(
                AgentSystemPromptEvent.create(
                    run_id="r1",
                    content=[{"type": "text", "text": "system material"}],
                    origin="before_conversation",
                    plugin_id="research",
                    plugin_name="ResearchPlugin",
                    plugin_role="plugin",
                    injection_name="inject_prompt",
                    metadata={"content_scope": "injected_segment"},
                )
            )
            agent.event_bus.publish(
                AgentContextInjectedEvent.create(
                    run_id="r1",
                    role="user",
                    content=[{"type": "text", "text": "context material"}],
                    hook_type="before_conversation",
                    plugin_id="research",
                    plugin_name="ResearchPlugin",
                    plugin_role="plugin",
                    injection_name="inject_context",
                    merge_target="user_message",
                    merge_position="before",
                    target_message_id="msg-1",
                    target_message_index=0,
                )
            )
            agent.event_bus.publish(
                AgentToolRuntimeContextInjectedEvent.create(
                    run_id="r1",
                    tool_name="inspect",
                    tool_call_id="tc-2",
                    parameter_name="context",
                    plugin_id="inspector",
                    plugin_name="InspectorPlugin",
                    plugin_role="tool_owner",
                    injection_name="runtime_context",
                )
            )
            agent.event_bus.publish(
                PluginEvent.message(
                    plugin_id="planner",
                    plugin_name="PlannerPlugin",
                    title="Plan",
                    message="Collected notes",
                    data={"count": 3},
                    run_id="r1",
                    message_id="plugin-msg-1",
                )
            )
            agent.event_bus.publish(
                PluginEvent.create(
                    "plugin.artifact.upsert",
                    plugin_id="planner",
                    plugin_name="PlannerPlugin",
                    payload={
                        "artifact": {
                            "id": "plan",
                            "type": "plan",
                            "title": "Plan",
                            "content": "# Plan\n",
                            "language": "markdown",
                        }
                    },
                    run_id="r1",
                )
            )
            sm._writer.wait_idle(timeout=2.0)

            entries = sm.read_message_history(sid)
            assert [entry["role"] for entry in entries] == ["event"] * 5
            event_types = [entry["metadata"]["event_type"] for entry in entries]
            assert event_types == [
                "agent.system_prompt",
                "agent.context_injected",
                "agent.tool_runtime_context_injected",
                "plugin.message",
                "plugin.artifact.upsert",
            ]
            context_payload = entries[1]["metadata"]["event_payload"]
            assert context_payload["plugin_id"] == "research"
            assert context_payload["merge_target"] == "user_message"
            assert context_payload["merge_position"] == "before"
            assert context_payload["target_message_id"] == "msg-1"
            plugin_payload = entries[3]["metadata"]["event_payload"]
            assert plugin_payload["plugin_id"] == "planner"
            assert plugin_payload["message"] == "Collected notes"
            artifact_payload = entries[4]["metadata"]["event_payload"]
            assert artifact_payload["artifact"]["id"] == "plan"

            manifest = json.loads(
                layout.manifest_path(layout.session_dir(session_root, sid)).read_text()
            )
            assert layout.COMPONENT_MESSAGE_HISTORY in manifest["components_present"]
        finally:
            sm.detach()

    def test_subagent_events_append_replayable_history(
        self,
        session_root: Path,
    ) -> None:
        agent = _StubAgent()
        runner = _StubAgentRunner()
        sm = SessionManager(root=session_root)
        sm.attach(agent, runner, event_bus=agent.event_bus)
        try:
            sid = sm.new_session()
            agent.event_bus.publish(
                SubAgentEvent.create(
                    "subagent.created",
                    subagent_id="sub_1",
                    subagent_name="worker-1",
                    subagent_role="worker",
                    status={"id": "sub_1", "state": "RUNNING"},
                )
            )
            agent.event_bus.publish(
                SubAgentEvent.create(
                    "subagent.event",
                    subagent_id="sub_1",
                    subagent_name="worker-1",
                    subagent_role="worker",
                    status={"id": "sub_1", "state": "RUNNING"},
                    child_event={"type": "model.content_block_delta", "delta": "ignored"},
                )
            )
            agent.event_bus.publish(
                SubAgentEvent.create(
                    "subagent.event",
                    subagent_id="sub_1",
                    subagent_name="worker-1",
                    subagent_role="worker",
                    status={"id": "sub_1", "state": "COMPLETED"},
                    child_event={"type": "agent.message_added", "run_id": "sub-run"},
                    message_entry={
                        "version": 1,
                        "run_id": "sub-run",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "done"}],
                    },
                )
            )
            sm._writer.wait_idle(timeout=2.0)

            entries = sm.read_message_history(sid)
            assert [entry["metadata"]["event_type"] for entry in entries] == [
                "subagent.created",
                "subagent.event",
            ]
            payload = entries[1]["metadata"]["event_payload"]
            assert payload["subagent_id"] == "sub_1"
            assert payload["message_entry"]["content"][0]["text"] == "done"
        finally:
            sm.detach()

    def test_user_visible_status_and_error_events_append_message_history(
        self,
        session_root: Path,
    ) -> None:
        agent = _StubAgent()
        runner = _StubAgentRunner()
        sm = SessionManager(root=session_root)
        sm.attach(agent, runner, event_bus=agent.event_bus)
        try:
            sid = sm.new_session()
            agent.event_bus.publish(
                AgentMessageAddedEvent.create(
                    run_id="r1",
                    role="user",
                    content=[{"type": "text", "text": "call the model"}],
                )
            )
            agent.event_bus.publish(
                ModelRetryEvent.create(
                    request_id="r1-1",
                    error_type="network",
                    attempt=1,
                    max_retries=10,
                    error_message="Anthropic connection error: Connection error.",
                )
            )
            agent.event_bus.publish(
                ModelErrorEvent.create(
                    DeniedError("Anthropic authentication failed: Error code: 401")
                )
            )
            agent.event_bus.publish(
                AgentErrorEvent.create(
                    run_id="r1",
                    error=ToolExecutionError("Tool execution failed loudly"),
                )
            )
            agent.event_bus.publish(
                AgentRunnerInterruptEvent.create("user", ["tc-1"])
            )
            agent.event_bus.publish(
                AgentInterruptEvent.create(interrupt_type="user", run_id="r1")
            )
            sm._writer.wait_idle(timeout=2.0)

            entries = sm.read_message_history(sid)
            assert [entry["role"] for entry in entries] == [
                "user",
                "system",
                "error",
                "error",
                "system",
                "system",
            ]
            assert entries[1]["content"][0]["text"].startswith("模型重试 1/10")
            assert entries[1]["metadata"]["display_message_type"] == "model_retry"
            assert (
                "Anthropic authentication failed"
                in entries[2]["content"][0]["text"]
            )
            assert entries[2]["metadata"]["code"] == "model_error"
            assert "Tool execution failed loudly" in entries[3]["content"][0]["text"]
            assert entries[3]["metadata"]["code"] == "agent_error"
            assert entries[4]["content"][0]["text"] == "执行被中断: user"
            assert entries[4]["metadata"]["interrupted_tool_calls"] == ["tc-1"]
            assert entries[5]["content"][0]["text"] == "Agent 中断: user"

            manifest = json.loads(
                layout.manifest_path(layout.session_dir(session_root, sid)).read_text()
            )
            assert layout.COMPONENT_MESSAGE_HISTORY in manifest["components_present"]
        finally:
            sm.detach()


# ---------------------------------------------------------------------------
# ExitHandler priority extensions
# ---------------------------------------------------------------------------


class TestExitHandlerPriority:
    def test_priority_constants_ordered(self) -> None:
        assert EXIT_PRIORITY_NORMAL < EXIT_PRIORITY_PLUGIN_TEARDOWN
        assert EXIT_PRIORITY_PLUGIN_TEARDOWN < EXIT_PRIORITY_SESSION_FLUSH

    def test_register_last_runs_after_others(self) -> None:
        handler = ExitHandler.get_instance()
        order: list[str] = []

        plugin_cb = lambda: order.append("plugin")
        last_cb = lambda: order.append("last")

        try:
            handler.register(plugin_cb, priority=EXIT_PRIORITY_PLUGIN_TEARDOWN)
            handler.register_last(last_cb, name="test-last")
            handler.execute_and_keep()
        finally:
            handler.unregister(plugin_cb)
            handler.unregister(last_cb)

        assert order.index("plugin") < order.index("last")

    def test_register_last_rejects_second_registration(self) -> None:
        handler = ExitHandler.get_instance()
        a = lambda: None
        b = lambda: None
        try:
            handler.register_last(a, name="first")
            with pytest.raises(RuntimeError):
                handler.register_last(b, name="second")
        finally:
            handler.unregister(a)
            handler.unregister(b)

    def test_unregister_matches_bound_method(self) -> None:
        handler = ExitHandler.get_instance()

        class Holder:
            def __init__(self) -> None:
                self.calls = 0

            def hook(self) -> None:
                self.calls += 1

        h = Holder()
        # Register once via one bound-method instance, unregister via a
        # *different* bound-method object — they must compare equal.
        handler.register(h.hook, priority=EXIT_PRIORITY_NORMAL)
        removed = handler.unregister(h.hook)
        assert removed is True
