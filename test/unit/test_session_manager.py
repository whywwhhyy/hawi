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
    Event,
    AgentInterruptEvent,
    AgentMessageAddedEvent,
    AgentSystemPromptEvent,
    AgentToolRuntimeContextInjectedEvent,
    EventBus,
    ModelErrorEvent,
    ModelMetadataEvent,
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
        self.loaded_runtime: dict[str, Any] | None = None
        self.loaded_steer: list[dict[str, Any]] | None = None

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
        self.loaded_runtime = data

    def snapshot_steer(self) -> list:
        return []

    def load_steer(self, data: list) -> None:  # pragma: no cover - exercised below
        self.loaded_steer = data


class _ControlStubRunner(_StubAgentRunner):
    def __init__(self) -> None:
        super().__init__()
        self.control: dict[str, Any] = {
            "paused": False,
            "pause_reason": None,
            "resumable": False,
            "paused_at": None,
            "last_error_message": None,
        }
        self.loaded_controls: list[dict[str, Any]] = []

    def control_snapshot(self) -> dict[str, Any]:
        return dict(self.control)

    def load_control_snapshot(self, data: dict[str, Any] | None) -> None:
        snapshot = dict(data or {})
        self.loaded_controls.append(snapshot)
        self.control = {
            "paused": bool(snapshot.get("paused")),
            "pause_reason": snapshot.get("pause_reason"),
            "resumable": bool(snapshot.get("resumable")),
            "paused_at": snapshot.get("paused_at"),
            "last_error_message": snapshot.get("last_error_message"),
        }

    @property
    def paused(self) -> bool:
        return bool(self.control.get("paused"))


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

    def test_pending_audit_persists_only_in_context_snapshot(self, stub_setup) -> None:
        sm, agent, _ = stub_setup
        agent.context.add_user_message("needs audit")
        agent.context._add_pending_tool_call(
            "audit-1",
            "reviewed_tool",
            {"path": "file.txt"},
        )
        sid = sm.new_session(name="audit")
        sm.save_now()

        session_dir = layout.session_dir(sm._root, sid)
        queues = json.loads(layout.queues_path(session_dir).read_text())
        context = json.loads(layout.context_path(session_dir).read_text())

        assert "pending_audit_tool_calls" not in queues
        assert len(context["pending_tool_calls"]) == 1
        pending = context["pending_tool_calls"][0]
        assert pending["tool_call_id"] == "audit-1"
        assert pending["tool_name"] == "reviewed_tool"
        assert pending["arguments"] == {"path": "file.txt"}
        assert isinstance(pending["requested_at"], float)

    def test_rename_current_session_updates_manifest(self, stub_setup) -> None:
        sm, agent, _ = stub_setup
        sid = sm.new_session(name="old name")
        agent.context.add_user_message("hello")
        sm.save_now()

        sm.rename_session(sid, "new name")

        manifest = json.loads(
            layout.manifest_path(layout.session_dir(sm._root, sid)).read_text()
        )
        assert manifest["name"] == "new name"
        assert manifest["last_checkpoint_event"] == "session_rename"
        assert manifest["title_auto_generated"] is False
        assert isinstance(manifest["title_user_edited_at"], str)
        assert sm.list_sessions()[0].name == "new name"

    def test_auto_title_current_session_updates_default_name(self, stub_setup) -> None:
        sm, agent, _ = stub_setup
        sid = sm.new_session()
        agent.context.add_user_message("hello")
        sm.save_now()

        assert sm.session_needs_auto_title(sid) is True
        assert sm.auto_title_session(sid, "Useful Title") is True

        manifest = json.loads(
            layout.manifest_path(layout.session_dir(sm._root, sid)).read_text()
        )
        assert manifest["name"] == "Useful Title"
        assert manifest["last_checkpoint_event"] == "session_auto_title"
        assert manifest["title_auto_generated"] is True
        assert isinstance(manifest["title_generated_at"], str)
        assert sm.session_needs_auto_title(sid) is False
        assert sm.list_sessions()[0].name == "Useful Title"

    def test_auto_title_does_not_override_manual_name(self, stub_setup) -> None:
        sm, agent, _ = stub_setup
        sid = sm.new_session(name="manual name")
        agent.context.add_user_message("hello")
        sm.save_now()

        assert sm.session_needs_auto_title(sid) is False
        assert sm.auto_title_session(sid, "Generated") is False

        manifest = json.loads(
            layout.manifest_path(layout.session_dir(sm._root, sid)).read_text()
        )
        assert manifest["name"] == "manual name"

    def test_rename_unloaded_session_updates_manifest(self, session_root: Path) -> None:
        agent = _StubAgent()
        runner = _StubAgentRunner()
        sm = SessionManager(root=session_root)
        sm.attach(agent, runner, event_bus=agent.event_bus)
        try:
            sid = sm.new_session(name="old name")
            agent.context.add_user_message("hello")
            sm.save_now()
        finally:
            sm.detach()

        sm2 = SessionManager(root=session_root)
        sm2.rename_session(sid, "new name")

        manifest = json.loads(
            layout.manifest_path(layout.session_dir(session_root, sid)).read_text()
        )
        assert manifest["name"] == "new name"
        assert sm2.list_sessions()[0].name == "new name"

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
            system_prompt_events: list[AgentSystemPromptEvent] = []

            def collect_system_prompt_event(event: Event) -> None:
                assert isinstance(event, AgentSystemPromptEvent)
                system_prompt_events.append(event)

            agent2.event_bus.subscribe_blocking(
                collect_system_prompt_event,
                event_types=["agent.system_prompt"],
            )
            await agent2._emit_system_prompt_event_if_changed(
                run_id="r1",
                origin="session_start",
                event_bus=agent2.event_bus,
            )
            await agent2._invoke_session_hook(
                "before_conversation",
                HookContext(run_id="r1", iteration=0),
            )

            assert plugin2.calls == 0
            assert system_prompt_events == []
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

    def test_load_session_restores_unpaused_runner_control(
        self,
        session_root: Path,
    ) -> None:
        agent = _StubAgent()
        runner = _ControlStubRunner()
        sm = SessionManager(root=session_root)
        sm.attach(agent, runner, event_bus=agent.event_bus)
        try:
            agent.context.add_user_message("paused")
            paused_sid = sm.new_session(name="paused")
            sm.save_now()

            agent.context.clear()
            agent.context.add_user_message("open")
            open_sid = sm.new_session(name="open")
            sm.save_now()
        finally:
            sm.detach()

        self._write_runner_control(session_root, paused_sid, paused=True)
        self._write_runner_control(
            session_root,
            open_sid,
            paused=False,
            include_stale_work=False,
        )

        agent2 = _StubAgent()
        runner2 = _ControlStubRunner()
        sm2 = SessionManager(root=session_root)
        sm2.attach(agent2, runner2, event_bus=agent2.event_bus)
        try:
            sm2.load_session(paused_sid)
            assert runner2.paused is True

            sm2.load_session(open_sid)
            assert runner2.paused is False
            assert agent2.loaded_steer == []
            assert agent2.loaded_runtime is not None
            assert agent2.loaded_runtime["current_tool_calls"] == []
        finally:
            sm2.detach()

    def test_fork_session_resets_copied_volatile_state_and_releases_lock(
        self,
        session_root: Path,
    ) -> None:
        agent = _StubAgent()
        runner = _ControlStubRunner()
        sm = SessionManager(root=session_root)
        sm.attach(agent, runner, event_bus=agent.event_bus)
        try:
            agent.context.add_user_message("fork source")
            source_sid = sm.new_session(name="source")
            sm.save_now()

            agent.context.clear()
            agent.context.add_user_message("target")
            target_sid = sm.new_session(name="target")
            sm.save_now()
        finally:
            sm.detach()

        self._write_runner_control(session_root, source_sid, paused=True)
        self._write_runtime_state(session_root, source_sid)
        self._write_pending_tool_call(session_root, source_sid)

        agent2 = _StubAgent()
        runner2 = _ControlStubRunner()
        sm2 = SessionManager(root=session_root)
        sm2.attach(agent2, runner2, event_bus=agent2.event_bus)
        try:
            forked_sid = sm2.fork_session(source_sid, name="forked")

            assert sm2.current_session_id == forked_sid
            assert runner2.paused is False
            assert agent2.loaded_steer == []
            assert agent2.loaded_runtime is not None
            assert agent2.loaded_runtime["current_tool_calls"] == []

            fork_dir = layout.session_dir(session_root, forked_sid)
            fork_queues = json.loads(layout.queues_path(fork_dir).read_text())
            assert fork_queues["runner"]["normal"] == []
            assert fork_queues["pending_steer_inputs"] == []
            assert "pending_audit_tool_calls" not in fork_queues
            assert fork_queues["runner_control"]["paused"] is False

            fork_runtime = json.loads(layout.runtime_path(fork_dir).read_text())
            assert fork_runtime["current_tool_calls"] == []
            assert fork_runtime["interrupted_tool_call_ids"] == []

            fork_context = json.loads(layout.context_path(fork_dir).read_text())
            assert fork_context["pending_tool_calls"] == []

            sm2.switch_to(target_sid)
            metas = {meta.session_id: meta for meta in sm2.list_sessions()}
            assert metas[forked_sid].locked is False
        finally:
            sm2.detach()

    @staticmethod
    def _write_runner_control(
        session_root: Path,
        session_id: str,
        *,
        paused: bool,
        include_stale_work: bool = True,
    ) -> None:
        session_dir = layout.session_dir(session_root, session_id)
        payload = {
            "version": layout.QUEUES_VERSION,
            "runner": {
                "version": 1,
                "urgent": None,
                "high_prio": [],
                "normal": [
                    {
                        "id": "queued-1",
                        "content": "stale queue item",
                        "queue_type": "NORMAL",
                        "created_at": 1.0,
                        "metadata": {},
                        "merged_tool_call_ids": [],
                    }
                ] if include_stale_work else [],
            },
            "pending_steer_inputs": [
                {
                    "id": "steer-1",
                    "content": [{"type": "text", "text": "stale steer"}],
                    "candidate_tool_call_ids": [],
                    "created_at": 1.0,
                    "preferred_merge_mode": None,
                }
            ] if include_stale_work else [],
            "pending_audit_tool_calls": [
                {
                    "tool_call_id": "audit-1",
                    "tool_name": "needs_review",
                    "arguments": {},
                    "requested_at": 1.0,
                }
            ] if include_stale_work else [],
            "runner_control": {
                "paused": paused,
                "pause_reason": "model_error" if paused else None,
                "resumable": paused,
                "paused_at": 1.0 if paused else None,
                "last_error_message": "boom" if paused else None,
            },
        }
        layout.atomic_write_text(
            layout.queues_path(session_dir),
            json.dumps(payload, ensure_ascii=False, indent=2),
            fsync=True,
        )

    @staticmethod
    def _write_runtime_state(session_root: Path, session_id: str) -> None:
        session_dir = layout.session_dir(session_root, session_id)
        payload = {
            "version": layout.RUNTIME_VERSION,
            "active_run_id": "run-stale",
            "iteration": 2,
            "current_tool_calls": [
                {
                    "type": "tool_call",
                    "id": "tc-stale",
                    "name": "tool",
                    "arguments": {},
                }
            ],
            "interrupted_tool_call_ids": ["tc-stale"],
            "last_unsent_tool_results": [
                {
                    "tool_call_id": "tc-stale",
                    "tool_name": "tool",
                    "content": "unsent",
                    "is_error": False,
                    "truncate_attempts": 0,
                }
            ],
            "last_interrupt_reason": "user",
            "tool_executor": {"version": 1, "queue": [], "requests": []},
        }
        layout.atomic_write_text(
            layout.runtime_path(session_dir),
            json.dumps(payload, ensure_ascii=False, indent=2),
            fsync=True,
        )

    @staticmethod
    def _write_pending_tool_call(session_root: Path, session_id: str) -> None:
        session_dir = layout.session_dir(session_root, session_id)
        context_path = layout.context_path(session_dir)
        payload = json.loads(context_path.read_text(encoding="utf-8"))
        payload["pending_tool_calls"] = [
            {
                "tool_call_id": "audit-1",
                "tool_name": "needs_review",
                "arguments": {},
                "requested_at": 1.0,
            }
        ]
        layout.atomic_write_text(
            context_path,
            json.dumps(payload, ensure_ascii=False, indent=2),
            fsync=True,
        )

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

    def test_fork_session_prunes_side_threads_after_context_boundary(
        self,
        stub_setup,
    ) -> None:
        sm, agent, _ = stub_setup
        agent.context.add_user_message("first")
        kept_id = agent.context.add_assistant_message([{"type": "text", "text": "reply"}])
        pruned_id = agent.context.add_user_message("second")
        agent.context.add_assistant_message([{"type": "text", "text": "later"}])
        sid = sm.new_session(name="source")
        sm.save_now()
        sm.upsert_side_thread(
            {
                "id": "keep",
                "session_id": sid,
                "anchor_context_message_id": kept_id,
                "status": "completed",
                "messages": [],
            },
            sid,
            fsync=True,
        )
        sm.upsert_side_thread(
            {
                "id": "prune",
                "session_id": sid,
                "anchor_context_message_id": pruned_id,
                "status": "completed",
                "messages": [],
            },
            sid,
            fsync=True,
        )

        result = sm.fork_session_after_message_id(
            session_id=sid,
            context_message_id=kept_id,
        )

        assert [thread["id"] for thread in sm.read_side_threads(result.session_id)] == ["keep"]

    def test_delete_side_thread_removes_only_target_thread(
        self,
        stub_setup,
    ) -> None:
        sm, _, _ = stub_setup
        sid = sm.new_session(name="source")
        sm.upsert_side_thread(
            {
                "id": "keep",
                "session_id": sid,
                "anchor_context_message_id": "ctx-a",
                "status": "completed",
                "messages": [],
            },
            sid,
            fsync=True,
        )
        sm.upsert_side_thread(
            {
                "id": "delete",
                "session_id": sid,
                "anchor_context_message_id": "ctx-b",
                "status": "completed",
                "messages": [],
            },
            sid,
            fsync=True,
        )

        assert sm.delete_side_thread("delete", session_id=sid) is True
        assert [thread["id"] for thread in sm.read_side_threads(sid)] == ["keep"]
        assert sm.delete_side_thread("missing", session_id=sid) is False
        assert [thread["id"] for thread in sm.read_side_threads(sid)] == ["keep"]

    def test_mark_stale_side_threads_skips_active_running_threads(
        self,
        stub_setup,
    ) -> None:
        sm, _, _ = stub_setup
        sid = sm.new_session(name="source")
        for thread_id, status in [
            ("stale", "running"),
            ("active", "running"),
            ("done", "completed"),
        ]:
            sm.upsert_side_thread(
                {
                    "id": thread_id,
                    "session_id": sid,
                    "anchor_context_message_id": f"ctx-{thread_id}",
                    "status": status,
                    "messages": [],
                },
                sid,
                fsync=True,
            )

        assert sm.mark_stale_side_threads(sid, active_ids={"active"}, fsync=True) is True

        statuses = {
            thread["id"]: thread["status"]
            for thread in sm.read_side_threads(sid)
        }
        assert statuses == {
            "stale": "stale",
            "active": "running",
            "done": "completed",
        }
        stale_thread = next(
            thread for thread in sm.read_side_threads(sid)
            if thread["id"] == "stale"
        )
        assert "engine stopped" in stale_thread["error"]

    def test_export_markdown_includes_side_threads(
        self,
        stub_setup,
    ) -> None:
        sm, agent, _ = stub_setup
        agent.context.add_user_message("main question")
        sid = sm.new_session(name="source")
        sm.save_now()
        sm.upsert_side_thread(
            {
                "id": "side-a",
                "session_id": sid,
                "anchor_context_message_id": "ctx-a",
                "anchor_preview": "main answer excerpt",
                "quoted_text": "selected quote",
                "quoted_preview": "selected quote",
                "status": "completed",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "side question"}],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "side answer"}],
                    },
                ],
            },
            sid,
            fsync=True,
        )

        export = sm.export_markdown(sid)

        assert "## Side Threads" in export.markdown
        assert "selected quote" in export.markdown
        assert "side question" in export.markdown
        assert "side answer" in export.markdown

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

    def test_edit_session_message_updates_user_text_and_keeps_later_history(
        self,
        stub_setup,
    ) -> None:
        sm, agent, _ = stub_setup
        agent.context.add_user_message("first")
        agent.context.add_assistant_message([{"type": "text", "text": "reply"}])
        target_id = agent.context.add_user_message("old user text")
        agent.context.add_assistant_message([{"type": "text", "text": "later reply"}])
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

        result = sm.edit_session_message(
            role="user",
            context_message_id=target_id,
            text="edited user text",
        )

        assert result["session_id"] == sid
        assert result["context_message_id"] == target_id
        assert result["message_index"] == 2
        assert agent.context.messages[2]["content"] == [
            {"type": "text", "text": "edited user text"}
        ]
        assert agent.context.messages[3]["content"][0]["text"] == "later reply"
        history = sm.read_message_history(sid)
        assert [entry["content"][0]["text"] for entry in history] == [
            "first",
            "reply",
            "edited user text",
            "later reply",
        ]

    def test_edit_session_message_replaces_user_text_and_attachments(
        self,
        stub_setup,
    ) -> None:
        sm, agent, _ = stub_setup
        target_id = agent.context.add_user_message([
            {"type": "text", "text": "old"},
            {
                "type": "image",
                "source": {
                    "kind": "blob",
                    "blob_id": "old-blob",
                    "uri": "hawi-blob://old-blob",
                    "mime_type": "image/png",
                    "filename": "old.png",
                },
            },
        ])
        sid = sm.new_session(name="attachments")
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
        next_parts = [
            {"type": "text", "text": "new"},
            {
                "type": "image",
                "source": {
                    "kind": "blob",
                    "blob_id": "new-blob",
                    "uri": "hawi-blob://new-blob",
                    "mime_type": "image/png",
                    "filename": "new.png",
                },
            },
        ]

        sm.edit_session_message(
            role="user",
            context_message_id=target_id,
            text="new",
            content_parts=next_parts,
        )

        assert agent.context.messages[0]["content"] == next_parts
        history = sm.read_message_history(sid)
        assert history[0]["content"] == next_parts

    def test_edit_session_message_updates_assistant_text_preserving_structure(
        self,
        stub_setup,
    ) -> None:
        sm, agent, _ = stub_setup
        agent.context.add_user_message("do work")
        assistant_id = agent.context.add_assistant_message([
            {"type": "reasoning", "reasoning": "private chain"},
            {"type": "text", "text": "old visible"},
            {
                "type": "tool_call",
                "id": "call-1",
                "name": "read_file",
                "arguments": {"path": "a.txt"},
            },
        ])
        sid = sm.new_session(name="assistant-edit")
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

        sm.edit_session_message(
            role="assistant",
            context_message_id=assistant_id,
            text="new visible",
        )

        content = agent.context.messages[1]["content"]
        assert content == [
            {"type": "reasoning", "reasoning": "private chain"},
            {"type": "text", "text": "new visible"},
            {
                "type": "tool_call",
                "id": "call-1",
                "name": "read_file",
                "arguments": {"path": "a.txt"},
            },
        ]
        history = sm.read_message_history(sid)
        assert history[1]["content"] == content

    def test_edit_session_message_rejects_bad_targets(
        self,
        stub_setup,
    ) -> None:
        sm, agent, _ = stub_setup
        user_id = agent.context.add_user_message("user")
        assistant_id = agent.context.add_assistant_message([
            {"type": "text", "text": "assistant"}
        ])
        sid = sm.new_session(name="bad-targets")
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

        with pytest.raises(KeyError):
            sm.edit_session_message(
                role="user",
                context_message_id="missing",
                text="nope",
            )
        with pytest.raises(ValueError, match="target role"):
            sm.edit_session_message(
                role="user",
                context_message_id=assistant_id,
                text="nope",
            )
        with pytest.raises(ValueError, match="user message edit must include"):
            sm.edit_session_message(
                role="user",
                context_message_id=user_id,
                text="",
                content_parts=[],
            )
        with pytest.raises(ValueError, match="assistant message edit text must be non-empty"):
            sm.edit_session_message(
                role="assistant",
                context_message_id=assistant_id,
                text="",
            )

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

    def test_model_metadata_appends_replayable_usage_history(
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
                ModelMetadataEvent.create(
                    request_id="req-usage",
                    usage={
                        "input_tokens": 12,
                        "output_tokens": 3,
                        "total_tokens": 15,
                        "cache_read_tokens": 4,
                        "cache_write_tokens": 1,
                    },
                    latency_ms=42,
                    context_tokens=15,
                    max_context_tokens=100,
                    context_ratio=0.15,
                    context_source="provider_usage",
                )
            )
            sm._writer.wait_idle(timeout=2.0)

            entries = sm.read_message_history(sid)
            assert [entry["role"] for entry in entries] == ["event"]
            assert entries[0]["metadata"]["event_type"] == "model.metadata"
            assert entries[0]["metadata"]["replay"] is True
            payload = entries[0]["metadata"]["event_payload"]
            assert payload["request_id"] == "req-usage"
            assert payload["input_tokens"] == 12
            assert payload["output_tokens"] == 3
            assert payload["total_tokens"] == 15
            assert payload["cache_read_tokens"] == 4
            assert payload["cache_write_tokens"] == 1
            assert payload["latency_ms"] == 42
            assert payload["context_tokens"] == 15
            assert payload["max_context_tokens"] == 100
            assert payload["context_ratio"] == 0.15
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
