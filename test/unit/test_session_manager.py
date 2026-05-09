"""Tests for hawi.session: layout, writer, manager.

Component snapshots are tested in their own files (test_queue_manager_snapshot,
test_agent_context_snapshot). This file focuses on the SessionManager glue
and the writer thread's atomicity / backpressure / failure semantics.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from hawi.agent.context import AgentContext
from hawi.agent.scheduler.queue import MessageQueueManager
from hawi.events import AgentMessageAddedEvent, EventBus, SessionWriteFailedEvent
from hawi.session import SessionManager, SessionWriter, WriteJob
from hawi.session import layout
from hawi.utils.lifecycle import (
    EXIT_PRIORITY_NORMAL,
    EXIT_PRIORITY_PLUGIN_TEARDOWN,
    EXIT_PRIORITY_SESSION_FLUSH,
    ExitHandler,
)


# ---------------------------------------------------------------------------
# Fixtures: stubs that mimic the agent / scheduler surface SessionManager uses
# ---------------------------------------------------------------------------


class _StubScheduler:
    def __init__(self) -> None:
        self._queue_manager = MessageQueueManager()


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


@pytest.fixture
def session_root(tmp_path: Path) -> Path:
    return tmp_path / "sessions"


@pytest.fixture
def stub_setup(session_root: Path):
    agent = _StubAgent()
    scheduler = _StubScheduler()
    sm = SessionManager(root=session_root)
    sm.attach(agent, scheduler, event_bus=agent.event_bus)
    yield sm, agent, scheduler
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
                        layout.COMPONENT_QUEUES: {"version": 1, "scheduler": {}},
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

    def test_save_now_skips_empty_session(self, stub_setup) -> None:
        sm, _, _ = stub_setup
        sid = sm.new_session(name="empty")
        sm.save_now()
        assert not layout.session_dir(sm._root, sid).exists()
        assert sm.list_sessions() == []

    def test_save_now_round_trip(self, session_root: Path) -> None:
        agent = _StubAgent()
        scheduler = _StubScheduler()
        sm = SessionManager(root=session_root)
        sm.attach(agent, scheduler, event_bus=agent.event_bus)
        try:
            agent.context.add_user_message("hello session")
            sid = sm.new_session(name="rt")
            sm.save_now()
            assert layout.context_path(layout.session_dir(session_root, sid)).exists()
        finally:
            sm.detach()

        # Fresh agent picks up the persisted state.
        agent2 = _StubAgent()
        scheduler2 = _StubScheduler()
        sm2 = SessionManager(root=session_root)
        sm2.attach(agent2, scheduler2, event_bus=agent2.event_bus)
        try:
            sm2.load_session(sid)
            assert len(agent2.context.messages) == 1
            assert (
                agent2.context.messages[0]["content"][0]["text"] == "hello session"
            )
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
        scheduler = _StubScheduler()
        sm = SessionManager(root=session_root)
        sm.attach(agent, scheduler, event_bus=agent.event_bus)
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
        scheduler2 = _StubScheduler()
        sm2 = SessionManager(root=session_root)
        sm2.attach(agent2, scheduler2, event_bus=agent2.event_bus)
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
        scheduler = _StubScheduler()
        sm = SessionManager(root=session_root)
        sm.attach(agent, scheduler, event_bus=agent.event_bus)
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

    def test_message_added_appends_visible_history_only(self, session_root: Path) -> None:
        agent = _StubAgent()
        scheduler = _StubScheduler()
        sm = SessionManager(root=session_root)
        sm.attach(agent, scheduler, event_bus=agent.event_bus)
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
