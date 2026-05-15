"""Snapshot round-trip tests for AgentContext, QueueManager, and plugin states.

Each component must serialize through ``json.dumps`` and rehydrate equivalent
state. These tests are independent of SessionManager; they pin down the
contract that SessionManager's writer relies on.
"""

from __future__ import annotations

import json

from hawi.agent.agent import SteerPartMergeMode
from hawi.agent.context import AgentContext
from hawi.agent.runner.queue import (
    MessageQueueManager,
    QueueType,
)
from hawi.events import EventBus
from hawi.tool.types import PendingToolCall
from hawi_plugins.plan_plugin.plugin import PlanPlugin
from hawi_plugins.python_interpreter import PythonInterpreterPlugin


class TestAgentContextSnapshot:
    def test_basic_round_trip(self) -> None:
        ctx = AgentContext()
        ctx.set_system_prompt("you are helpful")
        ctx.add_user_message("hello")
        ctx.add_assistant_message([{"type": "text", "text": "hi"}])

        snap = ctx.snapshot()
        encoded = json.dumps(snap, ensure_ascii=False)
        ctx2 = AgentContext()
        ctx2.load_snapshot(json.loads(encoded))

        assert ctx2.system_prompt == ctx.system_prompt
        assert ctx2.messages == ctx.messages

    def test_pending_tool_calls_persist(self) -> None:
        ctx = AgentContext()
        ctx._add_pending_tool_call("tc-1", "shell", {"cmd": "ls"})
        ctx._add_pending_tool_call("tc-2", "fetch", {"url": "x"})

        snap = ctx.snapshot()
        ctx2 = AgentContext()
        ctx2.load_snapshot(json.loads(json.dumps(snap)))

        pending = {p.tool_call_id: p for p in ctx2.get_pending_tool_calls()}
        assert set(pending) == {"tc-1", "tc-2"}
        assert pending["tc-1"].arguments == {"cmd": "ls"}

    def test_legacy_save_load_still_works(self, tmp_path) -> None:
        ctx = AgentContext()
        ctx.add_user_message("legacy")
        path = tmp_path / "ctx.json"
        ctx.save(path, format="json")
        ctx2 = AgentContext()
        ctx2.load(path)
        assert ctx2.messages[0]["content"][0]["text"] == "legacy"

    def test_unsupported_version_rejected(self) -> None:
        ctx = AgentContext()
        bad = {"version": "9.0", "messages": []}
        try:
            ctx.load_snapshot(bad)
        except ValueError as e:
            assert "version" in str(e).lower()
        else:
            raise AssertionError("expected ValueError")

    def test_truncate_after_user_message_pops_user(self) -> None:
        ctx = AgentContext()
        ctx.add_user_message("first")
        ctx.add_assistant_message([{"type": "text", "text": "reply"}])
        ctx.add_user_message("retry this")
        ctx.add_assistant_message([{"type": "text", "text": "later"}])

        result = ctx.truncate_after_message(2)

        assert result.target_role == "user"
        assert result.boundary_index == 2
        assert result.popped_user_message is not None
        assert result.popped_user_message["content"][0]["text"] == "retry this"
        assert [m["role"] for m in ctx.messages] == ["user", "assistant"]

    def test_truncate_after_tool_result_rejected(self) -> None:
        ctx = AgentContext()
        ctx.add_user_message("do it")
        ctx.add_assistant_message([
            {
                "type": "tool_call",
                "id": "call-1",
                "name": "tool",
                "arguments": {},
            }
        ])
        ctx.add_tool_result("call-1", "done")

        try:
            ctx.truncate_after_message(2)
        except ValueError as exc:
            assert "tool result" in str(exc)
        else:
            raise AssertionError("expected ValueError")

    def test_truncate_after_assistant_keeps_required_tool_results(self) -> None:
        ctx = AgentContext()
        ctx.add_user_message("do it")
        ctx.add_assistant_message([
            {
                "type": "tool_call",
                "id": "call-1",
                "name": "tool",
                "arguments": {},
            }
        ])
        ctx.add_tool_result("call-1", "done")
        ctx.add_user_message("next")

        result = ctx.truncate_after_message(1)

        assert result.target_role == "assistant"
        assert result.boundary_index == 3
        assert [m["role"] for m in ctx.messages] == ["user", "assistant", "tool"]


class TestQueueManagerSnapshot:
    def test_three_queues_round_trip(self) -> None:
        m = MessageQueueManager()
        m.enqueue_normal("n1")
        m.enqueue_high_prio("h1")
        m.enqueue_urgent("u1")

        snap = m.snapshot()
        encoded = json.dumps(snap)
        m2 = MessageQueueManager()
        m2.load_snapshot(json.loads(encoded))

        assert m2._pending_urgent is not None
        assert m2._pending_urgent.content == "u1"
        assert m2._high_prio_queue[0].content == "h1"
        assert m2._normal_queue[0].content == "n1"

    def test_queue_type_enum_preserved(self) -> None:
        m = MessageQueueManager()
        m.enqueue_high_prio("x")
        snap = json.loads(json.dumps(m.snapshot()))
        m2 = MessageQueueManager()
        m2.load_snapshot(snap)
        assert m2._high_prio_queue[0].queue_type is QueueType.HIGH_PRIO

    def test_event_bus_stripped_then_rebound(self) -> None:
        bus_a = EventBus()
        bus_b = EventBus()

        m = MessageQueueManager()
        m.enqueue_normal("x", event_bus=bus_a)
        snap = json.loads(json.dumps(m.snapshot()))
        m2 = MessageQueueManager()
        m2.load_snapshot(snap)

        assert m2._normal_queue[0].event_bus is None
        m2.rebind_event_bus(bus_b)
        assert m2._normal_queue[0].event_bus is bus_b

    def test_steer_merge_mode_enum_serialized(self) -> None:
        m = MessageQueueManager()
        m.enqueue_normal(
            "x",
            metadata={"steer_merge_mode": SteerPartMergeMode.APPEND_TO_TOOL_RESULT},
        )
        snap = json.loads(json.dumps(m.snapshot()))
        m2 = MessageQueueManager()
        m2.load_snapshot(snap)
        # str+Enum members serialize as their string value.
        assert (
            m2._normal_queue[0].metadata["steer_merge_mode"]
            == "append_to_tool_result"
        )


class TestPlanPluginState:
    def test_plan_state_round_trip(self) -> None:
        p = PlanPlugin(fold_completed_tasks=True)
        p._engine.add_plan_items(items=[{"content": "root task"}])
        state = p.save_state()
        encoded = json.dumps(state)
        p2 = PlanPlugin()
        p2.load_state(json.loads(encoded))
        assert [i.content for i in p2._engine.items] == ["root task"]
        assert p2._engine.fold_completed_tasks is True


class TestPythonInterpreterPluginState:
    def test_save_state_returns_work_dir_and_names(self, tmp_path) -> None:
        p = PythonInterpreterPlugin(work_dir=str(tmp_path))
        try:
            # _get_instance auto-creates DEFAULT_INSTANCE_NAME
            p._get_instance(None)
            state = p.save_state()
            assert state["work_dir"] == str(tmp_path)
            assert PythonInterpreterPlugin.DEFAULT_INSTANCE_NAME in state[
                "interpreter_names"
            ]
        finally:
            p.close()

    def test_load_state_recreates_named_slots(self, tmp_path) -> None:
        # Save → close → load on a fresh plugin, named slots come back empty.
        p = PythonInterpreterPlugin(work_dir=str(tmp_path))
        try:
            p._get_instance(None)
            p.create_interpreter(interpreter_name="custom")
            state = p.save_state()
        finally:
            p.close()

        p2 = PythonInterpreterPlugin(work_dir=str(tmp_path))
        try:
            p2.load_state(json.loads(json.dumps(state)))
            assert "custom" in p2.interpreters
            assert PythonInterpreterPlugin.DEFAULT_INSTANCE_NAME in p2.interpreters
        finally:
            p2.close()


class TestPendingToolCallSerialization:
    def test_round_trip(self) -> None:
        original = PendingToolCall(
            tool_call_id="tc",
            tool_name="run_shell",
            arguments={"cmd": "ls -la"},
        )
        ctx = AgentContext()
        ctx._pending_tool_calls[original.tool_call_id] = original
        snap = json.loads(json.dumps(ctx.snapshot()))
        ctx2 = AgentContext()
        ctx2.load_snapshot(snap)
        loaded = ctx2.get_pending_tool_calls()[0]
        assert loaded.tool_call_id == "tc"
        assert loaded.arguments == {"cmd": "ls -la"}
