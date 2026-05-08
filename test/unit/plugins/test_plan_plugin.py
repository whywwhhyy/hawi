from __future__ import annotations

from types import SimpleNamespace

from hawi.agent.context import AgentContext
from hawi.events import Event, EventBus
from hawi.plugin import HookContext
from hawi_plugins.plan_plugin import PlanPlugin
from hawi_plugins.plan_plugin.plugin import PLAN_PROMPT_BEGIN, PLAN_REMINDER_BEGIN


def test_plan_items_support_tree_and_recursive_completion() -> None:
    plugin = PlanPlugin()

    root = plugin.add_plan_item("Prepare release")
    assert root.success is True
    child = plugin.add_plan_item("Run tests", parent_id="P1")
    assert child.success is True

    listed = plugin.list_plan_items()
    assert listed.success is True
    state = listed.output
    assert isinstance(state, dict)
    assert state["pending_count"] == 2
    assert state["items"][0]["id"] == "P1"
    assert state["items"][0]["children"][0]["id"] == "P2"
    assert state["items"][0]["children"][0]["parent_id"] == "P1"

    completed = plugin.complete_plan_item("P1", mark_all_children=True)
    assert completed.success is True
    output = completed.output
    assert isinstance(output, str)
    assert "`P1` Prepare release" in output
    assert "`P2` Run tests" in output
    assert "Pending items: 0" in output
    listed = plugin.list_plan_items()
    assert isinstance(listed.output, dict)
    assert listed.output["pending_count"] == 0


def test_parent_completion_fails_with_unfinished_children_without_mark_all() -> None:
    plugin = PlanPlugin()
    plugin.add_plan_item("Prepare release")
    plugin.add_plan_item("Run tests", parent_id="P1")
    plugin.add_plan_item("Tag release", parent_id="P1")

    result = plugin.complete_plan_item("P1")

    assert result.success is False
    assert "unfinished child task" in result.error
    assert "`P2` Run tests" in result.error
    assert "`P3` Tag release" in result.error
    assert result.output is None
    listed = plugin.list_plan_items()
    assert isinstance(listed.output, dict)
    assert listed.output["pending_count"] == 3
    assert all(not item["completed"] for item in listed.output["flat_items"])


def test_mark_all_children_completes_parent_and_descendants() -> None:
    plugin = PlanPlugin()
    plugin.add_plan_item("Prepare release")
    plugin.add_plan_item("Run tests", parent_id="P1")
    plugin.add_plan_item("Tag release", parent_id="P1")

    result = plugin.complete_plan_item("P1", mark_all_children=True)

    assert result.success is True
    assert isinstance(result.output, str)
    assert "`P1` Prepare release" in result.output
    assert "`P2` Run tests" in result.output
    assert "`P3` Tag release" in result.output
    assert "Pending items: 0" in result.output
    listed = plugin.list_plan_items()
    assert isinstance(listed.output, dict)
    assert listed.output["pending_count"] == 0


def test_default_kind_is_exploratory_and_does_not_auto_complete_parent() -> None:
    plugin = PlanPlugin()
    plugin.add_plan_item("Prepare release")
    plugin.add_plan_item("Run tests", parent_id="P1")
    plugin.add_plan_item("Tag release", parent_id="P1")

    listed = plugin.list_plan_items()
    assert isinstance(listed.output, dict)
    assert listed.output["flat_items"][0]["kind"] == "exploratory"

    first = plugin.complete_plan_item("P2")
    assert first.success is True
    output = first.output
    assert isinstance(output, str)
    assert "`P2` Run tests" in output
    assert "Parent review required" not in output
    assert "Pending items: 2" in output

    second = plugin.complete_plan_item("P3")
    assert second.success is True
    output = second.output
    assert isinstance(output, str)
    assert "`P3` Tag release" in output
    assert "Pending items: 1" in output
    assert "Parent review required" in output
    assert "`P1` Prepare release" in output
    assert "exploratory" in output


def test_determinate_parent_auto_completes_when_children_done() -> None:
    plugin = PlanPlugin()
    plugin.add_plan_item("Prepare release", kind="determinate")
    plugin.add_plan_item("Run tests", parent_id="P1")
    plugin.add_plan_item("Tag release", parent_id="P1")

    plugin.complete_plan_item("P2")
    final = plugin.complete_plan_item("P3")
    assert final.success is True
    output = final.output
    assert isinstance(output, str)
    assert "`P3` Tag release" in output
    assert "`P1` Prepare release" in output
    assert "Parent review required" not in output
    assert "Pending items: 0" in output


def test_determinate_chain_stops_at_exploratory_ancestor() -> None:
    plugin = PlanPlugin()
    plugin.add_plan_item(
        items=[
            {
                "content": "Top exploratory",
                "children": [
                    {
                        "content": "Mid determinate",
                        "kind": "determinate",
                        "children": [{"content": "Leaf"}],
                    }
                ],
            }
        ]
    )

    result = plugin.complete_plan_item("P3")
    assert result.success is True
    output = result.output
    assert isinstance(output, str)
    assert "`P3` Leaf" in output
    assert "`P2` Mid determinate" in output
    assert "Parent review required" in output
    assert "`P1` Top exploratory" in output
    assert "Pending items: 1" in output


def test_auto_completion_emits_events_only_for_determinate_parents() -> None:
    bus = EventBus()
    received: list[Event] = []
    bus.subscribe_blocking(received.append)
    plugin = PlanPlugin()
    plugin.bind_plugin_identity(plugin_id="plan", plugin_name="PlanPlugin")
    plugin.bind_event_bus(bus)

    try:
        plugin.add_plan_item("Parent", kind="determinate")
        plugin.add_plan_item("Child A", parent_id="P1")
        plugin.add_plan_item("Child B", parent_id="P1")
        plugin.complete_plan_item("P2")
        plugin.complete_plan_item("P3")
    finally:
        bus.close(wait=True, timeout=2)

    completed_ids = [
        event.payload["item"]["id"]
        for event in received
        if event.type == "plugin.event"
        and event.payload.get("event_name") == "plan.item.updated"
        and event.payload.get("action") == "completed"
    ]
    assert completed_ids == ["P2", "P3", "P1"]


def test_add_plan_item_rejects_invalid_kind() -> None:
    plugin = PlanPlugin()
    result = plugin.add_plan_item("Task", kind="urgent")
    assert result.success is False
    assert "kind" in result.error

    tree_result = plugin.add_plan_item(
        items=[{"content": "Root", "kind": "weird"}]
    )
    assert tree_result.success is False
    assert "kind" in tree_result.error


def test_add_plan_item_rejects_top_level_kind_with_items() -> None:
    plugin = PlanPlugin()
    result = plugin.add_plan_item(
        items=[{"content": "Root"}], kind="determinate"
    )
    assert result.success is False
    assert "tree mode" in result.error.lower() or "kind" in result.error


def test_add_plan_item_rejects_unknown_parent() -> None:
    plugin = PlanPlugin()

    result = plugin.add_plan_item("Nested", parent_id="missing")

    assert result.success is False
    assert "Unknown parent" in result.error


def test_add_plan_item_can_create_entire_plan_tree() -> None:
    plugin = PlanPlugin()

    result = plugin.add_plan_item(
        items=[
            {
                "content": "Implement feature",
                "children": [
                    {"content": "Update backend"},
                    {
                        "content": "Update GUI",
                        "children": [{"content": "Add preview state"}],
                    },
                ],
            },
            {"content": "Run tests"},
        ]
    )

    assert result.success is True
    output = result.output
    assert isinstance(output, dict)
    assert [item["id"] for item in output["items"]] == ["P1", "P2", "P3", "P4", "P5"]
    assert output["pending_count"] == 5
    tree = output["tree"]
    assert tree[0]["content"] == "Implement feature"
    assert tree[0]["children"][0]["parent_id"] == "P1"
    assert tree[0]["children"][1]["children"][0]["content"] == "Add preview state"
    assert tree[1]["content"] == "Run tests"


def test_add_plan_item_tool_accepts_tree_items_argument() -> None:
    plugin = PlanPlugin()
    add_tool = next(tool for tool in plugin.tools if tool.name == "add_plan_item")

    result = add_tool.invoke(
        {"items": [{"content": "Root", "children": [{"content": "Child"}]}]}
    )

    assert result.success is True
    assert isinstance(result.output, dict)
    assert result.output["pending_count"] == 2


def test_add_plan_item_tree_rejects_invalid_nodes_without_partial_insert() -> None:
    plugin = PlanPlugin()

    result = plugin.add_plan_item(
        items=[
            {"content": "Valid"},
            {"content": "", "children": [{"content": "Would be partial"}]},
        ]
    )

    assert result.success is False
    assert "items[1].content" in result.error
    listed = plugin.list_plan_items()
    assert isinstance(listed.output, dict)
    assert listed.output["flat_items"] == []


def test_plan_update_emits_plugin_events_for_gui() -> None:
    bus = EventBus()
    received: list[Event] = []
    bus.subscribe_blocking(received.append)
    plugin = PlanPlugin()
    plugin.bind_plugin_identity(plugin_id="plan", plugin_name="PlanPlugin")
    plugin.bind_event_bus(bus)

    try:
        plugin.add_plan_item("Write implementation")
        plugin.complete_plan_item("P1")
    finally:
        bus.close(wait=True, timeout=2)

    artifact_events = [
        event for event in received if event.type == "plugin.artifact.upsert"
    ]
    update_events = [
        event
        for event in received
        if event.type == "plugin.event"
        and event.payload.get("event_name") == "plan.item.updated"
    ]
    assert len(artifact_events) >= 2
    assert [event.payload["action"] for event in update_events] == [
        "added",
        "completed",
    ]
    assert update_events[-1].plugin_id == "plan"
    assert update_events[-1].payload["state"]["pending_count"] == 0
    assert (
        artifact_events[-1].payload["artifact"]["metadata"]["items"][0]["completed"]
        is True
    )


def test_after_conversation_reinvokes_when_plan_is_unfinished() -> None:
    bus = EventBus()
    received: list[Event] = []
    bus.subscribe_blocking(received.append)
    plugin = PlanPlugin()
    plugin.bind_event_bus(bus)
    plugin.add_plan_item("Finish the task")

    try:
        result = plugin.notify_unfinished_plan(
            SimpleNamespace(),
            HookContext(run_id="run-1", iteration=1),
        )
    finally:
        bus.close(wait=True, timeout=2)

    assert result is not None
    assert result.action == "reinvoke"
    assert result.message is not None
    reminder = str(result.message)
    assert PLAN_REMINDER_BEGIN in reminder
    assert "origin: automatic runtime reminder, not a human user message" in reminder
    assert "Finish the task" in reminder
    assert any(event.type == "plugin.message" for event in received)


def test_after_conversation_does_not_reinvoke_when_done_or_paused() -> None:
    plugin = PlanPlugin()
    plugin.add_plan_item("Done item")
    plugin.complete_plan_item("all")

    result = plugin.notify_unfinished_plan(
        SimpleNamespace(),
        HookContext(run_id="run-1", iteration=1),
    )
    assert result is None

    plugin.add_plan_item("Deferred item")
    paused = plugin.plan_control("pause", "Waiting on external input")
    assert paused.success is True
    assert isinstance(paused.output, dict)
    assert paused.output["plan_paused"] is True
    assert paused.output["pause_reason"] == "Waiting on external input"

    result = plugin.notify_unfinished_plan(
        SimpleNamespace(),
        HookContext(run_id="run-2", iteration=1),
    )
    assert result is None

    resumed = plugin.plan_control("continue")
    assert resumed.success is True
    assert isinstance(resumed.output, dict)
    assert resumed.output["plan_paused"] is False

    result = plugin.notify_unfinished_plan(
        SimpleNamespace(),
        HookContext(run_id="run-3", iteration=1),
    )
    assert result is not None
    assert result.action == "reinvoke"


def test_plan_prompt_injection_is_idempotent() -> None:
    plugin = PlanPlugin()
    agent = SimpleNamespace(
        context=AgentContext(
            system_prompt=[
                {"type": "text", "text": "Base prompt"},
                {"type": "text", "text": f"{PLAN_PROMPT_BEGIN}\nstale"},
            ]
        )
    )

    plugin.inject_plan_instructions(agent, HookContext(run_id="run-1", iteration=0))
    plugin.inject_plan_instructions(agent, HookContext(run_id="run-2", iteration=0))

    prompts = agent.context.system_prompt
    assert prompts is not None
    plan_prompts = [
        part
        for part in prompts
        if isinstance(part, dict) and PLAN_PROMPT_BEGIN in str(part.get("text", ""))
    ]
    assert len(plan_prompts) == 1
    assert "parent_id" in plan_prompts[0]["text"]
    assert "whole plan tree" in plan_prompts[0]["text"]
    assert "items=[{content, children, kind}]" in plan_prompts[0]["text"]
    assert "exploratory" in plan_prompts[0]["text"]
    assert "determinate" in plan_prompts[0]["text"]
    assert "parent_review_required" in plan_prompts[0]["text"]
    assert "not a request to create a plan file" in plan_prompts[0]["text"]
    assert "Do not create, edit, or store plan.md" in plan_prompts[0]["text"]
    assert "plan_control" in plan_prompts[0]["text"]
    assert "mark_all_children" in plan_prompts[0]["text"]
    assert "handoff_notes" in plan_prompts[0]["text"]
    assert "action='abandon'" in plan_prompts[0]["text"]
    assert "Pass query to search folded contexts by keyword" in plan_prompts[0]["text"]
    assert "Use max_chars to limit" in plan_prompts[0]["text"]
    assert "message_start/message_end" in plan_prompts[0]["text"]
    assert PLAN_REMINDER_BEGIN in plan_prompts[0]["text"]
    assert "not human-user messages" in plan_prompts[0]["text"]
    assert "may appear in the conversation as user-role messages" in plan_prompts[0]["text"]
    assert "cancel_plan_notification" not in plan_prompts[0]["text"]


def test_plan_control_validates_pause_reason_and_action() -> None:
    plugin = PlanPlugin()

    missing_reason = plugin.plan_control("pause")
    assert missing_reason.success is False
    assert "reason is required" in missing_reason.error

    invalid = plugin.plan_control("stop")
    assert invalid.success is False
    assert "pause" in invalid.error
    assert "continue" in invalid.error
    assert "abandon" in invalid.error


def test_plan_control_tool_replaces_cancel_notification_tool() -> None:
    plugin = PlanPlugin()
    tool_names = {tool.name for tool in plugin.tools}

    assert "plan_control" in tool_names
    assert "cancel_plan_notification" not in tool_names


def test_plan_control_abandon_clears_current_plan() -> None:
    plugin = PlanPlugin()
    plugin.add_plan_item("Prepare release")
    plugin.add_plan_item("Run tests", parent_id="P1")
    plugin.plan_control("pause", "Need to stop this plan.")

    result = plugin.plan_control("abandon", "Wrong task.")

    assert result.success is True
    assert isinstance(result.output, dict)
    assert result.output["items"] == []
    assert result.output["flat_items"] == []
    assert result.output["pending_count"] == 0
    assert result.output["plan_paused"] is False
    assert result.output["pause_reason"] == ""
    assert result.output["abandon_reason"] == "Wrong task."
    assert [item["id"] for item in result.output["abandoned_items"]] == ["P1", "P2"]

    new_item = plugin.add_plan_item("New plan")
    assert isinstance(new_item.output, dict)
    assert new_item.output["item"]["id"] == "P1"


def test_complete_plan_tool_schema_uses_mark_all_children() -> None:
    plugin = PlanPlugin()
    tool = next(tool for tool in plugin.tools if tool.name == "complete_plan_item")

    assert "mark_all_children" in tool.parameters_schema["properties"]
    assert "handoff_notes" in tool.parameters_schema["properties"]
    assert "complete_children" not in tool.parameters_schema["properties"]


def test_context_folding_requires_summary_when_enabled() -> None:
    plugin = PlanPlugin(fold_completed_tasks=True)
    plugin.add_plan_item("Implement feature")

    result = plugin.complete_plan_item("P1")

    assert result.success is False
    assert "summary is required" in result.error
    listed = plugin.list_plan_items()
    assert isinstance(listed.output, dict)
    assert listed.output["flat_items"][0]["completed"] is False

    missing_handoff = plugin.complete_plan_item("P1", summary="Implementation is done.")
    assert missing_handoff.success is False
    assert "handoff_notes is required" in missing_handoff.error


def test_complete_plan_item_folds_context_and_can_read_it() -> None:
    plugin = PlanPlugin(fold_completed_tasks=True)
    plugin.add_plan_item("Implement feature")
    context = AgentContext(
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "Please implement the feature."}],
                "name": None,
                "metadata": None,
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "call-read",
                        "name": "read_file",
                        "description": "Read a source file",
                        "arguments": {"path": "annual_report_2025.md"},
                    }
                ],
                "name": None,
                "metadata": None,
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "I changed the backend."}],
                "name": None,
                "metadata": None,
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "call-complete",
                        "name": "complete_plan_item",
                        "arguments": {
                            "item_id": "P1",
                            "summary": "Backend implementation is done.",
                            "handoff_notes": "Remember the new API shape.",
                        },
                    }
                ],
                "name": None,
                "metadata": None,
            },
        ]
    )
    plugin._active_completion_tool_call_id = "call-complete"

    result = plugin.complete_plan_item(
        "P1",
        summary="Backend implementation is done.",
        handoff_notes="Remember the new API shape.",
        ctx=SimpleNamespace(context=context),
    )

    assert result.success is True
    assert len(context.messages) == 1
    assert context.messages[0]["content"][0]["id"] == "call-complete"
    assert isinstance(result.output, str)
    assert "Folded context:" in result.output
    assert "Fold id: `PF1`" in result.output
    assert "Item id: `P1`" in result.output
    assert "Folded messages: 3" in result.output
    assert "Folded message previews:" in result.output
    assert "1. user: Please implement the feature." in result.output
    assert "2. assistant: tool call `read_file` - Read a source file" in result.output
    assert "3. assistant: I changed the backend." in result.output
    assert 'read_completed_task_context(task_id="P1", message_start=1, message_end=3)' in result.output
    assert "Task summary:" in result.output
    assert "Backend implementation is done." in result.output
    assert "Information for later tasks:" in result.output
    assert "Remember the new API shape." in result.output

    read = plugin.read_completed_task_context("P1")
    assert read.success is True
    assert isinstance(read.output, dict)
    assert read.output["summary"] == "Backend implementation is done."
    assert read.output["handoff_notes"] == "Remember the new API shape."
    assert "Please implement the feature." in read.output["transcript"]
    assert "I changed the backend." in read.output["transcript"]

    ranged = plugin.read_completed_task_context(
        task_id="P1",
        message_start=2,
        message_end=2,
    )
    assert ranged.success is True
    assert isinstance(ranged.output, dict)
    assert ranged.output["message_start"] == 2
    assert ranged.output["message_end"] == 2
    assert ranged.output["selected_message_count"] == 1
    assert "Message 2 (assistant)" in ranged.output["transcript"]
    assert "name=read_file" in ranged.output["transcript"]
    assert "description=Read a source file" in ranged.output["transcript"]
    assert "Please implement the feature." not in ranged.output["transcript"]

    limited = plugin.read_completed_task_context("P1", max_chars=80)
    assert limited.success is True
    assert isinstance(limited.output, dict)
    assert limited.output["mode"] == "read"
    assert limited.output["truncated"] is True
    assert len(limited.output["transcript"]) <= 80

    search = plugin.read_completed_task_context(
        query="backend",
        max_chars=60,
        context_chars=20,
    )
    assert search.success is True
    assert isinstance(search.output, dict)
    assert search.output["mode"] == "search"
    assert search.output["query"] == "backend"
    assert search.output["searched_context_count"] == 1
    assert search.output["match_count"] >= 1
    assert search.output["matches_returned"] >= 1
    assert sum(len(match["snippet"]) for match in search.output["matches"]) <= 60
    assert "backend" in search.output["matches"][0]["snippet"].lower()


def test_read_completed_task_context_tool_schema_supports_search_and_limits() -> None:
    plugin = PlanPlugin(fold_completed_tasks=True)
    tool = next(tool for tool in plugin.tools if tool.name == "read_completed_task_context")
    properties = tool.parameters_schema["properties"]

    assert "query" in properties
    assert "task_id" in properties
    assert "message_start" in properties
    assert "message_end" in properties
    assert "message_range" in properties
    assert "case_sensitive" in properties
    assert "max_matches" in properties
    assert "context_chars" in properties
    assert "max_chars" in properties


def test_context_folding_preserves_previous_completion_marker() -> None:
    plugin = PlanPlugin(fold_completed_tasks=True)
    plugin.add_plan_item("First task")
    plugin.add_plan_item("Second task")
    context = AgentContext(
        messages=[
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "call-first",
                        "name": "complete_plan_item",
                        "arguments": {
                            "item_id": "P1",
                            "summary": "First task done.",
                            "handoff_notes": "First task note.",
                        },
                    }
                ],
                "name": None,
                "metadata": None,
            },
            {
                "role": "tool",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_call_id": "call-first",
                        "content": [{"type": "text", "text": "P1 folded."}],
                        "is_error": False,
                    }
                ],
                "name": None,
                "metadata": None,
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "Now do the second task."}],
                "name": None,
                "metadata": None,
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "call-second",
                        "name": "complete_plan_item",
                        "arguments": {
                            "item_id": "P2",
                            "summary": "Second task done.",
                            "handoff_notes": "Remember the follow-up detail.",
                        },
                    }
                ],
                "name": None,
                "metadata": None,
            },
        ]
    )
    plugin._active_completion_tool_call_id = "call-second"

    result = plugin.complete_plan_item(
        "P2",
        summary="Second task done.",
        handoff_notes="Remember the follow-up detail.",
        ctx=SimpleNamespace(context=context),
    )

    assert result.success is True
    assert [message["role"] for message in context.messages] == [
        "assistant",
        "tool",
        "assistant",
    ]
    assert context.messages[0]["content"][0]["id"] == "call-first"
    assert context.messages[2]["content"][0]["id"] == "call-second"
    read = plugin.read_completed_task_context("P2")
    assert isinstance(read.output, dict)
    assert "Now do the second task." in read.output["transcript"]


def test_complete_plan_tool_uses_runtime_context_injection() -> None:
    plugin = PlanPlugin(fold_completed_tasks=True)
    tool = next(tool for tool in plugin.tools if tool.name == "complete_plan_item")

    assert tool.context == "ctx"
    assert "ctx" not in tool.parameters_schema["properties"]
