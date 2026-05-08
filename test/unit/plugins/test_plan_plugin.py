from __future__ import annotations

from types import SimpleNamespace

from hawi.agent.context import AgentContext
from hawi.events import Event, EventBus
from hawi.plugin import HookContext
from hawi_plugins.plan_plugin import PlanPlugin
from hawi_plugins.plan_plugin.plugin import PLAN_PROMPT_BEGIN


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

    completed = plugin.complete_plan_item("P1", complete_children=True)
    assert completed.success is True
    output = completed.output
    assert isinstance(output, dict)
    assert output["pending_count"] == 0
    assert {item["id"] for item in output["completed"]} == {"P1", "P2"}


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
    assert isinstance(output, dict)
    assert output["parent_review_required"] == []
    assert output["pending_count"] == 2

    second = plugin.complete_plan_item("P3")
    assert second.success is True
    output = second.output
    assert isinstance(output, dict)
    assert {item["id"] for item in output["completed"]} == {"P3"}
    assert output["pending_count"] == 1
    assert len(output["parent_review_required"]) == 1
    review = output["parent_review_required"][0]
    assert review["id"] == "P1"
    assert review["kind"] == "exploratory"
    assert "exploratory" in review["reason"]


def test_determinate_parent_auto_completes_when_children_done() -> None:
    plugin = PlanPlugin()
    plugin.add_plan_item("Prepare release", kind="determinate")
    plugin.add_plan_item("Run tests", parent_id="P1")
    plugin.add_plan_item("Tag release", parent_id="P1")

    plugin.complete_plan_item("P2")
    final = plugin.complete_plan_item("P3")
    assert final.success is True
    output = final.output
    assert isinstance(output, dict)
    assert {item["id"] for item in output["completed"]} == {"P3", "P1"}
    assert output["parent_review_required"] == []
    assert output["pending_count"] == 0


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
    assert isinstance(output, dict)
    assert {item["id"] for item in output["completed"]} == {"P3", "P2"}
    assert [r["id"] for r in output["parent_review_required"]] == ["P1"]
    assert output["pending_count"] == 1


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
    assert "Finish the task" in str(result.message)
    assert any(event.type == "plugin.message" for event in received)


def test_after_conversation_does_not_reinvoke_when_done_or_cancelled() -> None:
    plugin = PlanPlugin()
    plugin.add_plan_item("Done item")
    plugin.complete_plan_item("all")

    result = plugin.notify_unfinished_plan(
        SimpleNamespace(),
        HookContext(run_id="run-1", iteration=1),
    )
    assert result is None

    plugin.add_plan_item("Deferred item")
    plugin.cancel_plan_notification("Waiting on external input")
    result = plugin.notify_unfinished_plan(
        SimpleNamespace(),
        HookContext(run_id="run-2", iteration=1),
    )
    assert result is None


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
