from __future__ import annotations

import asyncio

import pytest

from hawi.tool import ToolResult
from hawi_plugins.taskflow_plugin import TaskflowPlugin


def _create_two_step_workflow(plugin: TaskflowPlugin) -> ToolResult:
    return plugin.create_taskflow(
        title="Release Flow",
        mode="workflow",
        execution_policy="gated_graph",
        mutable=False,
        start_step_id="research",
        steps=[
            {
                "id": "research",
                "title": "Research",
                "instructions": "Gather the facts.",
                "review": {"type": "logger"},
            },
            {
                "id": "write",
                "title": "Write",
                "instructions": "Write the final answer.",
                "review": {"type": "logger"},
            },
        ],
        edges=[
            {
                "from": "research",
                "to": "write",
                "type": "transitions",
            }
        ],
    )


def test_taskflow_plan_supports_tree_and_auto_parent_completion() -> None:
    plugin = TaskflowPlugin()

    added = plugin.add_taskflow_steps(
        steps=[
            {
                "title": "Prepare release",
                "children": [
                    {"title": "Run tests"},
                    {"title": "Tag release"},
                ],
            }
        ]
    )
    assert added.success is True

    first = plugin.submit_taskflow_step(step_id="T2", output="Tests passed")
    assert first.success is True
    status = plugin.get_taskflow_status()
    assert status.output["taskflow"]["steps"][0]["status"] == "pending"
    assert status.output["open_step_count"] == 2

    second = plugin.submit_taskflow_step(step_id="T3", output="Tag created")
    assert second.success is True
    status = plugin.get_taskflow_status()
    steps = {
        step["id"]: step
        for step in status.output["taskflow"]["steps"]
    }
    assert steps["T1"]["status"] == "completed"
    assert steps["T2"]["status"] == "completed"
    assert steps["T3"]["status"] == "completed"
    assert status.output["open_step_count"] == 0


def test_taskflow_workflow_routes_logger_reviewed_steps() -> None:
    plugin = TaskflowPlugin()
    created = _create_two_step_workflow(plugin)
    assert created.success is True

    started = plugin.start_taskflow()
    assert started.success is True
    assert started.output["run"]["current_step_id"] == "research"

    completed = plugin.submit_taskflow_step(output="Research complete")
    assert completed.success is True
    state = completed.output["state"]
    assert state["run"]["current_step_id"] == "write"
    assert state["run"]["status"] == "running"
    assert state["taskflow"]["steps"][0]["status"] == "completed"
    assert state["taskflow"]["steps"][0]["review_records"][0]["reviewer_type"] == "logger"
    assert state["taskflow"]["steps"][1]["status"] == "active"

    terminal = plugin.submit_taskflow_step(output="Final answer")
    assert terminal.success is True
    assert terminal.output["state"]["run"]["status"] == "completed"
    assert terminal.output["state"]["run"]["current_step_id"] is None


def test_taskflow_sequential_workflow_routes_by_step_order() -> None:
    plugin = TaskflowPlugin()
    created = plugin.create_taskflow(
        title="Sequential Flow",
        mode="workflow",
        execution_policy="sequential",
        mutable=False,
        steps=[
            {"id": "one", "title": "One"},
            {"id": "two", "title": "Two"},
        ],
    )
    assert created.success is True
    assert plugin.start_taskflow().success is True

    first = plugin.submit_taskflow_step(output="one done")
    assert first.success is True
    assert first.output["state"]["run"]["current_step_id"] == "two"
    assert first.output["state"]["taskflow"]["steps"][1]["status"] == "active"

    second = plugin.submit_taskflow_step(output="two done")
    assert second.success is True
    assert second.output["state"]["run"]["status"] == "completed"


def test_taskflow_workflow_supports_conditional_loop_and_exit() -> None:
    plugin = TaskflowPlugin()
    created = plugin.create_taskflow(
        title="Loop Flow",
        mode="workflow",
        execution_policy="gated_graph",
        mutable=False,
        start_step_id="check",
        steps=[
            {"id": "check", "title": "Check quality", "instructions": "Inspect the result."},
            {"id": "finish", "title": "Finish", "instructions": "Return the final output."},
        ],
        edges=[
            {
                "from": "check",
                "to": "check",
                "type": "transitions",
                "label": "retry",
                "condition": "Output is not good enough yet.",
            },
            {
                "from": "check",
                "to": "finish",
                "type": "transitions",
                "label": "exit",
                "condition": "Output satisfies the quality bar.",
            },
        ],
    )
    assert created.success is True
    assert plugin.start_taskflow().success is True

    submitted = plugin.submit_taskflow_step(output="needs another pass")
    assert submitted.success is True
    assert "Multiple conditional transitions" in submitted.output["next_message"]
    assert submitted.output["state"]["run"]["current_step_id"] == "check"
    assert submitted.output["state"]["taskflow"]["steps"][0]["status"] == "completed"
    assert submitted.output["state"]["current_transitions"][0]["is_loop"] is True

    looped = plugin.select_next_taskflow_step(
        next_step_id="check",
        reason="Output is not good enough yet.",
    )
    assert looped.success is True
    assert looped.output["state"]["run"]["current_step_id"] == "check"
    check_step = looped.output["state"]["taskflow"]["steps"][0]
    assert check_step["status"] == "active"
    assert check_step["output"] is None
    assert check_step["completed_at"] is None

    submitted_again = plugin.submit_taskflow_step(output="good enough")
    assert submitted_again.success is True
    assert "Multiple conditional transitions" in submitted_again.output["next_message"]

    exited = plugin.select_next_taskflow_step(
        next_step_id="finish",
        reason="Output satisfies the quality bar.",
    )
    assert exited.success is True
    assert exited.output["state"]["run"]["current_step_id"] == "finish"
    assert exited.output["state"]["taskflow"]["steps"][1]["status"] == "active"

    done = plugin.submit_taskflow_step(output="final")
    assert done.success is True
    assert done.output["state"]["run"]["status"] == "completed"


def test_taskflow_rejects_mutation_of_immutable_workflow() -> None:
    plugin = TaskflowPlugin()
    created = _create_two_step_workflow(plugin)
    assert created.success is True

    added = plugin.add_taskflow_steps(steps=[{"title": "Sneak in another step"}])

    assert added.success is False
    assert "not mutable" in added.error


@pytest.mark.asyncio
async def test_taskflow_human_review_api_approves_pending_step() -> None:
    plugin = TaskflowPlugin()
    created = plugin.create_taskflow(
        title="Human Gate",
        mode="workflow",
        execution_policy="gated_graph",
        mutable=False,
        start_step_id="draft",
        steps=[
            {
                "id": "draft",
                "title": "Draft",
                "instructions": "Produce a draft.",
                "review": {"type": "human"},
            }
        ],
    )
    assert created.success is True
    assert plugin.start_taskflow().success is True

    submitted = plugin.submit_taskflow_step(output="rough draft")
    assert submitted.success is True
    assert submitted.output["review_pending"] is True
    assert submitted.output["review_type"] == "human"
    assert plugin._pending_human_reviews

    hook_result = await asyncio.wait_for(
        plugin.review_submitted_step(
            agent=None,
            tool_name="submit_taskflow_step",
            arguments={},
            result=submitted,
            ctx=None,
        ),
        timeout=0.1,
    )
    assert hook_result is None

    review_id = next(iter(plugin._pending_human_reviews))
    approved = plugin.approve_taskflow_review(
        review_id,
        feedback="Looks good.",
        modified_output="approved draft",
    )
    assert approved.success is True
    assert approved.output["approved"] is True
    assert not plugin._pending_human_reviews

    status = plugin.get_taskflow_status()
    step = status.output["taskflow"]["steps"][0]
    assert step["status"] == "completed"
    assert step["output"] == "approved draft"
    assert step["review_records"][0]["reviewer_type"] == "human"
    assert status.output["run"]["status"] == "completed"


def test_taskflow_save_and_load_round_trip(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HAWI_TASKFLOW_DIR", str(tmp_path))
    plugin = TaskflowPlugin()
    created = plugin.create_taskflow(
        title="Saved Flow",
        mode="plan",
        steps=[
            {"id": "a", "title": "Alpha"},
            {"id": "b", "title": "Beta"},
        ],
    )
    assert created.success is True
    saved = plugin.save_taskflow()
    assert saved.success is True

    other = TaskflowPlugin()
    listed = other.list_taskflows()
    assert listed.success is True
    assert listed.output["taskflows"][0]["name"] == "Saved Flow"

    loaded = other.load_taskflow("Saved Flow")
    assert loaded.success is True
    assert loaded.output["taskflow"]["title"] == "Saved Flow"
    assert len(loaded.output["taskflow"]["steps"]) == 2
