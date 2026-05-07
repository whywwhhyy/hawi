"""Unit tests for WorkflowPlugin — simplified agent tools + human APIs."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hawi.tool import ToolResult

from hawi_plugins.workflow_plugin.models import (
    NodeExecution,
    ReviewConfig,
    ReviewDecision,
    ReviewRecord,
    Workflow,
    WorkflowEdge,
    WorkflowNode,
    WorkflowRun,
)
from hawi_plugins.workflow_plugin.reviewers import (
    HumanReviewer,
    LoggerReviewer,
    SubAgentReviewer,
)
from hawi_plugins.workflow_plugin.plugin import (
    WorkflowPlugin,
    GATE_PROMPT_BEGIN,
    GATE_PROMPT_END,
)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _make_two_node_wf() -> Workflow:
    wf = Workflow(id="wf1", name="Test WF", start_node_id="research")
    wf.add_node(WorkflowNode(id="research", name="Research", prompt="Research.", review=ReviewConfig(type="logger")))
    wf.add_node(WorkflowNode(id="write", name="Write", prompt="Write.", review=ReviewConfig(type="logger")))
    wf.add_edge(WorkflowEdge("research", "write"))
    return wf


def _make_branching_wf() -> Workflow:
    wf = Workflow(id="wf_branch", name="Branch WF", start_node_id="research")
    wf.add_node(WorkflowNode(
        id="research",
        name="Research",
        prompt="Research and decide the path.",
        review=ReviewConfig(type="logger"),
    ))
    wf.add_node(WorkflowNode(
        id="write",
        name="Write",
        prompt="Write the final answer.",
        review=ReviewConfig(type="logger"),
    ))
    wf.add_node(WorkflowNode(
        id="escalate",
        name="Escalate",
        prompt="Escalate for deeper analysis.",
        review=ReviewConfig(type="logger"),
    ))
    wf.add_edge(WorkflowEdge("research", "write", label="ready"))
    wf.add_edge(WorkflowEdge("research", "escalate", label="needs_more_work"))
    return wf


def _inject_workflow(plugin: WorkflowPlugin, wf: Workflow) -> None:
    """Directly set the workflow on the plugin (bypasses load_workflow)."""
    plugin._workflow = wf


# ═══════════════════════════════════════════════════════════════════
# Plugin tools
# ═══════════════════════════════════════════════════════════════════


class TestLoadAndList:
    def test_load_workflow(self, tmp_path: Path, monkeypatch):
        from hawi_plugins.workflow_plugin import persistence
        monkeypatch.setattr(persistence, "_ensure_dir", lambda: tmp_path)
        monkeypatch.setattr(persistence, "_workflow_path",
                           lambda name: tmp_path / f"{name}.yaml")

        wf = _make_two_node_wf()
        persistence.save_workflow(wf)

        plugin = WorkflowPlugin()
        plugin._manual_read = True  # skip manual guard for test
        r = plugin.load_workflow("Test WF")
        assert r.success
        assert plugin._workflow.name == "Test WF"
        assert len(plugin._workflow.nodes) == 2

    def test_load_workflow_not_found(self):
        plugin = WorkflowPlugin()
        plugin._manual_read = True
        r = plugin.load_workflow("nonexistent")
        assert not r.success
        assert "not found" in r.error

    def test_list_workflows(self, tmp_path: Path, monkeypatch):
        from hawi_plugins.workflow_plugin import persistence
        monkeypatch.setattr(persistence, "_ensure_dir", lambda: tmp_path)
        monkeypatch.setattr(persistence, "_workflow_path",
                           lambda name: tmp_path / f"{name}.yaml")

        persistence.save_workflow(_make_two_node_wf())

        plugin = WorkflowPlugin()
        r = plugin.list_workflows()
        assert r.success
        assert len(r.output["workflows"]) >= 1

    def test_run_workflow_auto_loads(self, tmp_path: Path, monkeypatch):
        from hawi_plugins.workflow_plugin import persistence
        monkeypatch.setattr(persistence, "_ensure_dir", lambda: tmp_path)
        monkeypatch.setattr(persistence, "_workflow_path",
                           lambda name: tmp_path / f"{name}.yaml")

        persistence.save_workflow(_make_two_node_wf())

        plugin = WorkflowPlugin()
        plugin._manual_read = True
        r = plugin.run_workflow("Test WF", initial_input="Start.")
        assert r.success
        assert plugin._active_run.current_node_id == "research"
        assert plugin._active_run.status == "running"

    def test_run_workflow_uses_loaded(self):
        plugin = WorkflowPlugin()
        _inject_workflow(plugin, _make_two_node_wf())
        r = plugin.run_workflow("Test WF")
        assert r.success


class TestCompleteNode:
    def test_no_active_run(self):
        plugin = WorkflowPlugin()
        r = plugin.complete_workflow_node(output="done")
        assert not r.success
        assert "No active workflow" in r.error

    def test_empty_output(self):
        plugin = WorkflowPlugin()
        _inject_workflow(plugin, _make_two_node_wf())
        plugin.run_workflow("Test WF")
        r = plugin.complete_workflow_node(output="   ")
        assert not r.success

    def test_success(self):
        plugin = WorkflowPlugin()
        _inject_workflow(plugin, _make_two_node_wf())
        plugin.run_workflow("Test WF")
        r = plugin.complete_workflow_node(output="Research done.")
        assert r.success
        assert r.output["output_submitted"] is True
        assert r.output["review_pending"] is True


class TestSelectNextNode:
    def test_no_active_run(self):
        plugin = WorkflowPlugin()
        r = plugin.select_next_workflow_node(
            next_node_id="write",
            reason="The research is complete.",
        )
        assert not r.success
        assert "No active workflow" in r.error

    def test_requires_reason(self):
        plugin = WorkflowPlugin()
        _inject_workflow(plugin, _make_branching_wf())
        plugin.run_workflow("Branch WF")

        r = plugin.select_next_workflow_node(next_node_id="write", reason="  ")

        assert not r.success
        assert "reason" in r.error

    def test_rejects_non_downstream_node(self):
        plugin = WorkflowPlugin()
        wf = _make_branching_wf()
        wf.add_node(WorkflowNode(
            id="archive",
            name="Archive",
            prompt="Archive.",
            review=ReviewConfig(type="logger"),
        ))
        _inject_workflow(plugin, wf)
        plugin.run_workflow("Branch WF")

        r = plugin.select_next_workflow_node(
            next_node_id="archive",
            reason="This should not be reachable directly.",
        )

        assert not r.success
        assert "not an immediate downstream" in r.error

    def test_success_records_routing_decision(self):
        plugin = WorkflowPlugin()
        _inject_workflow(plugin, _make_branching_wf())
        plugin.run_workflow("Branch WF")

        r = plugin.select_next_workflow_node(
            next_node_id="escalate",
            reason="The research found an unresolved risk.",
        )

        assert r.success
        execution = plugin._active_run.current_execution()
        assert execution.selected_next_node_id == "escalate"
        assert execution.routing_reason == "The research found an unresolved risk."

        status = plugin.get_workflow_status()
        current = status.output["node_executions"]["research"]
        assert current["selected_next_node_id"] == "escalate"
        assert current["routing_reason"] == "The research found an unresolved risk."


class TestGetStatus:
    def test_idle_and_running(self):
        plugin = WorkflowPlugin()
        _inject_workflow(plugin, _make_two_node_wf())

        r = plugin.get_workflow_status()
        assert r.output["status"] == "idle"

        plugin.run_workflow("Test WF")
        r = plugin.get_workflow_status()
        assert r.output["status"] == "running"
        assert r.output["current_node_id"] == "research"

    def test_no_workflow(self):
        plugin = WorkflowPlugin()
        r = plugin.get_workflow_status()
        assert r.output["status"] == "no_workflow"


class TestGetPendingReviews:
    def test_shows_human_count_without_review_ids(self):
        plugin = WorkflowPlugin()
        _inject_workflow(plugin, _make_two_node_wf())
        plugin._pending_human_reviews["r1"] = asyncio.get_event_loop().create_future()

        r = plugin.get_pending_reviews()
        assert r.output["human_review_count"] >= 1
        for item in r.output["pending_reviews"]:
            assert "review_id" not in item


class TestClone:
    def test_clone_preserves_workflow_and_run(self):
        plugin = WorkflowPlugin()
        _inject_workflow(plugin, _make_two_node_wf())
        plugin.run_workflow("Test WF")
        clone = plugin.clone()
        assert clone._workflow.name == "Test WF"
        assert clone._active_run.workflow_id == plugin._active_run.workflow_id

    def test_clone_preserves_routing_decision(self):
        plugin = WorkflowPlugin()
        _inject_workflow(plugin, _make_branching_wf())
        plugin.run_workflow("Branch WF")
        plugin.select_next_workflow_node(
            next_node_id="escalate",
            reason="Needs a deeper pass.",
        )

        clone = plugin.clone()
        execution = clone._active_run.current_execution()
        assert execution.selected_next_node_id == "escalate"
        assert execution.routing_reason == "Needs a deeper pass."



# ═══════════════════════════════════════════════════════════════════
# Human review API (plain methods, NOT @tool)
# ═══════════════════════════════════════════════════════════════════


class TestHumanReviewAPI:
    def test_approve(self):
        plugin = WorkflowPlugin()
        plugin._pending_human_reviews["rid1"] = asyncio.get_event_loop().create_future()
        r = plugin.approve_workflow_node("rid1", feedback="Good")
        assert r.success
        assert plugin._pending_human_reviews["rid1"].result().approved

    def test_approve_already_resolved(self):
        plugin = WorkflowPlugin()
        plugin._pending_human_reviews["rid1"] = asyncio.get_event_loop().create_future()
        plugin.approve_workflow_node("rid1")
        r = plugin.approve_workflow_node("rid1")
        assert not r.success

    def test_reject(self):
        plugin = WorkflowPlugin()
        plugin._pending_human_reviews["rid2"] = asyncio.get_event_loop().create_future()
        r = plugin.reject_workflow_node("rid2", feedback="Try again.")
        assert r.success
        d = plugin._pending_human_reviews["rid2"].result()
        assert not d.approved
        assert d.feedback == "Try again."

    def test_reject_requires_feedback(self):
        plugin = WorkflowPlugin()
        plugin._pending_human_reviews["rid3"] = asyncio.get_event_loop().create_future()
        r = plugin.reject_workflow_node("rid3", feedback="  ")
        assert not r.success

    def test_unknown_review_id(self):
        plugin = WorkflowPlugin()
        r = plugin.approve_workflow_node("unknown")
        assert not r.success
        assert "No pending review" in r.error


# ═══════════════════════════════════════════════════════════════════
# Hook tests
# ═══════════════════════════════════════════════════════════════════


class TestHooks:
    def _setup(self) -> tuple[WorkflowPlugin, MagicMock]:
        plugin = WorkflowPlugin()
        _inject_workflow(plugin, Workflow(
            id="wf_hook", name="Hook WF", start_node_id="step1",
            nodes={
                "step1": WorkflowNode(id="step1", name="S1", prompt="Do step 1.", review=ReviewConfig(type="logger")),
                "step2": WorkflowNode(id="step2", name="S2", prompt="Do step 2.", review=ReviewConfig(type="logger")),
            },
            edges=[WorkflowEdge("step1", "step2")],
        ))
        plugin.run_workflow("Hook WF")
        agent = MagicMock()
        agent.context.system_prompt = [{"type": "text", "text": "original"}]
        return plugin, agent

    def test_before_conversation_injects_gate_prompt(self):
        plugin, agent = self._setup()
        plugin.inject_gate_context(agent, MagicMock())
        last_text = agent.context.system_prompt[-1]["text"]
        assert GATE_PROMPT_BEGIN in last_text
        assert "step1" in last_text
        assert "Do step 1." in last_text

    def test_before_conversation_idempotent(self):
        plugin, agent = self._setup()
        plugin.inject_gate_context(agent, MagicMock())
        plugin.inject_gate_context(agent, MagicMock())
        gate_blocks = [p for p in agent.context.system_prompt
                       if isinstance(p, dict) and GATE_PROMPT_BEGIN in str(p.get("text", ""))]
        assert len(gate_blocks) == 1

    def test_before_conversation_no_workflow_shows_guidance(self):
        plugin = WorkflowPlugin()
        agent = MagicMock()
        agent.context.system_prompt = [{"type": "text", "text": "original"}]
        plugin.inject_gate_context(agent, MagicMock())
        last_text = agent.context.system_prompt[-1]["text"]
        assert "workflow-plugin" in last_text
        assert "YAML format" in last_text
        assert "load_workflow" in last_text

    def test_validate_gate_call_rejects_when_no_run(self):
        plugin = WorkflowPlugin()
        result = plugin.validate_gate_call(MagicMock(), "complete_workflow_node", {}, MagicMock())
        assert result is not None
        assert result.action == "skip"

    def test_validate_gate_call_allows_when_active(self):
        plugin, _ = self._setup()
        result = plugin.validate_gate_call(MagicMock(), "complete_workflow_node", {}, MagicMock())
        assert result is None

    def test_validate_gate_call_ignores_other_tools(self):
        plugin = WorkflowPlugin()
        result = plugin.validate_gate_call(MagicMock(), "other_tool", {}, MagicMock())
        assert result is None

    @pytest.mark.asyncio
    async def test_gate_guard_logger_approves_and_advances(self):
        plugin, agent = self._setup()
        result_mock = ToolResult(success=True, output={})
        hr = await plugin.gate_guard(
            agent, "complete_workflow_node",
            {"output": "Step 1 done."}, result_mock, MagicMock(),
        )
        assert hr is not None
        assert hr.action == "reinvoke"
        assert "PASSED" in hr.message
        assert plugin._active_run.current_node_id == "step2"
        assert plugin._active_run.node_executions["step1"].status == "completed"

    @pytest.mark.asyncio
    async def test_gate_guard_uses_agent_selected_next_node(self):
        plugin = WorkflowPlugin()
        _inject_workflow(plugin, _make_branching_wf())
        plugin.run_workflow("Branch WF")
        plugin.select_next_workflow_node(
            next_node_id="escalate",
            reason="The research found an unresolved risk.",
        )

        hr = await plugin.gate_guard(
            MagicMock(), "complete_workflow_node",
            {"output": "Research found an unresolved risk."},
            ToolResult(success=True, output={}), MagicMock(),
        )

        assert hr is not None
        assert "Route selected by agent" in hr.message
        assert plugin._active_run.current_node_id == "escalate"
        assert plugin._active_run.node_executions["research"].status == "completed"
        assert plugin._active_run.node_executions["write"].status == "pending"

    @pytest.mark.asyncio
    async def test_gate_guard_ignores_other_tools(self):
        plugin, agent = self._setup()
        result = await plugin.gate_guard(agent, "other", {}, MagicMock(), MagicMock())
        assert result is None

    @pytest.mark.asyncio
    async def test_gate_guard_fails_on_max_retries(self):
        plugin = WorkflowPlugin()
        wf = Workflow(id="wf", name="Fail WF", start_node_id="s", nodes={
            "s": WorkflowNode(id="s", name="S", prompt="p", review=ReviewConfig(type="logger"), max_retries=1),
        })
        _inject_workflow(plugin, wf)
        plugin.run_workflow("Fail WF")
        plugin._active_run.node_executions["s"].attempt_count = 1

        with patch.object(plugin, "_resolve_reviewer") as mock_resolve:
            mock_reviewer = MagicMock()
            mock_reviewer.identity = "test"
            mock_reviewer.review = AsyncMock(return_value=ReviewDecision(approved=False, feedback="Bad."))
            mock_resolve.return_value = mock_reviewer

            hr = await plugin.gate_guard(
                MagicMock(), "complete_workflow_node",
                {"output": "bad"}, ToolResult(success=True, output={}), MagicMock(),
            )

        assert plugin._active_run.status == "failed"
        assert "FAILED" in hr.message

    @pytest.mark.asyncio
    async def test_gate_guard_terminal_node_completes(self):
        plugin = WorkflowPlugin()
        wf = Workflow(id="wf", name="Single WF", start_node_id="only", nodes={
            "only": WorkflowNode(id="only", name="Only", prompt="p", review=ReviewConfig(type="logger")),
        })
        _inject_workflow(plugin, wf)
        plugin.run_workflow("Single WF")

        hr = await plugin.gate_guard(
            MagicMock(), "complete_workflow_node",
            {"output": "Done."}, ToolResult(success=True, output={}), MagicMock(),
        )
        assert hr is None
        assert plugin._active_run.status == "completed"
