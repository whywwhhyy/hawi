"""WorkflowPlugin — agentic workflow with gated quality control.

Workflows are defined as YAML files in ``~/.hawi/workflows/`` (human-editable).
The plugin provides hooks for gate enforcement and tools for loading, listing,
running, and completing workflow gates.

Each node is a *gate*: the agent must call ``complete_workflow_node`` to
submit its output for review. Only after the reviewer approves does the
workflow advance to the next node.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from hawi.plugin import (
    HawiPlugin,
    HookResult,
    after_conversation,
    after_tool_calling,
    before_conversation,
    before_tool_calling,
    tool,
)
from hawi.tool import ToolResult

from hawi_plugins.workflow_plugin.models import (
    NodeExecution,
    ReviewDecision,
    ReviewRecord,
    Workflow,
    WorkflowNode,
    WorkflowRun,
)
from hawi_plugins.workflow_plugin.reviewers import (
    HumanReviewer,
    LoggerReviewer,
    SubAgentReviewer,
)

# ── Prompt markers ──

GATE_PROMPT_BEGIN = "<workflow-gate>"
GATE_PROMPT_END = "</workflow-gate>"


class WorkflowPlugin(HawiPlugin):
    """Workflow plugin for gated, reviewable agent execution.

    Hooks:
    - ``before_conversation``: injects current gate constraints + construction
      guidance into the system prompt.
    - ``after_tool_calling``: intercepts ``complete_workflow_node``, runs the
      configured reviewer, and either advances or sends feedback.

    Agent tools (8):
    - ``read_workflow_manual`` — read the workflow YAML format & usage guide
    - ``load_workflow``, ``list_workflows`` — discover & validate YAML workflows
    - ``run_workflow`` — start executing a workflow
    - ``select_next_workflow_node`` — choose the next downstream gate
    - ``complete_workflow_node`` — submit gate output (★ core)
    - ``get_workflow_status``, ``get_pending_reviews`` — inspect state

    Human review API (NOT agent tools — called by GUI/CLI):
    - ``approve_workflow_node(review_id, ...)``
    - ``reject_workflow_node(review_id, feedback)``

    Workflow construction:
    Workflows are YAML files.  Write them with filesystem tools, then call
    ``load_workflow`` to validate.  Humans can edit the YAML directly.
    """

    def __init__(self) -> None:
        self._workflow: Workflow | None = None
        self._active_run: WorkflowRun | None = None
        self._pending_human_reviews: dict[str, asyncio.Future[ReviewDecision]] = {}
        self._manual_read: bool = False

    # ═══════════════════════════════════════════════════════════════
    # GUI config
    # ═══════════════════════════════════════════════════════════════

    @classmethod
    def gui_config_schema(cls) -> dict:
        return {"type": "object", "properties": {}, "additionalProperties": False}

    @classmethod
    def gui_default_config(cls) -> dict:
        return {}

    def clone(self) -> "WorkflowPlugin":
        new = WorkflowPlugin()
        if self._workflow is not None:
            new._workflow = Workflow.from_dict(self._workflow.to_dict())
        if self._active_run is not None:
            new._active_run = WorkflowRun(
                id=self._active_run.id,
                workflow_id=self._active_run.workflow_id,
                status=self._active_run.status,
                current_node_id=self._active_run.current_node_id,
                node_executions={
                    nid: NodeExecution(
                        node_id=ne.node_id, status=ne.status, output=ne.output,
                        review_records=list(ne.review_records),
                        attempt_count=ne.attempt_count,
                        started_at=ne.started_at, completed_at=ne.completed_at,
                        selected_next_node_id=ne.selected_next_node_id,
                        routing_reason=ne.routing_reason,
                    )
                    for nid, ne in self._active_run.node_executions.items()
                },
                global_context=dict(self._active_run.global_context),
                created_at=self._active_run.created_at,
                completed_at=self._active_run.completed_at,
            )
        new._manual_read = self._manual_read
        return new

    def save_state(self) -> dict[str, Any] | None:
        """Persist workflow definition + active run for SessionManager.

        Pending human-review futures are deliberately NOT serialized — on load,
        a node in ``paused_awaiting_review`` stays paused and the GUI prompts
        the reviewer again.
        """
        if self._workflow is None and self._active_run is None:
            return None
        return {
            "workflow": self._workflow.to_dict() if self._workflow else None,
            "active_run": (
                self._active_run.to_dict() if self._active_run else None
            ),
            "manual_read": self._manual_read,
        }

    def load_state(self, data: dict[str, Any]) -> None:
        wf_dict = data.get("workflow")
        self._workflow = Workflow.from_dict(wf_dict) if wf_dict else None

        run_dict = data.get("active_run")
        if run_dict is None:
            self._active_run = None
        else:
            node_execs = {
                nid: NodeExecution(
                    node_id=ne.get("node_id", nid),
                    status=ne.get("status", "pending"),
                    output=ne.get("output"),
                    review_records=[
                        ReviewRecord(
                            node_id=r.get("node_id", ""),
                            reviewer_type=r.get("reviewer_type", "logger"),
                            approved=bool(r.get("approved", False)),
                            feedback=r.get("feedback", ""),
                            reviewer_identity=r.get("reviewer_identity"),
                            timestamp=float(r.get("timestamp", 0.0)),
                        )
                        for r in ne.get("review_records", [])
                    ],
                    attempt_count=int(ne.get("attempt_count", 0)),
                    started_at=ne.get("started_at"),
                    completed_at=ne.get("completed_at"),
                    selected_next_node_id=ne.get("selected_next_node_id"),
                    routing_reason=ne.get("routing_reason", ""),
                )
                for nid, ne in run_dict.get("node_executions", {}).items()
            }
            self._active_run = WorkflowRun(
                id=run_dict["id"],
                workflow_id=run_dict.get("workflow_id", ""),
                status=run_dict.get("status", "running"),
                current_node_id=run_dict.get("current_node_id", ""),
                node_executions=node_execs,
                global_context=dict(run_dict.get("global_context", {})),
                created_at=float(run_dict.get("created_at", 0.0)),
                completed_at=run_dict.get("completed_at"),
            )

        self._pending_human_reviews = {}
        self._manual_read = bool(data.get("manual_read", False))
    # ═══════════════════════════════════════════════════════════════

    @before_conversation
    def inject_gate_context(self, agent: Any, ctx: Any) -> None:
        """Inject current gate constraints + construction guidance."""
        system_prompt = list(agent.context.system_prompt or [])

        # Always inject construction guidance (if no workflow is loaded)
        if self._workflow is None:
            system_prompt = self._strip_gate_blocks(system_prompt)
            system_prompt.append({"type": "text", "text": self._construction_guidance()})
            agent.context.system_prompt = system_prompt
            return

        run = self._active_run
        if not run or run.status != "running":
            return

        node = self._current_node()
        if node is None:
            return

        prompt = self._build_gate_prompt(node, run)
        system_prompt = self._strip_gate_blocks(system_prompt)
        system_prompt.append({"type": "text", "text": prompt})
        agent.context.system_prompt = system_prompt

    # ═══════════════════════════════════════════════════════════════
    # Hook: after_tool_calling — gate guard
    # ═══════════════════════════════════════════════════════════════

    @after_tool_calling
    async def gate_guard(
        self, agent: Any, tool_name: str, arguments: dict, result: Any, ctx: Any
    ) -> HookResult | None:
        if tool_name != "complete_workflow_node" or not result.success:
            return None

        run = self._active_run
        node = self._current_node()
        if not run or node is None:
            return None

        execution = run.current_execution()
        if execution is None:
            return None

        output = arguments.get("output", "")
        execution.output = output
        execution.attempt_count += 1
        execution.status = "reviewing"
        run.status = "paused_awaiting_review"
        self._sync_artifact(run)

        reviewer = self._resolve_reviewer(node.review)
        decision = await reviewer.review(node, execution, run, agent)

        execution.review_records.append(ReviewRecord(
            node_id=node.id, reviewer_type=reviewer.identity,
            approved=decision.approved, feedback=decision.feedback,
            reviewer_identity=reviewer.identity,
        ))

        if decision.approved:
            return await self._on_approved(agent, run, node, execution, decision)
        else:
            return self._on_rejected(run, node, execution, decision)

    # ═══════════════════════════════════════════════════════════════
    # Hook: before_tool_calling — precondition check
    # ═══════════════════════════════════════════════════════════════

    @before_tool_calling
    def validate_gate_call(
        self, agent: Any, tool_name: str, arguments: dict, ctx: Any
    ) -> HookResult | None:
        if tool_name != "complete_workflow_node":
            return None
        if not self._active_run or not self._workflow:
            return HookResult.skip(ToolResult(
                success=False,
                error="No active workflow. Use run_workflow to start one.",
            ))
        return None

    # ═══════════════════════════════════════════════════════════════
    # Hook: after_conversation — keep working
    # ═══════════════════════════════════════════════════════════════

    @after_conversation
    def notify_running_workflow(self, agent: Any, ctx: Any) -> HookResult | None:
        if ctx.error is not None:
            return None
        run = self._active_run
        if not run or run.status != "running":
            return None
        node = self._current_node()
        if node is None:
            return None
        execution = run.current_execution()
        if execution is None:
            return None

        self._sync_artifact(run)
        return HookResult.reinvoke(
            f"Workflow '{self._workflow.name}' is still active.\n\n"
            f"Current gate: {node.name} ({node.id})\n"
            f"Attempt: {execution.attempt_count}/{node.max_retries}\n\n"
            f"Your task: {node.prompt}\n\n"
            "When you have completed this gate's work, call "
            "complete_workflow_node with your output."
        )

    # ═══════════════════════════════════════════════════════════════
    # Tools — Discovery & Validation
    # ═══════════════════════════════════════════════════════════════

    @tool(
        name="read_workflow_manual",
        description=(
            "Read the full workflow manual: YAML format, review types, best "
            "practices, and differences from Skills.  You MUST call this before "
            "writing or loading any workflow YAML file."
        ),
        parameters_schema={"type": "object", "properties": {}, "additionalProperties": False},
    )
    def read_workflow_manual(self) -> ToolResult:
        import os
        manual_path = os.path.join(os.path.dirname(__file__), "WORKFLOW_MANUAL.md")
        try:
            with open(manual_path, "r", encoding="utf-8") as f:
                content = f.read()
        except FileNotFoundError:
            return ToolResult(success=False, error="Manual not found at " + manual_path)
        self._manual_read = True
        return ToolResult(success=True, output={"manual": content})

    @tool(
        name="load_workflow",
        description=(
            "Load and validate a workflow YAML file from ~/.hawi/workflows/. "
            "Use this to verify a workflow definition before running it."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Workflow name (YAML stem)."},
            },
            "required": ["name"],
        },
    )
    def load_workflow(self, name: str) -> ToolResult:
        if not self._manual_read:
            return ToolResult(
                success=False,
                error=(
                    "You have not read the workflow manual yet. "
                    "Please call read_workflow_manual first to learn the YAML "
                    "format, review types, and best practices before working "
                    "with workflow files."
                ),
            )
        try:
            from hawi_plugins.workflow_plugin.persistence import load_workflow as _load
            self._workflow = _load(name.strip())
            self._active_run = None
            self._sync_artifact_definition()
            return ToolResult(success=True, output=self._workflow.to_dict())
        except FileNotFoundError:
            return ToolResult(success=False, error=f"Workflow '{name}' not found.")
        except Exception as e:
            return ToolResult(success=False, error=f"Load failed: {e}")

    @tool(
        name="list_workflows",
        description="List all workflow definitions in ~/.hawi/workflows/.",
        parameters_schema={"type": "object", "properties": {}, "additionalProperties": False},
    )
    def list_workflows(self) -> ToolResult:
        try:
            from hawi_plugins.workflow_plugin.persistence import list_workflows as _list
            return ToolResult(success=True, output={"workflows": _list()})
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    # ═══════════════════════════════════════════════════════════════
    # Tools — Runtime
    # ═══════════════════════════════════════════════════════════════

    @tool(
        name="run_workflow",
        description="Start executing a workflow. Auto-loads from disk if needed.",
        parameters_schema={
            "type": "object",
            "properties": {
                "workflow_name": {
                    "type": "string",
                    "description": "Name of the workflow to run.",
                },
                "initial_input": {
                    "type": "string",
                    "description": "Optional initial input/context for the first gate.",
                },
            },
            "required": ["workflow_name"],
        },
    )
    def run_workflow(self, workflow_name: str, initial_input: str | None = None) -> ToolResult:
        # Auto-load if needed
        if self._workflow is None or self._workflow.name != workflow_name:
            r = self.load_workflow(workflow_name)
            if not r.success:
                return r

        wf = self._workflow
        if not wf.start_node_id or wf.start_node_id not in wf.nodes:
            return ToolResult(success=False, error=f"Invalid start_node_id: {wf.start_node_id}")

        errors = wf.validate()
        if errors:
            return ToolResult(success=False, error="\n".join(errors))

        run_id = str(uuid.uuid4())[:8]
        now = time.time()
        run = WorkflowRun(id=run_id, workflow_id=wf.id, current_node_id=wf.start_node_id)
        for nid in wf.nodes:
            run.node_executions[nid] = NodeExecution(node_id=nid, status="pending")

        start_exec = run.node_executions[wf.start_node_id]
        start_exec.status = "active"
        start_exec.started_at = now
        self._active_run = run
        self._sync_artifact(run)

        self.emit_plugin_event("plugin.event", {
            "event_name": "workflow.run.started",
            "run_id": run_id, "workflow_name": wf.name,
            "start_node": wf.start_node_id,
            "title": f"Workflow started: {wf.name}",
            "message": f"Entering gate: {wf.nodes[wf.start_node_id].name}",
        })

        start_node = wf.nodes[wf.start_node_id]
        output = {
            "run_id": run_id, "workflow": wf.name,
            "start_node": start_node.name,
            "message": (
                f"Workflow '{wf.name}' started. "
                f"First gate: {start_node.name}\n\n{start_node.prompt}"
            ),
        }
        if initial_input:
            output["initial_input"] = initial_input
        return ToolResult(success=True, output=output)

    @tool(
        name="select_next_workflow_node",
        description=(
            "Choose which immediate downstream workflow gate should run after "
            "the current gate is approved, and explain why. This does not "
            "advance the workflow or skip review; you still must call "
            "complete_workflow_node with the current gate output."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "next_node_id": {
                    "type": "string",
                    "description": "ID of an immediate downstream gate to run next.",
                },
                "reason": {
                    "type": "string",
                    "description": "Why this next gate is the right route.",
                },
            },
            "required": ["next_node_id", "reason"],
        },
    )
    def select_next_workflow_node(self, next_node_id: str, reason: str) -> ToolResult:
        """Record the agent's routing choice for the current gate."""
        run = self._active_run
        wf = self._workflow
        if not run or not wf:
            return ToolResult(
                success=False,
                error="No active workflow. Use run_workflow to start one.",
            )
        if run.status != "running":
            return ToolResult(
                success=False,
                error=f"Cannot select next node while workflow status is '{run.status}'.",
            )

        node = self._current_node()
        execution = run.current_execution()
        if node is None or execution is None:
            return ToolResult(success=False, error="No current workflow node.")

        next_node_id = next_node_id.strip()
        reason = reason.strip()
        if not next_node_id:
            return ToolResult(success=False, error="next_node_id is required.")
        if not reason:
            return ToolResult(success=False, error="reason is required.")

        downstream = wf.downstream_node_ids(node.id)
        if not downstream:
            return ToolResult(
                success=False,
                error=f"Current node '{node.id}' is terminal; there is no next node to select.",
            )
        if next_node_id not in downstream:
            return ToolResult(
                success=False,
                error=(
                    f"Node '{next_node_id}' is not an immediate downstream node "
                    f"of '{node.id}'. Available next nodes: {downstream}"
                ),
            )

        execution.selected_next_node_id = next_node_id
        execution.routing_reason = reason
        self._sync_artifact(run)

        selected = wf.nodes[next_node_id]
        return ToolResult(success=True, output={
            "current_node_id": node.id,
            "selected_next_node_id": next_node_id,
            "selected_next_node_name": selected.name,
            "routing_reason": reason,
            "message": (
                f"Next gate selected: {selected.name} ({next_node_id}). "
                "This route will be used after the current gate is approved."
            ),
        })

    @tool(
        name="complete_workflow_node",
        description=(
            "Submit your output for the current workflow gate. Your output will be "
            "reviewed. If approved, the workflow advances. If rejected, you'll "
            "receive feedback and must revise."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "output": {
                    "type": "string",
                    "description": "Your complete output for this gate.",
                },
            },
            "required": ["output"],
        },
    )
    def complete_workflow_node(self, output: str) -> ToolResult:
        """Submit node output. Review/transition happens in after_tool_calling."""
        run = self._active_run
        if not run:
            return ToolResult(success=False, error="No active workflow run.")
        node = self._current_node()
        if node is None:
            return ToolResult(success=False, error="No current node.")
        execution = run.current_execution()
        if not output.strip():
            return ToolResult(success=False, error="Output cannot be empty.")

        return ToolResult(success=True, output={
            "node_id": node.id, "node_name": node.name,
            "output_submitted": True, "output_length": len(output),
            "output_preview": output[:200] + ("..." if len(output) > 200 else ""),
            "selected_next_node_id": execution.selected_next_node_id if execution else None,
            "routing_reason": execution.routing_reason if execution else "",
            "review_pending": True,
        })

    @tool(
        name="get_workflow_status",
        description="Get the current workflow execution status with all gate states.",
        parameters_schema={"type": "object", "properties": {}, "additionalProperties": False},
    )
    def get_workflow_status(self) -> ToolResult:
        if self._workflow is None:
            return ToolResult(success=True, output={"status": "no_workflow"})
        if self._active_run is None:
            return ToolResult(success=True, output={
                "status": "idle", "workflow": self._workflow.to_dict(),
            })
        run = self._active_run
        self._sync_artifact(run)
        return ToolResult(success=True, output={
            "status": run.status, "run_id": run.id,
            "workflow": self._workflow.to_dict(),
            "current_node_id": run.current_node_id,
            "node_executions": {
                nid: ne.to_dict() for nid, ne in run.node_executions.items()
            },
            "global_context": run.global_context,
        })

    @tool(
        name="get_pending_reviews",
        description="Check if there are nodes awaiting review (does not expose review IDs).",
        parameters_schema={"type": "object", "properties": {}, "additionalProperties": False},
    )
    def get_pending_reviews(self) -> ToolResult:
        """Return review status WITHOUT exposing actionable review IDs."""
        pending: list[dict[str, Any]] = []
        if self._active_run:
            for nid, ne in self._active_run.node_executions.items():
                if ne.status == "reviewing":
                    pending.append({
                        "node_id": nid, "status": ne.status,
                        "attempt": ne.attempt_count,
                        "output_preview": (ne.output or "")[:200],
                        "selected_next_node_id": ne.selected_next_node_id,
                        "routing_reason": ne.routing_reason,
                    })
        return ToolResult(success=True, output={
            "pending_count": len(pending),
            "human_review_count": len(self._pending_human_reviews),
            "pending_reviews": pending,
        })

    # ═══════════════════════════════════════════════════════════════
    # Human review API (NOT agent tools)
    # ═══════════════════════════════════════════════════════════════

    def approve_workflow_node(
        self, review_id: str, feedback: str = "", modified_output: str | None = None,
    ) -> ToolResult:
        """[GUI/CLI] Approve a pending human review."""
        future = self._pending_human_reviews.get(review_id)
        if future is None:
            return ToolResult(success=False,
                error=f"No pending review '{review_id}'. "
                f"Pending: {list(self._pending_human_reviews.keys())}")
        if future.done():
            return ToolResult(success=False, error=f"Review '{review_id}' already resolved.")
        future.set_result(ReviewDecision(
            approved=True, feedback=feedback, modified_output=modified_output))
        return ToolResult(success=True, output={"review_id": review_id, "approved": True})

    def reject_workflow_node(self, review_id: str, feedback: str) -> ToolResult:
        """[GUI/CLI] Reject a pending human review."""
        future = self._pending_human_reviews.get(review_id)
        if future is None:
            return ToolResult(success=False,
                error=f"No pending review '{review_id}'. "
                f"Pending: {list(self._pending_human_reviews.keys())}")
        if future.done():
            return ToolResult(success=False, error=f"Review '{review_id}' already resolved.")
        if not feedback.strip():
            return ToolResult(success=False, error="Feedback is required when rejecting.")
        future.set_result(ReviewDecision(approved=False, feedback=feedback))
        return ToolResult(success=True, output={"review_id": review_id, "rejected": True})

    # ═══════════════════════════════════════════════════════════════
    # Internal helpers
    # ═══════════════════════════════════════════════════════════════

    def _current_node(self) -> WorkflowNode | None:
        if not self._workflow or not self._active_run:
            return None
        return self._workflow.nodes.get(self._active_run.current_node_id)

    def _resolve_reviewer(self, config):
        if config.type == "human":
            return HumanReviewer(self)
        elif config.type == "sub_agent":
            return SubAgentReviewer(
                review_prompt=config.sub_agent_prompt,
                model=config.sub_agent_model,
            )
        else:
            return LoggerReviewer()

    @staticmethod
    def _strip_gate_blocks(parts: list) -> list:
        return [p for p in parts if not (
            isinstance(p, dict) and p.get("type") == "text"
            and GATE_PROMPT_BEGIN in str(p.get("text", ""))
        )]

    async def _on_approved(self, agent, run, node, execution, decision) -> HookResult | None:
        output = (execution.output or "").strip()

        # ── Detect STATUS: FAILED — execution-level failure with retry ──
        # This allows logger-reviewed gates to signal failure and retry,
        # respecting the node's max_retries before giving up gracefully.
        if "STATUS: FAILED" in output:
            if execution.attempt_count >= node.max_retries:
                execution.status = "rejected"
                run.status = "failed"
                run.completed_at = time.time()
                self._sync_artifact(run)
                self.emit_plugin_event("plugin.event", {
                    "event_name": "workflow.failed", "run_id": run.id,
                    "node_id": node.id, "node_name": node.name,
                    "title": f"Workflow failed: {self._workflow.name}",
                    "message": (
                        f"Gate '{node.name}' failed after "
                        f"{execution.attempt_count} attempts."
                    ),
                })
                return HookResult.reinvoke(
                    f"Gate '{node.name}' FAILED after {execution.attempt_count} "
                    f"attempts (max {node.max_retries}). ❌\n\n"
                    f"Workflow '{self._workflow.name}' has been stopped.\n\n"
                    f"Last output:\n{output[:2000]}\n\n"
                    f"Please fix the issue manually, then run the workflow again."
                )

            # Retry — reset execution to active
            execution.status = "active"
            self._sync_artifact(run)
            self.emit_plugin_event("plugin.event", {
                "event_name": "workflow.node.failed", "run_id": run.id,
                "node_id": node.id, "node_name": node.name,
                "attempt": execution.attempt_count,
                "title": f"Gate failed: {node.name}",
                "message": (
                    f"Gate '{node.name}' failed "
                    f"(attempt {execution.attempt_count}/{node.max_retries})."
                ),
            })
            return HookResult.reinvoke(
                f"Gate '{node.name}' encountered a failure. 🔄\n\n"
                f"Attempt {execution.attempt_count}/{node.max_retries}.\n\n"
                f"Output indicated failure:\n{output[:2000]}\n\n"
                f"Please review the failure reason above, fix the issue, "
                f"and call complete_workflow_node again."
            )

        # ── Normal approval path (gate passed successfully) ──
        execution.status = "completed"
        execution.completed_at = time.time()
        if decision.modified_output:
            execution.output = decision.modified_output
        run.global_context[node.id] = execution.output
        run.status = "running"

        downstream = self._workflow.downstream_node_ids(node.id)
        route_source = "reviewer" if decision.next_node_id else ""
        next_node_id = decision.next_node_id
        routing_reason = ""
        if next_node_id is None and execution.selected_next_node_id:
            next_node_id = execution.selected_next_node_id
            routing_reason = execution.routing_reason
            route_source = "agent"
        if next_node_id is None:
            next_node_id = downstream[0] if downstream else None
            route_source = "default" if next_node_id else ""

        if route_source == "agent" and next_node_id not in downstream:
            run.status = "failed"
            self._sync_artifact(run)
            return HookResult.reinvoke(
                f"Workflow error: selected next node '{next_node_id}' is no longer "
                f"an immediate downstream node of '{node.id}'. Workflow failed."
            )

        if next_node_id is None:
            run.status = "completed"
            run.completed_at = time.time()
            run.current_node_id = ""
            self._sync_artifact(run)
            self.emit_plugin_event("plugin.event", {
                "event_name": "workflow.completed", "run_id": run.id,
                "workflow_name": self._workflow.name,
                "title": f"Workflow completed: {self._workflow.name}",
                "message": f"All gates passed.",
            })
            return None

        if next_node_id not in self._workflow.nodes:
            run.status = "failed"
            self._sync_artifact(run)
            return HookResult.reinvoke(
                f"Workflow error: next node '{next_node_id}' not found. Workflow failed.")

        run.current_node_id = next_node_id
        next_node = self._workflow.nodes[next_node_id]
        next_exec = run.node_executions[next_node_id]
        next_exec.status = "active"
        next_exec.started_at = time.time()
        self._sync_artifact(run)

        self.emit_plugin_event("plugin.event", {
            "event_name": "workflow.node.completed", "run_id": run.id,
            "node_id": node.id, "node_name": node.name, "next_node": next_node_id,
            "route_source": route_source,
            "routing_reason": routing_reason,
            "title": f"Gate passed: {node.name}",
            "message": f"Gate '{node.name}' approved. Entering: {next_node.name}",
        })
        route_note = ""
        if route_source == "agent" and routing_reason:
            route_note = (
                f"\n\nRoute selected by agent: {next_node.name} ({next_node_id})"
                f"\nReason: {routing_reason}"
            )
        return HookResult.reinvoke(
            f"Gate '{node.name}' PASSED. ✅\n\n"
            f"Entering next gate: {next_node.name}{route_note}\n\n"
            f"Task: {next_node.prompt}"
        )

    def _on_rejected(self, run, node, execution, decision) -> HookResult | None:
        run.status = "running"
        if execution.attempt_count >= node.max_retries:
            execution.status = "rejected"
            run.status = "failed"
            run.completed_at = time.time()
            self._sync_artifact(run)
            self.emit_plugin_event("plugin.event", {
                "event_name": "workflow.failed", "run_id": run.id,
                "node_id": node.id, "node_name": node.name,
                "title": f"Workflow failed: {self._workflow.name}",
                "message": f"Gate '{node.name}' rejected after {execution.attempt_count} attempts.",
            })
            return HookResult.reinvoke(
                f"Gate '{node.name}' REJECTED after {execution.attempt_count} "
                f"attempts (max {node.max_retries}). ❌\n\n"
                f"Workflow '{self._workflow.name}' has FAILED.\n\n"
                f"Final feedback: {decision.feedback}"
            )

        execution.status = "active"
        self._sync_artifact(run)
        self.emit_plugin_event("plugin.event", {
            "event_name": "workflow.node.rejected", "run_id": run.id,
            "node_id": node.id, "node_name": node.name,
            "attempt": execution.attempt_count,
            "title": f"Gate rejected: {node.name}",
            "message": f"Gate '{node.name}' not passed (attempt {execution.attempt_count}/{node.max_retries}).",
        })
        return HookResult.reinvoke(
            f"Gate '{node.name}' NOT PASSED "
            f"(attempt {execution.attempt_count}/{node.max_retries}). ❌\n\n"
            f"Reviewer Feedback:\n{decision.feedback}\n\n"
            f"Please REVISE your output based on this feedback, then call "
            f"complete_workflow_node again with the improved output."
        )

    def _construction_guidance(self) -> str:
        """Guidance injected when no workflow is loaded."""
        return (
            "\n<workflow-plugin>\n"
            "WorkflowPlugin is active. "
            "Before writing any workflow file, call read_workflow_manual "
            "to learn the YAML format, review types, and best practices. "
            "Then use filesystem tools to write the YAML to "
            "~/.hawi/workflows/{name}.yaml, and call "
            "load_workflow to validate it.\n\n"
            "Available tools:\n"
            "- read_workflow_manual: read the full workflow guide (call this first!)\n"
            "- list_workflows: discover saved workflows\n"
            "- load_workflow(name): load & validate a YAML workflow\n"
            "- run_workflow(name, initial_input?): start execution\n"
            "- select_next_workflow_node(next_node_id, reason): choose a downstream route\n"
            "- complete_workflow_node(output): submit gate output for review\n"
            "- get_workflow_status: inspect current execution state\n"
            "- get_pending_reviews: check if reviews are pending\n"
            "</workflow-plugin>\n"
        )

    def _build_gate_prompt(self, node: WorkflowNode, run: WorkflowRun) -> str:
        wf = self._workflow
        execution = run.current_execution()

        upstream_parts: list[str] = []
        for nid, exec_ in run.node_executions.items():
            if nid == node.id:
                continue
            if exec_.output and exec_.status == "completed":
                upstream_parts.append(f"### Gate '{nid}' output:\n{exec_.output[:2000]}")
        upstream_text = "\n\n".join(upstream_parts) if upstream_parts else "(none)"

        downstream = wf.downstream_node_ids(node.id) if wf else []
        if wf and downstream:
            downstream_lines: list[str] = []
            for nid in downstream:
                next_node = wf.nodes.get(nid)
                if next_node is None:
                    downstream_lines.append(f"- {nid}")
                    continue
                edge = next(
                    (
                        edge for edge in wf.edges
                        if edge.from_node_id == node.id and edge.to_node_id == nid
                    ),
                    None,
                )
                details = []
                if edge and edge.label:
                    details.append(f"label: {edge.label}")
                if edge and edge.condition:
                    details.append(f"condition: {edge.condition}")
                suffix = f" ({'; '.join(details)})" if details else ""
                downstream_lines.append(
                    f"- {next_node.name} ({nid}){suffix}: "
                    f"{next_node.description or next_node.prompt[:120]}"
                )
            downstream_text = "\n".join(downstream_lines)
        else:
            downstream_text = "(terminal gate)"
        routing_instruction = ""
        if len(downstream) > 1:
            routing_instruction = (
                "\nMultiple next gates are available. Before calling "
                "`complete_workflow_node`, call `select_next_workflow_node` "
                "with your chosen downstream `next_node_id` and a concise "
                "reason based on this gate's result.\n"
            )
        elif len(downstream) == 1:
            routing_instruction = (
                "\nThere is one downstream gate. You may call "
                "`select_next_workflow_node` if you want to record why that "
                "route is appropriate; otherwise the workflow will use it by default.\n"
            )

        completed = [
            f"- {nid}: {ne.status}"
            for nid, ne in run.node_executions.items()
            if ne.status == "completed"
        ]

        return (
            f"\n{GATE_PROMPT_BEGIN}\n"
            f"## Workflow: {wf.name if wf else 'unknown'}\n"
            f"## Current Gate: {node.name} ({node.id})\n"
            f"### Description: {node.description or node.name}\n\n"
            f"### Completed Gates:\n"
            f"{chr(10).join(completed) if completed else '(none yet)'}\n\n"
            f"### Previous Gate Outputs:\n{upstream_text}\n\n"
            f"### Your Task at This Gate:\n{node.prompt}\n\n"
            f"### To Pass This Gate:\n"
            f"You MUST call `complete_workflow_node` with your final output.\n"
            f"Your output will be reviewed"
            f"{' by a human' if node.review.type == 'human' else ''}"
            f"{' by another agent' if node.review.type == 'sub_agent' else ''}"
            f"{' (auto-approved + logged)' if node.review.type == 'logger' else ''}"
            f".\n"
            f"If rejected, you will receive feedback and must revise.\n"
            f"You have {node.max_retries} attempt(s) "
            f"(current: {execution.attempt_count if execution else 0}).\n\n"
            f"### Available Next Gates After Passing:\n{downstream_text}\n"
            f"{routing_instruction}\n"
            f"⚠️ You CANNOT skip this gate. You CANNOT move to the next gate\n"
            f"without calling complete_workflow_node and passing review.\n"
            f"{GATE_PROMPT_END}\n"
        )

    def _sync_artifact(self, run: WorkflowRun | None = None) -> None:
        if not self._workflow:
            return
        status = "idle"
        content_lines = [f"# Workflow: {self._workflow.name}\n"]
        if run:
            status = run.status
            content_lines.append(f"**Status**: {run.status}\n")
            for nid, node in self._workflow.nodes.items():
                ne = run.node_executions.get(nid)
                mark = {
                    "completed": "✅", "active": "🔄", "reviewing": "⏳",
                    "rejected": "❌",
                }.get(ne.status if ne else "", "⬜")
                content_lines.append(f"- {mark} {node.name} ({nid})")
        else:
            content_lines.append("**Status**: idle (no active run)\n")
            for nid, node in self._workflow.nodes.items():
                content_lines.append(f"- ⬜ {node.name} ({nid})")

        self.upsert_artifact(
            "current-workflow", artifact_type="workflow",
            title=f"Workflow: {self._workflow.name}",
            content="\n".join(content_lines),
            language="markdown", mime_type="text/markdown", status=status,
            metadata={
                "workflow": self._workflow.to_dict(),
                "run": run.to_dict() if run else None,
            },
        )

    def _sync_artifact_definition(self) -> None:
        if not self._workflow:
            return
        lines = [
            f"# Workflow: {self._workflow.name}",
            f"**ID**: {self._workflow.id}",
            f"**Description**: {self._workflow.description}",
            f"**Start**: {self._workflow.start_node_id}",
            "", "## Nodes",
        ]
        for node in self._workflow.nodes.values():
            lines.append(
                f"- **{node.name}** (`{node.id}`): {node.description} "
                f"[review: {node.review.type}, retries: {node.max_retries}]"
            )
        lines.extend(["", "## Edges"])
        for edge in self._workflow.edges:
            label = f" ({edge.label})" if edge.label else ""
            lines.append(f"- {edge.from_node_id} → {edge.to_node_id}{label}")

        self.upsert_artifact(
            "current-workflow-def", artifact_type="workflow_definition",
            title=f"Workflow Definition: {self._workflow.name}",
            content="\n".join(lines), language="markdown",
            mime_type="text/markdown", status="defined",
            metadata={"workflow": self._workflow.to_dict()},
        )
