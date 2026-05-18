from __future__ import annotations

import json
import re
import time
import uuid
from copy import deepcopy
from typing import Any

from hawi.plugin import (
    HawiPlugin,
    HookResult,
    after_conversation,
    after_tool_calling,
    before_conversation,
    before_session,
    before_tool_calling,
    tool,
)
from hawi.review import RuntimeReviewDecision
from hawi.tool import ToolResult

from .models import (
    OPEN_STEP_STATUSES,
    PARKED_STEP_STATUSES,
    TERMINAL_STEP_STATUSES,
    TaskflowDefinition,
    TaskflowEdge,
    TaskflowFoldRecord,
    TaskflowReviewDecision,
    TaskflowReviewPolicy,
    TaskflowReviewRecord,
    TaskflowRun,
    TaskflowStep,
)


TASKFLOW_PROMPT_BEGIN = "<hawi-taskflow>"
TASKFLOW_PROMPT_END = "</hawi-taskflow>"
TASKFLOW_REMINDER_BEGIN = "<hawi-taskflow-reminder>"
TASKFLOW_REMINDER_END = "</hawi-taskflow-reminder>"
TASKFLOW_SUBMIT_CACHE_POINT_SOURCE = "taskflow.submit_taskflow_step"


def _empty_schema() -> dict[str, Any]:
    return {"type": "object", "properties": {}, "additionalProperties": False}


class TaskflowPlugin(HawiPlugin):
    """Unified task planning and gated workflow plugin."""

    name = "hawi/taskflow"
    display_name = "Taskflow"
    description = "Unified task plans and gated workflows with one step/run interface."
    dependencies = ()

    def __init__(self, fold_completed_steps: bool = False) -> None:
        self._taskflow: TaskflowDefinition | None = None
        self._active_run: TaskflowRun | None = None
        self._next_step_number = 1
        self._fold_completed_steps = bool(fold_completed_steps)
        self._fold_records: list[TaskflowFoldRecord] = []
        self._next_fold_number = 1
        self._active_submit_tool_call_id: str | None = None
        self._pending_human_reviews: dict[str, Any] = {}

    @classmethod
    def gui_config_schema(cls) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "fold_completed_steps": {
                    "type": "boolean",
                    "title": "Fold Completed Step Context",
                    "default": False,
                    "description": (
                        "When enabled, submit_taskflow_step may fold completed "
                        "step details out of active context when context_policy='fold'."
                    ),
                }
            },
            "additionalProperties": False,
        }

    @classmethod
    def gui_default_config(cls) -> dict[str, bool]:
        return {"fold_completed_steps": False}

    def clone(self) -> "TaskflowPlugin":
        clone = TaskflowPlugin(fold_completed_steps=self._fold_completed_steps)
        if self._taskflow is not None:
            clone._taskflow = TaskflowDefinition.from_dict(self._taskflow.to_dict())
        if self._active_run is not None:
            clone._active_run = TaskflowRun.from_dict(self._active_run.to_dict())
        clone._next_step_number = self._next_step_number
        clone._fold_records = [
            TaskflowFoldRecord.from_dict(record.to_dict())
            for record in self._fold_records
        ]
        clone._next_fold_number = self._next_fold_number
        clone._active_submit_tool_call_id = self._active_submit_tool_call_id
        return clone

    def save_state(self) -> dict[str, Any] | None:
        if self._taskflow is None and self._active_run is None:
            return None
        return {
            "taskflow": self._taskflow.to_dict() if self._taskflow else None,
            "active_run": self._active_run.to_dict() if self._active_run else None,
            "next_step_number": self._next_step_number,
            "fold_completed_steps": self._fold_completed_steps,
            "fold_records": [record.to_dict() for record in self._fold_records],
            "next_fold_number": self._next_fold_number,
            "active_submit_tool_call_id": self._active_submit_tool_call_id,
        }

    def load_state(self, data: dict[str, Any]) -> None:
        taskflow_data = data.get("taskflow")
        self._taskflow = (
            TaskflowDefinition.from_dict(taskflow_data)
            if isinstance(taskflow_data, dict)
            else None
        )
        run_data = data.get("active_run")
        self._active_run = (
            TaskflowRun.from_dict(run_data) if isinstance(run_data, dict) else None
        )
        self._next_step_number = int(data.get("next_step_number", 1) or 1)
        self._fold_completed_steps = bool(data.get("fold_completed_steps", False))
        self._fold_records = [
            TaskflowFoldRecord.from_dict(record)
            for record in data.get("fold_records", [])
            if isinstance(record, dict)
        ]
        self._next_fold_number = int(data.get("next_fold_number", 1) or 1)
        self._active_submit_tool_call_id = data.get("active_submit_tool_call_id")
        self._pending_human_reviews = {}

    @before_session(system_prompt_variability="hardcoded")
    def inject_taskflow_instructions(self, agent: Any, ctx: Any) -> None:
        prompt = self._build_taskflow_prompt()
        system_prompt = list(agent.context.system_prompt or [])
        system_prompt = [
            part
            for part in system_prompt
            if not (
                isinstance(part, dict)
                and part.get("type") == "text"
                and TASKFLOW_PROMPT_BEGIN in str(part.get("text", ""))
            )
        ]
        system_prompt.append({"type": "text", "text": prompt})
        agent.context.system_prompt = system_prompt

    @before_conversation
    def inject_active_step_context(self, agent: Any, ctx: Any) -> None:
        taskflow = self._taskflow
        run = self._active_run
        if taskflow is None or run is None or run.status != "running":
            return None
        if taskflow.execution_policy == "freeform":
            return None
        step = self._current_step()
        if step is None:
            return None
        agent.context.inject(
            {
                "role": "user",
                "content": [{"type": "text", "text": self._format_step_context(step)}],
                "name": None,
                "metadata": {
                    "source": "taskflow_plugin",
                    "injection": "taskflow_step_context",
                },
            },
            position=_find_last_user_insert_index(agent.context.messages),
        )
        return None

    @before_tool_calling
    def remember_submit_tool_call(
        self,
        agent: Any,
        tool_name: str,
        arguments: dict[str, Any],
        ctx: Any,
    ) -> None:
        if tool_name == "submit_taskflow_step":
            self._active_submit_tool_call_id = getattr(ctx, "tool_call_id", None)

    @after_tool_calling
    async def review_submitted_step(
        self,
        agent: Any,
        tool_name: str,
        arguments: dict[str, Any],
        result: ToolResult,
        ctx: Any,
    ) -> HookResult | None:
        try:
            if tool_name != "submit_taskflow_step" or not result.success:
                return None
            output = result.output
            if not isinstance(output, dict) or not output.get("review_pending"):
                return None
            step_id = str(output.get("step_id") or "")
            step = self._step(step_id)
            if step is None:
                return None
            decision = await self._review_step(agent, step, ctx)
            if decision is None:
                return None
            step.review_records.append(
                TaskflowReviewRecord(
                    step_id=step.id,
                    reviewer_type=step.review.type,
                    approved=decision.approved,
                    feedback=decision.feedback,
                    reviewer_identity=step.review.type,
                )
            )
            if decision.approved:
                if decision.modified_output is not None:
                    step.output = decision.modified_output
                hook_result, folded_context = self._approve_reviewed_step(
                    step,
                    decision,
                    ctx,
                )
                should_cache = self._should_mark_submit_cache_point(folded_context)
                if should_cache:
                    self._remove_previous_submit_cache_points(ctx)
                    result.cache_point = True
                    result.cache_point_source = TASKFLOW_SUBMIT_CACHE_POINT_SOURCE
                result.output = {
                    "review_pending": False,
                    "approved": True,
                    "step_id": step.id,
                    "review_type": step.review.type,
                    "folded_context": folded_context,
                    "next_message": hook_result.message if hook_result else None,
                    "state": self._state_dict(),
                }
                return hook_result
            hook_result = self._reject_reviewed_step(step, decision)
            result.output = {
                "review_pending": False,
                "approved": False,
                "rejected": True,
                "step_id": step.id,
                "review_type": step.review.type,
                "feedback": decision.feedback,
                "next_message": hook_result.message,
                "state": self._state_dict(),
            }
            return hook_result
        finally:
            if tool_name == "submit_taskflow_step":
                self._active_submit_tool_call_id = None

    @after_conversation
    def continue_active_taskflow(self, agent: Any, ctx: Any) -> HookResult | None:
        if ctx.error is not None:
            return None
        taskflow = self._taskflow
        run = self._active_run
        if taskflow is None or run is None or run.status != "running":
            return None
        self._sync_artifact()

        if taskflow.execution_policy == "freeform":
            open_steps = self._open_steps()
            if not open_steps:
                if self._has_parked_steps():
                    run.status = "paused"
                    run.pause_reason = "All unfinished taskflow steps are parked."
                    self._sync_artifact()
                elif self._has_only_terminal_or_parked_steps():
                    run.status = "completed"
                    run.completed_at = time.time()
                    self._sync_artifact()
                return None
            return HookResult.reinvoke(self._format_freeform_reminder(open_steps))

        step = self._current_step()
        if step is None:
            run.status = "completed"
            run.completed_at = time.time()
            self._sync_artifact()
            return None
        return HookResult.reinvoke(self._format_step_reminder(step))

    @tool(
        name="read_taskflow_manual",
        description="Read the Taskflow YAML and runtime usage guide.",
        parameters_schema=_empty_schema(),
    )
    def read_taskflow_manual(self) -> ToolResult:
        return ToolResult(success=True, output={"manual": _TASKFLOW_MANUAL})

    @tool(
        name="create_taskflow",
        description=(
            "Create or replace the current taskflow. Use mode='plan' for a mutable "
            "task plan and mode='workflow' for a gated DAG."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "mode": {"type": "string", "enum": ["plan", "workflow"]},
                "execution_policy": {
                    "type": "string",
                    "enum": ["freeform", "sequential", "gated_graph"],
                },
                "mutable": {"type": "boolean"},
                "description": {"type": "string"},
                "start_step_id": {"type": "string"},
                "steps": {"type": "array", "items": {"type": "object"}},
                "edges": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["title"],
        },
    )
    def create_taskflow(
        self,
        title: str,
        mode: str = "plan",
        execution_policy: str | None = None,
        mutable: bool | None = None,
        description: str = "",
        start_step_id: str | None = None,
        steps: list[dict[str, Any]] | None = None,
        edges: list[dict[str, Any]] | None = None,
    ) -> ToolResult:
        mode = self._normalize_mode(mode)
        if execution_policy is None:
            execution_policy = "freeform" if mode == "plan" else "gated_graph"
        execution_policy = self._normalize_execution_policy(execution_policy, mode)
        data = {
            "id": self._slug(title),
            "title": title.strip(),
            "mode": mode,
            "execution_policy": execution_policy,
            "mutable": bool(mode == "plan" if mutable is None else mutable),
            "description": description,
            "start_step_id": start_step_id,
            "steps": steps or [],
            "edges": edges or [],
        }
        taskflow = TaskflowDefinition.from_dict(data)
        errors = taskflow.validate()
        if steps and errors:
            return ToolResult(success=False, error="\n".join(errors))
        self._taskflow = taskflow
        self._active_run = None
        self._refresh_next_step_number()
        self._sync_artifact(status="defined")
        return ToolResult(success=True, output=self._state_dict())

    @tool(
        name="list_taskflows",
        description="List saved Taskflow YAML definitions in ~/.hawi/taskflows/.",
        parameters_schema=_empty_schema(),
    )
    def list_taskflows(self) -> ToolResult:
        try:
            from .persistence import list_taskflows

            return ToolResult(success=True, output={"taskflows": list_taskflows()})
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

    @tool(
        name="load_taskflow",
        description="Load a Taskflow YAML definition from ~/.hawi/taskflows/.",
        parameters_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    )
    def load_taskflow(self, name: str) -> ToolResult:
        try:
            from .persistence import load_taskflow

            self._taskflow = load_taskflow(name.strip())
            self._active_run = None
            self._refresh_next_step_number()
            self._sync_artifact(status="defined")
            return ToolResult(success=True, output=self._state_dict())
        except FileNotFoundError:
            return ToolResult(success=False, error=f"Taskflow {name!r} not found.")
        except Exception as exc:
            return ToolResult(success=False, error=f"Load failed: {exc}")

    @tool(
        name="save_taskflow",
        description="Save the current taskflow definition to ~/.hawi/taskflows/.",
        parameters_schema=_empty_schema(),
    )
    def save_taskflow(self) -> ToolResult:
        if self._taskflow is None:
            return ToolResult(success=False, error="No current taskflow to save.")
        try:
            from .persistence import save_taskflow

            path = save_taskflow(self._taskflow)
            return ToolResult(success=True, output={"path": path})
        except Exception as exc:
            return ToolResult(success=False, error=f"Save failed: {exc}")

    @tool(
        name="start_taskflow",
        description="Start executing the current taskflow, or load one by name first.",
        parameters_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "initial_input": {"type": "string"},
            },
        },
    )
    def start_taskflow(
        self,
        name: str | None = None,
        initial_input: str | None = None,
    ) -> ToolResult:
        if isinstance(name, str) and name.strip():
            loaded = self.load_taskflow(name)
            if not loaded.success:
                return loaded
        taskflow = self._taskflow
        if taskflow is None:
            return ToolResult(success=False, error="No taskflow. Call create_taskflow or load_taskflow first.")
        errors = taskflow.validate()
        if errors:
            return ToolResult(success=False, error="\n".join(errors))
        run = TaskflowRun(
            id=str(uuid.uuid4())[:8],
            taskflow_id=taskflow.id,
            status="running",
        )
        if initial_input:
            run.global_context["initial_input"] = initial_input
        if taskflow.execution_policy == "freeform":
            run.current_step_id = None
        else:
            self._reset_workflow_steps(taskflow)
            start_step_id = taskflow.start_step_id or next(iter(taskflow.steps))
            run.current_step_id = start_step_id
            step = taskflow.steps[start_step_id]
            self._enter_step(step)
        self._active_run = run
        self._sync_artifact()
        self._emit_taskflow_event(
            "taskflow.run.started",
            {
                "run_id": run.id,
                "taskflow_id": taskflow.id,
                "title": f"Taskflow started: {taskflow.title}",
                "message": self._start_message(),
            },
        )
        return ToolResult(success=True, output=self._state_dict())

    @tool(
        name="add_taskflow_steps",
        description=(
            "Add one or more steps to a mutable taskflow. Nested children create "
            "decomposition edges."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "parent_id": {"type": "string"},
                "steps": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["steps"],
        },
    )
    def add_taskflow_steps(
        self,
        steps: list[dict[str, Any]],
        parent_id: str | None = None,
    ) -> ToolResult:
        try:
            taskflow = self._ensure_mutable_taskflow()
        except ValueError as exc:
            return ToolResult(success=False, error=str(exc))
        if parent_id and parent_id not in taskflow.steps:
            return ToolResult(success=False, error=f"Unknown parent step id: {parent_id}")
        if not isinstance(steps, list) or not steps:
            return ToolResult(success=False, error="steps must contain at least one step.")
        created: list[TaskflowStep] = []
        for raw_step in steps:
            if not isinstance(raw_step, dict):
                return ToolResult(success=False, error="Each step must be an object.")
            error = self._add_step_tree(taskflow, raw_step, parent_id=parent_id, created=created)
            if error:
                return ToolResult(success=False, error=error)
        if self._active_run is None:
            self._active_run = TaskflowRun(
                id=str(uuid.uuid4())[:8],
                taskflow_id=taskflow.id,
                status="running",
            )
        else:
            self._active_run.status = "running"
            self._active_run.completed_at = None
            self._active_run.pause_reason = ""
        self._sync_artifact()
        for step in created:
            self._emit_step_event("added", step)
        return ToolResult(
            success=True,
            output={
                "created": [step.to_dict() for step in created],
                "state": self._state_dict(),
            },
        )

    @tool(
        name="update_taskflow_steps",
        description=(
            "Update one or more step statuses. Use blocked/deferred for parked "
            "work, canceled/obsolete/skipped for terminal non-completion, and "
            "pending/active to reopen."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "step_id": {"type": "string"},
                "step_ids": {"type": "array", "items": {"type": "string"}},
                "status": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["status"],
        },
    )
    def update_taskflow_steps(
        self,
        status: str,
        step_id: str | None = None,
        step_ids: list[str] | None = None,
        reason: str | None = None,
    ) -> ToolResult:
        steps, error = self._resolve_step_selection(step_id, step_ids)
        if error:
            return ToolResult(success=False, error=error)
        normalized_status = status.strip().lower() if isinstance(status, str) else ""
        allowed = {
            "pending",
            "active",
            "blocked",
            "deferred",
            "canceled",
            "obsolete",
            "failed",
            "skipped",
        }
        if normalized_status not in allowed:
            return ToolResult(success=False, error=f"status must be one of {sorted(allowed)}.")
        reason_text = reason.strip() if isinstance(reason, str) else ""
        for step in steps:
            step.status = normalized_status  # type: ignore[assignment]
            step.status_reason = reason_text or None
            if normalized_status in {"pending", "active"}:
                step.completed_at = None
                if (
                    self._taskflow is not None
                    and self._taskflow.execution_policy == "freeform"
                    and self._active_run is not None
                ):
                    self._active_run.status = "running"
                    self._active_run.completed_at = None
                    self._active_run.pause_reason = ""
            if normalized_status == "active":
                step.started_at = step.started_at or time.time()
            self._emit_step_event("status_updated", step)
        self._sync_artifact()
        return ToolResult(success=True, output=self._state_dict())

    @tool(
        name="submit_taskflow_step",
        description=(
            "Submit one step or a batch of steps as complete. Steps with human or "
            "sub_agent review enter review before they can complete."
        ),
        context="ctx",
        parameters_schema={
            "type": "object",
            "properties": {
                "step_id": {"type": "string"},
                "step_ids": {"type": "array", "items": {"type": "string"}},
                "output": {"type": "string"},
                "complete_descendants": {"type": "boolean", "default": False},
                "context_policy": {"type": "string", "enum": ["keep", "fold", "auto"]},
                "summary": {"type": "string"},
                "handoff_notes": {"type": "string"},
            },
        },
    )
    def submit_taskflow_step(
        self,
        step_id: str | None = None,
        step_ids: list[str] | None = None,
        output: str | None = None,
        complete_descendants: bool = False,
        context_policy: str | None = None,
        summary: str | None = None,
        handoff_notes: str | None = None,
        ctx: Any = None,
    ) -> ToolResult:
        taskflow = self._taskflow
        if taskflow is None:
            return ToolResult(success=False, error="No current taskflow.")
        if step_id is None and step_ids is None and self._active_run and self._active_run.current_step_id:
            step_id = self._active_run.current_step_id
        steps, error = self._resolve_step_selection(step_id, step_ids)
        if error:
            return ToolResult(success=False, error=error)
        steps, error = self._completion_closure(steps, complete_descendants)
        if error:
            return ToolResult(success=False, error=error)
        review_steps = [
            step
            for step in steps
            if step.review.type in {"human", "sub_agent"}
            and step.status != "completed"
        ]
        if review_steps:
            if len(steps) != 1:
                return ToolResult(
                    success=False,
                    error="Reviewed steps must be submitted one at a time.",
                )
            step = review_steps[0]
            step.output = output or ""
            step.attempt_count += 1
            step.status = "reviewing"
            if self._active_run is not None:
                self._active_run.status = "paused_awaiting_review"
            if step.review.type == "human" and not self._runtime_review_available(ctx):
                self._request_human_review(step)
            self._sync_artifact()
            return ToolResult(
                success=True,
                output={
                    "review_pending": True,
                    "step_id": step.id,
                    "step_title": step.title,
                    "review_type": step.review.type,
                    "attempt": step.attempt_count,
                    "max_retries": step.review.max_retries,
                    "output_preview": (step.output or "")[:400],
                },
            )

        completed = self._complete_steps(
            steps,
            output=output,
            reviewer_type="none",
        )
        folded_context = self._maybe_fold_for_submission(
            ctx,
            completed,
            context_policy=context_policy,
            summary=summary,
            handoff_notes=handoff_notes,
        )
        should_cache = self._should_mark_submit_cache_point(folded_context)
        if should_cache:
            self._remove_previous_submit_cache_points(ctx)
        next_message = None
        if (
            len(completed) == 1
            and self._active_run is not None
            and self._active_run.current_step_id == completed[0].id
        ):
            next_message = self._advance_after_step(completed[0], next_step_id=None)
        self._sync_artifact()
        return ToolResult(
            success=True,
            output={
                "completed": [step.to_dict() for step in completed],
                "folded_context": folded_context,
                "next_message": next_message,
                "state": self._state_dict(),
            },
            cache_point=should_cache,
            cache_point_source=TASKFLOW_SUBMIT_CACHE_POINT_SOURCE,
        )

    @tool(
        name="select_next_taskflow_step",
        description=(
            "Select an immediate downstream transition step. Use this before "
            "submitting a step, or after submission when a completed step has "
            "multiple conditional exits such as loop-back and exit edges."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "next_step_id": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["next_step_id", "reason"],
        },
    )
    def select_next_taskflow_step(self, next_step_id: str, reason: str) -> ToolResult:
        taskflow = self._taskflow
        step = self._current_step()
        if taskflow is None or step is None:
            return ToolResult(success=False, error="No active step to route from.")
        downstream = self._transition_downstream_ids(step.id)
        if next_step_id not in downstream:
            return ToolResult(
                success=False,
                error=(
                    f"{next_step_id!r} is not an immediate downstream transition "
                    f"from {step.id!r}. Available: {self._format_transition_options(step)}"
                ),
            )
        reason_text = reason.strip() if isinstance(reason, str) else ""
        if not reason_text:
            return ToolResult(success=False, error="reason is required.")
        step.selected_next_step_id = next_step_id
        step.routing_reason = reason_text
        next_message = None
        if (
            step.status == "completed"
            and self._active_run is not None
            and self._active_run.current_step_id == step.id
        ):
            next_message = self._advance_after_step(step, next_step_id=next_step_id)
        self._sync_artifact()
        return ToolResult(
            success=True,
            output={
                "current_step_id": step.id,
                "selected_next_step_id": next_step_id,
                "routing_reason": reason_text,
                "next_message": next_message,
                "state": self._state_dict(),
            },
        )

    @tool(
        name="get_taskflow_status",
        description="Return the current taskflow definition, run, steps, and reviews.",
        parameters_schema=_empty_schema(),
    )
    def get_taskflow_status(self) -> ToolResult:
        self._sync_artifact()
        return ToolResult(success=True, output=self._state_dict())

    @tool(
        name="get_pending_taskflow_reviews",
        description="Return pending taskflow reviews without exposing action ids to the model.",
        parameters_schema=_empty_schema(),
    )
    def get_pending_taskflow_reviews(self) -> ToolResult:
        pending = []
        taskflow = self._taskflow
        if taskflow is not None:
            for step in taskflow.steps.values():
                if step.status == "reviewing":
                    pending.append(
                        {
                            "step_id": step.id,
                            "title": step.title,
                            "attempt": step.attempt_count,
                            "review_type": step.review.type,
                            "output_preview": (step.output or "")[:400],
                        }
                    )
        return ToolResult(
            success=True,
            output={
                "pending_count": len(pending),
                "human_review_count": len(self._pending_human_reviews),
                "pending_reviews": pending,
            },
        )

    @tool(
        name="control_taskflow",
        description="Pause, resume, or clear the active taskflow run.",
        parameters_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["pause", "resume", "clear"]},
                "reason": {"type": "string"},
            },
            "required": ["action"],
        },
    )
    def control_taskflow(self, action: str, reason: str | None = None) -> ToolResult:
        normalized = action.strip().lower() if isinstance(action, str) else ""
        reason_text = reason.strip() if isinstance(reason, str) else ""
        if normalized == "pause":
            if not reason_text:
                return ToolResult(success=False, error="reason is required when pausing.")
            if self._active_run is None:
                return ToolResult(success=False, error="No active taskflow run.")
            self._active_run.status = "paused"
            self._active_run.pause_reason = reason_text
            self._sync_artifact()
            self._emit_taskflow_event(
                "taskflow.run.paused",
                {"reason": reason_text, "title": "Taskflow paused", "message": reason_text},
            )
            return ToolResult(success=True, output=self._state_dict())
        if normalized == "resume":
            if self._active_run is None:
                return ToolResult(success=False, error="No active taskflow run.")
            self._active_run.status = "running"
            self._active_run.pause_reason = ""
            self._sync_artifact()
            self._emit_taskflow_event(
                "taskflow.run.resumed",
                {"title": "Taskflow resumed", "message": "Taskflow execution resumed."},
            )
            return ToolResult(success=True, output=self._state_dict())
        if normalized == "clear":
            self._taskflow = None
            self._active_run = None
            self._next_step_number = 1
            self._fold_records.clear()
            self._next_fold_number = 1
            self.remove_artifact("current-taskflow")
            self._emit_taskflow_event(
                "taskflow.run.cleared",
                {"reason": reason_text, "title": "Taskflow cleared", "message": reason_text},
            )
            return ToolResult(success=True, output={"status": "cleared"})
        return ToolResult(success=False, error="action must be pause, resume, or clear.")

    @tool(
        name="recall_taskflow_context",
        description="Read or search folded context from completed taskflow steps.",
        parameters_schema={
            "type": "object",
            "properties": {
                "step_id": {"type": "string"},
                "fold_id": {"type": "string"},
                "query": {"type": "string"},
                "max_chars": {"type": "integer", "default": 20000},
            },
        },
    )
    def recall_taskflow_context(
        self,
        step_id: str = "",
        fold_id: str | None = None,
        query: str | None = None,
        max_chars: int = 20000,
    ) -> ToolResult:
        records = self._fold_records_for_lookup(step_id=step_id, fold_id=fold_id)
        if not records:
            return ToolResult(success=False, error="No matching folded context found.")
        query_text = query.strip() if isinstance(query, str) else ""
        if query_text:
            return ToolResult(
                success=True,
                output=self._search_fold_records(records, query_text, max_chars=max_chars),
            )
        record = records[0]
        transcript = self._truncate(
            self._format_folded_messages(record.messages),
            max_chars,
            marker="\n[Transcript truncated by max_chars.]",
        )[0]
        return ToolResult(
            success=True,
            output={
                "mode": "read",
                **record.reference_dict(),
                "message_count": len(record.messages),
                "transcript": transcript,
            },
        )

    def approve_taskflow_review(
        self,
        review_id: str,
        feedback: str = "",
        modified_output: str | None = None,
    ) -> ToolResult:
        review = self._pending_human_reviews.get(review_id)
        if review is None:
            return ToolResult(
                success=False,
                error=(
                    f"No pending review {review_id!r}. "
                    f"Pending: {list(self._pending_human_reviews)}"
                ),
            )
        if hasattr(review, "set_result"):
            if review.done():
                return ToolResult(success=False, error=f"Review {review_id!r} already resolved.")
            review.set_result(
                RuntimeReviewDecision(
                    approved=True,
                    feedback=feedback,
                    modified_output=modified_output,
                )
            )
            return ToolResult(
                success=True,
                output={"review_id": review_id, "approved": True},
            )
        self._pending_human_reviews.pop(review_id, None)
        step = self._step(str(review.get("step_id") or ""))
        if step is None:
            return ToolResult(success=False, error=f"Review {review_id!r} step is missing.")
        decision = TaskflowReviewDecision(
            approved=True,
            feedback=feedback,
            modified_output=modified_output,
        )
        step.review_records.append(
            TaskflowReviewRecord(
                step_id=step.id,
                reviewer_type="human",
                approved=True,
                feedback=decision.feedback,
                reviewer_identity="human",
            )
        )
        if decision.modified_output is not None:
            step.output = decision.modified_output
        next_message, folded_context = self._complete_reviewed_step(
            step,
            decision,
            ctx=None,
        )
        return ToolResult(
            success=True,
            output={
                "review_id": review_id,
                "approved": True,
                "folded_context": folded_context,
                "next_message": next_message,
                "state": self._state_dict(),
            },
        )

    def reject_taskflow_review(self, review_id: str, feedback: str) -> ToolResult:
        if not feedback.strip():
            return ToolResult(success=False, error="feedback is required when rejecting.")
        review = self._pending_human_reviews.get(review_id)
        if review is None:
            return ToolResult(
                success=False,
                error=(
                    f"No pending review {review_id!r}. "
                    f"Pending: {list(self._pending_human_reviews)}"
                ),
            )
        if hasattr(review, "set_result"):
            if review.done():
                return ToolResult(success=False, error=f"Review {review_id!r} already resolved.")
            review.set_result(RuntimeReviewDecision(approved=False, feedback=feedback))
            return ToolResult(
                success=True,
                output={"review_id": review_id, "rejected": True},
            )
        self._pending_human_reviews.pop(review_id, None)
        step = self._step(str(review.get("step_id") or ""))
        if step is None:
            return ToolResult(success=False, error=f"Review {review_id!r} step is missing.")
        decision = TaskflowReviewDecision(approved=False, feedback=feedback)
        step.review_records.append(
            TaskflowReviewRecord(
                step_id=step.id,
                reviewer_type="human",
                approved=False,
                feedback=decision.feedback,
                reviewer_identity="human",
            )
        )
        hook_result = self._reject_reviewed_step(step, decision)
        return ToolResult(
            success=True,
            output={
                "review_id": review_id,
                "rejected": True,
                "next_message": hook_result.message,
                "state": self._state_dict(),
            },
        )

    def _ensure_mutable_taskflow(self) -> TaskflowDefinition:
        if self._taskflow is None:
            self._taskflow = TaskflowDefinition(
                id="taskflow-plan",
                title="Taskflow Plan",
                mode="plan",
                execution_policy="freeform",
                mutable=True,
            )
        if not self._taskflow.mutable:
            raise ValueError("Current taskflow is not mutable.")
        return self._taskflow

    def _add_step_tree(
        self,
        taskflow: TaskflowDefinition,
        raw_step: dict[str, Any],
        *,
        parent_id: str | None,
        created: list[TaskflowStep],
    ) -> str:
        title = str(raw_step.get("title") or raw_step.get("content") or "").strip()
        if not title:
            return "Each step requires title or content."
        step_id = str(raw_step.get("id") or "").strip() or self._next_step_id()
        if step_id in taskflow.steps:
            return f"Duplicate step id: {step_id}"
        default_completion_policy = (
            "auto_when_children_complete" if raw_step.get("children") else "manual"
        )
        completion_policy = str(
            raw_step.get("completion_policy")
            or raw_step.get("completion_mode")
            or default_completion_policy
        ).strip().lower()
        if completion_policy == "auto_complete":
            completion_policy = "auto_when_children_complete"
        if completion_policy not in {"manual", "auto_when_children_complete"}:
            completion_policy = "manual"
        context_policy = str(raw_step.get("context_policy") or "keep").strip()
        if context_policy not in {"keep", "fold", "auto"}:
            context_policy = "keep"
        step = TaskflowStep(
            id=step_id,
            title=title,
            instructions=str(raw_step.get("instructions") or raw_step.get("prompt") or ""),
            description=str(raw_step.get("description") or ""),
            completion_policy=completion_policy,  # type: ignore[arg-type]
            context_policy=context_policy,  # type: ignore[arg-type]
            review=TaskflowReviewPolicy.from_dict(raw_step.get("review")),
        )
        taskflow.steps[step.id] = step
        created.append(step)
        if parent_id:
            taskflow.edges.append(
                TaskflowEdge(from_step_id=parent_id, to_step_id=step.id, type="decomposes")
            )
        for raw_child in raw_step.get("children") or []:
            if not isinstance(raw_child, dict):
                return "children must be step objects."
            error = self._add_step_tree(
                taskflow,
                raw_child,
                parent_id=step.id,
                created=created,
            )
            if error:
                return error
        return ""

    def _complete_steps(
        self,
        steps: list[TaskflowStep],
        *,
        output: str | None,
        reviewer_type: str,
    ) -> list[TaskflowStep]:
        now = time.time()
        completed: list[TaskflowStep] = []
        for index, step in enumerate(steps):
            if step.status == "completed":
                continue
            if output is not None and index == 0:
                step.output = output
            if reviewer_type == "logger" or step.review.type == "logger":
                step.review_records.append(
                    TaskflowReviewRecord(
                        step_id=step.id,
                        reviewer_type="logger",
                        approved=True,
                        reviewer_identity="logger",
                    )
                )
            step.status = "completed"
            step.completed_at = now
            step.status_reason = None
            completed.append(step)
            self._emit_step_event("completed", step)
        self._auto_complete_parents(completed)
        return completed

    def _approve_reviewed_step(
        self,
        step: TaskflowStep,
        decision: TaskflowReviewDecision,
        ctx: Any,
    ) -> tuple[HookResult | None, dict[str, Any] | None]:
        message, folded_context = self._complete_reviewed_step(step, decision, ctx=ctx)
        if message:
            return HookResult.reinvoke(message), folded_context
        return None, folded_context

    def _complete_reviewed_step(
        self,
        step: TaskflowStep,
        decision: TaskflowReviewDecision,
        *,
        ctx: Any,
    ) -> tuple[str | None, dict[str, Any] | None]:
        completed = self._complete_steps(
            [step],
            output=step.output,
            reviewer_type=step.review.type,
        )
        folded_context = self._maybe_fold_for_submission(
            ctx,
            completed,
            context_policy=step.context_policy,
            summary=f"Completed {step.id}: {step.title}",
            handoff_notes=decision.feedback or "No reviewer handoff notes.",
        )
        message = self._advance_after_step(step, next_step_id=decision.next_step_id)
        self._sync_artifact()
        return message, folded_context

    def _reject_reviewed_step(
        self,
        step: TaskflowStep,
        decision: TaskflowReviewDecision,
    ) -> HookResult:
        run = self._active_run
        if step.attempt_count >= step.review.max_retries:
            step.status = "failed"
            if run is not None:
                run.status = "failed"
                run.completed_at = time.time()
            self._sync_artifact()
            self._emit_taskflow_event(
                "taskflow.run.failed",
                {
                    "step_id": step.id,
                    "title": f"Taskflow failed: {step.title}",
                    "message": decision.feedback,
                },
            )
            return HookResult.reinvoke(
                f"Taskflow step {step.title} ({step.id}) failed after "
                f"{step.attempt_count} attempt(s).\n\nFeedback:\n{decision.feedback}"
            )
        step.status = "active"
        if run is not None:
            run.status = "running"
        self._sync_artifact()
        self._emit_step_event("rejected", step, data={"feedback": decision.feedback})
        return HookResult.reinvoke(
            f"Taskflow step {step.title} ({step.id}) was not approved.\n\n"
            f"Feedback:\n{decision.feedback}\n\nRevise the work and call "
            "submit_taskflow_step again."
        )

    def _advance_after_step(
        self,
        step: TaskflowStep,
        *,
        next_step_id: str | None,
    ) -> str | None:
        taskflow = self._taskflow
        run = self._active_run
        if taskflow is None or run is None:
            return None
        if taskflow.execution_policy == "freeform":
            if not self._open_steps():
                run.status = "completed"
                run.completed_at = time.time()
            else:
                run.status = "running"
            return None
        downstream = self._transition_downstream_ids(step.id)
        selected = next_step_id or step.selected_next_step_id
        if selected and selected not in downstream:
            run.status = "failed"
            return f"Taskflow route error: {selected!r} is not downstream of {step.id!r}."
        if selected is None:
            if len(downstream) > 1:
                run.status = "running"
                return (
                    f"Taskflow step {step.title} ({step.id}) passed.\n\n"
                    "Multiple conditional transitions are available. Call "
                    "select_next_taskflow_step with the matching next_step_id "
                    "and reason before continuing.\n\n"
                    f"{self._format_transition_options(step)}"
                )
            selected = downstream[0] if downstream else None
        if selected is None:
            run.status = "completed"
            run.current_step_id = None
            run.completed_at = time.time()
            self._emit_taskflow_event(
                "taskflow.run.completed",
                {"title": "Taskflow completed", "message": "All required steps completed."},
            )
            return None
        next_step = taskflow.steps[selected]
        run.status = "running"
        run.current_step_id = next_step.id
        self._enter_step(next_step)
        self._emit_step_event("entered", next_step)
        edge_text = self._transition_summary(step.id, next_step.id)
        return (
            f"Taskflow step {step.title} ({step.id}) passed.\n\n"
            f"Entering next step: {next_step.title} ({next_step.id})\n\n"
            f"{edge_text}\n\n"
            f"{next_step.instructions}"
        )

    async def _review_step(
        self,
        agent: Any,
        step: TaskflowStep,
        ctx: Any,
    ) -> TaskflowReviewDecision | None:
        if step.review.type == "human":
            return await self._human_review(step, ctx)
        if step.review.type == "sub_agent":
            return await self._sub_agent_review(agent, step)
        return TaskflowReviewDecision(approved=True)

    @staticmethod
    def _runtime_review_available(ctx: Any) -> bool:
        return getattr(ctx, "review", None) is not None

    async def _human_review(
        self,
        step: TaskflowStep,
        ctx: Any,
    ) -> TaskflowReviewDecision | None:
        broker = getattr(ctx, "review", None)
        if broker is None:
            return None
        review_id = self._pending_human_review_id(step) or str(uuid.uuid4())[:8]
        review_id = self._request_human_review(
            step,
            review_id=review_id,
            broker_future=broker.create(
                review_id,
                plugin_id=self.plugin_id,
                review_type="human",
                payload={
                    "step_id": step.id,
                    "step_title": step.title,
                    "output": step.output,
                },
            ).future,
        )
        try:
            raw_decision = await broker.wait(review_id)
        finally:
            broker.discard(review_id)
            self._pending_human_reviews.pop(review_id, None)
        return self._taskflow_review_decision_from_runtime(raw_decision)

    def _pending_human_review_id(self, step: TaskflowStep) -> str | None:
        for review_id, review in self._pending_human_reviews.items():
            if isinstance(review, dict) and review.get("step_id") == step.id:
                return review_id
        return None

    def _request_human_review(
        self,
        step: TaskflowStep,
        review_id: str | None = None,
        broker_future: Any = None,
    ) -> str:
        existing_review_id = self._pending_human_review_id(step)
        if existing_review_id is not None:
            return existing_review_id
        review_id = review_id or str(uuid.uuid4())[:8]
        self._pending_human_reviews[review_id] = broker_future or {
            "review_id": review_id,
            "step_id": step.id,
            "created_at": time.time(),
        }
        review_data = {
            "kind": "human_review_request",
            "review_id": review_id,
            "plugin_id": self.plugin_id,
            "approve_action": "approve_taskflow_review",
            "reject_action": "reject_taskflow_review",
            "taskflow_id": self._taskflow.id if self._taskflow else "",
            "taskflow_title": self._taskflow.title if self._taskflow else "",
            "step_id": step.id,
            "step_title": step.title,
            "instructions": step.instructions,
            "output": step.output,
            "output_preview": (step.output or "")[:500],
            "attempt": step.attempt_count,
            "max_retries": step.review.max_retries,
        }
        self._emit_taskflow_event(
            "taskflow.review.requested",
            {
                "review_id": review_id,
                "message_id": f"taskflow-review-{review_id}",
                "level": "warning",
                "step_id": step.id,
                "step_title": step.title,
                "output": step.output,
                "output_preview": (step.output or "")[:500],
                "attempt": step.attempt_count,
                "max_retries": step.review.max_retries,
                "title": f"Review required: {step.title}",
                "message": f"Step {step.title} is awaiting human review.",
                "data": review_data,
            },
        )
        return review_id

    @staticmethod
    def _taskflow_review_decision_from_runtime(raw: Any) -> TaskflowReviewDecision:
        if isinstance(raw, TaskflowReviewDecision):
            return raw
        if isinstance(raw, RuntimeReviewDecision):
            return TaskflowReviewDecision(
                approved=raw.approved,
                feedback=raw.feedback,
                modified_output=raw.modified_output,
                next_step_id=raw.next_step_id,
            )
        return TaskflowReviewDecision(
            approved=bool(getattr(raw, "approved", False)),
            feedback=str(getattr(raw, "feedback", "") or ""),
            modified_output=getattr(raw, "modified_output", None),
            next_step_id=getattr(raw, "next_step_id", None),
        )

    async def _sub_agent_review(self, agent: Any, step: TaskflowStep) -> TaskflowReviewDecision:
        sub = agent.clone()
        if step.review.model:
            sub.set_model(step.review.model)
        taskflow_title = self._taskflow.title if self._taskflow else "Taskflow"
        review_message = (
            "Review this Taskflow step output. Respond with one JSON object only: "
            '{"approved": true/false, "feedback": "..."}.\n\n'
            f"Taskflow: {taskflow_title}\n"
            f"Step: {step.title} ({step.id})\n"
            f"Instructions:\n{step.instructions}\n\n"
            f"Output:\n{step.output or '(empty)'}\n\n"
            f"Review criteria:\n{step.review.prompt or 'Approve if the output satisfies the step.'}"
        )
        try:
            result = await sub.arun(review_message)
        except Exception as exc:
            return TaskflowReviewDecision(
                approved=False,
                feedback=f"Review infrastructure error: {type(exc).__name__}: {exc}",
            )
        return self._parse_review_decision(result.text)

    @staticmethod
    def _parse_review_decision(text: str) -> TaskflowReviewDecision:
        match = re.search(r'\{[^{}]*"approved"[^{}]*\}', text, re.DOTALL)
        payload = match.group(0) if match else text.strip()
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            lowered = text.lower()
            approved = "approved" in lowered and "not approved" not in lowered
            return TaskflowReviewDecision(approved=approved, feedback=text[:500])
        return TaskflowReviewDecision(
            approved=bool(data.get("approved", False)),
            feedback=str(data.get("feedback") or ""),
            next_step_id=data.get("next_step_id"),
            modified_output=data.get("modified_output"),
        )

    def _completion_closure(
        self,
        requested: list[TaskflowStep],
        complete_descendants: bool,
    ) -> tuple[list[TaskflowStep], str]:
        taskflow = self._taskflow
        if taskflow is None:
            return [], "No current taskflow."
        result: list[TaskflowStep] = []
        seen: set[str] = set()

        def add(step: TaskflowStep) -> None:
            if step.id in seen:
                return
            seen.add(step.id)
            result.append(step)

        for step in requested:
            add(step)
            if complete_descendants:
                for descendant in taskflow.descendants_of(step.id):
                    add(descendant)
            elif self._open_children(step):
                children = ", ".join(child.id for child in self._open_children(step))
                return [], (
                    f"Step {step.id} has unfinished child step(s): {children}. "
                    "Pass complete_descendants=true only if they are genuinely done."
                )
        return result, ""

    def _auto_complete_parents(self, completed_steps: list[TaskflowStep]) -> None:
        taskflow = self._taskflow
        if taskflow is None:
            return
        queue = list(completed_steps)
        now = time.time()
        while queue:
            step = queue.pop(0)
            for parent_id in taskflow.upstream_ids(step.id, edge_type="decomposes"):
                parent = taskflow.steps.get(parent_id)
                if parent is None or parent.status == "completed":
                    continue
                if parent.completion_policy != "auto_when_children_complete":
                    continue
                children = taskflow.children_of(parent.id)
                if children and all(child.status == "completed" for child in children):
                    parent.status = "completed"
                    parent.completed_at = now
                    queue.append(parent)
                    self._emit_step_event("completed", parent)

    def _resolve_step_selection(
        self,
        step_id: str | None,
        step_ids: list[str] | None,
    ) -> tuple[list[TaskflowStep], str]:
        taskflow = self._taskflow
        if taskflow is None:
            return [], "No current taskflow."
        if step_id is not None and step_ids is not None:
            return [], "Provide either step_id or step_ids, not both."
        raw_ids: list[Any]
        if step_ids is not None:
            raw_ids = step_ids
        elif step_id is not None:
            raw_ids = [step_id]
        else:
            return [], "Provide step_id or step_ids."
        steps: list[TaskflowStep] = []
        missing: list[str] = []
        seen: set[str] = set()
        for raw_id in raw_ids:
            current_id = str(raw_id).strip()
            if not current_id or current_id in seen:
                continue
            seen.add(current_id)
            step = taskflow.steps.get(current_id)
            if step is None:
                missing.append(current_id)
            else:
                steps.append(step)
        if missing:
            return [], f"Unknown step id(s): {', '.join(missing)}"
        if not steps:
            return [], "Provide at least one step id."
        return steps, ""

    def _maybe_fold_for_submission(
        self,
        ctx: Any,
        completed: list[TaskflowStep],
        *,
        context_policy: str | None,
        summary: str | None,
        handoff_notes: str | None,
    ) -> dict[str, Any] | None:
        policy = context_policy
        if policy is None and completed:
            policy = completed[0].context_policy
        policy = (policy or "keep").strip().lower()
        if policy == "auto":
            policy = "fold" if self._fold_completed_steps else "keep"
        if policy != "fold":
            return None
        if not self._fold_completed_steps:
            return {"enabled": False, "reason": "fold_completed_steps is disabled."}
        summary_text = summary.strip() if isinstance(summary, str) else ""
        handoff_text = handoff_notes.strip() if isinstance(handoff_notes, str) else ""
        if not summary_text:
            return {"enabled": True, "skipped": True, "reason": "summary is required."}
        if not handoff_text:
            return {"enabled": True, "skipped": True, "reason": "handoff_notes is required."}
        context = getattr(ctx, "context", None)
        messages = getattr(context, "messages", None)
        if not isinstance(messages, list):
            return {"enabled": True, "skipped": True, "reason": "No runtime context."}
        current_index = self._find_current_submit_message_index(messages)
        if current_index is None:
            return {
                "enabled": True,
                "skipped": True,
                "reason": "Could not locate active submit_taskflow_step call.",
            }
        start_index = self._fold_start_index(messages, current_index)
        folded_messages = deepcopy(messages[start_index:current_index])
        if not folded_messages:
            completed_step_ids = [step.id for step in completed]
            if self._fold_records:
                previous = self._fold_records[-1]
                for step_id in completed_step_ids:
                    if step_id not in previous.completed_step_ids:
                        previous.completed_step_ids.append(step_id)
                folded_context = previous.reference_dict()
                folded_context["enabled"] = True
                folded_context["skipped"] = True
                folded_context["referenced_fold_id"] = previous.fold_id
                folded_context["reason"] = (
                    "No new messages appeared since the previous "
                    "submit_taskflow_step fold. Refer to "
                    f"`{previous.fold_id}` for the shared folded context."
                )
                return folded_context
            return {
                "enabled": True,
                "skipped": True,
                "reason": (
                    "No messages were available to fold for this "
                    "submit_taskflow_step call."
                ),
                "summary": summary_text,
                "handoff_notes": handoff_text,
                "completed_step_ids": completed_step_ids,
            }
        if start_index < current_index:
            del messages[start_index:current_index]
        record = TaskflowFoldRecord(
            fold_id=f"TF{self._next_fold_number}",
            step_id=completed[0].id if completed else "unknown",
            step_title=completed[0].title if completed else "Unknown step",
            summary=summary_text,
            messages=folded_messages,
            completed_step_ids=[step.id for step in completed],
            created_at=time.time(),
            handoff_notes=handoff_text,
        )
        self._next_fold_number += 1
        self._fold_records.append(record)
        self._emit_taskflow_event(
            "taskflow.context.folded",
            {
                "fold": record.reference_dict(),
                "title": "Taskflow context folded",
                "message": f"Folded {len(record.messages)} message(s).",
            },
        )
        folded_context = record.reference_dict()
        folded_context["enabled"] = True
        folded_context["skipped"] = False
        return folded_context

    @staticmethod
    def _should_mark_submit_cache_point(folded_context: dict[str, Any] | None) -> bool:
        return bool(
            folded_context
            and folded_context.get("enabled")
            and not folded_context.get("skipped")
        )

    @staticmethod
    def _remove_previous_submit_cache_points(ctx: Any) -> None:
        context = getattr(ctx, "context", None)
        remover = getattr(context, "remove_cache_points", None)
        if callable(remover):
            remover(source=TASKFLOW_SUBMIT_CACHE_POINT_SOURCE)

    def _find_current_submit_message_index(self, messages: list[dict[str, Any]]) -> int | None:
        if self._active_submit_tool_call_id:
            for index in range(len(messages) - 1, -1, -1):
                if self._assistant_has_tool_call_id(messages[index], self._active_submit_tool_call_id):
                    return index
        for index in range(len(messages) - 1, -1, -1):
            if self._assistant_has_tool_call_name(messages[index], "submit_taskflow_step"):
                return index
        return None

    @staticmethod
    def _fold_start_index(messages: list[dict[str, Any]], current_index: int) -> int:
        latest_marker_end = -1
        for index in range(current_index):
            if not TaskflowPlugin._assistant_has_tool_call_name(messages[index], "submit_taskflow_step"):
                continue
            marker_end = index
            while marker_end + 1 < current_index and messages[marker_end + 1].get("role") == "tool":
                marker_end += 1
            latest_marker_end = marker_end
        return latest_marker_end + 1

    @staticmethod
    def _assistant_has_tool_call_id(message: dict[str, Any], tool_call_id: str) -> bool:
        if message.get("role") != "assistant":
            return False
        for part in message.get("content", []):
            if isinstance(part, dict) and part.get("type") == "tool_call" and part.get("id") == tool_call_id:
                return True
        return False

    @staticmethod
    def _assistant_has_tool_call_name(message: dict[str, Any], tool_name: str) -> bool:
        if message.get("role") != "assistant":
            return False
        for part in message.get("content", []):
            if isinstance(part, dict) and part.get("type") == "tool_call" and part.get("name") == tool_name:
                return True
        return False

    def _fold_records_for_lookup(
        self,
        *,
        step_id: str,
        fold_id: str | None,
    ) -> list[TaskflowFoldRecord]:
        normalized_fold_id = fold_id.strip().lower() if isinstance(fold_id, str) else ""
        if normalized_fold_id:
            return [
                record
                for record in self._fold_records
                if record.fold_id.lower() == normalized_fold_id
            ]
        normalized_step_id = step_id.strip().lower() if isinstance(step_id, str) else ""
        if normalized_step_id:
            return [
                record
                for record in reversed(self._fold_records)
                if record.step_id.lower() == normalized_step_id
                or normalized_step_id in {item.lower() for item in record.completed_step_ids}
            ]
        return list(reversed(self._fold_records))

    def _search_fold_records(
        self,
        records: list[TaskflowFoldRecord],
        query: str,
        *,
        max_chars: int,
    ) -> dict[str, Any]:
        needle = query.lower()
        matches: list[dict[str, Any]] = []
        total_chars = 0
        for record in records:
            fields = [
                ("summary", record.summary),
                ("handoff_notes", record.handoff_notes or ""),
                ("transcript", self._format_folded_messages(record.messages)),
            ]
            for source, text in fields:
                haystack = text.lower()
                index = haystack.find(needle)
                if index < 0:
                    continue
                start = max(0, index - 240)
                end = min(len(text), index + len(query) + 240)
                snippet = text[start:end]
                if total_chars + len(snippet) > max_chars:
                    continue
                total_chars += len(snippet)
                matches.append(
                    {
                        "fold_id": record.fold_id,
                        "step_id": record.step_id,
                        "source": source,
                        "snippet": snippet,
                    }
                )
        return {
            "mode": "search",
            "query": query,
            "match_count": len(matches),
            "matches": matches,
        }

    def _format_folded_messages(self, messages: list[dict[str, Any]]) -> str:
        if not messages:
            return "[No detailed messages were folded.]"
        sections = []
        for index, message in enumerate(messages, 1):
            role = message.get("role", "unknown")
            sections.append(f"## Message {index} ({role})\n{self._format_content(message.get('content', []))}")
        return "\n\n".join(sections)

    def _format_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return self._safe_json(content)
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                parts.append(str(part))
                continue
            part_type = part.get("type")
            if part_type == "text":
                parts.append(str(part.get("text", "")))
            elif part_type == "tool_call":
                parts.append(f"[tool_call] name={part.get('name', '')} arguments={self._safe_json(part.get('arguments', {}))}")
            elif part_type == "tool_result":
                parts.append(f"[tool_result]\n{self._format_content(part.get('content', []))}")
            elif part_type == "reasoning":
                reasoning = part.get("reasoning") or ""
                if reasoning:
                    parts.append(f"[reasoning]\n{reasoning}")
            else:
                parts.append(self._safe_json(part))
        return "\n".join(text for text in parts if text)

    @staticmethod
    def _safe_json(value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, indent=2, default=str)
        except TypeError:
            return str(value)

    @staticmethod
    def _truncate(text: str, max_chars: int, *, marker: str) -> tuple[str, bool]:
        try:
            limit = max(1, int(max_chars))
        except (TypeError, ValueError):
            limit = 20000
        if len(text) <= limit:
            return text, False
        if limit <= len(marker):
            return marker[:limit], True
        return text[: limit - len(marker)].rstrip() + marker, True

    def _sync_artifact(self, status: str | None = None) -> None:
        taskflow = self._taskflow
        if taskflow is None:
            return
        run = self._active_run
        artifact_status = status or (run.status if run else "defined")
        self.upsert_artifact(
            "current-taskflow",
            artifact_type="taskflow",
            title=f"Taskflow: {taskflow.title}",
            content=self._format_artifact(),
            language="markdown",
            mime_type="text/markdown",
            status=artifact_status,
            metadata=self._state_dict(),
        )

    def _format_artifact(self) -> str:
        taskflow = self._taskflow
        run = self._active_run
        if taskflow is None:
            return "No taskflow."
        lines = [
            f"# Taskflow: {taskflow.title}",
            "",
            f"Mode: {taskflow.mode}",
            f"Execution: {taskflow.execution_policy}",
            f"Status: {run.status if run else 'defined'}",
            "",
        ]
        if taskflow.execution_policy == "freeform":
            for step in taskflow.children_of(None):
                self._append_step_tree(lines, step, depth=0)
        else:
            for step in taskflow.steps.values():
                lines.append(f"- {self._step_mark(step)} {step.title} ({step.id})")
        return "\n".join(lines)

    def _append_step_tree(self, lines: list[str], step: TaskflowStep, *, depth: int) -> None:
        indent = "  " * depth
        lines.append(f"{indent}- {self._step_mark(step)} {step.title} ({step.id})")
        taskflow = self._taskflow
        if taskflow is None:
            return
        for child in taskflow.children_of(step.id):
            self._append_step_tree(lines, child, depth=depth + 1)

    @staticmethod
    def _step_mark(step: TaskflowStep) -> str:
        return {
            "pending": "[ ]",
            "active": "[>]",
            "reviewing": "[?]",
            "completed": "[x]",
            "blocked": "[!]",
            "deferred": "[~]",
            "canceled": "[-]",
            "obsolete": "[-]",
            "failed": "[!]",
            "skipped": "[-]",
        }.get(step.status, "[ ]")

    def _state_dict(self) -> dict[str, Any]:
        return {
            "status": self._active_run.status if self._active_run else "idle",
            "taskflow": self._taskflow.to_dict() if self._taskflow else None,
            "run": self._active_run.to_dict() if self._active_run else None,
            "steps": self._steps_with_children(),
            "open_step_count": len(self._open_steps()),
            "current_transitions": self._current_transition_options(),
            "fold_completed_steps": self._fold_completed_steps,
            "folded_contexts": [record.reference_dict() for record in self._fold_records],
        }

    def _steps_with_children(self) -> list[dict[str, Any]]:
        taskflow = self._taskflow
        if taskflow is None:
            return []
        return [self._step_with_children(step) for step in taskflow.children_of(None)]

    def _step_with_children(self, step: TaskflowStep) -> dict[str, Any]:
        taskflow = self._taskflow
        children = taskflow.children_of(step.id) if taskflow else []
        return {
            **step.to_dict(),
            "children": [self._step_with_children(child) for child in children],
        }

    def _current_transition_options(self) -> list[dict[str, Any]]:
        step = self._current_step()
        taskflow = self._taskflow
        if step is None or taskflow is None:
            return []
        if taskflow.execution_policy == "sequential":
            return [
                {
                    "to_step_id": step_id,
                    "to_step_title": taskflow.steps[step_id].title,
                    "label": "",
                    "condition": None,
                    "is_loop": False,
                }
                for step_id in self._sequential_downstream_ids(step.id)
                if step_id in taskflow.steps
            ]
        return [
            {
                "to_step_id": edge.to_step_id,
                "to_step_title": taskflow.steps[edge.to_step_id].title,
                "label": edge.label,
                "condition": edge.condition,
                "is_loop": edge.to_step_id == step.id,
            }
            for edge in taskflow.transition_edges(step.id)
            if edge.to_step_id in taskflow.steps
        ]

    def _emit_step_event(
        self,
        action: str,
        step: TaskflowStep,
        data: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "event_name": "taskflow.step.updated",
            "action": action,
            "step": step.to_dict(),
            "state": self._state_dict(),
            "title": "Taskflow step updated",
            "message": f"{action}: {step.id}",
            "data": data or step.to_dict(),
        }
        self.emit_plugin_event("plugin.event", payload)

    def _emit_taskflow_event(self, event_name: str, payload: dict[str, Any]) -> None:
        self.emit_plugin_event(
            "plugin.event",
            {
                "event_name": event_name,
                "state": self._state_dict(),
                **payload,
            },
        )

    def _open_steps(self) -> list[TaskflowStep]:
        taskflow = self._taskflow
        if taskflow is None:
            return []
        return [
            step
            for step in taskflow.steps.values()
            if step.status in OPEN_STEP_STATUSES
        ]

    def _has_only_terminal_or_parked_steps(self) -> bool:
        taskflow = self._taskflow
        if taskflow is None:
            return False
        if not taskflow.steps:
            return False
        allowed = TERMINAL_STEP_STATUSES | PARKED_STEP_STATUSES
        return all(step.status in allowed for step in taskflow.steps.values())

    def _has_parked_steps(self) -> bool:
        taskflow = self._taskflow
        if taskflow is None:
            return False
        return any(step.status in PARKED_STEP_STATUSES for step in taskflow.steps.values())

    def _open_children(self, step: TaskflowStep) -> list[TaskflowStep]:
        taskflow = self._taskflow
        if taskflow is None:
            return []
        return [
            child
            for child in taskflow.children_of(step.id)
            if child.status in OPEN_STEP_STATUSES
        ]

    @staticmethod
    def _reset_workflow_steps(taskflow: TaskflowDefinition) -> None:
        for step in taskflow.steps.values():
            step.status = "pending"
            step.output = None
            step.attempt_count = 0
            step.review_records.clear()
            step.started_at = None
            step.completed_at = None
            step.selected_next_step_id = None
            step.routing_reason = ""
            step.status_reason = None

    @staticmethod
    def _enter_step(step: TaskflowStep) -> None:
        step.status = "active"
        step.output = None
        step.attempt_count = 0
        step.started_at = time.time()
        step.completed_at = None
        step.selected_next_step_id = None
        step.routing_reason = ""
        step.status_reason = None

    def _transition_downstream_ids(self, step_id: str) -> list[str]:
        taskflow = self._taskflow
        if taskflow is None:
            return []
        if taskflow.execution_policy == "sequential":
            return self._sequential_downstream_ids(step_id)
        return [edge.to_step_id for edge in taskflow.transition_edges(step_id)]

    def _sequential_downstream_ids(self, step_id: str) -> list[str]:
        taskflow = self._taskflow
        if taskflow is None:
            return []
        ids = list(taskflow.steps)
        try:
            index = ids.index(step_id)
        except ValueError:
            return []
        return ids[index + 1 : index + 2]

    def _transition_summary(self, from_step_id: str, to_step_id: str) -> str:
        taskflow = self._taskflow
        if taskflow is None or taskflow.execution_policy == "sequential":
            return ""
        for edge in taskflow.transition_edges(from_step_id):
            if edge.to_step_id != to_step_id:
                continue
            details = []
            if edge.label:
                details.append(f"label={edge.label}")
            if edge.condition:
                details.append(f"condition={edge.condition}")
            return "Transition: " + ", ".join(details) if details else ""
        return ""

    def _format_transition_options(self, step: TaskflowStep) -> str:
        taskflow = self._taskflow
        if taskflow is None:
            return "(no transitions)"
        if taskflow.execution_policy == "sequential":
            downstream = self._sequential_downstream_ids(step.id)
            if not downstream:
                return "(terminal step)"
            next_step = taskflow.steps[downstream[0]]
            return f"- {next_step.id}: {next_step.title}"
        edges = taskflow.transition_edges(step.id)
        if not edges:
            return "(terminal step)"
        lines = []
        for edge in edges:
            target = taskflow.steps.get(edge.to_step_id)
            target_title = target.title if target else edge.to_step_id
            loop_marker = " [loop]" if edge.to_step_id == step.id else ""
            detail = []
            if edge.label:
                detail.append(f"label: {edge.label}")
            if edge.condition:
                detail.append(f"condition: {edge.condition}")
            detail_text = f" ({'; '.join(detail)})" if detail else ""
            lines.append(f"- {edge.to_step_id}: {target_title}{loop_marker}{detail_text}")
        return "\n".join(lines)

    def _current_step(self) -> TaskflowStep | None:
        run = self._active_run
        if run is None or not run.current_step_id:
            return None
        return self._step(run.current_step_id)

    def _step(self, step_id: str) -> TaskflowStep | None:
        if self._taskflow is None:
            return None
        return self._taskflow.steps.get(step_id)

    def _next_step_id(self) -> str:
        while True:
            step_id = f"T{self._next_step_number}"
            self._next_step_number += 1
            if self._taskflow is None or step_id not in self._taskflow.steps:
                return step_id

    def _refresh_next_step_number(self) -> None:
        max_seen = 0
        if self._taskflow is not None:
            for step_id in self._taskflow.steps:
                if step_id.startswith("T") and step_id[1:].isdigit():
                    max_seen = max(max_seen, int(step_id[1:]))
        self._next_step_number = max_seen + 1

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        normalized = mode.strip().lower() if isinstance(mode, str) else "plan"
        return normalized if normalized in {"plan", "workflow"} else "plan"

    @staticmethod
    def _normalize_execution_policy(policy: str, mode: str) -> str:
        normalized = policy.strip().lower() if isinstance(policy, str) else ""
        if normalized in {"freeform", "sequential", "gated_graph"}:
            return normalized
        return "freeform" if mode == "plan" else "gated_graph"

    @staticmethod
    def _slug(value: str) -> str:
        lowered = value.strip().lower()
        slug = "".join(ch if ch.isalnum() else "_" for ch in lowered).strip("_")
        return slug or "taskflow"

    def _start_message(self) -> str:
        taskflow = self._taskflow
        if taskflow is None:
            return "Taskflow started."
        if taskflow.execution_policy == "freeform":
            return f"Taskflow '{taskflow.title}' started with {len(taskflow.steps)} step(s)."
        step = self._current_step()
        if step is None:
            return f"Taskflow '{taskflow.title}' started."
        return f"Taskflow '{taskflow.title}' started. Current step: {step.title}"

    def _build_taskflow_prompt(self) -> str:
        return (
            f"\n{TASKFLOW_PROMPT_BEGIN}\n"
            "TaskflowPlugin is enabled. A taskflow is a set of steps that can act "
            "as either a mutable plan or a gated workflow.\n\n"
            "Use plan mode for dynamic task decomposition. Use workflow mode for "
            "ordered or reviewed gates. Runtime reminders from TaskflowPlugin are "
            "automatic plugin messages, not human-user instructions.\n\n"
            "Tool quick reference:\n"
            "- create_taskflow/load_taskflow/list_taskflows/start_taskflow manage definitions and runs.\n"
            "- add_taskflow_steps adds mutable plan steps and child decomposition.\n"
            "- update_taskflow_steps parks, cancels, reopens, or marks step status.\n"
            "- submit_taskflow_step completes a step or submits it for review.\n"
            "- select_next_taskflow_step records a workflow route. Use it for conditional branches and loop exits.\n"
            "- get_taskflow_status and get_pending_taskflow_reviews inspect state.\n"
            "- recall_taskflow_context reads folded completed-step detail.\n"
            f"{TASKFLOW_PROMPT_END}\n"
        )

    def _format_step_context(self, step: TaskflowStep) -> str:
        transition_options = self._format_transition_options(step)
        return (
            f"{TASKFLOW_REMINDER_BEGIN}\n"
            "Taskflow current step context.\n\n"
            f"Step: {step.title} ({step.id})\n"
            f"Attempt: {step.attempt_count}/{step.review.max_retries}\n"
            f"Instructions:\n{step.instructions or step.description or step.title}\n\n"
            f"Available next transitions:\n{transition_options}\n"
            "When multiple transitions are available, choose the matching "
            "condition with select_next_taskflow_step before or after submitting.\n"
            "When this step is complete, call submit_taskflow_step with output.\n"
            f"{TASKFLOW_REMINDER_END}"
        )

    def _format_freeform_reminder(self, open_steps: list[TaskflowStep]) -> str:
        lines = [
            f"{TASKFLOW_REMINDER_BEGIN}",
            "Taskflow has unfinished steps. Continue the work or update step status.",
            "",
        ]
        for step in open_steps:
            lines.append(f"- {step.id}: {step.title}")
        lines.append(TASKFLOW_REMINDER_END)
        return "\n".join(lines)

    def _format_step_reminder(self, step: TaskflowStep) -> str:
        if step.status == "completed" and len(self._transition_downstream_ids(step.id)) > 1:
            return (
                f"{TASKFLOW_REMINDER_BEGIN}\n"
                f"Taskflow step {step.title} ({step.id}) is complete and needs "
                "a conditional route selection.\n\n"
                f"{self._format_transition_options(step)}\n\n"
                "Call select_next_taskflow_step with the matching next_step_id "
                "and reason.\n"
                f"{TASKFLOW_REMINDER_END}"
            )
        return (
            f"{TASKFLOW_REMINDER_BEGIN}\n"
            f"Taskflow is still running. Current step: {step.title} ({step.id}).\n\n"
            f"{step.instructions or step.description or step.title}\n\n"
            f"Available next transitions:\n{self._format_transition_options(step)}\n\n"
            "Complete the work and call submit_taskflow_step with output.\n"
            f"{TASKFLOW_REMINDER_END}"
        )


def _find_last_user_insert_index(messages: list[dict[str, Any]]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            return index
    return len(messages)


_TASKFLOW_MANUAL = """# Taskflow

Taskflow unifies mutable plans and gated workflows.

## YAML Shape

```yaml
id: release_flow
title: Release Flow
mode: workflow
execution_policy: gated_graph
start_step_id: test
steps:
  - id: test
    title: Run tests
    instructions: Run the full test suite.
    review:
      type: logger
      max_retries: 2
  - id: ship
    title: Ship release
    instructions: Prepare release notes.
    review:
      type: human
edges:
  - from: test
    to: ship
    type: transitions
```

Use mode `plan` with execution_policy `freeform` for dynamic task lists. Use
edge type `decomposes` for parent/child task breakdown, and `transitions` for
workflow routing.

Transition edges may contain cycles. Use a loop-back transition plus an exit
transition with clear `condition` text, then call `select_next_taskflow_step`
to choose the matching route when more than one transition is available.
"""
