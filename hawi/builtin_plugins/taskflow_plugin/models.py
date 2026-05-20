from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal


TaskflowMode = Literal["plan", "workflow"]
TaskflowExecutionPolicy = Literal["freeform", "sequential", "gated_graph"]
TaskflowEdgeType = Literal["decomposes", "transitions", "depends_on"]
TaskflowStepStatus = Literal[
    "pending",
    "active",
    "reviewing",
    "completed",
    "blocked",
    "deferred",
    "canceled",
    "obsolete",
    "failed",
    "skipped",
]
TaskflowRunStatus = Literal[
    "idle",
    "running",
    "paused",
    "paused_awaiting_review",
    "completed",
    "failed",
]
TaskflowReviewType = Literal["none", "logger", "sub_agent", "human"]
TaskflowCompletionPolicy = Literal["manual", "auto_when_children_complete"]
TaskflowContextPolicy = Literal["keep", "fold", "auto"]

OPEN_STEP_STATUSES = {"pending", "active"}
PARKED_STEP_STATUSES = {"blocked", "deferred"}
TERMINAL_STEP_STATUSES = {
    "completed",
    "canceled",
    "obsolete",
    "failed",
    "skipped",
}


@dataclass
class TaskflowReviewPolicy:
    type: TaskflowReviewType = "none"
    prompt: str = ""
    model: str | None = None
    max_retries: int = 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "prompt": self.prompt,
            "model": self.model,
            "max_retries": self.max_retries,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TaskflowReviewPolicy":
        data = dict(data or {})
        review_type = str(data.get("type") or "none").strip().lower()
        if review_type not in {"none", "logger", "sub_agent", "human"}:
            review_type = "none"
        try:
            max_retries = int(data.get("max_retries", 3))
        except (TypeError, ValueError):
            max_retries = 3
        return cls(
            type=review_type,  # type: ignore[arg-type]
            prompt=str(data.get("prompt") or data.get("sub_agent_prompt") or ""),
            model=data.get("model") or data.get("sub_agent_model"),
            max_retries=max(1, max_retries),
        )


@dataclass
class TaskflowReviewRecord:
    step_id: str
    reviewer_type: str
    approved: bool
    feedback: str = ""
    reviewer_identity: str | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "reviewer_type": self.reviewer_type,
            "approved": self.approved,
            "feedback": self.feedback,
            "reviewer_identity": self.reviewer_identity,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskflowReviewRecord":
        return cls(
            step_id=str(data.get("step_id") or ""),
            reviewer_type=str(data.get("reviewer_type") or "none"),
            approved=bool(data.get("approved", False)),
            feedback=str(data.get("feedback") or ""),
            reviewer_identity=data.get("reviewer_identity"),
            timestamp=float(data.get("timestamp", 0.0)),
        )


@dataclass
class TaskflowStep:
    id: str
    title: str
    instructions: str = ""
    description: str = ""
    status: TaskflowStepStatus = "pending"
    output: str | None = None
    completion_policy: TaskflowCompletionPolicy = "manual"
    context_policy: TaskflowContextPolicy = "keep"
    review: TaskflowReviewPolicy = field(default_factory=TaskflowReviewPolicy)
    attempt_count: int = 0
    review_records: list[TaskflowReviewRecord] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    selected_next_step_id: str | None = None
    routing_reason: str = ""
    status_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "instructions": self.instructions,
            "description": self.description,
            "status": self.status,
            "output": self.output,
            "completion_policy": self.completion_policy,
            "context_policy": self.context_policy,
            "review": self.review.to_dict(),
            "attempt_count": self.attempt_count,
            "review_records": [record.to_dict() for record in self.review_records],
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "selected_next_step_id": self.selected_next_step_id,
            "routing_reason": self.routing_reason,
            "status_reason": self.status_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskflowStep":
        status = str(data.get("status") or "pending").strip().lower()
        if status not in {
            "pending",
            "active",
            "reviewing",
            "completed",
            "blocked",
            "deferred",
            "canceled",
            "obsolete",
            "failed",
            "skipped",
        }:
            status = "pending"
        completion_policy = str(
            data.get("completion_policy") or data.get("completion_mode") or "manual"
        ).strip().lower()
        if completion_policy == "auto_complete":
            completion_policy = "auto_when_children_complete"
        if completion_policy not in {"manual", "auto_when_children_complete"}:
            completion_policy = "manual"
        context_policy = str(data.get("context_policy") or "keep").strip().lower()
        if context_policy not in {"keep", "fold", "auto"}:
            context_policy = "keep"
        return cls(
            id=str(data["id"]),
            title=str(data.get("title") or data.get("content") or data["id"]),
            instructions=str(data.get("instructions") or data.get("prompt") or ""),
            description=str(data.get("description") or ""),
            status=status,  # type: ignore[arg-type]
            output=data.get("output"),
            completion_policy=completion_policy,  # type: ignore[arg-type]
            context_policy=context_policy,  # type: ignore[arg-type]
            review=TaskflowReviewPolicy.from_dict(data.get("review")),
            attempt_count=int(data.get("attempt_count", 0) or 0),
            review_records=[
                TaskflowReviewRecord.from_dict(record)
                for record in data.get("review_records", [])
                if isinstance(record, dict)
            ],
            created_at=float(data.get("created_at", 0.0) or 0.0),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            selected_next_step_id=data.get("selected_next_step_id"),
            routing_reason=str(data.get("routing_reason") or ""),
            status_reason=data.get("status_reason"),
        )


@dataclass
class TaskflowEdge:
    from_step_id: str
    to_step_id: str
    type: TaskflowEdgeType = "transitions"
    label: str = ""
    condition: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.from_step_id,
            "to": self.to_step_id,
            "type": self.type,
            "label": self.label,
            "condition": self.condition,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskflowEdge":
        edge_type = str(data.get("type") or "transitions").strip().lower()
        if edge_type not in {"decomposes", "transitions", "depends_on"}:
            edge_type = "transitions"
        return cls(
            from_step_id=str(data.get("from") or data.get("from_step_id") or ""),
            to_step_id=str(data.get("to") or data.get("to_step_id") or ""),
            type=edge_type,  # type: ignore[arg-type]
            label=str(data.get("label") or ""),
            condition=data.get("condition"),
        )


@dataclass
class TaskflowDefinition:
    id: str
    title: str
    mode: TaskflowMode = "plan"
    execution_policy: TaskflowExecutionPolicy = "freeform"
    mutable: bool = True
    description: str = ""
    steps: dict[str, TaskflowStep] = field(default_factory=dict)
    edges: list[TaskflowEdge] = field(default_factory=list)
    start_step_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "mode": self.mode,
            "execution_policy": self.execution_policy,
            "mutable": self.mutable,
            "description": self.description,
            "start_step_id": self.start_step_id,
            "steps": [step.to_dict() for step in self.steps.values()],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskflowDefinition":
        mode = str(data.get("mode") or "plan").strip().lower()
        if mode not in {"plan", "workflow"}:
            mode = "plan"
        execution_policy = str(
            data.get("execution_policy")
            or ("gated_graph" if mode == "workflow" else "freeform")
        ).strip().lower()
        if execution_policy not in {"freeform", "sequential", "gated_graph"}:
            execution_policy = "freeform" if mode == "plan" else "gated_graph"
        taskflow = cls(
            id=str(data.get("id") or data.get("name") or "taskflow"),
            title=str(data.get("title") or data.get("name") or "Taskflow"),
            mode=mode,  # type: ignore[arg-type]
            execution_policy=execution_policy,  # type: ignore[arg-type]
            mutable=bool(data.get("mutable", mode == "plan")),
            description=str(data.get("description") or ""),
            start_step_id=data.get("start_step_id"),
        )
        for raw_step in data.get("steps", []):
            if not isinstance(raw_step, dict):
                continue
            _add_step_tree(taskflow, raw_step, parent_id=None)
        for raw_edge in data.get("edges", []):
            if not isinstance(raw_edge, dict):
                continue
            edge = TaskflowEdge.from_dict(raw_edge)
            if edge.from_step_id and edge.to_step_id:
                taskflow.edges.append(edge)
        return taskflow

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.steps:
            errors.append("Taskflow must contain at least one step.")
        if self.start_step_id and self.start_step_id not in self.steps:
            errors.append(f"start_step_id {self.start_step_id!r} is not a step id.")
        for edge in self.edges:
            if edge.from_step_id not in self.steps:
                errors.append(f"Edge from unknown step: {edge.from_step_id}")
            if edge.to_step_id not in self.steps:
                errors.append(f"Edge to unknown step: {edge.to_step_id}")
        errors.extend(self._decomposition_cycle_errors())
        return errors

    def transition_edges(self, step_id: str) -> list[TaskflowEdge]:
        return [
            edge
            for edge in self.edges
            if edge.from_step_id == step_id and edge.type == "transitions"
        ]

    def downstream_ids(
        self,
        step_id: str,
        edge_type: TaskflowEdgeType | None = None,
    ) -> list[str]:
        return [
            edge.to_step_id
            for edge in self.edges
            if edge.from_step_id == step_id
            and (edge_type is None or edge.type == edge_type)
        ]

    def upstream_ids(
        self,
        step_id: str,
        edge_type: TaskflowEdgeType | None = None,
    ) -> list[str]:
        return [
            edge.from_step_id
            for edge in self.edges
            if edge.to_step_id == step_id
            and (edge_type is None or edge.type == edge_type)
        ]

    def children_of(self, step_id: str | None) -> list[TaskflowStep]:
        if step_id is None:
            child_ids = {
                edge.to_step_id for edge in self.edges if edge.type == "decomposes"
            }
            return [step for step in self.steps.values() if step.id not in child_ids]
        ids = self.downstream_ids(step_id, edge_type="decomposes")
        return [self.steps[item_id] for item_id in ids if item_id in self.steps]

    def descendants_of(self, step_id: str) -> list[TaskflowStep]:
        result: list[TaskflowStep] = []
        for child in self.children_of(step_id):
            result.append(child)
            result.extend(self.descendants_of(child.id))
        return result

    def _decomposition_cycle_errors(self) -> list[str]:
        errors: list[str] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(step_id: str, path: list[str]) -> None:
            if step_id in visiting:
                cycle_path = " -> ".join([*path, step_id])
                errors.append(f"Decomposition cycle is not allowed: {cycle_path}")
                return
            if step_id in visited:
                return
            visiting.add(step_id)
            for child_id in self.downstream_ids(step_id, edge_type="decomposes"):
                visit(child_id, [*path, step_id])
            visiting.remove(step_id)
            visited.add(step_id)

        for step_id in self.steps:
            visit(step_id, [])
        return errors


@dataclass
class TaskflowRun:
    id: str
    taskflow_id: str
    status: TaskflowRunStatus = "running"
    current_step_id: str | None = None
    global_context: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    pause_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "taskflow_id": self.taskflow_id,
            "status": self.status,
            "current_step_id": self.current_step_id,
            "global_context": self.global_context,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "pause_reason": self.pause_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskflowRun":
        return cls(
            id=str(data["id"]),
            taskflow_id=str(data.get("taskflow_id") or ""),
            status=str(data.get("status") or "running"),  # type: ignore[arg-type]
            current_step_id=data.get("current_step_id"),
            global_context=dict(data.get("global_context", {})),
            created_at=float(data.get("created_at", 0.0) or 0.0),
            completed_at=data.get("completed_at"),
            pause_reason=str(data.get("pause_reason") or ""),
        )


@dataclass
class TaskflowReviewDecision:
    approved: bool
    feedback: str = ""
    next_step_id: str | None = None
    modified_output: str | None = None


@dataclass
class TaskflowFoldRecord:
    fold_id: str
    step_id: str
    step_title: str
    summary: str
    messages: list[dict[str, Any]]
    completed_step_ids: list[str]
    created_at: float
    handoff_notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "step_id": self.step_id,
            "step_title": self.step_title,
            "summary": self.summary,
            "messages": self.messages,
            "completed_step_ids": self.completed_step_ids,
            "created_at": self.created_at,
            "handoff_notes": self.handoff_notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskflowFoldRecord":
        return cls(
            fold_id=str(data["fold_id"]),
            step_id=str(data.get("step_id") or ""),
            step_title=str(data.get("step_title") or ""),
            summary=str(data.get("summary") or ""),
            messages=list(data.get("messages", [])),
            completed_step_ids=list(data.get("completed_step_ids", [])),
            created_at=float(data.get("created_at", 0.0) or 0.0),
            handoff_notes=data.get("handoff_notes"),
        )

    def reference_dict(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "step_id": self.step_id,
            "step_title": self.step_title,
            "summary": self.summary,
            "handoff_notes": self.handoff_notes,
            "completed_step_ids": self.completed_step_ids,
            "folded_message_count": len(self.messages),
            "read_tool": "recall_taskflow_context",
        }


def _add_step_tree(
    taskflow: TaskflowDefinition,
    raw_step: dict[str, Any],
    *,
    parent_id: str | None,
) -> None:
    step_data = raw_step
    if (
        raw_step.get("children")
        and "completion_policy" not in raw_step
        and "completion_mode" not in raw_step
    ):
        step_data = {**raw_step, "completion_policy": "auto_when_children_complete"}
    step = TaskflowStep.from_dict(step_data)
    taskflow.steps[step.id] = step
    if parent_id is not None:
        taskflow.edges.append(
            TaskflowEdge(
                from_step_id=parent_id,
                to_step_id=step.id,
                type="decomposes",
            )
        )
    for raw_child in raw_step.get("children", []) or []:
        if isinstance(raw_child, dict):
            _add_step_tree(taskflow, raw_child, parent_id=step.id)
