"""Workflow data models: DAG nodes, edges, execution state."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

# ── Review configuration types ──

ReviewType = Literal["human", "sub_agent", "logger", "none"]


@dataclass
class ReviewConfig:
    """Per-node review configuration.

    Attributes:
        type: Which reviewer to use.
        sub_agent_prompt: Prompt for SubAgentReviewer (only for type='sub_agent').
        sub_agent_model: Optional model override for the review sub-agent.
    """

    type: ReviewType = "logger"
    sub_agent_prompt: str = ""
    sub_agent_model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "sub_agent_prompt": self.sub_agent_prompt,
            "sub_agent_model": self.sub_agent_model,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReviewConfig":
        return cls(
            type=data.get("type", "logger"),
            sub_agent_prompt=data.get("sub_agent_prompt", ""),
            sub_agent_model=data.get("sub_agent_model"),
        )


# ── Workflow definition (static DAG) ──


@dataclass
class WorkflowNode:
    """A single node (gate) in a workflow DAG.

    Attributes:
        id: Unique node identifier within the workflow.
        name: Human-readable label.
        description: What this node accomplishes.
        prompt: Task instructions injected into the agent's context.
        review: Review configuration for this gate.
        input_schema: Expected input shape (future: enforce at runtime).
        output_schema: Expected output shape (future: enforce at runtime).
        max_retries: Maximum review rejections before the workflow fails.
        timeout_minutes: Optional per-node timeout.
    """

    id: str
    name: str
    description: str = ""
    prompt: str = ""
    review: ReviewConfig = field(default_factory=ReviewConfig)
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    max_retries: int = 3
    timeout_minutes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "prompt": self.prompt,
            "review": self.review.to_dict(),
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "max_retries": self.max_retries,
            "timeout_minutes": self.timeout_minutes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowNode":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            prompt=data.get("prompt", ""),
            review=ReviewConfig.from_dict(data.get("review", {})),
            input_schema=data.get("input_schema"),
            output_schema=data.get("output_schema"),
            max_retries=data.get("max_retries", 3),
            timeout_minutes=data.get("timeout_minutes"),
        )


@dataclass
class WorkflowEdge:
    """A directed edge between two workflow nodes.

    Attributes:
        from_node_id: Source node id.
        to_node_id: Target node id.
        label: Human-readable label (e.g. 'approved', 'escalated').
        condition: Reserved for future conditional routing.
    """

    from_node_id: str
    to_node_id: str
    label: str = ""
    condition: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.from_node_id,
            "to": self.to_node_id,
            "label": self.label,
            "condition": self.condition,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowEdge":
        return cls(
            from_node_id=data["from"],
            to_node_id=data["to"],
            label=data.get("label", ""),
            condition=data.get("condition"),
        )


@dataclass
class Workflow:
    """Complete workflow DAG definition.

    Attributes:
        id: Unique workflow identifier.
        name: Human-readable workflow name.
        description: What the workflow does.
        nodes: All nodes keyed by id.
        edges: Directed edges between nodes.
        start_node_id: The entry node.
        global_context_schema: Reserved for typed global context.
    """

    id: str
    name: str
    description: str = ""
    nodes: dict[str, WorkflowNode] = field(default_factory=dict)
    edges: list[WorkflowEdge] = field(default_factory=list)
    start_node_id: str = ""
    global_context_schema: dict[str, Any] | None = None

    def add_node(self, node: WorkflowNode) -> None:
        self.nodes[node.id] = node

    def add_edge(self, edge: WorkflowEdge) -> None:
        if edge.from_node_id not in self.nodes:
            raise ValueError(f"Unknown from_node_id: {edge.from_node_id}")
        if edge.to_node_id not in self.nodes:
            raise ValueError(f"Unknown to_node_id: {edge.to_node_id}")
        self.edges.append(edge)

    def remove_node(self, node_id: str) -> None:
        self.nodes.pop(node_id, None)
        self.edges = [
            e for e in self.edges
            if e.from_node_id != node_id and e.to_node_id != node_id
        ]

    def remove_edge(self, from_node_id: str, to_node_id: str) -> bool:
        before = len(self.edges)
        self.edges = [
            e for e in self.edges
            if not (e.from_node_id == from_node_id and e.to_node_id == to_node_id)
        ]
        return len(self.edges) < before

    def downstream_node_ids(self, node_id: str) -> list[str]:
        """Return ids of nodes reachable by a single edge from *node_id*."""
        return [e.to_node_id for e in self.edges if e.from_node_id == node_id]

    def upstream_node_ids(self, node_id: str) -> list[str]:
        """Return ids of nodes that point to *node_id*."""
        return [e.from_node_id for e in self.edges if e.to_node_id == node_id]

    def terminal_node_ids(self) -> list[str]:
        """Return ids of nodes with no outgoing edges."""
        all_from = {e.from_node_id for e in self.edges}
        return [nid for nid in self.nodes if nid not in all_from]

    def validate(self) -> list[str]:
        """Validate the workflow structure. Returns a list of error messages."""
        errors: list[str] = []
        if not self.nodes:
            errors.append("Workflow must have at least one node.")
        if self.start_node_id and self.start_node_id not in self.nodes:
            errors.append(f"start_node_id '{self.start_node_id}' not in nodes.")
        for edge in self.edges:
            if edge.from_node_id not in self.nodes:
                errors.append(
                    f"Edge '{edge.from_node_id}->{edge.to_node_id}': "
                    f"from_node_id not in nodes."
                )
            if edge.to_node_id not in self.nodes:
                errors.append(
                    f"Edge '{edge.from_node_id}->{edge.to_node_id}': "
                    f"to_node_id not in nodes."
                )
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "start_node_id": self.start_node_id,
            "global_context_schema": self.global_context_schema,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Workflow":
        nodes = {
            nd["id"]: WorkflowNode.from_dict(nd)
            for nd in data.get("nodes", [])
        }
        edges = [WorkflowEdge.from_dict(ed) for ed in data.get("edges", [])]
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            nodes=nodes,
            edges=edges,
            start_node_id=data.get("start_node_id", ""),
            global_context_schema=data.get("global_context_schema"),
        )


# ── Runtime execution state ──


@dataclass
class ReviewRecord:
    """A single review decision for audit trail."""

    node_id: str
    reviewer_type: str  # "human", "sub_agent", "logger"
    approved: bool
    feedback: str = ""
    reviewer_identity: str | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "reviewer_type": self.reviewer_type,
            "approved": self.approved,
            "feedback": self.feedback,
            "reviewer_identity": self.reviewer_identity,
            "timestamp": self.timestamp,
        }


@dataclass
class NodeExecution:
    """Runtime state for a single node during workflow execution."""

    node_id: str
    status: Literal[
        "pending", "active", "reviewing", "completed", "rejected", "skipped"
    ] = "pending"
    output: str | None = None
    review_records: list[ReviewRecord] = field(default_factory=list)
    attempt_count: int = 0
    started_at: float | None = None
    completed_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "status": self.status,
            "output": self.output,
            "review_records": [r.to_dict() for r in self.review_records],
            "attempt_count": self.attempt_count,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


WorkflowRunStatus = Literal[
    "running", "paused_awaiting_review", "completed", "rejected", "failed"
]


@dataclass
class WorkflowRun:
    """A single execution run of a workflow."""

    id: str
    workflow_id: str
    status: WorkflowRunStatus = "running"
    current_node_id: str = ""
    node_executions: dict[str, NodeExecution] = field(default_factory=dict)
    global_context: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None

    def current_execution(self) -> NodeExecution | None:
        if not self.current_node_id:
            return None
        return self.node_executions.get(self.current_node_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "status": self.status,
            "current_node_id": self.current_node_id,
            "node_executions": {
                nid: ne.to_dict() for nid, ne in self.node_executions.items()
            },
            "global_context": self.global_context,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


@dataclass
class ReviewDecision:
    """Outcome of a review.

    Attributes:
        approved: Whether the node output passed review.
        feedback: Feedback for the agent if not approved.
        next_node_id: Override the default next node.
        modified_output: Reviewer-corrected output (optional).
    """

    approved: bool
    feedback: str = ""
    next_node_id: str | None = None
    modified_output: str | None = None
