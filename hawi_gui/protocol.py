"""Hawi GUI Protocol — inter-thread message dataclasses.

UI → Scheduler via cmd_queue.
Scheduler → UI via ui_queue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

QueueKind = Literal["normal", "high_prio", "urgent"]

# ─── UI → Scheduler ──────────────────────────────────────────────────────────

@dataclass
class CmdEnqueue:
    """Send a user message to the scheduler."""
    content: str
    queue: QueueKind


@dataclass
class CmdClearContext:
    """Clear the agent's conversation history."""


@dataclass
class CmdClearQueue:
    """Clear one or all message queues."""
    queue: QueueKind | Literal["all"]


@dataclass
class CmdStop:
    """Shut down the scheduler thread."""


@dataclass
class CmdSwitchModel:
    """Hot-switch to a different model factory."""
    factory_name: str


# ─── Scheduler → UI ──────────────────────────────────────────────────────────

@dataclass
class UiReady:
    """Scheduler initialized and ready for messages."""
    factory_name: str


@dataclass
class UiStatusUpdate:
    """Periodic status: scheduler state, agent state, queue lengths."""
    scheduler_state: str
    agent_state: str
    queue_lengths: dict[str, int]


@dataclass
class UiRunStart:
    """An agent run has started."""
    run_id: str
    user_content: str
    queue_kind: QueueKind


@dataclass
class UiTextDelta:
    """Streaming text chunk from the model."""
    delta: str
    run_id: str


@dataclass
class UiRunStop:
    """An agent run has completed."""
    run_id: str
    stop_reason: str
    duration_ms: float


@dataclass
class UiToolCall:
    """A tool was invoked."""
    tool_name: str
    tool_call_id: str
    arguments: dict
    run_id: str


@dataclass
class UiToolResult:
    """A tool execution completed."""
    tool_call_id: str
    tool_name: str
    success: bool
    output: str
    duration_ms: float
    run_id: str


@dataclass
class UiInterrupt:
    """Agent execution was interrupted."""
    reason: str


@dataclass
class UiError:
    """An error occurred."""
    message: str
