"""Hawi GUI Protocol — inter-thread message dataclasses.

UI → Scheduler via cmd_queue.
Scheduler → UI via ui_queue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

QueueKind = Literal["normal", "high_prio", "urgent"]
PluginConfigs = dict[str, dict[str, Any]]
DEFAULT_SYSTEM_PROMPT = "你是Hawi，一个通用agent"

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
class CmdInterrupt:
    """Interrupt current agent execution without stopping scheduler thread."""
    reason: str = "user"


@dataclass
class CmdSwitchModel:
    """Hot-switch to a different model config."""
    model_name: str


@dataclass
class CmdApplyPlugins:
    """Apply plugin selection and plugin configs.

    Notes:
        The scheduler only applies this while idle.
    """
    selected_plugins: list[str]
    plugin_configs: PluginConfigs


@dataclass
class CmdSetSystemPrompt:
    """Update the agent system prompt for subsequent runs."""
    system_prompt: str


# ─── Scheduler → UI ──────────────────────────────────────────────────────────

@dataclass
class UiReady:
    """Scheduler initialized and ready for messages."""
    model_name: str
    selected_plugins: list[str]
    plugin_configs: PluginConfigs


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
class UiThinkingDelta:
    """Streaming thinking/reasoning chunk from the model."""
    delta: str
    run_id: str


@dataclass
class UiRunStop:
    """An agent run has completed."""
    run_id: str
    stop_reason: str
    duration_ms: float


@dataclass
class UiToolCallStart:
    """A tool call stream has started."""
    tool_name: str
    tool_call_id: str
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


@dataclass
class UiModelMetadata:
    """Model metadata (token usage, latency)."""
    run_id: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float | None


@dataclass
class UiModelRetry:
    """Model retry information."""
    run_id: str
    attempt: int
    max_retries: int
    error_type: str
    error_message: str


@dataclass
class UiToolCallDelta:
    """Tool call arguments streaming delta."""
    run_id: str
    tool_call_id: str
    delta: str


@dataclass
class UiToolCallStop:
    """Tool call arguments stream has completed."""
    run_id: str
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]


@dataclass
class UiAgentInterrupt:
    """Agent was interrupted."""
    run_id: str
    interrupt_type: str  # "user" | "scheduler" | "error"


@dataclass
class UiDebugInfo:
    """Debug information (stream start/stop, enqueue/dequeue, etc)."""
    message: str


@dataclass
class UiPluginsApplied:
    """Result of applying plugin configuration."""
    success: bool
    message: str
    selected_plugins: list[str]
    plugin_configs: PluginConfigs
