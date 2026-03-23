from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from hawi.tool.types import ToolResult

if TYPE_CHECKING:
    from hawi.tool.types import AgentTool
    from hawi.models import TokenUsage


@dataclass(frozen=True)
class HookContext:
    """Runtime context passed to every hook call.

    Fields are populated based on the hook type; unused fields default to None.
    """
    run_id: str
    """Unique ID for this arun() invocation."""

    iteration: int
    """Current loop iteration (0 for session/conversation-level hooks)."""

    tool_call_id: str | None = None
    """Tool call ID (tool hooks only)."""

    tool: AgentTool | None = None
    """Tool object (tool hooks only). Carries metadata beyond just the name."""

    duration_ms: float | None = None
    """Execution duration in milliseconds (after-* hooks only)."""

    usage: TokenUsage | None = None
    """Token usage for this call or cumulative usage (model/conversation hooks)."""

    stop_reason: str | None = None
    """Stop reason (after_model_call, after_conversation)."""

    error: Exception | None = None
    """Error that occurred, if any (after_conversation, after_session)."""


@dataclass(frozen=True)
class HookResult:
    """Return value from a hook to signal a control action.

    Return ``None`` from a hook to continue normal execution.
    Return a ``HookResult`` to signal a control action.

    When a hook in a chain returns a ``HookResult``, the chain stops and the
    result is processed by the agent immediately.
    """
    action: Literal["skip", "abort"]
    tool_result: ToolResult | None = None
    """Synthetic tool result used when action == 'skip'."""
    reason: str = ""
    """Human-readable reason used when action == 'abort'."""

    @staticmethod
    def skip(result: ToolResult) -> HookResult:
        """Return from ``before_tool_calling`` to skip tool execution.

        The provided ``result`` is used as the tool's output without running
        the actual tool.
        """
        return HookResult(action="skip", tool_result=result)

    @staticmethod
    def abort(reason: str = "") -> HookResult:
        """Return from any hook to terminate the agent loop early.

        The agent run will stop with ``stop_reason == 'hook_abort'``.
        """
        return HookResult(action="abort", reason=reason)
