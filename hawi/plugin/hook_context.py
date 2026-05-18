from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from hawi.tool.types import ToolResult

if TYPE_CHECKING:
    from hawi.tool.types import AgentTool
    from hawi.models.model import Model
    from hawi.models.message import ContentPart
    from hawi.agent.context import AgentContext
    from hawi.review import RuntimeReviewBroker


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

    context: AgentContext | None = None
    """Agent conversation context available to runtime hooks."""

    review: RuntimeReviewBroker | None = None
    """Runtime review broker for blocking human-in-the-loop decisions."""

    metadata: dict[str, Any] | None = None
    """Optional framework metadata for hook implementations."""

    duration_ms: float | None = None
    """Execution duration in milliseconds (after-* hooks only)."""

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
    action: Literal["skip", "abort", "replace_model", "reinvoke", "restart_turn"]
    tool_result: ToolResult | None = None
    """Synthetic tool result used when action == 'skip'."""
    reason: str = ""
    """Human-readable reason used when action == 'abort'."""
    model: Model | None = None
    """Replacement model used when action == 'replace_model'."""
    message: str | list[ContentPart] | None = None
    """Message to inject used when action == 'reinvoke'."""

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

    @staticmethod
    def replace_model(model: Model) -> HookResult:
        """Return from ``before_model_call`` to replace the model for this call.

        The agent will use ``model`` instead of the configured model for the
        current model invocation only.
        """
        return HookResult(action="replace_model", model=model)

    @staticmethod
    def reinvoke(message: str | list[ContentPart]) -> HookResult:
        """Return from any hook to inject a message and re-drive the agent.

        The provided ``message`` is appended to the context, the current run
        stops with ``stop_reason == 'hook_reinvoke'``, and ``arun()`` is called
        with the new message.
        """
        return HookResult(action="reinvoke", message=message)

    @staticmethod
    def restart_turn() -> HookResult:
        """Return from ``before_model_call`` to skip the model call entirely.

        The agent will skip the current model invocation and continue to the
        next loop iteration.
        """
        return HookResult(action="restart_turn")
