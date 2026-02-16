"""Agent execution results.

Non-streaming return values from agent runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hawi.agent.message import Message, TokenUsage
from hawi.tool.types import ToolResult


@dataclass
class ToolCallRecord:
    """Record of a single tool call execution.

    Attributes:
        tool_name: Name of the tool
        arguments: Arguments passed to the tool
        result: Execution result
        duration_ms: Execution duration in milliseconds
        tool_call_id: Unique identifier for this tool call
    """

    tool_name: str
    arguments: dict[str, Any]
    result: ToolResult
    duration_ms: float
    tool_call_id: str


@dataclass
class AgentRunResult:
    """Result of an agent execution.

    Contains complete execution state and history.
    Similar to Strands' AgentResult but adapted for Hawi.

    Attributes:
        stop_reason: Why execution stopped
                     ("end_turn", "tool_use", "max_iterations", "error", "user_interrupt")
        messages: Complete conversation history
        response: Final assistant message (last message in history, if assistant)
        usage: Token usage statistics
        tool_calls: Record of all tool calls made
        error: Error message if stop_reason is "error"
    """

    stop_reason: str
    messages: list[Message] = field(default_factory=list)
    response: Message | None = None
    usage: TokenUsage | None = None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    error: str | None = None

    def __str__(self) -> str:
        """Return string representation of the result.

        Returns:
            Text content from the final response, or error message
        """
        if self.error:
            return f"Error: {self.error}"

        if not self.response:
            return ""

        # Extract text from response content
        content = self.response.get("content", [])
        texts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(part.get("text", ""))

        return "\n".join(texts)

    @property
    def text(self) -> str:
        """Get response text content.

        Returns:
            Concatenated text from response message
        """
        return str(self)

    @property
    def reasoning_text(self) -> str:
        """Get response reasoning/thinking content.

        Returns:
            Concatenated reasoning content from response message
        """
        if not self.response:
            return ""

        # Extract reasoning from response content
        content = self.response.get("content", [])
        reasoning_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "reasoning":
                reasoning_parts.append(part.get("reasoning", ""))

        return "\n".join(reasoning_parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary.

        Returns:
            Serializable dictionary representation
        """
        return {
            "stop_reason": self.stop_reason,
            "messages": self.messages,
            "response": self.response,
            "usage": self.usage.model_dump() if self.usage else None,
            "tool_calls": [
                {
                    "tool_name": tc.tool_name,
                    "arguments": tc.arguments,
                    "result": {
                        "success": tc.result.success,
                        "output": tc.result.output,
                    },
                    "duration_ms": tc.duration_ms,
                    "tool_call_id": tc.tool_call_id,
                }
                for tc in self.tool_calls
            ],
            "error": self.error,
        }
