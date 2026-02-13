"""AgentContext implementation for HawiAgent.

Provides conversation state management and request preparation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from hawi.agent.messages import (
    ContentPart,
    Message,
    MessageRequest,
    TextPart,
    ToolDefinition,
    ToolCallPart,
)
from hawi.tool.types import AgentTool, PendingToolCall

if TYPE_CHECKING:
    from .agent import HawiAgent


@dataclass
class ToolCallContext:
    """Runtime context for tool execution.

    Simple data class providing access to agent runtime information.
    Can be extended with additional fields as needed.

    Attributes:
        agent: Reference to the HawiAgent instance
    """

    agent: HawiAgent


@dataclass
class AgentContext:
    """Conversation context for agent execution.

    Manages message history, tools, and system prompt.
    Provides methods for context manipulation.

    Attributes:
        messages: Conversation history (不支持 role="system")
        tools: Available tools (AgentTool instances)
        system_prompt: System prompt as list of ContentPart
        cache_tool_definitions: Whether to cache converted ToolDefinitions
    """

    messages: list[Message] = field(default_factory=list)
    tools: list[AgentTool] = field(default_factory=list)
    system_prompt: list[ContentPart] | None = None
    cache_tool_definitions: bool = True

    # Internal cache for ToolDefinition conversion
    _cached_tool_definitions: list[ToolDefinition] | None = field(
        default=None, repr=False, compare=False
    )

    # Pending tool calls for audit mechanism
    _pending_tool_calls: dict[str, PendingToolCall] = field(
        default_factory=dict, repr=False, compare=False
    )

    # Tool call context for runtime injection
    tool_call_context: ToolCallContext | None = field(
        default=None, repr=False, compare=False
    )

    def _add_pending_tool_call(self, tool_call_id: str, tool_name: str, arguments: dict[str, Any]) -> PendingToolCall:
        """Add a tool call to pending queue for audit (internal use)."""
        pending = PendingToolCall(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments=arguments,
        )
        self._pending_tool_calls[tool_call_id] = pending
        return pending

    def get_pending_tool_calls(self) -> list[PendingToolCall]:
        """Get all pending tool calls.

        Returns:
            List of pending tool calls (empty list if none)
        """
        return list(self._pending_tool_calls.values())

    def audit_pending_tool_calls(
        self,
        approve: list[str] | None = None,
        reject: list[str] | None = None,
    ) -> tuple[list[PendingToolCall], list[PendingToolCall]]:
        """Audit pending tool calls by approving or rejecting them.

        Args:
            approve: List of tool_call_ids to approve
            reject: List of tool_call_ids to reject

        Returns:
            Tuple of (approved_calls, rejected_calls)
        """
        approved: list[PendingToolCall] = []
        rejected: list[PendingToolCall] = []

        for tool_call_id in approve or []:
            if tool_call_id in self._pending_tool_calls:
                approved.append(self._pending_tool_calls.pop(tool_call_id))

        for tool_call_id in reject or []:
            if tool_call_id in self._pending_tool_calls:
                rejected.append(self._pending_tool_calls.pop(tool_call_id))

        return approved, rejected

    def clear_pending_tool_calls(self) -> None:
        """Clear all pending tool calls."""
        self._pending_tool_calls.clear()

    def _convert_tools_to_definitions(self) -> list[ToolDefinition]:
        """Convert AgentTool instances to ToolDefinition format.

        Returns:
            List of ToolDefinition for model consumption
        """
        return [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "schema": tool.parameters_schema,
            }
            for tool in self.tools
        ]

    def get_tool_definitions(self) -> list[ToolDefinition]:
        """Get ToolDefinition list for model request.

        Uses cache if cache_tool_definitions is True.

        Returns:
            List of ToolDefinition
        """
        if self.cache_tool_definitions:
            if self._cached_tool_definitions is None:
                self._cached_tool_definitions = self._convert_tools_to_definitions()
            return self._cached_tool_definitions
        return self._convert_tools_to_definitions()

    def invalidate_tool_cache(self) -> None:
        """Invalidate the tool definition cache.

        Call this after modifying tools list.
        """
        self._cached_tool_definitions = None

    def add_tool(self, tool: AgentTool) -> None:
        """Add a tool to the context.

        Args:
            tool: Tool to add
        """
        self.tools.append(tool)
        if self.cache_tool_definitions:
            self.invalidate_tool_cache()

    def remove_tool(self, tool_name: str) -> bool:
        """Remove a tool by name.

        Args:
            tool_name: Name of tool to remove

        Returns:
            True if tool was found and removed
        """
        for i, tool in enumerate(self.tools):
            if tool.name == tool_name:
                del self.tools[i]
                if self.cache_tool_definitions:
                    self.invalidate_tool_cache()
                return True
        return False

    def set_system_prompt(self, content: str | list[ContentPart]) -> None:
        """设置系统提示词。

        Args:
            content: 文本字符串或 ContentPart 列表
        """
        if isinstance(content, str):
            self.system_prompt = [{"type": "text", "text": content}]
        else:
            self.system_prompt = content

    def get_system_prompt(self) -> list[ContentPart] | None:
        """获取系统提示词。

        Returns:
            ContentPart 列表或 None
        """
        return self.system_prompt

    def get_tool(self, name: str) -> AgentTool | None:
        """Get a tool by name.

        Args:
            name: Tool name

        Returns:
            AgentTool if found, None otherwise
        """
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def prepare_request(self) -> MessageRequest:
        """Build MessageRequest from current context.

        Returns:
            MessageRequest ready for model invocation
        """
        return MessageRequest(
            messages=self.messages.copy(),
            system=self.system_prompt,
            tools=self.get_tool_definitions() if self.tools else None,
        )

    def add_message(self, message: Message) -> None:
        """Append a message to the conversation.

        Args:
            message: Message to append
        """
        self.messages.append(message)

    def add_user_message(self, content: str | list[ContentPart]) -> None:
        """Add a user message.

        Args:
            content: Text string or content parts
        """
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]

        self.messages.append({
            "role": "user",
            "content": content,
            "name": None,
            "tool_calls": None,
            "tool_call_id": None,
            "metadata": None,
        })

    def add_assistant_message(
        self,
        content: list[ContentPart],
        tool_calls: list[ToolCallPart] | None = None,
    ) -> None:
        """Add an assistant message.

        Args:
            content: Content parts
            tool_calls: Optional tool calls
        """
        self.messages.append({
            "role": "assistant",
            "content": content,
            "name": None,
            "tool_calls": tool_calls,
            "tool_call_id": None,
            "metadata": None,
        })

    def add_tool_result(
        self,
        tool_call_id: str,
        content: str | list[ContentPart],
        is_error: bool = False,
    ) -> None:
        """Add a tool result message.

        Args:
            tool_call_id: ID of the tool call
            content: Result content
            is_error: Whether this is an error result
        """
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]

        self.messages.append({
            "role": "tool",
            "content": content,
            "name": None,
            "tool_calls": None,
            "tool_call_id": tool_call_id,
            "metadata": None,
        })

    def truncate(self, keep_last: int) -> None:
        """Keep only the last N messages.

        Args:
            keep_last: Number of recent messages to keep
        """
        if keep_last < 0:
            return
        self.messages = self.messages[-keep_last:]

    def inject(self, message: Message, position: int = -1) -> None:
        """Insert a message at specified position.

        Args:
            message: Message to insert
            position: Position index (-1 for append)
        """
        if position == -1:
            self.messages.append(message)
        else:
            self.messages.insert(position, message)

    def collapse(self, start: int, end: int, summary: str) -> None:
        """Collapse a range of messages into a summary.

        Replaces messages[start:end] with a single system message
        containing the summary.

        Args:
            start: Start index (inclusive)
            end: End index (exclusive)
            summary: Summary text to replace the range
        """
        if start < 0 or end > len(self.messages) or start >= end:
            return

        # Remove the range
        del self.messages[start:end]

        # Insert summary as a system message (or user message if at start)
        summary_message: Message = {
            "role": "user",
            "content": [{"type": "text", "text": f"[Previous conversation summary: {summary}]"}],
            "name": None,
            "tool_calls": None,
            "tool_call_id": None,
            "metadata": None,
        }
        self.messages.insert(start, summary_message)

    def clear(self) -> None:
        """Clear all messages (preserve tools and system_prompt)."""
        self.messages.clear()

    def copy(self) -> AgentContext:
        """Create a deep copy of the context.

        Returns:
            New AgentContext with copied state
        """
        return AgentContext(
            messages=self.messages.copy(),
            tools=self.tools.copy(),
            system_prompt=self.system_prompt,
            cache_tool_definitions=self.cache_tool_definitions,
        )
