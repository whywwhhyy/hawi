"""AgentContext implementation for HawiAgent.

Provides conversation state management and request preparation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from hawi.models.message import (
    ContentPart,
    Message,
    MessageRequest,
    ToolDefinition,
    ToolCallPart,
    ToolResultPart,
)
from hawi.tool.types import PendingToolCall

if TYPE_CHECKING:
    from .agent import HawiAgent


class ToolCallContext:
    """工具执行时注入的有界 API。

    替代直接暴露 HawiAgent：工具通过此对象访问 agent 内部能力，
    接口清晰、意图明确。

    工具声明需要注入时，在类上设置 context 属性：
        class MyTool(AgentTool):
            context = "ctx"  # 参数名称
            def run(self, ..., ctx: ToolCallContext): ...

    属性:
        context: 对话上下文（消息历史、system prompt、历史操作）
        agent:   完整 agent（sub-agent 编排、工具管理等）
    """

    def __init__(self, agent: HawiAgent) -> None:
        self._agent = agent

    @property
    def context(self) -> AgentContext:
        """直接访问对话上下文（消息历史、system prompt）。"""
        return self._agent.context

    @property
    def agent(self) -> HawiAgent:
        """访问完整 agent（sub-agent 编排、动态工具注册等）。"""
        return self._agent


@dataclass
class RecoveredToolResult:
    """Synthetic tool result inserted to keep provider message history valid."""

    tool_call_id: str
    tool_name: str
    content: str


@dataclass
class AgentContext:
    """Conversation context for agent execution.

    Manages message history, tool definitions, and system prompt.
    Provides methods for context manipulation.

    Attributes:
        messages: Conversation history (不支持 role="system")
        tool_definitions: Available tool definitions for model consumption
        system_prompt: System prompt as list of ContentPart
    """

    messages: list[Message] = field(default_factory=list)
    tool_definitions: list[ToolDefinition] | None = None
    system_prompt: list[ContentPart] | None = None

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

    def prepare_request(self) -> MessageRequest:
        """Build MessageRequest from current context.

        Returns:
            MessageRequest ready for model invocation
        """
        return MessageRequest(
            messages=self.messages.copy(),
            system=self.system_prompt,
            tools=self.tool_definitions,
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
            "metadata": None,
        })

    def add_assistant_message(
        self,
        content: list[ContentPart],
    ) -> None:
        """Add an assistant message.

        Args:
            content: Content parts (text/reasoning/tool_call/etc)
                  Tool calls should already be included as ToolCallPart items in content.
        """
        self.messages.append({
            "role": "assistant",
            "content": content,
            "name": None,
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

        # 将 tool_call_id 嵌入到 ToolResultPart 中
        tool_result_part: ToolResultPart = {
            "type": "tool_result",
            "tool_call_id": tool_call_id,
            "content": content,
            "is_error": is_error,
        }

        self.messages.append({
            "role": "tool",
            "content": [tool_result_part],
            "name": None,
            "metadata": None,
        })

    def add_missing_tool_results(self, content: str) -> list[RecoveredToolResult]:
        """Insert error tool results for assistant tool calls with no response.

        OpenAI-compatible providers require every assistant message containing
        tool_calls to be followed by tool messages for each tool_call_id before
        any later user/assistant message. If an execution is cancelled during a
        tool call, the assistant tool_call may already be in history while its
        tool result is missing. This repairs that history in place.
        """
        recovered: list[RecoveredToolResult] = []
        index = 0

        while index < len(self.messages):
            message = self.messages[index]
            if message["role"] != "assistant":
                index += 1
                continue

            tool_calls = self._tool_call_parts(message)
            if not tool_calls:
                index += 1
                continue

            insert_at = index + 1
            responded_ids: set[str] = set()
            while insert_at < len(self.messages) and self.messages[insert_at]["role"] == "tool":
                responded_ids.update(self._tool_result_ids(self.messages[insert_at]))
                insert_at += 1

            for tool_call in tool_calls:
                tool_call_id = tool_call.get("id") or ""
                if not tool_call_id or tool_call_id in responded_ids:
                    continue
                tool_name = tool_call.get("name") or ""
                self.messages.insert(
                    insert_at,
                    self._make_tool_result_message(tool_call_id, content, is_error=True),
                )
                insert_at += 1
                responded_ids.add(tool_call_id)
                recovered.append(
                    RecoveredToolResult(
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        content=content,
                    )
                )

            index = insert_at

        return recovered

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

    @staticmethod
    def _make_tool_result_message(
        tool_call_id: str,
        content: str,
        *,
        is_error: bool,
    ) -> Message:
        return {
            "role": "tool",
            "content": [
                {
                    "type": "tool_result",
                    "tool_call_id": tool_call_id,
                    "content": [{"type": "text", "text": content}],
                    "is_error": is_error,
                }
            ],
            "name": None,
            "metadata": None,
        }

    @staticmethod
    def _tool_call_parts(message: Message) -> list[ToolCallPart]:
        return [
            part
            for part in message["content"]
            if isinstance(part, dict) and part.get("type") == "tool_call"
        ]

    @staticmethod
    def _tool_result_ids(message: Message) -> set[str]:
        return {
            str(part.get("tool_call_id") or "")
            for part in message["content"]
            if isinstance(part, dict)
            and part.get("type") == "tool_result"
            and part.get("tool_call_id")
        }

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
            tool_definitions=self.tool_definitions.copy() if self.tool_definitions else None,
            system_prompt=self.system_prompt,
        )

    def save(
        self,
        filepath: str | Path,
        format: Literal["markdown", "json"] = "markdown",
    ) -> None:
        """Save the entire context to a file.

        Supports two formats:
        - markdown: Human-readable format with formatted structure (default)
        - json: Complete serialization for state restoration

        Args:
            filepath: Path to the output file
            format: Output format - "markdown" or "json"
        """
        path = Path(filepath)

        if format == "json":
            self._save_json(path)
        else:
            self._save_markdown(path)

    def _save_json(self, path: Path) -> None:
        """Save context as JSON for complete state restoration."""
        data = {
            "version": "1.0",
            "saved_at": datetime.now().isoformat(),
            "system_prompt": self.system_prompt,
            "messages": self.messages,
        }

        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _save_markdown(self, path: Path) -> None:
        """Save context as human-readable markdown."""
        lines: list[str] = []

        # Header
        lines.append("# Agent Context History")
        lines.append(f"\n*Saved at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        lines.append("\n---\n")

        # System Prompt Section
        lines.append("## System Prompt\n")
        if self.system_prompt:
            for part in self.system_prompt:
                lines.extend(self._format_content_part(part))
        else:
            lines.append("*No system prompt set*")
        lines.append("\n---\n")

        # Tools Section
        lines.append("## Available Tools\n")
        if self.tool_definitions:
            lines.append(f"**Total tools:** {len(self.tool_definitions)}\n")
            for tool_def in self.tool_definitions:
                lines.extend(self._format_tool_definition_info(tool_def))
        else:
            lines.append("*No tools available*")
        lines.append("\n---\n")

        # Conversation History Section
        lines.append("## Conversation History\n")
        if self.messages:
            lines.append(f"**Total messages:** {len(self.messages)}\n")
            for i, msg in enumerate(self.messages, 1):
                lines.extend(self._format_message(i, msg))
        else:
            lines.append("*No messages in conversation*")

        # Write to file
        path.write_text("\n".join(lines), encoding="utf-8")

    def load(self, filepath: str | Path) -> None:
        """Load context state from a JSON file into this instance.

        Restores messages and system_prompt from the file.
        Tools and cache_tool_definitions remain unchanged.

        Args:
            filepath: Path to the JSON file

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON is invalid or has unsupported version
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Context file not found: {filepath}")

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in context file: {e}")

        # Validate version
        version = data.get("version", "1.0")
        if version != "1.0":
            raise ValueError(f"Unsupported context version: {version}")

        # Restore state into this instance (preserve tools and cache setting)
        self.messages = data.get("messages", [])
        self.system_prompt = data.get("system_prompt")

    def _format_content_part(self, part: ContentPart) -> list[str]:
        """Format a single ContentPart as markdown lines."""
        lines: list[str] = []
        part_type = part.get("type")

        if part_type == "text":
            lines.append(part.get("text", ""))
        elif part_type == "image":
            source = part.get("source", {})
            url = source.get("url", "")
            detail = source.get("detail", "auto")
            lines.append(f"*[Image: {url} (detail: {detail})]*")
        elif part_type == "document":
            source = part.get("source", {})
            url = source.get("url", "")
            title = part.get("title", "Untitled")
            mime_type = source.get("mime_type", "unknown")
            lines.append(f"*[Document: {title} ({mime_type}) - {url}]*")
        elif part_type == "audio":
            source = part.get("source", {})
            transcript = source.get("transcript", "")
            if transcript:
                lines.append(f"*[Audio: {transcript}]*")
            else:
                lines.append("*[Audio message]*")
        elif part_type == "video":
            source = part.get("source", {})
            url = source.get("url", "")
            fmt = source.get("format", "unknown")
            lines.append(f"*[Video: {url} (format: {fmt})]*")
        elif part_type == "tool_call":
            lines.append(f"\n\n**Tool Call:** `{part.get('name', 'unknown')}`")
            lines.append(f"- ID: `{part.get('id', 'N/A')}`")
            args = part.get("arguments", {})
            if args:
                lines.append("- Arguments:")
                for key, value in args.items():
                    value_str = str(value)
                    if '\n' in value_str:
                        # 有换行符的参数，冒号后换行
                        lines.append(f"  - **{key}**:")
                        lines.append(f"```\n{value_str}\n```")
                    else:
                        # 无换行符的参数，单行显示
                        lines.append(f"  - **{key}**: {value_str}")
        elif part_type == "tool_result":
            content = part.get("content", "")
            is_error = part.get("is_error", False)
            prefix = "**Tool Result**" if not is_error else "**Tool Result (Error)**"
            lines.append(f"{prefix} (ID: `{part.get('tool_call_id', 'N/A')}`):")
            if isinstance(content, list):
                # 多模态内容：递归格式化后用代码块包裹
                content_lines: list[str] = []
                for sub_part in content:
                    content_lines.extend(self._format_content_part(sub_part))
                content_str = "\n".join(content_lines)
                lines.append(f"```\n{content_str}\n```")
            else:
                lines.append(f"```\n{content}\n```")
        elif part_type == "steer":
            tool_call_id = part.get("tool_call_id")
            if tool_call_id:
                lines.append(f"**Steer** (tool_call_id: `{tool_call_id}`):")
            else:
                lines.append("**Steer**:")
            steer_lines: list[str] = []
            for sub_part in part.get("content", []):
                steer_lines.extend(self._format_content_part(sub_part))
            if steer_lines:
                steer_text = "\n".join(steer_lines)
                lines.append(f"```\n{steer_text}\n```")
        elif part_type == "reasoning":
            reasoning = part.get("reasoning", "")
            if reasoning:
                lines.append(f"> **Reasoning:** {reasoning}")
            elif part.get("redacted_content"):
                lines.append("> **[Redacted reasoning content]**")
        elif part_type == "refusal":
            lines.append(f"**Model Refusal:** {part.get('refusal', 'Unknown')}")
        elif part_type == "citation":
            citations = part.get("citations", [])
            lines.append(f"*[Citations: {len(citations)} reference(s)]*")
        elif part_type == "cache_control":
            lines.append("*[Cache control marker]*")
        else:
            lines.append(f"*[Content type: {part_type}]*")

        return lines

    def _format_tool_definition_info(self, tool_def: ToolDefinition) -> list[str]:
        """Format tool definition information as markdown lines."""
        lines: list[str] = []
        lines.append(f"### `{tool_def['name']}`\n")
        lines.append(f"**Description:** {tool_def.get('description', '')}\n")
        schema = tool_def.get('schema')
        if schema:
            try:
                schema_str = json.dumps(
                    schema,
                    ensure_ascii=False,
                    indent=2
                )
                lines.append(f"**Parameters Schema:**\n```json\n{schema_str}\n```\n")
            except (TypeError, ValueError):
                lines.append(f"**Parameters Schema:** {schema}\n")
        return lines

    def _format_message(self, index: int, msg: Message) -> list[str]:
        """Format a message as markdown lines."""
        lines: list[str] = []
        role = msg.get("role", "unknown")
        name = msg.get("name")
        metadata = msg.get("metadata")

        # Message header
        header = f"### Message {index}: **{role.upper()}**"
        if name:
            header += f" (name: `{name}`)"
        lines.append(header)

        # Metadata
        if metadata:
            meta_parts = []
            tokens = metadata.get("tokens")
            if tokens:
                meta_parts.append(f"tokens: {tokens}")
            timestamp = metadata.get("timestamp")
            if timestamp:
                ts = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                meta_parts.append(f"time: {ts}")
            importance = metadata.get("importance")
            if importance:
                meta_parts.append(f"importance: {importance:.2f}")
            source = metadata.get("source")
            if source:
                meta_parts.append(f"source: {source}")
            if metadata.get("summarized"):
                meta_parts.append("summarized: yes")
            if meta_parts:
                lines.append(f"*{', '.join(meta_parts)}*")

        lines.append("")

        # Message content
        content = msg.get("content", [])
        if content:
            for part in content:
                lines.extend(self._format_content_part(part))
        else:
            lines.append("*No content*")

        lines.append("\n---\n")
        return lines
