"""AgentContext implementation for HawiAgent.

Provides conversation state management and request preparation.
"""

from __future__ import annotations

import json
import math
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from hawi.models.message import (
    CachePoint,
    ContentPart,
    Message,
    MessageRequest,
    ToolDefinition,
    ToolCallPart,
    ToolResultPart,
    get_content_cache_point,
    normalize_cache_point,
)
from hawi.tool.types import PendingToolCall

if TYPE_CHECKING:
    from .agent import HawiAgent


CONTEXT_COMPACTION_PROMPT = (
    "You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff "
    "summary for another LLM that will resume the task. Include:\n"
    "- Current progress and key decisions made\n"
    "- Important context, constraints, or user preferences\n"
    "- What remains to be done (clear next steps)\n"
    "- Any critical data, examples, or references needed to continue\n\n"
    "Be concise, structured, and focused on helping the next LLM seamlessly "
    "continue the work."
)

CONTEXT_COMPACTION_SUMMARY_PREFIX = (
    "Another language model started to solve this problem and produced a "
    "summary of its work. Use this summary to build on the work that has "
    "already been done and avoid duplicating work."
)


def _safe_json_dumps(value: Any) -> str:
    """Serialize arbitrary values for approximate token estimation."""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def estimate_text_tokens(text: str) -> int:
    """Estimate token count from UTF-8 byte length.

    This mirrors the coarse heuristic used by many agent harnesses: roughly
    four bytes per token with a one-token floor for non-empty strings.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text.encode("utf-8")) / 4))


def estimate_content_part_tokens(part: ContentPart) -> int:
    """Estimate token count for one Hawi content part."""
    part_type = part.get("type")
    if part_type in {"cache_point", "cache_control"}:
        return 0
    if part_type == "text":
        return estimate_text_tokens(str(part.get("text", "")))
    if part_type == "reasoning":
        return estimate_text_tokens(str(part.get("reasoning") or ""))
    if part_type == "tool_call":
        return estimate_text_tokens(
            f"{part.get('name', '')} {_safe_json_dumps(part.get('arguments', {}))}"
        )
    if part_type == "tool_result":
        content = part.get("content", "")
        if isinstance(content, list):
            return estimate_content_tokens(content)
        return estimate_text_tokens(str(content))
    if part_type == "steer":
        return estimate_content_tokens(list(part.get("content", [])))
    if part_type == "image":
        # Small fixed cost for the reference itself. Provider-specific image
        # token accounting happens below the adapter layer.
        return 85 + estimate_text_tokens(_safe_json_dumps(part.get("source", {})))
    if part_type in {"document", "audio", "video", "file"}:
        return estimate_text_tokens(_safe_json_dumps(part))
    return estimate_text_tokens(_safe_json_dumps(part))


def estimate_content_tokens(content: list[ContentPart] | Any) -> int:
    """Estimate tokens for a content sequence."""
    if not isinstance(content, list):
        return estimate_text_tokens(str(content))
    return sum(
        estimate_content_part_tokens(cast(ContentPart, part))
        for part in content
        if isinstance(part, dict)
    )


def estimate_message_tokens(message: Message) -> int:
    """Estimate token count for a model-visible message."""
    metadata = message.get("metadata") or {}
    tokens = metadata.get("tokens")
    if isinstance(tokens, int) and tokens >= 0:
        return tokens
    # Small role/name framing overhead plus content.
    return 4 + estimate_text_tokens(str(message.get("role", ""))) + estimate_content_tokens(
        list(message.get("content", []))
    )


def _content_has_cache_point(content: list[ContentPart] | None) -> bool:
    if not content:
        return False
    return any(
        isinstance(part, dict) and get_content_cache_point(part) is not None
        for part in content
    )


def _messages_have_cache_point(messages: list[Message]) -> bool:
    return any(
        _content_has_cache_point(list(message.get("content", [])))
        for message in messages
    )


def _tools_have_cache_point(tools: list[ToolDefinition] | None) -> bool:
    if not tools:
        return False
    return any(tool.get("cache_point") or tool.get("cache_control") for tool in tools)


def _append_cache_point_marker(
    content: list[ContentPart],
    cache_point: CachePoint,
) -> list[ContentPart]:
    result = deepcopy(content)
    result.append({"type": "cache_point", "cache_point": deepcopy(cache_point)})
    return result


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
class ContextCompactionRecord:
    """Audit record for one context compaction operation."""

    summary: str
    replaced_messages: list[Message]
    kept_messages: int
    tokens_before: int
    tokens_after: int
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert the record to a JSON-serializable dictionary."""
        return {
            "summary": self.summary,
            "replaced_messages": self.replaced_messages,
            "kept_messages": self.kept_messages,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ContextUsageSnapshot:
    """Estimated context-window occupancy for the current conversation."""

    used_tokens: int
    max_context_tokens: int | None = None
    usage_ratio: float | None = None
    remaining_tokens: int | None = None
    source: Literal["estimate", "provider_usage"] = "estimate"

    def to_dict(self) -> dict[str, Any]:
        """Convert the snapshot to a JSON-serializable dictionary."""
        return {
            "used_tokens": self.used_tokens,
            "max_context_tokens": self.max_context_tokens,
            "usage_ratio": self.usage_ratio,
            "remaining_tokens": self.remaining_tokens,
            "source": self.source,
        }


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
    cache_point: CachePoint | None = None
    cache_tool_definitions: CachePoint | None = None
    auto_cache_static_prefix: CachePoint | None = None
    context_usage: ContextUsageSnapshot | None = None

    # Historical compaction audit records. These are intentionally kept out of
    # model-visible context and only used for debugging/persistence.
    compaction_records: list[ContextCompactionRecord] = field(
        default_factory=list, repr=False, compare=False
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

    def set_system_prompt(self, content: str | list[ContentPart]) -> None:
        """设置系统提示词。

        Args:
            content: 文本字符串或 ContentPart 列表
        """
        if isinstance(content, str):
            self.system_prompt = [{"type": "text", "text": content}]
        else:
            self.system_prompt = content

    def set_cache_point(
        self,
        cache_point: CachePoint | dict[str, Any] | bool | None = True,
        *,
        ttl: Literal["5m", "1h"] | None = None,
    ) -> None:
        """Set a provider-neutral top-level/automatic prompt cache point."""
        value: Any = cache_point
        if ttl is not None:
            value = dict(cache_point) if isinstance(cache_point, dict) else {"type": "ephemeral"}
            value["ttl"] = ttl
        self.cache_point = normalize_cache_point(value)

    def clear_cache_point(self) -> None:
        """Disable top-level/automatic prompt caching for this context."""
        self.cache_point = None

    def set_tool_cache_point(
        self,
        cache_point: CachePoint | dict[str, Any] | bool | None = True,
        *,
        ttl: Literal["5m", "1h"] | None = None,
    ) -> None:
        """Mark the tool-definition prefix as cacheable for providers that support it."""
        value: Any = cache_point
        if ttl is not None:
            value = dict(cache_point) if isinstance(cache_point, dict) else {"type": "ephemeral"}
            value["ttl"] = ttl
        self.cache_tool_definitions = normalize_cache_point(value)

    def clear_tool_cache_point(self) -> None:
        """Disable tool-definition cache marking for this context."""
        self.cache_tool_definitions = None

    def set_static_prefix_cache_point(
        self,
        cache_point: CachePoint | dict[str, Any] | bool | None = True,
        *,
        ttl: Literal["5m", "1h"] | None = None,
    ) -> None:
        """Automatically mark the static tools/system prefix cacheable."""
        value: Any = cache_point
        if ttl is not None:
            value = (
                dict(cache_point)
                if isinstance(cache_point, dict)
                else {"type": "ephemeral"}
            )
            value["ttl"] = ttl
        self.auto_cache_static_prefix = normalize_cache_point(value)

    def clear_static_prefix_cache_point(self) -> None:
        """Disable automatic static-prefix cache marking for this context."""
        self.auto_cache_static_prefix = None

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
        tools = self._tool_definitions_for_request()
        system = self._system_prompt_for_request(tools)
        return MessageRequest(
            messages=self.messages.copy(),
            system=system,
            cache_point=deepcopy(self.cache_point),
            cache_tool_definitions=deepcopy(self.cache_tool_definitions),
            tools=tools,
        )

    def _system_prompt_for_request(
        self,
        tools: list[ToolDefinition] | None,
    ) -> list[ContentPart] | None:
        """Return system prompt with agent-managed cache metadata applied."""
        system = deepcopy(self.system_prompt) if self.system_prompt else None
        cache_point = self.auto_cache_static_prefix
        if not cache_point:
            return system
        if self.cache_point or self.cache_tool_definitions:
            return system
        if _messages_have_cache_point(self.messages):
            return system
        if _content_has_cache_point(system) or _tools_have_cache_point(tools):
            return system

        if system:
            return _append_cache_point_marker(system, cache_point)
        if tools:
            tools[-1]["cache_point"] = deepcopy(cache_point)
        return system

    def _tool_definitions_for_request(self) -> list[ToolDefinition] | None:
        """Return tool definitions with context-level cache metadata applied."""
        if not self.tool_definitions:
            return None
        tools = deepcopy(self.tool_definitions)
        if self.cache_tool_definitions and not any(
            tool.get("cache_point") or tool.get("cache_control")
            for tool in tools
        ):
            tools[-1]["cache_point"] = deepcopy(self.cache_tool_definitions)
        return tools

    def estimate_tokens(
        self,
        *,
        include_system: bool = True,
        include_tools: bool = True,
    ) -> int:
        """Estimate model-visible input tokens with a lightweight heuristic.

        This intentionally avoids provider-specific tokenizers. If a message
        already has ``metadata.tokens`` set, that value is honored; otherwise
        text is approximated from UTF-8 byte length.
        """
        total = 0
        if include_system and self.system_prompt:
            total += estimate_content_tokens(self.system_prompt)
        if include_tools and self.tool_definitions:
            total += estimate_text_tokens(_safe_json_dumps(self.tool_definitions))
        for message in self.messages:
            total += estimate_message_tokens(message)
        return total

    def usage_snapshot(
        self,
        max_context_tokens: int | None = None,
        *,
        include_system: bool = True,
        include_tools: bool = True,
    ) -> ContextUsageSnapshot:
        """Return estimated context-window usage for UI and observability."""
        used_tokens = self.estimate_tokens(
            include_system=include_system,
            include_tools=include_tools,
        )
        if max_context_tokens is None or max_context_tokens <= 0:
            return ContextUsageSnapshot(used_tokens=used_tokens)
        return ContextUsageSnapshot(
            used_tokens=used_tokens,
            max_context_tokens=max_context_tokens,
            usage_ratio=min(1.0, used_tokens / max_context_tokens),
            remaining_tokens=max(0, max_context_tokens - used_tokens),
        )

    def set_context_usage(self, snapshot: ContextUsageSnapshot | None) -> None:
        """Store the most recent context usage metadata for session restore."""
        self.context_usage = snapshot

    def context_usage_snapshot(self) -> ContextUsageSnapshot | None:
        """Return the last persisted context usage metadata, if any."""
        return self.context_usage

    def add_message(self, message: Message) -> None:
        """Append a message to the conversation.

        Args:
            message: Message to append
        """
        self.messages.append(message)

    def add_user_message(
        self,
        content: str | list[ContentPart],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a user message.

        Args:
            content: Text string or content parts
            metadata: Optional message metadata
        """
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]

        self.messages.append({
            "role": "user",
            "content": content,
            "name": None,
            "metadata": dict(metadata) if metadata else None,
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

    def replace_tool_result_content(
        self,
        tool_call_id: str,
        content: str | list[ContentPart],
        *,
        is_error: bool | None = None,
    ) -> bool:
        """Replace the content of an existing tool result message.

        Returns ``True`` when a matching ``tool_call_id`` is found. This keeps
        the surrounding assistant/tool message structure intact, which matters
        for OpenAI-compatible providers.
        """
        replacement: list[ContentPart]
        if isinstance(content, str):
            replacement = [{"type": "text", "text": content}]
        else:
            replacement = deepcopy(content)

        for message in reversed(self.messages):
            if message["role"] != "tool":
                continue
            for part in message["content"]:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "tool_result"
                    and part.get("tool_call_id") == tool_call_id
                ):
                    part["content"] = replacement
                    if is_error is not None:
                        part["is_error"] = is_error
                    return True
        return False

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

    def compaction_tail_start(self, keep_last: int) -> int:
        """Return a safe start index for the recent suffix to keep.

        The returned boundary is moved left to the nearest user message so a
        compacted history does not begin inside a tool-call exchange.
        """
        if keep_last <= 0:
            desired_start = len(self.messages)
        else:
            desired_start = max(0, len(self.messages) - keep_last)

        if desired_start <= 0:
            return 0
        if desired_start >= len(self.messages):
            return len(self.messages)

        start = desired_start
        while start > 0 and self.messages[start]["role"] != "user":
            start -= 1
        return start

    def compact_with_summary(
        self,
        summary: str,
        *,
        keep_last: int = 8,
        summary_prefix: str = CONTEXT_COMPACTION_SUMMARY_PREFIX,
    ) -> ContextCompactionRecord | None:
        """Replace older history with a summary and keep recent messages.

        Returns ``None`` when there is no safe older span to compact.
        """
        tail_start = self.compaction_tail_start(keep_last)
        if tail_start <= 0:
            return None

        tokens_before = self.estimate_tokens()
        replaced_messages = deepcopy(self.messages[:tail_start])
        kept_tail = deepcopy(self.messages[tail_start:])
        summary_message = self._make_compaction_summary_message(
            summary,
            summary_prefix=summary_prefix,
        )
        self.messages = [summary_message, *kept_tail]
        tokens_after = self.estimate_tokens()

        record = ContextCompactionRecord(
            summary=summary,
            replaced_messages=replaced_messages,
            kept_messages=len(kept_tail),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
        )
        self.compaction_records.append(record)
        return record

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
            cast(ToolCallPart, part)
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
        self.context_usage = None

    def copy(self) -> AgentContext:
        """Create a deep copy of the context.

        Returns:
            New AgentContext with copied state
        """
        copied = AgentContext(
            messages=deepcopy(self.messages),
            tool_definitions=deepcopy(self.tool_definitions) if self.tool_definitions else None,
            system_prompt=deepcopy(self.system_prompt),
            cache_point=deepcopy(self.cache_point),
            cache_tool_definitions=deepcopy(self.cache_tool_definitions),
            auto_cache_static_prefix=deepcopy(self.auto_cache_static_prefix),
            context_usage=deepcopy(self.context_usage),
            compaction_records=deepcopy(self.compaction_records),
        )
        return copied

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

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of the full context state.

        Used by both :py:meth:`save` (for the legacy file API) and the
        SessionManager. Includes ``pending_tool_calls`` so audit state survives
        across restarts.
        """
        return {
            "version": "1.0",
            "saved_at": datetime.now().isoformat(),
            "system_prompt": self.system_prompt,
            "cache_point": self.cache_point,
            "cache_tool_definitions": self.cache_tool_definitions,
            "auto_cache_static_prefix": self.auto_cache_static_prefix,
            "context_usage": (
                self.context_usage.to_dict() if self.context_usage is not None else None
            ),
            "messages": self.messages,
            "compaction_records": [
                record.to_dict() for record in self.compaction_records
            ],
            "pending_tool_calls": [
                {
                    "tool_call_id": p.tool_call_id,
                    "tool_name": p.tool_name,
                    "arguments": p.arguments,
                    "requested_at": p.requested_at,
                }
                for p in self._pending_tool_calls.values()
            ],
        }

    def load_snapshot(self, data: dict[str, Any]) -> None:
        """Restore context state from a dict produced by :py:meth:`snapshot`.

        Tool definitions are intentionally NOT restored (they come from the
        live PluginManager and would otherwise drift). Caller is responsible
        for ensuring the plugin set matches what was active when the snapshot
        was captured.
        """
        version = data.get("version", "1.0")
        if version != "1.0":
            raise ValueError(f"Unsupported context snapshot version: {version}")

        self.messages = data.get("messages", [])
        self.system_prompt = data.get("system_prompt")
        self.cache_point = normalize_cache_point(data.get("cache_point"))
        self.cache_tool_definitions = normalize_cache_point(
            data.get("cache_tool_definitions")
        )
        self.auto_cache_static_prefix = normalize_cache_point(
            data.get("auto_cache_static_prefix")
        )
        self.context_usage = self._context_usage_from_snapshot(
            data.get("context_usage")
        )
        self.compaction_records = [
            ContextCompactionRecord(
                summary=record.get("summary", ""),
                replaced_messages=record.get("replaced_messages", []),
                kept_messages=record.get("kept_messages", 0),
                tokens_before=record.get("tokens_before", 0),
                tokens_after=record.get("tokens_after", 0),
                created_at=record.get("created_at", time.time()),
            )
            for record in data.get("compaction_records", [])
        ]

        self._pending_tool_calls = {
            entry["tool_call_id"]: PendingToolCall(
                tool_call_id=entry["tool_call_id"],
                tool_name=entry["tool_name"],
                arguments=entry.get("arguments", {}),
                requested_at=entry.get("requested_at", time.time()),
            )
            for entry in data.get("pending_tool_calls", [])
        }

    @staticmethod
    def _context_usage_from_snapshot(value: Any) -> ContextUsageSnapshot | None:
        if not isinstance(value, dict):
            return None
        used_tokens = value.get("used_tokens")
        if not isinstance(used_tokens, int):
            return None
        max_context_tokens = value.get("max_context_tokens")
        if not isinstance(max_context_tokens, int):
            max_context_tokens = None
        usage_ratio = value.get("usage_ratio")
        if not isinstance(usage_ratio, (int, float)):
            usage_ratio = None
        remaining_tokens = value.get("remaining_tokens")
        if not isinstance(remaining_tokens, int):
            remaining_tokens = None
        source = value.get("source")
        if source not in {"estimate", "provider_usage"}:
            source = "estimate"
        return ContextUsageSnapshot(
            used_tokens=used_tokens,
            max_context_tokens=max_context_tokens,
            usage_ratio=float(usage_ratio) if usage_ratio is not None else None,
            remaining_tokens=remaining_tokens,
            source=cast(Literal["estimate", "provider_usage"], source),
        )

    def _save_json(self, path: Path) -> None:
        """Save context as JSON for complete state restoration."""
        path.write_text(
            json.dumps(self.snapshot(), ensure_ascii=False, indent=2),
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

        Restores messages, system_prompt, cache-point settings, compaction
        records, and pending audit tool calls. Tool definitions remain unchanged.

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

        self.load_snapshot(data)

    @staticmethod
    def _make_compaction_summary_message(
        summary: str,
        *,
        summary_prefix: str,
    ) -> Message:
        text = summary.strip() or "(no summary available)"
        if summary_prefix:
            text = f"{summary_prefix}\n\n{text}"
        return {
            "role": "user",
            "content": [{"type": "text", "text": text}],
            "name": None,
            "metadata": {
                "source": "context_compaction",
                "summarized": True,
                "timestamp": time.time(),
                "compression_level": 1,
            },
        }

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
                if isinstance(sub_part, dict):
                    steer_lines.extend(
                        self._format_content_part(cast(ContentPart, sub_part))
                    )
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
        elif part_type in {"cache_point", "cache_control"}:
            lines.append("*[Cache point marker]*")
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
