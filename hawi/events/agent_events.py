"""
Agent events for Hawi Event System.

Agent events are produced by Agent implementations.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import field_serializer

from hawi.errors import AgentError
from hawi.model.message import ContentPart, TokenUsage

from .event import Event


class AgentRunStartEvent(Event):
    """Agent 开始执行"""
    run_id: str
    message_preview: str | None = None

    @classmethod
    def create(
        cls,
        run_id: str,
        message_preview: str | None = None,
    ) -> AgentRunStartEvent:
        return cls(
            type='agent.run_start',
            source='agent',
            run_id=run_id,
            message_preview=message_preview,
        )


class AgentRunStopEvent(Event):
    """Agent 执行结束"""
    run_id: str
    stop_reason: str
    duration_ms: float
    usage: "TokenUsage | None" = None

    @classmethod
    def create(
        cls,
        run_id: str,
        stop_reason: str,
        duration_ms: float,
        usage: "TokenUsage | None" = None,
    ) -> AgentRunStopEvent:
        return cls(
            type="agent.run_stop",
            source="agent",
            run_id=run_id,
            stop_reason=stop_reason,
            duration_ms=duration_ms,
            usage=usage,
        )


class AgentToolCallEvent(Event):
    """Agent 发起工具调用"""
    run_id: str
    tool_name: str
    arguments: dict[str, Any]
    tool_call_id: str

    @classmethod
    def create(
        cls,
        run_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        tool_call_id: str,
    ) -> AgentToolCallEvent:
        return cls(
            type="agent.tool_call",
            source="agent",
            run_id=run_id,
            tool_name=tool_name,
            arguments=arguments,
            tool_call_id=tool_call_id,
        )


class AgentToolResultEvent(Event):
    """Agent 收到工具结果"""
    run_id: str
    tool_name: str
    tool_call_id: str
    success: bool
    result_preview: str
    duration_ms: float
    arguments: dict[str, Any] | None = None

    @classmethod
    def create(
        cls,
        run_id: str,
        tool_name: str,
        tool_call_id: str,
        success: bool,
        result_preview: str,
        duration_ms: float,
        arguments: dict[str, Any] | None = None,
    ) -> AgentToolResultEvent:
        return cls(
            type="agent.tool_result",
            source="agent",
            run_id=run_id,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            success=success,
            result_preview=result_preview,
            duration_ms=duration_ms,
            arguments=arguments or {},
        )


class AgentMessageAddedEvent(Event):
    """消息被添加到上下文"""
    run_id: str
    role: Literal["user", "assistant", "tool"]
    content: list[ContentPart]
    message_preview: str

    @classmethod
    def create(
        cls,
        run_id: str,
        role: Literal["user", "assistant", "tool"],
        content: list[ContentPart],
        message_preview: str,
    ) -> AgentMessageAddedEvent:
        return cls(
            type="agent.message_added",
            source="agent",
            run_id=run_id,
            role=role,
            content=content,
            message_preview=message_preview,
        )


class AgentErrorEvent(Event):
    run_id: str
    error: "AgentError"

    @classmethod
    def create(cls, run_id: str, error: "AgentError"):
        return cls(
            type='agent.error',
            source='agent',
            run_id=run_id,
            error=error,
        )

    @field_serializer('error')
    def serialize_error(self, error: AgentError) -> dict[str, Any]:
        """将 AgentError 序列化为可 JSON 序列化的字典"""
        return {
            'type': error.error_type if hasattr(error, 'error_type') else 'unknown',
            'message': str(error),
            'class': error.__class__.__name__,
        }
