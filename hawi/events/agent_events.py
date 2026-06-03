"""
Agent events for Hawi Event System.

Agent events are produced by Agent implementations.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import field_serializer

from hawi.errors import AgentError
from hawi.models.message import ContentPart, TokenUsage
from hawi.tool.types import ToolResult

from .event import Event


class AgentRunStartEvent(Event):
    """Agent 开始执行"""
    run_id: str

    @classmethod
    def create(
        cls,
        run_id: str,
    ) -> AgentRunStartEvent:
        return cls(
            type='agent.run_start',
            source='agent',
            run_id=run_id,
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


class AgentToolResultPartEvent(Event):
    """Agent 收到工具结果的分片（用于异步生成器工具）

    当工具函数是异步生成器时，每产生一个结果片段就发送一次此事件。
    通过 tool_call_id 关联到之前的 AgentToolCallEvent。
    """
    run_id: str
    tool_call_id: str
    part: str           # 当前片段内容
    part_index: int     # 片段序号（从0开始）
    is_final: bool      # 是否是最后一个片段

    @classmethod
    def create(
        cls,
        run_id: str,
        tool_call_id: str,
        part: str,
        part_index: int,
        is_final: bool,
    ) -> AgentToolResultPartEvent:
        return cls(
            type="agent.tool_result_part",
            source="agent",
            run_id=run_id,
            tool_call_id=tool_call_id,
            part=part,
            part_index=part_index,
            is_final=is_final,
        )


class AgentToolResultEvent(Event):
    """Agent 收到工具结果

    通过 tool_call_id 关联到之前的 AgentToolCallEvent。
    工具名称和参数信息可以通过 AgentToolCallEvent 或 Printer 缓存获取。
    """
    run_id: str
    tool_call_id: str
    result_preview: str
    result: "ToolResult | None" = None
    success: bool
    duration_ms: float
    context_message_id: str | None = None
    interrupted: bool = False

    @classmethod
    def create(
        cls,
        run_id: str,
        tool_call_id: str,
        success: bool,
        result_preview: str,
        duration_ms: float,
        result_obj: "ToolResult | None" = None,
        context_message_id: str | None = None,
        interrupted: bool = False,
    ) -> AgentToolResultEvent:
        return cls(
            type="agent.tool_result",
            source="agent",
            run_id=run_id,
            tool_call_id=tool_call_id,
            result_preview=result_preview,
            result=result_obj,
            success=success,
            duration_ms=duration_ms,
            context_message_id=context_message_id,
            interrupted=interrupted,
        )

    @field_serializer('result')
    def serialize_result(self, result: "ToolResult | None") -> dict[str, Any] | None:
        """将 ToolResult 序列化为可 JSON 序列化的字典"""
        if result is None:
            return None
        return {
            'success': result.success,
            'output': result.output,
            'error': result.error,
        }


class AgentMessageAddedEvent(Event):
    """消息被添加到上下文"""
    run_id: str
    role: Literal["user", "assistant", "tool"]
    content: list[ContentPart]
    metadata: dict[str, Any] | None = None
    context_message_id: str | None = None

    @classmethod
    def create(
        cls,
        run_id: str,
        role: Literal["user", "assistant", "tool"],
        content: list[ContentPart],
        metadata: dict[str, Any] | None = None,
        context_message_id: str | None = None,
    ) -> AgentMessageAddedEvent:
        return cls(
            type="agent.message_added",
            source="agent",
            run_id=run_id,
            role=role,
            content=content,
            metadata=metadata,
            context_message_id=context_message_id,
        )


class AgentSystemPromptEvent(Event):
    """System prompt 可见内容被送入模型上下文"""

    run_id: str
    content: list[ContentPart]
    origin: str = "model_input"
    plugin_id: str | None = None
    plugin_name: str | None = None
    plugin_role: str = "framework"
    injection_name: str | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def create(
        cls,
        run_id: str,
        content: list[ContentPart],
        *,
        origin: str = "model_input",
        plugin_id: str | None = None,
        plugin_name: str | None = None,
        plugin_role: str = "framework",
        injection_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentSystemPromptEvent:
        return cls(
            type="agent.system_prompt",
            source="agent",
            run_id=run_id,
            content=content,
            origin=origin,
            plugin_id=plugin_id,
            plugin_name=plugin_name,
            plugin_role=plugin_role,
            injection_name=injection_name,
            metadata=metadata,
        )


class AgentContextInjectedEvent(Event):
    """Hook/plugin 向对话上下文注入了模型可见消息"""

    run_id: str
    role: Literal["user", "assistant", "tool", "system", "error"]
    content: list[ContentPart]
    hook_type: str | None = None
    position: int | None = None
    plugin_id: str | None = None
    plugin_name: str | None = None
    plugin_role: str = "framework"
    injection_name: str | None = None
    metadata: dict[str, Any] | None = None
    context_message_id: str | None = None
    merge_target: Literal["user_message"] | None = None
    merge_position: Literal["before", "after"] | None = None
    target_message_id: str | None = None
    target_message_index: int | None = None
    target_context_message_id: str | None = None

    @classmethod
    def create(
        cls,
        run_id: str,
        role: Literal["user", "assistant", "tool", "system", "error"],
        content: list[ContentPart],
        *,
        hook_type: str | None = None,
        position: int | None = None,
        plugin_id: str | None = None,
        plugin_name: str | None = None,
        plugin_role: str = "framework",
        injection_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        context_message_id: str | None = None,
        merge_target: Literal["user_message"] | None = None,
        merge_position: Literal["before", "after"] | None = None,
        target_message_id: str | None = None,
        target_message_index: int | None = None,
        target_context_message_id: str | None = None,
    ) -> AgentContextInjectedEvent:
        return cls(
            type="agent.context_injected",
            source="agent",
            run_id=run_id,
            role=role,
            content=content,
            hook_type=hook_type,
            position=position,
            plugin_id=plugin_id,
            plugin_name=plugin_name,
            plugin_role=plugin_role,
            injection_name=injection_name,
            metadata=metadata,
            context_message_id=context_message_id,
            merge_target=merge_target,
            merge_position=merge_position,
            target_message_id=target_message_id,
            target_message_index=target_message_index,
            target_context_message_id=target_context_message_id,
        )


class AgentToolRuntimeContextInjectedEvent(Event):
    """Hawi 运行时上下文被注入到工具实现参数中"""

    run_id: str
    tool_name: str
    tool_call_id: str
    parameter_name: str
    plugin_id: str | None = None
    plugin_name: str | None = None
    plugin_role: str = "tool_owner"
    injection_name: str | None = None

    @classmethod
    def create(
        cls,
        run_id: str,
        tool_name: str,
        tool_call_id: str,
        parameter_name: str,
        *,
        plugin_id: str | None = None,
        plugin_name: str | None = None,
        plugin_role: str = "tool_owner",
        injection_name: str | None = None,
    ) -> AgentToolRuntimeContextInjectedEvent:
        return cls(
            type="agent.tool_runtime_context_injected",
            source="agent",
            run_id=run_id,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            parameter_name=parameter_name,
            plugin_id=plugin_id,
            plugin_name=plugin_name,
            plugin_role=plugin_role,
            injection_name=injection_name,
        )


class AgentCompactStartEvent(Event):
    """Agent 开始压缩上下文"""
    run_id: str | None = None
    mode: Literal["manual", "auto"]
    keep_last_messages: int
    tokens_before: int
    message_count_before: int

    @classmethod
    def create(
        cls,
        *,
        run_id: str | None,
        mode: Literal["manual", "auto"],
        keep_last_messages: int,
        tokens_before: int,
        message_count_before: int,
    ) -> AgentCompactStartEvent:
        return cls(
            type="agent.compact_start",
            source="agent",
            run_id=run_id,
            mode=mode,
            keep_last_messages=keep_last_messages,
            tokens_before=tokens_before,
            message_count_before=message_count_before,
        )


class AgentCompactStopEvent(Event):
    """Agent 结束压缩上下文"""
    run_id: str | None = None
    mode: Literal["manual", "auto"]
    status: Literal["success", "skipped", "error"]
    duration_ms: float
    tokens_before: int | None = None
    tokens_after: int | None = None
    message_count_before: int | None = None
    message_count_after: int | None = None
    replaced_message_count: int | None = None
    kept_message_count: int | None = None
    error: str | None = None

    @classmethod
    def create(
        cls,
        *,
        run_id: str | None,
        mode: Literal["manual", "auto"],
        status: Literal["success", "skipped", "error"],
        duration_ms: float,
        tokens_before: int | None = None,
        tokens_after: int | None = None,
        message_count_before: int | None = None,
        message_count_after: int | None = None,
        replaced_message_count: int | None = None,
        kept_message_count: int | None = None,
        error: str | None = None,
    ) -> AgentCompactStopEvent:
        return cls(
            type="agent.compact_stop",
            source="agent",
            run_id=run_id,
            mode=mode,
            status=status,
            duration_ms=duration_ms,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            message_count_before=message_count_before,
            message_count_after=message_count_after,
            replaced_message_count=replaced_message_count,
            kept_message_count=kept_message_count,
            error=error,
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
