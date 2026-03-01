"""
Model events for Hawi Event System.

Model events are produced by Model implementations.
"""

from __future__ import annotations

from typing import Any, Literal, overload

from pydantic import field_serializer

from hawi.errors import ModelError
from hawi.model.message import ContentPart, StreamPart, TokenUsage

from .event import Event

from hawi.model.message import TextPart, ToolCallPart, ReasoningPart


class ModelStreamStartEvent(Event):
    """Model 开始流式响应"""
    request_id: str

    @classmethod
    def create(cls, request_id: str) -> ModelStreamStartEvent:
        return cls(
            type="model.stream_start",
            source="model",
            request_id=request_id,
        )


class ModelStreamStopEvent(Event):
    """Model 流式响应结束"""
    request_id: str
    stop_reason: str
    usage: TokenUsage | None = None

    @classmethod
    def create(
        cls,
        request_id: str,
        stop_reason: str,
        usage: TokenUsage | None = None,
    ) -> ModelStreamStopEvent:
        return cls(
            type="model.stream_stop",
            source="model",
            request_id=request_id,
            stop_reason=stop_reason,
            usage=usage,
        )


class ModelContentBlockStartEvent(Event):
    """内容块开始（统一 Anthropic 和 OpenAI 的非工具内容类型）

    block_type 说明:
    - text: 普通文本（Anthropic text, OpenAI content）
    - thinking: 推理内容（Anthropic thinking, OpenAI reasoning_content）
    - redacted_thinking: 被编辑的推理（Anthropic 特有）

    注意: tool_use 类型请使用 ModelToolUseBlockStartEvent
    """

    request_id: str
    block_index: int
    block_type: Literal["text", "thinking", "redacted_thinking"]

    @classmethod
    def create(
        cls,
        request_id: str,
        block_index: int,
        block_type: Literal["text", "thinking", "redacted_thinking"],
    ) -> ModelContentBlockStartEvent:
        """创建内容块开始事件

        Args:
            request_id: 请求 ID
            block_index: 内容块序号
            block_type: 内容块类型

        Returns:
            ModelContentBlockStartEvent 实例

        Example:
            # 文本块
            event = ModelContentBlockStartEvent.create("req-1", 0, "text")

            # 推理块
            event = ModelContentBlockStartEvent.create("req-1", 1, "thinking")
        """
        return cls(
            type="model.content_block_start",
            source="model",
            request_id=request_id,
            block_index=block_index,
            block_type=block_type,
        )


class ModelContentBlockDeltaEvent(Event):
    """内容块增量更新

    包含完整的原始 StreamPart，同时提供便捷属性访问常用字段。

    Attributes:
        request_id: 请求 ID
        block_index: 内容块序号
        delta_type: 增量类型（便捷属性，也可从 part 获取）
        delta: 增量内容（便捷属性，也可从 part 获取）
        part: 完整的 StreamPart，包含 is_start, is_end 等原始信息

    便捷属性:
        is_start: 是否是该内容块的开始
        is_end: 是否是该内容块的结束

    Example:
        # 简单用法：直接访问便捷属性
        if event.delta:
            print(event.delta)

        # 高级用法：访问完整 Part 信息
        if event.part["is_start"]:
            print("新块开始，准备 UI...")
        if event.part["is_end"]:
            print("块结束，刷新 UI...")
    """

    request_id: str
    block_index: int
    delta_type: Literal["text", "thinking", "tool_input", "signature"]
    delta: str
    part: StreamPart

    @property
    def is_start(self) -> bool:
        """是否是该内容块的开始"""
        return self.part.get("is_start", False)

    @property
    def is_end(self) -> bool:
        """是否是该内容块的结束"""
        return self.part.get("is_end", False)

    @classmethod
    def create(
        cls,
        request_id: str,
        part: StreamPart,
    ) -> ModelContentBlockDeltaEvent:
        """从 StreamPart 创建事件

        Args:
            request_id: 请求 ID
            part: StreamPart（StreamTextPart | StreamThinkingPart | StreamToolCallPart）

        Returns:
            ModelContentBlockDeltaEvent 实例
        """
        # 从 part 中提取信息（使用 get 避免类型错误）
        block_index = part.get("index", 0)
        part_type = part.get("type", "")

        # 映射 part type 到 delta_type
        delta_type_mapping = {
            "text_delta": "text",
            "thinking_delta": "thinking",
            "tool_call_delta": "tool_input",
        }
        delta_type = delta_type_mapping.get(part_type, "text")

        # 提取 delta 内容
        if part_type == "text_delta":
            delta = part.get("delta", "")
        elif part_type == "thinking_delta":
            delta = part.get("delta", "")
        elif part_type == "tool_call_delta":
            delta = part.get("arguments_delta", "")
        else:
            delta = ""

        return cls(
            type="model.content_block_delta",
            source="model",
            request_id=request_id,
            block_index=block_index,
            delta_type=delta_type,
            delta=delta,
            part=part,
        )


class ModelToolUseBlockStartEvent(Event):
    """工具调用内容块开始

    专门用于 tool_use 类型的内容块，包含 tool_call_id 和 tool_name。
    """

    request_id: str
    block_index: int
    tool_call_id: str
    tool_name: str

    @classmethod
    def create(
        cls,
        request_id: str,
        block_index: int,
        tool_call_id: str,
        tool_name: str,
    ) -> ModelToolUseBlockStartEvent:
        """创建工具调用内容块开始事件

        Args:
            request_id: 请求 ID
            block_index: 内容块序号
            tool_call_id: 工具调用 ID
            tool_name: 工具名称

        Returns:
            ModelToolUseBlockStartEvent 实例
        """
        return cls(
            type="model.tool_use_block_start",
            source="model",
            request_id=request_id,
            block_index=block_index,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
        )


class ModelToolUseBlockDeltaEvent(Event):
    """工具调用内容块增量

    专门用于 tool_use 类型的增量更新。
    """

    request_id: str
    block_index: int
    tool_call_id: str
    arguments_delta: str

    @classmethod
    def create(
        cls,
        request_id: str,
        block_index: int,
        tool_call_id: str,
        arguments_delta: str,
    ) -> ModelToolUseBlockDeltaEvent:
        """创建工具调用内容块增量事件

        Args:
            request_id: 请求 ID
            block_index: 内容块序号
            tool_call_id: 工具调用 ID
            arguments_delta: 参数增量（JSON 片段）

        Returns:
            ModelToolUseBlockDeltaEvent 实例
        """
        return cls(
            type="model.tool_use_block_delta",
            source="model",
            request_id=request_id,
            block_index=block_index,
            tool_call_id=tool_call_id,
            arguments_delta=arguments_delta,
        )


class ModelToolUseBlockStopEvent(Event):
    """工具调用内容块结束

    专门用于 tool_use 类型的结束事件，包含完整的参数。
    """

    request_id: str
    block_index: int
    tool_call_id: str
    arguments: str  # 完整的 JSON 参数

    @classmethod
    def create(
        cls,
        request_id: str,
        block_index: int,
        tool_call_id: str,
        arguments: str,
    ) -> ModelToolUseBlockStopEvent:
        """创建工具调用内容块结束事件

        Args:
            request_id: 请求 ID
            block_index: 内容块序号
            tool_call_id: 工具调用 ID
            arguments: 完整的参数 JSON 字符串

        Returns:
            ModelToolUseBlockStopEvent 实例
        """
        return cls(
            type="model.tool_use_block_stop",
            source="model",
            request_id=request_id,
            block_index=block_index,
            tool_call_id=tool_call_id,
            arguments=arguments,
        )


class ModelContentBlockStopEvent(Event):
    """内容块结束

    包含完整的 ContentPart，无论流式还是非流式，都通过此事件提供最终内容。
    流式场景下，Consumer 可以累积 Delta 事件的内容，在 Stop 事件中获取完整 Part。
    非流式场景下，直接通过 content 获取完整内容。

    Attributes:
        request_id: 请求 ID
        block_index: 内容块序号
        content: 完整的 ContentPart 列表（text/thinking/tool_call 等）

    Example:
        # 流式场景：累积 delta，最终在 stop 获取完整内容
        deltas = []
        async for event in agent.run("Hello", stream=True):
            if isinstance(event, ModelContentBlockDeltaEvent):
                deltas.append(event.delta)
            elif isinstance(event, ModelContentBlockStopEvent):
                # 获取完整 ContentPart
                content_part = event.content[0]  # TextPart, ReasoningPart, etc.
                print(f"完整内容: {content_part}")

        # 非流式场景：直接获取内容
        result = agent.run("Hello")  # 内部同样发送 Start/Delta/Stop 事件
    """

    request_id: str
    block_index: int
    content: list[ContentPart]

    @property
    def block_type(self) -> Literal["text", "thinking", "tool_use", "redacted_thinking"] | None:
        """内容块类型（从第一个 ContentPart 推断）"""
        if not self.content:
            return None
        part = self.content[0]
        return part['type']

    @classmethod
    def create(
        cls,
        request_id: str,
        block_index: int,
        content: list[ContentPart],
    ) -> ModelContentBlockStopEvent:
        """创建内容块结束事件

        Args:
            request_id: 请求 ID
            block_index: 内容块序号
            content: 完整的 ContentPart 列表

        Returns:
            ModelContentBlockStopEvent 实例
        """
        return cls(
            type='model.content_block_stop',
            source='model',
            request_id=request_id,
            block_index=block_index,
            content=content,
        )


class ModelMetadataEvent(Event):
    """Model 元数据（usage 等）"""
    request_id: str
    usage: TokenUsage | None = None
    latency_ms: float | None = None

    @classmethod
    def create(
        cls,
        request_id: str,
        usage: TokenUsage | None = None,
        latency_ms: float | None = None,
    ) -> ModelMetadataEvent:
        return cls(
            type='model.metadata',
            source='model',
            request_id=request_id,
            usage=usage,
            latency_ms=latency_ms,
        )


class ModelErrorEvent(Event):
    error: "ModelError"

    @classmethod
    def create(cls, error: "ModelError"):
        return cls(
            type='model.error',
            source='model',
            error=error,
        )

    @field_serializer('error')
    def serialize_error(self, error: ModelError) -> dict[str, Any]:
        """将 ModelError 序列化为可 JSON 序列化的字典"""
        return {
            'type': error.error_type if hasattr(error, 'error_type') else 'unknown',
            'message': str(error),
            'class': error.__class__.__name__,
        }
