"""
Model events for Hawi Event System.

Model events are produced by Model implementations.
"""

from __future__ import annotations

from typing import Any, Literal, overload

from pydantic import field_serializer

from hawi.errors import ModelError
from hawi.models.message import ContentPart, DeltaPart, TokenUsage, ContentPartType

from .event import Event


# 内容块类型 - 用于事件系统中 ContentBlock 相关的 block_type 字段
# 与 ContentPart.type 有所不同：
# - "reasoning" (ContentPart) -> "reasoning" (事件 block_type)
# - 不包含 tool_call、image 等无法作为内容块出现的类型
ContentBlockType = Literal["text", "reasoning", "redacted_thinking", "tool_use"]


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

    @classmethod
    def create(
        cls,
        request_id: str,
        stop_reason: str,
    ) -> ModelStreamStopEvent:
        return cls(
            type="model.stream_stop",
            source="model",
            request_id=request_id,
            stop_reason=stop_reason,
        )


class ModelContentBlockStartEvent(Event):
    """内容块开始（统一 Anthropic 和 OpenAI 的非工具内容类型）

    block_type 说明:
    - text: 普通文本（Anthropic text, OpenAI content）
    - thinking: 推理内容（Anthropic thinking, OpenAI reasoning_content）
    - redacted_thinking: 被编辑的推理（Anthropic 特有）

    注意: tool_use 类型请使用 ModelToolCallBlockStartEvent
    """

    request_id: str
    block_index: int
    block_type: Literal["text", "reasoning", "redacted_thinking"]

    @classmethod
    def create(
        cls,
        request_id: str,
        block_index: int,
        block_type: Literal["text", "reasoning", "redacted_thinking"],
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

    包含内容块增量的核心字段。不包含完整的 DeltaPart，只提取必要信息。

    关于 delta 的语义：
    - streaming=True: delta 表示每一个小碎块的信息
    - streaming=False: delta 表示完整的内容（非流式模式下一次性返回）

    Attributes:
        request_id: 请求 ID
        block_index: 内容块序号
        delta_type: 增量类型（"text", "reasoning", "tool_input", "signature"）
        delta: 增量内容
        is_streaming: 是否来自流式接口

    Example:
        # 流式模式：累积 delta
        if event.is_streaming:
            buffer += event.delta
        else:
            # 非流式模式：delta 就是完整内容
            print(event.delta)
    """

    request_id: str
    block_index: int
    delta_type: Literal["text", "reasoning", "tool_input", "signature"]
    delta: str
    is_streaming: bool = True

    @classmethod
    def create(
        cls,
        request_id: str,
        part: DeltaPart,
        is_streaming: bool = True,
    ) -> "ModelContentBlockDeltaEvent":
        """从 DeltaPart 创建事件

        Args:
            request_id: 请求 ID
            part: DeltaPart（DeltaTextPart | DeltaThinkingPart | DeltaToolCallPart）
            is_streaming: 是否来自流式接口（默认 True）

        Returns:
            ModelContentBlockDeltaEvent 实例
        """
        # 从 part 中提取信息
        block_index = part.get("index", 0)
        part_type = part.get("type", "")

        # 映射 part type 到 delta_type 并提取 delta 内容
        if part_type == "text_delta":
            delta_type: Literal["text", "reasoning", "tool_input", "signature"] = "text"
            delta = part.get("delta", "")
        elif part_type == "reasoning_delta":
            delta_type = "reasoning"
            delta = part.get("delta", "")
        elif part_type == "tool_call_delta":
            delta_type = "tool_input"
            delta = part.get("arguments_delta", "")
        elif part_type == "signature_delta":
            delta_type = "signature"
            delta = part.get("delta", "")
        else:
            delta_type = "text"
            delta = ""

        return cls(
            type="model.content_block_delta",
            source="model",
            request_id=request_id,
            block_index=block_index,
            delta_type=delta_type,
            delta=delta,
            is_streaming=is_streaming,
        )


class ModelToolCallBlockStartEvent(Event):
    """工具调用内容块开始

    专门用于 tool_call 类型的内容块，包含 tool_call_id 和 tool_name。
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
    ) -> ModelToolCallBlockStartEvent:
        """创建工具调用内容块开始事件

        Args:
            request_id: 请求 ID
            block_index: 内容块序号
            tool_call_id: 工具调用 ID
            tool_name: 工具名称

        Returns:
            ModelToolCallBlockStartEvent 实例
        """
        return cls(
            type="model.tool_call_block_start",
            source="model",
            request_id=request_id,
            block_index=block_index,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
        )


class ModelToolCallBlockDeltaEvent(Event):
    """工具调用内容块增量

    专门用于 tool_call 类型的增量更新。

    关于 arguments_delta 的语义：
    - streaming=True: arguments_delta 表示每一个小碎块的信息
    - streaming=False: arguments_delta 表示完整的参数 JSON

    Attributes:
        request_id: 请求 ID
        block_index: 内容块序号
        tool_call_id: 工具调用 ID
        arguments_delta: 参数增量（JSON 片段或完整 JSON）
        is_streaming: 是否来自流式接口
    """

    request_id: str
    block_index: int
    tool_call_id: str
    arguments_delta: str
    is_streaming: bool = True

    @classmethod
    def create(
        cls,
        request_id: str,
        block_index: int,
        tool_call_id: str,
        arguments_delta: str,
        is_streaming: bool = True,
    ) -> ModelToolCallBlockDeltaEvent:
        """创建工具调用内容块增量事件

        Args:
            request_id: 请求 ID
            block_index: 内容块序号
            tool_call_id: 工具调用 ID
            arguments_delta: 参数增量（JSON 片段或完整 JSON）
            is_streaming: 是否来自流式接口（默认 True）

        Returns:
            ModelToolCallBlockDeltaEvent 实例
        """
        return cls(
            type="model.tool_call_block_delta",
            source="model",
            request_id=request_id,
            block_index=block_index,
            tool_call_id=tool_call_id,
            arguments_delta=arguments_delta,
            is_streaming=is_streaming,
        )


class ModelToolCallBlockStopEvent(Event):
    """工具调用内容块结束

    专门用于 tool_call 类型的结束事件。
    包含完整的工具调用信息，包括工具名和解析后的参数。

    Attributes:
        request_id: 请求 ID
        block_index: 内容块序号
        tool_call_id: 工具调用 ID
        tool_name: 工具名称
        arguments: 完整解析后的参数（dict 类型）
    """

    request_id: str
    block_index: int
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]

    @classmethod
    def create(
        cls,
        request_id: str,
        block_index: int,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ModelToolCallBlockStopEvent:
        """创建工具调用内容块结束事件

        Args:
            request_id: 请求 ID
            block_index: 内容块序号
            tool_call_id: 工具调用 ID
            tool_name: 工具名称
            arguments: 完整解析后的参数

        Returns:
            ModelToolCallBlockStopEvent 实例
        """
        return cls(
            type="model.tool_call_block_stop",
            source="model",
            request_id=request_id,
            block_index=block_index,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
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
    def block_type(self) -> ContentPartType | None:
        """内容块类型（从第一个 ContentPart 推断）"""
        if not self.content:
            return None
        part = self.content[0]
        return part['type']

    @field_serializer('content')
    def serialize_content(self, content: list[ContentPart]) -> list[dict]:
        return [dict(item) for item in content]

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


class ModelContentMetadataEvent(Event):
    """内容元数据事件 - 携带内容块的元数据（如引用位置）

    用于在流式响应中传递 Citation 等元数据。
    元数据与对应的 content block 按 block_index 关联。

    Attributes:
        request_id: 请求 ID
        block_index: 内容块序号（对应 content 中的索引）
        metadata: 元数据内容（如 CitationPart）
        start_char: 被标注文本起始位置（可选，None 表示全文）
        end_char: 被标注文本结束位置（可选）

    Example:
        # 流式场景：获取 Citation
        async for event in agent.run("Hello", stream=True):
            if isinstance(event, ModelContentMetadataEvent):
                # 获取 Citation
                citations = event.metadata.get("citations", [])
                print(f"引用位置: {event.start_char}-{event.end_char}")
    """

    request_id: str
    block_index: int
    metadata: dict[str, Any]  # 元数据（如 {"citations": [...]}）
    start_char: int | None = None  # 被标注文本起始位置
    end_char: int | None = None    # 被标注文本结束位置

    @classmethod
    def create(
        cls,
        request_id: str,
        block_index: int,
        metadata: dict[str, Any],
        start_char: int | None = None,
        end_char: int | None = None,
    ) -> ModelContentMetadataEvent:
        """创建内容元数据事件

        Args:
            request_id: 请求 ID
            block_index: 内容块序号
            metadata: 元数据内容
            start_char: 被标注文本起始位置（可选）
            end_char: 被标注文本结束位置（可选）

        Returns:
            ModelContentMetadataEvent 实例
        """
        return cls(
            type='model.content_metadata',
            source='model',
            request_id=request_id,
            block_index=block_index,
            metadata=metadata,
            start_char=start_char,
            end_char=end_char,
        )


class ModelRetryEvent(Event):
    """Model 调用重试事件

    当模型调用因错误触发重试策略时产生此事件。
    允许监听器了解重试状态和进度。

    Attributes:
        request_id: 请求 ID
        error_type: 触发重试的错误类型
        attempt: 当前重试次数（从1开始，表示第几次重试）
        max_retries: 最大重试次数
        error_message: 错误信息
    """
    request_id: str
    error_type: str
    attempt: int
    max_retries: int
    error_message: str

    @classmethod
    def create(
        cls,
        request_id: str,
        error_type: str,
        attempt: int,
        max_retries: int,
        error_message: str,
    ) -> ModelRetryEvent:
        """创建重试事件

        Args:
            request_id: 请求 ID
            error_type: 触发重试的错误类型 (如 'network', 'throttle')
            attempt: 当前重试次数（1表示第1次重试）
            max_retries: 最大重试次数
            error_message: 错误信息

        Returns:
            ModelRetryEvent 实例
        """
        return cls(
            type='model.retry',
            source='model',
            request_id=request_id,
            error_type=error_type,
            attempt=attempt,
            max_retries=max_retries,
            error_message=error_message,
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
