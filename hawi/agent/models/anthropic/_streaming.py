"""
Anthropic 流式处理

处理同步和异步流式响应，将 Anthropic 特定的事件转换为统一的 StreamEvent。
遵循 content_block 事件模型（与 OpenAI 处理器保持一致）。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any, cast

from anthropic.types import (
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    TextDelta,
    InputJSONDelta,
    ThinkingDelta,
    SignatureDelta,
)

from hawi.agent.model import StreamEvent
from hawi.agent.messages import TextPart, ToolCallPart

if TYPE_CHECKING:
    from anthropic import Anthropic, AsyncAnthropic


def stream_response(
    client: Anthropic,
    request: dict[str, Any],
) -> Iterator[StreamEvent]:
    """同步流式响应处理

    将 Anthropic 原生事件转换为统一的 content_block 事件流。
    """
    # 工具调用状态累积
    current_tool_call: dict[str, Any] | None = None
    partial_json_parts: list[str] = []
    block_index: int = 0

    with client.messages.stream(**request) as stream:
        for event in stream:
            event_type = event.type

            if event_type == "content_block_start":
                start_event = cast(RawContentBlockStartEvent, event)
                block = start_event.content_block

                if block.type == "tool_use":
                    # 开始新的工具调用
                    current_tool_call = {
                        "id": block.id,
                        "name": block.name,
                    }
                    partial_json_parts = []

                    yield StreamEvent(
                        type="content_block_start",
                        block_type="tool_use",
                        block_index=block_index,
                        tool_call_id=block.id,
                        tool_name=block.name,
                    )
                elif block.type == "text":
                    yield StreamEvent(
                        type="content_block_start",
                        block_type="text",
                        block_index=block_index,
                    )
                elif block.type == "thinking":
                    yield StreamEvent(
                        type="content_block_start",
                        block_type="thinking",
                        block_index=block_index,
                    )
                elif block.type == "redacted_thinking":
                    yield StreamEvent(
                        type="content_block_start",
                        block_type="redacted_thinking",
                        block_index=block_index,
                    )

            elif event_type == "content_block_delta":
                delta_event = cast(RawContentBlockDeltaEvent, event)
                delta = delta_event.delta

                if isinstance(delta, TextDelta):
                    yield StreamEvent(
                        type="content_block_delta",
                        delta_type="text",
                        delta=delta.text,
                        block_index=block_index,
                    )
                elif isinstance(delta, InputJSONDelta):
                    # 累积 partial_json
                    partial_json_parts.append(delta.partial_json)
                    yield StreamEvent(
                        type="content_block_delta",
                        delta_type="tool_input",
                        delta=delta.partial_json,
                        block_index=block_index,
                    )
                elif isinstance(delta, ThinkingDelta):
                    yield StreamEvent(
                        type="content_block_delta",
                        delta_type="thinking",
                        delta=delta.thinking,
                        block_index=block_index,
                    )
                elif isinstance(delta, SignatureDelta):
                    # 签名是 thinking 的一部分，可选处理
                    pass

            elif event_type == "content_block_stop":
                stop_event = cast(RawContentBlockStopEvent, event)
                block = stop_event.content_block

                # 工具调用完成，组装完整数据
                if block.type == "tool_use" and current_tool_call is not None:
                    try:
                        arguments = json.loads("".join(partial_json_parts))
                    except json.JSONDecodeError:
                        arguments = {}  # 解析失败时返回空对象

                    yield StreamEvent(
                        type="content_block_stop",
                        block_type="tool_use",
                        block_index=block_index,
                        tool_call_id=current_tool_call["id"],
                        tool_name=current_tool_call["name"],
                        tool_arguments=arguments,
                        full_content="".join(partial_json_parts),
                    )
                    current_tool_call = None
                    partial_json_parts = []
                elif block.type == "text":
                    yield StreamEvent(
                        type="content_block_stop",
                        block_type="text",
                        block_index=block_index,
                    )
                elif block.type == "thinking":
                    yield StreamEvent(
                        type="content_block_stop",
                        block_type="thinking",
                        block_index=block_index,
                    )
                elif block.type == "redacted_thinking":
                    yield StreamEvent(
                        type="content_block_stop",
                        block_type="redacted_thinking",
                        block_index=block_index,
                    )

                block_index += 1

            elif event_type == "message_stop":
                usage = (
                    stream.current_message_snapshot.usage
                    if stream.current_message_snapshot
                    else None
                )
                if usage:
                    usage_dict: dict[str, int] = {
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                    }
                    # Include cache-related token counts if available
                    if hasattr(usage, "cache_creation_input_tokens") and usage.cache_creation_input_tokens is not None:
                        usage_dict["cache_creation_input_tokens"] = usage.cache_creation_input_tokens
                    if hasattr(usage, "cache_read_input_tokens") and usage.cache_read_input_tokens is not None:
                        usage_dict["cache_read_input_tokens"] = usage.cache_read_input_tokens
                    yield StreamEvent(
                        type="usage",
                        usage=usage_dict,
                    )
                yield StreamEvent(type="finish", stop_reason="end_turn")


async def stream_response_async(
    client: AsyncAnthropic,
    request: dict[str, Any],
) -> AsyncIterator[StreamEvent]:
    """异步流式响应处理

    将 Anthropic 原生事件转换为统一的 content_block 事件流。
    """
    # 工具调用状态累积
    current_tool_call: dict[str, Any] | None = None
    partial_json_parts: list[str] = []
    block_index: int = 0

    async with client.messages.stream(**request) as stream:
        async for event in stream:
            event_type = event.type

            if event_type == "content_block_start":
                start_event = cast(RawContentBlockStartEvent, event)
                block = start_event.content_block

                if block.type == "tool_use":
                    # 开始新的工具调用
                    current_tool_call = {
                        "id": block.id,
                        "name": block.name,
                    }
                    partial_json_parts = []

                    yield StreamEvent(
                        type="content_block_start",
                        block_type="tool_use",
                        block_index=block_index,
                        tool_call_id=block.id,
                        tool_name=block.name,
                    )
                elif block.type == "text":
                    yield StreamEvent(
                        type="content_block_start",
                        block_type="text",
                        block_index=block_index,
                    )
                elif block.type == "thinking":
                    yield StreamEvent(
                        type="content_block_start",
                        block_type="thinking",
                        block_index=block_index,
                    )
                elif block.type == "redacted_thinking":
                    yield StreamEvent(
                        type="content_block_start",
                        block_type="redacted_thinking",
                        block_index=block_index,
                    )

            elif event_type == "content_block_delta":
                delta_event = cast(RawContentBlockDeltaEvent, event)
                delta = delta_event.delta

                if isinstance(delta, TextDelta):
                    yield StreamEvent(
                        type="content_block_delta",
                        delta_type="text",
                        delta=delta.text,
                        block_index=block_index,
                    )
                elif isinstance(delta, InputJSONDelta):
                    # 累积 partial_json
                    partial_json_parts.append(delta.partial_json)
                    yield StreamEvent(
                        type="content_block_delta",
                        delta_type="tool_input",
                        delta=delta.partial_json,
                        block_index=block_index,
                    )
                elif isinstance(delta, ThinkingDelta):
                    yield StreamEvent(
                        type="content_block_delta",
                        delta_type="thinking",
                        delta=delta.thinking,
                        block_index=block_index,
                    )
                elif isinstance(delta, SignatureDelta):
                    # 签名是 thinking 的一部分，可选处理
                    pass

            elif event_type == "content_block_stop":
                stop_event = cast(RawContentBlockStopEvent, event)
                block = stop_event.content_block

                # 工具调用完成，组装完整数据
                if block.type == "tool_use" and current_tool_call is not None:
                    try:
                        arguments = json.loads("".join(partial_json_parts))
                    except json.JSONDecodeError:
                        arguments = {}  # 解析失败时返回空对象

                    yield StreamEvent(
                        type="content_block_stop",
                        block_type="tool_use",
                        block_index=block_index,
                        tool_call_id=current_tool_call["id"],
                        tool_name=current_tool_call["name"],
                        tool_arguments=arguments,
                        full_content="".join(partial_json_parts),
                    )
                    current_tool_call = None
                    partial_json_parts = []
                elif block.type == "text":
                    yield StreamEvent(
                        type="content_block_stop",
                        block_type="text",
                        block_index=block_index,
                    )
                elif block.type == "thinking":
                    yield StreamEvent(
                        type="content_block_stop",
                        block_type="thinking",
                        block_index=block_index,
                    )
                elif block.type == "redacted_thinking":
                    yield StreamEvent(
                        type="content_block_stop",
                        block_type="redacted_thinking",
                        block_index=block_index,
                    )

                block_index += 1

            elif event_type == "message_stop":
                usage = (
                    stream.current_message_snapshot.usage
                    if stream.current_message_snapshot
                    else None
                )
                if usage:
                    usage_dict: dict[str, int] = {
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                    }
                    # Include cache-related token counts if available
                    if hasattr(usage, "cache_creation_input_tokens") and usage.cache_creation_input_tokens is not None:
                        usage_dict["cache_creation_input_tokens"] = usage.cache_creation_input_tokens
                    if hasattr(usage, "cache_read_input_tokens") and usage.cache_read_input_tokens is not None:
                        usage_dict["cache_read_input_tokens"] = usage.cache_read_input_tokens
                    yield StreamEvent(
                        type="usage",
                        usage=usage_dict,
                    )
                yield StreamEvent(type="finish", stop_reason="end_turn")


def run_async_stream(
    async_gen: AsyncIterator[StreamEvent],
) -> Iterator[StreamEvent]:
    """将异步流式生成器转为同步迭代器"""
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)

        while True:
            try:
                event = loop.run_until_complete(async_gen.__anext__())
                yield event
            except StopAsyncIteration:
                break
    finally:
        loop.close()
