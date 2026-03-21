"""
Anthropic 流式处理

处理同步和异步流式响应，将 Anthropic 特定的事件转换为统一的 DeltaPart。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any, cast, Union

from anthropic.types import (
    ContentBlock,
    TextBlock,
    ThinkingBlock,
    RedactedThinkingBlock,
    ToolUseBlock,
    Usage,

    RawMessageStartEvent,
    RawMessageDeltaEvent,
    RawMessageStopEvent,
    RawContentBlockStartEvent,
    RawContentBlockDeltaEvent,
    RawContentBlockStopEvent,

    TextDelta,
    InputJSONDelta,
    CitationsDelta,
    ThinkingDelta,
    SignatureDelta,
)

from hawi.models.message import (
    DeltaPart,
    DeltaTextPart,
    DeltaThinkingPart,
    DeltaSignaturePart,
    DeltaMetadataPart,
    DeltaToolCallPart,
    DeltaFinishPart,
)

if TYPE_CHECKING:
    from anthropic import Anthropic, AsyncAnthropic
    from anthropic.lib.streaming._messages import MessageStream, AsyncMessageStream
    from anthropic.lib.streaming._types import MessageStreamEvent, ParsedMessageStreamEvent


class _AnthropicStreamHandler:
    """Anthropic 流事件处理器
    
    封装同步和异步流处理的公共状态与逻辑。
    """
    
    def __init__(self, stream: MessageStream | AsyncMessageStream) -> None:
        self._stream = stream
        self._content_blocks: dict[int, ContentBlock] = {}
        self._partial_json_parts: list[str] = []
    
    def handle_event(self, event: MessageStreamEvent | ParsedMessageStreamEvent[Any]) -> Iterator[DeltaPart]:
        """统一事件处理入口"""
        event_type = event.type
        
        if event_type == "content_block_start":
            yield from self._handle_content_block_start(cast(RawContentBlockStartEvent, event))
        elif event_type == "content_block_delta":
            yield from self._handle_content_block_delta(cast(RawContentBlockDeltaEvent, event))
        elif event_type == "content_block_stop":
            yield from self._handle_content_block_stop(cast(RawContentBlockStopEvent, event))
        elif event_type == "message_start":
            yield from self._handle_message_start(cast(RawMessageStartEvent, event))
        elif event_type == "message_delta":
            yield from self._handle_message_delta(cast(RawMessageDeltaEvent, event))
        elif event_type == "message_stop":
            # message_stop 事件在流级别处理，需要访问 stream 对象获取 usage
            # 这里不生成 DeltaPart，由外层函数处理
            pass
        elif event_type in ("text","input_json"):
            # 不需要处理snapshot信息，因为我们已经用delta信息拼接完毕
            pass
        elif event_type in ("thinking", "signature"):
            # DeepSeek Anthropic 兼容端点发送的非标准事件
            # 这些事件不包含增量数据，已由 content_block_start/delta/stop 处理
            pass
        elif event_type == "citation":
            # Citation 作为元数据块传递（通过 DeltaMetadataPart）
            pass
        else:
            print(f"unhandled event type: {event_type}")
    
    def _handle_message_start(self, _event: RawMessageStartEvent) -> Iterator[DeltaPart]:
        """处理 message_start 事件（消息初始元数据，流式处理中无需额外操作）"""
        return iter([])
    
    def _handle_message_delta(self, _event: RawMessageDeltaEvent) -> Iterator[DeltaPart]:
        """处理 message_delta 事件（消息级别增量，如 stop_reason 变化）"""
        return iter([])
    
    def _handle_content_block_start(
        self, event: RawContentBlockStartEvent
    ) -> Iterator[DeltaPart]:
        """处理 content_block_start 事件"""
        block_index = event.index
        block = event.content_block
        self._content_blocks[block_index] = block
        
        if block.type == "tool_use":
            self._partial_json_parts = []
            block = cast(ToolUseBlock, block)
            yield DeltaToolCallPart({
                "type": "tool_call_delta",
                "index": block_index,
                "id": block.id,
                "name": block.name,
                "arguments_delta": "",
                "is_start": True,
                "is_end": False,
            })
        elif block.type == "text":
            yield DeltaTextPart({
                "type": "text_delta",
                "index": block_index,
                "delta": "",
                "is_start": True,
                "is_end": False,
            })
        elif block.type == "thinking":
            yield DeltaThinkingPart({
                "type": "reasoning_delta",
                "index": block_index,
                "delta": "",
                "is_start": True,
                "is_end": False,
            })
        elif block.type == "redacted_thinking":
            yield DeltaThinkingPart({
                "type": "reasoning_delta",
                "index": block_index,
                "delta": "[Redacted thinking content]",
                "is_start": True,
                "is_end": True,
            })
        else:
            print(f"unhandled block type {block.type} in content_block_start event")
    
    def _handle_content_block_delta(
        self, event: RawContentBlockDeltaEvent
    ) -> Iterator[DeltaPart]:
        """处理 content_block_delta 事件"""
        block_index = event.index
        delta = event.delta
        block = self._content_blocks[block_index]
        
        if isinstance(delta, TextDelta):
            yield DeltaTextPart({
                "type": "text_delta",
                "index": block_index,
                "delta": delta.text,
                "is_start": False,
                "is_end": False,
            })
        elif isinstance(delta, InputJSONDelta):
            block = cast(ToolUseBlock, block)
            self._partial_json_parts.append(delta.partial_json)
            yield DeltaToolCallPart({
                "type": "tool_call_delta",
                "index": block_index,
                "id": None,
                "name": None,
                "arguments_delta": delta.partial_json,
                "is_start": False,
                "is_end": False,
            })
        elif isinstance(delta, ThinkingDelta):
            yield DeltaThinkingPart({
                "type": "reasoning_delta",
                "index": block_index,
                "delta": delta.thinking,
                "is_start": False,
                "is_end": False,
            })
        elif isinstance(delta, SignatureDelta):
            # 签名是 thinking 的一部分，作为独立块传输
            yield DeltaSignaturePart({
                "type": "signature_delta",
                "index": block_index,
                "delta": delta.signature,
                "is_start": False,
                "is_end": False,
            })
        elif isinstance(delta, CitationsDelta):
            # Citation 是对输出的位置标注（字符位置、页面位置等）
            # 作为元数据在 Delta 流中传递
            citation = delta.citation
            if hasattr(citation, 'model_dump'):
                citation_dict: dict[str, Any] = dict(citation.model_dump())
            elif hasattr(citation, '__dict__'):
                citation_dict = dict(citation.__dict__)
            else:
                citation_dict = {"text": str(citation)}
            
            # 从 citation 中提取字符位置
            start_char: int | None = citation_dict.get("start_char_index") or citation_dict.get("start_char")
            end_char: int | None = citation_dict.get("end_char_index") or citation_dict.get("end_char")
            
            yield DeltaMetadataPart({
                "type": "metadata_delta",
                "index": block_index,
                "metadata": {"citations": [citation_dict]},
                "start_char": start_char,
                "end_char": end_char,
                "is_start": False,
                "is_end": False,
            })
        else:
            print(f"unhandled delta type {type(delta).__name__} in content_block_delta event")
    
    def _handle_content_block_stop(
        self, event: RawContentBlockStopEvent
    ) -> Iterator[DeltaPart]:
        """处理 content_block_stop 事件"""
        block_index = event.index
        block = self._content_blocks[block_index]
        
        if block.type == "text":
            yield DeltaTextPart({
                "type": "text_delta",
                "index": block_index,
                "delta": "",
                "is_start": False,
                "is_end": True,
            })
        elif block.type == "tool_use":
            block = cast(ToolUseBlock, block)
            yield DeltaToolCallPart({
                "type": "tool_call_delta",
                "index": block_index,
                "id": block.id,
                "name": block.name,
                "arguments_delta": "",
                "is_start": False,
                "is_end": True,
            })
            self._partial_json_parts = []
        elif block.type == "thinking":
            yield DeltaThinkingPart({
                "type": "reasoning_delta",
                "index": block_index,
                "delta": "",
                "is_start": False,
                "is_end": True,
            })
        else:
            print(f"unhandled block type {block.type} in content_block_stop event")
    
    def _create_finish_part(self) -> DeltaPart:
        """创建流结束事件"""
        usage = (
            self._stream.current_message_snapshot.usage
            if self._stream.current_message_snapshot
            else None
        )
        usage_dict: dict[str, int] | None = None
        if usage:
            usage_dict = {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            }
            # Include cache-related token counts if available
            if hasattr(usage, "cache_creation_input_tokens") and usage.cache_creation_input_tokens is not None:
                usage_dict["cache_write_tokens"] = usage.cache_creation_input_tokens
            if hasattr(usage, "cache_read_input_tokens") and usage.cache_read_input_tokens is not None:
                usage_dict["cache_read_tokens"] = usage.cache_read_input_tokens
        
        return DeltaFinishPart({
            "type": "finish",
            "stop_reason": "end_turn",
            "usage": usage_dict,
        })


def stream_response(
    client: Anthropic,
    request: dict[str, Any],
) -> Iterator[DeltaPart]:
    """同步流式响应处理

    将 Anthropic 原生事件转换为统一的 DeltaPart 增量块流。
    """
    with client.messages.stream(**request) as stream:
        handler = _AnthropicStreamHandler(stream)
        for event in stream:
            yield from handler.handle_event(event)
        yield handler._create_finish_part()


async def stream_response_async(
    client: AsyncAnthropic,
    request: dict[str, Any],
) -> AsyncIterator[DeltaPart]:
    """异步流式响应处理

    将 Anthropic 原生事件转换为统一的 DeltaPart 增量块流。
    """
    async with client.messages.stream(**request) as stream:
        handler = _AnthropicStreamHandler(stream)
        async for event in stream:
            for part in handler.handle_event(event):
                yield part
        yield handler._create_finish_part()


def run_async_stream(
    async_iter: AsyncIterator[DeltaPart],
) -> Iterator[DeltaPart]:
    """将异步流式迭代器转为同步迭代器"""
    # 使用 cast 将 AsyncIterator 转为 AsyncGenerator 以访问 aclose 方法
    # 这是安全的，因为在实际运行时传入的是 AsyncGenerator
    async_gen = cast(AsyncGenerator[DeltaPart, None], async_iter)
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
        # Properly close the async generator to prevent warnings
        try:
            loop.run_until_complete(async_gen.aclose())
        except Exception:
            pass  # Ignore errors during cleanup
        
        # Cancel any remaining pending tasks to prevent warnings
        try:
            pending = asyncio.all_tasks(loop)
            if pending:
                for task in pending:
                    task.cancel()
                # Run the event loop briefly to let cancellations complete
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            pass  # Ignore errors during cleanup
        
        loop.close()
