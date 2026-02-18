"""
MiniMax Anthropic API 兼容模型

基于 AnthropicModel，适配 MiniMax API 的 Anthropic 兼容端点。

特殊处理:
- 处理 MiniMax 特有的 thinking 和 signature 事件
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any

from anthropic.types import (
    RawContentBlockStartEvent,
    RawContentBlockDeltaEvent,
    ToolUseBlock,
    ThinkingDelta,
    SignatureDelta,
)

from hawi.agent.models.anthropic import AnthropicModel
from hawi.agent.models.anthropic._streaming import (
    _AnthropicStreamHandler,
    run_async_stream,
)
from hawi.agent.models.anthropic._converters import needs_async_conversion
from hawi.agent.message import StreamPart, MessageRequest, MessageResponse

logger = logging.getLogger(__name__)


class MiniMaxAnthropicStreamHandler(_AnthropicStreamHandler):
    """
    MiniMax Anthropic 流事件处理器
    
    扩展标准 Anthropic 流处理器，处理 MiniMax 特有的事件类型。
    """
    
    def handle_event(self, event) -> Iterator[StreamPart]:
        """处理事件，包括 MiniMax 特有的 thinking 和 signature 事件"""
        event_type = event.type
        
        # 处理 MiniMax 特有的 thinking 事件（顶层事件，不是 content_block）
        if event_type == "thinking":
            logger.debug(f"MiniMax thinking event")
            return iter([])
        
        # 处理 signature 事件
        if event_type == "signature":
            logger.debug(f"MiniMax signature event")
            return iter([])
        
        # 其他事件使用父类处理
        return super().handle_event(event)
    
    def _handle_content_block_start(
        self, event: RawContentBlockStartEvent
    ) -> Iterator[StreamPart]:
        """处理 content_block_start 事件"""
        block_index = event.index
        block = event.content_block
        self._content_blocks[block_index] = block
        
        block_type = getattr(block, 'type', None)
        
        if block_type == "tool_use":
            self._partial_json_parts = []
            block = block  # type: ignore
            logger.debug(f"MiniMax tool_use start: id={block.id}, name={block.name}")
            yield {
                "type": "tool_call_delta",
                "index": block_index,
                "id": block.id,
                "name": block.name,
                "arguments_delta": "",
                "is_start": True,
                "is_end": False,
            }
        elif block_type == "text":
            yield {
                "type": "text_delta",
                "index": block_index,
                "delta": "",
                "is_start": True,
                "is_end": False,
            }
        elif block_type == "thinking":
            yield {
                "type": "thinking_delta",
                "index": block_index,
                "delta": "",
                "is_start": True,
                "is_end": False,
            }
        elif block_type == "redacted_thinking":
            yield {
                "type": "thinking_delta",
                "index": block_index,
                "delta": "[Redacted thinking content]",
                "is_start": True,
                "is_end": True,
            }
        else:
            logger.debug(f"MiniMax block type {block_type} in content_block_start event")
    
    def _handle_content_block_delta(
        self, event: RawContentBlockDeltaEvent
    ) -> Iterator[StreamPart]:
        """处理 content_block_delta 事件"""
        block_index = event.index
        delta = event.delta
        
        delta_type = getattr(delta, 'type', None)
        
        if delta_type == "thinking_delta" or isinstance(delta, ThinkingDelta):
            thinking = getattr(delta, 'thinking', '')
            yield {
                "type": "thinking_delta",
                "index": block_index,
                "delta": thinking,
                "is_start": False,
                "is_end": False,
            }
        elif delta_type == "signature_delta" or isinstance(delta, SignatureDelta):
            logger.debug(f"MiniMax signature delta")
        else:
            yield from super()._handle_content_block_delta(event)


def minimax_stream_response(
    client, request: dict[str, Any]
) -> Iterator[StreamPart]:
    """MiniMax 同步流式响应处理"""
    with client.messages.stream(**request) as stream:
        handler = MiniMaxAnthropicStreamHandler(stream)
        for event in stream:
            yield from handler.handle_event(event)


async def minimax_stream_response_async(
    client, request: dict[str, Any]
) -> AsyncIterator[StreamPart]:
    """MiniMax 异步流式响应处理"""
    async with client.messages.stream(**request) as stream:
        handler = MiniMaxAnthropicStreamHandler(stream)
        async for event in stream:
            for part in handler.handle_event(event):
                yield part


class MiniMaxAnthropicModel(AnthropicModel):
    """
    MiniMax Anthropic API 兼容模型

    使用 Anthropic SDK 格式，但底层是 MiniMax 模型。
    端点: https://api.minimaxi.com/anthropic

    特殊处理:
    - 处理 MiniMax 特有的 thinking 和 signature 事件

    Example:
        model = MiniMaxAnthropicModel(
            model_id="MiniMax-M2.5",
            api_key="sk-...",
            base_url="https://api.minimaxi.com/anthropic",
        )
    """

    def __init__(
        self,
        *,
        model_id: str = "MiniMax-M2.5",
        api_key: str | None = None,
        base_url: str = "https://api.minimaxi.com/anthropic",
        **params,
    ):
        """初始化 MiniMax Anthropic 模型"""
        super().__init__(
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
            **params
        )

    def _prepare_request_impl(self, request: MessageRequest) -> dict[str, Any]:
        """准备请求，处理 MiniMax 特殊需求"""
        req = super()._prepare_request_impl(request)
        
        # MiniMax 不支持某些参数，进行清理
        unsupported_params = ["metadata", "top_k"]
        for param in unsupported_params:
            if param in req:
                logger.debug(f"Removing unsupported param '{param}' for MiniMax")
                del req[param]
        
        return req

    def _stream_impl(self, request: MessageRequest) -> Iterator[StreamPart]:
        """同步流式调用 - 使用 MiniMax 专属的 handler"""
        if needs_async_conversion(
            request.messages, self.enable_image_download
        ):
            yield from run_async_stream(self._astream_impl(request))
            return

        req = self._prepare_request_sync(request)
        yield from minimax_stream_response(self.client, req)

    async def _astream_impl(
        self, request: MessageRequest
    ) -> AsyncIterator[StreamPart]:
        """异步流式调用 - 使用 MiniMax 专属的 handler"""
        req = await self._prepare_request_async(request)
        async for chunk in minimax_stream_response_async(self.async_client, req):
            yield chunk
