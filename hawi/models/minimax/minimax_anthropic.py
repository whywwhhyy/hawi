"""
MiniMax Anthropic API 兼容模型

基于 AnthropicModel，适配 MiniMax API 的 Anthropic 兼容端点。

特殊处理:
- 处理 MiniMax 特有的 thinking 和 signature 事件
- 处理 MiniMax 特有的错误码
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Iterator
from typing import Any, cast

from anthropic.types import (
    RawContentBlockStartEvent,
    RawContentBlockDeltaEvent,
    ToolUseBlock,
    ThinkingDelta,
    SignatureDelta,
    TextBlock,
    ThinkingBlock,
    RedactedThinkingBlock,
)

from hawi.models import TokenEstimate
from hawi.models.anthropic import AnthropicModel
from hawi.models.anthropic._streaming import (
    _AnthropicStreamHandler,
    run_async_stream,
)
from hawi.models._model_listing import (
    afetch_json_model_ids,
    bearer_auth_headers,
    fetch_json_model_ids,
)
from hawi.models.anthropic._converters import needs_async_conversion
from hawi.models.message import DeltaPart, MessageRequest, MessageResponse
from hawi.models.anthropic._model import _convert_anthropic_error as _base_convert_anthropic_error
from hawi.errors import RemoteError, ThrottleError, DeniedError, ValidationError

logger = logging.getLogger(__name__)


def _convert_minimax_anthropic_error(e: Exception) -> Exception:
    """Convert MiniMax-specific errors when using Anthropic SDK.
    
    This function extends the base Anthropic error conversion with MiniMax-specific
    error code handling.
    
    MiniMax Error Codes (via Anthropic endpoint):
        - 794, 1000: Temporary internal errors -> RemoteError (retryable)
        - 1002, 2045, 1041: Rate limiting -> ThrottleError
        - 1004, 2049: Authentication errors -> DeniedError
        - 1008: Insufficient balance -> DeniedError
        - 2013: Invalid parameters -> ValidationError
        - 1026, 1027: Sensitive content -> ValidationError
    
    Args:
        e: The exception from Anthropic SDK.
        
    Returns:
        A Hawi ModelError subclass instance, or original exception if not convertible.
    """
    # First try the base Anthropic error conversion
    converted = _base_convert_anthropic_error(e)
    
    # If base conversion handled it (or it's not an APIError), return as-is
    if converted is not e:
        return converted
    
    # Handle MiniMax-specific error codes
    try:
        from anthropic import APIError
        
        if isinstance(e, APIError):
            error_msg = str(e)
            
            # Check for MiniMax error code 794, 1000: temporary internal errors
            if "794" in error_msg or "1000" in error_msg:
                return RemoteError(f"MiniMax temporary error: {e}")
            
            # Check error body for structured error codes
            body = getattr(e, 'body', None)
            if isinstance(body, dict):
                error_body = body.get('error', {})
                if isinstance(error_body, dict):
                    code = error_body.get('code')
                    
                    # 794, 1000: temporary internal errors (retryable)
                    if code in (794, '794', 1000, '1000'):
                        return RemoteError(f"MiniMax temporary error ({code}): {e}")
                    
                    # 1004, 2049: authentication errors
                    if code in (1004, '1004', 2049, '2049'):
                        return DeniedError(f"MiniMax authentication failed ({code}): {e}")
                    
                    # 1002, 2045, 1041: rate limiting
                    if code in (1002, '1002', 2045, '2045', 1041, '1041'):
                        return ThrottleError(f"MiniMax rate limit ({code}): {e}")
                    
                    # 1008: insufficient balance
                    if code in (1008, '1008'):
                        return DeniedError(f"MiniMax insufficient balance: {e}")
                    
                    # 2013: invalid parameters
                    if code in (2013, '2013'):
                        return ValidationError(f"MiniMax invalid parameters ({code}): {e}")
                    
                    # 1026, 1027: sensitive content
                    if code in (1026, '1026', 1027, '1027'):
                        return ValidationError(f"MiniMax sensitive content ({code}): {e}")
        
        return e
        
    except ImportError:
        return e


class MiniMaxAnthropicStreamHandler(_AnthropicStreamHandler):
    """
    MiniMax Anthropic 流事件处理器
    
    扩展标准 Anthropic 流处理器，处理 MiniMax 特有的事件类型。
    """
    
    def handle_event(self, event) -> Iterator[DeltaPart]:
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
    ) -> Iterator[DeltaPart]:
        """处理 content_block_start 事件"""
        block_index = event.index
        block = event.content_block
        self._content_blocks[block_index] = block
        
        if block.type == "tool_use":
            self._partial_json_parts = []
            tool_use_block = cast(ToolUseBlock, block)
            logger.debug(f"MiniMax tool_use start: id={tool_use_block.id}, name={tool_use_block.name}")
            yield {
                "type": "tool_call_delta",
                "index": block_index,
                "id": tool_use_block.id,
                "name": tool_use_block.name,
                "arguments_delta": "",
                "is_start": True,
                "is_end": False,
            }
        elif block.type == "text":
            yield {
                "type": "text_delta",
                "index": block_index,
                "delta": "",
                "is_start": True,
                "is_end": False,
            }
        elif block.type == "thinking":
            yield {
                "type": "reasoning_delta",
                "index": block_index,
                "delta": "",
                "is_start": True,
                "is_end": False,
            }
        elif block.type == "redacted_thinking":
            yield {
                "type": "reasoning_delta",
                "index": block_index,
                "delta": "[Redacted thinking content]",
                "is_start": True,
                "is_end": True,
            }
        else:
            logger.debug(f"MiniMax block type {block.type} in content_block_start event")
    
    def _handle_content_block_delta(
        self, event: RawContentBlockDeltaEvent
    ) -> Iterator[DeltaPart]:
        """处理 content_block_delta 事件"""
        block_index = event.index
        delta = event.delta
        
        delta_type = getattr(delta, 'type', None)
        
        if delta_type == "reasoning_delta" or isinstance(delta, ThinkingDelta):
            thinking = getattr(delta, 'thinking', '')
            yield {
                "type": "reasoning_delta",
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
) -> Iterator[DeltaPart]:
    """MiniMax 同步流式响应处理"""
    with client.messages.stream(**request) as stream:
        handler = MiniMaxAnthropicStreamHandler(stream)
        for event in stream:
            yield from handler.handle_event(event)


async def minimax_stream_response_async(
    client, request: dict[str, Any]
) -> AsyncGenerator[DeltaPart, None]:
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
            model_id="MiniMax-M2.7",
            api_key="sk-...",
            base_url="https://api.minimaxi.com/anthropic",
        )
    """

    def __init__(
        self,
        *,
        model_id: str = "MiniMax-M2.7",
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

    def list_models(self) -> list[str]:
        """Query MiniMax's Anthropic-compatible model-list endpoint."""
        return fetch_json_model_ids(
            self._models_endpoint_url(),
            provider="MiniMax",
            headers=bearer_auth_headers(self.api_key),
            params={"limit": 100},
            timeout=self.timeout,
            paginate=True,
        )

    async def alist_models(self) -> list[str]:
        """Async model-list query for MiniMax's Anthropic-compatible adapter."""
        return await afetch_json_model_ids(
            self._models_endpoint_url(),
            provider="MiniMax",
            headers=bearer_auth_headers(self.api_key),
            params={"limit": 100},
            timeout=self.timeout,
            paginate=True,
        )

    def _models_endpoint_url(self) -> str:
        base_url = (self.base_url or "https://api.minimaxi.com/anthropic").rstrip("/")
        if base_url.endswith("/models"):
            return base_url
        if base_url.endswith("/v1"):
            return f"{base_url}/models"
        return f"{base_url}/v1/models"

    def _estimate_tokens_impl(
        self,
        request: MessageRequest,
    ) -> TokenEstimate:
        estimate = self._heuristic_token_estimate(request)
        estimate.provider = "minimax"
        estimate.details["provider_count_endpoint"] = "not_available_in_official_docs"
        estimate.details["recommended_exact_source"] = "response.usage"
        return estimate

    async def _aestimate_tokens_impl(
        self,
        request: MessageRequest,
    ) -> TokenEstimate:
        return self._estimate_tokens_impl(request)

    def _stream_impl(self, request: MessageRequest) -> Iterator[DeltaPart]:
        """同步流式调用 - 使用 MiniMax 专属的 handler"""
        if needs_async_conversion(
            request.messages, self.enable_image_download
        ):
            try:
                yield from run_async_stream(self._astream_impl(request))
            except Exception as e:
                converted = _convert_minimax_anthropic_error(e)
                if converted is not e:
                    raise converted from e
                raise
            return

        req = self._prepare_request_sync(request)
        try:
            yield from minimax_stream_response(self.client, req)
        except Exception as e:
            converted = _convert_minimax_anthropic_error(e)
            if converted is not e:
                raise converted from e
            raise

    async def _astream_impl(
        self, request: MessageRequest
    ) -> AsyncGenerator[DeltaPart]:
        """异步流式调用 - 使用 MiniMax 专属的 handler"""
        req = await self._prepare_request_async(request)
        try:
            async for chunk in minimax_stream_response_async(self.async_client, req):
                yield chunk
        except Exception as e:
            converted = _convert_minimax_anthropic_error(e)
            if converted is not e:
                raise converted from e
            raise
