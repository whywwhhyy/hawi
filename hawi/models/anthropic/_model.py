"""
Anthropic API 兼容模型主类
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Iterator
from typing import Any
import json

from anthropic import Anthropic, AsyncAnthropic

from hawi.models import Model
from hawi.models import (
    ContentPart,
    MessageRequest,
    MessageResponse,
    DeltaPart,
    TextPart,
    TokenUsage,
    ToolCallPart,
    ReasoningPart,
)
from hawi.errors import (
    NetworkError,
    RemoteError,
    ThrottleError,
    DeniedError,
    ValidationError,
    UnknownModelError,
)
from ._converters import (
    AsyncContentConverter,
    ContentConverter,
    needs_async_conversion,
)
from ._streaming import (
    run_async_stream,
    stream_response,
    stream_response_async,
    _AnthropicStreamHandler,
)
from ._utils import convert_system_prompt, map_stop_reason

logger = logging.getLogger(__name__)


def _convert_anthropic_error(e: Exception) -> Exception:
    """Convert Anthropic SDK errors to Hawi ModelError.
    
    This function converts Anthropic SDK exceptions to Hawi ModelError subclasses
    for consistent error handling across all model implementations.
    
    Anthropic Error Hierarchy:
        AnthropicError
        ├── APIError
        │   ├── APIConnectionError (network connection failed)
        │   │   └── APITimeoutError (request timeout)
        │   ├── APIResponseValidationError (invalid response from API)
        │   └── APIStatusError (HTTP status code based)
        │       ├── BadRequestError (400)
        │       ├── AuthenticationError (401)
        │       ├── PermissionDeniedError (403)
        │       ├── NotFoundError (404)
        │       ├── ConflictError (409)
        │       ├── UnprocessableEntityError (422)
        │       ├── RateLimitError (429)
        │       └── InternalServerError (5xx)
    
    Mapping to Hawi errors:
        - NetworkError: APIConnectionError, APITimeoutError (transport layer)
        - RemoteError: InternalServerError (5xx), other server-side errors
        - ThrottleError: RateLimitError (429)
        - DeniedError: AuthenticationError (401), PermissionDeniedError (403)
        - ValidationError: BadRequestError (400), UnprocessableEntityError (422),
          NotFoundError (404), ConflictError (409), APIResponseValidationError
        - UnknownModelError: other API errors
    
    Args:
        e: The Anthropic SDK exception.
        
    Returns:
        A Hawi ModelError subclass instance, or original exception if not convertible.
    """
    try:
        from anthropic import (
            APIError,
            APIConnectionError,
            APITimeoutError,
            APIResponseValidationError,
            APIStatusError,
            AuthenticationError,
            BadRequestError,
            ConflictError,
            InternalServerError,
            NotFoundError,
            PermissionDeniedError,
            RateLimitError,
            UnprocessableEntityError,
        )
        
        # Rate limit error (429) -> ThrottleError
        if isinstance(e, RateLimitError):
            return ThrottleError(f"Anthropic rate limit exceeded: {e}")
        
        # Authentication error (401) -> DeniedError
        if isinstance(e, AuthenticationError):
            return DeniedError(f"Anthropic authentication failed: {e}")
        
        # Permission denied (403) -> DeniedError
        if isinstance(e, PermissionDeniedError):
            return DeniedError(f"Anthropic permission denied: {e}")
        
        # Connection and timeout errors -> NetworkError (transport layer)
        if isinstance(e, (APIConnectionError, APITimeoutError)):
            return NetworkError(f"Anthropic connection error: {e}")
        
        # Internal server error (5xx) -> RemoteError (server-side)
        if isinstance(e, InternalServerError):
            return RemoteError(f"Anthropic server error: {e}")
        
        # Bad request (400) -> ValidationError
        if isinstance(e, BadRequestError):
            return ValidationError(f"Anthropic bad request: {e}")
        
        # Not found (404) -> ValidationError (usually invalid model ID)
        if isinstance(e, NotFoundError):
            return ValidationError(f"Anthropic resource not found: {e}")
        
        # Conflict (409) -> ValidationError
        if isinstance(e, ConflictError):
            return ValidationError(f"Anthropic conflict error: {e}")
        
        # Unprocessable entity (422) -> ValidationError
        if isinstance(e, UnprocessableEntityError):
            return ValidationError(f"Anthropic validation failed: {e}")
        
        # Response validation error -> ValidationError
        if isinstance(e, APIResponseValidationError):
            return ValidationError(f"Anthropic response validation error: {e}")
        
        # Generic APIStatusError with specific status code
        if isinstance(e, APIStatusError):
            status_code = getattr(e, 'status_code', None)
            if status_code == 401:
                return DeniedError(f"Anthropic authentication failed: {e}")
            if status_code == 403:
                return DeniedError(f"Anthropic permission denied: {e}")
            if status_code == 429:
                return ThrottleError(f"Anthropic rate limit exceeded: {e}")
            if status_code and 500 <= status_code < 600:
                return RemoteError(f"Anthropic server error ({status_code}): {e}")
        
        # Generic APIError with network-related keywords
        if isinstance(e, APIError):
            error_msg = str(e).lower()
            network_keywords = ['timeout', 'connection', 'network', 'dns', 'unreachable', 'refused']
            if any(kw in error_msg for kw in network_keywords):
                return NetworkError(f"Anthropic network error: {e}")
            
            # Default: unknown model error
            return UnknownModelError(f"Anthropic API error: {e}")
        
        # Not an Anthropic error, return as-is
        return e
        
    except ImportError:
        # anthropic module not available, return original exception
        return e


class AnthropicModel(Model):
    """
    Anthropic API 兼容模型

    支持 Claude 系列模型，包括：
    - 文本、图片、文档输入
    - Tool use / tool result
    - Prompt caching (cache_control)
    - 流式响应
    - 远程图片自动下载（异步）

    Example:
        model = AnthropicModel(
            model_id="claude-3-5-sonnet-20241022",
            api_key="sk-ant-...",
        )
        response = model.invoke(messages=[create_user_message("Hello")])
    """

    def __init__(
        self,
        *,
        model_id: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        enable_image_download: bool = True,
        thinking_budget: int | None = 8000,
        max_output_tokens: int | None = None,
        **params,
    ):
        """
        初始化 Anthropic 模型

        Args:
            model_id: 模型标识符，如 "claude-3-5-sonnet-20241022"
            api_key: API 密钥
            base_url: API 基础 URL
            timeout: 请求超时时间
            max_retries: 最大重试次数
            enable_image_download: 是否允许下载远程图片转为 base64
            thinking_budget: thinking 模式的 token 预算，0 或 None 表示禁用，默认 8000
            max_output_tokens: 最大输出 token 数，默认 None（使用 API 默认值或请求中的值）
            **params: 其他参数，如 temperature, max_output_tokens 等
        """
        self._model_id = model_id
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.enable_image_download = enable_image_download
        self.thinking_budget = thinking_budget
        self.max_output_tokens = max_output_tokens
        self.params = params
        self._client: Anthropic | None = None
        self._async_client: AsyncAnthropic | None = None

        # Clear env vars that may interfere with API calls when api_key is provided
        # Anthropic SDK reads ANTHROPIC_AUTH_TOKEN which can cause wrong API endpoint
        if api_key:
            import os
            os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
            os.environ.pop("ANTHROPIC_BASE_URL", None)

        # 初始化转换器
        self._converter = ContentConverter(enable_image_download)
        self._async_converter = AsyncContentConverter(enable_image_download)

    @property
    def model_id(self) -> str:
        """模型标识符"""
        return self._model_id

    @property
    def client(self) -> Anthropic:
        """获取或创建 Anthropic 客户端"""
        if self._client is None:
            client_args: dict[str, Any] = {
                "timeout": self.timeout,
                "max_retries": self.max_retries,
            }
            if self.api_key:
                client_args["api_key"] = self.api_key
            if self.base_url:
                client_args["base_url"] = self.base_url
            self._client = Anthropic(**client_args)
        return self._client

    @property
    def async_client(self) -> AsyncAnthropic:
        """获取或创建 Anthropic 异步客户端"""
        if self._async_client is None:
            client_args: dict[str, Any] = {
                "timeout": self.timeout,
                "max_retries": self.max_retries,
            }
            if self.api_key:
                client_args["api_key"] = self.api_key
            if self.base_url:
                client_args["base_url"] = self.base_url
            self._async_client = AsyncAnthropic(**client_args)
        return self._async_client

    def _get_params(self) -> dict[str, Any]:
        """获取模型参数"""
        return self.params

    # =======================================================================
    # 请求准备
    # =======================================================================

    def _prepare_request_impl(self, request: MessageRequest) -> dict[str, Any]:
        """将通用请求转换为 Anthropic 格式"""
        if needs_async_conversion(
            request.messages, self.enable_image_download
        ):
            return asyncio.run(self._prepare_request_async(request))
        return self._prepare_request_sync(request)

    def _prepare_request_sync(self, request: MessageRequest) -> dict[str, Any]:
        """同步请求准备"""
        anthropic_messages:list[dict[str,Any]] = []
        for m in request.messages:
            if m["role"] != "system":
                anthropic_message = self._converter.convert_message(m)
                if anthropic_message:
                    anthropic_messages.append(anthropic_message)

        return self._build_anthropic_request(
            messages=anthropic_messages,
            request=request,
        )

    async def _prepare_request_async(
        self, request: MessageRequest
    ) -> dict[str, Any]:
        """异步请求准备（支持图片下载）"""
        anthropic_messages = []
        for m in request.messages:
            if m["role"] == "system":
                continue
            msg = await self._async_converter.convert_message_async(m)
            if msg is not None:
                anthropic_messages.append(msg)

        return self._build_anthropic_request(
            messages=anthropic_messages,
            request=request,
        )

    def _build_anthropic_request(
        self,
        messages: list[dict[str,Any]],
        request: MessageRequest,
    ) -> dict[str, Any]:
        """构建 Anthropic API 请求"""
        # max_output_tokens: 请求级 > 实例级 > 默认值
        effective_max_tokens = request.max_output_tokens or self.max_output_tokens or 4096

        req: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": effective_max_tokens,
        }

        # System 内容 - Anthropic 使用顶级 system 字段
        system = convert_system_prompt(request.system)
        if system:
            req["system"] = system

        # 工具定义 (扁平格式: name, description, schema)
        if request.tools:
            req["tools"] = [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "input_schema": t["schema"],
                }
                for t in request.tools
            ]

        # 工具选择
        if request.tool_choice:
            tc = request.tool_choice
            tool_choice: dict[str, Any] = {"type": tc["type"]}
            if tc["type"] == "tool" and tc.get("name"):
                tool_choice["name"] = tc["name"]
            # 支持 disable_parallel_tool_use
            if request.parallel_tool_calls is not None:
                tool_choice["disable_parallel_tool_use"] = not request.parallel_tool_calls
            req["tool_choice"] = tool_choice

        # Thinking 模式 (thinking_budget 为 0 或 None 时禁用)
        # 优先级: 请求级 > 实例级
        effective_thinking_budget = request.thinking_budget if request.thinking_budget is not None else self.thinking_budget
        if effective_thinking_budget:
            req["thinking"] = {
                "type": "enabled",
                "budget_tokens": effective_thinking_budget,
            }

        # 可选参数
        if request.temperature is not None:
            req["temperature"] = request.temperature
        if request.top_p is not None:
            req["top_p"] = request.top_p
        if request.top_k is not None:
            req["top_k"] = request.top_k
        if request.stop_sequences is not None:
            req["stop_sequences"] = request.stop_sequences
        if request.metadata is not None:
            req["metadata"] = request.metadata

        return req

    # =======================================================================
    # 响应解析
    # =======================================================================

    def _parse_response_impl(
        self, response: dict[str, Any]
    ) -> MessageResponse:
        """将 Anthropic 响应转换为通用格式"""
        content = response.get("content", [])
        usage = response.get("usage", {})

        # 解析内容块
        parts: list[ContentPart] = []
        for block in content:
            block_type = block.get("type")

            if block_type == "text":
                parts.append(TextPart(type="text", text=block.get("text", "")))
            elif block_type == "tool_use":
                parts.append(
                    ToolCallPart(
                        type="tool_call",
                        id=block.get("id", ""),
                        name=block.get("name", ""),
                        arguments=block.get("input", {}),
                    )
                )
            elif block_type == "reasoning":
                parts.append(
                    ReasoningPart(
                        type="reasoning",
                        reasoning=block.get("thinking", ""),
                        signature=block.get("signature"),
                    )
                )
            elif block_type == "redacted_thinking":
                # Redacted thinking blocks contain sensitive reasoning
                # We include them as reasoning parts but mark as redacted
                parts.append(
                    ReasoningPart(
                        type="reasoning",
                        reasoning="[Redacted thinking block]",
                        signature=block.get("data"),
                    )
                )

        return MessageResponse(
            id=response.get("id", ""),
            content=parts,
            stop_reason=map_stop_reason(response.get("stop_reason")),
            usage=TokenUsage(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                cache_write_tokens=usage.get("cache_creation_input_tokens"),
                cache_read_tokens=usage.get("cache_read_input_tokens"),
            )
            if usage
            else None,
        )

    # =======================================================================
    # 调用实现
    # =======================================================================

    def _invoke_impl(
        self,
        request: MessageRequest,
    ) -> MessageResponse:
        """同步调用 Anthropic API"""
        if needs_async_conversion(
            request.messages, self.enable_image_download
        ):
            try:
                return asyncio.run(self._async_invoke_impl(request))
            except Exception as e:
                converted = _convert_anthropic_error(e)
                if converted is not e:
                    raise converted from e
                raise

        req = self._prepare_request_sync(request)
        try:
            response = self.client.messages.create(**req)
            result = self._parse_response_impl(response.model_dump())
            return result
        except Exception as e:
            converted = _convert_anthropic_error(e)
            if converted is not e:
                raise converted from e
            raise

    async def _async_invoke_impl(
        self,
        request: MessageRequest,
    ) -> MessageResponse:
        """异步辅助方法，用于同步调用中的异步转换"""
        req = await self._prepare_request_async(request)
        try:
            response = await self.async_client.messages.create(**req)
            return self._parse_response_impl(response.model_dump())
        except Exception as e:
            converted = _convert_anthropic_error(e)
            if converted is not e:
                raise converted from e
            raise

    async def _ainvoke_impl(
        self,
        request: MessageRequest,
    ) -> AsyncGenerator[DeltaPart, None]:
        """异步非流式调用 Anthropic API - 将完整响应拆分为 DeltaPart 序列

        Args:
            request: 消息请求

        Yields:
            DeltaPart 增量块序列
        """
        from typing import cast

        req = await self._prepare_request_async(request)
        try:
            response = await self.async_client.messages.create(**req)
        except Exception as e:
            converted = _convert_anthropic_error(e)
            if converted is not e:
                raise converted from e
            raise
        
        result = self._parse_response_impl(response.model_dump())

        # Yield content blocks as DeltaPart sequence
        for idx, part in enumerate(result.content):
            part_type = part["type"]

            if part_type == "text":
                text_part = cast(TextPart, part)
                from hawi.models.message import DeltaTextPart
                yield DeltaTextPart(
                    type="text_delta",
                    index=idx,
                    delta=text_part["text"],
                    is_start=True,
                    is_end=True,
                )

            elif part_type == "reasoning":
                reasoning_part = cast(ReasoningPart, part)
                from hawi.models.message import DeltaThinkingPart
                yield DeltaThinkingPart(
                    type="reasoning_delta",
                    index=idx,
                    delta=reasoning_part.get("reasoning") or "",
                    is_start=True,
                    is_end=True,
                )

            elif part_type == "tool_call":
                tool_part = cast(ToolCallPart, part)
                from hawi.models.message import DeltaToolCallPart
                yield DeltaToolCallPart(
                    type="tool_call_delta",
                    index=idx,
                    id=tool_part["id"],
                    name=tool_part["name"],
                    arguments_delta=json.dumps(tool_part["arguments"]),
                    is_start=True,
                    is_end=True,
                )

        # Yield finish part
        from hawi.models.message import DeltaFinishPart
        yield DeltaFinishPart(
            type="finish",
            stop_reason=result.stop_reason or "end_turn",
            usage=result.usage,
        )

    def _stream_impl(self, request: MessageRequest) -> Iterator[DeltaPart]:
        """同步流式调用"""
        if needs_async_conversion(
            request.messages, self.enable_image_download
        ):
            # Filter out Event types, only yield DeltaPart for sync streaming
            async def _filtered_stream():
                async for item in self._astream_impl(request):
                    if isinstance(item, dict):  # DeltaPart is a dict
                        yield item

            try:
                yield from run_async_stream(_filtered_stream())
            except Exception as e:
                converted = _convert_anthropic_error(e)
                if converted is not e:
                    raise converted from e
                raise
            return

        req = self._prepare_request_sync(request)
        try:
            yield from stream_response(self.client, req)
        except Exception as e:
            converted = _convert_anthropic_error(e)
            if converted is not e:
                raise converted from e
            raise

    async def _astream_impl(
        self, request: MessageRequest
    ) -> AsyncGenerator[DeltaPart, None]:
        """异步流式调用 - 实时转发 DeltaPart

        Args:
            request: 消息请求

        Yields:
            DeltaPart 流式增量块
        """
        req = await self._prepare_request_async(request)

        try:
            async with self.async_client.messages.stream(**req) as stream:
                handler = _AnthropicStreamHandler(stream)
                async for event in stream:
                    for delta_part in handler.handle_event(event):
                        yield delta_part
                yield handler._create_finish_part()
        except Exception as e:
            converted = _convert_anthropic_error(e)
            if converted is not e:
                raise converted from e
            raise
