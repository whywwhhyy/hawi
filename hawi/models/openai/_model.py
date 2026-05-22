"""
OpenAI 模型实现

提供 OpenAI API 兼容的模型调用实现。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, AsyncGenerator, Iterator, cast

from openai import OpenAI, AsyncOpenAI

from hawi.models import Model
from hawi.models import (
    MessageRequest,
    MessageResponse,
    ContentPart,
    DeltaPart,
    DeltaTextPart,
    DeltaThinkingPart,
    DeltaToolCallPart,
    DeltaFinishPart,
    TextPart,
    ToolCallPart,
    ReasoningPart,
    TokenEstimate,
)
from hawi.errors import (
    NetworkError,
    RemoteError,
    ThrottleError,
    DeniedError,
    ValidationError,
    ContextLengthError,
    UnknownModelError,
)
from hawi.models._auth import normalize_optional_api_key, sdk_api_key
from hawi.models.usage import normalize_openai_usage
from ._converters import (
    prepare_request,
    convert_openai_content_to_part,
    convert_message_to_openai,
    map_stop_reason,
)
from ._streaming import StreamProcessor

logger = logging.getLogger(__name__)


_CONTEXT_LENGTH_RE = re.compile(
    r"maximum context length is\s+([\d,]+)\s+tokens.*?"
    r"requested\s+([\d,]+)\s+tokens"
    r"(?:\s+\(([\d,]+)\s+in the messages,\s+([\d,]+)\s+in the completion\))?",
    re.IGNORECASE | re.DOTALL,
)


def _int_from_token_text(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value.replace(",", ""))
    except ValueError:
        return None


def _openai_error_message(e: Exception) -> str:
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if body.get("message"):
            return str(body["message"])

    response = getattr(e, "response", None)
    json_method = getattr(response, "json", None)
    if callable(json_method):
        try:
            payload = json_method()
        except Exception:
            payload = None
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict) and error.get("message"):
                return str(error["message"])
            if payload.get("message"):
                return str(payload["message"])

    return str(e)


def _context_length_error_from_text(text: str) -> ContextLengthError | None:
    if not text:
        return None
    lower_text = text.lower()
    if (
        "maximum context length" not in lower_text
        and "context length" not in lower_text
    ):
        return None
    if "requested" not in lower_text and "reduce the length" not in lower_text:
        return None

    match = _CONTEXT_LENGTH_RE.search(text)
    if match:
        max_context_tokens = _int_from_token_text(match.group(1))
        requested_tokens = _int_from_token_text(match.group(2))
        message_tokens = _int_from_token_text(match.group(3))
        completion_tokens = _int_from_token_text(match.group(4))
    else:
        max_context_tokens = None
        requested_tokens = None
        message_tokens = None
        completion_tokens = None

    return ContextLengthError(
        f"Context length exceeded: {text}",
        max_context_tokens=max_context_tokens,
        requested_tokens=requested_tokens,
        message_tokens=message_tokens,
        completion_tokens=completion_tokens,
    )


def _convert_openai_error(e: Exception) -> Exception:
    """Convert OpenAI SDK errors to Hawi ModelError.
    
    This function converts OpenAI SDK exceptions to Hawi ModelError subclasses
    for consistent error handling across all OpenAI-compatible models.
    
    OpenAI Error Hierarchy:
        OpenAIError
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
        ├── ContentFilterFinishReasonError (content filtered)
        └── LengthFinishReasonError (max_tokens reached)
    
    Mapping to Hawi errors:
        - NetworkError: APIConnectionError, APITimeoutError (transport layer)
        - RemoteError: InternalServerError (5xx), other server-side errors
        - ThrottleError: RateLimitError (429)
        - DeniedError: AuthenticationError (401), PermissionDeniedError (403)
        - ValidationError: BadRequestError (400), UnprocessableEntityError (422),
          NotFoundError (404), ConflictError (409), APIResponseValidationError
        - UnknownModelError: other API errors
    
    Args:
        e: The OpenAI SDK exception.
        
    Returns:
        A Hawi ModelError subclass instance, or original exception if not convertible.
    """
    try:
        from openai import (
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
            ContentFilterFinishReasonError,
            LengthFinishReasonError,
        )
        
        # Special finish reason errors - these are not really errors but signals
        # that generation stopped for a specific reason. Re-raise as-is.
        if isinstance(e, (ContentFilterFinishReasonError, LengthFinishReasonError)):
            return e
        
        # Rate limit error (429) -> ThrottleError
        if isinstance(e, RateLimitError):
            return ThrottleError(f"Rate limit exceeded: {e}")
        
        # Authentication error (401) -> DeniedError
        if isinstance(e, AuthenticationError):
            return DeniedError(f"Authentication failed: {e}")
        
        # Permission denied (403) -> DeniedError
        if isinstance(e, PermissionDeniedError):
            return DeniedError(f"Permission denied: {e}")
        
        # Connection and timeout errors -> NetworkError (transport layer)
        if isinstance(e, (APIConnectionError, APITimeoutError)):
            return NetworkError(f"Connection error: {e}")
        
        # Internal server error (5xx) -> RemoteError (server-side)
        if isinstance(e, InternalServerError):
            return RemoteError(f"Server error: {e}")
        
        # Bad request (400) -> ContextLengthError or ValidationError
        if isinstance(e, BadRequestError):
            context_length_error = _context_length_error_from_text(
                _openai_error_message(e)
            )
            if context_length_error is not None:
                return context_length_error
            return ValidationError(f"Bad request: {e}")
        
        # Not found (404) -> ValidationError (usually invalid model ID)
        if isinstance(e, NotFoundError):
            return ValidationError(f"Resource not found: {e}")
        
        # Conflict (409) -> ValidationError
        if isinstance(e, ConflictError):
            return ValidationError(f"Conflict error: {e}")
        
        # Unprocessable entity (422) -> ValidationError
        if isinstance(e, UnprocessableEntityError):
            return ValidationError(f"Validation failed: {e}")
        
        # Response validation error -> ValidationError
        if isinstance(e, APIResponseValidationError):
            return ValidationError(f"Response validation error: {e}")
        
        # Generic APIStatusError with specific status code
        if isinstance(e, APIStatusError):
            status_code = getattr(e, 'status_code', None)
            if status_code == 401:
                return DeniedError(f"Authentication failed: {e}")
            if status_code == 403:
                return DeniedError(f"Permission denied: {e}")
            if status_code == 429:
                return ThrottleError(f"Rate limit exceeded: {e}")
            if status_code and 500 <= status_code < 600:
                return RemoteError(f"Server error ({status_code}): {e}")
        
        # Generic APIError with network-related keywords
        if isinstance(e, APIError):
            error_msg = str(e).lower()
            network_keywords = ['timeout', 'connection', 'network', 'dns', 'unreachable', 'refused']
            if any(kw in error_msg for kw in network_keywords):
                return NetworkError(f"Network error: {e}")
            
            # Default: unknown model error
            return UnknownModelError(f"API error: {e}")
        
        # Not an OpenAI error, return as-is
        return e
        
    except ImportError:
        # openai module not available, return original exception
        return e


class OpenAIModel(Model):
    """OpenAI API 兼容模型

    支持 OpenAI 官方 API 及兼容 OpenAI 格式的第三方 API。

    Example:
        from hawi.models.openai import OpenAIModel

        model = OpenAIModel(
            model_id="gpt-4",
            api_key="sk-...",
        )
        response = model.invoke(messages=[{"role": "user", "content": [{"type": "text", "text": "Hello"}], "name": None, "tool_calls": None, "tool_call_id": None, "metadata": None}])
    """

    default_steer_merge_mode = "tool_result_assistant_template_and_user_message"

    def __init__(
        self,
        *,
        model_id: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        require_usage: bool = True,
        include_reasoning_in_context: bool = False,
        include_reasoning_in_tool_calls: bool | None = None,
        default_tool_call_reasoning_content: str = "",
        **params,
    ):
        """初始化 OpenAI 模型

        Args:
            model_id: 模型标识符，如 "gpt-4"
            api_key: API 密钥
            base_url: API 基础 URL
            timeout: 请求超时时间
            max_retries: 最大重试次数
            require_usage: 是否要求获取 token 使用量（用于计费），默认 True
            include_reasoning_in_context: 是否将历史 assistant 消息中的
                reasoning part/metadata 作为 OpenAI-compatible
                reasoning_content 字段回传
            include_reasoning_in_tool_calls: 是否在 assistant tool_call 历史
                中回传 reasoning_content。None 表示跟随
                include_reasoning_in_context
            default_tool_call_reasoning_content: tool_call 历史没有
                reasoning 时回传的默认 reasoning_content
            **params: 其他参数，如 temperature, max_tokens 等
        """
        self._model_id = model_id
        self.api_key = normalize_optional_api_key(api_key)
        self._api_key_explicit = api_key is not None
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.require_usage = require_usage
        self.include_reasoning_in_context = include_reasoning_in_context
        self.include_reasoning_in_tool_calls = (
            include_reasoning_in_context
            if include_reasoning_in_tool_calls is None
            else include_reasoning_in_tool_calls
        )
        self.default_tool_call_reasoning_content = (
            default_tool_call_reasoning_content
        )
        self.params = params
        self._client: OpenAI | None = None
        self._async_client: AsyncOpenAI | None = None

    @property
    def model_id(self) -> str:
        """模型标识符"""
        return self._model_id

    def reset(self) -> None:
        """重置模型状态，关闭并清除缓存的客户端连接。"""
        super().reset()
        # Close and clear cached clients
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        if self._async_client is not None:
            try:
                # Async client may need async close, but we can't await here
                # The client will be garbage collected
                pass
            except Exception:
                pass
            self._async_client = None

    @property
    def client(self) -> OpenAI:
        """获取或创建 OpenAI 客户端"""
        if self._client is None:
            client_args: dict[str, Any] = {
                "timeout": self.timeout,
                "max_retries": self.max_retries,
            }
            api_key = sdk_api_key(
                self.api_key,
                base_url=self.base_url,
                explicit_api_key=self._api_key_explicit,
            )
            if api_key is not None:
                client_args["api_key"] = api_key
            if self.base_url:
                client_args["base_url"] = self.base_url
            self._client = OpenAI(**client_args)
        return self._client

    @property
    def async_client(self) -> AsyncOpenAI:
        """获取或创建 OpenAI 异步客户端"""
        if self._async_client is None:
            client_args: dict[str, Any] = {
                "timeout": self.timeout,
                "max_retries": self.max_retries,
            }
            api_key = sdk_api_key(
                self.api_key,
                base_url=self.base_url,
                explicit_api_key=self._api_key_explicit,
            )
            if api_key is not None:
                client_args["api_key"] = api_key
            if self.base_url:
                client_args["base_url"] = self.base_url
            self._async_client = AsyncOpenAI(**client_args)
        return self._async_client

    def _get_params(self) -> dict[str, Any]:
        """获取模型参数（temperature, max_tokens 等）"""
        return self.params

    # ==================================================================
    # 请求/响应转换
    # ==================================================================

    def _convert_message_to_openai(self, message) -> list[dict[str, Any]]:
        """将通用消息转换为 OpenAI 格式（子类可覆盖）

        Args:
            message: 通用消息

        Returns:
            OpenAI 格式的消息字典列表
        """
        results = convert_message_to_openai(message)
        for result in results:
            if result.get("role") == "assistant":
                self._apply_reasoning_content(result, message)
        return results

    def _apply_reasoning_content(
        self,
        result: dict[str, Any],
        message: dict[str, Any],
    ) -> None:
        """Attach reasoning_content to assistant history when configured."""
        reasoning, has_reasoning = self._extract_request_reasoning(message)
        has_tool_calls = bool(result.get("tool_calls"))

        if has_tool_calls and self.include_reasoning_in_tool_calls:
            result["reasoning_content"] = (
                reasoning
                if has_reasoning
                else self.default_tool_call_reasoning_content
            )
            return

        if self.include_reasoning_in_context and has_reasoning:
            result["reasoning_content"] = reasoning

    def _extract_request_reasoning(
        self,
        message: dict[str, Any],
    ) -> tuple[str, bool]:
        """Extract reasoning_content from Hawi message content or metadata."""
        for part in message.get("content", []):
            if isinstance(part, dict) and part.get("type") == "reasoning":
                return part.get("reasoning") or "", True

        metadata = message.get("metadata")
        if isinstance(metadata, dict) and "reasoning_content" in metadata:
            return str(metadata.get("reasoning_content") or ""), True

        return "", False

    def _prepare_request_impl(self, request: MessageRequest) -> dict[str, Any]:
        """将通用请求转换为 OpenAI 格式"""
        return prepare_request(
            request=request,
            model_id=self.model_id,
            params=self.params,
            converter=self._convert_message_to_openai,
        )

    def _parse_response_impl(self, response: dict[str, Any]) -> MessageResponse:
        """将 OpenAI 响应转换为通用格式"""
        import json

        choice = response["choices"][0]
        message = choice.get("message") or {}

        content: list[ContentPart] = []

        # 处理 reasoning_content (OpenAI o1, o3 系列推理模型)
        reasoning_content, server_reasoning_present = (
            self._extract_response_reasoning(message)
        )

        # 处理消息内容
        msg_content = message.get("content")
        if msg_content:
            if isinstance(msg_content, str):
                # 对于结构化输出 (JSON mode)，尝试解析 JSON
                content.append({"type": "text", "text": msg_content})
            elif isinstance(msg_content, list):
                for part in msg_content:
                    content.extend(convert_openai_content_to_part(part))

        # 处理 tool_calls
        tool_calls = message.get("tool_calls") or []
        for tc in tool_calls:
            if tc.get("type") == "function":
                func = tc["function"]
                arguments = func.get("arguments", "{}")
                try:
                    parsed_args = json.loads(arguments) if isinstance(arguments, str) else arguments
                except json.JSONDecodeError:
                    logger.warning("Failed to parse tool call arguments: %s", arguments)
                    parsed_args = {}
                content.append({
                    "type": "tool_call",
                    "id": tc["id"],
                    "name": func["name"],
                    "arguments": parsed_args,
                })

        # 转换 usage (支持 prompt caching / reasoning token details)
        usage = normalize_openai_usage(response.get("usage"))

        # 解析 refusal (模型拒绝回答的情况)
        refusal = message.get("refusal")
        if refusal and not content:
            content.append({"type": "text", "text": f"[Refused: {refusal}]"})

        msg_response = MessageResponse(
            id=response["id"],
            content=content,
            stop_reason=map_stop_reason(choice.get("finish_reason")),
            usage=usage,
            reasoning_content=(
                reasoning_content if server_reasoning_present else None
            ),
        )
        if self._should_include_response_reasoning_part(
            server_reasoning_present=server_reasoning_present
        ):
            self._ensure_response_reasoning_part(
                msg_response,
                reasoning_content,
            )

        return msg_response

    def _extract_response_reasoning(
        self,
        message: dict[str, Any],
    ) -> tuple[str, bool]:
        """Extract reasoning from an OpenAI-compatible response message."""
        if "reasoning_content" in message:
            return message.get("reasoning_content") or "", True

        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") in {"reasoning", "thinking"}:
                    return (
                        block.get("reasoning")
                        or block.get("thinking")
                        or block.get("text")
                        or "",
                        True,
                    )

        return "", False

    def _should_include_response_reasoning_part(
        self,
        *,
        server_reasoning_present: bool,
    ) -> bool:
        """Whether response.reasoning_content should also become a content part."""
        return server_reasoning_present

    def _ensure_response_reasoning_part(
        self,
        response: MessageResponse,
        reasoning: str | None,
    ) -> None:
        """Ensure response.content starts with a reasoning part."""
        reasoning_text = reasoning or ""
        content = list(response.content)

        for part in content:
            if part.get("type") == "reasoning":
                rp = cast(ReasoningPart, part)
                rp["reasoning"] = rp.get("reasoning") or ""
                response.reasoning_content = rp["reasoning"]
                response.content = content
                return

        reasoning_part: ReasoningPart = {
            "type": "reasoning",
            "reasoning": reasoning_text,
            "signature": None,
        }
        response.reasoning_content = reasoning_text
        response.content = [reasoning_part] + content

    # ==================================================================
    # 调用实现
    # ==================================================================

    def _invoke_impl(
        self,
        request: MessageRequest,
    ) -> MessageResponse:
        """同步调用 OpenAI API

        Args:
            request: 消息请求

        Returns:
            MessageResponse: 完整的模型响应
        """
        req = self._prepare_request_impl(request)
        try:
            response = self.client.chat.completions.create(**req)
            return self._parse_response_impl(response.model_dump())
        except Exception as e:
            converted = _convert_openai_error(e)
            if converted is not e:
                raise converted from e
            raise

    def _estimate_tokens_impl(
        self,
        request: MessageRequest,
    ) -> TokenEstimate:
        estimate = super()._estimate_tokens_impl(request)
        estimate.provider = "openai_compatible"
        estimate.details["provider_count_endpoint"] = "not_available_in_official_docs"
        estimate.details["recommended_exact_source"] = "response.usage"
        return estimate

    def list_models(self) -> list[str]:
        """Query the OpenAI-compatible ``/models`` endpoint."""
        try:
            response = self.client.models.list()
        except Exception as e:
            converted = _convert_openai_error(e)
            if converted is not e:
                raise converted from e
            raise
        return self._coerce_model_id_list(response)

    async def alist_models(self) -> list[str]:
        """Async model-list query for OpenAI-compatible providers."""
        try:
            response = await self.async_client.models.list()
        except Exception as e:
            converted = _convert_openai_error(e)
            if converted is not e:
                raise converted from e
            raise
        return await self._acoerce_model_id_list(response)

    def _prepare_stream_request(self, request: MessageRequest) -> dict[str, Any]:
        """准备流式请求的通用配置

        子类可以调用此方法获取基础请求配置，然后添加自己的特殊处理。

        Args:
            request: 消息请求

        Returns:
            包含 stream=True 的请求字典
        """
        req = self._prepare_request_impl(request)
        req["stream"] = True
        # 根据 require_usage 设置 stream_options
        if self.require_usage:
            req["stream_options"] = {"include_usage": True}
        return req

    def _stream_impl(self, request: MessageRequest) -> Iterator[DeltaPart]:
        """同步流式调用 OpenAI API"""
        req = self._prepare_stream_request(request)

        processor = StreamProcessor(expect_usage=self.require_usage)

        try:
            stream = self.client.chat.completions.create(**req)
        except Exception as e:
            converted = _convert_openai_error(e)
            if converted is not e:
                raise converted from e
            raise

        for chunk in stream:
            chunk_dict = chunk.model_dump()
            yield from processor.process_chunk(chunk_dict)
        yield from processor.finalize()

    async def _ainvoke_impl(
        self,
        request: MessageRequest,
    ) -> AsyncGenerator[DeltaPart, None]:
        """异步非流式调用 OpenAI API - 将完整响应拆分为 DeltaPart 序列

        Args:
            request: 消息请求

        Yields:
            DeltaPart 增量块序列
        """
        import asyncio

        # 在线程池中执行同步 API 调用
        loop = asyncio.get_event_loop()
        req = self._prepare_request_impl(request)

        try:
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(**req)
            )
        except Exception as e:
            converted = _convert_openai_error(e)
            if converted is not e:
                raise converted from e
            raise

        result = self._parse_response_impl(response.model_dump())

        # Yield content blocks as DeltaPart sequence
        for idx, part in enumerate(result.content):
            part_type = part["type"]

            if part_type == "text":
                text_part = cast(TextPart, part)
                yield DeltaTextPart(
                    type="text_delta",
                    index=idx,
                    delta=text_part["text"],
                    is_start=True,
                    is_end=True,
                )

            elif part_type == "reasoning":
                reasoning_part = cast(ReasoningPart, part)
                yield DeltaThinkingPart(
                    type="reasoning_delta",
                    index=idx,
                    delta=reasoning_part.get("reasoning") or "",
                    is_start=True,
                    is_end=True,
                )

            elif part_type == "tool_call":
                tool_part = cast(ToolCallPart, part)
                yield DeltaToolCallPart(
                    type="tool_call_delta",
                    index=idx,
                    id=tool_part["id"],
                    name=tool_part["name"],
                    arguments_delta=json.dumps(tool_part["arguments"]),
                    is_start=True,
                    is_end=True,
                )

        # Yield finish part with usage
        yield DeltaFinishPart(
            type="finish",
            stop_reason=result.stop_reason or "end_turn",
            usage=result.usage,
        )

    async def _astream_impl(
        self, request: MessageRequest
    ) -> AsyncGenerator[DeltaPart, None]:
        """异步流式调用 OpenAI API - 实时转发 DeltaPart

        Args:
            request: 消息请求

        Yields:
            DeltaPart 流式增量块
        """
        req = self._prepare_stream_request(request)

        processor = StreamProcessor(expect_usage=self.require_usage)

        try:
            # OpenAI async streaming: await the coroutine first, then use async with
            stream = await self.async_client.chat.completions.create(**req)

            async with stream:
                async for chunk in stream:
                    chunk_dict = chunk.model_dump()
                    for delta_part in processor.process_chunk(chunk_dict):
                        yield delta_part
                for delta_part in processor.finalize():
                    yield delta_part
        except Exception as e:
            converted = _convert_openai_error(e)
            if converted is not e:
                raise converted from e
            raise
