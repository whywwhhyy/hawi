"""
OpenAI 模型实现

提供 OpenAI API 兼容的模型调用实现。
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Iterator

from openai import OpenAI, AsyncOpenAI

from hawi.agent.model import Model
from hawi.agent.message import (
    MessageRequest,
    MessageResponse,
    TokenUsage,
    ContentPart,
    StreamPart,
)
from ._converters import (
    prepare_request,
    convert_openai_content_to_part,
    convert_message_to_openai,
    map_stop_reason,
)
from ._streaming import StreamProcessor

logger = logging.getLogger(__name__)


class OpenAIModel(Model):
    """OpenAI API 兼容模型

    支持 OpenAI 官方 API 及兼容 OpenAI 格式的第三方 API。

    Example:
        from hawi.agent.models.openai import OpenAIModel

        model = OpenAIModel(
            model_id="gpt-4",
            api_key="sk-...",
        )
        response = model.invoke(messages=[{"role": "user", "content": [{"type": "text", "text": "Hello"}], "name": None, "tool_calls": None, "tool_call_id": None, "metadata": None}])
    """

    def __init__(
        self,
        *,
        model_id: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        **params,
    ):
        """初始化 OpenAI 模型

        Args:
            model_id: 模型标识符，如 "gpt-4"
            api_key: API 密钥
            base_url: API 基础 URL
            timeout: 请求超时时间
            max_retries: 最大重试次数
            **params: 其他参数，如 temperature, max_tokens 等
        """
        self._model_id = model_id
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.params = params
        self._client: OpenAI | None = None
        self._async_client: AsyncOpenAI | None = None

    @property
    def model_id(self) -> str:
        """模型标识符"""
        return self._model_id

    @property
    def client(self) -> OpenAI:
        """获取或创建 OpenAI 客户端"""
        if self._client is None:
            client_args: dict[str, Any] = {
                "timeout": self.timeout,
                "max_retries": self.max_retries,
            }
            if self.api_key:
                client_args["api_key"] = self.api_key
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
            if self.api_key:
                client_args["api_key"] = self.api_key
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

    def _convert_message_to_openai(self, message) -> dict[str, Any]:
        """将通用消息转换为 OpenAI 格式（子类可覆盖）

        Args:
            message: 通用消息

        Returns:
            OpenAI 格式的消息字典
        """
        return convert_message_to_openai(message)

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
        reasoning_content = message.get("reasoning_content")

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

        # 转换 usage (支持 prompt caching 字段)
        usage_data = response.get("usage")
        if usage_data:
            usage = TokenUsage(
                input_tokens=usage_data.get("prompt_tokens", 0),
                output_tokens=usage_data.get("completion_tokens", 0),
                cache_creation_input_tokens=usage_data.get(
                    "prompt_cache_creation_tokens"
                ),
                cache_read_input_tokens=usage_data.get(
                    "prompt_cache_read_tokens"
                ),
            )
        else:
            usage = None

        # 解析 refusal (模型拒绝回答的情况)
        refusal = message.get("refusal")
        if refusal and not content:
            content.append({"type": "text", "text": f"[Refused: {refusal}]"})

        return MessageResponse(
            id=response["id"],
            content=content,
            stop_reason=map_stop_reason(choice.get("finish_reason")),
            usage=usage,
            reasoning_content=reasoning_content,
        )

    # ==================================================================
    # 调用实现
    # ==================================================================

    def _invoke_impl(self, request: MessageRequest) -> MessageResponse:
        """同步调用 OpenAI API"""
        req = self._prepare_request_impl(request)
        response = self.client.chat.completions.create(**req)
        return self._parse_response_impl(response.model_dump())

    def _stream_impl(self, request: MessageRequest) -> Iterator[StreamPart]:
        """同步流式调用 OpenAI API"""
        req = self._prepare_request_impl(request)
        req["stream"] = True
        req["stream_options"] = {"include_usage": True}

        processor = StreamProcessor()

        for chunk in self.client.chat.completions.create(**req):
            chunk_dict = chunk.model_dump()
            yield from processor.process_chunk(chunk_dict)

    async def _ainvoke_impl(self, request: MessageRequest) -> MessageResponse:
        """异步调用 OpenAI API"""
        req = self._prepare_request_impl(request)
        response = await self.async_client.chat.completions.create(**req)
        return self._parse_response_impl(response.model_dump())

    async def _astream_impl(
        self, request: MessageRequest
    ) -> AsyncIterator[StreamPart]:
        """异步流式调用 OpenAI API"""
        req = self._prepare_request_impl(request)
        req["stream"] = True
        req["stream_options"] = {"include_usage": True}

        processor = StreamProcessor()

        # OpenAI async streaming: await the coroutine first, then use async with
        stream = await self.async_client.chat.completions.create(**req)
        async with stream:
            async for chunk in stream:
                chunk_dict = chunk.model_dump()
                for event in processor.process_chunk(chunk_dict):
                    yield event
