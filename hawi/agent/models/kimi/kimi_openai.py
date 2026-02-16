"""
Kimi/Moonshot API 兼容模型实现

基于 OpenAI API 格式，但支持 Kimi 特殊功能如 thinking 模式。
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Iterator

import httpx

from hawi.agent.model import BalanceInfo
from hawi.agent.message import StreamPart
from hawi.agent.models.openai import OpenAIModel
from hawi.agent.models.openai._streaming import StreamProcessor
from hawi.agent.message import MessageRequest, MessageResponse

logger = logging.getLogger(__name__)

# Kimi K2.5 模型的固定参数
KIMI_K25_FIXED_PARAMS = {
    "top_p": 0.95,
    "n": 1,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
}


class KimiOpenAIModel(OpenAIModel):
    """
    Kimi/Moonshot OpenAI API 兼容模型

    支持 Kimi 系列模型，包括 K2.5 的 thinking 模式。

    Kimi API 特殊处理:
    - K2.5 thinking 模式需要固定 temperature=1.0
    - 非 thinking 模式 temperature=0.6
    - 固定 top_p=0.95, n=1, presence_penalty=0.0, frequency_penalty=0.0

    Example:
        # 普通模型
        model = KimiOpenAIModel(
            model_id="kimi-k2",
            api_key="sk-...",
            base_url="https://api.moonshot.cn/v1",
        )

        # K2.5 thinking 模式（默认启用）
        model = KimiOpenAIModel(
            model_id="kimi-k2.5",
            api_key="sk-...",
            base_url="https://api.moonshot.cn/v1",
        )

        # K2.5 禁用 thinking 模式
        model = KimiOpenAIModel(
            model_id="kimi-k2.5",
            api_key="sk-...",
            base_url="https://api.moonshot.cn/v1",
            enable_thinking=False,
        )
    """

    def __init__(
        self,
        *,
        model_id: str = "kimi-k2.5",
        api_key: str | None = None,
        base_url: str = "https://api.moonshot.cn/v1",
        enable_thinking: bool = True,
        **params,
    ):
        """
        初始化 Kimi 模型

        Args:
            model_id: 模型标识符，默认为 "kimi-k2.5"
            api_key: API 密钥
            base_url: API 基础 URL，默认为 "https://api.moonshot.cn/v1"
            enable_thinking: 是否启用 thinking 模式（K2.5），默认为 True
            **params: 其他参数
        """
        super().__init__(
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
            **params
        )
        self.enable_thinking = enable_thinking

    # K2.5 系列 thinking 模型标识符（支持多种变体）
    _THINKING_MODELS = frozenset({
        "kimi-k2.5",
        "kimi-k2-5",
        "kimi-k2-0711-preview",
        "kimi-k2-0905-preview",
        "kimi-k2-thinking",
        "kimi-k2-thinking-turbo",
    })

    def _is_thinking_model(self) -> bool:
        """检查是否为 thinking 模型"""
        return self.model_id in self._THINKING_MODELS

    def _get_params(self) -> dict[str, Any]:
        """获取模型参数（K2.5 固定参数处理）"""
        params = dict(self.params)

        # 对 K2.5 模型应用固定参数
        if self._is_thinking_model():
            # 根据是否启用 thinking 设置 temperature
            if self.enable_thinking:
                params["temperature"] = 1.0
            else:
                params["temperature"] = 0.6

            # 应用其他固定参数
            params.update(KIMI_K25_FIXED_PARAMS)

            # K2 系列推荐使用 max_completion_tokens（包含 reasoning tokens）
            # 如果用户提供了 max_completion_tokens，则移除 max_tokens 以避免冲突
            if params.get("max_completion_tokens") is not None:
                params.pop("max_tokens", None)
                logger.debug(
                    "Kimi K2.5 使用 max_completion_tokens=%s",
                    params["max_completion_tokens"]
                )

            logger.debug(
                "Kimi K2.5 使用固定参数: temperature=%s, top_p=0.95",
                params["temperature"]
            )

        return params

    def _prepare_request_impl(self, request: MessageRequest) -> dict[str, Any]:
        """准备请求，处理 Kimi 特殊参数"""
        req = super()._prepare_request_impl(request)

        # 对 K2.5 模型，如果禁用 thinking，通过 extra_body 传递参数
        if self._is_thinking_model() and not self.enable_thinking:
            req["extra_body"] = {"thinking": {"type": "disabled"}}
            if "thinking" in req:
                del req["thinking"]

        if req.get("tool_choice") == "required":
            logger.warning(
                "Kimi API 不支持 tool_choice=required，已降级为 auto"
            )
            req["tool_choice"] = "auto"

        self._validate_request_params(req)

        return req

    def _validate_request_params(self, req: dict[str, Any]) -> None:
        temperature = req.get("temperature")
        if temperature is not None and (temperature < 0 or temperature > 1):
            raise ValueError("Kimi temperature 必须在 0 到 1 之间")

        n = req.get("n")
        if temperature is not None and n is not None:
            if temperature <= 0.001 and n > 1:
                raise ValueError(
                    "Kimi 在 temperature 接近 0 时不支持 n>1"
                )

    def _stream_impl(self, request: MessageRequest) -> Iterator[StreamPart]:
        """
        同步流式调用 Kimi API

        重写以处理 reasoning_content 的收集和保留，并使用 ToolCallAccumulator
        确保 tool_call 参数完整性。
        """
        req = self._prepare_request_impl(request)
        req["stream"] = True
        req["stream_options"] = {"include_usage": True}

        processor = StreamProcessor()

        for chunk in self.client.chat.completions.create(**req):
            chunk_dict = chunk.model_dump()
            yield from processor.process_chunk(chunk_dict)

    async def _astream_impl(
        self, request: MessageRequest
    ) -> AsyncIterator[StreamPart]:
        """异步流式调用 Kimi API

        重写以处理 reasoning_content 的收集和保留。
        """
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

    def _convert_message_to_openai(self, message) -> dict[str, Any]:
        """转换消息，处理 Kimi K2.5 reasoning_content"""
        result = super()._convert_message_to_openai(message)

        # 对于 K2.5 thinking 模型，从消息中提取 reasoning_content
        # 注意：reasoning_content 应该来自模型的实际响应，而非硬编码
        if self._is_thinking_model():
            for part in message.get("content", []):
                if part.get("type") == "reasoning":
                    result["reasoning_content"] = part.get("reasoning", "")
                    break

            # 如果消息元数据中有 reasoning_content，也提取出来
            metadata = message.get("metadata")
            if not result.get("reasoning_content") and metadata and metadata.get("reasoning_content"):
                result["reasoning_content"] = metadata["reasoning_content"]

            # Kimi K2.5 API 要求 tool call 消息必须有非空的 reasoning_content
            # 如果没有 reasoning_content 但有 tool_calls，添加一个默认的
            if not result.get("reasoning_content") and result.get("tool_calls"):
                result["reasoning_content"] = "Using tool to solve the problem..."

        return result

    def _parse_response_impl(self, response: dict[str, Any]) -> MessageResponse:
        """解析响应，提取 reasoning_content 并添加到 content 列表"""
        msg_response = super()._parse_response_impl(response)

        # 从原始响应中提取 reasoning_content
        choices = response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            reasoning = message.get("reasoning_content")
            if reasoning:
                msg_response.reasoning_content = reasoning
                # 将 reasoning_content 添加到 content 列表作为 ReasoningPart
                # 这样 HawiAgent 可以正确处理并显示它
                from hawi.agent.message import ReasoningPart
                reasoning_part: ReasoningPart = {
                    "type": "reasoning",
                    "reasoning": reasoning,
                    "signature": None,
                }
                # 添加到 content 末尾，保持 text 在开头（与测试期望一致）
                msg_response.content.append(reasoning_part)

        return msg_response

    def get_balance(self) -> list[BalanceInfo]:
        """
        查询 Kimi 账户余额

        Returns:
            BalanceInfo 对象列表（通常为 USD 一个条目）

        Raises:
            RuntimeError: 如果 API 调用失败或返回错误
        """
        if not self.api_key:
            raise RuntimeError("API key is required for balance query")

        url = f"{self.base_url}/users/me/balance"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            response = httpx.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            resp_data = response.json()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Balance query failed: HTTP {e.response.status_code}") from e
        except httpx.RequestError as e:
            raise RuntimeError(f"Balance query failed: network error - {e}") from e
        except Exception as e:
            raise RuntimeError(f"Balance query failed: {e}") from e

        code = resp_data.get("code")
        if code != 0:
            raise RuntimeError(f"Balance query failed: API error code {code}")

        data = resp_data.get("data", {})
        available_balance = data.get("available_balance", 0.0)
        voucher_balance = data.get("voucher_balance", 0.0)
        cash_balance = data.get("cash_balance", 0.0)

        # 当 available_balance <= 0 时不可用
        is_available = available_balance > 0

        total_balance = voucher_balance + max(cash_balance, 0)

        # Kimi API 不返回 currency 字段，置空表示未知
        return [
            BalanceInfo(
                currency="",
                available_balance=available_balance,
                total_balance=total_balance,
                is_available=is_available,
                details={
                    "voucher_balance": voucher_balance,
                    "cash_balance": cash_balance,
                },
            )
        ]
