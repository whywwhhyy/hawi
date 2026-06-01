"""
Kimi/Moonshot API 兼容模型实现

基于 OpenAI API 格式，但支持 Kimi 特殊功能如 thinking 模式。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any, Iterator

import httpx

from hawi.models import BalanceInfo
from hawi.models import DeltaPart
from hawi.models.openai import OpenAIModel
from hawi.models.openai._streaming import StreamProcessor
from hawi.models import MessageRequest
from ._token_estimate import KimiTokenEstimateMixin

logger = logging.getLogger(__name__)

# Kimi K2 thinking 模型的固定参数
KIMI_K2_THINKING_FIXED_PARAMS = {
    "top_p": 0.95,
    "n": 1,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
}


class KimiOpenAIModel(KimiTokenEstimateMixin, OpenAIModel):
    """
    Kimi/Moonshot OpenAI API 兼容模型

    支持 Kimi 系列模型，包括 K2 thinking 模式。

    Kimi API 特殊处理:
    - K2 thinking 模式需要固定 temperature=1.0
    - 非 thinking 模式 temperature=0.6
    - 固定 top_p=0.95, n=1, presence_penalty=0.0, frequency_penalty=0.0

    Example:
        # 普通模型
        model = KimiOpenAIModel(
            model_id="kimi-k2",
            api_key="sk-...",
            base_url="https://api.moonshot.cn/v1",
        )

        # K2 thinking 模式（默认启用）
        model = KimiOpenAIModel(
            model_id="kimi-k2.5",
            api_key="sk-...",
            base_url="https://api.moonshot.cn/v1",
        )

        # K2 thinking 禁用 thinking 模式
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
            enable_thinking: 是否启用 thinking 模式（K2 thinking），默认为 True
            **params: 其他参数
        """
        thinking_model = self._is_thinking_model_id(model_id)
        include_reasoning_in_context = params.pop(
            "include_reasoning_in_context",
            thinking_model,
        )
        include_reasoning_in_tool_calls = params.pop(
            "include_reasoning_in_tool_calls",
            thinking_model,
        )
        default_tool_call_reasoning_content = params.pop(
            "default_tool_call_reasoning_content",
            "Using tool to solve the problem...",
        )
        super().__init__(
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
            include_reasoning_in_context=include_reasoning_in_context,
            include_reasoning_in_tool_calls=include_reasoning_in_tool_calls,
            default_tool_call_reasoning_content=default_tool_call_reasoning_content,
            **params
        )
        self.enable_thinking = enable_thinking

    # K2 thinking 模型标识符（支持多种变体）
    _THINKING_MODELS = frozenset({
        "kimi-k2.5",
        "kimi-k2-5",
        "kimi-k2.6",
        "kimi-k2-6",
        "kimi-k2-0711-preview",
        "kimi-k2-0905-preview",
        "kimi-k2-thinking",
        "kimi-k2-thinking-turbo",
    })
    _THINKING_MODEL_PREFIXES = (
        "kimi-k2.5",
        "kimi-k2-5",
        "kimi-k2.6",
        "kimi-k2-6",
    )

    @staticmethod
    def _normalize_model_id(model_id: str) -> str:
        """Normalize provider-qualified model IDs for capability detection."""
        return model_id.rsplit("/", 1)[-1].strip().lower().replace("_", "-")

    @classmethod
    def _is_thinking_model_id(cls, model_id: str) -> bool:
        """检查模型 ID 是否为 thinking 模型"""
        normalized = cls._normalize_model_id(model_id)
        return (
            normalized in cls._THINKING_MODELS
            or normalized.startswith(cls._THINKING_MODEL_PREFIXES)
            or "thinking" in normalized
        )

    def _is_thinking_model(self) -> bool:
        """检查是否为 thinking 模型"""
        return self._is_thinking_model_id(self.model_id)

    def _get_params(self) -> dict[str, Any]:
        """获取模型参数（K2 thinking 固定参数处理）"""
        params = dict(self.params)

        # 对 K2 thinking 模型应用固定参数
        if self._is_thinking_model():
            # 根据是否启用 thinking 设置 temperature
            if self.enable_thinking:
                params["temperature"] = 1.0
            else:
                params["temperature"] = 0.6

            # 应用其他固定参数
            params.update(KIMI_K2_THINKING_FIXED_PARAMS)

            # K2 系列推荐使用 max_completion_tokens（包含 reasoning tokens）
            # 如果用户提供了 max_completion_tokens，则移除 max_tokens 以避免冲突
            if params.get("max_completion_tokens") is not None:
                params.pop("max_tokens", None)
                logger.debug(
                    "Kimi K2 thinking 使用 max_completion_tokens=%s",
                    params["max_completion_tokens"]
                )

            logger.debug(
                "Kimi K2 thinking 使用固定参数: temperature=%s, top_p=0.95",
                params["temperature"]
            )

        return params

    def _prepare_request_impl(self, request: MessageRequest) -> dict[str, Any]:
        """准备请求，处理 Kimi 特殊参数"""
        req = super()._prepare_request_impl(request)

        # 对 K2 thinking 模型，如果禁用 thinking，通过 extra_body 传递参数
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

    def _stream_impl(self, request: MessageRequest) -> Iterator[DeltaPart]:
        """
        同步流式调用 Kimi API

        重写以处理 reasoning_content 的收集和保留，并使用 ToolCallAccumulator
        确保 tool_call 参数完整性。
        """
        req = self._prepare_stream_request(request)

        processor = StreamProcessor()

        for chunk in self.client.chat.completions.create(**req):
            chunk_dict = chunk.model_dump()
            yield from processor.process_chunk(chunk_dict)

    async def _astream_impl(
        self, request: MessageRequest
    ) -> AsyncGenerator[DeltaPart, None]:
        """异步流式调用 Kimi API

        重写以处理 reasoning_content 的收集和保留。
        """
        req = self._prepare_stream_request(request)

        processor = StreamProcessor()

        # OpenAI async streaming: await the coroutine first, then use async with
        stream = await self.async_client.chat.completions.create(**req)
        async with stream:
            async for chunk in stream:
                chunk_dict = chunk.model_dump()
                for event in processor.process_chunk(chunk_dict):
                    yield event

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
