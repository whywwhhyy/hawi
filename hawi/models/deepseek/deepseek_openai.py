"""
DeepSeek API 兼容模型实现

基于 OpenAI API 格式，但修复了消息格式兼容性问题，并支持 Thinking Mode。

Tool Calling 支持:
- deepseek-chat: 支持 tool calling
- deepseek-reasoner: 从 V3.2 版本开始支持 tool calling + thinking mode

API 限制 (参考 https://api-docs.deepseek.com/guides/thinking_mode):
- reasoning_content: 普通对话中无需回传，但在 tool calling 场景下必须回传
- tool 消息 content: 必须是字符串，不支持数组格式
- Reasoner 模型: temperature/top_p 等参数会被忽略但不会报错
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator, Iterator
from typing import Any

import httpx

from hawi.models.openai import OpenAIModel
from hawi.models import DeltaPart, MessageRequest, MessageResponse
from hawi.models import BalanceInfo
from ._adaptive_reasoning import (
    awith_empty_reasoning_delta_if_missing,
    ensure_reasoning_part,
    is_reasoning_model,
    should_ensure_reasoning_part,
    with_empty_reasoning_delta_if_missing,
)

logger = logging.getLogger(__name__)

# DeepSeek Reasoner 模型不支持的参数 (设置了无效)
UNSUPPORTED_REASONER_PARAMS = {
    "temperature",
    "top_p",
    "top_k",
    "presence_penalty",
    "frequency_penalty",
}

# DeepSeek Reasoner 模型会报错的参数
ERROR_REASONER_PARAMS = {
    "logprobs",
    "top_logprobs",
}

# DeepSeek Reasoner 模型不支持的功能（已过时，保留用于文档参考）
# 从 DeepSeek-V3.2 开始，reasoner 模型支持 tool calling
UNSUPPORTED_REASONER_FEATURES: set[str] = set()


class DeepSeekOpenAIModel(OpenAIModel):
    """
    DeepSeek OpenAI API 兼容模型

    基于 OpenAIModel，但修复了消息格式兼容性问题，并支持 Thinking Mode。

    DeepSeek API 与 OpenAI API 的差异:
    - OpenAI: tool 消息的 content 可以是 str 或 数组
    - DeepSeek: tool 消息的 content 必须是 str

    自动根据 model_id 检测是否为 Reasoner 模型，进行参数过滤。

    Example:
        # 普通模型
        model = DeepSeekOpenAIModel(
            model_id="deepseek-chat",
            api_key="sk-...",
            base_url="https://api.deepseek.com",
        )

        # Reasoner 模型 (Thinking Mode) - 工具调用场景需回传 reasoning_content
        model = DeepSeekOpenAIModel(
            model_id="deepseek-reasoner",
            api_key="sk-...",
            base_url="https://api.deepseek.com",
            include_reasoning_in_context=True,  # 工具调用场景需要开启
        )
    """

    default_steer_merge_mode = "append_to_tool_result"

    def __init__(
        self,
        *,
        model_id: str = "deepseek-chat",
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com",
        include_reasoning_in_context: bool = False,
        **params,
    ):
        """
        初始化 DeepSeek 模型

        Args:
            model_id: 模型标识符，默认为 "deepseek-chat"
            api_key: API 密钥
            base_url: API 基础 URL，默认为 "https://api.deepseek.com"
            include_reasoning_in_context: 是否在非工具调用的 assistant 历史中也包含
                reasoning_content。工具调用 assistant 消息会始终回传 reasoning_content，
                这是 DeepSeek thinking mode 的协议要求。
            **params: 其他参数，如 temperature, max_tokens 等
        """
        super().__init__(
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
            **params
        )

        self.include_reasoning_in_context = include_reasoning_in_context

        # 如果是 Reasoner 模型，警告不支持的参数
        if self.model_id == "deepseek-reasoner":
            self._warn_reasoner_params()

    def _warn_reasoner_params(self) -> None:
        """警告 Reasoner 模型不支持的参数"""
        for param in ERROR_REASONER_PARAMS:
            if param in self.params:
                logger.warning("DeepSeek Reasoner 不支持 '%s' 参数，已移除", param)
        for param in UNSUPPORTED_REASONER_PARAMS:
            if param in self.params:
                logger.warning("DeepSeek Reasoner 不支持 '%s' 参数，设置无效", param)

    def _prepare_request_impl(self, request) -> dict[str, Any]:
        """准备请求，对 Reasoner 模型进行参数过滤"""
        req = super()._prepare_request_impl(request)

        # DeepSeek OpenAI 端点不支持 top_k，需统一剔除
        if "top_k" in req:
            logger.debug("Removing unsupported param 'top_k' for DeepSeek")
            del req["top_k"]

        # 对 Reasoner 模型进行参数校验
        if self.model_id == "deepseek-reasoner":
            req = self._filter_reasoner_params(req)

            # Tool calling is supported in deepseek-reasoner (V3.2+)
            # Just log a warning for older clients
            if req.get("tools"):
                logger.debug("deepseek-reasoner with tool calling - ensure reasoning_content is handled properly")

        return req

    def _filter_reasoner_params(self, req: dict[str, Any]) -> dict[str, Any]:
        """过滤 Reasoner 模型不支持的参数"""
        # 移除会报错的参数
        for param in ERROR_REASONER_PARAMS:
            if param in req:
                del req[param]

        # 警告无效参数
        for param in UNSUPPORTED_REASONER_PARAMS:
            if param in req:
                logger.warning("DeepSeek Reasoner 不支持 '%s' 参数，设置无效", param)

        return req

    @classmethod
    def format_request_tool_message(cls, tool_result: dict[str, Any]) -> dict[str, Any]:
        """
        格式化工具结果为 OpenAI 格式（DeepSeek 特殊版本）

        DeepSeek API 只接受字符串格式的 content，不接受数组格式。

        Args:
            tool_result: 工具结果，包含 toolUseId 和 content

        Returns:
            OpenAI 兼容的 tool 消息，content 为字符串格式
        """
        contents = [
            {"text": json.dumps(content["json"])} if "json" in content else content
            for content in tool_result["content"]
        ]

        # DeepSeek API 只接受字符串格式的 content
        text_parts = []
        for content in contents:
            if "text" in content:
                text_parts.append(content["text"])
            elif "image" in content:
                # 图片内容在 DeepSeek 中不被支持
                logger.warning("DeepSeek API 不支持 tool 消息中的图片内容，已忽略")
                text_parts.append("[图片内容]")
            else:
                text_parts.append(str(content))

        # DeepSeek API 不接受空的 content
        combined_content = "\n".join(text_parts) if text_parts else " "

        return {
            "role": "tool",
            "tool_call_id": tool_result["toolUseId"],
            "content": combined_content,
        }

    def _convert_message_to_openai(self, message) -> list[dict[str, Any]]:
        """转换消息，处理 DeepSeek 特殊格式"""
        results = super()._convert_message_to_openai(message)

        # 处理每条消息（父类可能返回多条，如混合内容拆分时）
        for result in results:
            content = result.get("content")
            if isinstance(content, list):
                # DeepSeek 不支持 image_url 结构，替换为文本占位
                result["content"] = self._sanitize_openai_content(content)

            # tool 消息特殊处理：确保 content 是字符串
            if result.get("role") == "tool":
                content = result.get("content", "")
                if isinstance(content, list):
                    result["content"] = self._serialize_content_to_string(content)

            if result.get("role") == "assistant":
                self._apply_reasoning_content(result, message)

        return results

    def _apply_reasoning_content(
        self,
        result: dict[str, Any],
        message: dict[str, Any],
    ) -> None:
        """Attach DeepSeek reasoning_content when the request protocol requires it."""
        reasoning, has_reasoning = self._extract_request_reasoning(message)
        has_tool_calls = bool(result.get("tool_calls"))

        # DeepSeek thinking mode requires assistant messages that contain
        # tool_calls to be passed back with reasoning_content. Adaptive thinking
        # may intentionally emit no reasoning, and the valid value is then "".
        if has_tool_calls:
            result["reasoning_content"] = reasoning if has_reasoning else ""
            return

        # Keep the opt-in legacy behavior for callers that explicitly want to
        # preserve reasoning on ordinary assistant turns.
        if self.include_reasoning_in_context and has_reasoning:
            result["reasoning_content"] = reasoning

    def _extract_request_reasoning(
        self,
        message: dict[str, Any],
    ) -> tuple[str, bool]:
        """Extract reasoning_content from Hawi message content or metadata."""
        for part in message.get("content", []):
            if part.get("type") == "reasoning":
                return part.get("reasoning") or "", True

        metadata = message.get("metadata")
        if isinstance(metadata, dict) and "reasoning_content" in metadata:
            return str(metadata.get("reasoning_content") or ""), True

        return "", False

    def _sanitize_openai_content(self, content: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sanitized: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                sanitized.append({"type": "text", "text": "[图片内容]"})
            else:
                sanitized.append(part)
        return sanitized

    def _serialize_content_to_string(self, content: list) -> str:
        """将 ContentPart 列表序列化为字符串（DeepSeek 专用）"""
        texts = []
        for part in content:
            p_type = part.get("type")

            if p_type == "text":
                texts.append(part.get("text", ""))
            elif p_type == "tool_result":
                # ToolResultPart: extract text from nested content
                nested_content = part.get("content", [])
                for nested_part in nested_content:
                    if nested_part.get("type") == "text":
                        texts.append(nested_part.get("text", ""))
            elif p_type in {"image", "image_url"}:
                # DeepSeek 不支持图片
                logger.warning("DeepSeek API 不支持图片内容，已忽略")
                texts.append("[图片内容]")
            else:
                texts.append(str(part))

        # DeepSeek API 不接受空的 content
        return "\n".join(texts) if texts else " "

    def _parse_response_impl(self, response: dict[str, Any]) -> MessageResponse:
        """解析响应，提取 reasoning_content"""
        msg_response = super()._parse_response_impl(response)

        # 从原始响应中提取 reasoning_content
        choices = response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            reasoning, server_reasoning_present = self._extract_response_reasoning(message)
            if should_ensure_reasoning_part(
                self.model_id,
                server_reasoning_present=server_reasoning_present,
            ):
                ensure_reasoning_part(msg_response, reasoning)

        return msg_response

    def _extract_response_reasoning(
        self,
        message: dict[str, Any],
    ) -> tuple[str, bool]:
        """Extract DeepSeek reasoning from OpenAI-compatible response message."""
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
                        or "",
                        True,
                    )

        return "", False

    def _stream_impl(self, request: MessageRequest) -> Iterator[DeltaPart]:
        """同步流式调用，适配 DeepSeek adaptive thinking 的空 reasoning 块。"""
        yield from with_empty_reasoning_delta_if_missing(
            super()._stream_impl(request),
            enabled=is_reasoning_model(self.model_id),
        )

    async def _astream_impl(
        self,
        request: MessageRequest,
    ) -> AsyncGenerator[DeltaPart, None]:
        """异步流式调用，适配 DeepSeek adaptive thinking 的空 reasoning 块。"""
        async for part in awith_empty_reasoning_delta_if_missing(
            super()._astream_impl(request),
            enabled=is_reasoning_model(self.model_id),
        ):
            yield part

    def get_balance(self) -> list[BalanceInfo]:
        """
        查询 DeepSeek 账户余额

        Returns:
            BalanceInfo 对象列表，每个币种一个条目

        Raises:
            RuntimeError: 如果 API 调用失败或返回错误
        """
        if not self.api_key:
            raise RuntimeError("API key is required for balance query")

        url = f"{self.base_url}/user/balance"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            response = httpx.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Balance query failed: HTTP {e.response.status_code}") from e
        except httpx.RequestError as e:
            raise RuntimeError(f"Balance query failed: network error - {e}") from e
        except Exception as e:
            raise RuntimeError(f"Balance query failed: {e}") from e

        is_available = data.get("is_available", True)
        balance_infos = data.get("balance_infos", [])

        if not balance_infos:
            raise RuntimeError("Balance query returned empty balance_infos")

        result = []
        for info in balance_infos:
            currency = info.get("currency", "UNKNOWN")
            total_balance = float(info.get("total_balance", "0"))
            granted_balance = float(info.get("granted_balance", "0"))
            topped_up_balance = float(info.get("topped_up_balance", "0"))

            # DeepSeek 的 available_balance = granted + topped_up = total_balance
            available_balance = total_balance

            result.append(
                BalanceInfo(
                    currency=currency,
                    available_balance=available_balance,
                    total_balance=total_balance,
                    is_available=is_available,
                    details={
                        "granted_balance": granted_balance,
                        "topped_up_balance": topped_up_balance,
                    },
                )
            )

        return result
