"""
DeepSeek Anthropic API 兼容模型

基于 AnthropicModel，适配 DeepSeek API 的 Anthropic 兼容端点。

特殊处理:
- 不支持 top_k 参数
- Reasoner 模型不支持 temperature, top_p
- 需要处理 thinking 参数的 budget_tokens 忽略
- Reasoner 模型 (V3.2+) 支持 tool calling + thinking mode
  注意：多轮对话中必须回传 reasoning_content
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Iterator
from typing import Any

from hawi.models.anthropic import AnthropicModel
from hawi.models.message import DeltaPart, MessageRequest, MessageResponse
from ._token_estimate import DeepSeekTokenEstimateMixin
from ._adaptive_reasoning import (
    awith_empty_reasoning_delta_if_missing,
    ensure_reasoning_part,
    is_reasoning_model,
    should_ensure_reasoning_part,
    with_empty_reasoning_delta_if_missing,
)

logger = logging.getLogger(__name__)

# DeepSeek 不支持的 Anthropic 特定参数
UNSUPPORTED_ANTHROPIC_PARAMS = {
    "top_k",
}

# DeepSeek Reasoner 模型不支持的参数
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


class DeepSeekAnthropicModel(DeepSeekTokenEstimateMixin, AnthropicModel):
    """
    DeepSeek Anthropic API 兼容模型

    使用 Anthropic SDK 格式，但底层是 DeepSeek 模型。
    端点: https://api.deepseek.com/anthropic

    自动根据 model_id 检测是否为 Reasoner 模型。

    Example:
        # 普通模型
        model = DeepSeekAnthropicModel(
            model_id="deepseek-chat",
            api_key="sk-...",
            base_url="https://api.deepseek.com/anthropic",
        )

        # Reasoner 模型
        model = DeepSeekAnthropicModel(
            model_id="deepseek-reasoner",
            api_key="sk-...",
            base_url="https://api.deepseek.com/anthropic",
        )
    """

    default_steer_merge_mode = "append_to_tool_result"

    def __init__(
        self,
        *,
        model_id: str = "deepseek-chat",
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com/anthropic",
        thinking_budget: int | None = None,
        max_output_tokens: int | None = None,
        include_reasoning_in_context: bool = False,
        **params,
    ):
        """初始化 DeepSeek Anthropic 模型"""
        include_reasoning_in_tool_calls = params.pop(
            "include_reasoning_in_tool_calls",
            True,
        )
        default_tool_call_reasoning_content = params.pop(
            "default_tool_call_reasoning_content",
            "",
        )
        super().__init__(
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
            thinking_budget=thinking_budget,
            max_output_tokens=max_output_tokens,
            include_reasoning_in_context=include_reasoning_in_context,
            include_reasoning_in_tool_calls=include_reasoning_in_tool_calls,
            default_tool_call_reasoning_content=default_tool_call_reasoning_content,
            **params
        )

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

    def _prepare_request_impl(self, request: MessageRequest) -> dict[str, Any]:
        """准备请求，清理 DeepSeek 不支持的参数"""
        req = super()._prepare_request_impl(request)

        # 清理 Anthropic 特定但不支持的参数
        for param in UNSUPPORTED_ANTHROPIC_PARAMS:
            if param in req:
                logger.debug("Removing unsupported param '%s' for DeepSeek", param)
                del req[param]

        if "tool_choice" in req:
            tool_choice = dict(req["tool_choice"])
            if "disable_parallel_tool_use" in tool_choice:
                # DeepSeek Anthropic 端点忽略该字段，避免下游误判
                logger.debug("DeepSeek 忽略 tool_choice.disable_parallel_tool_use 参数")
                del tool_choice["disable_parallel_tool_use"]
            req["tool_choice"] = tool_choice

        if "messages" in req:
            # 清理消息内容中的图片与文档块
            req["messages"] = [
                self._sanitize_message_content(m) for m in req["messages"]
            ]

        # 对 Reasoner 模型进行特殊处理
        if self.model_id == "deepseek-reasoner":
            req = self._clean_reasoner_params(req)

            # Tool calling is supported in deepseek-reasoner (V3.2+)
            if req.get("tools"):
                logger.debug("deepseek-reasoner with tool calling - ensure reasoning_content is handled properly")

        return req

    def _sanitize_message_content(self, message: dict[str, Any]) -> dict[str, Any]:
        content = message.get("content")
        if isinstance(content, list):
            # 复制字典，避免直接修改调用方传入的 message
            message = dict(message)
            message["content"] = self._sanitize_content_blocks(content)
        return message

    def _sanitize_content_blocks(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sanitized: list[dict[str, Any]] = []
        for block in blocks:
            block_type = block.get("type")
            if block_type == "image":
                # DeepSeek Anthropic 端点不支持图片块
                sanitized.append({"type": "text", "text": "[图片内容]"})
            elif block_type == "document":
                # DeepSeek Anthropic 端点不支持文档块
                sanitized.append({"type": "text", "text": "[文档内容]"})
            elif block_type == "tool_result" and isinstance(block.get("content"), list):
                # tool_result 内嵌内容同样需要递归清理
                updated = dict(block)
                updated["content"] = self._sanitize_content_blocks(block["content"])
                sanitized.append(updated)
            else:
                sanitized.append(block)
        return sanitized

    def _clean_reasoner_params(self, request: dict[str, Any]) -> dict[str, Any]:
        """清理 DeepSeek Reasoner 模型不支持的参数"""
        cleaned = dict(request)

        # 检查并移除会报错的参数
        for param in ERROR_REASONER_PARAMS:
            if param in cleaned:
                logger.warning("DeepSeek Reasoner 不支持 '%s' 参数，已移除", param)
                del cleaned[param]

        # 检查并警告不支持的参数（根据 DeepSeek 文档，这些参数会被忽略但不会报错，所以保留）
        for param in UNSUPPORTED_REASONER_PARAMS:
            if param in cleaned:
                logger.warning("DeepSeek Reasoner 不支持 '%s' 参数，设置无效", param)

        # 处理 thinking 参数中的 budget_tokens 警告
        if "thinking" in cleaned:
            thinking = dict(cleaned["thinking"])
            if "budget_tokens" in thinking:
                logger.debug("DeepSeek Reasoner 忽略 thinking.budget_tokens 参数")
            cleaned["thinking"] = thinking

        # 处理 tool_choice 中的 disable_parallel_tool_use
        if "tool_choice" in cleaned:
            tool_choice = dict(cleaned["tool_choice"])
            if "disable_parallel_tool_use" in tool_choice:
                logger.debug("DeepSeek Reasoner 忽略 tool_choice.disable_parallel_tool_use 参数")
                del tool_choice["disable_parallel_tool_use"]
            cleaned["tool_choice"] = tool_choice

        return cleaned

    def _parse_response_impl(self, response: dict[str, Any]) -> MessageResponse:
        """解析响应，提取 reasoning_content"""
        msg_response = super()._parse_response_impl(response)

        reasoning, server_reasoning_present = self._extract_response_reasoning(response)
        if should_ensure_reasoning_part(
            self.model_id,
            server_reasoning_present=server_reasoning_present,
        ):
            ensure_reasoning_part(msg_response, reasoning)

        return msg_response

    def _extract_response_reasoning(
        self,
        response: dict[str, Any],
        parts: list[Any] | None = None,
    ) -> tuple[str, bool]:
        """Extract DeepSeek reasoning from Anthropic-compatible response."""
        content = response.get("content", [])
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type in {"thinking", "reasoning"}:
                return (
                    block.get("thinking")
                    or block.get("reasoning")
                    or "",
                    True,
                )
            if block_type == "redacted_thinking":
                return "[Redacted thinking block]", True

        if "reasoning_content" in response:
            return response.get("reasoning_content") or "", True

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
