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
from typing import Any

from hawi.agent.models.anthropic import AnthropicModel
from hawi.agent.messages import MessageRequest, MessageResponse, ContentPart

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


class DeepSeekAnthropicModel(AnthropicModel):
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

    def __init__(
        self,
        *,
        model_id: str = "deepseek-chat",
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com/anthropic",
        **params,
    ):
        """初始化 DeepSeek Anthropic 模型"""
        super().__init__(
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
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

        # 对 Reasoner 模型进行特殊处理
        if self.model_id == "deepseek-reasoner":
            req = self._clean_reasoner_params(req)

            # Tool calling is supported in deepseek-reasoner (V3.2+)
            if req.get("tools"):
                logger.debug("deepseek-reasoner with tool calling - ensure reasoning_content is handled properly")

        return req

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

        # 从原始响应中提取 reasoning_content (DeepSeek Reasoner)
        content = response.get("content", [])
        for block in content:
            if block.get("type") == "thinking":
                # Anthropic format uses 'thinking' block
                msg_response.reasoning_content = block.get("thinking", "")
                break

        # Also check for direct reasoning_content field (DeepSeek specific)
        if not msg_response.reasoning_content:
            msg_response.reasoning_content = response.get("reasoning_content")

        return msg_response
