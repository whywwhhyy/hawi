"""
Kimi Anthropic API 兼容模型

基于 AnthropicModel，修复 Kimi API 返回内容导致的 Pydantic 序列化警告。

特殊处理:
- Kimi API 返回的 TextBlock 可能包含 citations 字段
- 需要自定义序列化避免 Pydantic 警告
"""

from __future__ import annotations

import logging
from typing import Any

from hawi.models.anthropic import AnthropicModel
from hawi.models.message import MessageResponse

logger = logging.getLogger(__name__)


class KimiAnthropicModel(AnthropicModel):
    """
    Kimi Anthropic API 兼容模型

    使用 Anthropic SDK 格式，端点为 Kimi API。
    端点: https://api.kimi.com/coding/

    特殊处理 citations 字段，避免 Pydantic 序列化警告。

    Example:
        model = KimiAnthropicModel(
            model_id="kimi-k2.5",
            api_key="sk-...",
            base_url="https://api.kimi.com/coding/",
        )
    """

    def __init__(
        self,
        *,
        model_id: str = "kimi-k2.5",
        api_key: str | None = None,
        base_url: str = "https://api.kimi.com/coding/",
        thinking_budget: int | None = 8000,
        max_output_tokens: int | None = None,
        **params,
    ):
        """初始化 Kimi Anthropic 模型"""
        thinking_enabled = bool(thinking_budget)
        include_reasoning_in_context = params.pop(
            "include_reasoning_in_context",
            thinking_enabled,
        )
        include_reasoning_in_tool_calls = params.pop(
            "include_reasoning_in_tool_calls",
            thinking_enabled,
        )
        default_tool_call_reasoning_content = params.pop(
            "default_tool_call_reasoning_content",
            "Using tool to solve the problem...",
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

    def _parse_response_impl(self, response: dict[str, Any]) -> MessageResponse:
        """解析响应，处理 citations 字段"""
        # 预处理 content blocks，提取 citations
        content = response.get("content", [])
        for block in content:
            if isinstance(block, dict) and "citations" in block:
                # citations 是 Kimi 特有的，我们保留它但在 metadata 中
                logger.debug("Detected citations in response: %s", block["citations"])

        return super()._parse_response_impl(response)

    def _serialize_content_block(self, block: dict[str, Any]) -> dict[str, Any]:
        """序列化内容块，处理 citations"""
        result = dict(block)

        # 如果 block 包含 citations，确保它被正确序列化
        if "citations" in result:
            citations = result["citations"]
            if isinstance(citations, list):
                result["citations"] = [
                    self._serialize_citation(c) if hasattr(c, "__dict__") else c
                    for c in citations
                ]

        return result

    def _serialize_citation(self, citation: Any) -> dict[str, Any]:
        """序列化引用字段"""
        if hasattr(citation, "__dict__"):
            return citation.__dict__
        if isinstance(citation, dict):
            return citation
        return {"value": str(citation)}
