"""
DeepSeek API 兼容模型实现

基于 OpenAI API 格式，但修复了消息格式兼容性问题，并支持 Thinking Mode。

Tool Calling 支持:
- deepseek-chat: 支持 tool calling
- deepseek-reasoner: 从 V3.2 版本开始支持 tool calling + thinking mode

API 限制 (参考 https://api-docs.deepseek.com/guides/thinking_mode):
- reasoning_content: 只能从响应中读取，请求中包含会导致 400 错误
- tool 消息 content: 必须是字符串，不支持数组格式
- Reasoner 模型: temperature/top_p 等参数会被忽略但不会报错
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from hawi.agent.models.openai import OpenAIModel
from hawi.agent.messages import MessageRequest, MessageResponse
from hawi.agent.model import BalanceInfo

logger = logging.getLogger(__name__)

# DeepSeek Reasoner 模型不支持的参数 (设置了无效)
UNSUPPORTED_REASONER_PARAMS = {
    "temperature",
    "top_p",
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

    def __init__(
        self,
        *,
        model_id: str = "deepseek-chat",
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com",
        **params,
    ):
        """
        初始化 DeepSeek 模型

        Args:
            model_id: 模型标识符，默认为 "deepseek-chat"
            api_key: API 密钥
            base_url: API 基础 URL，默认为 "https://api.deepseek.com"
            **params: 其他参数，如 temperature, max_tokens 等
        """
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

    def _prepare_request_impl(self, request) -> dict[str, Any]:
        """准备请求，对 Reasoner 模型进行参数过滤"""
        req = super()._prepare_request_impl(request)

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

    def _convert_message_to_openai(self, message) -> dict[str, Any]:
        """转换消息，处理 DeepSeek 特殊格式"""
        result = super()._convert_message_to_openai(message)

        # tool 消息特殊处理：确保 content 是字符串
        if result.get("role") == "tool":
            content = result.get("content", "")
            if isinstance(content, list):
                result["content"] = self._serialize_content_to_string(content)

        # 注意：根据 DeepSeek API 文档，请求中不能包含 reasoning_content 字段，
        # 否则会返回 400 错误。reasoning_content 只能从响应中读取。
        # 因此这里不将 reasoning_content 添加到请求中。

        return result

    def _serialize_content_to_string(self, content: list) -> str:
        """将 ContentPart 列表序列化为字符串（DeepSeek 专用）"""
        texts = []
        for part in content:
            p_type = part.get("type")

            if p_type == "text":
                texts.append(part.get("text", ""))
            elif p_type == "image":
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
            reasoning = message.get("reasoning_content")
            if reasoning:
                msg_response.reasoning_content = reasoning
                # 将 reasoning_content 添加到 content 列表作为 ReasoningPart
                # 这样 HawiAgent 可以正确处理并显示它
                from hawi.agent.messages import ReasoningPart
                reasoning_part: ReasoningPart = {
                    "type": "reasoning",
                    "reasoning": reasoning,
                    "signature": None,
                }
                # 插入到 content 开头，保持 reasoning 在 text 之前
                msg_response.content.insert(0, reasoning_part)

        return msg_response

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

