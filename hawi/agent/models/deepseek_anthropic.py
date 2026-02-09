"""DeepSeek Anthropic API 兼容性适配器

解决的问题:
1. DeepSeek API 通过 Anthropic 格式调用时的兼容性问题
2. 处理 DeepSeek 特定的响应字段和消息格式差异
3. 支持 DeepSeek Reasoner 模型的 thinking mode

基于 Anthropic API 格式，使用 base_url=https://api.deepseek.com/anthropic

DeepSeek Anthropic API 与标准 Anthropic API 的差异:
- 不支持 images、documents 输入
- 不支持 top_k 参数
- 不支持 anthropic-version 和 anthropic-beta header
- thinking 模式支持但 budget_tokens 被忽略
- tools/tool_choice 支持但 parallel tool use 被忽略

参考资料:
- https://api-docs.deepseek.com/guides/anthropic_api
- https://api-docs.deepseek.com/guides/thinking_mode
"""

import logging
from collections.abc import AsyncGenerator
from typing import Any, override

import anthropic

from strands.models.anthropic import AnthropicModel
from strands.types.content import Messages
from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolSpec, ToolChoice

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

# DeepSeek 不支持的 Anthropic 特定参数
UNSUPPORTED_ANTHROPIC_PARAMS = {
    "top_k",
}


class DeepSeekAnthropicModel(AnthropicModel):
    """DeepSeek Anthropic API 兼容模型

    基于 AnthropicModel，适配 DeepSeek API 的 Anthropic 兼容端点。

    DeepSeek API 的特殊情况:
    - 使用 Anthropic SDK 格式，但底层是 DeepSeek 模型
    - 支持模型: deepseek-chat, deepseek-reasoner
    - 不支持图片、文档输入
    - thinking 模式支持但 budget_tokens 被忽略

    使用示例:
        # 普通模型
        model = DeepSeekAnthropicModel(
            client_args={
                "api_key": "sk-...",
                "base_url": "https://api.deepseek.com/anthropic",
            },
            model_id="deepseek-chat",
            max_tokens=4096,
        )

        # Reasoner 模型 (Thinking Mode)
        model = DeepSeekAnthropicModel(
            client_args={
                "api_key": "sk-...",
                "base_url": "https://api.deepseek.com/anthropic",
            },
            model_id="deepseek-reasoner",
            max_tokens=4096,
            thinking={"type": "enabled", "budget_tokens": 1024},
        )
    """

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """流式处理响应，正确处理 DeepSeek API 的特殊字段"""
        logger.debug("formatting request")
        request = self.format_request(messages, tool_specs, system_prompt, tool_choice)
        logger.debug("request=<%s>", request)

        # 清理 DeepSeek 不支持的参数
        request = self._clean_request_params(request)

        logger.debug("invoking model")
        try:
            async with self.client.messages.stream(**request) as stream:
                logger.debug("got response from model")
                async for event in stream:
                    if event.type in AnthropicModel.EVENT_TYPES:
                        # 使用自定义的序列化方法，处理 DeepSeek 特定字段
                        event_dict = self._serialize_event(event)
                        yield self.format_chunk(event_dict)

                # 获取 usage 信息
                message = event.message  # type: ignore
                usage_dict = self._serialize_usage(message.usage)
                yield self.format_chunk({"type": "metadata", "usage": usage_dict})

        except anthropic.RateLimitError as error:
            raise ModelThrottledException(str(error)) from error
        except anthropic.BadRequestError as error:
            if any(
                overflow_message in str(error).lower()
                for overflow_message in AnthropicModel.OVERFLOW_MESSAGES
            ):
                raise ContextWindowOverflowException(str(error)) from error
            raise error

        logger.debug("finished streaming response from model")

    def _clean_request_params(self, request: dict[str, Any]) -> dict[str, Any]:
        """清理请求中 DeepSeek 不支持的参数

        Args:
            request: 原始请求字典

        Returns:
            清理后的请求字典
        """
        cleaned = dict(request)
        model_id = cleaned.get("model", "")

        # 清理 Anthropic 特定但不支持的参数
        for param in UNSUPPORTED_ANTHROPIC_PARAMS:
            if param in cleaned:
                logger.debug("Removing unsupported param '%s' for DeepSeek", param)
                del cleaned[param]

        # 对 Reasoner 模型进行特殊处理
        if model_id == "deepseek-reasoner":
            cleaned = self._clean_reasoner_params(cleaned)

        return cleaned

    def _clean_reasoner_params(self, request: dict[str, Any]) -> dict[str, Any]:
        """清理 DeepSeek Reasoner 模型不支持的参数

        Args:
            request: 原始请求字典

        Returns:
            清理后的请求字典
        """
        cleaned = dict(request)

        # 检查并移除会报错的参数
        for param in ERROR_REASONER_PARAMS:
            if param in cleaned:
                logger.warning(
                    "DeepSeek Reasoner 不支持 '%s' 参数，已移除", param
                )
                del cleaned[param]

        # 检查并警告不支持的参数
        for param in UNSUPPORTED_REASONER_PARAMS:
            if param in cleaned:
                logger.warning(
                    "DeepSeek Reasoner 不支持 '%s' 参数，设置无效", param
                )
                del cleaned[param]

        # 处理 thinking 参数中的 budget_tokens 警告
        if "thinking" in cleaned:
            thinking = dict(cleaned["thinking"])
            if "budget_tokens" in thinking:
                logger.debug(
                    "DeepSeek Reasoner 忽略 thinking.budget_tokens 参数"
                )
            cleaned["thinking"] = thinking

        # 处理 tool_choice 中的 disable_parallel_tool_use
        if "tool_choice" in cleaned:
            tool_choice = dict(cleaned["tool_choice"])
            if "disable_parallel_tool_use" in tool_choice:
                logger.debug(
                    "DeepSeek Reasoner 忽略 tool_choice.disable_parallel_tool_use 参数"
                )
                del tool_choice["disable_parallel_tool_use"]
            cleaned["tool_choice"] = tool_choice

        return cleaned

    def _serialize_event(self, event: Any) -> dict[str, Any]:
        """将 Anthropic SDK 事件序列化为字典，处理 DeepSeek API 的特殊字段

        Args:
            event: Anthropic SDK 事件对象

        Returns:
            事件的字典表示
        """
        # 基础事件字段
        result: dict[str, Any] = {"type": event.type}

        # 根据事件类型添加特定字段
        if hasattr(event, "index"):
            result["index"] = event.index

        if hasattr(event, "content_block"):
            result["content_block"] = self._serialize_content_block(event.content_block)

        if hasattr(event, "delta"):
            result["delta"] = self._serialize_delta(event.delta)

        if hasattr(event, "message"):
            result["message"] = self._serialize_message(event.message)

        return result

    def _serialize_content_block(self, content_block: Any) -> dict[str, Any]:
        """序列化内容块"""
        if content_block is None:
            return {}

        # 获取基础字段
        result: dict[str, Any] = {"type": content_block.type}

        # 根据内容块类型添加字段
        if hasattr(content_block, "text"):
            result["text"] = content_block.text

        if hasattr(content_block, "id"):
            result["id"] = content_block.id

        if hasattr(content_block, "name"):
            result["name"] = content_block.name

        if hasattr(content_block, "input"):
            result["input"] = content_block.input

        if hasattr(content_block, "thinking"):
            result["thinking"] = content_block.thinking

        if hasattr(content_block, "signature"):
            result["signature"] = content_block.signature

        return result

    def _serialize_delta(self, delta: Any) -> dict[str, Any]:
        """序列化 delta 字段"""
        if delta is None:
            return {}

        result: dict[str, Any] = {"type": delta.type}

        if hasattr(delta, "text"):
            result["text"] = delta.text

        if hasattr(delta, "partial_json"):
            result["partial_json"] = delta.partial_json

        if hasattr(delta, "thinking"):
            result["thinking"] = delta.thinking

        if hasattr(delta, "signature"):
            result["signature"] = delta.signature

        return result

    def _serialize_message(self, message: Any) -> dict[str, Any]:
        """序列化消息字段"""
        result: dict[str, Any] = {}

        if hasattr(message, "stop_reason") and message.stop_reason is not None:
            result["stop_reason"] = message.stop_reason

        return result

    def _serialize_usage(self, usage: Any) -> dict[str, Any]:
        """序列化 usage 字段"""
        result: dict[str, Any] = {}

        if hasattr(usage, "input_tokens"):
            result["input_tokens"] = usage.input_tokens

        if hasattr(usage, "output_tokens"):
            result["output_tokens"] = usage.output_tokens

        if hasattr(usage, "cache_creation_input_tokens"):
            result["cache_creation_input_tokens"] = usage.cache_creation_input_tokens

        if hasattr(usage, "cache_read_input_tokens"):
            result["cache_read_input_tokens"] = usage.cache_read_input_tokens

        return result


def create_deepseek_anthropic_model(
    api_key: str,
    model_id: str = "deepseek-chat",
    **kwargs
) -> DeepSeekAnthropicModel:
    """创建 DeepSeek Anthropic API 模型实例

    Args:
        api_key: DeepSeek API key
        model_id: 模型 ID，默认为 "deepseek-chat"
        **kwargs: 其他参数传递给 DeepSeekAnthropicModel

    Returns:
        DeepSeekAnthropicModel 实例
    """
    return DeepSeekAnthropicModel(
        client_args={
            "api_key": api_key,
            "base_url": "https://api.deepseek.com/anthropic",
        },
        model_id=model_id,
        **kwargs
    )


def create_deepseek_anthropic_reasoner(
    api_key: str,
    **kwargs
) -> DeepSeekAnthropicModel:
    """创建 DeepSeek Anthropic API Reasoner 模型实例 (Thinking Mode)

    注意: Reasoner 模型不支持 temperature, top_p, top_k 等参数

    Args:
        api_key: DeepSeek API key
        **kwargs: 其他参数传递给 DeepSeekAnthropicModel (建议只设置 max_tokens)

    Returns:
        DeepSeekAnthropicModel 实例 (model_id="deepseek-reasoner")
    """
    # 检查并警告不支持的参数
    params = kwargs.get("params", {})
    for param in UNSUPPORTED_REASONER_PARAMS:
        if param in params:
            logger.warning(
                "DeepSeek Reasoner 不支持 '%s' 参数，已自动忽略", param
            )

    return DeepSeekAnthropicModel(
        client_args={
            "api_key": api_key,
            "base_url": "https://api.deepseek.com/anthropic",
        },
        model_id="deepseek-reasoner",
        **kwargs
    )
