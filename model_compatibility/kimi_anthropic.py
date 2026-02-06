"""
Kimi Anthropic API 兼容性适配器

解决的问题:
1. Kimi API (api.kimi.com/coding) 使用 Anthropic API 格式，但返回的内容中
   包含一些非标准字段（如 citations），导致 Pydantic 序列化警告
   警告: PydanticSerializationUnexpectedValue(Expected `ParsedTextBlock[TypeVar]` ...)

根本原因:
- Kimi API 返回的 TextBlock 包含 citations 字段
- Anthropic SDK 的 ParsedTextBlock 是泛型类型 ParsedTextBlock[ResponseFormatT]
- 当使用 model_dump() 序列化时，Pydantic 发现实际类型与期望类型不匹配，发出警告

解决方案:
- 重写 stream 方法，在序列化前将事件转换为标准字典格式
- 确保 citations 等字段被正确处理，不触发 Pydantic 警告

参考资料:
- https://github.com/danny-avila/LibreChat/issues/11563
- https://platform.moonshot.ai/docs/guide/kimi-k2-5-quickstart
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


class KimiAnthropicModel(AnthropicModel):
    """Kimi Anthropic API 兼容模型

    基于 AnthropicModel，修复了 Kimi API 返回内容导致的 Pydantic 序列化警告。

    Kimi API 的特殊情况:
    - 返回的 TextBlock 可能包含 citations 字段
    - 这会导致 Pydantic 序列化时出现 UnexpectedValue 警告
    - 警告不影响功能，但会污染输出

    解决方案:
    - 重写 stream 方法，手动转换事件为字典，避免 Pydantic 序列化警告

    使用示例:
        model = KimiAnthropicModel(
            client_args={
                "api_key": "sk-...",
                "base_url": "https://api.kimi.com/coding/",
            },
            max_tokens=4096,
            model_id="kimi-k2.5",
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
        """流式处理响应，正确处理 Kimi API 的 citations 字段"""
        logger.debug("formatting request")
        request = self.format_request(messages, tool_specs, system_prompt, tool_choice)
        logger.debug("request=<%s>", request)

        logger.debug("invoking model")
        try:
            async with self.client.messages.stream(**request) as stream:
                logger.debug("got response from model")
                async for event in stream:
                    if event.type in AnthropicModel.EVENT_TYPES:
                        # 使用自定义的序列化方法，避免 Pydantic 警告
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

    def _serialize_event(self, event: Any) -> dict[str, Any]:
        """将 Anthropic SDK 事件序列化为字典，处理 Kimi API 的特殊字段

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
        """序列化内容块，处理 citations 等字段"""
        if content_block is None:
            return {}

        # 获取基础字段
        result: dict[str, Any] = {"type": content_block.type}

        # 根据内容块类型添加字段
        if hasattr(content_block, "text"):
            result["text"] = content_block.text

        if hasattr(content_block, "citations") and content_block.citations:
            # Kimi API 返回的 citations，转换为标准格式
            result["citations"] = [
                self._serialize_citation(c) for c in content_block.citations
            ]

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

    def _serialize_citation(self, citation: Any) -> dict[str, Any]:
        """序列化引用字段"""
        result: dict[str, Any] = {}

        if hasattr(citation, "type"):
            result["type"] = citation.type

        if hasattr(citation, "cited_text"):
            result["cited_text"] = citation.cited_text

        if hasattr(citation, "document_index"):
            result["document_index"] = citation.document_index

        if hasattr(citation, "document_title"):
            result["document_title"] = citation.document_title

        if hasattr(citation, "start_char_index"):
            result["start_char_index"] = citation.start_char_index

        if hasattr(citation, "end_char_index"):
            result["end_char_index"] = citation.end_char_index

        if hasattr(citation, "start_page_number"):
            result["start_page_number"] = citation.start_page_number

        if hasattr(citation, "end_page_number"):
            result["end_page_number"] = citation.end_page_number

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

        if hasattr(message, "stop_reason"):
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
