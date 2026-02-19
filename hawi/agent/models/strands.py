"""
Strands Model 到 Hawi Model 的适配器

将 strands 框架的 Model 实现转译为 hawi 框架的 Model 接口，
允许在 hawi 生态中复用 strands 的模型实现。

示例:
    # 包装 strands DeepSeek 模型
    from strands_models import DeepSeekOpenAIModel
    from hawi.agent.models.strands_adapter import StrandsModel

    strands_model = DeepSeekOpenAIModel(
        client_args={"api_key": "sk-...", "base_url": "https://api.deepseek.com"},
        model_id="deepseek-chat",
    )

    # 转译为 hawi Model
    hawi_model = StrandsModel(strands_model)

    # 使用 hawi API
    response = hawi_model.invoke(messages=[create_user_message("Hello")])
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Iterator, cast

from hawi.agent.model import BalanceInfo, Model
from hawi.agent.message import (
    ContentPart,
    Message,
    MessageRequest,
    MessageResponse,
    StreamPart,
    TextPart,
    ImagePart,
    DocumentPart,
    TokenUsage,
    ToolCallPart,
    ToolResultPart,
    ReasoningPart,
    ToolChoice,
    ToolDefinition,
    TokenUsage,
)

logger = logging.getLogger(__name__)


class StrandsModel(Model):
    """
    Strands Model 到 Hawi Model 的适配器

    将 strands 框架的 Model 实现包装为 hawi 的 Model 接口，
    实现两种消息格式之间的自动转换。

    Attributes:
        strands_model: 底层的 strands Model 实例
        model_id: 模型标识符（从 strands model 自动获取）

    Example:
        >>> from strands_models import DeepSeekOpenAIModel
        >>> from hawi.agent.models.strands_adapter import StrandsModel
        >>>
        >>> strands_model = DeepSeekOpenAIModel(...)
        >>> hawi_model = StrandsModel(strands_model)
        >>> response = hawi_model.invoke(messages=[{"role": "user", "content": [{"type": "text", "text": "Hello"}], "name": None, "tool_calls": None, "tool_call_id": None, "metadata": None}])
    """

    def __init__(self, strands_model: Any) -> None:
        """
        初始化适配器

        Args:
            strands_model: strands 框架的 Model 实例
                (如 DeepSeekOpenAIModel, KimiOpenAIModel 等)
        """
        self.strands_model = strands_model
        self._model_id = self._extract_model_id()

    def _extract_model_id(self) -> str:
        """从 strands model 提取 model_id"""
        # strands model 通常在 config 中存储 model_id
        if hasattr(self.strands_model, "config"):
            config = self.strands_model.config
            if isinstance(config, dict):
                return config.get("model_id", "unknown")
        # 或者直接有 model_id 属性
        if hasattr(self.strands_model, "model_id"):
            return self.strands_model.model_id
        return "unknown"

    @property
    def model_id(self) -> str:
        """模型标识符"""
        return self._model_id

    # ==========================================================================
    # 请求/响应转换
    # ==========================================================================

    def _prepare_request_impl(self, request: MessageRequest) -> dict[str, Any]:
        """
        将 hawi MessageRequest 转换为 strands 格式

        将 hawi 的通用消息格式转换为 strands 的 API 请求格式。
        """
        # 转换消息
        strands_messages = self._convert_messages_to_strands(request.messages)

        # 构建 strands 请求
        strands_request: dict[str, Any] = {
            "messages": strands_messages,
        }

        # 转换系统提示词
        if request.system:
            # list[ContentPart] 格式，提取文本内容
            system_texts = []
            for part in request.system:
                if part.get("type") == "text":
                    system_texts.append(cast(TextPart, part)["text"])
            if system_texts:
                strands_request["system_prompt"] = "\n".join(system_texts)

        # 转换工具定义
        if request.tools:
            strands_request["tool_specs"] = [
                self._convert_tool_definition_to_strands(tool)
                for tool in request.tools
            ]

        # 转换 tool_choice
        if request.tool_choice:
            strands_request["tool_choice"] = self._convert_tool_choice_to_strands(
                request.tool_choice
            )

        # 转换其他参数
        if request.max_tokens is not None:
            strands_request["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            strands_request["temperature"] = request.temperature
        if request.top_p is not None:
            strands_request["top_p"] = request.top_p

        return strands_request

    def _parse_response_impl(self, response: dict[str, Any]) -> MessageResponse:
        """
        将 strands 响应转换为 hawi MessageResponse

        将 strands 的 API 响应转换为 hawi 的通用响应格式。
        """
        # 提取 content
        content: list[ContentPart] = []

        # strands 响应可能包含多种内容块
        if "content" in response:
            for block in response["content"]:
                part = self._convert_strands_block_to_part(block)
                if part:
                    content.append(part)

        # 处理 tool calls
        if "tool_calls" in response:
            for tool_call in response["tool_calls"]:
                content.append(self._convert_strands_tool_call_to_part(tool_call))

        # 提取 usage (Strands使用camelCase字段名)
        usage = None
        if "usage" in response and response["usage"]:
            usage_data = response["usage"]
            usage = TokenUsage(
                input_tokens=usage_data.get("inputTokens", 0),
                output_tokens=usage_data.get("outputTokens", 0),
                cache_write_tokens=usage_data.get("cacheWriteInputTokens"),
                cache_read_tokens=usage_data.get("cacheReadInputTokens"),
            )

        # 提取 stop_reason
        stop_reason = response.get("stop_reason")
        if stop_reason:
            stop_reason = self._map_strands_stop_reason(stop_reason)

        # 提取 reasoning_content (DeepSeek Reasoner 等)
        reasoning_content = response.get("reasoning_content")

        return MessageResponse(
            id=response.get("id", ""),
            role="assistant",
            content=content,
            stop_reason=stop_reason,
            usage=usage,
            reasoning_content=reasoning_content,
        )

    # ==========================================================================
    # 调用实现
    # ==========================================================================

    def _invoke_impl(self, request: MessageRequest) -> MessageResponse:
        """同步调用实现"""
        # 准备 strands 格式的请求
        strands_request = self._prepare_request_impl(request)

        # 调用 strands model
        # strands model 通常有 run_sync 或类似方法
        if hasattr(self.strands_model, "run_sync"):
            strands_response = self.strands_model.run_sync(**strands_request)
        elif hasattr(self.strands_model, "invoke"):
            strands_response = self.strands_model.invoke(**strands_request)
        else:
            raise NotImplementedError(
                f"Strands model {type(self.strands_model)} does not support sync invocation"
            )

        # 转换响应
        return self._parse_response_impl(strands_response)

    def _stream_impl(self, request: MessageRequest) -> Iterator[StreamPart]:
        """同步流式实现"""
        strands_request = self._prepare_request_impl(request)

        # 获取 strands 流
        if hasattr(self.strands_model, "run_stream"):
            strands_stream = self.strands_model.run_stream(**strands_request)
        elif hasattr(self.strands_model, "stream"):
            strands_stream = self.strands_model.stream(**strands_request)
        else:
            raise NotImplementedError(
                f"Strands model {type(self.strands_model)} does not support streaming"
            )

        # 转换流事件
        yield from self._convert_strands_stream(strands_stream)

    async def _ainvoke_impl(self, request: MessageRequest) -> MessageResponse:
        """异步调用实现"""
        strands_request = self._prepare_request_impl(request)

        if hasattr(self.strands_model, "run_async"):
            strands_response = await self.strands_model.run_async(**strands_request)
        elif hasattr(self.strands_model, "ainvoke"):
            strands_response = await self.strands_model.ainvoke(**strands_request)
        else:
            # Fallback 到 sync 版本
            return self._invoke_impl(request)

        return self._parse_response_impl(strands_response)

    async def _astream_impl(self, request: MessageRequest) -> AsyncIterator[StreamPart]:
        """异步流式实现"""
        strands_request = self._prepare_request_impl(request)

        if hasattr(self.strands_model, "run_stream_async"):
            strands_stream = self.strands_model.run_stream_async(**strands_request)
        elif hasattr(self.strands_model, "astream"):
            strands_stream = self.strands_model.astream(**strands_request)
        else:
            # Fallback 到 sync 版本
            for chunk in self._stream_impl(request):
                yield chunk
            return

        async for event in strands_stream:
            for chunk in self._convert_strands_event_to_stream_part(event):
                yield chunk

    # ==========================================================================
    # 类型转换辅助方法
    # ==========================================================================

    def _convert_messages_to_strands(
        self, messages: list[Message]
    ) -> list[dict[str, Any]]:
        """将 hawi Message 列表转换为 strands 格式"""
        strands_messages = []

        for msg in messages:
            strands_msg = self._convert_single_message_to_strands(msg)
            strands_messages.append(strands_msg)

        return strands_messages

    def _convert_single_message_to_strands(self, msg: Message) -> dict[str, Any]:
        """转换单个 hawi Message 到 strands 格式"""
        role = msg["role"]

        # strands 使用 role, content 格式
        strands_msg: dict[str, Any] = {"role": role}

        # 转换 content
        if msg["content"]:
            strands_content = self._convert_content_to_strands(msg["content"])
            strands_msg["content"] = strands_content

        # 处理 tool_calls (assistant role)
        tool_calls = msg.get("tool_calls")
        if role == "assistant" and tool_calls:
            strands_msg["tool_calls"] = [
                self._convert_tool_call_part_to_strands(tc)
                for tc in tool_calls
            ]

        # 处理 tool_call_id (tool role)
        if role == "tool" and msg.get("tool_call_id"):
            strands_msg["tool_call_id"] = msg["tool_call_id"]

        # 处理 name
        if msg.get("name"):
            strands_msg["name"] = msg["name"]

        return strands_msg

    def _convert_content_to_strands(
        self, content: list[ContentPart]
    ) -> list[dict[str, Any]]:
        """将 hawi ContentPart 列表转换为 strands ContentBlock 列表"""
        strands_content = []

        for part in content:
            block = self._convert_part_to_strands_block(part)
            if block:
                strands_content.append(block)

        return strands_content

    def _convert_part_to_strands_block(self, part: ContentPart) -> dict[str, Any] | None:
        """转换单个 ContentPart 到 strands ContentBlock"""
        p_type = part.get("type")

        if p_type == "text":
            return {"text": part.get("text", "")}
        elif p_type == "image":
            part = cast(ImagePart, part)
            return {
                "image": {
                    "url": part["source"]["url"],
                    "detail": part["source"].get("detail"),
                }
            }
        elif p_type == "document":
            part = cast(DocumentPart, part)
            return {
                "document": {
                    "url": part["source"]["url"],
                    "mime_type": part["source"].get("mime_type"),
                    "title": part.get("title"),
                    "context": part.get("context"),
                }
            }
        elif p_type == "tool_call":
            part = cast(ToolCallPart, part)
            return {
                "toolUse": {
                    "toolUseId": part["id"],
                    "name": part["name"],
                    "input": part["arguments"],
                }
            }
        elif p_type == "tool_result":
            part = cast(ToolResultPart, part)
            return {
                "toolResult": {
                    "toolUseId": part["tool_call_id"],
                    "content": part["content"],
                    "is_error": part.get("is_error"),
                }
            }
        elif p_type == "reasoning":
            part = cast(ReasoningPart, part)
            return {
                "reasoningContent": {
                    "reasoningText": {
                        "text": part["reasoning"],
                        "signature": part.get("signature"),
                    }
                }
            }
        elif p_type == "cache_control":
            # strands 可能不支持 cache_control，跳过或转换
            logger.debug("CacheControlPart skipped in strands conversion")
            return None

        logger.warning(f"Unknown content part type: {p_type}")
        return None

    def _convert_strands_block_to_part(
        self, block: dict[str, Any]
    ) -> ContentPart | None:
        """转换 strands ContentBlock 到 hawi ContentPart"""
        if "text" in block:
            return {"type": "text", "text": block["text"]}
        elif "image" in block:
            image = block["image"]
            return {
                "type": "image",
                "source": {
                    "url": image.get("url", ""),
                    "detail": image.get("detail"),
                },
            }
        elif "document" in block:
            doc = block["document"]
            return {
                "type": "document",
                "source": {
                    "url": doc.get("url", ""),
                    "mime_type": doc.get("mime_type"),
                },
                "title": doc.get("title"),
                "context": doc.get("context"),
            }
        elif "reasoningContent" in block or "thinking" in block:
            # reasoningContent (DeepSeek) 或 thinking (Anthropic)
            reasoning = block.get("reasoningContent", block.get("thinking", {}))
            if "reasoningText" in reasoning:
                text = reasoning["reasoningText"].get("text", "")
                signature = reasoning["reasoningText"].get("signature")
            else:
                text = reasoning.get("text", "")
                signature = reasoning.get("signature")
            return {
                "type": "reasoning",
                "reasoning": text,
                "signature": signature,
            }

        logger.debug(f"Unknown strands block: {block.keys()}")
        return None

    def _convert_tool_definition_to_strands(
        self, tool: ToolDefinition
    ) -> dict[str, Any]:
        """转换 hawi ToolDefinition 到 strands ToolSpec"""
        return {
            "name": tool["name"],
            "description": tool["description"],
            "inputSchema": {
                "json": tool["schema"],
            },
        }

    def _convert_strands_tool_call_to_part(self, tool_call: dict[str, Any]) -> ToolCallPart:
        """转换 strands tool_call 到 hawi ToolCallPart"""
        # 解析参数
        arguments = tool_call.get("arguments", tool_call.get("input", "{}"))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

        return {
            "type": "tool_call",
            "id": tool_call.get("id", tool_call.get("toolUseId", "")),
            "name": tool_call.get("name", tool_call.get("function", {}).get("name", "")),
            "arguments": arguments,
        }

    def _convert_tool_call_part_to_strands(self, part: ToolCallPart) -> dict[str, Any]:
        """转换 hawi ToolCallPart 到 strands toolUse 格式"""
        return {
            "toolUse": {
                "toolUseId": part["id"],
                "name": part["name"],
                "input": part["arguments"],
            }
        }

    def _convert_tool_choice_to_strands(
        self, tool_choice: ToolChoice
    ) -> dict[str, Any]:
        """转换 hawi ToolChoice 到 strands ToolChoice"""
        tc_type = tool_choice.get("type", "auto")

        mapping = {
            "none": {"type": "none"},
            "auto": {"type": "auto"},
            "any": {"type": "any"},
            "tool": {"type": "tool", "name": tool_choice.get("name", "")},
        }

        return mapping.get(tc_type, {"type": "auto"})

    def _convert_strands_stream(
        self, strands_stream: Iterator[Any]
    ) -> Iterator[StreamPart]:
        """转换 strands 流到 StreamPart 流"""
        state = {"index": 0, "block_started": False, "pending_usage": None}

        for event in strands_stream:
            yield from self._convert_strands_event_to_stream_part(event, state)

    def _convert_strands_event_to_stream_part(
        self,
        event: Any,
        state: dict[str, Any],
    ) -> Iterator[StreamPart]:
        """转换单个 strands 事件到 StreamPart"""
        index = state["index"]
        block_started = state["block_started"]
        pending_usage = state["pending_usage"]

        # strands 事件可能是 dict 或对象
        if isinstance(event, dict):
            event_type = event.get("type", "")
            event_data = event
        else:
            # 处理对象形式的事件（直接访问属性）
            event_type = getattr(event, "type", "")
            event_data = {
                "content": getattr(event, "content", None),
                "reasoning": getattr(event, "reasoning", None),
                "tool_call": getattr(event, "tool_call", None),
                "usage": getattr(event, "usage", None),
                "stop_reason": getattr(event, "stop_reason", None),
            }

        if event_type == "content":
            content = event_data.get("content")
            if content:
                # 获取文本内容
                text = ""
                if isinstance(content, dict) and "text" in content:
                    text = content["text"]
                elif isinstance(content, str):
                    text = content

                if text:
                    # 如果是新的块，发送 start
                    if not block_started:
                        yield {
                            "type": "text_delta",
                            "index": index,
                            "delta": "",
                            "is_start": True,
                            "is_end": False,
                        }
                        block_started = True

                    yield {
                        "type": "text_delta",
                        "index": index,
                        "delta": text,
                        "is_start": False,
                        "is_end": False,
                    }

                    # 发送 end 标记（strands 的事件是离散的，每个 content 事件是完整的）
                    yield {
                        "type": "text_delta",
                        "index": index,
                        "delta": "",
                        "is_start": False,
                        "is_end": True,
                    }
                    block_started = False

        elif event_type == "reasoning":
            reasoning = event_data.get("reasoning", "")
            if reasoning:
                # 发送 start
                yield {
                    "type": "thinking_delta",
                    "index": index,
                    "delta": "",
                    "is_start": True,
                    "is_end": False,
                }
                # 发送内容
                yield {
                    "type": "thinking_delta",
                    "index": index,
                    "delta": reasoning,
                    "is_start": False,
                    "is_end": False,
                }
                # 发送 end
                yield {
                    "type": "thinking_delta",
                    "index": index,
                    "delta": "",
                    "is_start": False,
                    "is_end": True,
                }

        elif event_type == "tool_call":
            tool_call = event_data.get("tool_call")
            if tool_call:
                # 提取工具调用信息
                tc_id = tool_call.get("id", tool_call.get("toolUseId", "")) if isinstance(tool_call, dict) else None
                tc_name = ""
                tc_args = ""

                if isinstance(tool_call, dict):
                    tc_name = tool_call.get("name", tool_call.get("function", {}).get("name", ""))
                    args = tool_call.get("arguments", tool_call.get("input", {}))
                    if isinstance(args, dict):
                        tc_args = json.dumps(args)
                    else:
                        tc_args = str(args)

                # 发送 start
                yield {
                    "type": "tool_call_delta",
                    "index": index,
                    "id": tc_id or None,
                    "name": tc_name or None,
                    "arguments_delta": "",
                    "is_start": True,
                    "is_end": False,
                }
                # 发送参数
                if tc_args:
                    yield {
                        "type": "tool_call_delta",
                        "index": index,
                        "id": None,
                        "name": None,
                        "arguments_delta": tc_args,
                        "is_start": False,
                        "is_end": False,
                    }
                # 发送 end
                yield {
                    "type": "tool_call_delta",
                    "index": index,
                    "id": tc_id or None,
                    "name": tc_name or None,
                    "arguments_delta": "",
                    "is_start": False,
                    "is_end": True,
                }
                state["block_started"] = False
                state["index"] = index + 1

        elif event_type == "messageStop":
            # Strands消息结束事件
            stop_reason = event_data.get("stopReason", "end_turn")
            if isinstance(stop_reason, dict):
                stop_reason = stop_reason.get("stopReason", "end_turn")
            mapped_stop_reason = self._map_strands_stop_reason(stop_reason) if stop_reason else "end_turn"
            yield {
                "type": "finish",
                "stop_reason": mapped_stop_reason,
                "usage": pending_usage,
            }
            state["pending_usage"] = None

        elif event_type == "metadata":
            # Strands在metadata事件中返回usage
            metadata = event_data.get("metadata", {})
            usage = metadata.get("usage") if isinstance(metadata, dict) else None
            if usage:
                # 保存 usage 到 pending，等待 finish 事件
                if isinstance(usage, dict):
                    new_usage = {
                        "input_tokens": usage.get("inputTokens", 0),
                        "output_tokens": usage.get("outputTokens", 0),
                    }
                    if "cacheWriteInputTokens" in usage:
                        new_usage["cache_write_tokens"] = usage["cacheWriteInputTokens"]
                    if "cacheReadInputTokens" in usage:
                        new_usage["cache_read_tokens"] = usage["cacheReadInputTokens"]
                    state["pending_usage"] = new_usage

        # 保留对旧版自定义事件格式的兼容处理
        elif event_type == "finish":
            stop_reason = event_data.get("stop_reason", "end_turn")
            mapped_stop_reason = self._map_strands_stop_reason(stop_reason) if stop_reason else "end_turn"
            yield {
                "type": "finish",
                "stop_reason": mapped_stop_reason,
                "usage": pending_usage,
            }
            state["pending_usage"] = None

        else:
            # 未知事件类型，尝试通用处理
            logger.debug(f"Unknown strands event type: {event_type}")

    def _map_strands_stop_reason(self, reason: str) -> str:
        """映射 strands stop_reason 到 hawi 格式"""
        mapping = {
            "stop": "end_turn",
            "end_turn": "end_turn",
            "tool_calls": "tool_use",
            "tool_use": "tool_use",
            "length": "max_tokens",
            "max_tokens": "max_tokens",
            "content_filter": "content_filter",
            "pause_turn": "pause_turn",
        }
        return mapping.get(reason, reason)

    # ==========================================================================
    # 余额查询（可选）
    # ==========================================================================

    def get_balance(self) -> list[BalanceInfo]:
        """
        查询账户余额

        如果底层 strands model 支持余额查询，则委托给它。
        否则抛出 NotImplementedError。
        """
        # 检查 strands model 是否有 get_balance 方法
        if hasattr(self.strands_model, "get_balance"):
            return self.strands_model.get_balance()

        # 检查是否有 balance 属性
        if hasattr(self.strands_model, "balance"):
            balance = self.strands_model.balance
            if isinstance(balance, list):
                return balance
            elif isinstance(balance, BalanceInfo):
                return [balance]

        raise NotImplementedError(
            f"{self.__class__.__name__} does not support balance query"
        )
