"""
Strands Model 到 Hawi Model 的适配器

将 strands 框架的 Model 实现转译为 hawi 框架的 Model 接口，
允许在 hawi 生态中复用 strands 的模型实现。

示例:
    # 包装 strands DeepSeek 模型
    from strands_models import DeepSeekOpenAIModel
    from hawi.models.strands_adapter import StrandsModel

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
from typing import Any, AsyncGenerator, Iterator, cast

from hawi.models import BalanceInfo, Model
from hawi.models import (
    AudioPart,
    ContentPart,
    DocumentPart,
    ImagePart,
    Message,
    MessageRequest,
    MessageResponse,
    ReasoningPart,
    DeltaPart,
    TextPart,
    TokenUsage,
    ToolCallPart,
    ToolChoice,
    ToolDefinition,
    ToolResultPart,
    VideoPart,
    DeltaTextPart,
    DeltaThinkingPart,
    DeltaToolCallPart,
    DeltaFinishPart,
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
        >>> from hawi.models.strands_adapter import StrandsModel
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

        # 处理 tool calls (Strands中toolUse是content块的一部分)
        # 注意：_convert_strands_block_to_part 已经处理了 toolUse，所以这里不需要重复处理

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

    def _invoke_impl(
        self,
        request: MessageRequest,
    ) -> MessageResponse:
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

    def _stream_impl(self, request: MessageRequest) -> Iterator[DeltaPart]:
        """同步流式实现 - Strands 只支持异步流，同步方法不支持"""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support synchronous streaming. "
            "Use ainvoke(streaming=True) instead."
        )

    async def _ainvoke_impl(
        self,
        request: MessageRequest,
    ) -> AsyncGenerator[DeltaPart, None]:
        """异步非流式实现 - 将完整响应拆分为 DeltaPart 序列

        Args:
            request: 消息请求

        Yields:
            DeltaPart 增量块序列
        """
        from typing import cast

        strands_request = self._prepare_request_impl(request)

        if hasattr(self.strands_model, "run_async"):
            strands_response = await self.strands_model.run_async(**strands_request)
        elif hasattr(self.strands_model, "ainvoke"):
            strands_response = await self.strands_model.ainvoke(**strands_request)
        else:
            # Fallback: use streaming API and collect response
            full_text_parts: list[str] = []
            full_thinking_parts: list[str] = []
            tool_calls: list[DeltaToolCallPart] = []
            final_stop_reason = "end_turn"
            final_usage = None

            async for chunk in self._astream_impl(request):
                chunk_type = chunk["type"]
                if chunk_type == "text_delta":
                    full_text_parts.append(chunk["delta"])
                elif chunk_type == "thinking_delta":
                    full_thinking_parts.append(chunk["delta"])
                elif chunk_type == "tool_call_delta":
                    tool_calls.append(chunk)
                elif chunk_type == "finish":
                    final_stop_reason = chunk.get("stop_reason") or "end_turn"
                    final_usage = chunk.get("usage")

            # Yield collected content as complete parts
            if full_text_parts:
                yield DeltaTextPart(
                    type="text_delta",
                    index=0,
                    delta="".join(full_text_parts),
                    is_start=True,
                    is_end=True,
                )
            if full_thinking_parts:
                yield DeltaThinkingPart(
                    type="thinking_delta",
                    index=1 if full_text_parts else 0,
                    delta="".join(full_thinking_parts),
                    is_start=True,
                    is_end=True,
                )
            # TODO: Aggregate tool calls properly
            for tc in tool_calls:
                yield tc

            yield DeltaFinishPart(
                type="finish",
                stop_reason=final_stop_reason,
                usage=final_usage,
            )
            return

        result = self._parse_response_impl(strands_response)

        # Yield content blocks as DeltaPart
        for idx, part in enumerate(result.content):
            part_type = part["type"]

            if part_type == "text":
                text_part = cast(TextPart, part)
                yield DeltaTextPart(
                    type="text_delta",
                    index=idx,
                    delta=text_part["text"],
                    is_start=True,
                    is_end=True,
                )
            elif part_type == "reasoning":
                reasoning_part = cast(ReasoningPart, part)
                yield DeltaThinkingPart(
                    type="thinking_delta",
                    index=idx,
                    delta=reasoning_part.get("reasoning") or "",
                    is_start=True,
                    is_end=True,
                )
            elif part_type == "tool_call":
                tool_part = cast(ToolCallPart, part)
                yield DeltaToolCallPart(
                    type="tool_call_delta",
                    index=idx,
                    id=tool_part["id"],
                    name=tool_part["name"],
                    arguments_delta=json.dumps(tool_part["arguments"]),
                    is_start=True,
                    is_end=True,
                )

        # Yield finish part
        yield DeltaFinishPart(
            type="finish",
            stop_reason=result.stop_reason or "end_turn",
            usage=result.usage,
        )

    async def _astream_impl(self, request: MessageRequest) -> AsyncGenerator[DeltaPart, None]:
        """异步流式实现"""
        strands_request = self._prepare_request_impl(request)

        if hasattr(self.strands_model, "run_stream_async"):
            strands_stream = self.strands_model.run_stream_async(**strands_request)
        elif hasattr(self.strands_model, "astream"):
            strands_stream = self.strands_model.astream(**strands_request)
        else:
            # Fallback: call sync stream() but handle async generator properly
            strands_stream = self.strands_model.stream(**strands_request)

        state = {"index": 0, "block_started": False, "pending_usage": None}
        async for event in strands_stream:
            for chunk in self._convert_strands_event_to_stream_part(event, state):
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

        # 处理 tool_calls (assistant role) - extract from content as ToolCallPart
        if role == "assistant" and msg["content"]:
            tool_call_parts = [p for p in msg["content"] if p.get("type") == "tool_call"]
            if tool_call_parts:
                strands_msg["tool_calls"] = [
                    self._convert_tool_call_part_to_strands(cast(ToolCallPart, tc))
                    for tc in tool_call_parts
                ]

        # 处理 tool_call_id (tool role) - 从 content 中的 ToolResultPart 获取
        if role == "tool" and msg["content"]:
            for part in msg["content"]:
                if part.get("type") == "tool_result":
                    tool_call_id = cast(ToolResultPart, part).get("tool_call_id")
                    if tool_call_id:
                        strands_msg["tool_call_id"] = tool_call_id
                    break

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
                        "text": part.get("reasoning") or "",
                        "signature": part.get("signature"),
                    }
                }
            }
        elif p_type == "cache_control":
            # strands 可能不支持 cache_control，跳过或转换
            logger.debug("CacheControlPart skipped in strands conversion")
            return None
        elif p_type == "video":
            part = cast(VideoPart, part)
            return {
                "video": {
                    "source": {
                        "bytes": part["source"].get("data", ""),
                    },
                    "format": part["source"].get("format", "mp4"),
                }
            }
        elif p_type == "audio":
            part = cast(AudioPart, part)
            source = part["source"]
            # Strands 音频格式：优先使用 data，否则使用 url
            audio_data = source.get("data") or source.get("url") or ""
            return {
                "audio": {
                    "source": {"bytes": audio_data},
                    "format": source.get("format", "wav"),
                }
            }

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
        elif "reasoningContent" in block:
            # Strands reasoningContent格式
            reasoning = block["reasoningContent"]
            # 处理 redacted_content（加密的安全推理内容）
            if "redactedContent" in reasoning:
                redacted_data = reasoning["redactedContent"]
                if isinstance(redacted_data, str):
                    redacted_bytes = redacted_data.encode("utf-8")
                else:
                    redacted_bytes = redacted_data
                return cast(ReasoningPart, {
                    "type": "reasoning",
                    "reasoning": None,
                    "signature": None,
                    "redacted_content": redacted_bytes,
                })
            if "reasoningText" in reasoning:
                text = reasoning["reasoningText"].get("text", "")
                signature = reasoning["reasoningText"].get("signature")
            else:
                text = reasoning.get("text", "")
                signature = reasoning.get("signature")
            return cast(ReasoningPart, {
                "type": "reasoning",
                "reasoning": text,
                "signature": signature,
                "redacted_content": None,
            })
        elif "toolUse" in block:
            # Strands中toolUse也是content块的一部分
            return self._convert_strands_tool_use_to_part(block["toolUse"])
        elif "toolResult" in block:
            # toolResult块
            tool_result = block["toolResult"]
            return {
                "type": "tool_result",
                "tool_call_id": tool_result.get("toolUseId", ""),
                "content": tool_result.get("content", ""),
                "is_error": tool_result.get("status") == "error",
            }
        elif "video" in block:
            # Strands video content
            video = block["video"]
            source = video.get("source", {})
            video_data = source.get("bytes", "")
            return cast(VideoPart, {
                "type": "video",
                "source": {
                    "data": video_data if isinstance(video_data, str) else "",
                    "format": video.get("format", "mp4"),
                },
            })
        elif "audio" in block:
            # Strands audio content
            audio = block["audio"]
            source = audio.get("source", {})
            audio_data = source.get("bytes", "")
            return cast(AudioPart, {
                "type": "audio",
                "source": {
                    "data": audio_data if isinstance(audio_data, str) else "",
                    "format": audio.get("format", "wav"),
                },
            })

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

    def _convert_strands_tool_use_to_part(self, tool_use: dict[str, Any]) -> ToolCallPart:
        """转换 strands toolUse 到 hawi ToolCallPart"""
        # 解析参数
        input_data = tool_use.get("input", {})
        if isinstance(input_data, str):
            try:
                input_data = json.loads(input_data)
            except json.JSONDecodeError:
                input_data = {}

        return {
            "type": "tool_call",
            "id": tool_use.get("toolUseId", ""),
            "name": tool_use.get("name", ""),
            "arguments": input_data,
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
    ) -> Iterator[DeltaPart]:
        """转换 strands 流到 DeltaPart 流"""
        state = {"index": 0, "block_started": False, "pending_usage": None}

        for event in strands_stream:
            yield from self._convert_strands_event_to_stream_part(event, state)

    def _convert_strands_event_to_stream_part(
        self,
        event: Any,
        state: dict[str, Any],
    ) -> Iterator[DeltaPart]:
        """转换单个 strands 事件到 DeltaPart"""
        index = state["index"]
        block_started = state["block_started"]
        pending_usage = state["pending_usage"]

        # strands 事件格式: {event_type: event_data}
        # e.g., {'contentBlockDelta': {'delta': {'text': '...'}}}
        if isinstance(event, dict):
            # 找到事件类型键（不是 'delta', 'start' 等数据键）
            event_type_keys = [
                "contentBlockDelta", "contentBlockStart", "contentBlockStop",
                "messageStart", "messageStop", "metadata",
                "internalServerException", "modelStreamErrorException",
                "serviceUnavailableException", "throttlingException", "validationException",
                "finish",  # 向后兼容：旧版自定义事件格式
            ]
            event_type = ""
            event_data = {}
            for key in event_type_keys:
                if key in event:
                    event_type = key
                    event_data = event[key]
                    break
        else:
            # 处理对象形式的事件（直接访问属性）
            event_type = getattr(event, "type", "")
            event_data = {}
            # 尝试从对象获取标准Strands事件字段
            if hasattr(event, "delta"):
                event_data["delta"] = event.delta
            if hasattr(event, "start"):
                event_data["start"] = event.start
            if hasattr(event, "stopReason"):
                event_data["stopReason"] = event.stopReason
            if hasattr(event, "metadata"):
                event_data["metadata"] = event.metadata

        if event_type == "contentBlockDelta":
            # Strands标准事件: contentBlockDelta
            delta = event_data.get("delta", {})
            if isinstance(delta, dict):
                # 文本增量
                if "text" in delta:
                    text = delta["text"]
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
                            state["block_started"] = True
                            state["block_type"] = "text"

                        yield {
                            "type": "text_delta",
                            "index": index,
                            "delta": text,
                            "is_start": False,
                            "is_end": False,
                        }
                # reasoningContent 增量
                elif "reasoningContent" in delta:
                    reasoning = delta["reasoningContent"]
                    reasoning_text = ""
                    if isinstance(reasoning, dict):
                        reasoning_text = reasoning.get("text") or ""
                    if reasoning_text:
                        if not block_started:
                            yield {
                                "type": "thinking_delta",
                                "index": index,
                                "delta": "",
                                "is_start": True,
                                "is_end": False,
                            }
                            state["block_started"] = True
                            state["block_type"] = "thinking"

                        yield {
                            "type": "thinking_delta",
                            "index": index,
                            "delta": reasoning_text,
                            "is_start": False,
                            "is_end": False,
                        }
                # 工具输入增量
                elif "toolUse" in delta:
                    tool_input = delta["toolUse"].get("input", "")
                    # 工具块在contentBlockStart时已经初始化
                    # 这里只需要确保block_type被设置
                    if block_started:
                        state["block_type"] = "tool_use"
                    if tool_input:
                        yield {
                            "type": "tool_call_delta",
                            "index": index,
                            "id": None,
                            "name": None,
                            "arguments_delta": tool_input if isinstance(tool_input, str) else json.dumps(tool_input),
                            "is_start": False,
                            "is_end": False,
                        }

        elif event_type == "contentBlockStart":
            # Strands块开始事件
            start = event_data.get("start", {})
            if isinstance(start, dict):
                if "toolUse" in start:
                    tool = start["toolUse"]
                    yield {
                        "type": "tool_call_delta",
                        "index": index,
                        "id": tool.get("toolUseId"),
                        "name": tool.get("name"),
                        "arguments_delta": "",
                        "is_start": True,
                        "is_end": False,
                    }
                    state["block_started"] = True
                    state["block_type"] = "tool_use"

        elif event_type == "contentBlockStop":
            # Strands块结束事件 - 发送is_end标记
            if block_started:
                # 根据块类型发送适当的结束事件
                block_type = state.get("block_type", "text")
                if block_type == "tool_use":
                    yield {
                        "type": "tool_call_delta",
                        "index": index,
                        "id": None,
                        "name": None,
                        "arguments_delta": "",
                        "is_start": False,
                        "is_end": True,
                    }
                elif block_type == "thinking":
                    yield {
                        "type": "thinking_delta",
                        "index": index,
                        "delta": "",
                        "is_start": False,
                        "is_end": True,
                    }
                else:  # text or default
                    yield {
                        "type": "text_delta",
                        "index": index,
                        "delta": "",
                        "is_start": False,
                        "is_end": True,
                    }
                state["block_started"] = False
                state["block_type"] = None
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
            # event_data is already the metadata content when using new format
            # { "metadata": { "usage": {...} } } -> event_data = { "usage": {...} }
            usage = event_data.get("usage") if isinstance(event_data, dict) else None
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
