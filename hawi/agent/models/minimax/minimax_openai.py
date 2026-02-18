"""
MiniMax OpenAI API 兼容模型

基于 OpenAIModel，但特殊处理 MiniMax 的 <think> 标签格式 thinking 内容。
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator, Iterator
from typing import Any

from hawi.agent.models.openai import OpenAIModel
from hawi.agent.models.openai._streaming import StreamProcessor
from hawi.agent.message import MessageResponse, StreamPart, MessageRequest

logger = logging.getLogger(__name__)

# 匹配 <think> 标签内容的正则表达式
THINK_TAG_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
THINK_START_PATTERN = re.compile(r"<think>")
THINK_END_PATTERN = re.compile(r"</think>")


class MiniMaxOpenAIModel(OpenAIModel):
    """
    MiniMax OpenAI API 兼容模型

    基于 OpenAIModel，但特殊处理 MiniMax 的 <think> 标签格式 thinking 内容。

    MiniMax 与普通 OpenAI API 的差异:
    - MiniMax 将 thinking 内容包裹在 <think>...</think> 标签中
    - 需要解析并提取 thinking 内容

    Example:
        model = MiniMaxOpenAIModel(
            model_id="MiniMax-M2.5",
            api_key="sk-...",
            base_url="https://api.minimaxi.com/v1",
        )
    """

    def __init__(
        self,
        *,
        model_id: str = "MiniMax-M2.5",
        api_key: str | None = None,
        base_url: str = "https://api.minimaxi.com/v1",
        **params,
    ):
        """
        初始化 MiniMax OpenAI 模型

        Args:
            model_id: 模型标识符，默认为 "MiniMax-M2.5"
            api_key: API 密钥
            base_url: API 基础 URL，默认为 "https://api.minimaxi.com/v1"
            **params: 其他参数，如 temperature, max_tokens 等
        """
        super().__init__(
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
            **params
        )

    def _parse_response_impl(self, response: dict[str, Any]) -> MessageResponse:
        """
        解析响应，提取 <think> 标签中的 thinking 内容
        """
        # 先调用父类方法获取标准解析结果
        msg_response = super()._parse_response_impl(response)

        # 从内容中提取 <think> 标签
        thinking_content = None
        text_content = []

        for part in msg_response.content:
            if part.get("type") == "text":
                text = part.get("text", "")
                # 查找 <think> 标签
                match = THINK_TAG_PATTERN.search(text)
                if match:
                    thinking_content = match.group(1).strip()
                    # 移除 <think> 标签后的文本
                    cleaned_text = THINK_TAG_PATTERN.sub("", text).strip()
                    if cleaned_text:
                        text_content.append({
                            "type": "text",
                            "text": cleaned_text,
                        })
                else:
                    text_content.append(part)
            else:
                text_content.append(part)

        # 如果提取到了 thinking 内容
        if thinking_content:
            msg_response.reasoning_content = thinking_content
            # 将 thinking 内容添加到 content 列表作为 ReasoningPart
            from hawi.agent.message import ReasoningPart
            reasoning_part: ReasoningPart = {
                "type": "reasoning",
                "reasoning": thinking_content,
                "signature": None,
            }
            # 插入到 content 开头
            msg_response.content = [reasoning_part] + text_content
        else:
            msg_response.content = text_content

        return msg_response

    def _stream_impl(self, request: MessageRequest) -> Iterator[StreamPart]:
        """同步流式调用 - 处理 <think> 标签"""
        req = self._prepare_request_impl(request)
        req["stream"] = True
        req["stream_options"] = {"include_usage": True}

        processor = StreamProcessor()
        
        # 用于累积和处理 <think> 标签的状态
        buffer = ""
        in_thinking = False
        thinking_started = False
        thinking_content = ""
        
        for chunk in self.client.chat.completions.create(**req):
            chunk_dict = chunk.model_dump()
            
            # 提取 delta 内容
            choices = chunk_dict.get("choices", [])
            if not choices:
                # 非内容 chunk（如 usage），直接传递
                yield from processor.process_chunk(chunk_dict)
                continue
            
            delta = choices[0].get("delta", {})
            content = delta.get("content", "")
            
            if content:
                buffer += content
                
                # 检测 think 标签
                if not in_thinking and not thinking_started:
                    if "<think>" in buffer:
                        in_thinking = True
                        thinking_started = True
                        # 提取 <think> 之前的内容
                        before_think = buffer.split("<think>")[0]
                        if before_think:
                            # 输出 think 之前的内容
                            temp_chunk = dict(chunk_dict)
                            temp_chunk["choices"] = [{
                                "delta": {"content": before_think},
                                "index": 0,
                                "finish_reason": None
                            }]
                            yield from processor.process_chunk(temp_chunk)
                        buffer = buffer.split("<think>", 1)[1] if "<think>" in buffer else ""
                        
                        # 发送 thinking 开始事件
                        yield {
                            "type": "thinking_delta",
                            "index": 0,
                            "delta": "",
                            "is_start": True,
                            "is_end": False,
                        }
                
                elif in_thinking:
                    if "</think>" in buffer:
                        # think 结束
                        think_parts = buffer.split("</think>", 1)
                        thinking_content += think_parts[0]
                        
                        # 发送 thinking 内容
                        if think_parts[0]:
                            yield {
                                "type": "thinking_delta",
                                "index": 0,
                                "delta": think_parts[0],
                                "is_start": False,
                                "is_end": False,
                            }
                        
                        # 发送 thinking 结束事件
                        yield {
                            "type": "thinking_delta",
                            "index": 0,
                            "delta": "",
                            "is_start": False,
                            "is_end": True,
                        }
                        
                        in_thinking = False
                        buffer = think_parts[1] if len(think_parts) > 1 else ""
                        
                        # 输出 think 之后的内容
                        if buffer:
                            temp_chunk = dict(chunk_dict)
                            temp_chunk["choices"] = [{
                                "delta": {"content": buffer},
                                "index": 0,
                                "finish_reason": None
                            }]
                            yield from processor.process_chunk(temp_chunk)
                            buffer = ""
                    else:
                        # 仍在 think 中，发送内容
                        thinking_content += buffer
                        yield {
                            "type": "thinking_delta",
                            "index": 0,
                            "delta": buffer,
                            "is_start": False,
                            "is_end": False,
                        }
                        buffer = ""
                else:
                    # 不在 think 中，正常输出
                    yield from processor.process_chunk(chunk_dict)
                    buffer = ""
            else:
                # 无内容 delta，可能是 tool_calls 或其他
                yield from processor.process_chunk(chunk_dict)

    async def _astream_impl(
        self, request: MessageRequest
    ) -> AsyncIterator[StreamPart]:
        """异步流式调用 - 处理 <think> 标签"""
        req = self._prepare_request_impl(request)
        req["stream"] = True
        req["stream_options"] = {"include_usage": True}

        processor = StreamProcessor()
        
        # 用于累积和处理 <think> 标签的状态
        buffer = ""
        in_thinking = False
        thinking_started = False
        
        stream = await self.async_client.chat.completions.create(**req)
        async with stream:
            async for chunk in stream:
                chunk_dict = chunk.model_dump()
                
                # 提取 delta 内容
                choices = chunk_dict.get("choices", [])
                if not choices:
                    # 非内容 chunk（如 usage），直接传递
                    for event in processor.process_chunk(chunk_dict):
                        yield event
                    continue
                
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                
                if content:
                    buffer += content
                    
                    # 检测 think 标签
                    if not in_thinking and not thinking_started:
                        if "<think>" in buffer:
                            in_thinking = True
                            thinking_started = True
                            # 提取 <think> 之前的内容
                            before_think = buffer.split("<think>")[0]
                            if before_think:
                                # 输出 think 之前的内容
                                temp_chunk = dict(chunk_dict)
                                temp_chunk["choices"] = [{
                                    "delta": {"content": before_think},
                                    "index": 0,
                                    "finish_reason": None
                                }]
                                for event in processor.process_chunk(temp_chunk):
                                    yield event
                            buffer = buffer.split("<think>", 1)[1] if "<think>" in buffer else ""
                            
                            # 发送 thinking 开始事件
                            yield {
                                "type": "thinking_delta",
                                "index": 0,
                                "delta": "",
                                "is_start": True,
                                "is_end": False,
                            }
                    
                    elif in_thinking:
                        if "</think>" in buffer:
                            # think 结束
                            think_parts = buffer.split("</think>", 1)
                            
                            # 发送 thinking 内容
                            if think_parts[0]:
                                yield {
                                    "type": "thinking_delta",
                                    "index": 0,
                                    "delta": think_parts[0],
                                    "is_start": False,
                                    "is_end": False,
                                }
                            
                            # 发送 thinking 结束事件
                            yield {
                                "type": "thinking_delta",
                                "index": 0,
                                "delta": "",
                                "is_start": False,
                                "is_end": True,
                            }
                            
                            in_thinking = False
                            buffer = think_parts[1] if len(think_parts) > 1 else ""
                            
                            # 输出 think 之后的内容
                            if buffer:
                                temp_chunk = dict(chunk_dict)
                                temp_chunk["choices"] = [{
                                    "delta": {"content": buffer},
                                    "index": 0,
                                    "finish_reason": None
                                }]
                                for event in processor.process_chunk(temp_chunk):
                                    yield event
                                buffer = ""
                        else:
                            # 仍在 think 中，发送内容
                            yield {
                                "type": "thinking_delta",
                                "index": 0,
                                "delta": buffer,
                                "is_start": False,
                                "is_end": False,
                            }
                            buffer = ""
                    else:
                        # 不在 think 中，正常输出
                        for event in processor.process_chunk(chunk_dict):
                            yield event
                        buffer = ""
                else:
                    # 无内容 delta，可能是 tool_calls 或其他
                    for event in processor.process_chunk(chunk_dict):
                        yield event
