"""
OpenAI 流式响应处理器

处理 OpenAI 流式 API 响应，将其转换为统一的 StreamPart 流。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator

from hawi.agent.message import StreamPart

logger = logging.getLogger(__name__)


class StreamProcessor:
    """OpenAI 流式响应处理器

    将 OpenAI 流式响应转换为 StreamPart 增量块流。

    Example:
        processor = StreamProcessor()

        for chunk in client.chat.completions.create(stream=True):
            chunk_dict = chunk.model_dump()
            for part in processor.process_chunk(chunk_dict):
                yield part
    """

    def __init__(self) -> None:
        # 当前块状态
        self._current_block_type: str | None = None
        self._current_block_index: int = 0
        self._tool_call_state: dict[str, Any] | None = None
        # 累积内容
        self._text_buffer: str = ""
        self._thinking_buffer: str = ""
        self._tool_arguments_buffer: str = ""
        # 存储 usage 数据，在 finish 时一起发送
        self._pending_usage: dict[str, int] | None = None

    def process_chunk(
        self,
        chunk_dict: dict[str, Any]
    ) -> Iterator[StreamPart]:
        """处理单个流式 chunk

        Args:
            chunk_dict: OpenAI chunk 的字典表示

        Yields:
            StreamPart 增量块
        """
        choices = chunk_dict.get("choices", [])
        if not choices:
            # 某些 chunk 可能没有 choices（如 usage-only chunk）
            # 存储 usage 数据，在 finish 时一起发送
            if chunk_dict.get("usage"):
                usage = chunk_dict["usage"]
                # 标准化 usage 字段名
                self._pending_usage = {
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                }
                # 添加可选字段
                if "prompt_cache_creation_tokens" in usage:
                    self._pending_usage["cache_write_tokens"] = usage["prompt_cache_creation_tokens"]
                if "prompt_cache_read_tokens" in usage:
                    self._pending_usage["cache_read_tokens"] = usage["prompt_cache_read_tokens"]
            return

        choice = choices[0]
        delta = choice.get("delta") or {}

        # 处理 reasoning_content (OpenAI o1, o3 系列推理模型)
        reasoning_content = delta.get("reasoning_content")
        if reasoning_content:
            # 如果是新的 thinking 块，发送 start
            if self._current_block_type != "thinking":
                # 先结束之前的块
                if self._current_block_type == "text":
                    yield from self._close_text_block()
                elif self._current_block_type == "tool_use":
                    yield from self._close_tool_block()

                self._current_block_type = "thinking"
                yield {
                    "type": "thinking_delta",
                    "index": self._current_block_index,
                    "delta": "",
                    "is_start": True,
                    "is_end": False,
                }

            self._thinking_buffer += reasoning_content
            yield {
                "type": "thinking_delta",
                "index": self._current_block_index,
                "delta": reasoning_content,
                "is_start": False,
                "is_end": False,
            }

        # 处理普通内容
        content = delta.get("content")
        if content:
            # 如果是新的 text 块，发送 start
            if self._current_block_type != "text":
                # 先结束之前的块
                if self._current_block_type == "thinking":
                    yield from self._close_thinking_block()
                elif self._current_block_type == "tool_use":
                    yield from self._close_tool_block()

                self._current_block_type = "text"
                yield {
                    "type": "text_delta",
                    "index": self._current_block_index,
                    "delta": "",
                    "is_start": True,
                    "is_end": False,
                }

            self._text_buffer += content
            yield {
                "type": "text_delta",
                "index": self._current_block_index,
                "delta": content,
                "is_start": False,
                "is_end": False,
            }

        # 处理 tool_calls
        tool_calls = delta.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                # 如果是新的 tool_use 块，发送 start
                if self._current_block_type != "tool_use":
                    # 先结束之前的块
                    if self._current_block_type == "text":
                        yield from self._close_text_block()
                    elif self._current_block_type == "thinking":
                        yield from self._close_thinking_block()

                    self._current_block_type = "tool_use"
                    # 初始化 tool call 状态
                    self._tool_call_state = {
                        "id": tc.get("id", ""),
                        "name": tc.get("function", {}).get("name", ""),
                    }
                    self._tool_arguments_buffer = ""

                    yield {
                        "type": "tool_call_delta",
                        "index": self._current_block_index,
                        "id": self._tool_call_state["id"] or None,
                        "name": self._tool_call_state["name"] or None,
                        "arguments_delta": "",
                        "is_start": True,
                        "is_end": False,
                    }

                # 累积参数
                func = tc.get("function", {})
                if func.get("arguments"):
                    args_delta = func["arguments"]
                    self._tool_arguments_buffer += args_delta
                    yield {
                        "type": "tool_call_delta",
                        "index": self._current_block_index,
                        "id": None,
                        "name": None,
                        "arguments_delta": args_delta,
                        "is_start": False,
                        "is_end": False,
                    }

                # 更新 ID 和 name（如果这是第一个 chunk）
                assert self._tool_call_state is not None, "_tool_call_state must be set when processing tool_calls"
                if tc.get("id"):
                    self._tool_call_state["id"] = tc["id"]
                if func.get("name"):
                    self._tool_call_state["name"] = func["name"]

        # 处理完成
        finish_reason = choice.get("finish_reason")
        if finish_reason:
            # 结束当前块
            if self._current_block_type == "text":
                yield from self._close_text_block()
            elif self._current_block_type == "thinking":
                yield from self._close_thinking_block()
            elif self._current_block_type == "tool_use":
                yield from self._close_tool_block()

            yield {
                "type": "finish",
                "stop_reason": self._map_stop_reason(finish_reason),
                "usage": self._pending_usage,
            }
            self._pending_usage = None

    def _close_text_block(self) -> Iterator[StreamPart]:
        """关闭 text 块"""
        if self._current_block_type == "text":
            yield {
                "type": "text_delta",
                "index": self._current_block_index,
                "delta": "",
                "is_start": False,
                "is_end": True,
            }
            self._text_buffer = ""
            self._current_block_index += 1
            self._current_block_type = None

    def _close_thinking_block(self) -> Iterator[StreamPart]:
        """关闭 thinking 块"""
        if self._current_block_type == "thinking":
            yield {
                "type": "thinking_delta",
                "index": self._current_block_index,
                "delta": "",
                "is_start": False,
                "is_end": True,
            }
            self._thinking_buffer = ""
            self._current_block_index += 1
            self._current_block_type = None

    def _close_tool_block(self) -> Iterator[StreamPart]:
        """关闭 tool_use 块"""
        if self._current_block_type == "tool_use" and self._tool_call_state:
            yield {
                "type": "tool_call_delta",
                "index": self._current_block_index,
                "id": self._tool_call_state.get("id") or None,
                "name": self._tool_call_state.get("name") or None,
                "arguments_delta": "",
                "is_start": False,
                "is_end": True,
            }
            self._tool_arguments_buffer = ""
            self._tool_call_state = None
            self._current_block_index += 1
            self._current_block_type = None

    @staticmethod
    def _map_stop_reason(reason: str) -> str:
        """映射 OpenAI 停止原因到通用格式"""
        mapping = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "content_filter",
        }
        return mapping.get(reason, reason)
