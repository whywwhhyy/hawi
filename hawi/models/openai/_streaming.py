"""
OpenAI 流式响应处理器

处理 OpenAI 流式 API 响应，将其转换为统一的 DeltaPart 流。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator

from hawi.models.message import DeltaPart, TokenUsage
from hawi.models.usage import normalize_openai_usage

logger = logging.getLogger(__name__)


class StreamProcessor:
    """OpenAI 流式响应处理器

    将 OpenAI 流式响应转换为 DeltaPart 增量块流。

    Example:
        processor = StreamProcessor()

        for chunk in client.chat.completions.create(stream=True):
            chunk_dict = chunk.model_dump()
            for part in processor.process_chunk(chunk_dict):
                yield part
    """

    def __init__(self, *, expect_usage: bool = False) -> None:
        # 当前块状态
        self._current_block_type: str | None = None
        self._current_block_index: int = 0
        self._tool_call_states: dict[int, dict[str, Any]] = {}
        self._expect_usage = expect_usage
        # 累积内容
        self._text_buffer: str = ""
        self._thinking_buffer: str = ""
        # 存储每个块的完整内容
        self._block_contents: dict[int, dict[str, Any]] = {}
        # 存储 usage 数据，在 finish 时一起发送
        self._pending_usage: TokenUsage | None = None
        # 存储停止原因
        self._stop_reason: str | None = None
        self._pending_finish_reason: str | None = None
        self._finish_emitted = False

    @property
    def stop_reason(self) -> str | None:
        """获取停止原因"""
        return self._stop_reason

    @property
    def usage(self) -> TokenUsage | None:
        """获取 usage 数据"""
        return self._pending_usage

    def get_block_text(self, index: int) -> str:
        """获取指定索引文本块的完整内容"""
        return self._block_contents.get(index, {}).get("text", "")

    def get_block_thinking(self, index: int) -> str:
        """获取指定索引思考块的完整内容"""
        return self._block_contents.get(index, {}).get("reasoning", "")

    def get_tool_call_info(self, index: int) -> dict[str, Any]:
        """获取指定索引工具调用的完整信息"""
        info = self._block_contents.get(index, {})
        args_str = info.get("arguments", "")
        try:
            args = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            args = {}
        return {
            "id": info.get("id", ""),
            "name": info.get("name", ""),
            "arguments": args,
        }

    def process_chunk(
        self,
        chunk_dict: dict[str, Any]
    ) -> Iterator[DeltaPart]:
        """处理单个流式 chunk

        Args:
            chunk_dict: OpenAI chunk 的字典表示

        Yields:
            DeltaPart 增量块
        """
        usage = normalize_openai_usage(chunk_dict.get("usage"))
        if usage is not None:
            self._pending_usage = usage

        choices = chunk_dict.get("choices", [])
        if not choices:
            # 某些 chunk 没有 choices（如 OpenAI include_usage 的 usage-only chunk）。
            if self._pending_finish_reason is not None and self._pending_usage is not None:
                yield self._create_finish_part(self._pending_finish_reason)
            return

        choice = choices[0]
        delta = choice.get("delta") or {}

        # 处理 reasoning_content (OpenAI o1, o3 系列推理模型)
        reasoning_content = delta.get("reasoning_content")
        if reasoning_content:
            # 如果是新的 thinking 块，发送 start
            if self._current_block_type != "reasoning":
                # 先结束之前的块
                if self._current_block_type == "text":
                    yield from self._close_text_block()
                elif self._current_block_type == "tool_use":
                    yield from self._close_tool_blocks()

                self._current_block_type = "reasoning"
                yield {
                    "type": "reasoning_delta",
                    "index": self._current_block_index,
                    "delta": "",
                    "is_start": True,
                    "is_end": False,
                }

            self._thinking_buffer += reasoning_content
            # 存储思考块内容
            if self._current_block_index not in self._block_contents:
                self._block_contents[self._current_block_index] = {}
            self._block_contents[self._current_block_index]["reasoning"] = self._thinking_buffer
            yield {
                "type": "reasoning_delta",
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
                if self._current_block_type == "reasoning":
                    yield from self._close_thinking_block()
                elif self._current_block_type == "tool_use":
                    yield from self._close_tool_blocks()

                self._current_block_type = "text"
                yield {
                    "type": "text_delta",
                    "index": self._current_block_index,
                    "delta": "",
                    "is_start": True,
                    "is_end": False,
                }

            self._text_buffer += content
            # 存储文本块内容
            if self._current_block_index not in self._block_contents:
                self._block_contents[self._current_block_index] = {}
            self._block_contents[self._current_block_index]["text"] = self._text_buffer
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
                func = tc.get("function", {})

                # 如果是新的 tool_use 序列，先结束之前的非工具块。
                if self._current_block_type != "tool_use":
                    if self._current_block_type == "text":
                        yield from self._close_text_block()
                    elif self._current_block_type == "reasoning":
                        yield from self._close_thinking_block()

                    self._current_block_type = "tool_use"

                provider_index = self._tool_provider_index(tc)
                state = self._tool_call_states.get(provider_index)
                if state is None:
                    block_index = self._current_block_index
                    self._current_block_index += 1
                    state = {
                        "id": tc.get("id", ""),
                        "name": func.get("name", ""),
                        "arguments": "",
                        "block_index": block_index,
                    }
                    self._tool_call_states[provider_index] = state

                    yield {
                        "type": "tool_call_delta",
                        "index": block_index,
                        "id": state["id"] or None,
                        "name": state["name"] or None,
                        "arguments_delta": "",
                        "is_start": True,
                        "is_end": False,
                    }
                else:
                    if tc.get("id"):
                        state["id"] = tc["id"]
                    if func.get("name"):
                        state["name"] = func["name"]

                # 累积参数
                if func.get("arguments"):
                    args_delta = func["arguments"]
                    state["arguments"] += args_delta
                    yield {
                        "type": "tool_call_delta",
                        "index": state["block_index"],
                        "id": None,
                        "name": None,
                        "arguments_delta": args_delta,
                        "is_start": False,
                        "is_end": False,
                    }

                # 更新 ID 和 name（有些 OpenAI-compatible 服务会晚到）
                if tc.get("id"):
                    state["id"] = tc["id"]
                if func.get("name"):
                    state["name"] = func["name"]

                # 存储工具调用信息
                block_index = state["block_index"]
                if block_index not in self._block_contents:
                    self._block_contents[block_index] = {}
                self._block_contents[block_index].update({
                    "id": state["id"],
                    "name": state["name"],
                    "arguments": state["arguments"],
                })

        # 处理完成
        finish_reason = choice.get("finish_reason")
        if finish_reason:
            # 存储停止原因
            self._stop_reason = self._map_stop_reason(finish_reason)
            # 结束当前块
            if self._current_block_type == "text":
                yield from self._close_text_block()
            elif self._current_block_type == "reasoning":
                yield from self._close_thinking_block()
            elif self._current_block_type == "tool_use":
                yield from self._close_tool_blocks()

            if self._pending_usage is not None or not self._expect_usage:
                yield self._create_finish_part(self._stop_reason)
            else:
                self._pending_finish_reason = self._stop_reason

    def finalize(self) -> Iterator[DeltaPart]:
        """Emit a delayed finish part if the provider never sent usage."""
        if self._pending_finish_reason is not None and not self._finish_emitted:
            yield self._create_finish_part(self._pending_finish_reason)

    def _create_finish_part(self, stop_reason: str) -> DeltaPart:
        self._finish_emitted = True
        self._pending_finish_reason = None
        usage = self._pending_usage
        self._pending_usage = None
        return {
            "type": "finish",
            "stop_reason": stop_reason,
            "usage": usage,
        }

    def _close_text_block(self) -> Iterator[DeltaPart]:
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

    def _close_thinking_block(self) -> Iterator[DeltaPart]:
        """关闭 thinking 块"""
        if self._current_block_type == "reasoning":
            yield {
                "type": "reasoning_delta",
                "index": self._current_block_index,
                "delta": "",
                "is_start": False,
                "is_end": True,
            }
            self._thinking_buffer = ""
            self._current_block_index += 1
            self._current_block_type = None

    def _close_tool_blocks(self) -> Iterator[DeltaPart]:
        """关闭当前 tool_use 序列中的所有工具块"""
        if self._current_block_type == "tool_use" and self._tool_call_states:
            states = sorted(
                self._tool_call_states.values(),
                key=lambda state: int(state["block_index"]),
            )
            for state in states:
                yield {
                    "type": "tool_call_delta",
                    "index": state["block_index"],
                    "id": state.get("id") or None,
                    "name": state.get("name") or None,
                    "arguments_delta": "",
                    "is_start": False,
                    "is_end": True,
                }
            self._tool_call_states.clear()
            self._current_block_type = None

    @staticmethod
    def _tool_provider_index(tool_call_delta: dict[str, Any]) -> int:
        """Return the provider's tool-call index, defaulting to the first call."""
        raw_index = tool_call_delta.get("index", 0)
        try:
            return int(raw_index)
        except (TypeError, ValueError):
            return 0

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
