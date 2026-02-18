from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from hawi.agent.events import Event

logger = logging.getLogger(__name__)


class BasePrinter(ABC):
    """
    打印机基类，封装公共逻辑。
    """

    def __init__(
        self,
        *,
        show_reasoning: bool = True,
        show_tools: bool = True,
        show_errors: bool = True,
        max_arg_length: int = 80,
        max_result_length: int = 200,
    ):
        self.show_reasoning = show_reasoning
        self.show_tools = show_tools
        self.show_errors = show_errors
        self.max_arg_length = max_arg_length
        self.max_result_length = max_result_length

        self._current_block_type: str | None = None
        self._reasoning_buffer: str = ""
        self._active_tool_calls: dict[str, dict[str, Any]] = {}

    async def handle(self, event: Event) -> None:
        """处理事件"""
        handlers = {
            "model.content_block_start": self._on_content_block_start,
            "model.content_block_delta": self._on_content_block_delta,
            "model.content_block_stop": self._on_content_block_stop,
            "model.stream_start": self._on_stream_start,
            "model.stream_stop": self._on_stream_stop,
            "agent.run_start": self._on_run_start,
            "agent.run_stop": self._on_run_stop,
            "agent.tool_call": self._on_tool_call,
            "agent.tool_result": self._on_tool_result,
            "agent.error": self._on_error,
        }

        handler = handlers.get(event.type)
        if handler:
            await handler(event)

    async def _on_stream_start(self, event: Event) -> None:
        """Model 流式响应开始"""
        self._reasoning_buffer = ""
        self._active_tool_calls.clear()

    async def _on_stream_stop(self, event: Event) -> None:
        """Model 流式响应结束"""
        self._current_block_type = None

    @abstractmethod
    async def _on_content_block_start(self, event: Event) -> None:
        """内容块开始 - 子类实现"""
        pass

    @abstractmethod
    async def _on_content_block_delta(self, event: Event) -> None:
        """内容块增量 - 子类实现"""
        pass

    @abstractmethod
    async def _on_content_block_stop(self, event: Event) -> None:
        """内容块结束 - 子类实现"""
        pass

    async def _on_run_start(self, event: Event) -> None:
        """Agent 执行开始"""
        pass

    async def _on_run_stop(self, event: Event) -> None:
        """Agent 执行结束"""
        pass

    async def _on_tool_call(self, event: Event) -> None:
        """工具调用"""
        if not self.show_tools:
            return

        meta = event.metadata
        tool_name = meta.get("tool_name", "unknown")
        tool_call_id = meta.get("tool_call_id") or tool_name

        self._active_tool_calls[tool_call_id] = {
            "tool_name": tool_name,
            "arguments": meta.get("arguments", {}),
            "status": "running",
            "start_time": time.time(),
        }

    async def _on_tool_result(self, event: Event) -> None:
        """工具结果"""
        if not self.show_tools:
            return

        meta = event.metadata
        tool_name = meta.get("tool_name", "unknown")
        success = meta.get("success", False)
        result_preview = meta.get("result_preview", "")

        start_time = None
        arguments = {}
        tool_call_id = None

        for tid, info in list(self._active_tool_calls.items()):
            if info.get("tool_name") == tool_name:
                tool_call_id = tid
                start_time = info.get("start_time")
                arguments = info.get("arguments", {})
                break

        if tool_call_id:
            self._active_tool_calls.pop(tool_call_id, None)

        duration = (time.time() - start_time) * 1000 if start_time else 0

        await self._print_tool_result(tool_name, success, result_preview, duration, arguments)

    @abstractmethod
    async def _print_tool_result(
        self,
        tool_name: str,
        success: bool,
        result_preview: Any,
        duration: float,
        arguments: dict[str, Any] | None = None
    ) -> None:
        """打印工具结果 - 子类实现"""
        pass

    async def _on_error(self, event: Event) -> None:
        """错误处理"""
        if not self.show_errors:
            return

        meta = event.metadata
        error = meta.get("error", "Unknown error")
        await self._print_error(error)

    @abstractmethod
    async def _print_error(self, error: str) -> None:
        """打印错误 - 子类实现"""
        pass
