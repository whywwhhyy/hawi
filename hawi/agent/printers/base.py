from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from hawi.events import (
    Event,
    AgentToolCallEvent,
    AgentToolResultEvent,
    AgentErrorEvent,
    ModelErrorEvent,
    ModelToolUseBlockStartEvent,
    ModelToolUseBlockDeltaEvent,
    ModelToolUseBlockStopEvent,
)
from hawi.errors import AgentError, ModelError

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
        show_error_stack: bool = True,  # 新增：是否显示错误调用栈
        max_arg_length: int = 80,
        max_result_length: int = 200,
        show_full_tool_content: bool = True,
    ):
        self.show_reasoning = show_reasoning
        self.show_tools = show_tools
        self.show_errors = show_errors
        self.show_error_stack = show_error_stack
        self.max_arg_length = max_arg_length
        self.max_result_length = max_result_length
        self.show_full_tool_content = show_full_tool_content

        self._current_block_type: str | None = None
        self._reasoning_buffer: str = ""
        self._active_tool_calls: dict[str, dict[str, Any]] = {}

    def set_show_full_tool_content(self, value: bool) -> None:
        """设置是否显示完整的工具调用内容（不省略）。"""
        self.show_full_tool_content = value

    async def handle(self, event: Event) -> None:
        """处理事件"""
        handlers = {
            "model.content_block_start": self._on_content_block_start,
            "model.content_block_delta": self._on_content_block_delta,
            "model.content_block_stop": self._on_content_block_stop,
            "model.tool_use_block_start": self._on_tool_use_block_start,
            "model.tool_use_block_delta": self._on_tool_use_block_delta,
            "model.tool_use_block_stop": self._on_tool_use_block_stop,
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

    @abstractmethod
    async def _on_tool_use_block_start(self, event: Event) -> None:
        """工具调用块开始 - 子类实现"""
        pass

    @abstractmethod
    async def _on_tool_use_block_delta(self, event: Event) -> None:
        """工具调用块增量 - 子类实现"""
        pass

    @abstractmethod
    async def _on_tool_use_block_stop(self, event: Event) -> None:
        """工具调用块结束 - 子类实现"""
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

        # 类型断言以访问具体属性
        assert isinstance(event, AgentToolCallEvent)

        tool_name = event.tool_name
        tool_call_id = event.tool_call_id or tool_name

        self._active_tool_calls[tool_call_id] = {
            "tool_name": tool_name,
            "arguments": event.arguments,
            "status": "running",
            "start_time": time.time(),
        }

    async def _on_tool_result(self, event: Event) -> None:
        """工具结果"""
        if not self.show_tools:
            return

        # 类型断言以访问具体属性
        assert isinstance(event, AgentToolResultEvent)

        tool_name = event.tool_name
        success = event.success
        result_preview = event.result_preview
        duration = event.duration_ms
        arguments = event.arguments

        # 清理已完成的 tool call
        if event.tool_call_id in self._active_tool_calls:
            self._active_tool_calls.pop(event.tool_call_id, None)

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
        """错误处理 - 处理 AgentErrorEvent 和 ModelErrorEvent"""
        if not self.show_errors:
            return

        # 获取错误对象
        error_obj = getattr(event, 'error', None)
        if error_obj is None:
            return

        if isinstance(error_obj, (AgentError, ModelError)):
            # 有完整的结构化异常对象
            message = error_obj.message or "Unknown error"
            if self.show_error_stack and error_obj.stack_trace:
                full_message = f"{message}\n\n[Stack Trace]\n{error_obj.stack_trace}"
            else:
                full_message = message
            await self._print_error(full_message)
        elif isinstance(error_obj, Exception):
            # 其他异常对象
            message = str(error_obj)
            await self._print_error(message)
        else:
            # 兼容其他情况
            await self._print_error(str(error_obj))

    @abstractmethod
    async def _print_error(self, error: str) -> None:
        """打印错误 - 子类实现"""
        pass
