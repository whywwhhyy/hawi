"""
Hawi Printer Implementations

提供多种事件打印机实现：
- RichStreamingPrinter: 原始 ANSI 颜色流式打印
- MarkdownStreamingPrinter: Markdown 实时渲染打印机
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import sys

from hawi.agent.events import Event

logger = logging.getLogger(__name__)
_stdout = sys.stdout


# =============================================================================
# PlainPrinter - 朴素打印机
# =============================================================================


class PlainPrinter:
    """
    朴素打印机，完全不依赖 rich 库。

    这是最简单、最底层的实现，适合：
    - 不支持 ANSI 的终端
    - 日志文件输出
    - 最小依赖场景

    特性：
    - 逐字符实时输出
    - 纯文本格式，无颜色、无方框
    - 零 rich 依赖

    使用示例：
        printer = PlainPrinter()
        async for event in agent.arun("prompt", stream=True):
            await printer.handle(event)
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

        # 内部状态
        self._current_block_type: str | None = None
        self._reasoning_buffer: str = ""
        self._active_tool_calls: dict[str, dict[str, Any]] = {}
        self._block_wait_spinner: asyncio.Task | None = None
        self._block_has_received_delta: bool = False
        self._spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._spinner_index: int = 0

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

    async def _run_spinner(self) -> None:
        """运行等待动画"""
        while True:
            char = self._spinner_chars[self._spinner_index % len(self._spinner_chars)]
            self._spinner_index += 1
            _stdout.write(f"\r{char} 等待响应...")
            _stdout.flush()
            await asyncio.sleep(0.08)

    def _stop_spinner(self) -> None:
        """停止等待动画"""
        if self._block_wait_spinner is not None:
            self._block_wait_spinner.cancel()
            self._block_wait_spinner = None
            # 清除等待动画行
            _stdout.write("\r" + " " * 20 + "\r")
            _stdout.flush()

    async def _on_content_block_start(self, event: Event) -> None:
        """内容块开始"""
        meta = event.metadata
        block_type = meta.get("block_type")
        self._current_block_type = block_type
        self._block_has_received_delta = False

        # 对 text 和 thinking 类型的 block 显示等待动画
        if block_type in ("text", "thinking"):
            self._block_wait_spinner = asyncio.create_task(self._run_spinner())

    async def _on_content_block_delta(self, event: Event) -> None:
        """逐字符实时输出"""
        meta = event.metadata
        delta_type = meta.get("delta_type")
        delta = meta.get("delta", "")

        # 第一个 delta 到来时停止等待动画
        if not self._block_has_received_delta:
            self._block_has_received_delta = True
            self._stop_spinner()

        if not delta:
            return

        if delta_type == "text":
            _stdout.write(delta)
            _stdout.flush()
        elif delta_type == "thinking" and self.show_reasoning:
            self._reasoning_buffer += delta

    async def _on_content_block_stop(self, event: Event) -> None:
        """内容块结束"""
        # 确保等待动画已停止
        if not self._block_has_received_delta:
            self._stop_spinner()

        meta = event.metadata
        block_type = meta.get("block_type")

        if block_type == "thinking" and self.show_reasoning:
            if self._reasoning_buffer.strip():
                _stdout.write(f"\n[Thinking]\n{self._reasoning_buffer.strip()}\n[/Thinking]\n")
                _stdout.flush()
            self._reasoning_buffer = ""
        elif block_type == "tool_use":
            # 记录工具调用信息，供后续 tool_result 使用
            tool_call_id = meta.get("tool_call_id")
            tool_name = meta.get("tool_name")
            if tool_call_id and tool_name and self.show_tools:
                self._active_tool_calls[tool_call_id] = {
                    "tool_name": tool_name,
                    "arguments": meta.get("tool_arguments", {}),
                    "status": "running",
                    "start_time": time.time(),
                }
        self._current_block_type = None

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

        _stdout.write(f"\n[Tool Call: {tool_name}]\n")
        _stdout.flush()

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

        # 计算耗时
        start_time = None
        for tid, info in list(self._active_tool_calls.items()):
            if info.get("tool_name") == tool_name:
                start_time = info.get("start_time")
                del self._active_tool_calls[tid]
                break

        duration = (time.time() - start_time) * 1000 if start_time else 0

        status = "OK" if success else "FAILED"
        _stdout.write(f"[Tool Result: {tool_name}] {status} ({duration:.0f}ms)\n")

        if result_preview:
            preview = str(result_preview)
            if len(preview) > self.max_result_length:
                preview = preview[: self.max_result_length - 3] + "..."
            _stdout.write(f"  {preview}\n")
        _stdout.flush()

    async def _on_error(self, event: Event) -> None:
        """错误处理"""
        if not self.show_errors:
            return

        meta = event.metadata
        error = meta.get("error", "Unknown error")
        _stdout.write(f"\n[Error] {error}\n")
        _stdout.flush()


