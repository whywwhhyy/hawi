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
from hawi.agent.printers.base import BasePrinter

logger = logging.getLogger(__name__)
_stdout = sys.stdout


# =============================================================================
# PlainPrinter - 朴素打印机
# =============================================================================


class PlainPrinter(BasePrinter):
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

    SPINNER_CHARS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    SPINNER_DELAY = 0.08
    SPINNER_CLEAR_WIDTH = 20

    def __init__(
        self,
        *,
        show_reasoning: bool = True,
        show_tools: bool = True,
        show_errors: bool = True,
        max_arg_length: int = 80,
        max_result_length: int = 200,
    ):
        super().__init__(
            show_reasoning=show_reasoning,
            show_tools=show_tools,
            show_errors=show_errors,
            max_arg_length=max_arg_length,
            max_result_length=max_result_length,
        )

        self._block_wait_spinner: asyncio.Task | None = None
        self._block_has_received_delta: bool = False
        self._spinner_index: int = 0

    async def _run_spinner(self) -> None:
        """运行等待动画"""
        while True:
            char = self.SPINNER_CHARS[self._spinner_index % len(self.SPINNER_CHARS)]
            self._spinner_index += 1
            _stdout.write(f"\r{char} 等待响应...")
            _stdout.flush()
            await asyncio.sleep(self.SPINNER_DELAY)

    def _stop_spinner(self) -> None:
        """停止等待动画"""
        if self._block_wait_spinner is not None:
            self._block_wait_spinner.cancel()
            self._block_wait_spinner = None
            _stdout.write("\r" + " " * self.SPINNER_CLEAR_WIDTH + "\r")
            _stdout.flush()

    async def _on_content_block_start(self, event: Event) -> None:
        """内容块开始"""
        meta = event.metadata
        block_type = meta.get("block_type")
        self._current_block_type = block_type
        self._block_has_received_delta = False

        if block_type in ("text", "thinking"):
            self._block_wait_spinner = asyncio.create_task(self._run_spinner())

    async def _on_content_block_delta(self, event: Event) -> None:
        """逐字符实时输出"""
        meta = event.metadata
        delta_type = meta.get("delta_type")
        delta = meta.get("delta", "")

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

    async def _on_run_stop(self, event: Event) -> None:
        """Agent 执行结束"""

    async def _print_tool_result(
        self,
        tool_name: str,
        success: bool,
        result_preview: Any,
        duration: float,
        arguments: dict[str, Any] | None = None
    ) -> None:
        """打印工具结果"""
        status = "OK" if success else "FAILED"
        _stdout.write(f"[Tool Result: {tool_name}] {status} ({duration:.0f}ms)\n")

        if result_preview:
            preview = str(result_preview)
            if len(preview) > self.max_result_length:
                preview = preview[: self.max_result_length - 3] + "..."
            _stdout.write(f"  {preview}\n")
        _stdout.flush()

    async def _print_error(self, error: str) -> None:
        """打印错误"""
        _stdout.write(f"\n[Error] {error}\n")
        _stdout.flush()
