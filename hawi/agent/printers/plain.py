"""
Hawi Printer Implementations

提供多种事件打印机实现：
- RichStreamingPrinter: 原始 ANSI 颜色流式打印
- MarkdownStreamingPrinter: Markdown 实时渲染打印机
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import sys

from hawi.events import (
    Event,
    ModelContentBlockStartEvent,
    ModelContentBlockDeltaEvent,
    ModelContentBlockStopEvent,
    ModelToolCallBlockStartEvent,
    ModelToolCallBlockDeltaEvent,
    ModelToolCallBlockStopEvent,
)
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
        show_error_stack: bool = True,
        max_arg_length: int = 80,
        max_result_length: int = 200,
        show_full_tool_content: bool = False,
    ):
        super().__init__(
            show_reasoning=show_reasoning,
            show_tools=show_tools,
            show_errors=show_errors,
            show_error_stack=show_error_stack,
            max_arg_length=max_arg_length,
            max_result_length=max_result_length,
            show_full_tool_content=show_full_tool_content,
        )

        self._block_wait_spinner: asyncio.Task | None = None
        self._block_has_received_delta: bool = False
        self._spinner_index: int = 0
        self._block_count: int = 0

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
        assert isinstance(event, ModelContentBlockStartEvent)
        block_type = event.block_type
        self._current_block_type = block_type
        self._block_has_received_delta = False

        # 在每个 block 前添加额外换行（第一个除外）
        if self._block_count > 0:
            _stdout.write("\n")
            _stdout.flush()
        self._block_count += 1

        if block_type in ("text", "thinking"):
            self._block_wait_spinner = asyncio.create_task(self._run_spinner())

    async def _on_content_block_delta(self, event: Event) -> None:
        """逐字符实时输出"""
        assert isinstance(event, ModelContentBlockDeltaEvent)
        delta_type = event.delta_type
        delta = event.delta

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

        assert isinstance(event, ModelContentBlockStopEvent)
        block_type = event.block_type

        if block_type == "thinking" and self.show_reasoning:
            if self._reasoning_buffer.strip():
                _stdout.write(f"\n[Thinking]\n{self._reasoning_buffer.strip()}\n[/Thinking]\n")
                _stdout.flush()
            self._reasoning_buffer = ""

        self._current_block_type = None

    async def _on_tool_use_block_start(self, event: Event) -> None:
        """工具调用块开始"""
        assert isinstance(event, ModelToolCallBlockStartEvent)
        self._current_block_type = "tool_use"
        self._block_has_received_delta = False

    async def _on_tool_use_block_delta(self, event: Event) -> None:
        """工具调用块增量"""
        assert isinstance(event, ModelToolCallBlockDeltaEvent)
        # 工具调用参数增量不直接显示
        if not self._block_has_received_delta:
            self._block_has_received_delta = True

    async def _on_tool_use_block_stop(self, event: Event) -> None:
        """工具调用块结束"""
        assert isinstance(event, ModelToolCallBlockStopEvent)
        self._current_block_type = None

    async def _on_run_start(self, event: Event) -> None:
        """Agent 执行开始"""

    async def _on_run_stop(self, event: Event) -> None:
        """Agent 执行结束"""

    def _format_tool_arguments(self, arguments: dict[str, Any]) -> str:
        """格式化工具参数为易读的格式。

        - 无换行符的参数: arg: value
        - 有换行符的参数: arg:\nvalue
        """
        if not arguments:
            return ""

        lines: list[str] = []
        for key, value in arguments.items():
            value_str = str(value)
            if '\n' in value_str:
                # 有换行符的参数，冒号后换行
                lines.append(f"{key}:")
                lines.append(value_str)
            else:
                # 无换行符的参数，单行显示
                lines.append(f"{key}: {value_str}")

        full_text = "\n".join(lines)
        if not self.show_full_tool_content and len(full_text) > self.max_arg_length:
            full_text = full_text[:self.max_arg_length - 3] + "..."

        return full_text

    async def _print_tool_result(
        self,
        tool_name: str,
        success: bool,
        result_preview: Any,
        duration: float,
        arguments: dict[str, Any] | None = None
    ) -> None:
        """打印工具结果"""
        # 在 tool result 前添加额外换行，与前面的文本/block 分隔
        _stdout.write("\n")

        status = "OK" if success else "FAILED"
        _stdout.write(f"[Tool Result: {tool_name}] {status} ({duration:.0f}ms)\n")

        if arguments:
            for key, value in arguments.items():
                value_str = str(value)
                if '\n' in value_str:
                    # 有换行符的参数，冒号后换行
                    _stdout.write(f"  {key}:\n")
                    for line in value_str.split('\n'):
                        _stdout.write(f"    {line}\n")
                else:
                    # 无换行符的参数，单行显示
                    _stdout.write(f"  {key}: {value_str}\n")

        if result_preview:
            preview = str(result_preview)
            if not self.show_full_tool_content and len(preview) > self.max_result_length:
                preview = preview[: self.max_result_length - 3] + "..."
            _stdout.write(f"  → {preview}\n")
        _stdout.flush()

    async def _print_error(self, error: str) -> None:
        """打印错误"""
        _stdout.write(f"\n[Error] {error}\n")
        _stdout.flush()
