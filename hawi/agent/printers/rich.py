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

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from hawi.events import (
    Event,
    ModelContentBlockStartEvent,
    ModelContentBlockDeltaEvent,
    ModelContentBlockStopEvent,
    ModelToolUseBlockStartEvent,
    ModelToolUseBlockDeltaEvent,
    ModelToolUseBlockStopEvent,
)
from hawi.agent.printers.base import BasePrinter

logger = logging.getLogger(__name__)
_stdout = sys.stdout

_console = Console()

ANSI_COLORS = {
    "black": "\033[30m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "bright_black": "\033[90m",
    "bright_red": "\033[91m",
    "bright_green": "\033[92m",
    "bright_yellow": "\033[93m",
    "bright_blue": "\033[94m",
    "bright_magenta": "\033[95m",
    "bright_cyan": "\033[96m",
    "bright_white": "\033[97m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "italic": "\033[3m",
    "underline": "\033[4m",
    "reset": "\033[0m",
}


class RichStreamingPrinter(BasePrinter):
    """
    Rich 流式打印机 - 唯一推荐的交互式打印机。

    特性：
    - ANSI 转义码实现文本颜色/样式
    - 逐字符实时输出
    - rich Panel 显示 reasoning 和 tool 结果
    - 可选打字机效果

    使用示例：
        printer = RichStreamingPrinter(text_style="green")
        async for event in agent.arun("prompt", stream=True):
            await printer.handle(event)
    """

    def __init__(
        self,
        *,
        show_reasoning: bool = True,
        show_tools: bool = True,
        show_errors: bool = True,
        show_error_stack: bool = True,
        max_arg_length: int = 80,
        max_result_length: int = 200,
        console: Console | None = None,
        typing_delay: float = 0.0,
        text_style: str | None = "green",
        show_full_tool_content: bool = True,
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
        self._console = console or _console
        self.typing_delay = typing_delay
        self.text_style = text_style
        self._ansi_prefix = self._build_ansi_prefix() if text_style else ""

        self._block_has_received_delta: bool = False
        self._block_count: int = 0

    def _build_ansi_prefix(self) -> str:
        """构建 ANSI 转义码前缀"""
        if not self.text_style:
            return ""
        codes = []
        for style in self.text_style.lower().split():
            if style in ANSI_COLORS:
                codes.append(ANSI_COLORS[style])
        return "".join(codes)

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

    async def _on_content_block_delta(self, event: Event) -> None:
        """逐字符实时输出"""
        assert isinstance(event, ModelContentBlockDeltaEvent)
        delta_type = event.delta_type
        delta = event.delta

        if not self._block_has_received_delta:
            self._block_has_received_delta = True

        if not delta:
            return

        if delta_type == "text":
            if self._ansi_prefix:
                _stdout.write(self._ansi_prefix)
            for char in delta:
                _stdout.write(char)
                _stdout.flush()
                if self.typing_delay > 0:
                    await asyncio.sleep(self.typing_delay)
                if char == "\n" and self._ansi_prefix:
                    _stdout.write(ANSI_COLORS["reset"])

        elif delta_type == "thinking" and self.show_reasoning:
            if self._ansi_prefix:
                _stdout.write(ANSI_COLORS["reset"])
            self._reasoning_buffer += delta

    async def _on_content_block_stop(self, event: Event) -> None:
        """内容块结束"""
        assert isinstance(event, ModelContentBlockStopEvent)

        if self._ansi_prefix:
            _stdout.write(ANSI_COLORS["reset"])
            _stdout.flush()

        block_type = event.block_type

        if block_type == "reasoning" and self.show_reasoning:
            # 从 content 中提取 reasoning 文本
            reasoning_content = ""
            for part in event.content:
                if part.get("type") == "reasoning":
                    reasoning_content = part.get("reasoning") or ""
                    break
            self._print_thinking_panel(self._reasoning_buffer or reasoning_content)
            self._reasoning_buffer = ""

        self._current_block_type = None

    async def _on_tool_use_block_start(self, event: Event) -> None:
        """工具调用块开始"""
        assert isinstance(event, ModelToolUseBlockStartEvent)
        self._current_block_type = "tool_use"
        self._block_has_received_delta = False

    async def _on_tool_use_block_delta(self, event: Event) -> None:
        """工具调用块增量"""
        assert isinstance(event, ModelToolUseBlockDeltaEvent)
        # 工具调用参数增量不直接显示，在 stop 时显示完整信息
        if not self._block_has_received_delta:
            self._block_has_received_delta = True

    async def _on_tool_use_block_stop(self, event: Event) -> None:
        """工具调用块结束"""
        assert isinstance(event, ModelToolUseBlockStopEvent)
        self._current_block_type = None

    async def _on_stream_stop(self, event: Event) -> None:
        """Model 流式响应结束"""
        if self._ansi_prefix:
            _stdout.write(ANSI_COLORS["reset"])
            _stdout.flush()
        self._current_block_type = None

    def _print_thinking_panel(self, content: str) -> None:
        """打印 thinking 面板"""
        if not content.strip():
            return
        panel = Panel(
            Text(content.strip()),
            title="[bold yellow]🤔 Thinking[/bold yellow]",
            border_style="yellow",
            padding=(0, 1),
        )
        self._console.print(panel)

    async def _on_run_start(self, event: Event) -> None:
        """Agent 执行开始"""

    async def _on_run_stop(self, event: Event) -> None:
        """Agent 执行结束"""

    def _format_tool_arguments(self, arguments: dict[str, Any]) -> str:
        """格式化工具参数为易读的格式。

        - 无换行符的参数: **arg**: value
        - 有换行符的参数: **arg**:\nvalue
        """
        if not arguments:
            return ""

        lines: list[str] = []
        for key, value in arguments.items():
            value_str = str(value)
            if '\n' in value_str:
                # 有换行符的参数，冒号后换行
                lines.append(f"[bold]{key}[/bold]:")
                lines.append(value_str)
            else:
                # 无换行符的参数，单行显示
                lines.append(f"[bold]{key}[/bold]: {value_str}")

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
        from rich.table import Table

        status_emoji = "✅" if success else "❌"
        status_color = "green" if success else "red"
        status_text = "成功" if success else "失败"

        table = Table(show_header=False, box=None, expand=True, padding=(0, 1))
        table.add_column("label", width=10, style="dim cyan")
        table.add_column("content", ratio=1)

        table.add_row("工具", Text(tool_name, style="bold cyan"))
        if arguments:
            args_text = self._format_tool_arguments(arguments)
            table.add_row("参数", Text.from_markup(args_text))

        table.add_row("", "")
        table.add_row("结果", f"{status_emoji} {status_text}", style=f"bold {status_color}")

        if result_preview:
            preview = str(result_preview)
            if not self.show_full_tool_content and len(preview) > self.max_result_length:
                preview = preview[: self.max_result_length - 3] + "..."
            table.add_row("", Text(preview, style="white"))

        table.add_row("", "")
        table.add_row("耗时", Text(f"{duration:.0f}ms", style="dim"))

        # 在 tool result 前添加额外换行，与前面的文本/block 分隔
        self._console.print()

        panel = Panel(
            table,
            title=f"[bold {'blue' if success else 'red'}]🔧 Tool Call[/bold {'blue' if success else 'red'}]",
            border_style="blue" if success else "red",
            padding=(0, 0),
        )
        self._console.print(panel)

    async def _print_error(self, error: str) -> None:
        """打印错误"""
        panel = Panel(
            Text(error, style="red"),
            title="[bold red]❌ Error[/bold red]",
            border_style="red",
            padding=(0, 1),
        )
        self._console.print(panel)
