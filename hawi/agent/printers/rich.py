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

from rich.console import Console
from rich.panel import Panel
from rich.status import Status
from rich.text import Text

from hawi.agent.events import Event

logger = logging.getLogger(__name__)
_stdout = sys.stdout

_console = Console()

# =============================================================================
# 便捷函数
# =============================================================================


class RichStreamingPrinter:
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

    # ANSI 颜色映射
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

    def __init__(
        self,
        *,
        show_reasoning: bool = True,
        show_tools: bool = True,
        show_errors: bool = True,
        max_arg_length: int = 80,
        max_result_length: int = 200,
        console: Console | None = None,
        typing_delay: float = 0.0,
        text_style: str | None = "green",
    ):
        self.show_reasoning = show_reasoning
        self.show_tools = show_tools
        self.show_errors = show_errors
        self.max_arg_length = max_arg_length
        self.max_result_length = max_result_length
        self._console = console or _console
        self.typing_delay = typing_delay
        self.text_style = text_style
        self._ansi_prefix = self._build_ansi_prefix() if text_style else ""

        # 内部状态
        self._current_block_type: str | None = None
        self._reasoning_buffer: str = ""
        self._active_tool_calls: dict[str, dict[str, Any]] = {}
        self._status_context: Status | None = None
        self._block_wait_status: Status | None = None
        self._block_has_received_delta: bool = False

    def _build_ansi_prefix(self) -> str:
        """构建 ANSI 转义码前缀"""
        if not self.text_style:
            return ""
        codes = []
        for style in self.text_style.lower().split():
            if style in self.ANSI_COLORS:
                codes.append(self.ANSI_COLORS[style])
        return "".join(codes)

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
        if self._ansi_prefix:
            _stdout.write(self.ANSI_COLORS["reset"])
            _stdout.flush()
        self._current_block_type = None
        # NOTE: 暂时禁用 status spinner，避免与后续 panel 输出冲突
        # if self._status_context is not None:
        #     self._status_context.stop()
        #     self._status_context = None
        # # 清理块等待状态
        # if self._block_wait_status is not None:
        #     self._block_wait_status.stop()
        #     self._block_wait_status = None

    async def _on_content_block_start(self, event: Event) -> None:
        """内容块开始"""
        block_type = event.metadata.get("block_type")
        self._current_block_type = block_type
        self._block_has_received_delta = False

        # 对 text 和 thinking 类型的 block 显示等待动画
        # NOTE: 暂时禁用 status spinner，避免与后续 panel 输出冲突
        # if block_type in ("text", "thinking"):
        #     self._block_wait_status = self._console.status(
        #         "[bold green]⠋[/bold green] 等待响应...",
        #         spinner="dots2"
        #     )
        #     self._block_wait_status.start()

    async def _on_content_block_delta(self, event: Event) -> None:
        """逐字符实时输出"""
        meta = event.metadata
        delta_type = meta.get("delta_type")
        delta = meta.get("delta", "")

        # 第一个 delta 到来时停止等待动画
        if not self._block_has_received_delta:
            self._block_has_received_delta = True
            # NOTE: 暂时禁用 status spinner
            # if self._block_wait_status is not None:
            #     self._block_wait_status.stop()
            #     self._block_wait_status = None

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
                    _stdout.write(self.ANSI_COLORS["reset"])

        elif delta_type == "thinking" and self.show_reasoning:
            if self._ansi_prefix:
                _stdout.write(self.ANSI_COLORS["reset"])
            self._reasoning_buffer += delta

    async def _on_content_block_stop(self, event: Event) -> None:
        """内容块结束"""
        meta = event.metadata
        full_content = meta.get("full_content", "")

        # 确保等待动画已停止
        # NOTE: 暂时禁用 status spinner
        # if self._block_wait_status is not None:
        #     self._block_wait_status.stop()
        #     self._block_wait_status = None

        if self._ansi_prefix:
            _stdout.write(self.ANSI_COLORS["reset"])
            _stdout.flush()

        meta = event.metadata
        block_type = meta.get("block_type")

        if block_type == "thinking" and self.show_reasoning:
            self._print_thinking_panel(self._reasoning_buffer or full_content)
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
                # NOTE: 暂时禁用 status spinner，避免与后续 panel 输出冲突
                # if len(self._active_tool_calls) == 1 and self._status_context is None:
                #     self._status_context = self._console.status(
                #         f"[bold blue]🔧 正在执行 {len(self._active_tool_calls)} 个工具...",
                #         spinner="dots"
                #     )
                #     self._status_context.start()

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

        # NOTE: 暂时禁用 status spinner，避免与后续 panel 输出冲突
        # if len(self._active_tool_calls) == 1 and self._status_context is None:
        #     self._status_context = self._console.status(
        #         f"[bold blue]🔧 正在执行 {len(self._active_tool_calls)} 个工具...",
        #         spinner="dots"
        #     )
        #     self._status_context.start()

    async def _on_tool_result(self, event: Event) -> None:
        """工具结果"""
        if not self.show_tools:
            return

        meta = event.metadata
        tool_name = meta.get("tool_name", "unknown")
        success = meta.get("success", False)
        result_preview = meta.get("result_preview", "")

        # 查找并移除对应的工具调用
        tool_call_id = None
        for tid, info in list(self._active_tool_calls.items()):
            if info.get("tool_name") == tool_name:
                tool_call_id = tid
                break

        if tool_call_id:
            tool_info = self._active_tool_calls.pop(tool_call_id)
            start_time = tool_info.get("start_time", time.time())
            arguments = tool_info.get("arguments", {})
        else:
            start_time = time.time()
            arguments = {}

        duration = (time.time() - start_time) * 1000

        # NOTE: 暂时禁用 status spinner，避免与后续 panel 输出冲突
        # if len(self._active_tool_calls) == 0 and self._status_context is not None:
        #     self._status_context.stop()
        #     self._status_context = None

        self._print_tool_result(tool_name, success, result_preview, duration, arguments)

    def _print_tool_result(
        self,
        tool_name: str,
        success: bool,
        result_preview: Any,
        duration: float,
        arguments: dict[str, Any] | None = None
    ) -> None:
        """打印工具结果面板（上下布局）"""
        from rich.table import Table

        status_emoji = "✅" if success else "❌"
        status_color = "green" if success else "red"
        status_text = "成功" if success else "失败"

        # 创建内容表格（上下布局）
        table = Table(show_header=False, box=None, expand=True, padding=(0, 1))
        table.add_column("label", width=10, style="dim cyan")
        table.add_column("content", ratio=1)

        # 调用信息（上半部分）
        table.add_row("工具", Text(tool_name, style="bold cyan"))
        if arguments:
            args_str = str(arguments)
            if len(args_str) > self.max_arg_length:
                args_str = args_str[:self.max_arg_length - 3] + "..."
            table.add_row("参数", Text(args_str, style="dim"))

        # 分隔线
        table.add_row("", "")
        table.add_row("结果", f"{status_emoji} {status_text}", style=f"bold {status_color}")

        # 结果内容（下半部分）
        if result_preview:
            preview = str(result_preview)
            if len(preview) > self.max_result_length:
                preview = preview[: self.max_result_length - 3] + "..."
            table.add_row("", Text(preview, style="white"))

        # 时间信息
        table.add_row("", "")
        table.add_row("耗时", Text(f"{duration:.0f}ms", style="dim"))

        panel = Panel(
            table,
            title=f"[bold {'blue' if success else 'red'}]🔧 Tool Call[/bold {'blue' if success else 'red'}]",
            border_style="blue" if success else "red",
            padding=(0, 0),
        )
        self._console.print()
        self._console.print(panel)

    async def _on_error(self, event: Event) -> None:
        """错误处理"""
        if not self.show_errors:
            return
        error = event.metadata.get("error", "Unknown error")
        self._console.print(f"[bold red]Error:[/bold red] {error}")

