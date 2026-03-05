"""
Hawi Printer Implementations

提供多种事件打印机实现：
- RichPrinter: 动态流式渲染输出 (Markdown + Live)
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Optional

from rich.console import Console, RenderableType, Group

# For test mocking compatibility
_stdout = sys.stdout
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.json import JSON
from rich.rule import Rule

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

class RichPrinter(BasePrinter):
    """
    Rich Printer: 使用 rich 库输出真正的动态用户界面。
    
    特性：
    - 支持流式输出渲染过的 markdown 到终端
    - 实时流式显示输出（backtrack/re-render）
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
        show_full_tool_content: bool = True,
        console: Console | None = None,
        refresh_per_second: float = 12.5,
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
        self._console = console or Console()
        self._refresh_per_second = refresh_per_second
        
        self._live: Optional[Live] = None
        self._current_content = ""
        self._current_tool_name = ""
        # 缓存 tool args 用于 Live 持续显示
        self._pending_tool_calls: dict[str, str] = {} # tool_name -> args_json

    def _start_live(self, renderable: RenderableType) -> None:
        if self._live:
            self._live.stop()
        self._live = Live(
            renderable,
            console=self._console,
            refresh_per_second=self._refresh_per_second,
            transient=False, # Final output persists
        )
        self._live.start()

    def _update_live(self, renderable: RenderableType) -> None:
        if self._live:
            self._live.update(renderable, refresh=True)

    def _stop_live(self) -> None:
        if self._live:
            self._live.stop()
            self._live = None

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

    async def _on_content_block_start(self, event: Event) -> None:
        assert isinstance(event, ModelContentBlockStartEvent)
        self._current_content = ""
        
        if event.block_type == "text":
            self._start_live(Markdown(""))
        elif event.block_type == "thinking" and self.show_reasoning:
            self._start_live(Panel(Markdown(""), title="[bold yellow]🤔 Thinking[/bold yellow]", border_style="yellow"))

    async def _on_content_block_delta(self, event: Event) -> None:
        assert isinstance(event, ModelContentBlockDeltaEvent)
        if not event.delta: return

        if event.delta_type == "text":
            self._current_content += event.delta
            # Only update live if we have content
            if self._current_content:
                self._update_live(Markdown(self._current_content))
            
        elif event.delta_type == "thinking" and self.show_reasoning:
            self._current_content += event.delta
            if self._current_content:
                self._update_live(Panel(
                    Markdown(self._current_content),
                    title="[bold yellow]🤔 Thinking[/bold yellow]",
                    border_style="yellow"
                ))

    async def _on_content_block_stop(self, event: Event) -> None:
        self._stop_live()
        self._current_content = ""
        # Add a newline after text blocks if not present?
        # Rich's Live output usually ends with newline when stopped?
        # We might need explicit spacing.
        # self._console.print()

    async def _on_tool_use_block_start(self, event: Event) -> None:
        assert isinstance(event, ModelToolCallBlockStartEvent)
        self._current_tool_name = event.tool_name
        self._current_content = ""
        
        if self.show_tools:
            self._start_live(Panel(
                Text(""),
                title=f"[bold blue]🔧 Tool Call: {self._current_tool_name}[/bold blue]",
                border_style="blue"
            ))

    async def _on_tool_use_block_delta(self, event: Event) -> None:
        assert isinstance(event, ModelToolCallBlockDeltaEvent)
        self._current_content += event.arguments_delta
        
        if self.show_tools and self._live:
            # During delta, we only have partial JSON string, so we display it as is
            # Or try to format it if valid JSON? Usually delta is partial so invalid.
            self._update_live(Panel(
                Text(self._current_content),
                title=f"[bold blue]🔧 Tool Call: {self._current_tool_name}[/bold blue]",
                border_style="blue"
            ))

    async def _on_tool_use_block_stop(self, event: Event) -> None:
        """工具调用块结束 - 保持 Live 状态直到结果返回"""
        if self.show_tools and self._live:
            # Final update with pretty JSON and 'Executing...' status
            try:
                content = JSON(self._current_content)
            except:
                content = Text(self._current_content)
            
            # 保存 content 到 buffer，以便 print_result 时使用
            self._pending_tool_calls[self._current_tool_name] = self._current_content

            # 更新 Panel 显示 "Executing..."
            content_group = Group(
                content,
                Rule(style="dim"),
                Text("Executing...", style="dim italic")
            )
            
            self._update_live(Panel(
                content_group,
                title=f"[bold blue]🔧 Tool Call: {self._current_tool_name}[/bold blue]",
                border_style="blue"
            ))
            
        # 注意：这里我们故意不调用 self._stop_live()，保持 Live 活跃等待结果
        # 但这要求 tool_use_block_stop 和 tool_result 之间没有其他 block 输出
        # 如果有，其他 block 的 start 会先调用 _stop_live (在 _start_live 中)
        
        self._current_content = ""
        self._current_tool_name = ""

    def _print_tool_result(
        self,
        tool_name: str,
        success: bool,
        result_preview: Any,
        duration: float,
        arguments: dict[str, Any] | None = None
    ) -> None:
        """打印工具结果 - 更新 Live 并结束"""
        from rich.table import Table

        status_emoji = "✅" if success else "❌"
        status_color = "green" if success else "red"
        status_text = "OK" if success else "FAILED"

        # 1. 获取参数内容 (优先使用传入的 arguments 字典并格式化)
        if arguments:
            # 使用自定义格式化
            args_text = self._format_tool_arguments(arguments)
            args_content = Text.from_markup(args_text)
        else:
            # 尝试从 pending buffer 获取原始 JSON 字符串
            args_json_str = self._pending_tool_calls.pop(tool_name, "{}")
            try:
                args_content = JSON(args_json_str)
            except Exception:
                args_content = Text(args_json_str)

        # 2. 准备结果部分
        result_table = Table(show_header=False, box=None, expand=True, padding=(0, 1))
        result_table.add_column("label", width=10, style="dim cyan")
        result_table.add_column("content", ratio=1)

        result_table.add_row("Status", f"{status_emoji} {status_text} ({duration:.0f}ms)", style=f"bold {status_color}")

        if result_preview is not None:
            preview = str(result_preview)
            if not self.show_full_tool_content and len(preview) > self.max_result_length:
                preview = preview[: self.max_result_length - 3] + "..."
            result_table.add_row("Output", Text(preview, style="white"))

        # 3. 组合
        content_group = Group(
            args_content,
            Rule(style="dim"),
            result_table
        )

        panel = Panel(
            content_group,
            title=f"[bold blue]🔧 Tool Call: {tool_name}[/bold blue]",
            border_style="blue" if success else "red",
            padding=(0, 1),
        )
        
        if self._live:
            # 如果 Live 还在运行，更新它并停止
            self._update_live(panel)
            self._stop_live()
        else:
            # 如果 Live 已经停止（例如被打断），直接打印 Panel
            self._console.print(panel)

    def _print_error(self, error: str) -> None:
        """打印错误"""
        # 如果有正在运行的 Live，先停止它
        self._stop_live()
        
        panel = Panel(
            Text(error, style="red"),
            title="[bold red]❌ Error[/bold red]",
            border_style="red",
            padding=(0, 1),
        )
        self._console.print(panel)
