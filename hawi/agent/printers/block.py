"""
Block Printer Implementation
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from rich.console import Console, Group
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

class BlockPrinter(BasePrinter):
    """
    Block Printer: 使用 rich 库输出各类 content block。
    
    特性：
    - 借助 rich 库的功能支持在终端渲染 markdown
    - 不支持流式输出（即使对于流式响应，也仅当一个 content block 收集完成时才输出它）
    - 不需要撤销/修改已有内容
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
        self._tool_args_buffer = ""
        self._current_tool_name = ""
        # 缓存 tool args 以便在 result 时一起显示
        self._pending_tool_calls: dict[str, str] = {} # tool_name -> args_json

    def _format_tool_arguments(self, arguments: dict[str, Any]) -> str:
        """格式化工具参数为易读的格式。"""
        if not arguments:
            return ""

        lines: list[str] = []
        for key, value in arguments.items():
            value_str = str(value)
            if '\n' in value_str:
                lines.append(f"[bold]{key}[/bold]:")
                lines.append(value_str)
            else:
                lines.append(f"[bold]{key}[/bold]: {value_str}")

        full_text = "\n".join(lines)
        if not self.show_full_tool_content and len(full_text) > self.max_arg_length:
            full_text = full_text[:self.max_arg_length - 3] + "..."

        return full_text

    async def _on_content_block_start(self, event: Event) -> None:
        """内容块开始 - 忽略"""
        pass

    async def _on_content_block_delta(self, event: Event) -> None:
        """内容块增量 - 忽略"""
        pass

    async def _on_content_block_stop(self, event: Event) -> None:
        """内容块结束 - 输出完整块"""
        assert isinstance(event, ModelContentBlockStopEvent)
        
        if event.block_type == "text":
            text_content = ""
            for part in event.content:
                if part.get("type") == "text":
                    text_content += part.get("text", "")
            
            if text_content:
                self._console.print(Markdown(text_content))
                self._console.print() # 块后换行

        elif event.block_type == "thinking" and self.show_reasoning:
            reasoning_content = ""
            for part in event.content:
                if part.get("type") == "reasoning":
                    reasoning_content += part.get("reasoning", "")
            
            if reasoning_content:
                panel = Panel(
                    Markdown(reasoning_content),
                    title="[bold yellow]🤔 Thinking[/bold yellow]",
                    border_style="yellow",
                    padding=(0, 1),
                )
                self._console.print(panel)

    async def _on_tool_use_block_start(self, event: Event) -> None:
        """工具调用块开始 - 记录工具名"""
        assert isinstance(event, ModelToolCallBlockStartEvent)
        self._current_tool_name = event.tool_name
        self._tool_args_buffer = ""

    async def _on_tool_use_block_delta(self, event: Event) -> None:
        """工具调用块增量 - 累积参数"""
        assert isinstance(event, ModelToolCallBlockDeltaEvent)
        self._tool_args_buffer += event.arguments_delta

    async def _on_tool_use_block_stop(self, event: Event) -> None:
        """工具调用块结束 - 缓存参数，暂不输出"""
        assert isinstance(event, ModelToolCallBlockStopEvent)
        
        if self.show_tools:
            # 仅缓存，等待 result 一起输出
            # 注意：如果 result 不来（如 crash），这些信息会丢失。
            # 这是一个权衡。
            self._pending_tool_calls[self._current_tool_name] = self._tool_args_buffer
            
        self._current_tool_name = ""
        self._tool_args_buffer = ""

    def _print_tool_result(
        self,
        tool_name: str,
        success: bool,
        result_preview: Any,
        duration: float,
        arguments: dict[str, Any] | None = None
    ) -> None:
        """打印工具结果 - 合并参数和结果输出"""
        from rich.table import Table

        status_emoji = "✅" if success else "❌"
        status_color = "green" if success else "red"
        status_text = "OK" if success else "FAILED"

        # 1. 准备参数部分 (优先使用传入的 arguments 字典并格式化)
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

        if result_preview:
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
        self._console.print(panel)

    def _print_error(self, error: str) -> None:
        """打印错误"""
        panel = Panel(
            Text(error, style="red"),
            title="[bold red]❌ Error[/bold red]",
            border_style="red",
            padding=(0, 1),
        )
        self._console.print(panel)
