"""
Streaming Markdown Printer

基于块级增量解析的流式 Markdown 渲染器。

核心优化：
1. 块级分割 - 识别 Markdown 块边界（空行分隔）
2. 增量输出 - 完成的块立即输出，不等待流结束
3. 动态更新 - 当前未完成块使用 Live 实时更新
4. 自动清理 - 流结束时自动处理剩余内容

技术方案：
- 使用 markdown-it 解析块边界
- 使用 rich.live.Live 实现动态更新
- 使用 rich.markdown.Markdown 渲染内容
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from markdown_it import MarkdownIt
from rich.console import Console, RenderableType
from rich.live import Live
from rich.markdown import Markdown as RichMarkdown
from rich.panel import Panel
from rich.text import Text

from hawi.agent.events import (
    Event,
    ModelContentBlockDeltaEvent,
    ModelContentBlockStartEvent,
    ModelContentBlockStopEvent,
)
from hawi.agent.printers.base import BasePrinter

logger = logging.getLogger(__name__)


class StreamingMarkdownPrinter(BasePrinter):
    """
    流式 Markdown Printer
    
    使用增量解析策略，大幅提升长文档流式渲染性能。
    
    工作原理：
    1. 接收文本片段，累积到缓冲区
    2. 识别块边界（双换行符分隔）
    3. 已完成的块立即输出到终端
    4. 当前未完成块使用 Live 动态更新
    5. 流结束时，停止 Live 并输出剩余内容
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

        # Markdown 解析器
        self._md = MarkdownIt("commonmark").enable("table")

        # 流式状态
        self._buffer = ""  # 累积的原始文本
        self._is_thinking = False
        self._in_live_mode = False  # 是否处于 Live 模式

        # Live 显示
        self._live: Optional[Live] = None

        # 工具调用状态
        self._current_tool_name = ""
        self._current_tool_args = ""

    def _start_live(self) -> None:
        """启动 Live 显示"""
        if self._live:
            return
        self._live = Live(
            "",
            console=self._console,
            refresh_per_second=self._refresh_per_second,
            transient=True,  # 使用 transient，Live 停止时内容消失
            auto_refresh=True,
        )
        self._live.start()
        self._in_live_mode = True

    def _stop_live(self) -> None:
        """停止 Live 显示"""
        if self._live:
            self._live.stop()
            self._live = None
        self._in_live_mode = False

    def _update_live(self, content: RenderableType) -> None:
        """更新 Live 内容"""
        if self._live:
            self._live.update(content, refresh=True)

    def _feed_text(self, text: str) -> None:
        """
        接收文本片段，进行增量解析和渲染
        
        Args:
            text: 新接收的文本
        """
        self._buffer += text
        
        # 检查是否有完整的块（以 \n\n 结尾）
        while "\n\n" in self._buffer:
            # 找到第一个双换行
            idx = self._buffer.find("\n\n")
            if idx == -1:
                break
            
            # 提取完整块
            complete_block = self._buffer[:idx]
            self._buffer = self._buffer[idx + 2:]  # 跳过 \n\n
            
            if complete_block.strip():
                # 停止 Live，输出完成的块
                self._stop_live()
                self._render_and_print(complete_block, final=True)
        
        # 剩余内容使用 Live 更新
        if self._buffer.strip() or self._in_live_mode:
            self._start_live()
            self._render_and_print(self._buffer, final=False)

    def _render_and_print(self, text: str, final: bool = True) -> None:
        """渲染并打印/更新文本"""
        if not text.strip():
            return
        
        md = RichMarkdown(text)
        
        if self._is_thinking:
            content = Panel(
                md,
                title="[bold yellow]🤔 Thinking[/bold yellow]",
                border_style="yellow"
            )
        else:
            content = md
        
        if final:
            self._console.print(content)
        else:
            self._update_live(content)

    def _finalize(self) -> None:
        """最终化当前块（流结束时的处理）"""
        # 停止 Live（transient=True 意味着 Live 内容会消失）
        self._stop_live()
        
        # 输出剩余内容
        if self._buffer.strip():
            self._render_and_print(self._buffer, final=True)
        
        # 重置状态
        self._buffer = ""

    # ===== Event Handlers =====

    async def _on_content_block_start(self, event: Event) -> None:
        """内容块开始"""
        assert isinstance(event, ModelContentBlockStartEvent)
        
        # 重置状态
        self._buffer = ""
        self._is_thinking = event.block_type == "thinking"

    async def _on_content_block_delta(self, event: Event) -> None:
        """内容块增量"""
        assert isinstance(event, ModelContentBlockDeltaEvent)
        
        if not event.delta:
            return
        
        is_thinking = event.delta_type == "thinking"
        
        if event.delta_type == "text" or (is_thinking and self.show_reasoning):
            self._is_thinking = is_thinking
            self._feed_text(event.delta)

    async def _on_content_block_stop(self, event: Event) -> None:
        """内容块结束"""
        self._finalize()
        self._console.print()  # 空行分隔

    async def _on_tool_use_block_start(self, event: Event) -> None:
        """工具调用块开始"""
        from hawi.agent.events import ModelToolCallBlockStartEvent
        assert isinstance(event, ModelToolCallBlockStartEvent)
        
        self._current_tool_name = event.tool_name
        self._current_tool_args = ""
        
        if self.show_tools:
            timestamp = self._get_timestamp()
            self._start_live()
            self._update_live(Panel(
                Text(""),
                title=f"[bold blue]🔧 Tool Call: {self._current_tool_name} ({timestamp})[/bold blue]",
                border_style="blue"
            ))

    async def _on_tool_use_block_delta(self, event: Event) -> None:
        """工具调用块增量"""
        from hawi.agent.events import ModelToolCallBlockDeltaEvent
        assert isinstance(event, ModelToolCallBlockDeltaEvent)
        
        self._current_tool_args += event.arguments_delta
        
        if self.show_tools and self._live:
            timestamp = self._get_timestamp()
            self._update_live(Panel(
                Text(self._current_tool_args),
                title=f"[bold blue]🔧 Tool Call: {self._current_tool_name} ({timestamp})[/bold blue]",
                border_style="blue"
            ))

    async def _on_tool_use_block_stop(self, event: Event) -> None:
        """工具调用块结束"""
        self._stop_live()
        # 参数将在 _on_tool_call 中显示

    def _print_tool_result(
        self,
        tool_name: str,
        success: bool,
        result_preview: Any,
        duration: float,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        """打印工具结果"""
        from rich.json import JSON
        from rich.rule import Rule
        from rich.table import Table
        from rich.console import Group

        timestamp = self._get_timestamp()
        status_emoji = "✅" if success else "❌"
        status_color = "green" if success else "red"
        status_text = "OK" if success else "FAILED"

        # 参数部分
        if arguments:
            args_lines = []
            for key, value in arguments.items():
                value_str = str(value)
                if '\n' in value_str:
                    args_lines.append(f"[bold]{key}[/bold]:")
                    args_lines.append(value_str)
                else:
                    args_lines.append(f"[bold]{key}[/bold]: {value_str}")
            args_content = Text.from_markup("\n".join(args_lines))
        else:
            # 尝试解析累积的 JSON
            try:
                args_content = JSON(self._current_tool_args)
            except Exception:
                args_content = Text(self._current_tool_args or "(no arguments)", style="dim")

        # 结果表格
        result_table = Table(show_header=False, box=None, expand=True, padding=(0, 1))
        result_table.add_column("label", width=10, style="dim cyan")
        result_table.add_column("content", ratio=1)
        result_table.add_row("Status", f"{status_emoji} {status_text} ({duration:.0f}ms)", style=f"bold {status_color}")
        
        if result_preview is not None:
            preview = str(result_preview)
            if not self.show_full_tool_content and len(preview) > self.max_result_length:
                preview = preview[: self.max_result_length - 3] + "..."
            result_table.add_row("Output", Text(preview, style="white"))

        # 组合
        content_group = Group(
            args_content,
            Rule(style="dim"),
            result_table
        )

        panel = Panel(
            content_group,
            title=f"[bold blue]🔧 Tool Call: {tool_name} ({timestamp})[/bold blue]",
            border_style="blue" if success else "red",
            padding=(0, 1),
        )

        self._console.print(panel)
        self._console.print()

    def _print_error(self, error: str) -> None:
        """打印错误"""
        self._stop_live()
        
        timestamp = self._get_timestamp()
        panel = Panel(
            Text(error, style="red"),
            title=f"[bold red]❌ Error ({timestamp})[/bold red]",
            border_style="red",
            padding=(0, 1),
        )
        self._console.print(panel)
        self._console.print()

    def _print_usage(self, usage: Any) -> None:
        """打印 token 用量（简化，不显示）"""
        pass  # 流式 printer 不显示 usage，保持界面简洁
