"""
Rich Printer - 智能流式 Markdown 渲染器

支持两种工作模式：
1. Streaming 模式（默认）：实时动态更新当前块，适合标准终端
2. Non-streaming 模式：块确定后才打印，适合不支持动态更新的终端

自动检测终端能力：
- 检测 TTY 状态
- 检测终端类型（dumb/unknown 视为 non-streaming）
- 支持环境变量覆盖（HAWI_STREAMING=0/1）
- 支持参数强制指定（streaming=True/False）

技术方案：
- 块级分割：识别 Markdown 块边界（双换行分隔）
- 增量渲染：streaming 模式下当前块使用 Live 实时更新
- 延迟输出：non-streaming 模式下块完成才输出
"""

from __future__ import annotations

import logging
import os
import sys
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


def _detect_streaming_support() -> bool:
    """
    检测当前终端是否支持 streaming（Live 动态更新）
    
    检测逻辑（按优先级）：
    1. 非 TTY → 不支持
    2. dumb/unknown 终端类型 → 不支持
    3. CI 环境（CI=true） → 不支持
    4. 其他 → 支持
    """
    # 非 TTY 不支持
    if not sys.stdout.isatty():
        return False
    
    # 检查终端类型
    term = os.environ.get("TERM", "").lower()
    if term in ("dumb", "unknown", ""):
        return False
    
    # CI 环境通常不支持动态更新
    if os.environ.get("CI", "").lower() in ("true", "1", "yes"):
        return False
    
    # Jupyter/Notebook 环境检查
    if "JPY_PARENT_PID" in os.environ:
        return False
    
    return True


class RichPrinter(BasePrinter):
    """
    Rich Printer - 智能流式 Markdown 渲染器
    
    根据终端能力自动选择工作模式：
    
    **Streaming 模式**（动态更新）：
    - 已完成块立即输出
    - 当前块使用 Live 实时更新
    - 适合：标准终端、支持 ANSI 的终端
    
    **Non-streaming 模式**（块级输出）：
    - 块确定后（双换行）才输出
    - 无 Live 动态更新
    - 适合：管道、文件重定向、dumb 终端、CI 环境
    
    Args:
        streaming: 强制指定模式（None=自动检测, True=streaming, False=non-streaming）
        console: 自定义 Console 实例
        refresh_per_second: Live 刷新频率（streaming 模式有效）
    
    Example:
        # 自动检测模式
        printer = RichPrinter()
        
        # 强制 streaming 模式
        printer = RichPrinter(streaming=True)
        
        # 强制 non-streaming 模式
        printer = RichPrinter(streaming=False)
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
        streaming: bool | None = None,
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

        # 确定工作模式（参数 > 环境变量 > 自动检测）
        env_streaming = os.environ.get("HAWI_STREAMING")
        if streaming is not None:
            # 参数强制指定
            self._streaming_mode = streaming
        elif env_streaming is not None:
            # 环境变量覆盖
            self._streaming_mode = env_streaming.lower() in ("1", "true", "yes")
        else:
            # 自动检测
            self._streaming_mode = _detect_streaming_support()

        # Markdown 解析器
        self._md = MarkdownIt("commonmark").enable("table")

        # 状态
        self._buffer = ""
        self._is_thinking = False
        self._in_live_mode = False

        # Live 显示（仅 streaming 模式）
        self._live: Optional[Live] = None

        # 工具调用状态
        self._current_tool_name = ""
        self._current_tool_args = ""

    @property
    def streaming_mode(self) -> bool:
        """当前是否处于 streaming 模式"""
        return self._streaming_mode

    def _start_live(self) -> None:
        """启动 Live 显示（仅 streaming 模式）"""
        if not self._streaming_mode or self._live:
            return
        self._live = Live(
            "",
            console=self._console,
            refresh_per_second=self._refresh_per_second,
            transient=True,
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
        """更新 Live 内容（仅 streaming 模式）"""
        if self._live:
            self._live.update(content, refresh=True)

    def _feed_text(self, text: str) -> None:
        """
        接收文本片段，根据模式选择处理方式
        
        Streaming 模式：
        - 完整块立即输出
        - 当前块 Live 更新
        
        Non-streaming 模式：
        - 只累积，不输出
        - 块完成时一次性输出
        """
        self._buffer += text
        
        # 检查是否有完整的块（以 \n\n 结尾）
        while "\n\n" in self._buffer:
            idx = self._buffer.find("\n\n")
            if idx == -1:
                break
            
            complete_block = self._buffer[:idx]
            self._buffer = self._buffer[idx + 2:]
            
            if complete_block.strip():
                if self._streaming_mode:
                    self._stop_live()
                self._render_and_print(complete_block, final=True)
        
        # Streaming 模式：剩余内容使用 Live 更新
        if self._streaming_mode:
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
        elif self._streaming_mode:
            self._update_live(content)

    def _finalize(self) -> None:
        """最终化当前块（流结束时的处理）"""
        if self._streaming_mode:
            self._stop_live()
        
        # 输出剩余内容
        if self._buffer.strip():
            self._render_and_print(self._buffer, final=True)
        
        self._buffer = ""

    # ===== Event Handlers =====

    async def _on_content_block_start(self, event: Event) -> None:
        """内容块开始"""
        assert isinstance(event, ModelContentBlockStartEvent)
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
        self._console.print()

    async def _on_tool_use_block_start(self, event: Event) -> None:
        """工具调用块开始"""
        from hawi.agent.events import ModelToolCallBlockStartEvent
        assert isinstance(event, ModelToolCallBlockStartEvent)
        
        self._current_tool_name = event.tool_name
        self._current_tool_args = ""
        
        if self.show_tools:
            timestamp = self._get_timestamp()
            if self._streaming_mode:
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
        
        if self.show_tools and self._streaming_mode and self._live:
            timestamp = self._get_timestamp()
            self._update_live(Panel(
                Text(self._current_tool_args),
                title=f"[bold blue]🔧 Tool Call: {self._current_tool_name} ({timestamp})[/bold blue]",
                border_style="blue"
            ))

    async def _on_tool_use_block_stop(self, event: Event) -> None:
        """工具调用块结束"""
        self._stop_live()

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
        """打印 token 用量"""
        pass
