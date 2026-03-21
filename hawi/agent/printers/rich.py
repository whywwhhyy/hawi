"""
Rich Printer - 智能流式 Markdown 渲染器

特性：
- 支持 Streaming 模式（实时动态更新）
- 支持 Non-streaming 模式（块级输出）
- 自动检测终端能力
- 支持代码块语法高亮
- 支持 Thinking 内容显示

工作模式：
1. Streaming 模式：使用 Live 实时更新当前块，适合标准终端
2. Non-streaming 模式：块确定后才输出，适合管道、CI 等环境
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Optional

from rich.console import Console, RenderableType, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from hawi.agent.events import (
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


def _detect_streaming_support() -> bool:
    """
    检测当前终端是否支持 streaming（Live 动态更新）
    
    检测逻辑：
    1. 非 TTY → 不支持
    2. dumb/unknown 终端类型 → 不支持
    3. CI 环境 → 不支持
    4. Jupyter 环境 → 不支持
    5. 其他 → 支持
    """
    if not sys.stdout.isatty():
        return False
    
    term = os.environ.get("TERM", "").lower()
    if term in ("dumb", "unknown", ""):
        return False
    
    if os.environ.get("CI", "").lower() in ("true", "1", "yes"):
        return False
    
    if "JPY_PARENT_PID" in os.environ:
        return False
    
    return True


class RichPrinter(BasePrinter):
    """
    Rich Printer - 智能流式 Markdown 渲染器
    
    根据终端能力自动选择工作模式：
    - Streaming 模式：使用 Live 实时更新当前块
    - Non-streaming 模式：块确定后才输出
    
    Args:
        streaming: 强制指定模式（None=自动检测, True=streaming, False=non-streaming）
        console: 自定义 Console 实例
        code_theme: 代码高亮主题（默认 monokai）
        show_code_language: 是否显示代码语言标识（默认 True）
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
        code_theme: str = "friendly",
        show_code_language: bool = True,
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
        
        # Console 配置
        self._console = console or Console()
        self._code_theme = code_theme
        self._show_code_language = show_code_language

        # 确定工作模式
        env_streaming = os.environ.get("HAWI_STREAMING")
        if streaming is not None:
            self._streaming_mode = streaming
        elif env_streaming is not None:
            self._streaming_mode = env_streaming.lower() in ("1", "true", "yes")
        else:
            self._streaming_mode = _detect_streaming_support()

        # 状态
        self._buffer: str = ""
        self._block_type: str | None = None
        self._live: Live | None = None

        # 工具调用状态
        self._current_tool_name: str = ""
        self._current_tool_args: str = ""
        self._tool_calls: dict[str, dict[str, Any]] = {}

    @property
    def streaming_mode(self) -> bool:
        """当前是否处于 streaming 模式"""
        return self._streaming_mode

    # ===== Live 管理 =====

    def _start_live(self, initial_content: RenderableType = "") -> None:
        """启动 Live 显示"""
        if not self._streaming_mode or self._live is not None:
            return
        
        self._live = Live(
            initial_content,
            console=self._console,
            refresh_per_second=10,
            transient=False,  # 保留历史内容，允许滚轮查看
            vertical_overflow="visible",  # 允许内容自动滚动，而不是显示省略号
        )
        self._live.start()

    def _update_live(self, content: RenderableType) -> None:
        """更新 Live 内容"""
        if self._live:
            self._live.update(content)

    def _stop_live(self) -> None:
        """停止 Live 显示"""
        if self._live:
            self._live.stop()
            self._live = None

    # ===== 内容渲染 =====

    def _create_renderable(self, text: str, is_thinking: bool = False) -> RenderableType:
        """创建可渲染对象"""
        if not text.strip():
            return Text("")
        
        # 使用 Rich 内置的 Markdown 渲染
        md = Markdown(text, code_theme=self._code_theme)
        
        if is_thinking:
            return Panel(
                md,
                title="[bold yellow]🤔 Thinking[/bold yellow]",
                border_style="yellow",
                padding=(0, 1),  # 减少 padding 避免多余空行
            )
        
        return md

    def _render_buffer(self, final: bool = False) -> None:
        """渲染当前缓冲区内容"""
        if not self._buffer.strip():
            return
        
        is_thinking = self._block_type == "reasoning"
        content = self._create_renderable(self._buffer, is_thinking)
        
        if self._streaming_mode and not final:
            # Streaming 模式：更新 Live
            if self._live is None:
                self._start_live(content)
            else:
                self._update_live(content)
        elif final:
            # 最终输出：直接打印
            if self._streaming_mode and self._live:
                # 先更新最后一次内容
                self._update_live(content)
            else:
                # 对于 thinking block，不添加额外的换行
                self._console.print(content, end="")

    # ===== 事件处理 =====

    async def _on_content_block_start(self, event: Event) -> None:
        """内容块开始"""
        assert isinstance(event, ModelContentBlockStartEvent)
        
        # 停止之前的 Live（如果有）
        self._stop_live()
        
        # 重置状态
        self._buffer = ""
        self._block_type = event.block_type

    async def _on_content_block_delta(self, event: Event) -> None:
        """内容块增量"""
        assert isinstance(event, ModelContentBlockDeltaEvent)
        
        if not event.delta:
            return
        
        # 根据 delta_type 决定是否处理
        if event.delta_type == "text":
            self._buffer += event.delta
            self._render_buffer(final=False)
        elif event.delta_type == "reasoning" and self.show_reasoning:
            self._buffer += event.delta
            self._render_buffer(final=False)

    async def _on_content_block_stop(self, event: Event) -> None:
        """内容块结束"""
        # 最终渲染
        self._render_buffer(final=True)
        
        # 停止 Live
        self._stop_live()
        
        # 清空缓冲区
        self._buffer = ""
        self._block_type = None

    async def _on_tool_use_block_start(self, event: Event) -> None:
        """工具调用块开始"""
        assert isinstance(event, ModelToolCallBlockStartEvent)
        
        # 停止之前的 Live
        self._stop_live()
        
        self._current_tool_name = event.tool_name
        self._current_tool_args = ""
        
        # 记录工具调用信息
        self._tool_calls[event.tool_call_id] = {
            "tool_name": event.tool_name,
            "start_time": self._get_timestamp(),
        }
        
        if self.show_tools and self._streaming_mode:
            # Streaming 模式：显示初始状态
            timestamp = self._get_timestamp()
            panel = Panel(
                Text("Receiving arguments...", style="dim"),
                title=f"[bold blue]🔧 {event.tool_name} ({timestamp})[/bold blue]",
                border_style="blue",
            )
            self._start_live(panel)

    async def _on_tool_use_block_delta(self, event: Event) -> None:
        """工具调用块增量"""
        assert isinstance(event, ModelToolCallBlockDeltaEvent)
        
        self._current_tool_args += event.arguments_delta
        
        if self.show_tools and self._streaming_mode and self._live:
            # 更新 Live 显示
            timestamp = self._get_timestamp()
            panel = Panel(
                Text(self._current_tool_args or "..."),
                title=f"[bold blue]🔧 {self._current_tool_name} ({timestamp})[/bold blue]",
                border_style="blue",
            )
            self._update_live(panel)

    async def _on_tool_use_block_stop(self, event: Event) -> None:
        """工具调用块结束"""
        # 停止 Live（工具结果会由 _on_tool_result 处理）
        self._stop_live()

    # ===== 工具结果和错误 =====

    def _print_tool_result(
        self,
        tool_name: str,
        success: bool,
        result_preview: Any,
        duration: float,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        """打印工具结果"""
        timestamp = self._get_timestamp()
        status_emoji = "✅" if success else "❌"
        status_color = "green" if success else "red"
        status_text = "OK" if success else "FAILED"

        # 构建内容
        parts = []
        
        # 参数部分
        if arguments:
            args_text = Text()
            for key, value in arguments.items():
                value_str = str(value)
                # 截断长值
                if len(value_str) > 100:
                    value_str = value_str[:97] + "..."
                args_text.append(f"{key}: ", style="bold")
                args_text.append(f"{value_str}\n")
            parts.append(args_text)
            parts.append(Rule(style="dim"))
        
        # 结果部分
        result_table = Table(show_header=False, box=None, expand=True, padding=(0, 1))
        result_table.add_column("label", width=10, style="dim cyan")
        result_table.add_column("content", ratio=1)
        
        result_table.add_row(
            "Status",
            f"{status_emoji} {status_text} ({duration:.0f}ms)",
            style=f"bold {status_color}",
        )
        
        if result_preview is not None:
            preview = str(result_preview)
            if not self.show_full_tool_content and len(preview) > self.max_result_length:
                preview = preview[: self.max_result_length - 3] + "..."
            result_table.add_row("Output", Text(preview, style="white"))
        
        parts.append(result_table)

        # 创建面板
        panel = Panel(
            Group(*parts),
            title=f"[bold blue]🔧 {tool_name} ({timestamp})[/bold blue]",
            border_style="blue" if success else "red",
            padding=(0, 1),
        )

        self._console.print(panel)

    def _print_error(self, error: str) -> None:
        """打印错误"""
        # 停止 Live
        self._stop_live()
        
        timestamp = self._get_timestamp()
        panel = Panel(
            Text(error, style="red"),
            title=f"[bold red]❌ Error ({timestamp})[/bold red]",
            border_style="red",
            padding=(0, 1),
        )
        self._console.print(panel)

    def _print_usage(self, usage: Any) -> None:
        """打印 token 用量"""
        if usage is None:
            return
        
        # 提取 token 信息（兼容不同字段命名）
        input_tokens = getattr(usage, 'input_tokens', None) or getattr(usage, 'prompt_tokens', None)
        output_tokens = getattr(usage, 'output_tokens', None) or getattr(usage, 'completion_tokens', None)
        total_tokens = getattr(usage, 'total_tokens', None)
        
        # 缓存相关 tokens
        cache_read = getattr(usage, 'cache_read_tokens', None)
        cache_write = getattr(usage, 'cache_write_tokens', None)
        
        if not any([input_tokens, output_tokens, total_tokens, cache_read, cache_write]):
            return
        
        # 构建显示文本
        parts = []
        if input_tokens is not None:
            parts.append(f"↓ {input_tokens}")
        if output_tokens is not None:
            parts.append(f"↑ {output_tokens}")
        
        # 缓存 tokens
        cache_parts = []
        if cache_read:
            cache_parts.append(f"↺ {cache_read}")
        if cache_write:
            cache_parts.append(f"✎ {cache_write}")
        if cache_parts:
            parts.append(" ".join(cache_parts))
        
        if total_tokens is not None:
            parts.append(f"Σ {total_tokens}")
        
        if parts:
            usage_text = "  |  ".join(parts)
            self._console.print(
                f"[dim]Tokens: {usage_text}[/dim]",
                justify="right"
            )