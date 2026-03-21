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

代码块样式：
- 基于 markdown-it-py 解析，可自定义 fence 规则
- 支持 Pygments 主题选择（code_theme）
- 支持通过 Console 主题自定义整体样式

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
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
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


class CodeBlockRenderer:
    """
    自定义代码块渲染器
    
    利用 markdown-it-py 的 token 信息，为代码块添加自定义样式：
    - 边框
    - 背景色
    - 语言标识
    - 行号
    """
    
    def __init__(
        self,
        theme: str = "monokai",
        border_style: str = "dim",
        background_color: Optional[str] = None,
        show_language: bool = True,
        show_line_numbers: bool = False,
    ):
        self.theme = theme
        self.border_style = border_style
        self.background_color = background_color
        self.show_language = show_language
        self.show_line_numbers = show_line_numbers
    
    def render(self, code: str, language: Optional[str] = None) -> RenderableType:
        """渲染代码块为带样式的 Panel"""
        # 创建语法高亮
        if language:
            syntax = Syntax(
                code,
                language,
                theme=self.theme,
                background_color=self.background_color,
                line_numbers=self.show_line_numbers,
                word_wrap=True,
            )
        else:
            # 无语言时使用普通文本
            syntax = Text(
                code,
                style=f"on {self.background_color}" if self.background_color else None
            )
        
        # 构建标题
        title = None
        if self.show_language and language:
            title = f"📄 {language}"
        elif self.show_language:
            title = "📄 text"
        
        # 包装在 Panel 中
        return Panel(
            syntax,
            border_style=self.border_style,
            title=title,
            title_align="left",
            padding=(0, 1),
        )


class StyledMarkdown:
    """
    自定义 Markdown 渲染器，支持代码块样式自定义
    
    通过 markdown-it-py 解析 token，识别代码块并应用自定义样式
    """
    
    def __init__(
        self,
        markup: str,
        code_renderer: Optional[CodeBlockRenderer] = None,
        code_theme: str = "monokai",
    ):
        self.markup = markup
        self.code_renderer = code_renderer or CodeBlockRenderer(theme=code_theme)
        # 使用 markdown-it-py 解析
        self._md = MarkdownIt("commonmark").enable("table")
    
    def __rich__(self) -> RenderableType:
        """Rich 渲染协议"""
        from rich.console import Group
        
        tokens = self._md.parse(self.markup)
        renderables: list[RenderableType] = []
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            if token.type == "fence":
                # 代码块 - 使用自定义渲染器
                language = token.info.strip() if token.info else None
                code = token.content
                renderables.append(self.code_renderer.render(code, language))
                i += 1
            
            elif token.type == "paragraph_open":
                # 普通段落
                # 收集段落内容直到 paragraph_close
                content_parts = []
                i += 1
                while i < len(tokens) and tokens[i].type != "paragraph_close":
                    if tokens[i].type == "inline":
                        content_parts.append(tokens[i].content)
                    i += 1
                if content_parts:
                    # 使用 RichMarkdown 渲染段落
                    para_text = "".join(content_parts)
                    md = RichMarkdown(para_text, code_theme=self.code_renderer.theme)
                    renderables.append(md)
                i += 1  # 跳过 paragraph_close
            
            elif token.type == "heading_open":
                # 标题
                level = int(token.tag[1]) if token.tag.startswith("h") else 1
                i += 1
                if i < len(tokens) and tokens[i].type == "inline":
                    text = tokens[i].content
                    renderables.append(Text(text, style=f"bold {'#' * level}"))
                i += 2  # 跳过 inline 和 heading_close
            
            elif token.type == "bullet_list_open":
                # 无序列表
                list_items = []
                i += 1
                while i < len(tokens) and tokens[i].type != "bullet_list_close":
                    if tokens[i].type == "list_item_open":
                        i += 1
                        if i < len(tokens) and tokens[i].type == "inline":
                            list_items.append(f"• {tokens[i].content}")
                        i += 1  # 跳过 list_item_close
                    i += 1
                if list_items:
                    renderables.append(Text("\n".join(list_items)))
                i += 1  # 跳过 bullet_list_close
            
            elif token.type == "ordered_list_open":
                # 有序列表
                list_items = []
                num = 1
                i += 1
                while i < len(tokens) and tokens[i].type != "ordered_list_close":
                    if tokens[i].type == "list_item_open":
                        i += 1
                        if i < len(tokens) and tokens[i].type == "inline":
                            list_items.append(f"{num}. {tokens[i].content}")
                            num += 1
                        i += 1
                    i += 1
                if list_items:
                    renderables.append(Text("\n".join(list_items)))
                i += 1
            
            elif token.type == "blockquote_open":
                # 引用块
                content_parts = []
                i += 1
                while i < len(tokens) and tokens[i].type != "blockquote_close":
                    if tokens[i].type == "inline":
                        content_parts.append(tokens[i].content)
                    i += 1
                if content_parts:
                    text = " ".join(content_parts)
                    renderables.append(Panel(
                        Text(text, style="italic green"),
                        border_style="green",
                        title="Quote",
                        title_align="left",
                    ))
                i += 1
            
            elif token.type == "hr":
                # 分隔线
                renderables.append(Rule(style="dim"))
                i += 1
            
            else:
                i += 1
        
        if len(renderables) == 1:
            return renderables[0]
        return Group(*renderables) if renderables else Text("")


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
    
    代码块样式：
    - 基于 markdown-it-py 解析，支持自定义 fence 规则
    - 支持边框、背景色、语言标识、行号
    - 支持 Pygments 主题选择（code_theme）
    
    Args:
        streaming: 强制指定模式（None=自动检测, True=streaming, False=non-streaming）
        console: 自定义 Console 实例
        refresh_per_second: Live 刷新频率（streaming 模式有效）
        code_theme: 代码块语法高亮主题（默认 monokai）
        code_border_style: 代码块边框样式（默认 dim）
        code_background: 代码块背景色（默认 None）
        code_show_language: 是否显示代码语言标识（默认 True）
        code_line_numbers: 是否显示行号（默认 False）
    
    Example:
        # 自动检测模式
        printer = RichPrinter()
        
        # 自定义代码块样式
        printer = RichPrinter(
            code_theme="dracula",
            code_border_style="blue",
            code_background="#1e1e1e",
            code_show_language=True,
            code_line_numbers=True,
        )
    """

    # 可用的代码高亮主题
    CODE_THEMES = [
        "monokai", "dracula", "github-dark", "github-light",
        "one-dark", "solarized-dark", "solarized-light",
        "gruvbox-dark", "gruvbox-light",
    ]

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
        code_theme: str = "monokai",
        code_border_style: str = "dim",
        code_background: Optional[str] = None,
        code_show_language: bool = True,
        code_line_numbers: bool = False,
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
        
        # 代码主题验证
        if code_theme not in self.CODE_THEMES:
            logger.warning(f"Unknown code theme: {code_theme}, using 'monokai'")
            code_theme = "monokai"
        
        # 创建代码块渲染器
        self._code_renderer = CodeBlockRenderer(
            theme=code_theme,
            border_style=code_border_style,
            background_color=code_background,
            show_language=code_show_language,
            show_line_numbers=code_line_numbers,
        )
        self._code_theme = code_theme
        
        # Console 配置
        self._console = console or Console()
        self._refresh_per_second = refresh_per_second

        # 确定工作模式
        env_streaming = os.environ.get("HAWI_STREAMING")
        if streaming is not None:
            self._streaming_mode = streaming
        elif env_streaming is not None:
            self._streaming_mode = env_streaming.lower() in ("1", "true", "yes")
        else:
            self._streaming_mode = _detect_streaming_support()

        # Markdown 解析器
        self._md = MarkdownIt("commonmark").enable("table")

        # 状态
        self._buffer = ""
        self._is_thinking = False
        self._in_live_mode = False
        self._live: Optional[Live] = None

        # 工具调用状态
        self._current_tool_name = ""
        self._current_tool_args = ""

    @property
    def streaming_mode(self) -> bool:
        """当前是否处于 streaming 模式"""
        return self._streaming_mode

    @property
    def code_theme(self) -> str:
        """当前代码高亮主题"""
        return self._code_theme

    def _create_markdown(self, text: str) -> StyledMarkdown:
        """创建带样式的 Markdown 对象"""
        return StyledMarkdown(
            text,
            code_renderer=self._code_renderer,
            code_theme=self._code_theme,
        )

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
        """更新 Live 内容"""
        if self._live:
            self._live.update(content, refresh=True)

    def _feed_text(self, text: str, is_thinking: bool = False) -> None:
        """接收文本片段，进行增量解析
        
        Args:
            text: 文本内容
            is_thinking: 当前块是否为 thinking 类型（在调用时确定，避免状态竞争）
        """
        self._buffer += text
        
        while "\n\n" in self._buffer:
            idx = self._buffer.find("\n\n")
            if idx == -1:
                break
            
            complete_block = self._buffer[:idx]
            self._buffer = self._buffer[idx + 2:]
            
            if complete_block.strip():
                if self._streaming_mode:
                    self._stop_live()
                self._render_and_print(complete_block, is_thinking=is_thinking, final=True)
        
        if self._streaming_mode:
            if self._buffer.strip() or self._in_live_mode:
                self._start_live()
                self._render_and_print(self._buffer, is_thinking=is_thinking, final=False)

    def _render_and_print(self, text: str, is_thinking: bool = False, final: bool = True) -> None:
        """渲染并打印文本
        
        Args:
            text: 文本内容
            is_thinking: 是否为 thinking 类型内容
            final: 是否为最终输出（非 Live 更新）
        """
        if not text.strip():
            return
        
        md = self._create_markdown(text)
        
        if is_thinking:
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

    def _finalize(self, is_thinking: bool = False) -> None:
        """最终化当前块
        
        Args:
            is_thinking: 当前块是否为 thinking 类型
        """
        if self._streaming_mode:
            self._stop_live()
        
        if self._buffer.strip():
            self._render_and_print(self._buffer, is_thinking=is_thinking, final=True)
        
        self._buffer = ""

    # ===== Event Handlers =====

    async def _on_content_block_start(self, event: Event) -> None:
        """内容块开始"""
        assert isinstance(event, ModelContentBlockStartEvent)
        # 确保先清空之前的状态
        if self._streaming_mode:
            self._stop_live()
        self._buffer = ""
        # 根据 block_type 设置 thinking 状态
        self._is_thinking = event.block_type == "reasoning"

    async def _on_content_block_delta(self, event: Event) -> None:
        """内容块增量"""
        assert isinstance(event, ModelContentBlockDeltaEvent)
        
        if not event.delta:
            return
        
        # delta_type 应该与 block_type 一致
        # 传递 _is_thinking 状态，避免在 _feed_text 中读取时状态已改变
        if event.delta_type == "text":
            self._feed_text(event.delta, is_thinking=False)
        elif event.delta_type == "reasoning" and self.show_reasoning and self._is_thinking:
            self._feed_text(event.delta, is_thinking=True)

    async def _on_content_block_stop(self, event: Event) -> None:
        """内容块结束"""
        # 使用当前的 _is_thinking 状态来最终化
        self._finalize(is_thinking=self._is_thinking)
        self._console.print()
        # 重置状态，避免影响下一个 block
        self._is_thinking = False

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
        from rich.console import Group
        from rich.json import JSON

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
        """打印 token 用量（包含 cache read/write）"""
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
        
        # 缓存 tokens（如果存在）
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
