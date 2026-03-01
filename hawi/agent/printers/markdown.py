"""
Hawi Printer Implementations

提供多种事件打印机实现：
- RichStreamingPrinter: 原始 ANSI 颜色流式打印
- MarkdownStreamingPrinter: Markdown 实时渲染打印机
"""

from __future__ import annotations

import logging
import time
import re
from typing import Any

from rich.console import Console
from rich import box
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.live import Live
from rich.table import Table

from markdown_it import MarkdownIt
from mdit_py_plugins.tasklists import tasklists_plugin
from mdit_py_plugins.container import container_plugin
from mdit_py_plugins.admon import admon_plugin
from markdown_it.token import Token

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


class ConfigurableMarkdown(Markdown):
    """
    扩展 Rich Markdown 以支持自定义 parser。
    这允许我们显式使用 markdown-it-py 及其插件。
    """

    def __init__(self, markup: str, parser: MarkdownIt, **kwargs):
        super().__init__("", **kwargs)
        self.markup = markup
        self.parsed = parser.parse(markup)


class TokenMarkdown(Markdown):
    """
    支持直接传入 Token 列表的 Markdown 渲染类。
    用于流式增量渲染：我们手动解析 buffer，计算出差异，然后构造此对象进行渲染。
    """

    def __init__(self, tokens: list[Any], **kwargs):
        super().__init__("", **kwargs)
        self.parsed = tokens


class StreamMarkdownPrinter(BasePrinter):
    """
    StreamMarkdownPrinter - 支持流式 Markdown 输出的打印机

    特性：
    1. 正常文本立即流式输出
    2. 检测到特殊格式字符（*, `, [, ] 等）时开始缓冲
    3. 格式完整闭合后才一起渲染输出
    4. 块结束时如果格式未闭合，以纯文本输出缓冲内容
    """

    # 可能开启特殊格式的字符
    _FORMAT_START_CHARS = frozenset(['*', '`', '[', ']', '_', '~'])

    def __init__(
        self,
        console: Console | None = None,
        code_theme: str = "monokai",
        ellipsize_tables: bool = False,
        show_tools: bool = True,
        show_reasoning: bool = True,
        show_errors: bool = True,
        show_error_stack: bool = True,
        max_arg_length: int = 80,
        max_result_length: int = 200,
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
        self._console = console or Console()
        self._code_theme = code_theme
        self._ellipsize_tables = ellipsize_tables

        self._parser = (
            MarkdownIt("gfm-like")
            .enable("table")
            .enable("strikethrough")
            .use(tasklists_plugin)
            .use(container_plugin, name="warning")
            .use(admon_plugin)
        )

        # 主缓冲区 - 已确认的内容
        self._buffer = ""
        self._committed_tokens_len = 0

        # 格式检测缓冲区 - 可能包含未完成格式的内容
        self._format_buffer = ""
        self._in_format_detection = False

        self._live: Live | None = None
        self._current_block_type: str | None = None
        self._reasoning_buffer: str = ""
        self._block_count: int = 0

    async def handle(self, event: Event) -> None:
        handlers = {
            "model.content_block_start": self._on_content_block_start,
            "model.content_block_delta": self._on_content_block_delta,
            "model.content_block_stop": self._on_content_block_stop,
            "model.tool_use_block_start": self._on_tool_use_block_start,
            "model.tool_use_block_delta": self._on_tool_use_block_delta,
            "model.tool_use_block_stop": self._on_tool_use_block_stop,
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
        self._buffer = ""
        self._committed_tokens_len = 0
        self._format_buffer = ""
        self._in_format_detection = False
        self._block_count = 0

    async def _on_stream_stop(self, event: Event) -> None:
        """Model 流式响应结束"""
        # Flush any pending format buffer
        if self._format_buffer:
            self._flush_format_buffer()
        if self._live:
            self._live.stop()
            self._live = None
        self._current_block_type = None

    async def _on_content_block_start(self, event: Event) -> None:
        """内容块开始"""
        assert isinstance(event, ModelContentBlockStartEvent)
        block_type = event.block_type
        self._current_block_type = block_type

        # 在每个 block 前添加额外换行（第一个除外）
        if self._block_count > 0:
            self._buffer += "\n\n"
            self._update_display()
        self._block_count += 1

    async def _on_content_block_delta(self, event: Event) -> None:
        assert isinstance(event, ModelContentBlockDeltaEvent)
        delta = event.delta
        delta_type = event.delta_type

        if delta_type == "text":
            self._process_text_delta(delta)
        elif delta_type == "thinking" and self.show_reasoning:
            self._reasoning_buffer += delta

    def _process_text_delta(self, delta: str) -> None:
        """处理文本 delta，实现格式检测和缓冲逻辑"""
        for char in delta:
            if self._in_format_detection:
                self._format_buffer += char
                # 检查格式是否完整
                if self._is_format_complete(self._format_buffer):
                    # 格式完整，可以安全渲染
                    self._buffer += self._format_buffer
                    self._format_buffer = ""
                    self._in_format_detection = False
                    self._update_display()
                elif self._is_format_aborted(self._format_buffer):
                    # 格式不可能完成，刷新缓冲区为纯文本
                    self._flush_format_buffer()
            else:
                if char in self._FORMAT_START_CHARS:
                    # 检测到可能开启格式的字符，开始缓冲
                    self._in_format_detection = True
                    self._format_buffer = char
                else:
                    # 普通字符，直接加入缓冲区
                    self._buffer += char
                    self._update_display()

    def _is_format_complete(self, text: str) -> bool:
        """检查文本中的格式是否完整闭合"""
        # 检查粗体 **text**
        if re.match(r'\*\*[^*]+\*\*$', text):
            return True
        # 检查斜体 *text* (但不是 **)
        if re.match(r'\*[^*]+\*$', text) and not text.startswith('**'):
            return True
        # 检查行内代码 `text`
        if re.match(r'`[^`]+`$', text):
            return True
        # 检查链接 [text](url)
        if re.match(r'\[([^\]]+)\]\([^)]+\)$', text):
            return True
        # 检查删除线 ~~text~~
        if re.match(r'~~[^~]+~~$', text):
            return True
        return False

    def _is_format_aborted(self, text: str) -> bool:
        """检查格式是否不可能完成（如换行后）"""
        # 如果缓冲区中有换行，且格式跨行，则放弃
        if '\n' in text:
            # 检查是否是代码块（允许跨行）
            if text.startswith('```'):
                return False  # 代码块可以跨行
            # 检查行内格式跨行
            if text.startswith('**') or text.startswith('*') or text.startswith('`'):
                return True
        # 如果缓冲区太长且没有闭合，放弃
        if len(text) > 200:
            # 但代码块可以更长
            if not text.startswith('```'):
                return True
        return False

    def _flush_format_buffer(self) -> None:
        """将格式缓冲区作为纯文本刷新到主缓冲区"""
        if self._format_buffer:
            self._buffer += self._format_buffer
            self._format_buffer = ""
            self._in_format_detection = False
            self._update_display()

    async def _on_content_block_stop(self, event: Event) -> None:
        """内容块结束"""
        assert isinstance(event, ModelContentBlockStopEvent)

        # 刷新任何待处理的格式缓冲区
        if self._format_buffer:
            self._flush_format_buffer()

        if self._live:
            self._live.stop()
            self._live = None

        block_type = self._current_block_type

        if block_type == "tool_use" and self.show_tools:
            # 从 content 中提取工具调用信息
            for part in event.content:
                if part.get("type") == "tool_call":
                    tool_call_id = part.get("id", "")
                    tool_name = part.get("name", "")
                    tool_arguments = part.get("arguments", {})

                    if tool_call_id and tool_name:
                        self._active_tool_calls[tool_call_id] = {
                            "tool_name": tool_name,
                            "arguments": tool_arguments,
                            "status": "running",
                            "start_time": time.time(),
                        }

        if block_type == "thinking" and self.show_reasoning:
            # 从 content 中提取 reasoning 文本
            reasoning_content = ""
            for part in event.content:
                if part.get("type") == "reasoning":
                    reasoning_content = part.get("reasoning") or ""
                    break
            self._print_thinking_panel(self._reasoning_buffer or reasoning_content)
            self._reasoning_buffer = ""

        tokens = self._parser.parse(self._buffer)

        uncommitted_tokens = tokens[self._committed_tokens_len:]
        if uncommitted_tokens:
            self._print_tokens(uncommitted_tokens)

        self._buffer = ""
        self._committed_tokens_len = 0
        self._format_buffer = ""
        self._in_format_detection = False
        self._current_block_type = None

    async def _on_tool_use_block_start(self, event: Event) -> None:
        """工具调用块开始"""
        assert isinstance(event, ModelToolCallBlockStartEvent)
        self._current_block_type = "tool_use"

    async def _on_tool_use_block_delta(self, event: Event) -> None:
        """工具调用块增量"""
        assert isinstance(event, ModelToolCallBlockDeltaEvent)
        # 工具调用参数增量不直接显示，在 stop 时显示完整信息

    async def _on_tool_use_block_stop(self, event: Event) -> None:
        """工具调用块结束"""
        assert isinstance(event, ModelToolCallBlockStopEvent)

        # 工具调用信息由 agent.tool_call 事件处理
        # 这里只清理状态
        self._current_block_type = None

    def _update_display(self):
        """更新显示 - 只渲染已确认的、格式完整的内容"""
        if not self._buffer:
            return

        tokens = self._parser.parse(self._buffer)

        top_level_block_end_indices = []
        for i, token in enumerate(tokens):
            if token.level == 0 and token.block:
                if token.type.endswith('_close') or token.type in ('fence', 'hr', 'html_block', 'code'):
                    top_level_block_end_indices.append(i)

        new_committed_len = 0
        if len(top_level_block_end_indices) > 1:
            new_committed_len = top_level_block_end_indices[-2] + 1

        if new_committed_len > self._committed_tokens_len:
            if self._live:
                self._live.stop()
                self._live = None

            tokens_to_print = tokens[self._committed_tokens_len:new_committed_len]
            if tokens_to_print:
                self._print_tokens(tokens_to_print)

            self._committed_tokens_len = new_committed_len

        active_tokens = tokens[self._committed_tokens_len:]

        if active_tokens:
            first_token = active_tokens[0]
            buffered_types = {'table_open', 'fence', 'code_block', 'html_block'}

            should_live_stream = first_token.type not in buffered_types

            if should_live_stream:
                md = TokenMarkdown(active_tokens, code_theme=self._code_theme)

                if not self._live:
                    self._live = Live(md, console=self._console, auto_refresh=True, vertical_overflow="visible", transient=True)
                    self._live.start()
                else:
                    self._live.update(md)
            else:
                if self._live:
                    self._live.stop()
                    self._live = None

    def _print_tokens(self, tokens: list[Any]):
        buffer = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.type == 'table_open' and token.level == 0:
                if buffer:
                    self._console.print(TokenMarkdown(buffer, code_theme=self._code_theme))
                    buffer = []

                table_tokens = []
                j = i
                nesting = 0
                found_close = False

                while j < len(tokens):
                    t = tokens[j]
                    table_tokens.append(t)
                    if t.type == 'table_open':
                        nesting += 1
                    elif t.type == 'table_close':
                        nesting -= 1
                        if nesting == 0:
                            found_close = True
                            break
                    j += 1

                if found_close:
                    self._print_table(table_tokens)
                    i = j + 1
                else:
                    buffer.append(token)
                    i += 1
            else:
                buffer.append(token)
                i += 1

        if buffer:
            self._console.print(TokenMarkdown(buffer, code_theme=self._code_theme))

    def _print_table(self, tokens: list[Any]):
        rows_count = 0
        cols_count = 0
        first_row_cols = 0

        for t in tokens:
            if t.type == 'tr_open':
                rows_count += 1
            if t.type in ('th_open', 'td_open') and rows_count == 1:
                first_row_cols += 1
        cols_count = first_row_cols

        if self._ellipsize_tables:
            from rich.padding import Padding
            from rich.text import Text
            summary = f"📊 Table ({rows_count} rows x {cols_count} columns)"
            self._console.print(Padding(Text(summary, style="italic dim"), (0, 0, 1, 2)))
        else:
            min_col_width = 15
            estimated_width = cols_count * min_col_width
            console_width = self._console.width

            table_width = None
            if estimated_width > console_width:
                table_width = estimated_width

            table = Table(box=box.ROUNDED, show_lines=False, width=table_width)

            current_row = []
            in_header = False

            idx = 0
            while idx < len(tokens):
                t = tokens[idx]
                if t.type == 'thead_open':
                    in_header = True
                elif t.type == 'thead_close':
                    in_header = False
                elif t.type == 'tr_open':
                    current_row = []
                elif t.type == 'tr_close':
                    if in_header:
                        for cell in current_row:
                            table.add_column(cell)
                    else:
                        table.add_row(*current_row)
                    current_row = []
                elif t.type in ('th_open', 'td_open'):
                    cell_content = []
                    cell_content.append(Token('paragraph_open', 'p', 1))

                    idx += 1
                    while idx < len(tokens):
                        sub_t = tokens[idx]
                        if sub_t.type in ('th_close', 'td_close'):
                            break
                        if sub_t.type == 'inline':
                            if sub_t.children:
                                cell_content.extend(sub_t.children)
                            else:
                                txt = Token('text', '', 0)
                                txt.content = sub_t.content
                                cell_content.append(txt)
                        else:
                            cell_content.append(sub_t)
                        idx += 1

                    cell_content.append(Token('paragraph_close', 'p', -1))

                    cell_renderable = TokenMarkdown(cell_content, code_theme=self._code_theme)
                    current_row.append(cell_renderable)
                    continue

                idx += 1

            self._console.print(table, soft_wrap=True)

    def _format_tool_arguments(self, arguments: dict[str, Any]) -> Text:
        """格式化工具参数为易读的 Markdown 格式。

        - 无换行符的参数: **arg**: value
        - 有换行符的参数: **arg**:\nvalue
        """
        if not arguments:
            return Text("", style="dim")

        lines: list[str] = []
        for key, value in arguments.items():
            value_str = str(value)
            if '\n' in value_str:
                # 有换行符的参数，冒号后换行
                lines.append(f"**{key}**:")
                lines.append(value_str)
            else:
                # 无换行符的参数，单行显示
                lines.append(f"**{key}**: {value_str}")

        # 限制长度
        full_text = "\n".join(lines)
        if not self.show_full_tool_content and len(full_text) > self.max_arg_length:
            full_text = full_text[:self.max_arg_length - 3] + "..."

        # 返回支持 markdown 样式的 Text
        return Text.from_markup(full_text, style="dim")

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

    async def _print_tool_result(
        self,
        tool_name: str,
        success: bool,
        result_preview: Any,
        duration: float,
        arguments: dict[str, Any] | None = None
    ) -> None:
        """打印工具结果面板"""
        status_emoji = "✅" if success else "❌"
        status_color = "green" if success else "red"
        status_text = "成功" if success else "失败"

        table = Table(show_header=False, box=None, expand=True, padding=(0, 1))
        table.add_column("label", width=10, style="dim cyan")
        table.add_column("content", ratio=1)

        table.add_row("工具", Text(tool_name, style="bold cyan"))
        if arguments:
            args_text = self._format_tool_arguments(arguments)
            table.add_row("参数", args_text)

        table.add_row("", "")
        table.add_row("结果", f"{status_emoji} {status_text}", style=f"bold {status_color}")

        if result_preview:
            preview = str(result_preview)
            if not self.show_full_tool_content and len(preview) > self.max_result_length:
                preview = preview[: self.max_result_length - 3] + "..."
            table.add_row("", Text(preview, style="white"))

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

    async def _print_error(self, error: str) -> None:
        """打印错误"""
        panel = Panel(
            Text(error, style="red"),
            title="[bold red]❌ Error[/bold red]",
            border_style="red",
            padding=(0, 1),
        )
        self._console.print(panel)
