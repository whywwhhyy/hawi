"""
Hawi Printer Implementations

提供多种事件打印机实现：
- RichStreamingPrinter: 原始 ANSI 颜色流式打印
- MarkdownStreamingPrinter: Markdown 实时渲染打印机
"""

from __future__ import annotations

import logging
import time
from typing import Any

import sys

from rich.console import Console
from rich import box
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.live import Live

from markdown_it import MarkdownIt
from mdit_py_plugins.tasklists import tasklists_plugin
from mdit_py_plugins.container import container_plugin
from mdit_py_plugins.admon import admon_plugin

from hawi.agent.events import Event, EventHandler

logger = logging.getLogger(__name__)

class ConfigurableMarkdown(Markdown):
    """
    扩展 Rich Markdown 以支持自定义 parser。
    这允许我们显式使用 markdown-it-py 及其插件。
    """
    def __init__(self, markup: str, parser: MarkdownIt, **kwargs):
        # 初始化父类，传入空字符串以跳过默认解析
        super().__init__("", **kwargs)
        self.markup = markup
        # 使用自定义 parser 解析
        self.parsed = parser.parse(markup)


from rich.table import Table

class TokenMarkdown(Markdown):
    """
    支持直接传入 Token 列表的 Markdown 渲染类。
    用于流式增量渲染：我们手动解析 buffer，计算出差异，然后构造此对象进行渲染。
    """
    def __init__(self, tokens: list[Any], **kwargs):
        super().__init__("", **kwargs)
        self.parsed = tokens


from markdown_it.token import Token

class StreamMarkdownPrinter:
    """
    StreamMarkdownPrinter - 支持流式 Markdown 输出的打印机
    
    特性：
    1. 支持流式输入：逐字处理输入流。
    2. 普通文本立即输出：通过 Live Display 实现即时反馈。
    3. 块级结构智能渲染：利用 markdown-it 的解析能力，识别块的完整性。
       - 对于未完成的块（如正在输入的代码块），Live Display 会显示当前状态（可能是文本）。
       - 当块结构完成（如闭合代码块）或类型确定时，自动更新为正确的渲染样式。
    4. 显式集成 markdown-it-py：使用 Token 流进行增量渲染。
    5. 表格处理：支持省略表格（ellipsize_tables=True）或手动渲染宽表格。
    6. 工具调用显示：支持显示工具调用和结果。
    7. Thinking 块显示：支持显示模型思考过程。
    """
    
    def __init__(
        self, 
        console: Console | None = None,
        code_theme: str = "monokai",
        ellipsize_tables: bool = False,
        show_tools: bool = True,
        show_reasoning: bool = True,
        max_arg_length: int = 80,
        max_result_length: int = 200,
    ):
        self._console = console or Console()
        self._code_theme = code_theme
        self._ellipsize_tables = ellipsize_tables
        self._show_tools = show_tools
        self._show_reasoning = show_reasoning
        self._max_arg_length = max_arg_length
        self._max_result_length = max_result_length
        
        # 初始化自定义 markdown-it parser
        # 显式启用 GFM 表格支持和其他插件
        self._parser = (
            MarkdownIt("gfm-like")
            .enable("table")
            .enable("strikethrough")
            .use(tasklists_plugin)
            .use(container_plugin, name="warning")
            .use(admon_plugin)
        )
        
        # 缓冲区和状态
        self._buffer = ""
        self._committed_tokens_len = 0
        self._live: Live | None = None
        
        # 工具调用跟踪
        self._active_tool_calls: dict[str, dict[str, Any]] = {}
        self._current_block_type: str | None = None
        self._reasoning_buffer: str = ""
        
    async def handle(self, event: Event) -> None:
        handlers = {
            "model.content_block_start": self._on_content_block_start,
            "model.content_block_delta": self._on_delta,
            "model.content_block_stop": self._on_block_stop,
            "agent.tool_call": self._on_tool_call,
            "agent.tool_result": self._on_tool_result,
        }
        handler = handlers.get(event.type)
        if handler:
            await handler(event)

    async def _on_content_block_start(self, event: Event) -> None:
        """内容块开始"""
        block_type = event.metadata.get("block_type")
        self._current_block_type = block_type
            
    async def _on_delta(self, event: Event) -> None:
        delta = event.metadata.get("delta", "")
        delta_type = event.metadata.get("delta_type", "text")
        
        if delta_type == "text":
            self._buffer += delta
            self._update_display()
        elif delta_type == "thinking" and self._show_reasoning:
            self._reasoning_buffer += delta
            
    async def _on_block_stop(self, event: Event) -> None:
        # 块结束，强制刷新并停止 Live
        if self._live:
            self._live.stop()
            self._live = None
            
        meta = event.metadata
        block_type = self._current_block_type
        
        # 处理 tool_use 块
        if block_type == "tool_use" and self._show_tools:
            tool_call_id = meta.get("tool_call_id")
            tool_name = meta.get("tool_name")
            tool_arguments = meta.get("tool_arguments", {})
            
            if tool_call_id and tool_name:
                self._active_tool_calls[tool_call_id] = {
                    "tool_name": tool_name,
                    "arguments": tool_arguments,
                    "status": "running",
                    "start_time": time.time(),
                }
        
        # 处理 thinking 块
        if block_type == "thinking" and self._show_reasoning:
            full_content = meta.get("full_content", "")
            self._print_thinking_panel(self._reasoning_buffer or full_content)
            self._reasoning_buffer = ""
        
        # 打印最终完整内容（确保所有内容都被 committed）
        # 解析全部内容
        tokens = self._parser.parse(self._buffer)
        
        # 计算还未 commit 的部分
        uncommitted_tokens = tokens[self._committed_tokens_len:]
        if uncommitted_tokens:
            self._print_tokens(uncommitted_tokens)
            
        # 重置状态
        self._buffer = ""
        self._committed_tokens_len = 0
        self._current_block_type = None

    def _update_display(self):
        # 1. 解析当前缓冲区
        tokens = self._parser.parse(self._buffer)
        
        # 2. 识别 "Safe" (Committed) Tokens
        # 策略：找到最后一个 Top-Level Block 的开始位置。
        # 在此之前的所有 Block 都是安全的，可以 commit。
        # 最后一个 Block 是 "Active" 的，通过 Live 显示。
        
        top_level_block_end_indices = []
        for i, token in enumerate(tokens):
            if token.level == 0 and token.block:
                # 这是一个 top-level block token
                # 如果是 _close 或者是原子的 (fence, hr, html_block, code)
                if token.type.endswith('_close') or token.type in ('fence', 'hr', 'html_block', 'code'):
                    top_level_block_end_indices.append(i)
        
        # 决定 commit 多少
        new_committed_len = 0
        if len(top_level_block_end_indices) > 1:
            # 取倒数第二个 block 的结束位置作为 commit 点
            # tokens 索引是 inclusive 的，所以长度是 index + 1
            new_committed_len = top_level_block_end_indices[-2] + 1
        
        # 如果 new_committed_len > self._committed_tokens_len，说明有新的 block 完成了
        if new_committed_len > self._committed_tokens_len:
            # 1. 停止当前的 Live (它显示的是之前的 Active Block，现在已经完成了)
            if self._live:
                self._live.stop()
                self._live = None
            
            # 2. 打印新完成的 Block(s)
            tokens_to_print = tokens[self._committed_tokens_len:new_committed_len]
            if tokens_to_print:
                self._print_tokens(tokens_to_print)
            
            # 3. 更新 committed index
            self._committed_tokens_len = new_committed_len
            
        # 3. 处理剩下的 (Active) Tokens
        active_tokens = tokens[self._committed_tokens_len:]
        
        if active_tokens:
            # 根据 Active Block 类型决定是否使用 Live Display
            first_token = active_tokens[0]
            
            # 定义需要缓冲（不流式显示）的块类型
            buffered_types = {'table_open', 'fence', 'code_block', 'html_block'}
            
            should_live_stream = first_token.type not in buffered_types
            
            if should_live_stream:
                md = TokenMarkdown(active_tokens, code_theme=self._code_theme)
                
                if not self._live:
                    # transient=True 确保 Live 结束时清除显示，避免与 commit 的内容重复
                    self._live = Live(md, console=self._console, auto_refresh=True, vertical_overflow="visible", transient=True)
                    self._live.start()
                else:
                    self._live.update(md)
            else:
                # 如果是需要缓冲的类型，停止 Live (不显示中间状态)
                if self._live:
                    self._live.stop()
                    self._live = None

    def _print_tokens(self, tokens: list[Any]):
        """
        打印 Token 列表，处理表格的特殊渲染逻辑。
        如果遇到 Top-Level 表格：
          - ellipsize_tables=True: 打印摘要
          - ellipsize_tables=False: 手动构建 Rich Table 打印（支持更灵活的显示）
        其他内容：使用 TokenMarkdown 渲染
        """
        buffer = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            # 仅处理 Top-Level 表格
            if token.type == 'table_open' and token.level == 0:
                # 1. 先打印 buffer 中的内容
                if buffer:
                    self._console.print(TokenMarkdown(buffer, code_theme=self._code_theme))
                    buffer = []
                
                # 2. 提取表格 tokens
                table_tokens = []
                j = i
                nesting = 0
                found_close = False
                
                # 寻找匹配的 table_close
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
                    # 未找到结尾（理论上 committed tokens 应该是完整的），回退到 buffer
                    buffer.append(token)
                    i += 1
            else:
                buffer.append(token)
                i += 1
        
        # 打印剩余 buffer
        if buffer:
            self._console.print(TokenMarkdown(buffer, code_theme=self._code_theme))

    def _print_table(self, tokens: list[Any]):
        """
        手动渲染表格
        """
        # 预先分析行列信息
        rows_count = 0
        cols_count = 0
        first_row_cols = 0
        
        # 简单的扫描统计
        for t in tokens:
            if t.type == 'tr_open':
                rows_count += 1
            if t.type in ('th_open', 'td_open') and rows_count == 1:
                first_row_cols += 1
        cols_count = first_row_cols
        
        if self._ellipsize_tables:
            # 打印摘要
            summary = f"📊 Table ({rows_count} rows x {cols_count} columns)"
            # 使用 blockquote 样式
            from rich.padding import Padding
            from rich.text import Text
            self._console.print(Padding(Text(summary, style="italic dim"), (0, 0, 1, 2)))
        else:
            # 手动构建 Rich Table
            # 为了防止在窄终端下内容被过度挤压（导致省略），我们估算一个最小宽度
            # 假设每列至少需要 15 字符（包含 padding）
            min_col_width = 15
            estimated_width = cols_count * min_col_width
            # 获取当前 console 宽度（如果能获取到）
            console_width = self._console.width
            
            # 如果估算宽度超过 console 宽度，则强制设置表格宽度，配合 soft_wrap=True 实现水平滚动效果
            table_width = None
            if estimated_width > console_width:
                table_width = estimated_width
            
            table = Table(box=box.ROUNDED, show_lines=False, width=table_width)
            
            # 解析 tokens 构建表结构
            # 状态机：thead -> tr -> th; tbody -> tr -> td
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
                    # 添加行
                    if in_header:
                        # Rich Table 添加列定义
                        for cell in current_row:
                            # cell 是 TokenMarkdown 对象
                            # 我们可以直接把 renderable 传给 add_column? 
                            # 不，add_column 接受 header (str or Renderable)
                            table.add_column(cell)
                    else:
                        table.add_row(*current_row)
                    current_row = []
                elif t.type in ('th_open', 'td_open'):
                    # 提取单元格内容 tokens
                    # th_open -> inline -> th_close
                    # inline token 的 children 才是真正的内容
                    # 有时候可能有多个 token? markdown-it 表格单元格通常包含一个 inline token
                    
                    cell_content = []
                    # 必须包裹在 paragraph 中，因为 TokenMarkdown (rich.markdown) 需要块级元素作为容器
                    # 否则直接传入 inline tokens 会导致 Stack 为空 (Root 不接受 text) 或渲染错误
                    cell_content.append(Token('paragraph_open', 'p', 1))
                    
                    idx += 1
                    while idx < len(tokens):
                        sub_t = tokens[idx]
                        if sub_t.type in ('th_close', 'td_close'):
                            break
                        if sub_t.type == 'inline':
                            # 使用 children 渲染
                            if sub_t.children:
                                cell_content.extend(sub_t.children)
                            else:
                                # 如果没有 children 但有 content (纯文本)
                                # 构造一个 text token
                                txt = Token('text', '', 0)
                                txt.content = sub_t.content
                                cell_content.append(txt)
                        else:
                            # 其他块级元素？表格单元格内通常是 inline
                            cell_content.append(sub_t)
                        idx += 1
                    
                    cell_content.append(Token('paragraph_close', 'p', -1))
                    
                    # 渲染单元格
                    cell_renderable = TokenMarkdown(cell_content, code_theme=self._code_theme)
                    current_row.append(cell_renderable)
                    # 此时 idx 指向 close token，循环会自动处理
                    continue
                    
                idx += 1
            
            # 使用 soft_wrap=True 允许表格超出终端宽度，避免内容被过度挤压
            self._console.print(table, soft_wrap=True)

    async def _on_tool_call(self, event: Event) -> None:
        """工具调用"""
        if not self._show_tools:
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

    async def _on_tool_result(self, event: Event) -> None:
        """工具结果"""
        if not self._show_tools:
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
            if len(args_str) > self._max_arg_length:
                args_str = args_str[:self._max_arg_length - 3] + "..."
            table.add_row("参数", Text(args_str, style="dim"))

        # 分隔线
        table.add_row("", "")
        table.add_row("结果", f"{status_emoji} {status_text}", style=f"bold {status_color}")

        # 结果内容（下半部分）
        if result_preview:
            preview = str(result_preview)
            if len(preview) > self._max_result_length:
                preview = preview[: self._max_result_length - 3] + "..."
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
