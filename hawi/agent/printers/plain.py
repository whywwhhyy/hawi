"""
PlainPrinter - 纯文本事件打印机

用于非终端环境的简单文本输出。
"""

from __future__ import annotations

import logging
from typing import Any

import sys

from hawi.agent.events import (
    Event,
    ModelContentBlockStartEvent,
    ModelContentBlockDeltaEvent,
    ModelContentBlockStopEvent,
    ModelToolCallBlockStartEvent,
    ModelToolCallBlockDeltaEvent,
    ModelToolCallBlockStopEvent,
    AgentToolCallEvent,
)
from hawi.agent.printers.base import BasePrinter

logger = logging.getLogger(__name__)
_stdout = sys.stdout


# =============================================================================
# PlainPrinter - 朴素打印机
# =============================================================================


class PlainPrinter(BasePrinter):
    """
    朴素打印机，使用纯文本输出原文（带标签）（包括错误信息）。
    
    特性：
    - 支持 non-tty 输出
    - 支持流式输出，响应最快
    - 不需要撤销/修改已有内容
    - 包含 [Thinking], [Tool Call] 等标签
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
        show_full_tool_content: bool = False,
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
        self._has_printed_first_block = False

    async def _on_content_block_start(self, event: Event) -> None:
        """内容块开始"""
        assert isinstance(event, ModelContentBlockStartEvent)
        self._current_block_type = event.block_type
        timestamp = self._get_timestamp()

        # 块之间添加换行（除了第一个）
        if self._has_printed_first_block:
            _stdout.write("\n")
        self._has_printed_first_block = True

        if event.block_type == "reasoning" and self.show_reasoning:
            _stdout.write(f"[{timestamp}] [Thinking]\n")
            _stdout.flush()

    async def _on_content_block_delta(self, event: Event) -> None:
        """逐字符实时输出"""
        assert isinstance(event, ModelContentBlockDeltaEvent)
        delta = event.delta
        if not delta:
            return

        if event.delta_type == "text":
            _stdout.write(delta)
            _stdout.flush()
        elif event.delta_type == "reasoning" and self.show_reasoning:
            _stdout.write(delta)
            _stdout.flush()

    async def _on_content_block_stop(self, event: Event) -> None:
        """内容块结束"""
        assert isinstance(event, ModelContentBlockStopEvent)
        
        if event.block_type == "reasoning" and self.show_reasoning:
            _stdout.write("\n[/Thinking]\n")
            _stdout.flush()
        elif event.block_type == "text":
            # 文本块结束，确保换行（如果需要）
            _stdout.write("\n")
            _stdout.flush()
        
        self._current_block_type = None

    async def _on_tool_use_block_start(self, event: Event) -> None:
        """工具调用块开始 - 记录状态，不打印"""
        assert isinstance(event, ModelToolCallBlockStartEvent)
        self._current_block_type = "tool_use"

    async def _on_tool_use_block_delta(self, event: Event) -> None:
        """工具调用块增量 - 忽略，在 _on_tool_call 中统一打印"""
        pass

    async def _on_tool_use_block_stop(self, event: Event) -> None:
        """工具调用块结束 - 忽略，在 _on_tool_call 中统一打印"""
        pass

    async def _on_tool_call(self, event: Event) -> None:
        """工具调用 - 立即打印工具名、格式化参数和执行状态"""
        if not self.show_tools:
            return

        from hawi.agent.events import AgentToolCallEvent
        assert isinstance(event, AgentToolCallEvent)

        timestamp = self._get_timestamp()
        tool_name = event.tool_name
        arguments = event.arguments

        if self._has_printed_first_block:
            _stdout.write("\n")
        self._has_printed_first_block = True

        _stdout.write(f"[{timestamp}] [Tool Call: {tool_name}]\n")
        if arguments:
            for key, value in arguments.items():
                _stdout.write(f"  {key}: {value}\n")
        _stdout.write("  Status: Executing...")
        _stdout.flush()

        # 标记为等待第一个结果片段
        self._awaiting_first_part = True

    async def _on_tool_result_part(self, event: Event) -> None:
        """处理流式工具结果片段（异步生成器工具）"""
        if not self.show_tools:
            return

        from hawi.agent.events import AgentToolResultPartEvent
        assert isinstance(event, AgentToolResultPartEvent)

        part = event.part
        is_final = event.is_final

        # 第一次收到片段时，清除 Executing... 状态
        if getattr(self, '_awaiting_first_part', False):
            _stdout.write("\r  Status: Receiving...\n")
            _stdout.write("  Output: ")
            self._awaiting_first_part = False

        # 打印片段（不换行，连续输出）
        if part:
            _stdout.write(part)
            _stdout.flush()

        # 如果是最终片段，添加换行
        if is_final:
            _stdout.write("\n")
            _stdout.flush()

    def _print_tool_result(
        self,
        tool_name: str,
        success: bool,
        result_preview: Any,
        duration: float,
        arguments: dict[str, Any] | None = None
    ) -> None:
        """打印工具结果"""
        timestamp = self._get_timestamp()
        status = "OK" if success else "FAILED"

        # 在同一行更新状态
        _stdout.write(f"\r  Status: {status} ({duration:.0f}ms)\n")

        if result_preview is not None:
            preview = str(result_preview)
            if not self.show_full_tool_content and len(preview) > self.max_result_length:
                preview = preview[: self.max_result_length - 3] + "..."
            _stdout.write(f"  Result: {preview}\n")

        _stdout.write("[/Tool Call]\n")
        _stdout.flush()

    def _print_error(self, error: str) -> None:
        """打印错误"""
        timestamp = self._get_timestamp()
        _stdout.write(f"\n[{timestamp}] [Error] {error}\n")
        _stdout.flush()

    def _print_usage(self, usage) -> None:
        """打印 token 用量"""
        timestamp = self._get_timestamp()
        parts = [f"in={usage['input_tokens']}", f"out={usage['output_tokens']}"]
        cache_write = usage.get('cache_write_tokens')
        cache_read = usage.get('cache_read_tokens')
        if cache_write:
            parts.append(f"cache_write={cache_write}")
        if cache_read:
            parts.append(f"cache_read={cache_read}")
        _stdout.write(f"[{timestamp}] [Usage] {', '.join(parts)}\n")
        _stdout.flush()
