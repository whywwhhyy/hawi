"""
Hawi Event System V2

统一事件系统：
- Event: 只读、非阻塞，由 Model 和 Agent 产生
- Hook: 阻塞、可修改，仅由 Agent 产生

命名规范：
- Model*Event: 由 Model 产生的事件
- Agent*Event: 由 Agent 产生的事件
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any, Literal

import sys

from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.rule import Rule
from rich.status import Status

from hawi.agent.messages import TokenUsage
from hawi.tool.types import ToolResult

# 创建 rich console 实例用于美化输出
_console = Console()

# 标准输出用于逐字打印
_stdout = sys.stdout

logger = logging.getLogger(__name__)

# =============================================================================
# Event 基类（只读、不可变）
# =============================================================================


@dataclass(frozen=True, slots=True)
class Event:
    """
    统一事件基类。只读、不可变。

    特性：
    - frozen=True + slots=True: 内存高效且防止修改
    - 异步处理: 消费者不能阻塞主流程
    - 多播: 可被多个消费者同时监听
    """
    type: str
    source: Literal["model", "agent"]
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)


# =============================================================================
# Model 事件（由 Model 产生）
# =============================================================================


def model_stream_start_event(request_id: str, **metadata: Any) -> Event:
    """Model 开始流式响应"""
    return Event(
        type="model.stream_start",
        source="model",
        metadata={"request_id": request_id, **metadata},
    )


def model_stream_stop_event(
    request_id: str, stop_reason: str, usage: TokenUsage | None = None, **metadata: Any
) -> Event:
    """Model 流式响应结束"""
    return Event(
        type="model.stream_stop",
        source="model",
        metadata={
            "request_id": request_id,
            "stop_reason": stop_reason,
            "usage": usage,
            **metadata,
        },
    )


def model_content_block_start_event(
    request_id: str,
    block_index: int,
    block_type: Literal["text", "thinking", "tool_use", "redacted_thinking"],
    **metadata: Any,
) -> Event:
    """内容块开始（统一 Anthropic 和 OpenAI 的所有内容类型）

    block_type 说明:
    - text: 普通文本（Anthropic text, OpenAI content）
    - thinking: 推理内容（Anthropic thinking, OpenAI reasoning_content）
    - tool_use: 工具调用（Anthropic tool_use, OpenAI tool_calls）
    - redacted_thinking: 被编辑的推理（Anthropic 特有）
    """
    return Event(
        type="model.content_block_start",
        source="model",
        metadata={
            "request_id": request_id,
            "block_index": block_index,
            "block_type": block_type,
            **metadata,
        },
    )


def model_content_block_delta_event(
    request_id: str,
    block_index: int,
    delta_type: Literal["text", "thinking", "tool_input", "signature"],
    delta: str,
    **metadata: Any,
) -> Event:
    """内容块增量更新

    delta_type 说明:
    - text: 文本增量
    - thinking: 推理增量
    - tool_input: 工具参数 JSON 片段
    - signature: thinking 签名（Anthropic 特有）
    """
    return Event(
        type="model.content_block_delta",
        source="model",
        metadata={
            "request_id": request_id,
            "block_index": block_index,
            "delta_type": delta_type,
            "delta": delta,
            **metadata,
        },
    )


def model_content_block_stop_event(
    request_id: str,
    block_index: int,
    block_type: Literal["text", "thinking", "tool_use", "redacted_thinking"] | None = None,
    full_content: str | None = None,
    **metadata: Any,
) -> Event:
    """内容块结束

    对于 tool_use 类型，metadata 中应包含:
    - tool_call_id: 工具调用 ID
    - tool_name: 工具名称
    - tool_arguments: 解析后的参数 dict
    """
    return Event(
        type="model.content_block_stop",
        source="model",
        metadata={
            "request_id": request_id,
            "block_index": block_index,
            "block_type": block_type,
            "full_content": full_content,
            **metadata,
        },
    )


def model_metadata_event(
    request_id: str,
    usage: TokenUsage | None = None,
    latency_ms: float | None = None,
    **metadata: Any,
) -> Event:
    """Model 元数据（usage 等）"""
    return Event(
        type="model.metadata",
        source="model",
        metadata={
            "request_id": request_id,
            "usage": usage,
            "latency_ms": latency_ms,
            **metadata,
        },
    )


# =============================================================================
# Agent 事件（由 Agent 产生）
# =============================================================================


def agent_run_start_event(
    run_id: str, message_preview: str | None = None, **metadata: Any
) -> Event:
    """Agent 开始执行"""
    return Event(
        type="agent.run_start",
        source="agent",
        metadata={
            "run_id": run_id,
            "message_preview": message_preview,
            **metadata,
        },
    )


def agent_run_stop_event(
    run_id: str, stop_reason: str, duration_ms: float, usage: TokenUsage | None = None, **metadata: Any
) -> Event:
    """Agent 执行结束"""
    return Event(
        type="agent.run_stop",
        source="agent",
        metadata={
            "run_id": run_id,
            "stop_reason": stop_reason,
            "duration_ms": duration_ms,
            "usage": usage,
            **metadata,
        },
    )


def agent_tool_call_event(
    run_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    tool_call_id: str,
    **metadata: Any,
) -> Event:
    """Agent 发起工具调用"""
    return Event(
        type="agent.tool_call",
        source="agent",
        metadata={
            "run_id": run_id,
            "tool_name": tool_name,
            "arguments": arguments,
            "tool_call_id": tool_call_id,
            **metadata,
        },
    )


def agent_tool_result_event(
    run_id: str,
    tool_name: str,
    tool_call_id: str,
    success: bool,
    result_preview: str,
    duration_ms: float,
    arguments: dict[str, Any] | None = None,
    **metadata: Any,
) -> Event:
    """Agent 收到工具结果"""
    return Event(
        type="agent.tool_result",
        source="agent",
        metadata={
            "run_id": run_id,
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "success": success,
            "result_preview": result_preview,
            "duration_ms": duration_ms,
            "arguments": arguments or {},
            **metadata,
        },
    )


def agent_message_added_event(
    run_id: str,
    role: Literal["user", "assistant", "tool"],
    message_preview: str,
    **metadata: Any,
) -> Event:
    """消息被添加到上下文"""
    return Event(
        type="agent.message_added",
        source="agent",
        metadata={
            "run_id": run_id,
            "role": role,
            "message_preview": message_preview,
            **metadata,
        },
    )


def agent_error_event(
    run_id: str,
    error_type: str,
    error_message: str,
    recoverable: bool = False,
    **metadata: Any,
) -> Event:
    """Agent 执行错误"""
    return Event(
        type="agent.error",
        source="agent",
        metadata={
            "run_id": run_id,
            "error_type": error_type,
            "error_message": error_message,
            "recoverable": recoverable,
            **metadata,
        },
    )


# =============================================================================
# Event Bus（事件总线）
# =============================================================================


EventHandler = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """
    事件总线：负责 Event 的多播分发。

    特性：
    - 异步广播：消费者不能阻塞生产者
    - 背压处理：慢消费者可选择丢弃或缓冲
    - 类型过滤：消费者可按事件类型订阅
    """

    def __init__(self):
        self._subscribers: dict[str, list[EventHandler]] = {}
        self._wildcards: list[EventHandler] = []
        self._closed = False

    def subscribe(
        self,
        callback: EventHandler,
        event_types: list[str] | None = None,
    ) -> None:
        """
        订阅事件。

        Args:
            callback: 异步回调函数，不能阻塞
            event_types: 订阅的事件类型列表，None 表示订阅所有
        """
        if self._closed:
            raise RuntimeError("EventBus is closed")

        if event_types is None:
            self._wildcards.append(callback)
        else:
            for et in event_types:
                self._subscribers.setdefault(et, []).append(callback)

    def unsubscribe(
        self,
        callback: EventHandler,
        event_types: list[str] | None = None,
    ) -> bool:
        """
        取消订阅。

        Returns:
            是否成功移除
        """
        removed = False
        if event_types is None:
            if callback in self._wildcards:
                self._wildcards.remove(callback)
                removed = True
        else:
            for et in event_types:
                if et in self._subscribers and callback in self._subscribers[et]:
                    self._subscribers[et].remove(callback)
                    removed = True
        return removed

    async def publish(self, event: Event) -> None:
        """
        发布事件。异步广播，不等待消费者完成。

        使用 asyncio.create_task 确保非阻塞。
        """
        if self._closed:
            return

        callbacks = self._wildcards.copy()
        callbacks.extend(self._subscribers.get(event.type, []))

        # Fire and forget - 不阻塞主流程
        for callback in callbacks:
            asyncio.create_task(self._invoke_safe(callback, event))

    async def _invoke_safe(self, callback: EventHandler, event: Event) -> None:
        """安全调用，捕获异常"""
        try:
            await callback(event)
        except Exception as e:
            logger.warning(f"Event handler error for {event.type}: {e}")

    def close(self) -> None:
        """关闭事件总线，清理订阅"""
        self._closed = True
        self._subscribers.clear()
        self._wildcards.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# =============================================================================
# PlainPrinter - 朴素打印机
# =============================================================================


class PlainPrinter:
    """
    朴素打印机，完全不依赖 rich 库。

    这是最简单、最底层的实现，适合：
    - 不支持 ANSI 的终端
    - 日志文件输出
    - 最小依赖场景

    特性：
    - 逐字符实时输出
    - 纯文本格式，无颜色、无方框
    - 零 rich 依赖

    使用示例：
        printer = PlainPrinter()
        async for event in agent.arun("prompt", stream=True):
            await printer.handle(event)
    """

    def __init__(
        self,
        *,
        show_reasoning: bool = True,
        show_tools: bool = True,
        show_errors: bool = True,
        max_arg_length: int = 80,
        max_result_length: int = 200,
    ):
        self.show_reasoning = show_reasoning
        self.show_tools = show_tools
        self.show_errors = show_errors
        self.max_arg_length = max_arg_length
        self.max_result_length = max_result_length

        # 内部状态
        self._current_block_type: str | None = None
        self._reasoning_buffer: str = ""
        self._active_tool_calls: dict[str, dict[str, Any]] = {}
        self._block_wait_spinner: asyncio.Task | None = None
        self._block_has_received_delta: bool = False
        self._spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._spinner_index: int = 0

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
        self._current_block_type = None

    async def _run_spinner(self) -> None:
        """运行等待动画"""
        while True:
            char = self._spinner_chars[self._spinner_index % len(self._spinner_chars)]
            self._spinner_index += 1
            _stdout.write(f"\r{char} 等待响应...")
            _stdout.flush()
            await asyncio.sleep(0.08)

    def _stop_spinner(self) -> None:
        """停止等待动画"""
        if self._block_wait_spinner is not None:
            self._block_wait_spinner.cancel()
            self._block_wait_spinner = None
            # 清除等待动画行
            _stdout.write("\r" + " " * 20 + "\r")
            _stdout.flush()

    async def _on_content_block_start(self, event: Event) -> None:
        """内容块开始"""
        meta = event.metadata
        block_type = meta.get("block_type")
        self._current_block_type = block_type
        self._block_has_received_delta = False

        # 对 text 和 thinking 类型的 block 显示等待动画
        if block_type in ("text", "thinking"):
            self._block_wait_spinner = asyncio.create_task(self._run_spinner())

    async def _on_content_block_delta(self, event: Event) -> None:
        """逐字符实时输出"""
        meta = event.metadata
        delta_type = meta.get("delta_type")
        delta = meta.get("delta", "")

        # 第一个 delta 到来时停止等待动画
        if not self._block_has_received_delta:
            self._block_has_received_delta = True
            self._stop_spinner()

        if not delta:
            return

        if delta_type == "text":
            _stdout.write(delta)
            _stdout.flush()
        elif delta_type == "thinking" and self.show_reasoning:
            self._reasoning_buffer += delta

    async def _on_content_block_stop(self, event: Event) -> None:
        """内容块结束"""
        # 确保等待动画已停止
        if not self._block_has_received_delta:
            self._stop_spinner()

        meta = event.metadata
        block_type = meta.get("block_type")

        if block_type == "thinking" and self.show_reasoning:
            if self._reasoning_buffer.strip():
                _stdout.write(f"\n[Thinking]\n{self._reasoning_buffer.strip()}\n[/Thinking]\n")
                _stdout.flush()
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
        self._current_block_type = None

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

        _stdout.write(f"\n[Tool Call: {tool_name}]\n")
        _stdout.flush()

        tool_call_id = meta.get("tool_call_id") or tool_name
        self._active_tool_calls[tool_call_id] = {
            "tool_name": tool_name,
            "arguments": meta.get("arguments", {}),
            "status": "running",
            "start_time": time.time(),
        }

    async def _on_tool_result(self, event: Event) -> None:
        """工具结果"""
        if not self.show_tools:
            return

        meta = event.metadata
        tool_name = meta.get("tool_name", "unknown")
        success = meta.get("success", False)
        result_preview = meta.get("result_preview", "")

        # 计算耗时
        start_time = None
        for tid, info in list(self._active_tool_calls.items()):
            if info.get("tool_name") == tool_name:
                start_time = info.get("start_time")
                del self._active_tool_calls[tid]
                break

        duration = (time.time() - start_time) * 1000 if start_time else 0

        status = "OK" if success else "FAILED"
        _stdout.write(f"[Tool Result: {tool_name}] {status} ({duration:.0f}ms)\n")

        if result_preview:
            preview = str(result_preview)
            if len(preview) > self.max_result_length:
                preview = preview[: self.max_result_length - 3] + "..."
            _stdout.write(f"  {preview}\n")
        _stdout.flush()

    async def _on_error(self, event: Event) -> None:
        """错误处理"""
        if not self.show_errors:
            return

        meta = event.metadata
        error = meta.get("error", "Unknown error")
        _stdout.write(f"\n[Error] {error}\n")
        _stdout.flush()


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


def create_event_printer(
    *,
    show_reasoning: bool = True,
    show_tools: bool = True,
    show_errors: bool = True,
) -> EventHandler:
    """
    创建默认的事件打印处理器。

    使用 RichStreamingPrinter 作为默认实现。

    Args:
        show_reasoning: 是否显示 reasoning 内容
        show_tools: 是否显示工具调用信息
        show_errors: 是否显示错误信息

    Returns:
        异步事件处理函数
    """
    printer = RichStreamingPrinter(
        show_reasoning=show_reasoning,
        show_tools=show_tools,
        show_errors=show_errors,
    )
    return printer.handle


# 类型别名（向后兼容）
ModelStreamStartEvent = Event
ModelStreamStopEvent = Event
ModelContentBlockStartEvent = Event
ModelContentBlockDeltaEvent = Event
ModelContentBlockStopEvent = Event
ModelMetadataEvent = Event
AgentRunStartEvent = Event
AgentRunStopEvent = Event
AgentToolCallEvent = Event
AgentToolResultEvent = Event
AgentMessageAddedEvent = Event
AgentErrorEvent = Event

# 类名别名（向后兼容）
ConversationPrinter = RichStreamingPrinter
