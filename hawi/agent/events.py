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

from hawi.agent.message import TokenUsage

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
    Type = Literal[
        "model.stream_start",
        "model.stream_stop",
        "model.content_block_start",
        "model.content_block_delta",
        "model.content_block_stop",
        "model.metadata",
        "agent.run_start",
        "agent.run_stop",
        "agent.tool_call",
        "agent.tool_result",
        "agent.message_added",
        "agent.error",
    ]

    type: Event.Type
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

