"""
Event base class for Hawi Event System.

Events are:
- Read-only and immutable
- Produced by Model and Agent
- Non-blocking, multi-consumer
"""


from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)


EventSource = Literal['model', 'agent']

ModelEventType = Literal[
    'model.stream_start',
    'model.content_block_start',
    'model.content_block_delta',
    'model.content_block_stop',
    'model.metadata',
    'model.stream_stop',
    'model.error',
]

AgentEventType = Literal[
    'agent.run_start',
    'agent.run_stop',
    'agent.tool_call',
    'agent.tool_result',
    'agent.message_added',
    'agent.error',
]

EventType = ModelEventType | AgentEventType


class Event(BaseModel):
    """
    统一事件基类。只读、不可变。

    特性：
    - frozen=True: 不可变，防止意外修改
    - extra="forbid": 禁止额外字段，确保类型安全
    - arbitrary_types_allowed=True: 允许任意类型（如 AgentError/ModelError）
    - 抽象基类: 不能直接实例化，必须使用子类
    - 异步处理: 消费者不能阻塞主流程
    - 多播: 可被多个消费者同时监听
    """
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    type: EventType
    source: EventSource
    timestamp: float = Field(default_factory=time.time)

    def __init__(self, **data):
        # 防止直接实例化 Event 基类
        if self.__class__ is Event:
            raise TypeError(
                "Event is an abstract base class and cannot be instantiated directly. "
                "Use concrete subclasses like ModelStreamStartEvent, AgentToolCallEvent, etc."
            )
        super().__init__(**data)

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

