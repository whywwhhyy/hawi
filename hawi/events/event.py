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
from typing import Any, Literal

from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)


EventSource = Literal['model', 'agent']

ModelEventType = Literal[
    'model.stream_start',
    'model.content_block_start',
    'model.content_block_delta',
    'model.content_block_stop',
    'model.tool_use_block_start',
    'model.tool_use_block_delta',
    'model.tool_use_block_stop',
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
    - 队列模式：每个消费者独立队列，防止生产者被阻塞
    - 阻塞模式：关键消费者可阻塞生产者，确保事件被处理
    - 背压处理：队列满时丢弃事件并记录警告
    - 类型过滤：消费者可按事件类型订阅
    - 优雅关闭：支持等待队列消费完毕
    """

    def __init__(self):
        # 队列模式：每个消费者独立队列和任务
        self._queues: dict[int, asyncio.Queue] = {}
        self._consumer_tasks: dict[int, asyncio.Task] = {}
        self._consumer_filters: dict[int, list[str] | None] = {}

        # 阻塞模式：直接存储 handler
        self._blocking_handlers: list[tuple[EventHandler, list[str] | None]] = []

        self._closed = False

    def subscribe(
        self,
        callback: EventHandler,
        event_types: list[str] | None = None,
        blocking: bool = False,
        maxsize: int = 100,
    ) -> None:
        """
        订阅事件。

        Args:
            callback: 异步回调函数
            event_types: 订阅的事件类型列表，None 表示订阅所有
            blocking: 是否阻塞模式。阻塞模式下生产者等待消费者处理完成
            maxsize: 队列大小（仅非阻塞模式有效）

        Raises:
            RuntimeError: 如果 EventBus 已关闭
        """
        if self._closed:
            raise RuntimeError("EventBus is closed")

        if blocking:
            # 阻塞模式：直接存储 handler，不使用队列
            self._blocking_handlers.append((callback, event_types))
        else:
            # 非阻塞模式：创建独立队列和消费者任务
            queue = asyncio.Queue(maxsize=maxsize)
            handler_id = id(callback)
            self._queues[handler_id] = queue
            self._consumer_filters[handler_id] = event_types

            # 启动消费者任务
            task = asyncio.create_task(
                self._consume_queue(queue, callback, event_types),
                name=f"EventBus-consumer-{handler_id}"
            )
            self._consumer_tasks[handler_id] = task

    async def _consume_queue(
        self,
        queue: asyncio.Queue,
        callback: EventHandler,
        event_types: list[str] | None,
    ) -> None:
        """队列消费者循环。"""
        while True:
            event = await queue.get()
            if event is None:  # 结束信号
                break

            # 类型过滤
            if event_types is not None and event.type not in event_types:
                continue

            try:
                await callback(event)
            except Exception as e:
                logger.warning(f"Event handler error for {event.type}: {e}")

    async def unsubscribe(
        self,
        callback: EventHandler,
        wait: bool = False,
        timeout: float | None = None,
    ) -> bool:
        """
        取消订阅，停止对应的消费者任务。

        Args:
            callback: 要取消的回调函数
            wait: 是否等待队列中的事件消费完毕
            timeout: 等待超时时间（秒），None 表示无限等待

        Returns:
            是否成功取消订阅
        """
        handler_id = id(callback)

        # 处理阻塞订阅者
        for i, (handler, _) in enumerate(self._blocking_handlers):
            if handler is callback:
                self._blocking_handlers.pop(i)
                return True

        # 处理队列订阅者
        if handler_id not in self._queues:
            return False

        queue = self._queues[handler_id]
        task = self._consumer_tasks.get(handler_id)

        if wait:
            # 等待队列消费完毕或超时
            try:
                await asyncio.wait_for(
                    self._wait_queue_empty(queue),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"Unsubscribe timeout for handler {handler_id}")

        # 发送结束信号
        try:
            queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

        # 如果等待，确保任务完成
        if wait and task is not None and not task.done():
            try:
                await asyncio.wait_for(task, timeout=1.0)
            except asyncio.TimeoutError:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # 清理
        del self._queues[handler_id]
        del self._consumer_filters[handler_id]
        if handler_id in self._consumer_tasks:
            del self._consumer_tasks[handler_id]

        return True

    async def _wait_queue_empty(self, queue: asyncio.Queue) -> None:
        """等待队列变空。"""
        while not queue.empty():
            await asyncio.sleep(0.01)

    async def publish(self, event: Event) -> None:
        """
        发布事件。

        处理顺序：
        1. 先执行阻塞订阅者（确保关键操作如 DumpManager 完成）
        2. 再分发给非阻塞队列

        Args:
            event: 要发布的事件
        """
        if self._closed:
            return

        # 1. 阻塞订阅者（生产者等待）
        for handler, event_types in self._blocking_handlers:
            if event_types is not None and event.type not in event_types:
                continue
            try:
                await handler(event)
            except Exception as e:
                logger.warning(f"Blocking handler error for {event.type}: {e}")

        # 2. 非阻塞队列（fire and forget）
        for handler_id, queue in list(self._queues.items()):
            # 检查类型过滤
            event_types = self._consumer_filters.get(handler_id)
            if event_types is not None and event.type not in event_types:
                continue

            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # 队列满，消费者太慢，丢弃事件
                logger.warning(f"Queue full for handler {handler_id}, dropping event {event.type}")

    async def close(self, wait: bool = True, timeout: float | None = None) -> None:
        """
        关闭事件总线，优雅停止所有消费者。

        Args:
            wait: 是否等待消费者处理完队列中的事件
            timeout: 等待超时时间（秒），None 表示无限等待
        """
        if self._closed:
            return

        # 1. 发送结束信号到所有队列
        for queue in self._queues.values():
            try:
                queue.put_nowait(None)
            except asyncio.QueueFull:
                pass

        # 2. 等待消费者任务完成
        if wait and self._consumer_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._consumer_tasks.values(), return_exceptions=True),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"EventBus close timeout, cancelling {len(self._consumer_tasks)} tasks")
                for task in self._consumer_tasks.values():
                    task.cancel()

        # 3. 清理
        self._queues.clear()
        self._consumer_tasks.clear()
        self._consumer_filters.clear()
        self._blocking_handlers.clear()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 同步上下文管理器无法等待异步 close
        self._closed = True
        self._queues.clear()
        self._consumer_tasks.clear()
        self._consumer_filters.clear()
        self._blocking_handlers.clear()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
