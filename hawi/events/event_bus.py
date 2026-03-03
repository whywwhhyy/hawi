from __future__ import annotations

import asyncio
import inspect
import logging
import queue
import threading
import time
from collections.abc import Callable
from typing import Any, Awaitable

from .event import Event

logger = logging.getLogger(__name__)


# =============================================================================
# Type Definitions
# =============================================================================

SyncEventHandler = Callable[[Event], None]
AsyncEventHandler = Callable[[Event], Awaitable[None]]
EventHandler = SyncEventHandler | AsyncEventHandler


# =============================================================================
# Event Bus (Refactored)
# =============================================================================

class EventBus:
    """
    事件总线：负责 Event 的多播分发。

    特性：
    - 单 worker 线程串行执行所有 non-blocking subscriber
    - blocking subscriber 在 publish 调用线程同步执行
    - subscribe 为同步方法，支持注册 sync/async subscriber
    - 支持阻塞和非阻塞两种订阅模式（通过 blocking 参数区分）
    - publish 提供同步和异步版本
    - 类型过滤：可按事件类型订阅
    - 优雅关闭：支持等待任务处理完毕
    """

    def __init__(self):
        # 任务队列，用于主线程与 worker 线程通信
        # 任务格式: event 或 None(停止信号)
        self._task_queue: queue.Queue = queue.Queue()

        # Worker 线程
        self._worker_thread: threading.Thread | None = None
        self._running = False
        self._closed = False

        # Subscriber 注册表: (handler, event_types, blocking)
        self._subscribers: list[tuple[EventHandler, list[str] | None, bool]] = []
        self._subscribers_lock = threading.RLock()

        # Worker 线程的事件循环
        self._loop: asyncio.AbstractEventLoop | None = None

    def _ensure_started(self) -> None:
        """确保 worker 线程已启动。"""
        if self._running:
            return

        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="EventBus-worker",
            daemon=True,
        )
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        """Worker 线程主循环，串行处理所有任务。"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._process_tasks())
        except Exception as e:
            logger.error(f"EventBus worker error: {e}")
        finally:
            self._loop.close()
            self._loop = None

    async def _process_tasks(self) -> None:
        """异步任务处理循环。"""
        while self._running:
            try:
                # 使用 run_in_executor 从同步队列获取任务
                task = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, self._get_next_task
                    ),
                    timeout=0.1
                )

                if not self._running:
                    break
                if task is None:
                    # 收到停止信号
                    break

                # 执行任务（串行执行 non-blocking subscribers）
                await self._execute_task(task)

            except asyncio.TimeoutError:
                continue
            except RuntimeError:
                if not self._running:
                    break
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"EventBus task error: {e}")

    def _get_next_task(self) -> Any:
        """从任务队列获取任务（同步方法，在 executor 中运行）。"""
        try:
            return self._task_queue.get(timeout=0.1)
        except queue.Empty:
            return None

    async def _execute_task(self, event: Event) -> None:
        """执行任务：分发事件给所有 non-blocking subscriber。

        Args:
            event: 要处理的事件
        """
        # 复制 subscriber 列表（在锁保护下）以避免遍历时被修改
        with self._subscribers_lock:
            subscribers = self._subscribers.copy()

        # 串行执行所有 non-blocking subscriber
        for handler, event_types, blocking in subscribers:
            # 跳过 blocking subscriber（已在 publish 线程中执行）
            if blocking:
                continue

            # 类型过滤
            if event_types is not None and event.type not in event_types:
                continue

            try:
                # 执行 handler
                result = handler(event)
                if result is not None and asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning(f"Event handler error for {event.type}: {e}")

    def _execute_blocking_subscribers(self, event: Event) -> None:
        """在调用线程中同步执行 blocking subscribers。

        Args:
            event: 要处理的事件
        """
        with self._subscribers_lock:
            subscribers = self._subscribers.copy()

        for handler, event_types, blocking in subscribers:
            # 只执行 blocking subscriber
            if not blocking:
                continue

            # 类型过滤
            if event_types is not None and event.type not in event_types:
                continue

            try:
                handler(event)
            except Exception as e:
                logger.warning(f"Blocking handler error for {event.type}: {e}")

    def subscribe(
        self,
        callback: EventHandler,
        event_types: list[str] | None = None,
        _maxsize: int = 100,  # 兼容旧API，但不使用
    ) -> None:
        """订阅事件（非阻塞模式）。

        Args:
            callback: 事件处理函数，可以是 sync 或 async
            event_types: 订阅的事件类型列表，None 表示订阅所有
            maxsize: 兼容旧API参数，不使用

        Raises:
            RuntimeError: 如果 EventBus 已关闭
        """
        if self._closed:
            raise RuntimeError("EventBus is closed")

        self._ensure_started()
        with self._subscribers_lock:
            self._subscribers.append((callback, event_types, False))

    def subscribe_blocking(
        self,
        callback: SyncEventHandler,
        event_types: list[str] | None = None,
    ) -> None:
        """订阅事件（阻塞模式，仅支持同步 handler）。

        事件在发布线程同步执行，handler 执行完成后 publish 才返回。
        仅支持同步 handler。

        Args:
            callback: 同步事件处理函数
            event_types: 订阅的事件类型列表，None 表示订阅所有

        Raises:
            RuntimeError: 如果 EventBus 已关闭
            ValueError: 如果 callback 是 async 函数
        """
        if self._closed:
            raise RuntimeError("EventBus is closed")

        if inspect.iscoroutinefunction(callback):
            raise ValueError(
                "subscribe_blocking only supports synchronous handlers. "
                "Use subscribe() for async handlers."
            )

        self._ensure_started()
        with self._subscribers_lock:
            self._subscribers.append((callback, event_types, True))

    def unsubscribe(
        self,
        callback: EventHandler,
    ) -> bool:
        """取消订阅。

        Args:
            callback: 要取消的回调函数

        Returns:
            是否成功取消订阅
        """
        if self._closed:
            return False

        with self._subscribers_lock:
            for i, (handler, _, _) in enumerate(self._subscribers):
                if handler is callback:
                    self._subscribers.pop(i)
                    return True
            return False

    def publish(self, event: Event) -> None:
        """发布事件（同步版本）。

        执行顺序：
        1. 在调用线程同步执行 blocking subscribers
        2. 将事件放入队列，由 worker 线程异步执行 non-blocking subscribers

        Args:
            event: 要发布的事件

        Raises:
            RuntimeError: 如果 EventBus 已关闭
        """
        if self._closed:
            raise RuntimeError("EventBus is closed")

        self._ensure_started()

        # 1. 在调用线程同步执行 blocking subscribers
        self._execute_blocking_subscribers(event)

        # 2. 将事件放入队列，由 worker 线程异步执行 non-blocking subscribers
        self._task_queue.put(event)

    async def publish_async(self, event: Event) -> None:
        """发布事件（异步版本）。

        执行顺序：
        1. 在调用线程同步执行 blocking subscribers
        2. 等待 worker 线程完成 non-blocking subscribers 的执行

        Args:
            event: 要发布的事件

        Raises:
            RuntimeError: 如果 EventBus 已关闭
        """
        if self._closed:
            raise RuntimeError("EventBus is closed")

        self._ensure_started()

        # 1. 在调用线程同步执行 blocking subscribers
        self._execute_blocking_subscribers(event)

        # 2. 等待 worker 线程完成 non-blocking subscribers
        done_event = threading.Event()

        # 包装任务以设置完成信号
        async def wrapped_task():
            try:
                await self._execute_task(event)
            finally:
                done_event.set()

        # 提交到 worker 线程
        if self._loop is not None:
            asyncio.run_coroutine_threadsafe(wrapped_task(), self._loop)

            # 等待完成
            while not done_event.is_set():
                await asyncio.sleep(0.001)
        else:
            # Worker 未启动，直接执行
            await self._execute_task(event)

    def close(self, wait: bool = True, timeout: float | None = None) -> None:
        """关闭事件总线，优雅停止 worker 线程。

        Args:
            wait: 是否等待队列中的任务处理完毕
            timeout: 等待超时时间（秒），None 表示无限等待
        """
        if self._closed:
            return

        # 等待队列中的任务处理完毕
        if wait:
            self.flush(timeout=timeout)

        self._closed = True
        self._running = False

        # 发送停止信号
        try:
            self._task_queue.put_nowait(None)
        except queue.Full:
            pass

        # 等待 worker 线程结束
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout if timeout else 5.0)

        # 清理
        with self._subscribers_lock:
            self._subscribers.clear()

    def flush(self, timeout: float | None = None) -> bool:
        """等待所有待处理的事件被处理完毕。

        Args:
            timeout: 超时时间（秒），None 表示无限等待

        Returns:
            True 如果所有事件都已处理，False 如果超时
        """
        if self._closed:
            return True

        timeout_val = timeout if timeout else 30.0
        start_time = time.time()

        # 等待任务队列变空
        while not self._task_queue.empty():
            if time.time() - start_time > timeout_val:
                return False
            time.sleep(0.01)

        return True

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()
