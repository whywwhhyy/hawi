from __future__ import annotations

import asyncio
import atexit
import logging
import queue
import threading
import time
from collections.abc import Callable
from typing import Any, Literal

from .event import Event

logger = logging.getLogger(__name__)



# =============================================================================
# Event Bus（事件总线）
# =============================================================================


EventHandler = Callable[[Event], None]


class EventBus:
    """
    事件总线：负责 Event 的多播分发。

    特性：
    - 内部 worker 线程处理所有异步操作
    - 对外暴露同步接口
    - 队列模式：每个消费者独立队列，防止生产者被阻塞
    - 阻塞模式：关键消费者可阻塞生产者，确保事件被处理
    - 背压处理：队列满时丢弃事件并记录警告
    - 类型过滤：消费者可按事件类型订阅
    - 优雅关闭：支持等待队列消费完毕
    """

    def __init__(self):
        # 内部队列用于 worker 线程与外部交互
        self._task_queue: queue.Queue = queue.Queue()
        self._worker_thread: threading.Thread | None = None
        self._running = False
        self._closed = False
        self._stopping = False  # 关闭中标志

        # 队列模式：每个消费者独立队列和任务
        self._queues: dict[int, asyncio.Queue] = {}
        self._consumer_tasks: dict[int, asyncio.Task] = {}
        self._consumer_filters: dict[int, list[str] | None] = {}

        # 阻塞模式：直接存储 handler
        self._blocking_handlers: list[tuple[EventHandler, list[str] | None]] = []

        # 事件循环，用于在 worker 线程中运行 asyncio
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
        """Worker 线程主循环，处理任务队列。"""
        # 创建新的事件循环
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            # 运行事件循环，处理队列任务
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
                # 使用 run_in_executor 获取任务，但捕获 RuntimeError
                # 这是因为事件循环关闭时可能会有竞态条件
                task = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, self._get_next_task
                    ),
                    timeout=0.1
                )
                # 关闭过程中忽略所有任务
                if not self._running:
                    break
                if task is None:
                    # 收到停止信号，立即退出，不处理待处理任务
                    break
                # 如果任务是协程，则 await 它
                if asyncio.iscoroutine(task):
                    await task
                # 如果任务是可调用的（lambda 等），调用它并检查返回值是否是协程
                elif callable(task):
                    result = task()
                    if asyncio.iscoroutine(result):
                        await result
            except asyncio.TimeoutError:
                # 超时继续循环检查 _running
                continue
            except RuntimeError as e:
                # 事件循环关闭时，run_in_executor 会失败
                # 这是一个正常情况，不需要记录错误
                if not self._running:
                    break
                # 继续循环
                continue
            except Exception as e:
                # 其他错误，记录一下
                if self._running:
                    logger.error(f"EventBus task error: {e}")

    def _get_next_task(self) -> Any:
        """从任务队列获取任务（同步）。"""
        try:
            return self._task_queue.get(timeout=0.1)
        except queue.Empty:
            return None

    def subscribe(
        self,
        callback: EventHandler,
        event_types: list[str] | None = None,
        blocking: bool = False,
        maxsize: int = 100,
    ) -> None:
        """
        订阅事件（同步接口）。

        Args:
            callback: 同步回调函数
            event_types: 订阅的事件类型列表，None 表示订阅所有
            blocking: 是否阻塞模式。阻塞模式下生产者等待消费者处理完成
            maxsize: 队列大小（仅非阻塞模式有效）

        Raises:
            RuntimeError: 如果 EventBus 已关闭
        """
        if self._closed:
            raise RuntimeError("EventBus is closed")

        self._ensure_started()

        if blocking:
            # 阻塞模式：直接存储 handler，不使用队列
            self._blocking_handlers.append((callback, event_types))
        else:
            # 将任务提交到 worker 线程，使用 lambda 包装确保可序列化
            self._task_queue.put(lambda: self._subscribe_async(callback, event_types, maxsize))

    async def _subscribe_async(
        self,
        callback: EventHandler,
        event_types: list[str] | None,
        maxsize: int,
    ) -> None:
        """异步订阅实现。"""
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
        while self._running:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                # 检查是否还在运行
                if not self._running:
                    break
                continue

            if event is None:  # 结束信号
                break

            # 关闭过程中忽略新事件
            if not self._running:
                break

            # 类型过滤
            if event_types is not None and event.type not in event_types:
                continue

            try:
                callback(event)  # 同步调用
            except Exception as e:
                logger.warning(f"Event handler error for {event.type}: {e}")

    def unsubscribe(
        self,
        callback: EventHandler,
        wait: bool = False,
        timeout: float | None = None,
    ) -> bool:
        """
        取消订阅，停止对应的消费者任务（同步接口）。

        Args:
            callback: 要取消的回调函数
            wait: 是否等待队列中的事件消费完毕
            timeout: 等待超时时间（秒），None 表示无限等待

        Returns:
            是否成功取消订阅
        """
        if self._closed:
            return False

        # 直接同步处理阻塞订阅者（不需要 worker 线程）
        for i, (handler, _) in enumerate(self._blocking_handlers):
            if handler is callback:
                self._blocking_handlers.pop(i)
                return True

        # 对于队列订阅者，需要通过 worker 线程处理
        handler_id = id(callback)
        
        # 检查是否已订阅（通过检查队列）
        if handler_id not in self._queues:
            return False

        # 如果需要等待，先等待队列变空
        if wait:
            queue = self._queues[handler_id]
            timeout_val = timeout if timeout else 30.0
            start_time = time.time()
            while not queue.empty():
                if time.time() - start_time > timeout_val:
                    break
                time.sleep(0.01)

        # 同步移除队列和任务引用
        if handler_id in self._queues:
            q = self._queues[handler_id]
            del self._queues[handler_id]
            # 发送结束信号到队列
            if self._loop is not None:
                try:
                    asyncio.run_coroutine_threadsafe(q.put(None), self._loop)
                except Exception:
                    pass
        if handler_id in self._consumer_filters:
            del self._consumer_filters[handler_id]
        if handler_id in self._consumer_tasks:
            task = self._consumer_tasks[handler_id]
            # 取消任务
            task.cancel()
            del self._consumer_tasks[handler_id]

        return True

    async def _unsubscribe_async(
        self,
        callback: EventHandler,
        wait: bool,
        timeout: float | None,
        result_queue: queue.Queue,
    ) -> bool:
        """异步取消订阅实现。"""
        handler_id = id(callback)
        result = True

        # 处理阻塞订阅者
        for i, (handler, _) in enumerate(self._blocking_handlers):
            if handler is callback:
                self._blocking_handlers.pop(i)
                result_queue.put(True)
                return True

        # 处理队列订阅者
        if handler_id not in self._queues:
            result_queue.put(False)
            return False

        queue = self._queues[handler_id]
        task = self._consumer_tasks.get(handler_id)

        if wait:
            # 等待队列消费完毕或超时
            try:
                await asyncio.wait_for(
                    self._wait_queue_empty(queue),
                    timeout=timeout if timeout else 30.0
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

        result_queue.put(True)
        return True

    async def _wait_queue_empty(self, queue: asyncio.Queue) -> None:
        """等待队列变空。"""
        while not queue.empty():
            await asyncio.sleep(0.01)

    def publish(self, event: Event) -> None:
        """
        发布事件（同步接口）。

        处理顺序：
        1. 先执行阻塞订阅者（确保关键操作如 DumpManager 完成）
        2. 再分发给非阻塞队列

        Args:
            event: 要发布的事件
        """
        if self._closed:
            return

        self._ensure_started()

        # 将任务提交到 worker 线程，使用 lambda 包装确保可序列化
        self._task_queue.put(lambda: self._publish_async(event))

    async def _publish_async(self, event: Event) -> None:
        """异步发布实现。"""
        if self._closed:
            return

        # 1. 阻塞订阅者（生产者等待）
        for handler, event_types in self._blocking_handlers:
            if event_types is not None and event.type not in event_types:
                continue
            try:
                handler(event)  # 同步调用
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

    def close(self, wait: bool = True, timeout: float | None = None) -> None:
        """
        关闭事件总线，优雅停止所有消费者（同步接口）。

        Args:
            wait: 是否等待消费者处理完队列中的事件
            timeout: 等待超时时间（秒），None 表示无限等待
        """
        if self._closed:
            return

        self._closed = True
        self._stopping = True  # 设置关闭标志
        self._running = False

        # 先取消所有消费者任务
        for task in list(self._consumer_tasks.values()):
            task.cancel()

        # 发送停止信号到主队列（这样 worker 线程会退出）
        try:
            self._task_queue.put_nowait(None)
        except queue.Full:
            pass

        # 等待 worker 线程结束
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout if timeout else 5.0)

        # 清理
        self._queues.clear()
        self._consumer_tasks.clear()
        self._consumer_filters.clear()
        self._blocking_handlers.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.close()
