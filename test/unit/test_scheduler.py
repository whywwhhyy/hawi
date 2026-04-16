"""Tests for HawiScheduler module."""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from hawi.events import EventBus
from hawi.agent.scheduler import (
    QueueType,
    QueuedMessage,
    MessageQueueManager,
    EventMode,
    EventInterceptor,
    AgentExecutor,
    SchedulerState,
    ErrorAction,
    HawiScheduler,
    SchedulerError,
)
from hawi.agent.agent import SteerPartMergeMode
from hawi.events import (
    SchedulerEnqueueEvent,
    SchedulerDequeueEvent,
    SchedulerInterruptEvent,
)
from hawi.models.message import (
    ContentPart
)


class TestQueueType:
    """Test QueueType enum."""

    def test_queue_type_values(self):
        assert QueueType.NORMAL
        assert QueueType.HIGH_PRIO
        assert QueueType.URGENT


class TestQueuedMessage:
    """Test QueuedMessage dataclass."""

    def test_create_message(self):
        msg = QueuedMessage.create("test content", QueueType.NORMAL)
        assert msg.id
        assert len(msg.id) == 8  # UUID[:8]
        assert msg.content == "test content"
        assert msg.queue_type == QueueType.NORMAL
        assert msg.created_at > 0
        assert msg.metadata == {}

    def test_create_message_with_metadata(self):
        msg = QueuedMessage.create("test", QueueType.HIGH_PRIO, {"key": "value"})
        assert msg.metadata == {"key": "value"}

    def test_get_content_preview_text(self):
        msg = QueuedMessage.create("short text", QueueType.NORMAL)
        assert msg.get_content_preview() == "short text"

    def test_get_content_preview_truncated(self):
        long_text = "a" * 200
        msg = QueuedMessage.create(long_text, QueueType.NORMAL)
        preview = msg.get_content_preview(100)
        assert len(preview) == 100
        assert preview.endswith("...")

    def test_get_content_preview_content_parts(self):
        content:list[ContentPart] = [{"type": "text", "text": "hello world"}]
        msg = QueuedMessage.create(content, QueueType.NORMAL)
        assert msg.get_content_preview() == "hello world"


class TestMessageQueueManager:
    """Test MessageQueueManager."""

    def test_enqueue_normal(self):
        qm = MessageQueueManager()
        msg = qm.enqueue_normal("normal message")
        assert msg.queue_type == QueueType.NORMAL
        assert qm.has_normal()

    def test_enqueue_high_prio(self):
        qm = MessageQueueManager()
        msg = qm.enqueue_high_prio("high prio message")
        assert msg.queue_type == QueueType.HIGH_PRIO
        assert qm.has_high_prio()

    def test_enqueue_urgent(self):
        qm = MessageQueueManager()
        msg = qm.enqueue_urgent("urgent message")
        assert msg.queue_type == QueueType.URGENT
        assert qm.has_urgent()

    def test_urgent_single_slot(self):
        """Urgent queue should replace old message."""
        qm = MessageQueueManager()
        msg1 = qm.enqueue_urgent("first")
        msg2 = qm.enqueue_urgent("second")
        assert qm.get_queue_lengths()["urgent"] == 1
        urgent_message = qm.dequeue_urgent()
        assert urgent_message and urgent_message.id == msg2.id

    def test_dequeue_order(self):
        qm = MessageQueueManager()
        msg1 = qm.enqueue_high_prio("first")
        msg2 = qm.enqueue_high_prio("second")
        msg = qm.dequeue_high_prio()
        assert msg and msg.id == msg1.id
        msg = qm.dequeue_high_prio()
        assert msg and msg.id == msg2.id

    def test_dequeue_empty_returns_none(self):
        qm = MessageQueueManager()
        assert qm.dequeue_normal() is None
        assert qm.dequeue_high_prio() is None
        assert qm.dequeue_urgent() is None

    def test_peek_high_prio(self):
        qm = MessageQueueManager()
        msg = qm.enqueue_high_prio("peek me")
        peeked = qm.peek_high_prio()
        assert peeked and peeked.id == msg.id
        assert qm.has_high_prio()  # Should not remove

    def test_insert_front_normal(self):
        qm = MessageQueueManager()
        msg1 = qm.enqueue_normal("first")
        msg2 = QueuedMessage.create("second", QueueType.NORMAL)
        qm.insert_front_normal(msg2)
        msg = qm.dequeue_normal()
        assert msg and msg.id == msg2.id
        msg = qm.dequeue_normal()
        assert msg and msg.id == msg1.id

    def test_get_queue_lengths(self):
        qm = MessageQueueManager()
        qm.enqueue_normal("n1")
        qm.enqueue_normal("n2")
        qm.enqueue_high_prio("h1")
        qm.enqueue_urgent("u1")
        lengths = qm.get_queue_lengths()
        assert lengths == {"normal": 2, "high_prio": 1, "urgent": 1}

    def test_remove_message_by_id(self):
        qm = MessageQueueManager()
        msg = qm.enqueue_normal("to remove")
        assert qm.remove_message(msg.id) is True
        assert not qm.has_normal()

    def test_remove_message_not_found(self):
        qm = MessageQueueManager()
        assert qm.remove_message("nonexistent") is False

    def test_remove_messages_by_filter(self):
        qm = MessageQueueManager()
        msg1 = qm.enqueue_normal("keep")
        msg2 = qm.enqueue_normal("remove")
        removed = qm.remove_messages(lambda m: m.content == "remove")
        assert len(removed) == 1
        assert msg2.id in removed
        assert qm.get_queue_lengths()["normal"] == 1

    def test_clear_queue(self):
        qm = MessageQueueManager()
        qm.enqueue_normal("n1")
        qm.enqueue_normal("n2")
        count = qm.clear_queue(QueueType.NORMAL)
        assert count == 2
        assert not qm.has_normal()

    def test_clear_all_queues(self):
        qm = MessageQueueManager()
        qm.enqueue_normal("n")
        qm.enqueue_high_prio("h")
        qm.enqueue_urgent("u")
        result = qm.clear_all_queues()
        assert result == {"normal": 1, "high_prio": 1, "urgent": 1}
        assert qm.get_queue_lengths() == {"normal": 0, "high_prio": 0, "urgent": 0}


class TestEventMode:
    """Test EventMode enum."""

    def test_event_modes(self):
        assert EventMode.PASS_THROUGH
        assert EventMode.INTERCEPT
        assert EventMode.REPROCESS
        assert EventMode.SUPPRESS


class TestEventInterceptor:
    """Test EventInterceptor."""

    def test_register_handler(self):
        mock_scheduler = MagicMock()
        interceptor = EventInterceptor(mock_scheduler)
        handler = lambda e: EventMode.PASS_THROUGH
        interceptor.register_handler("test.event", handler)
        assert "test.event" in interceptor._handlers

    def test_register_transform(self):
        mock_scheduler = MagicMock()
        interceptor = EventInterceptor(mock_scheduler)
        transform = lambda e: e
        interceptor.register_transform("test.event", transform)
        assert "test.event" in interceptor._transforms

    def test_unregister_handler(self):
        mock_scheduler = MagicMock()
        interceptor = EventInterceptor(mock_scheduler)
        interceptor.register_handler("test.event", lambda e: EventMode.PASS_THROUGH)
        assert interceptor.unregister_handler("test.event") is True
        assert interceptor.unregister_handler("test.event") is False


class TestSchedulerState:
    """Test SchedulerState enum."""

    def test_states(self):
        assert SchedulerState.IDLE
        assert SchedulerState.READY
        assert SchedulerState.RUNNING
        assert SchedulerState.INTERRUPTING


class TestErrorAction:
    """Test ErrorAction enum."""

    def test_actions(self):
        assert ErrorAction.RETRY
        assert ErrorAction.ABORT
        assert ErrorAction.CONTINUE


class TestHawiSchedulerBasic:
    """Test HawiScheduler basic functionality."""

    @pytest.fixture
    def mock_agent(self):
        agent = MagicMock()
        agent.event_bus = EventBus()
        agent.event_bus.subscribe = MagicMock()
        agent.event_bus.unsubscribe = MagicMock(return_value=True)
        agent.interrupt = MagicMock(return_value=[])
        agent.clear_interrupt_state = MagicMock()
        agent.context = MagicMock()
        agent.context.messages = []
        agent.subscribe = MagicMock()
        agent.unsubscribe = MagicMock(return_value=True)
        agent._emit_event = AsyncMock()
        agent.has_active_tool_calls = False
        agent.steer = MagicMock(return_value="steer-1234")
        return agent

    def test_scheduler_init(self, mock_agent):
        scheduler = HawiScheduler(mock_agent)
        assert scheduler.agent is mock_agent
        assert scheduler.state == SchedulerState.IDLE

    def test_scheduler_reuses_agent_event_bus(self, mock_agent):
        scheduler = HawiScheduler(mock_agent)
        assert scheduler.event_bus is mock_agent.event_bus

    def test_scheduler_enqueue_normal(self, mock_agent):
        scheduler = HawiScheduler(mock_agent)
        msg_id = scheduler.enqueue("test message", "normal")
        assert msg_id
        assert scheduler.get_queue_lengths()["normal"] == 1

    def test_scheduler_enqueue_high_prio(self, mock_agent):
        scheduler = HawiScheduler(mock_agent)
        msg_id = scheduler.enqueue("test message", "high_prio")
        assert msg_id
        assert scheduler.get_queue_lengths()["high_prio"] == 1

    def test_scheduler_steers_high_prio_message_during_active_tool(self, mock_agent):
        scheduler = HawiScheduler(mock_agent)
        scheduler._executor._state = SchedulerState.RUNNING
        mock_agent.has_active_tool_calls = True

        msg_id = scheduler.enqueue(
            "new steer",
            "high_prio",
            metadata={"steer_merge_mode": SteerPartMergeMode.USER_MESSAGE_TEMPLATE},
        )

        assert msg_id == "steer-1234"
        assert scheduler.get_queue_lengths()["high_prio"] == 0
        mock_agent.steer.assert_called_once_with(
            "new steer",
            merge_mode=SteerPartMergeMode.USER_MESSAGE_TEMPLATE,
        )

    def test_scheduler_enqueue_urgent(self, mock_agent):
        scheduler = HawiScheduler(mock_agent)
        msg_id = scheduler.enqueue("test message", "urgent")
        assert msg_id
        assert scheduler.get_queue_lengths()["urgent"] == 1

    def test_scheduler_remove_message(self, mock_agent):
        scheduler = HawiScheduler(mock_agent)
        msg_id = scheduler.enqueue("test", "normal")
        assert scheduler.remove_message(msg_id) is True
        assert scheduler.get_queue_lengths()["normal"] == 0

    def test_scheduler_clear_queue(self, mock_agent):
        scheduler = HawiScheduler(mock_agent)
        scheduler.enqueue("test", "normal")
        count = scheduler.clear_queue("normal")
        assert count == 1
        assert scheduler.get_queue_lengths()["normal"] == 0

    def test_scheduler_clear_all_queues(self, mock_agent):
        scheduler = HawiScheduler(mock_agent)
        scheduler.enqueue("n", "normal")
        scheduler.enqueue("h", "high_prio")
        scheduler.enqueue("u", "urgent")
        result = scheduler.clear_all_queues()
        assert result == {"normal": 1, "high_prio": 1, "urgent": 1}
        assert scheduler.get_queue_lengths() == {"normal": 0, "high_prio": 0, "urgent": 0}

    def test_scheduler_subscribe_proxies_to_agent(self, mock_agent):
        scheduler = HawiScheduler(mock_agent)

        def handler(_event):
            return None

        scheduler.subscribe(handler, ["scheduler.enqueue"])
        mock_agent.subscribe.assert_called_once_with(handler, ["scheduler.enqueue"])

    def test_scheduler_unsubscribe_proxies_to_agent(self, mock_agent):
        scheduler = HawiScheduler(mock_agent)

        def handler(_event):
            return None

        assert scheduler.unsubscribe(handler) is True
        mock_agent.unsubscribe.assert_called_once_with(handler)

    @pytest.mark.asyncio
    async def test_scheduler_emits_dequeue_and_passes_event_bus_to_agent(self, mock_agent):
        scheduler = HawiScheduler(mock_agent)
        override_bus = EventBus()
        msg = scheduler._queue_manager.enqueue_normal("test", event_bus=override_bus)

        executed_messages: list[QueuedMessage] = []

        def fake_execute(message: QueuedMessage):
            executed_messages.append(message)
            return object()

        scheduler._executor.execute = MagicMock(side_effect=fake_execute)

        started = await scheduler._start_message_execution(msg)

        assert started is True
        assert executed_messages == [msg]
        mock_agent._emit_event.assert_awaited_once()
        emitted_event = mock_agent._emit_event.await_args.args[0]
        emitted_bus = mock_agent._emit_event.await_args.args[1]
        assert emitted_event.type == "scheduler.dequeue"
        assert emitted_event.message_id == msg.id
        assert emitted_bus is override_bus


class TestHawiSchedulerErrorHooks:
    """Test HawiScheduler error hooks."""

    @pytest.fixture
    def mock_agent(self):
        agent = MagicMock()
        agent.event_bus = EventBus()
        agent.event_bus.subscribe = MagicMock()
        agent.interrupt = MagicMock(return_value=[])
        agent.clear_interrupt_state = MagicMock()
        agent.context = MagicMock()
        agent.context.messages = []
        agent.subscribe = MagicMock()
        agent.unsubscribe = MagicMock(return_value=True)
        agent._emit_event = AsyncMock()
        agent.has_active_tool_calls = False
        agent.steer = MagicMock(return_value="steer-1234")
        return agent

    @pytest.mark.asyncio
    async def test_on_model_error_default(self, mock_agent):
        scheduler = HawiScheduler(mock_agent)
        action = await scheduler._on_model_error(Exception("test"), None)
        assert action == ErrorAction.CONTINUE

    @pytest.mark.asyncio
    async def test_on_agent_error_default(self, mock_agent):
        scheduler = HawiScheduler(mock_agent)
        from hawi.agent.scheduler.queue import QueuedMessage
        msg = QueuedMessage.create("test", QueueType.NORMAL)
        action = await scheduler._on_agent_error(Exception("test"), msg)
        assert action == ErrorAction.CONTINUE

    @pytest.mark.asyncio
    async def test_on_scheduler_error_default(self, mock_agent):
        scheduler = HawiScheduler(mock_agent)
        action = await scheduler._on_scheduler_error(Exception("test"))
        assert action == ErrorAction.CONTINUE


class TestAgentExecutorEventBus:
    @pytest.mark.asyncio
    async def test_execute_passes_message_event_bus_to_agent(self):
        agent = MagicMock()
        agent._arun_internal = AsyncMock(return_value=MagicMock())
        agent.clear_interrupt_state = MagicMock()
        scheduler = MagicMock()
        executor = AgentExecutor(agent, scheduler)
        event_bus = EventBus()
        message = QueuedMessage.create("hello", QueueType.NORMAL, event_bus=event_bus)

        task = executor.execute(message)
        assert task is not None
        await task

        agent._arun_internal.assert_awaited_once_with("hello", event_bus=event_bus)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
