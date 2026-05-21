"""Tests for AgentRunner module."""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from hawi.events import EventBus
from hawi.agent.runner import (
    QueueType,
    QueuedMessage,
    MessageQueueManager,
    EventMode,
    EventInterceptor,
    AgentExecutor,
    AgentRunnerState,
    ErrorAction,
    AgentRunner,
    AgentRunnerError,
)
from hawi.agent.agent import SteerPartMergeMode
from hawi.errors import ConfigurationError
from hawi.events import (
    AgentRunnerEnqueueEvent,
    AgentRunnerDequeueEvent,
    AgentRunnerInterruptEvent,
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

    def test_dequeue_all_high_prio(self):
        qm = MessageQueueManager()
        first = qm.enqueue_high_prio("first")
        second = qm.enqueue_high_prio("second")

        messages = qm.dequeue_all_high_prio()

        assert [msg.id for msg in messages] == [first.id, second.id]
        assert not qm.has_high_prio()

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

    def test_get_queue_messages(self):
        qm = MessageQueueManager()
        normal = qm.enqueue_normal("normal message", {"source": "test"})
        high = qm.enqueue_high_prio("high message")
        urgent = qm.enqueue_urgent("urgent message")

        messages = qm.get_queue_messages()

        assert messages["normal"][0]["id"] == normal.id
        assert messages["normal"][0]["queue"] == "normal"
        assert messages["normal"][0]["content_preview"] == "normal message"
        assert messages["normal"][0]["metadata"] == {"source": "test"}
        assert messages["high_prio"][0]["id"] == high.id
        assert messages["urgent"][0]["id"] == urgent.id

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
        mock_runner = MagicMock()
        interceptor = EventInterceptor(mock_runner)
        handler = lambda e: EventMode.PASS_THROUGH
        interceptor.register_handler("test.event", handler)
        assert "test.event" in interceptor._handlers

    def test_register_transform(self):
        mock_runner = MagicMock()
        interceptor = EventInterceptor(mock_runner)
        transform = lambda e: e
        interceptor.register_transform("test.event", transform)
        assert "test.event" in interceptor._transforms

    def test_unregister_handler(self):
        mock_runner = MagicMock()
        interceptor = EventInterceptor(mock_runner)
        interceptor.register_handler("test.event", lambda e: EventMode.PASS_THROUGH)
        assert interceptor.unregister_handler("test.event") is True
        assert interceptor.unregister_handler("test.event") is False


class TestAgentRunnerState:
    """Test AgentRunnerState enum."""

    def test_states(self):
        assert AgentRunnerState.IDLE
        assert AgentRunnerState.READY
        assert AgentRunnerState.RUNNING
        assert AgentRunnerState.INTERRUPTING


class TestErrorAction:
    """Test ErrorAction enum."""

    def test_actions(self):
        assert ErrorAction.RETRY
        assert ErrorAction.ABORT
        assert ErrorAction.CONTINUE


class TestAgentRunnerBasic:
    """Test AgentRunner basic functionality."""

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

    def test_runner_init(self, mock_agent):
        runner = AgentRunner(mock_agent)
        assert runner.agent is mock_agent
        assert runner.state == AgentRunnerState.IDLE

    def test_runner_reuses_agent_event_bus(self, mock_agent):
        runner = AgentRunner(mock_agent)
        assert runner.event_bus is mock_agent.event_bus

    def test_runner_enqueue_normal(self, mock_agent):
        runner = AgentRunner(mock_agent)
        msg_id = runner.enqueue("test message", "normal")
        assert msg_id
        assert runner.get_queue_lengths()["normal"] == 1

    def test_runner_enqueue_high_prio(self, mock_agent):
        runner = AgentRunner(mock_agent)
        msg_id = runner.enqueue("test message", "high_prio")
        assert msg_id
        assert runner.get_queue_lengths()["high_prio"] == 1

    def test_runner_steers_high_prio_message_during_active_tool(self, mock_agent):
        runner = AgentRunner(mock_agent)
        runner._executor._state = AgentRunnerState.RUNNING
        mock_agent.has_active_tool_calls = True

        msg_id = runner.enqueue(
            "new steer",
            "high_prio",
            metadata={"steer_merge_mode": SteerPartMergeMode.USER_MESSAGE_TEMPLATE},
        )

        assert msg_id == "steer-1234"
        assert runner.get_queue_lengths()["high_prio"] == 0
        mock_agent.steer.assert_called_once_with(
            "new steer",
            merge_mode=SteerPartMergeMode.USER_MESSAGE_TEMPLATE,
        )

    def test_runner_resolves_new_steer_merge_mode_from_string(self, mock_agent):
        runner = AgentRunner(mock_agent)
        runner._executor._state = AgentRunnerState.RUNNING
        mock_agent.has_active_tool_calls = True

        msg_id = runner.enqueue(
            "new steer",
            "high_prio",
            metadata={
                "steer_merge_mode": (
                    "tool_result_assistant_template_and_user_message"
                ),
            },
        )

        assert msg_id == "steer-1234"
        mock_agent.steer.assert_called_once_with(
            "new steer",
            merge_mode=(
                SteerPartMergeMode.TOOL_RESULT_ASSISTANT_TEMPLATE_AND_USER_MESSAGE
            ),
        )

    def test_runner_leaves_missing_steer_merge_mode_to_model(self, mock_agent):
        runner = AgentRunner(mock_agent)
        runner._executor._state = AgentRunnerState.RUNNING
        mock_agent.has_active_tool_calls = True

        msg_id = runner.enqueue("new steer", "high_prio")

        assert msg_id == "steer-1234"
        mock_agent.steer.assert_called_once_with(
            "new steer",
            merge_mode=None,
        )

    def test_runner_rejects_invalid_steer_merge_mode(self, mock_agent):
        runner = AgentRunner(mock_agent)
        runner._executor._state = AgentRunnerState.RUNNING
        mock_agent.has_active_tool_calls = True

        with pytest.raises(ConfigurationError, match="Invalid high_prio steer_merge_mode"):
            runner.enqueue(
                "new steer",
                "high_prio",
                metadata={"steer_merge_mode": "missing_default"},
            )

    def test_runner_enqueue_urgent(self, mock_agent):
        runner = AgentRunner(mock_agent)
        msg_id = runner.enqueue("test message", "urgent")
        # In a sync (non-async) context, urgent enqueue falls back to the
        # legacy path: the message goes to the urgent queue.
        assert msg_id
        assert runner.get_queue_lengths()["urgent"] == 1

    def test_runner_remove_message(self, mock_agent):
        runner = AgentRunner(mock_agent)
        msg_id = runner.enqueue("test", "normal")
        assert runner.remove_message(msg_id) is True
        assert runner.get_queue_lengths()["normal"] == 0

    def test_runner_clear_queue(self, mock_agent):
        runner = AgentRunner(mock_agent)
        runner.enqueue("test", "normal")
        count = runner.clear_queue("normal")
        assert count == 1
        assert runner.get_queue_lengths()["normal"] == 0

    def test_runner_clear_all_queues(self, mock_agent):
        runner = AgentRunner(mock_agent)
        runner.enqueue("n", "normal")
        runner.enqueue("h", "high_prio")
        # urgent enqueue in async context no longer populates the legacy queue
        # Use queue manager directly to populate urgent for test
        runner._queue_manager.enqueue_urgent("u")
        result = runner.clear_all_queues()
        assert result == {"normal": 1, "high_prio": 1, "urgent": 1}
        assert runner.get_queue_lengths() == {"normal": 0, "high_prio": 0, "urgent": 0}

    def test_runner_subscribe_proxies_to_agent(self, mock_agent):
        runner = AgentRunner(mock_agent)

        def handler(_event):
            return None

        runner.subscribe(handler, ["runner.enqueue"])
        mock_agent.subscribe.assert_called_once_with(handler, ["runner.enqueue"])

    def test_runner_unsubscribe_proxies_to_agent(self, mock_agent):
        runner = AgentRunner(mock_agent)

        def handler(_event):
            return None

        assert runner.unsubscribe(handler) is True
        mock_agent.unsubscribe.assert_called_once_with(handler)

    @pytest.mark.asyncio
    async def test_runner_emits_dequeue_and_passes_event_bus_to_agent(self, mock_agent):
        runner = AgentRunner(mock_agent)
        override_bus = EventBus()
        msg = runner._queue_manager.enqueue_normal("test", event_bus=override_bus)

        executed_messages: list[QueuedMessage] = []

        def fake_execute(message: QueuedMessage):
            executed_messages.append(message)
            return object()

        runner._executor.execute = MagicMock(side_effect=fake_execute)

        started = await runner._start_message_execution(msg)

        assert started is True
        assert executed_messages == [msg]
        mock_agent._emit_event.assert_awaited_once()
        emitted_event = mock_agent._emit_event.await_args.args[0]
        emitted_bus = mock_agent._emit_event.await_args.args[1]
        assert emitted_event.type == "runner.dequeue"
        assert emitted_event.message_id == msg.id
        assert emitted_bus is override_bus

    @pytest.mark.asyncio
    async def test_runner_starts_pending_inputs_before_queued_messages(self, mock_agent):
        runner = AgentRunner(mock_agent)
        runner._executor.execute_pending_inputs = MagicMock(return_value=object())

        started = await runner._start_pending_input_execution()

        assert started is True
        runner._executor.execute_pending_inputs.assert_called_once_with()
        mock_agent._emit_event.assert_awaited_once()
        emitted_event = mock_agent._emit_event.await_args.args[0]
        assert emitted_event.type == "runner.dequeue"
        assert emitted_event.message_id == "pending-inputs"
        assert emitted_event.queue_type == "high_prio"

    @pytest.mark.asyncio
    async def test_runner_executes_urgent_message_after_consuming_queue(self, mock_agent):
        runner = AgentRunner(mock_agent)
        runner.enqueue("stop now", "urgent")
        runner._executor.execute = MagicMock(return_value=object())

        async def stop_soon() -> None:
            await asyncio.sleep(0.01)
            runner.stop()

        await asyncio.gather(
            runner.run_forever(poll_interval=0.001),
            stop_soon(),
        )

        # Urgent enqueue now downgrades to stop_execution with message.
        # The message is submitted as high_prio via submit_immediate_message,
        # and the urgent queue stays empty.
        assert runner.get_queue_lengths()["urgent"] == 0
        runner._executor.execute.assert_called_once()
        executed_message = runner._executor.execute.call_args.args[0]
        assert executed_message.content == "stop now"
        # With the new stop-with-message path, messages go through high_prio,
        # not the legacy urgent queue.
        assert executed_message.queue_type == QueueType.HIGH_PRIO
        assert executed_message.metadata.get("intent") == "stop_with_message"

    @pytest.mark.asyncio
    async def test_runner_merges_queued_high_prio_messages_at_execution_point(self, mock_agent):
        runner = AgentRunner(mock_agent)
        first_id = runner.enqueue("first steer", "high_prio", metadata={"source": "gui"})
        second_id = runner.enqueue("second steer", "high_prio")
        executed_messages: list[QueuedMessage] = []

        def fake_execute(message: QueuedMessage):
            executed_messages.append(message)
            runner.stop()
            return object()

        runner._executor.execute = MagicMock(side_effect=fake_execute)

        await runner.run_forever(poll_interval=0.001)

        assert len(executed_messages) == 1
        executed = executed_messages[0]
        assert executed.id == first_id
        assert executed.queue_type == QueueType.HIGH_PRIO
        assert executed.content == [
            {"type": "text", "text": "first steer"},
            {"type": "text", "text": "\n\n"},
            {"type": "text", "text": "second steer"},
        ]
        assert executed.metadata["source"] == "gui"
        assert executed.metadata["merged_message_ids"] == [first_id, second_id]
        assert executed.metadata["merged_message_count"] == 2


class TestAgentRunnerErrorHooks:
    """Test AgentRunner error hooks."""

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
        runner = AgentRunner(mock_agent)
        action = await runner._on_model_error(Exception("test"), None)
        assert action == ErrorAction.CONTINUE

    @pytest.mark.asyncio
    async def test_on_agent_error_default(self, mock_agent):
        runner = AgentRunner(mock_agent)
        from hawi.agent.runner.queue import QueuedMessage
        msg = QueuedMessage.create("test", QueueType.NORMAL)
        action = await runner._on_agent_error(Exception("test"), msg)
        assert action == ErrorAction.CONTINUE

    @pytest.mark.asyncio
    async def test_on_runner_error_default(self, mock_agent):
        runner = AgentRunner(mock_agent)
        action = await runner._on_runner_error(Exception("test"))
        assert action == ErrorAction.CONTINUE


class TestAgentExecutorEventBus:
    @pytest.mark.asyncio
    async def test_execute_passes_message_event_bus_to_agent(self):
        agent = MagicMock()
        agent._arun_internal = AsyncMock(return_value=MagicMock())
        agent.clear_interrupt_state = MagicMock()
        runner = MagicMock()
        executor = AgentExecutor(agent, runner)
        event_bus = EventBus()
        message = QueuedMessage.create("hello", QueueType.NORMAL, event_bus=event_bus)

        task = executor.execute(message)
        assert task is not None
        await task

        agent._arun_internal.assert_awaited_once_with(
            "hello",
            event_bus=event_bus,
            message_metadata={
                "message_id": message.id,
                "queue": "normal",
                "display_message_type": "normal",
            },
        )

    @pytest.mark.asyncio
    async def test_execute_marks_high_prio_plain_message_as_normal(self):
        agent = MagicMock()
        agent._arun_internal = AsyncMock(return_value=MagicMock())
        agent.clear_interrupt_state = MagicMock()
        runner = MagicMock()
        executor = AgentExecutor(agent, runner)
        message = QueuedMessage.create(
            "hello",
            QueueType.HIGH_PRIO,
            metadata={"client": "gui"},
        )

        task = executor.execute(message)
        assert task is not None
        await task

        agent._arun_internal.assert_awaited_once_with(
            "hello",
            event_bus=None,
            message_metadata={
                "client": "gui",
                "message_id": message.id,
                "queue": "normal",
                "display_message_type": "normal",
                "source_queue": "high_prio",
                "materialized_as": "plain_user_message",
            },
        )

    @pytest.mark.asyncio
    async def test_execute_pending_inputs_runs_without_new_message(self):
        agent = MagicMock()
        agent._arun_internal = AsyncMock(return_value=MagicMock())
        agent.clear_interrupt_state = MagicMock()
        runner = MagicMock()
        executor = AgentExecutor(agent, runner)
        event_bus = EventBus()

        task = executor.execute_pending_inputs(event_bus=event_bus)
        assert task is not None
        await task

        agent._arun_internal.assert_awaited_once_with(
            None,
            event_bus=event_bus,
            message_metadata=None,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
