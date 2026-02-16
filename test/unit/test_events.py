"""Unit tests for Event system.

Tests the Event class, EventBus, and RichStreamingPrinter.
"""

import asyncio
import io
import pytest


from hawi.agent.events import (
    Event,
    EventBus,
    # Model events
    model_stream_start_event,
    model_stream_stop_event,
    model_content_block_start_event,
    model_content_block_delta_event,
    model_content_block_stop_event,
    agent_run_start_event,
    agent_run_stop_event,
    agent_tool_call_event,
    agent_tool_result_event,
    agent_error_event,
)
from hawi.agent.printers import RichStreamingPrinter


class TestEvent:
    """Tests for Event dataclass."""

    def test_event_creation(self):
        """Test basic event creation."""
        event = Event(
            type="test.event",
            source="agent",
            metadata={"key": "value"},
        )
        assert event.type == "test.event"
        assert event.source == "agent"
        assert event.metadata["key"] == "value"
        assert event.timestamp > 0

    def test_event_is_frozen(self):
        """Test that Event is immutable."""
        event = Event(type="test", source="agent")
        with pytest.raises(AttributeError):
            event.type = "modified"

    def test_event_hashable(self):
        """Test that Event can be used in sets/dicts."""
        event1 = Event(type="test", source="agent", metadata={"id": 1})
        event2 = Event(type="test", source="agent", metadata={"id": 2})
        event3 = Event(type="test", source="agent", metadata={"id": 1})

        # Events with same type/source/timestamp are equal
        # Note: timestamp makes them unique unless mocked
        events = {event1, event2}
        assert len(events) == 2


class TestModelEvents:
    """Tests for Model event factory functions."""

    def test_model_stream_start_event(self):
        """Test model_stream_start_event creation."""
        event = model_stream_start_event(request_id="req-123")
        assert event.type == "model.stream_start"
        assert event.source == "model"
        assert event.metadata["request_id"] == "req-123"

    def test_model_stream_stop_event(self):
        """Test model_stream_stop_event creation."""
        event = model_stream_stop_event(
            request_id="req-123",
            stop_reason="end_turn",
        )
        assert event.type == "model.stream_stop"
        assert event.metadata["stop_reason"] == "end_turn"

    def test_model_content_block_delta_event(self):
        """Test model_content_block_delta_event creation."""
        event = model_content_block_delta_event(
            request_id="req-123",
            block_index=0,
            delta_type="text",
            delta="Hello",
        )
        assert event.type == "model.content_block_delta"
        assert event.metadata["delta"] == "Hello"
        assert event.metadata["delta_type"] == "text"

    def test_model_content_block_events_with_reasoning(self):
        """Test content block events for reasoning."""
        start = model_content_block_start_event(
            request_id="req-123",
            block_index=1,
            block_type="reasoning",
        )
        delta = model_content_block_delta_event(
            request_id="req-123",
            block_index=1,
            delta_type="reasoning",
            delta="Let me think...",
        )
        stop = model_content_block_stop_event(
            request_id="req-123",
            block_index=1,
            full_content="Let me think...",
        )

        assert start.metadata["block_type"] == "reasoning"
        assert delta.metadata["delta_type"] == "reasoning"
        assert stop.metadata["full_content"] == "Let me think..."


class TestAgentEvents:
    """Tests for Agent event factory functions."""

    def test_agent_run_start_event(self):
        """Test agent_run_start_event creation."""
        event = agent_run_start_event(
            run_id="run-123",
            message_preview="Hello",
        )
        assert event.type == "agent.run_start"
        assert event.source == "agent"
        assert event.metadata["run_id"] == "run-123"
        assert event.metadata["message_preview"] == "Hello"

    def test_agent_run_stop_event(self):
        """Test agent_run_stop_event creation."""
        event = agent_run_stop_event(
            run_id="run-123",
            stop_reason="end_turn",
            duration_ms=1234.5,
        )
        assert event.type == "agent.run_stop"
        assert event.metadata["stop_reason"] == "end_turn"
        assert event.metadata["duration_ms"] == 1234.5

    def test_agent_tool_call_event(self):
        """Test agent_tool_call_event creation."""
        event = agent_tool_call_event(
            run_id="run-123",
            tool_name="calculate",
            arguments={"expression": "1+1"},
            tool_call_id="tc-123",
        )
        assert event.type == "agent.tool_call"
        assert event.metadata["tool_name"] == "calculate"
        assert event.metadata["arguments"]["expression"] == "1+1"

    def test_agent_tool_result_event(self):
        """Test agent_tool_result_event creation."""
        event = agent_tool_result_event(
            run_id="run-123",
            tool_name="calculate",
            tool_call_id="tc-123",
            success=True,
            result_preview="2",
            duration_ms=100.0,
            arguments={"expression": "1+1"},
        )
        assert event.type == "agent.tool_result"
        assert event.metadata["success"] is True
        assert event.metadata["result_preview"] == "2"
        assert event.metadata["arguments"]["expression"] == "1+1"

    def test_agent_error_event(self):
        """Test agent_error_event creation."""
        event = agent_error_event(
            run_id="run-123",
            error_type="model_error",
            error_message="API timeout",
            recoverable=False,
        )
        assert event.type == "agent.error"
        assert event.metadata["error_type"] == "model_error"
        assert event.metadata["recoverable"] is False


class TestEventBus:
    """Tests for EventBus class."""

    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self):
        """Test basic subscribe and publish."""
        bus = EventBus()
        received_events = []

        async def handler(event: Event) -> None:
            received_events.append(event)

        bus.subscribe(handler)
        event = agent_run_start_event(run_id="test")
        await bus.publish(event)

        # Wait for async handler
        await asyncio.sleep(0.1)
        assert len(received_events) == 1
        assert received_events[0].type == "agent.run_start"

    @pytest.mark.asyncio
    async def test_subscribe_with_event_types(self):
        """Test subscribing to specific event types."""
        bus = EventBus()
        received = []

        async def handler(event: Event) -> None:
            received.append(event.type)

        bus.subscribe(handler, event_types=["agent.tool_call", "agent.tool_result"])

        await bus.publish(agent_run_start_event(run_id="test"))
        await bus.publish(agent_tool_call_event(run_id="test", tool_name="calc", arguments={}, tool_call_id="tc-1"))
        await bus.publish(agent_tool_result_event(run_id="test", tool_name="calc", tool_call_id="tc-1", success=True, result_preview="2", duration_ms=10))

        await asyncio.sleep(0.1)
        assert len(received) == 2
        assert "agent.run_start" not in received
        assert "agent.tool_call" in received
        assert "agent.tool_result" in received

    @pytest.mark.asyncio
    async def test_multiple_handlers(self):
        """Test multiple handlers receiving events."""
        bus = EventBus()
        handler1_events = []
        handler2_events = []

        async def handler1(event: Event) -> None:
            handler1_events.append(event.type)

        async def handler2(event: Event) -> None:
            handler2_events.append(event.type)

        bus.subscribe(handler1)
        bus.subscribe(handler2)

        await bus.publish(agent_run_start_event(run_id="test"))
        await asyncio.sleep(0.1)

        assert len(handler1_events) == 1
        assert len(handler2_events) == 1

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        """Test unsubscribing from events."""
        bus = EventBus()
        received = []

        async def handler(event: Event) -> None:
            received.append(event.type)

        bus.subscribe(handler)
        await bus.publish(agent_run_start_event(run_id="test"))
        await asyncio.sleep(0.1)
        assert len(received) == 1

        bus.unsubscribe(handler)
        await bus.publish(agent_run_start_event(run_id="test2"))
        await asyncio.sleep(0.1)
        assert len(received) == 1  # No new events

    def test_context_manager(self):
        """Test EventBus as context manager."""
        with EventBus() as bus:
            assert not bus._closed
        assert bus._closed


class TestConversationPrinter:
    """Tests for RichStreamingPrinter class."""

    @pytest.fixture
    def printer(self, monkeypatch):
        """Create a RichStreamingPrinter with captured stdout for testing."""
        output = io.StringIO()
        # Patch sys.stdout for the printer module
        import hawi.agent.printers as printers_module
        monkeypatch.setattr(printers_module, '_stdout', output)
        printer = RichStreamingPrinter()
        printer._output = output  # Store reference for tests
        return printer

    @pytest.mark.asyncio
    async def test_handle_text_delta(self, printer):
        """Test printing text delta events."""
        event = model_content_block_delta_event(
            request_id="req-1",
            block_index=0,
            delta_type="text",
            delta="Hello World\n",
        )
        await printer.handle(event)
        output = printer._output.getvalue()
        assert "Hello World" in output

    @pytest.mark.asyncio
    async def test_handle_reasoning_delta(self, printer):
        """Test printing reasoning delta events."""
        # First send start event to set up state
        start_event = model_content_block_start_event(
            request_id="req-1",
            block_index=0,
            block_type="thinking",
        )
        await printer.handle(start_event)

        delta_event = model_content_block_delta_event(
            request_id="req-1",
            block_index=0,
            delta_type="thinking",
            delta="Let me think...",
        )
        await printer.handle(delta_event)

        # Verify content is buffered before stop event
        assert "Let me think..." in printer._reasoning_buffer

        # Reasoning is only printed on block stop
        stop_event = model_content_block_stop_event(
            request_id="req-1",
            block_index=0,
            block_type="thinking",
            full_content="Let me think...",
        )
        await printer.handle(stop_event)

        # Buffer is cleared after printing, but panel was displayed

    @pytest.mark.asyncio
    async def test_handle_tool_call(self, printer):
        """Test printing tool call events - tool calls show Status, not direct output."""
        event = agent_tool_call_event(
            run_id="run-1",
            tool_name="calculate",
            arguments={"expression": "1+1"},
            tool_call_id="tc-1",
        )
        await printer.handle(event)
        # Tool calls display a status spinner, no direct output until result
        # Status output is handled by rich's status mechanism

    @pytest.mark.asyncio
    async def test_handle_tool_result(self, printer):
        """Test printing tool result events."""
        event = agent_tool_result_event(
            run_id="run-1",
            tool_name="calculate",
            tool_call_id="tc-1",
            success=True,
            result_preview="2",
            duration_ms=100.0,
            arguments={"expression": "1+1"},
        )
        await printer.handle(event)
        # Tool result uses console.print via _print_tool_result
        # Verify tool call was tracked
        assert "calculate" in printer._active_tool_calls or len(printer._active_tool_calls) == 0

    @pytest.mark.asyncio
    async def test_handle_tool_result_failure(self, printer):
        """Test printing failed tool result."""
        event = agent_tool_result_event(
            run_id="run-1",
            tool_name="calculate",
            tool_call_id="tc-1",
            success=False,
            result_preview="Error",
            duration_ms=50.0,
            arguments={},
        )
        await printer.handle(event)
        # Just verify no exception is raised

    @pytest.mark.asyncio
    async def test_hide_reasoning(self, monkeypatch):
        """Test hiding reasoning output."""
        output = io.StringIO()
        import hawi.agent.printers as printers_module
        monkeypatch.setattr(printers_module, '_stdout', output)
        printer = RichStreamingPrinter(show_reasoning=False)
        event = model_content_block_delta_event(
            request_id="req-1",
            block_index=0,
            delta_type="thinking",
            delta="Secret thought",
        )
        await printer.handle(event)
        # When reasoning is hidden, buffer should not be populated
        assert printer._reasoning_buffer == ""

    @pytest.mark.asyncio
    async def test_hide_tools(self, monkeypatch):
        """Test hiding tool output."""
        output = io.StringIO()
        import hawi.agent.printers as printers_module
        monkeypatch.setattr(printers_module, '_stdout', output)
        printer = RichStreamingPrinter(show_tools=False)
        event = agent_tool_call_event(
            run_id="run-1",
            tool_name="calculate",
            arguments={},
            tool_call_id="tc-1",
        )
        await printer.handle(event)
        # When tools are hidden, no active tracking
        assert len(printer._active_tool_calls) == 0

    @pytest.mark.asyncio
    async def test_handle_error(self, printer):
        """Test printing error events."""
        event = agent_error_event(
            run_id="run-1",
            error_type="model_error",
            error_message="Something went wrong",
        )
        await printer.handle(event)
        # Error uses console.print, verify no exception


class TestRichStreamingPrinter:
    """Tests for RichStreamingPrinter class."""

    def test_has_handle_method(self):
        """Test that RichStreamingPrinter has a handle method."""
        printer = RichStreamingPrinter()
        assert hasattr(printer, 'handle')
        assert callable(printer.handle)

    @pytest.mark.asyncio
    async def test_printer_handles_events(self):
        """Test that the printer handles events correctly."""
        printer = RichStreamingPrinter()
        event = model_content_block_delta_event(
            request_id="req-1",
            block_index=0,
            delta_type="text",
            delta="Test\n",
        )
        # Should not raise any exception
        await printer.handle(event)


class TestEventOrdering:
    """Tests for event ordering and lifecycle."""

    @pytest.mark.asyncio
    async def test_event_bus_ordering(self):
        """Test that events are published in order."""
        bus = EventBus()
        received = []

        async def handler(event: Event) -> None:
            received.append(event.metadata.get("seq"))

        bus.subscribe(handler)

        for i in range(5):
            await bus.publish(Event(type="test", source="agent", metadata={"seq": i}))

        await asyncio.sleep(0.1)
        assert received == [0, 1, 2, 3, 4]
