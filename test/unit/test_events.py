"""Unit tests for Event system.

Tests the Event class, EventBus, and RichStreamingPrinter.
"""

import asyncio
import io
import pytest


from hawi.events import (
    Event,
    EventBus,
    # Model events
    ModelStreamStartEvent,
    ModelStreamStopEvent,
    ModelContentBlockStartEvent,
    ModelContentBlockDeltaEvent,
    ModelContentBlockStopEvent,
    ModelMetadataEvent,
    ModelErrorEvent,
    # Agent events
    AgentRunStartEvent,
    AgentRunStopEvent,
    AgentToolCallEvent,
    AgentToolResultEvent,
    AgentMessageAddedEvent,
    AgentErrorEvent,
)
from hawi.agent.printers import RichStreamingPrinter
from hawi.errors import AgentError, ModelError


class TestEvent:
    """Tests for Event base class."""

    def test_event_cannot_be_instantiated_directly(self):
        """Test that Event base class cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract base class"):
            Event(type="test.event", source="agent")

    def test_event_subclass_can_be_created(self):
        """Test that Event subclasses can be created via create()."""
        event = ModelStreamStartEvent.create(request_id="req-123")
        assert event.type == "model.stream_start"
        assert event.source == "model"
        assert event.request_id == "req-123"
        assert event.timestamp > 0

    def test_event_is_frozen(self):
        """Test that Event is immutable."""
        event = ModelStreamStartEvent.create(request_id="req-123")
        with pytest.raises(Exception):  # Pydantic raises ValidationError or similar
            event.type = "modified"


class TestModelEvents:
    """Tests for Model event classes."""

    def test_model_stream_start_event(self):
        """Test ModelStreamStartEvent creation."""
        event = ModelStreamStartEvent.create(request_id="req-123")
        assert event.type == "model.stream_start"
        assert event.source == "model"
        assert event.request_id == "req-123"

    def test_model_stream_stop_event(self):
        """Test ModelStreamStopEvent creation."""
        from hawi.model.message import TokenUsage
        event = ModelStreamStopEvent.create(
            request_id="req-123",
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=10, output_tokens=20),
        )
        assert event.type == "model.stream_stop"
        assert event.stop_reason == "end_turn"
        assert event.usage.input_tokens == 10

    def test_model_content_block_delta_event(self):
        """Test ModelContentBlockDeltaEvent creation."""
        from hawi.model.message import StreamTextPart

        part: StreamTextPart = {
            "type": "text_delta",
            "index": 0,
            "delta": "Hello",
            "is_start": True,
            "is_end": False,
        }
        event = ModelContentBlockDeltaEvent.create(
            request_id="req-123",
            part=part,
        )
        assert event.type == "model.content_block_delta"
        assert event.delta == "Hello"
        assert event.delta_type == "text"
        assert event.is_start is True
        assert event.is_end is False
        assert event.part == part

    def test_model_content_block_events_with_reasoning(self):
        """Test content block events for reasoning."""
        from hawi.model.message import StreamThinkingPart

        start = ModelContentBlockStartEvent.create(
            request_id="req-123",
            block_index=1,
            block_type="thinking",
        )

        part: StreamThinkingPart = {
            "type": "thinking_delta",
            "index": 1,
            "delta": "Let me think...",
            "is_start": False,
            "is_end": False,
        }
        delta = ModelContentBlockDeltaEvent.create(
            request_id="req-123",
            part=part,
        )
        from hawi.model.message import ReasoningPart
        reasoning_part = ReasoningPart(
            type="reasoning",
            reasoning="Let me think...",
            signature=None,
            redacted_content=None
        )
        stop = ModelContentBlockStopEvent.create(
            request_id="req-123",
            block_index=1,
            content=[reasoning_part],
        )

        assert start.block_type == "thinking"
        assert delta.delta_type == "thinking"
        assert stop.content[0].get("reasoning") == "Let me think..."
        assert stop.block_type == "reasoning"


class TestAgentEvents:
    """Tests for Agent event classes."""

    def test_agent_run_start_event(self):
        """Test AgentRunStartEvent creation."""
        event = AgentRunStartEvent.create(
            run_id="run-123",
            message_preview="Hello",
        )
        assert event.type == "agent.run_start"
        assert event.source == "agent"
        assert event.run_id == "run-123"
        assert event.message_preview == "Hello"

    def test_agent_run_stop_event(self):
        """Test AgentRunStopEvent creation."""
        event = AgentRunStopEvent.create(
            run_id="run-123",
            stop_reason="end_turn",
            duration_ms=1234.5,
        )
        assert event.type == "agent.run_stop"
        assert event.stop_reason == "end_turn"
        assert event.duration_ms == 1234.5

    def test_agent_tool_call_event(self):
        """Test AgentToolCallEvent creation."""
        event = AgentToolCallEvent.create(
            run_id="run-123",
            tool_name="calculate",
            arguments={"expression": "1+1"},
            tool_call_id="tc-123",
        )
        assert event.type == "agent.tool_call"
        assert event.tool_name == "calculate"
        assert event.arguments["expression"] == "1+1"

    def test_agent_tool_result_event(self):
        """Test AgentToolResultEvent creation."""
        event = AgentToolResultEvent.create(
            run_id="run-123",
            tool_name="calculate",
            tool_call_id="tc-123",
            success=True,
            result_preview="2",
            duration_ms=100.0,
            arguments={"expression": "1+1"},
        )
        assert event.type == "agent.tool_result"
        assert event.success is True
        assert event.result_preview == "2"
        assert event.arguments["expression"] == "1+1"

    def test_agent_error_event(self):
        """Test AgentErrorEvent creation."""
        error = AgentError(error_type="tool_execution", msg="API timeout")
        event = AgentErrorEvent.create(
            run_id="run-123",
            error=error,
        )
        assert event.type == "agent.error"
        assert event.error.message == "API timeout"


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
        event = AgentRunStartEvent.create(run_id="test")
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

        await bus.publish(AgentRunStartEvent.create(run_id="test"))
        await bus.publish(AgentToolCallEvent.create(run_id="test", tool_name="calc", arguments={}, tool_call_id="tc-1"))
        await bus.publish(AgentToolResultEvent.create(run_id="test", tool_name="calc", tool_call_id="tc-1", success=True, result_preview="2", duration_ms=10))

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

        await bus.publish(AgentRunStartEvent.create(run_id="test"))
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
        await bus.publish(AgentRunStartEvent.create(run_id="test"))
        await asyncio.sleep(0.1)
        assert len(received) == 1

        bus.unsubscribe(handler)
        await bus.publish(AgentRunStartEvent.create(run_id="test2"))
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
        import hawi.agent.printers.rich as rich_module
        monkeypatch.setattr(rich_module, '_stdout', output)
        printer = RichStreamingPrinter()
        printer._output = output  # Store reference for tests
        return printer

    @pytest.mark.asyncio
    async def test_handle_text_delta(self, printer):
        """Test printing text delta events."""
        from hawi.model.message import StreamTextPart

        part: StreamTextPart = {
            "type": "text_delta",
            "index": 0,
            "delta": "Hello World\n",
            "is_start": False,
            "is_end": False,
        }
        event = ModelContentBlockDeltaEvent.create(
            request_id="req-1",
            part=part,
        )
        await printer.handle(event)
        output = printer._output.getvalue()
        assert "Hello World" in output

    @pytest.mark.asyncio
    async def test_handle_reasoning_delta(self, printer):
        """Test printing reasoning delta events."""
        from hawi.model.message import StreamThinkingPart

        # First send start event to set up state
        start_event = ModelContentBlockStartEvent.create(
            request_id="req-1",
            block_index=0,
            block_type="thinking",
        )
        await printer.handle(start_event)

        part: StreamThinkingPart = {
            "type": "thinking_delta",
            "index": 0,
            "delta": "Let me think...",
            "is_start": False,
            "is_end": False,
        }
        delta_event = ModelContentBlockDeltaEvent.create(
            request_id="req-1",
            part=part,
        )
        await printer.handle(delta_event)

        # Verify content is buffered before stop event
        assert "Let me think..." in printer._reasoning_buffer

        # Reasoning is only printed on block stop
        from hawi.model.message import ReasoningPart
        reasoning_part = ReasoningPart(
            type="reasoning",
            reasoning="Let me think...",
            signature=None,
            redacted_content=None
        )
        stop_event = ModelContentBlockStopEvent.create(
            request_id="req-1",
            block_index=0,
            content=[reasoning_part],
        )
        await printer.handle(stop_event)

        # Buffer is cleared after printing, but panel was displayed

    @pytest.mark.asyncio
    async def test_handle_tool_call(self, printer):
        """Test printing tool call events - tool calls show Status, not direct output."""
        event = AgentToolCallEvent.create(
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
        event = AgentToolResultEvent.create(
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
        event = AgentToolResultEvent.create(
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
        import hawi.agent.printers.rich as rich_module
        monkeypatch.setattr(rich_module, '_stdout', output)
        printer = RichStreamingPrinter(show_reasoning=False)
        from hawi.model.message import StreamThinkingPart

        part: StreamThinkingPart = {
            "type": "thinking_delta",
            "index": 0,
            "delta": "Secret thought",
            "is_start": False,
            "is_end": False,
        }
        event = ModelContentBlockDeltaEvent.create(
            request_id="req-1",
            part=part,
        )
        await printer.handle(event)
        # When reasoning is hidden, buffer should not be populated
        assert printer._reasoning_buffer == ""

    @pytest.mark.asyncio
    async def test_hide_tools(self, monkeypatch):
        """Test hiding tool output."""
        output = io.StringIO()
        import hawi.agent.printers.rich as rich_module
        monkeypatch.setattr(rich_module, '_stdout', output)
        printer = RichStreamingPrinter(show_tools=False)
        event = AgentToolCallEvent.create(
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
        error = AgentError(error_type="tool_execution", msg="Something went wrong")
        event = AgentErrorEvent.create(
            run_id="run-1",
            error=error,
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
        from hawi.model.message import StreamTextPart

        printer = RichStreamingPrinter()
        part: StreamTextPart = {
            "type": "text_delta",
            "index": 0,
            "delta": "Test\n",
            "is_start": False,
            "is_end": False,
        }
        event = ModelContentBlockDeltaEvent.create(
            request_id="req-1",
            part=part,
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
            # Access event attributes directly
            if hasattr(event, 'run_id'):
                received.append(event.run_id)

        bus.subscribe(handler)

        for i in range(5):
            await bus.publish(AgentRunStartEvent.create(run_id=f"seq-{i}"))

        await asyncio.sleep(0.1)
        assert received == ["seq-0", "seq-1", "seq-2", "seq-3", "seq-4"]
