"""Unit tests for event system advanced features.

Tests EventBus edge cases, event filtering, and advanced event handling patterns.
"""

import asyncio
import pytest
from typing import List

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
    ModelProfileEvent,
    # Agent events
    AgentRunStartEvent,
    AgentRunStopEvent,
    AgentToolCallEvent,
    AgentToolResultEvent,
    AgentMessageAddedEvent,
    AgentErrorEvent,
)
from hawi.agent.printers import RichPrinter as ConversationPrinter
from hawi.errors import AgentError


class TestEventBusAdvanced:
    """Advanced tests for EventBus."""

    @pytest.mark.asyncio
    async def test_concurrent_publish(self):
        """Test publishing events concurrently."""
        bus = EventBus()
        received: List[str] = []

        def handler(event: Event) -> None:
            if isinstance(event, AgentRunStartEvent):
                received.append(event.run_id)

        bus.subscribe(handler)

        # Publish multiple events concurrently (sync publish)
        for i in range(10):
            bus.publish(AgentRunStartEvent.create(run_id=f"event-{i}"))

        await asyncio.sleep(0.1)
        assert len(received) == 10
        # All events should be received
        assert all(f"event-{i}" in received for i in range(10))

    @pytest.mark.asyncio
    async def test_handler_exception_isolated(self):
        """Test that exception in one handler doesn't affect others."""
        bus = EventBus()

        handler1_calls = []
        handler2_calls = []

        def failing_handler(event: Event) -> None:
            handler1_calls.append(event.type)
            raise ValueError("Handler error")

        def good_handler(event: Event) -> None:
            handler2_calls.append(event.type)

        bus.subscribe(failing_handler)
        bus.subscribe(good_handler)

        bus.publish(AgentRunStartEvent.create(run_id="test"))
        await asyncio.sleep(0.1)

        # Both handlers should have been called
        assert len(handler1_calls) == 1
        assert len(handler2_calls) == 1


class TestEventLifecycle:
    """Tests for event lifecycle scenarios."""

    def test_event_cannot_be_instantiated_directly(self):
        """Test that Event base class cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract base class"):
            Event(type="test.event", source="agent")

    def test_event_subclass_can_be_created(self):
        """Test that Event subclasses can be created via create()."""
        event = ModelStreamStartEvent.create(request_id="req-123")
        assert event.type == "model.stream_start"
        assert event.source == "model"

    def test_event_immutability(self):
        """Test that events are immutable (Pydantic frozen model)."""
        event = ModelStreamStartEvent.create(request_id="req-123")

        # Should not be able to modify frozen fields
        with pytest.raises(Exception):
            event.timestamp = 0.0

        with pytest.raises(Exception):
            event.source = "agent"

    def test_event_equality(self):
        """Test event equality based on attributes."""
        # Events with same attributes (explicit timestamp)
        event1 = AgentRunStartEvent.create(run_id="test")
        event2 = AgentRunStartEvent.create(run_id="test")

        # Events are not equal because they have different timestamps
        assert event1 != event2

        # Create events with same all fields
        event3 = AgentRunStartEvent(
            type="agent.run_start",
            source="agent",
            timestamp=1.0,
            run_id="test",
        )
        event4 = AgentRunStartEvent(
            type="agent.run_start",
            source="agent",
            timestamp=1.0,
            run_id="test",
        )
        assert event3 == event4


class TestModelEventClasses:
    """Tests for model event classes."""

    def test_model_stream_start_event(self):
        """Test ModelStreamStartEvent.create()."""
        event = ModelStreamStartEvent.create(
            request_id="req-123",
        )
        assert event.type == "model.stream_start"
        assert event.source == "model"
        assert event.request_id == "req-123"

    def test_model_stream_stop_event(self):
        """Test ModelStreamStopEvent.create()."""
        event = ModelStreamStopEvent.create(
            request_id="req-123",
            stop_reason="tool_calls",
        )
        assert event.type == "model.stream_stop"
        assert event.stop_reason == "tool_calls"

    def test_model_content_block_start_event(self):
        """Test ModelContentBlockStartEvent.create()."""
        event = ModelContentBlockStartEvent.create(
            request_id="req-123",
            block_index=0,
            block_type="text",
        )
        assert event.type == "model.content_block_start"
        assert event.block_index == 0
        assert event.block_type == "text"

    def test_model_content_block_delta_event(self):
        """Test ModelContentBlockDeltaEvent.create()."""
        from hawi.models.message import DeltaTextPart

        part: DeltaTextPart = {
            "type": "text_delta",
            "index": 0,
            "delta": "Hello",
            "is_start": False,
            "is_end": False,
        }
        event = ModelContentBlockDeltaEvent.create(
            request_id="req-123",
            part=part,
        )
        assert event.type == "model.content_block_delta"
        assert event.delta == "Hello"
        assert event.delta_type == "text"
        assert event.block_index == 0

    def test_model_content_block_stop_event(self):
        """Test ModelContentBlockStopEvent.create()."""
        from hawi.models.message import TextPart

        text_part = TextPart(type="text", text="Hello World")
        event = ModelContentBlockStopEvent.create(
            request_id="req-123",
            block_index=0,
            content=[text_part],
        )
        assert event.type == "model.content_block_stop"
        assert event.content[0].get("text") == "Hello World"
        assert event.block_type == "text"

    def test_model_metadata_event(self):
        """Test ModelMetadataEvent.create()."""
        from hawi.models.message import TokenUsage
        usage: TokenUsage = {"input_tokens": 10, "output_tokens": 20, "cache_write_tokens": None, "cache_read_tokens": None}
        event = ModelMetadataEvent.create(
            request_id="req-123",
            usage=usage,
            latency_ms=500.0,
            ttft_ms=120.0,
            prefill_tokens_per_second=83.3,
            decode_tokens_per_second=52.6,
        )
        assert event.type == "model.metadata"
        assert event.usage is not None
        assert event.usage["output_tokens"] == 20
        assert event.latency_ms == 500.0
        assert event.ttft_ms == 120.0
        assert event.prefill_tokens_per_second == 83.3
        assert event.decode_tokens_per_second == 52.6

    def test_model_profile_event(self):
        """Test ModelProfileEvent.create()."""
        event = ModelProfileEvent.create(
            request_id="req-123",
            cache_tokens=10,
            prefill_ms=246.0,
            prefill_tokens=20,
            decode_tokens=5,
            decode_tokens_per_second=42.5,
        )
        assert event.type == "model.profile"
        assert event.source == "model"
        assert event.request_id == "req-123"
        assert event.cache_tokens == 10
        assert event.prefill_ms == 246.0
        assert event.prefill_tokens == 20
        assert event.decode_tokens == 5
        assert event.decode_tokens_per_second == 42.5


class TestAgentEventClasses:
    """Tests for agent event classes."""

    def test_agent_run_start_event(self):
        """Test AgentRunStartEvent.create()."""
        event = AgentRunStartEvent.create(
            run_id="run-123",
        )
        assert event.type == "agent.run_start"
        assert event.source == "agent"
        assert event.run_id == "run-123"

    def test_agent_run_stop_event(self):
        """Test AgentRunStopEvent.create()."""
        from hawi.models.message import TokenUsage
        usage: TokenUsage = {"input_tokens": 10, "output_tokens": 20, "cache_write_tokens": None, "cache_read_tokens": None}
        event = AgentRunStopEvent.create(
            run_id="run-123",
            stop_reason="max_iterations",
            duration_ms=1234.5,
            usage=usage
        )
        assert event.type == "agent.run_stop"
        assert event.stop_reason == "max_iterations"
        assert event.duration_ms == 1234.5

    def test_agent_tool_call_event(self):
        """Test AgentToolCallEvent.create()."""
        event = AgentToolCallEvent.create(
            run_id="run-123",
            tool_name="calculator",
            arguments={"expression": "1+1"},
            tool_call_id="tc-789"
        )
        assert event.type == "agent.tool_call"
        assert event.tool_name == "calculator"
        assert event.arguments["expression"] == "1+1"
        assert event.tool_call_id == "tc-789"

    def test_agent_tool_result_event(self):
        """Test AgentToolResultEvent.create()."""
        event = AgentToolResultEvent.create(
            run_id="run-123",
            tool_call_id="tc-789",
            success=True,
            result_preview="2",
            duration_ms=50.5,
        )
        assert event.type == "agent.tool_result"
        assert event.success is True
        assert event.result_preview == "2"
        assert event.duration_ms == 50.5

    def test_agent_tool_result_event_failure(self):
        """Test AgentToolResultEvent.create() for failure case."""
        event = AgentToolResultEvent.create(
            run_id="run-123",
            tool_call_id="tc-789",
            success=False,
            result_preview="Error: Division by zero",
            duration_ms=10.0,
        )
        assert event.type == "agent.tool_result"
        assert event.success is False
        assert event.result_preview == "Error: Division by zero"

    def test_agent_message_added_event(self):
        """Test AgentMessageAddedEvent.create()."""
        event = AgentMessageAddedEvent.create(
            run_id="run-123",
            role="assistant",
            content=[{"type": "text", "text": "Hello world"}],
        )
        assert event.type == "agent.message_added"
        assert event.role == "assistant"
        assert event.content == [{"type": "text", "text": "Hello world"}]

    def test_agent_error_event(self):
        """Test AgentErrorEvent.create()."""
        error = AgentError(error_type="tool_execution", msg="API timeout")
        event = AgentErrorEvent.create(
            run_id="run-123",
            error=error
        )
        assert event.type == "agent.error"
        assert event.error.error_type == "tool_execution"
        assert event.error.message == "API timeout"


class TestConversationPrinterAdvanced:
    """Advanced tests for ConversationPrinter."""

    @pytest.mark.asyncio
    async def test_printer_reasoning_visibility(self):
        """Test reasoning visibility toggle."""
        from rich.console import Console
        import io
        output = io.StringIO()
        console = Console(file=output, force_terminal=True, width=80)

        # With reasoning shown (default)
        printer_with = ConversationPrinter(show_reasoning=True, console=console)

        from hawi.models.message import ReasoningPart

        # handle is now sync, no await needed
        await printer_with.handle(ModelContentBlockStartEvent.create(
            request_id="r1", block_index=0, block_type="reasoning"
        ))
        await printer_with.handle(ModelContentBlockDeltaEvent.create(
            request_id="r1", part={
                "type": "reasoning_delta",
                "index": 0,
                "delta": "Thinking...",
                "is_start": False,
                "is_end": False,
            }
        ))

        # Verify reasoning was captured in buffer before stop event
        assert "Thinking..." in printer_with._buffer

        # Reasoning is buffered and displayed on stop event
        reasoning_part = ReasoningPart(
            type="reasoning",
            reasoning="Thinking...",
            signature=None,
            redacted_content=None
        )
        await printer_with.handle(ModelContentBlockStopEvent.create(
            request_id="r1", block_index=0, content=[reasoning_part]
        ))

        # Buffer is cleared after printing

    @pytest.mark.asyncio
    async def test_printer_reasoning_hidden(self):
        """Test reasoning hidden."""
        from rich.console import Console
        import io
        output = io.StringIO()
        console = Console(file=output, force_terminal=True, width=80)

        # With reasoning hidden
        printer_without = ConversationPrinter(show_reasoning=False, console=console)

        await printer_without.handle(ModelContentBlockDeltaEvent.create(
            request_id="r1", part={
                "type": "reasoning_delta",
                "index": 0,
                "delta": "Secret thought",
                "is_start": False,
                "is_end": False,
            }
        ))

        # Verify reasoning buffer is empty when hidden
        assert printer_without._buffer == ""

    @pytest.mark.asyncio
    async def test_printer_tool_visibility(self):
        """Test tool visibility toggle."""
        from rich.console import Console
        import io
        output = io.StringIO()
        console = Console(file=output, force_terminal=True, width=80)

        printer_hidden = ConversationPrinter(show_tools=False, console=console)

        await printer_hidden.handle(AgentToolCallEvent.create(
            run_id="r1", tool_name="test", arguments={}, tool_call_id="tc1"
        ))

        # Verify no active tool calls when hidden
        assert len(printer_hidden._active_tool_calls) == 0

    @pytest.mark.asyncio
    async def test_printer_stream_lifecycle(self):
        """Test complete stream lifecycle handling."""
        from rich.console import Console
        import io
        output = io.StringIO()
        console = Console(file=output, force_terminal=True, width=80)
        printer = ConversationPrinter(console=console)

        # Stream start (handle is now sync)
        await printer.handle(ModelStreamStartEvent.create(request_id="r1"))

        # Content block start
        await printer.handle(ModelContentBlockStartEvent.create(
            request_id="r1", block_index=0, block_type="text"
        ))

        # Deltas
        await printer.handle(ModelContentBlockDeltaEvent.create(
            request_id="r1", part={
                "type": "text_delta",
                "index": 0,
                "delta": "Hello",
                "is_start": False,
                "is_end": False,
            }
        ))
        await printer.handle(ModelContentBlockDeltaEvent.create(
            request_id="r1", part={
                "type": "text_delta",
                "index": 0,
                "delta": " World",
                "is_start": False,
                "is_end": False,
            }
        ))

        # Content block stop
        from hawi.models.message import TextPart
        text_part = TextPart(type="text", text="Hello World")
        await printer.handle(ModelContentBlockStopEvent.create(
            request_id="r1", block_index=0, content=[text_part]
        ))

        # Stream stop
        await printer.handle(ModelStreamStopEvent.create(
            request_id="r1", stop_reason="end_turn"
        ))

        # Verify text was written to stdout
        result = output.getvalue()
        assert "Hello" in result
        assert "World" in result
