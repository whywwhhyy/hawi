"""Unit tests for event system advanced features.

Tests EventBus edge cases, event filtering, and advanced event handling patterns.
"""

import asyncio
import pytest
from typing import List

from hawi.agent.events import (
    Event,
    EventBus,
    ConversationPrinter,
    # Model events
    model_stream_start_event,
    model_stream_stop_event,
    model_content_block_start_event,
    model_content_block_delta_event,
    model_content_block_stop_event,
    model_metadata_event,
    # Agent events
    agent_run_start_event,
    agent_run_stop_event,
    agent_tool_call_event,
    agent_tool_result_event,
    agent_message_added_event,
    agent_error_event,
)


class TestEventBusAdvanced:
    """Advanced tests for EventBus."""

    @pytest.mark.asyncio
    async def test_concurrent_publish(self):
        """Test publishing events concurrently."""
        bus = EventBus()
        received: List[str] = []
        lock = asyncio.Lock()

        async def handler(event: Event) -> None:
            async with lock:
                received.append(event.metadata["seq"])

        bus.subscribe(handler)

        # Publish multiple events concurrently
        await asyncio.gather(*[
            bus.publish(Event(type="test", source="agent", metadata={"seq": f"event-{i}"}))
            for i in range(10)
        ])

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

        async def failing_handler(event: Event) -> None:
            handler1_calls.append(event.type)
            raise ValueError("Handler error")

        async def good_handler(event: Event) -> None:
            handler2_calls.append(event.type)

        bus.subscribe(failing_handler)
        bus.subscribe(good_handler)

        await bus.publish(agent_run_start_event(run_id="test"))
        await asyncio.sleep(0.1)

        # Both handlers should have been called
        assert len(handler1_calls) == 1
        assert len(handler2_calls) == 1


class TestEventLifecycle:
    """Tests for event lifecycle scenarios."""

    def test_event_immutability(self):
        """Test that events are immutable (frozen dataclass)."""
        event = Event(
            type="test.event",
            source="agent",
            metadata={"key": "value"}
        )

        # Should not be able to modify frozen fields
        with pytest.raises(AttributeError):
            event.type = "modified"

        with pytest.raises(AttributeError):
            event.source = "model"

        # Note: metadata dict itself is not frozen (only the field reference)
        # The frozen=True prevents reassigning event.metadata, but modifying
        # the dict contents is still possible since dict is mutable
        event.metadata["new_key"] = "new_value"  # This works
        assert event.metadata["new_key"] == "new_value"

    def test_event_equality(self):
        """Test event equality based on attributes."""
        # Events with same attributes
        event1 = Event(type="test", source="agent", timestamp=1.0, metadata={})
        event2 = Event(type="test", source="agent", timestamp=1.0, metadata={})

        assert event1 == event2

        # Events with different attributes
        event3 = Event(type="test", source="agent", timestamp=2.0, metadata={})
        assert event1 != event3


class TestModelEventFactories:
    """Tests for model event factory functions."""

    def test_model_stream_start_event(self):
        """Test model_stream_start_event factory."""
        event = model_stream_start_event(
            request_id="req-123",
            model_id="gpt-4",
            extra="value"
        )
        assert event.type == "model.stream_start"
        assert event.source == "model"
        assert event.metadata["request_id"] == "req-123"
        assert event.metadata["model_id"] == "gpt-4"
        assert event.metadata["extra"] == "value"

    def test_model_stream_stop_event(self):
        """Test model_stream_stop_event factory."""
        event = model_stream_stop_event(
            request_id="req-123",
            stop_reason="tool_calls",
            usage={"prompt_tokens": 10, "completion_tokens": 20}
        )
        assert event.type == "model.stream_stop"
        assert event.metadata["stop_reason"] == "tool_calls"
        assert event.metadata["usage"]["prompt_tokens"] == 10

    def test_model_content_block_start_event(self):
        """Test model_content_block_start_event factory."""
        event = model_content_block_start_event(
            request_id="req-123",
            block_index=0,
            block_type="text"
        )
        assert event.type == "model.content_block_start"
        assert event.metadata["block_index"] == 0
        assert event.metadata["block_type"] == "text"

    def test_model_content_block_delta_event(self):
        """Test model_content_block_delta_event factory."""
        event = model_content_block_delta_event(
            request_id="req-123",
            block_index=0,
            delta_type="text",
            delta="Hello",
            is_complete=False
        )
        assert event.type == "model.content_block_delta"
        assert event.metadata["delta"] == "Hello"
        assert event.metadata["delta_type"] == "text"

    def test_model_content_block_stop_event(self):
        """Test model_content_block_stop_event factory."""
        event = model_content_block_stop_event(
            request_id="req-123",
            block_index=0,
            full_content="Hello World"
        )
        assert event.type == "model.content_block_stop"
        assert event.metadata["full_content"] == "Hello World"

    def test_model_metadata_event(self):
        """Test model_metadata_event factory."""
        event = model_metadata_event(
            request_id="req-123",
            usage={"total_tokens": 100},
            latency_ms=500.0
        )
        assert event.type == "model.metadata"
        assert event.metadata["usage"]["total_tokens"] == 100
        assert event.metadata["latency_ms"] == 500.0


class TestAgentEventFactories:
    """Tests for agent event factory functions."""

    def test_agent_run_start_event(self):
        """Test agent_run_start_event factory."""
        event = agent_run_start_event(
            run_id="run-123",
            message_preview="Hello, world!",
            user_id="user-456"
        )
        assert event.type == "agent.run_start"
        assert event.source == "agent"
        assert event.metadata["run_id"] == "run-123"
        assert event.metadata["message_preview"] == "Hello, world!"
        assert event.metadata["user_id"] == "user-456"

    def test_agent_run_stop_event(self):
        """Test agent_run_stop_event factory."""
        event = agent_run_stop_event(
            run_id="run-123",
            stop_reason="max_iterations",
            duration_ms=1234.5,
            iteration_count=5
        )
        assert event.type == "agent.run_stop"
        assert event.metadata["stop_reason"] == "max_iterations"
        assert event.metadata["duration_ms"] == 1234.5
        assert event.metadata["iteration_count"] == 5

    def test_agent_tool_call_event(self):
        """Test agent_tool_call_event factory."""
        event = agent_tool_call_event(
            run_id="run-123",
            tool_name="calculator",
            arguments={"expression": "1+1"},
            tool_call_id="tc-789",
            pending_audit=True
        )
        assert event.type == "agent.tool_call"
        assert event.metadata["tool_name"] == "calculator"
        assert event.metadata["arguments"]["expression"] == "1+1"
        assert event.metadata["tool_call_id"] == "tc-789"
        assert event.metadata["pending_audit"] is True

    def test_agent_tool_result_event(self):
        """Test agent_tool_result_event factory."""
        event = agent_tool_result_event(
            run_id="run-123",
            tool_name="calculator",
            tool_call_id="tc-789",
            success=True,
            result_preview="2",
            duration_ms=50.5,
            arguments={"expression": "1+1"}
        )
        assert event.type == "agent.tool_result"
        assert event.metadata["success"] is True
        assert event.metadata["result_preview"] == "2"
        assert event.metadata["duration_ms"] == 50.5

    def test_agent_tool_result_event_failure(self):
        """Test agent_tool_result_event for failure case."""
        event = agent_tool_result_event(
            run_id="run-123",
            tool_name="calculator",
            tool_call_id="tc-789",
            success=False,
            result_preview="Error: Division by zero",
            duration_ms=10.0,
            arguments={"expression": "1/0"},
            error_message="Division by zero"
        )
        assert event.type == "agent.tool_result"
        assert event.metadata["success"] is False
        assert event.metadata["error_message"] == "Division by zero"

    def test_agent_message_added_event(self):
        """Test agent_message_added_event factory."""
        event = agent_message_added_event(
            run_id="run-123",
            role="assistant",
            message_preview="Hello world",
            message_index=5,
            has_tool_calls=True
        )
        assert event.type == "agent.message_added"
        assert event.metadata["role"] == "assistant"
        assert event.metadata["message_preview"] == "Hello world"
        assert event.metadata["message_index"] == 5
        assert event.metadata["has_tool_calls"] is True

    def test_agent_error_event(self):
        """Test agent_error_event factory."""
        event = agent_error_event(
            run_id="run-123",
            error_type="model_error",
            error_message="API timeout",
            recoverable=True,
            retry_count=2
        )
        assert event.type == "agent.error"
        assert event.metadata["error_type"] == "model_error"
        assert event.metadata["error_message"] == "API timeout"
        assert event.metadata["recoverable"] is True
        assert event.metadata["retry_count"] == 2


class TestConversationPrinterAdvanced:
    """Advanced tests for ConversationPrinter."""

    @pytest.mark.asyncio
    async def test_printer_reasoning_visibility(self, monkeypatch):
        """Test reasoning visibility toggle."""
        import io
        import hawi.agent.events as events_module
        output = io.StringIO()
        monkeypatch.setattr(events_module, '_stdout', output)

        # With reasoning shown (default)
        printer_with = ConversationPrinter(show_reasoning=True)

        await printer_with.handle(model_content_block_start_event(
            request_id="r1", block_index=0, block_type="thinking"
        ))
        await printer_with.handle(model_content_block_delta_event(
            request_id="r1", block_index=0, delta_type="thinking", delta="Thinking..."
        ))

        # Verify reasoning was captured in buffer before stop event
        assert "Thinking..." in printer_with._reasoning_buffer

        # Reasoning is buffered and displayed on stop event
        await printer_with.handle(model_content_block_stop_event(
            request_id="r1", block_index=0, block_type="thinking", full_content="Thinking..."
        ))

        # Buffer is cleared after printing

    @pytest.mark.asyncio
    async def test_printer_reasoning_hidden(self, monkeypatch):
        """Test reasoning hidden."""
        import io
        import hawi.agent.events as events_module
        output = io.StringIO()
        monkeypatch.setattr(events_module, '_stdout', output)

        # With reasoning hidden
        printer_without = ConversationPrinter(show_reasoning=False)

        await printer_without.handle(model_content_block_delta_event(
            request_id="r1", block_index=0, delta_type="reasoning", delta="Secret thought"
        ))

        # Verify reasoning buffer is empty when hidden
        assert printer_without._reasoning_buffer == ""

    @pytest.mark.asyncio
    async def test_printer_tool_visibility(self, monkeypatch):
        """Test tool visibility toggle."""
        import io
        import hawi.agent.events as events_module
        output = io.StringIO()
        monkeypatch.setattr(events_module, '_stdout', output)

        printer_hidden = ConversationPrinter(show_tools=False)

        await printer_hidden.handle(agent_tool_call_event(
            run_id="r1", tool_name="test", arguments={}, tool_call_id="tc1"
        ))

        # Verify no active tool calls when hidden
        assert len(printer_hidden._active_tool_calls) == 0

    @pytest.mark.asyncio
    async def test_printer_stream_lifecycle(self, monkeypatch):
        """Test complete stream lifecycle handling."""
        import io
        import hawi.agent.events as events_module
        output = io.StringIO()
        monkeypatch.setattr(events_module, '_stdout', output)

        printer = ConversationPrinter()

        # Stream start
        await printer.handle(model_stream_start_event(request_id="r1"))

        # Content block start
        await printer.handle(model_content_block_start_event(
            request_id="r1", block_index=0, block_type="text"
        ))

        # Deltas
        await printer.handle(model_content_block_delta_event(
            request_id="r1", block_index=0, delta_type="text", delta="Hello"
        ))
        await printer.handle(model_content_block_delta_event(
            request_id="r1", block_index=0, delta_type="text", delta=" World"
        ))

        # Content block stop
        await printer.handle(model_content_block_stop_event(
            request_id="r1", block_index=0, full_content="Hello World"
        ))

        # Stream stop
        await printer.handle(model_stream_stop_event(
            request_id="r1", stop_reason="end_turn"
        ))

        # Verify text was written to stdout
        result = output.getvalue()
        assert "Hello" in result
        assert "World" in result
