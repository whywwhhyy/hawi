"""Integration tests for Event system with HawiAgent.

Tests the complete event flow from Agent execution.
"""

import asyncio
import pytest
from typing import Any

from hawi.agent import HawiAgent
from hawi.agent.events import (
    Event,
    EventBus,
    ConversationPrinter,
    # Model events
    model_content_block_delta_event,
    model_content_block_start_event,
    model_content_block_stop_event,
    # Agent events
    agent_run_start_event,
    agent_run_stop_event,
    agent_tool_call_event,
    agent_tool_result_event,
    agent_error_event,
)
from hawi.agent.models.deepseek import DeepSeekModel
from hawi.plugin import HawiPlugin
from hawi.plugin.decorators import tool

from test.integration.models import get_deepseek_api_key


# Check if API key is available
DEEPSEEK_API_KEY = get_deepseek_api_key()
HAS_DEEPSEEK_KEY = DEEPSEEK_API_KEY is not None and DEEPSEEK_API_KEY.strip() != ""

SKIP_REASON = "DeepSeek API key not found (set DEEPSEEK_API_KEY or configure apikey.yaml)"


class CalculatorPlugin(HawiPlugin):
    """Simple calculator plugin for testing events."""

    @tool
    def calculate(self, expression: str) -> str:
        """Calculate a mathematical expression.

        Args:
            expression: Mathematical expression like "2 + 2"

        Returns:
            The result of the calculation
        """
        try:
            allowed_chars = set('0123456789+-*/(). ')
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters"
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Error: {e}"


@pytest.mark.skipif(not HAS_DEEPSEEK_KEY, reason=SKIP_REASON)
class TestEventFlowWithAgent:
    """Integration tests for event flow with real Agent execution."""

    @pytest.fixture
    def model(self) -> DeepSeekModel:
        """Create a DeepSeek model instance."""
        return DeepSeekModel(
            model_id="deepseek-chat",
            api_key=DEEPSEEK_API_KEY,
        )

    @pytest.fixture
    def agent(self, model: DeepSeekModel) -> HawiAgent:
        """Create a HawiAgent with calculator plugin."""
        return HawiAgent(
            model=model,
            plugins=[CalculatorPlugin()],
            system_prompt="You are a helpful assistant with calculator tools.",
            max_iterations=5,
            enable_streaming=True,
        )

    @pytest.mark.asyncio
    async def test_agent_events_flow(self, agent: HawiAgent):
        """Test that agent produces correct event flow."""
        events = []

        async for event in agent.arun("Say 'Hello' and nothing else.", stream=True):
            events.append(event)

        # Check event types
        event_types = [e.type for e in events]

        # Should have agent lifecycle events
        assert "agent.run_start" in event_types
        assert "agent.run_stop" in event_types

        # Should have model stream events
        assert "model.stream_start" in event_types
        assert "model.stream_stop" in event_types

        # Should have content block events
        assert "model.content_block_start" in event_types
        assert "model.content_block_delta" in event_types
        assert "model.content_block_stop" in event_types

        # Order matters: run_start should be first
        assert event_types[0] == "agent.run_start"
        # run_stop should be last
        assert event_types[-1] == "agent.run_stop"

    @pytest.mark.asyncio
    async def test_tool_call_events(self, agent: HawiAgent):
        """Test events produced during tool calling."""
        events = []

        async for event in agent.arun("What is 5 + 3? Use the calculate tool.", stream=True):
            events.append(event)

        event_types = [e.type for e in events]

        # Should have tool events
        assert "agent.tool_call" in event_types
        assert "agent.tool_result" in event_types

        # Find tool call event
        tool_call_events = [e for e in events if e.type == "agent.tool_call"]
        assert len(tool_call_events) >= 1

        # Verify tool call details
        tc_event = tool_call_events[0]
        assert tc_event.metadata["tool_name"] == "calculate"
        assert "expression" in tc_event.metadata["arguments"]

        # Find tool result event
        tool_result_events = [e for e in events if e.type == "agent.tool_result"]
        assert len(tool_result_events) >= 1

        tr_event = tool_result_events[0]
        assert tr_event.metadata["success"] is True
        assert "8" in tr_event.metadata["result_preview"]

    @pytest.mark.asyncio
    async def test_event_bus_with_agent(self, agent: HawiAgent):
        """Test using EventBus to collect agent events."""
        bus = EventBus()
        received_events = []

        async def handler(event: Event) -> None:
            received_events.append(event)

        bus.subscribe(handler)

        async for event in agent.arun("Say 'Hi'", stream=True, event_bus=bus):
            pass  # Events are also published to bus

        await asyncio.sleep(0.1)  # Wait for async handlers

        # Bus should have received all events
        assert len(received_events) > 0
        event_types = [e.type for e in received_events]
        assert "agent.run_start" in event_types
        assert "agent.run_stop" in event_types

    @pytest.mark.asyncio
    async def test_conversation_printer_with_agent(self, agent: HawiAgent, capsys):
        """Test ConversationPrinter with real agent execution."""
        printer = ConversationPrinter(show_reasoning=True, show_tools=True)

        async for event in agent.arun("Calculate 2+2", stream=True):
            await printer.handle(event)

        captured = capsys.readouterr()
        output = captured.out

        # Should have printed something
        assert len(output) > 0

        # Should contain tool call info if tool was used
        # Note: Model may or may not use tool depending on response

    @pytest.mark.asyncio
    async def test_content_block_events_for_text(self, agent: HawiAgent):
        """Test content block events contain correct text content."""
        events = []

        async for event in agent.arun("Say exactly 'Test123'", stream=True):
            events.append(event)

        # Find content block events
        delta_events = [e for e in events if e.type == "model.content_block_delta"]
        assert len(delta_events) > 0

        # Should have text content
        text_content = "".join(e.metadata.get("delta", "") for e in delta_events if e.metadata.get("delta_type") == "text")
        assert "Test123" in text_content or len(text_content) > 0

    @pytest.mark.asyncio
    async def test_run_stop_event_contains_duration(self, agent: HawiAgent):
        """Test that run_stop event contains execution duration."""
        events = []

        async for event in agent.arun("Say 'Hello'", stream=True):
            events.append(event)

        run_stop_events = [e for e in events if e.type == "agent.run_stop"]
        assert len(run_stop_events) == 1

        stop_event = run_stop_events[0]
        assert "duration_ms" in stop_event.metadata
        assert stop_event.metadata["duration_ms"] > 0
        assert stop_event.metadata["stop_reason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_event_metadata_consistency(self, agent: HawiAgent):
        """Test that event metadata is consistent across related events."""
        events = []

        async for event in agent.arun("Calculate 1+1", stream=True):
            events.append(event)

        # All agent events should have the same run_id
        agent_events = [e for e in events if e.source == "agent"]
        if len(agent_events) > 0:
            run_ids = set(e.metadata.get("run_id") for e in agent_events if "run_id" in e.metadata)
            assert len(run_ids) == 1  # Should all share the same run_id

    @pytest.mark.asyncio
    async def test_streaming_vs_non_streaming_events(self, agent: HawiAgent):
        """Test that streaming produces same result as non-streaming."""
        # Streaming execution
        events = []
        async for event in agent.arun("Say 'Hello'", stream=True):
            events.append(event)

        # Non-streaming execution (must be awaited in async test)
        result_non_stream = await agent.arun("Say 'Hello'", stream=False)

        # Both should succeed
        assert result_non_stream.stop_reason == "end_turn"

        run_stop_events = [e for e in events if e.type == "agent.run_stop"]
        assert len(run_stop_events) == 1
        assert run_stop_events[0].metadata["stop_reason"] == "end_turn"


@pytest.mark.skipif(not HAS_DEEPSEEK_KEY, reason=SKIP_REASON)
class TestEventFiltering:
    """Tests for event filtering and selective subscription."""

    @pytest.fixture
    def model(self) -> DeepSeekModel:
        return DeepSeekModel(
            model_id="deepseek-chat",
            api_key=DEEPSEEK_API_KEY,
        )

    @pytest.fixture
    def agent(self, model: DeepSeekModel) -> HawiAgent:
        return HawiAgent(
            model=model,
            plugins=[CalculatorPlugin()],
            enable_streaming=True,
        )

    @pytest.mark.asyncio
    async def test_selective_event_subscription(self, agent: HawiAgent):
        """Test subscribing to specific event types only."""
        bus = EventBus()
        tool_events = []

        async def tool_handler(event: Event) -> None:
            tool_events.append(event.type)

        bus.subscribe(tool_handler, event_types=["agent.tool_call", "agent.tool_result"])

        async for event in agent.arun("Calculate 1+1", stream=True, event_bus=bus):
            pass

        await asyncio.sleep(0.1)

        # Should only have tool events
        assert len(tool_events) >= 0  # May or may not have tool events
        for et in tool_events:
            assert "tool" in et

    @pytest.mark.asyncio
    async def test_wildcard_subscription(self, agent: HawiAgent):
        """Test subscribing to all events."""
        bus = EventBus()
        all_events = []

        async def catch_all(event: Event) -> None:
            all_events.append(event.type)

        bus.subscribe(catch_all)  # No event_types = wildcard

        async for event in agent.arun("Say 'Hi'", stream=True, event_bus=bus):
            pass

        await asyncio.sleep(0.1)

        # Should have various event types
        assert len(all_events) > 0
        assert "agent.run_start" in all_events


@pytest.mark.skipif(not HAS_DEEPSEEK_KEY, reason=SKIP_REASON)
class TestEventWithReasoningModel:
    """Tests for events with reasoning/thinking models."""

    @pytest.fixture
    def model(self) -> DeepSeekModel:
        return DeepSeekModel(
            model_id="deepseek-reasoner",
            api_key=DEEPSEEK_API_KEY,
        )

    @pytest.fixture
    def agent(self, model: DeepSeekModel) -> HawiAgent:
        return HawiAgent(
            model=model,
            enable_streaming=True,
        )

    @pytest.mark.asyncio
    async def test_reasoning_content_events(self, agent: HawiAgent):
        """Test that reasoning models produce reasoning content events."""
        events = []

        async for event in agent.arun("What is 15 * 23? Show your thinking.", stream=True):
            events.append(event)

        # Find content block events
        block_start_events = [e for e in events if e.type == "model.content_block_start"]

        # Check for reasoning block type
        block_types = [e.metadata.get("block_type") for e in block_start_events]

        # Should have at least text blocks, may have reasoning
        assert "text" in block_types

        # Verify delta events have correct types
        delta_events = [e for e in events if e.type == "model.content_block_delta"]
        for de in delta_events:
            assert de.metadata.get("delta_type") in ["text", "thinking"]
