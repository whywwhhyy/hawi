"""Integration tests for HawiAgent.

Tests the complete agent workflow with real API calls.
Requires DEEPSEEK_API_KEY environment variable or apikey.yaml configuration.
"""

import pytest
from typing import cast

from hawi.agent import HawiAgent
from hawi.agent.model import Model
from hawi.agent.models.deepseek import DeepSeekModel
from hawi.plugin import HawiPlugin
from hawi.plugin.decorators import tool, before_conversation, after_conversation

from test.integration.models import get_deepseek_api_key


# Check if API key is available
DEEPSEEK_API_KEY = get_deepseek_api_key()
HAS_DEEPSEEK_KEY = DEEPSEEK_API_KEY is not None and DEEPSEEK_API_KEY.strip() != ""

SKIP_REASON = "DeepSeek API key not found (set DEEPSEEK_API_KEY or configure apikey.yaml)"


class CalculatorPlugin(HawiPlugin):
    """Simple calculator plugin for testing."""

    def __init__(self):
        self.events = []

    @before_conversation
    def on_start(self, agent:HawiAgent):
        self.events.append("before_conversation")

    @after_conversation
    def on_end(self, agent):
        self.events.append("after_conversation")

    @tool
    def calculate(self, expression: str) -> str:
        """Calculate a mathematical expression.

        Args:
            expression: Mathematical expression like "2 + 2" or "10 * 5"

        Returns:
            The result of the calculation
        """
        try:
            # Safe evaluation for simple math
            allowed_chars = set('0123456789+-*/(). ')
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression"

            result = eval(expression, {"__builtins__": {}}, {})
            return f"{result}"
        except Exception as e:
            return f"Error: {e}"

    @tool
    def get_current_time(self) -> str:
        """Get the current system time.

        Returns:
            Current time as a string
        """
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@pytest.mark.skipif(not HAS_DEEPSEEK_KEY, reason=SKIP_REASON)
class TestHawiAgentIntegration:
    """Integration tests requiring real DeepSeek API access."""

    @pytest.fixture
    def model(self) -> Model:
        """Create a DeepSeek model instance."""
        return DeepSeekModel(
            model_id="deepseek-chat",
            api_key=DEEPSEEK_API_KEY,
        )

    @pytest.fixture
    def calculator_plugin(self) -> CalculatorPlugin:
        """Create a calculator plugin."""
        return CalculatorPlugin()

    @pytest.fixture
    def agent(self, model: Model, calculator_plugin: CalculatorPlugin) -> HawiAgent:
        """Create a HawiAgent with calculator tools."""
        return HawiAgent(
            model=model,
            plugins=[calculator_plugin],
            system_prompt="You are a helpful assistant with access to calculator tools.",
            max_iterations=5,
            enable_streaming=False,
        )

    def test_simple_conversation(self, agent: HawiAgent, calculator_plugin: CalculatorPlugin):
        """Test basic conversation without tool calls."""
        result = agent.run("Say 'Hello, World!' and nothing else.")

        assert result.stop_reason == "end_turn"
        assert result.error is None
        assert len(result.messages) == 2  # user + assistant
        assert "Hello" in result.text or "World" in result.text

        # Verify hooks were triggered
        assert "before_conversation" in calculator_plugin.events
        assert "after_conversation" in calculator_plugin.events

    def test_tool_calling(self, agent: HawiAgent, calculator_plugin: CalculatorPlugin):
        """Test agent using calculator tool."""
        result = agent.run("What is 15 + 27? Use the calculate tool.")

        assert result.stop_reason == "end_turn"
        assert result.error is None
        assert len(result.tool_calls) >= 1

        # Find the calculator tool call
        calc_calls = [tc for tc in result.tool_calls if tc.tool_name == "CalculatorPlugin__calculate"]
        assert len(calc_calls) >= 1

        # Verify the calculation was performed
        calc_call = calc_calls[0]
        assert "expression" in calc_call.arguments
        assert calc_call.result.success is True
        assert "42" in cast(str, calc_call.result.output)  # 15 + 27 = 42

    def test_multi_turn_conversation(self, agent: HawiAgent):
        """Test multi-turn conversation with context retention."""
        # First turn
        result1 = agent.run("My name is Alice.")
        assert result1.stop_reason == "end_turn"

        # Second turn - agent should remember the name
        result2 = agent.run("What's my name?")
        assert result2.stop_reason == "end_turn"
        assert "Alice" in result2.text

    def test_clone_functionality(self, agent: HawiAgent):
        """Test agent cloning creates independent copy."""
        # Add some context to original agent
        agent.run("The secret number is 42.")

        # Clone the agent
        cloned = agent.clone()

        # Verify clone has same context
        assert len(cloned.context.messages) == len(agent.context.messages)

        # Modify clone
        cloned.run("Actually, the secret number is 99.")

        # Original should not be affected
        original_result = agent.run("What is the secret number?")
        assert "42" in original_result.text

        # Clone should have new value
        cloned_result = cloned.run("What is the secret number?")
        assert "99" in cloned_result.text

    def test_streaming_response(self, model: Model, calculator_plugin: CalculatorPlugin):
        """Test streaming event generation."""
        agent = HawiAgent(
            model=model,
            plugins=[calculator_plugin],
            enable_streaming=True,  # Enable streaming
        )

        events = list(agent.run("Count from 1 to 3.", stream=True))

        # Should have start, message(s), and finish events
        event_types = [e.type for e in events]
        assert "agent.run_start" in event_types
        assert "agent.run_stop" in event_types
        assert "model.content_block_delta" in event_types

    def test_max_iterations_limit(self, model: Model, calculator_plugin: CalculatorPlugin):
        """Test max_iterations stops infinite loops."""
        agent = HawiAgent(
            model=model,
            plugins=[calculator_plugin],
            max_iterations=2,  # Very low limit
            enable_streaming=False,
        )

        # A complex request might trigger multiple tool calls
        result = agent.run(
            "Calculate these: 1+1, 2+2, 3+3, 4+4, 5+5. Use the tool for each.",
        )

        # Should either complete normally or hit max iterations
        assert result.stop_reason in ["end_turn", "error"]
        assert len(result.tool_calls) <= 3  # Should be limited by max_iterations


@pytest.mark.skipif(not HAS_DEEPSEEK_KEY, reason=SKIP_REASON)
class TestHawiAgentAsync:
    """Async integration tests."""

    @pytest.fixture
    def agent(self) -> HawiAgent:
        """Create a HawiAgent."""
        model = DeepSeekModel(
            model_id="deepseek-chat",
            api_key=DEEPSEEK_API_KEY,
        )
        plugin = CalculatorPlugin()
        return HawiAgent(
            model=model,
            plugins=[plugin],
            enable_streaming=False,
        )

    @pytest.mark.asyncio
    async def test_async_run(self, agent: HawiAgent):
        """Test async execution."""
        result = await agent.arun("Say hi!")

        assert result.stop_reason == "end_turn"
        assert result.error is None
        assert len(result.text) > 0

    @pytest.mark.asyncio
    async def test_async_streaming(self, agent: HawiAgent):
        """Test async streaming."""
        events = []
        async for event in agent.arun("Count to 2.", stream=True):
            events.append(event.type)

        assert "agent.run_start" in events
        assert "agent.run_stop" in events
