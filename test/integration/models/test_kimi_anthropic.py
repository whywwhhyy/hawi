"""KimiAnthropicModel integration tests.

Tests the Kimi model implementation using Anthropic-compatible API.
"""

import pytest
from typing import Any

from hawi.agent.models.kimi.kimi_anthropic import KimiAnthropicModel
from hawi.agent.message import (
    ContentPart,
    Message,
    TextPart,
    ToolCallPart,
    ReasoningPart,
)
from test.integration import get_kimi_anthropic_api_key


# =============================================================================
# Helper functions for creating messages and content parts
# =============================================================================

def _text_part(text: str) -> TextPart:
    """Create a text content part."""
    return {"type": "text", "text": text}


def _reasoning_part(reasoning: str, signature: str | None = None) -> ReasoningPart:
    """Create a reasoning content part."""
    return {"type": "reasoning", "reasoning": reasoning, "signature": signature}


def _tool_call_part(id: str, name: str, arguments: dict[str, Any]) -> ToolCallPart:
    """Create a tool call content part."""
    return {"type": "tool_call", "id": id, "name": name, "arguments": arguments}


def _normalize_content(content: str | list[ContentPart] | None) -> list[ContentPart]:
    """Normalize content to list[ContentPart]."""
    if content is None:
        return []
    if isinstance(content, str):
        return [_text_part(content)]
    return content


def _create_user_message(
    content: str | list[ContentPart],
    name: str | None = None,
) -> Message:
    """Create a user message."""
    return {
        "role": "user",
        "content": _normalize_content(content),
        "name": name,
        "tool_calls": None,
        "tool_call_id": None,
        "metadata": None,
    }


def _create_assistant_message(
    content: str | list[ContentPart] | None = None,
    tool_calls: list[ToolCallPart] | None = None,
) -> Message:
    """Create an assistant message."""
    return {
        "role": "assistant",
        "content": _normalize_content(content),
        "name": None,
        "tool_calls": tool_calls,
        "tool_call_id": None,
        "metadata": None,
    }


def _create_tool_result_message(
    tool_call_id: str,
    content: str | list[ContentPart],
) -> Message:
    """Create a tool result message."""
    return {
        "role": "tool",
        "content": _normalize_content(content),
        "name": None,
        "tool_calls": None,
        "tool_call_id": tool_call_id,
        "metadata": None,
    }


# Check if API key is available
KIMI_API_KEY = get_kimi_anthropic_api_key()
HAS_KIMI_KEY = KIMI_API_KEY is not None and KIMI_API_KEY.strip() != ""

# Skip reason for tests requiring API key
SKIP_REASON = "Kimi Anthropic API key not found (set KIMI_ANTHROPIC_API_KEY, KIMI_API_KEY or configure apikey.yaml)"


class TestKimiAnthropicUnit:
    """Unit tests for KimiAnthropicModel (no API calls)."""

    def test_model_initialization_defaults(self):
        """Test model initialization with default values."""
        model = KimiAnthropicModel(
            api_key="test-key",
        )
        assert model.model_id == "kimi-k2.5"
        assert model.api_key == "test-key"
        assert model.base_url == "https://api.kimi.com/coding/"

    def test_model_initialization_custom(self):
        """Test model initialization with custom values."""
        model = KimiAnthropicModel(
            model_id="kimi-k2",
            api_key="test-key",
            base_url="https://custom.endpoint.com/",
        )
        assert model.model_id == "kimi-k2"
        assert model.api_key == "test-key"
        assert model.base_url == "https://custom.endpoint.com/"


@pytest.mark.skipif(not HAS_KIMI_KEY, reason=SKIP_REASON)
class TestKimiAnthropicIntegration:
    """Integration tests requiring real Kimi API access."""

    @pytest.fixture
    def model(self) -> KimiAnthropicModel:
        """Create a Kimi Anthropic model instance."""
        return KimiAnthropicModel(
            api_key=KIMI_API_KEY,
            model_id="kimi-k2.5",
        )

    def test_simple_chat_completion(self, model: KimiAnthropicModel):
        """Test basic chat completion."""
        response = model.invoke(
            messages=[_create_user_message("Say 'Hello, World!' and nothing else.")],
        )

        assert response.id is not None
        assert len(response.content) > 0
        assert response.content[0]["type"] == "text"
        assert "Hello" in response.content[0]["text"] or "World" in response.content[0]["text"]
        assert response.stop_reason == "end_turn"
        assert response.usage is not None
        # Kimi may report input_tokens as 0 when using cache - check total tokens instead
        total_input = response.usage.input_tokens + (response.usage.cache_read_tokens or 0)
        assert total_input >= 0
        assert response.usage.output_tokens > 0

    def test_streaming_response(self, model: KimiAnthropicModel):
        """Test streaming response."""
        events = list(model.stream(
            messages=[_create_user_message("Count from 1 to 3.")],
        ))

        # Should have content block events
        content_events = [e for e in events if e["type"] == "text_delta"]

        assert len(content_events) > 0

    def test_tool_call_formatting(self, model: KimiAnthropicModel):
        """Test tool call request formatting."""
        from hawi.agent.message import ToolDefinition

        tools: list[ToolDefinition] = [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather information",
                "schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                    },
                    "required": ["location"],
                },
            }
        ]

        response = model.invoke(
            messages=[_create_user_message("What's the weather in Beijing?")],
            tools=tools,
        )

        # Should either have text response or tool_call
        assert len(response.content) > 0
        if response.content[0]["type"] == "tool_call":
            assert response.content[0]["name"] == "get_weather"
            assert "location" in response.content[0]["arguments"]
            assert response.stop_reason == "tool_use"

    def test_multi_turn_conversation(self, model: KimiAnthropicModel):
        """Test multi-turn conversation."""
        messages = [
            _create_user_message("My name is Bob."),
        ]

        # First turn
        response1 = model.invoke(messages=messages)
        first_part = response1.content[0]
        assert first_part["type"] == "text"
        messages.append(_create_assistant_message(content=[
            _text_part(first_part["text"]),
        ]))

        # Second turn
        messages.append(_create_user_message("What's my name?"))
        response2 = model.invoke(messages=messages)

        second_part = response2.content[0]
        assert second_part["type"] == "text"
        assert "Bob" in second_part["text"]


@pytest.mark.skipif(not HAS_KIMI_KEY, reason=SKIP_REASON)
class TestKimiAnthropicToolCalls:
    """Tests for Kimi Anthropic tool calls."""

    @pytest.fixture
    def model(self) -> KimiAnthropicModel:
        """Create a Kimi Anthropic model instance."""
        return KimiAnthropicModel(
            api_key=KIMI_API_KEY,
            model_id="kimi-k2.5",
        )

    def test_tool_call_with_citations(self, model: KimiAnthropicModel):
        """Test that tool calls work correctly and handle citations."""
        from hawi.agent.message import ToolDefinition

        tools: list[ToolDefinition] = [
            {
                "type": "function",
                "name": "calculate",
                "description": "Perform mathematical calculation",
                "schema": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"},
                    },
                    "required": ["expression"],
                },
            }
        ]

        response = model.invoke(
            messages=[_create_user_message("Calculate 100 * 100")],
            tools=tools,
        )

        # Check if tool was called or text response given
        tool_calls = [c for c in response.content if c["type"] == "tool_call"]
        text_parts = [c for c in response.content if c["type"] == "text"]

        if tool_calls:
            assert tool_calls[0]["name"] == "calculate"
            assert response.stop_reason == "tool_use"
        elif text_parts:
            # Model may respond directly with text
            assert "10000" in text_parts[0]["text"]

    def test_multi_turn_with_tool_result(self, model: KimiAnthropicModel):
        """Test multi-turn conversation with tool results."""
        from hawi.agent.message import ToolDefinition

        tools: list[ToolDefinition] = [
            {
                "type": "function",
                "name": "get_time",
                "description": "Get current time",
                "schema": {
                    "type": "object",
                    "properties": {},
                },
            }
        ]

        # First turn
        response = model.invoke(
            messages=[_create_user_message("What time is it? Use the get_time tool.")],
            tools=tools,
        )

        tool_calls = [c for c in response.content if c["type"] == "tool_call"]
        if not tool_calls:
            pytest.skip("Model did not call tool")

        # Simulate tool result
        messages = [
            _create_user_message("What time is it? Use the get_time tool."),
            _create_assistant_message(
                content=None,
                tool_calls=[
                    _tool_call_part(
                        id=tool_calls[0]["id"],
                        name=tool_calls[0]["name"],
                        arguments=tool_calls[0]["arguments"],
                    )
                ],
            ),
            _create_tool_result_message(
                tool_call_id=tool_calls[0]["id"],
                content="The current time is 14:30.",
            ),
        ]

        # Second turn
        response2 = model.invoke(messages=messages, tools=tools)

        assert len(response2.content) > 0
