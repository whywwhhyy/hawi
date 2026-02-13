"""
KimiOpenAIModel integration tests.

Tests the new Kimi model implementation based on hawi.agent.models.openai.
"""

import os
import pytest
from typing import Any

from hawi.agent.models.kimi.kimi_openai import KimiOpenAIModel
from hawi.agent.messages import (
    ContentPart,
    Message,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    ReasoningPart,
)
from test.integration.models import get_kimi_openai_api_key


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
KIMI_API_KEY = get_kimi_openai_api_key()
HAS_KIMI_KEY = KIMI_API_KEY is not None and KIMI_API_KEY.strip() != ""

# Skip reason for tests requiring API key
SKIP_REASON = "Kimi API key not found (set KIMI_API_KEY or configure apikey.yaml)"


class TestKimiOpenAIUnit:
    """Unit tests for KimiOpenAIModel (no API calls)."""

    def test_model_initialization_defaults(self):
        """Test model initialization with default values."""
        model = KimiOpenAIModel(
            api_key="test-key",
        )
        assert model.model_id == "kimi-k2.5"
        assert model.api_key == "test-key"
        assert model.base_url == "https://api.moonshot.cn/v1"
        assert model.enable_thinking is True

    def test_model_initialization_disabled_thinking(self):
        """Test model initialization with thinking disabled."""
        model = KimiOpenAIModel(
            api_key="test-key",
            enable_thinking=False,
        )
        assert model.enable_thinking is False

    def test_k25_fixed_params_with_thinking(self):
        """Test K2.5 fixed parameters when thinking is enabled."""
        model = KimiOpenAIModel(
            api_key="test-key",
            model_id="kimi-k2.5",
            enable_thinking=True,
        )

        params = model._get_params()

        assert params["temperature"] == 1.0
        assert params["top_p"] == 0.95
        assert params["n"] == 1
        assert params["presence_penalty"] == 0.0
        assert params["frequency_penalty"] == 0.0

    def test_k25_fixed_params_without_thinking(self):
        """Test K2.5 fixed parameters when thinking is disabled."""
        model = KimiOpenAIModel(
            api_key="test-key",
            model_id="kimi-k2.5",
            enable_thinking=False,
        )

        params = model._get_params()

        assert params["temperature"] == 0.6
        assert params["top_p"] == 0.95

    def test_non_k25_model_no_fixed_params(self):
        """Test non-K2.5 models don't get fixed parameters."""
        model = KimiOpenAIModel(
            api_key="test-key",
            model_id="kimi-k2",
            temperature=0.5,
            enable_thinking=True,
        )

        params = model._get_params()

        assert params["temperature"] == 0.5  # Not overridden
        assert "top_p" not in params  # Not set

    def test_prepare_request_with_disabled_thinking(self):
        """Test request preparation with disabled thinking."""
        from hawi.agent.messages import MessageRequest

        model = KimiOpenAIModel(
            api_key="test-key",
            model_id="kimi-k2.5",
            enable_thinking=False,
        )

        request = MessageRequest(
            messages=[_create_user_message("Hello")],
        )

        req = model._prepare_request_impl(request)

        assert "extra_body" in req
        assert req["extra_body"]["thinking"]["type"] == "disabled"

    def test_convert_message_with_reasoning(self):
        """Test message conversion extracts reasoning_content."""
        model = KimiOpenAIModel(
            api_key="test-key",
            model_id="kimi-k2.5",
        )

        msg = _create_assistant_message(content=[
            _reasoning_part("Analyzing the problem..."),
            _text_part("The answer is 42"),
        ])

        result = model._convert_message_to_openai(msg)

        assert result["role"] == "assistant"
        assert result.get("reasoning_content") == "Analyzing the problem..."

    def test_convert_message_tool_call_requires_reasoning(self):
        """Test that tool call messages get default reasoning_content."""
        model = KimiOpenAIModel(
            api_key="test-key",
            model_id="kimi-k2.5",
        )

        msg = _create_assistant_message(
            content=None,
            tool_calls=[
                _tool_call_part(
                    id="call_123",
                    name="get_weather",
                    arguments={"location": "Beijing"},
                )
            ],
        )

        result = model._convert_message_to_openai(msg)

        assert result["role"] == "assistant"
        assert result.get("tool_calls") is not None
        # Tool call messages must have non-empty reasoning_content for K2.5
        assert result.get("reasoning_content")
        assert isinstance(result["reasoning_content"], str)
        assert len(result["reasoning_content"]) > 0


@pytest.mark.skipif(not HAS_KIMI_KEY, reason=SKIP_REASON)
class TestKimiOpenAIIntegration:
    """Integration tests requiring real Kimi API access."""

    @pytest.fixture
    def model(self) -> KimiOpenAIModel:
        """Create a Kimi model instance with default settings."""
        return KimiOpenAIModel(
            api_key=KIMI_API_KEY,
            model_id="kimi-k2.5",
            enable_thinking=True,
        )

    @pytest.fixture
    def model_no_thinking(self) -> KimiOpenAIModel:
        """Create a Kimi model instance with thinking disabled."""
        return KimiOpenAIModel(
            api_key=KIMI_API_KEY,
            model_id="kimi-k2.5",
            enable_thinking=False,
        )

    def test_simple_chat_with_thinking(self, model: KimiOpenAIModel):
        """Test basic chat completion with thinking enabled."""
        response = model.invoke(
            messages=[_create_user_message("What is 2+2? Answer with just the number.")],
        )

        assert response.id is not None
        assert len(response.content) > 0
        assert response.content[0]["type"] == "text"
        assert response.usage is not None
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

    def test_simple_chat_without_thinking(self, model_no_thinking: KimiOpenAIModel):
        """Test basic chat completion with thinking disabled."""
        response = model_no_thinking.invoke(
            messages=[_create_user_message("What is 2+2? Answer with just the number.")],
        )

        assert response.id is not None
        assert len(response.content) > 0
        assert response.content[0]["type"] == "text"
        # May or may not have reasoning_content when thinking is disabled

    def test_streaming_response(self, model: KimiOpenAIModel):
        """Test streaming response with reasoning."""
        events = list(model.stream(
            messages=[_create_user_message("Count from 1 to 3.")],
        ))

        # Should have content block events and finish event
        content_events = [e for e in events if e.type == "content_block_delta" and e.delta_type == "text"]
        reasoning_events = [e for e in events if e.type == "content_block_delta" and e.delta_type == "thinking"]
        finish_events = [e for e in events if e.type == "finish"]

        assert len(content_events) > 0
        assert len(finish_events) == 1
        # May have reasoning events depending on model response

    def test_multi_turn_conversation(self, model: KimiOpenAIModel):
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

    def test_balance_query(self, model: KimiOpenAIModel):
        """Test balance query functionality."""
        balances = model.get_balance()

        assert len(balances) > 0
        for balance in balances:
            # Kimi returns empty currency string
            assert balance.available_balance >= 0
            assert balance.total_balance is not None


@pytest.mark.skipif(not HAS_KIMI_KEY, reason=SKIP_REASON)
class TestKimiK25ToolCalls:
    """Tests for Kimi K2.5 tool calls with reasoning."""

    @pytest.fixture
    def model(self) -> KimiOpenAIModel:
        """Create a Kimi K2.5 model instance."""
        return KimiOpenAIModel(
            api_key=KIMI_API_KEY,
            model_id="kimi-k2.5",
            enable_thinking=True,
        )

    def test_tool_call_with_reasoning(self, model: KimiOpenAIModel):
        """Test that tool calls work correctly with reasoning enabled."""
        from hawi.agent.messages import ToolDefinition

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

    def test_multi_turn_with_tool_result(self, model: KimiOpenAIModel):
        """Test multi-turn conversation with tool results."""
        from hawi.agent.messages import ToolDefinition

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
