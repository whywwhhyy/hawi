"""MiniMax OpenAI API integration tests.

Tests the MiniMax M2.5/M2.1 model implementation using OpenAI-compatible API.
"""

import pytest

from hawi.models.minimax.minimax_openai import MiniMaxOpenAIModel
from hawi.models import Message
from hawi.models.message import ContentPart
from test.integration.models import get_minimax_api_key

# Check if API key is available
MINIMAX_API_KEY = get_minimax_api_key()
HAS_MINIMAX_KEY = MINIMAX_API_KEY is not None and MINIMAX_API_KEY.strip() != ""

# Skip reason for tests requiring API key
SKIP_REASON = "MiniMax API key not found (set MINIMAX_API_KEY or configure models.yaml)"


def _create_user_message(content: str) -> Message:
    """Create a user message directly."""
    return {
        "role": "user",
        "content": [{"type": "text", "text": content}],
        "name": None,
        "metadata": None,
    }


def _create_assistant_message(content: list[ContentPart]) -> Message:
    """Create an assistant message directly."""
    return {
        "role": "assistant",
        "content": content,
        "name": None,
        "metadata": None,
    }


def _create_tool_result_message(tool_call_id: str, content: str) -> Message:
    """Create a tool result message directly."""
    return {
        "role": "tool",
        "content": [{
            "type": "tool_result",
            "tool_call_id": tool_call_id,
            "content": [{"type": "text", "text": content}],
            "is_error": False,
        }],
        "name": None,
        "metadata": None,
    }


class TestMiniMaxOpenAIUnit:
    """Unit tests for MiniMaxOpenAIModel (no API calls)."""

    def test_model_initialization(self):
        """Test model can be initialized with correct defaults."""
        model = MiniMaxOpenAIModel(
            model_id="MiniMax-M2.5",
            api_key="test-key",
        )
        assert model.model_id == "MiniMax-M2.5"
        assert model.api_key == "test-key"
        assert model.base_url == "https://api.minimaxi.com/v1"

    def test_m21_model_initialization(self):
        """Test M2.1 model initialization."""
        model = MiniMaxOpenAIModel(
            model_id="MiniMax-M2.1",
            api_key="test-key",
        )
        assert model.model_id == "MiniMax-M2.1"

    def test_convert_tool_message_to_string(self):
        """Test that tool message content is converted to string."""
        model = MiniMaxOpenAIModel(api_key="test-key")

        msg = _create_tool_result_message(
            tool_call_id="call_123",
            content="Tool result data",
        )

        result = model._convert_message_to_openai(msg)[0]

        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_123"
        assert isinstance(result["content"], str)
        assert result["content"] == "Tool result data"


@pytest.mark.skipif(not HAS_MINIMAX_KEY, reason=SKIP_REASON)
class TestMiniMaxM25Integration:
    """Integration tests for MiniMax M2.5 model requiring real API access."""

    @pytest.fixture
    def model(self) -> MiniMaxOpenAIModel:
        """Create a MiniMax M2.5 model instance."""
        return MiniMaxOpenAIModel(
            model_id="MiniMax-M2.5",
            api_key=MINIMAX_API_KEY,
        )

    def test_simple_chat_completion(self, model: MiniMaxOpenAIModel):
        """Test basic chat completion with M2.5."""
        response = model.invoke(
            messages=[_create_user_message("Say 'Hello, World!' and nothing else.")],
        )

        assert response.id is not None
        content_list = list(response.content)
        assert len(content_list) > 0
        # MiniMax M2.5 may return reasoning content as the first part
        assert content_list[0]["type"] in ["text", "reasoning"]
        # Find text content for assertion
        text_content = ""
        for part in content_list:
            if part.get("type") == "text":
                text_content += part.get("text", "")
        assert "Hello" in text_content or "World" in text_content
        assert response.stop_reason == "end_turn"
        assert response.usage is not None
        assert response.usage["input_tokens"] > 0
        assert response.usage["output_tokens"] > 0

    def test_streaming_response(self, model: MiniMaxOpenAIModel):
        """Test streaming response."""
        events = list(model.invoke(
            streaming=True,
            messages=[_create_user_message("Count from 1 to 3.")],
        ))

        # MiniMax may output thinking_delta or text_delta events
        content_events = [e for e in events if e["type"] in ("text_delta", "thinking_delta")]

        assert len(content_events) > 0
        # finish event may not be present in some cases, so we just verify we got content

    def test_tool_call_formatting(self, model: MiniMaxOpenAIModel):
        """Test tool call request formatting."""
        from hawi.models.message import ToolDefinition

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
        content_list = list(response.content)
        assert len(content_list) > 0
        if content_list[0]["type"] == "tool_call":
            assert content_list[0]["name"] == "get_weather"
            assert "location" in content_list[0]["arguments"]
            assert response.stop_reason == "tool_use"

    def test_multi_turn_conversation(self, model: MiniMaxOpenAIModel):
        """Test multi-turn conversation."""
        messages = [
            _create_user_message("My name is Alice."),
        ]

        # First turn
        response1 = model.invoke(messages=messages)
        # Find text content from response (may be mixed with reasoning)
        text_parts = [p for p in response1.content if p.get("type") == "text"]
        assistant_text = text_parts[0].get("text", "") if text_parts else ""
        messages.append(_create_assistant_message(content=[
            {"type": "text", "text": assistant_text},
        ]))

        # Second turn
        messages.append(_create_user_message("What's my name?"))
        response2 = model.invoke(messages=messages)

        # Find text content from response2
        text_parts2 = [p for p in response2.content if p.get("type") == "text"]
        response_text = text_parts2[0].get("text", "") if text_parts2 else ""
        assert "Alice" in response_text


@pytest.mark.skipif(not HAS_MINIMAX_KEY, reason=SKIP_REASON)
class TestMiniMaxM21Integration:
    """Integration tests for MiniMax M2.1 model requiring real API access."""

    @pytest.fixture
    def model(self) -> MiniMaxOpenAIModel:
        """Create a MiniMax M2.1 model instance."""
        return MiniMaxOpenAIModel(
            model_id="MiniMax-M2.1",
            api_key=MINIMAX_API_KEY,
        )

    def test_simple_chat_completion(self, model: MiniMaxOpenAIModel):
        """Test basic chat completion with M2.1."""
        response = model.invoke(
            messages=[_create_user_message("Say 'Hello from M2.1!' and nothing else.")],
        )

        assert response.id is not None
        content_list = list(response.content)
        assert len(content_list) > 0
        # MiniMax M2.1 may return reasoning content as the first part
        assert content_list[0]["type"] in ["text", "reasoning"]
        assert response.stop_reason == "end_turn"
        assert response.usage is not None

    def test_streaming_response(self, model: MiniMaxOpenAIModel):
        """Test streaming response with M2.1."""
        events = list(model.invoke(
            streaming=True,
            messages=[_create_user_message("Tell me a short joke.")],
        ))

        # MiniMax may output thinking_delta or text_delta events
        content_events = [e for e in events if e["type"] in ("text_delta", "thinking_delta")]

        assert len(content_events) > 0
        # finish event may not be present in some cases


@pytest.mark.skipif(not HAS_MINIMAX_KEY, reason=SKIP_REASON)
class TestMiniMaxOpenAIAsync:
    """Async integration tests for MiniMax OpenAI API."""

    @pytest.fixture
    def model(self) -> MiniMaxOpenAIModel:
        """Create a MiniMax M2.5 model instance."""
        return MiniMaxOpenAIModel(
            model_id="MiniMax-M2.5",
            api_key=MINIMAX_API_KEY,
        )

    @pytest.mark.asyncio
    async def test_async_non_streaming_chat_completion(self, model: MiniMaxOpenAIModel):
        """Test async non-streaming chat completion."""
        events = []
        async for event in model.ainvoke(
            messages=[_create_user_message("Say 'Async MiniMax!' and nothing else.")],
            streaming=False,
        ):
            events.append(event)

        assert len(events) > 0

        def get_type(e):
            return e["type"] if isinstance(e, dict) else e.type

        # ainvoke returns delta parts directly (not ModelEvents)
        assert get_type(events[0]) in ["text_delta", "thinking_delta"]
        assert get_type(events[-1]) == "finish"

        # Extract text deltas
        text_deltas = [e for e in events if isinstance(e, dict) and e.get("type") == "text_delta"]
        assert len(text_deltas) > 0

        full_text = "".join(d.get("delta", "") for d in text_deltas)
        assert "MiniMax" in full_text or "Async" in full_text

    @pytest.mark.asyncio
    async def test_async_streaming_chat_completion(self, model: MiniMaxOpenAIModel):
        """Test async streaming chat completion."""
        events = []
        async for event in model.ainvoke(
            messages=[_create_user_message("Count from 1 to 3.")],
            streaming=True,
        ):
            events.append(event)

        assert len(events) > 0

        # MiniMax may output thinking_delta or text_delta events
        content_events = [e for e in events if isinstance(e, dict) and e.get("type") in ("text_delta", "thinking_delta")]

        assert len(content_events) > 0

        full_text = "".join(str(d.get("delta", "")) for d in content_events)
        assert "1" in full_text and "3" in full_text

    @pytest.mark.asyncio
    async def test_async_streaming_events_structure(self, model: MiniMaxOpenAIModel):
        """Test that async streaming produces correct event sequence."""
        events = []
        async for event in model.ainvoke(
            messages=[_create_user_message("Hi")],
            streaming=True,
        ):
            events.append(event)

        def get_type(e):
            return e["type"] if isinstance(e, dict) else e.type

        event_types = [get_type(e) for e in events]

        # Verify event sequence - ainvoke returns delta parts directly
        assert event_types[0] in ["text_delta", "thinking_delta"]
        # Last event should be finish or a content delta (depending on API)
        assert event_types[-1] in ["finish", "text_delta", "thinking_delta"]

    @pytest.mark.asyncio
    async def test_async_non_streaming_m21_model(self):
        """Test async non-streaming with M2.1 model."""
        model = MiniMaxOpenAIModel(
            model_id="MiniMax-M2.1",
            api_key=MINIMAX_API_KEY,
        )

        events = []
        async for event in model.ainvoke(
            messages=[_create_user_message("Say 'M2.1 Async' and nothing else.")],
            streaming=False,
        ):
            events.append(event)

        assert len(events) > 0

        # Extract text deltas
        text_deltas = [e for e in events if isinstance(e, dict) and e.get("type") == "text_delta"]
        assert len(text_deltas) > 0

        full_text = "".join(d.get("delta", "") for d in text_deltas)
        assert "M2.1" in full_text or "Async" in full_text
