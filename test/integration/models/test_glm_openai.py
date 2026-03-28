"""GLM (智谱 AI) OpenAI API integration tests.

Tests GLM API using the OpenAIModel.
GLM API is OpenAI-compatible at https://open.bigmodel.cn/api/paas/v4
"""

import pytest

from hawi.models.openai import OpenAIModel
from hawi.models import Message
from hawi.models.message import ContentPart
from test.integration.models import (
    has_factory, 
    create_model,
    skip_on_rate_limit,
    async_skip_on_rate_limit,
)


# Factory names
GLM_FACTORY = "glm-4.7-flash-openai"

# Check if factories are available
HAS_GLM = has_factory(GLM_FACTORY)

# Skip reason for tests requiring factory
SKIP_REASON = f"Factory '{GLM_FACTORY}' not found in models.yaml"


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


class TestGLMOpenAIUnit:
    """Unit tests for GLM using standard OpenAIModel (no API calls)."""

    def test_model_initialization(self):
        """Test standard OpenAIModel can be initialized for GLM."""
        model = OpenAIModel(
            model_id="glm-4-flash",
            api_key="test-key",
            base_url="https://open.bigmodel.cn/api/paas/v4",
        )
        assert model.model_id == "glm-4-flash"
        assert model.api_key == "test-key"
        assert model.base_url == "https://open.bigmodel.cn/api/paas/v4"

    def test_glm4_model_initialization(self):
        """Test GLM-4 model with standard OpenAIModel."""
        model = OpenAIModel(
            model_id="glm-4",
            api_key="test-key",
            base_url="https://open.bigmodel.cn/api/paas/v4",
        )
        assert model.model_id == "glm-4"

    def test_glm4_air_model_initialization(self):
        """Test GLM-4-Air model with standard OpenAIModel."""
        model = OpenAIModel(
            model_id="glm-4-air",
            api_key="test-key",
            base_url="https://open.bigmodel.cn/api/paas/v4",
        )
        assert model.model_id == "glm-4-air"


@pytest.mark.skipif(not HAS_GLM, reason=SKIP_REASON)
class TestGLMOpenAIIntegration:
    """Integration tests for GLM using standard OpenAIModel."""

    @pytest.fixture
    def model(self) -> OpenAIModel:
        """Create a GLM-4-Flash model instance from registry."""
        return create_model(GLM_FACTORY)

    @pytest.fixture
    def model_glm4(self) -> OpenAIModel:
        """Create a GLM-4 model instance from registry."""
        return create_model(GLM_FACTORY, model_id="glm-4")

    @skip_on_rate_limit
    def test_simple_chat_completion(self, model: OpenAIModel):
        """Test basic chat completion with GLM-4-Flash using standard OpenAIModel."""
        response = model.invoke(
            messages=[_create_user_message("Say 'Hello from GLM Standard OpenAI!' and nothing else.")],
        )

        assert response.id is not None
        content_list = list(response.content)
        assert len(content_list) > 0
        assert content_list[0]["type"] == "text"
        assert "GLM" in content_list[0]["text"] or "Hello" in content_list[0]["text"]
        assert response.stop_reason == "end_turn"
        assert response.usage is not None
        assert response.usage["input_tokens"] > 0
        assert response.usage["output_tokens"] > 0

    @skip_on_rate_limit
    def test_streaming_response(self, model: OpenAIModel):
        """Test streaming response with standard OpenAIModel."""
        events = list(model.invoke(
            streaming=True,
            messages=[_create_user_message("Count from 1 to 3.")],
        ))

        # Should have content block events and finish event
        content_events = [e for e in events if e["type"] == "text_delta"]
        finish_events = [e for e in events if e["type"] == "finish"]

        assert len(content_events) > 0
        assert len(finish_events) == 1
        assert finish_events[0]["stop_reason"] == "end_turn"

    @skip_on_rate_limit
    def test_tool_call_formatting(self, model: OpenAIModel):
        """Test tool call request formatting with standard OpenAIModel."""
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

    @skip_on_rate_limit
    def test_multi_turn_conversation(self, model: OpenAIModel):
        """Test multi-turn conversation with standard OpenAIModel."""
        messages = [
            _create_user_message("My name is Alice."),
        ]

        # First turn
        response1 = model.invoke(messages=messages)
        response1_content = list(response1.content)
        messages.append(_create_assistant_message(content=[
            {"type": "text", "text": response1_content[0].get("text", "")},
        ]))

        # Second turn
        messages.append(_create_user_message("What's my name?"))
        response2 = model.invoke(messages=messages)
        response2_content = list(response2.content)

        assert "Alice" in response2_content[0].get("text", "")

    @skip_on_rate_limit
    def test_glm4_chat_completion(self, model_glm4: OpenAIModel):
        """Test GLM-4 model with standard OpenAIModel."""
        response = model_glm4.invoke(
            messages=[_create_user_message("Say 'Hello from GLM-4!' and nothing else.")],
        )

        assert response.id is not None
        content_list = list(response.content)
        assert len(content_list) > 0
        assert content_list[0]["type"] == "text"
        assert "GLM" in content_list[0]["text"] or "Hello" in content_list[0]["text"]
        assert response.stop_reason == "end_turn"
        assert response.usage is not None


@pytest.mark.skipif(not HAS_GLM, reason=SKIP_REASON)
class TestGLMOpenAIAsync:
    """Async integration tests for GLM using standard OpenAIModel."""

    @pytest.fixture
    def model(self) -> OpenAIModel:
        """Create a GLM model instance from registry."""
        return create_model(GLM_FACTORY)

    @pytest.mark.asyncio
    @async_skip_on_rate_limit
    async def test_async_non_streaming_chat_completion(self, model: OpenAIModel):
        """Test async non-streaming chat completion with standard OpenAIModel."""
        events = []
        async for event in model.ainvoke(
            messages=[_create_user_message("Say 'Async GLM Standard!' and nothing else.")],
            streaming=False,
        ):
            events.append(event)

        assert len(events) > 0

        def get_type(e):
            return e["type"] if isinstance(e, dict) else e.type

        # ainvoke returns delta parts directly (not ModelEvents)
        assert get_type(events[0]) in ["text_delta", "reasoning_delta"]
        assert get_type(events[-1]) == "finish"

        # Extract text deltas
        text_deltas = [e for e in events if isinstance(e, dict) and e.get("type") == "text_delta"]
        assert len(text_deltas) > 0

        full_text = "".join(d.get("delta", "") for d in text_deltas)
        assert "GLM" in full_text or "Async" in full_text

    @pytest.mark.asyncio
    @async_skip_on_rate_limit
    async def test_async_streaming_chat_completion(self, model: OpenAIModel):
        """Test async streaming chat completion with standard OpenAIModel."""
        events = []
        async for event in model.ainvoke(
            messages=[_create_user_message("Count from 1 to 3.")],
            streaming=True,
        ):
            events.append(event)

        assert len(events) > 0

        # Extract text deltas
        text_deltas = [e for e in events if isinstance(e, dict) and e.get("type") == "text_delta"]
        assert len(text_deltas) > 0

        full_text = "".join(d.get("delta", "") for d in text_deltas)
        assert "1" in full_text and "3" in full_text

    @pytest.mark.asyncio
    @async_skip_on_rate_limit
    async def test_async_streaming_events_structure(self, model: OpenAIModel):
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
        assert event_types[0] in ["text_delta", "reasoning_delta"]
        assert event_types[-1] == "finish"

    @pytest.mark.asyncio
    @async_skip_on_rate_limit
    async def test_async_non_streaming_default(self, model: OpenAIModel):
        """Test that ainvoke defaults to non-streaming."""
        events = []
        async for event in model.ainvoke(
            messages=[_create_user_message("Say 'Default Async GLM' and nothing else.")],
        ):
            events.append(event)

        # Default should be non-streaming (produces ModelEvents)
        assert len(events) > 0

        def get_type(e):
            return e["type"] if isinstance(e, dict) else e.type

        # ainvoke returns delta parts directly (not ModelEvents)
        assert get_type(events[0]) in ["text_delta", "reasoning_delta"]
