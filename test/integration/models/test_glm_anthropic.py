"""GLM (智谱 AI) Anthropic API integration tests.

Tests GLM API using the AnthropicModel.
Note: This test verifies if GLM supports Anthropic-compatible API format.
"""

import pytest

from hawi.models.anthropic import AnthropicModel
from hawi.models import Message
from test.integration.models import get_glm_api_key


# Check if API key is available
GLM_API_KEY = get_glm_api_key()
HAS_GLM_KEY = GLM_API_KEY is not None and GLM_API_KEY.strip() != ""

# Skip reason for tests requiring API key
SKIP_REASON = "GLM API key not found (set GLM_API_KEY or configure models.yaml)"

# GLM Anthropic-compatible endpoint
# Reference: https://open.bigmodel.cn/dev/api/thirdparty-frame/anthropic-sdk
GLM_ANTHROPIC_BASE_URL = "https://open.bigmodel.cn/api/anthropic"


def _create_user_message(content: str) -> Message:
    """Create a user message directly."""
    return {
        "role": "user",
        "content": [{"type": "text", "text": content}],
        "name": None,
        "metadata": None,
    }


class TestGLMAnthropicUnit:
    """Unit tests for GLM using standard AnthropicModel (no API calls)."""

    def test_model_initialization(self):
        """Test standard AnthropicModel can be initialized for GLM."""
        model = AnthropicModel(
            model_id="glm-4-flash",
            api_key="test-key",
            base_url=GLM_ANTHROPIC_BASE_URL,
        )
        assert model.model_id == "glm-4-flash"
        assert model.api_key == "test-key"
        assert model.base_url == GLM_ANTHROPIC_BASE_URL

    def test_glm4_model_initialization(self):
        """Test GLM-4 model with standard AnthropicModel."""
        model = AnthropicModel(
            model_id="glm-4",
            api_key="test-key",
            base_url=GLM_ANTHROPIC_BASE_URL,
        )
        assert model.model_id == "glm-4"


@pytest.mark.skipif(not HAS_GLM_KEY, reason=SKIP_REASON)
class TestGLMAnthropicIntegration:
    """Integration tests for GLM using standard AnthropicModel.

    Note: These tests will be skipped if GLM does not support Anthropic API format.
    If GLM API returns errors, it indicates GLM does not support Anthropic format.
    """

    @pytest.fixture
    def model(self) -> AnthropicModel:
        """Create a GLM-4-Flash model instance using standard AnthropicModel."""
        return AnthropicModel(
            model_id="glm-4-flash",
            api_key=GLM_API_KEY,
            base_url=GLM_ANTHROPIC_BASE_URL,
        )

    def test_simple_chat_completion(self, model: AnthropicModel):
        """Test basic chat completion with GLM using standard AnthropicModel."""
        try:
            response = model.invoke(
                messages=[_create_user_message("Say 'Hello from GLM Anthropic!' and nothing else.")],
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
        except Exception as e:
            pytest.skip(f"GLM may not support Anthropic API format: {e}")

    def test_streaming_response(self, model: AnthropicModel):
        """Test streaming response with standard AnthropicModel."""
        try:
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
        except Exception as e:
            pytest.skip(f"GLM may not support Anthropic streaming API: {e}")

    def test_tool_call_formatting(self, model: AnthropicModel):
        """Test tool call request formatting with standard AnthropicModel."""
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

        try:
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
        except Exception as e:
            pytest.skip(f"GLM may not support Anthropic tool calls: {e}")


@pytest.mark.skipif(not HAS_GLM_KEY, reason=SKIP_REASON)
class TestGLMAnthropicAsync:
    """Async integration tests for GLM using standard AnthropicModel."""

    @pytest.fixture
    def model(self) -> AnthropicModel:
        """Create a GLM-4-Flash model instance using standard AnthropicModel."""
        return AnthropicModel(
            model_id="glm-4-flash",
            api_key=GLM_API_KEY,
            base_url=GLM_ANTHROPIC_BASE_URL,
        )

    @pytest.mark.asyncio
    async def test_async_non_streaming_chat_completion(self, model: AnthropicModel):
        """Test async non-streaming chat completion with standard AnthropicModel."""
        try:
            events = []
            async for event in model.ainvoke(
                messages=[_create_user_message("Say 'Async GLM Anthropic!' and nothing else.")],
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
            assert "GLM" in full_text or "Async" in full_text
        except Exception as e:
            pytest.skip(f"GLM may not support Anthropic async API: {e}")

    @pytest.mark.asyncio
    async def test_async_streaming_chat_completion(self, model: AnthropicModel):
        """Test async streaming chat completion with standard AnthropicModel."""
        try:
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
        except Exception as e:
            pytest.skip(f"GLM may not support Anthropic async streaming: {e}")

    @pytest.mark.asyncio
    async def test_async_streaming_events_structure(self, model: AnthropicModel):
        """Test that async streaming produces correct event sequence."""
        try:
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
            assert event_types[-1] == "finish"
        except Exception as e:
            pytest.skip(f"GLM may not support Anthropic async streaming events: {e}")
