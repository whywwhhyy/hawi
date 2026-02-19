"""DeepSeekAnthropicModel integration tests.

Tests the DeepSeek model implementation using Anthropic-compatible API.
"""

import pytest

from hawi.agent.models.deepseek.deepseek_anthropic import DeepSeekAnthropicModel
from hawi.agent.message import Message, ContentPart
from test.integration import get_deepseek_api_key

# Check if API key is available
DEEPSEEK_API_KEY = get_deepseek_api_key()
HAS_DEEPSEEK_KEY = DEEPSEEK_API_KEY is not None and DEEPSEEK_API_KEY.strip() != ""

# Skip reason for tests requiring API key
SKIP_REASON = "DeepSeek API key not found (set DEEPSEEK_API_KEY or configure apikey.yaml)"


def _create_user_message(content: str) -> Message:
    """Create a user message directly."""
    return {
        "role": "user",
        "content": [{"type": "text", "text": content}],
        "name": None,
        "tool_calls": None,
        "tool_call_id": None,
        "metadata": None,
    }


def _create_assistant_message(content: list[ContentPart]) -> Message:
    """Create an assistant message directly."""
    return {
        "role": "assistant",
        "content": content,
        "name": None,
        "tool_calls": None,
        "tool_call_id": None,
        "metadata": None,
    }


def _create_tool_result_message(tool_call_id: str, content: str) -> Message:
    """Create a tool result message directly."""
    return {
        "role": "tool",
        "content": [{"type": "text", "text": content}],
        "name": None,
        "tool_calls": None,
        "tool_call_id": tool_call_id,
        "metadata": None,
    }


class TestDeepSeekAnthropicUnit:
    """Unit tests for DeepSeekAnthropicModel (no API calls)."""

    def test_model_initialization(self):
        """Test model can be initialized with correct defaults."""
        model = DeepSeekAnthropicModel(
            model_id="deepseek-chat",
            api_key="test-key",
        )
        assert model.model_id == "deepseek-chat"
        assert model.api_key == "test-key"
        assert model.base_url == "https://api.deepseek.com/anthropic"

    def test_reasoner_model_initialization(self):
        """Test Reasoner model initialization with warnings."""
        model = DeepSeekAnthropicModel(
            model_id="deepseek-reasoner",
            api_key="test-key",
            temperature=0.7,  # Will be warned as unsupported
            logprobs=True,    # Will be warned and removed
        )
        assert model.model_id == "deepseek-reasoner"
        assert "temperature" in model.params
        assert "logprobs" in model.params

    def test_prepare_request_filters_reasoner_params(self):
        """Test that Reasoner model parameters are filtered correctly."""
        from hawi.agent.message import MessageRequest

        model = DeepSeekAnthropicModel(
            model_id="deepseek-reasoner",
            api_key="test-key",
            logprobs=True,
            top_logprobs=5,
        )

        request = MessageRequest(
            messages=[_create_user_message("Hello")],
        )

        req = model._prepare_request_impl(request)

        # Error params should be removed
        assert "logprobs" not in req
        assert "top_logprobs" not in req

    def test_prepare_request_removes_top_k(self):
        """Test that top_k parameter is removed for DeepSeek."""
        from hawi.agent.message import MessageRequest

        model = DeepSeekAnthropicModel(
            model_id="deepseek-chat",
            api_key="test-key",
            top_k=50,
        )

        request = MessageRequest(
            messages=[_create_user_message("Hello")],
        )

        req = model._prepare_request_impl(request)

        # top_k should be removed
        assert "top_k" not in req

    def test_prepare_request_sanitizes_image_content(self):
        """Test that image content is sanitized for DeepSeek."""
        from hawi.agent.message import MessageRequest

        model = DeepSeekAnthropicModel(
            model_id="deepseek-chat",
            api_key="test-key",
        )

        request = MessageRequest(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at this image:"},
                    {"type": "image", "source": {"url": "data:image/png;base64,abc", "detail": "auto"}},
                ],
                "name": None,
                "tool_calls": None,
                "tool_call_id": None,
                "metadata": None,
            }],
        )

        req = model._prepare_request_impl(request)

        # Image content should be converted to text placeholder
        content = req["messages"][0]["content"]
        assert len(content) == 2
        assert content[1]["type"] == "text"
        assert "[图片内容]" in content[1]["text"]

    def test_prepare_request_sanitizes_document_content(self):
        """Test that document content is sanitized for DeepSeek."""
        from hawi.agent.message import MessageRequest

        model = DeepSeekAnthropicModel(
            model_id="deepseek-chat",
            api_key="test-key",
        )

        request = MessageRequest(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Read this document:"},
                    {"type": "document", "source": {"url": "data:application/pdf;base64,abc", "mime_type": "application/pdf"}, "title": None, "context": None},
                ],
                "name": None,
                "tool_calls": None,
                "tool_call_id": None,
                "metadata": None,
            }],
        )

        req = model._prepare_request_impl(request)

        # Document content should be converted to text placeholder
        content = req["messages"][0]["content"]
        assert len(content) == 2
        assert content[1]["type"] == "text"
        assert "[文档内容]" in content[1]["text"]


@pytest.mark.skipif(not HAS_DEEPSEEK_KEY, reason=SKIP_REASON)
class TestDeepSeekAnthropicIntegration:
    """Integration tests requiring real DeepSeek API access."""

    @pytest.fixture
    def model(self) -> DeepSeekAnthropicModel:
        """Create a DeepSeek Anthropic model instance."""
        return DeepSeekAnthropicModel(
            model_id="deepseek-chat",
            api_key=DEEPSEEK_API_KEY,
        )

    @pytest.fixture
    def reasoner_model(self) -> DeepSeekAnthropicModel:
        """Create a DeepSeek Reasoner model instance."""
        return DeepSeekAnthropicModel(
            model_id="deepseek-reasoner",
            api_key=DEEPSEEK_API_KEY,
        )

    def test_simple_chat_completion(self, model: DeepSeekAnthropicModel):
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
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

    def test_reasoner_chat_completion(self, reasoner_model: DeepSeekAnthropicModel):
        """Test Reasoner model chat completion with reasoning."""
        response = reasoner_model.invoke(
            messages=[_create_user_message("What is 15 + 27?")],
        )

        assert response.id is not None
        assert len(response.content) > 0
        # Reasoner model may have reasoning content
        assert response.usage is not None

    def test_streaming_response(self, model: DeepSeekAnthropicModel):
        """Test streaming response."""
        events = list(model.stream(
            messages=[_create_user_message("Count from 1 to 3.")],
        ))

        # Should have content block events
        content_events = [e for e in events if e["type"] == "text_delta"]

        assert len(content_events) > 0

    def test_tool_call_formatting(self, model: DeepSeekAnthropicModel):
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

    def test_multi_turn_conversation(self, model: DeepSeekAnthropicModel):
        """Test multi-turn conversation."""
        messages = [
            _create_user_message("My name is Alice."),
        ]

        # First turn
        response1 = model.invoke(messages=messages)
        messages.append(_create_assistant_message(content=[
            {"type": "text", "text": response1.content[0].get("text", "")},
        ]))

        # Second turn
        messages.append(_create_user_message("What's my name?"))
        response2 = model.invoke(messages=messages)

        assert "Alice" in response2.content[0].get("text", "")


@pytest.mark.skipif(not HAS_DEEPSEEK_KEY, reason=SKIP_REASON)
class TestDeepSeekAnthropicReasonerMultiTurn:
    """Tests for Reasoner model multi-turn with reasoning content."""

    @pytest.fixture
    def reasoner_model(self) -> DeepSeekAnthropicModel:
        """Create a DeepSeek Reasoner model."""
        return DeepSeekAnthropicModel(
            model_id="deepseek-reasoner",
            api_key=DEEPSEEK_API_KEY,
        )

    def test_reasoner_with_tool_call(self, reasoner_model: DeepSeekAnthropicModel):
        """Test Reasoner model supports tool calls (V3.2+).

        DeepSeek-V3.2 and later versions support tool calling with reasoning mode.
        Note: When using tool calls with Reasoner, reasoning_content must be
        properly handled in multi-turn conversations.
        """
        from hawi.agent.message import ToolDefinition

        tools: list[ToolDefinition] = [
            {
                "type": "function",
                "name": "calculate",
                "description": "Perform calculation",
                "schema": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"},
                    },
                    "required": ["expression"],
                },
            }
        ]

        # Tool calling with Reasoner should now work (V3.2+)
        # The API will return either a text response or tool_calls
        response = reasoner_model.invoke(
            messages=[_create_user_message("Calculate 123 * 456")],
            tools=tools,
        )

        # Should have some response content
        assert len(response.content) > 0
        # Response could be text, tool_call, or reasoning (for Reasoner model)
        assert response.content[0]["type"] in ["text", "tool_call", "reasoning"]