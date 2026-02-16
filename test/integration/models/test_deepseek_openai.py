"""DeepSeekOpenAIModel integration tests.

Tests the new DeepSeek model implementation based on hawi.agent.models.openai.
"""

import pytest

from hawi.agent.models.deepseek.deepseek_openai import DeepSeekOpenAIModel
from hawi.agent.message import Message, ContentPart
from test.integration.models import get_deepseek_api_key

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


class TestDeepSeekOpenAIUnit:
    """Unit tests for DeepSeekOpenAIModel (no API calls)."""

    def test_model_initialization(self):
        """Test model can be initialized with correct defaults."""
        model = DeepSeekOpenAIModel(
            model_id="deepseek-chat",
            api_key="test-key",
        )
        assert model.model_id == "deepseek-chat"
        assert model.api_key == "test-key"
        assert model.base_url == "https://api.deepseek.com"

    def test_reasoner_model_initialization(self):
        """Test Reasoner model initialization with warnings."""
        model = DeepSeekOpenAIModel(
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

        model = DeepSeekOpenAIModel(
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

    def test_convert_message_with_reasoning(self):
        """Test message conversion with reasoning content.

        Note: According to DeepSeek API docs, reasoning_content should NOT be
        sent in requests. It can only be read from responses.
        """
        model = DeepSeekOpenAIModel(
            model_id="deepseek-reasoner",
            api_key="test-key",
        )

        # Create message with reasoning part directly as TypedDict
        msg = _create_assistant_message(content=[
            {"type": "reasoning", "reasoning": "Let me think about this...", "signature": None},
            {"type": "text", "text": "Here's my answer"},
        ])

        result = model._convert_message_to_openai(msg)

        assert result["role"] == "assistant"
        # According to DeepSeek API docs, reasoning_content should NOT be sent in requests
        # It can only be read from responses
        assert "reasoning_content" not in result, "reasoning_content should not be in request"

    def test_convert_tool_message_to_string(self):
        """Test that tool message content is converted to string."""
        model = DeepSeekOpenAIModel(api_key="test-key")

        msg = _create_tool_result_message(
            tool_call_id="call_123",
            content="Tool result data",
        )

        result = model._convert_message_to_openai(msg)

        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_123"
        assert isinstance(result["content"], str)
        assert result["content"] == "Tool result data"


@pytest.mark.skipif(not HAS_DEEPSEEK_KEY, reason=SKIP_REASON)
class TestDeepSeekOpenAIIntegration:
    """Integration tests requiring real DeepSeek API access."""

    @pytest.fixture
    def model(self) -> DeepSeekOpenAIModel:
        """Create a DeepSeek model instance."""
        return DeepSeekOpenAIModel(
            model_id="deepseek-chat",
            api_key=DEEPSEEK_API_KEY,
        )

    @pytest.fixture
    def reasoner_model(self) -> DeepSeekOpenAIModel:
        """Create a DeepSeek Reasoner model instance."""
        return DeepSeekOpenAIModel(
            model_id="deepseek-reasoner",
            api_key=DEEPSEEK_API_KEY,
        )

    def test_simple_chat_completion(self, model: DeepSeekOpenAIModel):
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

    def test_reasoner_chat_completion(self, reasoner_model: DeepSeekOpenAIModel):
        """Test Reasoner model chat completion with reasoning."""
        response = reasoner_model.invoke(
            messages=[_create_user_message("What is 15 + 27?")],
        )

        assert response.id is not None
        assert len(response.content) > 0
        # Reasoner model may have reasoning_content
        # Note: reasoning_content might be in the response or None depending on API
        assert response.usage is not None

    def test_streaming_response(self, model: DeepSeekOpenAIModel):
        """Test streaming response."""
        events = list(model.stream(
            messages=[_create_user_message("Count from 1 to 3.")],
        ))

        # Should have content block events and finish event
        content_events = [e for e in events if e["type"] == "text_delta"]
        finish_events = [e for e in events if e["type"] == "finish"]

        assert len(content_events) > 0
        assert len(finish_events) == 1
        assert finish_events[0]["stop_reason"] == "end_turn"

    def test_tool_call_formatting(self, model: DeepSeekOpenAIModel):
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

    def test_multi_turn_conversation(self, model: DeepSeekOpenAIModel):
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

    def test_balance_query(self, model: DeepSeekOpenAIModel):
        """Test balance query functionality."""
        balances = model.get_balance()

        assert len(balances) > 0
        for balance in balances:
            assert balance.currency is not None
            assert balance.available_balance >= 0
            assert balance.total_balance is not None


@pytest.mark.skipif(not HAS_DEEPSEEK_KEY, reason=SKIP_REASON)
class TestDeepSeekReasonerMultiTurn:
    """Tests for Reasoner model multi-turn with reasoning content."""

    @pytest.fixture
    def reasoner_model(self) -> DeepSeekOpenAIModel:
        """Create a DeepSeek Reasoner model."""
        return DeepSeekOpenAIModel(
            model_id="deepseek-reasoner",
            api_key=DEEPSEEK_API_KEY,
        )

    def test_reasoner_with_tool_call(self, reasoner_model: DeepSeekOpenAIModel):
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

    def test_reasoner_tool_call_with_reasoning_content(self, reasoner_model: DeepSeekOpenAIModel):
        """Test Reasoner model handles reasoning_content in tool call scenarios.

        When using tool calls with deepseek-reasoner, the API returns reasoning_content
        which must be preserved in multi-turn conversations.
        """
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

        # First turn: Request that may trigger tool call
        response = reasoner_model.invoke(
            messages=[_create_user_message("What's the weather in Beijing?")],
            tools=tools,
        )

        # Verify response structure
        assert response.id is not None
        assert len(response.content) > 0

        # Reasoner model may have reasoning_content
        # This is important for multi-turn tool calling scenarios
        # where reasoning_content must be passed back to the API
