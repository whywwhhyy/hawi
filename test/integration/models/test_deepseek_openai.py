"""DeepSeekOpenAIModel integration tests.

Tests the new DeepSeek model implementation based on hawi.agent.models.openai.
Uses Model Registry for model creation.
"""

import pytest

from hawi.models.deepseek.deepseek_openai import DeepSeekOpenAIModel
from hawi.models import Message
from hawi.models.message import ContentPart
from test.integration.models import has_factory, create_model, skip_on_rate_limit, async_skip_on_rate_limit

# Factory names
DEEPSEEK_CHAT_FACTORY = "deepseek-chat-openai"
DEEPSEEK_REASONER_FACTORY = "deepseek-reasoner-openai"

# Check if factories are available
HAS_DEEPSEEK_CHAT = has_factory(DEEPSEEK_CHAT_FACTORY)
HAS_DEEPSEEK_REASONER = has_factory(DEEPSEEK_REASONER_FACTORY)

# Skip reason for tests requiring factory
SKIP_REASON = f"Factory '{DEEPSEEK_CHAT_FACTORY}' not found in models.yaml"


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
        from hawi.models.message import MessageRequest

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

        result = model._convert_message_to_openai(msg)[0]

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

        result = model._convert_message_to_openai(msg)[0]

        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_123"
        assert isinstance(result["content"], str)
        assert result["content"] == "Tool result data"


@pytest.mark.skipif(not HAS_DEEPSEEK_CHAT, reason=SKIP_REASON)
class TestDeepSeekOpenAIIntegration:
    """Integration tests requiring real DeepSeek API access."""

    @pytest.fixture
    def model(self) -> DeepSeekOpenAIModel:
        """Create a DeepSeek model instance from registry."""
        return create_model(DEEPSEEK_CHAT_FACTORY)

    @pytest.fixture
    def reasoner_model(self) -> DeepSeekOpenAIModel:
        """Create a DeepSeek Reasoner model instance from registry."""
        return create_model(DEEPSEEK_REASONER_FACTORY)

    @skip_on_rate_limit
    def test_simple_chat_completion(self, model: DeepSeekOpenAIModel):
        """Test basic chat completion."""
        response = model.invoke(
            messages=[_create_user_message("Say 'Hello, World!' and nothing else.")],
        )

        assert response.id is not None
        content_list = list(response.content)
        assert len(content_list) > 0
        assert content_list[0]["type"] == "text"
        assert "Hello" in content_list[0]["text"] or "World" in content_list[0]["text"]
        assert response.stop_reason == "end_turn"
        assert response.usage is not None
        assert response.usage["input_tokens"] > 0
        assert response.usage["output_tokens"] > 0

    @skip_on_rate_limit
    def test_reasoner_chat_completion(self, reasoner_model: DeepSeekOpenAIModel):
        """Test Reasoner model chat completion with reasoning."""
        response = reasoner_model.invoke(
            messages=[_create_user_message("What is 15 + 27?")],
        )

        assert response.id is not None
        assert len(list(response.content)) > 0
        # Reasoner model may have reasoning_content
        # Note: reasoning_content might be in the response or None depending on API
        assert response.usage is not None

    @skip_on_rate_limit
    def test_streaming_response(self, model: DeepSeekOpenAIModel):
        """Test streaming response."""
        events = list(model.invoke(
            messages=[_create_user_message("Count from 1 to 3.")],
            streaming=True,
        ))

        # Should have content block events and finish event
        content_events = [e for e in events if e["type"] == "text_delta"]
        finish_events = [e for e in events if e["type"] == "finish"]

        assert len(content_events) > 0
        assert len(finish_events) == 1
        assert finish_events[0]["stop_reason"] == "end_turn"

    @skip_on_rate_limit
    def test_tool_call_formatting(self, model: DeepSeekOpenAIModel):
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

    @skip_on_rate_limit
    def test_multi_turn_conversation(self, model: DeepSeekOpenAIModel):
        """Test multi-turn conversation."""
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
    def test_balance_query(self, model: DeepSeekOpenAIModel):
        """Test balance query functionality."""
        balances = model.get_balance()

        assert len(balances) > 0
        for balance in balances:
            assert balance.currency is not None
            assert balance.available_balance >= 0
            assert balance.total_balance is not None


@pytest.mark.skipif(not HAS_DEEPSEEK_REASONER, reason=f"Factory '{DEEPSEEK_REASONER_FACTORY}' not found")  
class TestDeepSeekReasonerMultiTurn:
    """Tests for Reasoner model multi-turn with reasoning content."""

    @pytest.fixture
    def reasoner_model(self) -> DeepSeekOpenAIModel:
        """Create a DeepSeek Reasoner model from registry."""
        return create_model(DEEPSEEK_REASONER_FACTORY)

    @skip_on_rate_limit
    def test_reasoner_with_tool_call(self, reasoner_model: DeepSeekOpenAIModel):
        """Test Reasoner model supports tool calls (V3.2+).

        DeepSeek-V3.2 and later versions support tool calling with reasoning mode.
        Note: When using tool calls with Reasoner, reasoning_content must be
        properly handled in multi-turn conversations.
        """
        from hawi.models.message import ToolDefinition

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
        content_list = list(response.content)
        assert len(content_list) > 0
        # Response could be text, tool_call, or reasoning (for Reasoner model)
        assert content_list[0]["type"] in ["text", "tool_call", "reasoning"]

    @skip_on_rate_limit
    def test_reasoner_tool_call_with_reasoning_content(self, reasoner_model: DeepSeekOpenAIModel):
        """Test Reasoner model handles reasoning_content in tool call scenarios.

        When using tool calls with deepseek-reasoner, the API returns reasoning_content
        which must be preserved in multi-turn conversations.
        """
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

        # First turn: Request that may trigger tool call
        response = reasoner_model.invoke(
            messages=[_create_user_message("What's the weather in Beijing?")],
            tools=tools,
        )

        # Verify response structure
        assert response.id is not None
        assert len(list(response.content)) > 0

        # Reasoner model may have reasoning_content
        # This is important for multi-turn tool calling scenarios
        # where reasoning_content must be passed back to the API


@pytest.mark.skipif(not HAS_DEEPSEEK_CHAT, reason=SKIP_REASON)
class TestDeepSeekOpenAIAsync:
    """Async integration tests for DeepSeek OpenAI API."""

    @pytest.fixture
    def model(self) -> DeepSeekOpenAIModel:
        """Create a DeepSeek model instance from registry."""
        return create_model(DEEPSEEK_CHAT_FACTORY)

    @pytest.mark.asyncio
    @async_skip_on_rate_limit
    async def test_async_non_streaming_chat_completion(self, model: DeepSeekOpenAIModel):
        """Test async non-streaming chat completion."""
        events = []
        async for event in model.ainvoke(
            messages=[_create_user_message("Say 'Async Hello!' and nothing else.")],
            streaming=False,
        ):
            events.append(event)

        # Should have ModelEvent types for non-streaming
        assert len(events) > 0

        # Check event types
        def get_type(e):
            return e["type"] if isinstance(e, dict) else e.type

        # First event should be text_delta (DeepSeek OpenAI returns delta parts directly)
        assert get_type(events[0]) in ["text_delta", "reasoning_delta"]
        # Last event should be finish
        assert get_type(events[-1]) == "finish"

        # Extract text delta events
        text_deltas = [e for e in events if isinstance(e, dict) and e.get("type") == "text_delta"]
        assert len(text_deltas) > 0

        # Verify content
        full_text = "".join(d.get("delta", "") for d in text_deltas)
        assert "Async" in full_text or "Hello" in full_text

    @pytest.mark.asyncio
    @async_skip_on_rate_limit
    async def test_async_streaming_chat_completion(self, model: DeepSeekOpenAIModel):
        """Test async streaming chat completion."""
        events = []
        async for event in model.ainvoke(
            messages=[_create_user_message("Count from 1 to 3 quickly.")],
            streaming=True,
        ):
            events.append(event)

        # Should have events including text deltas
        assert len(events) > 0

        # Extract text delta events
        text_deltas = [e for e in events if isinstance(e, dict) and e.get("type") == "text_delta"]
        assert len(text_deltas) > 0

        # Verify content includes numbers
        full_text = "".join(d.get("delta", "") for d in text_deltas)
        assert "1" in full_text and "3" in full_text

    @pytest.mark.asyncio
    @async_skip_on_rate_limit
    async def test_async_streaming_events_structure(self, model: DeepSeekOpenAIModel):
        """Test that async streaming produces correct event structure."""
        events = []
        async for event in model.ainvoke(
            messages=[_create_user_message("Hi")],
            streaming=True,
        ):
            events.append(event)

        # Check event structure
        def get_type(e):
            return e["type"] if isinstance(e, dict) else e.type

        event_types = [get_type(e) for e in events]

        # Should start with delta and end with finish
        assert event_types[0] in ["text_delta", "reasoning_delta"]
        assert event_types[-1] == "finish"

        # Verify finish event structure (usage may or may not be present depending on API)
        finish_event = events[-1]
        if isinstance(finish_event, dict):
            assert finish_event.get("type") == "finish"
            assert finish_event.get("stop_reason") is not None

    @pytest.mark.asyncio
    @async_skip_on_rate_limit
    async def test_async_non_streaming_default(self, model: DeepSeekOpenAIModel):
        """Test that ainvoke defaults to non-streaming when streaming parameter omitted."""
        events = []
        async for event in model.ainvoke(
            messages=[_create_user_message("Say 'Default Async' and nothing else.")],
        ):
            events.append(event)

        # Default should be non-streaming (produces ModelEvents)
        assert len(events) > 0

        def get_type(e):
            return e["type"] if isinstance(e, dict) else e.type

        # ainvoke returns delta parts directly (not ModelEvents)
        assert get_type(events[0]) in ["text_delta", "reasoning_delta"]
