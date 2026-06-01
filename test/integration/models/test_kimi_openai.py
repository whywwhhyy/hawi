"""
KimiOpenAIModel integration tests.

Tests the new Kimi model implementation based on hawi.agent.models.openai.
"""

import pytest
from typing import Any

from hawi.models.kimi.kimi_openai import KimiOpenAIModel
from hawi.models import Message
from hawi.models.message import (
    ContentPart,
    TextPart,
    ToolCallPart,
    ReasoningPart,
)
from test.integration.models import has_model, create_model, skip_on_rate_limit, async_skip_on_rate_limit


# Model name
KIMI_OPENAI_MODEL = "moonshot/kimi-k2.5"

# Check if model is available
HAS_KIMI = has_model(KIMI_OPENAI_MODEL)

# Skip reason for tests requiring model
SKIP_REASON = f"Model '{KIMI_OPENAI_MODEL}' not found in models.yaml"


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
        "metadata": None,
    }


def _create_assistant_message(
    content: str | list[ContentPart] | None = None,
    tool_calls: list[ToolCallPart] | None = None,
) -> Message:
    """Create an assistant message."""
    # 合并 content 和 tool_calls 到 content 中
    normalized_content = _normalize_content(content)
    if tool_calls:
        normalized_content = normalized_content + tool_calls
    return {
        "role": "assistant",
        "content": normalized_content,
        "name": None,
        "metadata": None,
    }


def _create_tool_result_message(
    tool_call_id: str,
    content: str | list[ContentPart],
) -> Message:
    """Create a tool result message."""
    return {
        "role": "tool",
        "content": [{
            "type": "tool_result",
            "tool_call_id": tool_call_id,
            "content": _normalize_content(content),
            "is_error": False,
        }],
        "name": None,
        "metadata": None,
    }




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

    @pytest.mark.parametrize(
        "model_id",
        [
            "kimi-k2.6",
            "kimi-k2-6",
            "moonshot/kimi-k2.6",
            "KIMI_K2_6",
        ],
    )
    def test_k26_thinking_model_detection_variants(self, model_id: str):
        """Test K2.6 model ID variants are detected as thinking models."""
        assert KimiOpenAIModel._is_thinking_model_id(model_id)

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
        from hawi.models.message import MessageRequest

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

        result = model._convert_message_to_openai(msg)[0]

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

        result = model._convert_message_to_openai(msg)[0]

        assert result["role"] == "assistant"
        assert result.get("tool_calls") is not None
        # Tool call messages must have non-empty reasoning_content for K2.5
        assert result.get("reasoning_content")
        assert isinstance(result["reasoning_content"], str)
        assert len(result["reasoning_content"]) > 0

    def test_k26_mixed_tool_call_history_includes_reasoning(self):
        """Test K2.6 split tool-call history preserves reasoning_content."""
        model = KimiOpenAIModel(
            api_key="test-key",
            model_id="kimi-k2.6",
        )

        msg = _create_assistant_message(content=[
            _reasoning_part("Need to inspect the project files."),
            _text_part("I'll inspect the project."),
            _tool_call_part(
                id="call_123",
                name="list_dir",
                arguments={"path": "."},
            ),
        ])

        results = model._convert_message_to_openai(msg)
        tool_call_message = next(item for item in results if item.get("tool_calls"))

        assert tool_call_message["role"] == "assistant"
        assert tool_call_message["content"] is None
        assert (
            tool_call_message["reasoning_content"]
            == "Need to inspect the project files."
        )


@pytest.mark.skipif(not HAS_KIMI, reason=SKIP_REASON)
class TestKimiOpenAIIntegration:
    """Integration tests requiring real Kimi API access."""

    @pytest.fixture
    def model(self) -> KimiOpenAIModel:
        """Create a Kimi model instance with default settings from registry."""
        model = create_model(KIMI_OPENAI_MODEL)
        assert isinstance(model, KimiOpenAIModel)
        return model

    @pytest.fixture
    def model_no_thinking(self) -> KimiOpenAIModel:
        """Create a Kimi model instance with thinking disabled."""
        model = create_model(KIMI_OPENAI_MODEL, enable_thinking=False)
        assert isinstance(model, KimiOpenAIModel)
        return model

    @skip_on_rate_limit
    def test_simple_chat_with_thinking(self, model: KimiOpenAIModel):
        """Test basic chat completion with thinking enabled."""
        response = model.invoke(
            messages=[_create_user_message("What is 2+2? Answer with just the number.")],
        )

        assert response.id is not None
        content_list = list(response.content)
        assert len(content_list) > 0
        # When thinking is enabled, reasoning_content comes first
        assert content_list[0]["type"] in ["text", "reasoning"]
        assert response.usage is not None
        assert response.usage["input_tokens"] > 0
        assert response.usage["output_tokens"] > 0

    @skip_on_rate_limit
    def test_simple_chat_without_thinking(self, model_no_thinking: KimiOpenAIModel):
        """Test basic chat completion with thinking disabled."""
        response = model_no_thinking.invoke(
            messages=[_create_user_message("What is 2+2? Answer with just the number.")],
        )

        assert response.id is not None
        content_list = list(response.content)
        assert len(content_list) > 0
        # When thinking is disabled, content[0] should be text
        assert content_list[0]["type"] == "text"
        # May or may not have reasoning_content when thinking is disabled

    @skip_on_rate_limit
    def test_streaming_response(self, model: KimiOpenAIModel):
        """Test streaming response with reasoning."""
        events = list(model.invoke(
            streaming=True,
            messages=[_create_user_message("Count from 1 to 3.")],
        ))

        # Should have content block events and finish event
        content_events = [e for e in events if e["type"] == "text_delta"]
        finish_events = [e for e in events if e["type"] == "finish"]

        assert len(content_events) > 0
        assert len(finish_events) == 1

    @skip_on_rate_limit
    def test_multi_turn_conversation(self, model: KimiOpenAIModel):
        """Test multi-turn conversation."""
        messages = [
            _create_user_message("My name is Bob."),
        ]

        # First turn
        response1 = model.invoke(messages=messages)
        # Find text part (reasoning may come first when thinking is enabled)
        text_parts = [p for p in response1.content if p["type"] == "text"]
        assert len(text_parts) > 0
        first_part = text_parts[0]
        messages.append(_create_assistant_message(content=[
            _text_part(first_part["text"]),
        ]))

        # Second turn
        messages.append(_create_user_message("What's my name?"))
        response2 = model.invoke(messages=messages)

        # Find text part (reasoning may come first when thinking is enabled)
        text_parts = [p for p in response2.content if p["type"] == "text"]
        assert len(text_parts) > 0
        second_part = text_parts[0]
        assert "Bob" in second_part["text"]

    @skip_on_rate_limit
    def test_balance_query(self, model: KimiOpenAIModel):
        """Test balance query functionality."""
        balances = model.get_balance()

        assert len(balances) > 0
        for balance in balances:
            # Kimi returns empty currency string
            assert balance.available_balance >= 0
            assert balance.total_balance is not None


@pytest.mark.skipif(not HAS_KIMI, reason=SKIP_REASON)
class TestKimiK25ToolCalls:
    """Tests for Kimi K2.5 tool calls with reasoning."""

    @pytest.fixture
    def model(self) -> KimiOpenAIModel:
        """Create a Kimi K2.5 model instance from registry."""
        model = create_model(KIMI_OPENAI_MODEL)
        assert isinstance(model, KimiOpenAIModel)
        return model

    @skip_on_rate_limit
    def test_tool_call_with_reasoning(self, model: KimiOpenAIModel):
        """Test that tool calls work correctly with reasoning enabled."""
        from hawi.models.message import ToolDefinition

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
        content_list = list(response.content)
        tool_calls = [c for c in content_list if c["type"] == "tool_call"]
        text_parts = [c for c in content_list if c["type"] == "text"]

        if tool_calls:
            assert tool_calls[0]["name"] == "calculate"
            assert response.stop_reason == "tool_use"
        elif text_parts:
            # Model may respond directly with text
            assert "10000" in text_parts[0]["text"]

    @skip_on_rate_limit
    def test_multi_turn_with_tool_result(self, model: KimiOpenAIModel):
        """Test multi-turn conversation with tool results."""
        from hawi.models.message import ToolDefinition

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

        content_list = list(response.content)
        tool_calls = [c for c in content_list if c["type"] == "tool_call"]
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


@pytest.mark.skipif(not HAS_KIMI, reason=SKIP_REASON)
class TestKimiOpenAIAsync:
    """Async integration tests for Kimi OpenAI API."""

    @pytest.fixture
    def model(self) -> KimiOpenAIModel:
        """Create a Kimi model instance from registry."""
        model = create_model(KIMI_OPENAI_MODEL)
        assert isinstance(model, KimiOpenAIModel)
        return model

    @pytest.mark.asyncio
    @async_skip_on_rate_limit
    async def test_async_non_streaming_chat_completion(self, model: KimiOpenAIModel):
        """Test async non-streaming chat completion."""
        events = []
        async for event in model.ainvoke(
            messages=[_create_user_message("Say 'Async Kimi!' and nothing else.")],
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
        assert "Kimi" in full_text or "Async" in full_text

    @pytest.mark.asyncio
    @async_skip_on_rate_limit
    async def test_async_streaming_chat_completion(self, model: KimiOpenAIModel):
        """Test async streaming chat completion with reasoning."""
        events = []
        async for event in model.ainvoke(
            messages=[_create_user_message("Count from 1 to 3.")],
            streaming=True,
        ):
            events.append(event)

        assert len(events) > 0

        # Check for text deltas (reasoning deltas may also appear)
        text_deltas = [e for e in events if isinstance(e, dict) and e.get("type") == "text_delta"]

        assert len(text_deltas) > 0

        full_text = "".join(d.get("delta", "") for d in text_deltas)
        assert "1" in full_text and "3" in full_text

    @pytest.mark.asyncio
    @async_skip_on_rate_limit
    async def test_async_streaming_events_structure(self, model: KimiOpenAIModel):
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
    async def test_async_non_streaming_with_thinking_disabled(self):
        """Test async non-streaming with thinking disabled."""
        model = create_model(KIMI_OPENAI_MODEL, enable_thinking=False)

        events = []
        async for event in model.ainvoke(
            messages=[_create_user_message("Say 'No Thinking' and nothing else.")],
            streaming=False,
        ):
            events.append(event)

        assert len(events) > 0

        # Extract text deltas
        text_deltas = [e for e in events if isinstance(e, dict) and e.get("type") == "text_delta"]
        assert len(text_deltas) > 0

        full_text = "".join(d.get("delta", "") for d in text_deltas)
        assert "No" in full_text or "Thinking" in full_text
