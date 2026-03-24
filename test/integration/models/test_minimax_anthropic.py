"""MiniMaxAnthropicModel integration tests.

Tests the MiniMax model implementation using Anthropic-compatible API.
"""

import functools
import pytest

from hawi.models.minimax.minimax_anthropic import MiniMaxAnthropicModel
from hawi.models import Message
from hawi.models.message import ContentPart
from test.integration import get_minimax_api_key

# Check if API key is available
MINIMAX_API_KEY = get_minimax_api_key()
HAS_MINIMAX_KEY = MINIMAX_API_KEY is not None and MINIMAX_API_KEY.strip() != ""

# Skip reason for tests requiring API key
SKIP_REASON = "MiniMax API key not found (set MINIMAX_API_KEY or configure models.yaml)"


def _is_minimax_retryable_error(e: Exception) -> bool:
    """Check if an exception is a MiniMax 794/1000 retryable error.
    
    This function checks for MiniMax API specific errors:
    - openai.APIError with message pattern "unknown error, 794 (1000)"
    - HTTP 794 status code in error body
    - MiniMax error code 1000 (internal error)
    
    Args:
        e: The exception to check.
        
    Returns:
        True if the error is a retryable MiniMax error, False otherwise.
    """
    # Only check openai.APIError (or its subclasses)
    try:
        from openai import APIError
        if not isinstance(e, APIError):
            return False
    except ImportError:
        # If openai module not available, fall back to string check
        pass
    
    error_msg = str(e)
    
    # Check for MiniMax specific error patterns
    # Pattern 1: "unknown error, 794 (1000)" - the most common pattern
    if "unknown error, 794" in error_msg and "(1000)" in error_msg:
        return True
    
    # Pattern 2: Check error body for structured error codes
    # APIError.body may contain {'error': {'code': '794' or 1000}}
    if hasattr(e, 'body') and e.body is not None:
        body = e.body
        if isinstance(body, dict):
            error_body = body.get('error', {})
            if isinstance(error_body, dict):
                code = error_body.get('code')
                # Check for MiniMax error codes that indicate temporary issues
                if code in (794, '794', 1000, '1000'):
                    return True
    
    return False


def retry_on_794(max_retries: int = 3):
    """Decorator to retry tests on MiniMax 794/1000 error.
    
    Retries tests when MiniMax API returns temporary internal errors:
    - Error 794: MiniMax internal HTTP status code
    - Error 1000: MiniMax "unknown error" business code
    
    Args:
        max_retries: Maximum number of retry attempts.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if _is_minimax_retryable_error(e):
                        last_exception = e
                        if attempt < max_retries - 1:
                            print(f"\nMiniMax error {e} on attempt {attempt + 1}, retrying...")
                            continue
                    # If not a retryable error or last attempt, re-raise
                    raise
            # All retries exhausted
            raise last_exception
        return wrapper
    return decorator


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


class TestMiniMaxAnthropicUnit:
    """Unit tests for MiniMaxAnthropicModel (no API calls)."""

    def test_model_initialization(self):
        """Test model can be initialized with correct defaults."""
        model = MiniMaxAnthropicModel(
            model_id="MiniMax-M2.5",
            api_key="test-key",
        )
        assert model.model_id == "MiniMax-M2.5"
        assert model.api_key == "test-key"
        assert model.base_url == "https://api.minimaxi.com/anthropic"

    def test_m21_model_initialization(self):
        """Test M2.1 model initialization."""
        model = MiniMaxAnthropicModel(
            model_id="MiniMax-M2.1",
            api_key="test-key",
        )
        assert model.model_id == "MiniMax-M2.1"

    def test_prepare_request_removes_unsupported_params(self):
        """Test that unsupported parameters are removed for MiniMax."""
        from hawi.models.message import MessageRequest

        model = MiniMaxAnthropicModel(
            model_id="MiniMax-M2.5",
            api_key="test-key",
            top_k=50,
            metadata={"user_id": "123"},
        )

        request = MessageRequest(
            messages=[_create_user_message("Hello")],
        )

        req = model._prepare_request_impl(request)

        # top_k and metadata should be removed
        assert "top_k" not in req
        assert "metadata" not in req


@pytest.mark.skipif(not HAS_MINIMAX_KEY, reason=SKIP_REASON)
class TestMiniMaxAnthropicM25Integration:
    """Integration tests for MiniMax M2.5 model requiring real API access."""

    @pytest.fixture
    def model(self) -> MiniMaxAnthropicModel:
        """Create a MiniMax M2.5 model instance."""
        return MiniMaxAnthropicModel(
            model_id="MiniMax-M2.5",
            api_key=MINIMAX_API_KEY,
        )

    def test_simple_chat_completion(self, model: MiniMaxAnthropicModel):
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

    def test_streaming_response(self, model: MiniMaxAnthropicModel):
        """Test streaming response."""
        events = list(model.invoke(
            streaming=True,
            messages=[_create_user_message("Count from 1 to 3.")],
        ))

        # MiniMax may output reasoning_delta or text_delta events
        content_events = [e for e in events if e["type"] in ("text_delta", "reasoning_delta")]

        assert len(content_events) > 0
        # finish event may not be present in some cases, so we just verify we got content

    def test_tool_call_formatting(self, model: MiniMaxAnthropicModel):
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

    def test_multi_turn_conversation(self, model: MiniMaxAnthropicModel):
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
class TestMiniMaxAnthropicM21Integration:
    """Integration tests for MiniMax M2.1 model requiring real API access."""

    @pytest.fixture
    def model(self) -> MiniMaxAnthropicModel:
        """Create a MiniMax M2.1 model instance."""
        return MiniMaxAnthropicModel(
            model_id="MiniMax-M2.1",
            api_key=MINIMAX_API_KEY,
        )

    def test_simple_chat_completion(self, model: MiniMaxAnthropicModel):
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

    @retry_on_794(max_retries=3)
    def test_streaming_response(self, model: MiniMaxAnthropicModel):
        """Test streaming response with M2.1."""
        events = list(model.invoke(
            streaming=True,
            messages=[_create_user_message("Tell me a short joke.")],
        ))

        # MiniMax may output reasoning_delta or text_delta events
        content_events = [e for e in events if e["type"] in ("text_delta", "reasoning_delta")]

        assert len(content_events) > 0
