"""Tests for StrandsModel adapter.

Tests the Hawi StrandsModel adapter to ensure it correctly:
1. Maps TokenUsage fields from Strands camelCase format
2. Parses tool calls from content blocks (not tool_calls field)
3. Handles streaming events (contentBlockDelta, metadata, messageStop)
4. Converts between Hawi and Strands message formats
"""

import pytest
from unittest.mock import Mock, MagicMock

from hawi.agent.models.strands import StrandsModel
from hawi.agent.message import (
    MessageRequest,
    TokenUsage,
    ContentPart,
    ToolCallPart,
    ToolResultPart,
)


class TestStrandsModelTokenUsage:
    """Test TokenUsage field mapping from Strands camelCase format."""

    def test_parse_response_with_camel_case_usage(self):
        """Strands uses camelCase field names for usage."""
        mock_strands_model = Mock()
        mock_strands_model.config = {"model_id": "test-model"}
        
        model = StrandsModel(mock_strands_model)
        
        # Strands response with camelCase usage fields
        strands_response = {
            "id": "test-123",
            "content": [{"text": "Hello"}],
            "usage": {
                "inputTokens": 100,
                "outputTokens": 50,
                "totalTokens": 150,
                "cacheWriteInputTokens": 80,
                "cacheReadInputTokens": 20,
            },
            "stop_reason": "end_turn",
        }
        
        response = model._parse_response_impl(strands_response)
        
        assert response.usage is not None
        assert response.usage.input_tokens == 100
        assert response.usage.output_tokens == 50
        assert response.usage.cache_write_tokens == 80
        assert response.usage.cache_read_tokens == 20

    def test_parse_response_without_cache_tokens(self):
        """Response without cache tokens should have None for cache fields."""
        mock_strands_model = Mock()
        mock_strands_model.config = {"model_id": "test-model"}
        
        model = StrandsModel(mock_strands_model)
        
        strands_response = {
            "id": "test-123",
            "content": [{"text": "Hello"}],
            "usage": {
                "inputTokens": 100,
                "outputTokens": 50,
                "totalTokens": 150,
            },
            "stop_reason": "end_turn",
        }
        
        response = model._parse_response_impl(strands_response)
        
        assert response.usage is not None
        assert response.usage.input_tokens == 100
        assert response.usage.output_tokens == 50
        assert response.usage.cache_write_tokens is None
        assert response.usage.cache_read_tokens is None

    def test_parse_response_without_usage(self):
        """Response without usage should have None usage."""
        mock_strands_model = Mock()
        mock_strands_model.config = {"model_id": "test-model"}
        
        model = StrandsModel(mock_strands_model)
        
        strands_response = {
            "id": "test-123",
            "content": [{"text": "Hello"}],
            "stop_reason": "end_turn",
        }
        
        response = model._parse_response_impl(strands_response)
        
        assert response.usage is None


class TestStrandsModelToolCalls:
    """Test tool call parsing from content blocks."""

    def test_parse_tool_use_from_content_block(self):
        """Strands returns toolUse as content block, not in tool_calls field."""
        mock_strands_model = Mock()
        mock_strands_model.config = {"model_id": "test-model"}
        
        model = StrandsModel(mock_strands_model)
        
        strands_response = {
            "id": "test-123",
            "content": [
                {"text": "I'll help you with that."},
                {
                    "toolUse": {
                        "toolUseId": "tool-123",
                        "name": "calculator",
                        "input": {"expression": "1 + 1"},
                    }
                },
            ],
            "stop_reason": "tool_use",
        }
        
        response = model._parse_response_impl(strands_response)
        
        # Should have 2 content parts: text + tool_call (tool_use is parsed from content block)
        tool_call_parts = [p for p in response.content if p.get("type") == "tool_call"]
        assert len(tool_call_parts) == 1
        
        tool_call_part = tool_call_parts[0]
        assert tool_call_part["id"] == "tool-123"
        assert tool_call_part["name"] == "calculator"
        assert tool_call_part["arguments"] == {"expression": "1 + 1"}

    def test_parse_multiple_tool_uses(self):
        """Parse multiple toolUse blocks from content."""
        mock_strands_model = Mock()
        mock_strands_model.config = {"model_id": "test-model"}
        
        model = StrandsModel(mock_strands_model)
        
        strands_response = {
            "id": "test-123",
            "content": [
                {"toolUse": {"toolUseId": "tool-1", "name": "weather", "input": {"city": "Beijing"}}},
                {"toolUse": {"toolUseId": "tool-2", "name": "time", "input": {"timezone": "UTC"}}},
            ],
            "stop_reason": "tool_use",
        }
        
        response = model._parse_response_impl(strands_response)
        
        # Should have exactly 2 tool_call parts (no duplicates)
        tool_call_parts = [p for p in response.content if p.get("type") == "tool_call"]
        assert len(tool_call_parts) == 2
        assert tool_call_parts[0]["name"] == "weather"
        assert tool_call_parts[1]["name"] == "time"

    def test_parse_tool_use_with_string_input(self):
        """Tool input might be a JSON string instead of dict."""
        mock_strands_model = Mock()
        mock_strands_model.config = {"model_id": "test-model"}
        
        model = StrandsModel(mock_strands_model)
        
        strands_response = {
            "id": "test-123",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "tool-123",
                        "name": "search",
                        "input": '{"query": "test"}',  # String input
                    }
                },
            ],
            "stop_reason": "tool_use",
        }
        
        response = model._parse_response_impl(strands_response)
        
        tool_call = response.content[0]
        assert tool_call["arguments"] == {"query": "test"}


class TestStrandsModelStreaming:
    """Test streaming event conversion."""

    def test_content_block_delta_text(self):
        """Convert contentBlockDelta with text to text_delta."""
        mock_strands_model = Mock()
        mock_strands_model.config = {"model_id": "test-model"}
        
        model = StrandsModel(mock_strands_model)
        
        # First event: contentBlockDelta starts the block
        event1 = {
            "type": "contentBlockDelta",
            "delta": {"text": "Hello world"},
        }
        
        state = {"index": 0, "block_started": False, "pending_usage": None}
        parts = list(model._convert_strands_event_to_stream_part(event1, state))
        
        # Should yield: start, content (no end - that comes from contentBlockStop)
        assert len(parts) == 2
        assert parts[0]["type"] == "text_delta"
        assert parts[0]["is_start"] is True
        assert parts[1]["delta"] == "Hello world"
        assert parts[1]["is_end"] is False
        
        # Second event: contentBlockStop ends the block
        event2 = {"type": "contentBlockStop"}
        parts = list(model._convert_strands_event_to_stream_part(event2, state))
        
        assert len(parts) == 1
        assert parts[0]["type"] == "text_delta"
        assert parts[0]["is_end"] is True

    def test_content_block_start_tool_use(self):
        """Convert contentBlockStart with toolUse to tool_call_delta start."""
        mock_strands_model = Mock()
        mock_strands_model.config = {"model_id": "test-model"}
        
        model = StrandsModel(mock_strands_model)
        
        event = {
            "type": "contentBlockStart",
            "start": {
                "toolUse": {
                    "toolUseId": "tool-123",
                    "name": "calculator",
                }
            },
        }
        
        state = {"index": 0, "block_started": False, "pending_usage": None}
        parts = list(model._convert_strands_event_to_stream_part(event, state))
        
        assert len(parts) == 1
        assert parts[0]["type"] == "tool_call_delta"
        assert parts[0]["is_start"] is True
        assert parts[0]["id"] == "tool-123"
        assert parts[0]["name"] == "calculator"

    def test_content_block_delta_tool_input(self):
        """Convert contentBlockDelta with toolUse input to tool_call_delta."""
        mock_strands_model = Mock()
        mock_strands_model.config = {"model_id": "test-model"}
        
        model = StrandsModel(mock_strands_model)
        
        event = {
            "type": "contentBlockDelta",
            "delta": {"toolUse": {"input": '{"x": 1}'}},
        }
        
        state = {"index": 0, "block_started": False, "pending_usage": None}
        parts = list(model._convert_strands_event_to_stream_part(event, state))
        
        assert len(parts) == 1
        assert parts[0]["type"] == "tool_call_delta"
        assert parts[0]["arguments_delta"] == '{"x": 1}'

    def test_metadata_event_with_usage(self):
        """Extract usage from metadata event with camelCase fields."""
        mock_strands_model = Mock()
        mock_strands_model.config = {"model_id": "test-model"}
        
        model = StrandsModel(mock_strands_model)
        
        event = {
            "type": "metadata",
            "metadata": {
                "usage": {
                    "inputTokens": 100,
                    "outputTokens": 50,
                    "cacheWriteInputTokens": 80,
                    "cacheReadInputTokens": 20,
                }
            },
        }
        
        state = {"index": 0, "block_started": False, "pending_usage": None}
        list(model._convert_strands_event_to_stream_part(event, state))
        
        # Usage should be stored in state
        assert state["pending_usage"] is not None
        assert state["pending_usage"]["input_tokens"] == 100
        assert state["pending_usage"]["output_tokens"] == 50
        assert state["pending_usage"]["cache_write_tokens"] == 80
        assert state["pending_usage"]["cache_read_tokens"] == 20

    def test_message_stop_event(self):
        """Convert messageStop to finish event."""
        mock_strands_model = Mock()
        mock_strands_model.config = {"model_id": "test-model"}
        
        model = StrandsModel(mock_strands_model)
        
        pending_usage = {"input_tokens": 100, "output_tokens": 50}
        state = {"index": 0, "block_started": False, "pending_usage": pending_usage}
        
        event = {
            "type": "messageStop",
            "stopReason": "end_turn",
        }
        
        parts = list(model._convert_strands_event_to_stream_part(event, state))
        
        assert len(parts) == 1
        assert parts[0]["type"] == "finish"
        assert parts[0]["stop_reason"] == "end_turn"
        assert parts[0]["usage"] == pending_usage
        assert state["pending_usage"] is None  # Should be cleared

    def test_stop_reason_mapping(self):
        """Test stop reason mapping from Strands to Hawi format."""
        mock_strands_model = Mock()
        mock_strands_model.config = {"model_id": "test-model"}
        
        model = StrandsModel(mock_strands_model)
        
        test_cases = [
            ("end_turn", "end_turn"),
            ("tool_use", "tool_use"),
            ("max_tokens", "max_tokens"),
            ("stop", "end_turn"),  # Mapped
            ("tool_calls", "tool_use"),  # Mapped
            ("length", "max_tokens"),  # Mapped
        ]
        
        for strands_reason, expected_hawi in test_cases:
            result = model._map_strands_stop_reason(strands_reason)
            assert result == expected_hawi, f"Failed for {strands_reason}"


class TestStrandsModelMessageConversion:
    """Test message format conversion."""

    def test_convert_hawi_message_to_strands(self):
        """Convert Hawi Message to Strands format."""
        mock_strands_model = Mock()
        mock_strands_model.config = {"model_id": "test-model"}
        
        model = StrandsModel(mock_strands_model)
        
        hawi_message = {
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}],
            "name": None,
            "tool_calls": None,
            "tool_call_id": None,
            "metadata": None,
        }
        
        strands_msg = model._convert_single_message_to_strands(hawi_message)
        
        assert strands_msg["role"] == "user"
        assert strands_msg["content"] == [{"text": "Hello"}]

    def test_convert_tool_call_message(self):
        """Convert Hawi assistant message with tool_calls."""
        mock_strands_model = Mock()
        mock_strands_model.config = {"model_id": "test-model"}
        
        model = StrandsModel(mock_strands_model)
        
        hawi_message = {
            "role": "assistant",
            "content": [],
            "tool_calls": [
                {"type": "tool_call", "id": "tool-1", "name": "weather", "arguments": {"city": "Beijing"}}
            ],
            "name": None,
            "tool_call_id": None,
            "metadata": None,
        }
        
        strands_msg = model._convert_single_message_to_strands(hawi_message)
        
        assert strands_msg["role"] == "assistant"
        assert "tool_calls" in strands_msg
        assert strands_msg["tool_calls"][0]["toolUse"]["toolUseId"] == "tool-1"

    def test_convert_tool_result_message(self):
        """Convert Hawi tool result message."""
        mock_strands_model = Mock()
        mock_strands_model.config = {"model_id": "test-model"}
        
        model = StrandsModel(mock_strands_model)
        
        hawi_message = {
            "role": "tool",
            "content": [{"type": "text", "text": "Result: 25°C"}],
            "tool_call_id": "tool-1",
            "name": None,
            "tool_calls": None,
            "metadata": None,
        }
        
        strands_msg = model._convert_single_message_to_strands(hawi_message)
        
        assert strands_msg["role"] == "tool"
        assert strands_msg["tool_call_id"] == "tool-1"

    def test_convert_system_prompt(self):
        """Convert Hawi system prompt to Strands format."""
        mock_strands_model = Mock()
        mock_strands_model.config = {"model_id": "test-model"}
        
        model = StrandsModel(mock_strands_model)
        
        request = MessageRequest(
            messages=[],
            system=[{"type": "text", "text": "You are helpful."}],
        )
        
        strands_req = model._prepare_request_impl(request)
        
        assert strands_req["system_prompt"] == "You are helpful."


class TestStrandsModelEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_content_response(self):
        """Handle response with empty content."""
        mock_strands_model = Mock()
        mock_strands_model.config = {"model_id": "test-model"}
        
        model = StrandsModel(mock_strands_model)
        
        strands_response = {
            "id": "test-123",
            "content": [],
            "stop_reason": "end_turn",
        }
        
        response = model._parse_response_impl(strands_response)
        
        assert response.content == []

    def test_unknown_content_block(self):
        """Handle unknown content block types gracefully."""
        mock_strands_model = Mock()
        mock_strands_model.config = {"model_id": "test-model"}
        
        model = StrandsModel(mock_strands_model)
        
        strands_response = {
            "id": "test-123",
            "content": [
                {"unknownBlock": {"data": "test"}},
                {"text": "Hello"},
            ],
            "stop_reason": "end_turn",
        }
        
        response = model._parse_response_impl(strands_response)
        
        # Unknown block should be skipped, text should be preserved
        assert len(response.content) == 1
        assert response.content[0]["type"] == "text"

    def test_backward_compatible_finish_event(self):
        """Still support old custom finish event format."""
        mock_strands_model = Mock()
        mock_strands_model.config = {"model_id": "test-model"}
        
        model = StrandsModel(mock_strands_model)
        
        pending_usage = {"input_tokens": 100, "output_tokens": 50}
        state = {"index": 0, "block_started": False, "pending_usage": pending_usage}
        
        # Old format event
        event = {
            "type": "finish",
            "stop_reason": "end_turn",
        }
        
        parts = list(model._convert_strands_event_to_stream_part(event, state))
        
        assert parts[0]["type"] == "finish"
        assert parts[0]["stop_reason"] == "end_turn"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
