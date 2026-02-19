"""
StrandsModel 适配器单元测试

测试 strands 到 hawi 的转译适配器功能。
"""

import pytest
from typing import Any

from hawi.agent.models.strands import StrandsModel
from hawi.agent.message import (
    Message,
    ToolDefinition,
    ToolChoice,
)


class MockStrandsModel:
    """Mock strands model for testing"""

    def __init__(self, model_id: str = "mock-model") -> None:
        self.config = {"model_id": model_id}
        self.model_id = model_id
        self.last_call: dict[str, Any] | None = None

    def run_sync(self, **kwargs: Any) -> dict[str, Any]:
        self.last_call = kwargs
        return {
            "id": "msg_123",
            "content": [{"text": "Hello from strands"}],
            "stop_reason": "stop",
            "usage": {
                "inputTokens": 10,
                "outputTokens": 5,
            },
        }

    def run_stream(self, **kwargs: Any):
        self.last_call = kwargs
        # Standard Strands streaming events
        yield {"type": "contentBlockDelta", "delta": {"text": "Hello"}}
        yield {"type": "messageStop", "stopReason": "end_turn"}


class TestStrandsModel:
    """StrandsModel 基础测试"""

    def test_initialization(self):
        """测试适配器初始化"""
        strands_model = MockStrandsModel(model_id="test-model")
        adapter = StrandsModel(strands_model)

        assert adapter.model_id == "test-model"
        assert adapter.strands_model is strands_model

    def test_extract_model_id_from_config(self):
        """测试从 config 提取 model_id"""
        strands_model = MockStrandsModel()
        strands_model.config = {"model_id": "config-model"}
        del strands_model.model_id  # Remove attribute

        adapter = StrandsModel(strands_model)
        assert adapter.model_id == "config-model"

    def test_extract_model_id_fallback(self):
        """测试 model_id 回退"""
        strands_model = object()  # No config or model_id
        adapter = StrandsModel(strands_model)
        assert adapter.model_id == "unknown"


class TestMessageConversion:
    """消息格式转换测试"""

    def test_convert_user_message(self):
        """测试转换用户消息"""
        strands_model = MockStrandsModel()
        adapter = StrandsModel(strands_model)

        messages: list[Message] = [{
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}],
            "name": None,
            "tool_calls": None,
            "tool_call_id": None,
            "metadata": None,
        }]
        result = adapter._convert_messages_to_strands(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == [{"text": "Hello"}]

    def test_convert_assistant_message(self):
        """测试转换助手消息"""
        strands_model = MockStrandsModel()
        adapter = StrandsModel(strands_model)

        messages: list[Message] = [{
            "role": "assistant",
            "content": [{"type": "text", "text": "Hi there"}],
            "name": None,
            "tool_calls": None,
            "tool_call_id": None,
            "metadata": None,
        }]
        result = adapter._convert_messages_to_strands(messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == [{"text": "Hi there"}]

    def test_convert_image_part(self):
        """测试转换图片内容"""
        strands_model = MockStrandsModel()
        adapter = StrandsModel(strands_model)

        messages: list[Message] = [{
            "role": "user",
            "content": [{"type": "image", "source": {"url": "https://example.com/img.jpg", "detail": "auto"}}],
            "name": None,
            "tool_calls": None,
            "tool_call_id": None,
            "metadata": None,
        }]
        result = adapter._convert_messages_to_strands(messages)

        assert result[0]["content"] == [
            {"image": {"url": "https://example.com/img.jpg", "detail": "auto"}}
        ]

    def test_convert_tool_calls(self):
        """测试转换 tool_calls"""
        strands_model = MockStrandsModel()
        adapter = StrandsModel(strands_model)

        messages: list[Message] = [{
            "role": "assistant",
            "content": [],
            "name": None,
            "tool_calls": [
                {"type": "tool_call", "id": "call_123", "name": "get_weather", "arguments": {"city": "Beijing"}}
            ],
            "tool_call_id": None,
            "metadata": None,
        }]
        result = adapter._convert_messages_to_strands(messages)

        assert result[0]["role"] == "assistant"
        assert result[0]["tool_calls"] == [
            {
                "toolUse": {
                    "toolUseId": "call_123",
                    "name": "get_weather",
                    "input": {"city": "Beijing"},
                }
            }
        ]


class TestToolConversion:
    """工具定义转换测试"""

    def test_convert_tool_definition(self):
        """测试转换工具定义"""
        strands_model = MockStrandsModel()
        adapter = StrandsModel(strands_model)

        tool: ToolDefinition = {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather info",
            "schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        }

        result = adapter._convert_tool_definition_to_strands(tool)

        assert result["name"] == "get_weather"
        assert result["description"] == "Get weather info"
        assert result["inputSchema"]["json"]["type"] == "object"

    def test_convert_tool_choice_none(self):
        """测试转换 tool_choice none"""
        strands_model = MockStrandsModel()
        adapter = StrandsModel(strands_model)

        tool_choice: ToolChoice = {"type": "none", "name": None}
        result = adapter._convert_tool_choice_to_strands(tool_choice)

        assert result == {"type": "none"}

    def test_convert_tool_choice_auto(self):
        """测试转换 tool_choice auto"""
        strands_model = MockStrandsModel()
        adapter = StrandsModel(strands_model)

        tool_choice: ToolChoice = {"type": "auto", "name": None}
        result = adapter._convert_tool_choice_to_strands(tool_choice)

        assert result == {"type": "auto"}

    def test_convert_tool_choice_tool(self):
        """测试转换 tool_choice tool"""
        strands_model = MockStrandsModel()
        adapter = StrandsModel(strands_model)

        tool_choice: ToolChoice = {"type": "tool", "name": "get_weather"}
        result = adapter._convert_tool_choice_to_strands(tool_choice)

        assert result == {"type": "tool", "name": "get_weather"}


class TestResponseConversion:
    """响应转换测试"""

    def test_convert_simple_response(self):
        """测试转换简单响应"""
        strands_model = MockStrandsModel()
        adapter = StrandsModel(strands_model)

        strands_response = {
            "id": "msg_456",
            "content": [{"text": "Hello world"}],
            "stop_reason": "stop",
            "usage": {
                "inputTokens": 10,
                "outputTokens": 5,
            },
        }

        result = adapter._parse_response_impl(strands_response)

        assert result.id == "msg_456"
        assert len(result.content) == 1
        assert result.content[0]["type"] == "text"
        assert result.content[0]["text"] == "Hello world"  # type: ignore
        assert result.stop_reason == "end_turn"
        assert result.usage is not None
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5

    def test_convert_tool_call_response(self):
        """测试转换 tool call 响应 - Strands使用content块中的toolUse"""
        strands_model = MockStrandsModel()
        adapter = StrandsModel(strands_model)

        strands_response = {
            "id": "msg_789",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "call_123",
                        "name": "get_weather",
                        "input": {"city": "Beijing"},
                    }
                }
            ],
            "stop_reason": "tool_use",
        }

        result = adapter._parse_response_impl(strands_response)

        assert len(result.content) == 1
        assert result.content[0]["type"] == "tool_call"
        assert result.content[0]["name"] == "get_weather"  # type: ignore
        assert result.content[0]["arguments"] == {"city": "Beijing"}  # type: ignore
        assert result.stop_reason == "tool_use"


class TestStreamConversion:
    """流式事件转换测试"""

    def test_convert_content_block_delta_event(self):
        """测试转换 contentBlockDelta 事件"""
        strands_model = MockStrandsModel()
        adapter = StrandsModel(strands_model)

        state = {"index": 0, "block_started": False, "pending_usage": None}
        event = {"type": "contentBlockDelta", "delta": {"text": "Hello"}}
        result = list(adapter._convert_strands_event_to_stream_part(event, state))

        # contentBlockDelta 产生 start + delta (end由contentBlockStop产生)
        assert len(result) == 2
        assert result[0]["type"] == "text_delta"
        assert result[0]["is_start"] is True
        assert result[1]["type"] == "text_delta"
        assert result[1]["delta"] == "Hello"

    def test_convert_message_stop_event(self):
        """测试转换 messageStop 事件"""
        strands_model = MockStrandsModel()
        adapter = StrandsModel(strands_model)

        state = {"index": 0, "block_started": False, "pending_usage": None}
        event = {"type": "messageStop", "stopReason": "end_turn"}
        result = list(adapter._convert_strands_event_to_stream_part(event, state))

        assert len(result) == 1
        assert result[0]["type"] == "finish"
        assert result[0]["stop_reason"] == "end_turn"

    def test_convert_legacy_finish_event(self):
        """测试转换旧版 finish 事件（向后兼容）"""
        strands_model = MockStrandsModel()
        adapter = StrandsModel(strands_model)

        state = {"index": 0, "block_started": False, "pending_usage": None}
        event = {"type": "finish", "stop_reason": "stop"}
        result = list(adapter._convert_strands_event_to_stream_part(event, state))

        assert len(result) == 1
        assert result[0]["type"] == "finish"
        # stop_reason 会被映射为 end_turn
        assert result[0]["stop_reason"] == "end_turn"


class TestIntegration:
    """集成测试"""

    def test_invoke_delegation(self):
        """测试 invoke 委托给 strands model"""
        strands_model = MockStrandsModel()
        adapter = StrandsModel(strands_model)

        messages: list[Message] = [{
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}],
            "name": None,
            "tool_calls": None,
            "tool_call_id": None,
            "metadata": None,
        }]
        response = adapter.invoke(messages=messages)

        # Verify strands model was called
        assert strands_model.last_call is not None
        assert strands_model.last_call["messages"][0]["role"] == "user"

        # Verify response
        assert response.id == "msg_123"
        assert response.content[0]["type"] == "text"

    def test_stream_delegation(self):
        """测试 stream 委托给 strands model"""
        strands_model = MockStrandsModel()
        adapter = StrandsModel(strands_model)

        messages: list[Message] = [{
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}],
            "name": None,
            "tool_calls": None,
            "tool_call_id": None,
            "metadata": None,
        }]
        events = list(adapter.stream(messages=messages))

        # Verify strands model was called
        assert strands_model.last_call is not None

        # Verify events (start + delta from contentBlockDelta + finish from messageStop = 3 events)
        assert len(events) == 3
        assert events[0]["type"] == "text_delta"
        assert events[0]["is_start"] is True
        assert events[1]["type"] == "text_delta"
        assert events[1]["delta"] == "Hello"
        assert events[2]["type"] == "finish"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
