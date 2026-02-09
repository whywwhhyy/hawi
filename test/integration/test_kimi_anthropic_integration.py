"""
KimiAnthropicModel 集成测试

API Key 来源 (按优先级):
1. KIMI_ANTHROPIC_API_KEY 或 KIMI_API_KEY 环境变量
2. apikey.yaml 文件中的 kimi-anthropic 配置

参考文档:
- https://platform.moonshot.ai/docs/guide/kimi-k2-5-quickstart
"""

import os
import sys
import pytest
from typing import Generator
from unittest.mock import Mock, patch, MagicMock, AsyncMock

from strands.types.content import Message

from hawi.agent.models.kimi_anthropic import KimiAnthropicModel
from test.integration import get_kimi_anthropic_api_key

pytest.skip("非必要不测试这个api", allow_module_level=True)

# 检查是否设置了 API Key (从环境变量或 apikey.yaml)
KIMI_ANTHROPIC_API_KEY = get_kimi_anthropic_api_key()
HAS_KIMI_ANTHROPIC_KEY = KIMI_ANTHROPIC_API_KEY is not None and KIMI_ANTHROPIC_API_KEY.strip() != ""


class TestKimiAnthropicModelUnit:
    """KimiAnthropicModel 单元测试 (无需真实 API)"""

    def test_serialize_event_basic(self):
        """测试基础事件序列化"""
        model = KimiAnthropicModel(
            client_args={"api_key": "test"},
            model_id="kimi-k2.5",
            max_tokens=1024
        )

        # 模拟事件对象，不包含 content_block 或 delta
        mock_event = Mock(spec=["type", "index"])
        mock_event.type = "content_block_start"
        mock_event.index = 0

        result = model._serialize_event(mock_event)

        assert result["type"] == "content_block_start"
        assert result["index"] == 0

    def test_serialize_content_block_text(self):
        """测试文本内容块序列化"""
        model = KimiAnthropicModel(
            client_args={"api_key": "test"},
            model_id="kimi-k2.5",
            max_tokens=1024
        )

        mock_block = Mock()
        mock_block.type = "text"
        mock_block.text = "Hello world"
        mock_block.citations = None

        result = model._serialize_content_block(mock_block)

        assert result["type"] == "text"
        assert result["text"] == "Hello world"

    def test_serialize_content_block_with_citations(self):
        """测试带 citations 的内容块序列化"""
        model = KimiAnthropicModel(
            client_args={"api_key": "test"},
            model_id="kimi-k2.5",
            max_tokens=1024
        )

        mock_citation = Mock()
        mock_citation.type = "char_location"
        mock_citation.cited_text = "Source text"
        mock_citation.document_index = 0
        mock_citation.document_title = "Doc 1"
        mock_citation.start_char_index = 10
        mock_citation.end_char_index = 20

        mock_block = Mock()
        mock_block.type = "text"
        mock_block.text = "Some text"
        mock_block.citations = [mock_citation]

        result = model._serialize_content_block(mock_block)

        assert result["type"] == "text"
        assert "citations" in result
        assert len(result["citations"]) == 1
        assert result["citations"][0]["cited_text"] == "Source text"

    def test_serialize_content_block_tool_use(self):
        """测试 tool_use 内容块序列化"""
        model = KimiAnthropicModel(
            client_args={"api_key": "test"},
            model_id="kimi-k2.5",
            max_tokens=1024
        )

        mock_block = Mock()
        mock_block.type = "tool_use"
        mock_block.id = "tool_123"
        mock_block.name = "test_tool"
        mock_block.input = {"param": "value"}
        mock_block.citations = None

        result = model._serialize_content_block(mock_block)

        assert result["type"] == "tool_use"
        assert result["id"] == "tool_123"
        assert result["name"] == "test_tool"
        assert result["input"] == {"param": "value"}

    def test_serialize_content_block_thinking(self):
        """测试 thinking 内容块序列化"""
        model = KimiAnthropicModel(
            client_args={"api_key": "test"},
            model_id="kimi-k2.5",
            max_tokens=1024
        )

        mock_block = Mock()
        mock_block.type = "thinking"
        mock_block.thinking = "Step 1..."
        mock_block.signature = "sig123"
        mock_block.citations = None

        result = model._serialize_content_block(mock_block)

        assert result["type"] == "thinking"
        assert result["thinking"] == "Step 1..."
        assert result["signature"] == "sig123"

    def test_serialize_citation_full(self):
        """测试完整引用序列化"""
        model = KimiAnthropicModel(
            client_args={"api_key": "test"},
            model_id="kimi-k2.5",
            max_tokens=1024
        )

        mock_citation = Mock()
        mock_citation.type = "page_location"
        mock_citation.cited_text = "Page content"
        mock_citation.document_index = 1
        mock_citation.document_title = "Document Title"
        mock_citation.start_page_number = 5
        mock_citation.end_page_number = 6
        mock_citation.start_char_index = 100
        mock_citation.end_char_index = 200

        result = model._serialize_citation(mock_citation)

        assert result["type"] == "page_location"
        assert result["cited_text"] == "Page content"
        assert result["document_index"] == 1
        assert result["document_title"] == "Document Title"
        assert result["start_page_number"] == 5
        assert result["end_page_number"] == 6

    def test_serialize_delta_text(self):
        """测试文本 delta 序列化"""
        model = KimiAnthropicModel(
            client_args={"api_key": "test"},
            model_id="kimi-k2.5",
            max_tokens=1024
        )

        mock_delta = Mock()
        mock_delta.type = "text_delta"
        mock_delta.text = "More text"

        result = model._serialize_delta(mock_delta)

        assert result["type"] == "text_delta"
        assert result["text"] == "More text"

    def test_serialize_delta_partial_json(self):
        """测试 partial_json delta 序列化"""
        model = KimiAnthropicModel(
            client_args={"api_key": "test"},
            model_id="kimi-k2.5",
            max_tokens=1024
        )

        mock_delta = Mock()
        mock_delta.type = "input_json_delta"
        mock_delta.partial_json = '{"key": "value"}'

        result = model._serialize_delta(mock_delta)

        assert result["type"] == "input_json_delta"
        assert result["partial_json"] == '{"key": "value"}'

    def test_serialize_usage(self):
        """测试 usage 序列化"""
        model = KimiAnthropicModel(
            client_args={"api_key": "test"},
            model_id="kimi-k2.5",
            max_tokens=1024
        )

        mock_usage = Mock()
        mock_usage.input_tokens = 100
        mock_usage.output_tokens = 50
        mock_usage.cache_creation_input_tokens = 20
        mock_usage.cache_read_input_tokens = 10

        result = model._serialize_usage(mock_usage)

        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert result["cache_creation_input_tokens"] == 20
        assert result["cache_read_input_tokens"] == 10

    def test_serialize_none_values(self):
        """测试 None 值处理"""
        model = KimiAnthropicModel(
            client_args={"api_key": "test"},
            model_id="kimi-k2.5",
            max_tokens=1024
        )

        assert model._serialize_content_block(None) == {}
        assert model._serialize_delta(None) == {}

    def test_serialize_event_with_content_block(self):
        """测试带 content_block 的事件序列化"""
        model = KimiAnthropicModel(
            client_args={"api_key": "test"},
            model_id="kimi-k2.5",
            max_tokens=1024
        )

        mock_block = Mock()
        mock_block.type = "text"
        mock_block.text = "Hello"
        mock_block.citations = None

        mock_event = Mock()
        mock_event.type = "content_block_start"
        mock_event.index = 0
        mock_event.content_block = mock_block

        result = model._serialize_event(mock_event)

        assert result["type"] == "content_block_start"
        assert result["index"] == 0
        assert "content_block" in result
        assert result["content_block"]["type"] == "text"

    def test_serialize_event_with_delta(self):
        """测试带 delta 的事件序列化"""
        model = KimiAnthropicModel(
            client_args={"api_key": "test"},
            model_id="kimi-k2.5",
            max_tokens=1024
        )

        mock_delta = Mock(spec=["type", "text", "partial_json", "thinking", "signature"])
        mock_delta.type = "text_delta"
        mock_delta.text = " delta"
        mock_delta.partial_json = None
        mock_delta.thinking = None
        mock_delta.signature = None

        mock_event = Mock(spec=["type", "index", "delta", "content_block", "message"])
        mock_event.type = "content_block_delta"
        mock_event.index = 0
        mock_event.delta = mock_delta
        mock_event.content_block = None
        mock_event.message = None

        result = model._serialize_event(mock_event)

        assert result["type"] == "content_block_delta"
        assert "delta" in result
        assert result["delta"]["text"] == " delta"


@pytest.mark.skipif(not HAS_KIMI_ANTHROPIC_KEY, reason="KIMI_ANTHROPIC_API_KEY not set")
class TestKimiAnthropicModelIntegration:
    """KimiAnthropicModel 集成测试 (需要真实 API Key)"""

    @pytest.fixture
    def model(self) -> Generator[KimiAnthropicModel, None, None]:
        """创建 Kimi Anthropic 模型实例"""
        m = KimiAnthropicModel(
            client_args={
                "api_key": KIMI_ANTHROPIC_API_KEY,
                "base_url": "https://api.kimi.com/coding/",
            },
            model_id="kimi-k2.5",
            max_tokens=1024
        )
        yield m

    @pytest.mark.asyncio
    async def test_simple_chat_completion(self, model: KimiAnthropicModel):
        """测试简单对话完成"""
        messages: list[Message] = [{"role": "user", "content": [{"text": "Say hello"}]}]

        chunks = []
        async for chunk in model.stream(messages):
            chunks.append(chunk)

        # 验证收到了消息
        assert len(chunks) > 0
        # 应该包含 message_start 和 message_stop
        chunk_types = [c.get("chunk_type") for c in chunks]
        assert "message_start" in chunk_types
        assert "message_stop" in chunk_types

    @pytest.mark.asyncio
    async def test_stream_response_no_pydantic_warnings(self, model: KimiAnthropicModel):
        """测试流式响应不产生 Pydantic 警告"""
        import warnings

        messages: list[Message] = [{"role": "user", "content": [{"text": "What is 2+2?"}]}]

        # 捕获所有警告
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            chunks = []
            async for chunk in model.stream(messages):
                chunks.append(chunk)

            # 检查没有 Pydantic 序列化警告
            pydantic_warnings = [
                warning for warning in w
                if "PydanticSerialization" in str(warning.category) or
                   "UnexpectedValue" in str(warning.message)
            ]

            assert len(pydantic_warnings) == 0, f"Found Pydantic warnings: {pydantic_warnings}"

    @pytest.mark.asyncio
    async def test_stream_content_blocks(self, model: KimiAnthropicModel):
        """测试流式内容块处理"""
        messages: list[Message] = [{"role": "user", "content": [{"text": "Count to 3"}]}]

        content_blocks = []
        text_deltas = []

        async for chunk in model.stream(messages):
            chunk_type = chunk.get("chunk_type")
            if chunk_type == "content_start":
                content_blocks.append(chunk)
            elif chunk_type == "content_delta":
                text_deltas.append(chunk)

        # 应该收到内容块
        assert len(content_blocks) > 0 or len(text_deltas) > 0

    @pytest.mark.asyncio
    async def test_usage_in_response(self, model: KimiAnthropicModel):
        """测试响应中包含 usage 信息"""
        messages: list[Message] = [{"role": "user", "content": [{"text": "Hello"}]}]

        usage_chunks = []
        async for chunk in model.stream(messages):
            if chunk.get("chunk_type") == "metadata":
                usage_chunks.append(chunk)

        # 应该收到 usage 信息
        if usage_chunks:
            usage = usage_chunks[-1].get("data", {})
            # 可能有 input_tokens 或 output_tokens
            assert "input_tokens" in usage or "output_tokens" in usage


class TestKimiAnthropicModelEdgeCases:
    """KimiAnthropicModel 边界情况测试"""

    def test_serialize_message_with_stop_reason(self):
        """测试带 stop_reason 的消息序列化"""
        model = KimiAnthropicModel(
            client_args={"api_key": "test"},
            model_id="kimi-k2.5",
            max_tokens=1024
        )

        mock_message = Mock()
        mock_message.stop_reason = "end_turn"

        result = model._serialize_message(mock_message)
        assert result["stop_reason"] == "end_turn"

    def test_serialize_message_minimal(self):
        """测试最小化消息序列化"""
        model = KimiAnthropicModel(
            client_args={"api_key": "test"},
            model_id="kimi-k2.5",
            max_tokens=1024
        )

        mock_message = Mock()
        mock_message.stop_reason = None

        result = model._serialize_message(mock_message)
        assert result == {}

    def test_serialize_citation_minimal(self):
        """测试最小化引用序列化"""
        model = KimiAnthropicModel(
            client_args={"api_key": "test"},
            model_id="kimi-k2.5",
            max_tokens=1024
        )

        # 使用 spec=True 防止 Mock 自动创建属性
        mock_citation = Mock(spec=True)
        mock_citation.type = None
        mock_citation.cited_text = None

        result = model._serialize_citation(mock_citation)
        assert result == {}


class TestKimiAnthropicMultiTurnToolCalls:
    """Kimi Anthropic 多轮对话带工具调用测试 (无需真实 API)"""

    def test_multi_turn_conversation_formatting(self):
        """测试多轮对话的消息格式化"""
        from strands.types.content import Message

        messages: list[Message] = [
            # Round 1
            {"role": "user", "content": [{"text": "What is 15 * 6?"}]},
            {
                "role": "assistant",
                "content": [
                    {"text": "Let me calculate that for you."},
                    {"toolUse": {"toolUseId": "call-1", "name": "calculator", "input": {"expression": "15*6"}}}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"toolResult": {"toolUseId": "call-1", "status": "success", "content": [{"json": {"result": 90}}]}}
                ]
            },
            {"role": "assistant", "content": [{"text": "15 * 6 = 90"}]},
            # Round 2
            {"role": "user", "content": [{"text": "Now divide by 3"}]},
        ]

        # 使用 format_request 来格式化完整请求
        model = KimiAnthropicModel(
            client_args={"api_key": "test"},
            model_id="kimi-k2.5",
            max_tokens=1024
        )

        formatted = model.format_request(messages)

        # 验证请求结构
        assert "messages" in formatted
        assert len(formatted["messages"]) >= 4

    def test_tool_use_content_block_serialization(self):
        """测试 tool_use 内容块的序列化（多轮场景）"""
        model = KimiAnthropicModel(
            client_args={"api_key": "test"},
            model_id="kimi-k2.5",
            max_tokens=1024
        )

        # 模拟 tool_use 内容块
        mock_block = Mock()
        mock_block.type = "tool_use"
        mock_block.id = "tool_call_123"
        mock_block.name = "calculator"
        mock_block.input = {"expression": "15*6"}

        result = model._serialize_content_block(mock_block)

        assert result["type"] == "tool_use"
        assert result["id"] == "tool_call_123"
        assert result["name"] == "calculator"
        assert result["input"] == {"expression": "15*6"}

    def test_multi_turn_with_citations(self):
        """测试多轮对话中带 citations 的内容处理"""
        model = KimiAnthropicModel(
            client_args={"api_key": "test"},
            model_id="kimi-k2.5",
            max_tokens=1024
        )

        # 模拟带 citation 的文本块（Kimi API 特有）
        mock_citation = Mock()
        mock_citation.type = "char_location"
        mock_citation.cited_text = "According to the document..."
        mock_citation.document_index = 0
        mock_citation.document_title = "Reference Doc"

        mock_block = Mock()
        mock_block.type = "text"
        mock_block.text = "Based on the reference, the answer is 42."
        mock_block.citations = [mock_citation]

        result = model._serialize_content_block(mock_block)

        assert result["type"] == "text"
        assert "citations" in result
        assert len(result["citations"]) == 1
        assert result["citations"][0]["document_title"] == "Reference Doc"

    def test_stream_event_with_tool_use_delta(self):
        """测试工具调用 delta 事件的序列化"""
        model = KimiAnthropicModel(
            client_args={"api_key": "test"},
            model_id="kimi-k2.5",
            max_tokens=1024
        )

        # 模拟 input_json_delta（工具参数增量）
        mock_delta = Mock()
        mock_delta.type = "input_json_delta"
        mock_delta.partial_json = '{"expre'

        result = model._serialize_delta(mock_delta)

        assert result["type"] == "input_json_delta"
        assert result["partial_json"] == '{"expre'


@pytest.mark.skipif(not HAS_KIMI_ANTHROPIC_KEY, reason="KIMI_ANTHROPIC_API_KEY not set")
class TestKimiAnthropicMultiTurnToolCallsIntegration:
    """Kimi Anthropic 多轮工具调用集成测试 (需要真实 API)"""

    @pytest.fixture
    def model(self) -> Generator[KimiAnthropicModel, None, None]:
        """创建 Kimi Anthropic 模型实例"""
        m = KimiAnthropicModel(
            client_args={
                "api_key": KIMI_ANTHROPIC_API_KEY,
                "base_url": "https://api.kimi.com/coding/",
            },
            model_id="kimi-k2.5",
            max_tokens=1024
        )
        yield m

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, model: KimiAnthropicModel):
        """测试多轮对话"""
        from strands.types.content import Message

        # 第一轮
        messages: list[Message] = [
            {"role": "user", "content": [{"text": "What is 25 * 4?"}]}
        ]

        first_chunks = []
        async for chunk in model.stream(messages):
            first_chunks.append(chunk)

        assert len(first_chunks) > 0

        # 第二轮
        messages.extend([
            {"role": "assistant", "content": [{"text": "25 * 4 = 100"}]},
            {"role": "user", "content": [{"text": "Add 50 to that result"}]}
        ])

        second_chunks = []
        async for chunk in model.stream(messages):
            second_chunks.append(chunk)

        assert len(second_chunks) > 0

    @pytest.mark.asyncio
    async def test_stream_no_pydantic_warnings_multi_turn(self, model: KimiAnthropicModel):
        """测试多轮对话不产生 Pydantic 警告"""
        from strands.types.content import Message
        import warnings

        messages: list[Message] = [
            {"role": "user", "content": [{"text": "Hello"}]},
            {"role": "assistant", "content": [{"text": "Hi there!"}]},
            {"role": "user", "content": [{"text": "How are you?"}]}
        ]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            chunks = []
            async for chunk in model.stream(messages):
                chunks.append(chunk)

            pydantic_warnings = [
                warning for warning in w
                if "PydanticSerialization" in str(warning.category) or
                   "UnexpectedValue" in str(warning.message)
            ]

            assert len(pydantic_warnings) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
