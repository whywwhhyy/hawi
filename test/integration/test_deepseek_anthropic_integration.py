"""
DeepSeekAnthropicModel 集成测试

API Key 来源 (按优先级):
1. DEEPSEEK_API_KEY 环境变量
2. apikey.yaml 文件中的 deepseek 配置

参考文档:
- https://api-docs.deepseek.com/guides/anthropic_api
- https://api-docs.deepseek.com/guides/thinking_mode
"""

import os
import sys
import pytest
from typing import Generator, Any
from unittest.mock import Mock, patch, MagicMock, AsyncMock

from strands.types.content import Message
from strands.types.tools import ToolResult

from hawi.agent.models.deepseek_anthropic import (
    DeepSeekAnthropicModel,
    create_deepseek_anthropic_model,
    create_deepseek_anthropic_reasoner,
    UNSUPPORTED_REASONER_PARAMS,
    ERROR_REASONER_PARAMS,
)
from test.integration import get_deepseek_api_key

# 检查是否设置了 API Key (从环境变量或 apikey.yaml)
DEEPSEEK_API_KEY = get_deepseek_api_key()
HAS_DEEPSEEK_KEY = DEEPSEEK_API_KEY is not None and DEEPSEEK_API_KEY.strip() != ""


class TestDeepSeekAnthropicModelUnit:
    """DeepSeekAnthropicModel 单元测试 (无需真实 API)"""

    def test_clean_request_params_removes_unsupported(self):
        """测试清理请求中不支持的参数"""
        model = DeepSeekAnthropicModel(
            client_args={"api_key": "test"},
            model_id="deepseek-chat",
            max_tokens=1024
        )

        request = {
            "model": "deepseek-chat",
            "messages": [],
            "top_k": 40,  # 不支持的参数
            "max_tokens": 1024
        }

        cleaned = model._clean_request_params(request)

        assert "top_k" not in cleaned
        assert cleaned["max_tokens"] == 1024

    def test_clean_reasoner_params_removes_error_params(self):
        """测试 Reasoner 模型清理会报错的参数"""
        model = DeepSeekAnthropicModel(
            client_args={"api_key": "test"},
            model_id="deepseek-reasoner",
            max_tokens=1024
        )

        request = {
            "model": "deepseek-reasoner",
            "messages": [],
            "logprobs": True,
            "top_logprobs": 5,
            "temperature": 0.7,
            "max_tokens": 1024
        }

        cleaned = model._clean_reasoner_params(request)

        # 会报错的参数应该被移除
        assert "logprobs" not in cleaned
        assert "top_logprobs" not in cleaned
        # 不支持的参数也应该被移除
        assert "temperature" not in cleaned

    def test_clean_reasoner_params_handles_tool_choice(self):
        """测试 Reasoner 模型处理 tool_choice 中的 disable_parallel_tool_use"""
        model = DeepSeekAnthropicModel(
            client_args={"api_key": "test"},
            model_id="deepseek-reasoner",
            max_tokens=1024
        )

        request = {
            "model": "deepseek-reasoner",
            "tool_choice": {
                "type": "auto",
                "disable_parallel_tool_use": True
            }
        }

        cleaned = model._clean_reasoner_params(request)

        assert "disable_parallel_tool_use" not in cleaned["tool_choice"]
        assert cleaned["tool_choice"]["type"] == "auto"

    def test_serialize_event_basic(self):
        """测试基础事件序列化"""
        model = DeepSeekAnthropicModel(
            client_args={"api_key": "test"},
            model_id="deepseek-chat",
            max_tokens=1024
        )

        mock_event = Mock(spec=["type", "index"])
        mock_event.type = "content_block_start"
        mock_event.index = 0

        result = model._serialize_event(mock_event)

        assert result["type"] == "content_block_start"
        assert result["index"] == 0

    def test_serialize_content_block_text(self):
        """测试文本内容块序列化"""
        model = DeepSeekAnthropicModel(
            client_args={"api_key": "test"},
            model_id="deepseek-chat",
            max_tokens=1024
        )

        mock_block = Mock()
        mock_block.type = "text"
        mock_block.text = "Hello world"

        result = model._serialize_content_block(mock_block)

        assert result["type"] == "text"
        assert result["text"] == "Hello world"

    def test_serialize_content_block_tool_use(self):
        """测试 tool_use 内容块序列化"""
        model = DeepSeekAnthropicModel(
            client_args={"api_key": "test"},
            model_id="deepseek-chat",
            max_tokens=1024
        )

        mock_block = Mock()
        mock_block.type = "tool_use"
        mock_block.id = "tool_123"
        mock_block.name = "calculator"
        mock_block.input = {"expression": "2+2"}

        result = model._serialize_content_block(mock_block)

        assert result["type"] == "tool_use"
        assert result["id"] == "tool_123"
        assert result["name"] == "calculator"

    def test_serialize_thinking_block(self):
        """测试 thinking 内容块序列化"""
        model = DeepSeekAnthropicModel(
            client_args={"api_key": "test"},
            model_id="deepseek-reasoner",
            max_tokens=1024
        )

        mock_block = Mock()
        mock_block.type = "thinking"
        mock_block.thinking = "Step 1: Analyze the problem"
        mock_block.signature = "sig123"

        result = model._serialize_content_block(mock_block)

        assert result["type"] == "thinking"
        assert result["thinking"] == "Step 1: Analyze the problem"
        assert result["signature"] == "sig123"

    def test_create_deepseek_anthropic_model_helper(self):
        """测试 create_deepseek_anthropic_model 辅助函数"""
        model = create_deepseek_anthropic_model(
            api_key="test-key",
            model_id="deepseek-chat",
            max_tokens=2048
        )

        assert model.config["model_id"] == "deepseek-chat"
        # AnthropicModel 使用 client 存储配置
        assert model.client.api_key == "test-key"
        # URL 对象需要转换为字符串比较，且可能有尾部斜杠
        assert str(model.client.base_url).rstrip("/") == "https://api.deepseek.com/anthropic"

    def test_create_deepseek_anthropic_reasoner_helper(self):
        """测试 create_deepseek_anthropic_reasoner 辅助函数"""
        model = create_deepseek_anthropic_reasoner(
            api_key="test-key",
            max_tokens=4096
        )

        assert model.config["model_id"] == "deepseek-reasoner"
        # URL 对象需要转换为字符串比较，且可能有尾部斜杠
        assert str(model.client.base_url).rstrip("/") == "https://api.deepseek.com/anthropic"


class TestDeepSeekAnthropicMultiTurnToolCalls:
    """DeepSeek Anthropic 多轮对话带工具调用测试 (无需真实 API)"""

    def test_multi_turn_conversation_formatting(self):
        """测试多轮对话的消息格式化"""
        model = DeepSeekAnthropicModel(
            client_args={"api_key": "test"},
            model_id="deepseek-chat",
            max_tokens=1024
        )

        messages: list[Message] = [
            # Round 1
            {"role": "user", "content": [{"text": "What is 15 * 6?"}]},
            {
                "role": "assistant",
                "content": [
                    {"text": "Let me calculate that."},
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

        formatted = model.format_request(messages)

        assert "messages" in formatted
        assert len(formatted["messages"]) >= 4
        assert formatted["model"] == "deepseek-chat"

    def test_reasoner_multi_turn_with_thinking(self):
        """测试 Reasoner 模型的多轮对话（带 thinking）"""
        model = DeepSeekAnthropicModel(
            client_args={"api_key": "test"},
            model_id="deepseek-reasoner",
            max_tokens=1024
        )

        messages: list[Message] = [
            {"role": "user", "content": [{"text": "Solve step by step: 25 * 4"}]},
            {
                "role": "assistant",
                "content": [
                    {"text": "I need to multiply 25 by 4."}
                ]
            },
            {"role": "user", "content": [{"text": "Continue"}]},
        ]

        formatted = model.format_request(messages)

        # Reasoner 模型不应该有不支持的参数
        assert "temperature" not in formatted
        assert "top_p" not in formatted
        assert formatted["model"] == "deepseek-reasoner"

    def test_serialize_delta_partial_json(self):
        """测试工具参数增量序列化（多轮场景）"""
        model = DeepSeekAnthropicModel(
            client_args={"api_key": "test"},
            model_id="deepseek-chat",
            max_tokens=1024
        )

        mock_delta = Mock()
        mock_delta.type = "input_json_delta"
        mock_delta.partial_json = '{"param": "val'

        result = model._serialize_delta(mock_delta)

        assert result["type"] == "input_json_delta"
        assert result["partial_json"] == '{"param": "val'

    def test_serialize_usage(self):
        """测试 usage 序列化"""
        model = DeepSeekAnthropicModel(
            client_args={"api_key": "test"},
            model_id="deepseek-chat",
            max_tokens=1024
        )

        mock_usage = Mock()
        mock_usage.input_tokens = 100
        mock_usage.output_tokens = 50

        result = model._serialize_usage(mock_usage)

        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50


class TestDeepSeekAnthropicConstants:
    """DeepSeek Anthropic 常量定义测试"""

    def test_unsupported_reasoner_params_defined(self):
        """测试 UNSUPPORTED_REASONER_PARAMS 常量定义"""
        expected = {"temperature", "top_p", "top_k", "presence_penalty", "frequency_penalty"}
        assert UNSUPPORTED_REASONER_PARAMS == expected

    def test_error_reasoner_params_defined(self):
        """测试 ERROR_REASONER_PARAMS 常量定义"""
        expected = {"logprobs", "top_logprobs"}
        assert ERROR_REASONER_PARAMS == expected


@pytest.mark.skipif(not HAS_DEEPSEEK_KEY, reason="DEEPSEEK_API_KEY not set")
class TestDeepSeekAnthropicModelIntegration:
    """DeepSeekAnthropicModel 集成测试 (需要真实 API Key)"""

    @pytest.fixture
    def model(self) -> Generator[DeepSeekAnthropicModel, None, None]:
        """创建 DeepSeek Anthropic 模型实例"""
        assert DEEPSEEK_API_KEY is not None
        m = create_deepseek_anthropic_model(
            api_key=DEEPSEEK_API_KEY,
            model_id="deepseek-chat",
            max_tokens=1024
        )
        yield m

    @pytest.fixture
    def reasoner_model(self) -> Generator[DeepSeekAnthropicModel, None, None]:
        """创建 DeepSeek Reasoner 模型实例"""
        assert DEEPSEEK_API_KEY is not None
        m = create_deepseek_anthropic_reasoner(
            api_key=DEEPSEEK_API_KEY,
            max_tokens=2048
        )
        yield m

    @pytest.mark.asyncio
    async def test_simple_chat_completion(self, model: DeepSeekAnthropicModel):
        """测试简单对话完成"""
        messages: list[Message] = [
            {"role": "user", "content": [{"text": "Say hello"}]}
        ]

        chunks = []
        async for chunk in model.stream(messages):
            chunks.append(chunk)

        assert len(chunks) > 0
        # 验证有 content_block_delta 或 message_stop
        has_content = any("contentBlockDelta" in c or c.get("type") == "content_block_delta" for c in chunks)
        has_stop = any("messageStop" in c or c.get("type") == "message_stop" for c in chunks)
        assert has_content or has_stop

    @pytest.mark.asyncio
    async def test_reasoner_chat_completion(self, reasoner_model: DeepSeekAnthropicModel):
        """测试 Reasoner 模型对话完成"""
        messages: list[Message] = [
            {"role": "user", "content": [{"text": "What is 2+2? Explain briefly."}]}
        ]

        chunks = []
        async for chunk in reasoner_model.stream(messages):
            chunks.append(chunk)

        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, model: DeepSeekAnthropicModel):
        """测试多轮对话"""
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
            {"role": "user", "content": [{"text": "Add 50 to that"}]}
        ])

        second_chunks = []
        async for chunk in model.stream(messages):
            second_chunks.append(chunk)

        assert len(second_chunks) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
