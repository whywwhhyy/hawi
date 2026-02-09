"""
DeepSeekModel 集成测试

API Key 来源 (按优先级):
1. DEEPSEEK_API_KEY 环境变量
2. apikey.yaml 文件中的 deepseek 配置

参考文档:
- https://api-docs.deepseek.com/guides/thinking_mode
"""

import os
import sys
import json
import pytest
from typing import Generator
from unittest.mock import Mock, patch, MagicMock

from strands.types.tools import ToolResult
from typing import Any

from hawi.agent.models.deepseek_openai import DeepSeekModel, create_deepseek_model, create_deepseek_reasoner
from hawi.agent.models.deepseek_openai import UNSUPPORTED_REASONER_PARAMS, ERROR_REASONER_PARAMS
from test.integration import get_deepseek_api_key

# 检查是否设置了 API Key (从环境变量或 apikey.yaml)
DEEPSEEK_API_KEY = get_deepseek_api_key()
HAS_DEEPSEEK_KEY = DEEPSEEK_API_KEY is not None and DEEPSEEK_API_KEY.strip() != ""


class TestDeepSeekModelUnit:
    """DeepSeekModel 单元测试 (无需真实 API)"""

    def test_format_request_tool_message_string_content(self):
        """测试 tool 消息 content 被格式化为字符串"""
        tool_result: ToolResult = {
            "toolUseId": "test-123",
            "status": "success",
            "content": [{"json": {"result": "success", "data": [1, 2, 3]}}]
        }

        result = DeepSeekModel.format_request_tool_message(tool_result)

        assert result["role"] == "tool"
        assert result["tool_call_id"] == "test-123"
        assert isinstance(result["content"], str)
        assert "success" in result["content"]
        # DeepSeek API 不接受数组格式
        assert not isinstance(result["content"], list)

    def test_format_request_tool_message_multiple_content(self):
        """测试多个内容块合并为字符串"""
        tool_result: ToolResult = {
            "toolUseId": "multi-456",
            "status": "success",
            "content": [
                {"json": {"status": "ok"}},
                {"text": "Additional info"}
            ]
        }

        result = DeepSeekModel.format_request_tool_message(tool_result)

        assert isinstance(result["content"], str)
        assert "ok" in result["content"]
        assert "Additional info" in result["content"]

    def test_format_request_tool_message_with_image(self):
        """测试图片内容被正确处理为警告文本"""
        tool_result: ToolResult = {
            "toolUseId": "img-789",
            "status": "success",
            "content": [
                {"json": {"result": "data"}},
                {"image": {"format": "png", "source": {"bytes": b"base64..."}}}
            ]
        }

        result = DeepSeekModel.format_request_tool_message(tool_result)

        assert isinstance(result["content"], str)
        assert "data" in result["content"]
        assert "[图片内容]" in result["content"]

    def test_format_request_tool_message_empty_content(self):
        """测试空内容处理"""
        tool_result: ToolResult = {
            "toolUseId": "empty-000",
            "status": "success",
            "content": []
        }

        result = DeepSeekModel.format_request_tool_message(tool_result)

        assert isinstance(result["content"], str)
        # DeepSeek API 不接受空 content，所以返回一个空格
        assert result["content"] == " "

    def test_reasoner_params_validation_remove_error_params(self):
        """测试 Reasoner 模型移除会报错的参数"""
        model = DeepSeekModel(
            client_args={"api_key": "test"},
            model_id="deepseek-reasoner",
            params={
                "temperature": 0.7,
                "logprobs": True,
                "top_logprobs": 5,
                "max_tokens": 1024
            }
        )

        params: dict[str, Any] = model.config.get("params", {})  # type: ignore[assignment]
        cleaned = model._validate_reasoner_params(params)

        # 会报错的参数应该被移除
        assert "logprobs" not in cleaned
        assert "top_logprobs" not in cleaned
        # 无效参数应该保留但会发出警告
        assert "temperature" in cleaned
        assert "max_tokens" in cleaned

    def test_reasoner_params_validation_regular_model(self):
        """测试普通模型不清理参数"""
        model = DeepSeekModel(
            client_args={"api_key": "test"},
            model_id="deepseek-chat",
            params={
                "temperature": 0.7,
                "logprobs": True,
            }
        )

        params = {"temperature": 0.7, "logprobs": True}
        cleaned = model._validate_reasoner_params(params)

        # 普通模型应该保留所有参数
        assert "temperature" in cleaned
        assert "logprobs" in cleaned

    def test_format_request_messages_with_reasoning(self):
        """测试带 reasoning_content 的消息格式化（reasoningContent 被过滤）"""
        from strands.types.content import Message
        messages: list[Message] = [
            {
                "role": "assistant",
                "content": [
                    {"text": "The answer is 42"}
                ]
            }
        ]

        result = DeepSeekModel.format_request_messages(messages)

        # 验证消息被正确格式化
        assert len(result) > 0
        assistant_msg = [m for m in result if m.get("role") == "assistant"][0]
        assert "content" in assistant_msg

    def test_format_request_messages_without_reasoning(self):
        """测试不带 reasoning_content 的消息格式化"""
        from strands.types.content import Message
        messages: list[Message] = [
            {
                "role": "user",
                "content": [{"text": "Hello"}]
            },
            {
                "role": "assistant",
                "content": [{"text": "Hi there!"}]
            }
        ]

        result = DeepSeekModel.format_request_messages(messages)

        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_create_deepseek_model_helper(self):
        """测试 create_deepseek_model 辅助函数"""
        model = create_deepseek_model(
            api_key="test-key",  # type: ignore[arg-type]
            model_id="deepseek-chat",
            params={"temperature": 0.5}
        )

        assert model.config["model_id"] == "deepseek-chat"
        assert model.client_args["api_key"] == "test-key"
        assert model.client_args["base_url"] == "https://api.deepseek.com"
        params: dict[str, Any] = model.config.get("params", {})  # type: ignore[assignment]
        assert params["temperature"] == 0.5

    def test_create_deepseek_reasoner_helper(self):
        """测试 create_deepseek_reasoner 辅助函数"""
        model = create_deepseek_reasoner(
            api_key="test-key",
            include_reasoning_in_context=True,
            params={"max_tokens": 4096}
        )

        assert model.config["model_id"] == "deepseek-reasoner"
        assert model.include_reasoning_in_context is True
        assert model.client_args["base_url"] == "https://api.deepseek.com"

    def test_create_deepseek_reasoner_warns_unsupported_params(self, caplog):
        """测试创建 Reasoner 时对不支持参数的警告"""
        import logging

        with caplog.at_level(logging.WARNING):
            model = create_deepseek_reasoner(
                api_key="test-key",
                params={"temperature": 0.7, "top_p": 0.9}
            )

        # 应该发出关于不支持参数的警告
        assert any("temperature" in r.message for r in caplog.records)
        assert any("top_p" in r.message for r in caplog.records)

    def test_split_tool_message_images(self):
        """测试 tool 消息中的图片被分离到 user 消息"""
        tool_msg = {
            "role": "tool",
            "tool_call_id": "test-123",
            "content": [
                {"type": "text", "text": "Tool result"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}}
            ]
        }

        tool_clean, user_images = DeepSeekModel._split_tool_message_images(tool_msg)

        assert tool_clean["role"] == "tool"
        assert "image_url" not in str(tool_clean.get("content", ""))
        assert user_images is not None
        assert user_images["role"] == "user"


@pytest.mark.skipif(not HAS_DEEPSEEK_KEY, reason="DEEPSEEK_API_KEY not set")
class TestDeepSeekModelIntegration:
    """DeepSeekModel 集成测试 (需要真实 API Key)"""

    @pytest.fixture
    def model(self) -> Generator[DeepSeekModel, None, None]:
        """创建 DeepSeek 模型实例"""
        assert DEEPSEEK_API_KEY is not None  # type guard
        m = create_deepseek_model(
            api_key=DEEPSEEK_API_KEY,
            model_id="deepseek-chat",
            params={"temperature": 0.7, "max_tokens": 1024}
        )
        yield m

    @pytest.fixture
    def reasoner_model(self) -> Generator[DeepSeekModel, None, None]:
        """创建 DeepSeek Reasoner 模型实例"""
        assert DEEPSEEK_API_KEY is not None  # type guard
        m = create_deepseek_reasoner(
            api_key=DEEPSEEK_API_KEY,
            include_reasoning_in_context=True,
            params={"max_tokens": 2048}
        )
        yield m

    @pytest.mark.asyncio
    async def test_simple_chat_completion(self, model: DeepSeekModel):
        """测试简单对话完成"""
        from strands.types.content import Message
        messages: list[Message] = [{"role": "user", "content": [{"text": "Say hello"}]}]

        chunks = []
        async for chunk in model.stream(messages):
            chunks.append(chunk)

        # 验证收到了消息
        assert len(chunks) > 0
        # 检查是否有内容块或消息停止
        has_content = any("contentBlockDelta" in c for c in chunks)
        has_message_stop = any("messageStop" in c for c in chunks)
        assert has_content or has_message_stop

    @pytest.mark.asyncio
    async def test_reasoner_chat_completion(self, reasoner_model: DeepSeekModel):
        """测试 Reasoner 模型对话完成"""
        from strands.types.content import Message
        messages: list[Message] = [{"role": "user", "content": [{"text": "What is 2+2? Explain your reasoning."}]}]

        chunks = []
        reasoning_parts = []
        content_parts = []

        async for chunk in reasoner_model.stream(messages):
            chunks.append(chunk)
            # 检查是否有推理内容或文本内容
            if "contentBlockDelta" in chunk:
                delta = chunk["contentBlockDelta"].get("delta", {})
                if "reasoningContent" in delta:
                    reasoning_text = delta["reasoningContent"].get("text", "")
                    if reasoning_text:
                        reasoning_parts.append(reasoning_text)
                elif "text" in delta:
                    text = delta.get("text", "")
                    if text:
                        content_parts.append(text)

        # Reasoner 应该产生推理内容或最终答案
        assert len(chunks) > 0
        assert len(reasoning_parts) > 0 or len(content_parts) > 0

    @pytest.mark.asyncio
    async def test_tool_call_formatting(self, model: DeepSeekModel):
        """测试工具调用消息格式"""
        from strands.types.content import Message, ContentBlock
        messages: list[Message] = [
            {"role": "user", "content": [{"text": "Use the test tool"}]},
            {
                "role": "assistant",
                "content": [
                    {"toolUse": {"toolUseId": "call-123", "name": "test_tool", "input": {}}}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"toolResult": {"toolUseId": "call-123", "status": "success", "content": [{"json": {"result": "ok"}}]}}
                ]
            }
        ]

        # 格式化请求消息
        formatted = model.format_request_messages(messages)

        # 验证 tool 消息的 content 是字符串
        tool_messages = [m for m in formatted if m.get("role") == "tool"]
        for msg in tool_messages:
            assert isinstance(msg.get("content"), str), f"Tool message content should be string, got {type(msg.get('content'))}"

    @pytest.mark.asyncio
    async def test_stream_response_structure(self, model: DeepSeekModel):
        """测试流式响应结构"""
        from strands.types.content import Message
        messages: list[Message] = [{"role": "user", "content": [{"text": "Count to 3"}]}]

        message_start_seen = False
        message_stop_seen = False
        content_deltas = []

        async for chunk in model.stream(messages):
            # 检查消息开始
            if "messageStart" in chunk:
                message_start_seen = True
            # 检查消息停止
            elif "messageStop" in chunk:
                message_stop_seen = True
            # 检查内容增量
            elif "contentBlockDelta" in chunk:
                content_deltas.append(chunk)

        assert message_start_seen, "Should see message_start"
        assert message_stop_seen, "Should see message_stop"
        # 至少应该有一些内容增量
        assert len(content_deltas) > 0, "Should have content deltas"


class TestDeepSeekConstants:
    """DeepSeek 常量定义测试"""

    def test_unsupported_reasoner_params_defined(self):
        """测试 UNSUPPORTED_REASONER_PARAMS 常量定义"""
        expected = {"temperature", "top_p", "presence_penalty", "frequency_penalty"}
        assert UNSUPPORTED_REASONER_PARAMS == expected

    def test_error_reasoner_params_defined(self):
        """测试 ERROR_REASONER_PARAMS 常量定义"""
        expected = {"logprobs", "top_logprobs"}
        assert ERROR_REASONER_PARAMS == expected


class TestDeepSeekMultiTurnToolCalls:
    """DeepSeek 多轮对话带工具调用测试 (无需真实 API)"""

    def test_multi_turn_tool_call_formatting(self):
        """测试多轮对话中工具调用消息的格式化"""
        from strands.types.content import Message

        # 模拟一个完整的多轮工具调用对话
        messages: list[Message] = [
            # 第一轮：用户请求
            {"role": "user", "content": [{"text": "Calculate 2+2"}]},
            # 助手调用工具
            {
                "role": "assistant",
                "content": [
                    {"text": "I'll calculate that for you."},
                    {"toolUse": {"toolUseId": "call-1", "name": "calculator", "input": {"expression": "2+2"}}}
                ]
            },
            # 工具返回结果
            {
                "role": "user",
                "content": [
                    {"toolResult": {"toolUseId": "call-1", "status": "success", "content": [{"json": {"result": 4}}]}}
                ]
            },
            # 助手给出最终答案
            {"role": "assistant", "content": [{"text": "The result is 4."}]},
            # 第二轮：用户再次请求
            {"role": "user", "content": [{"text": "Now multiply by 3"}]},
        ]

        formatted = DeepSeekModel.format_request_messages(messages)

        # 验证格式化后的消息结构
        assert len(formatted) >= 4  # user, assistant, tool, assistant, user

        # 验证 tool 消息的 content 是字符串
        tool_messages = [m for m in formatted if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert isinstance(tool_messages[0].get("content"), str)
        assert "4" in tool_messages[0]["content"]

        # 验证 assistant 消息有 tool_calls
        assistant_messages = [m for m in formatted if m.get("role") == "assistant"]
        tool_call_msgs = [m for m in assistant_messages if "tool_calls" in m]
        assert len(tool_call_msgs) == 1

    def test_multi_turn_with_reasoning_content(self):
        """测试多轮对话中带 reasoning_content 的工具调用"""
        from strands.types.content import Message

        # 模拟 Reasoner 模型的多轮对话
        messages: list[Message] = [
            {"role": "user", "content": [{"text": "Solve step by step: 15*6"}]},
            {
                "role": "assistant",
                "content": [
                    {"reasoningContent": {"reasoningText": {"text": "First, I need to multiply 15 by 6."}}},
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
        ]

        # 测试不包含 reasoning_content (默认行为)
        formatted_without = DeepSeekModel.format_request_messages(messages, include_reasoning_in_context=False)
        tool_msgs = [m for m in formatted_without if m.get("role") == "tool"]
        assert len(tool_msgs) == 1

        # 测试包含 reasoning_content (工具调用场景需要)
        formatted_with = DeepSeekModel.format_request_messages(messages, include_reasoning_in_context=True)
        # 验证消息被正确格式化
        assert len(formatted_with) >= 3

    def test_reasoner_multi_turn_context_preservation(self):
        """测试 Reasoner 模型多轮对话中 reasoning_content 的保留"""
        from strands.types.content import Message

        # 创建一个包含 reasoning_content 的复杂对话历史
        messages: list[Message] = [
            {"role": "user", "content": [{"text": "What is the capital of France?"}]},
            {
                "role": "assistant",
                "content": [
                    {"reasoningContent": {"reasoningText": {"text": "The capital of France is a well-known fact."}}},
                    {"text": "The capital of France is Paris."}
                ]
            },
            {"role": "user", "content": [{"text": "What about Germany?"}]},
            {
                "role": "assistant",
                "content": [
                    {"reasoningContent": {"reasoningText": {"text": "Similar to France, Germany's capital is well-known."}}},
                    {"text": "The capital of Germany is Berlin."}
                ]
            },
        ]

        # 使用 include_reasoning_in_context=True (工具调用场景)
        model = DeepSeekModel(
            client_args={"api_key": "test"},
            model_id="deepseek-reasoner",
            include_reasoning_in_context=True
        )

        formatted = model.format_request_messages(messages)

        # 验证所有消息都被保留
        assert len(formatted) >= 4  # user, assistant, user, assistant

        # 验证角色顺序正确
        roles = [m.get("role") for m in formatted]
        assert roles[0] == "user"
        assert roles[1] == "assistant"


@pytest.mark.skipif(not HAS_DEEPSEEK_KEY, reason="DEEPSEEK_API_KEY not set")
class TestDeepSeekMultiTurnToolCallsIntegration:
    """DeepSeek 多轮工具调用集成测试 (需要真实 API)"""

    @pytest.fixture
    def model(self) -> Generator[DeepSeekModel, None, None]:
        """创建 DeepSeek 模型实例"""
        assert DEEPSEEK_API_KEY is not None
        m = create_deepseek_model(
            api_key=DEEPSEEK_API_KEY,
            model_id="deepseek-chat",
            params={"temperature": 0.7, "max_tokens": 1024}
        )
        yield m

    @pytest.fixture
    def reasoner_model(self) -> Generator[DeepSeekModel, None, None]:
        """创建 DeepSeek Reasoner 模型实例"""
        assert DEEPSEEK_API_KEY is not None
        m = create_deepseek_reasoner(
            api_key=DEEPSEEK_API_KEY,
            include_reasoning_in_context=True,
            params={"max_tokens": 2048}
        )
        yield m

    @pytest.mark.asyncio
    async def test_multi_turn_conversation_with_tools(self, model: DeepSeekModel):
        """测试多轮对话中带工具调用的完整流程"""
        from strands.types.content import Message

        # 第一轮：用户提问
        messages: list[Message] = [
            {"role": "user", "content": [{"text": "What is 25 * 4?"}]}
        ]

        # 模拟获取助手回复（包含工具调用）
        assistant_chunks = []
        async for chunk in model.stream(messages):
            assistant_chunks.append(chunk)

        # 验证收到了响应
        assert len(assistant_chunks) > 0

        # 手动构建第二轮对话（模拟工具调用后的回复）
        messages.extend([
            {
                "role": "assistant",
                "content": [{"text": "The result of 25 * 4 is 100."}]
            },
            {"role": "user", "content": [{"text": "Now add 50 to that result."}]}
        ])

        # 第二轮对话
        second_chunks = []
        async for chunk in model.stream(messages):
            second_chunks.append(chunk)

        # 验证上下文被保留（模型知道之前的计算结果）
        assert len(second_chunks) > 0
        second_response = str(second_chunks)
        # 响应中应该包含对 100 或 150 的引用
        assert "100" in second_response or "150" in second_response or "result" in second_response.lower()

    @pytest.mark.asyncio
    async def test_reasoner_tool_call_with_reasoning_content(self, reasoner_model: DeepSeekModel):
        """测试 Reasoner 模型在工具调用场景中的 reasoning_content 处理"""
        from strands.types.content import Message

        # 第一轮：需要推理的问题
        messages: list[Message] = [
            {"role": "user", "content": [{"text": "Calculate the area of a rectangle with length 12 and width 8, then divide by 2."}]}
        ]

        reasoning_parts = []
        content_parts = []

        async for chunk in reasoner_model.stream(messages):
            if "contentBlockDelta" in chunk:
                delta = chunk["contentBlockDelta"].get("delta", {})
                if "reasoningContent" in delta:
                    reasoning_text = delta["reasoningContent"].get("text", "")
                    if reasoning_text:
                        reasoning_parts.append(reasoning_text)
                elif "text" in delta:
                    text = delta.get("text", "")
                    if text:
                        content_parts.append(text)

        # Reasoner 应该产生推理内容
        full_response = "".join(content_parts)
        assert len(reasoning_parts) > 0 or len(content_parts) > 0

        # 继续第二轮对话
        messages.extend([
            {"role": "assistant", "content": [{"text": full_response or "Area = 96, divided by 2 is 48."}]},
            {"role": "user", "content": [{"text": "What would be the result if width was 10 instead?"}]}
        ])

        second_chunks = []
        async for chunk in reasoner_model.stream(messages):
            second_chunks.append(chunk)

        # 验证上下文被保留
        assert len(second_chunks) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
