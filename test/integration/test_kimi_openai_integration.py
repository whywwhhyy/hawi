"""
KimiOpenAIModel 集成测试

API Key 来源 (按优先级):
1. KIMI_API_KEY 环境变量
2. apikey.yaml 文件中的 kimi-openai 配置

参考文档:
- https://platform.moonshot.ai/docs/guide/kimi-k2-5-quickstart
"""

import os
import sys
import pytest
from typing import Generator, Any
from unittest.mock import Mock, patch, MagicMock, AsyncMock

from strands.types.content import ContentBlock, Message
from strands.types.tools import ToolResult
from strands.types.streaming import StreamEvent

from hawi.agent.models.kimi_openai import KimiOpenAIModel, create_kimi_model
from test.integration import get_kimi_openai_api_key

# 检查是否设置了 API Key (从环境变量或 apikey.yaml)
KIMI_API_KEY: str | None = get_kimi_openai_api_key()
HAS_KIMI_KEY: bool = KIMI_API_KEY is not None and KIMI_API_KEY.strip() != ""


class TestKimiOpenAIModelUnit:
    """KimiOpenAIModel 单元测试 (无需真实 API)"""

    def test_extract_reasoning_content(self):
        """测试从内容块中提取 reasoning_content"""
        contents: list[ContentBlock] = [
            {"reasoningContent": {"reasoningText": {"text": "Let me think..."}}},
            {"text": "The answer is 42"}
        ]

        result = KimiOpenAIModel._extract_reasoning_content(contents)

        assert result == "Let me think..."

    def test_extract_reasoning_content_empty(self):
        """测试没有 reasoning_content 时返回 None"""
        contents: list[ContentBlock] = [{"text": "Just a simple response"}]

        result = KimiOpenAIModel._extract_reasoning_content(contents)

        assert result is None

    def test_extract_reasoning_content_partial(self):
        """测试不完整的 reasoning_content 结构"""
        contents: list[ContentBlock] = [{"reasoningContent": {"other": "data"}}]  # type: ignore[typeddict-item]

        result = KimiOpenAIModel._extract_reasoning_content(contents)

        # 当 reasoningContent 存在但没有 reasoningText.text 时返回空字符串
        assert result == ""

    def test_format_regular_messages_with_reasoning(self):
        """测试格式化带 reasoning_content 的常规消息"""
        messages: list[Message] = [
            {
                "role": "assistant",
                "content": [
                    {"reasoningContent": {"reasoningText": {"text": "Step 1..."}}},
                    {"text": "Final answer"}
                ]
            }
        ]

        result = KimiOpenAIModel._format_regular_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert "reasoning_content" in result[0]
        assert result[0]["reasoning_content"] == "Step 1..."

    def test_format_regular_messages_tool_call_requires_reasoning(self):
        """测试 tool call 消息需要 reasoning_content 字段"""
        messages: list[Message] = [
            {
                "role": "assistant",
                "content": [
                    {"toolUse": {"toolUseId": "call-123", "name": "test_tool", "input": {}}}
                ]
            }
        ]

        result = KimiOpenAIModel._format_regular_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        # Tool call 消息必须有 reasoning_content（即使是空字符串）
        assert "reasoning_content" in result[0]
        assert result[0]["reasoning_content"] == ""

    def test_format_regular_messages_filters_reasoning(self):
        """测试常规消息内容过滤 reasoningContent"""
        messages: list[Message] = [
            {
                "role": "assistant",
                "content": [
                    {"reasoningContent": {"reasoningText": {"text": "Thinking..."}}},
                    {"text": "Response"},
                    {"toolUse": {"toolUseId": "call-1", "name": "tool", "input": {}}}
                ]
            }
        ]

        result = KimiOpenAIModel._format_regular_messages(messages)

        # reasoningContent 应该被过滤出 content，但作为 reasoning_content 字段保留
        assistant_msg = result[0]
        assert "reasoning_content" in assistant_msg

    def test_format_request_messages(self):
        """测试完整的消息格式化"""
        messages: list[Message] = [
            {"role": "user", "content": [{"text": "Hello"}]},
            {"role": "assistant", "content": [{"text": "Hi!"}]}
        ]

        result = KimiOpenAIModel.format_request_messages(messages)

        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_format_request_with_system_prompt(self):
        """测试带系统提示的消息格式化"""
        messages: list[Message] = [{"role": "user", "content": [{"text": "Hello"}]}]

        result = KimiOpenAIModel.format_request_messages(
            messages,
            system_prompt="You are a helpful assistant."
        )

        # 应该添加 system 消息
        assert result[0]["role"] == "system"
        assert "helpful assistant" in str(result[0].get("content", ""))

    def test_format_request_disables_thinking(self):
        """测试禁用 thinking 模式的请求格式化"""
        model = KimiOpenAIModel(
            client_args={"api_key": "test"},
            model_id="kimi-k2.5",
            params={"temperature": 0.6, "thinking": {"type": "disabled"}}
        )

        messages: list[Message] = [{"role": "user", "content": [{"text": "Hello"}]}]
        request = model.format_request(messages)

        # 禁用 thinking 时应该通过 extra_body 传递
        assert "extra_body" in request
        assert request["extra_body"]["thinking"]["type"] == "disabled"

    def test_format_request_keeps_thinking_enabled(self):
        """测试启用 thinking 模式的请求格式化"""
        model = KimiOpenAIModel(
            client_args={"api_key": "test"},
            model_id="kimi-k2.5",
            params={"max_tokens": 1024}
        )

        messages: list[Message] = [{"role": "user", "content": [{"text": "Hello"}]}]
        request = model.format_request(messages)

        # 默认启用 thinking，不应该有 extra_body
        assert "extra_body" not in request or "thinking" not in request.get("extra_body", {})

    def test_create_kimi_model_helper(self):
        """测试 create_kimi_model 辅助函数"""
        model = create_kimi_model(
            api_key="test-key",
            model_id="kimi-k2.5",
            enable_thinking=True
        )

        assert model.config["model_id"] == "kimi-k2.5"
        assert model.client_args["api_key"] == "test-key"
        assert model.client_args["base_url"] == "https://api.moonshot.cn/v1"

    def test_create_kimi_model_disabled_thinking(self):
        """测试创建禁用 thinking 的 Kimi 模型"""
        model = create_kimi_model(
            api_key="test-key",
            model_id="kimi-k2.5",
            enable_thinking=False,
            params={"temperature": 0.5}
        )

        params: dict[str, Any] = model.config.get("params", {})  # type: ignore[assignment]
        assert params.get("thinking", {}).get("type") == "disabled"
        assert params.get("temperature") == 0.5

    def test_tool_message_format(self):
        """测试 tool 消息格式化"""
        tool_result: ToolResult = {
            "toolUseId": "test-456",
            "content": [{"json": {"result": "success"}}],
            "status": "success"
        }

        formatted = KimiOpenAIModel.format_request_tool_message(tool_result)

        # 验证基本结构
        assert formatted["role"] == "tool"
        assert formatted["tool_call_id"] == "test-456"
        assert "content" in formatted


@pytest.mark.skipif(not HAS_KIMI_KEY, reason="KIMI_API_KEY not set")
class TestKimiOpenAIModelIntegration:
    """KimiOpenAIModel 集成测试 (需要真实 API Key)"""

    @pytest.fixture
    def model(self) -> Generator[KimiOpenAIModel, None, None]:
        """创建 Kimi 模型实例 (启用 thinking)"""
        assert KIMI_API_KEY is not None  # type narrowing for mypy
        m = create_kimi_model(
            api_key=KIMI_API_KEY,
            model_id="kimi-k2.5",
            enable_thinking=True,
            params={"max_tokens": 1024}
        )
        yield m

    @pytest.fixture
    def model_no_thinking(self) -> Generator[KimiOpenAIModel, None, None]:
        """创建 Kimi 模型实例 (禁用 thinking)"""
        assert KIMI_API_KEY is not None  # type narrowing for mypy
        m = create_kimi_model(
            api_key=KIMI_API_KEY,
            model_id="kimi-k2.5",
            enable_thinking=False,
            params={"temperature": 0.6, "max_tokens": 1024}
        )
        yield m

    @pytest.mark.asyncio
    async def test_simple_chat_with_thinking(self, model: KimiOpenAIModel):
        """测试启用 thinking 模式的简单对话"""
        messages: list[Message] = [{"role": "user", "content": [{"text": "What is 2+2?"}]}]

        chunks: list[StreamEvent] = []
        reasoning_parts: list[str] = []
        content_parts: list[str] = []

        async for chunk in model.stream(messages):
            chunks.append(chunk)  # type: ignore[arg-type]
            # 检查是否有内容增量
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

        # 应该收到响应
        assert len(chunks) > 0
        # Thinking 模式应该产生推理内容或文本内容
        assert len(reasoning_parts) > 0 or len(content_parts) > 0

    @pytest.mark.asyncio
    async def test_simple_chat_without_thinking(self, model_no_thinking: KimiOpenAIModel):
        """测试禁用 thinking 模式的简单对话"""
        messages: list[Message] = [{"role": "user", "content": [{"text": "Say hello"}]}]

        chunks: list[StreamEvent] = []
        async for chunk in model_no_thinking.stream(messages):
            chunks.append(chunk)  # type: ignore[arg-type]

        # 应该收到文本响应
        assert len(chunks) > 0
        content_chunks = [c for c in chunks if c.get("data_type") == "text"]
        assert len(content_chunks) >= 0  # 可能有文本内容

    @pytest.mark.asyncio
    async def test_reasoning_content_preserved_in_context(self, model: KimiOpenAIModel):
        """测试 reasoning_content 在多轮对话中被保留"""
        # 第一轮对话
        messages: list[Message] = [{"role": "user", "content": [{"text": "What is 15*6?"}]}]

        assistant_response: list[str] = []
        async for chunk in model.stream(messages):
            if "contentBlockDelta" in chunk:
                delta = chunk["contentBlockDelta"].get("delta", {})
                if "text" in delta:
                    text = delta.get("text", "")
                    if text:
                        assistant_response.append(text)

        answer = "".join(assistant_response)

        # 第二轮对话，应该保留上下文
        messages.extend([
            {"role": "assistant", "content": [{"text": answer}]},
            {"role": "user", "content": [{"text": "Now divide by 3"}]}
        ])

        chunks: list[StreamEvent] = []
        async for chunk in model.stream(messages):
            chunks.append(chunk)  # type: ignore[arg-type]

        assert len(chunks) > 0


class TestKimiOpenAIModelEdgeCases:
    """KimiOpenAIModel 边界情况测试"""

    def test_empty_messages(self):
        """测试空消息列表"""
        result = KimiOpenAIModel.format_request_messages([])
        assert result == []

    def test_message_with_only_tool_use(self):
        """测试只有 toolUse 的消息"""
        messages: list[Message] = [
            {
                "role": "assistant",
                "content": [
                    {"toolUse": {"toolUseId": "call-1", "name": "tool", "input": {}}}
                ]
            }
        ]

        result = KimiOpenAIModel._format_regular_messages(messages)

        assert len(result) == 1
        assert "tool_calls" in result[0]
        # Tool call 必须有 reasoning_content
        assert result[0].get("reasoning_content") == ""

    def test_message_with_tool_result(self):
        """测试 toolResult 消息"""
        messages: list[Message] = [
            {
                "role": "user",
                "content": [
                    {"toolResult": {"toolUseId": "call-1", "content": [{"text": "Result"}], "status": "success"}}
                ]
            }
        ]

        result = KimiOpenAIModel._format_regular_messages(messages)

        # Tool result 应该转换为 tool 角色消息
        assert any(m.get("role") == "tool" for m in result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])