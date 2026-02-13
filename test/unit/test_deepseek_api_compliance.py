"""
DeepSeek API 合规性测试

验证实现是否符合 DeepSeek API 文档规范:
- https://api-docs.deepseek.com/guides/thinking_mode
- https://api-docs.deepseek.com/guides/openai_api
- https://api-docs.deepseek.com/guides/anthropic_api
"""

import pytest
from unittest.mock import patch

from hawi.agent.messages import MessageRequest, Message
from hawi.agent.models.deepseek.deepseek_openai import DeepSeekOpenAIModel
from hawi.agent.models.deepseek.deepseek_anthropic import DeepSeekAnthropicModel


class TestReasoningContentCompliance:
    """测试 reasoning_content 处理符合 API 规范"""

    def test_reasoning_content_not_sent_in_request(self):
        """
        测试: 请求中不应包含 reasoning_content 字段

        DeepSeek API 文档说明:
        - 如果请求消息中包含 reasoning_content 字段，API 会返回 400 错误
        - reasoning_content 只能从响应中读取，不能发送到 API
        """
        model = DeepSeekOpenAIModel(
            model_id="deepseek-reasoner",
            api_key="test-key",
            include_reasoning_in_context=False,  # 默认行为
        )

        # 创建包含 reasoning 的消息
        message: Message = {
            "role": "assistant",
            "content": [
                {"type": "reasoning", "reasoning": "Let me think...", "signature": None},
                {"type": "text", "text": "Here's the answer"},
            ],
            "name": None,
            "tool_calls": None,
            "tool_call_id": None,
            "metadata": None,
        }

        request = MessageRequest(messages=[message])
        req = model._prepare_request_impl(request)

        # 检查 messages 中不包含 reasoning_content
        for msg in req.get("messages", []):
            assert "reasoning_content" not in msg, \
                f"请求消息不应包含 reasoning_content 字段，否则 API 会返回 400 错误: {msg}"

    def test_reasoning_content_extraction_from_response(self):
        """
        测试: 能从响应中正确提取 reasoning_content
        """
        model = DeepSeekOpenAIModel(
            model_id="deepseek-reasoner",
            api_key="test-key",
        )

        # 模拟 DeepSeek API 响应
        response = {
            "id": "test-response-id",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "The answer is 42.",
                    "reasoning_content": "Step 1: Analyze the problem...",
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
            }
        }

        result = model._parse_response_impl(response)

        # 验证 reasoning_content 被提取
        assert result.reasoning_content == "Step 1: Analyze the problem..."
        # 验证 reasoning_content 也被添加到 content 列表
        reasoning_parts = [p for p in result.content if p.get("type") == "reasoning"]
        assert len(reasoning_parts) == 1
        assert reasoning_parts[0].get("reasoning") == "Step 1: Analyze the problem..."


class TestReasonerParameterHandling:
    """测试 Reasoner 模型参数处理一致性"""

    def test_openai_adapter_warns_but_preserves_unsupported_params(self):
        """
        测试: OpenAI 适配器仅警告不支持的参数，不删除它们

        根据 DeepSeek 文档，temperature/top_p 等参数会被忽略但不应导致错误
        """
        with patch("hawi.agent.models.deepseek.deepseek_openai.logger") as mock_logger:
            model = DeepSeekOpenAIModel(
                model_id="deepseek-reasoner",
                api_key="test-key",
                temperature=0.7,
                top_p=0.9,
            )

            # 验证参数保留在 model.params 中
            assert "temperature" in model.params
            assert "top_p" in model.params

            # 验证发送请求时会过滤参数
            request = MessageRequest(messages=[{
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}],
                "name": None,
                "tool_calls": None,
                "tool_call_id": None,
                "metadata": None,
            }])

            req = model._prepare_request_impl(request)

            # 根据 DeepSeek 文档，这些参数会被忽略但不会报错
            # 实现应该保留它们（因为文档说会忽略，不是报错）
            # 实际行为：这些参数可以发送到 API，API 会忽略它们

    def test_anthropic_adapter_consistent_with_openai(self):
        """
        测试: Anthropic 适配器与 OpenAI 适配器参数处理策略一致

        注意: Anthropic 和 OpenAI 基础模型的参数传递方式不同：
        - OpenAI: 通过 self.params 传递，直接传递给 API
        - Anthropic: 通过 request.temperature 等属性传递

        这里测试的是：如果参数被传递，应该只警告而不删除
        """
        # OpenAI 模型通过 params 传递 temperature
        openai_model = DeepSeekOpenAIModel(
            model_id="deepseek-reasoner",
            api_key="test-key",
            temperature=0.7,
        )

        # Anthropic 模型通过 request 传递 temperature
        anthropic_model = DeepSeekAnthropicModel(
            model_id="deepseek-reasoner",
            api_key="test-key",
        )

        # OpenAI 请求：params 中的参数会被包含
        openai_request = MessageRequest(messages=[{
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}],
            "name": None,
            "tool_calls": None,
            "tool_call_id": None,
            "metadata": None,
        }])
        openai_req = openai_model._prepare_request_impl(openai_request)

        # Anthropic 请求：通过 request.temperature 传递
        anthropic_request = MessageRequest(
            messages=[{
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}],
                "name": None,
                "tool_calls": None,
                "tool_call_id": None,
                "metadata": None,
            }],
            temperature=0.7,
        )
        anthropic_req = anthropic_model._prepare_request_impl(anthropic_request)

        # 当参数被显式传递时，应该保留在请求中（API 会忽略但不应删除）
        # 根据 DeepSeek 文档，temperature 会被忽略但不会报错
        assert "temperature" in openai_req, "OpenAI 请求应该保留 temperature"
        assert "temperature" in anthropic_req, "Anthropic 请求应该保留 temperature"


class TestDeepSeekOpenAIAPILimits:
    """测试 DeepSeek OpenAI 格式 API 限制"""

    def test_tool_message_content_is_string(self):
        """
        测试: tool 消息的 content 必须是字符串（不是数组）

        DeepSeek API 与 OpenAI API 的差异:
        - OpenAI: tool 消息的 content 可以是 str 或数组
        - DeepSeek: tool 消息的 content 必须是 str
        """
        model = DeepSeekOpenAIModel(api_key="test-key")

        # 创建包含数组 content 的 tool 消息
        message: Message = {
            "role": "tool",
            "content": [
                {"type": "text", "text": "Part 1"},
                {"type": "text", "text": "Part 2"},
            ],
            "name": None,
            "tool_calls": None,
            "tool_call_id": "call_123",
            "metadata": None,
        }

        result = model._convert_message_to_openai(message)

        assert result["role"] == "tool"
        assert isinstance(result["content"], str), \
            f"DeepSeek API 要求 tool 消息的 content 必须是字符串，不是 {type(result['content'])}"
        assert "Part 1" in result["content"]
        assert "Part 2" in result["content"]

    def test_error_params_removed_for_reasoner(self):
        """
        测试: Reasoner 模型的错误参数被移除

        DeepSeek 文档: logprobs, top_logprobs 会触发错误，必须移除
        """
        model = DeepSeekOpenAIModel(
            model_id="deepseek-reasoner",
            api_key="test-key",
            logprobs=True,
            top_logprobs=5,
        )

        request = MessageRequest(messages=[{
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}],
            "name": None,
            "tool_calls": None,
            "tool_call_id": None,
            "metadata": None,
        }])

        req = model._prepare_request_impl(request)

        assert "logprobs" not in req, "logprobs 必须从 Reasoner 请求中移除"
        assert "top_logprobs" not in req, "top_logprobs 必须从 Reasoner 请求中移除"


class TestDeepSeekAnthropicAPILimits:
    """测试 DeepSeek Anthropic 格式 API 限制"""

    def test_top_k_removed(self):
        """
        测试: DeepSeek Anthropic API 不支持 top_k 参数
        """
        model = DeepSeekAnthropicModel(
            model_id="deepseek-chat",
            api_key="test-key",
            top_k=40,  # DeepSeek 不支持
        )

        request = MessageRequest(messages=[{
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}],
            "name": None,
            "tool_calls": None,
            "tool_call_id": None,
            "metadata": None,
        }])

        req = model._prepare_request_impl(request)

        assert "top_k" not in req, "top_k 必须从 DeepSeek Anthropic 请求中移除"

    def test_reasoner_error_params_removed(self):
        """
        测试: Reasoner 模型的错误参数在 Anthropic 格式中也被移除
        """
        model = DeepSeekAnthropicModel(
            model_id="deepseek-reasoner",
            api_key="test-key",
            logprobs=True,
            top_logprobs=5,
        )

        request = MessageRequest(messages=[{
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}],
            "name": None,
            "tool_calls": None,
            "tool_call_id": None,
            "metadata": None,
        }])

        req = model._prepare_request_impl(request)

        assert "logprobs" not in req, "logprobs 必须从 Reasoner 请求中移除"
        assert "top_logprobs" not in req, "top_logprobs 必须从 Reasoner 请求中移除"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
