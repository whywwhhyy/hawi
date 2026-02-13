"""Tests for system_prompt handling across different model adapters.

This test suite verifies the refactored system_prompt design:
- Hawi uses list[ContentPart] for system_prompt
- OpenAI: converts to first message with role="system" (or "developer" for o1/o3)
- Anthropic: uses top-level system field with multi-block support
"""

import pytest
from typing import Any

from hawi.agent.messages import MessageRequest, ContentPart, TextPart
from hawi.agent.context import AgentContext
from hawi.agent import HawiAgent
from hawi.agent.models.openai._converters import prepare_request as openai_prepare_request
from hawi.agent.models.anthropic._utils import convert_system_prompt


class TestSystemPromptTypes:
    """Test system_prompt type handling in AgentContext and MessageRequest."""

    def test_message_request_accepts_content_part_list(self):
        """MessageRequest.system should accept list[ContentPart]."""
        system: list[ContentPart] = [
            {"type": "text", "text": "You are helpful."}
        ]
        request = MessageRequest(
            messages=[],
            system=system,
        )
        assert request.system == system

    def test_message_request_accepts_none(self):
        """MessageRequest.system should accept None."""
        request = MessageRequest(
            messages=[],
            system=None,
        )
        assert request.system is None

    def test_agent_context_default_system_prompt(self):
        """AgentContext should default system_prompt to None."""
        ctx = AgentContext()
        assert ctx.system_prompt is None

    def test_agent_context_set_system_prompt_string(self):
        """AgentContext.set_system_prompt should convert string to ContentPart list."""
        ctx = AgentContext()
        ctx.set_system_prompt("You are helpful.")

        assert ctx.system_prompt == [{"type": "text", "text": "You are helpful."}]

    def test_agent_context_set_system_prompt_list(self):
        """AgentContext.set_system_prompt should accept ContentPart list."""
        ctx = AgentContext()
        system: list[ContentPart] = [
            {"type": "text", "text": "You are helpful."},
            {"type": "text", "text": "Be concise."},
        ]
        ctx.set_system_prompt(system)

        assert ctx.system_prompt == system

    def test_agent_context_get_system_prompt(self):
        """AgentContext.get_system_prompt should return ContentPart list."""
        ctx = AgentContext()
        ctx.set_system_prompt("Test prompt")

        result = ctx.get_system_prompt()
        assert result == [{"type": "text", "text": "Test prompt"}]

    def test_agent_context_prepare_request_with_system(self):
        """AgentContext.prepare_request should include system_prompt."""
        ctx = AgentContext()
        ctx.set_system_prompt("You are helpful.")
        ctx.add_user_message("Hello")

        request = ctx.prepare_request()
        assert request.system == [{"type": "text", "text": "You are helpful."}]
        assert len(request.messages) == 1


class TestHawiAgentSystemPrompt:
    """Test HawiAgent system_prompt handling."""

    def test_hawi_agent_accepts_string_system_prompt(self):
        """HawiAgent should accept string system_prompt."""
        from hawi.agent.models.deepseek import DeepSeekModel

        agent = HawiAgent(
            model=DeepSeekModel(api_key="test"),
            system_prompt="You are helpful.",
        )

        assert agent.context.system_prompt == [{"type": "text", "text": "You are helpful."}]

    def test_hawi_agent_accepts_list_system_prompt(self):
        """HawiAgent should accept list[ContentPart] system_prompt."""
        from hawi.agent.models.deepseek import DeepSeekModel

        system: list[ContentPart] = [
            {"type": "text", "text": "You are helpful."},
            {"type": "text", "text": "Be concise."},
        ]
        agent = HawiAgent(
            model=DeepSeekModel(api_key="test"),
            system_prompt=system,
        )

        assert agent.context.system_prompt == system

    def test_hawi_agent_accepts_none_system_prompt(self):
        """HawiAgent should accept None system_prompt."""
        from hawi.agent.models.deepseek import DeepSeekModel

        agent = HawiAgent(
            model=DeepSeekModel(api_key="test"),
            system_prompt=None,
        )

        assert agent.context.system_prompt is None

    def test_hawi_agent_clone_preserves_system_prompt(self):
        """Cloned agent should preserve system_prompt."""
        from hawi.agent.models.deepseek import DeepSeekModel

        agent = HawiAgent(
            model=DeepSeekModel(api_key="test"),
            system_prompt="You are helpful.",
        )

        cloned = agent.clone()
        assert cloned.context.system_prompt == [{"type": "text", "text": "You are helpful."}]


class TestOpenAISystemPromptConversion:
    """Test OpenAI converter system_prompt handling."""

    def _create_request(self, system: list[ContentPart] | None) -> MessageRequest:
        """Helper to create a MessageRequest."""
        return MessageRequest(
            messages=[{
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}],
                "name": None,
                "tool_calls": None,
                "tool_call_id": None,
                "metadata": None,
            }],
            system=system,
        )

    def test_system_prompt_converted_to_system_message(self):
        """system_prompt should be converted to first message with role='system'."""
        request = self._create_request([{"type": "text", "text": "You are helpful."}])
        result = openai_prepare_request(request, "gpt-4", {})

        assert result["messages"][0] == {"role": "system", "content": "You are helpful."}
        assert result["messages"][1]["role"] == "user"

    def test_o1_model_uses_developer_role(self):
        """o1 models should use role='developer' for system_prompt."""
        request = self._create_request([{"type": "text", "text": "You are helpful."}])
        result = openai_prepare_request(request, "o1-preview", {})

        assert result["messages"][0] == {"role": "developer", "content": "You are helpful."}

    def test_o3_model_uses_developer_role(self):
        """o3 models should use role='developer' for system_prompt."""
        request = self._create_request([{"type": "text", "text": "You are helpful."}])
        result = openai_prepare_request(request, "o3-mini", {})

        assert result["messages"][0] == {"role": "developer", "content": "You are helpful."}

    def test_no_system_prompt_no_extra_message(self):
        """When system_prompt is None, no system message should be added."""
        request = self._create_request(None)
        result = openai_prepare_request(request, "gpt-4", {})

        assert result["messages"][0]["role"] == "user"
        assert len(result["messages"]) == 1

    def test_multi_part_system_prompt(self):
        """Multi-part system_prompt should be converted to content list."""
        request = self._create_request([
            {"type": "text", "text": "You are helpful."},
            {"type": "text", "text": "Be concise."},
        ])
        result = openai_prepare_request(request, "gpt-4", {})

        assert result["messages"][0]["role"] == "system"
        assert isinstance(result["messages"][0]["content"], list)
        assert len(result["messages"][0]["content"]) == 2


class TestAnthropicSystemPromptConversion:
    """Test Anthropic system_prompt handling."""

    def test_single_text_system_prompt_returns_string(self):
        """Single text part should return simple string."""
        result = convert_system_prompt([{"type": "text", "text": "You are helpful."}])

        assert result == "You are helpful."

    def test_multi_part_system_prompt_returns_list(self):
        """Multi-part system_prompt should return list of blocks."""
        result = convert_system_prompt([
            {"type": "text", "text": "You are helpful."},
            {"type": "text", "text": "Be concise."},
        ])

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == {"type": "text", "text": "You are helpful."}

    def test_cache_control_attached_to_previous_block(self):
        """Cache control marker should be attached to previous text block."""
        result = convert_system_prompt([
            {"type": "text", "text": "Long document..."},
            {"type": "cache_control", "cache_control": {"type": "ephemeral"}},
        ])

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "Long document..."
        assert result[0]["cache_control"] == {"type": "ephemeral"}

    def test_empty_system_prompt_returns_none(self):
        """Empty system_prompt should return None."""
        result = convert_system_prompt([])
        assert result is None

    def test_none_system_prompt_returns_none(self):
        """None system_prompt should return None."""
        result = convert_system_prompt(None)
        assert result is None


class TestMessageRoleValidation:
    """Test that Message role no longer supports 'system'."""

    def test_message_role_accepts_user(self):
        """Message role should accept 'user'."""
        from hawi.agent.messages import Message

        msg: Message = {
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}],
            "name": None,
            "tool_calls": None,
            "tool_call_id": None,
            "metadata": None,
        }
        assert msg["role"] == "user"

    def test_message_role_accepts_assistant(self):
        """Message role should accept 'assistant'."""
        from hawi.agent.messages import Message

        msg: Message = {
            "role": "assistant",
            "content": [{"type": "text", "text": "Hi"}],
            "name": None,
            "tool_calls": None,
            "tool_call_id": None,
            "metadata": None,
        }
        assert msg["role"] == "assistant"

    def test_message_role_accepts_tool(self):
        """Message role should accept 'tool'."""
        from hawi.agent.messages import Message

        msg: Message = {
            "role": "tool",
            "content": [{"type": "text", "text": "result"}],
            "name": None,
            "tool_calls": None,
            "tool_call_id": "123",
            "metadata": None,
        }
        assert msg["role"] == "tool"

    def test_message_role_accepts_developer(self):
        """Message role should accept 'developer' for OpenAI o1/o3 models."""
        from hawi.agent.messages import Message

        # Note: 'developer' role is supported for OpenAI o1/o3 models
        # but system prompts should generally be passed via MessageRequest.system
        msg: Message = {
            "role": "developer",
            "content": [{"type": "text", "text": "Dev message"}],
            "name": None,
            "tool_calls": None,
            "tool_call_id": None,
            "metadata": None,
        }
        assert msg["role"] == "developer"


class TestSystemPromptWithCacheControl:
    """Test system_prompt with prompt caching."""

    def test_anthropic_cache_control_in_system_prompt(self):
        """Anthropic should support cache_control in system_prompt."""
        system: list[ContentPart] = [
            {"type": "text", "text": "Large document content..."},
            {"type": "cache_control", "cache_control": {"type": "ephemeral"}},
        ]

        result = convert_system_prompt(system)

        assert isinstance(result, list)
        assert result[0].get("cache_control") == {"type": "ephemeral"}

    def test_multi_block_with_selective_caching(self):
        """Support caching only specific blocks in multi-block system prompt."""
        system: list[ContentPart] = [
            {"type": "text", "text": "Always relevant context."},
            {"type": "cache_control", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "Large document to cache."},
            {"type": "cache_control", "cache_control": {"type": "ephemeral"}},
        ]

        result = convert_system_prompt(system)

        assert isinstance(result, list)
        assert len(result) == 2
        # Both blocks should have cache_control
        assert result[0].get("cache_control") == {"type": "ephemeral"}
        assert result[1].get("cache_control") == {"type": "ephemeral"}
