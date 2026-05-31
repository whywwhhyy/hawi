"""Tests for system_prompt handling across different model adapters.

This test suite verifies the refactored system_prompt design:
- Hawi uses list[ContentPart] for system_prompt
- OpenAI: converts to first message with role="system" (or "developer" for o1/o3)
- Anthropic: uses top-level system field with multi-block support
"""


import pytest

from hawi.models.message import MessageRequest, ContentPart
from hawi.agent.context import AgentContext
from hawi.agent import HawiAgent
from hawi.models import Model
from hawi.models.openai._converters import prepare_request as openai_prepare_request
from hawi.models.anthropic._utils import convert_system_prompt
from hawi.plugin import HawiPlugin, HookContext, before_conversation, before_session


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

    def test_message_request_normalizes_cache_point_booleans(self):
        """MessageRequest cache settings should accept convenience booleans."""
        request = MessageRequest(
            messages=[],
            cache_point=True,
            cache_tool_definitions=False,
        )

        assert request.cache_point == {"type": "ephemeral"}
        assert request.cache_tool_definitions is None

    def test_agent_context_default_system_prompt(self):
        """AgentContext should default system_prompt to None."""
        ctx = AgentContext()
        assert ctx.system_prompt is None

    def test_agent_context_set_system_prompt_string(self):
        """AgentContext.set_system_prompt should convert string to ContentPart list."""
        ctx = AgentContext()
        ctx.set_system_prompt("You are helpful.")

        assert ctx.system_prompt == [{"type": "text", "text": "You are helpful."}]
        assert ctx.get_base_system_prompt() == [{"type": "text", "text": "You are helpful."}]

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

    def test_agent_context_keeps_base_system_prompt_separate_from_injections(self):
        """Plugin-injected prompt parts should not overwrite the configured base prompt."""
        ctx = AgentContext(system_prompt=[{"type": "text", "text": "Base prompt"}])

        system_prompt = list(ctx.system_prompt or [])
        system_prompt.append({"type": "text", "text": "Injected prompt"})
        ctx.system_prompt = system_prompt

        assert ctx.get_base_system_prompt() == [{"type": "text", "text": "Base prompt"}]
        assert ctx.get_system_prompt() == [
            {"type": "text", "text": "Base prompt"},
            {"type": "text", "text": "Injected prompt"},
        ]

    def test_agent_context_prepare_request_with_system(self):
        """AgentContext.prepare_request should include system_prompt."""
        ctx = AgentContext()
        ctx.set_system_prompt("You are helpful.")
        ctx.add_user_message("Hello")

        request = ctx.prepare_request()
        assert request.system == [{"type": "text", "text": "You are helpful."}]
        assert len(request.messages) == 1

    def test_agent_context_prepare_request_with_cache_points(self):
        """AgentContext should carry cache point settings into MessageRequest."""
        ctx = AgentContext(
            tool_definitions=[
                {
                    "type": "function",
                    "name": "example",
                    "description": "Example tool",
                    "schema": {"type": "object", "properties": {}},
                }
            ]
        )
        ctx.set_cache_point(True)
        ctx.set_tool_cache_point({"type": "ephemeral", "ttl": "1h"})

        request = ctx.prepare_request()

        assert request.cache_point == {"type": "ephemeral"}
        assert request.cache_tool_definitions == {"type": "ephemeral", "ttl": "1h"}
        assert request.tools is not None
        assert request.tools[0]["cache_point"] == {"type": "ephemeral", "ttl": "1h"}

    def test_agent_context_auto_cache_static_prefix_marks_system_ir(self):
        """AgentContext should manage automatic static-prefix cache points in Hawi IR."""
        ctx = AgentContext(
            system_prompt=[{"type": "text", "text": "You are helpful."}],
            tool_definitions=[_tool_definition("example")],
        )
        ctx.set_static_prefix_cache_point(True)

        request = ctx.prepare_request()

        assert request.system == [
            {"type": "text", "text": "You are helpful."},
            {"type": "cache_point", "cache_point": {"type": "ephemeral"}},
        ]
        assert request.tools is not None
        assert "cache_point" not in request.tools[0]
        assert request.cache_point is None
        assert request.cache_tool_definitions is None

    def test_agent_context_auto_cache_static_prefix_marks_tools_without_system(self):
        """AgentContext should fall back to tool definition cache metadata."""
        ctx = AgentContext(tool_definitions=[_tool_definition("example")])
        ctx.set_static_prefix_cache_point(True)

        request = ctx.prepare_request()

        assert request.system is None
        assert request.tools is not None
        assert request.tools[0]["cache_point"] == {"type": "ephemeral"}

    def test_agent_context_auto_cache_static_prefix_preserves_explicit_markers(self):
        """Automatic cache point management should not add duplicate markers."""
        system: list[ContentPart] = [
            {"type": "text", "text": "Large system prompt"},
            {"type": "cache_point", "cache_point": {"type": "ephemeral"}},
        ]
        ctx = AgentContext(
            system_prompt=system,
            tool_definitions=[_tool_definition("example")],
        )
        ctx.set_static_prefix_cache_point(True)

        request = ctx.prepare_request()

        assert request.system == system
        assert request.tools is not None
        assert "cache_point" not in request.tools[0]


class TestHawiAgentSystemPrompt:
    """Test HawiAgent system_prompt handling."""

    def test_hawi_agent_accepts_string_system_prompt(self):
        """HawiAgent should accept string system_prompt."""
        from hawi.models.deepseek import DeepSeekModel

        agent = HawiAgent(
            model=DeepSeekModel(api_key="test"),
            system_prompt="You are helpful.",
        )

        assert agent.context.system_prompt == [{"type": "text", "text": "You are helpful."}]

    def test_hawi_agent_accepts_list_system_prompt(self):
        """HawiAgent should accept list[ContentPart] system_prompt."""
        from hawi.models.deepseek import DeepSeekModel

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
        from hawi.models.deepseek import DeepSeekModel

        agent = HawiAgent(
            model=DeepSeekModel(api_key="test"),
            system_prompt=None,
        )

        assert agent.context.system_prompt is None

    def test_hawi_agent_clone_preserves_system_prompt(self):
        """Cloned agent should preserve system_prompt."""
        from hawi.models.deepseek import DeepSeekModel

        agent = HawiAgent(
            model=DeepSeekModel(api_key="test"),
            system_prompt="You are helpful.",
        )

        cloned = agent.clone()
        assert cloned.context.system_prompt == [{"type": "text", "text": "You are helpful."}]

    def test_hawi_agent_preserves_system_cache_points_for_model_call(self):
        """Agent model calls should not strip cache point markers from system IR."""
        model = _CaptureSystemModel()
        system: list[ContentPart] = [
            {"type": "text", "text": "Large system prompt"},
            {"type": "cache_point", "cache_point": {"type": "ephemeral"}},
        ]
        agent = HawiAgent(model=model, system_prompt=system)

        agent.run("hi")

        assert model.system_seen == system

    def test_hawi_agent_enables_static_prefix_cache_point_by_default(self):
        """HawiAgent should manage default static-prefix cache points before model calls."""
        model = _CaptureSystemModel()
        agent = HawiAgent(model=model, system_prompt="You are helpful.")

        agent.run("hi")

        assert model.system_seen == [
            {"type": "text", "text": "You are helpful."},
            {"type": "cache_point", "cache_point": {"type": "ephemeral"}},
        ]

    def test_hawi_agent_can_disable_static_prefix_cache_point(self):
        """HawiAgent should allow disabling automatic static-prefix cache points."""
        model = _CaptureSystemModel()
        agent = HawiAgent(
            model=model,
            system_prompt="You are helpful.",
            auto_cache_static_prefix=False,
        )

        agent.run("hi")

        assert model.system_seen == [{"type": "text", "text": "You are helpful."}]

    def test_hawi_agent_system_prompt_event_reads_updated_context_prompt(self):
        """System-prompt display events should come from AgentContext, not stale agent state."""
        model = _CaptureSystemModel()
        agent = HawiAgent(model=model, system_prompt="old prompt")
        events = []
        agent.subscribe_blocking(events.append, ["agent.system_prompt"])

        agent.context.set_system_prompt("new prompt")
        agent.run("hi")

        assert events
        assert events[0].content == [{"type": "text", "text": "new prompt"}]

    @pytest.mark.asyncio
    async def test_declared_system_prompt_hooks_sort_across_session_phases(self):
        class VariableSessionPlugin(HawiPlugin):
            @before_session(system_prompt_variability="time_hour")
            def inject_variable(self, agent, ctx):
                system_prompt = list(agent.context.system_prompt or [])
                system_prompt.append({"type": "text", "text": "variable"})
                agent.context.system_prompt = system_prompt

        class StableConversationPlugin(HawiPlugin):
            @before_conversation(system_prompt_variability="hardcoded")
            def inject_stable(self, agent, ctx):
                system_prompt = list(agent.context.system_prompt or [])
                system_prompt.append({"type": "text", "text": "stable"})
                agent.context.system_prompt = system_prompt

        agent = HawiAgent(
            model=object(),
            plugins=[VariableSessionPlugin(), StableConversationPlugin()],
            system_prompt="base",
        )

        await agent._invoke_session_hook(
            "before_session",
            HookContext(run_id="r1", iteration=0),
        )
        await agent._invoke_session_hook(
            "before_conversation",
            HookContext(run_id="r1", iteration=0),
        )

        assert agent.context.system_prompt == [
            {"type": "text", "text": "base"},
            {"type": "text", "text": "stable"},
            {"type": "text", "text": "variable"},
        ]

    @pytest.mark.asyncio
    async def test_declared_system_prompt_hooks_run_once_per_agent_session(self):
        class StableConversationPlugin(HawiPlugin):
            def __init__(self):
                self.calls = 0

            @before_conversation(system_prompt_variability="hardcoded")
            def inject_stable(self, agent, ctx):
                self.calls += 1
                system_prompt = list(agent.context.system_prompt or [])
                system_prompt.append(
                    {"type": "text", "text": f"stable {self.calls}"}
                )
                agent.context.system_prompt = system_prompt

        plugin = StableConversationPlugin()
        agent = HawiAgent(
            model=object(),
            plugins=[plugin],
            system_prompt="base",
        )

        await agent._invoke_session_hook(
            "before_conversation",
            HookContext(run_id="r1", iteration=0),
        )
        await agent._invoke_session_hook(
            "before_conversation",
            HookContext(run_id="r2", iteration=0),
        )

        assert plugin.calls == 1
        assert agent.context.system_prompt == [
            {"type": "text", "text": "base"},
            {"type": "text", "text": "stable 1"},
        ]

        agent.reset_system_prompt_injection_hooks()
        await agent._invoke_session_hook(
            "before_conversation",
            HookContext(run_id="r3", iteration=0),
        )

        assert plugin.calls == 2
        assert agent.context.system_prompt == [
            {"type": "text", "text": "base"},
            {"type": "text", "text": "stable 1"},
            {"type": "text", "text": "stable 2"},
        ]


class TestOpenAISystemPromptConversion:
    """Test OpenAI converter system_prompt handling."""

    def _create_request(self, system: list[ContentPart] | None) -> MessageRequest:
        """Helper to create a MessageRequest."""
        return MessageRequest(
            messages=[{
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}],
                "name": None,
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
        from hawi.models.message import Message

        msg: Message = {
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}],
            "name": None,
            "metadata": None,
        }
        assert msg["role"] == "user"

    def test_message_role_accepts_assistant(self):
        """Message role should accept 'assistant'."""
        from hawi.models.message import Message

        msg: Message = {
            "role": "assistant",
            "content": [{"type": "text", "text": "Hi"}],
            "name": None,
            "metadata": None,
        }
        assert msg["role"] == "assistant"

    def test_message_role_accepts_tool(self):
        """Message role should accept 'tool'."""
        from hawi.models.message import Message

        msg: Message = {
            "role": "tool",
            "content": [{
                "type": "tool_result",
                "tool_call_id": "123",
                "content": [{"type": "text", "text": "result"}],
                "is_error": False,
            }],
            "name": None,
            "metadata": None,
        }
        assert msg["role"] == "tool"

    def test_message_role_accepts_developer(self):
        """Message role should accept 'developer' for OpenAI o1/o3 models."""
        from hawi.models.message import Message

        # Note: 'developer' role is supported for OpenAI o1/o3 models
        # but system prompts should generally be passed via MessageRequest.system
        # Using dict annotation to allow 'developer' role which is not in the standard Message type
        msg: dict = {
            "role": "developer",  # type: ignore[assignment]
            "content": [{"type": "text", "text": "Dev message"}],
            "name": None,
            "metadata": None,
        }
        assert msg["role"] == "developer"


class TestSystemPromptWithCacheControl:
    """Test system_prompt with prompt caching."""

    def test_anthropic_cache_control_in_system_prompt(self):
        """Anthropic should support cache_control in system_prompt."""
        system: list[ContentPart] = [
            {"type": "text", "text": "Large document content..."},
            {"type": "cache_point", "cache_point": {"type": "ephemeral"}},
        ]

        result = convert_system_prompt(system)

        assert isinstance(result, list)
        assert result[0].get("cache_control") == {"type": "ephemeral"}

    def test_legacy_cache_control_in_system_prompt(self):
        """Legacy cache_control marker should still work."""
        system: list[ContentPart] = [
            {"type": "text", "text": "Large document content..."},
            {"type": "cache_control", "cache_control": {"type": "ephemeral"}},
        ]

        result = convert_system_prompt(system)

        assert isinstance(result, list)
        assert result[0].get("cache_control") == {"type": "ephemeral"}

    def test_leading_cache_point_attaches_to_next_system_block(self):
        """Leading cache marker keeps compatibility with the old converter behavior."""
        system: list[ContentPart] = [
            {"type": "cache_point", "cache_point": {"type": "ephemeral"}},
            {"type": "text", "text": "Large document content..."},
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


class _CaptureSystemModel(Model):
    default_steer_merge_mode = "tool_result_assistant_template_and_user_message"

    def __init__(self) -> None:
        super().__init__()
        self.system_seen = None

    @property
    def model_id(self) -> str:
        return "capture"

    async def ainvoke(self, messages, *, streaming=False, system=None, tools=None, tool_choice=None, **kwargs):
        self.system_seen = system
        yield {
            "type": "finish",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

    def _prepare_request_impl(self, request):
        return {}

    def _parse_response_impl(self, response):
        raise NotImplementedError

    def _invoke_impl(self, request):
        raise NotImplementedError


def _tool_definition(name: str) -> dict:
    return {
        "type": "function",
        "name": name,
        "description": f"{name} description",
        "schema": {"type": "object", "properties": {}},
    }
