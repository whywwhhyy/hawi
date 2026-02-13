"""Unit tests for plugin hook system.

Tests the plugin lifecycle hooks including before/after session, model call,
and tool calling hooks.
"""

import asyncio
import pytest
from unittest.mock import MagicMock

from hawi.plugin import HawiPlugin
from hawi.plugin.decorators import (
    before_session,
    after_session,
    before_conversation,
    after_conversation,
    before_model_call,
    after_model_call,
    before_tool_calling,
    after_tool_calling,
    tool as tool_decorator,
)
from hawi.tool.types import AgentTool, ToolResult


class MockAgent:
    """Mock agent for testing."""
    def __init__(self):
        self.context = MagicMock()
        self.context.messages = []


class SimplePlugin(HawiPlugin):
    """Simple plugin for testing."""

    def __init__(self):
        self.called_hooks = []

    @before_session
    def on_before_session(self, agent):
        self.called_hooks.append("before_session")

    @after_session
    def on_after_session(self, agent):
        self.called_hooks.append("after_session")


class AsyncPlugin(HawiPlugin):
    """Plugin with async hooks."""

    def __init__(self):
        self.called_hooks = []

    @before_tool_calling
    async def on_before_tool(self, agent, tool_name, arguments):
        self.called_hooks.append(f"before_tool_calling:{tool_name}")
        # Simulate async operation
        await asyncio.sleep(0)

    @after_tool_calling
    async def on_after_tool(self, agent, tool_name, arguments, result):
        self.called_hooks.append(f"after_tool_calling:{tool_name}")


class AllHooksPlugin(HawiPlugin):
    """Plugin implementing all hook types."""

    def __init__(self):
        self.hook_calls = []

    @before_session
    def on_before_session(self, agent):
        self.hook_calls.append(("before_session", agent))

    @after_session
    def on_after_session(self, agent):
        self.hook_calls.append(("after_session", agent))

    @before_conversation
    def on_before_conversation(self, agent):
        self.hook_calls.append(("before_conversation", agent))

    @after_conversation
    def on_after_conversation(self, agent):
        self.hook_calls.append(("after_conversation", agent))

    @before_model_call
    def on_before_model(self, agent, context, model):
        self.hook_calls.append(("before_model_call", agent, context, model))

    @after_model_call
    def on_after_model(self, agent, context, response):
        self.hook_calls.append(("after_model_call", agent, context, response))

    @before_tool_calling
    def on_before_tool(self, agent, tool_name, arguments):
        self.hook_calls.append(("before_tool_calling", agent, tool_name, arguments))

    @after_tool_calling
    def on_after_tool(self, agent, tool_name, arguments, result):
        self.hook_calls.append(("after_tool_calling", agent, tool_name, arguments, result))


class ToolInterventionPlugin(HawiPlugin):
    """Plugin that intervenes in tool calls."""

    def __init__(self):
        self.interventions = []

    @before_tool_calling
    def on_before_tool(self, agent, tool_name, arguments):
        # Add timeout to dangerous tools
        if tool_name == "dangerous":
            arguments["timeout"] = 30
            self.interventions.append(f"added_timeout:{tool_name}")

    @after_tool_calling
    def on_after_tool(self, agent, tool_name, arguments, result):
        # Log all tool results
        self.interventions.append(f"logged:{tool_name}:{result.success}")


class ModelInterceptorPlugin(HawiPlugin):
    """Plugin that modifies model calls."""

    @before_model_call
    def on_before_model(self, agent, context, model):
        # Add system message modifier
        if context.system_prompt:
            context.system_prompt.append({"type": "text", "text": "[Modified by plugin]"})


class TestConvenienceDecorators:
    """Tests for convenience decorators."""

    def test_decorator_registers_hook(self):
        """Test that decorator registers hook method."""
        plugin = SimplePlugin()
        hooks = plugin.hooks

        assert "before_session" in hooks
        assert "after_session" in hooks
        assert callable(hooks["before_session"])
        assert callable(hooks["after_session"])

    def test_decorator_preserves_method(self):
        """Test that decorated method still works as normal method."""
        plugin = SimplePlugin()
        agent = MockAgent()

        # Can still call directly
        plugin.on_before_session(agent)
        assert "before_session" in plugin.called_hooks

    def test_before_session_decorator(self):
        """Test @before_session decorator."""
        class TestPlugin(HawiPlugin):
            @before_session
            def on_start(self, agent):
                self.called = True

        plugin = TestPlugin()
        assert "before_session" in plugin.hooks

        agent = MockAgent()
        plugin.hooks["before_session"](agent)
        assert plugin.called

    def test_after_session_decorator(self):
        """Test @after_session decorator."""
        class TestPlugin(HawiPlugin):
            @after_session
            def on_end(self, agent):
                self.called = True

        plugin = TestPlugin()
        assert "after_session" in plugin.hooks

    def test_before_conversation_decorator(self):
        """Test @before_conversation decorator."""
        class TestPlugin(HawiPlugin):
            @before_conversation
            def on_conv_start(self, agent):
                pass

        plugin = TestPlugin()
        assert "before_conversation" in plugin.hooks

    def test_after_conversation_decorator(self):
        """Test @after_conversation decorator."""
        class TestPlugin(HawiPlugin):
            @after_conversation
            def on_conv_end(self, agent):
                pass

        plugin = TestPlugin()
        assert "after_conversation" in plugin.hooks

    def test_before_model_call_decorator(self):
        """Test @before_model_call decorator."""
        class TestPlugin(HawiPlugin):
            @before_model_call
            def on_model_start(self, agent, context, model):
                pass

        plugin = TestPlugin()
        assert "before_model_call" in plugin.hooks

    def test_after_model_call_decorator(self):
        """Test @after_model_call decorator."""
        class TestPlugin(HawiPlugin):
            @after_model_call
            def on_model_end(self, agent, context, response):
                pass

        plugin = TestPlugin()
        assert "after_model_call" in plugin.hooks

    def test_before_tool_calling_decorator(self):
        """Test @before_tool_calling decorator."""
        class TestPlugin(HawiPlugin):
            @before_tool_calling
            def on_tool_start(self, agent, tool_name, arguments):
                pass

        plugin = TestPlugin()
        assert "before_tool_calling" in plugin.hooks

    def test_after_tool_calling_decorator(self):
        """Test @after_tool_calling decorator."""
        class TestPlugin(HawiPlugin):
            @after_tool_calling
            def on_tool_end(self, agent, tool_name, arguments, result):
                pass

        plugin = TestPlugin()
        assert "after_tool_calling" in plugin.hooks


class TestHawiPluginBase:
    """Tests for HawiPlugin base class."""

    def test_empty_plugin_has_no_hooks(self):
        """Test that empty plugin has no hooks."""
        class EmptyPlugin(HawiPlugin):
            pass

        plugin = EmptyPlugin()
        assert plugin.hooks == {}

    def test_empty_plugin_has_no_tools(self):
        """Test that empty plugin has no tools."""
        class EmptyPlugin(HawiPlugin):
            pass

        plugin = EmptyPlugin()
        assert plugin.tools == []

    def test_empty_plugin_has_no_resources(self):
        """Test that empty plugin has no resources."""
        class EmptyPlugin(HawiPlugin):
            pass

        plugin = EmptyPlugin()
        assert plugin.resources == []

    def test_plugin_with_tools(self):
        """Test plugin with tools."""
        class MockTool(AgentTool):
            @property
            def name(self): return "mock_tool"
            @property
            def description(self): return "A mock tool"
            @property
            def parameters_schema(self): return {}
            def run(self, **kwargs): return ToolResult(True)

        class ToolPlugin(HawiPlugin):
            def __init__(self):
                self._tools = [MockTool()]

            @property
            def tools(self):
                return self._tools

        plugin = ToolPlugin()
        assert len(plugin.tools) == 1
        assert plugin.tools[0].name == "mock_tool"


class TestHookExecution:
    """Tests for hook execution."""

    def test_before_session_hook_execution(self):
        """Test before_session hook execution."""
        plugin = SimplePlugin()
        agent = MockAgent()

        hook_fn = plugin.hooks["before_session"]
        hook_fn(agent)

        assert "before_session" in plugin.called_hooks

    def test_tool_intervention_hook(self):
        """Test tool intervention via before_tool_calling hook."""
        plugin = ToolInterventionPlugin()
        agent = MockAgent()
        arguments = {}

        hook_fn = plugin.hooks["before_tool_calling"]
        hook_fn(agent, "dangerous", arguments)

        assert arguments["timeout"] == 30
        assert "added_timeout:dangerous" in plugin.interventions

    def test_after_tool_logging_hook(self):
        """Test after_tool_calling hook for logging."""
        plugin = ToolInterventionPlugin()
        agent = MockAgent()
        result = ToolResult(success=True, output="done")

        hook_fn = plugin.hooks["after_tool_calling"]
        hook_fn(agent, "test_tool", {}, result)

        assert "logged:test_tool:True" in plugin.interventions

    def test_model_modification_hook(self):
        """Test before_model_call hook modifying context."""
        plugin = ModelInterceptorPlugin()
        agent = MockAgent()
        context = MagicMock()
        context.system_prompt = [{"type": "text", "text": "Original"}]
        model = MagicMock()

        hook_fn = plugin.hooks["before_model_call"]
        hook_fn(agent, context, model)

        # Check that modifier was added
        assert len(context.system_prompt) == 2
        assert "[Modified by plugin]" in context.system_prompt[1]["text"]

    def test_all_hooks_plugin(self):
        """Test plugin with all hook types."""
        plugin = AllHooksPlugin()
        agent = MockAgent()
        context = MagicMock()
        model = MagicMock()
        response = MagicMock()

        hooks = plugin.hooks

        # Execute all hooks
        hooks["before_session"](agent)
        hooks["before_conversation"](agent)
        hooks["before_model_call"](agent, context, model)
        hooks["after_model_call"](agent, context, response)
        hooks["before_tool_calling"](agent, "test_tool", {})
        hooks["after_tool_calling"](agent, "test_tool", {}, ToolResult(True))
        hooks["after_conversation"](agent)
        hooks["after_session"](agent)

        assert len(plugin.hook_calls) == 8
        assert plugin.hook_calls[0][0] == "before_session"
        assert plugin.hook_calls[-1][0] == "after_session"


class TestAsyncHooks:
    """Tests for async hook execution."""

    @pytest.mark.asyncio
    async def test_async_before_tool_hook(self):
        """Test async before_tool_calling hook."""
        plugin = AsyncPlugin()
        agent = MockAgent()

        hook_fn = plugin.hooks["before_tool_calling"]
        await hook_fn(agent, "my_tool", {})

        assert "before_tool_calling:my_tool" in plugin.called_hooks

    @pytest.mark.asyncio
    async def test_async_after_tool_hook(self):
        """Test async after_tool_calling hook."""
        plugin = AsyncPlugin()
        agent = MockAgent()
        result = ToolResult(success=True)

        hook_fn = plugin.hooks["after_tool_calling"]
        await hook_fn(agent, "my_tool", {}, result)

        assert "after_tool_calling:my_tool" in plugin.called_hooks


class TestMultiplePlugins:
    """Tests for multiple plugins with hooks."""

    def test_multiple_plugins_hooks(self):
        """Test that multiple plugins can have different hooks."""
        plugin1 = SimplePlugin()
        plugin2 = ToolInterventionPlugin()

        # Each plugin has its own hooks
        assert "before_session" in plugin1.hooks
        assert "before_session" not in plugin2.hooks
        assert "before_tool_calling" in plugin2.hooks
        assert "before_tool_calling" not in plugin1.hooks

    def test_hook_isolation(self):
        """Test that plugin hooks are isolated."""
        plugin1 = SimplePlugin()
        plugin2 = SimplePlugin()
        agent = MockAgent()

        # Call hook on plugin1
        plugin1.hooks["before_session"](agent)

        # Only plugin1 should have the call recorded
        assert "before_session" in plugin1.called_hooks
        assert "before_session" not in plugin2.called_hooks


class TestToolDecoratorIntegration:
    """Tests for @tool decorator in plugins."""

    def test_tool_decorator_creates_tool(self):
        """Test that @tool decorator marks function for plugin discovery."""

        @tool_decorator()
        def calculator(expression: str) -> float:
            """Calculate expression."""
            return eval(expression)

        # tool_decorator returns the original function with markers
        assert callable(calculator)
        assert calculator.__name__ == "calculator"
        assert hasattr(calculator, '_is_agent_tool')
        assert calculator._is_agent_tool is True

    def test_tool_decorator_tool_invocation(self):
        """Test that decorated function can still be called directly."""

        @tool_decorator()
        def calculator(expression: str) -> float:
            """Calculate expression."""
            return eval(expression)

        # The decorated function remains callable
        result = calculator("1 + 2")
        assert result == 3.0

    def test_tool_decorator_creates_plugin_compatible_tool(self):
        """Test that @tool decorator creates plugin-compatible tool."""

        @tool_decorator()
        def calculator(expression: str) -> float:
            """Calculate expression."""
            return eval(expression)

        # Function is marked for plugin discovery
        assert hasattr(calculator, '_is_agent_tool')
        assert calculator._is_agent_tool is True
        # Tool parameters are stored for later use
        assert hasattr(calculator, '_agent_tool_parameters')
