"""Tests for the plugin manager."""

import pytest
from typing import Sequence
from hawi.plugin.manager import DynamicPlugin
from hawi.tool.types import AgentTool, ToolResult


class MockTool(AgentTool):
    @property
    def name(self):
        return "mock_tool"

    @property
    def description(self):
        return "A mock tool"

    @property
    def parameters_schema(self):
        return {}

    def run(self, **kwargs):
        return ToolResult(True)

    def clone(self):
        return MockTool()


def test_dynamic_plugin_add_tool():
    dp = DynamicPlugin()
    tool = MockTool()
    dp.add_tool(tool)
    assert len(dp.tools) == 1
    assert dp.tools[0].name == "mock_tool"


def test_dynamic_plugin_remove_tool():
    dp = DynamicPlugin()
    tool = MockTool()
    dp.add_tool(tool)
    assert dp.remove_tool("mock_tool") is True
    assert len(dp.tools) == 0


def test_dynamic_plugin_remove_nonexistent_tool():
    dp = DynamicPlugin()
    assert dp.remove_tool("nonexistent") is False


def test_dynamic_plugin_add_hook():
    dp = DynamicPlugin()

    def my_hook(agent, ctx):
        return None

    dp.add_hook("before_conversation", my_hook)
    assert len(dp.get_hooks("before_conversation")) == 1


def test_dynamic_plugin_remove_hook():
    dp = DynamicPlugin()

    def my_hook(agent, ctx):
        return None

    dp.add_hook("before_conversation", my_hook)
    assert dp.remove_hook("before_conversation", my_hook) is True
    assert len(dp.get_hooks("before_conversation")) == 0


def test_dynamic_plugin_remove_nonexistent_hook():
    dp = DynamicPlugin()

    def my_hook(agent, ctx):
        return None

    assert dp.remove_hook("before_conversation", my_hook) is False


def test_dynamic_plugin_multiple_hooks_same_type():
    dp = DynamicPlugin()

    def hook1(agent, ctx):
        return None

    def hook2(agent, ctx):
        return None

    dp.add_hook("before_tool_call", hook1)
    dp.add_hook("before_tool_call", hook2)
    hooks = dp.get_hooks("before_tool_call")
    assert len(hooks) == 2
    assert hook1 in hooks
    assert hook2 in hooks


def test_dynamic_plugin_multiple_hook_types():
    dp = DynamicPlugin()

    def hook1(agent, ctx):
        return None

    def hook2(agent, ctx):
        return None

    dp.add_hook("before_conversation", hook1)
    dp.add_hook("after_conversation", hook2)
    assert len(dp.get_hooks("before_conversation")) == 1
    assert len(dp.get_hooks("after_conversation")) == 1
    assert dp.get_hooks("before_tool_call") == []


# =============================================================================
# PluginManager Tests
# =============================================================================

from hawi.plugin import HawiPlugin
from hawi.plugin.manager import PluginManager


class TestPluginManager:
    def test_init_empty(self):
        pm = PluginManager()
        assert pm.get_plugins() == []
        assert pm.get_tools() == []

    def test_init_with_plugins(self):
        class SimplePlugin(HawiPlugin):
            pass
        plugin = SimplePlugin()
        pm = PluginManager(plugins=[plugin])
        assert pm.get_plugins() == [plugin]


class TestPluginManagerTools:
    def test_add_tool(self):
        pm = PluginManager()
        tool = MockTool()
        pm.add_tool(tool)
        assert pm.get_tool("mock_tool") == tool

    def test_get_tools_filters_masked(self):
        pm = PluginManager()
        tool = MockTool()
        pm.add_tool(tool)
        pm.mask_tool("mock_tool")
        assert pm.get_tools() == []
        assert pm.get_tool("mock_tool") == tool  # get_tool still returns

    def test_remove_tool(self):
        pm = PluginManager()
        tool = MockTool()
        pm.add_tool(tool)
        assert pm.remove_tool("mock_tool") is True
        assert pm.get_tool("mock_tool") is None

    def test_add_tool_shadows_plugin_tool_warning(self):
        """Test that adding a dynamic tool with same name as plugin tool warns."""
        class ToolPlugin(HawiPlugin):
            @property
            def tools(self) -> list[AgentTool]:
                return [MockTool()]

        pm = PluginManager(plugins=[ToolPlugin()])
        new_tool = MockTool()
        with pytest.warns(UserWarning, match="shadows plugin tool"):
            pm.add_tool(new_tool)

    def test_unmask_tool(self):
        pm = PluginManager()
        tool = MockTool()
        pm.add_tool(tool)
        pm.mask_tool("mock_tool")
        assert pm.get_tools() == []
        pm.unmask_tool("mock_tool")
        assert pm.get_tools() == [tool]

    def test_is_masked(self):
        pm = PluginManager()
        assert pm.is_masked("mock_tool") is False
        pm.mask_tool("mock_tool")
        assert pm.is_masked("mock_tool") is True

    def test_get_tool_definitions(self):
        pm = PluginManager()
        tool = MockTool()
        pm.add_tool(tool)
        defs = pm.get_tool_definitions()
        assert len(defs) == 1
        assert defs[0]["name"] == "mock_tool"
        assert defs[0]["type"] == "function"

    def test_dynamic_tool_overrides_plugin_tool(self):
        """Test that dynamic tools take precedence over plugin tools."""
        class MockTool2(AgentTool):
            @property
            def name(self):
                return "mock_tool"

            @property
            def description(self):
                return "A different mock tool"

            @property
            def parameters_schema(self):
                return {}

            def run(self, **kwargs):
                return ToolResult(True)

        class ToolPlugin(HawiPlugin):
            @property
            def tools(self) -> list[AgentTool]:
                return [MockTool()]

        pm = PluginManager(plugins=[ToolPlugin()])
        # Initially returns plugin tool
        mock_tool = pm.get_tool("mock_tool")
        assert mock_tool
        assert mock_tool.description == "A mock tool"

        # Add dynamic tool with same name
        dynamic_tool = MockTool2()
        with pytest.warns(UserWarning):
            pm.add_tool(dynamic_tool)

        # Now returns dynamic tool
        mock_tool = pm.get_tool("mock_tool")
        assert mock_tool
        assert mock_tool.description == "A different mock tool"


class TestPluginManagerHooks:
    def test_get_hooks_empty(self):
        pm = PluginManager()
        assert pm.get_hooks("before_conversation") == []

    def test_add_hook(self):
        pm = PluginManager()

        def my_hook(agent, ctx):
            return None

        pm.add_hook("before_conversation", my_hook)
        assert my_hook in pm.get_hooks("before_conversation")

    def test_remove_hook(self):
        pm = PluginManager()

        def my_hook(agent, ctx):
            return None

        pm.add_hook("before_conversation", my_hook)
        assert pm.remove_hook("before_conversation", my_hook) is True
        assert my_hook not in pm.get_hooks("before_conversation")


class TestPluginManagerClone:
    def test_clone_plugins(self):
        class SimplePlugin(HawiPlugin):
            def clone(self):
                return SimplePlugin()
        plugin = SimplePlugin()
        pm = PluginManager(plugins=[plugin])
        pm2 = pm.clone()
        assert len(pm2.get_plugins()) == 1
        assert pm2.get_plugins()[0] is not plugin  # different instance

    def test_clone_dynamic_tools(self):
        pm = PluginManager()
        tool = MockTool()
        pm.add_tool(tool)
        pm2 = pm.clone()
        assert len(pm2.get_tools()) == 1
        assert pm2.get_tool("mock_tool") is not tool  # different instance

    def test_clone_mask(self):
        pm = PluginManager()
        pm.mask_tool("test_tool")
        pm2 = pm.clone()
        assert pm2.is_masked("test_tool")


# =============================================================================
# HawiAgent Integration Tests
# =============================================================================

class TestHawiAgentPluginManagerIntegration:
    """Integration tests for HawiAgent using PluginManager."""

    def test_agent_uses_plugin_manager(self):
        """HawiAgent initializes with PluginManager for plugin/tool/hook management."""
        from hawi.agent import HawiAgent

        agent = HawiAgent(model="deepseek-chat")

        # Agent should have a PluginManager
        assert hasattr(agent, '_plugin_manager')
        assert hasattr(agent, 'plugins')
        assert agent.plugins is agent._plugin_manager

    def test_agent_plugins_property_returns_plugin_manager(self):
        """agent.plugins returns the PluginManager instance."""
        from hawi.agent import HawiAgent

        agent = HawiAgent(model="deepseek-chat")

        # plugins property should return the PluginManager
        assert isinstance(agent.plugins, PluginManager)

    def test_agent_add_tool_via_plugins(self):
        """Tools can be added to agent via agent.plugins.add_tool()."""
        from hawi.agent import HawiAgent

        agent = HawiAgent(model="deepseek-chat")
        tool = MockTool()

        agent.plugins.add_tool(tool)

        assert agent.plugins.get_tool("mock_tool") is tool

    def test_agent_get_tool_definitions_via_plugins(self):
        """Tool definitions can be retrieved via agent.plugins.get_tool_definitions()."""
        from hawi.agent import HawiAgent

        agent = HawiAgent(model="deepseek-chat")
        tool = MockTool()

        agent.plugins.add_tool(tool)
        defs = agent.plugins.get_tool_definitions()

        assert len(defs) == 1
        assert defs[0]["name"] == "mock_tool"
        assert defs[0]["type"] == "function"

    def test_agent_clone_copies_plugin_manager(self):
        """Cloning an agent clones the PluginManager."""
        from hawi.agent import HawiAgent

        agent = HawiAgent(model="deepseek-chat")
        tool = MockTool()
        agent.plugins.add_tool(tool)

        cloned = agent.clone()

        # Cloned agent should have its own PluginManager
        assert cloned._plugin_manager is not agent._plugin_manager
        # But it should have the same tools
        assert cloned.plugins.get_tool("mock_tool") is not None
        assert cloned.plugins.get_tool("mock_tool") is not tool  # Different instance

    def test_agent_with_plugin_factories(self):
        """HawiAgent correctly initializes PluginManager with plugin factories."""
        from hawi.agent import HawiAgent
        from hawi.plugin import HawiPlugin

        class TestPlugin(HawiPlugin):
            def __init__(self):
                self._tools = [MockTool()]

            @property
            def tools(self) -> Sequence[AgentTool]:
                return self._tools

        def factory():
            return TestPlugin()

        agent = HawiAgent(model="deepseek-chat", plugin_factories=[factory])

        # Plugin should be created via factory
        assert len(agent.plugins.get_plugins()) == 1
        assert agent.plugins.get_tool("mock_tool") is not None

    def test_agent_with_plugins(self):
        """HawiAgent correctly initializes PluginManager with plugins."""
        from hawi.agent import HawiAgent
        from hawi.plugin import HawiPlugin

        class TestPlugin(HawiPlugin):
            @property
            def tools(self):
                return [MockTool()]

        plugin = TestPlugin()
        agent = HawiAgent(model="deepseek-chat", plugins=[plugin])

        # Plugin should be registered
        assert len(agent.plugins.get_plugins()) == 1
        assert agent.plugins.get_tool("mock_tool") is not None
