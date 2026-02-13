from __future__ import annotations
from typing import TYPE_CHECKING

from .types import PluginHooks

if TYPE_CHECKING:
    from hawi.tool.types import AgentTool
    from hawi.resources import HawiResource

class HawiPlugin:
    """Base class for Hawi plugins.

    Plugins can provide:
    - hooks: Lifecycle hooks
    - tools: Custom tools for agents to use
    - resources: Contextual data/resources for agents (MCP-compatible)
    """
    _cached_hooks:PluginHooks
    _cached_tools:list[AgentTool]

    def _collect_items(self):
        def _parse_object(obj: "HawiPlugin") -> tuple[PluginHooks, list[AgentTool]]:
            """Parse a plugin object to extract hooks and tools.

            Args:
                obj: The plugin object to parse.

            Returns:
                Tuple of (hooks dict, list of tools).
            """
            from hawi.tool.function_tool import FunctionAgentTool
            from hawi.tool import tool as create_tool

            hooks: PluginHooks = {}
            tools: list[AgentTool] = []

            # Skip these properties to avoid triggering recursion
            _skip_names = {"hooks", "tools", "resources", "_cached_hooks", "_cached_tools"}

            for name in dir(obj):
                if name in _skip_names:
                    continue
                member = getattr(obj, name, None)
                if getattr(member, "_is_hook", None) is True:
                    hook_type = getattr(member, "_hook_type")
                    hooks[hook_type] = member
                if getattr(member, "_is_agent_tool", None) is True and callable(member):
                    agent_tools_kwargs = getattr(member, "_agent_tool_parameters", {})
                    tools.append(create_tool(**agent_tools_kwargs)(member))
            return hooks, tools
        if not hasattr(self, "_cached_hooks") or not hasattr(self, "_cached_tools"):
            self._cached_hooks,self._cached_tools = _parse_object(self)

    @property
    def hooks(self) -> PluginHooks:
        """Lifecycle hooks."""
        self._collect_items()
        return self._cached_hooks

    @property
    def tools(self) -> list[AgentTool]:
        """Tools provided by this plugin."""
        self._collect_items()
        return self._cached_tools

    @property
    def resources(self) -> list[HawiResource]:
        """Resources provided by this plugin (MCP-compatible).

        Resources provide contextual data to agents, identified by URI.
        They can be text or binary, static or dynamic.
        """
        return []
