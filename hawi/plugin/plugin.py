from __future__ import annotations
from typing import TYPE_CHECKING

from .types import PluginHooks

if TYPE_CHECKING:
    from hawi.tool.types import AgentTool
    from .resource import HawiResource

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
        from hawi.tool import tool as create_tool

        self._cached_hooks = {}
        self._cached_tools = []

        # Skip these properties to avoid triggering recursion
        _skip_names = {"hooks", "tools", "resources", "_cached_hooks", "_cached_tools"}

        for name in dir(self):
            if name in _skip_names:
                continue
            member = getattr(self, name, None)
            if getattr(member, "_is_hook", None) is True:
                hook_type = getattr(member, "_hook_type")
                self._cached_hooks[hook_type] = member
            if getattr(member, "_is_agent_tool", None) is True and callable(member):
                agent_tools_kwargs = getattr(member, "_agent_tool_parameters", {})
                self._cached_tools.append(create_tool(**agent_tools_kwargs)(member))

    @property
    def hooks(self) -> PluginHooks:
        """Lifecycle hooks."""
        if not hasattr(self, "_cached_hooks"):
            self._collect_items()
        return self._cached_hooks

    @property
    def tools(self) -> list[AgentTool]:
        """Tools provided by this plugin."""
        if not hasattr(self, "_cached_tools"):
            self._collect_items()
        return self._cached_tools

    @property
    def resources(self) -> list[HawiResource]:
        """Resources provided by this plugin (MCP-compatible).

        Resources provide contextual data to agents, identified by URI.
        They can be text or binary, static or dynamic.
        """
        return []
