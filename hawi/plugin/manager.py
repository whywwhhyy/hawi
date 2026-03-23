"""Plugin manager for dynamic tools and hooks."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, cast, Callable, TypeVar

from hawi.tool.types import AgentTool
from hawi.plugin.types import HookReturnType
from hawi.models.message import ToolDefinition

if TYPE_CHECKING:
    from hawi.plugin import HawiPlugin

P = TypeVar("P", bound="HawiPlugin")


class DynamicPlugin:
    """Internal container that holds all dynamically-added tools and hooks."""

    def __init__(self) -> None:
        self._tools: dict[str, AgentTool] = {}
        self._hooks: dict[str, list[Callable[..., HookReturnType]]] = {}

    # --- Tool management ---
    def add_tool(self, tool: AgentTool) -> None:
        """Add a tool to the dynamic plugin."""
        self._tools[tool.name] = tool

    def remove_tool(self, name: str) -> bool:
        """Remove a tool by name. Returns True if removed, False if not found."""
        return self._tools.pop(name, None) is not None

    @property
    def tools(self) -> list[AgentTool]:
        """Get all tools as a list."""
        return list(self._tools.values())

    # --- Hook management ---
    def add_hook(
        self,
        hook_type: str,
        hook_fn: Callable[..., HookReturnType],
    ) -> None:
        """Add a hook function for a specific hook type."""
        self._hooks.setdefault(hook_type, []).append(hook_fn)

    def remove_hook(
        self,
        hook_type: str,
        hook_fn: Callable[..., HookReturnType],
    ) -> bool:
        """Remove a hook function. Returns True if removed, False if not found."""
        hooks = self._hooks.get(hook_type, [])
        if hook_fn in hooks:
            hooks.remove(hook_fn)
            return True
        return False

    def get_hooks(self, hook_type: str) -> list[Callable[..., HookReturnType]]:
        """Get all hooks for a specific hook type."""
        return list(self._hooks.get(hook_type, []))


class PluginManager:
    """Centralized manager for plugins, tools, and hooks."""

    def __init__(
        self,
        plugins: list[HawiPlugin] | None = None,
        plugin_factories: list[Callable[[], HawiPlugin]] | None = None,
    ) -> None:
        from hawi.plugin import HawiPlugin

        self._plugin_factories = plugin_factories or []
        factory_plugins = [f() for f in self._plugin_factories]
        self._plugins: list[HawiPlugin] = factory_plugins + list(plugins) if plugins else factory_plugins

        # Collect hooks from plugins (aggregate from PluginHooks TypedDict to list)
        self._hooks: dict[str, list[Callable[..., HookReturnType]]] = {}
        self._collect_plugin_hooks()

        # Dynamic management (tools + hooks)
        self._dynamic = DynamicPlugin()
        self._masked_names: set[str] = set()

        # Caches
        self._tools_cache: list[AgentTool] | None = None
        self._tool_defs_cache: list[ToolDefinition] | None = None

    def _collect_plugin_hooks(self) -> None:
        """Collect hooks from all plugins, building hook chains."""
        for plugin in self._plugins:
            plugin_hooks = plugin.hooks
            for hook_type, hook_fn in plugin_hooks.items():
                if hook_fn:
                    self._hooks.setdefault(hook_type, []).append(cast(Callable[..., HookReturnType], hook_fn))

    def get_plugins(self) -> list[HawiPlugin]:
        """Return all plugins (as a copy)."""
        return list(self._plugins)

    def _invalidate_cache(self) -> None:
        """Invalidate tool and tool definition caches."""
        self._tools_cache = None
        self._tool_defs_cache = None

    # --- Plugin Query ---
    def get_plugin(self, plugin_type: type[P]) -> P | None:
        """Find the first plugin instance matching the given type."""
        for plugin in self._plugins:
            if isinstance(plugin, plugin_type):
                return plugin
        return None

    # --- Tool Query ---
    def get_tool(self, name: str) -> AgentTool | None:
        """Get a tool by name (dynamic tools take precedence over plugin tools)."""
        # 1. Check dynamic tools first
        for tool in self._dynamic.tools:
            if tool.name == name:
                return tool
        # 2. Check plugin tools
        for plugin in self._plugins:
            for tool in plugin.tools:
                if tool.name == name:
                    return tool
        return None

    def get_tools(self) -> list[AgentTool]:
        """Get all unmasked tools (cached)."""
        if self._tools_cache is None:
            all_tools = self._collect_all_tools()
            self._tools_cache = [t for t in all_tools if t.name not in self._masked_names]
        return self._tools_cache

    def get_tool_definitions(self) -> list[ToolDefinition]:
        """Get tool definitions for LLM API (cached)."""
        if self._tool_defs_cache is None:
            tools = self.get_tools()
            self._tool_defs_cache = [
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "schema": t.parameters_schema,
                }
                for t in tools
            ]
        return self._tool_defs_cache

    # --- Dynamic Tool Management ---
    def add_tool(self, tool: AgentTool) -> None:
        """Add a tool to dynamic. Warns if it shadows a plugin tool."""
        # Check if shadowing a plugin tool
        for plugin in self._plugins:
            for pt in plugin.tools:
                if pt.name == tool.name:
                    warnings.warn(
                        f"Tool '{tool.name}' shadows plugin tool from {plugin.__class__.__name__}",
                        UserWarning,
                    )
                    break
        self._dynamic.add_tool(tool)
        self._invalidate_cache()

    def remove_tool(self, name: str) -> bool:
        """Remove a tool from dynamic. Cannot remove plugin tools."""
        if self._dynamic.remove_tool(name):
            self._invalidate_cache()
            return True
        return False

    # --- Mask Mechanism ---
    def mask_tool(self, name: str) -> None:
        """Mask a tool (hide from model). Silently ignored if tool doesn't exist."""
        self._masked_names.add(name)
        self._invalidate_cache()

    def unmask_tool(self, name: str) -> None:
        """Unmask a tool. Silently ignored if not masked."""
        self._masked_names.discard(name)
        self._invalidate_cache()

    def is_masked(self, name: str) -> bool:
        """Check if a tool is masked."""
        return name in self._masked_names

    def _collect_all_tools(self) -> list[AgentTool]:
        """Collect all tools (unfiltered), dynamic tools override plugin tools."""
        tools: dict[str, AgentTool] = {}
        # Plugin tools (in registration order, later overrides earlier)
        for plugin in self._plugins:
            for tool in plugin.tools:
                tools[tool.name] = tool
        # Dynamic tools (override plugin tools)
        for tool in self._dynamic.tools:
            tools[tool.name] = tool
        return list(tools.values())

    # --- Dynamic Hook Management ---
    def add_hook(
        self,
        hook_type: str,
        hook_fn: Callable[..., HookReturnType],
    ) -> None:
        """Add hook to dynamic. Dynamic hooks execute after plugin hooks."""
        self._dynamic.add_hook(hook_type, hook_fn)

    def remove_hook(
        self,
        hook_type: str,
        hook_fn: Callable[..., HookReturnType],
    ) -> bool:
        """Remove hook from dynamic. Returns True if removed, False if not found."""
        return self._dynamic.remove_hook(hook_type, hook_fn)

    # --- Hook Query ---
    def get_hooks(self, hook_type: str) -> list[Callable[..., HookReturnType]]:
        """Get hook chain for a specific type (plugin hooks + dynamic hooks)."""
        plugin_hooks = self._hooks.get(hook_type, [])
        dynamic_hooks = self._dynamic.get_hooks(hook_type)
        return plugin_hooks + dynamic_hooks

    # --- Clone ---
    def clone(self) -> "PluginManager":
        """Create an independent copy of the plugin manager.

        Creates a new PluginManager with:
        - Cloned plugins (using each plugin's clone() method)
        - Copied plugin factories
        - Cloned dynamic tools (if they have clone() method, otherwise shared)
        - Shared dynamic hooks (function references)
        - Copied mask state

        Returns:
            A new PluginManager instance that is independent of the original.
        """
        # 1. Clone all plugins
        cloned_plugins = [p.clone() for p in self._plugins]

        # 2. Create new manager
        new_manager = PluginManager(
            plugins=cloned_plugins,
            plugin_factories=self._plugin_factories.copy(),
        )

        # 3. Clone dynamic tools
        for tool in self._dynamic.tools:
            # dynamic tools are registered as insatnces, thus here we use same instances
            new_manager._dynamic.add_tool(tool)

        # 4. Copy dynamic hooks (function references, shared same function object)
        for hook_type, hooks in self._dynamic._hooks.items():
            for hook_fn in hooks:
                new_manager._dynamic.add_hook(hook_type, hook_fn)

        # 5. Copy mask state
        new_manager._masked_names = self._masked_names.copy()

        return new_manager
