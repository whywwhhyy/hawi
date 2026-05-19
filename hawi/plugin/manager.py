"""Plugin manager for dynamic tools and hooks."""

from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING, Any, cast, Callable, TypeVar

from hawi.tool.types import (
    AgentTool,
    ToolParameterInjection,
    ToolParameterInjectionHandler,
    ToolParameterInjectionPredicate,
)
from hawi.plugin.types import HookReturnType, system_prompt_variability_rank
from hawi.models.message import ToolDefinition
from hawi.permission import (
    PermissionChecker,
    PermissionDeclared,
    PermissionSet,
    FrozenPermissionSet,
    build_tool_permission_map,
    collect_plugin_permissions,
    filter_tools,
)
from hawi.permission.types import PermissionPolicy

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
        """Initialize PluginManager"""
        self._plugin_factories = plugin_factories or []
        factory_plugins = [f() for f in self._plugin_factories]
        self._factory_plugin_count = len(factory_plugins)
        self._plugins: list[HawiPlugin] = factory_plugins + (plugins or [])

        # Collect hooks from plugins (aggregate from PluginHooks TypedDict to list)
        self._hooks: dict[str, list[Callable[..., HookReturnType]]] = {}
        self._collect_plugin_hooks()

        # Dynamic management (tools + hooks)
        self._dynamic = DynamicPlugin()
        self._masked_names: set[str] = set()
        self._parameter_injections: list[ToolParameterInjection] = []

        # Caches
        self._tools_cache: list[AgentTool] | None = None
        self._tool_defs_cache: list[ToolDefinition] | None = None
        self._event_bus: Any | None = None

        # Permission system
        self._permission_checker = PermissionChecker()
        self._tool_permissions: dict[str, list[PermissionDeclared]] = {}
        self._build_permission_map()

    def _collect_plugin_hooks(self) -> None:
        """Collect hooks from all plugins, building hook chains."""
        for plugin in self._plugins:
            plugin_hooks = plugin.hooks
            for hook_type, hook_fn in plugin_hooks.items():
                if hook_fn:
                    self._hooks.setdefault(hook_type, []).append(cast(Callable[..., HookReturnType], hook_fn))

    # --- Permission System ---

    def _build_permission_map(self) -> None:
        """Rebuild the tool → permission mapping from all loaded plugins.

        Plugin tools are named like ``PluginClass__method_name`` (derived
        from ``__qualname__``), but plugins declare permissions with short
        names.  This method resolves short names to full tool names by
        consulting each plugin's registered tools.
        """
        self._tool_permissions = {}
        for plugin in self._plugins:
            # Build short-name → full-name lookup for this plugin
            short_to_full: dict[str, str] = {}
            for tool in plugin.tools:
                if "__" in tool.name:
                    short_name = tool.name.split("__", 1)[1]
                else:
                    short_name = tool.name
                short_to_full[short_name] = tool.name

            perms = getattr(plugin, "permissions", None)
            if perms is None:
                continue
            if callable(perms):
                perms = perms()
            for decl in perms:
                for tool_name in decl.tool_names:
                    # Resolve short name to actual tool name registered
                    # by _collect_items; fall back to the declared name
                    actual_name = short_to_full.get(tool_name, tool_name)
                    self._tool_permissions.setdefault(actual_name, []).append(decl)

    def set_permission_set(
        self,
        permission_set: PermissionSet | FrozenPermissionSet | None,
    ) -> None:
        """Set or clear the active permission set for tool filtering.

        When *permission_set* is ``None``, all tools are visible (backwards
        compatible).  Setting a non-None set invalidates caches so that the
        next call to :meth:`get_tools` or :meth:`get_tool_definitions`
        reflects the new permissions.
        """
        self._permission_checker.set_permission_set(permission_set)
        self._invalidate_cache()

    @property
    def permission_set(self) -> PermissionSet | FrozenPermissionSet | None:
        """The active permission set, if any."""
        return self._permission_checker.permission_set

    def check_tool_permission(self, tool_name: str) -> PermissionPolicy:
        """Return the effective permission policy for *tool_name*."""
        return self._permission_checker.check_tool_permission(
            tool_name,
            tool_permissions=self._tool_permissions,
        )

    def get_permission_checker(self) -> PermissionChecker:
        """Return the internal :class:`PermissionChecker` instance."""
        return self._permission_checker

    def get_tool_permissions_map(self) -> dict[str, list[PermissionDeclared]]:
        """Return the ``{tool_name: [PermissionDeclared]}`` mapping."""
        return dict(self._tool_permissions)

    def get_plugins(self) -> list[HawiPlugin]:
        """Return all plugins (as a copy)."""
        return list(self._plugins)

    def add_plugin(self, plugin: HawiPlugin) -> None:
        """Add a plugin instance at runtime."""
        self._plugins.append(plugin)
        plugin.bind_event_bus(self._event_bus)
        for hook_type, hook_fn in plugin.hooks.items():
            if hook_fn:
                self._hooks.setdefault(hook_type, []).append(
                    cast(Callable[..., HookReturnType], hook_fn)
                )
        self._build_permission_map()
        self._invalidate_cache()

    def add_plugin_factory(self, factory: Callable[[], HawiPlugin]) -> HawiPlugin:
        """Create and add a plugin from a factory at runtime."""
        self._plugin_factories.append(factory)
        plugin = factory()
        self._plugins.insert(self._factory_plugin_count, plugin)
        self._factory_plugin_count += 1
        plugin.bind_event_bus(self._event_bus)
        for hook_type, hook_fn in plugin.hooks.items():
            if hook_fn:
                self._hooks.setdefault(hook_type, []).append(
                    cast(Callable[..., HookReturnType], hook_fn)
                )
        self._build_permission_map()
        self._invalidate_cache()
        return plugin

    def bind_event_bus(self, event_bus: Any | None) -> None:
        """Bind an event bus to all managed plugins that support plugin events."""
        self._event_bus = event_bus
        for plugin in self._plugins:
            plugin.bind_event_bus(event_bus)

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

    def get_tool_owner(self, name: str) -> HawiPlugin | None:
        """Return the plugin instance that owns a tool, if known."""
        owner_by_name: dict[str, HawiPlugin] = {}
        for plugin in self._plugins:
            for tool in plugin.tools:
                owner_by_name[tool.name] = plugin
        for tool in self._dynamic.tools:
            if tool.name == name:
                owner = getattr(tool, "_hawi_plugin", None)
                return owner if owner is not None else None
        return owner_by_name.get(name)

    def get_tools(self) -> list[AgentTool]:
        """Get all unmasked and permission-allowed tools (cached)."""
        if self._tools_cache is None:
            all_tools = self._collect_all_tools()
            unmasked = [t for t in all_tools if t.name not in self._masked_names]
            self._tools_cache = filter_tools(
                unmasked,
                self._permission_checker,
                self._tool_permissions,
            )
        return self._tools_cache

    def get_tool_definitions(self) -> list[ToolDefinition]:
        """Get tool definitions for LLM API (cached)."""
        if self._tool_defs_cache is None:
            tools = self.get_tools()
            self._tool_defs_cache = [
                {
                    "type": "function",
                    "name": t.name,
                    "description": self.get_tool_description(t),
                    "schema": self.get_parameters_schema(t),
                }
                for t in tools
            ]
        return self._tool_defs_cache

    def get_tool_description(self, tool: AgentTool) -> str:
        """Return a tool description augmented with injected parameter docs."""
        description = tool.description
        injections = self.get_tool_parameter_injections(tool)
        if not injections:
            return description

        lines = [
            "",
            "Injected framework parameters (Hawi consumes these before invoking the real tool):",
        ]
        for injection in injections:
            schema = injection.schema
            schema_type = schema.get("type")
            type_label = str(schema_type) if schema_type else "any"
            required = ", required" if injection.required else ""
            parameter_description = schema.get("description")
            if parameter_description:
                lines.append(
                    f"- {injection.name} ({type_label}{required}): {parameter_description}"
                )
            else:
                lines.append(f"- {injection.name} ({type_label}{required})")
        suffix = "\n".join(lines)
        if not description.strip():
            return suffix.lstrip()
        return description.rstrip() + suffix

    def get_parameters_schema(self, tool: AgentTool) -> dict[str, Any]:
        """Return a tool schema augmented with registered injected parameters."""
        schema = copy.deepcopy(tool.parameters_schema or {})
        injections = self.get_tool_parameter_injections(tool)
        if not injections:
            return schema

        if not schema:
            schema = {"type": "object", "properties": {}, "required": []}
        if schema.get("type", "object") != "object":
            raise ValueError(
                f"Cannot inject parameters into non-object schema for tool '{tool.name}'"
            )

        properties = schema.setdefault("properties", {})
        if not isinstance(properties, dict):
            raise ValueError(
                f"Tool '{tool.name}' parameters_schema.properties must be a dict"
            )

        required = schema.get("required")
        if required is None:
            required_list: list[str] = []
        elif isinstance(required, list):
            required_list = list(required)
        else:
            required_list = list(required)

        for injection in injections:
            if injection.name in properties:
                raise ValueError(
                    f"Injected tool parameter '{injection.name}' conflicts with "
                    f"an existing parameter on tool '{tool.name}'"
                )
            properties[injection.name] = injection.schema_copy()
            if injection.required and injection.name not in required_list:
                required_list.append(injection.name)

        if required_list:
            schema["required"] = required_list
        else:
            schema.pop("required", None)
        return schema

    # --- Tool Parameter Injection ---
    def add_tool_parameter_injection(
        self,
        injection: ToolParameterInjection | None = None,
        *,
        name: str | None = None,
        schema: dict[str, Any] | None = None,
        required: bool = False,
        handler: ToolParameterInjectionHandler | None = None,
        applies_to: ToolParameterInjectionPredicate | None = None,
    ) -> ToolParameterInjection:
        """Register a framework-level parameter for matching tool schemas.

        The parameter is exposed to the model, then stripped before the actual
        tool implementation is invoked.
        """
        if injection is None:
            if name is None or schema is None:
                raise ValueError(
                    "Provide either a ToolParameterInjection or both name and schema"
                )
            injection = ToolParameterInjection(
                name=name,
                schema=schema,
                required=required,
                handler=handler,
                applies_to=applies_to,
            )

        if any(existing.name == injection.name for existing in self._parameter_injections):
            raise ValueError(
                f"Injected tool parameter '{injection.name}' is already registered"
            )

        self._parameter_injections.append(injection)
        self._invalidate_cache()
        return injection

    def remove_tool_parameter_injection(self, name: str) -> bool:
        """Remove a registered injected parameter by name."""
        original_count = len(self._parameter_injections)
        self._parameter_injections = [
            injection
            for injection in self._parameter_injections
            if injection.name != name
        ]
        removed = len(self._parameter_injections) != original_count
        if removed:
            self._invalidate_cache()
        return removed

    def get_tool_parameter_injections(
        self,
        tool: AgentTool | None = None,
    ) -> list[ToolParameterInjection]:
        """Return registered injected parameters, optionally filtered for a tool."""
        if tool is None:
            return list(self._parameter_injections)
        return [
            injection
            for injection in self._parameter_injections
            if injection.applies_to_tool(tool)
        ]

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
        hooks = plugin_hooks + dynamic_hooks
        if hook_type in {"before_session", "before_conversation"}:
            return [
                hook
                for _, hook in sorted(
                    enumerate(hooks),
                    key=lambda item: (system_prompt_variability_rank(item[1]), item[0]),
                )
            ]
        return hooks

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
        explicit_plugins = self._plugins[self._factory_plugin_count:]
        cloned_plugins = [p.clone() for p in explicit_plugins]
        for source, clone in zip(explicit_plugins, cloned_plugins):
            clone.bind_plugin_identity(
                plugin_id=getattr(source, "_plugin_id", None),
                plugin_name=getattr(source, "_plugin_name", None),
            )

        # 2. Create new manager
        new_manager = PluginManager(
            plugins=cloned_plugins,
            plugin_factories=self._plugin_factories.copy(),
        )
        new_manager.bind_event_bus(self._event_bus)

        # 3. Clone dynamic tools
        for tool in self._dynamic.tools:
            # dynamic tools are registered as instances, thus here we use same instances
            new_manager._dynamic.add_tool(tool)

        # 4. Copy dynamic hooks (function references, shared same function object)
        for hook_type, hooks in self._dynamic._hooks.items():
            for hook_fn in hooks:
                new_manager._dynamic.add_hook(hook_type, hook_fn)

        # 5. Copy mask state
        new_manager._masked_names = self._masked_names.copy()

        # 6. Copy injected parameter definitions (immutable dataclasses)
        new_manager._parameter_injections = self._parameter_injections.copy()

        # 7. Copy permission state
        ps = self._permission_checker.permission_set
        if ps is not None:
            new_manager._permission_checker.set_permission_set(
                ps.freeze() if hasattr(ps, "freeze") else ps
            )
        new_manager._tool_permissions = {
            k: list(v) for k, v in self._tool_permissions.items()
        }

        return new_manager
