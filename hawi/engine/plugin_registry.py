"""Single source of truth for engine-loadable Hawi plugins."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from importlib import import_module
from typing import Any

PLUGIN_FILESYSTEM = "hawi/filesystem"
PLUGIN_SHELL = "hawi/shell"
PLUGIN_WEB = "hawi/web"
PLUGIN_SKILLS = "hawi/skills"
PLUGIN_PYTHON_INTERPRETER = "hawi/python-interpreter"
PLUGIN_MCP = "hawi/mcp"
PLUGIN_TASKFLOW = "hawi/taskflow"
PLUGIN_PLAN = "hawi/plan"
PLUGIN_WORKFLOW = "hawi/workflow"
PLUGIN_ENVIRON_PROMPT = "hawi/environ-prompt"
PLUGIN_SUBAGENT = "hawi/subagent"

PluginFactory = Callable[[dict[str, Any]], Any | Awaitable[Any]]


@dataclass(frozen=True)
class PluginDescriptor:
    """Descriptor for a plugin the engine can advertise and instantiate."""

    key: str
    import_path: str
    factory: PluginFactory | None = None

    def load_class(self) -> type:
        """Load the plugin class lazily from ``module:ClassName``."""
        module_name, _, class_name = self.import_path.partition(":")
        if not module_name or not class_name:
            raise ValueError(f"Invalid plugin import path: {self.import_path!r}")
        module = import_module(module_name)
        plugin_cls = getattr(module, class_name)
        if not isinstance(plugin_cls, type):
            raise TypeError(f"Plugin import path is not a class: {self.import_path}")
        return plugin_cls

    @property
    def name(self) -> str:
        """Canonical plugin name advertised to the GUI and persisted config."""
        return str(getattr(self.load_class(), "name", None) or self.key)

    @property
    def dependencies(self) -> tuple[str, ...]:
        """Canonical plugin names this plugin depends on."""
        dependencies = getattr(self.load_class(), "dependencies", ()) or ()
        return tuple(str(dependency) for dependency in dependencies)

    @property
    def display_name(self) -> str:
        """Human-readable plugin name for GUI labels and plugin events."""
        plugin_cls = self.load_class()
        return str(getattr(plugin_cls, "display_name", None) or plugin_cls.__name__)

    @property
    def description(self) -> str:
        """User-facing plugin description for GUI help text."""
        return str(getattr(self.load_class(), "description", None) or "")

    async def create(self, config: dict[str, Any] | None = None) -> Any:
        """Instantiate this plugin using its descriptor-owned factory."""
        cfg = dict(config or {})
        if self.factory is None:
            plugin = self.load_class()()
        else:
            plugin = self.factory(cfg)
            if inspect.isawaitable(plugin):
                plugin = await plugin
        return plugin

    def gui_config_schema(self) -> dict[str, Any]:
        """Return the plugin GUI config schema."""
        return self.load_class().gui_config_schema()

    def gui_default_config(self) -> dict[str, Any]:
        """Return the plugin GUI default config."""
        return self.load_class().gui_default_config()

    @property
    def permissions(self) -> list[dict[str, Any]]:
        """Return the plugin's declared permissions for GUI display."""
        plugin_cls = self.load_class()
        perm_prop = getattr(plugin_cls, "permissions", None)
        if perm_prop is None:
            return []
        # Create a temporary instance to read the property
        try:
            instance = plugin_cls()
            perms = instance.permissions
        except Exception:
            return []
        result: list[dict[str, Any]] = []
        for decl in perms:
            perm = getattr(decl, "permission", None)
            if perm is None:
                continue
            result.append({
                "id": str(getattr(perm, "id", "")),
                "description": str(getattr(perm, "description", "")),
                "risk_level": str(getattr(perm, "risk_level", "medium")),
                "default_policy": str(getattr(perm, "default_policy", "deny")),
                "tool_names": list(getattr(decl, "tool_names", ())),
            })
        return result


def _create_skills_plugin(config: dict[str, Any]) -> Any:
    from hawi.builtin_plugins.skills_plugin import SkillsPlugin

    skills_dir = str(config.get("skills_dir") or ".skills")
    return SkillsPlugin(skills_dir=skills_dir)


def _create_python_interpreter_plugin(config: dict[str, Any]) -> Any:
    from hawi.builtin_plugins.python_interpreter import PythonInterpreterPlugin

    work_dir_raw = config.get("work_dir")
    work_dir = str(work_dir_raw).strip() if isinstance(work_dir_raw, str) else None
    return PythonInterpreterPlugin(
        work_dir=work_dir or None,
        print_execution=bool(config.get("print_execution", False)),
    )


async def _create_mcp_plugin(config: dict[str, Any]) -> Any:
    from hawi.builtin_plugins.mcp_plugin import MCPPlugin

    config_path = str(config.get("config_path") or "").strip()
    if not config_path:
        raise ValueError("MCP plugin requires 'config_path'.")
    plugin = MCPPlugin(config_path=config_path)
    await plugin.connect()
    return plugin


def _create_plan_plugin(config: dict[str, Any]) -> Any:
    from hawi.builtin_plugins.plan_plugin import PlanPlugin

    return PlanPlugin(
        fold_completed_tasks=bool(config.get("fold_completed_tasks", False))
    )


def _create_environ_prompt_plugin(config: dict[str, Any]) -> Any:
    from hawi.builtin_plugins.environ_prompt_plugin import EnvironPromptPlugin

    config_path = str(config.get("config_path") or "").strip() or None
    return EnvironPromptPlugin(config_path=config_path)


PLUGIN_REGISTRY: dict[str, PluginDescriptor] = {
    # EnvironPromptPlugin first because it injects session/env context into
    # the prompt before normal tool plugins run.
    PLUGIN_ENVIRON_PROMPT: PluginDescriptor(
        key=PLUGIN_ENVIRON_PROMPT,
        import_path="hawi.builtin_plugins.environ_prompt_plugin:EnvironPromptPlugin",
        factory=_create_environ_prompt_plugin,
    ),
    PLUGIN_FILESYSTEM: PluginDescriptor(
        key=PLUGIN_FILESYSTEM,
        import_path="hawi.builtin_plugins.filesystem_plugin:FileSystemPlugin",
    ),
    PLUGIN_SHELL: PluginDescriptor(
        key=PLUGIN_SHELL,
        import_path="hawi.builtin_plugins.shell_plugin:ShellPlugin",
    ),
    PLUGIN_WEB: PluginDescriptor(
        key=PLUGIN_WEB,
        import_path="hawi.builtin_plugins.web:WebPlugin",
    ),
    PLUGIN_SKILLS: PluginDescriptor(
        key=PLUGIN_SKILLS,
        import_path="hawi.builtin_plugins.skills_plugin:SkillsPlugin",
        factory=_create_skills_plugin,
    ),
    PLUGIN_PYTHON_INTERPRETER: PluginDescriptor(
        key=PLUGIN_PYTHON_INTERPRETER,
        import_path="hawi.builtin_plugins.python_interpreter:PythonInterpreterPlugin",
        factory=_create_python_interpreter_plugin,
    ),
    PLUGIN_MCP: PluginDescriptor(
        key=PLUGIN_MCP,
        import_path="hawi.builtin_plugins.mcp_plugin:MCPPlugin",
        factory=_create_mcp_plugin,
    ),
    PLUGIN_TASKFLOW: PluginDescriptor(
        key=PLUGIN_TASKFLOW,
        import_path="hawi.builtin_plugins.taskflow_plugin:TaskflowPlugin",
    ),
    PLUGIN_PLAN: PluginDescriptor(
        key=PLUGIN_PLAN,
        import_path="hawi.builtin_plugins.plan_plugin:PlanPlugin",
        factory=_create_plan_plugin,
    ),
    PLUGIN_WORKFLOW: PluginDescriptor(
        key=PLUGIN_WORKFLOW,
        import_path="hawi.builtin_plugins.workflow_plugin:WorkflowPlugin",
    ),
    PLUGIN_SUBAGENT: PluginDescriptor(
        key=PLUGIN_SUBAGENT,
        import_path="hawi.builtin_plugins.subagent_plugin:SubAgentPlugin",
    ),
}

KNOWN_PLUGINS: frozenset[str] = frozenset(PLUGIN_REGISTRY)
PLUGIN_DISPLAY_NAMES: dict[str, str] = {
    key: descriptor.display_name for key, descriptor in PLUGIN_REGISTRY.items()
}
PLUGIN_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    key: descriptor.dependencies for key, descriptor in PLUGIN_REGISTRY.items()
}


def iter_plugin_descriptors() -> Iterable[PluginDescriptor]:
    """Yield plugin descriptors in GUI/runtime display order."""
    return PLUGIN_REGISTRY.values()


def get_plugin_descriptor(key: str) -> PluginDescriptor:
    """Return the descriptor for ``key`` or raise a clear ValueError."""
    try:
        return PLUGIN_REGISTRY[key]
    except KeyError as exc:
        raise ValueError(f"Unknown plugin key: {key}") from exc


def expand_plugin_dependencies(selected_plugins: Iterable[str]) -> list[str]:
    """Normalize plugin references and include transitive dependencies first."""
    result: list[str] = []
    seen: set[str] = set()
    visiting: set[str] = set()

    def visit(key: str) -> None:
        if key not in PLUGIN_REGISTRY:
            raise ValueError(f"Unknown plugin key: {key}")
        if key in seen:
            return
        if key in visiting:
            raise ValueError(f"Plugin dependency cycle detected at: {key}")
        visiting.add(key)
        descriptor = PLUGIN_REGISTRY[key]
        for dependency in descriptor.dependencies:
            visit(dependency)
        visiting.remove(key)
        seen.add(key)
        result.append(key)

    for plugin in selected_plugins:
        visit(plugin)
    return result


async def create_plugin(
    key: str,
    config: dict[str, Any] | None = None,
) -> Any:
    """Create a plugin instance from the registry."""
    return await get_plugin_descriptor(key).create(config)


def plugin_catalog() -> list[dict[str, Any]]:
    """Return GUI-facing plugin catalog entries from the registry."""
    return [
        {
            "key": descriptor.key,
            "name": descriptor.name,
            "display_name": descriptor.display_name,
            "description": descriptor.description,
            "dependencies": list(descriptor.dependencies),
            "schema": descriptor.gui_config_schema(),
            "defaults": descriptor.gui_default_config(),
            "permissions": descriptor.permissions,
        }
        for descriptor in iter_plugin_descriptors()
    ]
