"""Single source of truth for engine-loadable Hawi plugins."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from importlib import import_module
from typing import Any

PLUGIN_FILESYSTEM = "filesystem"
PLUGIN_SHELL = "shell"
PLUGIN_WEB = "web"
PLUGIN_SKILLS = "skills"
PLUGIN_PYTHON_INTERPRETER = "python_interpreter"
PLUGIN_MCP = "mcp"
PLUGIN_PLAN = "plan"
PLUGIN_WORKFLOW = "workflow"
PLUGIN_ENVIRON_PROMPT = "environ_prompt"
PLUGIN_SUBAGENT = "subagent"

PluginFactory = Callable[[dict[str, Any]], Any | Awaitable[Any]]


@dataclass(frozen=True)
class PluginDescriptor:
    """Descriptor for a plugin the engine can advertise and instantiate."""

    key: str
    label: str
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


def _create_skills_plugin(config: dict[str, Any]) -> Any:
    from hawi_plugins.skills_plugin import SkillsPlugin

    skills_dir = str(config.get("skills_dir") or ".skills")
    return SkillsPlugin(skills_dir=skills_dir)


def _create_python_interpreter_plugin(config: dict[str, Any]) -> Any:
    from hawi_plugins.python_interpreter import PythonInterpreterPlugin

    work_dir_raw = config.get("work_dir")
    work_dir = str(work_dir_raw).strip() if isinstance(work_dir_raw, str) else None
    return PythonInterpreterPlugin(
        work_dir=work_dir or None,
        print_execution=bool(config.get("print_execution", False)),
    )


async def _create_mcp_plugin(config: dict[str, Any]) -> Any:
    from hawi_plugins.mcp_plugin import MCPPlugin

    config_path = str(config.get("config_path") or "").strip()
    if not config_path:
        raise ValueError("MCP plugin requires 'config_path'.")
    plugin = MCPPlugin(config_path=config_path)
    await plugin.connect()
    return plugin


def _create_plan_plugin(config: dict[str, Any]) -> Any:
    from hawi_plugins.plan_plugin import PlanPlugin

    return PlanPlugin(
        fold_completed_tasks=bool(config.get("fold_completed_tasks", False))
    )


def _create_environ_prompt_plugin(config: dict[str, Any]) -> Any:
    from hawi_plugins.environ_prompt_plugin import EnvironPromptPlugin

    config_path = str(config.get("config_path") or "").strip() or None
    return EnvironPromptPlugin(config_path=config_path)


PLUGIN_REGISTRY: dict[str, PluginDescriptor] = {
    # EnvironPromptPlugin first because it injects session/env context into
    # the prompt before normal tool plugins run.
    PLUGIN_ENVIRON_PROMPT: PluginDescriptor(
        key=PLUGIN_ENVIRON_PROMPT,
        label="EnvironPromptPlugin",
        import_path="hawi_plugins.environ_prompt_plugin:EnvironPromptPlugin",
        factory=_create_environ_prompt_plugin,
    ),
    PLUGIN_FILESYSTEM: PluginDescriptor(
        key=PLUGIN_FILESYSTEM,
        label="FileSystemPlugin",
        import_path="hawi_plugins.filesystem_plugin:FileSystemPlugin",
    ),
    PLUGIN_SHELL: PluginDescriptor(
        key=PLUGIN_SHELL,
        label="ShellPlugin",
        import_path="hawi_plugins.shell_plugin:ShellPlugin",
    ),
    PLUGIN_WEB: PluginDescriptor(
        key=PLUGIN_WEB,
        label="WebPlugin",
        import_path="hawi_plugins.web:WebPlugin",
    ),
    PLUGIN_SKILLS: PluginDescriptor(
        key=PLUGIN_SKILLS,
        label="SkillsPlugin",
        import_path="hawi_plugins.skills_plugin:SkillsPlugin",
        factory=_create_skills_plugin,
    ),
    PLUGIN_PYTHON_INTERPRETER: PluginDescriptor(
        key=PLUGIN_PYTHON_INTERPRETER,
        label="PythonInterpreterPlugin",
        import_path="hawi_plugins.python_interpreter:PythonInterpreterPlugin",
        factory=_create_python_interpreter_plugin,
    ),
    PLUGIN_MCP: PluginDescriptor(
        key=PLUGIN_MCP,
        label="MCPPlugin",
        import_path="hawi_plugins.mcp_plugin:MCPPlugin",
        factory=_create_mcp_plugin,
    ),
    PLUGIN_PLAN: PluginDescriptor(
        key=PLUGIN_PLAN,
        label="PlanPlugin",
        import_path="hawi_plugins.plan_plugin:PlanPlugin",
        factory=_create_plan_plugin,
    ),
    PLUGIN_WORKFLOW: PluginDescriptor(
        key=PLUGIN_WORKFLOW,
        label="WorkflowPlugin",
        import_path="hawi_plugins.workflow_plugin:WorkflowPlugin",
    ),
    PLUGIN_SUBAGENT: PluginDescriptor(
        key=PLUGIN_SUBAGENT,
        label="SubAgentPlugin",
        import_path="hawi_plugins.subagent_plugin:SubAgentPlugin",
    ),
}

KNOWN_PLUGINS: frozenset[str] = frozenset(PLUGIN_REGISTRY)
PLUGIN_LABELS: dict[str, str] = {
    key: descriptor.label for key, descriptor in PLUGIN_REGISTRY.items()
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
            "label": descriptor.label,
            "schema": descriptor.gui_config_schema(),
            "defaults": descriptor.gui_default_config(),
        }
        for descriptor in iter_plugin_descriptors()
    ]
