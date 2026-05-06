"""Inspection metadata for external GUI clients."""

from __future__ import annotations

from typing import Any

from hawi.models import model_registry

from .protocol import VERSION, to_json_safe
from .runtime import (
    DEFAULT_SYSTEM_PROMPT,
    PLUGIN_FILESYSTEM,
    PLUGIN_MCP,
    PLUGIN_PYTHON_INTERPRETER,
    PLUGIN_SHELL,
    PLUGIN_SKILLS,
    PLUGIN_WEB,
)


def build_inspect_payload() -> dict[str, Any]:
    """Return metadata needed by non-Python GUI clients."""
    plugin_catalog = []
    for key, label, plugin_cls in _plugin_entries():
        plugin_catalog.append(
            {
                "key": key,
                "label": label,
                "schema": to_json_safe(plugin_cls.gui_config_schema()),
                "defaults": to_json_safe(plugin_cls.gui_default_config()),
            }
        )

    return {
        "version": VERSION,
        "models": model_registry.list_models(),
        "plugin_catalog": plugin_catalog,
        "default_system_prompt": DEFAULT_SYSTEM_PROMPT,
    }


def _plugin_entries() -> list[tuple[str, str, type]]:
    from hawi_plugins.filesystem_plugin import FileSystemPlugin
    from hawi_plugins.mcp_plugin import MCPPlugin
    from hawi_plugins.python_interpreter import PythonInterpreterPlugin
    from hawi_plugins.shell_plugin import ShellPlugin
    from hawi_plugins.skills_plugin import SkillsPlugin
    from hawi_plugins.web import WebPlugin

    return [
        (PLUGIN_FILESYSTEM, "FileSystemPlugin", FileSystemPlugin),
        (PLUGIN_SHELL, "ShellPlugin", ShellPlugin),
        (PLUGIN_WEB, "WebPlugin", WebPlugin),
        (PLUGIN_SKILLS, "SkillsPlugin", SkillsPlugin),
        (PLUGIN_PYTHON_INTERPRETER, "PythonInterpreterPlugin", PythonInterpreterPlugin),
        (PLUGIN_MCP, "MCPPlugin", MCPPlugin),
    ]
