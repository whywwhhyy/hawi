from __future__ import annotations

from hawi.builtin_plugins.filesystem_plugin import FileSystemPlugin
from hawi.builtin_plugins.mcp_plugin import MCPPlugin
from hawi.builtin_plugins.plan_plugin import PlanPlugin
from hawi.builtin_plugins.python_interpreter import PythonInterpreterPlugin
from hawi.builtin_plugins.shell_plugin import ShellPlugin
from hawi.builtin_plugins.skills_plugin import SkillsPlugin
from hawi.builtin_plugins.taskflow_plugin import TaskflowPlugin
from hawi.builtin_plugins.web import WebPlugin


def test_plugins_expose_gui_schema_and_defaults():
    plugin_classes = [
        FileSystemPlugin,
        ShellPlugin,
        WebPlugin,
        SkillsPlugin,
        PythonInterpreterPlugin,
        MCPPlugin,
        TaskflowPlugin,
        PlanPlugin,
    ]
    for plugin_cls in plugin_classes:
        schema = plugin_cls.gui_config_schema()
        defaults = plugin_cls.gui_default_config()
        assert isinstance(schema, dict)
        assert isinstance(defaults, dict)
        assert schema.get("type") == "object"
        assert "properties" in schema


def test_mcp_schema_requires_config_path():
    schema = MCPPlugin.gui_config_schema()
    assert schema["properties"]["config_path"]["type"] == "string"
    assert "config_path" in schema.get("required", [])


def test_skills_schema_has_skills_dir():
    schema = SkillsPlugin.gui_config_schema()
    defaults = SkillsPlugin.gui_default_config()
    assert schema["properties"]["skills_dir"]["type"] == "string"
    assert defaults["skills_dir"] == ".skills"


def test_filesystem_schema_has_seek_style_dropdown():
    schema = FileSystemPlugin.gui_config_schema()
    defaults = FileSystemPlugin.gui_default_config()
    assert schema["properties"]["seek_style"]["enum"] == ["line", "char"]
    assert defaults["seek_style"] == "line"


def test_plan_schema_has_context_folding_toggle():
    schema = PlanPlugin.gui_config_schema()
    defaults = PlanPlugin.gui_default_config()
    assert schema["properties"]["fold_completed_tasks"]["type"] == "boolean"
    assert defaults["fold_completed_tasks"] is False


def test_taskflow_schema_has_context_folding_toggle():
    schema = TaskflowPlugin.gui_config_schema()
    defaults = TaskflowPlugin.gui_default_config()
    assert schema["properties"]["fold_completed_steps"]["type"] == "boolean"
    assert defaults["fold_completed_steps"] is False
