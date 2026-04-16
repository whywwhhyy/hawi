from __future__ import annotations

from hawi_plugins.filesystem_plugin import FileSystemPlugin
from hawi_plugins.mcp_plugin import MCPPlugin
from hawi_plugins.python_interpreter import PythonInterpreterPlugin
from hawi_plugins.shell_plugin import ShellPlugin
from hawi_plugins.skills_plugin import SkillsPlugin
from hawi_plugins.web import WebPlugin


def test_plugins_expose_gui_schema_and_defaults():
    plugin_classes = [
        FileSystemPlugin,
        ShellPlugin,
        WebPlugin,
        SkillsPlugin,
        PythonInterpreterPlugin,
        MCPPlugin,
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
