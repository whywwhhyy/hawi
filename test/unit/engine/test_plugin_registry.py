from __future__ import annotations

import pytest

from hawi.engine.plugin_registry import (
    KNOWN_PLUGINS,
    PLUGIN_DISPLAY_NAMES,
    create_plugin,
    expand_plugin_dependencies,
    get_plugin_descriptor,
    iter_plugin_descriptors,
    plugin_catalog,
    plugin_tool_preview,
)


def test_registry_catalog_matches_known_plugins() -> None:
    descriptors = list(iter_plugin_descriptors())
    catalog = plugin_catalog()

    assert [item["key"] for item in catalog] == [d.key for d in descriptors]
    assert {item["key"] for item in catalog} == set(KNOWN_PLUGINS)
    assert [item["name"] for item in catalog] == [d.name for d in descriptors]
    assert [item["display_name"] for item in catalog] == [
        d.display_name for d in descriptors
    ]
    assert [item["description"] for item in catalog] == [
        d.description for d in descriptors
    ]
    assert PLUGIN_DISPLAY_NAMES == {d.key: d.display_name for d in descriptors}
    assert all("dependencies" in item for item in catalog)
    assert all("schema" in item and "defaults" in item for item in catalog)


def test_descriptor_metadata_comes_from_plugin_classes() -> None:
    for descriptor in iter_plugin_descriptors():
        plugin_cls = descriptor.load_class()
        assert descriptor.name == plugin_cls.name
        assert descriptor.display_name == plugin_cls.display_name
        assert descriptor.description == plugin_cls.description
        assert descriptor.dependencies == tuple(plugin_cls.dependencies)


def test_descriptor_loads_plugin_schema() -> None:
    descriptor = get_plugin_descriptor("hawi/subagent")

    assert descriptor.display_name == "Subagent"
    assert descriptor.gui_config_schema()["type"] == "object"
    assert descriptor.gui_default_config() == {}


def test_plugin_dependencies_expand_before_dependents() -> None:
    assert expand_plugin_dependencies(["hawi/skills"]) == [
        "hawi/filesystem",
        "hawi/shell",
        "hawi/skills",
    ]


def test_unknown_plugin_key_has_clear_error() -> None:
    with pytest.raises(ValueError, match="Unknown plugin key"):
        get_plugin_descriptor("missing")


@pytest.mark.asyncio
async def test_create_plugin_binds_to_descriptor_factory() -> None:
    plugin = await create_plugin("hawi/subagent", {})

    assert plugin.__class__.__name__ == "SubAgentPlugin"


@pytest.mark.asyncio
async def test_create_filesystem_plugin_passes_seek_style_config() -> None:
    plugin = await create_plugin("hawi/filesystem", {"seek_style": "char"})

    assert plugin.seek_style == "char"


@pytest.mark.asyncio
async def test_plugin_tool_preview_loads_temporary_plugin_tools() -> None:
    preview = await plugin_tool_preview("hawi/web", {})

    tools = {tool["short_name"]: tool for tool in preview["tools"]}
    assert preview["key"] == "hawi/web"
    assert "fetch" in tools
    assert tools["fetch"]["name"] == "WebPlugin__fetch"
    assert "url" in tools["fetch"]["schema"]["properties"]


@pytest.mark.asyncio
async def test_filesystem_tool_preview_reflects_seek_style_config() -> None:
    preview = await plugin_tool_preview("hawi/filesystem", {"seek_style": "char"})

    tools = {tool["short_name"]: tool for tool in preview["tools"]}
    schema = tools["read_file"]["schema"]["properties"]
    assert "offset" in schema
    assert "limit" in schema
    assert "start_line" not in schema
