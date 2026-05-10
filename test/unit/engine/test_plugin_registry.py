from __future__ import annotations

import pytest

from hawi_engine.plugin_registry import (
    KNOWN_PLUGINS,
    PLUGIN_LABELS,
    create_plugin,
    get_plugin_descriptor,
    iter_plugin_descriptors,
    plugin_catalog,
)


def test_registry_catalog_matches_known_plugins() -> None:
    descriptors = list(iter_plugin_descriptors())
    catalog = plugin_catalog()

    assert [item["key"] for item in catalog] == [d.key for d in descriptors]
    assert {item["key"] for item in catalog} == set(KNOWN_PLUGINS)
    assert PLUGIN_LABELS == {d.key: d.label for d in descriptors}
    assert all("schema" in item and "defaults" in item for item in catalog)


def test_descriptor_loads_plugin_schema() -> None:
    descriptor = get_plugin_descriptor("subagent")

    assert descriptor.label == "SubAgentPlugin"
    assert descriptor.gui_config_schema()["type"] == "object"
    assert descriptor.gui_default_config() == {}


def test_unknown_plugin_key_has_clear_error() -> None:
    with pytest.raises(ValueError, match="Unknown plugin key"):
        get_plugin_descriptor("missing")


@pytest.mark.asyncio
async def test_create_plugin_binds_to_descriptor_factory() -> None:
    plugin = await create_plugin("subagent", {})

    assert plugin.__class__.__name__ == "SubAgentPlugin"
