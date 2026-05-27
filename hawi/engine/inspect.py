"""Inspection metadata for external GUI clients."""

from __future__ import annotations

from typing import Any

from hawi.models import model_registry

from .plugin_registry import plugin_catalog, plugin_tool_preview
from .protocol import VERSION, to_json_safe
from .runtime import DEFAULT_SYSTEM_PROMPT


def build_inspect_payload() -> dict[str, Any]:
    """Return metadata needed by non-Python GUI clients."""
    return {
        "version": VERSION,
        "models": model_registry.list_models(),
        "model_provider_configs": to_json_safe(_model_provider_config_previews()),
        "plugin_catalog": to_json_safe(plugin_catalog()),
        "default_system_prompt": DEFAULT_SYSTEM_PROMPT,
    }


async def build_plugin_tool_preview_payload(
    plugin_key: str,
    plugin_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return tool metadata for one plugin after temporary initialization."""
    preview = await plugin_tool_preview(plugin_key, plugin_config)
    return {
        "version": VERSION,
        "plugin_key": plugin_key,
        "plugin_name": preview["name"],
        "display_name": preview["display_name"],
        "description": preview["description"],
        "tools": preview["tools"],
    }


def _model_provider_config_previews() -> dict[str, dict[str, Any]]:
    previews: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for provider_name in model_registry.list_providers():
        if provider_name in seen:
            continue
        seen.add(provider_name)
        provider_configs = model_registry.get_provider(provider_name) or []
        previews[provider_name] = _summarize_provider_configs(provider_configs)
    return previews


def _summarize_provider_configs(provider_configs: list[Any]) -> dict[str, Any]:
    adapters: list[str] = []
    model_count = 0
    property_values: dict[str, list[Any]] = {}

    for provider in provider_configs:
        if provider.adapter not in adapters:
            adapters.append(provider.adapter)
        model_count += len(provider.model_ids)
        for key, value in provider.properties.items():
            preview_value = _preview_config_value(key, value)
            values = property_values.setdefault(key, [])
            if not any(existing == preview_value for existing in values):
                values.append(preview_value)

    return {
        "adapter": ", ".join(adapters),
        "model_count": model_count,
        "properties": {
            key: _summarize_config_values(values)
            for key, values in property_values.items()
        },
    }


def _summarize_config_values(values: list[Any]) -> Any:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return " | ".join(str(value) for value in values)


def _preview_config_value(key: str, value: Any) -> Any:
    if not _is_sensitive_config_key(key):
        return value
    if value is None:
        return None
    text = str(value)
    if not text:
        return ""
    if len(text) <= 8:
        return "****"
    return f"{text[:3]}...{text[-4:]}"


def _is_sensitive_config_key(key: str) -> bool:
    normalized = key.lower()
    return any(
        marker in normalized
        for marker in ("key", "token", "secret", "password", "authorization", "credential")
    )
