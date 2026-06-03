"""Round-trip persistence helpers for model provider configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap


class ModelProviderConfigPersistenceError(ValueError):
    """Raised when a provider config cannot be written back safely."""


def persist_provider_properties(
    provider: str,
    properties: Mapping[str, Any],
    config_paths: Iterable[str | Path],
) -> Path:
    """Persist provider property updates to the first config containing provider.

    Existing provider and property key order is preserved by updating the
    round-tripped YAML node in place. New property keys are appended to the
    existing ``properties`` mapping.
    """
    provider_name = provider.strip()
    if not provider_name:
        raise ModelProviderConfigPersistenceError("provider is required")
    if not isinstance(properties, Mapping):
        raise ModelProviderConfigPersistenceError("properties must be a mapping")

    paths = [Path(path).expanduser() for path in config_paths]
    for path in paths:
        if not path.exists():
            continue
        yaml = _round_trip_yaml()
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.load(fh) or CommentedMap()
        node = _find_provider_node(data, provider_name)
        if node is None:
            continue
        _update_provider_properties(node, properties)
        with path.open("w", encoding="utf-8") as fh:
            yaml.dump(data, fh)
        return path

    searched = ", ".join(str(path) for path in paths) or "no config paths"
    raise ModelProviderConfigPersistenceError(
        f"Provider '{provider_name}' was not found in loaded model configs: {searched}"
    )


def _round_trip_yaml() -> YAML:
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.default_flow_style = False
    return yaml


def _find_provider_node(data: Any, provider: str) -> CommentedMap | None:
    if not isinstance(data, Mapping):
        raise ModelProviderConfigPersistenceError("models config root must be a mapping")
    providers = data.get("providers")
    if providers is None:
        return None
    if not isinstance(providers, list):
        raise ModelProviderConfigPersistenceError("models config providers must be a list")
    for item in providers:
        if isinstance(item, Mapping) and str(item.get("name", "")).strip() == provider:
            if not isinstance(item, CommentedMap):
                return CommentedMap(item)
            return item
    return None


def _update_provider_properties(
    provider_node: CommentedMap,
    properties: Mapping[str, Any],
) -> None:
    current = provider_node.get("properties")
    if current is None:
        current = CommentedMap()
        provider_node["properties"] = current
    if not isinstance(current, Mapping):
        raise ModelProviderConfigPersistenceError("provider properties must be a mapping")
    if not isinstance(current, CommentedMap):
        converted = CommentedMap(current)
        provider_node["properties"] = converted
        current = converted
    for key, value in properties.items():
        current[str(key)] = _round_trip_value(value)


def _round_trip_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        converted = CommentedMap()
        for key, item in value.items():
            converted[str(key)] = _round_trip_value(item)
        return converted
    if isinstance(value, list):
        return [_round_trip_value(item) for item in value]
    return value
