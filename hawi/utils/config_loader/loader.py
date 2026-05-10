"""File parsing, merging, and directory-chain config loading."""

from __future__ import annotations

import json
import re
import tomllib
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .config import Config, ConfigLoaderError, ConfigValue, PathLike

_SUPPORTED_SUFFIXES = {".json", ".yaml", ".yml", ".toml"}
_BARE_TOML_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


class ConfigLoader:
    """Load and merge structured config files.

    ``directories`` passed to :meth:`load_from_directory_chain` are ordered by
    precedence: files found in earlier directories override files found in
    later directories. Existing files are loaded in reverse discovery order so
    raw values are merged from lowest to highest precedence before
    substitution runs.
    """

    def __init__(self, filenames: str | Sequence[str]) -> None:
        self.filenames = _normalize_filenames(filenames)

    def find_files(self, directories: Iterable[PathLike]) -> list[Path]:
        """Return existing config files in precedence order."""
        found: list[Path] = []
        for directory in directories:
            base = Path(directory).expanduser()
            if not base.is_dir():
                continue
            for filename in self.filenames:
                path = base / filename
                if path.is_file():
                    found.append(path)
        return found

    def load_from_directory_chain(
        self,
        directories: Iterable[PathLike],
        *,
        defaults: Mapping[str, Any] | None = None,
    ) -> Config:
        """Load and reverse-merge config values into a :class:`Config`."""
        merged: ConfigValue = deepcopy(dict(defaults or {}))
        for path in reversed(self.find_files(directories)):
            merged = deep_merge(merged, load_config_file(path))
        return Config(merged)


def load_config_from_directory_chain(
    directories: Iterable[PathLike],
    filenames: str | Sequence[str],
    *,
    defaults: Mapping[str, Any] | None = None,
) -> Config:
    """Convenience wrapper around :class:`ConfigLoader`."""
    return ConfigLoader(filenames).load_from_directory_chain(
        directories,
        defaults=defaults,
    )


def load_config_file(path: PathLike) -> ConfigValue:
    """Load one JSON, YAML, or TOML mapping file based on suffix."""
    resolved = Path(path).expanduser()
    suffix = resolved.suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        raise ConfigLoaderError(
            f"Unsupported config file suffix for {resolved}: {suffix or '<none>'}"
        )

    try:
        if suffix == ".json":
            with resolved.open("r", encoding="utf-8") as fh:
                data = json.load(fh) or {}
        elif suffix == ".toml":
            with resolved.open("rb") as fh:
                data = tomllib.load(fh) or {}
        else:
            data = _load_yaml_file(resolved)
    except ConfigLoaderError:
        raise
    except Exception as exc:
        raise ConfigLoaderError(f"Failed to load config file {resolved}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigLoaderError(f"Config root must be a mapping: {resolved}")
    return dict(data)


def save_config_file(path: PathLike, config: Mapping[str, Any]) -> None:
    """Save a config mapping as JSON, YAML, or TOML based on suffix."""
    resolved = Path(path).expanduser()
    suffix = resolved.suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        raise ConfigLoaderError(
            f"Unsupported config file suffix for {resolved}: {suffix or '<none>'}"
        )

    resolved.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".json":
        with resolved.open("w", encoding="utf-8") as fh:
            json.dump(dict(config), fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        return
    if suffix == ".toml":
        resolved.write_text(_dump_toml(config), encoding="utf-8")
        return
    _save_yaml_file(resolved, config)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> ConfigValue:
    """Return ``base`` recursively merged with ``override``.

    Only mappings are merged recursively. Lists and scalar values are replaced
    wholesale by the higher-precedence value.
    """
    merged: ConfigValue = deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _normalize_filenames(filenames: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(filenames, str):
        names = [filenames]
    else:
        names = list(filenames)
    normalized = tuple(name for name in names if isinstance(name, str) and name)
    if not normalized:
        raise ConfigLoaderError("At least one config filename is required")
    return normalized


def _load_yaml_file(path: Path) -> Any:
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ConfigLoaderError(
            "PyYAML is required to load YAML config files."
        ) from exc
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _save_yaml_file(path: Path, config: Mapping[str, Any]) -> None:
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ConfigLoaderError(
            "PyYAML is required to save YAML config files."
        ) from exc
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(
            dict(config),
            fh,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )


def _dump_toml(config: Mapping[str, Any]) -> str:
    lines: list[str] = []
    _append_toml_table(lines, dict(config), ())
    return "\n".join(lines).rstrip() + "\n"


def _append_toml_table(
    lines: list[str],
    table: Mapping[str, Any],
    prefix: tuple[str, ...],
) -> None:
    scalar_items: list[tuple[str, Any]] = []
    nested_items: list[tuple[str, Mapping[str, Any]]] = []
    for key, value in table.items():
        if isinstance(value, Mapping):
            nested_items.append((str(key), value))
        else:
            scalar_items.append((str(key), value))

    if prefix:
        lines.append(f"[{'.'.join(_toml_key(part) for part in prefix)}]")
    for key, value in scalar_items:
        lines.append(f"{_toml_key(key)} = {_toml_value(value)}")
    if prefix and scalar_items:
        lines.append("")

    for key, value in nested_items:
        _append_toml_table(lines, value, (*prefix, key))


def _toml_key(value: str) -> str:
    if _BARE_TOML_KEY.match(value):
        return value
    return json.dumps(value, ensure_ascii=False)


def _toml_value(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if value is None:
        raise ConfigLoaderError("TOML config files cannot represent null values")
    raise ConfigLoaderError(
        f"Unsupported TOML value type: {type(value).__name__}"
    )
