"""Config object and recursive substitution support."""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

ConfigValue = dict[str, Any]
PathLike = str | Path

_SUBSTITUTION_PATTERN = re.compile(r"(?<!\\)\{([^{}]+)\}")
_ESCAPED_OPEN_BRACE = "\\{"


class ConfigLoaderError(ValueError):
    """Base error for config loading, saving, and substitution failures."""


class ConfigSubstitutionError(ConfigLoaderError):
    """Raised when a substitution reference cannot be resolved safely."""


@dataclass
class Config:
    """A loaded config with both raw and substituted values.

    ``raw`` keeps placeholders exactly as loaded so callers can update it at
    runtime and call :meth:`resubstitute`. ``data`` is the resolved tree used by
    runtime code.
    """

    raw: Mapping[str, Any] = field(default_factory=dict)
    data: ConfigValue = field(init=False)

    def __post_init__(self) -> None:
        self.raw = deepcopy(dict(self.raw))
        self.resubstitute()

    @classmethod
    def from_file(cls, path: PathLike) -> "Config":
        """Load a single config file into a :class:`Config` object."""
        from .loader import load_config_file

        return cls(load_config_file(path))

    def resubstitute(self) -> ConfigValue:
        """Recompute ``data`` from the current ``raw`` tree."""
        self.data = substitute_config(self.raw)
        return self.data

    def set_raw(self, path: str | Sequence[str | int], value: Any) -> None:
        """Set one raw value by dotted path, then recompute substitutions."""
        parts = _normalize_path(path)
        if not parts:
            raise ConfigLoaderError("Raw config path cannot be empty")

        current: Any = self.raw
        for part in parts[:-1]:
            if isinstance(current, dict):
                current = current.setdefault(part, {})
            elif isinstance(current, list) and isinstance(part, int):
                current = current[part]
            else:
                raise ConfigLoaderError(f"Cannot set raw config path: {path}")

        last = parts[-1]
        if isinstance(current, dict):
            current[last] = value
        elif isinstance(current, list) and isinstance(last, int):
            current[last] = value
        else:
            raise ConfigLoaderError(f"Cannot set raw config path: {path}")
        self.resubstitute()

    def save(self, path: PathLike, *, substituted: bool = False) -> None:
        """Save ``raw`` by default, or ``data`` when ``substituted`` is true."""
        from .loader import save_config_file

        save_config_file(path, self.data if substituted else self.raw)

    def as_dict(self) -> ConfigValue:
        """Return a deep copy of the substituted config tree."""
        return deepcopy(self.data)

    def raw_dict(self) -> ConfigValue:
        """Return a deep copy of the raw config tree."""
        return deepcopy(dict(self.raw))


def _normalize_path(path: str | Sequence[str | int]) -> list[str | int]:
    if isinstance(path, str):
        return [part for part in path.split(".") if part]
    return list(path)


def substitute_config(config: Mapping[str, Any]) -> ConfigValue:
    """Resolve substitutions across a config mapping."""
    root = deepcopy(dict(config))
    value = _substitute_value(root, root, [])
    if not isinstance(value, dict):
        raise ConfigSubstitutionError("Substituted config root must be a mapping")
    return value


def _substitute_value(value: Any, root: ConfigValue, parents: list[Any]) -> Any:
    if isinstance(value, str):
        return _substitute_string(value, root, parents)
    if isinstance(value, dict):
        return {
            key: _substitute_value(item, root, [*parents, value])
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _substitute_value(item, root, [*parents, value])
            for item in value
        ]
    return deepcopy(value)


def _substitute_string(value: str, root: ConfigValue, parents: list[Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        raw_ref = match.group(1).strip()
        resolved, resolved_parents = _resolve_reference(raw_ref, root, parents)
        if isinstance(resolved, (dict, list)):
            raise ConfigSubstitutionError(
                f"Substitution reference must resolve to a scalar: {raw_ref}"
            )
        substituted = _substitute_value(resolved, root, resolved_parents)
        if isinstance(substituted, (dict, list)):
            raise ConfigSubstitutionError(
                f"Substitution reference must resolve to a scalar: {raw_ref}"
            )
        return _stringify_scalar(substituted)

    return _SUBSTITUTION_PATTERN.sub(replace, value).replace(
        _ESCAPED_OPEN_BRACE,
        "{",
    )


def _resolve_reference(
    raw_ref: str,
    root: ConfigValue,
    parents: list[Any],
) -> tuple[Any, list[Any]]:
    if not raw_ref:
        raise ConfigSubstitutionError("Empty substitution reference")

    if raw_ref.startswith("."):
        dot_count = len(raw_ref) - len(raw_ref.lstrip("."))
        key_path = raw_ref[dot_count:]
        if not key_path:
            raise ConfigSubstitutionError(
                f"Relative substitution requires a key path: {raw_ref}"
            )
        if not parents:
            raise ConfigSubstitutionError(
                f"Relative substitution has no parent scope: {raw_ref}"
            )
        up_count = dot_count - 1
        if up_count >= len(parents):
            raise ConfigSubstitutionError(
                f"Relative substitution escapes config root: {raw_ref}"
            )
        base_index = len(parents) - 1 - up_count
        base = parents[base_index]
        base_parents = parents[:base_index]
        return _resolve_path(base, key_path, base_parents, raw_ref)

    return _resolve_path(root, raw_ref, [], raw_ref)


def _resolve_path(
    base: Any,
    path: str,
    parents: list[Any],
    raw_ref: str,
) -> tuple[Any, list[Any]]:
    current = base
    current_parents = list(parents)
    for part in path.split("."):
        if not part:
            raise ConfigSubstitutionError(
                f"Invalid substitution reference: {raw_ref}"
            )
        parent = current
        if isinstance(current, Mapping):
            if part not in current:
                raise ConfigSubstitutionError(
                    f"Unknown substitution reference: {raw_ref}"
                )
            current = current[part]
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                raise ConfigSubstitutionError(
                    f"Unknown substitution reference: {raw_ref}"
                ) from None
        else:
            raise ConfigSubstitutionError(
                f"Cannot traverse scalar substitution reference: {raw_ref}"
            )
        current_parents.append(parent)
    return current, current_parents


def _stringify_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
