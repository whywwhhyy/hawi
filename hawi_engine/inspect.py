"""Inspection metadata for external GUI clients."""

from __future__ import annotations

from typing import Any

from hawi.models import model_registry

from .plugin_registry import plugin_catalog
from .protocol import VERSION, to_json_safe
from .runtime import DEFAULT_SYSTEM_PROMPT


def build_inspect_payload() -> dict[str, Any]:
    """Return metadata needed by non-Python GUI clients."""
    return {
        "version": VERSION,
        "models": model_registry.list_models(),
        "plugin_catalog": to_json_safe(plugin_catalog()),
        "default_system_prompt": DEFAULT_SYSTEM_PROMPT,
    }
