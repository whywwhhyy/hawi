"""Shared helpers for append-only visible message history records."""

from __future__ import annotations

import logging
from typing import Any

from hawi.events import Event

logger = logging.getLogger(__name__)


def message_history_entry_from_event(event: Event) -> dict[str, Any] | None:
    """Build a stable message-history record from ``agent.message_added``."""
    if event.type != "agent.message_added":
        return None
    try:
        data = event.model_dump(mode="json")
    except Exception:
        logger.exception("failed to serialize message history event")
        return None

    role = data.get("role")
    if role not in {"user", "assistant", "tool"}:
        return None
    metadata = data.get("metadata")
    if not should_persist_message(metadata):
        return None
    content = data.get("content")
    if not isinstance(content, list) or not content:
        return None

    return {
        "version": 1,
        "timestamp": data.get("timestamp"),
        "run_id": data.get("run_id"),
        "role": role,
        "content": content,
        "metadata": metadata if isinstance(metadata, dict) else None,
    }


def should_persist_message(metadata: Any) -> bool:
    """Return whether a message should be kept in display history."""
    if not isinstance(metadata, dict):
        return True
    if metadata.get("hidden") is True:
        return False
    for key in ("display", "visible", "persist", "persist_session"):
        if metadata.get(key) is False:
            return False
    display_type = metadata.get("display_message_type")
    if isinstance(display_type, str) and display_type in {
        "hidden",
        "internal",
        "none",
    }:
        return False
    return True
