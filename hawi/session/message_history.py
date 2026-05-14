"""Shared helpers for append-only visible message history records."""

from __future__ import annotations

import logging
from typing import Any

from hawi.events import Event

logger = logging.getLogger(__name__)


def message_history_entry_from_event(event: Event) -> dict[str, Any] | None:
    """Build a stable message-history record from a user-visible event."""
    if event.type == "model.retry":
        return _model_retry_entry(event)
    if event.type in {"model.error", "agent.error"}:
        return _error_entry(event)
    if event.type in {"runner.interrupt", "agent.interrupt"}:
        return _interrupt_entry(event)
    if event.type != "agent.message_added":
        return None
    try:
        data = event.model_dump(mode="json")
    except Exception:
        logger.exception("failed to serialize message history event")
        return None

    role = data.get("role")
    if role not in {"user", "assistant", "tool", "system", "error"}:
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


def _model_retry_entry(event: Event) -> dict[str, Any] | None:
    try:
        data = event.model_dump(mode="json")
    except Exception:
        logger.exception("failed to serialize model retry history event")
        return None
    attempt = data.get("attempt", "")
    max_retries = data.get("max_retries", "")
    error_type = data.get("error_type", "")
    error_message = data.get("error_message", "")
    text = f"模型重试 {attempt}/{max_retries}: [{error_type}] {error_message}"
    return {
        "version": 1,
        "timestamp": data.get("timestamp"),
        "run_id": data.get("run_id") or data.get("request_id"),
        "role": "system",
        "content": [{"type": "text", "text": text}],
        "metadata": {
            "display_message_type": "model_retry",
            "request_id": data.get("request_id"),
            "error_type": error_type,
            "attempt": attempt,
            "max_retries": max_retries,
            "persist_session": True,
        },
    }


def _error_entry(event: Event) -> dict[str, Any] | None:
    try:
        data = event.model_dump(mode="json")
    except Exception:
        logger.exception("failed to serialize error history event")
        return None
    raw_error = data.get("error")
    error = raw_error if isinstance(raw_error, dict) else {}
    message = str(error.get("message") or "Unknown error")
    return {
        "version": 1,
        "timestamp": data.get("timestamp"),
        "run_id": data.get("run_id"),
        "role": "error",
        "content": [{"type": "text", "text": message}],
        "metadata": {
            "display_message_type": event.type,
            "code": "model_error" if event.type == "model.error" else "agent_error",
            "error": error,
            "persist_session": True,
        },
    }


def _interrupt_entry(event: Event) -> dict[str, Any] | None:
    try:
        data = event.model_dump(mode="json")
    except Exception:
        logger.exception("failed to serialize interrupt history event")
        return None
    if event.type == "runner.interrupt":
        text = f"执行被中断: {data.get('reason', '')}"
    else:
        text = f"Agent 中断: {data.get('interrupt_type', '')}"
    return {
        "version": 1,
        "timestamp": data.get("timestamp"),
        "run_id": data.get("run_id"),
        "role": "system",
        "content": [{"type": "text", "text": text}],
        "metadata": {
            "display_message_type": event.type,
            "interrupted_tool_calls": data.get("interrupted_tool_calls"),
            "persist_session": True,
        },
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
