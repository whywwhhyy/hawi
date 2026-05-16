"""Shared helpers for append-only visible message history records."""

from __future__ import annotations

import logging
from typing import Any

from hawi.events import Event

logger = logging.getLogger(__name__)

REPLAYABLE_AGENT_EVENT_TYPES = {
    "agent.system_prompt",
    "agent.context_injected",
    "agent.tool_parameter_injected",
    "agent.tool_runtime_context_injected",
}

REPLAYABLE_PLUGIN_EVENT_TYPES = {
    "plugin.event",
    "plugin.message",
    "plugin.status",
    "plugin.tool_progress",
    "plugin.artifact.upsert",
    "plugin.artifact.delta",
    "plugin.artifact.remove",
    "plugin.artifact.clear",
}


def message_history_entry_from_event(event: Event) -> dict[str, Any] | None:
    """Build a stable message-history record from a user-visible event."""
    if event.type in REPLAYABLE_AGENT_EVENT_TYPES:
        return _replayable_agent_event_entry(event)
    if event.type in REPLAYABLE_PLUGIN_EVENT_TYPES:
        return _replayable_plugin_event_entry(event)
    if event.type == "model.retry":
        return _model_retry_entry(event)
    if event.type in {"model.error", "agent.error"}:
        return _error_entry(event)
    if event.type in {"runner.interrupt", "agent.interrupt"}:
        return _interrupt_entry(event)
    if event.type in {"agent.compact_start", "agent.compact_stop"}:
        return _context_compaction_entry(event)
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


def _context_compaction_entry(event: Event) -> dict[str, Any] | None:
    try:
        data = event.model_dump(mode="json")
    except Exception:
        logger.exception("failed to serialize context compaction history event")
        return None

    metadata = {
        "display_message_type": "context_compaction",
        "event_type": event.type,
        "mode": data.get("mode"),
        "status": data.get("status"),
        "tokens_before": data.get("tokens_before"),
        "tokens_after": data.get("tokens_after"),
        "message_count_before": data.get("message_count_before"),
        "message_count_after": data.get("message_count_after"),
        "replaced_message_count": data.get("replaced_message_count"),
        "kept_message_count": data.get("kept_message_count"),
        "error": data.get("error"),
        "persist_session": True,
    }
    text = (
        "Compressing context..."
        if event.type == "agent.compact_start"
        else _context_compaction_stop_text(data)
    )
    return {
        "version": 1,
        "timestamp": data.get("timestamp"),
        "run_id": data.get("run_id"),
        "role": "event",
        "content": [{"type": "text", "text": text}],
        "metadata": metadata,
    }


def _replayable_agent_event_entry(event: Event) -> dict[str, Any] | None:
    try:
        data = event.model_dump(mode="json")
    except Exception:
        logger.exception("failed to serialize replayable agent history event")
        return None
    payload = _agent_event_payload(event.type, data)
    return _replayable_event_entry(
        event.type,
        timestamp=data.get("timestamp"),
        run_id=data.get("run_id"),
        content=_event_content(payload),
        payload=payload,
    )


def _replayable_plugin_event_entry(event: Event) -> dict[str, Any] | None:
    try:
        data = event.model_dump(mode="json")
    except Exception:
        logger.exception("failed to serialize replayable plugin history event")
        return None
    payload = _plugin_event_payload(data)
    return _replayable_event_entry(
        event.type,
        timestamp=data.get("timestamp"),
        run_id=payload.get("run_id"),
        content=_event_content(payload),
        payload=payload,
    )


def _replayable_event_entry(
    event_type: str,
    *,
    timestamp: Any,
    run_id: Any,
    content: list[dict[str, Any]],
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "version": 1,
        "timestamp": timestamp,
        "run_id": run_id,
        "role": "event",
        "content": content,
        "metadata": {
            "display_message_type": "core_event",
            "event_type": event_type,
            "event_payload": payload,
            "persist_session": True,
            "replay": True,
        },
    }


def _agent_event_payload(event_type: str, data: dict[str, Any]) -> dict[str, Any]:
    if event_type == "agent.system_prompt":
        return {
            "run_id": data.get("run_id"),
            "content": data.get("content"),
            "text": _content_text(data.get("content")),
            "origin": data.get("origin"),
            "plugin_id": data.get("plugin_id"),
            "plugin_name": data.get("plugin_name"),
            "plugin_role": data.get("plugin_role"),
            "injection_name": data.get("injection_name"),
            "metadata": data.get("metadata"),
        }
    if event_type == "agent.context_injected":
        return {
            "run_id": data.get("run_id"),
            "role": data.get("role"),
            "content": data.get("content"),
            "text": _content_text(data.get("content")),
            "hook_type": data.get("hook_type"),
            "position": data.get("position"),
            "plugin_id": data.get("plugin_id"),
            "plugin_name": data.get("plugin_name"),
            "plugin_role": data.get("plugin_role"),
            "injection_name": data.get("injection_name"),
            "metadata": data.get("metadata"),
            "merge_target": data.get("merge_target"),
            "merge_position": data.get("merge_position"),
            "target_message_id": data.get("target_message_id"),
            "target_message_index": data.get("target_message_index"),
        }
    if event_type == "agent.tool_parameter_injected":
        return {
            "run_id": data.get("run_id"),
            "tool_name": data.get("tool_name"),
            "tool_call_id": data.get("tool_call_id"),
            "parameters": data.get("parameters") or {},
            "plugin_id": data.get("plugin_id"),
            "plugin_name": data.get("plugin_name"),
            "plugin_role": data.get("plugin_role"),
            "injection_name": data.get("injection_name"),
        }
    return {
        "run_id": data.get("run_id"),
        "tool_name": data.get("tool_name"),
        "tool_call_id": data.get("tool_call_id"),
        "parameter_name": data.get("parameter_name"),
        "plugin_id": data.get("plugin_id"),
        "plugin_name": data.get("plugin_name"),
        "plugin_role": data.get("plugin_role"),
        "injection_name": data.get("injection_name"),
    }


def _plugin_event_payload(data: dict[str, Any]) -> dict[str, Any]:
    raw_payload = data.get("payload")
    payload = dict(raw_payload) if isinstance(raw_payload, dict) else {}
    plugin_name = str(data.get("plugin_name") or "")
    plugin_id = str(data.get("plugin_id") or plugin_name)
    payload.update(
        {
            "plugin_id": plugin_id,
            "plugin_name": plugin_name,
            "run_id": data.get("run_id") or "",
            "tool_call_id": data.get("tool_call_id") or "",
        }
    )
    message_id = data.get("message_id")
    if message_id:
        payload["message_id"] = message_id
    return payload


def _event_content(payload: dict[str, Any]) -> list[dict[str, Any]]:
    content = payload.get("content")
    if isinstance(content, list) and content:
        return content
    text = (
        payload.get("text")
        or payload.get("message")
        or payload.get("title")
        or payload.get("event_name")
        or payload.get("status")
        or payload.get("tool_name")
        or "event"
    )
    return [{"type": "text", "text": str(text)}]


def _content_text(content: Any) -> str:
    if not isinstance(content, list):
        return ""
    chunks: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            text = part.get("text")
            if text:
                chunks.append(str(text))
    return "\n\n".join(chunks)


def _context_compaction_stop_text(data: dict[str, Any]) -> str:
    status = data.get("status")
    if status == "success":
        return "Context compacted"
    if status == "skipped":
        return "Context compaction skipped"
    return "Context compaction failed"


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
