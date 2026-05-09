"""Stable JSON protocol for the Hawi core process."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

VERSION = "hawi.core.v1"

COMMAND_TYPES = {
    "hello",
    "enqueue",
    "interrupt",
    "clear_context",
    "clear_queue",
    "set_system_prompt",
    "switch_model",
    "apply_plugins",
    "get_status",
    "shutdown",
    "ping",
    "session_list",
    "session_new",
    "session_load",
    "session_switch",
    "session_delete",
    "session_save_now",
}

EVENT_TYPES = {
    "core.ready",
    "core.status",
    "ack",
    "error",
    "pong",
    "run.start",
    "run.text_delta",
    "run.thinking_delta",
    "run.stop",
    "tool.call_start",
    "tool.call_delta",
    "tool.call_stop",
    "tool.result",
    "model.metadata",
    "model.retry",
    "agent.interrupt",
    "scheduler.interrupt",
    "debug.info",
    "plugin.event",
    "plugin.message",
    "plugin.status",
    "plugin.tool_progress",
    "plugin.artifact.upsert",
    "plugin.artifact.delta",
    "plugin.artifact.remove",
    "plugin.artifact.clear",
}


class ProtocolError(ValueError):
    """Raised when an incoming JSON frame does not match the v1 protocol."""

    def __init__(self, message: str, code: str = "bad_request") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class CoreCommand:
    """Parsed command envelope."""

    type: str
    payload: dict[str, Any]
    id: str | None = None
    version: str = VERSION


def parse_frame(raw: str | bytes) -> CoreCommand:
    """Parse a JSON command frame.

    The input frame is one JSON object. For NDJSON transports, callers pass one
    decoded line; for WebSocket transports, callers pass one text message.
    """
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolError("Frame must be valid UTF-8") from exc

    text = raw.strip()
    if not text:
        raise ProtocolError("Frame is empty")

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Frame is not valid JSON: {exc.msg}") from exc

    if not isinstance(data, dict):
        raise ProtocolError("Frame must be a JSON object")

    version = data.get("version", VERSION)
    if version != VERSION:
        raise ProtocolError(f"Unsupported protocol version: {version}", "unsupported_version")

    command_type = data.get("type")
    if not isinstance(command_type, str) or not command_type:
        raise ProtocolError("Frame field 'type' must be a non-empty string")
    if command_type not in COMMAND_TYPES:
        raise ProtocolError(f"Unknown command type: {command_type}", "unknown_command")

    payload = data.get("payload", {})
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ProtocolError("Frame field 'payload' must be an object")

    request_id = data.get("id")
    if request_id is not None and not isinstance(request_id, str):
        raise ProtocolError("Frame field 'id' must be a string when present")

    return CoreCommand(
        type=command_type,
        payload=payload,
        id=request_id,
        version=version,
    )


def make_frame(
    event_type: str,
    payload: dict[str, Any] | None = None,
    *,
    request_id: str | None = None,
    timestamp: float | None = None,
) -> dict[str, Any]:
    """Create a JSON-serializable protocol envelope."""
    if event_type not in EVENT_TYPES:
        raise ValueError(f"Unknown event type: {event_type}")
    return {
        "version": VERSION,
        "type": event_type,
        "id": request_id,
        "ts": time.time() if timestamp is None else timestamp,
        "payload": to_json_safe(payload or {}),
    }


def make_ack(
    command: str,
    *,
    request_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ack_payload = {"command": command, "ok": True}
    if payload:
        ack_payload.update(payload)
    return make_frame("ack", ack_payload, request_id=request_id)


def make_error(
    message: str,
    *,
    request_id: str | None = None,
    code: str = "error",
    details: Any | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": False,
        "code": code,
        "message": message,
    }
    if details is not None:
        payload["details"] = to_json_safe(details)
    return make_frame("error", payload, request_id=request_id)


def json_dumps(frame: dict[str, Any]) -> str:
    """Serialize one protocol frame without ASCII escaping."""
    return json.dumps(to_json_safe(frame), ensure_ascii=False, separators=(",", ":"))


def to_json_safe(value: Any) -> Any:
    """Convert common Python objects into JSON-safe data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_safe(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return to_json_safe(value.model_dump(mode="json"))
        except TypeError:
            return to_json_safe(value.model_dump())
    if hasattr(value, "__dict__"):
        return to_json_safe(vars(value))
    return str(value)
