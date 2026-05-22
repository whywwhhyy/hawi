"""Stable JSON protocol for the Hawi core process.

Wire envelope (unchanged across plans):
    {"version": "hawi.core.v1", "type": <str>, "id": <str|null>, "payload": <obj>}

Hello payload (added in Plan 2 — capability negotiation, additive only):
    {
        "token": <optional str, when --token is set>,
        "client_caps": <optional list[str]>  # capabilities the client wishes to use
    }

Hello ack payload (added in Plan 2):
    {
        "command": "hello",
        "ok": true,
        "authenticated": true,
        "server_caps": <list[str]>,   # full set the server speaks (sorted)
        "negotiated": <list[str]>     # client_caps & server_caps (sorted)
    }

Old clients that omit `client_caps` get an empty `negotiated` set; old servers
that don't return `server_caps`/`negotiated` are simply ignored by future-aware
clients. SERVER_CAPS lives in `hawi.engine.runtime` and is empty in Plan 2;
Plans 3-5 add concrete capability strings (`tlv_v1`, `binary_frames`, etc.).
"""

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
    "stop",
    "resume",
    "clear_context",
    "compact_context",
    "set_auto_compact",
    "clear_queue",
    "set_system_prompt",
    "switch_model",
    "refresh_models",
    "apply_plugins",
    "plugin_action",
    "get_status",
    "shutdown",
    "ping",
    "session_list",
    "session_new",
    "session_fork",
    "session_rewind",
    "session_load",
    "session_switch",
    "session_delete",
    "session_rename",
    "session_save_now",
    "session_history",
    "session_export_markdown",
    "queue_task_add",
    "queue_task_update",
    "queue_task_remove",
    "queue_task_reorder",
    "blob.upload_init",
    "blob.upload_chunk",
    "blob.upload_finalize",
    "blob.has",
    "blob.fetch",
    "blob.release",
    "blob.request_retransmit",
}

EVENT_TYPES = {
    "core.ready",
    "core.status",
    "ack",
    "error",
    "pong",
    "run.start",
    "run.message_committed",
    "run.text_delta",
    "run.thinking_delta",
    "run.stop",
    "tool.call_start",
    "tool.call_delta",
    "tool.call_stop",
    "tool.result",
    "model.metadata",
    "model.retry",
    "agent.system_prompt",
    "agent.context_injected",
    "agent.tool_runtime_context_injected",
    "agent.compact_start",
    "agent.compact_stop",
    "agent.interrupt",
    "runner.interrupt",
    "runner.paused",
    "runner.resumed",
    "debug.info",
    "plugin.event",
    "plugin.message",
    "plugin.status",
    "plugin.tool_progress",
    "plugin.artifact.upsert",
    "plugin.artifact.delta",
    "plugin.artifact.remove",
    "plugin.artifact.clear",
    "subagent.created",
    "subagent.event",
    "subagent.closed",
    "blob.chunk",
    "blob.complete",
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
