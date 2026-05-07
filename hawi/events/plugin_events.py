"""Plugin-originated events for UI/runtime integrations.

Plugin events are intentionally a thin, structured pass-through channel. Core
agent logic should not interpret the payload; clients such as the GUI can render
messages, statuses, and artifacts from the stable ``plugin.*`` event types.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from .event import Event, PluginEventType


PLUGIN_EVENT_TYPES: set[str] = {
    "plugin.event",
    "plugin.message",
    "plugin.status",
    "plugin.tool_progress",
    "plugin.artifact.upsert",
    "plugin.artifact.delta",
    "plugin.artifact.remove",
    "plugin.artifact.clear",
}


class PluginEvent(Event):
    """Generic event emitted by a Hawi plugin.

    The event envelope carries plugin identity and current execution context.
    ``payload`` remains plugin-defined and JSON-safe so transports can forward
    it without knowing the plugin-specific schema.
    """

    type: PluginEventType
    source: Literal["plugin"]
    plugin_name: str
    plugin_id: str | None = None
    run_id: str | None = None
    tool_call_id: str | None = None
    message_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create(
        cls,
        event_type: PluginEventType,
        *,
        plugin_name: str,
        plugin_id: str | None = None,
        payload: dict[str, Any] | None = None,
        run_id: str | None = None,
        tool_call_id: str | None = None,
        message_id: str | None = None,
    ) -> "PluginEvent":
        return cls(
            type=event_type,
            source="plugin",
            plugin_name=plugin_name,
            plugin_id=plugin_id,
            run_id=run_id,
            tool_call_id=tool_call_id,
            message_id=message_id,
            payload=_json_safe_dict(payload or {}),
        )

    @classmethod
    def message(
        cls,
        *,
        plugin_name: str,
        message: str,
        plugin_id: str | None = None,
        level: Literal["debug", "info", "warning", "error"] = "info",
        title: str | None = None,
        data: Any | None = None,
        run_id: str | None = None,
        tool_call_id: str | None = None,
        message_id: str | None = None,
    ) -> "PluginEvent":
        payload: dict[str, Any] = {"level": level, "message": message}
        if title:
            payload["title"] = title
        if data is not None:
            payload["data"] = data
        return cls.create(
            "plugin.message",
            plugin_name=plugin_name,
            plugin_id=plugin_id,
            payload=payload,
            run_id=run_id,
            tool_call_id=tool_call_id,
            message_id=message_id,
        )


def _json_safe_dict(value: dict[str, Any]) -> dict[str, Any]:
    safe = _json_safe(value)
    return safe if isinstance(safe, dict) else {}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump(mode="json"))
        except TypeError:
            return _json_safe(value.model_dump())
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return str(value)


__all__ = [
    "PLUGIN_EVENT_TYPES",
    "PluginEvent",
]
