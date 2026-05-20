"""Sub-agent subsystem events.

Sub-agents are a first-class runtime subsystem. The plugin named
``hawi/subagent`` only exposes tool-facing controls for this subsystem; the
observable lifecycle and transcript stream use dedicated ``subagent.*`` events.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from .event import Event, SubAgentEventType


SUBAGENT_EVENT_TYPES: set[str] = {
    "subagent.created",
    "subagent.event",
    "subagent.closed",
}


class SubAgentEvent(Event):
    """Lifecycle and child-event notification for a managed sub-agent."""

    type: SubAgentEventType  # type: ignore[reportIncompatibleVariableOverride]
    source: Literal["subagent"]  # type: ignore[reportIncompatibleVariableOverride]
    subagent_id: str
    subagent_name: str
    subagent_role: str
    status: dict[str, Any] = Field(default_factory=dict)
    child_event: dict[str, Any] | None = None
    message_entry: dict[str, Any] | None = None
    reason: str | None = None

    @classmethod
    def create(
        cls,
        event_type: SubAgentEventType,
        *,
        subagent_id: str,
        subagent_name: str,
        subagent_role: str,
        status: dict[str, Any] | None = None,
        child_event: dict[str, Any] | None = None,
        message_entry: dict[str, Any] | None = None,
        reason: str | None = None,
    ) -> "SubAgentEvent":
        return cls(
            type=event_type,
            source="subagent",
            subagent_id=subagent_id,
            subagent_name=subagent_name,
            subagent_role=subagent_role,
            status=_json_safe_dict(status or {}),
            child_event=_json_safe_dict(child_event) if child_event is not None else None,
            message_entry=_json_safe_dict(message_entry) if message_entry is not None else None,
            reason=reason,
        )


def _json_safe_dict(value: dict[str, Any] | None) -> dict[str, Any]:
    safe = _json_safe(value or {})
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
    "SUBAGENT_EVENT_TYPES",
    "SubAgentEvent",
]
