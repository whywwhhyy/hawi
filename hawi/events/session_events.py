"""
Session events for Hawi Event System.

Emitted by SessionManager / SessionWriter to surface persistence activity to
GUI and observability consumers.
"""

from __future__ import annotations

from .event import Event


class SessionCheckpointRequestedEvent(Event):
    """A boundary event triggered SessionManager to capture snapshots."""

    session_id: str
    trigger: str
    components: list[str]

    @classmethod
    def create(
        cls,
        session_id: str,
        trigger: str,
        components: list[str],
    ) -> SessionCheckpointRequestedEvent:
        return cls(
            type="session.checkpoint_requested",
            source="agent",
            session_id=session_id,
            trigger=trigger,
            components=list(components),
        )


class SessionWriteFailedEvent(Event):
    """Writer thread failed to persist one or more components."""

    session_id: str
    component: str
    error: str

    @classmethod
    def create(
        cls,
        session_id: str,
        component: str,
        error: str,
    ) -> SessionWriteFailedEvent:
        return cls(
            type="session.write_failed",
            source="agent",
            session_id=session_id,
            component=component,
            error=error,
        )


class SessionLoadedEvent(Event):
    """A session has been loaded into the live agent."""

    session_id: str
    components_loaded: list[str]

    @classmethod
    def create(
        cls,
        session_id: str,
        components_loaded: list[str],
    ) -> SessionLoadedEvent:
        return cls(
            type="session.loaded",
            source="agent",
            session_id=session_id,
            components_loaded=list(components_loaded),
        )


class SessionSwitchedEvent(Event):
    """The agent switched from one session to another."""

    from_session_id: str | None
    to_session_id: str

    @classmethod
    def create(
        cls,
        from_session_id: str | None,
        to_session_id: str,
    ) -> SessionSwitchedEvent:
        return cls(
            type="session.switched",
            source="agent",
            from_session_id=from_session_id,
            to_session_id=to_session_id,
        )
