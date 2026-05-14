
"""Event interceptor for AgentRunner.

Provides event filtering and transformation capabilities:
- PASS_THROUGH: Let event through unchanged
- INTERCEPT: Block the event (do not forward)
- REPROCESS: Transform and emit new event
- SUPPRESS: Discard the event
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, Callable

from hawi.events import Event

if TYPE_CHECKING:
    from .runner import AgentRunner


class EventMode(Enum):
    """Event handling modes."""

    PASS_THROUGH = auto()  # Let event through unchanged
    INTERCEPT = auto()  # Block the event
    REPROCESS = auto()  # Transform and emit new event
    SUPPRESS = auto()  # Discard the event


class EventInterceptor:
    """Intercepts and processes events from the EventBus."""

    def __init__(self, runner: "AgentRunner") -> None:
        """Initialize the event interceptor."""
        self._runner = runner
        self._handlers: dict[str, Callable[[Event], EventMode]] = {}
        self._transforms: dict[str, Callable[[Event], Event | None]] = {}

    def register_handler(
        self,
        event_type: str,
        handler: Callable[[Event], EventMode],
    ) -> None:
        """Register a handler for a specific event type.

        Args:
            event_type: Event type to handle
            handler: Handler function returning EventMode
        """
        self._handlers[event_type] = handler

    def register_transform(
        self,
        event_type: str,
        transform: Callable[[Event], Event | None],
    ) -> None:
        """Register a transform for a specific event type.

        Args:
            event_type: Event type to transform
            transform: Transform function returning new event or None
        """
        self._transforms[event_type] = transform

    def unregister_handler(self, event_type: str) -> bool:
        """Unregister a handler for a specific event type.

        Returns:
            True if handler was found and removed
        """
        if event_type in self._handlers:
            del self._handlers[event_type]
            return True
        return False

    def unregister_transform(self, event_type: str) -> bool:
        """Unregister a transform for a specific event type.

        Returns:
            True if transform was found and removed
        """
        if event_type in self._transforms:
            del self._transforms[event_type]
            return True
        return False

    async def handle(self, event: Event) -> EventMode:
        """Handle an event.

        Args:
            event: Event to handle

        Returns:
            EventMode indicating how to handle the event
        """
        event_type = event.type

        # Check for transform first
        if event_type in self._transforms:
            transform = self._transforms[event_type]
            new_event = transform(event)
            if new_event is None:
                return EventMode.SUPPRESS
            if new_event is not event:
                # Emit the transformed event
                await self._runner._emit_event(new_event)
                return EventMode.REPROCESS

        # Check for handler
        if event_type in self._handlers:
            return self._handlers[event_type](event)

        return EventMode.PASS_THROUGH
