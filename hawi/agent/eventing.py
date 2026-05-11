"""Agent event subscription and publishing component."""

from __future__ import annotations

from typing import Callable, Protocol

from hawi.events import DumpManager, Event, EventBus, EventHandler, SyncEventHandler


class AgentEventOwner(Protocol):
    _event_bus: EventBus
    _dump_manager: DumpManager | None


class AgentEvents:
    """Explicit eventing component owned by HawiAgent."""

    def __init__(self, owner: AgentEventOwner) -> None:
        self._owner = owner

    @property
    def event_bus(self) -> EventBus:
        """Return the current agent EventBus."""
        return self._owner._event_bus

    def subscribe(
        self,
        callback: EventHandler,
        event_types: list[str] | None = None,
        maxsize: int = 100,
    ) -> None:
        """Subscribe to agent events."""
        self._owner._event_bus.subscribe(callback, event_types, maxsize)

    def subscribe_blocking(
        self,
        callback: SyncEventHandler,
        event_types: list[str] | None = None,
    ) -> None:
        """Subscribe to agent events with a blocking sync handler."""
        self._owner._event_bus.subscribe_blocking(callback, event_types)

    def unsubscribe(
        self,
        callback: Callable[[Event], None],
    ) -> bool:
        """Unsubscribe from agent events."""
        return self._owner._event_bus.unsubscribe(callback)

    async def emit(
        self,
        event: Event,
        event_bus: EventBus | None,
    ) -> Event:
        """Emit event to the agent bus and an optional external bus."""
        await self._owner._event_bus.publish_async(event)
        if event_bus is not None and event_bus is not self._owner._event_bus:
            await event_bus.publish_async(event)
        if self._owner._dump_manager is not None:
            self._owner._dump_manager.dump(event)
        return event
