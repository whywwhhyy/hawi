
"""Agent executor for HawiScheduler.

Manages agent execution lifecycle and interruption.
"""

from __future__ import annotations

import asyncio
from enum import Enum, auto
from typing import TYPE_CHECKING

from hawi.agent.result import AgentRunResult
from hawi.events.scheduler_events import SchedulerInterruptEvent

if TYPE_CHECKING:
    from hawi.agent.agent import HawiAgent
    from .scheduler import HawiScheduler
    from .queue import QueuedMessage


class SchedulerState(Enum):
    """Scheduler execution states."""

    IDLE = auto()  # Waiting for messages
    READY = auto()  # Has message, checking priority
    RUNNING = auto()  # Normal execution
    INTERRUPTING = auto()  # Being interrupted


class ErrorAction(Enum):
    """Error handling actions."""

    RETRY = auto()
    ABORT = auto()
    CONTINUE = auto()


class AgentExecutor:
    """Executes agent runs with support for interruption."""

    def __init__(self, agent: HawiAgent, scheduler: HawiScheduler) -> None:
        """Initialize the agent executor."""
        self._agent = agent
        self._scheduler = scheduler
        self._state = SchedulerState.IDLE
        self._current_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    @property
    def state(self) -> SchedulerState:
        """Get current execution state."""
        return self._state

    @property
    def is_idle(self) -> bool:
        """Check if executor is idle."""
        return self._state == SchedulerState.IDLE

    @property
    def is_running(self) -> bool:
        """Check if executor is running."""
        return self._state == SchedulerState.RUNNING

    def _set_state(self, state: SchedulerState) -> None:
        """Update execution state."""
        self._state = state

    async def interrupt(self, reason: str) -> list[str]:
        """Interrupt current agent execution.

        Args:
            reason: Reason for interruption

        Returns:
            List of interrupted tool call IDs
        """
        async with self._lock:
            if self._state != SchedulerState.RUNNING:
                return []

            self._set_state(SchedulerState.INTERRUPTING)

            # Call agent interrupt
            interrupted_ids = self._agent.interrupt(reason)

            # Emit interrupt event
            await self._scheduler._emit_event(
                SchedulerInterruptEvent.create(
                    reason=reason,
                    interrupted_tool_calls=interrupted_ids,
                )
            )

            # Cancel current task if running
            if self._current_task and not self._current_task.done():
                self._current_task.cancel()
                try:
                    await self._current_task
                except asyncio.CancelledError:
                    pass

            self._set_state(SchedulerState.IDLE)
            return interrupted_ids

    async def execute(self, message: QueuedMessage) -> AgentRunResult | None:
        """Execute a queued message.

        Args:
            message: Message to execute

        Returns:
            Agent run result or None if execution failed
        """
        async with self._lock:
            # Clear any previous interrupt state
            self._agent.clear_interrupt_state()

            self._set_state(SchedulerState.RUNNING)

            try:
                # Create task for agent execution
                self._current_task = asyncio.create_task(
                    self._agent.arun(message.content)
                )
                result = await self._current_task
                return result
            except asyncio.CancelledError:
                # Execution was cancelled (interrupted)
                return None
            except Exception as e:
                # Handle error through scheduler error hook
                action = await self._scheduler._on_agent_error(e, message)
                if action == ErrorAction.RETRY:
                    # Retry execution
                    return await self.execute(message)
                elif action == ErrorAction.ABORT:
                    raise
                else:  # CONTINUE
                    return None
            finally:
                self._current_task = None
                self._set_state(SchedulerState.IDLE)
