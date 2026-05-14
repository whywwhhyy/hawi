"""Agent executor for AgentRunner.

Manages agent execution lifecycle and interruption.
"""

from __future__ import annotations

import asyncio
from enum import Enum, auto
from typing import TYPE_CHECKING

from hawi.agent.result import AgentRunResult
from hawi.events import EventBus
from hawi.events.runner_events import AgentRunnerInterruptEvent
from hawi.models.message import ContentPart
from .queue import QueuedMessage, QueueType

if TYPE_CHECKING:
    from hawi.agent.agent import HawiAgent
    from .runner import AgentRunner


class AgentRunnerState(Enum):
    """AgentRunner execution states."""

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

    def __init__(self, agent: HawiAgent, runner: AgentRunner) -> None:
        """Initialize the agent executor."""
        self._agent = agent
        self._runner = runner
        self._state = AgentRunnerState.IDLE
        self._current_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._last_result: AgentRunResult | None = None
        self._current_event_bus: EventBus | None = None

    @property
    def state(self) -> AgentRunnerState:
        """Get current execution state."""
        return self._state

    @property
    def is_idle(self) -> bool:
        """Check if executor is idle."""
        return self._state == AgentRunnerState.IDLE and (
            self._current_task is None or self._current_task.done()
        )

    @property
    def is_running(self) -> bool:
        """Check if executor is running."""
        return self._state == AgentRunnerState.RUNNING

    @property
    def last_result(self) -> AgentRunResult | None:
        """Get last execution result."""
        return self._last_result

    @property
    def current_event_bus(self) -> EventBus | None:
        """Get the event bus override for the current execution."""
        return self._current_event_bus

    def _set_state(self, state: AgentRunnerState) -> None:
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
            if self._state == AgentRunnerState.IDLE:
                return []

            self._set_state(AgentRunnerState.INTERRUPTING)

            # Call agent interrupt
            interrupted_ids = self._agent.interrupt(reason)

            # Emit interrupt event
            await self._runner._emit_event(
                AgentRunnerInterruptEvent.create(
                    reason=reason,
                    interrupted_tool_calls=interrupted_ids,
                ),
                self._current_event_bus,
            )

            # Cancel current task if running
            current_task = self._current_task
            if current_task and not current_task.done():
                current_task.cancel()
                await asyncio.gather(current_task, return_exceptions=True)

            self._set_state(AgentRunnerState.IDLE)
            return interrupted_ids

    def execute(self, message: QueuedMessage) -> asyncio.Task | None:
        """Execute a queued message (non-blocking).

        This method starts the execution as a background task and returns
        immediately, allowing the runner to continue processing.

        Args:
            message: Message to execute

        Returns:
            Task object for the execution, or None if couldn't start
        """
        return self._start_execution(message, message.content)

    def execute_pending_inputs(
        self,
        event_bus: EventBus | None = None,
    ) -> asyncio.Task | None:
        """Execute pending agent steer inputs without adding a new message."""
        message = QueuedMessage.create(
            "",
            QueueType.HIGH_PRIO,
            {"source": "pending_inputs"},
            event_bus=event_bus,
        )
        return self._start_execution(message, None)

    def _start_execution(
        self,
        message: QueuedMessage,
        content: str | list[ContentPart] | None,
    ) -> asyncio.Task | None:
        if not self.is_idle:
            return None

        # Clear any previous interrupt state
        self._agent.clear_interrupt_state()
        self._set_state(AgentRunnerState.RUNNING)
        self._last_result = None
        self._current_event_bus = message.event_bus

        # Create and start task
        self._current_task = asyncio.create_task(
            self._execute_with_error_handling(message, content)
        )
        return self._current_task

    async def _execute_with_error_handling(
        self,
        message: QueuedMessage,
        content: str | list[ContentPart] | None,
    ) -> None:
        """Execute message with error handling."""
        try:
            result = await self._agent._arun_internal(
                content,
                event_bus=message.event_bus,
                message_metadata=(
                    self._message_metadata_for_execution(message)
                    if content is not None
                    else None
                ),
            )
            self._last_result = result
        except asyncio.CancelledError:
            # Execution was cancelled (interrupted)
            self._last_result = None
        except Exception as e:
            # Handle error through runner error hook
            self._last_result = None
            action = await self._runner._on_agent_error(e, message)
            if action == ErrorAction.RETRY:
                # Retry execution
                await self._execute_with_error_handling(message, content)
                return
            elif action == ErrorAction.ABORT:
                raise
            # CONTINUE — trigger pause to prevent automatic queue consumption
            await self._runner._on_execution_error(e, message)
        finally:
            self._set_state(AgentRunnerState.IDLE)
            self._current_event_bus = None

    @staticmethod
    def _message_metadata_for_execution(message: QueuedMessage) -> dict[str, object]:
        metadata: dict[str, object] = dict(message.metadata)
        queue = message.queue_type.name.lower()
        metadata.setdefault("message_id", message.id)
        if queue == "high_prio":
            metadata.setdefault("queue", "normal")
            metadata.setdefault("display_message_type", "normal")
            metadata.setdefault("source_queue", "high_prio")
            metadata.setdefault("materialized_as", "plain_user_message")
        else:
            metadata.setdefault("queue", queue)
            metadata.setdefault(
                "display_message_type",
                "urgent" if queue == "urgent" else "normal",
            )
        return metadata

    async def wait_for_complete(self) -> AgentRunResult | None:
        """Wait for current execution to complete.

        Returns:
            Last execution result
        """
        if self._current_task and not self._current_task.done():
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass
        return self._last_result
