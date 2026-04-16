
"""HawiScheduler - Message scheduling and agent orchestration.

Provides always-on agent capabilities with:
- Three-tier message queue (NORMAL, HIGH_PRIO, URGENT)
- Tool call interruption
- Event interception
- Multi-agent coordination support
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Literal

from hawi.agent.agent import HawiAgent
from hawi.agent.agent import SteerPartMergeMode
from hawi.agent.result import AgentRunResult
from hawi.events import Event, EventBus
from hawi.events.scheduler_events import (
    SchedulerEnqueueEvent,
    SchedulerDequeueEvent,
)
from hawi.models.message import ContentPart

from .queue import QueueType, QueuedMessage, MessageQueueManager
from .interceptor import EventMode, EventInterceptor
from .executor import AgentExecutor, SchedulerState, ErrorAction

logger = logging.getLogger(__name__)


class SchedulerError(Exception):
    """Error raised by HawiScheduler."""
    pass


# Error hook protocols
class ModelErrorHook:
    """Protocol for model error handling hook."""

    async def on_model_error(
        self, error: Exception, context: Any
    ) -> ErrorAction:
        """Handle model call error."""
        return ErrorAction.CONTINUE


class AgentErrorHook:
    """Protocol for agent error handling hook."""

    async def on_agent_error(
        self, error: Exception, message: QueuedMessage, context: Any
    ) -> ErrorAction:
        """Handle agent execution error."""
        return ErrorAction.CONTINUE


class SchedulerErrorHook:
    """Protocol for scheduler error handling hook."""

    async def on_scheduler_error(self, error: Exception) -> ErrorAction:
        """Handle scheduler internal error."""
        return ErrorAction.CONTINUE


class HawiScheduler:
    """Scheduler for managing agent execution with message queues.

    Features:
    - Three-tier message queue system (NORMAL, HIGH_PRIO, URGENT)
    - Urgent message interruption
    - High priority message merging
    - Event interception
    - Always-on mode with run_forever()
    """

    def __init__(
        self,
        agent: HawiAgent,
    ) -> None:
        """Initialize the scheduler.

        Args:
            agent: Agent to schedule
        """
        self._agent = agent

        # Queue management
        self._queue_manager = MessageQueueManager()

        # Event interceptor
        self._interceptor = EventInterceptor(self)

        # Agent executor
        self._executor = AgentExecutor(agent, self)

        # State
        self._running = False
        self._state = SchedulerState.IDLE

        # Error hooks
        self._model_error_hook: ModelErrorHook | None = None
        self._agent_error_hook: AgentErrorHook | None = None
        self._scheduler_error_hook: SchedulerErrorHook | None = None

        # Result callback for multi-agent coordination
        self._result_callback: Callable[[str, AgentRunResult], None] | None = None

        # Subscribe to agent events
        self._agent.event_bus.subscribe(self._on_agent_event)

    @property
    def agent(self) -> HawiAgent:
        """Get the managed agent."""
        return self._agent

    @property
    def event_bus(self) -> EventBus:
        """Get the underlying agent event bus."""
        return self._agent.event_bus

    @property
    def state(self) -> SchedulerState:
        """Get current scheduler state."""
        return self._state

    def set_model_error_hook(self, hook: ModelErrorHook) -> None:
        """Set model error handling hook."""
        self._model_error_hook = hook

    def set_agent_error_hook(self, hook: AgentErrorHook) -> None:
        """Set agent error handling hook."""
        self._agent_error_hook = hook

    def set_scheduler_error_hook(self, hook: SchedulerErrorHook) -> None:
        """Set scheduler error handling hook."""
        self._scheduler_error_hook = hook

    def set_result_callback(
        self, callback: Callable[[str, AgentRunResult], None]
    ) -> None:
        """Set result callback for multi-agent coordination."""
        self._result_callback = callback

    # Queue operations

    def enqueue(
        self,
        content: str | list[ContentPart],
        queue: Literal["normal", "high_prio", "urgent"] = "normal",
        event_bus: EventBus | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Enqueue a message.

        Args:
            content: Message content
            queue: Queue type ("normal", "high_prio", "urgent")
            event_bus: Optional event bus override for this queued execution
            metadata: Optional metadata

        Returns:
            Message ID

        Raises:
            SchedulerError: If enqueue fails
        """
        if queue == "urgent":
            msg = self._queue_manager.enqueue_urgent(
                content,
                metadata,
                event_bus=event_bus,
            )
            # Urgent: trigger immediate interruption
            if not self._executor.is_idle:
                try:
                    asyncio.get_running_loop()
                    asyncio.create_task(self._executor.interrupt("urgent"))
                except RuntimeError:
                    # No event loop running, will be handled on next loop iteration
                    pass
        elif queue == "high_prio":
            if not self._executor.is_idle:
                merge_mode = self._resolve_steer_merge_mode(metadata)
                msg_id = self._agent.steer(content, merge_mode=merge_mode)
                self._emit_enqueue_event(
                    message_id=msg_id,
                    queue_type=queue,
                    content=content,
                    event_bus=event_bus,
                )
                return msg_id
            msg = self._queue_manager.enqueue_high_prio(
                content,
                metadata,
                event_bus=event_bus,
            )
        else:
            msg = self._queue_manager.enqueue_normal(
                content,
                metadata,
                event_bus=event_bus,
            )

        self._emit_enqueue_event(
            message_id=msg.id,
            queue_type=queue,
            content=content,
            event_bus=event_bus,
        )

        return msg.id

    def remove_message(self, message_id: str) -> bool:
        """Remove a message by ID.

        Args:
            message_id: Message ID to remove

        Returns:
            True if message was found and removed
        """
        return self._queue_manager.remove_message(message_id)

    def remove_messages(
        self, filter_fn: Callable[[QueuedMessage], bool]
    ) -> list[str]:
        """Remove messages matching filter.

        Args:
            filter_fn: Filter function returning True for messages to remove

        Returns:
            List of removed message IDs
        """
        return self._queue_manager.remove_messages(filter_fn)

    def clear_queue(self, queue: Literal["normal", "high_prio", "urgent"]) -> int:
        """Clear a specific queue.

        Args:
            queue: Queue type to clear

        Returns:
            Number of messages cleared
        """
        if queue == "urgent":
            return self._queue_manager.clear_queue(QueueType.URGENT)
        elif queue == "high_prio":
            return self._queue_manager.clear_queue(QueueType.HIGH_PRIO)
        else:
            return self._queue_manager.clear_queue(QueueType.NORMAL)

    def clear_all_queues(self) -> dict[str, int]:
        """Clear all queues.

        Returns:
            Dictionary with counts of cleared messages per queue
        """
        return self._queue_manager.clear_all_queues()

    def get_queue_lengths(self) -> dict[str, int]:
        """Get current queue lengths.

        Returns:
            Dictionary with queue lengths
        """
        return self._queue_manager.get_queue_lengths()

    # Event handling

    async def _on_agent_event(self, event: Event) -> None:
        """Handle events from agent.

        Args:
            event: Agent event
        """
        # First pass through interceptor
        mode = await self._interceptor.handle(event)
        if mode == EventMode.SUPPRESS:
            return
        if mode != EventMode.PASS_THROUGH:
            # Event was reprocessed or intercepted
            return

        if event.type == "agent.run_stop":
            await self._on_agent_idle()

    async def _on_agent_idle(self) -> None:
        """Handle agent idle state.

        Called when agent run stops. Only updates scheduler state,
        actual message execution is handled by run_forever() main loop.
        """
        # Just update state, let run_forever handle next message
        self._state = SchedulerState.IDLE

    # Error handling

    async def _on_model_error(self, error: Exception, context: Any) -> ErrorAction:
        """Handle model error."""
        if self._model_error_hook:
            action = await self._model_error_hook.on_model_error(error, context)
            if action is not None:
                return action
        logger.error(f"Model error: {error}")
        return ErrorAction.CONTINUE

    async def _on_agent_error(
        self, error: Exception, message: QueuedMessage
    ) -> ErrorAction:
        """Handle agent execution error."""
        if self._agent_error_hook:
            action = await self._agent_error_hook.on_agent_error(
                error, message, self._agent.context
            )
            if action is not None:
                return action
        logger.error(f"Agent error: {error}")
        return ErrorAction.CONTINUE

    async def _on_scheduler_error(self, error: Exception) -> ErrorAction:
        """Handle scheduler internal error."""
        if self._scheduler_error_hook:
            action = await self._scheduler_error_hook.on_scheduler_error(error)
            if action is not None:
                return action
        logger.error(f"Scheduler error: {error}")
        return ErrorAction.CONTINUE

    # Event emission

    async def _emit_event(
        self,
        event: Event,
        event_bus: EventBus | None = None,
    ) -> None:
        """Emit scheduler event via the agent's event pipeline."""
        await self._agent._emit_event(event, event_bus)

    def _resolve_steer_merge_mode(
        self,
        metadata: dict[str, Any] | None,
    ) -> SteerPartMergeMode:
        """Resolve the steer merge mode from queue metadata."""
        if metadata is None:
            return SteerPartMergeMode.APPEND_TO_TOOL_RESULT

        raw_mode = metadata.get("steer_merge_mode")
        if isinstance(raw_mode, SteerPartMergeMode):
            return raw_mode
        if isinstance(raw_mode, str):
            try:
                return SteerPartMergeMode(raw_mode)
            except ValueError:
                logger.warning("Unknown steer merge mode '%s', falling back to append.", raw_mode)
        return SteerPartMergeMode.APPEND_TO_TOOL_RESULT

    def _build_content_preview(self, content: str | list[ContentPart]) -> str:
        """Build a short content preview without creating a queued message."""
        if isinstance(content, str):
            text = content
        else:
            texts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            text = " ".join(texts)
        return text[:97] + "..." if len(text) > 100 else text

    def _emit_enqueue_event(
        self,
        *,
        message_id: str,
        queue_type: str,
        content: str | list[ContentPart],
        event_bus: EventBus | None = None,
    ) -> None:
        """Emit enqueue event when an event loop is available."""
        try:
            asyncio.get_running_loop()
            asyncio.create_task(
                self._emit_event(
                    SchedulerEnqueueEvent.create(
                        message_id=message_id,
                        queue_type=queue_type,
                        content_preview=self._build_content_preview(content),
                    ),
                    event_bus,
                )
            )
        except RuntimeError:
            pass

    async def _start_message_execution(self, msg: QueuedMessage) -> bool:
        """Emit dequeue event and hand the message to the executor."""
        queue_name = msg.queue_type.name.lower()
        await self._emit_event(
            SchedulerDequeueEvent.create(
                message_id=msg.id,
                queue_type=queue_name,
            ),
            msg.event_bus,
        )

        task = self._executor.execute(msg)
        if task:
            self._state = SchedulerState.RUNNING
            return True
        return False

    # Main loop

    async def run_forever(self, poll_interval: float = 0.1) -> None:
        """Run scheduler in always-on mode.

        Args:
            poll_interval: Interval between queue checks (seconds)
        """
        self._running = True

        while self._running:
            try:
                # Check urgent first (always process, even if busy)
                if self._queue_manager.has_urgent():
                    msg = self._queue_manager.dequeue_urgent()
                    if msg:
                        # Interrupt current execution if any
                        if not self._executor.is_idle:
                            await self._executor.interrupt("urgent")
                        if await self._start_message_execution(msg):
                            continue

                # Check high priority (only when idle)
                if self._executor.is_idle and self._queue_manager.has_high_prio():
                    msg = self._queue_manager.dequeue_high_prio()
                    if msg:
                        if await self._start_message_execution(msg):
                            continue

                # Check normal (only when idle)
                if self._executor.is_idle and self._queue_manager.has_normal():
                    msg = self._queue_manager.dequeue_normal()
                    if msg:
                        if await self._start_message_execution(msg):
                            continue

                # Update state to idle if executor is idle
                if self._executor.is_idle:
                    self._state = SchedulerState.IDLE

                # No messages - wait
                await asyncio.sleep(poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                action = await self._on_scheduler_error(e)
                if action == ErrorAction.ABORT:
                    break

    def stop(self) -> None:
        """Stop the scheduler loop."""
        self._running = False

    async def interrupt(self, reason: str = "user") -> list[str]:
        """Interrupt current executor run without stopping scheduler loop."""
        return await self._executor.interrupt(reason)

    # Multi-agent coordination

    async def receive_signal(self, signal: str, payload: Any) -> None:
        """Receive external control signal.

        Args:
            signal: Signal type (e.g., "suspend", "resume", "reset")
            payload: Signal payload
        """
        if signal == "suspend":
            await self._executor.interrupt("signal")
        elif signal == "reset":
            self.clear_all_queues()
            await self._executor.interrupt("reset")

    # Convenience methods

    def subscribe(
        self,
        callback: Callable[[Event], None],
        event_types: list[str] | None = None,
    ) -> None:
        """Subscribe to the wrapped agent's event stream."""
        self._agent.subscribe(callback, event_types)

    def unsubscribe(self, callback: Callable[[Event], None]) -> bool:
        """Unsubscribe from the wrapped agent's event stream."""
        return self._agent.unsubscribe(callback)
