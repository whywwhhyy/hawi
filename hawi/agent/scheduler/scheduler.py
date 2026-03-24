
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
from hawi.agent.result import AgentRunResult
from hawi.events import Event, EventBus
from hawi.events.agent_events import AgentToolResultEvent, AgentRunStopEvent
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
        *,
        event_bus: EventBus | None = None,
    ) -> None:
        """Initialize the scheduler.

        Args:
            agent: Agent to schedule
            event_bus: Optional event bus for publishing events
        """
        self._agent = agent
        self._event_bus = event_bus or EventBus()

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
        """Get the scheduler event bus."""
        return self._event_bus

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
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Enqueue a message.

        Args:
            content: Message content
            queue: Queue type ("normal", "high_prio", "urgent")
            metadata: Optional metadata

        Returns:
            Message ID

        Raises:
            SchedulerError: If enqueue fails
        """
        if queue == "urgent":
            msg = self._queue_manager.enqueue_urgent(content, metadata)
            # Urgent: trigger immediate interruption
            if not self._executor.is_idle:
                try:
                    asyncio.get_running_loop()
                    asyncio.create_task(self._executor.interrupt("urgent"))
                except RuntimeError:
                    # No event loop running, will be handled on next loop iteration
                    pass
        elif queue == "high_prio":
            msg = self._queue_manager.enqueue_high_prio(content, metadata)
        else:
            msg = self._queue_manager.enqueue_normal(content, metadata)

        # Emit enqueue event (safely handle no event loop case)
        try:
            asyncio.get_running_loop()
            asyncio.create_task(
                self._emit_event(
                    SchedulerEnqueueEvent.create(
                        message_id=msg.id,
                        queue_type=queue,
                        content_preview=msg.get_content_preview(),
                    )
                )
            )
        except RuntimeError:
            # No event loop running, skip event emission
            pass

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

        # Handle specific events
        if event.type == "agent.tool_result":
            await self._on_tool_call_end(event)
        elif event.type == "agent.run_stop":
            await self._on_agent_idle()

    async def _on_tool_call_end(self, event: AgentToolResultEvent) -> None:
        """Handle tool call completion.

        Check HIGH_PRIO queue and merge or queue at front.
        """
        if not self._queue_manager.has_high_prio():
            return

        high_prio_msg = self._queue_manager.dequeue_high_prio()
        if high_prio_msg is None:
            return

        # Check if last message is tool result
        messages = self._agent.context.messages
        if messages and messages[-1].get("role") == "tool":
            # Merge into tool result
            await self._merge_high_prio_to_tool_result(high_prio_msg)
        else:
            # Queue at front of normal queue
            self._queue_manager.insert_front_normal(high_prio_msg)

    async def _on_agent_idle(self) -> None:
        """Handle agent idle state.

        Check NORMAL queue and execute next message.
        """
        # Check for high priority messages first
        if self._queue_manager.has_high_prio():
            msg = self._queue_manager.dequeue_high_prio()
            if msg:
                await self._executor.execute(msg)
                return

        # Then check normal queue
        if self._queue_manager.has_normal():
            msg = self._queue_manager.dequeue_normal()
            if msg:
                await self._executor.execute(msg)
                return

        # No messages - scheduler is idle
        self._state = SchedulerState.IDLE

    async def _merge_high_prio_to_tool_result(
        self,
        high_prio_msg: QueuedMessage,
        tool_call_id: str | None = None,
        append_mode: Literal["text", "content_part", "auto"] = "auto",
    ) -> bool:
        """Merge high priority message into tool result.

        Args:
            high_prio_msg: Message to merge
            tool_call_id: Target tool call ID (None for last one)
            append_mode: How to append ("text", "content_part", or "auto")

        Returns:
            True if merge succeeded
        """
        messages = self._agent.context.messages
        if not messages:
            return False

        # Find target tool result message
        target_idx = len(messages) - 1
        if tool_call_id:
            # Find specific tool result
            for i, msg in enumerate(reversed(messages)):
                if msg.get("role") == "tool":
                    content = msg.get("content", [])
                    for part in content:
                        if (
                            isinstance(part, dict)
                            and part.get("type") == "tool_result"
                            and part.get("tool_call_id") == tool_call_id
                        ):
                            target_idx = len(messages) - 1 - i
                            break

        # Get content to merge
        if isinstance(high_prio_msg.content, str):
            merge_content = high_prio_msg.content
        else:
            # Extract text from content parts
            texts = []
            for part in high_prio_msg.content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part.get("text", ""))
            merge_content = "\\n\\n".join(texts)

        # Merge into tool result
        target_msg = messages[target_idx]
        target_content = list(target_msg.get("content", []))

        for part in target_content:
            if isinstance(part, dict) and part.get("type") == "tool_result":
                current_content = part.get("content", "")
                if isinstance(current_content, str):
                    new_content = f"{current_content}\\n\\n[Additional context: {merge_content}]"
                else:
                    # Append as new content part
                    new_content = list(current_content)
                    new_content.append({"type": "text", "text": merge_content})
                part["content"] = new_content
                break

        return True

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

    async def _emit_event(self, event: Event) -> None:
        """Emit event to event bus."""
        await self._event_bus.publish_async(event)

    # Main loop

    async def run_forever(self, poll_interval: float = 0.1) -> None:
        """Run scheduler in always-on mode.

        Args:
            poll_interval: Interval between queue checks (seconds)
        """
        self._running = True

        while self._running:
            try:
                # Check urgent first
                if self._queue_manager.has_urgent():
                    msg = self._queue_manager.dequeue_urgent()
                    if msg:
                        await self._executor.execute(msg)
                        continue

                # Check high priority
                if self._queue_manager.has_high_prio() and self._executor.is_idle:
                    msg = self._queue_manager.dequeue_high_prio()
                    if msg:
                        await self._executor.execute(msg)
                        continue

                # Check normal (only when idle)
                if self._queue_manager.has_normal() and self._executor.is_idle:
                    msg = self._queue_manager.dequeue_normal()
                    if msg:
                        await self._executor.execute(msg)
                        continue

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
        """Subscribe to scheduler events."""
        self._event_bus.subscribe(callback, event_types)

    def unsubscribe(self, callback: Callable[[Event], None]) -> bool:
        """Unsubscribe from scheduler events."""
        return self._event_bus.unsubscribe(callback)
