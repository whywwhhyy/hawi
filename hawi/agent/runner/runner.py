
"""AgentRunner - Message scheduling and agent orchestration.

Provides always-on agent capabilities with:
- Three-tier message queue (NORMAL, HIGH_PRIO, URGENT)
- Tool call interruption
- Event interception
- Multi-agent coordination support
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Literal

from hawi.agent.agent import HawiAgent
from hawi.agent.content_utils import merge_content_parts
from hawi.agent.state import SteerPartMergeMode
from hawi.errors import ConfigurationError
from hawi.agent.result import AgentRunResult
from hawi.events import Event, EventBus
from hawi.events.runner_events import (
    AgentRunnerEnqueueEvent,
    AgentRunnerDequeueEvent,
)
from hawi.models.message import ContentPart

from .queue import QueueMessageSnapshot, QueueType, QueuedMessage, MessageQueueManager
from .interceptor import EventMode, EventInterceptor
from .executor import AgentExecutor, AgentRunnerState, ErrorAction

logger = logging.getLogger(__name__)


class AgentRunnerError(Exception):
    """Error raised by AgentRunner."""
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


class AgentRunnerErrorHook:
    """Protocol for runner error handling hook."""

    async def on_runner_error(self, error: Exception) -> ErrorAction:
        """Handle runner internal error."""
        return ErrorAction.CONTINUE


class AgentRunner:
    """AgentRunner for managing agent execution with message queues.

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
        """Initialize the runner.

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
        self._state = AgentRunnerState.IDLE

        # Error hooks
        self._model_error_hook: ModelErrorHook | None = None
        self._agent_error_hook: AgentErrorHook | None = None
        self._runner_error_hook: AgentRunnerErrorHook | None = None

        # Result callback for multi-agent coordination
        self._result_callback: Callable[[str, AgentRunResult], None] | None = None

        # Pause/control state
        self._paused = False
        self._pause_reason: str | None = None
        self._paused_at: float | None = None
        self._last_pause_error: str | None = None

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
    def state(self) -> AgentRunnerState:
        """Get current runner state."""
        return self._state

    @property
    def executor_state(self) -> AgentRunnerState:
        """Get the current executor state."""
        return self._executor.state

    @property
    def agent_state(self) -> AgentRunnerState:
        """Get the current managed agent execution state."""
        return self._executor.state

    @property
    def executor_is_idle(self) -> bool:
        """Whether the underlying executor is idle."""
        return self._executor.is_idle

    @property
    def is_idle(self) -> bool:
        """Whether the runner has no active agent execution."""
        return self._executor.is_idle

    @property
    def queue_manager(self) -> MessageQueueManager:
        """Expose queue persistence and inspection for session storage."""
        return self._queue_manager

    @property
    def last_result(self) -> AgentRunResult | None:
        """Return the last completed agent result, if any."""
        return self._executor.last_result

    def set_model_error_hook(self, hook: ModelErrorHook) -> None:
        """Set model error handling hook."""
        self._model_error_hook = hook

    def set_agent_error_hook(self, hook: AgentErrorHook) -> None:
        """Set agent error handling hook."""
        self._agent_error_hook = hook

    def set_runner_error_hook(self, hook: AgentRunnerErrorHook) -> None:
        """Set runner error handling hook."""
        self._runner_error_hook = hook

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
            AgentRunnerError: If enqueue fails
        """
        # Determine intent from metadata
        intent = (metadata or {}).get("intent", "legacy")

        # Urgent: downgrade to stop with message (compatibility path)
        if queue == "urgent":
            try:
                asyncio.get_running_loop()
                asyncio.create_task(self.stop_execution(
                    reason="urgent",
                    message=content,
                    pause=False,
                    event_bus=event_bus,
                    metadata=metadata,
                ))
            except RuntimeError:
                # No event loop running; fall back to old urgent behavior
                msg = self._queue_manager.enqueue_urgent(
                    content, metadata, event_bus=event_bus,
                )
                if not self._executor.is_idle:
                    try:
                        asyncio.create_task(self._executor.interrupt("urgent"))
                    except RuntimeError:
                        pass
                return msg.id
            # Return a placeholder id; actual message id from stop()
            return ""

        # Resume intent: clear pause before enqueue
        if intent in ("user_send", "resume") and self._paused:
            self._resume_internal()

        if queue == "high_prio":
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

    def get_queue_messages(self) -> dict[str, list[QueueMessageSnapshot]]:
        """Get current queued message previews grouped by queue kind."""
        return self._queue_manager.get_queue_messages()

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

        Called when agent run stops. Only updates runner state,
        actual message execution is handled by run_forever() main loop.
        """
        # Just update state, let run_forever handle next message
        self._state = AgentRunnerState.IDLE

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

    async def _on_runner_error(self, error: Exception) -> ErrorAction:
        """Handle runner internal error."""
        if self._runner_error_hook:
            action = await self._runner_error_hook.on_runner_error(error)
            if action is not None:
                return action
        logger.error(f"AgentRunner error: {error}")
        return ErrorAction.CONTINUE

    # Event emission

    async def _emit_event(
        self,
        event: Event,
        event_bus: EventBus | None = None,
    ) -> None:
        """Emit runner event via the agent's event pipeline."""
        await self._agent._emit_event(event, event_bus)

    def _resolve_steer_merge_mode(
        self,
        metadata: dict[str, Any] | None,
    ) -> SteerPartMergeMode | None:
        """Resolve an explicit steer merge mode from queue metadata."""
        if metadata is None:
            return None

        raw_mode = metadata.get("steer_merge_mode")
        if raw_mode is None:
            return None
        if isinstance(raw_mode, SteerPartMergeMode):
            return raw_mode
        if isinstance(raw_mode, str):
            try:
                return SteerPartMergeMode(raw_mode)
            except ValueError:
                pass
        valid = ", ".join(mode.value for mode in SteerPartMergeMode)
        raise ConfigurationError(
            f"Invalid high_prio steer_merge_mode: {raw_mode!r}. "
            f"Valid values are: {valid}. "
            "Omit steer_merge_mode to use the model's declared mode."
        )

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
                    AgentRunnerEnqueueEvent.create(
                        message_id=message_id,
                        queue_type=queue_type,
                        content_preview=self._build_content_preview(content),
                    ),
                    event_bus,
                )
            )
        except RuntimeError:
            pass

    # === Pause / Resume Control ===

    def is_paused(self) -> bool:
        """Check if the runner is in paused state.

        When paused, the runner does not automatically consume any queued
        messages or pending steer inputs.
        """
        return self._paused

    def pause(
        self,
        reason: str,
        *,
        error_message: str | None = None,
    ) -> None:
        """Pause the runner, stopping automatic queue consumption.

        Args:
            reason: Pause reason (e.g. "user_interrupt", "model_error")
            error_message: Optional error description for display
        """
        self._paused = True
        self._pause_reason = reason
        self._paused_at = time.time()
        self._last_pause_error = error_message

    def resume(self) -> None:
        """Resume the runner from paused state.

        Only clears control state — does not send messages or start execution.
        Use :meth:`submit_immediate_message` or :meth:`resume_with_prompt` to
        both resume and enqueue a message.
        """
        self._resume_internal()

    def has_pending_immediate_work(self) -> bool:
        """Return whether resume can continue already queued immediate work."""
        return (
            self._queue_manager.has_urgent()
            or self._queue_manager.has_high_prio()
            or self._agent_has_pending_inputs()
        )

    def _resume_internal(self) -> None:
        """Internal: clear pause state without emitting events."""
        self._paused = False
        self._pause_reason = None
        self._paused_at = None
        self._last_pause_error = None

    def control_snapshot(self) -> dict[str, Any]:
        """Return current control state as a JSON-friendly dict."""
        return {
            "paused": self._paused,
            "pause_reason": self._pause_reason,
            "resumable": self._paused and not self._executor.is_running,
            "paused_at": self._paused_at,
            "last_error_message": self._last_pause_error,
        }

    def submit_immediate_message(
        self,
        content: str | list[ContentPart],
        *,
        intent: str = "user_send",
        event_bus: EventBus | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Submit a message for immediate execution, clearing pause first.

        This is the preferred way to send a user message or resume prompt.
        It clears pause, enqueues with ``high_prio``, and returns the message ID.

        Args:
            content: Message content
            intent: Message intent ("user_send", "resume", "stop_with_message")
            event_bus: Optional event bus override
            metadata: Optional metadata (intent is auto-injected)

        Returns:
            Message ID
        """
        # Clear pause first
        self._resume_internal()

        merged_metadata = dict(metadata or {})
        merged_metadata.setdefault("intent", intent)
        merged_metadata.setdefault("display_message_type", intent)

        msg = self._queue_manager.enqueue_high_prio(
            content,
            merged_metadata,
            event_bus=event_bus,
        )
        self._emit_enqueue_event(
            message_id=msg.id,
            queue_type="high_prio",
            content=content,
            event_bus=event_bus,
        )
        return msg.id

    async def stop_execution(
        self,
        reason: str = "user",
        *,
        message: str | list[ContentPart] | None = None,
        pause: bool | None = None,
        event_bus: EventBus | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Stop current execution, with optional follow-up message.

        Pure stop (``message=None``): Interrupts current run and pauses the
        runner. Queued normal tasks are preserved but not automatically consumed.

        Stop with message (``message is not None``): Interrupts current run,
        does NOT pause (or pauses briefly and resumes), and submits the message
        as ``stop_with_message`` intent.

        Args:
            reason: Reason for stopping ("user", "urgent")
            message: Optional message to execute after stopping
            pause: Force pause behavior. Default: True if message is None
            event_bus: Optional event bus for the follow-up message
            metadata: Optional metadata for the follow-up message

        Returns:
            Dict with keys:
                - ``interrupted_tool_calls``: list[str]
                - ``message_id``: str | None
                - ``control``: dict from control_snapshot()
        """
        if pause is None:
            pause = message is None

        # Interrupt current execution
        interrupted_tool_calls = await self._executor.interrupt(reason)

        if message is not None:
            # Stop with message: submit immediately
            merged_metadata = dict(metadata or {})
            merged_metadata.setdefault("intent", "stop_with_message")
            msg_id = self.submit_immediate_message(
                message,
                intent="stop_with_message",
                event_bus=event_bus,
                metadata=merged_metadata,
            )
            return {
                "interrupted_tool_calls": interrupted_tool_calls,
                "message_id": msg_id,
                "control": self.control_snapshot(),
            }
        else:
            # Pure stop
            if pause:
                self.pause(
                    "user_interrupt" if reason == "user" else reason,
                    error_message=None,
                )
            return {
                "interrupted_tool_calls": interrupted_tool_calls,
                "message_id": None,
                "control": self.control_snapshot(),
            }

    async def _on_execution_error(
        self, error: Exception, message: QueuedMessage
    ) -> None:
        """Called by the executor when a non-recoverable error finishes a run.

        Classifies the error and sets the appropriate pause state so the
        runner does not automatically continue consuming the queue.
        """
        reason = self._classify_pause_reason(error)
        self.pause(reason, error_message=str(error))

    @staticmethod
    def _classify_pause_reason(error: Exception) -> str:
        """Classify an exception into a pause reason string."""
        err_str = str(error).lower()
        type_name = type(error).__name__.lower()
        combined = f"{type_name} {err_str}"
        if any(kw in combined for kw in ("connection", "timeout", "network", "econnrefused", "econnreset")):
            return "network_error"
        if any(kw in combined for kw in ("model", "api", "rate_limit", "token")):
            return "model_error"
        return "runtime_error"

    # === Message execution ===

    async def _start_message_execution(self, msg: QueuedMessage) -> bool:
        """Emit dequeue event and hand the message to the executor."""
        queue_name = msg.queue_type.name.lower()
        await self._emit_event(
            AgentRunnerDequeueEvent.create(
                message_id=msg.id,
                queue_type=queue_name,
            ),
            msg.event_bus,
        )

        task = self._executor.execute(msg)
        if task:
            self._state = AgentRunnerState.RUNNING
            return True
        return False

    def _dequeue_merged_high_prio_message(self) -> QueuedMessage | None:
        messages = self._queue_manager.dequeue_all_high_prio()
        if not messages:
            return None
        if len(messages) == 1:
            return messages[0]

        first = messages[0]
        merged_metadata = dict(first.metadata)
        merged_metadata["merged_message_ids"] = [msg.id for msg in messages]
        merged_metadata["merged_message_count"] = len(messages)
        return QueuedMessage(
            id=first.id,
            content=merge_content_parts(msg.content for msg in messages),
            queue_type=QueueType.HIGH_PRIO,
            created_at=first.created_at,
            event_bus=first.event_bus,
            metadata=merged_metadata,
        )

    async def _start_pending_input_execution(self) -> bool:
        """Run agent pending steer inputs before moving on to queued messages."""
        await self._emit_event(
            AgentRunnerDequeueEvent.create(
                message_id="pending-inputs",
                queue_type="high_prio",
            ),
            None,
        )

        task = self._executor.execute_pending_inputs()
        if task:
            self._state = AgentRunnerState.RUNNING
            return True
        return False

    def _agent_has_pending_inputs(self) -> bool:
        getter = getattr(self._agent, "has_pending_inputs", None)
        return callable(getter) and getter() is True

    # === Main loop ===

    async def run_forever(self, poll_interval: float = 0.1) -> None:
        """Run runner in always-on mode.

        When paused, the loop will not consume any queued messages or pending
        steer inputs. Only ``resume()`` / ``submit_immediate_message()`` /
        ``stop(message=...)`` can reactivate execution.

        Args:
            poll_interval: Interval between queue checks (seconds)
        """
        self._running = True

        while self._running:
            try:
                # Paused: do NOT consume any queue or pending inputs
                if self._paused:
                    await asyncio.sleep(poll_interval)
                    continue

                # Check urgent first (legacy path, only if no event loop)
                if self._queue_manager.has_urgent():
                    msg = self._queue_manager.dequeue_urgent()
                    if msg:
                        if not self._executor.is_idle:
                            await self._executor.interrupt("urgent")
                        if await self._start_message_execution(msg):
                            continue

                # Drain steered inputs that survived an interruption before
                # continuing with runner-owned queues.
                if self._executor.is_idle and self._agent_has_pending_inputs():
                    if await self._start_pending_input_execution():
                        continue

                # Check high priority (only when idle)
                if self._executor.is_idle and self._queue_manager.has_high_prio():
                    msg = self._dequeue_merged_high_prio_message()
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
                    self._state = AgentRunnerState.IDLE

                # No messages - wait
                await asyncio.sleep(poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                action = await self._on_runner_error(e)
                if action == ErrorAction.ABORT:
                    break

    def stop(self) -> None:
        """Stop the runner loop (synchronous, backward compat).

        Sets the ``_running`` flag so ``run_forever()`` exits on its next
        iteration. Does not interrupt current execution.
        """
        self._running = False

    def _stop_loop(self) -> None:
        """Stop the runner loop (internal alias)."""
        self._running = False

    async def interrupt(
        self,
        reason: str = "user",
        *,
        pause: bool = False,
        message: str | list[ContentPart] | None = None,
    ) -> list[str]:
        """Interrupt current executor run.

        Args:
            reason: Reason for interruption
            pause: If True, enter paused state after interrupt (pure stop)
            message: If provided, treat as stop-with-message

        Returns:
            List of interrupted tool call IDs

        Note:
            Prefer the higher-level :meth:`stop_execution` method. This is kept
            for backward compatibility.
        """
        if message is not None:
            result = await self.stop_execution(
                reason=reason,
                message=message,
                pause=pause,
            )
            return result.get("interrupted_tool_calls", [])

        interrupted = await self._executor.interrupt(reason)
        if pause:
            self.pause("user_interrupt" if reason == "user" else reason)
        return interrupted

    # === Multi-agent coordination ===

    async def receive_signal(self, signal: str, payload: Any) -> None:
        """Receive external control signal.

        Args:
            signal: Signal type (e.g., "suspend", "resume", "reset")
            payload: Signal payload
        """
        if signal == "suspend":
            await self._executor.interrupt("signal")
            self.pause("signal_suspend")
        elif signal == "resume":
            self._resume_internal()
        elif signal == "reset":
            self.clear_all_queues()
            await self._executor.interrupt("reset")
            self._resume_internal()

    # === Convenience methods ===

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
