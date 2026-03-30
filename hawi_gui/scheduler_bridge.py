"""SchedulerThread — runs asyncio + HawiScheduler in a daemon thread.

Communicates with the UI via:
  ui_queue  (scheduler → UI)
  cmd_queue (UI → scheduler)
"""

from __future__ import annotations

import asyncio
import queue
import threading
from typing import TYPE_CHECKING, cast

from hawi.agent import HawiAgent, HawiScheduler
import hawi.events
from hawi.events import Event
from hawi.models import model_registry

from .protocol import (
    CmdClearContext,
    CmdClearQueue,
    CmdEnqueue,
    CmdStop,
    CmdSwitchModel,
    QueueKind,
    UiAgentInterrupt,
    UiDebugInfo,
    UiError,
    UiInterrupt,
    UiModelMetadata,
    UiModelRetry,
    UiReady,
    UiRunStart,
    UiRunStop,
    UiStatusUpdate,
    UiTextDelta,
    UiThinkingDelta,
    UiToolCall,
    UiToolCallDelta,
    UiToolResult,
)


class SchedulerThread(threading.Thread):
    """Daemon thread hosting an asyncio event loop with HawiScheduler."""

    def __init__(
        self,
        ui_queue: "queue.Queue",
        cmd_queue: "queue.Queue",
        model_name: str,
    ):
        super().__init__(daemon=True, name="SchedulerThread")
        self.ui_queue = ui_queue
        self.cmd_queue = cmd_queue
        self.model_name = model_name
        self.loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._scheduler: HawiScheduler | None = None

    # ─── Thread entry ─────────────────────────────────────────────────────────

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._main())
        except Exception as exc:
            self._send_ui(UiError(message=f"Scheduler thread crashed: {exc}"))
        finally:
            self.loop.close()

    def _send_ui(self, msg):
        """Thread-safe push to the UI queue."""
        self.ui_queue.put(msg)

    # ─── Async main ───────────────────────────────────────────────────────────

    async def _main(self):
        await self._init_scheduler(self.model_name)
        self._ready.set()

        self._run_forever_task: asyncio.Task | None = None
        cmd_task = asyncio.create_task(self._process_commands())
        status_task = asyncio.create_task(self._status_loop())

        await asyncio.gather(cmd_task, status_task, return_exceptions=True)

    async def _init_scheduler(self, model_name: str):
        """Create agent and scheduler, subscribe to events."""
        model = model_registry.create_model(model_name)

        from hawi_plugins.skills_plugin import SkillsPlugin
        from hawi_plugins.web import WebPlugin

        agent = HawiAgent(
            model=model,
            plugins=[SkillsPlugin(skills_dir=".skills"), WebPlugin()],
            system_prompt="You are a helpful AI assistant.",
            max_iterations=None,
            streaming=True,
        )
        self._scheduler = HawiScheduler(agent)

        # Event tracking state
        self._active_run_id: str | None = None
        self._active_tool_calls: dict[str, dict] = {}
        self._last_enqueued_kind: QueueKind = "normal"
        self._pending_run: dict[str, QueueKind] = {}
        self._tool_call_buffers: dict[str, str] = {}  # tool_call_id -> accumulated arguments

        agent.event_bus.subscribe(self._on_agent_event)
        self._scheduler.event_bus.subscribe(self._on_scheduler_event)

        # Start the scheduler loop as a task we can cancel later
        self._run_forever_task = asyncio.create_task(
            self._scheduler.run_forever(poll_interval=0.1)
        )

    async def _teardown_scheduler(self):
        """Stop current scheduler and clean up."""
        if self._scheduler:
            self._scheduler.stop()
            # Give run_forever a moment to exit cleanly
            if self._run_forever_task and not self._run_forever_task.done():
                self._run_forever_task.cancel()
                try:
                    await asyncio.wait_for(self._run_forever_task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

    # ─── Async loops ─────────────────────────────────────────────────────────

    async def _status_loop(self):
        while True:
            await asyncio.sleep(0.3)
            if self._scheduler is None:
                continue
            try:
                lengths = self._scheduler.get_queue_lengths()
                sched_state = self._scheduler.state.name
                exec_state = self._scheduler._executor.state.name
                self._send_ui(UiStatusUpdate(
                    scheduler_state=sched_state,
                    agent_state=exec_state,
                    queue_lengths=lengths,
                ))
            except Exception:
                pass

    async def _process_commands(self):
        loop = asyncio.get_event_loop()
        while True:
            try:
                cmd = await loop.run_in_executor(
                    None, lambda: self.cmd_queue.get(timeout=0.1)
                )
            except queue.Empty:
                continue

            if isinstance(cmd, CmdStop):
                await self._teardown_scheduler()
                break

            elif isinstance(cmd, CmdEnqueue):
                if self._scheduler:
                    msg_id = self._scheduler.enqueue(
                        cmd.content, cmd.queue,
                        metadata={"queue_kind": cmd.queue},
                    )
                    self._last_enqueued_kind = cmd.queue

            elif isinstance(cmd, CmdClearContext):
                if self._scheduler:
                    self._scheduler.agent.context.clear()

            elif isinstance(cmd, CmdClearQueue):
                if self._scheduler:
                    if cmd.queue == "all":
                        self._scheduler.clear_all_queues()
                    else:
                        self._scheduler.clear_queue(cmd.queue)

            elif isinstance(cmd, CmdSwitchModel):
                # Switch model without recreating scheduler - preserves context
                if self._scheduler:
                    try:
                        self._scheduler.agent.set_model(cmd.model_name)
                        self.model_name = cmd.model_name
                        self._send_ui(UiReady(model_name=cmd.model_name))
                    except Exception as exc:
                        self._send_ui(UiError(message=f"切换模型失败: {exc}"))

    # ─── Event handlers ───────────────────────────────────────────────────────

    def _on_agent_event(self, event: Event):
        etype = event.type

        if etype == "agent.run_start":
            event = cast(hawi.events.AgentRunStartEvent, event)
            self._active_run_id = event.run_id
            self._pending_run[event.run_id] = self._last_enqueued_kind

        elif etype == "agent.message_added":
            event = cast(hawi.events.AgentMessageAddedEvent, event)
            if event.role == "user":
                text = ""
                for part in event.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text += part.get("text", "")
                if text:
                    qk = self._pending_run.get(event.run_id, "normal")
                    self._send_ui(UiRunStart(
                        run_id=event.run_id,
                        user_content=text,
                        queue_kind=qk,
                    ))

        elif etype == "model.content_block_delta":
            event = cast(hawi.events.ModelContentBlockDeltaEvent, event)
            if not self._active_run_id:
                return
            if event.delta_type == "text" and event.delta:
                self._send_ui(UiTextDelta(delta=event.delta, run_id=self._active_run_id))
            elif event.delta_type == "reasoning" and event.delta:
                self._send_ui(UiThinkingDelta(delta=event.delta, run_id=self._active_run_id))

        elif etype == "model.metadata":
            event = cast(hawi.events.ModelMetadataEvent, event)
            run_id = self._active_run_id or ""
            usage = event.usage or {}
            self._send_ui(UiModelMetadata(
                run_id=run_id,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                latency_ms=event.latency_ms,
            ))

        elif etype == "model.retry":
            event = cast(hawi.events.ModelRetryEvent, event)
            run_id = self._active_run_id or ""
            self._send_ui(UiModelRetry(
                run_id=run_id,
                attempt=event.attempt,
                max_retries=event.max_retries,
                error_type=event.error_type,
                error_message=event.error_message,
            ))

        elif etype == "model.error":
            event = cast(hawi.events.ModelErrorEvent, event)
            self._send_ui(UiError(message=f"[Model Error] {event.error.error_type}: {event.error.message}"))

        elif etype == "agent.error":
            event = cast(hawi.events.AgentErrorEvent, event)
            self._send_ui(UiError(message=f"[Agent Error] {event.error.error_type}: {event.error.message}"))

        elif etype == "model.tool_call_block_delta":
            event = cast(hawi.events.ModelToolCallBlockDeltaEvent, event)
            run_id = self._active_run_id or ""
            tool_call_id = event.tool_call_id
            delta = event.arguments_delta
            # Accumulate and send
            if tool_call_id not in self._tool_call_buffers:
                self._tool_call_buffers[tool_call_id] = ""
            self._tool_call_buffers[tool_call_id] += delta
            self._send_ui(UiToolCallDelta(
                run_id=run_id,
                tool_call_id=tool_call_id,
                delta=delta,
            ))

        elif etype == "agent.interrupt":
            event = cast(hawi.events.AgentInterruptEvent, event)
            self._send_ui(UiAgentInterrupt(
                run_id=event.run_id,
                interrupt_type=event.interrupt_type,
            ))

        # Debug events
        elif etype == "model.stream_start":
            self._send_ui(UiDebugInfo(message=f"Model stream started"))

        elif etype == "model.stream_stop":
            event = cast(hawi.events.ModelStreamStopEvent, event)
            self._send_ui(UiDebugInfo(message=f"Model stream stopped: {event.stop_reason}"))

        elif etype == "model.content_block_start":
            event = cast(hawi.events.ModelContentBlockStartEvent, event)
            self._send_ui(UiDebugInfo(message=f"Content block start: {event.block_type}"))

        elif etype == "model.content_block_stop":
            event = cast(hawi.events.ModelContentBlockStopEvent, event)
            block_type = event.block_type if event.content else "unknown"
            self._send_ui(UiDebugInfo(message=f"Content block stop: {block_type}"))

        elif etype == "agent.run_stop":
            event = cast(hawi.events.AgentRunStopEvent, event)
            self._pending_run.pop(event.run_id, None)
            if self._active_run_id == event.run_id:
                self._active_run_id = None
            self._send_ui(UiRunStop(
                run_id=event.run_id,
                stop_reason=event.stop_reason,
                duration_ms=event.duration_ms,
            ))

        elif etype == "agent.tool_call":
            event = cast(hawi.events.AgentToolCallEvent, event)
            run_id = self._active_run_id or ""
            self._active_tool_calls[event.tool_call_id] = {
                "tool_name": event.tool_name,
                "arguments": event.arguments,
                "run_id": run_id,
            }
            self._send_ui(UiToolCall(
                tool_name=event.tool_name,
                tool_call_id=event.tool_call_id,
                arguments=event.arguments,
                run_id=run_id,
            ))

        elif etype == "agent.tool_result":
            event = cast(hawi.events.AgentToolResultEvent, event)
            call_info = self._active_tool_calls.pop(event.tool_call_id, {})
            tool_name = call_info.get("tool_name", event.tool_call_id)
            run_id = call_info.get("run_id", self._active_run_id or "")
            output = (event.result.output if event.result else None) or event.result_preview
            self._send_ui(UiToolResult(
                tool_call_id=event.tool_call_id,
                tool_name=tool_name,
                success=event.success,
                output=str(output) if output else "",
                duration_ms=event.duration_ms,
                run_id=run_id,
            ))

    def _on_scheduler_event(self, event: Event):
        if event.type == "scheduler.interrupt":
            event = cast(hawi.events.SchedulerInterruptEvent, event)
            self._send_ui(UiInterrupt(reason=event.reason))
        elif event.type == "scheduler.dequeue":
            event = cast(hawi.events.SchedulerDequeueEvent, event)
            qk = event.queue_type
            if qk in ("normal", "high_prio", "urgent"):
                self._last_enqueued_kind = qk
            self._send_ui(UiDebugInfo(message=f"Dequeue from {qk}"))
        elif event.type == "scheduler.enqueue":
            event = cast(hawi.events.SchedulerEnqueueEvent, event)
            self._send_ui(UiDebugInfo(message=f"Enqueue to {event.queue_type}: {event.content_preview[:30]}..."))

    # ─── External API ─────────────────────────────────────────────────────────

    def wait_ready(self, timeout: float = 10.0) -> bool:
        return self._ready.wait(timeout)
