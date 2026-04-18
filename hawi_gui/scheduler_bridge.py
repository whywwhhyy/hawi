"""SchedulerThread — runs asyncio + HawiScheduler in a daemon thread.

Communicates with the UI via:
  ui_queue  (scheduler -> UI)
  cmd_queue (UI -> scheduler)
"""

from __future__ import annotations

import asyncio
import queue
import threading
from typing import Any, cast

import hawi.events
from hawi.agent import HawiAgent, HawiScheduler
from hawi.agent.context import AgentContext, ToolCallContext
from hawi.events import Event
from hawi.models import model_registry

from .protocol import (
    CmdApplyPlugins,
    CmdClearContext,
    CmdClearQueue,
    CmdEnqueue,
    CmdInterrupt,
    CmdSetSystemPrompt,
    CmdStop,
    CmdSwitchModel,
    DEFAULT_SYSTEM_PROMPT,
    PluginConfigs,
    QueueKind,
    UiAgentInterrupt,
    UiDebugInfo,
    UiError,
    UiInterrupt,
    UiModelMetadata,
    UiModelRetry,
    UiPluginsApplied,
    UiReady,
    UiRunStart,
    UiRunStop,
    UiStatusUpdate,
    UiTextDelta,
    UiThinkingDelta,
    UiToolCallStart,
    UiToolCallDelta,
    UiToolCallStop,
    UiToolResult,
)


PLUGIN_FILESYSTEM = "filesystem"
PLUGIN_SHELL = "shell"
PLUGIN_WEB = "web"
PLUGIN_SKILLS = "skills"
PLUGIN_PYTHON_INTERPRETER = "python_interpreter"
PLUGIN_MCP = "mcp"


class SchedulerThread(threading.Thread):
    """Daemon thread hosting an asyncio event loop with HawiScheduler."""

    def __init__(
        self,
        ui_queue: "queue.Queue",
        cmd_queue: "queue.Queue",
        model_name: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        selected_plugins: list[str] | None = None,
        plugin_configs: PluginConfigs | None = None,
    ):
        super().__init__(daemon=True, name="SchedulerThread")
        self.ui_queue = ui_queue
        self.cmd_queue = cmd_queue
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._scheduler: HawiScheduler | None = None
        self._run_forever_task: asyncio.Task | None = None
        self._selected_plugins: list[str] = list(selected_plugins or [])
        self._plugin_configs: PluginConfigs = {
            name: dict(cfg) for name, cfg in (plugin_configs or {}).items()
        }

        # Event tracking state
        self._active_run_id: str | None = None
        self._active_tool_calls: dict[str, dict[str, Any]] = {}
        self._last_enqueued_kind: QueueKind = "normal"
        self._pending_run: dict[str, QueueKind] = {}
        self._tool_call_buffers: dict[str, str] = {}
        self._tool_call_id_by_block: dict[int, str] = {}

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

    def _send_ui(self, msg: Any):
        """Thread-safe push to the UI queue."""
        self.ui_queue.put(msg)

    # ─── Async main ───────────────────────────────────────────────────────────

    async def _main(self):
        init_ok = await self._replace_scheduler(
            model_name=self.model_name,
            selected_plugins=self._selected_plugins,
            plugin_configs=self._plugin_configs,
            preserve_context=None,
            emit_ready=True,
        )
        if not init_ok:
            self._send_ui(UiError(message="Scheduler initialization failed."))

        self._ready.set()
        cmd_task = asyncio.create_task(self._process_commands())
        status_task = asyncio.create_task(self._status_loop())
        await cmd_task
        status_task.cancel()
        await asyncio.gather(status_task, return_exceptions=True)

    # ─── Plugin setup ─────────────────────────────────────────────────────────

    async def _create_plugins(
        self,
        selected_plugins: list[str],
        plugin_configs: PluginConfigs,
    ) -> list[Any]:
        """Instantiate plugins from selected plugin keys and saved config."""
        plugins: list[Any] = []

        for plugin_key in selected_plugins:
            cfg = dict(plugin_configs.get(plugin_key, {}))

            if plugin_key == PLUGIN_FILESYSTEM:
                from hawi_plugins.filesystem_plugin import FileSystemPlugin

                plugins.append(FileSystemPlugin())
            elif plugin_key == PLUGIN_SHELL:
                from hawi_plugins.shell_plugin import ShellPlugin

                plugins.append(ShellPlugin())
            elif plugin_key == PLUGIN_WEB:
                from hawi_plugins.web import WebPlugin

                plugins.append(WebPlugin())
            elif plugin_key == PLUGIN_SKILLS:
                from hawi_plugins.skills_plugin import SkillsPlugin

                skills_dir = str(cfg.get("skills_dir") or ".skills")
                plugins.append(SkillsPlugin(skills_dir=skills_dir))
            elif plugin_key == PLUGIN_PYTHON_INTERPRETER:
                from hawi_plugins.python_interpreter import PythonInterpreterPlugin

                work_dir_raw = cfg.get("work_dir")
                work_dir = str(work_dir_raw).strip() if isinstance(work_dir_raw, str) else None
                if not work_dir:
                    work_dir = None
                print_execution = bool(cfg.get("print_execution", False))
                plugins.append(
                    PythonInterpreterPlugin(
                        work_dir=work_dir,
                        print_execution=print_execution,
                    )
                )
            elif plugin_key == PLUGIN_MCP:
                from hawi_plugins.mcp_plugin import MCPPlugin

                config_path = str(cfg.get("config_path") or "").strip()
                if not config_path:
                    raise ValueError("MCP plugin requires 'config_path'.")
                mcp = MCPPlugin(config_path=config_path)
                await mcp.connect()
                plugins.append(mcp)
            else:
                raise ValueError(f"Unknown plugin key: {plugin_key}")

        return plugins

    async def _build_scheduler(
        self,
        model_name: str,
        selected_plugins: list[str],
        plugin_configs: PluginConfigs,
        context_to_restore: AgentContext | None,
    ) -> tuple[HawiScheduler, asyncio.Task]:
        model = model_registry.create_model(model_name)
        plugins = await self._create_plugins(selected_plugins, plugin_configs)
        agent = HawiAgent(
            model=model,
            plugins=plugins,
            system_prompt=self.system_prompt,
            max_iterations=None,
            streaming=True,
        )
        if context_to_restore is not None:
            agent.set_context(context_to_restore.copy())
            agent.context.tool_call_context = ToolCallContext(agent)

        scheduler = HawiScheduler(agent)
        agent.event_bus.subscribe(self._on_agent_event)
        scheduler.event_bus.subscribe(self._on_scheduler_event)
        run_task = asyncio.create_task(scheduler.run_forever(poll_interval=0.1))
        return scheduler, run_task

    async def _stop_scheduler(
        self,
        scheduler: HawiScheduler | None,
        run_task: asyncio.Task | None,
    ) -> None:
        if scheduler is None:
            return

        scheduler.agent.event_bus.unsubscribe(self._on_agent_event)
        scheduler.event_bus.unsubscribe(self._on_scheduler_event)
        scheduler.stop()
        if run_task and not run_task.done():
            run_task.cancel()
            try:
                await asyncio.wait_for(run_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

    async def _replace_scheduler(
        self,
        *,
        model_name: str,
        selected_plugins: list[str],
        plugin_configs: PluginConfigs,
        preserve_context: AgentContext | None,
        emit_ready: bool,
    ) -> bool:
        """Replace scheduler atomically (all-or-nothing)."""
        old_scheduler = self._scheduler
        old_task = self._run_forever_task

        try:
            new_scheduler, new_task = await self._build_scheduler(
                model_name=model_name,
                selected_plugins=selected_plugins,
                plugin_configs=plugin_configs,
                context_to_restore=preserve_context,
            )
        except Exception as exc:
            self._send_ui(UiError(message=f"Failed to apply scheduler update: {exc}"))
            return False

        await self._stop_scheduler(old_scheduler, old_task)
        self._scheduler = new_scheduler
        self._run_forever_task = new_task
        self.model_name = model_name
        self._selected_plugins = list(selected_plugins)
        self._plugin_configs = {name: dict(cfg) for name, cfg in plugin_configs.items()}
        if self._scheduler is not None and self._scheduler.agent.context.system_prompt:
            text_parts = [
                str(part.get("text", ""))
                for part in self._scheduler.agent.context.system_prompt
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            if text_parts:
                self.system_prompt = "\n".join(text_parts)

        if emit_ready:
            self._send_ui(
                UiReady(
                    model_name=model_name,
                    selected_plugins=list(self._selected_plugins),
                    plugin_configs={k: dict(v) for k, v in self._plugin_configs.items()},
                )
            )
        return True

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
                self._send_ui(
                    UiStatusUpdate(
                        scheduler_state=sched_state,
                        agent_state=exec_state,
                        queue_lengths=lengths,
                    )
                )
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
                await self._stop_scheduler(self._scheduler, self._run_forever_task)
                self._scheduler = None
                self._run_forever_task = None
                break

            elif isinstance(cmd, CmdEnqueue):
                if self._scheduler:
                    self._scheduler.enqueue(
                        cmd.content,
                        cmd.queue,
                        metadata={"queue_kind": cmd.queue},
                    )
                    self._last_enqueued_kind = cmd.queue

            elif isinstance(cmd, CmdInterrupt):
                if self._scheduler:
                    try:
                        await self._scheduler.interrupt(cmd.reason)
                    except Exception as exc:
                        self._send_ui(UiError(message=f"中断失败: {exc}"))

            elif isinstance(cmd, CmdClearContext):
                if self._scheduler:
                    self._scheduler.agent.context.clear()

            elif isinstance(cmd, CmdSetSystemPrompt):
                self.system_prompt = cmd.system_prompt
                if self._scheduler:
                    self._scheduler.agent.context.set_system_prompt(cmd.system_prompt)

            elif isinstance(cmd, CmdClearQueue):
                if self._scheduler:
                    if cmd.queue == "all":
                        self._scheduler.clear_all_queues()
                    else:
                        self._scheduler.clear_queue(cmd.queue)

            elif isinstance(cmd, CmdSwitchModel):
                if self._scheduler:
                    try:
                        self._scheduler.agent.set_model(cmd.model_name)
                        self.model_name = cmd.model_name
                        self._send_ui(
                            UiReady(
                                model_name=cmd.model_name,
                                selected_plugins=list(self._selected_plugins),
                                plugin_configs={k: dict(v) for k, v in self._plugin_configs.items()},
                            )
                        )
                    except Exception as exc:
                        self._send_ui(UiError(message=f"切换模型失败: {exc}"))

            elif isinstance(cmd, CmdApplyPlugins):
                if self._scheduler is None:
                    continue

                if not self._scheduler._executor.is_idle:
                    self._send_ui(
                        UiPluginsApplied(
                            success=False,
                            message="Agent is running. Please wait until idle before applying plugins.",
                            selected_plugins=list(self._selected_plugins),
                            plugin_configs={k: dict(v) for k, v in self._plugin_configs.items()},
                        )
                    )
                    continue

                context_copy = self._scheduler.agent.context.copy()
                success = await self._replace_scheduler(
                    model_name=self.model_name,
                    selected_plugins=list(cmd.selected_plugins),
                    plugin_configs={k: dict(v) for k, v in cmd.plugin_configs.items()},
                    preserve_context=context_copy,
                    emit_ready=True,
                )
                if success:
                    self._send_ui(
                        UiPluginsApplied(
                            success=True,
                            message="Plugins applied successfully.",
                            selected_plugins=list(self._selected_plugins),
                            plugin_configs={k: dict(v) for k, v in self._plugin_configs.items()},
                        )
                    )
                else:
                    self._send_ui(
                        UiPluginsApplied(
                            success=False,
                            message="Failed to apply plugin configuration.",
                            selected_plugins=list(self._selected_plugins),
                            plugin_configs={k: dict(v) for k, v in self._plugin_configs.items()},
                        )
                    )

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
                    self._send_ui(
                        UiRunStart(
                            run_id=event.run_id,
                            user_content=text,
                            queue_kind=qk,
                        )
                    )

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
            self._send_ui(
                UiModelMetadata(
                    run_id=run_id,
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                    latency_ms=event.latency_ms,
                )
            )

        elif etype == "model.retry":
            event = cast(hawi.events.ModelRetryEvent, event)
            run_id = self._active_run_id or ""
            self._send_ui(
                UiModelRetry(
                    run_id=run_id,
                    attempt=event.attempt,
                    max_retries=event.max_retries,
                    error_type=event.error_type,
                    error_message=event.error_message,
                )
            )

        elif etype == "model.error":
            event = cast(hawi.events.ModelErrorEvent, event)
            self._send_ui(UiError(message=f"[Model Error] {event.error.error_type}: {event.error.message}"))

        elif etype == "agent.error":
            event = cast(hawi.events.AgentErrorEvent, event)
            self._send_ui(UiError(message=f"[Agent Error] {event.error.error_type}: {event.error.message}"))

        elif etype == "model.tool_call_block_delta":
            event = cast(hawi.events.ModelToolCallBlockDeltaEvent, event)
            run_id = self._active_run_id or ""
            tool_call_id = event.tool_call_id or self._tool_call_id_by_block.get(event.block_index, "")
            if not tool_call_id:
                return
            delta = event.arguments_delta
            if tool_call_id not in self._tool_call_buffers:
                self._tool_call_buffers[tool_call_id] = ""
            self._tool_call_buffers[tool_call_id] += delta
            self._send_ui(
                UiToolCallDelta(
                    run_id=run_id,
                    tool_call_id=tool_call_id,
                    delta=delta,
                )
            )

        elif etype == "model.tool_call_block_start":
            event = cast(hawi.events.ModelToolCallBlockStartEvent, event)
            run_id = self._active_run_id or ""
            if event.tool_call_id:
                self._tool_call_id_by_block[event.block_index] = event.tool_call_id
            self._active_tool_calls[event.tool_call_id] = {
                "tool_name": event.tool_name,
                "arguments": {},
                "run_id": run_id,
            }
            self._send_ui(
                UiToolCallStart(
                    tool_name=event.tool_name,
                    tool_call_id=event.tool_call_id,
                    run_id=run_id,
                )
            )

        elif etype == "model.tool_call_block_stop":
            event = cast(hawi.events.ModelToolCallBlockStopEvent, event)
            run_id = self._active_run_id or ""
            tool_call_id = event.tool_call_id or self._tool_call_id_by_block.get(event.block_index, "")
            if not tool_call_id:
                return
            if tool_call_id:
                self._tool_call_id_by_block[event.block_index] = tool_call_id
            self._active_tool_calls.setdefault(
                tool_call_id,
                {
                    "tool_name": event.tool_name,
                    "arguments": event.arguments,
                    "run_id": run_id,
                },
            )
            self._active_tool_calls[tool_call_id].update(
                {
                    "tool_name": event.tool_name,
                    "arguments": event.arguments,
                    "run_id": run_id,
                }
            )
            self._send_ui(
                UiToolCallStop(
                    run_id=run_id,
                    tool_call_id=tool_call_id,
                    tool_name=event.tool_name,
                    arguments=event.arguments,
                )
            )

        elif etype == "agent.interrupt":
            event = cast(hawi.events.AgentInterruptEvent, event)
            self._send_ui(
                UiAgentInterrupt(
                    run_id=event.run_id,
                    interrupt_type=event.interrupt_type,
                )
            )

        elif etype == "model.stream_start":
            self._send_ui(UiDebugInfo(message="Model stream started"))

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
            self._tool_call_buffers.clear()
            self._tool_call_id_by_block.clear()
            self._send_ui(
                UiRunStop(
                    run_id=event.run_id,
                    stop_reason=event.stop_reason,
                    duration_ms=event.duration_ms,
                )
            )

        elif etype == "agent.tool_call":
            event = cast(hawi.events.AgentToolCallEvent, event)
            run_id = self._active_run_id or ""
            # Fallback: in rare cases without model.tool_call_block_start,
            # keep tool-call metadata so tool_result still resolves correctly.
            self._active_tool_calls.setdefault(
                event.tool_call_id,
                {
                    "tool_name": event.tool_name,
                    "arguments": event.arguments,
                    "run_id": run_id,
                },
            )

        elif etype == "agent.tool_result":
            event = cast(hawi.events.AgentToolResultEvent, event)
            call_info = self._active_tool_calls.pop(event.tool_call_id, {})
            tool_name = str(call_info.get("tool_name", event.tool_call_id))
            run_id = str(call_info.get("run_id", self._active_run_id or ""))
            output = (event.result.output if event.result else None) or event.result_preview
            self._send_ui(
                UiToolResult(
                    tool_call_id=event.tool_call_id,
                    tool_name=tool_name,
                    success=event.success,
                    output=str(output) if output else "",
                    duration_ms=event.duration_ms,
                    run_id=run_id,
                )
            )

    def _on_scheduler_event(self, event: Event):
        if event.type == "scheduler.interrupt":
            event = cast(hawi.events.SchedulerInterruptEvent, event)
            self._send_ui(UiInterrupt(reason=event.reason))
        elif event.type == "scheduler.dequeue":
            event = cast(hawi.events.SchedulerDequeueEvent, event)
            qk = event.queue_type
            if qk in ("normal", "high_prio", "urgent"):
                self._last_enqueued_kind = cast(QueueKind, qk)
            self._send_ui(UiDebugInfo(message=f"Dequeue from {qk}"))
        elif event.type == "scheduler.enqueue":
            event = cast(hawi.events.SchedulerEnqueueEvent, event)
            self._send_ui(
                UiDebugInfo(
                    message=f"Enqueue to {event.queue_type}: {event.content_preview[:30]}..."
                )
            )

    # ─── External API ─────────────────────────────────────────────────────────

    def wait_ready(self, timeout: float = 10.0) -> bool:
        return self._ready.wait(timeout)
