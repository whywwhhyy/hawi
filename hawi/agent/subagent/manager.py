"""Sub-agent lifecycle manager."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

from hawi.events import Event, EventBus, SubAgentEvent, SubAgentEventType
from hawi.models import ContentPart, Message
from hawi.session import layout as session_layout
from hawi.session.markdown_export import (
    MarkdownExport,
    export_message_history_to_markdown,
    write_markdown_export_bundle,
)
from hawi.session.message_history import message_history_entry_from_event

from ..context import ToolCallContext
from ..result import AgentRunResult
from .prompts import (
    ROLE_SYSTEM_PROMPTS,
    SUBAGENT_IDENTITY_PROMPT,
    SUBAGENT_SHARED_CONTEXT_TASK_PROMPT_TEMPLATE,
    SUBAGENT_TASK_PROMPT_TEMPLATE,
)
from .types import (
    SubAgentError,
    SubAgentHandle,
    SubAgentLifecycleState,
    SubAgentLimits,
    SubAgentPluginPolicy,
    SubAgentQueue,
    SubAgentSpec,
    SubAgentStatus,
    SubAgentTimeoutAction,
)
from .utils import (
    drop_trailing_unanswered_tool_call_turn,
    event_summary,
    normalize_system_prompt,
)

if TYPE_CHECKING:
    from ..agent import HawiAgent

logger = logging.getLogger(__name__)


class SubAgentManager:
    """Create, drive, inspect, and close sub-agents for one parent agent."""

    def __init__(
        self,
        parent: HawiAgent,
        *,
        max_children: int = 8,
        poll_interval: float = 0.05,
    ) -> None:
        self._parent = parent
        self._max_children = max_children
        self._poll_interval = poll_interval
        self._handles: dict[str, SubAgentHandle] = {}
        self._lock = asyncio.Lock()
        self._session_root: Path | None = None
        self._session_id_provider: Callable[[], str | None] | None = None

    def configure_session_storage(
        self,
        *,
        root: Path | str,
        session_id_provider: Callable[[], str | None],
    ) -> None:
        """Bind sub-agent histories to the parent SessionManager layout."""
        self._session_root = Path(root).expanduser()
        self._session_id_provider = session_id_provider

    def list(self) -> list[SubAgentStatus]:
        """Return status snapshots for all known sub-agents."""
        return [self.status(subagent_id) for subagent_id in self._handles]

    async def spawn(
        self,
        spec: SubAgentSpec | None = None,
        **kwargs: Any,
    ) -> SubAgentHandle:
        """Create a sub-agent, start its runner, and optionally enqueue work."""
        spec = self._coerce_spec(spec, kwargs)
        async with self._lock:
            self._validate_spawn(spec)

            subagent_id = f"sub_{uuid.uuid4().hex[:8]}"
            spec.name = spec.name or f"{spec.role}-{subagent_id[-4:]}"
            child_event_bus = EventBus()
            child_agent = self._create_child_agent(spec, child_event_bus)

            from ..runner import AgentRunner

            runner = AgentRunner(child_agent)
            runner_task = asyncio.create_task(
                runner.run_forever(poll_interval=self._poll_interval),
                name=f"hawi-subagent-{subagent_id}",
            )

            handle = SubAgentHandle(
                id=subagent_id,
                spec=spec,
                agent=child_agent,
                runner=runner,
                runner_task=runner_task,
                event_bus=child_event_bus,
                state=SubAgentLifecycleState.IDLE,
                parent_session_id=self._current_parent_session_id(),
            )
            handle.event_handler = self._make_event_handler(handle)
            child_event_bus.subscribe(handle.event_handler)
            self._handles[subagent_id] = handle

            if spec.limits.max_runtime_seconds is not None:
                handle.monitor_task = asyncio.create_task(
                    self._enforce_runtime_limit(
                        subagent_id,
                        spec.limits.max_runtime_seconds,
                    ),
                    name=f"hawi-subagent-timeout-{subagent_id}",
                )

            initial = self._initial_message(spec)
            if initial is not None:
                self.send(subagent_id, initial, metadata={"source": "subagent.spawn"})
                handle.state = SubAgentLifecycleState.RUNNING
                handle.updated_at = time.time()

            await self._emit_manager_event(
                handle,
                "subagent.created",
                {"status": self.status(subagent_id).to_dict()},
            )
            return handle

    def send(
        self,
        subagent_id: str,
        message: str | list[ContentPart],
        queue: SubAgentQueue = "normal",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Send a message to a running sub-agent runner."""
        handle = self._require_handle(subagent_id)
        if handle.state == SubAgentLifecycleState.CLOSED:
            raise SubAgentError(f"Sub-agent is closed: {subagent_id}")

        message_id = handle.runner.enqueue(
            message,
            queue=queue,
            metadata={
                "subagent_id": subagent_id,
                "subagent_role": handle.role,
                "working_dir": handle.spec.working_dir,
                **(metadata or {}),
            },
        )
        if handle.state in {
            SubAgentLifecycleState.IDLE,
            SubAgentLifecycleState.COMPLETED,
        }:
            handle.state = SubAgentLifecycleState.RUNNING
        handle.updated_at = time.time()
        return message_id

    def status(self, subagent_id: str) -> SubAgentStatus:
        """Return a serializable status snapshot."""
        handle = self._require_handle(subagent_id)
        result = self._latest_result(handle)
        runner_state = handle.runner.state.name
        executor_state = handle.runner.executor_state.name
        queue_lengths = handle.runner.get_queue_lengths()
        state = self._derive_state(handle, queue_lengths, executor_state)
        model_id = getattr(handle.agent.model, "model_id", None)
        return SubAgentStatus(
            id=handle.id,
            name=handle.name,
            role=handle.role,
            state=state.value,
            runner_state=runner_state,
            executor_state=executor_state,
            queue_lengths=queue_lengths,
            created_at=handle.created_at,
            updated_at=handle.updated_at,
            closed_at=handle.closed_at,
            model_id=str(model_id) if model_id is not None else None,
            working_dir=handle.spec.working_dir,
            mode=handle.spec.mode,
            shared_context=handle.spec.mode == "fork",
            last_result_text=result.text if result is not None else None,
            last_error=handle.last_error,
        )

    async def wait(
        self,
        subagent_id: str,
        timeout: float | None = None,
        *,
        raise_on_error: bool = False,
    ) -> AgentRunResult | None:
        """Wait until the sub-agent has no queued or active work."""
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be greater than or equal to 0")
        handle = self._require_handle(subagent_id)
        deadline = time.monotonic() + timeout if timeout is not None else None

        while True:
            self._latest_result(handle)

            queue_lengths = handle.runner.get_queue_lengths()
            if self._is_settled(handle, queue_lengths):
                if handle.last_error and raise_on_error:
                    raise SubAgentError(handle.last_error)
                return handle.last_result

            if handle.last_error and raise_on_error:
                raise SubAgentError(handle.last_error)

            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for sub-agent: {subagent_id}")
            sleep_for = self._poll_interval
            if deadline is not None:
                sleep_for = max(0.0, min(sleep_for, deadline - time.monotonic()))
            await asyncio.sleep(sleep_for)

    async def wait_report(
        self,
        subagent_id: str,
        timeout: float | None = None,
        *,
        timeout_action: SubAgentTimeoutAction = "status",
        raise_on_error: bool = False,
    ) -> dict[str, Any]:
        """Wait like a shell job and return a structured status report.

        Unlike :meth:`wait`, timeout is reported as data by default so agent
        tools can tell the model to check again instead of surfacing a tool
        error.
        """
        timed_out = False
        result: AgentRunResult | None = None
        try:
            result = await self.wait(
                subagent_id,
                timeout=timeout,
                raise_on_error=raise_on_error,
            )
        except TimeoutError:
            timed_out = True
            if timeout_action == "raise":
                raise
            if timeout_action == "interrupt":
                await self.interrupt(subagent_id, reason="wait_timeout")
            elif timeout_action == "close":
                await self.close(subagent_id, reason="wait_timeout", interrupt=True)

        status = self.status(subagent_id).to_dict()
        result_text = result.text if result is not None else status.get("last_result_text")
        report = {
            "subagent_id": subagent_id,
            "timed_out": timed_out,
            "timeout_action": timeout_action,
            "status": status,
            "result_text": result_text,
            "last_error": status.get("last_error"),
        }
        handle = self._require_handle(subagent_id)
        if not timed_out and handle.message_history:
            export = self.export_markdown(subagent_id)
            report["export"] = self._export_query_payload(export)
            report["query_hint"] = (
                "Use read_subagent(view='markdown') for the readable report, "
                "read_subagent(view='export') for paths, or read_subagent(view='ref', "
                "ref_path='<filename>') for folded tool arguments/results."
            )
        if timed_out and timeout_action == "status":
            report["next_action"] = (
                "Sub-agent is still running. Call wait_subagent with the same "
                "subagent_id and a positive notify_timeout to wait again, or "
                "read_subagent for a non-blocking status snapshot."
            )
        return report

    def recent_events(self, subagent_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent child event summaries."""
        handle = self._require_handle(subagent_id)
        if limit <= 0:
            return []
        return handle.recent_events[-limit:]

    async def interrupt(self, subagent_id: str, reason: str = "parent") -> list[str]:
        """Interrupt the sub-agent's current run."""
        handle = self._require_handle(subagent_id)
        interrupted = await handle.runner.interrupt(reason)
        handle.state = SubAgentLifecycleState.INTERRUPTING
        handle.updated_at = time.time()
        return interrupted

    async def close(
        self,
        subagent_id: str,
        *,
        reason: str = "closed",
        interrupt: bool = True,
    ) -> SubAgentStatus:
        """Stop runner tasks and close a sub-agent."""
        handle = self._require_handle(subagent_id)
        if handle.state == SubAgentLifecycleState.CLOSED:
            return self.status(subagent_id)

        if interrupt:
            with contextlib.suppress(asyncio.CancelledError):
                await handle.runner.interrupt(reason)

        handle.runner.stop()
        if handle.monitor_task and not handle.monitor_task.done():
            handle.monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await handle.monitor_task
        if not handle.runner_task.done():
            handle.runner_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await handle.runner_task

        if handle.event_handler is not None:
            handle.event_bus.unsubscribe(handle.event_handler)
        handle.event_bus.close(wait=True, timeout=2.0)
        handle.state = SubAgentLifecycleState.CLOSED
        handle.closed_at = time.time()
        handle.updated_at = handle.closed_at
        await self._emit_manager_event(
            handle,
            "subagent.closed",
            {"reason": reason, "status": self.status(subagent_id).to_dict()},
        )
        return self.status(subagent_id)

    async def close_all(self, *, reason: str = "closed") -> list[SubAgentStatus]:
        """Close all managed sub-agents."""
        statuses: list[SubAgentStatus] = []
        for subagent_id in list(self._handles):
            statuses.append(await self.close(subagent_id, reason=reason))
        return statuses

    async def run_subagent(
        self,
        spec: SubAgentSpec | None = None,
        *,
        timeout: float | None = None,
        close: bool = True,
        **kwargs: Any,
    ) -> AgentRunResult | None:
        """Convenience API: spawn, wait, and optionally close."""
        handle = await self.spawn(spec, **kwargs)
        try:
            return await self.wait(handle.id, timeout=timeout)
        finally:
            if close:
                await self.close(handle.id, reason="run_subagent_complete")

    def read(
        self,
        subagent_id: str,
        *,
        view: Literal[
            "status",
            "summary",
            "events",
            "context_tail",
            "markdown",
            "export",
            "ref",
        ] = "summary",
        limit: int = 20,
        ref_path: str | None = None,
    ) -> dict[str, Any]:
        """Read a controlled view of sub-agent state."""
        handle = self._require_handle(subagent_id)
        status = self.status(subagent_id).to_dict()
        if view == "status":
            return {"status": status}
        if view == "events":
            return {"status": status, "events": self.recent_events(subagent_id, limit)}
        if view == "context_tail":
            messages = deepcopy(handle.agent.context.messages)
            partial = self._partial_assistant_message(handle)
            if partial is not None:
                messages.append(partial)
            return {
                "status": status,
                "messages": messages[-limit:] if limit > 0 else [],
            }
        if view == "markdown":
            export = self.export_markdown(subagent_id)
            return {
                "status": status,
                "markdown": export.markdown,
                "export": self._export_query_payload(export, include_references=True),
            }
        if view == "export":
            export = self.export_markdown(subagent_id)
            return {
                "status": status,
                "export": self._export_query_payload(export, include_references=True),
            }
        if view == "ref":
            return {
                "status": status,
                "reference": self._read_export_reference(handle, ref_path),
            }
        return {
            "status": status,
            "recent_events": self.recent_events(subagent_id, min(limit, 10)),
        }

    def export_markdown(self, subagent_id: str) -> MarkdownExport:
        """Create or refresh the session-internal Markdown export for a child."""
        handle = self._require_handle(subagent_id)
        history_path = self._subagent_history_path(handle)
        export = export_message_history_to_markdown(
            handle.message_history,
            kind="subagent",
            subject_id=handle.id,
            title=f"SubAgent {handle.name}",
            model=str(getattr(handle.agent.model, "model_id", "")) or None,
            system_prompt=handle.agent.context.get_system_prompt(),
            metadata={
                "name": handle.name,
                "role": handle.role,
                "state": self.status(handle.id).state,
                "parent_session_id": handle.parent_session_id,
                "working_dir": handle.spec.working_dir,
                "result_contract": handle.spec.result_contract,
                "message_count": len(handle.message_history),
            },
            raw_history_path=str(history_path) if history_path is not None else None,
        )
        export_dir = self._subagent_export_dir(handle, export.export_id)
        if export_dir is not None:
            export = write_markdown_export_bundle(
                export,
                export_dir=export_dir,
                source_jsonl_path=history_path,
                message_history=handle.message_history,
            )
        handle.last_export = self._export_query_payload(
            export,
            include_references=True,
        )
        return export

    def _coerce_spec(
        self,
        spec: SubAgentSpec | None,
        kwargs: dict[str, Any],
    ) -> SubAgentSpec:
        if spec is None:
            spec = SubAgentSpec()
        if kwargs:
            policy = kwargs.pop("plugin_policy", None)
            if policy is not None:
                kwargs["plugin_policy"] = policy
            spec = replace(spec, **kwargs)
        if spec.mode not in {"fork", "fresh"}:
            raise ValueError("Sub-agent mode must be 'fork' or 'fresh'")
        if not isinstance(spec.plugin_policy, SubAgentPluginPolicy):
            raise TypeError("plugin_policy must be a SubAgentPluginPolicy")
        if not isinstance(spec.limits, SubAgentLimits):
            raise TypeError("limits must be a SubAgentLimits")
        return spec

    def _validate_spawn(self, spec: SubAgentSpec) -> None:
        active = [
            handle
            for handle in self._handles.values()
            if handle.state != SubAgentLifecycleState.CLOSED
        ]
        max_children = spec.limits.max_children or self._max_children
        if len(active) >= max_children:
            raise SubAgentError(f"Sub-agent limit reached: {max_children}")
        if spec.limits.max_recursion_depth < 0:
            raise ValueError("max_recursion_depth must be >= 0")

    def _create_child_agent(
        self,
        spec: SubAgentSpec,
        event_bus: EventBus,
    ) -> HawiAgent:
        policy = spec.plugin_policy
        if spec.mode == "fork":
            child = (
                self._parent.clone()
                if policy.inherit
                else self._new_agent_without_inherited_plugins(spec, event_bus)
            )
            if not policy.inherit:
                child.set_context(self._parent.context.copy())
            drop_trailing_unanswered_tool_call_turn(child.context.messages)
        else:
            child = (
                self._parent.clone()
                if policy.inherit
                else self._new_agent_without_inherited_plugins(spec, event_bus)
            )
            child.context.clear()
            child.set_system_prompt(None)

        self._rebind_agent_event_bus(child, event_bus)
        self._apply_plugin_policy(child, policy)

        # --- Apply permission set ---
        # If spec explicitly provides a permission_set, use it; otherwise
        # inherit the parent's permission set (already cloned).
        if spec.permission_set is not None:
            child.set_permissions(spec.permission_set)

        if spec.model is not None:
            child.set_model(spec.model)

        system_prompt = self._system_prompt_for_spec(spec, child)
        if system_prompt is not None:
            child.set_system_prompt(system_prompt)

        if spec.limits.max_iterations is not None:
            child._max_iterations = spec.limits.max_iterations

        return child

    def _new_agent_without_inherited_plugins(
        self,
        spec: SubAgentSpec,
        event_bus: EventBus,
    ) -> HawiAgent:
        from ..agent import HawiAgent

        return HawiAgent(
            model=spec.model or self._parent.model,
            plugins=[],
            plugin_factories=[],
            system_prompt=None,
            max_iterations=spec.limits.max_iterations or self._parent._max_iterations,
            model_error_policy=self._parent._model_error_policy,
            event_bus=event_bus,
            streaming=self._parent._streaming,
            auto_compact=self._parent._auto_compact,
        )

    def _rebind_agent_event_bus(self, agent: HawiAgent, event_bus: EventBus) -> None:
        agent._event_bus = event_bus
        agent._plugin_manager.bind_event_bus(event_bus)
        agent.context.tool_call_context = ToolCallContext(agent=agent)

    def _apply_plugin_policy(
        self,
        child: HawiAgent,
        policy: SubAgentPluginPolicy,
    ) -> None:
        for plugin in policy.extra_plugins:
            child.plugins.add_plugin(plugin)
        for factory in policy.extra_factories:
            child.plugins.add_plugin_factory(factory)
        defs = child.plugins.get_tool_definitions()
        child.context.tool_definitions = defs if defs else None

    def _system_prompt_for_spec(
        self,
        spec: SubAgentSpec,
        child: HawiAgent,
    ) -> list[ContentPart] | None:
        if spec.system_prompt is not None:
            return normalize_system_prompt(spec.system_prompt)

        role_prompt = ROLE_SYSTEM_PROMPTS.get(
            str(spec.role),
            ROLE_SYSTEM_PROMPTS["general"],
        )
        base = deepcopy(child.context.get_system_prompt() or [])
        base.append({"type": "text", "text": SUBAGENT_IDENTITY_PROMPT})
        base.append({"type": "text", "text": role_prompt})
        if spec.working_dir:
            base.append({
                "type": "text",
                "text": f"Logical working directory: {spec.working_dir}",
            })
        if spec.ownership:
            base.append({
                "type": "text",
                "text": "Declared ownership:\n" + json.dumps(
                    spec.ownership,
                    ensure_ascii=False,
                    indent=2,
                ),
            })
        return base or None

    def _initial_message(
        self,
        spec: SubAgentSpec,
    ) -> str | list[ContentPart] | None:
        if spec.initial_prompt is None and spec.initial_plan is None:
            return None

        if spec.initial_plan is None:
            return self._subagent_task_message(spec, spec.initial_prompt)

        if isinstance(spec.initial_plan, str):
            plan_text = spec.initial_plan
        else:
            plan_text = json.dumps(spec.initial_plan, ensure_ascii=False, indent=2)

        if spec.initial_prompt is None:
            return self._subagent_task_message(
                spec,
                f"Execute the following initial plan:\n\n{plan_text}"
            )
        if isinstance(spec.initial_prompt, str):
            return self._subagent_task_message(
                spec,
                f"{spec.initial_prompt}\n\nInitial plan:\n{plan_text}"
            )
        content = deepcopy(spec.initial_prompt)
        content.append({"type": "text", "text": f"Initial plan:\n{plan_text}"})
        return self._subagent_task_message(spec, content)

    def _subagent_task_message(
        self,
        spec: SubAgentSpec,
        task: str | list[ContentPart] | None,
    ) -> str | list[ContentPart] | None:
        if task is None:
            return None
        template = (
            SUBAGENT_SHARED_CONTEXT_TASK_PROMPT_TEMPLATE
            if spec.mode == "fork"
            else SUBAGENT_TASK_PROMPT_TEMPLATE
        )
        if isinstance(task, str):
            return template.format(task=task.strip())
        content = [
            {
                "type": "text",
                "text": template.format(
                    task="See the following content parts."
                ),
            }
        ]
        content.extend(deepcopy(task))
        return content

    def _make_event_handler(
        self,
        handle: SubAgentHandle,
    ) -> Callable[[Event], Any]:
        async def on_child_event(event: Event) -> None:
            summary = event_summary(event)
            handle.recent_events.append(summary)
            if len(handle.recent_events) > 200:
                del handle.recent_events[:-200]
            handle.updated_at = time.time()
            entry = message_history_entry_from_event(event)
            if entry is not None:
                handle.message_history.append(entry)
                self._append_subagent_history(handle, [entry])

            if event.type == "agent.run_start":
                handle.state = SubAgentLifecycleState.RUNNING
                self._clear_partial_assistant(handle)
            elif event.type == "model.stream_start":
                self._clear_partial_assistant(handle)
            elif event.type == "model.content_block_delta":
                self._append_partial_assistant_delta(handle, summary)
            elif event.type == "agent.run_stop":
                handle.state = SubAgentLifecycleState.COMPLETED
                self._clear_partial_assistant(handle)
            elif event.type == "agent.error":
                handle.state = SubAgentLifecycleState.FAILED
                handle.last_error = summary.get("error") or "Sub-agent failed"
                self._clear_partial_assistant(handle)
            elif event.type == "runner.interrupt":
                handle.state = SubAgentLifecycleState.INTERRUPTING

            await self._emit_manager_event(
                handle,
                "subagent.event",
                {
                    "child_event": summary,
                    **({"message_entry": entry} if entry is not None else {}),
                    "status": self.status(handle.id).to_dict(),
                },
            )

        return on_child_event

    async def _emit_manager_event(
        self,
        handle: SubAgentHandle,
        event_name: str,
        payload: dict[str, Any],
    ) -> None:
        await self._parent._emit_event(
            SubAgentEvent.create(
                cast(SubAgentEventType, event_name),
                subagent_id=handle.id,
                subagent_name=handle.name,
                subagent_role=handle.role,
                status=payload.get("status") if isinstance(payload.get("status"), dict) else {},
                child_event=(
                    payload.get("child_event")
                    if isinstance(payload.get("child_event"), dict)
                    else None
                ),
                message_entry=(
                    payload.get("message_entry")
                    if isinstance(payload.get("message_entry"), dict)
                    else None
                ),
                reason=payload.get("reason") if isinstance(payload.get("reason"), str) else None,
            ),
            None,
        )

    def _latest_result(self, handle: SubAgentHandle) -> AgentRunResult | None:
        result = handle.runner.last_result
        if result is not None:
            handle.last_result = result
            return result
        return handle.last_result

    def _current_parent_session_id(self) -> str | None:
        if self._session_id_provider is None:
            return None
        try:
            return self._session_id_provider()
        except Exception:
            logger.debug("subagent session id provider failed", exc_info=True)
            return None

    def _parent_session_dir(self, handle: SubAgentHandle) -> Path | None:
        if self._session_root is None or not handle.parent_session_id:
            return None
        return session_layout.session_dir(self._session_root, handle.parent_session_id)

    def _subagent_dir(self, handle: SubAgentHandle) -> Path | None:
        parent_dir = self._parent_session_dir(handle)
        if parent_dir is None:
            return None
        return session_layout.subagent_dir(parent_dir, handle.id)

    def _subagent_history_path(self, handle: SubAgentHandle) -> Path | None:
        subagent_dir = self._subagent_dir(handle)
        if subagent_dir is None:
            return None
        return session_layout.message_history_path(subagent_dir)

    def _subagent_export_dir(
        self,
        handle: SubAgentHandle,
        export_id: str,
    ) -> Path | None:
        subagent_dir = self._subagent_dir(handle)
        if subagent_dir is None:
            return None
        return session_layout.export_dir(subagent_dir, export_id)

    def _append_subagent_history(
        self,
        handle: SubAgentHandle,
        entries: list[dict[str, Any]],
    ) -> None:
        path = self._subagent_history_path(handle)
        if path is None:
            return
        try:
            session_layout.append_jsonl(path, entries, fsync=False)
        except Exception:
            logger.warning("failed to append subagent history %s", path, exc_info=True)

    def _export_query_payload(
        self,
        export: MarkdownExport,
        *,
        include_references: bool = False,
    ) -> dict[str, Any]:
        payload = {
            "export_id": export.export_id,
            "markdown_path": export.session_markdown_path,
            "message_history_path": export.session_jsonl_path,
            "reference_dir_name": export.reference_dir_name,
            "query": {
                "tool": "read_subagent",
                "markdown": {
                    "view": "markdown",
                    "subagent_id": export.subject_id,
                },
                "export": {
                    "view": "export",
                    "subagent_id": export.subject_id,
                },
                "reference": {
                    "view": "ref",
                    "subagent_id": export.subject_id,
                    "ref_path": "<filename>",
                },
            },
        }
        if include_references:
            payload["references"] = [
                {
                    "filename": ref.filename,
                    "mime_type": ref.mime_type,
                }
                for ref in export.references
            ]
        return payload

    def _read_export_reference(
        self,
        handle: SubAgentHandle,
        ref_path: str | None,
    ) -> dict[str, Any] | None:
        if not ref_path:
            return None
        export_info = handle.last_export
        if export_info is None:
            export = self.export_markdown(handle.id)
            export_info = self._export_query_payload(export, include_references=True)
        markdown_path = export_info.get("markdown_path") if isinstance(export_info, dict) else None
        ref_dir_name = export_info.get("reference_dir_name") if isinstance(export_info, dict) else None
        if not isinstance(markdown_path, str) or not isinstance(ref_dir_name, str):
            return None
        filename = Path(ref_path).name
        path = Path(markdown_path).parent / ref_dir_name / filename
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            return None
        return {
            "filename": filename,
            "path": str(path),
            "content": content,
        }

    def _clear_partial_assistant(self, handle: SubAgentHandle) -> None:
        handle.partial_text = ""
        handle.partial_reasoning = ""
        handle.partial_updated_at = None

    def _append_partial_assistant_delta(
        self,
        handle: SubAgentHandle,
        summary: dict[str, Any],
    ) -> None:
        delta = summary.get("delta")
        if not isinstance(delta, str) or delta == "":
            return
        delta_type = summary.get("delta_type")
        if delta_type == "reasoning":
            handle.partial_reasoning += delta
        elif delta_type == "text":
            handle.partial_text += delta
        else:
            return
        handle.partial_updated_at = time.time()

    def _partial_assistant_message(
        self,
        handle: SubAgentHandle,
    ) -> Message | None:
        content: list[ContentPart] = []
        if handle.partial_reasoning:
            content.append({
                "type": "reasoning",
                "reasoning": handle.partial_reasoning,
            })
        if handle.partial_text:
            content.append({
                "type": "text",
                "text": handle.partial_text,
            })
        if not content:
            return None
        return cast(Message, {
            "role": "assistant",
            "content": content,
            "name": None,
            "metadata": {
                "subagent_id": handle.id,
                "subagent_partial": True,
                "updated_at": handle.partial_updated_at,
            },
        })

    async def _enforce_runtime_limit(
        self,
        subagent_id: str,
        max_runtime_seconds: float,
    ) -> None:
        await asyncio.sleep(max_runtime_seconds)
        handle = self._handles.get(subagent_id)
        if handle is None or handle.state in {
            SubAgentLifecycleState.CLOSED,
            SubAgentLifecycleState.COMPLETED,
            SubAgentLifecycleState.FAILED,
            SubAgentLifecycleState.CANCELLED,
        }:
            return
        await handle.runner.interrupt("subagent_runtime_limit")
        handle.state = SubAgentLifecycleState.CANCELLED
        handle.last_error = "Sub-agent runtime limit exceeded"
        handle.updated_at = time.time()

    def _derive_state(
        self,
        handle: SubAgentHandle,
        queue_lengths: dict[str, int],
        executor_state: str,
    ) -> SubAgentLifecycleState:
        if handle.state == SubAgentLifecycleState.CLOSED:
            return SubAgentLifecycleState.CLOSED
        if handle.last_error:
            return handle.state
        if executor_state in {"RUNNING", "INTERRUPTING"}:
            return SubAgentLifecycleState(executor_state)
        if any(queue_lengths.values()):
            return SubAgentLifecycleState.RUNNING
        if handle.state == SubAgentLifecycleState.COMPLETED:
            return SubAgentLifecycleState.COMPLETED
        return SubAgentLifecycleState.IDLE

    def _is_settled(
        self,
        handle: SubAgentHandle,
        queue_lengths: dict[str, int],
    ) -> bool:
        if handle.state in {
            SubAgentLifecycleState.CLOSED,
            SubAgentLifecycleState.FAILED,
            SubAgentLifecycleState.CANCELLED,
        }:
            return True
        if handle.state in {
            SubAgentLifecycleState.CREATED,
            SubAgentLifecycleState.RUNNING,
            SubAgentLifecycleState.INTERRUPTING,
        }:
            return False
        return handle.runner.executor_is_idle and not any(queue_lengths.values())

    def _require_handle(self, subagent_id: str) -> SubAgentHandle:
        try:
            return self._handles[subagent_id]
        except KeyError as exc:
            raise SubAgentError(f"Unknown sub-agent id: {subagent_id}") from exc
