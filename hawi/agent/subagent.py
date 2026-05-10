"""Sub-agent lifecycle management for HawiAgent.

This module keeps sub-agents as a core runtime concept. Agent tools and
plugins can wrap this API, but the lifecycle itself lives under
``HawiAgent.subagents``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Literal

from hawi.events import Event, EventBus, PluginEvent
from hawi.models import ContentPart, Model
from hawi.plugin import HawiPlugin

from .context import ToolCallContext
from .result import AgentRunResult

if TYPE_CHECKING:
    from .agent import HawiAgent
    from .scheduler import HawiScheduler


SubAgentMode = Literal["fork", "fresh"]
SubAgentRole = Literal[
    "general",
    "planner",
    "reviewer",
    "explorer",
    "implementer",
    "critic",
    "summarizer",
]
SubAgentQueue = Literal["normal", "high_prio", "urgent"]
SubAgentResultContract = Literal[
    "text",
    "json",
    "plan",
    "review",
    "diff",
    "artifact",
]


class SubAgentLifecycleState(str, Enum):
    """Lifecycle state for a managed sub-agent."""

    CREATED = "CREATED"
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    INTERRUPTING = "INTERRUPTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    CLOSED = "CLOSED"


@dataclass
class SubAgentLimits:
    """Safety limits for a sub-agent.

    Only ``max_runtime_seconds`` and ``max_children`` are enforced in the first
    implementation. The rest are explicit API placeholders for the next
    scheduler/tool-call budget pass.
    """

    max_runtime_seconds: float | None = None
    max_tool_calls: int | None = None
    max_iterations: int | None = None
    max_recursion_depth: int = 1
    max_children: int | None = None


@dataclass
class SubAgentPluginPolicy:
    """Plugin inheritance and extension policy for child agents."""

    inherit: bool = True
    extra_plugins: list[HawiPlugin] = field(default_factory=list)
    extra_factories: list[Callable[[], HawiPlugin]] = field(default_factory=list)
    allowlist: list[str] | None = None
    denylist: list[str] | None = None
    tool_allowlist: list[str] | None = None
    tool_denylist: list[str] | None = None


@dataclass
class SubAgentSpec:
    """Configuration for creating a sub-agent."""

    mode: SubAgentMode = "fork"
    name: str | None = None
    role: SubAgentRole | str = "general"
    model: Model | str | None = None
    system_prompt: str | list[ContentPart] | None = None
    plugin_policy: SubAgentPluginPolicy = field(default_factory=SubAgentPluginPolicy)
    working_dir: str | None = None
    initial_prompt: str | list[ContentPart] | None = None
    initial_plan: str | dict[str, Any] | list[Any] | None = None
    limits: SubAgentLimits = field(default_factory=SubAgentLimits)
    result_contract: SubAgentResultContract | str = "text"
    ownership: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    description: str | None = None


@dataclass
class SubAgentStatus:
    """Serializable status snapshot for a sub-agent."""

    id: str
    name: str
    role: str
    state: str
    scheduler_state: str
    executor_state: str
    queue_lengths: dict[str, int]
    created_at: float
    updated_at: float
    closed_at: float | None = None
    model_id: str | None = None
    working_dir: str | None = None
    last_result_text: str | None = None
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe status dict."""
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "state": self.state,
            "scheduler_state": self.scheduler_state,
            "executor_state": self.executor_state,
            "queue_lengths": self.queue_lengths,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "closed_at": self.closed_at,
            "model_id": self.model_id,
            "working_dir": self.working_dir,
            "last_result_text": self.last_result_text,
            "last_error": self.last_error,
        }


@dataclass
class SubAgentHandle:
    """Runtime handle for a managed sub-agent."""

    id: str
    spec: SubAgentSpec
    agent: HawiAgent
    scheduler: HawiScheduler
    scheduler_task: asyncio.Task[None]
    event_bus: EventBus
    state: SubAgentLifecycleState = SubAgentLifecycleState.CREATED
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    closed_at: float | None = None
    last_result: AgentRunResult | None = None
    last_error: str | None = None
    recent_events: list[dict[str, Any]] = field(default_factory=list)
    event_handler: Callable[[Event], Any] | None = None
    monitor_task: asyncio.Task[None] | None = None

    @property
    def name(self) -> str:
        return self.spec.name or self.id

    @property
    def role(self) -> str:
        return str(self.spec.role)


ROLE_SYSTEM_PROMPTS: dict[str, str] = {
    "general": (
        "You are a focused sub-agent. Complete the assigned task independently, "
        "state important assumptions, and return a concise handoff."
    ),
    "planner": (
        "You are a planning sub-agent. Produce an executable plan with "
        "dependencies, risks, and acceptance checks."
    ),
    "reviewer": (
        "You are a reviewer sub-agent. Prioritize defects, regressions, missing "
        "tests, and unclear assumptions. Put findings before summary."
    ),
    "explorer": (
        "You are an explorer sub-agent. Inspect the requested material without "
        "making changes, and report evidence with file paths or artifact ids."
    ),
    "implementer": (
        "You are an implementer sub-agent. Work within the declared ownership, "
        "make focused changes, and report changed files."
    ),
    "critic": (
        "You are a critic sub-agent. Look for counterexamples, boundary cases, "
        "incorrect assumptions, and places where the plan could fail."
    ),
    "summarizer": (
        "You are a summarizer sub-agent. Compress context into decisions, "
        "constraints, progress, and clear next steps."
    ),
}


class SubAgentError(RuntimeError):
    """Raised when sub-agent operations fail."""


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

    def list(self) -> list[SubAgentStatus]:
        """Return status snapshots for all known sub-agents."""
        return [self.status(subagent_id) for subagent_id in self._handles]

    async def spawn(
        self,
        spec: SubAgentSpec | None = None,
        **kwargs: Any,
    ) -> SubAgentHandle:
        """Create a sub-agent, start its scheduler, and optionally enqueue work."""
        spec = self._coerce_spec(spec, kwargs)
        async with self._lock:
            self._validate_spawn(spec)

            subagent_id = f"sub_{uuid.uuid4().hex[:8]}"
            spec.name = spec.name or f"{spec.role}-{subagent_id[-4:]}"
            child_event_bus = EventBus()
            child_agent = self._create_child_agent(spec, child_event_bus)

            from .scheduler import HawiScheduler

            scheduler = HawiScheduler(child_agent)
            scheduler_task = asyncio.create_task(
                scheduler.run_forever(poll_interval=self._poll_interval),
                name=f"hawi-subagent-{subagent_id}",
            )

            handle = SubAgentHandle(
                id=subagent_id,
                spec=spec,
                agent=child_agent,
                scheduler=scheduler,
                scheduler_task=scheduler_task,
                event_bus=child_event_bus,
                state=SubAgentLifecycleState.IDLE,
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
        """Send a message to a running sub-agent scheduler."""
        handle = self._require_handle(subagent_id)
        if handle.state == SubAgentLifecycleState.CLOSED:
            raise SubAgentError(f"Sub-agent is closed: {subagent_id}")

        message_id = handle.scheduler.enqueue(
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
        result = handle.last_result or handle.scheduler.last_result
        if result is not None:
            handle.last_result = result
        scheduler_state = handle.scheduler.state.name
        executor_state = handle.scheduler.executor_state.name
        queue_lengths = handle.scheduler.get_queue_lengths()
        state = self._derive_state(handle, queue_lengths, executor_state)
        model_id = getattr(handle.agent.model, "model_id", None)
        return SubAgentStatus(
            id=handle.id,
            name=handle.name,
            role=handle.role,
            state=state.value,
            scheduler_state=scheduler_state,
            executor_state=executor_state,
            queue_lengths=queue_lengths,
            created_at=handle.created_at,
            updated_at=handle.updated_at,
            closed_at=handle.closed_at,
            model_id=str(model_id) if model_id is not None else None,
            working_dir=handle.spec.working_dir,
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
        handle = self._require_handle(subagent_id)
        deadline = time.monotonic() + timeout if timeout is not None else None

        while True:
            result = handle.scheduler.last_result
            if result is not None:
                handle.last_result = result

            queue_lengths = handle.scheduler.get_queue_lengths()
            if self._is_settled(handle, queue_lengths):
                if handle.last_error and raise_on_error:
                    raise SubAgentError(handle.last_error)
                return handle.last_result

            if handle.last_error and raise_on_error:
                raise SubAgentError(handle.last_error)

            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for sub-agent: {subagent_id}")
            await asyncio.sleep(self._poll_interval)

    def recent_events(self, subagent_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent child event summaries."""
        handle = self._require_handle(subagent_id)
        if limit <= 0:
            return []
        return handle.recent_events[-limit:]

    async def interrupt(self, subagent_id: str, reason: str = "parent") -> list[str]:
        """Interrupt the sub-agent's current run."""
        handle = self._require_handle(subagent_id)
        interrupted = await handle.scheduler.interrupt(reason)
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
        """Stop scheduler tasks and close a sub-agent."""
        handle = self._require_handle(subagent_id)
        if handle.state == SubAgentLifecycleState.CLOSED:
            return self.status(subagent_id)

        if interrupt:
            with contextlib.suppress(asyncio.CancelledError):
                await handle.scheduler.interrupt(reason)

        handle.scheduler.stop()
        if handle.monitor_task and not handle.monitor_task.done():
            handle.monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await handle.monitor_task
        if not handle.scheduler_task.done():
            handle.scheduler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await handle.scheduler_task

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
        view: Literal["status", "summary", "events", "context_tail"] = "summary",
        limit: int = 20,
    ) -> dict[str, Any]:
        """Read a controlled view of sub-agent state."""
        handle = self._require_handle(subagent_id)
        status = self.status(subagent_id).to_dict()
        if view == "status":
            return {"status": status}
        if view == "events":
            return {"status": status, "events": self.recent_events(subagent_id, limit)}
        if view == "context_tail":
            return {
                "status": status,
                "messages": deepcopy(handle.agent.context.messages[-limit:]),
            }
        return {
            "status": status,
            "recent_events": self.recent_events(subagent_id, min(limit, 10)),
        }

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
            _drop_trailing_unanswered_tool_call_turn(child.context.messages)
        else:
            child = (
                self._parent.clone()
                if policy.inherit
                else self._new_agent_without_inherited_plugins(spec, event_bus)
            )
            child.context.clear()

        self._rebind_agent_event_bus(child, event_bus)
        self._apply_plugin_policy(child, policy)

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
        from .agent import HawiAgent

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
        if policy.allowlist or policy.denylist or policy.tool_allowlist or policy.tool_denylist:
            raise NotImplementedError(
                "Sub-agent plugin allow/deny policies are reserved for the next "
                "tool-permission pass. Use inherit and extra plugins for now."
            )
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
            return _normalize_system_prompt(spec.system_prompt)

        role_prompt = ROLE_SYSTEM_PROMPTS.get(str(spec.role), ROLE_SYSTEM_PROMPTS["general"])
        base = deepcopy(child.context.get_system_prompt() or [])
        if spec.mode == "fork" and str(spec.role) == "general" and base:
            return base
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
            return spec.initial_prompt

        if isinstance(spec.initial_plan, str):
            plan_text = spec.initial_plan
        else:
            plan_text = json.dumps(spec.initial_plan, ensure_ascii=False, indent=2)

        if spec.initial_prompt is None:
            return f"Execute the following initial plan:\n\n{plan_text}"
        if isinstance(spec.initial_prompt, str):
            return f"{spec.initial_prompt}\n\nInitial plan:\n{plan_text}"
        content = deepcopy(spec.initial_prompt)
        content.append({"type": "text", "text": f"Initial plan:\n{plan_text}"})
        return content

    def _make_event_handler(
        self,
        handle: SubAgentHandle,
    ) -> Callable[[Event], Any]:
        async def on_child_event(event: Event) -> None:
            summary = _event_summary(event)
            handle.recent_events.append(summary)
            if len(handle.recent_events) > 200:
                del handle.recent_events[:-200]
            handle.updated_at = time.time()

            if event.type == "agent.run_start":
                handle.state = SubAgentLifecycleState.RUNNING
            elif event.type == "agent.run_stop":
                handle.state = SubAgentLifecycleState.COMPLETED
            elif event.type == "agent.error":
                handle.state = SubAgentLifecycleState.FAILED
                handle.last_error = summary.get("error") or "Sub-agent failed"
            elif event.type == "scheduler.interrupt":
                handle.state = SubAgentLifecycleState.INTERRUPTING

            await self._emit_manager_event(
                handle,
                "subagent.event",
                {
                    "child_event": summary,
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
            PluginEvent.create(
                "plugin.event",
                plugin_name="SubAgent",
                plugin_id="subagent",
                payload={
                    "event_name": event_name,
                    "subagent_id": handle.id,
                    "subagent_name": handle.name,
                    "subagent_role": handle.role,
                    **payload,
                },
            ),
            None,
        )

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
        await handle.scheduler.interrupt("subagent_runtime_limit")
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
        return handle.scheduler.executor_is_idle and not any(queue_lengths.values())

    def _require_handle(self, subagent_id: str) -> SubAgentHandle:
        try:
            return self._handles[subagent_id]
        except KeyError as exc:
            raise SubAgentError(f"Unknown sub-agent id: {subagent_id}") from exc


def _drop_trailing_unanswered_tool_call_turn(messages: list[dict[str, Any]]) -> int:
    """Drop the trailing parent tool-call turn if it is still in progress.

    Forking can happen while a parent tool is still executing. At that moment
    the parent context already contains the assistant tool_call message, but
    its matching tool result has not been appended yet. The forked child should
    see the last stable context plus its own new task message, not the parent's
    half-finished tool-calling turn.
    """
    if not messages:
        return 0

    assistant_index = len(messages) - 1
    while assistant_index >= 0 and messages[assistant_index].get("role") == "tool":
        assistant_index -= 1
    if assistant_index < 0:
        return 0

    assistant = messages[assistant_index]
    if assistant.get("role") != "assistant":
        return 0

    content = assistant.get("content")
    if not isinstance(content, list):
        return 0

    tool_call_ids = {
        str(part.get("id"))
        for part in content
        if isinstance(part, dict)
        and part.get("type") == "tool_call"
        and part.get("id")
    }
    if not tool_call_ids:
        return 0

    responded_ids: set[str] = set()
    for message in messages[assistant_index + 1:]:
        tool_content = message.get("content")
        if not isinstance(tool_content, list):
            continue
        responded_ids.update(
            str(part.get("tool_call_id"))
            for part in tool_content
            if isinstance(part, dict)
            and part.get("type") == "tool_result"
            and part.get("tool_call_id")
        )

    if tool_call_ids <= responded_ids:
        return 0

    removed = len(messages) - assistant_index
    del messages[assistant_index:]
    return removed


def _normalize_system_prompt(
    value: str | list[ContentPart],
) -> list[ContentPart]:
    if isinstance(value, str):
        return [{"type": "text", "text": value}]
    return deepcopy(value)


def _event_summary(event: Event) -> dict[str, Any]:
    data = event.model_dump(mode="json", exclude_none=True)
    summary: dict[str, Any] = {
        "type": data.get("type"),
        "source": data.get("source"),
        "timestamp": data.get("timestamp"),
    }
    for key in (
        "run_id",
        "tool_call_id",
        "tool_name",
        "message_id",
        "queue_type",
        "stop_reason",
        "reason",
    ):
        if key in data:
            summary[key] = data[key]
    if "error" in data:
        summary["error"] = str(data["error"])
    if event.type == "agent.message_added" and "content" in data:
        summary["content_preview"] = _content_preview(data["content"])
    return summary


def _content_preview(content: Any, max_chars: int = 160) -> str:
    if not isinstance(content, list):
        text = str(content)
    else:
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        text = " ".join(parts)
    return text[: max_chars - 3] + "..." if len(text) > max_chars else text
