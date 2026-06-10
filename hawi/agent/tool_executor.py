"""Asynchronous tool execution for Hawi agents.

The executor owns the mechanics of running tool calls: hook dispatch,
framework-parameter stripping, audit queuing, runtime context injection,
streaming tool outputs, result persistence, and tool-call events.  The agent
run loop stays responsible for model orchestration and deciding when a batch
of tool calls should be executed.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TYPE_CHECKING, cast

from hawi.errors import ToolExecutionError, ToolNotFoundError
from hawi.events import (
    AgentToolCallEvent,
    AgentToolResultEvent,
    AgentToolResultPartEvent,
    AgentToolRuntimeContextInjectedEvent,
    Event,
    EventBus,
)
from hawi.models import ContentPart, ToolCallPart
from hawi.models.message import CachePoint
from hawi.plugin import PluginManager
from hawi.plugin.hook_context import HookContext, HookResult
from hawi.permission.types import PermissionPolicy
from hawi.permission.audit import PermissionAuditSink
from hawi.review import RuntimeReviewBroker, RuntimeReviewDecision
from hawi.tool.types import (
    AgentTool,
    ToolParameterInjectionContext,
    ToolResult,
)

from .context import AgentContext
from .result import ToolCallRecord
from .state import AddedToolResultMessages

if TYPE_CHECKING:
    from .agent import HawiAgent

logger = logging.getLogger(__name__)

TOOL_RESULT_MAX_BYTES = 50 * 1024
TOOL_RESULT_TRUNCATION_WARNING = (
    "Tool result was truncated before being written to context. "
    "Narrow the request, use pagination, or write large output to a file/artifact instead."
)
TOOL_RESULT_TRUNCATION_SUFFIX = (
    f"\n\n[Hawi warning: {TOOL_RESULT_TRUNCATION_WARNING}]"
)


class EmitEventCallback(Protocol):
    def __call__(
        self,
        event: Event,
        event_bus: EventBus | None,
    ) -> Awaitable[Event]:
        ...


class AddToolResultCallback(Protocol):
    def __call__(
        self,
        tool_call_id: str,
        content: str | list[ContentPart],
        *,
        is_error: bool = False,
        materialize_pending_steer: bool = True,
        cache_point: CachePoint | dict[str, Any] | bool | None = None,
        cache_point_source: str | None = None,
    ) -> AddedToolResultMessages:
        ...


class EmitToolResultMessageCallback(Protocol):
    def __call__(
        self,
        *,
        run_id: str,
        tool_call_id: str,
        content: str | list[ContentPart],
        is_error: bool,
        context_message_id: str,
        event_bus: EventBus | None,
    ) -> Awaitable[None]:
        ...


class EmitMaterializedSteerCallback(Protocol):
    def __call__(
        self,
        run_id: str,
        materialized_messages: list[Any],
        event_bus: EventBus | None,
    ) -> Awaitable[None]:
        ...


@dataclass
class PreparedToolArguments:
    """Arguments split into tool-visible and framework-visible parts."""

    tool_arguments: dict[str, Any]
    injected_arguments: dict[str, Any] = field(default_factory=dict)
    short_circuit_result: ToolResult | None = None
    output_prefix: str | None = None


ToolCallRequestStatus = Literal[
    "queued",
    "running",
    "completed",
    "failed",
    "cancelled",
]


TOOL_CALL_PURPOSE_PARAMETER = "tool_call_purpose"
MISSING_TOOL_CALL_PURPOSE_OUTPUT_PREFIX = (
    "Error: tool_call_purpose 字段必填；未指定会导致用户误解，并影响自动审核 agent 的判断准确度。"
)


@dataclass
class ToolCallRequest:
    """A queued request to execute one model-produced tool call.

    ``blocked_by`` accepts either another request id or another tool call id.
    This keeps dependency edges stable for the executor while still letting
    agent/runtime snapshots talk in provider-facing tool call ids.
    """

    tool_call: ToolCallPart
    run_id: str
    iteration: int
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    blocked_by: str | None = None
    event_bus: EventBus | None = field(default=None, repr=False, compare=False)
    materialize_pending_steer: bool = True
    add_to_context: bool = True
    emit_final_event: bool = True
    run_injection_handlers: bool = True
    audit_action: Literal["queue", "execute"] = "queue"
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tool_call_id(self) -> str:
        return str(self.tool_call.get("id", ""))

    @property
    def tool_name(self) -> str:
        return str(self.tool_call.get("name", ""))

    def snapshot(self, status: ToolCallRequestStatus) -> dict[str, Any]:
        """Return a JSON-serializable view of this request."""

        return {
            "request_id": self.request_id,
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "tool_call": dict(self.tool_call),
            "run_id": self.run_id,
            "iteration": self.iteration,
            "blocked_by": self.blocked_by,
            "status": status,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }


class ToolCallPromise:
    """Awaitable handle returned when a tool request is enqueued."""

    def __init__(
        self,
        request: ToolCallRequest,
        future: asyncio.Future[ToolCallRecord],
    ) -> None:
        self.request = request
        self._future = future

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def tool_call_id(self) -> str:
        return self.request.tool_call_id

    @property
    def blocked_by(self) -> str | None:
        return self.request.blocked_by

    @property
    def done(self) -> bool:
        return self._future.done()

    @property
    def cancelled(self) -> bool:
        return self._future.cancelled()

    def cancel(self) -> bool:
        return self._future.cancel()

    def result(self) -> ToolCallRecord:
        return self._future.result()

    async def wait(self) -> ToolCallRecord:
        return await self._future

    def __await__(self):
        return self.wait().__await__()


@dataclass
class ToolExecutionOutcome:
    """Internal result of one tool call before/after context persistence."""

    record: ToolCallRecord
    audit_pending: bool
    result_content: str
    control: HookResult | None = None
    context_message_id: str | None = None


@dataclass
class ToolExecutionBatchResult:
    """Result of executing one model-produced tool-call batch."""

    records: list[ToolCallRecord] = field(default_factory=list)
    completed_tool_call_ids: list[str] = field(default_factory=list)
    control: HookResult | None = None


class ToolExecutor:
    """Execute and manage agent tool calls asynchronously."""

    def __init__(
        self,
        *,
        agent: HawiAgent,
        plugin_manager: PluginManager,
        context: AgentContext,
        emit_event: EmitEventCallback,
        render_tool_result: Callable[[ToolResult], str],
        add_tool_result: AddToolResultCallback,
        emit_tool_result_message: EmitToolResultMessageCallback,
        emit_materialized_steer_events: EmitMaterializedSteerCallback,
        current_tool_calls: list[ToolCallPart] | None = None,
    ) -> None:
        self._agent = agent
        self._plugin_manager = plugin_manager
        self._context = context
        self._emit_event = emit_event
        self._render_tool_result = render_tool_result
        self._add_tool_result = add_tool_result
        self._emit_tool_result_message = emit_tool_result_message
        self._emit_materialized_steer_events = emit_materialized_steer_events
        self._current_tool_calls = current_tool_calls
        self._requests: dict[str, ToolCallRequest] = {}
        self._request_queue: list[str] = []
        self._promises: dict[str, ToolCallPromise] = {}
        self._request_status: dict[str, ToolCallRequestStatus] = {}
        self._request_outcomes: dict[str, ToolExecutionOutcome] = {}
        self._request_by_tool_call_id: dict[str, str] = {}
        self._completed_request_ids: set[str] = set()
        self._released_request_ids: set[str] = set()
        self._dispatcher_task: asyncio.Task[None] | None = None

    async def execute_call(
        self,
        tool_call: ToolCallPart,
        *,
        run_id: str,
        iteration: int,
        event_bus: EventBus | None = None,
        materialize_pending_steer: bool = True,
        add_to_context: bool = True,
        emit_final_event: bool = True,
        run_injection_handlers: bool = True,
        audit_action: Literal["queue", "execute"] = "queue",
    ) -> ToolCallRecord:
        """Execute one tool call and optionally persist the tool result."""

        request = ToolCallRequest(
            tool_call=tool_call,
            run_id=run_id,
            iteration=iteration,
            event_bus=event_bus,
            audit_action=audit_action,
            materialize_pending_steer=materialize_pending_steer,
            add_to_context=add_to_context,
            emit_final_event=emit_final_event,
            run_injection_handlers=run_injection_handlers,
        )
        promise = self.enqueue_call(request)
        try:
            outcome = await self._drain_until_outcome(promise)
        except asyncio.CancelledError:
            self.cancel_requests([promise.request_id])
            raise
        return outcome.record

    async def execute_batch(
        self,
        tool_calls: list[ToolCallPart],
        *,
        run_id: str,
        iteration: int,
        event_bus: EventBus | None = None,
        is_interrupted: Callable[[], bool] | None = None,
        materialize_pending_steer: bool = True,
    ) -> ToolExecutionBatchResult:
        """Execute multiple tool calls in model order.

        Tool calls emitted in one assistant turn may depend on earlier calls
        through side effects or tool-visible context.  Hawi therefore commits
        each result before starting the next call.
        """

        if not tool_calls:
            return ToolExecutionBatchResult()

        promises = self.enqueue_batch(
            tool_calls,
            run_id=run_id,
            iteration=iteration,
            event_bus=event_bus,
            materialize_pending_steer=False,
            add_to_context=False,
            emit_final_event=False,
            run_injection_handlers=True,
            audit_action="queue",
            chain_blocked=True,
        )
        return await self.resolve_batch(
            promises,
            run_id=run_id,
            iteration=iteration,
            event_bus=event_bus,
            is_interrupted=is_interrupted,
            materialize_pending_steer=materialize_pending_steer,
        )

    async def resolve_batch(
        self,
        promises: list[ToolCallPromise],
        *,
        run_id: str,
        iteration: int,
        event_bus: EventBus | None = None,
        is_interrupted: Callable[[], bool] | None = None,
        materialize_pending_steer: bool = True,
    ) -> ToolExecutionBatchResult:
        """Resolve queued tool promises and commit their results in order."""

        if not promises:
            return ToolExecutionBatchResult()

        outcomes: list[ToolExecutionOutcome] = []
        try:
            control: HookResult | None = None
            for index, promise in enumerate(promises):
                if is_interrupted is not None and is_interrupted():
                    interrupted = self._synthesize_stopped_tool_results(
                        [item.request.tool_call for item in promises[index:]],
                        run_id=run_id,
                        iteration=iteration,
                        reason="interrupted",
                    )
                    outcomes.extend(interrupted)
                    for interrupted_promise, interrupted_outcome in zip(
                        promises[index:],
                        interrupted,
                    ):
                        await self._commit_outcomes(
                            [interrupted_outcome],
                            run_id=run_id,
                            event_bus=event_bus,
                            materialize_pending_steer=False,
                        )
                        await self._emit_final_result_event(
                            interrupted_outcome.record,
                            run_id,
                            event_bus,
                            context_message_id=interrupted_outcome.context_message_id,
                        )
                        self._complete_request(
                            interrupted_promise.request_id,
                            interrupted_outcome,
                            release=True,
                        )
                    break

                outcome = await self._drain_until_outcome(promise)
                outcomes.append(outcome)
                await self._commit_outcome_if_needed(
                    promise,
                    outcome,
                    run_id=run_id,
                    event_bus=event_bus,
                )

                if outcome.control is not None:
                    control = outcome.control
                    remaining_promises = promises[index + 1 :]
                    skipped = self._synthesize_stopped_tool_results(
                        [item.request.tool_call for item in remaining_promises],
                        run_id=run_id,
                        iteration=iteration,
                        reason=self._control_reason(control),
                    )
                    outcomes.extend(skipped)
                    for skipped_promise, skipped_outcome in zip(
                        remaining_promises,
                        skipped,
                    ):
                        await self._commit_outcomes(
                            [skipped_outcome],
                            run_id=run_id,
                            event_bus=event_bus,
                            materialize_pending_steer=False,
                        )
                        await self._emit_final_result_event(
                            skipped_outcome.record,
                            run_id,
                            event_bus,
                            context_message_id=skipped_outcome.context_message_id,
                        )
                        self._complete_request(
                            skipped_promise.request_id,
                            skipped_outcome,
                            release=True,
                        )
                    break
        except asyncio.CancelledError:
            self.cancel_requests([promise.request_id for promise in promises])
            raise
        finally:
            self.cancel_requests(
                [
                    promise.request_id
                    for promise in promises
                    if not promise.done
                ]
            )

        if (
            materialize_pending_steer
            and outcomes
            and not (is_interrupted is not None and is_interrupted())
        ):
            await self._materialize_pending_steer(
                [outcome.record.tool_call_id for outcome in outcomes],
                run_id=run_id,
                event_bus=event_bus,
            )

        return ToolExecutionBatchResult(
            records=[outcome.record for outcome in outcomes],
            completed_tool_call_ids=[
                outcome.record.tool_call_id for outcome in outcomes
            ],
            control=control,
        )

    def enqueue_call(self, request: ToolCallRequest) -> ToolCallPromise:
        """Queue a tool call request and return an awaitable promise."""

        if request.request_id in self._requests:
            raise ValueError(
                f"Tool call request already exists: {request.request_id}"
            )

        loop = asyncio.get_running_loop()
        future: asyncio.Future[ToolCallRecord] = loop.create_future()
        promise = ToolCallPromise(request, future)
        self._requests[request.request_id] = request
        self._promises[request.request_id] = promise
        self._request_status[request.request_id] = "queued"
        self._request_queue.append(request.request_id)
        if request.tool_call_id:
            self._request_by_tool_call_id[request.tool_call_id] = request.request_id
        self._register_active_tool_calls([request.tool_call])
        self.start_dispatcher()
        return promise

    def enqueue_batch(
        self,
        tool_calls: list[ToolCallPart],
        *,
        run_id: str,
        iteration: int,
        event_bus: EventBus | None = None,
        materialize_pending_steer: bool = True,
        add_to_context: bool = True,
        emit_final_event: bool = True,
        run_injection_handlers: bool = True,
        audit_action: Literal["queue", "execute"] = "queue",
        chain_blocked: bool = True,
    ) -> list[ToolCallPromise]:
        """Queue a batch of tool calls.

        When ``chain_blocked`` is true, every request after the first is
        blocked by the previous request.  This preserves model order while the
        executor now owns dependency edges explicitly.
        """

        promises: list[ToolCallPromise] = []
        blocked_by: str | None = None
        for tool_call in tool_calls:
            request = ToolCallRequest(
                tool_call=tool_call,
                run_id=run_id,
                iteration=iteration,
                blocked_by=blocked_by,
                event_bus=event_bus,
                materialize_pending_steer=materialize_pending_steer,
                add_to_context=add_to_context,
                emit_final_event=emit_final_event,
                run_injection_handlers=run_injection_handlers,
                audit_action=audit_action,
            )
            promise = self.enqueue_call(request)
            promises.append(promise)
            if chain_blocked:
                blocked_by = promise.request_id
        return promises

    async def drain_until_complete(
        self,
        promises: list[ToolCallPromise] | None = None,
    ) -> list[ToolCallRecord]:
        """Run queued requests until the selected promises are resolved."""

        if promises is None:
            target_ids = list(self._request_queue)
            promises = [
                self._promises[request_id]
                for request_id in target_ids
                if request_id in self._promises
            ]

        outcomes: list[ToolExecutionOutcome] = []
        for promise in promises:
            if not promise.done:
                outcomes.append(await self._drain_until_outcome(promise))
                self._release_request(promise.request_id)
            elif promise.request_id in self._request_outcomes:
                outcomes.append(self._request_outcomes[promise.request_id])
                self._release_request(promise.request_id)

        return [outcome.record for outcome in outcomes]

    def start_dispatcher(self) -> None:
        """Start the background dispatcher if there is queued work."""

        if not self._request_queue:
            return
        if self._dispatcher_task is not None and not self._dispatcher_task.done():
            return
        self._dispatcher_task = asyncio.create_task(self._dispatch_queued())

    async def wait_for_dispatcher(self) -> None:
        """Wait for the current dispatcher task, if any."""

        task = self._dispatcher_task
        if task is not None and not task.done():
            await task

    def cancel_requests(self, request_ids: list[str]) -> None:
        """Cancel queued requests and forget unresolved promises."""

        for request_id in request_ids:
            promise = self._promises.get(request_id)
            request = self._requests.get(request_id)
            if promise is None or request is None or promise.done:
                continue
            if request_id in self._request_queue:
                self._request_queue.remove(request_id)
            self._request_status[request_id] = "cancelled"
            promise.cancel()
            self._unregister_active_tool_calls([request.tool_call])

    def clear(self) -> None:
        """Clear queued executor state.

        Completed request metadata is discarded because promises are runtime
        handles, not persisted session objects.
        """

        if self._dispatcher_task is not None and not self._dispatcher_task.done():
            self._dispatcher_task.cancel()
        self.cancel_requests(list(self._promises))
        self._requests.clear()
        self._request_queue.clear()
        self._promises.clear()
        self._request_status.clear()
        self._request_outcomes.clear()
        self._request_by_tool_call_id.clear()
        self._completed_request_ids.clear()
        self._released_request_ids.clear()
        self._dispatcher_task = None

    def snapshot(self) -> dict[str, Any]:
        """Return a serializable view of queued and recently completed work."""

        return {
            "version": 1,
            "queue": list(self._request_queue),
            "requests": [
                request.snapshot(
                    self._request_status.get(request_id, "queued")
                )
                for request_id, request in self._requests.items()
            ],
            "completed_request_ids": list(self._completed_request_ids),
            "released_request_ids": list(self._released_request_ids),
        }

    async def prepare_tool_arguments(
        self,
        tool: AgentTool,
        arguments: dict[str, Any],
        *,
        tool_call_id: str,
        run_id: str,
        iteration: int,
        run_injection_handlers: bool,
    ) -> PreparedToolArguments:
        """Validate and strip framework-injected parameters before tool calls."""

        tool_arguments = dict(arguments)
        injections = self._plugin_manager.get_tool_parameter_injections(tool)
        if not injections:
            return PreparedToolArguments(tool_arguments=tool_arguments)

        from hawi.tool._utils import validate_parameters

        injected_schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                injection.name: injection.schema_copy()
                for injection in injections
            },
        }
        validation_arguments = dict(arguments)
        missing_default_none_injections: set[str] = set()
        output_prefixes: list[str] = []
        required: list[str] = []
        for injection in injections:
            if not injection.required:
                continue
            schema = injection.schema_copy()
            missing_or_none = (
                injection.name not in validation_arguments
                or validation_arguments.get(injection.name) is None
            )
            if missing_or_none and schema.get("default", object()) is None:
                missing_default_none_injections.add(injection.name)
                validation_arguments.pop(injection.name, None)
                if injection.name == TOOL_CALL_PURPOSE_PARAMETER:
                    output_prefixes.append(MISSING_TOOL_CALL_PURPOSE_OUTPUT_PREFIX)
                continue
            required.append(injection.name)
        if required:
            injected_schema["required"] = required

        is_valid, errors = validate_parameters(validation_arguments, injected_schema)
        if not is_valid:
            return PreparedToolArguments(
                tool_arguments=tool_arguments,
                short_circuit_result=ToolResult(
                    success=False,
                    error=(
                        "Injected parameter validation failed: "
                        f"{'; '.join(errors)}"
                    ),
                ),
            )

        injected_arguments: dict[str, Any] = {}
        for injection in injections:
            if injection.name in tool_arguments:
                injected_arguments[injection.name] = tool_arguments.pop(injection.name)
            elif injection.name in missing_default_none_injections:
                injected_arguments[injection.name] = None

        prepared = PreparedToolArguments(
            tool_arguments=tool_arguments,
            injected_arguments=injected_arguments,
            output_prefix="\n".join(output_prefixes) if output_prefixes else None,
        )
        if not run_injection_handlers:
            return prepared

        handler_context = ToolParameterInjectionContext(
            agent=self._agent,
            tool=tool,
            tool_name=tool.name,
            tool_call_id=tool_call_id,
            run_id=run_id,
            iteration=iteration,
            arguments=dict(arguments),
            injected_arguments=dict(injected_arguments),
        )

        for injection in injections:
            if injection.name not in injected_arguments or injection.handler is None:
                continue
            try:
                handler_result = injection.handler(
                    handler_context,
                    injected_arguments[injection.name],
                )
                if inspect.isawaitable(handler_result):
                    handler_result = await handler_result
            except Exception as e:
                prepared.short_circuit_result = ToolResult(
                    success=False,
                    error=(
                        "Injected parameter handler failed: "
                        f"{type(e).__name__}: {e}"
                    ),
                )
                break
            if isinstance(handler_result, ToolResult):
                prepared.short_circuit_result = handler_result
                break

        return prepared

    @staticmethod
    def _prepend_tool_result_output(result: ToolResult, prefix: str | None) -> ToolResult:
        if not prefix:
            return result
        output_str = (
            result.output
            if isinstance(result.output, str)
            else str(result.output)
            if result.output is not None
            else ""
        )
        output = prefix if not output_str else f"{prefix}\n{output_str}"
        return ToolResult(
            success=result.success,
            output=output,
            error=result.error,
            cache_point=result.cache_point,
            cache_point_source=result.cache_point_source,
        )

    def inject_tool_runtime_context(
        self,
        tool: AgentTool,
        tool_arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Inject Hawi's runtime tool-call context into tool-visible arguments."""

        prepared = dict(tool_arguments)
        context_param = getattr(tool, "context", None)
        if context_param and self._context.tool_call_context:
            prepared[context_param] = self._context.tool_call_context
        return prepared

    async def _drain_until_outcome(
        self,
        promise: ToolCallPromise,
    ) -> ToolExecutionOutcome:
        self.start_dispatcher()
        while not promise.done:
            task = self._dispatcher_task
            if task is None or task.done():
                if task is not None:
                    exc = self._task_exception(task)
                    if exc is not None:
                        raise exc
                self.start_dispatcher()
                task = self._dispatcher_task
            if task is None or task.done():
                raise RuntimeError(
                    "No ready tool call requests; unresolved blockers: "
                    f"{self._blocked_request_debug()}"
                )
            done, _ = await asyncio.wait(
                {promise._future, task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if promise._future in done:
                if promise.cancelled:
                    raise asyncio.CancelledError
                exc = promise._future.exception()
                if exc is not None:
                    raise exc
                break
            if task in done:
                exc = self._task_exception(task)
                if exc is not None:
                    raise exc

        if promise.cancelled:
            raise asyncio.CancelledError
        if promise.done:
            exc = promise._future.exception()
            if exc is not None:
                raise exc
        if promise.request_id not in self._request_outcomes:
            raise RuntimeError(
                f"Tool call promise resolved without outcome: {promise.request_id}"
            )
        return self._request_outcomes[promise.request_id]

    async def _dispatch_queued(self) -> None:
        while True:
            request_id = self._next_ready_request_id()
            if request_id is None:
                return
            outcome = await self._execute_request(request_id)
            if outcome.control is not None:
                return

    @staticmethod
    def _task_exception(task: asyncio.Task[Any]) -> BaseException | None:
        if task.cancelled():
            return asyncio.CancelledError()
        return task.exception()

    def _next_ready_request_id(self) -> str | None:
        for request_id in list(self._request_queue):
            request = self._requests.get(request_id)
            if request is None:
                self._request_queue.remove(request_id)
                continue
            if self._request_status.get(request_id) != "queued":
                continue
            if self._is_unblocked(request):
                return request_id
        return None

    def _is_unblocked(self, request: ToolCallRequest) -> bool:
        blocker = request.blocked_by
        if not blocker:
            return True
        if blocker in self._released_request_ids:
            return True
        request_id = self._request_by_tool_call_id.get(blocker)
        return bool(request_id and request_id in self._released_request_ids)

    def _blocked_request_debug(self) -> list[dict[str, str | None]]:
        blocked: list[dict[str, str | None]] = []
        for request_id in self._request_queue:
            request = self._requests.get(request_id)
            if request is None:
                continue
            if self._request_status.get(request_id) != "queued":
                continue
            blocked.append(
                {
                    "request_id": request_id,
                    "tool_call_id": request.tool_call_id,
                    "blocked_by": request.blocked_by,
                }
            )
        return blocked

    async def _execute_request(
        self,
        request_id: str,
    ) -> ToolExecutionOutcome:
        request = self._requests[request_id]
        if request_id in self._request_queue:
            self._request_queue.remove(request_id)
        self._request_status[request_id] = "running"
        try:
            outcome = await self._run_tool_call(
                request.tool_call,
                run_id=request.run_id,
                iteration=request.iteration,
                event_bus=request.event_bus,
                run_injection_handlers=request.run_injection_handlers,
                audit_action=request.audit_action,
            )
            if request.add_to_context:
                await self._commit_outcomes(
                    [outcome],
                    run_id=request.run_id,
                    event_bus=request.event_bus,
                    materialize_pending_steer=request.materialize_pending_steer,
                )
            if request.emit_final_event:
                await self._emit_final_result_event(
                    outcome.record,
                    request.run_id,
                    request.event_bus,
                    context_message_id=outcome.context_message_id,
                )
            self._complete_request(
                request_id,
                outcome,
                release=request.add_to_context,
            )
            return outcome
        except asyncio.CancelledError:
            self._request_status[request_id] = "cancelled"
            self.cancel_requests([request_id])
            raise
        except Exception as exc:
            self._request_status[request_id] = "failed"
            self._unregister_active_tool_calls([request.tool_call])
            promise = self._promises.get(request_id)
            if promise is not None and not promise.done:
                promise._future.set_exception(exc)
            raise

    def _complete_request(
        self,
        request_id: str,
        outcome: ToolExecutionOutcome,
        *,
        release: bool = False,
    ) -> None:
        request = self._requests.get(request_id)
        promise = self._promises.get(request_id)
        if request is None or promise is None:
            return
        if request_id in self._request_queue:
            self._request_queue.remove(request_id)
        self._request_outcomes[request_id] = outcome
        self._request_status[request_id] = "completed"
        self._completed_request_ids.add(request_id)
        if release:
            self._released_request_ids.add(request_id)
        self._unregister_active_tool_calls([request.tool_call])
        if not promise.done:
            promise._future.set_result(outcome.record)

    async def _commit_outcome_if_needed(
        self,
        promise: ToolCallPromise,
        outcome: ToolExecutionOutcome,
        *,
        run_id: str,
        event_bus: EventBus | None,
    ) -> None:
        if outcome.context_message_id is None:
            await self._commit_outcomes(
                [outcome],
                run_id=run_id,
                event_bus=event_bus,
                materialize_pending_steer=False,
            )
            await self._emit_final_result_event(
                outcome.record,
                run_id,
                event_bus,
                context_message_id=outcome.context_message_id,
            )
        self._release_request(promise.request_id)

    def _release_request(self, request_id: str) -> None:
        if request_id not in self._completed_request_ids:
            return
        self._released_request_ids.add(request_id)
        self.start_dispatcher()

    async def _run_tool_call(
        self,
        tool_call: ToolCallPart,
        *,
        run_id: str,
        iteration: int,
        event_bus: EventBus | None,
        run_injection_handlers: bool,
        audit_action: Literal["queue", "execute"],
    ) -> ToolExecutionOutcome:
        tool_name = tool_call["name"]
        arguments = dict(tool_call["arguments"])
        tool_call_id = tool_call["id"]
        start_time = time.time()

        await self._emit_event(
            AgentToolCallEvent.create(
                run_id=run_id,
                tool_name=tool_name,
                arguments=arguments,
                tool_call_id=tool_call_id,
            ),
            event_bus,
        )

        tool = self._plugin_manager.get_tool(tool_name)
        audit_pending = False
        result_output_prefix: str | None = None

        before_ctx = HookContext(
            run_id=run_id,
            iteration=iteration,
            tool_call_id=tool_call_id,
            tool=tool,
            context=self._context,
            review=getattr(self._agent, "review_broker", None),
        )
        hook_result = await self._invoke_before_tool_calling(
            tool_name,
            arguments,
            before_ctx,
        )
        control: HookResult | None = None
        if hook_result:
            if hook_result.action == "skip":
                result = hook_result.tool_result or ToolResult(
                    success=False,
                    error="Hook skipped tool without providing a result",
                )
            elif hook_result.action == "abort":
                control = hook_result
                reason = hook_result.reason or "no reason provided"
                result = ToolResult(
                    success=False,
                    error=f"Aborted by before_tool_calling hook: {reason}",
                )
            else:
                result = ToolResult(
                    success=False,
                    error=(
                        "Unsupported before_tool_calling hook action: "
                        f"{hook_result.action}"
                    ),
                )
        elif tool is None:
            err = ToolNotFoundError(f"Tool '{tool_name}' not found")
            result = ToolResult(
                success=False,
                error=f"{err.__class__.__name__}: {err.message}",
            )
        else:
            # --- Permission check (secondary gate) ---
            # Tools might have been hidden from the model via PluginManager
            # filtering, but a stale assistant turn or dynamic tool could
            # still request a denied tool.  This secondary check catches
            # those edge cases and produces an audit record.
            permission_policy = self._plugin_manager.check_tool_permission_raw(tool_name)

            if permission_policy == PermissionPolicy.deny:
                # Hard deny — tool is blocked unconditionally.
                result = ToolResult(
                    success=False,
                    error=(
                        f"Permission denied for tool '{tool_name}': "
                        f"policy is '{permission_policy.value}'"
                    ),
                )
                self._record_permission_audit(
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    run_id=run_id,
                    policy=permission_policy,
                    allowed=False,
                )

            elif permission_policy == PermissionPolicy.human_review:
                # Human-in-the-loop review.
                # Phase 1 fallback: deny when no broker is available.
                result = await self._handle_human_review(
                    tool=tool,
                    tool_name=tool_name,
                    arguments=arguments,
                    tool_call_id=tool_call_id,
                    run_id=run_id,
                    iteration=iteration,
                )

            elif permission_policy == PermissionPolicy.agent_review:
                # Sub-agent review.
                # Phase 1 fallback: allow when no review mechanism available.
                result = await self._handle_agent_review(
                    tool=tool,
                    tool_name=tool_name,
                    arguments=arguments,
                    tool_call_id=tool_call_id,
                    run_id=run_id,
                    iteration=iteration,
                )

            else:  # PermissionPolicy.allow
                # --- Normal tool execution ---
                tool_owner = self._plugin_manager.get_tool_owner(tool_name)
                owner_plugin_id = getattr(tool_owner, "plugin_id", None)
                owner_plugin_name = getattr(tool_owner, "plugin_name", None)
                prepared = await self.prepare_tool_arguments(
                    tool,
                    arguments,
                    tool_call_id=tool_call_id,
                    run_id=run_id,
                    iteration=iteration,
                    run_injection_handlers=run_injection_handlers,
                )
                result_output_prefix = prepared.output_prefix
                if prepared.short_circuit_result is not None:
                    result = prepared.short_circuit_result
                elif getattr(tool, "audit", False) and audit_action == "queue":
                    self._context._add_pending_tool_call(
                        tool_call_id,
                        tool_name,
                        arguments,
                    )
                    audit_pending = True
                    result = ToolResult(
                        success=True,
                        output=(
                            f"[AUDIT PENDING] Tool '{tool_name}' has been submitted "
                            "for review. Use review_pending_tools() to check status "
                            "and approve/reject."
                        ),
                    )
                else:
                    context_param = getattr(tool, "context", None)
                    has_runtime_context = bool(
                        context_param and self._context.tool_call_context
                    )
                    tool_arguments = self.inject_tool_runtime_context(
                        tool,
                        prepared.tool_arguments,
                    )
                    if has_runtime_context:
                        await self._emit_event(
                            AgentToolRuntimeContextInjectedEvent.create(
                                run_id=run_id,
                                tool_name=tool_name,
                                tool_call_id=tool_call_id,
                                parameter_name=str(context_param),
                                plugin_id=owner_plugin_id,
                                plugin_name=owner_plugin_name,
                                plugin_role="tool_owner" if tool_owner is not None else "dynamic_tool",
                                injection_name=str(context_param),
                            ),
                            event_bus,
                        )
                    result = await self._execute_agent_tool(
                        tool,
                        tool_name,
                        tool_call_id,
                        tool_arguments,
                        run_id=run_id,
                        iteration=iteration,
                        event_bus=event_bus,
                    )

        duration_ms = (time.time() - start_time) * 1000
        result = self._prepend_tool_result_output(result, result_output_prefix)
        result = self._enforce_tool_result_size(
            result,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            run_id=run_id,
        )
        after_hook_result = await self._invoke_after_tool_calling(
            tool_name,
            arguments,
            result,
            HookContext(
                run_id=run_id,
                iteration=iteration,
                tool_call_id=tool_call_id,
                tool=tool,
                context=self._context,
                review=getattr(self._agent, "review_broker", None),
                duration_ms=duration_ms,
            ),
        )
        if after_hook_result and after_hook_result.action in {"abort", "reinvoke"}:
            control = after_hook_result

        record = ToolCallRecord(
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            duration_ms=duration_ms,
            tool_call_id=tool_call_id,
        )
        return ToolExecutionOutcome(
            record=record,
            audit_pending=audit_pending,
            result_content=self._render_tool_result(result),
            control=control,
        )

    def _synthesize_stopped_tool_results(
        self,
        tool_calls: list[ToolCallPart],
        *,
        run_id: str,
        iteration: int,
        reason: str,
    ) -> list[ToolExecutionOutcome]:
        """Create error tool results for unexecuted calls in the same batch.

        Providers require every assistant tool_call to receive a matching
        tool_result. When a hook stops a batch early, these synthetic results
        preserve that protocol without running the remaining tools.
        """
        outcomes: list[ToolExecutionOutcome] = []
        for tool_call in tool_calls:
            result = ToolResult(
                success=False,
                error=f"Tool call skipped because the tool batch stopped: {reason}",
            )
            record = ToolCallRecord(
                tool_name=tool_call["name"],
                arguments=dict(tool_call["arguments"]),
                result=result,
                duration_ms=0.0,
                tool_call_id=tool_call["id"],
            )
            outcomes.append(
                ToolExecutionOutcome(
                    record=record,
                    audit_pending=False,
                    result_content=self._render_tool_result(result),
                )
            )
        return outcomes

    @staticmethod
    def _control_reason(control: HookResult) -> str:
        if control.action == "reinvoke":
            return "after_tool_calling requested reinvoke"
        return control.reason or f"{control.action} requested by hook"

    async def _execute_agent_tool(
        self,
        tool: AgentTool,
        tool_name: str,
        tool_call_id: str,
        tool_arguments: dict[str, Any],
        *,
        run_id: str,
        iteration: int,
        event_bus: EventBus | None,
    ) -> ToolResult:
        try:
            is_valid, errors = tool.validate_parameters(tool_arguments)
            if not is_valid:
                return ToolResult(
                    success=False,
                    error=f"Parameter validation failed: {'; '.join(errors)}",
                )

            owner = self._plugin_manager.get_tool_owner(tool_name)
            event_scope = (
                owner.plugin_event_context(
                    run_id=run_id,
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    iteration=iteration,
                )
                if owner is not None
                else contextlib.nullcontext()
            )
            with event_scope:
                raw_result = await tool.arun(**tool_arguments)

                if inspect.isasyncgen(raw_result):
                    parts: list[str] = []
                    final_result: ToolResult | None = None
                    async_gen = cast(AsyncGenerator[Any, None], raw_result)
                    async for part in async_gen:
                        if isinstance(part, ToolResult):
                            final_result = part
                            continue
                        part_text = str(part)
                        parts.append(part_text)
                        await self._emit_event(
                            AgentToolResultPartEvent.create(
                                run_id=run_id,
                                tool_call_id=tool_call_id,
                                part=part_text,
                                part_index=len(parts) - 1,
                                is_final=False,
                            ),
                            event_bus,
                        )
                    await self._emit_event(
                        AgentToolResultPartEvent.create(
                            run_id=run_id,
                            tool_call_id=tool_call_id,
                            part="",
                            part_index=len(parts),
                            is_final=True,
                        ),
                        event_bus,
                    )
                    if final_result is not None:
                        if final_result.output is None and parts:
                            final_result.output = "".join(parts)
                        return final_result
                    return ToolResult(success=True, output="".join(parts))

                if isinstance(raw_result, ToolResult):
                    return raw_result
                return ToolResult(success=True, output=raw_result)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            err = ToolExecutionError(
                f"Tool '{tool_name}' execution failed: {e}",
                details={"original": e},
            )
            return ToolResult(
                success=False,
                error=f"{err.__class__.__name__}: {err.message}",
            )

    def _enforce_tool_result_size(
        self,
        result: ToolResult,
        *,
        tool_name: str,
        tool_call_id: str,
        run_id: str,
    ) -> ToolResult:
        serialized = self._serialize_tool_result_for_limit(result)
        size_bytes = len(serialized.encode("utf-8"))
        if size_bytes <= TOOL_RESULT_MAX_BYTES:
            return result

        message = (
            f"Tool result from '{tool_name}' ({tool_call_id}) is {size_bytes} bytes, "
            f"exceeding limit {TOOL_RESULT_MAX_BYTES} bytes"
        )
        logger.warning("%s; returning truncated tool result", message)
        return self._truncated_tool_result(
            result,
            size_bytes=size_bytes,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            run_id=run_id,
        )

    @staticmethod
    def _serialize_tool_result_for_limit(result: ToolResult) -> str:
        try:
            return json.dumps(
                {
                    "success": result.success,
                    "output": result.output,
                    "error": result.error,
                },
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
        except Exception:
            return repr(result)

    @staticmethod
    def _truncate_utf8(text: str, max_bytes: int) -> str:
        if len(text.encode("utf-8")) <= max_bytes:
            return text
        return ToolExecutor._truncate_utf8_with_suffix(
            text,
            max_bytes,
            TOOL_RESULT_TRUNCATION_SUFFIX,
        )

    @staticmethod
    def _truncate_utf8_with_suffix(text: str, max_bytes: int, suffix: str) -> str:
        encoded = text.encode("utf-8")
        suffix_bytes = suffix.encode("utf-8")
        if max_bytes <= len(suffix_bytes):
            return suffix[:max(max_bytes, 0)]
        prefix = encoded[: max_bytes - len(suffix_bytes)]
        return prefix.decode("utf-8", errors="ignore") + suffix

    @staticmethod
    def _tool_output_text(output: Any) -> str:
        if output is None:
            return ""
        if isinstance(output, str):
            return output
        try:
            return json.dumps(output, ensure_ascii=False, indent=2, default=str)
        except Exception:
            return str(output)

    def _truncated_tool_result(
        self,
        result: ToolResult,
        *,
        size_bytes: int,
        tool_name: str,
        tool_call_id: str,
        run_id: str,
    ) -> ToolResult:
        warning_message = (
            f"{TOOL_RESULT_TRUNCATION_WARNING} "
            f"Original size: {size_bytes} bytes; limit: {TOOL_RESULT_MAX_BYTES} bytes; "
            f"tool: {tool_name}; tool_call_id: {tool_call_id}; run_id: {run_id}."
        )
        warning_suffix = f"\n\n[Hawi warning: {warning_message}]"
        source_text = self._tool_output_text(result.output)
        error_text = result.error
        max_payload_bytes = max(
            0,
            TOOL_RESULT_MAX_BYTES
            - len(warning_suffix.encode("utf-8"))
            - len(error_text.encode("utf-8"))
            - 512,
        )
        output = self._truncate_utf8_with_suffix(
            source_text,
            max_payload_bytes,
            warning_suffix,
        )
        candidate = ToolResult(
            success=result.success,
            output=output,
            error=error_text,
            cache_point=result.cache_point,
            cache_point_source=result.cache_point_source,
        )
        for _ in range(8):
            if (
                len(self._serialize_tool_result_for_limit(candidate).encode("utf-8"))
                <= TOOL_RESULT_MAX_BYTES
            ):
                return candidate
            max_payload_bytes = max(0, max_payload_bytes - 512)
            candidate.output = self._truncate_utf8_with_suffix(
                source_text,
                max_payload_bytes,
                warning_suffix,
            )

        if candidate.error:
            candidate.error = self._truncate_utf8_with_suffix(
                candidate.error,
                1024,
                "\n\n[truncated]",
            )
        if (
            len(self._serialize_tool_result_for_limit(candidate).encode("utf-8"))
            <= TOOL_RESULT_MAX_BYTES
        ):
            return candidate

        return ToolResult(
            success=result.success,
            output=warning_suffix.strip(),
            error="",
            cache_point=result.cache_point,
            cache_point_source=result.cache_point_source,
        )

    async def _commit_outcomes(
        self,
        outcomes: list[ToolExecutionOutcome],
        *,
        run_id: str,
        event_bus: EventBus | None,
        materialize_pending_steer: bool,
    ) -> None:
        completed_tool_call_ids: list[str] = []
        for outcome in outcomes:
            record = outcome.record
            completed_tool_call_ids.append(record.tool_call_id)
            if outcome.audit_pending:
                continue

            added_messages = self._add_tool_result(
                tool_call_id=record.tool_call_id,
                content=outcome.result_content,
                is_error=not record.result.success,
                materialize_pending_steer=False,
                cache_point=getattr(record.result, "cache_point", None),
                cache_point_source=getattr(record.result, "cache_point_source", None),
            )
            outcome.context_message_id = added_messages.context_message_id
            await self._emit_tool_result_message(
                run_id=run_id,
                tool_call_id=record.tool_call_id,
                content=outcome.result_content,
                is_error=not record.result.success,
                context_message_id=added_messages.context_message_id,
                event_bus=event_bus,
            )
            if added_messages.materialized_messages:
                await self._emit_materialized_steer_events(
                    run_id,
                    added_messages.materialized_messages,
                    event_bus,
                )

        if materialize_pending_steer and completed_tool_call_ids:
            await self._materialize_pending_steer(
                completed_tool_call_ids,
                run_id=run_id,
                event_bus=event_bus,
            )

    async def _materialize_pending_steer(
        self,
        tool_call_ids: list[str],
        *,
        run_id: str,
        event_bus: EventBus | None,
    ) -> None:
        materialized_messages = self._agent._materialize_pending_steer_for_tool_results(
            tool_call_ids
        )
        await self._emit_materialized_steer_events(
            run_id,
            materialized_messages,
            event_bus,
        )

    async def _emit_final_result_event(
        self,
        record: ToolCallRecord,
        run_id: str,
        event_bus: EventBus | None,
        context_message_id: str | None = None,
    ) -> None:
        await self._emit_event(
            AgentToolResultEvent.create(
                run_id=run_id,
                tool_call_id=record.tool_call_id,
                success=record.result.success,
                result_preview=str(record.result.output),
                duration_ms=record.duration_ms,
                result_obj=record.result,
                context_message_id=context_message_id,
            ),
            event_bus,
        )

    def _register_active_tool_calls(
        self,
        tool_calls: list[ToolCallPart],
    ) -> list[ToolCallPart]:
        current_tool_calls = self._active_tool_call_list()
        if current_tool_calls is None:
            return []
        active_batch_tool_calls = [
            tool_call
            for tool_call in tool_calls
            if tool_call not in current_tool_calls
        ]
        current_tool_calls.extend(active_batch_tool_calls)
        return active_batch_tool_calls

    def _unregister_active_tool_calls(
        self,
        active_batch_tool_calls: list[ToolCallPart],
    ) -> None:
        current_tool_calls = self._active_tool_call_list()
        if current_tool_calls is None:
            return
        for tool_call in active_batch_tool_calls:
            if tool_call in current_tool_calls:
                current_tool_calls.remove(tool_call)

    def _active_tool_call_list(self) -> list[ToolCallPart] | None:
        return getattr(self._agent, "_current_tool_calls", self._current_tool_calls)

    def _record_permission_audit(
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        run_id: str,
        policy: "PermissionPolicy",
        allowed: bool,
    ) -> None:
        """Record a permission decision to the agent's audit sink."""
        from hawi.permission import PermissionAuditSink

        sink: PermissionAuditSink | None = getattr(
            self._agent, "_permission_audit_sink", None
        )
        if sink is None:
            return

        from hawi.permission.types import (
            PermissionAuditRecord,
            PermissionId,
            PermissionPolicy,
        )

        pid = PermissionId(f"tool:{tool_name}")
        decision = "allowed" if allowed else "denied"
        sink.record(
            PermissionAuditRecord(
                permission_id=pid,
                tool_name=tool_name,
                effective_policy=policy,
                decision=decision,
                tool_call_id=tool_call_id,
                run_id=run_id,
            )
        )

    async def _handle_human_review(
        self,
        *,
        tool: "AgentTool",
        tool_name: str,
        arguments: dict[str, Any],
        tool_call_id: str,
        run_id: str,
        iteration: int,
    ) -> ToolResult:
        """Handle a ``human_review`` permission policy.

        When a :class:`RuntimeReviewBroker` is available (via the agent's
        ``review_broker``), this method creates a review request, waits for
        the human to approve or reject, and then either executes the tool
        or returns a deny result.

        When no broker is available (standalone / headless usage), falls back
        to Phase 1 semantics: **deny**.
        """
        broker: RuntimeReviewBroker | None = getattr(
            self._agent, "review_broker", None
        )
        if broker is None:
            # Phase 1 fallback: deny
            self._record_permission_audit(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                run_id=run_id,
                policy=PermissionPolicy.human_review,
                allowed=False,
            )
            return ToolResult(
                success=False,
                error=(
                    f"Tool '{tool_name}' requires human review, but no "
                    "review broker is available (headless mode)."
                ),
            )

        import uuid
        review_id = f"perm-{tool_name}-{uuid.uuid4().hex[:8]}"
        broker.create(
            review_id,
            plugin_id="hawi/permission",
            review_type="human_review",
            payload={
                "kind": "permission_review",
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "arguments": arguments,
                "permission_policy": "human_review",
            },
        )

        # Emit review-requested event for GUI observability
        # The payload matches the human_review_request protocol that the
        # GUI's PluginMessageActions component already understands.
        from hawi.events import PluginEvent
        review_payload = {
            "event_name": "permission.review.requested",
            "kind": "human_review_request",       # ← GUI key
            "review_id": review_id,
            "plugin_id": "hawi/permission",       # ← GUI key
            "approve_action": "approve_permission_review",  # ← GUI key
            "reject_action": "reject_permission_review",    # ← GUI key
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "review_type": "human",
            "output_preview": str(arguments)[:400],
            "level": "info",
            "message": (
                f"Tool '{tool_name}' requires human review. "
                f"Arguments: {str(arguments)[:200]}"
            ),
        }
        await self._emit_event(
            PluginEvent.create(
                "plugin.event",
                plugin_name="Permission",
                plugin_id="hawi/permission",
                payload=review_payload,
            ),
            None,
        )

        try:
            raw_decision = await broker.wait(review_id)
            decision = self._normalize_review_decision(raw_decision)
        except asyncio.CancelledError:
            broker.discard(review_id)
            raise
        finally:
            broker.discard(review_id)

        if decision.approved:
            self._record_permission_audit(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                run_id=run_id,
                policy=PermissionPolicy.human_review,
                allowed=True,
            )
            # Execute the tool normally (approved by human)
            tool_owner = self._plugin_manager.get_tool_owner(tool_name)
            owner_plugin_id = getattr(tool_owner, "plugin_id", None)
            owner_plugin_name = getattr(tool_owner, "plugin_name", None)
            prepared = await self.prepare_tool_arguments(
                tool,
                arguments,
                tool_call_id=tool_call_id,
                run_id=run_id,
                iteration=iteration,
                run_injection_handlers=True,
            )
            if prepared.short_circuit_result is not None:
                return prepared.short_circuit_result
            tool_arguments = self.inject_tool_runtime_context(
                tool, prepared.tool_arguments
            )
            result = await self._execute_agent_tool(
                tool,
                tool_name,
                tool_call_id,
                tool_arguments,
                run_id=run_id,
                iteration=iteration,
                event_bus=None,
            )
            return self._prepend_tool_result_output(result, prepared.output_prefix)
        else:
            self._record_permission_audit(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                run_id=run_id,
                policy=PermissionPolicy.human_review,
                allowed=False,
            )
            return ToolResult(
                success=False,
                error=(
                    f"Human review rejected tool '{tool_name}'. "
                    f"Feedback: {decision.feedback or 'no feedback provided'}"
                ),
            )

    async def _handle_agent_review(
        self,
        *,
        tool: "AgentTool",
        tool_name: str,
        arguments: dict[str, Any],
        tool_call_id: str,
        run_id: str,
        iteration: int,
    ) -> ToolResult:
        """Handle an ``agent_review`` permission policy.

        Phase 1 fallback: **allow** (execute the tool unconditionally).
        Future phases will route through a reviewer sub-agent.
        """
        self._record_permission_audit(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            run_id=run_id,
            policy=PermissionPolicy.agent_review,
            allowed=True,
        )
        # Phase 1: allow — execute normally
        tool_owner = self._plugin_manager.get_tool_owner(tool_name)
        owner_plugin_id = getattr(tool_owner, "plugin_id", None)
        owner_plugin_name = getattr(tool_owner, "plugin_name", None)
        prepared = await self.prepare_tool_arguments(
            tool,
            arguments,
            tool_call_id=tool_call_id,
            run_id=run_id,
            iteration=iteration,
            run_injection_handlers=True,
        )
        if prepared.short_circuit_result is not None:
            return prepared.short_circuit_result
        tool_arguments = self.inject_tool_runtime_context(
            tool, prepared.tool_arguments
        )
        result = await self._execute_agent_tool(
            tool,
            tool_name,
            tool_call_id,
            tool_arguments,
            run_id=run_id,
            iteration=iteration,
            event_bus=None,
        )
        return self._prepend_tool_result_output(result, prepared.output_prefix)

    @staticmethod
    def _normalize_review_decision(raw: Any) -> RuntimeReviewDecision:
        """Normalize a raw review result into a RuntimeReviewDecision."""
        if isinstance(raw, RuntimeReviewDecision):
            return raw
        approved = bool(getattr(raw, "approved", False))
        feedback = str(getattr(raw, "feedback", ""))
        return RuntimeReviewDecision(approved=approved, feedback=feedback)

    async def _invoke_before_tool_calling(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        ctx: HookContext,
    ) -> Any:
        for hook in self._plugin_manager.get_hooks("before_tool_calling"):
            result = hook(self._agent, tool_name, arguments, ctx)
            if inspect.isawaitable(result):
                result = await result
            if result is not None:
                return result
        return None

    async def _invoke_after_tool_calling(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        tool_result: ToolResult,
        ctx: HookContext,
    ) -> Any:
        for hook in self._plugin_manager.get_hooks("after_tool_calling"):
            result = hook(self._agent, tool_name, arguments, tool_result, ctx)
            if inspect.isawaitable(result):
                result = await result
            if result is not None:
                return result
        return None
