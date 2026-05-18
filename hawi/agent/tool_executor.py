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
import time
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
from hawi.tool.types import (
    AgentTool,
    ToolParameterInjectionContext,
    ToolResult,
)

from .context import AgentContext
from .result import ToolCallRecord

if TYPE_CHECKING:
    from .agent import HawiAgent


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
    ) -> list[Any]:
        ...


class EmitToolResultMessageCallback(Protocol):
    def __call__(
        self,
        *,
        run_id: str,
        tool_call_id: str,
        content: str | list[ContentPart],
        is_error: bool,
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


@dataclass
class ToolExecutionOutcome:
    """Internal result of one tool call before/after context persistence."""

    record: ToolCallRecord
    audit_pending: bool
    result_content: str
    control: HookResult | None = None


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

        outcome = await self._run_tool_call(
            tool_call,
            run_id=run_id,
            iteration=iteration,
            event_bus=event_bus,
            run_injection_handlers=run_injection_handlers,
            audit_action=audit_action,
        )
        if add_to_context:
            await self._commit_outcomes(
                [outcome],
                run_id=run_id,
                event_bus=event_bus,
                materialize_pending_steer=materialize_pending_steer,
            )
        if emit_final_event:
            await self._emit_final_result_event(outcome.record, run_id, event_bus)
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

        active_batch_tool_calls = self._register_active_tool_calls(tool_calls)
        outcomes: list[ToolExecutionOutcome] = []
        try:
            control: HookResult | None = None
            for index, tool_call in enumerate(tool_calls):
                if is_interrupted is not None and is_interrupted():
                    break

                outcome = await self._run_tool_call(
                    tool_call,
                    run_id=run_id,
                    iteration=iteration,
                    event_bus=event_bus,
                    run_injection_handlers=True,
                    audit_action="queue",
                )
                outcomes.append(outcome)

                if tool_call in active_batch_tool_calls:
                    self._unregister_active_tool_calls([tool_call])

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
                )
                if outcome.control is not None:
                    control = outcome.control
                    remaining = tool_calls[index + 1 :]
                    skipped = self._synthesize_stopped_tool_results(
                        remaining,
                        run_id=run_id,
                        iteration=iteration,
                        reason=self._control_reason(control),
                    )
                    outcomes.extend(skipped)
                    if skipped:
                        await self._commit_outcomes(
                            skipped,
                            run_id=run_id,
                            event_bus=event_bus,
                            materialize_pending_steer=False,
                        )
                        for skipped_outcome in skipped:
                            await self._emit_final_result_event(
                                skipped_outcome.record,
                                run_id,
                                event_bus,
                            )
                    break
        finally:
            self._unregister_active_tool_calls(active_batch_tool_calls)

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
        required = [injection.name for injection in injections if injection.required]
        if required:
            injected_schema["required"] = required

        is_valid, errors = validate_parameters(arguments, injected_schema)
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

        prepared = PreparedToolArguments(
            tool_arguments=tool_arguments,
            injected_arguments=injected_arguments,
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
                    async_gen = cast(AsyncGenerator[Any, None], raw_result)
                    async for part in async_gen:
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

            materialized_messages = self._add_tool_result(
                tool_call_id=record.tool_call_id,
                content=outcome.result_content,
                is_error=not record.result.success,
                materialize_pending_steer=False,
                cache_point=getattr(record.result, "cache_point", None),
                cache_point_source=getattr(record.result, "cache_point_source", None),
            )
            await self._emit_tool_result_message(
                run_id=run_id,
                tool_call_id=record.tool_call_id,
                content=outcome.result_content,
                is_error=not record.result.success,
                event_bus=event_bus,
            )
            if materialized_messages:
                await self._emit_materialized_steer_events(
                    run_id,
                    materialized_messages,
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
    ) -> None:
        await self._emit_event(
            AgentToolResultEvent.create(
                run_id=run_id,
                tool_call_id=record.tool_call_id,
                success=record.result.success,
                result_preview=str(record.result.output),
                duration_ms=record.duration_ms,
                result_obj=record.result,
            ),
            event_bus,
        )

    def _register_active_tool_calls(
        self,
        tool_calls: list[ToolCallPart],
    ) -> list[ToolCallPart]:
        if self._current_tool_calls is None:
            return []
        active_batch_tool_calls = [
            tool_call
            for tool_call in tool_calls
            if tool_call not in self._current_tool_calls
        ]
        self._current_tool_calls.extend(active_batch_tool_calls)
        return active_batch_tool_calls

    def _unregister_active_tool_calls(
        self,
        active_batch_tool_calls: list[ToolCallPart],
    ) -> None:
        if self._current_tool_calls is None:
            return
        for tool_call in active_batch_tool_calls:
            if tool_call in self._current_tool_calls:
                self._current_tool_calls.remove(tool_call)

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
