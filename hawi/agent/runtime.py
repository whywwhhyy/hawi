"""Interrupt, steer, and runtime snapshot component for HawiAgent."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, cast

from hawi.errors import ContextLengthError
from hawi.events import AgentMessageAddedEvent, AgentToolResultEvent, EventBus
from hawi.models import CachePoint, ContentPart
from hawi.tool.types import ToolResult

from .content_utils import (
    merge_content_parts,
    normalize_content_parts,
    serialize_content_parts,
    tool_result_content,
    truncate_preview,
)
from .context_retry import (
    context_retry_needed_reduction_chars,
    context_retry_tool_result_target_chars,
    truncate_tool_result_for_retry,
)
from .result import AgentRunResult, ToolCallRecord
from .state import (
    AddedToolResultMessages,
    MaterializedSteerMessage,
    PendingInput,
    SteerPartMergeMode,
    _RecentToolResult,
)


class AgentRuntime:
    """Explicit runtime component owned by HawiAgent."""

    def __init__(self, owner: Any) -> None:
        self._owner = owner

    def interrupt(self, reason: str = "user") -> list[str]:
        """Interrupt current agent execution."""
        agent = self._owner
        self.on_interrupt(reason)
        agent._cancel_event.set()
        agent._last_interrupt_reason = reason
        interrupted_ids = [tc.get("id", "") for tc in agent._current_tool_calls]
        agent._interrupted_tool_call_ids.extend(interrupted_ids)
        return interrupted_ids

    def on_interrupt(self, reason: str = "user") -> None:
        """Interrupt hook (no-op by default)."""
        return None

    def clear_interrupt_state(self) -> None:
        """Clear interrupt state for a fresh execution."""
        agent = self._owner
        tool_executor = getattr(agent, "_tool_executor", None)
        if tool_executor is not None:
            tool_executor.clear()
        agent._cancel_event.clear()
        agent._interrupted_tool_call_ids.clear()
        agent._current_tool_calls.clear()

    def check_interrupt(self) -> bool:
        """Check if an interrupt has been requested."""
        return self._owner._cancel_event.is_set()

    @staticmethod
    def interrupt_tool_result_content(reason: str) -> str:
        return f"Tool call interrupted before completion (reason: {reason})."

    async def recover_unanswered_tool_calls(
        self,
        *,
        run_id: str | None,
        event_bus: EventBus | None,
        reason: str,
        emit_events: bool,
    ) -> None:
        agent = self._owner
        content = self.interrupt_tool_result_content(reason)
        recovered = agent._context.add_missing_tool_results(content)
        if recovered:
            refresher = getattr(agent, "_refresh_context_usage_snapshot", None)
            if callable(refresher):
                refresher()
        agent._last_interrupt_reason = None
        if not emit_events or not run_id:
            return
        for item in recovered:
            normalized = self.normalize_content_parts(item.content)
            await agent._emit_event(
                AgentToolResultEvent.create(
                    run_id=run_id,
                    tool_call_id=item.tool_call_id,
                    success=False,
                    result_preview=content,
                    duration_ms=0.0,
                    result_obj=ToolResult(success=False, error=content),
                    context_message_id=item.context_message_id,
                ),
                event_bus,
            )
            await agent._emit_event(
                AgentMessageAddedEvent.create(
                    run_id=run_id,
                    role="tool",
                    content=[
                        {
                            "type": "tool_result",
                            "tool_call_id": item.tool_call_id,
                            "content": normalized,
                            "is_error": True,
                        }
                    ],
                    context_message_id=item.context_message_id,
                ),
                event_bus,
            )

    @property
    def has_active_tool_calls(self) -> bool:
        """Whether the agent is currently waiting on one or more tool calls."""
        return len(self._owner._current_tool_calls) > 0

    def snapshot_runtime(self) -> dict[str, Any]:
        """Capture in-flight run state for SessionManager persistence."""
        agent = self._owner
        state = agent._active_execution_state
        return {
            "version": 1,
            "active_run_id": state.run_id if state else None,
            "iteration": state.iteration if state else 0,
            "current_tool_calls": list(agent._current_tool_calls),
            "interrupted_tool_call_ids": list(agent._interrupted_tool_call_ids),
            "last_unsent_tool_results": [
                {
                    "tool_call_id": r.tool_call_id,
                    "tool_name": r.tool_name,
                    "content": r.content,
                    "is_error": r.is_error,
                    "truncate_attempts": r.truncate_attempts,
                }
                for r in agent._last_unsent_tool_results
            ],
            "last_interrupt_reason": agent._last_interrupt_reason,
            "tool_executor": (
                agent._tool_executor.snapshot()
                if hasattr(agent, "_tool_executor")
                else {"version": 1, "queue": [], "requests": []}
            ),
        }

    def load_runtime(self, data: dict[str, Any]) -> None:
        """Restore in-flight runtime state from :py:meth:`snapshot_runtime`."""
        agent = self._owner
        version = data.get("version", 1)
        if version != 1:
            raise ValueError(f"Unsupported runtime snapshot version: {version}")
        tool_executor = getattr(agent, "_tool_executor", None)
        if tool_executor is not None:
            tool_executor.clear()
        agent._current_tool_calls = list(data.get("current_tool_calls", []))
        agent._interrupted_tool_call_ids = list(
            data.get("interrupted_tool_call_ids", [])
        )
        agent._last_unsent_tool_results = [
            _RecentToolResult(
                tool_call_id=entry["tool_call_id"],
                tool_name=entry["tool_name"],
                content=entry.get("content", ""),
                is_error=entry.get("is_error", False),
                truncate_attempts=entry.get("truncate_attempts", 0),
            )
            for entry in data.get("last_unsent_tool_results", [])
        ]
        agent._last_interrupt_reason = data.get("last_interrupt_reason")

    def snapshot_steer(self) -> list[dict[str, Any]]:
        """Capture pending steer inputs."""
        agent = self._owner
        with agent._steer_lock:
            return [
                {
                    "id": p.id,
                    "content": p.content,
                    "candidate_tool_call_ids": list(p.candidate_tool_call_ids),
                    "created_at": p.created_at,
                    "preferred_merge_mode": (
                        p.preferred_merge_mode.value
                        if p.preferred_merge_mode is not None
                        else None
                    ),
                }
                for p in agent._pending_inputs
            ]

    def load_steer(self, data: list[dict[str, Any]]) -> None:
        """Restore pending steer inputs."""
        agent = self._owner
        with agent._steer_lock:
            agent._pending_inputs = [
                PendingInput(
                    id=entry["id"],
                    content=entry.get("content", []),
                    candidate_tool_call_ids=tuple(
                        entry.get("candidate_tool_call_ids", [])
                    ),
                    created_at=entry.get("created_at", time.time()),
                    preferred_merge_mode=(
                        SteerPartMergeMode(entry["preferred_merge_mode"])
                        if entry.get("preferred_merge_mode")
                        else None
                    ),
                )
                for entry in data
            ]

    def steer(
        self,
        content: str | list[ContentPart],
        *,
        merge_mode: SteerPartMergeMode | None = None,
    ) -> str:
        """Queue steer content for later materialization."""
        agent = self._owner
        steer_content = self.normalize_content_parts(content)
        steer_id = str(uuid.uuid4())[:8]
        should_start_new_loop = False
        with agent._steer_lock:
            candidate_tool_call_ids = tuple(
                tc.get("id", "")
                for tc in agent._current_tool_calls
                if tc.get("id")
            )
            agent._pending_inputs.append(
                PendingInput(
                    id=steer_id,
                    content=steer_content,
                    candidate_tool_call_ids=candidate_tool_call_ids,
                    created_at=time.time(),
                    preferred_merge_mode=merge_mode,
                )
            )
            if self.has_active_tool_calls:
                return steer_id
            with agent._session_lock:
                should_start_new_loop = not agent._session_active

        if should_start_new_loop:
            agent._ensure_pending_turn_loop()
        return steer_id

    def get_pending_input_messages(self) -> list[dict[str, Any]]:
        """Return read-only previews of pending steer inputs for observability."""
        agent = self._owner
        with agent._steer_lock:
            pending_inputs = list(agent._pending_inputs)
        return [
            {
                "id": pending.id,
                "queue": "high_prio",
                "content_preview": self.truncate_preview(
                    self.serialize_content_parts(pending.content),
                    240,
                ),
                "created_at": pending.created_at,
                "metadata": {
                    "candidate_tool_call_ids": list(pending.candidate_tool_call_ids),
                    "merge_mode": (
                        pending.preferred_merge_mode.value
                        if pending.preferred_merge_mode is not None
                        else None
                    ),
                },
            }
            for pending in pending_inputs
        ]

    def has_pending_inputs(self) -> bool:
        """Return whether queued steer inputs still need materialization."""
        agent = self._owner
        with agent._steer_lock:
            return bool(agent._pending_inputs)

    normalize_content_parts = staticmethod(normalize_content_parts)
    truncate_preview = staticmethod(truncate_preview)
    serialize_content_parts = staticmethod(serialize_content_parts)
    tool_result_content = staticmethod(tool_result_content)

    def recent_tool_result_from_record(
        self,
        record: ToolCallRecord,
    ) -> _RecentToolResult:
        return _RecentToolResult(
            tool_call_id=record.tool_call_id,
            tool_name=record.tool_name,
            content=self.tool_result_content(record.result),
            is_error=not record.result.success,
        )

    def mark_tool_results_unsent(
        self,
        records: list[ToolCallRecord],
    ) -> None:
        """Remember tool results that still need a successful model round-trip."""
        self._owner._last_unsent_tool_results = [
            self.recent_tool_result_from_record(record)
            for record in records
        ]

    async def truncate_last_unsent_tool_results_for_context_retry(
        self,
        error: ContextLengthError,
    ) -> bool:
        """Shrink the longest latest tool results before retrying a model call."""
        agent = self._owner
        if not agent._last_unsent_tool_results:
            return False

        candidates = sorted(
            (
                item
                for item in agent._last_unsent_tool_results
                if len(item.content) > 1
            ),
            key=lambda item: len(item.content),
            reverse=True,
        )
        if not candidates:
            return False

        needed_reduction = self.context_retry_needed_reduction_chars(error)
        selected: list[_RecentToolResult] = []
        planned_reduction = 0
        for item in candidates:
            selected.append(item)
            target_chars = self.context_retry_tool_result_target_chars(
                len(item.content),
                error,
                attempt=item.truncate_attempts,
            )
            planned_reduction += max(1, len(item.content) - target_chars)
            if needed_reduction is None or planned_reduction >= needed_reduction:
                break

        changed = False
        for item in selected:
            truncated = self.truncate_tool_result_for_retry(
                item.content,
                error,
                attempt=item.truncate_attempts,
            )
            if truncated == item.content:
                continue
            if not agent._context.replace_tool_result_content(
                item.tool_call_id,
                truncated,
                is_error=item.is_error,
            ):
                continue
            item.content = truncated
            item.truncate_attempts += 1
            changed = True
        return changed

    truncate_tool_result_for_retry = staticmethod(truncate_tool_result_for_retry)
    context_retry_tool_result_target_chars = staticmethod(
        context_retry_tool_result_target_chars
    )
    context_retry_needed_reduction_chars = staticmethod(
        context_retry_needed_reduction_chars
    )

    async def drain_pending_inputs_to_context(
        self,
        run_id: str,
        event_bus: EventBus | None,
    ) -> bool:
        """Move queued pending inputs into the conversation as plain user messages."""
        agent = self._owner
        with agent._steer_lock:
            pending_inputs = agent._pending_inputs[:]
            agent._pending_inputs.clear()

        if not pending_inputs:
            return False

        merged_content = merge_content_parts(
            pending.content for pending in pending_inputs
        )
        metadata = {
            "message_id": pending_inputs[0].id,
            "queue": "normal",
            "display_message_type": "normal",
            "source_queue": "high_prio",
            "materialized_as": "plain_user_message",
        }
        if len(pending_inputs) > 1:
            metadata["merged_message_ids"] = [
                pending.id for pending in pending_inputs
            ]
            metadata["merged_message_count"] = len(pending_inputs)
        context_message_id = agent._context.add_user_message(
            merged_content,
            metadata=metadata,
        )
        refresher = getattr(agent, "_refresh_context_usage_snapshot", None)
        if callable(refresher):
            refresher()
        await agent._emit_event(
            AgentMessageAddedEvent.create(
                run_id=run_id,
                role="user",
                content=merged_content,
                metadata=metadata,
                context_message_id=context_message_id,
            ),
            event_bus,
        )
        return True

    def clear_autonomous_run_task(self, task: asyncio.Task[AgentRunResult]) -> None:
        """Drop the autonomous run task reference after completion."""
        agent = self._owner
        with agent._session_lock:
            if agent._autonomous_run_task is task:
                agent._autonomous_run_task = None

    async def run_pending_turns(self) -> AgentRunResult:
        """Execute queued turns using the agent's current configuration."""
        return await self._owner._arun_internal(message=None)

    def ensure_pending_turn_loop(self) -> None:
        """Start a new loop to process queued pending turns when idle."""
        agent = self._owner
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(agent._run_pending_turns())
            return

        with agent._session_lock:
            if agent._session_active:
                return
            if (
                agent._autonomous_run_task is not None
                and not agent._autonomous_run_task.done()
            ):
                return
            task = loop.create_task(agent._run_pending_turns())
            task.add_done_callback(agent._clear_autonomous_run_task)
            agent._autonomous_run_task = task

    def add_tool_result_with_pending_steer(
        self,
        tool_call_id: str,
        content: str | list[ContentPart],
        *,
        is_error: bool = False,
        materialize_pending_steer: bool = True,
        cache_point: CachePoint | dict[str, Any] | bool | None = None,
        cache_point_source: str | None = None,
    ) -> AddedToolResultMessages:
        """Add a tool result and materialize one matching pending input as steer."""
        agent = self._owner
        tool_result_content = self.normalize_content_parts(content)
        context_message_id = agent._context.add_tool_result(
            tool_call_id=tool_call_id,
            content=tool_result_content,
            is_error=is_error,
            cache_point=cache_point,
            cache_point_source=cache_point_source,
        )
        refresher = getattr(agent, "_refresh_context_usage_snapshot", None)
        if callable(refresher):
            refresher()

        if materialize_pending_steer:
            return AddedToolResultMessages(
                context_message_id=context_message_id,
                materialized_messages=self.materialize_pending_steer_for_tool_results(
                    [tool_call_id]
                ),
            )
        return AddedToolResultMessages(context_message_id=context_message_id)

    async def emit_tool_result_message_event(
        self,
        *,
        run_id: str,
        tool_call_id: str,
        content: str | list[ContentPart],
        is_error: bool,
        context_message_id: str,
        event_bus: EventBus | None,
    ) -> None:
        """Emit AgentMessageAddedEvent for a persisted tool result."""
        agent = self._owner
        normalized = self.normalize_content_parts(content)
        await agent._emit_event(
            AgentMessageAddedEvent.create(
                run_id=run_id,
                role="tool",
                content=[
                    {
                        "type": "tool_result",
                        "tool_call_id": tool_call_id,
                        "content": normalized,
                        "is_error": is_error,
                    }
                ],
                context_message_id=context_message_id,
            ),
            event_bus,
        )

    def materialize_pending_steer_for_tool_results(
        self,
        tool_call_ids: list[str],
    ) -> list[MaterializedSteerMessage]:
        """Append pending steer messages after a completed tool-result batch."""
        agent = self._owner
        if not tool_call_ids:
            return []

        tool_call_id_set = set(tool_call_ids)
        fallback_tool_call_id = tool_call_ids[0]
        materialized: list[tuple[PendingInput, str]] = []
        with agent._steer_lock:
            remaining: list[PendingInput] = []
            for item in agent._pending_inputs:
                matched_tool_call_id = next(
                    (
                        candidate
                        for candidate in item.candidate_tool_call_ids
                        if candidate in tool_call_id_set
                    ),
                    None,
                )
                if matched_tool_call_id is None and not item.candidate_tool_call_ids:
                    matched_tool_call_id = fallback_tool_call_id
                if matched_tool_call_id is None:
                    remaining.append(item)
                    continue
                materialized.append((item, matched_tool_call_id))
            agent._pending_inputs = remaining

        grouped: dict[str, list[PendingInput]] = {}
        for pending_input, matched_tool_call_id in materialized:
            grouped.setdefault(matched_tool_call_id, []).append(pending_input)

        materialized_messages: list[MaterializedSteerMessage] = []
        for matched_tool_call_id, pending_group in grouped.items():
            first_pending = pending_group[0]
            merge_modes = {item.preferred_merge_mode for item in pending_group}
            preferred_merge_mode = (
                first_pending.preferred_merge_mode
                if len(merge_modes) == 1
                else None
            )
            steer_part: ContentPart = {
                "type": "steer",
                "content": merge_content_parts(
                    item.content for item in pending_group
                ),
                "tool_call_id": matched_tool_call_id,
                "preferred_merge_mode": (
                    preferred_merge_mode.value
                    if preferred_merge_mode is not None
                    else None
                ),
            }
            content = [steer_part]
            metadata = self.steer_message_metadata(first_pending, matched_tool_call_id)
            if len(pending_group) > 1:
                metadata["merged_message_ids"] = [item.id for item in pending_group]
                metadata["merged_message_count"] = len(pending_group)
                if len(merge_modes) != 1:
                    metadata["merge_mode"] = None
            context_message_id = agent._context.add_user_message(
                content,
                metadata=metadata,
            )
            materialized_messages.append(
                MaterializedSteerMessage(
                    content=cast(list[ContentPart], content),
                    metadata=metadata,
                    context_message_id=context_message_id,
                )
            )
        if materialized_messages:
            refresher = getattr(agent, "_refresh_context_usage_snapshot", None)
            if callable(refresher):
                refresher()
        return materialized_messages

    @staticmethod
    def steer_message_metadata(
        pending_input: PendingInput,
        matched_tool_call_id: str,
    ) -> dict[str, Any]:
        return {
            "message_id": pending_input.id,
            "queue": "high_prio",
            "display_message_type": "steer",
            "source_queue": "high_prio",
            "materialized_as": "steer",
            "tool_call_id": matched_tool_call_id,
            "merge_mode": (
                pending_input.preferred_merge_mode.value
                if pending_input.preferred_merge_mode is not None
                else None
            ),
        }

    async def emit_materialized_steer_events(
        self,
        run_id: str,
        materialized_messages: list[MaterializedSteerMessage],
        event_bus: EventBus | None,
    ) -> None:
        agent = self._owner
        for message in materialized_messages:
            await agent._emit_event(
                AgentMessageAddedEvent.create(
                    run_id=run_id,
                    role="user",
                    content=message.content,
                    metadata=message.metadata,
                    context_message_id=message.context_message_id,
                ),
                event_bus,
            )
