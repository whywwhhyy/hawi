from __future__ import annotations

from typing import Any

from hawi.plugin import (
    HawiPlugin,
    HookResult,
    after_conversation,
    after_tool_calling,
    before_session,
    before_tool_calling,
    tool,
)
from hawi.tool import ToolResult

from .engine import PlanEngine
from .models import (
    PLAN_ITEM_COMPLETION_MODES,
    PLAN_ITEM_DEFAULT_COMPLETION_MODE,
    PLAN_ITEM_DEFAULT_STATUS,
    PLAN_ITEM_STATUSES,
    PlanEngineResult,
    PlanFoldRecord,
    PlanItem,
)
from .presentation import (
    ADD_PLAN_ITEMS_DESCRIPTION,
    ADD_PLAN_ITEMS_SCHEMA,
    COMPLETE_PLAN_ITEM_DESCRIPTION,
    COMPLETE_PLAN_ITEM_SCHEMA,
    LIST_PLAN_ITEMS_SCHEMA,
    PLAN_CONTROL_SCHEMA,
    PLAN_PROMPT_BEGIN,
    PLAN_PROMPT_END,
    PLAN_REMINDER_BEGIN,
    PLAN_REMINDER_END,
    RECALL_COMPLETED_TASK_SCHEMA,
    UPDATE_PLAN_ITEMS_SCHEMA,
    build_plan_prompt,
    format_completion_error_markdown,
    format_completion_markdown,
    format_runtime_reminder,
    gui_config_schema,
    gui_default_config,
)


PLAN_COMPLETION_CACHE_POINT_SOURCE = "plan.complete_plan_item"

__all__ = [
    "PLAN_ITEM_COMPLETION_MODES",
    "PLAN_ITEM_DEFAULT_COMPLETION_MODE",
    "PLAN_ITEM_DEFAULT_STATUS",
    "PLAN_ITEM_STATUSES",
    "PLAN_PROMPT_BEGIN",
    "PLAN_PROMPT_END",
    "PLAN_REMINDER_BEGIN",
    "PLAN_REMINDER_END",
    "PlanFoldRecord",
    "PlanItem",
    "PlanPlugin",
]


class PlanPlugin(HawiPlugin):
    """Plan mode plugin.

    Provides tools for maintaining an explicit task plan and a conversation hook
    that nudges the agent to continue while open plan items remain.
    """

    def __init__(self, fold_completed_tasks: bool = False) -> None:
        self._engine = PlanEngine(fold_completed_tasks=fold_completed_tasks)

    @property
    def _items(self) -> list[PlanItem]:
        return self._engine.items

    @_items.setter
    def _items(self, value: list[PlanItem]) -> None:
        self._engine.items = value

    @property
    def _next_item_number(self) -> int:
        return self._engine.next_item_number

    @_next_item_number.setter
    def _next_item_number(self, value: int) -> None:
        self._engine.next_item_number = value

    @property
    def _plan_paused(self) -> bool:
        return self._engine.plan_paused

    @_plan_paused.setter
    def _plan_paused(self, value: bool) -> None:
        self._engine.plan_paused = value

    @property
    def _pause_reason(self) -> str:
        return self._engine.pause_reason

    @_pause_reason.setter
    def _pause_reason(self, value: str) -> None:
        self._engine.pause_reason = value

    @property
    def _fold_completed_tasks(self) -> bool:
        return self._engine.fold_completed_tasks

    @_fold_completed_tasks.setter
    def _fold_completed_tasks(self, value: bool) -> None:
        self._engine.fold_completed_tasks = bool(value)

    @property
    def _fold_records(self) -> list[PlanFoldRecord]:
        return self._engine.fold_records

    @_fold_records.setter
    def _fold_records(self, value: list[PlanFoldRecord]) -> None:
        self._engine.fold_records = value

    @property
    def _next_fold_number(self) -> int:
        return self._engine.next_fold_number

    @_next_fold_number.setter
    def _next_fold_number(self, value: int) -> None:
        self._engine.next_fold_number = value

    @property
    def _active_completion_tool_call_id(self) -> str | None:
        return self._engine.active_completion_tool_call_id

    @_active_completion_tool_call_id.setter
    def _active_completion_tool_call_id(self, value: str | None) -> None:
        self._engine.active_completion_tool_call_id = value

    @classmethod
    def gui_config_schema(cls) -> dict:
        return gui_config_schema()

    @classmethod
    def gui_default_config(cls) -> dict:
        return gui_default_config()

    def clone(self) -> "PlanPlugin":
        new_plugin = PlanPlugin(
            fold_completed_tasks=self._engine.fold_completed_tasks
        )
        new_plugin._engine = self._engine.clone()
        return new_plugin

    def save_state(self) -> dict[str, Any]:
        """Capture the engine's persistent state for SessionManager."""
        return {
            "fold_completed_tasks": self._engine.fold_completed_tasks,
            "items": [item.to_dict() for item in self._engine.items],
            "next_item_number": self._engine.next_item_number,
            "plan_paused": self._engine.plan_paused,
            "pause_reason": self._engine.pause_reason,
            "fold_records": [
                {
                    "fold_id": r.fold_id,
                    "item_id": r.item_id,
                    "item_content": r.item_content,
                    "summary": r.summary,
                    "messages": r.messages,
                    "completed_item_ids": list(r.completed_item_ids),
                    "created_at": r.created_at,
                    "handoff_notes": r.handoff_notes,
                }
                for r in self._engine.fold_records
            ],
            "next_fold_number": self._engine.next_fold_number,
            "active_completion_tool_call_id": (
                self._engine.active_completion_tool_call_id
            ),
        }

    def load_state(self, data: dict[str, Any]) -> None:
        """Restore the engine from a :py:meth:`save_state` payload."""
        engine = self._engine
        engine.fold_completed_tasks = bool(data.get("fold_completed_tasks", False))
        engine.items = [
            PlanItem(
                id=item["id"],
                content=item.get("content", ""),
                parent_id=item.get("parent_id"),
                status=item.get(
                    "status",
                    "completed" if bool(item.get("completed", False)) else PLAN_ITEM_DEFAULT_STATUS,
                ),
                completed=bool(item.get("completed", False)),
                created_at=float(item.get("created_at", 0.0)),
                completed_at=item.get("completed_at"),
                completion_mode=item.get(
                    "completion_mode",
                    PLAN_ITEM_DEFAULT_COMPLETION_MODE,
                ),
                completion_summary=item.get("completion_summary"),
                status_reason=item.get("status_reason"),
            )
            for item in data.get("items", [])
        ]
        engine.next_item_number = int(data.get("next_item_number", 1))
        engine.plan_paused = bool(data.get("plan_paused", False))
        engine.pause_reason = str(data.get("pause_reason", ""))
        engine.fold_records = [
            PlanFoldRecord(
                fold_id=r["fold_id"],
                item_id=r["item_id"],
                item_content=r.get("item_content", ""),
                summary=r.get("summary", ""),
                messages=list(r.get("messages", [])),
                completed_item_ids=list(r.get("completed_item_ids", [])),
                created_at=float(r.get("created_at", 0.0)),
                handoff_notes=r.get("handoff_notes"),
            )
            for r in data.get("fold_records", [])
        ]
        engine.next_fold_number = int(data.get("next_fold_number", 1))
        engine.active_completion_tool_call_id = data.get(
            "active_completion_tool_call_id"
        )

    @before_session(system_prompt_variability="hardcoded")
    def inject_plan_instructions(self, agent: Any, ctx: Any) -> None:
        """Inject plan mode guidance into the system prompt."""
        prompt = build_plan_prompt(
            fold_completed_tasks=self._engine.fold_completed_tasks
        )
        system_prompt = list(agent.context.system_prompt or [])
        system_prompt = [
            part
            for part in system_prompt
            if not (
                isinstance(part, dict)
                and part.get("type") == "text"
                and PLAN_PROMPT_BEGIN in str(part.get("text", ""))
            )
        ]
        system_prompt.append({"type": "text", "text": prompt})
        agent.context.system_prompt = system_prompt

    @before_tool_calling
    def remember_completion_tool_call(
        self,
        agent: Any,
        tool_name: str,
        arguments: dict[str, Any],
        ctx: Any,
    ) -> None:
        if tool_name == "complete_plan_item":
            self._engine.active_completion_tool_call_id = getattr(
                ctx, "tool_call_id", None
            )

    @after_tool_calling
    def clear_completion_tool_call(
        self,
        agent: Any,
        tool_name: str,
        arguments: dict[str, Any],
        result: ToolResult,
        ctx: Any,
    ) -> None:
        if tool_name == "complete_plan_item":
            self._engine.active_completion_tool_call_id = None

    @after_conversation
    def notify_unfinished_plan(self, agent: Any, ctx: Any) -> HookResult | None:
        """Re-drive the agent while plan items remain unfinished."""
        if ctx.error is not None or self._engine.plan_paused:
            return None
        if not self._engine.items:
            return None

        incomplete = self._engine.incomplete_items()
        if not incomplete:
            self._sync_artifact(status=self._engine.plan_status())
            return None

        plan_text = self._engine.format_plan_list()
        self._sync_artifact(status="active")
        self.emit_message(
            "Plan has unfinished items; asking the agent to continue.",
            title="Plan reminder",
            data={"pending_count": len(incomplete)},
            run_id=ctx.run_id,
        )
        return HookResult.reinvoke(format_runtime_reminder(plan_text))

    @tool(
        name="add_plan_items",
        description=ADD_PLAN_ITEMS_DESCRIPTION,
        parameters_schema=ADD_PLAN_ITEMS_SCHEMA,
    )
    def add_plan_items(
        self,
        parent_id: str | None = None,
        items: list[dict[str, Any]] | None = None,
    ) -> ToolResult:
        result = self._engine.add_plan_items(
            parent_id=parent_id,
            items=items,
        )
        return self._finish_engine_result(result)

    @tool(
        name="complete_plan_item",
        description=COMPLETE_PLAN_ITEM_DESCRIPTION,
        context="ctx",
        parameters_schema=COMPLETE_PLAN_ITEM_SCHEMA,
    )
    def complete_plan_item(
        self,
        item_id: str | None = None,
        item_ids: list[str] | None = None,
        mark_all_children: bool = False,
        fold_context: bool = False,
        summary: str | None = None,
        handoff_notes: str | None = None,
        ctx: Any = None,
    ) -> ToolResult:
        result = self._engine.complete_plan_item(
            item_id,
            item_ids=item_ids,
            mark_all_children=mark_all_children,
            fold_context=fold_context,
            summary=summary,
            handoff_notes=handoff_notes,
        )
        if not result.success:
            return ToolResult(
                success=False,
                error=format_completion_error_markdown(result),
            )

        self._sync_artifact(status=self._engine.plan_status())
        self._emit_item_events(result)
        if result.fold_request is not None:
            folded_context = self._maybe_fold_completed_context(
                ctx,
                **result.fold_request,
            )
            result.output["folded_context"] = folded_context
        self._remove_previous_completion_cache_points(ctx)
        return ToolResult(
            success=True,
            output=format_completion_markdown(result.output),
            cache_point=True,
            cache_point_source=PLAN_COMPLETION_CACHE_POINT_SOURCE,
        )

    @tool(
        name="update_plan_items",
        description=(
            "Change one or more plan item statuses without marking them complete. "
            "Use blocked/deferred for parked work, canceled/obsolete for work that "
            "should no longer be done, and open to reopen items."
        ),
        parameters_schema=UPDATE_PLAN_ITEMS_SCHEMA,
    )
    def update_plan_items(
        self,
        item_id: str | None = None,
        item_ids: list[str] | None = None,
        status: str = "open",
        reason: str | None = None,
    ) -> ToolResult:
        result = self._engine.update_plan_items_status(
            item_id,
            item_ids=item_ids,
            status=status,
            reason=reason,
        )
        return self._finish_engine_result(result)

    @tool(
        name="list_plan_items",
        description="List the current plan items and notification state.",
        parameters_schema=LIST_PLAN_ITEMS_SCHEMA,
    )
    def list_plan_items(self) -> ToolResult:
        self._sync_artifact(status=self._engine.plan_status())
        return ToolResult(success=True, output=self._engine.state_dict())

    @tool(
        name="recall_completed_task",
        description=(
            "Read or search detailed messages that PlanPlugin folded out of active "
            "context for a completed plan item."
        ),
        parameters_schema=RECALL_COMPLETED_TASK_SCHEMA,
    )
    def recall_completed_task(
        self,
        item_id: str = "",
        fold_id: str | None = None,
        message_start: int | None = None,
        message_end: int | None = None,
        query: str | None = None,
        case_sensitive: bool = False,
        max_matches: int = 20,
        context_chars: int = 240,
        max_chars: int = 20000,
    ) -> ToolResult:
        normalized_item_id = item_id
        query_text = query.strip() if isinstance(query, str) else ""
        if query_text:
            result = self._engine.search_folded_contexts(
                query=query_text,
                item_id=normalized_item_id,
                fold_id=fold_id,
                case_sensitive=bool(case_sensitive),
                max_matches=max_matches,
                context_chars=context_chars,
                max_chars=max_chars,
            )
            return self._finish_read_result(result)

        if not (normalized_item_id or fold_id):
            return ToolResult(
                success=False,
                error=(
                    "Provide item_id or fold_id to read a folded task context, "
                    "or provide query to search folded task contexts."
                ),
            )
        record = self._engine.find_fold_record(
            item_id=normalized_item_id,
            fold_id=fold_id,
        )
        if record is None:
            if not self._engine.fold_records:
                return ToolResult(
                    success=False,
                    error="No completed task contexts have been folded yet.",
                )
            return ToolResult(
                success=False,
                error=(
                    "No folded context found for "
                    f"item_id={normalized_item_id!r}, fold_id={fold_id!r}."
                ),
            )
        selected_messages, selected_start, selected_end, range_error = (
            self._engine.select_folded_messages(
                record.messages,
                message_start=message_start,
                message_end=message_end,
            )
        )
        if range_error:
            return ToolResult(success=False, error=range_error)

        max_chars = self._engine.normalize_int(
            max_chars,
            default=20000,
            minimum=1,
            maximum=200000,
        )
        transcript = self._engine.format_folded_messages(
            selected_messages,
            start_index=selected_start,
        )
        transcript, truncated = self._engine.truncate_text(
            transcript,
            max_chars,
            marker=(
                "\n\n[Transcript truncated. Call recall_completed_task "
                "with a larger max_chars value for more detail.]"
            ),
        )
        return ToolResult(
            success=True,
            output={
                "mode": "read",
                "fold_id": record.fold_id,
                "item_id": record.item_id,
                "item_content": record.item_content,
                "summary": record.summary,
                "handoff_notes": record.handoff_notes,
                "completed_item_ids": record.completed_item_ids,
                "message_count": len(record.messages),
                "selected_message_count": len(selected_messages),
                "message_start": selected_start,
                "message_end": selected_end,
                "transcript": transcript,
                "truncated": truncated,
                "max_chars": max_chars,
            },
        )

    @tool(
        name="plan_control",
        description=(
            "Pause, continue, or clear plan execution. Pause temporarily exits "
            "plan state and stops automatic reminders; continue resumes plan "
            "execution; clear discards the current plan."
        ),
        parameters_schema=PLAN_CONTROL_SCHEMA,
    )
    def plan_control(self, action: str, reason: str | None = None) -> ToolResult:
        result = self._engine.control(action=action, reason=reason)
        return self._finish_engine_result(result)

    def _finish_engine_result(self, result: PlanEngineResult) -> ToolResult:
        if not result.success:
            return self._failed_tool_result(result)

        self._sync_artifact(status=self._engine.plan_status())
        self._emit_item_events(result)
        if result.plugin_event is not None:
            self._emit_engine_plugin_event(result.plugin_event)
        return ToolResult(success=True, output=result.output)

    def _finish_read_result(self, result: PlanEngineResult) -> ToolResult:
        if not result.success:
            return self._failed_tool_result(result)
        return ToolResult(success=True, output=result.output)

    @staticmethod
    def _failed_tool_result(result: PlanEngineResult) -> ToolResult:
        return ToolResult(
            success=False,
            output=result.output or None,
            error=result.error,
        )

    def _emit_item_events(self, result: PlanEngineResult) -> None:
        for event in result.item_events:
            self._emit_plan_item_update(event["action"], event["item"])

    def _emit_engine_plugin_event(self, payload: dict[str, Any]) -> None:
        self.emit_plugin_event(
            "plugin.event",
            {
                **payload,
                "state": self._engine.state_dict(),
            },
        )

    def _maybe_fold_completed_context(
        self,
        ctx: Any,
        *,
        item_id: str,
        item_content: str,
        summary: str,
        completed_item_ids: list[str],
        handoff_notes: str | None = None,
    ) -> dict[str, Any]:
        folded_context, record = self._engine.fold_completed_context(
            ctx,
            item_id=item_id,
            item_content=item_content,
            summary=summary,
            handoff_notes=handoff_notes,
            completed_item_ids=completed_item_ids,
        )
        if record is not None:
            self.emit_plugin_event(
                "plugin.event",
                {
                    "event_name": "plan.context.folded",
                    "fold": record.reference_dict(),
                    "title": "Plan context folded",
                    "message": (
                        f"Folded {len(record.messages)} message(s) for {item_id}."
                    ),
                    "data": {
                        "fold_id": record.fold_id,
                        "item_id": item_id,
                        "folded_message_count": len(record.messages),
                    },
                },
            )
        return folded_context

    @staticmethod
    def _remove_previous_completion_cache_points(ctx: Any) -> None:
        context = getattr(ctx, "context", None)
        remover = getattr(context, "remove_cache_points", None)
        if callable(remover):
            remover(source=PLAN_COMPLETION_CACHE_POINT_SOURCE)

    def _sync_artifact(self, *, status: str) -> None:
        self.upsert_artifact(
            "current-plan",
            artifact_type="plan",
            title="Current Plan",
            content=f"{self._engine.format_plan_artifact()}\n",
            language="markdown",
            mime_type="text/markdown",
            status=status,
            metadata={
                "pending_count": len(self._engine.incomplete_items()),
                "plan_paused": self._engine.plan_paused,
                "pause_reason": self._engine.pause_reason,
                "items": self._engine.tree_items(),
            },
        )

    def _emit_plan_item_update(
        self, action: str, item: PlanItem | dict[str, Any]
    ) -> None:
        item_data = item.to_dict() if isinstance(item, PlanItem) else item
        self.emit_plugin_event(
            "plugin.event",
            {
                "event_name": "plan.item.updated",
                "action": action,
                "item": item_data,
                "state": self._engine.state_dict(),
                "title": "Plan updated",
                "message": f"{action.capitalize()} {item_data.get('id', 'plan item')}",
                "data": item_data,
            },
        )
