from __future__ import annotations

from typing import Any

from hawi.plugin import (
    HawiPlugin,
    HookResult,
    after_conversation,
    after_tool_calling,
    before_conversation,
    before_tool_calling,
    tool,
)
from hawi.tool import ToolResult

from .engine import (
    PLAN_ITEM_DEFAULT_KIND,
    PLAN_ITEM_KINDS,
    PlanEngine,
    PlanEngineResult,
    PlanFoldRecord,
    PlanItem,
)


PLAN_PROMPT_BEGIN = "<hawi-plan-mode>"
PLAN_PROMPT_END = "</hawi-plan-mode>"
PLAN_REMINDER_BEGIN = "<hawi-plan-runtime-reminder>"
PLAN_REMINDER_END = "</hawi-plan-runtime-reminder>"

__all__ = [
    "PLAN_ITEM_DEFAULT_KIND",
    "PLAN_ITEM_KINDS",
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
        return {
            "type": "object",
            "properties": {
                "fold_completed_tasks": {
                    "type": "boolean",
                    "title": "Fold Completed Task Context",
                    "default": False,
                    "description": (
                        "When enabled, complete_plan_item requires a summary and "
                        "moves detailed messages since the previous completed plan "
                        "item into PlanPlugin memory."
                    ),
                }
            },
            "additionalProperties": False,
        }

    @classmethod
    def gui_default_config(cls) -> dict:
        return {"fold_completed_tasks": False}

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
                completed=bool(item.get("completed", False)),
                created_at=float(item.get("created_at", 0.0)),
                completed_at=item.get("completed_at"),
                kind=item.get("kind", "exploratory"),
                completion_summary=item.get("completion_summary"),
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

    @before_conversation
    def inject_plan_instructions(self, agent: Any, ctx: Any) -> None:
        """Inject plan mode guidance into the system prompt."""
        folding_guidance = ""
        if self._engine.fold_completed_tasks:
            folding_guidance = (
                "\n"
                "Completed-task context folding is enabled. When you call "
                "complete_plan_item, you must provide summary and handoff_notes. "
                "The summary must briefly state what was completed in this task. "
                "The handoff_notes must state the information later tasks must "
                "remember; if there is nothing lasting, say so explicitly. After a "
                "completion, PlanPlugin will move detailed messages since the "
                "previous completion out of the active context and keep only the "
                "completion tool call/result marker with the task id, summary, "
                "handoff notes, and read-back instructions. If later work needs folded details, call read_completed_task_context "
                "with the relevant item_id or fold_id.\n"
            )
        prompt = (
            f"\n{PLAN_PROMPT_BEGIN}\n"
            "Plan mode is enabled. This is a runtime/UI planning channel, not a "
            "request to create a plan file. Do not create, edit, or store plan.md, "
            "TODO.md, or any other plan file for the plan itself unless the user "
            "explicitly asks for such a file. Maintain the current plan only by "
            "calling the plan tools.\n"
            "\n"
            "Use the plan tools for work that has multiple steps, unclear ordering, "
            "or a risk of losing track of unfinished work.\n"
            "\n"
            "Available plan tools:\n"
            "- add_plan_item: add one concrete task item, or create a whole plan tree "
            "in one call by passing items=[{content, children, kind}].\n"
            "  Pass parent_id to create a single child item or attach batch root items "
            "under an existing plan item.\n"
            "  Each item has a kind that controls how its completion is decided:\n"
            "    * 'exploratory' (default): a parent whose completion requires your "
            "judgment (e.g. 'Investigate root cause', 'Decide auth strategy'). When all "
            "of its children are completed, complete_plan_item will return a "
            "parent_review_required entry pointing at this parent - for each entry, "
            "decide whether the work is truly done (call complete_plan_item on it) or "
            "whether new follow-up children should be added (call add_plan_item).\n"
            "    * 'determinate': a mechanical parent whose completion is fully implied "
            "by its children (e.g. 'Run all unit tests' broken into concrete sub-tests). "
            "When every child completes, this item is auto-completed, and that auto-"
            "completion can chain upward into other determinate ancestors.\n"
            "  Leave kind unspecified when unsure - exploratory is the safer default.\n"
            "  Leaf items can use either kind; kind only matters when the item has "
            "children.\n"
            "- complete_plan_item: mark a plan item complete as soon as it is actually "
            "done. Pass item_id='all' only when every open item is complete. After each "
            "call, inspect parent_review_required in the result and act on every entry "
            "before moving on. When completed-task context folding is enabled, summary "
            "and handoff_notes are required. summary is a concise statement of what "
            "was completed; handoff_notes are the details later tasks must remember. "
            "If completing a parent with unfinished children, pass "
            "mark_all_children=true only when every child should also be marked done.\n"
            "- list_plan_items: inspect the current plan state.\n"
            "- read_completed_task_context: read or search details that were folded "
            "out of the active context after a previous complete_plan_item call. "
            "Pass query to search folded contexts by keyword. Use max_chars to "
            "limit transcript or search-snippet output length. Use task_id with "
            "message_start/message_end to read a precise folded message range.\n"
            "- plan_control: pause, continue, or abandon plan execution. Use action='pause' "
            "to temporarily exit plan state when plan-driven continuation should stop "
            "for now; include a reason. Use action='continue' to resume plan execution "
            "and automatic reminders. Use action='abandon' to completely discard the "
            "current plan and clear plan memory.\n"
            f"{folding_guidance}"
            "\n"
            "Keep plan items actionable. Do not leave completed work unchecked. When Hawi "
            "reminds you about unfinished plan items, either continue the work, mark completed "
            "items complete, or call plan_control with action='pause' and a clear reason "
            "to temporarily leave plan execution.\n"
            "\n"
            "Important message-origin rule: messages enclosed in "
            f"{PLAN_REMINDER_BEGIN}...{PLAN_REMINDER_END} are automatic runtime "
            "reminders generated by PlanPlugin. They may appear in the conversation "
            "as user-role messages because Hawi reinvokes the model through the "
            "normal message channel, but they are not human-user messages. Treat "
            "them as plugin control guidance, do not attribute them to the user, "
            "and do not infer that the user said or requested their wording.\n"
            f"{PLAN_PROMPT_END}\n"
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
            self._sync_artifact(status="complete")
            return None

        plan_text = self._engine.format_plan_list()
        self._sync_artifact(status="active")
        self.emit_message(
            "Plan has unfinished items; asking the agent to continue.",
            title="Plan reminder",
            data={"pending_count": len(incomplete)},
            run_id=ctx.run_id,
        )
        return HookResult.reinvoke(self._format_runtime_reminder(plan_text))

    @tool(
        name="add_plan_item",
        description=(
            "Add one concrete task item or a tree of task items to the current plan. "
            "Each item has a kind: 'exploratory' (default, requires your judgment to close) "
            "or 'determinate' (auto-completes when all children complete)."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Concrete task to add as a single plan item.",
                },
                "parent_id": {
                    "type": "string",
                    "description": "Optional parent plan item id for nested plan items.",
                },
                "kind": {
                    "type": "string",
                    "enum": list(PLAN_ITEM_KINDS),
                    "description": (
                        "How this item's completion is decided. 'exploratory' (default): "
                        "completion requires your judgment; when all children complete you "
                        "will be reminded via parent_review_required to confirm or extend. "
                        "'determinate': completion is fully implied by children, so the item "
                        "auto-completes when every child completes. Pick determinate only for "
                        "mechanical parents whose work is truly the union of their children."
                    ),
                },
                "items": {
                    "type": "array",
                    "description": (
                        "Optional tree of plan items for creating an entire plan in one call. "
                        "Each item has content, optional children, and optional kind "
                        "(exploratory by default)."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Concrete task for this plan item.",
                            },
                            "kind": {
                                "type": "string",
                                "enum": list(PLAN_ITEM_KINDS),
                                "description": (
                                    "Same semantics as the top-level kind. Defaults to "
                                    "'exploratory' when omitted."
                                ),
                            },
                            "children": {
                                "type": "array",
                                "description": "Optional child plan items.",
                                "items": {"type": "object"},
                            },
                        },
                        "required": ["content"],
                    },
                },
            },
        },
    )
    def add_plan_item(
        self,
        content: str | None = None,
        parent_id: str | None = None,
        items: list[dict[str, Any]] | None = None,
        kind: str | None = None,
    ) -> ToolResult:
        result = self._engine.add_plan_item(
            content=content,
            parent_id=parent_id,
            items=items,
            kind=kind,
        )
        return self._finish_engine_result(result)

    @tool(
        name="complete_plan_item",
        description=(
            "Mark a plan item complete. Use item_id='all' only when every open item "
            "is actually complete. When completed-task context folding is enabled, "
            "summary and handoff_notes are required and become the handoff note "
            "left in active context."
        ),
        context="ctx",
        parameters_schema={
            "type": "object",
            "properties": {
                "item_id": {
                    "type": "string",
                    "description": "Plan item id from list_plan_items, or 'all'.",
                },
                "mark_all_children": {
                    "type": "boolean",
                    "description": (
                        "When true, also mark all descendant items complete. Required "
                        "to complete a parent that still has unfinished children."
                    ),
                    "default": False,
                },
                "summary": {
                    "type": "string",
                    "description": (
                        "Required when completed-task context folding is enabled. "
                        "Briefly summarize what was completed in this task."
                    ),
                },
                "handoff_notes": {
                    "type": "string",
                    "description": (
                        "Required when completed-task context folding is enabled. "
                        "State information later tasks must remember, or explicitly "
                        "say there are no lasting notes."
                    ),
                },
            },
            "required": ["item_id"],
        },
    )
    def complete_plan_item(
        self,
        item_id: str,
        mark_all_children: bool = False,
        summary: str | None = None,
        handoff_notes: str | None = None,
        ctx: Any = None,
        complete_children: bool | None = None,
    ) -> ToolResult:
        result = self._engine.complete_plan_item(
            item_id,
            mark_all_children=mark_all_children,
            summary=summary,
            handoff_notes=handoff_notes,
            complete_children=complete_children,
        )
        if not result.success:
            return ToolResult(
                success=False,
                error=self._format_completion_error_markdown(result),
            )

        self._sync_artifact(status=self._engine.plan_status())
        self._emit_item_events(result)
        if result.fold_request is not None:
            folded_context = self._maybe_fold_completed_context(
                ctx,
                **result.fold_request,
            )
            result.output["folded_context"] = folded_context
        return ToolResult(
            success=True,
            output=self._format_completion_markdown(result.output),
        )

    @tool(
        name="list_plan_items",
        description="List the current plan items and notification state.",
        parameters_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    )
    def list_plan_items(self) -> ToolResult:
        self._sync_artifact(status=self._engine.plan_status())
        return ToolResult(success=True, output=self._engine.state_dict())

    @tool(
        name="read_completed_task_context",
        description=(
            "Read detailed messages that PlanPlugin folded out of active context "
            "after a completed plan item, or search folded task contexts by keyword. "
            "Use this when a later task needs details from a previous item."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "item_id": {
                    "type": "string",
                    "description": (
                        "Completed plan item id, such as P1. Omit this when using "
                        "query to search every folded context."
                    ),
                },
                "task_id": {
                    "type": "string",
                    "description": (
                        "Alias for item_id. Use this with message_start/message_end "
                        "to read a precise range from a completed task context."
                    ),
                },
                "fold_id": {
                    "type": "string",
                    "description": "Specific folded context id from a completion result.",
                },
                "message_start": {
                    "type": "integer",
                    "description": (
                        "Optional 1-based first folded message index to read. "
                        "Indexes come from the Folded context preview list."
                    ),
                },
                "message_end": {
                    "type": "integer",
                    "description": (
                        "Optional 1-based last folded message index to read, inclusive."
                    ),
                },
                "message_range": {
                    "type": "string",
                    "description": (
                        "Optional message range shorthand such as '2-4' or '3'. "
                        "Equivalent to message_start/message_end."
                    ),
                },
                "query": {
                    "type": "string",
                    "description": (
                        "Optional case-insensitive keyword search. When provided, "
                        "the tool returns matching snippets instead of a full transcript. "
                        "If item_id and fold_id are omitted, all folded contexts are searched."
                    ),
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether query matching should be case-sensitive.",
                    "default": False,
                },
                "max_matches": {
                    "type": "integer",
                    "description": "Maximum number of search matches to return.",
                    "default": 20,
                },
                "context_chars": {
                    "type": "integer",
                    "description": (
                        "Characters of surrounding context to include before and "
                        "after each search match."
                    ),
                    "default": 240,
                },
                "max_chars": {
                    "type": "integer",
                    "description": (
                        "Maximum transcript characters to return in read mode, or "
                        "maximum aggregate snippet characters to return in search mode."
                    ),
                    "default": 20000,
                },
            },
            "additionalProperties": False,
        },
    )
    def read_completed_task_context(
        self,
        item_id: str = "",
        task_id: str = "",
        fold_id: str | None = None,
        message_start: int | None = None,
        message_end: int | None = None,
        message_range: str | None = None,
        query: str | None = None,
        case_sensitive: bool = False,
        max_matches: int = 20,
        context_chars: int = 240,
        max_chars: int = 20000,
    ) -> ToolResult:
        normalized_item_id = item_id or task_id
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
                    "Provide task_id, item_id, or fold_id to read a folded task "
                    "context, or provide query to search folded task contexts."
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
                    f"task_id={normalized_item_id!r}, fold_id={fold_id!r}."
                ),
            )
        selected_messages, selected_start, selected_end, range_error = (
            self._select_folded_messages(
                record.messages,
                message_start=message_start,
                message_end=message_end,
                message_range=message_range,
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
                "\n\n[Transcript truncated. Call read_completed_task_context "
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
            "Pause, continue, or abandon plan execution. Pause temporarily exits "
            "plan state and stops automatic reminders; continue resumes plan "
            "execution; abandon clears the current plan."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["pause", "continue", "abandon"],
                    "description": (
                        "Use 'pause' to temporarily stop plan-driven continuation; "
                        "use 'continue' to resume plan execution; use 'abandon' "
                        "to completely clear the current plan."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": "Required when action is 'pause'; optional for abandon.",
                },
            },
            "required": ["action"],
        },
    )
    def plan_control(self, action: str, reason: str | None = None) -> ToolResult:
        result = self._engine.control(action=action, reason=reason)
        return self._finish_engine_result(result)

    def _select_folded_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        message_start: int | None,
        message_end: int | None,
        message_range: str | None,
    ) -> tuple[list[dict[str, Any]], int, int, str]:
        total = len(messages)
        if total == 0:
            return [], 1, 0, ""

        start = message_start
        end = message_end
        range_text = message_range.strip() if isinstance(message_range, str) else ""
        if range_text:
            parsed_start, parsed_end, error = self._parse_message_range(range_text)
            if error:
                return [], 1, total, error
            start = parsed_start
            end = parsed_end

        start = 1 if start is None else start
        end = total if end is None else end
        if not isinstance(start, int) or not isinstance(end, int):
            return [], 1, total, "message_start and message_end must be integers."
        if start < 1 or end < 1:
            return [], 1, total, "message_start and message_end are 1-based and must be positive."
        if start > end:
            return [], 1, total, "message_start must be less than or equal to message_end."
        if start > total:
            return [], 1, total, f"message_start exceeds folded message count ({total})."
        end = min(end, total)
        return messages[start - 1 : end], start, end, ""

    @staticmethod
    def _parse_message_range(message_range: str) -> tuple[int | None, int | None, str]:
        if "-" in message_range:
            raw_start, raw_end = message_range.split("-", 1)
            try:
                start = int(raw_start.strip())
                end = int(raw_end.strip())
            except ValueError:
                return None, None, "message_range must look like '2-4' or '3'."
            return start, end, ""
        try:
            value = int(message_range)
        except ValueError:
            return None, None, "message_range must look like '2-4' or '3'."
        return value, value, ""

    def _format_runtime_reminder(self, plan_text: str) -> str:
        return (
            f"{PLAN_REMINDER_BEGIN}\n"
            "source: PlanPlugin\n"
            "origin: automatic runtime reminder, not a human user message\n"
            "purpose: continue plan execution while unfinished plan items remain\n"
            "\n"
            "The following plan items are still unfinished:\n\n"
            f"{plan_text}\n\n"
            "Continue executing the remaining work. If some items are already done, "
            "call complete_plan_item for each completed item before proceeding. If all "
            "remaining items are impossible, obsolete, blocked by missing information, or "
            "intentionally deferred for now, call plan_control with action='pause' and "
            "a clear reason to temporarily exit plan execution. When plan execution "
            "should resume, call plan_control with action='continue'. If the current "
            "plan should be completely discarded, call plan_control with "
            "action='abandon'. Otherwise keep working on the plan.\n"
            f"{PLAN_REMINDER_END}"
        )

    def _format_completion_markdown(self, output: dict[str, Any]) -> str:
        completed = output.get("completed", [])
        parent_review_required = output.get("parent_review_required", [])
        pending_count = int(output.get("pending_count") or 0)
        completion_summary = output.get("completion_summary")
        handoff_notes = output.get("handoff_notes")
        lines = [
            "Plan item completed." if completed else "No new plan items were completed.",
            "",
        ]

        if completed:
            lines.append("Completed:")
            for item in completed:
                lines.append(f"- {self._format_item_reference(item)}")
            lines.append("")

        lines.append(f"Pending items: {pending_count}")

        if completion_summary:
            lines.extend(["", "Task summary:"])
            lines.append(str(completion_summary))

        if handoff_notes:
            lines.extend(["", "Information for later tasks:"])
            lines.append(str(handoff_notes))

        if parent_review_required:
            lines.extend(["", "Parent review required:"])
            for item in parent_review_required:
                lines.append(f"- {self._format_item_reference(item)}")
                reason = item.get("reason")
                if reason:
                    lines.append(f"  {reason}")

        folded_context = output.get("folded_context")
        if isinstance(folded_context, dict) and folded_context.get("enabled"):
            lines.extend(["", "Folded context:"])
            if folded_context.get("skipped"):
                lines.append(f"- Not folded: {folded_context.get('reason', 'No details.')}")
            else:
                fold_id = folded_context.get("fold_id")
                item_id = folded_context.get("item_id")
                folded_message_count = folded_context.get("folded_message_count")
                if fold_id:
                    lines.append(f"- Fold id: `{fold_id}`")
                if item_id:
                    lines.append(f"- Item id: `{item_id}`")
                if folded_message_count is not None:
                    lines.append(f"- Folded messages: {folded_message_count}")
                message_previews = folded_context.get("message_previews", [])
                if message_previews:
                    lines.extend(["", "Folded message previews:"])
                    for preview in message_previews:
                        index = preview.get("index")
                        role = preview.get("role", "unknown")
                        text = preview.get("preview", "")
                        lines.append(f"{index}. {role}: {text}")
                if fold_id or item_id:
                    lines.extend(["", "How to read folded details:"])
                if item_id:
                    lines.append(
                        "- Full task context: "
                        f"`read_completed_task_context(task_id=\"{item_id}\")`."
                    )
                if item_id and folded_message_count:
                    range_end = min(int(folded_message_count), 3)
                    lines.append(
                        "- Message range: "
                        f"`read_completed_task_context(task_id=\"{item_id}\", "
                        f"message_start=1, message_end={range_end})`."
                    )
                if fold_id:
                    lines.append(
                        "- Fold id lookup: "
                        f"`read_completed_task_context(fold_id=\"{fold_id}\")`."
                    )
                if item_id:
                    lines.append(
                        "- Search this task: "
                        f"`read_completed_task_context(task_id=\"{item_id}\", "
                        "query=\"keyword\")`."
                    )

        lines.extend(["", f"Next action: {self._completion_next_action(output)}"])
        return "\n".join(lines)

    def _format_completion_error_markdown(self, result: PlanEngineResult) -> str:
        output = result.output or {}
        unfinished_children = output.get("unfinished_children", [])
        item = output.get("item", {})
        if unfinished_children:
            item_id = item.get("id", "the requested item")
            lines = [
                f"Cannot complete `{item_id}` because it has unfinished child task(s).",
                "",
                "Unfinished children:",
            ]
            for child in unfinished_children:
                lines.append(f"- {self._format_item_reference(child)}")
            pending_count = output.get("pending_count")
            if pending_count is not None:
                lines.extend(["", f"Pending items: {pending_count}"])
            lines.extend(
                [
                    "",
                    "Next action: Complete the child tasks first, or call "
                    "`complete_plan_item` with `mark_all_children=true` only if every "
                    "unfinished child should also be marked done.",
                ]
            )
            return "\n".join(lines)
        return result.error

    @staticmethod
    def _format_item_reference(item: dict[str, Any]) -> str:
        item_id = item.get("id", "unknown")
        content = item.get("content", "")
        return f"`{item_id}` {content}".rstrip()

    @staticmethod
    def _completion_next_action(output: dict[str, Any]) -> str:
        if output.get("parent_review_required"):
            return "Resolve every parent review item before moving on."
        pending_count = int(output.get("pending_count") or 0)
        if pending_count > 0:
            return "Continue with the remaining plan items."
        return "The plan is complete."

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
