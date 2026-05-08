from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import dataclass
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


PLAN_PROMPT_BEGIN = "<hawi-plan-mode>"
PLAN_PROMPT_END = "</hawi-plan-mode>"

PLAN_ITEM_KINDS = ("exploratory", "determinate")
PLAN_ITEM_DEFAULT_KIND = "exploratory"


@dataclass
class PlanItem:
    id: str
    content: str
    parent_id: str | None = None
    completed: bool = False
    created_at: float = 0.0
    completed_at: float | None = None
    kind: str = PLAN_ITEM_DEFAULT_KIND
    completion_summary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "parent_id": self.parent_id,
            "completed": self.completed,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "kind": self.kind,
            "completion_summary": self.completion_summary,
        }


@dataclass
class PlanFoldRecord:
    fold_id: str
    item_id: str
    item_content: str
    summary: str
    messages: list[dict[str, Any]]
    completed_item_ids: list[str]
    created_at: float

    def reference_dict(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "item_id": self.item_id,
            "item_content": self.item_content,
            "summary": self.summary,
            "completed_item_ids": self.completed_item_ids,
            "folded_message_count": len(self.messages),
            "read_tool": "read_completed_task_context",
            "note": (
                "Plan context folding is enabled. Detailed messages from this "
                "completed task were moved out of the active model context and "
                "stored in PlanPlugin memory. Use read_completed_task_context "
                "with this item_id or fold_id if later work needs the details."
            ),
        }


class PlanPlugin(HawiPlugin):
    """Plan mode plugin.

    Provides tools for maintaining an explicit task plan and a conversation hook
    that nudges the agent to continue while open plan items remain.
    """

    def __init__(self, fold_completed_tasks: bool = False) -> None:
        self._items: list[PlanItem] = []
        self._next_item_number = 1
        self._notification_cancelled = False
        self._cancel_reason = ""
        self._fold_completed_tasks = bool(fold_completed_tasks)
        self._fold_records: list[PlanFoldRecord] = []
        self._next_fold_number = 1
        self._active_completion_tool_call_id: str | None = None

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
        new_plugin = PlanPlugin(fold_completed_tasks=self._fold_completed_tasks)
        new_plugin._items = [
            PlanItem(
                id=item.id,
                content=item.content,
                parent_id=item.parent_id,
                completed=item.completed,
                created_at=item.created_at,
                completed_at=item.completed_at,
                kind=item.kind,
                completion_summary=item.completion_summary,
            )
            for item in self._items
        ]
        new_plugin._next_item_number = self._next_item_number
        new_plugin._notification_cancelled = self._notification_cancelled
        new_plugin._cancel_reason = self._cancel_reason
        new_plugin._fold_records = [
            PlanFoldRecord(
                fold_id=record.fold_id,
                item_id=record.item_id,
                item_content=record.item_content,
                summary=record.summary,
                messages=deepcopy(record.messages),
                completed_item_ids=list(record.completed_item_ids),
                created_at=record.created_at,
            )
            for record in self._fold_records
        ]
        new_plugin._next_fold_number = self._next_fold_number
        return new_plugin

    @before_conversation
    def inject_plan_instructions(self, agent: Any, ctx: Any) -> None:
        """Inject plan mode guidance into the system prompt."""
        folding_guidance = ""
        if self._fold_completed_tasks:
            folding_guidance = (
                "\n"
                "Completed-task context folding is enabled. When you call "
                "complete_plan_item, you must provide summary. The summary is a "
                "handoff note for future tasks and must briefly cover: (1) what "
                "happened since the previous plan item was completed, and (2) any "
                "details worth remembering for the rest of the work. After a "
                "completion, PlanPlugin will move detailed messages since the "
                "previous completion out of the active context and keep only the "
                "completion tool call/result marker with the summary and a task id. "
                "If later work needs folded details, call read_completed_task_context "
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
            "parent_review_required entry pointing at this parent — for each entry, "
            "decide whether the work is truly done (call complete_plan_item on it) or "
            "whether new follow-up children should be added (call add_plan_item).\n"
            "    * 'determinate': a mechanical parent whose completion is fully implied "
            "by its children (e.g. 'Run all unit tests' broken into concrete sub-tests). "
            "When every child completes, this item is auto-completed, and that auto-"
            "completion can chain upward into other determinate ancestors.\n"
            "  Leave kind unspecified when unsure — exploratory is the safer default.\n"
            "  Leaf items can use either kind; kind only matters when the item has "
            "children.\n"
            "- complete_plan_item: mark a plan item complete as soon as it is actually "
            "done. Pass item_id='all' only when every open item is complete. After each "
            "call, inspect parent_review_required in the result and act on every entry "
            "before moving on. When completed-task context folding is enabled, summary "
            "is required.\n"
            "- list_plan_items: inspect the current plan state.\n"
            "- read_completed_task_context: read details that were folded out of the "
            "active context after a previous complete_plan_item call.\n"
            "- cancel_plan_notification: stop automatic plan reminders only when remaining "
            "items are impossible, obsolete, or intentionally deferred; include a reason.\n"
            f"{folding_guidance}"
            "\n"
            "Keep plan items actionable. Do not leave completed work unchecked. When Hawi "
            "reminds you about unfinished plan items, either continue the work, mark completed "
            "items complete, or cancel the notification with a clear reason.\n"
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
            self._active_completion_tool_call_id = getattr(ctx, "tool_call_id", None)

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
            self._active_completion_tool_call_id = None

    @after_conversation
    def notify_unfinished_plan(self, agent: Any, ctx: Any) -> HookResult | None:
        """Re-drive the agent while plan items remain unfinished."""
        if ctx.error is not None or self._notification_cancelled:
            return None
        if not self._items:
            return None

        incomplete = self._incomplete_items()
        if not incomplete:
            self._sync_artifact(status="complete")
            return None

        plan_text = self._format_plan_list()
        self._sync_artifact(status="active")
        self.emit_message(
            "Plan has unfinished items; asking the agent to continue.",
            title="Plan reminder",
            data={"pending_count": len(incomplete)},
            run_id=ctx.run_id,
        )
        return HookResult.reinvoke(
            (
                "Plan reminder: the following plan items are still unfinished.\n\n"
                f"{plan_text}\n\n"
                "Continue executing the remaining work. If some items are already done, "
                "call complete_plan_item for each completed item before proceeding. If all "
                "remaining items are impossible, obsolete, blocked by missing information, or "
                "intentionally deferred, call cancel_plan_notification with a clear reason. "
                "Otherwise keep working on the plan."
            )
        )

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
        has_content = isinstance(content, str) and bool(content.strip())
        has_items = items is not None
        if has_content and has_items:
            return ToolResult(
                success=False,
                error="Provide either content for a single item or items for a plan tree, not both.",
            )
        if not has_content and not has_items:
            return ToolResult(
                success=False,
                error="Provide content for a single item or items for a plan tree.",
            )
        if has_items and kind is not None:
            return ToolResult(
                success=False,
                error=(
                    "Top-level kind is only valid when adding a single item. "
                    "For tree mode, set kind on each node inside items."
                ),
            )

        normalized_kind, kind_error = self._normalize_kind(kind, path="kind")
        if kind_error:
            return ToolResult(success=False, error=kind_error)

        normalized_parent_id = (
            parent_id.strip()
            if isinstance(parent_id, str) and parent_id.strip()
            else None
        )
        if (
            normalized_parent_id is not None
            and self._find_item(normalized_parent_id) is None
        ):
            return ToolResult(
                success=False,
                error=f"Unknown parent plan item id: {normalized_parent_id}",
            )

        now = time.time()
        if has_items:
            created, error = self._add_plan_item_tree(
                items or [],
                parent_id=normalized_parent_id,
                created_at=now,
            )
            if error:
                return ToolResult(success=False, error=error)
            self._notification_cancelled = False
            self._cancel_reason = ""
            self._sync_artifact(status="active")
            for item in created:
                self._emit_plan_item_update("added", item)
            return ToolResult(
                success=True,
                output={
                    "items": [item.to_dict() for item in created],
                    "tree": self._tree_items(),
                    "pending_count": len(self._incomplete_items()),
                },
            )

        item = self._create_plan_item(
            content=content.strip() if isinstance(content, str) else "",
            parent_id=normalized_parent_id,
            created_at=now,
            kind=normalized_kind,
        )
        self._notification_cancelled = False
        self._cancel_reason = ""
        self._sync_artifact(status="active")
        self._emit_plan_item_update("added", item)
        return ToolResult(
            success=True,
            output={
                "item": item.to_dict(),
                "tree": self._tree_items(),
                "pending_count": len(self._incomplete_items()),
            },
        )

    @tool(
        name="complete_plan_item",
        description=(
            "Mark a plan item complete. Use item_id='all' only when every open item "
            "is actually complete. When completed-task context folding is enabled, "
            "summary is required and becomes the handoff note left in active context."
        ),
        context="ctx",
        parameters_schema={
            "type": "object",
            "properties": {
                "item_id": {
                    "type": "string",
                    "description": "Plan item id from list_plan_items, or 'all'.",
                },
                "complete_children": {
                    "type": "boolean",
                    "description": "When true, also mark all descendant items complete.",
                    "default": False,
                },
                "summary": {
                    "type": "string",
                    "description": (
                        "Required when completed-task context folding is enabled. "
                        "Briefly state what happened since the previous completed "
                        "plan item and what details future tasks should remember."
                    ),
                },
            },
            "required": ["item_id"],
        },
    )
    def complete_plan_item(
        self,
        item_id: str,
        complete_children: bool = False,
        summary: str | None = None,
        ctx: Any = None,
    ) -> ToolResult:
        item_id = item_id.strip()
        summary_text = summary.strip() if isinstance(summary, str) else ""
        if self._fold_completed_tasks and not summary_text:
            return ToolResult(
                success=False,
                error=(
                    "summary is required when completed-task context folding is "
                    "enabled. Provide a concise handoff summary covering what "
                    "happened since the previous completed item and what should "
                    "be remembered for later work."
                ),
            )
        now = time.time()

        if item_id.lower() == "all":
            completed = []
            for item in self._items:
                if not item.completed:
                    item.completed = True
                    item.completed_at = now
                    if summary_text:
                        item.completion_summary = summary_text
                    completed.append(item.to_dict())
            self._sync_artifact(status="complete")
            for item_data in completed:
                self._emit_plan_item_update("completed", item_data)
            folded_context = self._maybe_fold_completed_context(
                ctx,
                item_id="all",
                item_content="All open plan items",
                summary=summary_text,
                completed_item_ids=[item["id"] for item in completed],
            )
            return ToolResult(
                success=True,
                output={
                    "completed": completed,
                    "tree": self._tree_items(),
                    "pending_count": len(self._incomplete_items()),
                    "parent_review_required": [],
                    "completion_summary": summary_text or None,
                    "folded_context": folded_context,
                },
            )

        item = self._find_item(item_id)
        if item is None:
            return ToolResult(success=False, error=f"Unknown plan item id: {item_id}")

        items_to_complete = [item]
        if complete_children:
            items_to_complete.extend(self._descendants(item.id))
        completed = []
        for current in items_to_complete:
            if not current.completed:
                current.completed = True
                current.completed_at = now
                if summary_text:
                    current.completion_summary = summary_text
                completed.append(current.to_dict())
        auto_completed, review_required = self._propagate_completion_upward(item, now)
        completed.extend(auto_completed)
        self._sync_artifact(status="complete" if not self._incomplete_items() else "active")
        for item_data in completed:
            self._emit_plan_item_update("completed", item_data)
        folded_context = self._maybe_fold_completed_context(
            ctx,
            item_id=item.id,
            item_content=item.content,
            summary=summary_text,
            completed_item_ids=[item_data["id"] for item_data in completed],
        )
        return ToolResult(
            success=True,
            output={
                "item": item.to_dict(),
                "completed": completed,
                "tree": self._tree_items(),
                "pending_count": len(self._incomplete_items()),
                "parent_review_required": review_required,
                "completion_summary": summary_text or None,
                "folded_context": folded_context,
            },
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
        self._sync_artifact(
            status="cancelled"
            if self._notification_cancelled
            else "complete"
            if self._items and not self._incomplete_items()
            else "active"
            if self._items
            else "empty"
        )
        return ToolResult(success=True, output=self._state_dict())

    @tool(
        name="read_completed_task_context",
        description=(
            "Read detailed messages that PlanPlugin folded out of active context "
            "after a completed plan item. Use this when a later task needs details "
            "from a previously completed item."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "item_id": {
                    "type": "string",
                    "description": (
                        "Completed plan item id, such as P1. Use this unless you "
                        "have a specific fold_id."
                    ),
                },
                "fold_id": {
                    "type": "string",
                    "description": "Specific folded context id from a completion result.",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum transcript characters to return.",
                    "default": 20000,
                },
            },
            "additionalProperties": False,
        },
    )
    def read_completed_task_context(
        self,
        item_id: str = "",
        fold_id: str | None = None,
        max_chars: int = 20000,
    ) -> ToolResult:
        if not (item_id or fold_id):
            return ToolResult(
                success=False,
                error="Provide item_id or fold_id to read a folded task context.",
            )
        record = self._find_fold_record(item_id=item_id, fold_id=fold_id)
        if record is None:
            if not self._fold_records:
                return ToolResult(
                    success=False,
                    error="No completed task contexts have been folded yet.",
                )
            return ToolResult(
                success=False,
                error=(
                    "No folded context found for "
                    f"item_id={item_id!r}, fold_id={fold_id!r}."
                ),
            )

        max_chars = max(1000, int(max_chars or 20000))
        transcript = self._format_folded_messages(record.messages)
        truncated = len(transcript) > max_chars
        if truncated:
            transcript = transcript[: max_chars - 120].rstrip() + (
                "\n\n[Transcript truncated. Call read_completed_task_context "
                "with a larger max_chars value for more detail.]"
            )
        return ToolResult(
            success=True,
            output={
                "fold_id": record.fold_id,
                "item_id": record.item_id,
                "item_content": record.item_content,
                "summary": record.summary,
                "completed_item_ids": record.completed_item_ids,
                "message_count": len(record.messages),
                "transcript": transcript,
                "truncated": truncated,
            },
        )

    @tool(
        name="cancel_plan_notification",
        description=(
            "Cancel automatic reminders for unfinished plan items when they cannot "
            "or should not be completed in this run."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why remaining plan items should not continue now.",
                }
            },
            "required": ["reason"],
        },
    )
    def cancel_plan_notification(self, reason: str) -> ToolResult:
        reason = reason.strip()
        if not reason:
            return ToolResult(success=False, error="Cancellation reason cannot be empty.")
        self._notification_cancelled = True
        self._cancel_reason = reason
        self._sync_artifact(status="cancelled")
        self.emit_plugin_event(
            "plugin.event",
            {
                "event_name": "plan.notification.cancelled",
                "reason": reason,
                "state": self._state_dict(),
                "title": "Plan notification cancelled",
                "message": reason,
                "data": {"reason": reason},
            },
        )
        return ToolResult(success=True, output=self._state_dict())

    def _normalize_kind(
        self, kind: Any, *, path: str
    ) -> tuple[str, str]:
        if kind is None:
            return PLAN_ITEM_DEFAULT_KIND, ""
        if not isinstance(kind, str):
            return "", f"{path} must be a string when provided."
        normalized = kind.strip().lower()
        if not normalized:
            return PLAN_ITEM_DEFAULT_KIND, ""
        if normalized not in PLAN_ITEM_KINDS:
            allowed = ", ".join(repr(k) for k in PLAN_ITEM_KINDS)
            return "", f"{path} must be one of {allowed}."
        return normalized, ""

    def _create_plan_item(
        self,
        *,
        content: str,
        parent_id: str | None,
        created_at: float,
        kind: str = PLAN_ITEM_DEFAULT_KIND,
    ) -> PlanItem:
        item = PlanItem(
            id=f"P{self._next_item_number}",
            content=content,
            parent_id=parent_id,
            created_at=created_at,
            kind=kind,
        )
        self._next_item_number += 1
        self._items.append(item)
        return item

    def _add_plan_item_tree(
        self,
        nodes: list[dict[str, Any]],
        *,
        parent_id: str | None,
        created_at: float,
    ) -> tuple[list[PlanItem], str]:
        validation_error = self._validate_plan_item_nodes(nodes, path="items")
        if validation_error:
            return [], validation_error

        created: list[PlanItem] = []

        def add_nodes(current_nodes: list[dict[str, Any]], current_parent_id: str | None) -> None:
            for node in current_nodes:
                node_kind, _ = self._normalize_kind(
                    node.get("kind"), path="kind"
                )
                item = self._create_plan_item(
                    content=str(node["content"]).strip(),
                    parent_id=current_parent_id,
                    created_at=created_at,
                    kind=node_kind,
                )
                created.append(item)
                children = node.get("children") or []
                add_nodes(children, item.id)

        add_nodes(nodes, parent_id)
        return created, ""

    def _validate_plan_item_nodes(self, nodes: Any, *, path: str) -> str:
        if not isinstance(nodes, list):
            return f"{path} must be a list of plan item objects."
        if path == "items" and not nodes:
            return "items must contain at least one plan item."

        for index, node in enumerate(nodes):
            item_path = f"{path}[{index}]"
            if not isinstance(node, dict):
                return f"{item_path} must be an object."
            content = node.get("content")
            if not isinstance(content, str) or not content.strip():
                return f"{item_path}.content must be a non-empty string."
            if "kind" in node:
                _, kind_error = self._normalize_kind(
                    node.get("kind"), path=f"{item_path}.kind"
                )
                if kind_error:
                    return kind_error
            children = node.get("children", [])
            if children is None:
                continue
            if not isinstance(children, list):
                return f"{item_path}.children must be a list when provided."
            child_error = self._validate_plan_item_nodes(
                children,
                path=f"{item_path}.children",
            )
            if child_error:
                return child_error
        return ""

    def _maybe_fold_completed_context(
        self,
        ctx: Any,
        *,
        item_id: str,
        item_content: str,
        summary: str,
        completed_item_ids: list[str],
    ) -> dict[str, Any]:
        if not self._fold_completed_tasks:
            return {
                "enabled": False,
                "summary": summary or None,
            }
        if not completed_item_ids:
            return {
                "enabled": True,
                "skipped": True,
                "reason": "No plan items were newly completed by this call.",
                "summary": summary,
            }

        context = getattr(ctx, "context", None)
        messages = getattr(context, "messages", None)
        if not isinstance(messages, list):
            return {
                "enabled": True,
                "skipped": True,
                "reason": (
                    "No runtime AgentContext was available. Folding only runs "
                    "during agent tool execution."
                ),
                "summary": summary,
            }

        current_index = self._find_current_completion_message_index(messages)
        if current_index is None:
            return {
                "enabled": True,
                "skipped": True,
                "reason": (
                    "Could not locate the active complete_plan_item tool call in "
                    "the current context, so no messages were folded."
                ),
                "summary": summary,
            }

        start_index = self._fold_start_index(messages, current_index)
        folded_messages = deepcopy(messages[start_index:current_index])
        if start_index < current_index:
            del messages[start_index:current_index]

        record = PlanFoldRecord(
            fold_id=f"PF{self._next_fold_number}",
            item_id=item_id,
            item_content=item_content,
            summary=summary,
            messages=folded_messages,
            completed_item_ids=list(completed_item_ids),
            created_at=time.time(),
        )
        self._next_fold_number += 1
        self._fold_records.append(record)
        self.emit_plugin_event(
            "plugin.event",
            {
                "event_name": "plan.context.folded",
                "fold": record.reference_dict(),
                "title": "Plan context folded",
                "message": (
                    f"Folded {len(folded_messages)} message(s) for {item_id}."
                ),
                "data": {
                    "fold_id": record.fold_id,
                    "item_id": item_id,
                    "folded_message_count": len(folded_messages),
                },
            },
        )

        folded_context = record.reference_dict()
        folded_context["enabled"] = True
        folded_context["skipped"] = False
        return folded_context

    def _find_current_completion_message_index(
        self,
        messages: list[dict[str, Any]],
    ) -> int | None:
        if self._active_completion_tool_call_id:
            for index in range(len(messages) - 1, -1, -1):
                message = messages[index]
                if self._assistant_has_tool_call_id(
                    message,
                    self._active_completion_tool_call_id,
                ):
                    return index

        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if self._assistant_has_tool_call_name(message, "complete_plan_item"):
                return index
        return None

    def _fold_start_index(
        self,
        messages: list[dict[str, Any]],
        current_index: int,
    ) -> int:
        latest_marker_end = -1
        for index in range(current_index):
            message = messages[index]
            if not self._assistant_has_tool_call_name(message, "complete_plan_item"):
                continue
            marker_end = index
            while (
                marker_end + 1 < current_index
                and messages[marker_end + 1].get("role") == "tool"
            ):
                marker_end += 1
            latest_marker_end = marker_end
        return latest_marker_end + 1

    @staticmethod
    def _assistant_has_tool_call_id(message: dict[str, Any], tool_call_id: str) -> bool:
        if message.get("role") != "assistant":
            return False
        for part in message.get("content", []):
            if (
                isinstance(part, dict)
                and part.get("type") == "tool_call"
                and part.get("id") == tool_call_id
            ):
                return True
        return False

    @staticmethod
    def _assistant_has_tool_call_name(message: dict[str, Any], tool_name: str) -> bool:
        if message.get("role") != "assistant":
            return False
        for part in message.get("content", []):
            if (
                isinstance(part, dict)
                and part.get("type") == "tool_call"
                and part.get("name") == tool_name
            ):
                return True
        return False

    def _find_fold_record(
        self,
        *,
        item_id: str,
        fold_id: str | None,
    ) -> PlanFoldRecord | None:
        normalized_fold_id = fold_id.strip().lower() if isinstance(fold_id, str) else ""
        if normalized_fold_id:
            for record in reversed(self._fold_records):
                if record.fold_id.lower() == normalized_fold_id:
                    return record

        normalized_item_id = item_id.strip().lower() if isinstance(item_id, str) else ""
        if normalized_item_id:
            for record in reversed(self._fold_records):
                if record.item_id.lower() == normalized_item_id or any(
                    completed_id.lower() == normalized_item_id
                    for completed_id in record.completed_item_ids
                ):
                    return record
        return None

    def _format_folded_messages(self, messages: list[dict[str, Any]]) -> str:
        if not messages:
            return "[No detailed messages were folded for this completion.]"

        sections: list[str] = []
        for index, message in enumerate(messages, 1):
            role = message.get("role", "unknown")
            name = message.get("name")
            title = f"Message {index} ({role})"
            if name:
                title += f" name={name}"
            content = self._format_content(message.get("content", []))
            sections.append(f"## {title}\n{content}")
        return "\n\n".join(sections)

    def _format_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return self._safe_json(content)

        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                parts.append(str(part))
                continue
            part_type = part.get("type")
            if part_type == "text":
                parts.append(str(part.get("text", "")))
            elif part_type == "reasoning":
                reasoning = part.get("reasoning") or ""
                if reasoning:
                    parts.append(f"[reasoning]\n{reasoning}")
            elif part_type == "tool_call":
                parts.append(
                    "[tool_call] "
                    f"id={part.get('id', '')} "
                    f"name={part.get('name', '')} "
                    f"arguments={self._safe_json(part.get('arguments', {}))}"
                )
            elif part_type == "tool_result":
                result_content = part.get("content", [])
                parts.append(
                    "[tool_result] "
                    f"tool_call_id={part.get('tool_call_id', '')} "
                    f"is_error={part.get('is_error', False)}\n"
                    f"{self._format_content(result_content)}"
                )
            elif part_type == "steer":
                parts.append(f"[steer]\n{self._format_content(part.get('content', []))}")
            else:
                parts.append(self._safe_json(part))
        return "\n".join(text for text in parts if text)

    @staticmethod
    def _safe_json(value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            return str(value)

    def _state_dict(self) -> dict[str, Any]:
        incomplete = self._incomplete_items()
        return {
            "items": self._tree_items(),
            "flat_items": [item.to_dict() for item in self._items],
            "pending_count": len(incomplete),
            "notification_cancelled": self._notification_cancelled,
            "cancel_reason": self._cancel_reason,
            "context_folding_enabled": self._fold_completed_tasks,
            "folded_contexts": [
                record.reference_dict() for record in self._fold_records
            ],
        }

    def _find_item(self, item_id: str) -> PlanItem | None:
        normalized = item_id.strip().lower()
        for item in self._items:
            if item.id.lower() == normalized:
                return item
        return None

    def _children_of(self, parent_id: str | None) -> list[PlanItem]:
        return [item for item in self._items if item.parent_id == parent_id]

    def _descendants(self, parent_id: str) -> list[PlanItem]:
        descendants: list[PlanItem] = []
        for child in self._children_of(parent_id):
            descendants.append(child)
            descendants.extend(self._descendants(child.id))
        return descendants

    def _incomplete_items(self) -> list[PlanItem]:
        return [item for item in self._items if not item.completed]

    def _propagate_completion_upward(
        self, start: PlanItem, now: float
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        auto_completed: list[dict[str, Any]] = []
        review_required: list[dict[str, Any]] = []
        cursor = start
        while cursor.parent_id is not None:
            parent = self._find_item(cursor.parent_id)
            if parent is None or parent.completed:
                break
            children = self._children_of(parent.id)
            if not children or not all(child.completed for child in children):
                break
            if parent.kind == "determinate":
                parent.completed = True
                parent.completed_at = now
                auto_completed.append(parent.to_dict())
                cursor = parent
                continue
            review_required.append(
                {
                    "id": parent.id,
                    "content": parent.content,
                    "kind": parent.kind,
                    "reason": (
                        "All children are complete, but this item is exploratory — "
                        "decide whether the work is truly done (call complete_plan_item) "
                        "or add follow-up children (call add_plan_item)."
                    ),
                }
            )
            break
        return auto_completed, review_required

    def _tree_items(self) -> list[dict[str, Any]]:
        return [self._tree_item(item) for item in self._children_of(None)]

    def _tree_item(self, item: PlanItem) -> dict[str, Any]:
        return {
            **item.to_dict(),
            "children": [self._tree_item(child) for child in self._children_of(item.id)],
        }

    def _format_plan_list(self) -> str:
        if not self._items:
            return "No plan items."
        lines: list[str] = []
        for item in self._children_of(None):
            self._format_plan_item(item, lines, depth=0)
        return "\n".join(lines)

    def _format_plan_item(self, item: PlanItem, lines: list[str], *, depth: int) -> None:
        mark = "x" if item.completed else " "
        indent = "  " * depth
        lines.append(f"{indent}- [{mark}] {item.id}: {item.content}")
        for child in self._children_of(item.id):
            self._format_plan_item(child, lines, depth=depth + 1)

    def _format_plan_artifact(self) -> str:
        if not self._items:
            return "No plan items."
        lines: list[str] = []
        for item in self._children_of(None):
            self._format_plan_artifact_item(item, lines, depth=0)
        return "\n".join(lines)

    def _format_plan_artifact_item(
        self, item: PlanItem, lines: list[str], *, depth: int
    ) -> None:
        indent = "  " * depth
        text = f"{item.id}: {item.content}"
        rendered = f"~~{text}~~" if item.completed else text
        lines.append(f"{indent}- {rendered}")
        for child in self._children_of(item.id):
            self._format_plan_artifact_item(child, lines, depth=depth + 1)

    def _sync_artifact(self, *, status: str) -> None:
        self.upsert_artifact(
            "current-plan",
            artifact_type="plan",
            title="Current Plan",
            content=f"{self._format_plan_artifact()}\n",
            language="markdown",
            mime_type="text/markdown",
            status=status,
            metadata={
                "pending_count": len(self._incomplete_items()),
                "notification_cancelled": self._notification_cancelled,
                "cancel_reason": self._cancel_reason,
                "items": self._tree_items(),
            },
        )

    def _emit_plan_item_update(self, action: str, item: PlanItem | dict[str, Any]) -> None:
        item_data = item.to_dict() if isinstance(item, PlanItem) else item
        self.emit_plugin_event(
            "plugin.event",
            {
                "event_name": "plan.item.updated",
                "action": action,
                "item": item_data,
                "state": self._state_dict(),
                "title": "Plan updated",
                "message": f"{action.capitalize()} {item_data.get('id', 'plan item')}",
                "data": item_data,
            },
        )
