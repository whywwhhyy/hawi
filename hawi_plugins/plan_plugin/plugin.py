from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from hawi.plugin import HawiPlugin, HookResult, after_conversation, before_conversation, tool
from hawi.tool import ToolResult


PLAN_PROMPT_BEGIN = "<hawi-plan-mode>"
PLAN_PROMPT_END = "</hawi-plan-mode>"


@dataclass
class PlanItem:
    id: str
    content: str
    parent_id: str | None = None
    completed: bool = False
    created_at: float = 0.0
    completed_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "parent_id": self.parent_id,
            "completed": self.completed,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


class PlanPlugin(HawiPlugin):
    """Plan mode plugin.

    Provides tools for maintaining an explicit task plan and a conversation hook
    that nudges the agent to continue while open plan items remain.
    """

    def __init__(self) -> None:
        self._items: list[PlanItem] = []
        self._next_item_number = 1
        self._notification_cancelled = False
        self._cancel_reason = ""

    @classmethod
    def gui_config_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    @classmethod
    def gui_default_config(cls) -> dict:
        return {}

    def clone(self) -> "PlanPlugin":
        new_plugin = PlanPlugin()
        new_plugin._items = [
            PlanItem(
                id=item.id,
                content=item.content,
                parent_id=item.parent_id,
                completed=item.completed,
                created_at=item.created_at,
                completed_at=item.completed_at,
            )
            for item in self._items
        ]
        new_plugin._next_item_number = self._next_item_number
        new_plugin._notification_cancelled = self._notification_cancelled
        new_plugin._cancel_reason = self._cancel_reason
        return new_plugin

    @before_conversation
    def inject_plan_instructions(self, agent: Any, ctx: Any) -> None:
        """Inject plan mode guidance into the system prompt."""
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
            "- add_plan_item: add one concrete task item before or during multi-step work.\n"
            "  Pass parent_id to create a child item under an existing plan item.\n"
            "- complete_plan_item: mark a plan item complete as soon as it is actually done. "
            "Pass item_id='all' only when every open item is complete.\n"
            "- list_plan_items: inspect the current plan state.\n"
            "- cancel_plan_notification: stop automatic plan reminders only when remaining "
            "items are impossible, obsolete, or intentionally deferred; include a reason.\n"
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
        description="Add one concrete task item to the current plan.",
        parameters_schema={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Concrete task to add to the plan.",
                },
                "parent_id": {
                    "type": "string",
                    "description": "Optional parent plan item id for nested plan items.",
                },
            },
            "required": ["content"],
        },
    )
    def add_plan_item(self, content: str, parent_id: str | None = None) -> ToolResult:
        content = content.strip()
        if not content:
            return ToolResult(success=False, error="Plan item content cannot be empty.")
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

        item = PlanItem(
            id=f"P{self._next_item_number}",
            content=content,
            parent_id=normalized_parent_id,
            created_at=time.time(),
        )
        self._next_item_number += 1
        self._items.append(item)
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
            "is actually complete."
        ),
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
            },
            "required": ["item_id"],
        },
    )
    def complete_plan_item(self, item_id: str, complete_children: bool = False) -> ToolResult:
        item_id = item_id.strip()
        now = time.time()

        if item_id.lower() == "all":
            completed = []
            for item in self._items:
                if not item.completed:
                    item.completed = True
                    item.completed_at = now
                    completed.append(item.to_dict())
            self._sync_artifact(status="complete")
            for item_data in completed:
                self._emit_plan_item_update("completed", item_data)
            return ToolResult(
                success=True,
                output={
                    "completed": completed,
                    "tree": self._tree_items(),
                    "pending_count": len(self._incomplete_items()),
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
                completed.append(current.to_dict())
        self._sync_artifact(status="complete" if not self._incomplete_items() else "active")
        for item_data in completed:
            self._emit_plan_item_update("completed", item_data)
        return ToolResult(
            success=True,
            output={
                "item": item.to_dict(),
                "completed": completed,
                "tree": self._tree_items(),
                "pending_count": len(self._incomplete_items()),
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

    def _state_dict(self) -> dict[str, Any]:
        incomplete = self._incomplete_items()
        return {
            "items": self._tree_items(),
            "flat_items": [item.to_dict() for item in self._items],
            "pending_count": len(incomplete),
            "notification_cancelled": self._notification_cancelled,
            "cancel_reason": self._cancel_reason,
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

    def _sync_artifact(self, *, status: str) -> None:
        self.upsert_artifact(
            "current-plan",
            artifact_type="plan",
            title="Current Plan",
            content=f"# Current Plan\n\n{self._format_plan_list()}\n",
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
