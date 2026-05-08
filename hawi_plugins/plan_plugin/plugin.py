from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from hawi.plugin import HawiPlugin, HookResult, after_conversation, before_conversation, tool
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "parent_id": self.parent_id,
            "completed": self.completed,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "kind": self.kind,
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
                kind=item.kind,
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
            "before moving on.\n"
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
                    "parent_review_required": [],
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
        auto_completed, review_required = self._propagate_completion_upward(item, now)
        completed.extend(auto_completed)
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
                "parent_review_required": review_required,
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
