from __future__ import annotations

import time
from copy import deepcopy
from typing import Any

from .folding import PlanFoldingMixin
from .models import (
    PLAN_ITEM_COMPLETION_MODES,
    PLAN_ITEM_DEFAULT_COMPLETION_MODE,
    PLAN_ITEM_STATUSES,
    PlanEngineResult,
    PlanFoldRecord,
    PlanItem,
)


class PlanEngine(PlanFoldingMixin):
    """Stateful plan execution engine used by PlanPlugin."""

    def __init__(self, fold_completed_tasks: bool = False) -> None:
        self.items: list[PlanItem] = []
        self.next_item_number = 1
        self.plan_paused = False
        self.pause_reason = ""
        self.fold_completed_tasks = bool(fold_completed_tasks)
        self.fold_records: list[PlanFoldRecord] = []
        self.next_fold_number = 1
        self.active_completion_tool_call_id: str | None = None

    def clone(self) -> "PlanEngine":
        engine = PlanEngine(fold_completed_tasks=self.fold_completed_tasks)
        engine.items = [
            PlanItem(
                id=item.id,
                content=item.content,
                parent_id=item.parent_id,
                status=item.status,
                completed=item.completed,
                created_at=item.created_at,
                completed_at=item.completed_at,
                completion_mode=item.completion_mode,
                completion_summary=item.completion_summary,
                status_reason=item.status_reason,
            )
            for item in self.items
        ]
        engine.next_item_number = self.next_item_number
        engine.plan_paused = self.plan_paused
        engine.pause_reason = self.pause_reason
        engine.fold_records = [
            PlanFoldRecord(
                fold_id=record.fold_id,
                item_id=record.item_id,
                item_content=record.item_content,
                summary=record.summary,
                messages=deepcopy(record.messages),
                completed_item_ids=list(record.completed_item_ids),
                created_at=record.created_at,
                handoff_notes=record.handoff_notes,
            )
            for record in self.fold_records
        ]
        engine.next_fold_number = self.next_fold_number
        engine.active_completion_tool_call_id = self.active_completion_tool_call_id
        return engine

    def add_plan_items(
        self,
        parent_id: str | None = None,
        items: list[dict[str, Any]] | None = None,
    ) -> PlanEngineResult:
        normalized_parent_id = (
            parent_id.strip()
            if isinstance(parent_id, str) and parent_id.strip()
            else None
        )
        if (
            normalized_parent_id is not None
            and self.find_item(normalized_parent_id) is None
        ):
            return PlanEngineResult(
                success=False,
                error=f"Unknown parent plan item id: {normalized_parent_id}",
            )

        now = time.time()
        created, error = self._add_plan_items_tree(
            items,
            parent_id=normalized_parent_id,
            created_at=now,
        )
        if error:
            return PlanEngineResult(success=False, error=error)
        return PlanEngineResult(
            success=True,
            output={
                "items": [item.to_dict() for item in created],
                "tree": self.tree_items(),
                "pending_count": len(self.incomplete_items()),
            },
            item_events=[
                {"action": "added", "item": item.to_dict()} for item in created
            ],
        )

    def complete_plan_item(
        self,
        item_id: str | None = None,
        *,
        item_ids: list[str] | None = None,
        mark_all_children: bool = False,
        fold_context: bool = False,
        summary: str | None = None,
        handoff_notes: str | None = None,
    ) -> PlanEngineResult:
        should_fold_context = bool(fold_context)
        summary_text = summary.strip() if isinstance(summary, str) else ""
        handoff_notes_text = (
            handoff_notes.strip() if isinstance(handoff_notes, str) else ""
        )
        if should_fold_context and not self.fold_completed_tasks:
            return PlanEngineResult(
                success=False,
                error=(
                    "fold_context requires completed-task context folding to be "
                    "enabled in PlanPlugin config."
                ),
            )
        if should_fold_context and not summary_text:
            return PlanEngineResult(
                success=False,
                error=(
                    "summary is required when fold_context is true. Provide a "
                    "concise summary of what was completed in this task."
                ),
            )
        if should_fold_context and not handoff_notes_text:
            return PlanEngineResult(
                success=False,
                error=(
                    "handoff_notes is required when fold_context is true. "
                    "Provide the details later tasks must remember, or explicitly "
                    "say that there are no lasting notes."
                ),
            )
        now = time.time()

        requested_ids, requested_ids_error = self._normalize_completion_item_ids(
            item_id,
            item_ids,
        )
        if requested_ids_error:
            return PlanEngineResult(success=False, error=requested_ids_error)

        if requested_ids == ["all"]:
            completed = []
            for item in self.incomplete_items():
                self._mark_item_completed(item, now, summary_text=summary_text)
                completed.append(item.to_dict())
            return PlanEngineResult(
                success=True,
                output={
                    "completed": completed,
                    "tree": self.tree_items(),
                    "pending_count": len(self.incomplete_items()),
                    "parent_review_required": [],
                    "completion_summary": summary_text or None,
                    "handoff_notes": handoff_notes_text or None,
                    "fold_context": should_fold_context,
                    "folded_context": None,
                },
                item_events=[
                    {"action": "completed", "item": item_data}
                    for item_data in completed
                ],
                fold_request=(
                    {
                        "item_id": "all",
                        "item_content": "All open plan items",
                        "summary": summary_text,
                        "handoff_notes": handoff_notes_text,
                        "completed_item_ids": [item["id"] for item in completed],
                    }
                    if should_fold_context
                    else None
                ),
            )

        if any(current_id.lower() == "all" for current_id in requested_ids):
            return PlanEngineResult(
                success=False,
                error="Use item_id='all' by itself, not inside item_ids.",
            )

        requested_items: list[PlanItem] = []
        missing_ids: list[str] = []
        for current_id in requested_ids:
            item = self.find_item(current_id)
            if item is None:
                missing_ids.append(current_id)
            else:
                requested_items.append(item)
        if missing_ids:
            return PlanEngineResult(
                success=False,
                error=f"Unknown plan item id(s): {', '.join(missing_ids)}",
            )

        items_to_complete = self._completion_closure(
            requested_items,
            mark_all_children=mark_all_children,
        )
        requested_id_set = {item.id for item in requested_items}
        forced_descendants = [
            item
            for item in items_to_complete
            if item.id not in requested_id_set and not item.completed
        ]
        closure_ids = {item.id for item in items_to_complete}
        blocked_children_by_item: list[tuple[PlanItem, list[PlanItem]]] = []
        if not mark_all_children:
            for item in requested_items:
                incomplete_descendants = [
                    descendant
                    for descendant in self.incomplete_descendants(item.id)
                    if descendant.id not in closure_ids
                ]
                if incomplete_descendants:
                    blocked_children_by_item.append((item, incomplete_descendants))
        if blocked_children_by_item:
            blocked_item, incomplete_descendants = blocked_children_by_item[0]
            return PlanEngineResult(
                success=False,
                error=(
                    f"Plan item {blocked_item.id} has unfinished child task(s). "
                    "Pass mark_all_children=true only if these child tasks should "
                    "also be marked complete: "
                    f"{self._format_unfinished_child_list(incomplete_descendants)}"
                ),
                output={
                    "item": blocked_item.to_dict(),
                    "items": [item.to_dict() for item in requested_items],
                    "unfinished_children": [
                        child.to_dict() for child in incomplete_descendants
                    ],
                    "pending_count": len(self.incomplete_items()),
                },
            )

        completed = []
        for current in items_to_complete:
            if not current.completed:
                self._mark_item_completed(
                    current,
                    now,
                    summary_text=summary_text,
                )
                completed.append(current.to_dict())
        auto_completed, review_required = self._propagate_batch_completion_upward(
            items_to_complete,
            now,
        )
        seen_completed_ids = {item_data["id"] for item_data in completed}
        for item_data in auto_completed:
            if item_data["id"] not in seen_completed_ids:
                completed.append(item_data)
                seen_completed_ids.add(item_data["id"])

        fold_item_id, fold_item_content = self._fold_identity_for_items(
            requested_items,
        )
        return PlanEngineResult(
            success=True,
            output={
                "item": requested_items[0].to_dict() if len(requested_items) == 1 else None,
                "items": [item.to_dict() for item in requested_items],
                "completed": completed,
                "tree": self.tree_items(),
                "pending_count": len(self.incomplete_items()),
                "parent_review_required": review_required,
                "completion_summary": summary_text or None,
                "handoff_notes": handoff_notes_text or None,
                "fold_context": should_fold_context,
                "marked_by_mark_all_children": [
                    item.to_dict() for item in forced_descendants
                ],
                "folded_context": None,
            },
            item_events=[
                {"action": "completed", "item": item_data}
                for item_data in completed
            ],
            fold_request=(
                {
                    "item_id": fold_item_id,
                    "item_content": fold_item_content,
                    "summary": summary_text,
                    "handoff_notes": handoff_notes_text,
                    "completed_item_ids": [item_data["id"] for item_data in completed],
                }
                if should_fold_context
                else None
            ),
        )

    def control(self, action: str, reason: str | None = None) -> PlanEngineResult:
        action = action.strip().lower() if isinstance(action, str) else ""
        reason_text = reason.strip() if isinstance(reason, str) else ""

        if action == "pause":
            if not reason_text:
                return PlanEngineResult(
                    success=False,
                    error="reason is required when pausing plan execution.",
                )
            self.plan_paused = True
            self.pause_reason = reason_text
            return PlanEngineResult(
                success=True,
                output=self.state_dict(),
                plugin_event={
                    "event_name": "plan.execution.paused",
                    "reason": reason_text,
                    "title": "Plan execution paused",
                    "message": reason_text,
                    "data": {"reason": reason_text},
                },
            )

        if action == "continue":
            self.plan_paused = False
            self.pause_reason = ""
            return PlanEngineResult(
                success=True,
                output=self.state_dict(),
                plugin_event={
                    "event_name": "plan.execution.resumed",
                    "title": "Plan execution resumed",
                    "message": "Plan execution has resumed.",
                    "data": {},
                },
            )

        if action == "clear":
            cleared_items = [item.to_dict() for item in self.items]
            cleared_fold_count = len(self.fold_records)
            self.clear_plan_state()
            output = self.state_dict()
            output["cleared_items"] = cleared_items
            output["cleared_fold_count"] = cleared_fold_count
            output["clear_reason"] = reason_text or None
            return PlanEngineResult(
                success=True,
                output=output,
                plugin_event={
                    "event_name": "plan.execution.cleared",
                    "reason": reason_text,
                    "title": "Plan cleared",
                    "message": reason_text or "Current plan has been cleared.",
                    "data": {
                        "reason": reason_text,
                        "cleared_items": cleared_items,
                        "cleared_fold_count": cleared_fold_count,
                    },
                },
            )

        return PlanEngineResult(
            success=False,
            error="action must be 'pause', 'continue', or 'clear'.",
        )

    def update_plan_items_status(
        self,
        item_id: str | None = None,
        *,
        item_ids: list[str] | None = None,
        status: str,
        reason: str | None = None,
    ) -> PlanEngineResult:
        normalized_status, status_error = self._normalize_status(
            status,
            path="status",
            allowed={"open", "blocked", "deferred", "canceled", "obsolete"},
        )
        if status_error:
            return PlanEngineResult(success=False, error=status_error)

        requested_ids, requested_ids_error = self._normalize_completion_item_ids(
            item_id,
            item_ids,
        )
        if requested_ids_error:
            return PlanEngineResult(success=False, error=requested_ids_error)

        if any(current_id.lower() == "all" for current_id in requested_ids):
            if requested_ids != ["all"]:
                return PlanEngineResult(
                    success=False,
                    error="Use item_id='all' by itself, not inside item_ids.",
                )
            items_to_update = list(self.items)
        else:
            items_to_update = []
            missing_ids: list[str] = []
            for current_id in requested_ids:
                item = self.find_item(current_id)
                if item is None:
                    missing_ids.append(current_id)
                else:
                    items_to_update.append(item)
            if missing_ids:
                return PlanEngineResult(
                    success=False,
                    error=f"Unknown plan item id(s): {', '.join(missing_ids)}",
                )

        reason_text = reason.strip() if isinstance(reason, str) else ""
        updated = []
        for item in items_to_update:
            if normalized_status == "open":
                item.status = "open"
                item.completed = False
                item.completed_at = None
                item.completion_summary = None
                item.status_reason = reason_text or None
            else:
                item.status = normalized_status
                item.completed = False
                item.completed_at = None
                item.status_reason = reason_text or None
            updated.append(item.to_dict())

        return PlanEngineResult(
            success=True,
            output={
                "updated": updated,
                "status": normalized_status,
                "reason": reason_text or None,
                "tree": self.tree_items(),
                "pending_count": len(self.incomplete_items()),
            },
            item_events=[
                {"action": "status_updated", "item": item_data}
                for item_data in updated
            ],
        )

    def state_dict(self) -> dict[str, Any]:
        incomplete = self.incomplete_items()
        return {
            "items": self.tree_items(),
            "flat_items": [item.to_dict() for item in self.items],
            "pending_count": len(incomplete),
            "plan_paused": self.plan_paused,
            "pause_reason": self.pause_reason,
            "context_folding_enabled": self.fold_completed_tasks,
            "folded_contexts": [
                record.reference_dict() for record in self.fold_records
            ],
        }

    def plan_status(self) -> str:
        if self.plan_paused:
            return "paused"
        if self.items and not self.incomplete_items():
            if any(item.status in ("blocked", "deferred") for item in self.items):
                return "paused"
            return "complete"
        if self.items:
            return "active"
        return "empty"

    def find_item(self, item_id: str) -> PlanItem | None:
        normalized = item_id.strip().lower()
        for item in self.items:
            if item.id.lower() == normalized:
                return item
        return None

    def children_of(self, parent_id: str | None) -> list[PlanItem]:
        return [item for item in self.items if item.parent_id == parent_id]

    def descendants(self, parent_id: str) -> list[PlanItem]:
        descendants: list[PlanItem] = []
        for child in self.children_of(parent_id):
            descendants.append(child)
            descendants.extend(self.descendants(child.id))
        return descendants

    def incomplete_descendants(self, parent_id: str) -> list[PlanItem]:
        return [
            item
            for item in self.descendants(parent_id)
            if item.status == "open"
        ]

    def incomplete_items(self) -> list[PlanItem]:
        return [item for item in self.items if item.status == "open"]

    def clear_plan_state(self) -> None:
        self.items.clear()
        self.next_item_number = 1
        self.plan_paused = False
        self.pause_reason = ""
        self.fold_records.clear()
        self.next_fold_number = 1
        self.active_completion_tool_call_id = None

    def tree_items(self) -> list[dict[str, Any]]:
        return [self._tree_item(item) for item in self.children_of(None)]

    def format_plan_list(self) -> str:
        if not self.items:
            return "No plan items."
        lines: list[str] = []
        for item in self.children_of(None):
            self._format_plan_item(item, lines, depth=0)
        return "\n".join(lines)

    def format_plan_artifact(self) -> str:
        if not self.items:
            return "No plan items."
        lines: list[str] = []
        for item in self.children_of(None):
            self._format_plan_artifact_item(item, lines, depth=0)
        return "\n".join(lines)

    def _normalize_completion_mode(
        self,
        completion_mode: Any,
        *,
        path: str,
    ) -> tuple[str, str]:
        if completion_mode is None:
            return PLAN_ITEM_DEFAULT_COMPLETION_MODE, ""
        if not isinstance(completion_mode, str):
            return "", f"{path} must be a string when provided."
        normalized = completion_mode.strip().lower()
        if not normalized:
            return PLAN_ITEM_DEFAULT_COMPLETION_MODE, ""
        if normalized not in PLAN_ITEM_COMPLETION_MODES:
            allowed = ", ".join(repr(mode) for mode in PLAN_ITEM_COMPLETION_MODES)
            return "", f"{path} must be one of {allowed}."
        return normalized, ""

    def _normalize_status(
        self,
        status: Any,
        *,
        path: str,
        allowed: set[str] | None = None,
    ) -> tuple[str, str]:
        if not isinstance(status, str):
            return "", f"{path} must be a string."
        normalized = status.strip().lower()
        allowed_statuses = set(PLAN_ITEM_STATUSES) if allowed is None else allowed
        if normalized not in allowed_statuses:
            allowed_values = ", ".join(repr(value) for value in sorted(allowed_statuses))
            return "", f"{path} must be one of {allowed_values}."
        return normalized, ""

    def _normalize_completion_item_ids(
        self,
        item_id: Any,
        item_ids: Any,
    ) -> tuple[list[str], str]:
        has_item_id = item_id is not None
        has_item_ids = item_ids is not None
        if has_item_id and has_item_ids:
            return [], "Provide either item_id or item_ids, not both."
        if not has_item_id and not has_item_ids:
            return [], "Provide item_id for one item or item_ids for multiple items."

        raw_ids: list[Any]
        if has_item_ids:
            if not isinstance(item_ids, list):
                return [], "item_ids must be a list of plan item ids."
            raw_ids = item_ids
        else:
            if not isinstance(item_id, str):
                return [], "item_id must be a string when provided."
            raw_ids = [item_id]

        normalized: list[str] = []
        seen: set[str] = set()
        for index, raw_id in enumerate(raw_ids):
            if not isinstance(raw_id, str):
                return [], f"item_ids[{index}] must be a string."
            current_id = raw_id.strip()
            if not current_id:
                return [], f"item_ids[{index}] must be a non-empty string."
            dedupe_key = current_id.lower()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            normalized.append(current_id)

        if not normalized:
            return [], "Provide at least one plan item id."
        return normalized, ""

    def _completion_closure(
        self,
        requested_items: list[PlanItem],
        *,
        mark_all_children: bool,
    ) -> list[PlanItem]:
        items: list[PlanItem] = []
        seen: set[str] = set()

        def add_item(item: PlanItem) -> None:
            if item.id in seen:
                return
            seen.add(item.id)
            items.append(item)

        for item in requested_items:
            add_item(item)
            if mark_all_children:
                for descendant in self.descendants(item.id):
                    add_item(descendant)
        return items

    @staticmethod
    def _fold_identity_for_items(items: list[PlanItem]) -> tuple[str, str]:
        if not items:
            return "unknown", "Unknown completed plan item"
        if len(items) == 1:
            return items[0].id, items[0].content
        item_summaries = "; ".join(
            f"{item.id}: {item.content}" for item in items
        )
        return items[0].id, f"Multiple plan items: {item_summaries}"

    def _create_plan_item(
        self,
        *,
        content: str,
        parent_id: str | None,
        created_at: float,
        completion_mode: str = PLAN_ITEM_DEFAULT_COMPLETION_MODE,
    ) -> PlanItem:
        item = PlanItem(
            id=f"P{self.next_item_number}",
            content=content,
            parent_id=parent_id,
            created_at=created_at,
            completion_mode=completion_mode,
        )
        self.next_item_number += 1
        self.items.append(item)
        return item

    @staticmethod
    def _mark_item_completed(
        item: PlanItem,
        completed_at: float,
        *,
        summary_text: str = "",
    ) -> None:
        item.status = "completed"
        item.completed = True
        item.completed_at = completed_at
        item.status_reason = None
        if summary_text:
            item.completion_summary = summary_text

    def _add_plan_items_tree(
        self,
        nodes: Any,
        *,
        parent_id: str | None,
        created_at: float,
    ) -> tuple[list[PlanItem], str]:
        validation_error = self._validate_plan_item_nodes(nodes, path="items")
        if validation_error:
            return [], validation_error

        created: list[PlanItem] = []

        def add_nodes(
            current_nodes: list[dict[str, Any]], current_parent_id: str | None
        ) -> None:
            for node in current_nodes:
                node_completion_mode, _ = self._normalize_completion_mode(
                    node.get("completion_mode"),
                    path="completion_mode",
                )
                item = self._create_plan_item(
                    content=str(node["content"]).strip(),
                    parent_id=current_parent_id,
                    created_at=created_at,
                    completion_mode=node_completion_mode,
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
            if "completion_mode" in node:
                _, completion_mode_error = self._normalize_completion_mode(
                    node.get("completion_mode"),
                    path=f"{item_path}.completion_mode",
                )
                if completion_mode_error:
                    return completion_mode_error
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

    def _format_unfinished_child_list(self, children: list[PlanItem]) -> str:
        return "; ".join(f"{child.id}: {child.content}" for child in children)

    def _propagate_completion_upward(
        self, start: PlanItem, now: float
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        auto_completed: list[dict[str, Any]] = []
        review_required: list[dict[str, Any]] = []
        cursor = start
        while cursor.parent_id is not None:
            parent = self.find_item(cursor.parent_id)
            if parent is None or parent.completed or parent.status != "open":
                break
            children = self.children_of(parent.id)
            if not children or not all(child.completed for child in children):
                break
            if parent.completion_mode == "auto_complete":
                self._mark_item_completed(parent, now)
                auto_completed.append(parent.to_dict())
                cursor = parent
                continue
            review_required.append(
                {
                    "id": parent.id,
                    "content": parent.content,
                    "completion_mode": parent.completion_mode,
                    "reason": (
                        "All children are complete, but this item uses manual_mark. "
                        "Decide whether the parent is now done (call complete_plan_item) "
                        "or needs follow-up children (call add_plan_items)."
                    ),
                }
            )
            break
        return auto_completed, review_required

    def _propagate_batch_completion_upward(
        self,
        completed_items: list[PlanItem],
        now: float,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        auto_completed: list[dict[str, Any]] = []
        review_required: list[dict[str, Any]] = []
        seen_auto_ids: set[str] = set()
        seen_review_ids: set[str] = set()
        for item in completed_items:
            current_auto, current_review = self._propagate_completion_upward(
                item,
                now,
            )
            for item_data in current_auto:
                item_id = item_data.get("id", "")
                if item_id in seen_auto_ids:
                    continue
                seen_auto_ids.add(item_id)
                auto_completed.append(item_data)
            for item_data in current_review:
                item_id = item_data.get("id", "")
                if item_id in seen_review_ids:
                    continue
                seen_review_ids.add(item_id)
                review_required.append(item_data)
        return auto_completed, review_required

    def _tree_item(self, item: PlanItem) -> dict[str, Any]:
        return {
            **item.to_dict(),
            "children": [
                self._tree_item(child) for child in self.children_of(item.id)
            ],
        }

    def _format_plan_item(
        self, item: PlanItem, lines: list[str], *, depth: int
    ) -> None:
        mark = "x" if item.completed else " "
        indent = "  " * depth
        suffix = self._status_suffix(item)
        lines.append(f"{indent}- [{mark}] {item.id}: {item.content}{suffix}")
        for child in self.children_of(item.id):
            self._format_plan_item(child, lines, depth=depth + 1)

    def _format_plan_artifact_item(
        self, item: PlanItem, lines: list[str], *, depth: int
    ) -> None:
        indent = "  " * depth
        text = f"{item.id}: {item.content}{self._status_suffix(item)}"
        rendered = f"~~{text}~~" if item.completed else text
        lines.append(f"{indent}- {rendered}")
        for child in self.children_of(item.id):
            self._format_plan_artifact_item(child, lines, depth=depth + 1)

    @staticmethod
    def _status_suffix(item: PlanItem) -> str:
        if item.status in ("open", "completed"):
            return ""
        if item.status_reason:
            return f" ({item.status}: {item.status_reason})"
        return f" ({item.status})"
