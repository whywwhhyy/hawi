from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


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
    handoff_notes: str | None = None

    def reference_dict(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "item_id": self.item_id,
            "item_content": self.item_content,
            "summary": self.summary,
            "handoff_notes": self.handoff_notes,
            "completed_item_ids": self.completed_item_ids,
            "folded_message_count": len(self.messages),
            "message_previews": self.message_previews(),
            "read_tool": "read_completed_task_context",
            "note": (
                "Plan context folding is enabled. Detailed messages from this "
                "completed task were moved out of the active model context and "
                "stored in PlanPlugin memory. Use read_completed_task_context "
                "with this item_id or fold_id if later work needs the details."
            ),
        }

    def message_previews(self, max_chars: int = 100) -> list[dict[str, Any]]:
        previews: list[dict[str, Any]] = []
        for index, message in enumerate(self.messages, 1):
            previews.append(
                {
                    "index": index,
                    "role": message.get("role", "unknown"),
                    "preview": self._message_preview(message, max_chars=max_chars),
                }
            )
        return previews

    @classmethod
    def _message_preview(cls, message: dict[str, Any], *, max_chars: int) -> str:
        tool_preview = cls._tool_call_preview(message)
        if tool_preview:
            return tool_preview
        content = cls._plain_content_preview(message.get("content", []))
        if len(content) > max_chars:
            return content[:max_chars].rstrip() + "..."
        return content

    @staticmethod
    def _tool_call_preview(message: dict[str, Any]) -> str:
        content = message.get("content", [])
        if not isinstance(content, list):
            return ""
        parts = [
            part
            for part in content
            if isinstance(part, dict) and part.get("type") == "tool_call"
        ]
        if not parts:
            return ""
        rendered = []
        for part in parts:
            name = part.get("name", "unknown_tool")
            description = part.get("description")
            suffix = f" - {description}" if description else ""
            rendered.append(f"tool call `{name}`{suffix}")
        return "; ".join(rendered)

    @classmethod
    def _plain_content_preview(cls, content: Any) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return str(content)
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                parts.append(str(part))
                continue
            part_type = part.get("type")
            if part_type == "text":
                parts.append(str(part.get("text", "")))
            elif part_type == "tool_result":
                parts.append(cls._plain_content_preview(part.get("content", [])))
            elif part_type == "reasoning":
                reasoning = part.get("reasoning") or ""
                if reasoning:
                    parts.append(str(reasoning))
            elif part_type == "steer":
                parts.append(cls._plain_content_preview(part.get("content", [])))
            elif part_type != "tool_call":
                parts.append(str(part))
        return " ".join(text.strip() for text in parts if text and text.strip())


@dataclass
class PlanEngineResult:
    success: bool
    output: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    item_events: list[dict[str, Any]] = field(default_factory=list)
    plugin_event: dict[str, Any] | None = None
    fold_request: dict[str, Any] | None = None


class PlanEngine:
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
                completed=item.completed,
                created_at=item.created_at,
                completed_at=item.completed_at,
                kind=item.kind,
                completion_summary=item.completion_summary,
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

    def add_plan_item(
        self,
        content: str | None = None,
        parent_id: str | None = None,
        items: list[dict[str, Any]] | None = None,
        kind: str | None = None,
    ) -> PlanEngineResult:
        has_content = isinstance(content, str) and bool(content.strip())
        has_items = items is not None
        if has_content and has_items:
            return PlanEngineResult(
                success=False,
                error="Provide either content for a single item or items for a plan tree, not both.",
            )
        if not has_content and not has_items:
            return PlanEngineResult(
                success=False,
                error="Provide content for a single item or items for a plan tree.",
            )
        if has_items and kind is not None:
            return PlanEngineResult(
                success=False,
                error=(
                    "Top-level kind is only valid when adding a single item. "
                    "For tree mode, set kind on each node inside items."
                ),
            )

        normalized_kind, kind_error = self._normalize_kind(kind, path="kind")
        if kind_error:
            return PlanEngineResult(success=False, error=kind_error)

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
        if has_items:
            created, error = self._add_plan_item_tree(
                items or [],
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

        item = self._create_plan_item(
            content=content.strip() if isinstance(content, str) else "",
            parent_id=normalized_parent_id,
            created_at=now,
            kind=normalized_kind,
        )
        return PlanEngineResult(
            success=True,
            output={
                "item": item.to_dict(),
                "tree": self.tree_items(),
                "pending_count": len(self.incomplete_items()),
            },
            item_events=[{"action": "added", "item": item.to_dict()}],
        )

    def complete_plan_item(
        self,
        item_id: str,
        *,
        mark_all_children: bool = False,
        summary: str | None = None,
        handoff_notes: str | None = None,
        complete_children: bool | None = None,
    ) -> PlanEngineResult:
        item_id = item_id.strip() if isinstance(item_id, str) else ""
        if complete_children is not None:
            mark_all_children = bool(complete_children)
        summary_text = summary.strip() if isinstance(summary, str) else ""
        handoff_notes_text = (
            handoff_notes.strip() if isinstance(handoff_notes, str) else ""
        )
        if self.fold_completed_tasks and not summary_text:
            return PlanEngineResult(
                success=False,
                error=(
                    "summary is required when completed-task context folding is "
                    "enabled. Provide a concise summary of what was completed "
                    "in this task."
                ),
            )
        if self.fold_completed_tasks and not handoff_notes_text:
            return PlanEngineResult(
                success=False,
                error=(
                    "handoff_notes is required when completed-task context "
                    "folding is enabled. Provide the details later tasks must "
                    "remember, or explicitly say that there are no lasting notes."
                ),
            )
        now = time.time()

        if item_id.lower() == "all":
            completed = []
            for item in self.items:
                if not item.completed:
                    item.completed = True
                    item.completed_at = now
                    if summary_text:
                        item.completion_summary = summary_text
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
                    "folded_context": None,
                },
                item_events=[
                    {"action": "completed", "item": item_data}
                    for item_data in completed
                ],
                fold_request={
                    "item_id": "all",
                    "item_content": "All open plan items",
                    "summary": summary_text,
                    "handoff_notes": handoff_notes_text,
                    "completed_item_ids": [item["id"] for item in completed],
                },
            )

        item = self.find_item(item_id)
        if item is None:
            return PlanEngineResult(
                success=False,
                error=f"Unknown plan item id: {item_id}",
            )

        incomplete_descendants = self.incomplete_descendants(item.id)
        if incomplete_descendants and not mark_all_children:
            return PlanEngineResult(
                success=False,
                error=(
                    f"Plan item {item.id} has unfinished child task(s). "
                    "Pass mark_all_children=true only if these child tasks should "
                    "also be marked complete: "
                    f"{self._format_unfinished_child_list(incomplete_descendants)}"
                ),
                output={
                    "item": item.to_dict(),
                    "unfinished_children": [
                        child.to_dict() for child in incomplete_descendants
                    ],
                    "pending_count": len(self.incomplete_items()),
                },
            )

        items_to_complete = [item]
        if mark_all_children:
            items_to_complete.extend(self.descendants(item.id))
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
        return PlanEngineResult(
            success=True,
            output={
                "item": item.to_dict(),
                "completed": completed,
                "tree": self.tree_items(),
                "pending_count": len(self.incomplete_items()),
                "parent_review_required": review_required,
                "completion_summary": summary_text or None,
                "handoff_notes": handoff_notes_text or None,
                "folded_context": None,
            },
            item_events=[
                {"action": "completed", "item": item_data}
                for item_data in completed
            ],
            fold_request={
                "item_id": item.id,
                "item_content": item.content,
                "summary": summary_text,
                "handoff_notes": handoff_notes_text,
                "completed_item_ids": [item_data["id"] for item_data in completed],
            },
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

        if action == "abandon":
            abandoned_items = [item.to_dict() for item in self.items]
            abandoned_fold_count = len(self.fold_records)
            self.clear_plan_state()
            output = self.state_dict()
            output["abandoned_items"] = abandoned_items
            output["abandoned_fold_count"] = abandoned_fold_count
            output["abandon_reason"] = reason_text or None
            return PlanEngineResult(
                success=True,
                output=output,
                plugin_event={
                    "event_name": "plan.execution.abandoned",
                    "reason": reason_text,
                    "title": "Plan abandoned",
                    "message": reason_text or "Current plan has been abandoned.",
                    "data": {
                        "reason": reason_text,
                        "abandoned_items": abandoned_items,
                        "abandoned_fold_count": abandoned_fold_count,
                    },
                },
            )

        return PlanEngineResult(
            success=False,
            error="action must be 'pause', 'continue', or 'abandon'.",
        )

    def fold_completed_context(
        self,
        ctx: Any,
        *,
        item_id: str,
        item_content: str,
        summary: str,
        completed_item_ids: list[str],
        handoff_notes: str | None = None,
    ) -> tuple[dict[str, Any], PlanFoldRecord | None]:
        if not self.fold_completed_tasks:
            return {
                "enabled": False,
                "summary": summary or None,
                "handoff_notes": handoff_notes or None,
            }, None
        if not completed_item_ids:
            return {
                "enabled": True,
                "skipped": True,
                "reason": "No plan items were newly completed by this call.",
                "summary": summary,
                "handoff_notes": handoff_notes or None,
            }, None

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
                "handoff_notes": handoff_notes or None,
            }, None

        current_index = self.find_current_completion_message_index(messages)
        if current_index is None:
            return {
                "enabled": True,
                "skipped": True,
                "reason": (
                    "Could not locate the active complete_plan_item tool call in "
                    "the current context, so no messages were folded."
                ),
                "summary": summary,
                "handoff_notes": handoff_notes or None,
            }, None

        start_index = self.fold_start_index(messages, current_index)
        folded_messages = deepcopy(messages[start_index:current_index])
        if start_index < current_index:
            del messages[start_index:current_index]

        record = PlanFoldRecord(
            fold_id=f"PF{self.next_fold_number}",
            item_id=item_id,
            item_content=item_content,
            summary=summary,
            messages=folded_messages,
            completed_item_ids=list(completed_item_ids),
            created_at=time.time(),
            handoff_notes=handoff_notes or None,
        )
        self.next_fold_number += 1
        self.fold_records.append(record)

        folded_context = record.reference_dict()
        folded_context["enabled"] = True
        folded_context["skipped"] = False
        return folded_context, record

    def find_current_completion_message_index(
        self,
        messages: list[dict[str, Any]],
    ) -> int | None:
        if self.active_completion_tool_call_id:
            for index in range(len(messages) - 1, -1, -1):
                message = messages[index]
                if self._assistant_has_tool_call_id(
                    message,
                    self.active_completion_tool_call_id,
                ):
                    return index

        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if self._assistant_has_tool_call_name(message, "complete_plan_item"):
                return index
        return None

    def fold_start_index(
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

    def find_fold_record(
        self,
        *,
        item_id: str,
        fold_id: str | None,
    ) -> PlanFoldRecord | None:
        normalized_fold_id = fold_id.strip().lower() if isinstance(fold_id, str) else ""
        if normalized_fold_id:
            for record in reversed(self.fold_records):
                if record.fold_id.lower() == normalized_fold_id:
                    return record

        normalized_item_id = item_id.strip().lower() if isinstance(item_id, str) else ""
        if normalized_item_id:
            for record in reversed(self.fold_records):
                if record.item_id.lower() == normalized_item_id or any(
                    completed_id.lower() == normalized_item_id
                    for completed_id in record.completed_item_ids
                ):
                    return record
        return None

    def search_folded_contexts(
        self,
        *,
        query: str,
        item_id: str = "",
        fold_id: str | None = None,
        case_sensitive: bool = False,
        max_matches: int = 20,
        context_chars: int = 240,
        max_chars: int = 20000,
    ) -> PlanEngineResult:
        query_text = query.strip() if isinstance(query, str) else ""
        if not query_text:
            return PlanEngineResult(
                success=False,
                error="query must be a non-empty string when searching folded contexts.",
            )

        records = self._fold_records_for_lookup(item_id=item_id, fold_id=fold_id)
        if not records:
            if not self.fold_records:
                return PlanEngineResult(
                    success=False,
                    error="No completed task contexts have been folded yet.",
                )
            return PlanEngineResult(
                success=False,
                error=(
                    "No folded context found for "
                    f"item_id={item_id!r}, fold_id={fold_id!r}."
                ),
            )

        max_matches = self.normalize_int(
            max_matches,
            default=20,
            minimum=1,
            maximum=100,
        )
        context_chars = self.normalize_int(
            context_chars,
            default=240,
            minimum=0,
            maximum=4000,
        )
        max_chars = self.normalize_int(
            max_chars,
            default=20000,
            minimum=1,
            maximum=200000,
        )

        returned_matches: list[dict[str, Any]] = []
        match_count = 0
        snippet_chars = 0
        snippet_limit_reached = False
        match_limit_reached = False

        for record in records:
            for candidate in self._iter_fold_record_matches(
                record,
                query=query_text,
                case_sensitive=case_sensitive,
                context_chars=context_chars,
            ):
                match_count += 1
                if len(returned_matches) >= max_matches:
                    match_limit_reached = True
                    continue

                snippet = str(candidate["snippet"])
                remaining_chars = max_chars - snippet_chars
                if remaining_chars <= 0:
                    snippet_limit_reached = True
                    continue
                if len(snippet) > remaining_chars:
                    snippet, _ = self.truncate_text(
                        snippet,
                        remaining_chars,
                        marker="\n[Snippet truncated by max_chars.]",
                    )
                    snippet_limit_reached = True

                candidate["snippet"] = snippet
                returned_matches.append(candidate)
                snippet_chars += len(snippet)

        truncated = match_limit_reached or snippet_limit_reached
        truncation_reasons = []
        if match_limit_reached:
            truncation_reasons.append("max_matches")
        if snippet_limit_reached:
            truncation_reasons.append("max_chars")

        return PlanEngineResult(
            success=True,
            output={
                "mode": "search",
                "query": query_text,
                "case_sensitive": case_sensitive,
                "searched_context_count": len(records),
                "searched_fold_ids": [record.fold_id for record in records],
                "match_count": match_count,
                "matches_returned": len(returned_matches),
                "matches": returned_matches,
                "max_matches": max_matches,
                "max_chars": max_chars,
                "context_chars": context_chars,
                "truncated": truncated,
                "truncation_reason": (
                    ", ".join(truncation_reasons) if truncation_reasons else None
                ),
            },
        )

    def format_folded_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        start_index: int = 1,
    ) -> str:
        if not messages:
            return "[No detailed messages were folded for this completion.]"

        sections: list[str] = []
        for index, message in enumerate(messages, start_index):
            role = message.get("role", "unknown")
            name = message.get("name")
            title = f"Message {index} ({role})"
            if name:
                title += f" name={name}"
            content = self._format_content(message.get("content", []))
            sections.append(f"## {title}\n{content}")
        return "\n\n".join(sections)

    @staticmethod
    def truncate_text(
        text: str,
        max_chars: int,
        *,
        marker: str,
    ) -> tuple[str, bool]:
        if len(text) <= max_chars:
            return text, False
        if max_chars <= 0:
            return "", True
        if max_chars <= len(marker):
            return marker[:max_chars], True
        return text[: max_chars - len(marker)].rstrip() + marker, True

    @staticmethod
    def normalize_int(
        value: Any,
        *,
        default: int,
        minimum: int,
        maximum: int,
    ) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return min(max(parsed, minimum), maximum)

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
            if not item.completed
        ]

    def incomplete_items(self) -> list[PlanItem]:
        return [item for item in self.items if not item.completed]

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
            id=f"P{self.next_item_number}",
            content=content,
            parent_id=parent_id,
            created_at=created_at,
            kind=kind,
        )
        self.next_item_number += 1
        self.items.append(item)
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

        def add_nodes(
            current_nodes: list[dict[str, Any]], current_parent_id: str | None
        ) -> None:
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
                description = part.get("description")
                fields = [
                    "[tool_call]",
                    f"id={part.get('id', '')}",
                    f"name={part.get('name', '')}",
                ]
                if description:
                    fields.append(f"description={description}")
                fields.append(
                    f"arguments={self._safe_json(part.get('arguments', {}))}"
                )
                parts.append(" ".join(fields))
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

    def _fold_records_for_lookup(
        self,
        *,
        item_id: str,
        fold_id: str | None,
    ) -> list[PlanFoldRecord]:
        if item_id or fold_id:
            record = self.find_fold_record(item_id=item_id, fold_id=fold_id)
            return [record] if record is not None else []
        return list(reversed(self.fold_records))

    def _iter_fold_record_matches(
        self,
        record: PlanFoldRecord,
        *,
        query: str,
        case_sensitive: bool,
        context_chars: int,
    ) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        fields: list[tuple[str, int | None, str, str]] = [
            ("item_content", None, "plan_item", record.item_content),
            ("summary", None, "summary", record.summary),
        ]
        if record.handoff_notes:
            fields.append(("handoff_notes", None, "handoff_notes", record.handoff_notes))
        for index, message in enumerate(record.messages, 1):
            role = str(message.get("role", "unknown"))
            content = self._format_content(message.get("content", []))
            fields.append(("message", index, role, content))

        for source, message_index, role, text in fields:
            for start in self._find_query_positions(
                text,
                query=query,
                case_sensitive=case_sensitive,
            ):
                snippet = self._make_search_snippet(
                    text,
                    start=start,
                    query_length=len(query),
                    context_chars=context_chars,
                )
                matches.append(
                    {
                        "fold_id": record.fold_id,
                        "item_id": record.item_id,
                        "item_content": record.item_content,
                        "source": source,
                        "message_index": message_index,
                        "role": role,
                        "start": start,
                        "end": start + len(query),
                        "snippet": snippet,
                    }
                )
        return matches

    @staticmethod
    def _find_query_positions(
        text: str,
        *,
        query: str,
        case_sensitive: bool,
    ) -> list[int]:
        if not text:
            return []
        haystack = text if case_sensitive else text.lower()
        needle = query if case_sensitive else query.lower()
        positions: list[int] = []
        start = 0
        while True:
            index = haystack.find(needle, start)
            if index < 0:
                return positions
            positions.append(index)
            start = index + max(1, len(needle))

    @staticmethod
    def _make_search_snippet(
        text: str,
        *,
        start: int,
        query_length: int,
        context_chars: int,
    ) -> str:
        snippet_start = max(0, start - context_chars)
        snippet_end = min(len(text), start + query_length + context_chars)
        prefix = "..." if snippet_start > 0 else ""
        suffix = "..." if snippet_end < len(text) else ""
        return f"{prefix}{text[snippet_start:snippet_end]}{suffix}"

    @staticmethod
    def _safe_json(value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            return str(value)

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
            if parent is None or parent.completed:
                break
            children = self.children_of(parent.id)
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
                        "All children are complete, but this item is exploratory - "
                        "decide whether the work is truly done (call complete_plan_item) "
                        "or add follow-up children (call add_plan_item)."
                    ),
                }
            )
            break
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
        lines.append(f"{indent}- [{mark}] {item.id}: {item.content}")
        for child in self.children_of(item.id):
            self._format_plan_item(child, lines, depth=depth + 1)

    def _format_plan_artifact_item(
        self, item: PlanItem, lines: list[str], *, depth: int
    ) -> None:
        indent = "  " * depth
        text = f"{item.id}: {item.content}"
        rendered = f"~~{text}~~" if item.completed else text
        lines.append(f"{indent}- {rendered}")
        for child in self.children_of(item.id):
            self._format_plan_artifact_item(child, lines, depth=depth + 1)
