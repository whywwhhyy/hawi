from __future__ import annotations

import json
import time
from copy import deepcopy
from typing import Any

from .models import PlanEngineResult, PlanFoldRecord


class PlanFoldingMixin:
    """Completed-task context folding and recall helpers for PlanEngine."""

    fold_completed_tasks: bool
    fold_records: list[PlanFoldRecord]
    next_fold_number: int
    active_completion_tool_call_id: str | None

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
        if not folded_messages and self.fold_records:
            previous = self.fold_records[-1]
            for completed_item_id in completed_item_ids:
                if completed_item_id not in previous.completed_item_ids:
                    previous.completed_item_ids.append(completed_item_id)
            folded_context = previous.reference_dict()
            folded_context["enabled"] = True
            folded_context["skipped"] = True
            folded_context["referenced_fold_id"] = previous.fold_id
            folded_context["reason"] = (
                "No new messages appeared since the previous complete_plan_item "
                f"fold. Refer to `{previous.fold_id}` for the shared folded context."
            )
            return folded_context, None
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

    def select_folded_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        message_start: int | None,
        message_end: int | None,
    ) -> tuple[list[dict[str, Any]], int, int, str]:
        total = len(messages)
        if total == 0:
            return [], 1, 0, ""

        start = 1 if message_start is None else message_start
        end = total if message_end is None else message_end
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
