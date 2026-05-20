from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


PLAN_ITEM_COMPLETION_MODES = (
    "auto_complete",
    "manual_mark",
)
PLAN_ITEM_DEFAULT_COMPLETION_MODE = "auto_complete"
PLAN_ITEM_STATUSES = (
    "open",
    "completed",
    "blocked",
    "deferred",
    "canceled",
    "obsolete",
)
PLAN_ITEM_DEFAULT_STATUS = "open"


@dataclass
class PlanItem:
    id: str
    content: str
    parent_id: str | None = None
    status: str = PLAN_ITEM_DEFAULT_STATUS
    completed: bool = False
    created_at: float = 0.0
    completed_at: float | None = None
    completion_mode: str = PLAN_ITEM_DEFAULT_COMPLETION_MODE
    completion_summary: str | None = None
    status_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "parent_id": self.parent_id,
            "status": self.status,
            "completed": self.completed,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "completion_mode": self.completion_mode,
            "completion_summary": self.completion_summary,
            "status_reason": self.status_reason,
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
            "read_tool": "recall_completed_task",
            "note": (
                "Plan context folding is enabled. Detailed messages from this "
                "completed task were moved out of the active model context and "
                "stored in PlanPlugin memory. Use recall_completed_task "
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
