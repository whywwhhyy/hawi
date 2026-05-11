"""Utility helpers for managed sub-agents."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from hawi.events import Event
from hawi.models import ContentPart


def drop_trailing_unanswered_tool_call_turn(messages: list[dict[str, Any]]) -> int:
    """Drop the trailing parent tool-call turn if it is still in progress.

    Forking can happen while a parent tool is still executing. At that moment
    the parent context already contains the assistant tool_call message, but
    its matching tool result has not been appended yet. The forked child should
    see the last stable context plus its own new task message, not the parent's
    half-finished tool-calling turn.
    """
    if not messages:
        return 0

    assistant_index = len(messages) - 1
    while assistant_index >= 0 and messages[assistant_index].get("role") == "tool":
        assistant_index -= 1
    if assistant_index < 0:
        return 0

    assistant = messages[assistant_index]
    if assistant.get("role") != "assistant":
        return 0

    content = assistant.get("content")
    if not isinstance(content, list):
        return 0

    tool_call_ids = {
        str(part.get("id"))
        for part in content
        if isinstance(part, dict)
        and part.get("type") == "tool_call"
        and part.get("id")
    }
    if not tool_call_ids:
        return 0

    responded_ids: set[str] = set()
    for message in messages[assistant_index + 1:]:
        tool_content = message.get("content")
        if not isinstance(tool_content, list):
            continue
        responded_ids.update(
            str(part.get("tool_call_id"))
            for part in tool_content
            if isinstance(part, dict)
            and part.get("type") == "tool_result"
            and part.get("tool_call_id")
        )

    if tool_call_ids <= responded_ids:
        return 0

    removed = len(messages) - assistant_index
    del messages[assistant_index:]
    return removed


def normalize_system_prompt(
    value: str | list[ContentPart],
) -> list[ContentPart]:
    if isinstance(value, str):
        return [{"type": "text", "text": value}]
    return deepcopy(value)


def event_summary(event: Event) -> dict[str, Any]:
    data = event.model_dump(mode="json", exclude_none=True)
    summary: dict[str, Any] = {
        "type": data.get("type"),
        "source": data.get("source"),
        "timestamp": data.get("timestamp"),
    }
    for key in (
        "run_id",
        "tool_call_id",
        "tool_name",
        "message_id",
        "queue_type",
        "stop_reason",
        "reason",
    ):
        if key in data:
            summary[key] = data[key]
    if "error" in data:
        summary["error"] = str(data["error"])
    if event.type == "agent.message_added" and "content" in data:
        summary["content_preview"] = content_preview(data["content"])
    return summary


def content_preview(content: Any, max_chars: int = 160) -> str:
    if not isinstance(content, list):
        text = str(content)
    else:
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        text = " ".join(parts)
    return text[: max_chars - 3] + "..." if len(text) > max_chars else text
