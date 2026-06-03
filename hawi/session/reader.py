"""Read-only helpers for browsing persisted Hawi sessions."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from . import layout
from .lock import probe_session_lock, read_lock_owner
from .manager import SessionManager
from .message_history import should_persist_message

logger = logging.getLogger(__name__)


class ReadOnlySessionBrowser:
    """Search and inspect persisted sessions without acquiring locks."""

    def __init__(self, root: Path | str | None = None) -> None:
        self._root = Path(root).expanduser() if root else layout.DEFAULT_ROOT

    def list_sessions(self) -> list[dict[str, Any]]:
        """Return visible session manifests sorted from newest to oldest."""
        if not self._root.exists():
            return []
        sessions: list[dict[str, Any]] = []
        for child in sorted(self._root.iterdir(), key=lambda path: path.name):
            if child.name.startswith(".") or not child.is_dir():
                continue
            manifest = self._read_manifest_from_dir(child)
            if not manifest or not self._session_dir_has_visible_messages(child):
                continue
            session_id = str(manifest.get("session_id") or child.name)
            lock_path = layout.session_lock_path(child)
            lock_info = probe_session_lock(lock_path, owner_token="readonly")
            sessions.append({
                "session_id": session_id,
                "name": str(manifest.get("name") or session_id),
                "created_at": str(manifest.get("created_at") or ""),
                "updated_at": str(manifest.get("updated_at") or ""),
                "last_checkpoint_event": manifest.get("last_checkpoint_event"),
                "components_present": list(manifest.get("components_present", [])),
                "locked": lock_info.locked,
                "lock_owner": read_lock_owner(lock_path) if lock_info.locked else None,
                "gui_launch_profile": (
                    manifest.get("gui_launch_profile")
                    if isinstance(manifest.get("gui_launch_profile"), dict)
                    else None
                ),
                "last_cwd": (
                    manifest.get("last_cwd")
                    if isinstance(manifest.get("last_cwd"), str)
                    else None
                ),
            })
        return sorted(
            sessions,
            key=lambda item: _parse_time(
                str(item.get("updated_at") or item.get("created_at") or "")
            ),
            reverse=True,
        )

    def read_message_history(self, session_id: str) -> list[dict[str, Any]]:
        """Read and annotate append-only message history for one session."""
        if not session_id:
            raise ValueError("session_id must be a non-empty string")
        session_dir = layout.session_dir(self._root, session_id)
        manifest_path = layout.manifest_path(session_dir)
        if not manifest_path.exists():
            raise FileNotFoundError(f"session not found: {session_id}")
        entries = layout.read_jsonl(layout.message_history_path(session_dir))
        context_snapshot = self._read_context_snapshot(session_id)
        messages = context_snapshot.get("messages")
        if isinstance(messages, list):
            return SessionManager._history_with_context_indices(entries, messages)
        return entries

    def search(
        self,
        query: str,
        *,
        limit: int = 100,
        session_id: str | None = None,
        case_sensitive: bool = False,
        whole_word: bool = False,
    ) -> dict[str, Any]:
        """Search visible message records from newest to oldest."""
        normalized_query = query.strip()
        if not normalized_query:
            return {
                "query": query,
                "case_sensitive": case_sensitive,
                "whole_word": whole_word,
                "results": [],
                "total_matches": 0,
                "truncated": False,
            }
        limit = max(1, min(int(limit), 500))

        metas = self.list_sessions()
        if session_id:
            metas = [item for item in metas if item.get("session_id") == session_id]

        matches: list[dict[str, Any]] = []
        total_matches = 0
        for meta_order, meta in enumerate(metas):
            sid = str(meta.get("session_id") or "")
            if not sid:
                continue
            try:
                history = self.read_message_history(sid)
            except (OSError, json.JSONDecodeError):
                logger.warning("failed to read history for session %s", sid, exc_info=True)
                continue
            for message_index in range(len(history) - 1, -1, -1):
                record = history[message_index]
                if _is_system_prompt_record(record):
                    continue
                text = message_record_text(record)
                if not text:
                    continue
                ranges = _match_ranges(
                    text,
                    normalized_query,
                    case_sensitive=case_sensitive,
                    whole_word=whole_word,
                )
                if not ranges:
                    continue
                total_matches += 1
                sort_timestamp = _record_sort_timestamp(record, meta)
                matches.append({
                    "session_id": sid,
                    "session_name": meta.get("name") or sid,
                    "session_created_at": meta.get("created_at") or "",
                    "session_updated_at": meta.get("updated_at") or "",
                    "last_cwd": meta.get("last_cwd"),
                    "message_index": message_index,
                    "context_message_id": record.get("context_message_id"),
                    "context_message_index": record.get("context_message_index"),
                    "run_id": record.get("run_id"),
                    "role": record.get("role") or "message",
                    "timestamp": record.get("timestamp"),
                    "sort_timestamp": sort_timestamp,
                    "text": text,
                    "snippet": _snippet(text, ranges[0]),
                    "match_ranges": [
                        {"start": start, "end": end} for start, end in ranges[:20]
                    ],
                    "_sort": (sort_timestamp, -meta_order, message_index),
                })

        matches.sort(key=lambda item: item["_sort"], reverse=True)
        limited_matches = matches[:limit]
        for item in limited_matches:
            item.pop("_sort", None)
        return {
            "query": query,
            "case_sensitive": case_sensitive,
            "whole_word": whole_word,
            "results": limited_matches,
            "total_matches": total_matches,
            "truncated": total_matches > len(limited_matches),
        }

    def _read_manifest_from_dir(self, session_dir: Path) -> dict[str, Any]:
        manifest_path = layout.manifest_path(session_dir)
        if not manifest_path.exists():
            return {}
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("skipping unreadable session manifest %s", manifest_path)
            return {}
        return data if isinstance(data, dict) else {}

    def _read_context_snapshot(self, session_id: str) -> dict[str, Any]:
        ctx_path = layout.context_path(layout.session_dir(self._root, session_id))
        if not ctx_path.exists():
            return {}
        try:
            data = json.loads(ctx_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("failed to read context snapshot %s", ctx_path)
            return {}
        return data if isinstance(data, dict) else {}

    def _session_dir_has_visible_messages(self, session_dir: Path) -> bool:
        history_path = layout.message_history_path(session_dir)
        try:
            if history_path.exists() and layout.read_jsonl(history_path):
                return True
        except (OSError, json.JSONDecodeError):
            logger.warning("could not inspect message history %s", history_path)

        context_snapshot = self._read_context_snapshot(session_dir.name)
        messages = context_snapshot.get("messages")
        return isinstance(messages, list) and any(
            _message_is_visible_entry(message) for message in messages
        )


def message_record_text(record: dict[str, Any]) -> str:
    """Return searchable plain text for one message-history record."""
    role = record.get("role")
    content = _content_text(record.get("content"))
    if content:
        return content
    if role:
        return str(role)
    return ""


def _is_system_prompt_record(record: dict[str, Any]) -> bool:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return False
    return metadata.get("event_type") == "agent.system_prompt"


def _message_is_visible_entry(message: Any) -> bool:
    if not isinstance(message, dict):
        return False
    if message.get("role") not in {"user", "assistant", "tool"}:
        return False
    content = message.get("content")
    return isinstance(content, list) and bool(content) and should_persist_message(
        message.get("metadata")
    )


def _content_text(value: Any) -> str:
    if isinstance(value, list):
        chunks = [_content_text(part) for part in value]
        return "\n\n".join(chunk for chunk in chunks if chunk)
    if isinstance(value, dict):
        part_type = value.get("type")
        if part_type == "text":
            return str(value.get("text") or "")
        if part_type == "reasoning":
            return str(value.get("reasoning") or value.get("text") or "")
        if part_type in {"steer", "tool_result"}:
            return _content_text(value.get("content"))
        if part_type == "tool_call":
            parts = [str(value.get("name") or "tool_call")]
            arguments = value.get("arguments", value.get("args"))
            if arguments not in (None, ""):
                parts.append(_json_text(arguments))
            return "\n".join(parts)
        return _json_text(value)
    if value is None:
        return ""
    return str(value)


def _json_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _match_ranges(
    text: str,
    needle: str,
    *,
    case_sensitive: bool = False,
    whole_word: bool = False,
) -> list[tuple[int, int]]:
    if not needle:
        return []
    haystack = text if case_sensitive else text.lower()
    target = needle if case_sensitive else needle.lower()
    ranges: list[tuple[int, int]] = []
    start = 0
    while True:
        index = haystack.find(target, start)
        if index < 0:
            return ranges
        end = index + len(target)
        if not whole_word or _has_non_english_letter_boundaries(text, index, end):
            ranges.append((index, end))
        start = index + max(1, len(target))


def _has_non_english_letter_boundaries(text: str, start: int, end: int) -> bool:
    left = text[start - 1] if start > 0 else ""
    right = text[end] if end < len(text) else ""
    return not _is_english_letter(left) and not _is_english_letter(right)


def _is_english_letter(char: str) -> bool:
    return len(char) == 1 and (("a" <= char <= "z") or ("A" <= char <= "Z"))


def _snippet(text: str, match_range: tuple[int, int], *, radius: int = 80) -> str:
    start, end = match_range
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    snippet = text[left:right].replace("\n", " ").strip()
    if left > 0:
        snippet = "..." + snippet
    if right < len(text):
        snippet += "..."
    return snippet


def _record_sort_timestamp(record: dict[str, Any], meta: dict[str, Any]) -> float:
    timestamp = _coerce_timestamp(record.get("timestamp"))
    if timestamp is not None:
        return timestamp
    for key in ("updated_at", "created_at"):
        value = meta.get(key)
        if isinstance(value, str):
            parsed = _parse_time(value)
            if parsed:
                return parsed
    return 0.0


def _coerce_timestamp(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        parsed = _parse_time(value)
        return parsed or None
    return None


def _parse_time(value: str) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0
