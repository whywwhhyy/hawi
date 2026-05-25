from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from hawi.engine.protocol import VERSION
from hawi.engine.readonly import ReadOnlyRuntime
from hawi.session import layout
from hawi.session.reader import ReadOnlySessionBrowser


@dataclass(eq=False)
class FakeClient:
    id: str = "client"
    authenticated: bool = True
    negotiated_caps: set[str] = field(default_factory=set)
    sent: list[dict[str, Any]] = field(default_factory=list)
    closed: bool = False

    async def send(self, frame: dict[str, Any]) -> None:
        self.sent.append(frame)

    async def close(self) -> None:
        self.closed = True


def write_session(
    root: Path,
    session_id: str,
    *,
    name: str,
    created_at: str,
    updated_at: str,
    history: list[dict[str, Any]],
    context_messages: list[dict[str, Any]] | None = None,
) -> None:
    session_dir = layout.session_dir(root, session_id)
    session_dir.mkdir(parents=True)
    layout.atomic_write_text(
        layout.manifest_path(session_dir),
        json.dumps(
            {
                "version": 1,
                "session_id": session_id,
                "name": name,
                "created_at": created_at,
                "updated_at": updated_at,
                "components_present": ["message_history", "context"],
            }
        ),
        fsync=False,
    )
    layout.write_jsonl(layout.message_history_path(session_dir), history, fsync=False)
    layout.atomic_write_text(
        layout.context_path(session_dir),
        json.dumps(
            {
                "version": "1.0",
                "messages": context_messages if context_messages is not None else [
                    {
                        "role": item["role"],
                        "content": item["content"],
                        "metadata": item.get("metadata"),
                        "context_message_id": f"ctx-{session_id}-{index}",
                    }
                    for index, item in enumerate(history)
                    if item.get("role") in {"user", "assistant", "tool"}
                ],
            }
        ),
        fsync=False,
    )


def text_record(text: str, *, role: str = "user", timestamp: float = 0.0) -> dict[str, Any]:
    return {
        "version": 1,
        "timestamp": timestamp,
        "run_id": f"run-{timestamp}",
        "role": role,
        "content": [{"type": "text", "text": text}],
        "metadata": None,
    }


def test_readonly_browser_searches_messages_from_newest_to_oldest(tmp_path: Path) -> None:
    old_history = [text_record("alpha old match", timestamp=10)]
    new_history = [
        text_record("ordinary message", timestamp=20),
        text_record("alpha new match", role="assistant", timestamp=30),
    ]
    write_session(
        tmp_path,
        "old",
        name="Old",
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:01:00",
        history=old_history,
    )
    write_session(
        tmp_path,
        "new",
        name="New",
        created_at="2026-01-02T00:00:00",
        updated_at="2026-01-02T00:01:00",
        history=new_history,
    )

    result = ReadOnlySessionBrowser(tmp_path).search("alpha", limit=10)

    assert result["total_matches"] == 2
    assert [item["session_id"] for item in result["results"]] == ["new", "old"]
    assert result["results"][0]["message_index"] == 1
    assert result["results"][0]["context_message_id"] == "ctx-new-1"


def test_readonly_browser_search_respects_case_sensitive(tmp_path: Path) -> None:
    write_session(
        tmp_path,
        "case-session",
        name="Case",
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:01:00",
        history=[
            text_record("Alpha token", timestamp=10),
            text_record("alpha token", timestamp=20),
        ],
    )
    browser = ReadOnlySessionBrowser(tmp_path)

    default_result = browser.search("alpha", limit=10)
    sensitive_result = browser.search("alpha", limit=10, case_sensitive=True)

    assert default_result["total_matches"] == 2
    assert sensitive_result["case_sensitive"] is True
    assert sensitive_result["total_matches"] == 1
    assert sensitive_result["results"][0]["text"] == "alpha token"


def test_readonly_browser_search_matches_whole_words_by_english_letters(
    tmp_path: Path,
) -> None:
    write_session(
        tmp_path,
        "word-session",
        name="Words",
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:01:00",
        history=[
            text_record("scatter", timestamp=10),
            text_record("catA", timestamp=20),
            text_record("Acat", timestamp=30),
            text_record("cat!", timestamp=40),
            text_record("cat中文", timestamp=50),
        ],
    )

    result = ReadOnlySessionBrowser(tmp_path).search("cat", limit=10, whole_word=True)

    assert result["whole_word"] is True
    assert result["total_matches"] == 2
    assert [item["text"] for item in result["results"]] == ["cat中文", "cat!"]


@pytest.mark.asyncio
async def test_readonly_runtime_returns_search_results(tmp_path: Path) -> None:
    write_session(
        tmp_path,
        "session-a",
        name="A",
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:01:00",
        history=[text_record("find this keyword", timestamp=123)],
    )
    runtime = ReadOnlyRuntime(session_root=tmp_path)
    client = FakeClient()

    await runtime.handle_frame(
        client,
        json.dumps(
            {
                "version": VERSION,
                "type": "session_search",
                "id": "search-1",
                "payload": {
                    "query": "keyword",
                    "limit": 5,
                    "case_sensitive": True,
                    "whole_word": True,
                },
            }
        ),
    )

    payload = client.sent[-1]["payload"]
    assert client.sent[-1]["type"] == "ack"
    assert payload["command"] == "session_search"
    assert payload["read_only"] is True
    assert payload["case_sensitive"] is True
    assert payload["whole_word"] is True
    assert payload["results"][0]["session_id"] == "session-a"


@pytest.mark.asyncio
async def test_readonly_runtime_requires_session_id_for_history(tmp_path: Path) -> None:
    runtime = ReadOnlyRuntime(session_root=tmp_path)
    client = FakeClient()

    await runtime.handle_frame(
        client,
        json.dumps(
            {
                "version": VERSION,
                "type": "session_history",
                "id": "history-1",
                "payload": {},
            }
        ),
    )

    assert client.sent[-1]["type"] == "error"
    assert client.sent[-1]["payload"]["code"] == "command_failed"
