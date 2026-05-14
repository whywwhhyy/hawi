"""Markdown export for Hawi message history."""

from __future__ import annotations

import json
import re
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from . import layout

EXPORT_VERSION = 1
DEFAULT_TOOL_LINE_LIMIT = 100

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class MarkdownReference:
    """A folded export payload written beside the Markdown file."""

    filename: str
    content: str
    mime_type: str = "text/plain"

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "content": self.content,
            "mime_type": self.mime_type,
        }


@dataclass(frozen=True)
class MarkdownExport:
    """Rendered Markdown plus internal/export metadata."""

    export_id: str
    kind: Literal["session", "subagent"]
    subject_id: str
    suggested_filename: str
    reference_dir_name: str
    markdown: str
    references: list[MarkdownReference] = field(default_factory=list)
    session_export_dir: str | None = None
    session_markdown_path: str | None = None
    session_jsonl_path: str | None = None
    manifest_path: str | None = None

    def to_dict(self, *, include_markdown: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "export_id": self.export_id,
            "kind": self.kind,
            "subject_id": self.subject_id,
            "suggested_filename": self.suggested_filename,
            "reference_dir_name": self.reference_dir_name,
            "references": [ref.to_dict() for ref in self.references],
            "session_export_dir": self.session_export_dir,
            "session_markdown_path": self.session_markdown_path,
            "session_jsonl_path": self.session_jsonl_path,
            "manifest_path": self.manifest_path,
        }
        if include_markdown:
            payload["markdown"] = self.markdown
        return payload


def new_export_id() -> str:
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def export_message_history_to_markdown(
    message_history: list[dict[str, Any]],
    *,
    kind: Literal["session", "subagent"],
    subject_id: str,
    title: str,
    export_id: str | None = None,
    model: str | None = None,
    system_prompt: Any = None,
    metadata: dict[str, Any] | None = None,
    raw_history_path: str | None = None,
    tool_line_limit: int = DEFAULT_TOOL_LINE_LIMIT,
) -> MarkdownExport:
    """Render visible ``message_history`` records to a readable Markdown file."""
    export_id = export_id or new_export_id()
    base_name = sanitize_filename(f"hawi-{kind}-{subject_id}-{export_id}") or "hawi-export"
    suggested_filename = f"{base_name}.md"
    reference_dir_name = f"{base_name}-ref"
    refs: list[MarkdownReference] = []
    ref_counter = _ReferenceCounter()

    lines: list[str] = [
        f"# {title}",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Export kind | `{kind}` |",
        f"| ID | `{subject_id}` |",
        f"| Export ID | `{export_id}` |",
        f"| Exported at | `{datetime.now().isoformat(timespec='seconds')}` |",
    ]
    if model:
        lines.append(f"| Model | `{_escape_table(model)}` |")
    if raw_history_path:
        lines.append(f"| Raw history | `{_escape_table(raw_history_path)}` |")
    for key, value in sorted((metadata or {}).items()):
        if value is None:
            continue
        lines.append(f"| {_escape_table(str(key))} | {_format_table_value(value)} |")

    if system_prompt:
        lines.extend(["", "## System Prompt", ""])
        lines.append(_render_regular_content(_coerce_content_list(system_prompt)) or "_Empty._")

    for index, record in enumerate(message_history, start=1):
        role = str(record.get("role") or "message")
        timestamp = str(record.get("timestamp") or "")
        run_id = str(record.get("run_id") or "")
        content_raw = record.get("content")
        content: list[Any] = content_raw if isinstance(content_raw, list) else []
        metadata_raw = record.get("metadata")
        metadata_obj: dict[str, Any] = metadata_raw if isinstance(metadata_raw, dict) else {}

        lines.extend(["", "---", ""])
        heading = f"## {index:03d} · {_role_label(role)}"
        if timestamp:
            heading += f" · {timestamp}"
        lines.append(heading)
        details = _message_details(run_id, metadata_obj)
        if details:
            lines.extend(["", details])
        lines.extend(_render_message_content(
            role=role,
            content=content,
            refs=refs,
            ref_counter=ref_counter,
            reference_dir_name=reference_dir_name,
            line_limit=tool_line_limit,
        ))

    if not message_history:
        lines.extend(["", "---", "", "_No visible messages in this export._"])

    markdown = "\n".join(lines).rstrip() + "\n"
    return MarkdownExport(
        export_id=export_id,
        kind=kind,
        subject_id=subject_id,
        suggested_filename=suggested_filename,
        reference_dir_name=reference_dir_name,
        markdown=markdown,
        references=refs,
    )


def write_markdown_export_bundle(
    export: MarkdownExport,
    *,
    export_dir: Path,
    source_jsonl_path: Path | None = None,
    message_history: list[dict[str, Any]] | None = None,
) -> MarkdownExport:
    """Write a complete session-internal export bundle."""
    export_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = export_dir / export.suggested_filename
    ref_dir = export_dir / export.reference_dir_name
    layout.atomic_write_text(markdown_path, export.markdown, fsync=True)
    if export.references:
        ref_dir.mkdir(parents=True, exist_ok=True)
        for ref in export.references:
            layout.atomic_write_text(ref_dir / ref.filename, ref.content, fsync=True)

    jsonl_path = export_dir / layout.MESSAGE_HISTORY_FILENAME
    if source_jsonl_path is not None and source_jsonl_path.exists():
        shutil.copy2(source_jsonl_path, jsonl_path)
    elif message_history is not None:
        layout.atomic_write_text(
            jsonl_path,
            "".join(
                json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n"
                for entry in message_history
            ),
            fsync=True,
        )

    manifest_path = export_dir / "manifest.json"
    manifest = {
        "version": EXPORT_VERSION,
        "export_id": export.export_id,
        "kind": export.kind,
        "subject_id": export.subject_id,
        "suggested_filename": export.suggested_filename,
        "reference_dir_name": export.reference_dir_name,
        "markdown_path": str(markdown_path),
        "message_history_path": str(jsonl_path) if jsonl_path.exists() else None,
        "references": [
            {
                "filename": ref.filename,
                "mime_type": ref.mime_type,
                "path": str(ref_dir / ref.filename),
            }
            for ref in export.references
        ],
    }
    layout.atomic_write_text(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        fsync=True,
    )
    return MarkdownExport(
        export_id=export.export_id,
        kind=export.kind,
        subject_id=export.subject_id,
        suggested_filename=export.suggested_filename,
        reference_dir_name=export.reference_dir_name,
        markdown=export.markdown,
        references=export.references,
        session_export_dir=str(export_dir),
        session_markdown_path=str(markdown_path),
        session_jsonl_path=str(jsonl_path) if jsonl_path.exists() else None,
        manifest_path=str(manifest_path),
    )


def sanitize_filename(value: str) -> str:
    cleaned = _SAFE_FILENAME_RE.sub("-", value.strip()).strip(".-")
    return cleaned[:160]


class _ReferenceCounter:
    def __init__(self) -> None:
        self.tool_call = 0
        self.tool_result = 0

    def next_tool_call(self, suffix: str) -> str:
        self.tool_call += 1
        return f"tool_call_{self.tool_call:04d}_arguments.{suffix}"

    def next_tool_result(self, suffix: str) -> str:
        self.tool_result += 1
        return f"tool_result_{self.tool_result:04d}.{suffix}"


def _render_message_content(
    *,
    role: str,
    content: list[Any],
    refs: list[MarkdownReference],
    ref_counter: _ReferenceCounter,
    reference_dir_name: str,
    line_limit: int,
) -> list[str]:
    rendered: list[str] = [""]
    if role == "assistant":
        regular_parts: list[Any] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "reasoning":
                reasoning = str(part.get("reasoning") or part.get("text") or "")
                if reasoning:
                    rendered.extend(["", "> **Thinking**"])
                    rendered.extend(_quote_lines(reasoning))
            elif isinstance(part, dict) and part.get("type") == "tool_call":
                regular = _render_regular_content(regular_parts)
                if regular:
                    rendered.extend(["", regular])
                regular_parts = []
                rendered.extend(_render_tool_call(
                    part,
                    refs=refs,
                    ref_counter=ref_counter,
                    reference_dir_name=reference_dir_name,
                    line_limit=line_limit,
                ))
            else:
                regular_parts.append(part)
        regular = _render_regular_content(regular_parts)
        if regular:
            rendered.extend(["", regular])
    elif role == "tool":
        for part in content:
            if isinstance(part, dict) and part.get("type") == "tool_result":
                rendered.extend(_render_tool_result(
                    part,
                    refs=refs,
                    ref_counter=ref_counter,
                    reference_dir_name=reference_dir_name,
                    line_limit=line_limit,
                ))
            else:
                rendered.extend(["", _fenced(_json_dumps(part), "json")])
    else:
        rendered.append(_render_regular_content(content) or "_Empty message._")
    return rendered


def _render_tool_call(
    part: dict[str, Any],
    *,
    refs: list[MarkdownReference],
    ref_counter: _ReferenceCounter,
    reference_dir_name: str,
    line_limit: int,
) -> list[str]:
    name = str(part.get("name") or "tool")
    tool_call_id = str(part.get("id") or "")
    args = part.get("arguments", part.get("args", {}))
    args_text = _json_dumps(args)
    body, ref = _maybe_fold(
        args_text,
        filename=ref_counter.next_tool_call("json"),
        mime_type="application/json",
        refs=refs,
        line_limit=line_limit,
    )
    lines = ["", f"### Tool Call · `{name}`"]
    if tool_call_id:
        lines.extend(["", f"Tool call ID: `{tool_call_id}`"])
    if ref:
        lines.extend([
            "",
            f"> Full arguments: [{ref.filename}]({reference_dir_name}/{ref.filename})",
        ])
    lines.extend(["", _fenced(body, "json")])
    return lines


def _render_tool_result(
    part: dict[str, Any],
    *,
    refs: list[MarkdownReference],
    ref_counter: _ReferenceCounter,
    reference_dir_name: str,
    line_limit: int,
) -> list[str]:
    tool_call_id = str(part.get("tool_call_id") or "tool_result")
    is_error = part.get("is_error") is True
    result_text = _content_text(part.get("content"))
    body, ref = _maybe_fold(
        result_text,
        filename=ref_counter.next_tool_result("txt"),
        mime_type="text/plain",
        refs=refs,
        line_limit=line_limit,
    )
    lines = ["", f"### Tool Result · `{tool_call_id}`"]
    lines.extend(["", f"Status: `{'error' if is_error else 'ok'}`"])
    if ref:
        lines.extend([
            "",
            f"> Full result: [{ref.filename}]({reference_dir_name}/{ref.filename})",
        ])
    lines.extend(["", _fenced(body, "text")])
    return lines


def _maybe_fold(
    text: str,
    *,
    filename: str,
    mime_type: str,
    refs: list[MarkdownReference],
    line_limit: int,
) -> tuple[str, MarkdownReference | None]:
    lines = text.splitlines()
    if len(lines) <= line_limit:
        return text, None
    ref = MarkdownReference(filename=filename, content=text, mime_type=mime_type)
    refs.append(ref)
    return "\n".join(lines[:line_limit] + ["..."]), ref


def _render_regular_content(content: list[Any]) -> str:
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            parts.append(str(part))
            continue
        part_type = part.get("type")
        if part_type == "text":
            parts.append(str(part.get("text") or ""))
        elif part_type == "steer" and isinstance(part.get("content"), list):
            parts.append(_render_regular_content(part["content"]))
        elif part_type == "reasoning":
            parts.append(str(part.get("reasoning") or part.get("text") or ""))
        elif part_type == "tool_result":
            parts.append(_content_text(part.get("content")))
        elif part_type == "tool_call":
            continue
        else:
            parts.append(_fenced(_json_dumps(part), "json"))
    return "\n\n".join(part for part in parts if part)


def _content_text(value: Any) -> str:
    if isinstance(value, list):
        return _render_regular_content(value)
    if isinstance(value, (dict, list)):
        return _json_dumps(value)
    if value is None:
        return ""
    return str(value)


def _coerce_content_list(value: Any) -> list[Any]:
    if isinstance(value, str):
        return [{"type": "text", "text": value}]
    if isinstance(value, list):
        return value
    return [{"type": "text", "text": str(value)}]


def _message_details(run_id: str, metadata: dict[str, Any]) -> str:
    details: list[str] = []
    if run_id:
        details.append(f"Run: `{run_id}`")
    queue = metadata.get("queue")
    if isinstance(queue, str):
        details.append(f"Queue: `{queue}`")
    message_id = metadata.get("message_id")
    if isinstance(message_id, str):
        details.append(f"Message: `{message_id}`")
    return " · ".join(details)


def _quote_lines(text: str) -> list[str]:
    return ["> " + line if line else ">" for line in text.splitlines()]


def _role_label(role: str) -> str:
    return {
        "user": "User",
        "assistant": "Assistant",
        "tool": "Tool",
        "event": "Event",
    }.get(role, role.title())


def _format_table_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return "`" + _escape_table(json.dumps(value, ensure_ascii=False)) + "`"
    return "`" + _escape_table(str(value)) + "`"


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _fenced(value: str, language: str) -> str:
    fence = "```"
    while fence in value:
        fence += "`"
    return f"{fence}{language}\n{value}\n{fence}"


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
