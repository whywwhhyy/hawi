"""Streaming JSON helpers for tool-call argument rendering."""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any


PLACEHOLDER = "…"
_KEY_RE = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"\s*:')
_DANGLING_KEY_RE = re.compile(r'(?:,?\s*"[^"\\]*(?:\\.[^"\\]*)*"\s*:\s*)$')


@dataclass
class StreamingJsonState:
    """State for one tool-call arguments stream."""

    raw_buffer: str = ""
    final_arguments: dict[str, Any] | None = None

    def feed(self, delta: str) -> Any:
        self.raw_buffer += delta
        return self.snapshot_tree()

    def finalize(self, arguments: dict[str, Any]) -> Any:
        self.final_arguments = arguments
        self.raw_buffer = json.dumps(arguments, ensure_ascii=False)
        return arguments

    def snapshot_tree(self) -> Any:
        if self.final_arguments is not None:
            return self.final_arguments
        return best_effort_json_tree(self.raw_buffer)


def best_effort_json_tree(raw: str) -> Any:
    text = raw.strip()
    if not text:
        return {}

    parsed = _try_json_loads(text)
    if parsed is not None:
        return parsed

    repaired = _repair_partial_json(text)
    parsed = _try_json_loads(repaired)
    if parsed is not None:
        return parsed

    keys = _extract_keys(text)
    if keys:
        return {k: PLACEHOLDER for k in keys}

    if text.startswith("["):
        return [PLACEHOLDER]
    return PLACEHOLDER


def render_json_tree_html(tree: Any) -> str:
    return _render_any(tree, root=True)


def _try_json_loads(text: str) -> Any | None:
    try:
        return json.loads(text)
    except JSONDecodeError:
        return None


def _extract_keys(text: str) -> list[str]:
    keys: list[str] = []
    for m in _KEY_RE.finditer(text):
        key = _safe_unescape(m.group(1))
        if key not in keys:
            keys.append(key)
    return keys


def _safe_unescape(text: str) -> str:
    try:
        return json.loads(f'"{text}"')
    except Exception:
        return text


def _repair_partial_json(text: str) -> str:
    repaired = text

    if repaired.endswith("\\"):
        repaired = repaired[:-1]

    repaired = repaired.rstrip()

    in_string, stack = _scan_json_structure(repaired)
    if in_string:
        repaired += '"'

    # Drop a dangling key/value prefix before closing braces are appended.
    repaired = _DANGLING_KEY_RE.sub("", repaired)

    closing = "".join("}" if ch == "{" else "]" for ch in reversed(stack))
    repaired = repaired + closing

    # Remove trailing commas before braces/brackets.
    prev = None
    while prev != repaired:
        prev = repaired
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

    return repaired


def _scan_json_structure(text: str) -> tuple[bool, list[str]]:
    stack: list[str] = []
    in_string = False
    escaped = False

    for ch in text:
        if in_string:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{" or ch == "[":
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()

    return in_string, stack


def _render_any(value: Any, *, root: bool = False) -> str:
    if isinstance(value, dict):
        if not value:
            return '<ul class="tool-args"><li><strong>(empty)</strong></li></ul>' if root else "<span>(empty)</span>"
        rows: list[str] = []
        for key, item in value.items():
            label = f"<strong>{html.escape(str(key))}</strong>"
            rendered = _render_child(item)
            rows.append(f"<li>{label}: {rendered}</li>")
        cls = "tool-args"
        return f'<ul class="{cls}">{"".join(rows)}</ul>'

    if isinstance(value, list):
        if not value:
            return '<ul class="tool-args"><li><strong>(empty)</strong></li></ul>' if root else "<span>(empty)</span>"
        rows = []
        for idx, item in enumerate(value):
            label = f"<strong>[{idx}]</strong>"
            rows.append(f"<li>{label}: {_render_child(item)}</li>")
        return f'<ul class="tool-args">{"".join(rows)}</ul>'

    scalar = _render_scalar(value)
    if root:
        return f'<ul class="tool-args"><li><strong>value</strong>: {scalar}</li></ul>'
    return scalar


def _render_child(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return _render_any(value, root=False)
    return _render_scalar(value)


def _render_scalar(value: Any) -> str:
    if value is PLACEHOLDER or value == PLACEHOLDER:
        return f'<span class="placeholder">{PLACEHOLDER}</span>'
    if isinstance(value, str):
        return html.escape(value)
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return html.escape(str(value))


__all__ = [
    "PLACEHOLDER",
    "StreamingJsonState",
    "best_effort_json_tree",
    "render_json_tree_html",
]
