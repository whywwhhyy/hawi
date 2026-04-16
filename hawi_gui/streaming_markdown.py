"""Streaming Markdown parser that emits operation stream and HTML renderer."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Literal


InlineStyle = Literal["bold", "italic", "code"]
ListKind = Literal["ul", "ol"]


@dataclass(frozen=True)
class TextAppendOp:
    text: str
    styles: tuple[InlineStyle, ...] = ()
    href: str | None = None


@dataclass(frozen=True)
class ParagraphStartOp:
    pass


@dataclass(frozen=True)
class ParagraphEndOp:
    pass


@dataclass(frozen=True)
class HeadingStartOp:
    level: int


@dataclass(frozen=True)
class HeadingEndOp:
    level: int


@dataclass(frozen=True)
class BlockquoteDepthOp:
    depth: int


@dataclass(frozen=True)
class ListStartOp:
    depth: int
    kind: ListKind
    ordered_start: int | None = None


@dataclass(frozen=True)
class ListEndOp:
    depth: int


@dataclass(frozen=True)
class ListItemStartOp:
    depth: int
    ordered_index: int | None = None


@dataclass(frozen=True)
class ListItemEndOp:
    depth: int


@dataclass(frozen=True)
class CodeBlockStartOp:
    language: str


@dataclass(frozen=True)
class CodeTextAppendOp:
    text: str


@dataclass(frozen=True)
class CodeBlockEndOp:
    pass


@dataclass(frozen=True)
class TableStartOp:
    pass


@dataclass(frozen=True)
class TableRowStartOp:
    row_index: int
    is_header: bool


@dataclass(frozen=True)
class TableCellUpdateOp:
    column_index: int
    text: str
    styles: tuple[InlineStyle, ...] = ()
    href: str | None = None


@dataclass(frozen=True)
class TableRowEndOp:
    pass


@dataclass(frozen=True)
class TableEndOp:
    pass


@dataclass(frozen=True)
class ThematicBreakOp:
    pass


@dataclass(frozen=True)
class LineBreakOp:
    pass


MarkdownOp = (
    TextAppendOp
    | ParagraphStartOp
    | ParagraphEndOp
    | HeadingStartOp
    | HeadingEndOp
    | BlockquoteDepthOp
    | ListStartOp
    | ListEndOp
    | ListItemStartOp
    | ListItemEndOp
    | CodeBlockStartOp
    | CodeTextAppendOp
    | CodeBlockEndOp
    | TableStartOp
    | TableRowStartOp
    | TableCellUpdateOp
    | TableRowEndOp
    | TableEndOp
    | ThematicBreakOp
    | LineBreakOp
)


_LIST_RE = re.compile(r"^(?P<indent>\s*)(?:(?P<unordered>[-+*])|(?P<ordered>\d+)\.)\s+(?P<text>.*)$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE_RE = re.compile(r"^\s*(?P<marker>`{3,}|~{3,})\s*(?P<lang>\S*)\s*$")
_THEMATIC_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
_TABLE_SEPARATOR_RE = re.compile(r"^\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?$")


@dataclass
class _ListState:
    depth: int
    kind: ListKind


class StreamingMarkdownOpParser:
    """Parse markdown delta stream into structural operations."""

    def __init__(self) -> None:
        self._line_buf = ""
        self._preview_len = 0

        self._in_paragraph = False
        self._blockquote_depth = 0

        self._in_code = False
        self._fence_marker = ""

        self._lists: list[_ListState] = []
        self._open_list_items: list[int] = []

        self._in_table = False
        self._table_row_index = 0
        self._table_header_written = False
        self._table_separator_seen = False

    def feed(self, delta: str) -> list[MarkdownOp]:
        if not delta:
            return []

        ops: list[MarkdownOp] = []
        for ch in delta:
            self._line_buf += ch
            if ch == "\n":
                line = self._line_buf[:-1]
                preview_len = self._preview_len
                self._line_buf = ""
                self._preview_len = 0
                ops.extend(self._process_line(line, preview_len, terminated=True))
                continue

            ops.extend(self._emit_preview())

        return ops

    def flush(self) -> list[MarkdownOp]:
        ops: list[MarkdownOp] = []

        if self._line_buf:
            ops.extend(self._process_line(self._line_buf, self._preview_len, terminated=False))
            self._line_buf = ""
            self._preview_len = 0

        if self._in_code:
            ops.append(CodeBlockEndOp())
            self._in_code = False
            self._fence_marker = ""

        ops.extend(self._close_table())
        ops.extend(self._close_paragraph())
        ops.extend(self._close_lists())
        ops.extend(self._set_blockquote_depth(0))
        return ops

    def reset(self) -> None:
        self.__init__()

    def _emit_preview(self) -> list[MarkdownOp]:
        if self._in_code:
            return []
        if not self._line_buf or self._preview_len >= len(self._line_buf):
            return []
        if not self._can_preview_paragraph(self._line_buf):
            return []

        ops: list[MarkdownOp] = []
        if not self._in_paragraph:
            ops.append(ParagraphStartOp())
            self._in_paragraph = True

        fragment = self._line_buf[self._preview_len :]
        ops.extend(self._inline_to_ops(fragment))
        self._preview_len = len(self._line_buf)
        return ops

    def _can_preview_paragraph(self, line: str) -> bool:
        stripped = line.lstrip()
        if not stripped:
            return False
        if self._in_table or self._lists:
            return False
        if any(ch in line for ch in ("`", "*", "_", "[", "]")):
            return False
        if stripped.startswith(("#", ">", "-", "+", "|", "`", "~")):
            return False
        if re.match(r"^\\d+\\.", stripped):
            return False
        if _FENCE_RE.match(stripped):
            return False
        if _HEADING_RE.match(stripped):
            return False
        if _THEMATIC_RE.match(stripped):
            return False
        if stripped.startswith(">"):
            return False
        if stripped.startswith("|"):
            return False
        if _LIST_RE.match(stripped):
            return False
        return True

    def _process_line(self, line: str, preview_len: int, *, terminated: bool) -> list[MarkdownOp]:
        ops: list[MarkdownOp] = []

        if self._in_code:
            stripped = line.strip()
            if stripped.startswith(self._fence_marker):
                ops.append(CodeBlockEndOp())
                self._in_code = False
                self._fence_marker = ""
            else:
                payload = line
                if terminated:
                    payload += "\n"
                if payload:
                    ops.append(CodeTextAppendOp(payload))
            return ops

        depth, content = self._split_blockquote(line)
        ops.extend(self._set_blockquote_depth(depth))

        if content.strip() == "":
            ops.extend(self._close_table())
            ops.extend(self._close_paragraph())
            if not self._in_table:
                ops.extend(self._close_lists())
            return ops

        stripped = content.lstrip()
        fence_match = _FENCE_RE.match(stripped)
        if fence_match:
            ops.extend(self._close_table())
            ops.extend(self._close_paragraph())
            ops.extend(self._close_lists())
            marker = fence_match.group("marker")
            lang = fence_match.group("lang") or ""
            self._in_code = True
            self._fence_marker = marker[0] * 3
            ops.append(CodeBlockStartOp(language=lang))
            return ops

        if self._is_table_line(content):
            ops.extend(self._close_paragraph())
            ops.extend(self._close_lists())
            if not self._in_table:
                self._in_table = True
                self._table_row_index = 0
                self._table_header_written = False
                self._table_separator_seen = False
                ops.append(TableStartOp())

            if _TABLE_SEPARATOR_RE.match(content.strip()):
                self._table_separator_seen = True
                return ops

            is_header = not self._table_separator_seen and not self._table_header_written
            ops.append(TableRowStartOp(row_index=self._table_row_index, is_header=is_header))
            for col, cell in enumerate(self._split_table_cells(content)):
                cell_text = cell.strip()
                span_ops = self._inline_to_ops(cell_text)
                if not span_ops:
                    ops.append(TableCellUpdateOp(column_index=col, text=""))
                    continue
                for span in span_ops:
                    assert isinstance(span, TextAppendOp)
                    ops.append(
                        TableCellUpdateOp(
                            column_index=col,
                            text=span.text,
                            styles=span.styles,
                            href=span.href,
                        )
                    )
            ops.append(TableRowEndOp())

            if is_header:
                self._table_header_written = True
            self._table_row_index += 1
            return ops

        ops.extend(self._close_table())

        list_match = _LIST_RE.match(content)
        if list_match:
            ops.extend(self._close_paragraph())
            indent = len(list_match.group("indent").expandtabs(2))
            depth_index = indent // 2 + 1
            ordered_raw = list_match.group("ordered")
            kind: ListKind = "ol" if ordered_raw else "ul"
            ordered_start = int(ordered_raw) if ordered_raw else None
            ops.extend(self._close_open_list_items(depth_index + 1))
            ops.extend(self._sync_list_stack(depth_index, kind, ordered_start))
            ops.extend(self._close_open_list_items(depth_index))
            text = list_match.group("text")
            ops.append(ListItemStartOp(depth=depth_index, ordered_index=ordered_start))
            ops.extend(self._inline_to_ops(text))
            self._open_list_items.append(depth_index)
            return ops

        ops.extend(self._close_lists())

        heading_match = _HEADING_RE.match(content)
        if heading_match:
            ops.extend(self._close_paragraph())
            hashes, heading_text = heading_match.groups()
            level = len(hashes)
            ops.append(HeadingStartOp(level=level))
            ops.extend(self._inline_to_ops(heading_text))
            ops.append(HeadingEndOp(level=level))
            return ops

        if _THEMATIC_RE.match(content):
            ops.extend(self._close_paragraph())
            ops.append(ThematicBreakOp())
            return ops

        if not self._in_paragraph:
            self._in_paragraph = True
            ops.append(ParagraphStartOp())

        suffix = content[preview_len:]
        if suffix:
            ops.extend(self._inline_to_ops(suffix))
        if terminated:
            ops.append(LineBreakOp())
        return ops

    def _split_blockquote(self, line: str) -> tuple[int, str]:
        idx = 0
        depth = 0
        n = len(line)
        while idx < n:
            original = idx
            while idx < n and line[idx] == " ":
                idx += 1
            if idx < n and line[idx] == ">":
                depth += 1
                idx += 1
                if idx < n and line[idx] == " ":
                    idx += 1
            else:
                idx = original
                break
        return depth, line[idx:]

    def _set_blockquote_depth(self, depth: int) -> list[MarkdownOp]:
        if depth == self._blockquote_depth:
            return []
        self._blockquote_depth = depth
        return [BlockquoteDepthOp(depth=depth)]

    def _close_paragraph(self) -> list[MarkdownOp]:
        if not self._in_paragraph:
            return []
        self._in_paragraph = False
        return [ParagraphEndOp()]

    def _close_lists(self) -> list[MarkdownOp]:
        ops: list[MarkdownOp] = []
        ops.extend(self._close_open_list_items(1))
        while self._lists:
            state = self._lists.pop()
            ops.append(ListEndOp(depth=state.depth))
        return ops

    def _sync_list_stack(self, target_depth: int, kind: ListKind, ordered_start: int | None) -> list[MarkdownOp]:
        ops: list[MarkdownOp] = []

        while len(self._lists) > target_depth:
            depth = self._lists[-1].depth
            ops.extend(self._close_open_list_items(depth))
            state = self._lists.pop()
            ops.append(ListEndOp(depth=state.depth))

        if self._lists and self._lists[-1].depth == target_depth:
            if self._lists[-1].kind != kind:
                ops.extend(self._close_open_list_items(target_depth))
                state = self._lists.pop()
                ops.append(ListEndOp(depth=state.depth))

        while len(self._lists) < target_depth:
            depth = len(self._lists) + 1
            start = ordered_start if kind == "ol" and depth == target_depth else None
            self._lists.append(_ListState(depth=depth, kind=kind))
            ops.append(ListStartOp(depth=depth, kind=kind, ordered_start=start))

        if self._lists and self._lists[-1].depth == target_depth and self._lists[-1].kind != kind:
            ops.extend(self._close_open_list_items(target_depth))
            state = self._lists.pop()
            ops.append(ListEndOp(depth=state.depth))
            self._lists.append(_ListState(depth=target_depth, kind=kind))
            ops.append(ListStartOp(depth=target_depth, kind=kind, ordered_start=ordered_start if kind == "ol" else None))

        return ops

    def _close_open_list_items(self, min_depth: int) -> list[MarkdownOp]:
        ops: list[MarkdownOp] = []
        while self._open_list_items and self._open_list_items[-1] >= min_depth:
            depth = self._open_list_items.pop()
            ops.append(ListItemEndOp(depth=depth))
        return ops

    def _close_table(self) -> list[MarkdownOp]:
        if not self._in_table:
            return []
        self._in_table = False
        self._table_row_index = 0
        self._table_header_written = False
        self._table_separator_seen = False
        return [TableEndOp()]

    def _is_table_line(self, content: str) -> bool:
        stripped = content.strip()
        return stripped.startswith("|") and "|" in stripped[1:]

    def _split_table_cells(self, line: str) -> list[str]:
        stripped = line.strip()
        if stripped.startswith("|"):
            stripped = stripped[1:]
        if stripped.endswith("|"):
            stripped = stripped[:-1]
        return stripped.split("|")

    def _inline_to_ops(self, text: str) -> list[TextAppendOp]:
        if not text:
            return []

        ops: list[TextAppendOp] = []
        i = 0
        bold = False
        italic = False
        code = False

        def push(fragment: str, href: str | None = None) -> None:
            if not fragment:
                return
            styles: list[InlineStyle] = []
            if bold:
                styles.append("bold")
            if italic:
                styles.append("italic")
            if code:
                styles.append("code")
            ops.append(TextAppendOp(text=fragment, styles=tuple(styles), href=href))

        while i < len(text):
            if not code and text.startswith("**", i):
                bold = not bold
                i += 2
                continue
            if not code and text[i] == "*":
                italic = not italic
                i += 1
                continue
            if text[i] == "`":
                code = not code
                i += 1
                continue

            if not code and text[i] == "[":
                close_bracket = text.find("]", i + 1)
                if close_bracket > i and close_bracket + 1 < len(text) and text[close_bracket + 1] == "(":
                    close_paren = text.find(")", close_bracket + 2)
                    if close_paren > close_bracket:
                        label = text[i + 1 : close_bracket]
                        href = text[close_bracket + 2 : close_paren]
                        push(label, href=href)
                        i = close_paren + 1
                        continue

            j = i + 1
            while j < len(text):
                if not code and text.startswith("**", j):
                    break
                if text[j] in ("*", "`"):
                    break
                if not code and text[j] == "[":
                    break
                j += 1
            push(text[i:j])
            i = j

        return ops


class MarkdownOpHtmlRenderer:
    """Apply markdown operations and build final HTML fragment."""

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._in_paragraph = False
        self._heading_level: int | None = None
        self._blockquote_depth = 0
        self._list_stack: list[ListKind] = []
        self._in_code = False

        self._in_table = False
        self._row_is_header = False
        self._row_cells: dict[int, list[str]] = {}

    def apply(self, ops: list[MarkdownOp]) -> None:
        for op in ops:
            if isinstance(op, ParagraphStartOp):
                if not self._in_paragraph:
                    self._parts.append("<p>")
                    self._in_paragraph = True
                continue

            if isinstance(op, ParagraphEndOp):
                if self._in_paragraph:
                    # Drop trailing line break when a paragraph is closed.
                    # Markdown often uses blank lines only as block separators;
                    # we should not render an extra visible blank line before the next block.
                    if self._parts and self._parts[-1] == "<br/>":
                        self._parts.pop()
                    self._parts.append("</p>")
                    self._in_paragraph = False
                continue

            if isinstance(op, HeadingStartOp):
                self._parts.append(f"<h{op.level}>")
                self._heading_level = op.level
                continue

            if isinstance(op, HeadingEndOp):
                level = op.level if self._heading_level is None else self._heading_level
                self._parts.append(f"</h{level}>")
                self._heading_level = None
                continue

            if isinstance(op, BlockquoteDepthOp):
                while self._blockquote_depth > op.depth:
                    self._parts.append("</blockquote>")
                    self._blockquote_depth -= 1
                while self._blockquote_depth < op.depth:
                    self._parts.append("<blockquote>")
                    self._blockquote_depth += 1
                continue

            if isinstance(op, ListStartOp):
                if op.kind == "ol":
                    start_attr = f' start="{op.ordered_start}"' if op.ordered_start else ""
                    self._parts.append(f"<ol{start_attr}>")
                else:
                    self._parts.append("<ul>")
                self._list_stack.append(op.kind)
                continue

            if isinstance(op, ListEndOp):
                if self._list_stack:
                    kind = self._list_stack.pop()
                    self._parts.append("</ol>" if kind == "ol" else "</ul>")
                continue

            if isinstance(op, ListItemStartOp):
                self._parts.append("<li>")
                continue

            if isinstance(op, ListItemEndOp):
                self._parts.append("</li>")
                continue

            if isinstance(op, CodeBlockStartOp):
                lang = html.escape(op.language)
                class_attr = f' class="language-{lang}"' if lang else ""
                self._parts.append(f"<pre><code{class_attr}>")
                self._in_code = True
                continue

            if isinstance(op, CodeTextAppendOp):
                self._parts.append(html.escape(op.text))
                continue

            if isinstance(op, CodeBlockEndOp):
                if self._in_code:
                    self._parts.append("</code></pre>")
                    self._in_code = False
                continue

            if isinstance(op, TableStartOp):
                self._parts.append('<table class="md-table">')
                self._in_table = True
                continue

            if isinstance(op, TableRowStartOp):
                self._row_is_header = op.is_header
                self._row_cells = {}
                continue

            if isinstance(op, TableCellUpdateOp):
                self._row_cells.setdefault(op.column_index, []).append(
                    self._format_inline(op.text, op.styles, op.href)
                )
                continue

            if isinstance(op, TableRowEndOp):
                if not self._in_table:
                    continue
                tag = "th" if self._row_is_header else "td"
                self._parts.append("<tr>")
                for col in sorted(self._row_cells.keys()):
                    cell_content = "".join(self._row_cells[col])
                    self._parts.append(f"<{tag}>{cell_content}</{tag}>")
                self._parts.append("</tr>")
                self._row_cells = {}
                continue

            if isinstance(op, TableEndOp):
                if self._in_table:
                    self._parts.append("</table>")
                    self._in_table = False
                continue

            if isinstance(op, ThematicBreakOp):
                self._parts.append("<hr/>")
                continue

            if isinstance(op, LineBreakOp):
                if self._in_paragraph:
                    self._parts.append("<br/>")
                continue

            if isinstance(op, TextAppendOp):
                self._parts.append(self._format_inline(op.text, op.styles, op.href))

    def html(self) -> str:
        return "".join(self._parts)

    def reset(self) -> None:
        self.__init__()

    def _format_inline(
        self,
        text: str,
        styles: tuple[InlineStyle, ...],
        href: str | None,
    ) -> str:
        out = html.escape(text)

        if "code" in styles:
            out = f"<code>{out}</code>"
        if "italic" in styles:
            out = f"<em>{out}</em>"
        if "bold" in styles:
            out = f"<strong>{out}</strong>"

        if href:
            safe_href = html.escape(href, quote=True)
            out = f'<a href="{safe_href}">{out}</a>'

        return out


__all__ = [
    "MarkdownOp",
    "TextAppendOp",
    "ParagraphStartOp",
    "ParagraphEndOp",
    "HeadingStartOp",
    "HeadingEndOp",
    "BlockquoteDepthOp",
    "ListStartOp",
    "ListEndOp",
    "ListItemStartOp",
    "ListItemEndOp",
    "CodeBlockStartOp",
    "CodeTextAppendOp",
    "CodeBlockEndOp",
    "TableStartOp",
    "TableRowStartOp",
    "TableCellUpdateOp",
    "TableRowEndOp",
    "TableEndOp",
    "ThematicBreakOp",
    "LineBreakOp",
    "StreamingMarkdownOpParser",
    "MarkdownOpHtmlRenderer",
]
