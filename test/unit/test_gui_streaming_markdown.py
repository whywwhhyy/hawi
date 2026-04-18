"""Unit tests for GUI streaming markdown parser and renderer."""

from __future__ import annotations

from hawi_gui.streaming_markdown import (
    ListStartOp,
    MarkdownOp,
    MarkdownOpHtmlRenderer,
    StreamingMarkdownOpParser,
    TableEndOp,
    TableRowStartOp,
    TableStartOp,
)


def _collect_ops(text: str, chunk: int) -> list[MarkdownOp]:
    parser = StreamingMarkdownOpParser()
    ops: list[MarkdownOp] = []
    for i in range(0, len(text), chunk):
        ops.extend(parser.feed(text[i : i + chunk]))
    ops.extend(parser.flush())
    return ops


def _render_html(text: str, chunk: int) -> str:
    parser = StreamingMarkdownOpParser()
    renderer = MarkdownOpHtmlRenderer()
    for i in range(0, len(text), chunk):
        renderer.apply(parser.feed(text[i : i + chunk]))
    renderer.apply(parser.flush())
    return renderer.html()


def test_chunking_keeps_same_final_html_for_core_gfm() -> None:
    text = (
        "# Title\n"
        "Hello **bold** and *italic* with `code` [link](https://example.com)\n\n"
        "- one\n"
        "  - nested\n\n"
        "| k | v |\n"
        "| --- | --- |\n"
        "| a | 1 |\n\n"
        "```py\n"
        "print('x')\n"
        "```\n"
    )

    html_chunk_1 = _render_html(text, 1)
    html_chunk_7 = _render_html(text, 7)
    html_chunk_all = _render_html(text, len(text))

    assert html_chunk_1 == html_chunk_7 == html_chunk_all
    assert "<h1>Title</h1>" in html_chunk_1
    assert "<strong>bold</strong>" in html_chunk_1
    assert "<em>italic</em>" in html_chunk_1
    assert '<a href="https://example.com">link</a>' in html_chunk_1


def test_nested_list_emits_depth_two_list_start() -> None:
    ops = _collect_ops("- one\n  - nested\n", 1)
    starts = [op for op in ops if isinstance(op, ListStartOp)]
    assert any(op.depth == 2 for op in starts)


def test_table_lifecycle_ops_are_emitted() -> None:
    ops = _collect_ops("| a | b |\n| --- | --- |\n| 1 | 2 |\n\n", 2)
    assert any(isinstance(op, TableStartOp) for op in ops)
    assert any(isinstance(op, TableRowStartOp) for op in ops)
    assert any(isinstance(op, TableEndOp) for op in ops)


def test_paragraph_to_list_does_not_emit_extra_visual_blank_line() -> None:
    html = _render_html("普通文本\n\n- 列表项\n", 1)
    assert "<p>普通文本</p>" in html
    assert "<p>普通文本<br/></p>" not in html
    assert "</p><ul>" in html


def test_ordered_list_does_not_preview_duplicate_marker() -> None:
    html = _render_html("1. 外貌特征\n2. 社交达人\n", 1)
    assert "<p>1.</p>" not in html
    assert "<p>2.</p>" not in html
    assert '<ol start="1"><li>外貌特征</li><li>社交达人</li></ol>' in html
