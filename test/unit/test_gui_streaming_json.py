"""Unit tests for GUI streaming json argument rendering."""

from __future__ import annotations

from hawi_gui.streaming_json import (
    PLACEHOLDER,
    StreamingJsonState,
    best_effort_json_tree,
    render_json_tree_html,
)


def test_best_effort_partial_json_preserves_discovered_shape() -> None:
    tree = best_effort_json_tree('{"query": "weather", "filters": {"city": "Shanghai", "days":')
    assert isinstance(tree, dict)
    assert "query" in tree
    assert "filters" in tree


def test_streaming_json_state_finalize_uses_final_arguments() -> None:
    state = StreamingJsonState()
    state.feed('{"a":')
    tree = state.finalize({"a": 1, "b": {"c": True}})
    assert tree == {"a": 1, "b": {"c": True}}
    assert state.snapshot_tree() == {"a": 1, "b": {"c": True}}


def test_render_json_tree_html_uses_bold_keys_and_placeholder() -> None:
    html = render_json_tree_html({"alpha": PLACEHOLDER, "nested": {"beta": 2}})
    assert "<strong>alpha</strong>" in html
    assert "<strong>nested</strong>" in html
    assert "<strong>beta</strong>" in html
    assert "placeholder" in html
