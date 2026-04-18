"""GUI app tests for agent content boundaries around tool calls."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from hawi_gui.app import HawiGuiApp
from hawi_gui.protocol import (
    CmdSetSystemPrompt,
    DEFAULT_SYSTEM_PROMPT,
    UiRunStart,
    UiTextDelta,
    UiToolCallStart,
    UiToolCallStop,
    UiToolResult,
)


_APP = QApplication.instance() or QApplication([])


def test_tool_call_splits_agent_content_into_new_bubble() -> None:
    gui = HawiGuiApp("dummy/model")

    try:
        gui._dispatch(UiRunStart(run_id="run-1", user_content="hi", queue_kind="normal"))
        gui._dispatch(UiTextDelta(delta="前半段", run_id="run-1"))
        gui._dispatch(UiToolCallStart(tool_name="web.search", tool_call_id="tc-1", run_id="run-1"))
        gui._dispatch(
            UiToolCallStop(
                run_id="run-1",
                tool_call_id="tc-1",
                tool_name="web.search",
                arguments={"q": "weather"},
            )
        )
        gui._dispatch(
            UiToolResult(
                tool_call_id="tc-1",
                tool_name="web.search",
                success=True,
                output="ok",
                duration_ms=12.0,
                run_id="run-1",
            )
        )
        gui._dispatch(UiTextDelta(delta="后半段", run_id="run-1"))

        assert [node.kind for node in gui._chat_nodes] == ["user", "agent", "tool", "agent"]
        assert "前半段" in gui._chat_nodes[1].html
        assert "后半段" in gui._chat_nodes[3].html
    finally:
        gui.close()


def test_tool_before_text_does_not_create_empty_agent_placeholder() -> None:
    gui = HawiGuiApp("dummy/model")

    try:
        gui._dispatch(UiRunStart(run_id="run-2", user_content="hi", queue_kind="normal"))
        gui._dispatch(UiToolCallStart(tool_name="web.search", tool_call_id="tc-2", run_id="run-2"))
        gui._dispatch(
            UiToolResult(
                tool_call_id="tc-2",
                tool_name="web.search",
                success=True,
                output="ok",
                duration_ms=8.0,
                run_id="run-2",
            )
        )
        gui._dispatch(UiTextDelta(delta="最终回答", run_id="run-2"))

        assert [node.kind for node in gui._chat_nodes] == ["user", "tool", "agent"]
        assert "最终回答" in gui._chat_nodes[2].html
    finally:
        gui.close()


def test_chat_style_uses_compact_qt_list_indent() -> None:
    gui = HawiGuiApp("dummy/model")

    try:
        style = gui._chat_style_block()
        assert ".md-content ul," in style
        assert ".md-content ol {" in style
        assert "margin: 0 0 8px 0;" in style
        assert "-qt-list-indent: 1;" in style
    finally:
        gui.close()


def test_system_prompt_box_uses_default_text() -> None:
    gui = HawiGuiApp("dummy/model")

    try:
        assert gui._system_prompt_box.toPlainText() == DEFAULT_SYSTEM_PROMPT
    finally:
        gui.close()


def test_apply_system_prompt_enqueues_update_command() -> None:
    gui = HawiGuiApp("dummy/model")

    try:
        gui._system_prompt_box.setPlainText("新的 system prompt")
        gui._apply_system_prompt()

        cmd = gui.cmd_queue.get_nowait()
        assert isinstance(cmd, CmdSetSystemPrompt)
        assert cmd.system_prompt == "新的 system prompt"
        assert gui._system_prompt_text == "新的 system prompt"
    finally:
        gui.close()
