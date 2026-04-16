"""Scheduler bridge tests for tool-call streaming UI events."""

from __future__ import annotations

import queue

import hawi.events as events
from hawi_gui.protocol import UiToolCallDelta, UiToolCallStart, UiToolCallStop, UiToolResult
from hawi_gui.scheduler_bridge import SchedulerThread


def _drain(q: "queue.Queue") -> list[object]:
    out: list[object] = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            break
    return out


def test_tool_call_stream_events_are_forwarded_from_model_blocks() -> None:
    ui_q: "queue.Queue" = queue.Queue()
    cmd_q: "queue.Queue" = queue.Queue()
    bridge = SchedulerThread(ui_queue=ui_q, cmd_queue=cmd_q, model_name="dummy/model")

    bridge._on_agent_event(events.AgentRunStartEvent.create(run_id="run-1"))
    bridge._on_agent_event(
        events.ModelToolCallBlockStartEvent.create(
            request_id="run-1-0",
            block_index=0,
            tool_call_id="tc-1",
            tool_name="web.search",
        )
    )
    bridge._on_agent_event(
        events.ModelToolCallBlockDeltaEvent.create(
            request_id="run-1-0",
            block_index=0,
            tool_call_id="tc-1",
            arguments_delta='{"q":"weather"',
            is_streaming=True,
        )
    )
    bridge._on_agent_event(
        events.ModelToolCallBlockStopEvent.create(
            request_id="run-1-0",
            block_index=0,
            tool_call_id="tc-1",
            tool_name="web.search",
            arguments={"q": "weather"},
        )
    )
    bridge._on_agent_event(
        events.AgentToolResultEvent.create(
            run_id="run-1",
            tool_call_id="tc-1",
            success=True,
            result_preview="ok",
            duration_ms=12.0,
            result_obj=None,
        )
    )

    messages = _drain(ui_q)

    assert any(isinstance(msg, UiToolCallStart) and msg.tool_call_id == "tc-1" for msg in messages)
    assert any(isinstance(msg, UiToolCallDelta) and msg.tool_call_id == "tc-1" for msg in messages)
    assert any(isinstance(msg, UiToolCallStop) and msg.tool_call_id == "tc-1" for msg in messages)
    assert any(isinstance(msg, UiToolResult) and msg.tool_call_id == "tc-1" for msg in messages)
