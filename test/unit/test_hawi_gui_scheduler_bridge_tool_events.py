"""Scheduler bridge tests for tool-call streaming UI events."""

from __future__ import annotations

import asyncio
import queue
from types import SimpleNamespace

import hawi.events as events
from hawi_gui.protocol import CmdSetSystemPrompt, UiToolCallDelta, UiToolCallStart, UiToolCallStop, UiToolResult
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


def test_set_system_prompt_command_updates_scheduler_context() -> None:
    ui_q: "queue.Queue" = queue.Queue()
    cmd_q: "queue.Queue" = queue.Queue()
    bridge = SchedulerThread(ui_queue=ui_q, cmd_queue=cmd_q, model_name="dummy/model")

    class _DummyContext:
        def __init__(self) -> None:
            self.value: str | None = None

        def set_system_prompt(self, prompt: str) -> None:
            self.value = prompt

    dummy_context = _DummyContext()
    bridge._scheduler = SimpleNamespace(agent=SimpleNamespace(context=dummy_context))
    cmd_q.put(CmdSetSystemPrompt(system_prompt="新的系统提示词"))
    cmd_q.put(object())

    async def _run_once() -> None:
        task = asyncio.create_task(bridge._process_commands())
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(_run_once())

    assert bridge.system_prompt == "新的系统提示词"
    assert dummy_context.value == "新的系统提示词"
