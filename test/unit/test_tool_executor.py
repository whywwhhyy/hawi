from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

import pytest

import hawi.agent.tool_executor as tool_executor_module
from hawi.agent import HawiAgent, ToolCallRequest, ToolExecutor
from hawi.agent.agent import _ExecutionState
from hawi.events import AgentToolResultPartEvent
from hawi.models import Model
from hawi.models.message import DeltaPart, MessageRequest, MessageResponse, TokenUsage
from hawi.plugin import HawiPlugin, tool
from hawi.tool import ToolResult


class OrderedToolModel(Model):
    default_steer_merge_mode = "tool_result_assistant_template_and_user_message"

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "ordered-tool-model"

    def _prepare_request_impl(self, request: MessageRequest) -> dict[str, Any]:
        return {}

    def _parse_response_impl(self, response: dict[str, Any]) -> MessageResponse:
        return MessageResponse(
            id="response",
            role="assistant",
            content=[],
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=1, output_tokens=1),
        )

    def _invoke_impl(self, request: MessageRequest) -> MessageResponse:
        raise NotImplementedError

    async def _astream_impl(
        self,
        request: MessageRequest,
    ) -> AsyncGenerator[DeltaPart, None]:
        self.calls += 1
        if self.calls == 1:
            yield {
                "type": "tool_call_delta",
                "index": 0,
                "id": "call-slow",
                "name": "slow_tool",
                "arguments_delta": "{}",
                "is_start": True,
                "is_end": True,
            }
            yield {
                "type": "tool_call_delta",
                "index": 1,
                "id": "call-fast",
                "name": "fast_tool",
                "arguments_delta": "{}",
                "is_start": True,
                "is_end": True,
            }
            yield {"type": "finish", "stop_reason": "tool_use", "usage": None}
            return

        yield {
            "type": "text_delta",
            "index": 0,
            "delta": "done",
            "is_start": True,
            "is_end": True,
        }
        yield {"type": "finish", "stop_reason": "end_turn", "usage": None}


class DelayedSecondToolModel(OrderedToolModel):
    def __init__(self) -> None:
        super().__init__()
        self.before_second_tool_at: float | None = None

    async def _astream_impl(
        self,
        request: MessageRequest,
    ) -> AsyncGenerator[DeltaPart, None]:
        self.calls += 1
        if self.calls == 1:
            yield {
                "type": "tool_call_delta",
                "index": 0,
                "id": "call-slow",
                "name": "slow_tool",
                "arguments_delta": "{}",
                "is_start": True,
                "is_end": True,
            }
            await asyncio.sleep(0.12)
            self.before_second_tool_at = time.perf_counter()
            yield {
                "type": "tool_call_delta",
                "index": 1,
                "id": "call-fast",
                "name": "fast_tool",
                "arguments_delta": "{}",
                "is_start": True,
                "is_end": True,
            }
            yield {"type": "finish", "stop_reason": "tool_use", "usage": None}
            return

        yield {
            "type": "text_delta",
            "index": 0,
            "delta": "done",
            "is_start": True,
            "is_end": True,
        }
        yield {"type": "finish", "stop_reason": "end_turn", "usage": None}


class OrderedToolPlugin(HawiPlugin):
    def __init__(self) -> None:
        self.events: list[tuple[str, str, float]] = []
        self.shared_state: str | None = None

    @tool(
        name="slow_tool",
        description="Slow async tool",
        parameters_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    )
    async def slow_tool(self) -> str:
        self.events.append(("start", "slow", time.perf_counter()))
        await asyncio.sleep(0.2)
        self.shared_state = "ready"
        self.events.append(("finish", "slow", time.perf_counter()))
        return "slow result"

    @tool(
        name="fast_tool",
        description="Fast async tool",
        parameters_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    )
    async def fast_tool(self) -> str:
        self.events.append(("start", "fast", time.perf_counter()))
        await asyncio.sleep(0.02)
        self.events.append(("finish", "fast", time.perf_counter()))
        return f"fast saw {self.shared_state}"


class OversizeToolPlugin(HawiPlugin):
    @tool(
        name="big_text_tool",
        description="Return a large text result",
        parameters_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    )
    async def big_text_tool(self) -> str:
        return "x" * 5000

    @tool(
        name="big_dict_tool",
        description="Return a large structured result",
        parameters_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    )
    async def big_dict_tool(self) -> dict[str, str]:
        return {"payload": "x" * 5000}


class StreamingToolPlugin(HawiPlugin):
    @tool(
        name="streaming_tool",
        description="Stream partial output and then return a final result",
        parameters_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    )
    async def streaming_tool(self) -> AsyncGenerator[str | ToolResult, None]:
        yield "alpha"
        await asyncio.sleep(0)
        yield " beta"
        yield ToolResult(
            success=False,
            output="final output",
            error="final error",
        )


def _tool_result_ids(agent: HawiAgent) -> list[str]:
    ids: list[str] = []
    for message in agent.context.messages:
        if message["role"] != "tool":
            continue
        part = message["content"][0]
        ids.append(str(part.get("tool_call_id")))
    return ids


def test_tool_executor_is_exported() -> None:
    assert ToolExecutor.__name__ == "ToolExecutor"


def test_tool_call_request_is_exported() -> None:
    assert ToolCallRequest.__name__ == "ToolCallRequest"


@pytest.mark.asyncio
async def test_agent_executes_tool_batch_sequentially_in_model_order() -> None:
    plugin = OrderedToolPlugin()
    agent = HawiAgent(
        model=OrderedToolModel(),
        plugins=[plugin],
        streaming=True,
        max_iterations=3,
    )

    result = await agent.arun("run both tools")

    assert result.error is None
    assert result.text == "done"
    assert [record.tool_call_id for record in result.tool_calls] == [
        "call-slow",
        "call-fast",
    ]
    assert result.tool_calls[1].result.output == "fast saw ready"
    assert _tool_result_ids(agent) == ["call-slow", "call-fast"]
    assert [(event, name) for event, name, _ in plugin.events] == [
        ("start", "slow"),
        ("finish", "slow"),
        ("start", "fast"),
        ("finish", "fast"),
    ]
    slow_finish = plugin.events[1][2]
    fast_start = plugin.events[2][2]
    assert fast_start >= slow_finish


@pytest.mark.asyncio
async def test_agent_starts_tool_call_as_soon_as_stream_block_completes() -> None:
    model = DelayedSecondToolModel()
    plugin = OrderedToolPlugin()
    agent = HawiAgent(
        model=model,
        plugins=[plugin],
        streaming=True,
        max_iterations=3,
    )

    result = await agent.arun("run both tools")

    assert result.error is None
    assert model.before_second_tool_at is not None
    assert plugin.events[0][0:2] == ("start", "slow")
    assert plugin.events[0][2] < model.before_second_tool_at
    assert _tool_result_ids(agent) == ["call-slow", "call-fast"]


@pytest.mark.asyncio
async def test_tool_executor_promises_honor_blocked_by_tool_call_id() -> None:
    plugin = OrderedToolPlugin()
    agent = HawiAgent(
        model=OrderedToolModel(),
        plugins=[plugin],
        streaming=True,
        max_iterations=3,
    )
    executor = agent.tool_executor

    fast_request = ToolCallRequest(
        request_id="req-fast",
        blocked_by="call-slow",
        run_id="run",
        iteration=1,
        tool_call={
            "type": "tool_call",
            "id": "call-fast",
            "name": "fast_tool",
            "arguments": {},
        },
    )
    slow_request = ToolCallRequest(
        request_id="req-slow",
        run_id="run",
        iteration=1,
        tool_call={
            "type": "tool_call",
            "id": "call-slow",
            "name": "slow_tool",
            "arguments": {},
        },
    )

    fast_promise = executor.enqueue_call(fast_request)
    slow_promise = executor.enqueue_call(slow_request)
    snapshot = agent.snapshot_runtime()["tool_executor"]

    assert snapshot["queue"] == ["req-fast", "req-slow"]
    assert {
        request["request_id"]: request["blocked_by"]
        for request in snapshot["requests"]
    } == {"req-fast": "call-slow", "req-slow": None}

    records = await executor.drain_until_complete([fast_promise, slow_promise])

    assert [record.tool_call_id for record in records] == [
        "call-fast",
        "call-slow",
    ]
    assert fast_promise.done
    assert slow_promise.done
    assert fast_promise.result().result.output == "fast saw ready"
    assert [(event, name) for event, name, _ in plugin.events] == [
        ("start", "slow"),
        ("finish", "slow"),
        ("start", "fast"),
        ("finish", "fast"),
    ]


def test_tool_result_default_limit_is_context_friendly() -> None:
    assert tool_executor_module.TOOL_RESULT_MAX_BYTES == 50 * 1024


@pytest.mark.asyncio
async def test_tool_executor_converts_oversized_text_result_to_error_with_preview(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("HAWI_DEBUG", raising=False)
    monkeypatch.setattr(tool_executor_module, "TOOL_RESULT_MAX_BYTES", 3000)
    caplog.set_level(logging.WARNING, logger=tool_executor_module.__name__)
    agent = HawiAgent(
        model=OrderedToolModel(),
        plugins=[OversizeToolPlugin()],
        streaming=True,
    )

    record = await agent._execute_tool(
        {
            "type": "tool_call",
            "id": "call-big",
            "name": "big_text_tool",
            "arguments": {},
        },
        _ExecutionState(run_id="run-big", iteration=1),
    )

    assert record.result.success is False
    assert "Tool result from 'big_text_tool'" in record.result.error
    assert isinstance(record.result.output, dict)
    assert record.result.output["hawi_oversized_tool_result"] is True
    assert record.result.output["original_output_type"] == "str"
    assert record.result.output["output_preview"].startswith("xxx")
    assert "Hawi warning:" in record.result.output["output_preview"]
    serialized = tool_executor_module.ToolExecutor._serialize_tool_result_for_limit(
        record.result
    )
    assert len(serialized.encode("utf-8")) <= 3000
    assert "exceeding limit 3000 bytes" in caplog.text
    tool_part = agent.context.messages[-1]["content"][0]
    assert tool_part["tool_call_id"] == "call-big"
    assert tool_part["is_error"] is True
    tool_text = tool_part["content"][0]["text"]
    assert "Output before error:" in tool_text
    assert "Error: Tool result from 'big_text_tool'" in tool_text


@pytest.mark.asyncio
async def test_tool_executor_converts_oversized_structured_result_to_error_with_preview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HAWI_DEBUG", raising=False)
    monkeypatch.setattr(tool_executor_module, "TOOL_RESULT_MAX_BYTES", 3000)
    agent = HawiAgent(
        model=OrderedToolModel(),
        plugins=[OversizeToolPlugin()],
        streaming=True,
    )

    record = await agent._execute_tool(
        {
            "type": "tool_call",
            "id": "call-big-dict",
            "name": "big_dict_tool",
            "arguments": {},
        },
        _ExecutionState(run_id="run-big", iteration=1),
    )

    assert record.result.success is False
    assert "Tool result from 'big_dict_tool'" in record.result.error
    assert isinstance(record.result.output, dict)
    assert record.result.output["hawi_oversized_tool_result"] is True
    assert record.result.output["original_output_type"] == "dict"
    assert "serialized_preview" in record.result.output
    assert "payload" in record.result.output["serialized_preview"]
    serialized = tool_executor_module.ToolExecutor._serialize_tool_result_for_limit(
        record.result
    )
    assert len(serialized.encode("utf-8")) <= 3000


@pytest.mark.asyncio
async def test_tool_executor_does_not_raise_on_oversized_tool_result_in_debug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HAWI_DEBUG", "1")
    monkeypatch.setattr(tool_executor_module, "TOOL_RESULT_MAX_BYTES", 3000)
    agent = HawiAgent(
        model=OrderedToolModel(),
        plugins=[OversizeToolPlugin()],
        streaming=True,
    )

    record = await agent._execute_tool(
        {
            "type": "tool_call",
            "id": "call-big-debug",
            "name": "big_text_tool",
            "arguments": {},
        },
        _ExecutionState(run_id="run-big", iteration=1),
    )

    assert record.result.success is False
    assert "Tool result from 'big_text_tool'" in record.result.error
    assert isinstance(record.result.output, dict)
    assert record.result.output["output_preview"].startswith("xxx")


@pytest.mark.asyncio
async def test_tool_executor_streams_parts_and_uses_generator_final_result() -> None:
    agent = HawiAgent(
        model=OrderedToolModel(),
        plugins=[StreamingToolPlugin()],
        streaming=True,
    )
    part_events: list[AgentToolResultPartEvent] = []
    agent.subscribe_blocking(
        lambda event: part_events.append(event),
        ["agent.tool_result_part"],
    )

    record = await agent._execute_tool(
        {
            "type": "tool_call",
            "id": "call-stream",
            "name": "streaming_tool",
            "arguments": {},
        },
        _ExecutionState(run_id="run-stream", iteration=1),
    )

    assert [event.part for event in part_events] == ["alpha", " beta", ""]
    assert [event.is_final for event in part_events] == [False, False, True]
    assert record.result.success is False
    assert record.result.output == "final output"
    assert record.result.error == "final error"
