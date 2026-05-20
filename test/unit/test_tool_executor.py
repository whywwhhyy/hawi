from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from hawi.agent import HawiAgent, ToolCallRequest, ToolExecutor
from hawi.models import Model
from hawi.models.message import DeltaPart, MessageRequest, MessageResponse, TokenUsage
from hawi.plugin import HawiPlugin, tool


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
