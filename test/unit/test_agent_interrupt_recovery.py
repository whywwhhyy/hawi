from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from hawi.agent import HawiAgent
from hawi.agent.context import AgentContext
from hawi.models import Model
from hawi.models.message import DeltaPart, MessageRequest, MessageResponse, TokenUsage
from hawi.tool.types import AgentTool, ToolResult


class ToolCallModel(Model):
    default_steer_merge_mode = "tool_result_assistant_template_and_user_message"

    @property
    def model_id(self) -> str:
        return "tool-call-model"

    def _prepare_request_impl(self, request: MessageRequest) -> dict[str, Any]:
        return {}

    def _parse_response_impl(self, response: dict[str, Any]) -> MessageResponse:
        return MessageResponse(
            id="response",
            content=[],
            stop_reason="tool_use",
            usage=TokenUsage(input_tokens=1, output_tokens=1),
        )

    def _invoke_impl(self, request: MessageRequest) -> MessageResponse:
        return self._parse_response_impl({})

    async def _astream_impl(
        self,
        request: MessageRequest,
    ) -> AsyncGenerator[DeltaPart, None]:
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
            "type": "finish",
            "stop_reason": "tool_use",
            "usage": None,
        }


class SlowTool(AgentTool):
    @property
    def name(self) -> str:
        return "slow_tool"

    @property
    def description(self) -> str:
        return "A slow tool"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "additionalProperties": False}

    async def arun(self, **kwargs: Any) -> ToolResult:
        await asyncio.sleep(60)
        return ToolResult(success=True, output="done")


def test_context_inserts_missing_tool_results_before_later_messages() -> None:
    context = AgentContext()
    context.add_user_message("start")
    context.add_assistant_message([
        {"type": "tool_call", "id": "call-1", "name": "first", "arguments": {}},
        {"type": "tool_call", "id": "call-2", "name": "second", "arguments": {}},
    ])
    context.add_tool_result("call-1", "ok")
    context.add_user_message("urgent follow-up")

    recovered = context.add_missing_tool_results("Tool call interrupted.")

    assert [item.tool_call_id for item in recovered] == ["call-2"]
    assert [message["role"] for message in context.messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "user",
    ]
    tool_result = context.messages[3]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_call_id"] == "call-2"

    assert context.add_missing_tool_results("Tool call interrupted.") == []


@pytest.mark.asyncio
async def test_cancelled_tool_call_adds_error_tool_result_to_context() -> None:
    agent = HawiAgent(model=ToolCallModel(), streaming=True)
    agent._plugin_manager.add_tool(SlowTool())

    task = asyncio.create_task(agent.arun("run slow tool"))
    for _ in range(100):
        if agent.has_active_tool_calls:
            break
        await asyncio.sleep(0.01)

    assert agent.has_active_tool_calls
    agent.interrupt("urgent")
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert [message["role"] for message in agent.context.messages] == [
        "user",
        "assistant",
        "tool",
    ]
    tool_message = agent.context.messages[2]
    tool_result = tool_message["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_call_id"] == "call-slow"
    assert tool_result["is_error"] is True
    nested_text = tool_result["content"][0]
    assert nested_text["type"] == "text"
    assert "reason: urgent" in nested_text["text"]
