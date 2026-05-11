from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import pytest

from hawi.agent import HawiAgent
from hawi.models import Model
from hawi.models.message import DeltaPart, MessageRequest, MessageResponse
from hawi.plugin import HawiPlugin, HookResult, after_tool_calling, before_tool_calling, tool


class ToolThenDoneModel(Model):
    default_steer_merge_mode = "tool_result_assistant_template_and_user_message"

    def __init__(self, *, two_tools: bool = False) -> None:
        super().__init__()
        self.calls = 0
        self.two_tools = two_tools

    @property
    def model_id(self) -> str:
        return "tool-then-done"

    def _prepare_request_impl(self, request: MessageRequest) -> dict[str, Any]:
        return {}

    def _parse_response_impl(self, response: dict[str, Any]) -> MessageResponse:
        raise NotImplementedError

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
                "id": "call-echo",
                "name": "echo_tool",
                "arguments_delta": '{"text": "hello"}',
                "is_start": True,
                "is_end": True,
            }
            if self.two_tools:
                yield {
                    "type": "tool_call_delta",
                    "index": 1,
                    "id": "call-other",
                    "name": "other_tool",
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


class ToolPlugin(HawiPlugin):
    def __init__(self) -> None:
        self.calls: list[str] = []

    @tool(name="echo_tool")
    def echo_tool(self, text: str) -> str:
        self.calls.append(f"echo:{text}")
        return f"echoed {text}"

    @tool(name="other_tool")
    def other_tool(self) -> str:
        self.calls.append("other")
        return "other"


class AbortBeforeToolPlugin(ToolPlugin):
    @before_tool_calling
    def abort_before(self, agent, tool_name, arguments, ctx):
        return HookResult.abort("blocked before tool")


class AbortAfterToolPlugin(ToolPlugin):
    @after_tool_calling
    def abort_after(self, agent, tool_name, arguments, result, ctx):
        return HookResult.abort("blocked after tool")


class ReinvokeAfterToolPlugin(ToolPlugin):
    @after_tool_calling
    def reinvoke_after(self, agent, tool_name, arguments, result, ctx):
        return HookResult.reinvoke("continue with X")


@pytest.mark.asyncio
async def test_before_tool_calling_abort_writes_synthetic_results_and_stops() -> None:
    model = ToolThenDoneModel(two_tools=True)
    plugin = AbortBeforeToolPlugin()
    agent = HawiAgent(model=model, plugins=[plugin])

    result = await agent.arun("start")

    assert result.stop_reason == "hook_abort"
    assert plugin.calls == []
    assert model.calls == 1
    assert [record.tool_call_id for record in result.tool_calls] == [
        "call-echo",
        "call-other",
    ]
    assert "Aborted by before_tool_calling" in str(result.tool_calls[0].result.error)
    assert "tool batch stopped" in str(result.tool_calls[1].result.error)


@pytest.mark.asyncio
async def test_after_tool_calling_abort_writes_result_then_stops() -> None:
    model = ToolThenDoneModel()
    plugin = AbortAfterToolPlugin()
    agent = HawiAgent(model=model, plugins=[plugin])

    result = await agent.arun("start")

    assert result.stop_reason == "hook_abort"
    assert plugin.calls == ["echo:hello"]
    assert model.calls == 1
    assert any(message["role"] == "tool" for message in agent.context.messages)


@pytest.mark.asyncio
async def test_after_tool_calling_reinvoke_appends_message_and_continues() -> None:
    model = ToolThenDoneModel()
    plugin = ReinvokeAfterToolPlugin()
    agent = HawiAgent(model=model, plugins=[plugin])

    result = await agent.arun("start")

    assert result.stop_reason == "end_turn"
    assert plugin.calls == ["echo:hello"]
    assert model.calls == 2
    user_text = str(
        [
            part.get("text")
            for message in agent.context.messages
            if message["role"] == "user"
            for part in message["content"]
            if isinstance(part, dict)
        ]
    )
    assert "continue with X" in user_text
