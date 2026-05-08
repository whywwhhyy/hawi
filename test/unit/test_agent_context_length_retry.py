from __future__ import annotations

from collections.abc import AsyncGenerator
from copy import deepcopy
from typing import Any

import pytest

from hawi.agent import HawiAgent
from hawi.errors import ContextLengthError
from hawi.models import Model
from hawi.models.message import DeltaPart, Message, MessageRequest, MessageResponse
from hawi.plugin import HawiPlugin, tool


class ContextLengthAfterToolsModel(Model):
    default_steer_merge_mode = "tool_result_assistant_template_and_user_message"

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.requests: list[list[Message]] = []

    @property
    def model_id(self) -> str:
        return "context-length-after-tools"

    def _prepare_request_impl(self, request: MessageRequest) -> dict[str, Any]:
        return {}

    def _parse_response_impl(self, response: dict[str, Any]) -> MessageResponse:
        return MessageResponse(
            id="response",
            role="assistant",
            content=[],
            stop_reason="end_turn",
            usage=None,
        )

    def _invoke_impl(self, request: MessageRequest) -> MessageResponse:
        raise NotImplementedError

    async def _astream_impl(
        self,
        request: MessageRequest,
    ) -> AsyncGenerator[DeltaPart, None]:
        self.calls += 1
        self.requests.append(deepcopy(request.messages))

        if self.calls == 1:
            yield {
                "type": "tool_call_delta",
                "index": 0,
                "id": "call-long",
                "name": "long_tool",
                "arguments_delta": "{}",
                "is_start": True,
                "is_end": True,
            }
            yield {
                "type": "tool_call_delta",
                "index": 1,
                "id": "call-short",
                "name": "short_tool",
                "arguments_delta": "{}",
                "is_start": True,
                "is_end": True,
            }
            yield {"type": "finish", "stop_reason": "tool_use", "usage": None}
            return

        if self.calls == 2:
            raise ContextLengthError(
                "Context length exceeded",
                max_context_tokens=100,
                requested_tokens=101,
            )

        yield {
            "type": "text_delta",
            "index": 0,
            "delta": "ok",
            "is_start": True,
            "is_end": True,
        }
        yield {"type": "finish", "stop_reason": "end_turn", "usage": None}


class LargeToolPlugin(HawiPlugin):
    @tool(
        name="long_tool",
        description="Return a long result",
        parameters_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    )
    def long_tool(self) -> str:
        return "L" * 6_000

    @tool(
        name="short_tool",
        description="Return a short result",
        parameters_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    )
    def short_tool(self) -> str:
        return "short result"


def _tool_result_text(messages: list[Message], tool_call_id: str) -> str:
    for message in messages:
        if message["role"] != "tool":
            continue
        for part in message["content"]:
            if (
                isinstance(part, dict)
                and part.get("type") == "tool_result"
                and part.get("tool_call_id") == tool_call_id
            ):
                content = part.get("content")
                if isinstance(content, list):
                    return "\n".join(
                        str(item.get("text", ""))
                        for item in content
                        if isinstance(item, dict)
                    )
                return str(content)
    raise AssertionError(f"tool result not found: {tool_call_id}")


@pytest.mark.asyncio
async def test_context_length_retry_truncates_longest_unsent_tool_result() -> None:
    model = ContextLengthAfterToolsModel()
    agent = HawiAgent(
        model=model,
        plugins=[LargeToolPlugin()],
        streaming=True,
    )

    result = await agent.arun("run tools")

    assert result.error is None
    assert model.calls == 3
    assert agent._last_unsent_tool_results == []

    retried_messages = model.requests[2]
    long_result = _tool_result_text(retried_messages, "call-long")
    short_result = _tool_result_text(retried_messages, "call-short")

    assert "Hawi truncated this tool result" in long_result
    assert len(long_result) < 6_000
    assert short_result == "short result"
