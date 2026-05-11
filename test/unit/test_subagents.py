from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

import pytest

from hawi.agent import HawiAgent, SubAgentSpec, ToolCallContext
from hawi.models import Model
from hawi.models.message import DeltaPart, MessageRequest, MessageResponse
from hawi_plugins.subagent_plugin import SubAgentPlugin


class EchoModel(Model):
    default_steer_merge_mode = "tool_result_assistant_template_and_user_message"

    @property
    def model_id(self) -> str:
        return "echo-model"

    def _prepare_request_impl(self, request: MessageRequest) -> dict[str, Any]:
        return {}

    def _parse_response_impl(self, response: dict[str, Any]) -> MessageResponse:
        return MessageResponse(
            id="echo",
            content=[{"type": "text", "text": response.get("text", "")}],
            stop_reason="end_turn",
        )

    def _invoke_impl(self, request: MessageRequest) -> MessageResponse:
        return MessageResponse(
            id="echo",
            content=[{"type": "text", "text": self._last_user_text(request)}],
            stop_reason="end_turn",
        )

    async def _astream_impl(
        self,
        request: MessageRequest,
    ) -> AsyncIterator[DeltaPart]:
        text = "echo: " + self._last_user_text(request)
        yield {
            "type": "text_delta",
            "index": 0,
            "delta": text,
            "is_start": True,
            "is_end": True,
        }
        yield {
            "type": "finish",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    def _last_user_text(self, request: MessageRequest) -> str:
        for message in reversed(request.messages):
            if message.get("role") != "user":
                continue
            parts = message.get("content") or []
            return " ".join(
                str(part.get("text", ""))
                for part in parts
                if isinstance(part, dict) and part.get("type") == "text"
            )
        return ""


class SlowEchoModel(EchoModel):
    def __init__(self, delay: float = 0.2) -> None:
        super().__init__()
        self.delay = delay

    async def _astream_impl(
        self,
        request: MessageRequest,
    ) -> AsyncIterator[DeltaPart]:
        await asyncio.sleep(self.delay)
        async for part in super()._astream_impl(request):
            yield part


class DelayedStreamingModel(EchoModel):
    def __init__(self, delay: float = 0.2) -> None:
        super().__init__()
        self.delay = delay

    async def _astream_impl(
        self,
        request: MessageRequest,
    ) -> AsyncIterator[DeltaPart]:
        text = "streaming: " + self._last_user_text(request)
        midpoint = max(1, len(text) // 2)
        yield {
            "type": "text_delta",
            "index": 0,
            "delta": text[:midpoint],
            "is_start": True,
            "is_end": False,
        }
        await asyncio.sleep(self.delay)
        yield {
            "type": "text_delta",
            "index": 0,
            "delta": text[midpoint:],
            "is_start": False,
            "is_end": True,
        }
        yield {
            "type": "finish",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }


@pytest.mark.asyncio
async def test_spawn_fork_and_fresh_context_modes() -> None:
    agent = HawiAgent(model=EchoModel())
    agent.context.add_user_message("parent context")

    forked = await agent.subagents.spawn(SubAgentSpec(mode="fork", role="reviewer"))
    fresh = await agent.subagents.spawn(SubAgentSpec(mode="fresh", role="explorer"))

    try:
        assert len(forked.agent.context.messages) == len(agent.context.messages)
        assert fresh.agent.context.messages == []
        assert fresh.agent.context.tool_call_context is not None
        assert forked.agent.context.tool_call_context is not None
    finally:
        await agent.subagents.close_all(reason="test_cleanup")


@pytest.mark.asyncio
async def test_fork_drops_trailing_unanswered_parent_tool_call_turn() -> None:
    agent = HawiAgent(model=EchoModel())
    agent.context.add_user_message("parent context")
    agent.context.add_assistant_message([
        {"type": "text", "text": "I will use tools."},
        {"type": "tool_call", "id": "call_done", "name": "done_tool", "arguments": {}},
        {"type": "tool_call", "id": "call_pending", "name": "pending_tool", "arguments": {}},
    ])
    agent.context.add_tool_result("call_done", "done")

    forked = await agent.subagents.spawn(mode="fork")

    try:
        assert [message["role"] for message in forked.agent.context.messages] == ["user"]
        assert [message["role"] for message in agent.context.messages] == [
            "user",
            "assistant",
            "tool",
        ]
    finally:
        await agent.subagents.close_all(reason="test_cleanup")


@pytest.mark.asyncio
async def test_fork_drops_empty_assistant_message_when_all_tool_calls_pending() -> None:
    agent = HawiAgent(model=EchoModel())
    agent.context.add_user_message("parent context")
    agent.context.add_assistant_message([
        {"type": "tool_call", "id": "call_pending", "name": "pending_tool", "arguments": {}},
    ])

    forked = await agent.subagents.spawn(mode="fork")

    try:
        assert [message["role"] for message in forked.agent.context.messages] == ["user"]
        assert [message["role"] for message in agent.context.messages] == [
            "user",
            "assistant",
        ]
    finally:
        await agent.subagents.close_all(reason="test_cleanup")


@pytest.mark.asyncio
async def test_fork_keeps_completed_trailing_tool_call_turn() -> None:
    agent = HawiAgent(model=EchoModel())
    agent.context.add_user_message("parent context")
    agent.context.add_assistant_message([
        {"type": "tool_call", "id": "call_done", "name": "done_tool", "arguments": {}},
    ])
    agent.context.add_tool_result("call_done", "done")

    forked = await agent.subagents.spawn(mode="fork")

    try:
        assert [message["role"] for message in forked.agent.context.messages] == [
            "user",
            "assistant",
            "tool",
        ]
    finally:
        await agent.subagents.close_all(reason="test_cleanup")


@pytest.mark.asyncio
async def test_subagent_runs_initial_prompt_in_background() -> None:
    agent = HawiAgent(model=EchoModel())

    handle = await agent.subagents.spawn(
        mode="fresh",
        role="summarizer",
        initial_prompt="hello subagent",
    )

    try:
        result = await agent.subagents.wait(handle.id, timeout=2)
        assert result is not None
        assert "echo: hello subagent" in result.text

        status = agent.subagents.status(handle.id)
        assert status.state == "COMPLETED"
        assert status.last_result_text is not None
        assert "hello subagent" in status.last_result_text
    finally:
        await agent.subagents.close(handle.id, reason="test_cleanup")


@pytest.mark.asyncio
async def test_send_subagent_message_and_read_events() -> None:
    agent = HawiAgent(model=EchoModel())
    handle = await agent.subagents.spawn(mode="fresh")

    try:
        message_id = agent.subagents.send(handle.id, "second task")
        assert message_id
        result = await agent.subagents.wait(handle.id, timeout=2)
        assert result is not None
        assert "second task" in result.text

        data = agent.subagents.read(handle.id, view="events", limit=10)
        assert data["status"]["id"] == handle.id
        assert any(event["type"] == "agent.run_stop" for event in data["events"])
    finally:
        await agent.subagents.close(handle.id, reason="test_cleanup")


@pytest.mark.asyncio
async def test_subagent_status_tracks_latest_completed_result() -> None:
    agent = HawiAgent(model=EchoModel())
    handle = await agent.subagents.spawn(mode="fresh", initial_prompt="first task")

    try:
        await agent.subagents.wait(handle.id, timeout=2)
        assert "first task" in (agent.subagents.status(handle.id).last_result_text or "")

        agent.subagents.send(handle.id, "second task")
        await agent.subagents.wait(handle.id, timeout=2)

        status = agent.subagents.status(handle.id)
        assert status.last_result_text is not None
        assert "second task" in status.last_result_text
        assert "first task" not in status.last_result_text
    finally:
        await agent.subagents.close(handle.id, reason="test_cleanup")


@pytest.mark.asyncio
async def test_subagent_read_exposes_streaming_delta_and_partial_context() -> None:
    agent = HawiAgent(model=DelayedStreamingModel(delay=0.2))
    handle = await agent.subagents.spawn(mode="fresh", initial_prompt="live task")

    try:
        partial = await wait_for_partial_context(agent, handle.id)
        assert partial is not None
        assert partial["metadata"]["subagent_partial"] is True
        content = partial["content"]
        assert isinstance(content, list)
        assert any(
            isinstance(part, dict)
            and part.get("type") == "text"
            and "streaming:" in str(part.get("text"))
            for part in content
        )

        events = agent.subagents.read(handle.id, view="events", limit=20)
        assert any(
            event.get("type") == "model.content_block_delta"
            and "streaming:" in str(event.get("delta", ""))
            for event in events["events"]
        )

        await agent.subagents.wait(handle.id, timeout=2)
    finally:
        await agent.subagents.close(handle.id, reason="test_cleanup")


@pytest.mark.asyncio
async def test_subagent_plugin_exposes_lifecycle_tools() -> None:
    agent = HawiAgent(model=EchoModel(), plugins=[SubAgentPlugin()])
    definitions = agent.plugins.get_tool_definitions()
    names = {definition["name"] for definition in definitions}

    assert {
        "create_subagent",
        "send_subagent_message",
        "wait_subagent",
        "read_subagent",
        "close_subagent",
    }.issubset(names)
    create_schema = next(d for d in definitions if d["name"] == "create_subagent")["schema"]
    assert "ctx" not in create_schema.get("properties", {})

    create_tool = agent.plugins.get_tool("create_subagent")
    assert create_tool is not None
    created = await create_tool.arun(
        mode="fresh",
        role="reviewer",
        initial_prompt="plugin task",
        ctx=ToolCallContext(agent),
    )
    assert created.success is True
    subagent_id = created.output["subagent_id"]  # type: ignore[index]

    try:
        await agent.subagents.wait(subagent_id, timeout=2)
        read_tool = agent.plugins.get_tool("read_subagent")
        assert read_tool is not None
        read = await read_tool.arun(
            subagent_id=subagent_id,
            view="summary",
            ctx=ToolCallContext(agent),
        )
        assert read.success is True
        assert read.output["status"]["id"] == subagent_id  # type: ignore[index]
    finally:
        await agent.subagents.close(subagent_id, reason="test_cleanup")


async def wait_for_partial_context(
    agent: HawiAgent,
    subagent_id: str,
    *,
    timeout: float = 1,
) -> dict[str, Any] | None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        data = agent.subagents.read(subagent_id, view="context_tail", limit=5)
        for message in data["messages"]:
            if (
                isinstance(message, dict)
                and isinstance(message.get("metadata"), dict)
                and message["metadata"].get("subagent_partial") is True
            ):
                return message
        await asyncio.sleep(0.01)
    return None


@pytest.mark.asyncio
async def test_wait_subagent_tool_returns_running_status_on_timeout() -> None:
    agent = HawiAgent(model=SlowEchoModel(), plugins=[SubAgentPlugin()])
    create_tool = agent.plugins.get_tool("create_subagent")
    wait_tool = agent.plugins.get_tool("wait_subagent")
    assert create_tool is not None
    assert wait_tool is not None

    created = await create_tool.arun(
        mode="fresh",
        initial_prompt="slow task",
        ctx=ToolCallContext(agent),
    )
    assert created.success is True
    subagent_id = created.output["subagent_id"]  # type: ignore[index]

    try:
        timed_out = await wait_tool.arun(
            subagent_id=subagent_id,
            notify_timeout=0.01,
            ctx=ToolCallContext(agent),
        )
        assert timed_out.success is True
        assert timed_out.output["timed_out"] is True  # type: ignore[index]
        assert timed_out.output["status"]["state"] == "RUNNING"  # type: ignore[index]
        assert "next_action" in timed_out.output  # type: ignore[operator]

        completed = await wait_tool.arun(
            subagent_id=subagent_id,
            notify_timeout=2,
            ctx=ToolCallContext(agent),
        )
        assert completed.success is True
        assert completed.output["timed_out"] is False  # type: ignore[index]
        assert "slow task" in completed.output["result_text"]  # type: ignore[index]
    finally:
        await agent.subagents.close(subagent_id, reason="test_cleanup")
