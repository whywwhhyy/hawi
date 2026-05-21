from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

import pytest

from hawi.agent import HawiAgent, SubAgentSpec, ToolCallContext
from hawi.agent.subagent.prompts import (
    ROLE_SYSTEM_PROMPTS,
    SUBAGENT_IDENTITY_PROMPT,
)
from hawi.models import Model
from hawi.models.message import DeltaPart, MessageRequest, MessageResponse
from hawi.builtin_plugins.subagent_plugin import SubAgentPlugin
from hawi.plugin import HawiPlugin, tool as plugin_tool


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


class DummyToolsPlugin(HawiPlugin):
    name = "hawi/dummy-tools"
    display_name = "Dummy Tools"

    @plugin_tool(name="dummy_tool")
    def dummy_tool(self, value: str) -> dict[str, str]:
        """Return a dummy value."""
        return {"value": value}


def prompt_text(parts: list[dict[str, Any]] | None) -> str:
    return "\n".join(
        str(part.get("text", ""))
        for part in parts or []
        if isinstance(part, dict) and part.get("type") == "text"
    )


@pytest.mark.asyncio
async def test_spawn_fork_and_fresh_context_modes() -> None:
    agent = HawiAgent(model=EchoModel(), system_prompt="parent prompt")
    agent.context.add_user_message("parent context")

    forked = await agent.subagents.spawn(SubAgentSpec(mode="fork", role="reviewer"))
    fresh = await agent.subagents.spawn(SubAgentSpec(mode="fresh", role="explorer"))

    try:
        assert len(forked.agent.context.messages) == len(agent.context.messages)
        assert fresh.agent.context.messages == []
        assert fresh.agent.context.tool_call_context is not None
        assert forked.agent.context.tool_call_context is not None
        assert agent.subagents.status(forked.id).mode == "fork"
        assert agent.subagents.status(forked.id).shared_context is True
        assert agent.subagents.status(fresh.id).mode == "fresh"
        assert agent.subagents.status(fresh.id).shared_context is False
        assert "parent prompt" in prompt_text(forked.agent.context.get_system_prompt())
        assert SUBAGENT_IDENTITY_PROMPT in prompt_text(
            forked.agent.context.get_system_prompt()
        )
        fresh_prompt = prompt_text(fresh.agent.context.get_system_prompt())
        assert "parent prompt" not in fresh_prompt
        assert SUBAGENT_IDENTITY_PROMPT in fresh_prompt
        assert ROLE_SYSTEM_PROMPTS["explorer"] in fresh_prompt
    finally:
        await agent.subagents.close_all(reason="test_cleanup")


@pytest.mark.asyncio
async def test_spawn_defaults_to_fresh_context_without_parent_system_prompt() -> None:
    agent = HawiAgent(model=EchoModel(), system_prompt="parent prompt")
    agent.context.add_user_message("parent context")

    handle = await agent.subagents.spawn()

    try:
        assert handle.spec.mode == "fresh"
        assert handle.agent.context.messages == []
        child_prompt = prompt_text(handle.agent.context.get_system_prompt())
        assert "parent prompt" not in child_prompt
        assert SUBAGENT_IDENTITY_PROMPT in child_prompt
        assert ROLE_SYSTEM_PROMPTS["general"] in child_prompt
    finally:
        await agent.subagents.close_all(reason="test_cleanup")


@pytest.mark.asyncio
async def test_spawn_accepts_parent_controlled_system_prompt() -> None:
    agent = HawiAgent(model=EchoModel(), system_prompt="parent prompt")

    handle = await agent.subagents.spawn(system_prompt="child controlled prompt")

    try:
        child_prompt = prompt_text(handle.agent.context.get_system_prompt())
        assert child_prompt == "child controlled prompt"
        assert "parent prompt" not in child_prompt
        assert ROLE_SYSTEM_PROMPTS["general"] not in child_prompt
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
        assert "You are a managed Hawi sub-agent" in result.text
        assert "Your task from the parent agent" in result.text
        assert "hello subagent" in result.text

        status = agent.subagents.status(handle.id)
        assert status.state == "COMPLETED"
        assert status.last_result_text is not None
        assert "hello subagent" in status.last_result_text
    finally:
        await agent.subagents.close(handle.id, reason="test_cleanup")


@pytest.mark.asyncio
async def test_fork_initial_prompt_clarifies_shared_context_handoff() -> None:
    agent = HawiAgent(model=EchoModel())
    agent.context.add_user_message("parent context that should be only background")

    handle = await agent.subagents.spawn(
        mode="fork",
        initial_prompt="review only this assigned task",
    )

    try:
        result = await agent.subagents.wait(handle.id, timeout=2)
        assert result is not None
        assert "The messages before this one are inherited parent-agent context" in (
            result.text
        )
        assert "Treat them only as background material" in result.text
        assert "responsible only for the task below" in result.text
        assert "tell the parent agent the result of your work" in result.text
        assert "review only this assigned task" in result.text
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
async def test_subagent_run_stop_event_reports_completed_status() -> None:
    agent = HawiAgent(model=EchoModel())
    events: list[Any] = []
    agent.subscribe_blocking(events.append, ["subagent.event"])

    handle = await agent.subagents.spawn(mode="fresh", initial_prompt="finish task")

    try:
        await agent.subagents.wait(handle.id, timeout=2)
        run_stop_events = [
            event
            for event in events
            if (event.child_event or {}).get("type") == "agent.run_stop"
        ]
        assert run_stop_events
        assert run_stop_events[-1].status["state"] == "COMPLETED"
        deadline = asyncio.get_running_loop().time() + 2
        settled_events: list[Any] = []
        while asyncio.get_running_loop().time() < deadline:
            settled_events = [
                event
                for event in events
                if (event.child_event or {}).get("type") == "subagent.status"
            ]
            if settled_events:
                break
            await asyncio.sleep(0.01)
        assert settled_events
        settled_status = settled_events[-1].status
        assert settled_status["state"] == "COMPLETED"
        assert settled_status["runner_state"] == "IDLE"
        assert settled_status["executor_state"] == "IDLE"
    finally:
        await agent.subagents.close(handle.id, reason="test_cleanup")


@pytest.mark.asyncio
async def test_subagent_markdown_export_uses_message_history(tmp_path) -> None:
    agent = HawiAgent(model=EchoModel())
    agent.subagents.configure_session_storage(
        root=tmp_path,
        session_id_provider=lambda: "parent-session",
    )
    handle = await agent.subagents.spawn(mode="fresh", initial_prompt="export task")

    try:
        await agent.subagents.wait(handle.id, timeout=2)
        assert handle.message_history

        data = agent.subagents.read(handle.id, view="markdown")
        assert "markdown" in data
        assert "export task" in data["markdown"]
        export = data["export"]
        assert export["markdown_path"]
        assert export["message_history_path"]
        assert "read_subagent" == export["query"]["tool"]
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
    agent = HawiAgent(model=EchoModel(), plugins=[SubAgentPlugin(), DummyToolsPlugin()])
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
    assert create_schema["properties"]["mode"]["default"] == "fresh"
    assert create_schema["properties"]["share_context"]["default"] is False
    assert "initial_prompt" in create_schema.get("required", [])
    assert "plugins" not in create_schema.get("required", [])
    assert create_schema["properties"]["plugins"]["default"] is None
    plugin_description = create_schema["properties"]["plugins"]["description"]
    assert "null/None" in plugin_description
    assert "[]" in plugin_description
    assert "share_context" in plugin_description

    create_tool = agent.plugins.get_tool("create_subagent")
    assert create_tool is not None
    assert "plugins argument controls the child's tools" in create_tool.description
    created = await create_tool.arun(
        mode="fresh",
        role="reviewer",
        initial_prompt="plugin task",
        plugins=["hawi/dummy-tools"],
        ctx=ToolCallContext(agent),
    )
    assert created.success is True
    subagent_id = created.output["subagent_id"]  # type: ignore[index]

    try:
        handle = agent.subagents._handles[subagent_id]  # type: ignore[attr-defined]
        assert handle.agent.plugins.get_tool("dummy_tool") is not None
        assert handle.agent.plugins.get_tool("create_subagent") is None
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


@pytest.mark.asyncio
async def test_create_subagent_tool_requires_initial_user_prompt() -> None:
    agent = HawiAgent(model=EchoModel(), plugins=[SubAgentPlugin()])
    create_tool = agent.plugins.get_tool("create_subagent")
    assert create_tool is not None

    created = await create_tool.arun(ctx=ToolCallContext(agent))

    assert created.success is False
    assert "initial_prompt" in created.error


@pytest.mark.asyncio
async def test_create_subagent_plugins_none_inherits_and_empty_list_disables_tools() -> None:
    agent = HawiAgent(model=EchoModel(), plugins=[SubAgentPlugin(), DummyToolsPlugin()])
    create_tool = agent.plugins.get_tool("create_subagent")
    assert create_tool is not None

    inherited = await create_tool.arun(
        initial_prompt="inherit tools",
        ctx=ToolCallContext(agent),
    )
    assert inherited.success is True
    inherited_id = inherited.output["subagent_id"]  # type: ignore[index]

    no_tools = await create_tool.arun(
        initial_prompt="no tools",
        plugins=[],
        ctx=ToolCallContext(agent),
    )
    assert no_tools.success is True
    no_tools_id = no_tools.output["subagent_id"]  # type: ignore[index]

    try:
        inherited_handle = agent.subagents._handles[inherited_id]  # type: ignore[attr-defined]
        no_tools_handle = agent.subagents._handles[no_tools_id]  # type: ignore[attr-defined]
        assert inherited_handle.agent.plugins.get_tool("dummy_tool") is not None
        assert inherited_handle.agent.plugins.get_tool("create_subagent") is not None
        assert no_tools_handle.agent.plugins.get_tool_definitions() == []
    finally:
        await agent.subagents.close(inherited_id, reason="test_cleanup")
        await agent.subagents.close(no_tools_id, reason="test_cleanup")


@pytest.mark.asyncio
async def test_create_subagent_shared_context_must_inherit_plugins() -> None:
    agent = HawiAgent(model=EchoModel(), plugins=[SubAgentPlugin(), DummyToolsPlugin()])
    create_tool = agent.plugins.get_tool("create_subagent")
    assert create_tool is not None

    rejected = await create_tool.arun(
        share_context=True,
        initial_prompt="shared task",
        plugins=["hawi/dummy-tools"],
        ctx=ToolCallContext(agent),
    )
    assert rejected.success is False
    assert "inherit the parent plugin setup" in rejected.error

    rejected_no_tools = await create_tool.arun(
        share_context=True,
        initial_prompt="shared task",
        plugins=[],
        ctx=ToolCallContext(agent),
    )
    assert rejected_no_tools.success is False
    assert "inherit the parent plugin setup" in rejected_no_tools.error

    accepted = await create_tool.arun(
        share_context=True,
        initial_prompt="shared task",
        ctx=ToolCallContext(agent),
    )
    assert accepted.success is True
    subagent_id = accepted.output["subagent_id"]  # type: ignore[index]

    try:
        handle = agent.subagents._handles[subagent_id]  # type: ignore[attr-defined]
        assert handle.spec.mode == "fork"
        assert handle.agent.plugins.get_tool("dummy_tool") is not None
        assert handle.agent.plugins.get_tool("create_subagent") is not None
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
        plugins=["hawi/subagent"],
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
