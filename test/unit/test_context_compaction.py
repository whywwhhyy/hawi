from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, cast

import pytest

from hawi.agent import AutoCompactConfig, HawiAgent
from hawi.agent.context import AgentContext, ContextUsageSnapshot
from hawi.events import Event, EventBus
from hawi.models import Model
from hawi.models.message import DeltaPart, Message, MessageRequest, MessageResponse


class CompactingModel(Model):
    default_steer_merge_mode = "tool_result_assistant_template_and_user_message"

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[list[Message]] = []

    @property
    def model_id(self) -> str:
        return "compacting-model"

    def _prepare_request_impl(self, request: MessageRequest) -> dict[str, Any]:
        return {}

    def _parse_response_impl(self, response: dict[str, Any]) -> MessageResponse:
        return MessageResponse(id="response", content=[])

    def _invoke_impl(self, request: MessageRequest) -> MessageResponse:
        return MessageResponse(id="response", content=[])

    async def _ainvoke_impl(
        self,
        request: MessageRequest,
    ) -> AsyncGenerator[DeltaPart, None]:
        self.requests.append(request.messages)
        first_part = cast(dict[str, Any], request.messages[0]["content"][0])
        text = str(first_part.get("text", ""))
        if "Summarize the following Hawi conversation transcript" in text:
            yield {
                "type": "text_delta",
                "index": 0,
                "delta": "Summary: preserve decisions and continue the current task.",
                "is_start": True,
                "is_end": True,
            }
            yield {
                "type": "finish",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
            return

        yield {
            "type": "text_delta",
            "index": 0,
            "delta": "done",
            "is_start": True,
            "is_end": True,
        }
        yield {
            "type": "finish",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }


class ContextTokenModel(Model):
    default_steer_merge_mode = "tool_result_assistant_template_and_user_message"

    @property
    def model_id(self) -> str:
        return "context-token-model"

    def _prepare_request_impl(self, request: MessageRequest) -> dict[str, Any]:
        return {}

    def _parse_response_impl(self, response: dict[str, Any]) -> MessageResponse:
        return MessageResponse(id="response", content=[])

    def _invoke_impl(self, request: MessageRequest) -> MessageResponse:
        return MessageResponse(id="response", content=[])

    async def _ainvoke_impl(
        self,
        request: MessageRequest,
    ) -> AsyncGenerator[DeltaPart, None]:
        yield {
            "type": "text_delta",
            "index": 0,
            "delta": "done",
            "is_start": True,
            "is_end": True,
        }
        yield {
            "type": "finish",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "context_tokens": 60, "output_tokens": 5},
        }


def test_context_compaction_keeps_tool_exchange_intact() -> None:
    context = AgentContext()
    context.add_user_message("old request")
    context.add_assistant_message([{"type": "text", "text": "old answer"}])
    context.add_user_message("new request")
    context.add_assistant_message([
        {"type": "tool_call", "id": "call-1", "name": "lookup", "arguments": {}}
    ])
    context.add_tool_result("call-1", "lookup result")
    context.add_assistant_message([{"type": "text", "text": "final answer"}])

    record = context.compact_with_summary("Old work summary", keep_last=2)

    assert record is not None
    metadata = cast(dict[str, Any], context.messages[0].get("metadata") or {})
    assert metadata["source"] == "context_compaction"
    assert [message["role"] for message in context.messages] == [
        "user",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    tool_call = context.messages[2]["content"][0]
    tool_result = context.messages[3]["content"][0]
    assert tool_call["type"] == "tool_call"
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_call_id"] == "call-1"


def test_context_usage_snapshot_reports_context_ratio() -> None:
    context = AgentContext()
    context.add_user_message("hello " * 20)

    snapshot = context.usage_snapshot(max_context_tokens=1000)

    assert snapshot.used_tokens > 0
    assert snapshot.max_context_tokens == 1000
    assert snapshot.usage_ratio is not None
    assert 0 < snapshot.usage_ratio < 1
    assert snapshot.remaining_tokens == 1000 - snapshot.used_tokens
    assert snapshot.source == "estimate"


def test_context_usage_snapshot_round_trips_in_context_snapshot() -> None:
    context = AgentContext()
    context.set_context_usage(
        ContextUsageSnapshot(
            used_tokens=500,
            max_context_tokens=1000,
            usage_ratio=0.5,
            remaining_tokens=500,
            source="provider_usage",
        )
    )

    restored = AgentContext()
    restored.load_snapshot(context.snapshot())

    assert restored.context_usage_snapshot() == ContextUsageSnapshot(
        used_tokens=500,
        max_context_tokens=1000,
        usage_ratio=0.5,
        remaining_tokens=500,
        source="provider_usage",
    )


def test_model_metadata_uses_normalized_provider_context_tokens() -> None:
    events: list[Event] = []
    bus = EventBus()
    bus.subscribe_blocking(events.append, event_types=["model.metadata"])
    model = ContextTokenModel()
    model.configure_max_context_tokens(100)
    agent = HawiAgent(model=model, event_bus=bus, streaming=False)

    try:
        agent.run("hi")
    finally:
        bus.close()

    metadata = cast(Any, events[-1])
    assert metadata.context_tokens == 60
    assert metadata.context_ratio == 0.6
    assert metadata.context_source == "provider_usage"
    assert agent.context.context_usage_snapshot() == ContextUsageSnapshot(
        used_tokens=60,
        max_context_tokens=100,
        usage_ratio=0.6,
        remaining_tokens=40,
        source="provider_usage",
    )


@pytest.mark.asyncio
async def test_agent_auto_compacts_before_model_call() -> None:
    model = CompactingModel()
    events: list[Event] = []
    bus = EventBus()
    bus.subscribe_blocking(
        events.append,
        event_types=["agent.compact_start", "agent.compact_stop"],
    )
    agent = HawiAgent(
        model=model,
        event_bus=bus,
        streaming=False,
        auto_compact=AutoCompactConfig(
            enabled=True,
            trigger_tokens=20,
            keep_last_messages=2,
            min_messages=3,
        ),
    )
    for idx in range(6):
        agent.context.add_user_message(f"old message {idx} " + ("x" * 80))
        agent.context.add_assistant_message([{"type": "text", "text": "old answer"}])

    try:
        result = await agent.arun("finish now")
    finally:
        bus.close()

    assert result.text == "done"
    assert len(agent.context.compaction_records) == 1
    metadata = cast(dict[str, Any], agent.context.messages[0].get("metadata") or {})
    assert metadata["source"] == "context_compaction"
    first_part = cast(dict[str, Any], agent.context.messages[0]["content"][0])
    assert "Summary: preserve decisions" in str(first_part.get("text", ""))
    # First model request is the summarization request; second is the actual run.
    assert len(model.requests) == 2
    assert len(model.requests[-1]) < 14
    assert [event.type for event in events] == [
        "agent.compact_start",
        "agent.compact_stop",
    ]
    start = events[0]
    stop = events[1]
    assert getattr(start, "mode") == "auto"
    assert getattr(stop, "mode") == "auto"
    assert getattr(stop, "status") == "success"
    assert getattr(start, "run_id") == getattr(stop, "run_id")
    assert getattr(stop, "replaced_message_count") > 0
    assert getattr(stop, "tokens_after") < getattr(stop, "tokens_before")


@pytest.mark.asyncio
async def test_agent_auto_compact_is_enabled_from_model_context_window() -> None:
    model = CompactingModel()
    model.configure_max_context_tokens(20)
    agent = HawiAgent(model=model, streaming=False)
    for idx in range(6):
        agent.context.add_user_message(f"old message {idx} " + ("x" * 80))
        agent.context.add_assistant_message([{"type": "text", "text": "old answer"}])

    result = await agent.arun("finish now")

    assert result.text == "done"
    assert len(agent.context.compaction_records) == 1
    assert len(model.requests) == 2
