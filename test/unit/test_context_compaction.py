from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import Any, cast

import pytest

from hawi.agent import AutoCompactConfig, HawiAgent
from hawi.agent.context import AgentContext, ContextUsageSnapshot
from hawi.errors import ContextLengthError
from hawi.events import Event, EventBus
from hawi.models import Model
from hawi.models.message import DeltaPart, Message, MessageRequest, MessageResponse
from hawi.plugin import HawiPlugin
from hawi.plugin.decorators import before_model_call, tool


class CompactingModel(Model):
    default_steer_merge_mode = "tool_result_assistant_template_and_user_message"

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[list[Message]] = []
        self.system_prompts: list[Any] = []

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
        self.system_prompts.append(request.system)
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


class ContextLengthThenCompactModel(CompactingModel):
    def __init__(self) -> None:
        super().__init__()
        self.real_calls = 0

    @property
    def model_id(self) -> str:
        return "context-length-then-compact-model"

    async def _ainvoke_impl(
        self,
        request: MessageRequest,
    ) -> AsyncGenerator[DeltaPart, None]:
        self.requests.append(request.messages)
        self.system_prompts.append(request.system)
        first_part = cast(dict[str, Any], request.messages[0]["content"][0])
        text = str(first_part.get("text", ""))
        if "Summarize the following Hawi conversation transcript" in text:
            yield {
                "type": "text_delta",
                "index": 0,
                "delta": "Summary after provider context-length error.",
                "is_start": True,
                "is_end": True,
            }
            yield {"type": "finish", "stop_reason": "end_turn", "usage": None}
            return

        self.real_calls += 1
        if self.real_calls == 1:
            raise ContextLengthError(
                "Context length exceeded",
                max_context_tokens=100,
                requested_tokens=150,
            )

        yield {
            "type": "text_delta",
            "index": 0,
            "delta": "done",
            "is_start": True,
            "is_end": True,
        }
        yield {"type": "finish", "stop_reason": "end_turn", "usage": None}


class CancelDuringCompactionModel(CompactingModel):
    @property
    def model_id(self) -> str:
        return "cancel-during-compaction-model"

    async def _ainvoke_impl(
        self,
        request: MessageRequest,
    ) -> AsyncGenerator[DeltaPart, None]:
        self.requests.append(request.messages)
        self.system_prompts.append(request.system)
        first_part = cast(dict[str, Any], request.messages[0]["content"][0])
        text = str(first_part.get("text", ""))
        if "Summarize the following Hawi conversation transcript" in text:
            raise asyncio.CancelledError()

        yield {
            "type": "text_delta",
            "index": 0,
            "delta": "done",
            "is_start": True,
            "is_end": True,
        }
        yield {"type": "finish", "stop_reason": "end_turn", "usage": None}


class VerboseCompactionModel(CompactingModel):
    @property
    def model_id(self) -> str:
        return "verbose-compaction-model"

    async def _ainvoke_impl(
        self,
        request: MessageRequest,
    ) -> AsyncGenerator[DeltaPart, None]:
        self.requests.append(request.messages)
        self.system_prompts.append(request.system)
        first_part = cast(dict[str, Any], request.messages[0]["content"][0])
        text = str(first_part.get("text", ""))
        if "Summarize the following Hawi conversation transcript" in text:
            yield {
                "type": "text_delta",
                "index": 0,
                "delta": "S" * 10_000,
                "is_start": True,
                "is_end": True,
            }
            yield {"type": "finish", "stop_reason": "end_turn", "usage": None}
            return

        yield {
            "type": "text_delta",
            "index": 0,
            "delta": "done",
            "is_start": True,
            "is_end": True,
        }
        yield {"type": "finish", "stop_reason": "end_turn", "usage": None}


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


class FinalOverflowModel(CompactingModel):
    @property
    def model_id(self) -> str:
        return "final-overflow-model"

    async def _ainvoke_impl(
        self,
        request: MessageRequest,
    ) -> AsyncGenerator[DeltaPart, None]:
        self.requests.append(request.messages)
        self.system_prompts.append(request.system)
        first_part = cast(dict[str, Any], request.messages[0]["content"][0])
        text = str(first_part.get("text", ""))
        if "Summarize the following Hawi conversation transcript" in text:
            yield {
                "type": "text_delta",
                "index": 0,
                "delta": "Summary: compact after final answer.",
                "is_start": True,
                "is_end": True,
            }
            yield {"type": "finish", "stop_reason": "end_turn", "usage": None}
            return

        yield {
            "type": "text_delta",
            "index": 0,
            "delta": "final " + ("x" * 400),
            "is_start": True,
            "is_end": True,
        }
        yield {"type": "finish", "stop_reason": "end_turn", "usage": None}


class ToolCallingContextModel(Model):
    default_steer_merge_mode = "tool_result_assistant_template_and_user_message"

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "tool-calling-context-model"

    def _prepare_request_impl(self, request: MessageRequest) -> dict[str, Any]:
        return {}

    def _parse_response_impl(self, response: dict[str, Any]) -> MessageResponse:
        return MessageResponse(id="response", content=[])

    def _invoke_impl(self, request: MessageRequest) -> MessageResponse:
        return MessageResponse(id="response", content=[])

    async def _astream_impl(
        self,
        request: MessageRequest,
    ) -> AsyncGenerator[DeltaPart, None]:
        self.calls += 1
        if self.calls == 1:
            yield {
                "type": "tool_call_delta",
                "index": 0,
                "id": "call-ctx",
                "name": "context_probe",
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


class TimingStreamingModel(Model):
    default_steer_merge_mode = "tool_result_assistant_template_and_user_message"

    @property
    def model_id(self) -> str:
        return "timing-streaming-model"

    def _prepare_request_impl(self, request: MessageRequest) -> dict[str, Any]:
        return {}

    def _parse_response_impl(self, response: dict[str, Any]) -> MessageResponse:
        return MessageResponse(id="response", content=[])

    def _invoke_impl(self, request: MessageRequest) -> MessageResponse:
        return MessageResponse(id="response", content=[])

    async def _astream_impl(
        self,
        request: MessageRequest,
    ) -> AsyncGenerator[DeltaPart, None]:
        await asyncio.sleep(0.01)
        yield {
            "type": "text_delta",
            "index": 0,
            "delta": "done",
            "is_start": True,
            "is_end": False,
        }
        await asyncio.sleep(0.01)
        yield {
            "type": "text_delta",
            "index": 0,
            "delta": "",
            "is_start": False,
            "is_end": True,
        }
        yield {
            "type": "finish",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "context_tokens": 60,
                "cache_read_tokens": 20,
                "output_tokens": 5,
            },
        }


class BeforeModelMessagePlugin(HawiPlugin):
    def __init__(self) -> None:
        self.added_at: float | None = None

    @before_model_call
    def add_plugin_message(self, agent: HawiAgent, model: Model, ctx: Any) -> None:
        self.added_at = time.time()
        agent.context.add_user_message("plugin supplied input")


class ContextProbePlugin(HawiPlugin):
    @tool(
        name="context_probe",
        description="Inspect context usage before returning",
        parameters_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    )
    def context_probe(self) -> str:
        return "ok"


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
        "assistant",
        "tool",
        "assistant",
    ]
    tool_call = context.messages[1]["content"][0]
    tool_result = context.messages[2]["content"][0]
    assert tool_call["type"] == "tool_call"
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_call_id"] == "call-1"
    assert [message["role"] for message in record.replaced_messages] == [
        "user",
        "assistant",
        "user",
    ]


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


def test_auto_compact_default_trigger_reserves_compression_budget() -> None:
    large_window = AutoCompactConfig(max_context_tokens=128_000)
    small_window = AutoCompactConfig(max_context_tokens=32_000)
    huge_window = AutoCompactConfig(max_context_tokens=1_000_000)
    explicit = AutoCompactConfig(max_context_tokens=128_000, trigger_tokens=50_000)

    assert large_window.compression_budget == 20_000
    assert large_window.summary_max_output_tokens == 1024
    assert large_window.summary_max_chars == 4_000
    assert large_window.max_transcript_chars == 12_000
    assert large_window.token_limit() == 108_000
    assert small_window.token_limit() == 25_600
    assert huge_window.token_limit() == 950_000
    assert explicit.token_limit() == 50_000


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
    assert metadata.context_tokens == 65
    assert metadata.context_ratio == 0.65
    assert metadata.context_source == "provider_usage"
    assert agent.context.context_usage_snapshot() == ContextUsageSnapshot(
        used_tokens=65,
        max_context_tokens=100,
        usage_ratio=0.65,
        remaining_tokens=35,
        source="provider_usage",
    )


def test_model_metadata_includes_ttft_and_speed_estimates() -> None:
    events: list[Event] = []
    bus = EventBus()
    bus.subscribe_blocking(events.append, event_types=["model.metadata"])
    agent = HawiAgent(model=TimingStreamingModel(), event_bus=bus, streaming=True)

    try:
        agent.run("hi")
    finally:
        bus.close()

    metadata = cast(Any, events[-1])
    assert metadata.started_at is not None
    assert metadata.first_token_at is not None
    assert metadata.completed_at is not None
    assert metadata.started_at <= metadata.first_token_at <= metadata.completed_at
    assert metadata.ttft_ms > 0
    assert metadata.decode_ms > 0
    assert metadata.prefill_tokens == 40
    assert metadata.decode_tokens == 5
    assert metadata.context_tokens == 65
    assert metadata.prefill_tokens_per_second > 0
    assert metadata.decode_tokens_per_second > 0


@pytest.mark.asyncio
async def test_context_usage_snapshot_refreshes_before_tool_execution() -> None:
    model = ToolCallingContextModel()
    bus = EventBus()
    agent = HawiAgent(
        model=model,
        plugins=[ContextProbePlugin()],
        event_bus=bus,
        streaming=True,
        max_iterations=3,
    )
    snapshots: list[tuple[ContextUsageSnapshot | None, int]] = []

    def on_tool_call(_event: Event) -> None:
        snapshots.append(
            (
                agent.context.context_usage_snapshot(),
                agent.context.estimate_tokens(),
            )
        )

    bus.subscribe_blocking(on_tool_call, event_types=["agent.tool_call"])
    try:
        result = await agent.arun("use the tool")
    finally:
        bus.close()

    assert result.text == "done"
    assert snapshots
    snapshot, estimated_tokens = snapshots[0]
    assert snapshot is not None
    assert snapshot.used_tokens >= estimated_tokens


def test_ttft_timer_uses_plugin_added_message_start() -> None:
    events: list[Event] = []
    bus = EventBus()
    bus.subscribe_blocking(events.append, event_types=["model.metadata"])
    plugin = BeforeModelMessagePlugin()
    agent = HawiAgent(
        model=TimingStreamingModel(),
        plugins=[plugin],
        event_bus=bus,
        streaming=True,
    )

    try:
        agent.run(None)
    finally:
        bus.close()

    metadata = cast(Any, events[-1])
    assert plugin.added_at is not None
    assert metadata.started_at >= plugin.added_at
    assert metadata.ttft_ms > 0


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
async def test_compaction_summary_only_includes_replaced_prefix() -> None:
    model = CompactingModel()
    agent = HawiAgent(
        model=model,
        streaming=False,
        auto_compact=AutoCompactConfig(
            enabled=True,
            trigger_tokens=20,
            keep_last_messages=1,
            min_messages=3,
        ),
    )
    for idx in range(4):
        agent.context.add_user_message(f"old message {idx} " + ("x" * 80))
        agent.context.add_assistant_message([{"type": "text", "text": "old answer"}])

    result = await agent.arun("live prompt marker qwerty")

    assert result.text == "done"
    summary_request = model.requests[0][0]
    summary_text = cast(dict[str, Any], summary_request["content"][0])["text"]
    assert "old message 0" in str(summary_text)
    assert "live prompt marker qwerty" not in str(summary_text)


@pytest.mark.asyncio
async def test_compaction_stop_is_emitted_when_summary_is_cancelled() -> None:
    model = CancelDuringCompactionModel()
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
            keep_last_messages=1,
            min_messages=3,
        ),
    )
    for idx in range(4):
        agent.context.add_user_message(f"old message {idx} " + ("x" * 80))
        agent.context.add_assistant_message([{"type": "text", "text": "old answer"}])

    try:
        with pytest.raises(asyncio.CancelledError):
            await agent.arun("finish now")
    finally:
        bus.close()

    assert [event.type for event in events] == [
        "agent.compact_start",
        "agent.compact_stop",
    ]
    stop = events[1]
    assert getattr(stop, "status") == "error"
    assert getattr(stop, "error") == "cancelled"


@pytest.mark.asyncio
async def test_compaction_summary_is_hard_capped() -> None:
    model = VerboseCompactionModel()
    agent = HawiAgent(
        model=model,
        streaming=False,
        auto_compact=AutoCompactConfig(
            enabled=True,
            trigger_tokens=20,
            keep_last_messages=1,
            summary_max_chars=2_000,
        ),
    )
    for idx in range(4):
        agent.context.add_user_message(f"old message {idx} " + ("x" * 80))
        agent.context.add_assistant_message([{"type": "text", "text": "old answer"}])

    result = await agent.arun("finish now")

    assert result.text == "done"
    assert agent.context.compaction_records
    summary = agent.context.compaction_records[0].summary
    assert len(summary) <= 2_000
    assert "Compaction summary truncated" in summary


@pytest.mark.asyncio
async def test_compaction_transcript_abbreviates_large_tool_results() -> None:
    model = CompactingModel()
    agent = HawiAgent(model=model, streaming=False)
    agent.context.add_user_message("please inspect a file")
    agent.context.add_assistant_message([
        {"type": "tool_call", "id": "call-big", "name": "read_file", "arguments": {}}
    ])
    agent.context.add_tool_result("call-big", "line\n" + ("x" * 20_000))

    await agent._generate_compaction_summary(
        model,
        prompt=agent._auto_compact.prompt,
        compression_budget=20_000,
        max_output_tokens=1024,
        max_summary_chars=4_000,
        max_transcript_chars=12_000,
    )

    summary_request = model.requests[0][0]
    summary_text = str(cast(dict[str, Any], summary_request["content"][0])["text"])
    assert len(summary_text) < 8_000
    assert "tool result truncated for compaction" in summary_text
    assert "x" * 10_000 not in summary_text


@pytest.mark.asyncio
async def test_agent_auto_compact_token_pressure_overrides_min_messages() -> None:
    model = CompactingModel()
    agent = HawiAgent(
        model=model,
        streaming=False,
        auto_compact=AutoCompactConfig(
            enabled=True,
            trigger_tokens=20,
            keep_last_messages=1,
            min_messages=12,
        ),
    )
    for idx in range(2):
        agent.context.add_user_message(f"old message {idx} " + ("x" * 80))
        agent.context.add_assistant_message([{"type": "text", "text": "old answer"}])
    assert len(agent.context.messages) < 12

    result = await agent.arun("finish now")

    assert result.text == "done"
    assert len(agent.context.compaction_records) == 1
    assert len(model.requests) == 2


@pytest.mark.asyncio
async def test_agent_auto_compacts_after_final_answer_crosses_threshold() -> None:
    model = FinalOverflowModel()
    agent = HawiAgent(
        model=model,
        streaming=False,
        auto_compact=AutoCompactConfig(
            enabled=True,
            trigger_tokens=100,
            keep_last_messages=2,
            min_messages=12,
        ),
    )
    agent.context.add_user_message("old request")
    agent.context.add_assistant_message([{"type": "text", "text": "old answer"}])
    before_tokens = agent.context.estimate_tokens()
    assert before_tokens < 100

    result = await agent.arun("finish now")

    assert "final" in result.text
    assert len(agent.context.compaction_records) >= 1
    assert len(model.requests) >= 2
    metadata = cast(dict[str, Any], agent.context.messages[0].get("metadata") or {})
    assert metadata["source"] == "context_compaction"


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


@pytest.mark.asyncio
async def test_explicit_auto_compact_config_is_clamped_to_model_context_window() -> None:
    model = CompactingModel()
    model.configure_max_context_tokens(100)
    agent = HawiAgent(
        model=model,
        streaming=False,
        auto_compact=AutoCompactConfig(
            enabled=True,
            keep_last_messages=2,
        ),
    )
    assert agent._auto_compact.max_context_tokens == 100
    for idx in range(6):
        agent.context.add_user_message(f"old message {idx} " + ("x" * 80))
        agent.context.add_assistant_message([{"type": "text", "text": "old answer"}])

    result = await agent.arun("finish now")

    assert result.text == "done"
    assert len(agent.context.compaction_records) == 1
    assert len(model.requests) == 2


@pytest.mark.asyncio
async def test_context_length_error_forces_compaction_before_retry() -> None:
    model = ContextLengthThenCompactModel()
    agent = HawiAgent(
        model=model,
        streaming=False,
        auto_compact=AutoCompactConfig(
            enabled=True,
            max_context_tokens=128_000,
            keep_last_messages=2,
        ),
    )
    for idx in range(6):
        agent.context.add_user_message(f"old message {idx}")
        agent.context.add_assistant_message([{"type": "text", "text": "old answer"}])

    result = await agent.arun("finish now")

    assert result.text == "done"
    assert model.real_calls == 2
    assert model.get_max_context_tokens() == 100
    assert agent._auto_compact.max_context_tokens == 100
    assert len(agent.context.compaction_records) == 1
    assert len(model.requests) == 3


@pytest.mark.asyncio
async def test_agent_auto_compact_uses_saved_provider_context_usage() -> None:
    model = CompactingModel()
    agent = HawiAgent(
        model=model,
        streaming=False,
        auto_compact=AutoCompactConfig(
            enabled=True,
            max_context_tokens=2_000,
            trigger_tokens=1_000,
            compression_budget=12_345,
            keep_last_messages=2,
            min_messages=3,
        ),
    )
    for idx in range(4):
        agent.context.add_user_message(f"old message {idx}")
        agent.context.add_assistant_message([{"type": "text", "text": "old answer"}])
    assert agent.context.estimate_tokens() < 1_000
    agent.context.set_context_usage(
        ContextUsageSnapshot(
            used_tokens=1_500,
            max_context_tokens=2_000,
            usage_ratio=0.75,
            remaining_tokens=500,
            source="provider_usage",
        )
    )

    result = await agent.arun("finish now")

    assert result.text == "done"
    assert len(agent.context.compaction_records) >= 1
    summary_system = cast(list[dict[str, Any]], model.system_prompts[0])
    summary_prompt = str(summary_system[0]["text"])
    assert "compression budget is 12,345 tokens" in summary_prompt
    assert "Preserve the original language of the conversation" in summary_prompt
