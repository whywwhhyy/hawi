from __future__ import annotations

from typing import Any, cast

from hawi.events import (
    AgentCompactStartEvent,
    AgentCompactStopEvent,
    AgentContextInjectedEvent,
    AgentMessageAddedEvent,
    AgentRunStartEvent,
    AgentRunStopEvent,
    AgentSystemPromptEvent,
    AgentToolCallEvent,
    AgentToolResultEvent,
    AgentToolRuntimeContextInjectedEvent,
    ModelContentBlockDeltaEvent,
    ModelErrorEvent,
    ModelMetadataEvent,
    ModelProfileEvent,
    ModelStreamStartEvent,
    ModelStreamStopEvent,
    ModelToolCallBlockDeltaEvent,
    ModelToolCallBlockStartEvent,
    ModelToolCallBlockStopEvent,
    PluginEvent,
    SubAgentEvent,
    AgentRunnerDequeueEvent,
    AgentRunnerEnqueueEvent,
    AgentRunnerInterruptEvent,
)
from hawi.errors import ModelError
from hawi.models.message import DeltaPart, TokenUsage
from hawi.tool.types import ToolResult
from hawi.engine.event_mapper import SemanticEventMapper


def with_timestamp(event, timestamp: float):
    return event.model_copy(update={"timestamp": timestamp})


def test_mapper_only_logs_high_priority_message_on_enqueue() -> None:
    mapper = SemanticEventMapper()

    frames = mapper.map(
        AgentRunnerEnqueueEvent.create("steer-1", "high_prio", "new priority")
    )

    assert [frame["type"] for frame in frames] == ["debug.info"]
    assert frames[0]["payload"]["message_id"] == "steer-1"


def test_mapper_does_not_display_normal_message_on_enqueue() -> None:
    mapper = SemanticEventMapper()

    frames = mapper.map(
        AgentRunnerEnqueueEvent.create("msg-1", "normal", "queued normal")
    )

    assert [frame["type"] for frame in frames] == ["debug.info"]


def test_mapper_emits_run_start_with_queue_kind() -> None:
    mapper = SemanticEventMapper()

    mapper.map(AgentRunStartEvent.create("run-1"))
    frames = mapper.map(
        AgentMessageAddedEvent.create(
            "run-1",
            "user",
            [{"type": "text", "text": "hello"}],
            metadata={
                "message_id": "msg-1",
                "queue": "urgent",
                "display_message_type": "urgent",
            },
            context_message_id="ctxmsg_user_1",
        )
    )

    assert frames[0]["type"] == "run.start"
    assert frames[0]["payload"]["run_id"] == "run-1"
    assert frames[0]["payload"]["message_id"] == "msg-1"
    assert frames[0]["payload"]["queue"] == "urgent"
    assert frames[0]["payload"]["display_message_type"] == "urgent"
    assert frames[0]["payload"]["user_content"] == "hello"
    assert frames[0]["payload"]["context_message_id"] == "ctxmsg_user_1"
    assert frames[0]["payload"]["content"] == [{"type": "text", "text": "hello"}]


def test_mapper_emits_run_start_for_media_only_user_message() -> None:
    mapper = SemanticEventMapper()
    image_part = {
        "type": "image",
        "source": {
            "kind": "blob",
            "blob_id": "a" * 64,
            "uri": "hawi-blob://" + "a" * 64,
            "mime_type": "image/png",
            "filename": "screen.png",
        },
    }

    mapper.map(AgentRunStartEvent.create("run-media"))
    frames = mapper.map(
        AgentMessageAddedEvent.create(
            "run-media",
            "user",
            [cast(Any, image_part)],
            metadata={"message_id": "msg-media", "queue": "high_prio"},
            context_message_id="ctxmsg_media_1",
        )
    )

    assert frames[0]["type"] == "run.start"
    assert frames[0]["payload"]["message_id"] == "msg-media"
    assert "screen.png" in frames[0]["payload"]["user_content"]
    assert frames[0]["payload"]["content"] == [image_part]


def test_mapper_emits_assistant_commit_context_message_id() -> None:
    mapper = SemanticEventMapper()

    mapper.map(AgentRunStartEvent.create("run-commit"))
    frames = mapper.map(
        AgentMessageAddedEvent.create(
            "run-commit",
            "assistant",
            [{"type": "text", "text": "answer"}],
            context_message_id="ctxmsg_assistant_1",
        )
    )

    assert frames[0]["type"] == "run.message_committed"
    assert frames[0]["payload"]["run_id"] == "run-commit"
    assert frames[0]["payload"]["role"] == "assistant"
    assert frames[0]["payload"]["context_message_id"] == "ctxmsg_assistant_1"


def test_mapper_emits_ttft_debug_before_first_model_delta() -> None:
    mapper = SemanticEventMapper()
    mapper.map(AgentRunStartEvent.create("run-ttft"))
    mapper.map(
        with_timestamp(
            AgentMessageAddedEvent.create(
                "run-ttft",
                "user",
                [{"type": "text", "text": "hello"}],
            ),
            100.0,
        )
    )
    mapper.map(
        with_timestamp(
            ModelStreamStartEvent.create("req-ttft"),
            100.1,
        )
    )

    frames = mapper.map(
        with_timestamp(
            ModelContentBlockDeltaEvent.create(
                "req-ttft",
                {
                    "type": "text_delta",
                    "index": 0,
                    "delta": "hi",
                    "is_start": False,
                    "is_end": False,
                },
            ),
            100.25,
        )
    )

    assert [frame["type"] for frame in frames] == ["debug.info", "run.text_delta"]
    assert frames[0]["payload"]["message"] == "TTFT 250ms"
    assert frames[0]["payload"]["elapsed_ms"] == 250.0
    assert frames[1]["payload"]["delta"] == "hi"


def test_mapper_emits_elapsed_wait_before_model_error() -> None:
    mapper = SemanticEventMapper()
    mapper.map(AgentRunStartEvent.create("run-error"))
    mapper.map(
        with_timestamp(
            AgentMessageAddedEvent.create(
                "run-error",
                "user",
                [{"type": "text", "text": "hello"}],
            ),
            200.0,
        )
    )
    mapper.map(
        with_timestamp(
            ModelStreamStartEvent.create("req-error"),
            200.1,
        )
    )

    frames = mapper.map(
        with_timestamp(
            ModelErrorEvent.create(ModelError("network", "connection failed")),
            201.0,
        )
    )

    assert [frame["type"] for frame in frames] == ["debug.info", "error"]
    assert frames[0]["payload"]["message"] == "TTFT unavailable after 1000ms"
    assert frames[0]["payload"]["elapsed_ms"] == 1000.0
    assert frames[1]["payload"]["code"] == "model_error"


def test_mapper_falls_back_to_run_queue_without_message_metadata() -> None:
    mapper = SemanticEventMapper()

    mapper.map(AgentRunnerDequeueEvent.create("msg-plain", "high_prio"))
    mapper.map(AgentRunStartEvent.create("run-plain"))
    frames = mapper.map(
        AgentMessageAddedEvent.create(
            "run-plain",
            "user",
            [{"type": "text", "text": "plain high-priority message"}],
        )
    )

    assert frames[0]["type"] == "run.start"
    assert frames[0]["payload"]["message_id"] == ""
    assert frames[0]["payload"]["queue"] == "high_prio"
    assert frames[0]["payload"]["display_message_type"] == "normal"
    assert frames[0]["payload"]["user_content"] == "plain high-priority message"


def test_mapper_displays_materialized_high_priority_message_as_normal() -> None:
    mapper = SemanticEventMapper()

    mapper.map(AgentRunnerDequeueEvent.create("msg-plain", "high_prio"))
    frames = mapper.map(
        AgentMessageAddedEvent.create(
            "run-plain",
            "user",
            [{"type": "text", "text": "plain high-priority message"}],
            metadata={
                "message_id": "msg-plain",
                "queue": "normal",
                "display_message_type": "normal",
                "source_queue": "high_prio",
                "materialized_as": "plain_user_message",
            },
        )
    )

    assert frames[0]["type"] == "run.start"
    assert frames[0]["payload"]["message_id"] == "msg-plain"
    assert frames[0]["payload"]["queue"] == "normal"
    assert frames[0]["payload"]["display_message_type"] == "normal"
    assert frames[0]["payload"]["user_content"] == "plain high-priority message"


def test_mapper_displays_materialized_steer_message() -> None:
    mapper = SemanticEventMapper()

    frames = mapper.map(
        AgentMessageAddedEvent.create(
            "run-steer",
            "user",
            [
                {
                    "type": "steer",
                    "content": [{"type": "text", "text": "new steer"}],
                    "tool_call_id": "call-1",
                    "preferred_merge_mode": "append_to_tool_result",
                }
            ],
            metadata={
                "message_id": "steer-1",
                "queue": "high_prio",
                "display_message_type": "steer",
                "source_queue": "high_prio",
                "materialized_as": "steer",
                "tool_call_id": "call-1",
            },
        )
    )

    assert frames[0]["type"] == "run.start"
    assert frames[0]["payload"]["message_id"] == "steer-1"
    assert frames[0]["payload"]["queue"] == "high_prio"
    assert frames[0]["payload"]["display_message_type"] == "steer"
    assert frames[0]["payload"]["user_content"] == "new steer"


def test_mapper_uses_message_metadata_queue_override() -> None:
    mapper = SemanticEventMapper()

    mapper.map(AgentRunnerDequeueEvent.create("msg-2", "urgent"))
    mapper.map(AgentRunStartEvent.create("run-2"))
    frames = mapper.map(
        AgentMessageAddedEvent.create(
            "run-2",
            "user",
            [{"type": "text", "text": "plain drained steer"}],
            metadata={"queue": "normal"},
        )
    )

    assert frames[0]["type"] == "run.start"
    assert frames[0]["payload"]["queue"] == "normal"
    assert frames[0]["payload"]["display_message_type"] == "normal"


def test_mapper_uses_message_metadata_message_id_override() -> None:
    mapper = SemanticEventMapper()

    mapper.map(AgentRunnerDequeueEvent.create("pending-inputs", "high_prio"))
    mapper.map(AgentRunStartEvent.create("run-3"))
    frames = mapper.map(
        AgentMessageAddedEvent.create(
            "run-3",
            "user",
            [{"type": "text", "text": "plain drained steer"}],
            metadata={"queue": "normal", "message_id": "steer-1"},
        )
    )

    assert frames[0]["type"] == "run.start"
    assert frames[0]["payload"]["message_id"] == "steer-1"
    assert frames[0]["payload"]["queue"] == "normal"
    assert frames[0]["payload"]["display_message_type"] == "normal"


def test_mapper_emits_text_delta_and_run_stop() -> None:
    mapper = SemanticEventMapper()
    mapper.map(AgentRunStartEvent.create("run-2"))

    frames = mapper.map(
        ModelContentBlockDeltaEvent.create(
            "req-1",
            cast(
                DeltaPart,
                {
                    "type": "text_delta",
                    "index": 0,
                    "delta": "chunk",
                    "is_start": True,
                    "is_end": False,
                },
            ),
        )
    )

    assert frames[0]["type"] == "run.text_delta"
    assert frames[0]["payload"] == {"run_id": "run-2", "delta": "chunk"}

    stop = mapper.map(AgentRunStopEvent.create("run-2", "end_turn", 12.5))
    assert stop[0]["type"] == "run.stop"
    assert stop[0]["payload"]["duration_ms"] == 12.5


def test_mapper_emits_tool_events_and_result() -> None:
    mapper = SemanticEventMapper()
    mapper.map(AgentRunStartEvent.create("run-3"))

    start = mapper.map(
        ModelToolCallBlockStartEvent.create("req-2", 0, "tc-1", "calc")
    )
    assert start[0]["type"] == "tool.call_start"
    assert start[0]["payload"]["tool_name"] == "calc"
    assert start[0]["payload"]["status"] == "pending"

    running = mapper.map(
        AgentToolCallEvent.create("run-3", "calc", {"expression": "2+2"}, "tc-1")
    )
    assert running[0]["type"] == "tool.call_start"
    assert running[0]["payload"]["tool_call_id"] == "tc-1"
    assert running[0]["payload"]["status"] == "running"
    assert running[0]["payload"]["arguments"] == {"expression": "2+2"}

    result = mapper.map(
        AgentToolResultEvent.create(
            "run-3",
            "tc-1",
            True,
            "4",
            3.0,
            ToolResult(success=True, output={"answer": 4}),
        )
    )
    assert result[0]["type"] == "tool.result"
    assert result[0]["payload"]["tool_name"] == "calc"
    assert result[0]["payload"]["output"] == {"answer": 4}


def test_mapper_forwards_framework_injection_events() -> None:
    mapper = SemanticEventMapper()
    mapper.map(AgentRunStartEvent.create("run-injected"))

    system_prompt = mapper.map(
        AgentSystemPromptEvent.create(
            "run-injected",
            [{"type": "text", "text": "system guidance"}],
            origin="model_input",
            plugin_id="env",
            plugin_name="Environment",
            plugin_role="plugin",
            injection_name="inject_system",
        )
    )
    context = mapper.map(
        AgentContextInjectedEvent.create(
            "run-injected",
            "user",
            [{"type": "text", "text": "environment block"}],
            hook_type="before_conversation",
            position=1,
            metadata={"source": "test_plugin"},
            merge_target="user_message",
            merge_position="after",
            target_message_id="msg-1",
            target_message_index=2,
            plugin_id="env",
            plugin_name="Environment",
            plugin_role="plugin",
            injection_name="inject_user",
        )
    )
    runtime_context = mapper.map(
        AgentToolRuntimeContextInjectedEvent.create(
            "run-injected",
            "stateful_tool",
            "tc-2",
            "ctx",
            plugin_id="stateful",
            plugin_name="Stateful",
            injection_name="ctx",
        )
    )

    assert system_prompt[0]["type"] == "agent.system_prompt"
    assert system_prompt[0]["payload"]["text"] == "system guidance"
    assert system_prompt[0]["payload"]["plugin_id"] == "env"
    assert system_prompt[0]["payload"]["plugin_role"] == "plugin"
    assert system_prompt[0]["payload"]["injection_name"] == "inject_system"
    assert context[0]["type"] == "agent.context_injected"
    assert context[0]["payload"]["text"] == "environment block"
    assert context[0]["payload"]["merge_target"] == "user_message"
    assert context[0]["payload"]["merge_position"] == "after"
    assert context[0]["payload"]["target_message_id"] == "msg-1"
    assert context[0]["payload"]["plugin_id"] == "env"
    assert context[0]["payload"]["injection_name"] == "inject_user"
    assert runtime_context[0]["type"] == "agent.tool_runtime_context_injected"
    assert runtime_context[0]["payload"]["parameter_name"] == "ctx"
    assert runtime_context[0]["payload"]["plugin_id"] == "stateful"


def test_mapper_extracts_tool_call_purpose_from_arguments() -> None:
    mapper = SemanticEventMapper()
    mapper.map(AgentRunStartEvent.create("run-described"))
    mapper.map(ModelToolCallBlockStartEvent.create("req-described", 0, "tc-described", "read"))

    stop = mapper.map(
        ModelToolCallBlockStopEvent.create(
            "req-described",
            0,
            "tc-described",
            "read",
            {
                "path": "notes.md",
                "tool_call_purpose": "Read the current design notes.",
            },
        )
    )

    assert stop[0]["payload"]["tool_call_purpose"] == "Read the current design notes."
    assert stop[0]["payload"]["arguments"] == {"path": "notes.md"}

    result = mapper.map(
        AgentToolResultEvent.create(
            "run-described",
            "tc-described",
            True,
            "ok",
            3.0,
            ToolResult(success=True, output="ok"),
        )
    )
    assert result[0]["payload"]["tool_call_purpose"] == "Read the current design notes."


def test_mapper_marks_full_tool_argument_snapshot() -> None:
    mapper = SemanticEventMapper()
    mapper.map(AgentRunStartEvent.create("run-tool"))
    mapper.map(ModelToolCallBlockStartEvent.create("req-tool", 0, "tc-full", "calc"))

    frames = mapper.map(
        ModelToolCallBlockDeltaEvent.create(
            "req-tool",
            0,
            "tc-full",
            '{"expression":"1+1"}',
            is_streaming=False,
        )
    )

    assert frames[0]["type"] == "tool.call_delta"
    assert frames[0]["payload"]["delta"] == '{"expression":"1+1"}'
    assert frames[0]["payload"]["is_streaming"] is False


def test_mapper_includes_tool_error_output() -> None:
    mapper = SemanticEventMapper()
    mapper.map(AgentRunStartEvent.create("run-fail"))
    mapper.map(ModelToolCallBlockStartEvent.create("req-fail", 0, "tc-fail", "calc"))

    frames = mapper.map(
        AgentToolResultEvent.create(
            "run-fail",
            "tc-fail",
            False,
            "None",
            3.0,
            ToolResult(success=False, error="Parameter validation failed"),
        )
    )

    assert frames[0]["type"] == "tool.result"
    assert frames[0]["payload"]["success"] is False
    assert frames[0]["payload"]["output"] == "Parameter validation failed"
    assert frames[0]["payload"]["error"] == "Parameter validation failed"


def test_mapper_drops_block_start_without_tool_call_id() -> None:
    """Defense in depth: empty-id block_start events are now suppressed at
    the stream accumulator (deferred until id arrives). If one slips through
    to the mapper the right behavior is to drop it rather than forward an
    ambiguous frame to the GUI under a placeholder id."""
    mapper = SemanticEventMapper()
    mapper.map(AgentRunStartEvent.create("run-pending-tool"))

    frames = mapper.map(
        ModelToolCallBlockStartEvent.create("req-pending", 0, "", "WebPlugin__fetch")
    )
    assert frames == []

    # Real flow continues normally once a downstream stop carries the id.
    stop = mapper.map(
        ModelToolCallBlockStopEvent.create(
            "req-pending",
            0,
            "tc-real",
            "WebPlugin__fetch",
            {"url": "https://example.com"},
        )
    )
    assert stop[0]["payload"]["tool_call_id"] == "tc-real"

    running = mapper.map(
        AgentToolCallEvent.create(
            "run-pending-tool",
            "WebPlugin__fetch",
            {"url": "https://example.com"},
            "tc-real",
        )
    )
    assert running[0]["payload"]["tool_call_id"] == "tc-real"
    assert running[0]["payload"]["status"] == "running"

    result = mapper.map(
        AgentToolResultEvent.create(
            "run-pending-tool",
            "tc-real",
            False,
            "",
            3.0,
            ToolResult(success=False, error="DNS failed"),
        )
    )
    assert result[0]["payload"]["tool_call_id"] == "tc-real"
    assert result[0]["payload"]["tool_name"] == "WebPlugin__fetch"
    assert result[0]["payload"]["success"] is False
    assert result[0]["payload"]["interrupted"] is False

    interrupted = mapper.map(
        AgentToolResultEvent.create(
            "run-pending-tool",
            "tc-interrupted",
            False,
            "Tool call interrupted before completion (reason: user).",
            0.0,
            ToolResult(
                success=False,
                error="Tool call interrupted before completion (reason: user).",
            ),
            interrupted=True,
        )
    )
    assert interrupted[0]["payload"]["interrupted"] is True


def test_stream_accumulator_defers_start_until_tool_call_id_known() -> None:
    """The root fix for the pending: workaround: when the underlying provider
    streams a tool_call without id at is_start, the accumulator must defer
    the ModelToolCallBlockStartEvent until a later chunk reveals the id.
    Downstream consumers therefore never see an empty-id block_start."""
    from hawi.agent.stream_accumulator import StreamBlockAccumulator

    acc = StreamBlockAccumulator.create_tool_handler()

    # First chunk: is_start with empty id (mimics OpenAI streaming variants
    # that supply id later).
    [(_, events)] = acc.handle(
        {
            "type": "tool_call_delta",
            "index": 0,
            "id": None,
            "name": "fetch",
            "arguments_delta": "",
            "is_start": True,
            "is_end": False,
        },
        request_id="req-A",
    )
    start_events = cast(
        list[ModelToolCallBlockStartEvent],
        [e for e in events if e.type == "model.tool_call_block_start"],
    )
    assert start_events == [], "StartEvent must be deferred"

    # Subsequent chunk carrying the real id: StartEvent flushes now.
    [(_, events)] = acc.handle(
        {
            "type": "tool_call_delta",
            "index": 0,
            "id": "tc-real",
            "name": None,
            "arguments_delta": '{"q":',
            "is_start": False,
            "is_end": False,
        },
        request_id="req-A",
    )
    start_events = cast(
        list[ModelToolCallBlockStartEvent],
        [e for e in events if e.type == "model.tool_call_block_start"],
    )
    assert len(start_events) == 1
    assert start_events[0].tool_call_id == "tc-real"
    assert start_events[0].tool_name == "fetch"

    # Closing the block does not re-emit a Start.
    [(_, events)] = acc.handle(
        {
            "type": "tool_call_delta",
            "index": 0,
            "id": "tc-real",
            "name": "fetch",
            "arguments_delta": "",
            "is_start": False,
            "is_end": True,
        },
        request_id="req-A",
    )
    start_events = cast(
        list[ModelToolCallBlockStartEvent],
        [e for e in events if e.type == "model.tool_call_block_start"],
    )
    assert start_events == []
    stop_events = cast(
        list[ModelToolCallBlockStopEvent],
        [e for e in events if e.type == "model.tool_call_block_stop"],
    )
    assert len(stop_events) == 1
    assert stop_events[0].tool_call_id == "tc-real"


def test_stream_accumulator_flushes_pending_start_on_block_end() -> None:
    """If the id only arrives in the is_end chunk, StartEvent must be flushed
    before StopEvent so consumers see Start → Stop in order."""
    from hawi.agent.stream_accumulator import StreamBlockAccumulator

    acc = StreamBlockAccumulator.create_tool_handler()
    acc.handle(
        {
            "type": "tool_call_delta",
            "index": 0,
            "id": None,
            "name": "fetch",
            "arguments_delta": "",
            "is_start": True,
            "is_end": False,
        },
        request_id="req-B",
    )
    [(_, events)] = acc.handle(
        {
            "type": "tool_call_delta",
            "index": 0,
            "id": "tc-late",
            "name": "fetch",
            "arguments_delta": "{}",
            "is_start": False,
            "is_end": True,
        },
        request_id="req-B",
    )
    types = [e.type for e in events]
    assert "model.tool_call_block_start" in types
    assert "model.tool_call_block_stop" in types
    assert types.index("model.tool_call_block_start") < types.index(
        "model.tool_call_block_stop"
    )


def test_mapper_emits_model_metadata_and_runner_interrupt() -> None:
    mapper = SemanticEventMapper()
    mapper.map(AgentRunStartEvent.create("run-4"))

    metadata = mapper.map(
        ModelMetadataEvent.create(
            "req-3",
            usage=TokenUsage(
                input_tokens=2,
                output_tokens=5,
                total_tokens=9,
                cache_read_tokens=1,
                reasoning_tokens=2,
            ),
            latency_ms=10.0,
            started_at=100.0,
            first_token_at=100.2,
            completed_at=101.0,
            ttft_ms=200.0,
            decode_ms=800.0,
            prefill_tokens=2,
            prefill_total_tokens=4,
            decode_tokens=5,
            prefill_tokens_per_second=10.0,
            decode_tokens_per_second=6.25,
            context_tokens=100,
            max_context_tokens=1000,
            context_ratio=0.1,
            context_source="provider_usage",
        )
    )
    assert metadata[0]["type"] == "model.metadata"
    assert metadata[0]["payload"]["total_tokens"] == 9
    assert metadata[0]["payload"]["cache_read_tokens"] == 1
    assert metadata[0]["payload"]["reasoning_tokens"] == 2
    assert metadata[0]["payload"]["context_tokens"] == 100
    assert metadata[0]["payload"]["context_ratio"] == 0.1
    assert metadata[0]["payload"]["context_source"] == "provider_usage"
    assert metadata[0]["payload"]["started_at"] == 100.0
    assert metadata[0]["payload"]["first_token_at"] == 100.2
    assert metadata[0]["payload"]["completed_at"] == 101.0
    assert metadata[0]["payload"]["ttft_ms"] == 200.0
    assert metadata[0]["payload"]["decode_ms"] == 800.0
    assert metadata[0]["payload"]["prefill_tokens"] == 2
    assert metadata[0]["payload"]["prefill_total_tokens"] == 4
    assert metadata[0]["payload"]["decode_tokens"] == 5
    assert metadata[0]["payload"]["prefill_tokens_per_second"] == 10.0
    assert metadata[0]["payload"]["decode_tokens_per_second"] == 6.25
    assert len(metadata) == 1

    interrupted = mapper.map(AgentRunnerInterruptEvent.create("user", ["tc-9"]))
    assert [frame["type"] for frame in interrupted] == [
        "runner.interrupt",
        "tool.interrupted",
    ]
    assert interrupted[0]["payload"]["interrupted_tool_calls"] == ["tc-9"]
    assert interrupted[1]["payload"]["tool_call_id"] == "tc-9"
    assert interrupted[1]["payload"]["reason"] == "user"


def test_mapper_emits_model_profile_update() -> None:
    mapper = SemanticEventMapper()
    mapper.map(AgentRunStartEvent.create("run-profile"))

    frames = mapper.map(
        ModelProfileEvent.create(
            "req-profile",
            cache_tokens=8,
            prefill_ms=246.0,
            prefill_tokens=12,
            prefill_total_tokens=48,
            prefill_tokens_per_second=48.8,
            ttft_ms=698.0,
            decode_ms=123.0,
            decode_tokens=5,
            decode_tokens_per_second=40.7,
            peak_decode_tokens_per_second=52.1,
        )
    )

    assert [frame["type"] for frame in frames] == ["model.profile"]
    payload = frames[0]["payload"]
    assert payload["run_id"] == "run-profile"
    assert payload["request_id"] == "req-profile"
    assert payload["cache_tokens"] == 8
    assert payload["prefill_ms"] == 246.0
    assert payload["prefill_tokens"] == 12
    assert payload["prefill_total_tokens"] == 48
    assert payload["prefill_tokens_per_second"] == 48.8
    assert payload["ttft_ms"] == 698.0
    assert payload["decode_ms"] == 123.0
    assert payload["decode_tokens"] == 5
    assert payload["decode_tokens_per_second"] == 40.7
    assert payload["peak_decode_tokens_per_second"] == 52.1


def test_mapper_emits_interrupted_for_active_tool_calls() -> None:
    mapper = SemanticEventMapper()
    mapper.map(AgentRunStartEvent.create("run-interrupt"))
    mapper.map(ModelStreamStartEvent.create("req-interrupt"))
    mapper.map(
        ModelToolCallBlockStartEvent.create(
            "req-interrupt",
            0,
            "tc-pending",
            "search",
        )
    )
    mapper.map(
        AgentToolCallEvent.create(
            "run-interrupt",
            "search",
            {"query": "hawi"},
            "tc-running",
        )
    )

    frames = mapper.map(AgentRunnerInterruptEvent.create("user", ["tc-running"]))

    assert [frame["type"] for frame in frames] == [
        "runner.interrupt",
        "tool.interrupted",
        "tool.interrupted",
        "model.interrupted",
        "debug.info",
    ]
    interrupted_tools = [
        frame["payload"]["tool_call_id"]
        for frame in frames
        if frame["type"] == "tool.interrupted"
    ]
    assert interrupted_tools == ["tc-pending", "tc-running"]
    model_interrupted = [
        frame["payload"]
        for frame in frames
        if frame["type"] == "model.interrupted"
    ]
    assert model_interrupted == [
        {
            "run_id": "run-interrupt",
            "request_id": "req-interrupt",
            "reason": "user",
            "stop_reason": "interrupted",
        }
    ]
    assert frames[-1]["payload"]["event_type"] == "model.stream_stop"
    assert frames[-1]["payload"]["stop_reason"] == "interrupted"

    later_stop = mapper.map(ModelStreamStopEvent.create("req-interrupt", "interrupted"))
    assert [frame["type"] for frame in later_stop] == ["debug.info"]


def test_mapper_emits_model_interrupted_from_stream_stop() -> None:
    mapper = SemanticEventMapper()
    mapper.map(AgentRunStartEvent.create("run-stream-interrupt"))
    mapper.map(ModelStreamStartEvent.create("req-stream-interrupt"))

    frames = mapper.map(
        ModelStreamStopEvent.create("req-stream-interrupt", "interrupted")
    )

    assert [frame["type"] for frame in frames] == [
        "debug.info",
        "model.interrupted",
    ]
    assert frames[1]["payload"] == {
        "run_id": "run-stream-interrupt",
        "request_id": "req-stream-interrupt",
        "reason": "interrupted",
        "stop_reason": "interrupted",
    }


def test_mapper_forwards_agent_compact_events() -> None:
    mapper = SemanticEventMapper()

    start = mapper.map(
        AgentCompactStartEvent.create(
            run_id="run-compact",
            mode="auto",
            keep_last_messages=4,
            tokens_before=1000,
            message_count_before=20,
        )
    )
    stop = mapper.map(
        AgentCompactStopEvent.create(
            run_id="run-compact",
            mode="auto",
            status="success",
            duration_ms=12.5,
            tokens_before=1000,
            tokens_after=250,
            message_count_before=20,
            message_count_after=5,
            replaced_message_count=16,
            kept_message_count=4,
        )
    )

    assert start[0]["type"] == "agent.compact_start"
    assert start[0]["payload"]["tokens_before"] == 1000
    assert stop[0]["type"] == "agent.compact_stop"
    assert stop[0]["payload"]["status"] == "success"
    assert stop[0]["payload"]["tokens_after"] == 250
    assert stop[0]["payload"]["replaced_message_count"] == 16


def test_mapper_forwards_plugin_artifact_and_tool_progress_events() -> None:
    mapper = SemanticEventMapper()
    mapper.map(AgentRunStartEvent.create("run-plugin"))

    artifact = mapper.map(
        PluginEvent.create(
            "plugin.artifact.upsert",
            plugin_id="plan",
            plugin_name="PlanPlugin",
            payload={
                "artifact": {
                    "id": "current",
                    "type": "plan",
                    "title": "Current Plan",
                }
            },
        )
    )

    assert artifact[0]["type"] == "plugin.artifact.upsert"
    assert artifact[0]["payload"]["plugin_id"] == "plan"
    assert artifact[0]["payload"]["run_id"] == "run-plugin"
    assert artifact[0]["payload"]["artifact"]["title"] == "Current Plan"

    progress = mapper.map(
        PluginEvent.create(
            "plugin.tool_progress",
            plugin_id="plan",
            plugin_name="PlanPlugin",
            tool_call_id="tc-plan",
            payload={"progress": 0.8, "message": "Reviewing"},
        )
    )

    assert progress[0]["type"] == "plugin.tool_progress"
    assert progress[0]["payload"]["tool_call_id"] == "tc-plan"
    assert progress[0]["payload"]["progress"] == 0.8


def test_mapper_forwards_subagent_events_on_dedicated_channel() -> None:
    mapper = SemanticEventMapper()
    frames = mapper.map(
        SubAgentEvent.create(
            "subagent.event",
            subagent_id="sub_1",
            subagent_name="worker-1",
            subagent_role="worker",
            status={"id": "sub_1", "state": "RUNNING"},
            child_event={"type": "agent.message_added", "run_id": "run-sub"},
            message_entry={
                "version": 1,
                "run_id": "run-sub",
                "role": "assistant",
                "content": [{"type": "text", "text": "done"}],
            },
        )
    )

    assert frames[0]["type"] == "subagent.event"
    assert frames[0]["payload"]["subagent_id"] == "sub_1"
    assert frames[0]["payload"]["status"]["state"] == "RUNNING"
    assert frames[0]["payload"]["message_entry"]["role"] == "assistant"
