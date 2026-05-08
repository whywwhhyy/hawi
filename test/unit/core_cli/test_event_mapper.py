from __future__ import annotations

from typing import cast

from hawi.events import (
    AgentMessageAddedEvent,
    AgentRunStartEvent,
    AgentRunStopEvent,
    AgentToolCallEvent,
    AgentToolResultEvent,
    ModelContentBlockDeltaEvent,
    ModelMetadataEvent,
    ModelToolCallBlockDeltaEvent,
    ModelToolCallBlockStartEvent,
    ModelToolCallBlockStopEvent,
    PluginEvent,
    SchedulerDequeueEvent,
    SchedulerEnqueueEvent,
    SchedulerInterruptEvent,
)
from hawi.models.message import DeltaPart, TokenUsage
from hawi.tool.types import ToolResult
from hawi_core_cli.event_mapper import SemanticEventMapper


def test_mapper_only_logs_high_priority_message_on_enqueue() -> None:
    mapper = SemanticEventMapper()

    frames = mapper.map(
        SchedulerEnqueueEvent.create("steer-1", "high_prio", "new priority")
    )

    assert [frame["type"] for frame in frames] == ["debug.info"]
    assert frames[0]["payload"]["message_id"] == "steer-1"


def test_mapper_does_not_display_normal_message_on_enqueue() -> None:
    mapper = SemanticEventMapper()

    frames = mapper.map(
        SchedulerEnqueueEvent.create("msg-1", "normal", "queued normal")
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
        )
    )

    assert frames[0]["type"] == "run.start"
    assert frames[0]["payload"]["run_id"] == "run-1"
    assert frames[0]["payload"]["message_id"] == "msg-1"
    assert frames[0]["payload"]["queue"] == "urgent"
    assert frames[0]["payload"]["display_message_type"] == "urgent"
    assert frames[0]["payload"]["user_content"] == "hello"


def test_mapper_falls_back_to_run_queue_without_message_metadata() -> None:
    mapper = SemanticEventMapper()

    mapper.map(SchedulerDequeueEvent.create("msg-plain", "high_prio"))
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

    mapper.map(SchedulerDequeueEvent.create("msg-plain", "high_prio"))
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

    mapper.map(SchedulerDequeueEvent.create("msg-2", "urgent"))
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

    mapper.map(SchedulerDequeueEvent.create("pending-inputs", "high_prio"))
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


def test_mapper_extracts_tool_call_description_from_arguments() -> None:
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
                "tool_call_description": "Read the current design notes.",
            },
        )
    )

    assert stop[0]["payload"]["tool_call_description"] == "Read the current design notes."
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
    assert result[0]["payload"]["tool_call_description"] == "Read the current design notes."


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


def test_mapper_reconciles_pending_tool_call_id() -> None:
    mapper = SemanticEventMapper()
    mapper.map(AgentRunStartEvent.create("run-pending-tool"))

    start = mapper.map(
        ModelToolCallBlockStartEvent.create("req-pending", 0, "", "WebPlugin__fetch")
    )
    display_id = start[0]["payload"]["tool_call_id"]
    assert display_id.startswith("pending:req-pending:0")
    assert start[0]["payload"]["actual_tool_call_id"] == ""
    assert start[0]["payload"]["status"] == "pending"

    delta = mapper.map(
        ModelToolCallBlockDeltaEvent.create(
            "req-pending",
            0,
            "",
            '{"url":"https://this-domain-does-not-exist-123456789.com"}',
        )
    )
    assert delta[0]["payload"]["tool_call_id"] == display_id

    stop = mapper.map(
        ModelToolCallBlockStopEvent.create(
            "req-pending",
            0,
            "tc-real",
            "WebPlugin__fetch",
            {"url": "https://this-domain-does-not-exist-123456789.com"},
        )
    )
    assert stop[0]["payload"]["tool_call_id"] == display_id
    assert stop[0]["payload"]["actual_tool_call_id"] == "tc-real"

    running = mapper.map(
        AgentToolCallEvent.create(
            "run-pending-tool",
            "WebPlugin__fetch",
            {"url": "https://this-domain-does-not-exist-123456789.com"},
            "tc-real",
        )
    )
    assert running[0]["payload"]["tool_call_id"] == display_id
    assert running[0]["payload"]["status"] == "running"
    result = mapper.map(
        AgentToolResultEvent.create(
            "run-pending-tool",
            "tc-real",
            False,
            "",
            3.0,
            ToolResult(success=False, error="抓取失败: DNS lookup failed"),
        )
    )

    assert result[0]["type"] == "tool.result"
    assert result[0]["payload"]["tool_call_id"] == display_id
    assert result[0]["payload"]["actual_tool_call_id"] == "tc-real"
    assert result[0]["payload"]["tool_name"] == "WebPlugin__fetch"
    assert result[0]["payload"]["success"] is False
    assert result[0]["payload"]["error"] == "抓取失败: DNS lookup failed"


def test_mapper_emits_model_metadata_and_scheduler_interrupt() -> None:
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

    interrupted = mapper.map(SchedulerInterruptEvent.create("user", ["tc-9"]))
    assert interrupted[0]["type"] == "scheduler.interrupt"
    assert interrupted[0]["payload"]["interrupted_tool_calls"] == ["tc-9"]


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
