"""
AgentRunner events for Hawi Event System.

AgentRunner events are produced by the AgentRunner for message queue
operations, interruptions, and multi-agent coordination.
"""

from __future__ import annotations

from typing import Any

from .event import Event


class AgentRunnerEnqueueEvent(Event):
    """Message enqueued to runner"""
    message_id: str
    queue_type: str
    content_preview: str

    @classmethod
    def create(
        cls,
        message_id: str,
        queue_type: str,
        content_preview: str,
    ) -> AgentRunnerEnqueueEvent:
        return cls(
            type="runner.enqueue",
            source="runner",
            message_id=message_id,
            queue_type=queue_type,
            content_preview=content_preview,
        )


class AgentRunnerDequeueEvent(Event):
    """Message dequeued from runner"""
    message_id: str
    queue_type: str

    @classmethod
    def create(
        cls,
        message_id: str,
        queue_type: str,
    ) -> AgentRunnerDequeueEvent:
        return cls(
            type="runner.dequeue",
            source="runner",
            message_id=message_id,
            queue_type=queue_type,
        )


class AgentRunnerInterruptEvent(Event):
    """AgentRunner interrupted agent execution"""
    reason: str
    interrupted_tool_calls: list[str]

    @classmethod
    def create(
        cls,
        reason: str,
        interrupted_tool_calls: list[str],
    ) -> AgentRunnerInterruptEvent:
        return cls(
            type="runner.interrupt",
            source="runner",
            reason=reason,
            interrupted_tool_calls=interrupted_tool_calls,
        )


class AgentInterruptEvent(Event):
    """Agent was requested to interrupt"""
    interrupt_type: str  # "user" | "runner" | "error"
    run_id: str

    @classmethod
    def create(
        cls,
        interrupt_type: str,
        run_id: str,
    ) -> AgentInterruptEvent:
        return cls(
            type="agent.interrupt",
            source="agent",
            interrupt_type=interrupt_type,
            run_id=run_id,
        )


class AgentRunnerYieldEvent(Event):
    """AgentRunner yields control, waiting for external input

    Used for multi-agent coordination.
    """
    runner_id: str
    yield_reason: str  # "waiting_message", "waiting_signal", "idle"

    @classmethod
    def create(
        cls,
        runner_id: str,
        yield_reason: str,
    ) -> AgentRunnerYieldEvent:
        return cls(
            type="runner.yield",
            source="runner",
            runner_id=runner_id,
            yield_reason=yield_reason,
        )


class AgentRunnerResumeEvent(Event):
    """External source resumed AgentRunner execution

    Used for multi-agent coordination.
    """
    runner_id: str
    resume_data: dict[str, Any]

    @classmethod
    def create(
        cls,
        runner_id: str,
        resume_data: dict[str, Any] | None = None,
    ) -> AgentRunnerResumeEvent:
        return cls(
            type="runner.resume",
            source="runner",
            runner_id=runner_id,
            resume_data=resume_data or {},
        )
