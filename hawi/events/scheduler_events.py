"""
Scheduler events for Hawi Event System.

Scheduler events are produced by the HawiScheduler for message queue
operations, interruptions, and multi-agent coordination.
"""

from __future__ import annotations

from typing import Any

from .event import Event


class SchedulerEnqueueEvent(Event):
    """Message enqueued to scheduler"""
    message_id: str
    queue_type: str
    content_preview: str

    @classmethod
    def create(
        cls,
        message_id: str,
        queue_type: str,
        content_preview: str,
    ) -> SchedulerEnqueueEvent:
        return cls(
            type="scheduler.enqueue",
            source="scheduler",
            message_id=message_id,
            queue_type=queue_type,
            content_preview=content_preview,
        )


class SchedulerDequeueEvent(Event):
    """Message dequeued from scheduler"""
    message_id: str
    queue_type: str

    @classmethod
    def create(
        cls,
        message_id: str,
        queue_type: str,
    ) -> SchedulerDequeueEvent:
        return cls(
            type="scheduler.dequeue",
            source="scheduler",
            message_id=message_id,
            queue_type=queue_type,
        )


class SchedulerInterruptEvent(Event):
    """Scheduler interrupted agent execution"""
    reason: str
    interrupted_tool_calls: list[str]

    @classmethod
    def create(
        cls,
        reason: str,
        interrupted_tool_calls: list[str],
    ) -> SchedulerInterruptEvent:
        return cls(
            type="scheduler.interrupt",
            source="scheduler",
            reason=reason,
            interrupted_tool_calls=interrupted_tool_calls,
        )


class AgentInterruptEvent(Event):
    """Agent was requested to interrupt"""
    interrupt_type: str  # "user" | "scheduler" | "error"
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


class SchedulerYieldEvent(Event):
    """Scheduler yields control, waiting for external input

    Used for multi-agent coordination.
    """
    scheduler_id: str
    yield_reason: str  # "waiting_message", "waiting_signal", "idle"

    @classmethod
    def create(
        cls,
        scheduler_id: str,
        yield_reason: str,
    ) -> SchedulerYieldEvent:
        return cls(
            type="scheduler.yield",
            source="scheduler",
            scheduler_id=scheduler_id,
            yield_reason=yield_reason,
        )


class SchedulerResumeEvent(Event):
    """External source resumed Scheduler execution

    Used for multi-agent coordination.
    """
    scheduler_id: str
    resume_data: dict[str, Any]

    @classmethod
    def create(
        cls,
        scheduler_id: str,
        resume_data: dict[str, Any] | None = None,
    ) -> SchedulerResumeEvent:
        return cls(
            type="scheduler.resume",
            source="scheduler",
            scheduler_id=scheduler_id,
            resume_data=resume_data or {},
        )
