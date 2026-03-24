"""Hawi Scheduler - Message queue management and agent orchestration.

This module provides the HawiScheduler for managing agent execution
with support for complex message processing, prioritization, and interruption.
"""

from __future__ import annotations

from .queue import QueueType, QueuedMessage, MessageQueueManager
from .interceptor import EventMode, EventInterceptor
from .executor import AgentExecutor, SchedulerState, ErrorAction
from .scheduler import HawiScheduler, SchedulerError

__all__ = [
    # Queue
    "QueueType",
    "QueuedMessage",
    "MessageQueueManager",
    # Interceptor
    "EventMode",
    "EventInterceptor",
    # Executor
    "AgentExecutor",
    "SchedulerState",
    "ErrorAction",
    # Scheduler
    "HawiScheduler",
    "SchedulerError",
]
