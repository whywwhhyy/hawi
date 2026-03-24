
"""Hawi Agent module."""

from .agent import HawiAgent, ModelErrorPolicy, ModelErrorRetryPolicy, ModelErrorNotifyPolicy, ModelErrorStopPolicy
from .context import AgentContext, ToolCallContext
from .result import AgentRunResult, ToolCallRecord

# Re-export scheduler classes
from .scheduler import (
    HawiScheduler,
    SchedulerError,
    QueueType,
    QueuedMessage,
    MessageQueueManager,
    EventMode,
    EventInterceptor,
    AgentExecutor,
    SchedulerState,
    ErrorAction,
)

__all__ = [
    # Agent
    "HawiAgent",
    "ModelErrorPolicy",
    "ModelErrorRetryPolicy",
    "ModelErrorNotifyPolicy",
    "ModelErrorStopPolicy",
    # Context
    "AgentContext",
    "ToolCallContext",
    # Result
    "AgentRunResult",
    "ToolCallRecord",
    # Scheduler
    "HawiScheduler",
    "SchedulerError",
    "QueueType",
    "QueuedMessage",
    "MessageQueueManager",
    "EventMode",
    "EventInterceptor",
    "AgentExecutor",
    "SchedulerState",
    "ErrorAction",
]
