
"""Hawi Agent module."""

from .agent import (
    HawiAgent,
    ModelErrorPolicy,
    ModelErrorRetryPolicy,
    ModelErrorNotifyPolicy,
    ModelErrorStopPolicy,
    SteerPartMergeMode,
    AutoCompactConfig,
)
from .context import AgentContext, ContextCompactionRecord, ToolCallContext
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
    "SteerPartMergeMode",
    "AutoCompactConfig",
    # Context
    "AgentContext",
    "ContextCompactionRecord",
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
