
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
from .context import (
    AgentContext,
    ContextCompactionRecord,
    ContextUsageSnapshot,
    ToolCallContext,
)
from .result import AgentRunResult, ToolCallRecord
from .subagent import (
    SubAgentError,
    SubAgentHandle,
    SubAgentLimits,
    SubAgentLifecycleState,
    SubAgentManager,
    SubAgentPluginPolicy,
    SubAgentSpec,
    SubAgentStatus,
)

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
    "ContextUsageSnapshot",
    "ToolCallContext",
    # Result
    "AgentRunResult",
    "ToolCallRecord",
    # Subagents
    "SubAgentError",
    "SubAgentHandle",
    "SubAgentLimits",
    "SubAgentLifecycleState",
    "SubAgentManager",
    "SubAgentPluginPolicy",
    "SubAgentSpec",
    "SubAgentStatus",
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
