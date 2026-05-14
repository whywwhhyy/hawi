
"""Hawi Agent module."""

from .agent import HawiAgent
from .config import (
    AutoCompactConfig,
    ModelErrorPolicy,
    ModelErrorRetryPolicy,
    ModelErrorNotifyPolicy,
    ModelErrorStopPolicy,
)
from .state import SteerPartMergeMode
from .context import (
    AgentContext,
    ContextCompactionRecord,
    ContextUsageSnapshot,
    ToolCallContext,
)
from .result import AgentRunResult, ToolCallRecord
from .tool_executor import ToolExecutionBatchResult, ToolExecutor
from .subagent import (
    SubAgentError,
    SubAgentHandle,
    SubAgentLimits,
    SubAgentLifecycleState,
    SubAgentManager,
    SubAgentPluginPolicy,
    SubAgentSpec,
    SubAgentStatus,
    SubAgentTimeoutAction,
)

# Re-export runner classes
from .runner import (
    AgentErrorHook,
    AgentRunner,
    AgentRunnerError,
    AgentRunnerErrorHook,
    QueueType,
    QueuedMessage,
    MessageQueueManager,
    EventMode,
    EventInterceptor,
    AgentExecutor,
    AgentRunnerState,
    ErrorAction,
    ModelErrorHook,
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
    "ToolExecutor",
    "ToolExecutionBatchResult",
    # Subagents
    "SubAgentError",
    "SubAgentHandle",
    "SubAgentLimits",
    "SubAgentLifecycleState",
    "SubAgentManager",
    "SubAgentPluginPolicy",
    "SubAgentSpec",
    "SubAgentStatus",
    "SubAgentTimeoutAction",
    # AgentRunner
    "AgentErrorHook",
    "AgentRunner",
    "AgentRunnerError",
    "AgentRunnerErrorHook",
    "QueueType",
    "QueuedMessage",
    "MessageQueueManager",
    "EventMode",
    "EventInterceptor",
    "AgentExecutor",
    "AgentRunnerState",
    "ErrorAction",
    "ModelErrorHook",
]
