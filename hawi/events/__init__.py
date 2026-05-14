
"""
Hawi Event System

统一事件系统：
- Event: 只读、非阻塞，由 Model 和 Agent 产生
- Hook: 阻塞、可修改，仅由 Agent 产生

命名规范：
- Model*Event: 由 Model 产生的事件
- Agent*Event: 由 Agent 产生的事件
- AgentRunner*Event: 由 AgentRunner 产生的事件
"""

from .event import (
    Event,
    EventSource,
    EventType,
    ModelEventType,
    AgentEventType,
    AgentRunnerEventType,
    PluginEventType,
    SessionEventType,
)
from .model_events import (
    ModelStreamStartEvent,
    ModelStreamStopEvent,
    ModelContentBlockStartEvent,
    ModelContentBlockDeltaEvent,
    ModelContentBlockStopEvent,
    ModelToolCallBlockStartEvent,
    ModelToolCallBlockDeltaEvent,
    ModelToolCallBlockStopEvent,
    ModelMetadataEvent,
    ModelContentMetadataEvent,
    ModelErrorEvent,
    ModelRetryEvent,
)
from .agent_events import (
    AgentRunStartEvent,
    AgentRunStopEvent,
    AgentToolCallEvent,
    AgentToolResultPartEvent,
    AgentToolResultEvent,
    AgentMessageAddedEvent,
    AgentCompactStartEvent,
    AgentCompactStopEvent,
    AgentErrorEvent,
)
from .runner_events import (
    AgentRunnerEnqueueEvent,
    AgentRunnerDequeueEvent,
    AgentRunnerInterruptEvent,
    AgentInterruptEvent,
    AgentRunnerYieldEvent,
    AgentRunnerResumeEvent,
)
from .plugin_events import (
    PLUGIN_EVENT_TYPES,
    PluginEvent,
)
from .session_events import (
    SessionCheckpointRequestedEvent,
    SessionWriteFailedEvent,
    SessionLoadedEvent,
    SessionSwitchedEvent,
)
from .event_bus import (
    EventBus,
    EventHandler,
    SyncEventHandler,
    AsyncEventHandler,
 )
from .dump_manager import DumpManager

__all__ = [
    # Base
    "Event",
    "EventBus",
    "EventHandler",
    "SyncEventHandler",
    "AsyncEventHandler",
    "EventSource",
    "EventType",
    "ModelEventType",
    "AgentEventType",
    "AgentRunnerEventType",
    "PluginEventType",
    "SessionEventType",
    # Model
    "ModelStreamStartEvent",
    "ModelStreamStopEvent",
    "ModelContentBlockStartEvent",
    "ModelContentBlockDeltaEvent",
    "ModelContentBlockStopEvent",
    "ModelToolCallBlockStartEvent",
    "ModelToolCallBlockDeltaEvent",
    "ModelToolCallBlockStopEvent",
    "ModelMetadataEvent",
    "ModelContentMetadataEvent",
    "ModelErrorEvent",
    "ModelRetryEvent",
    # Agent
    "AgentRunStartEvent",
    "AgentRunStopEvent",
    "AgentToolCallEvent",
    "AgentToolResultPartEvent",
    "AgentToolResultEvent",
    "AgentMessageAddedEvent",
    "AgentCompactStartEvent",
    "AgentCompactStopEvent",
    "AgentErrorEvent",
    # AgentRunner
    "AgentRunnerEnqueueEvent",
    "AgentRunnerDequeueEvent",
    "AgentRunnerInterruptEvent",
    "AgentInterruptEvent",
    "AgentRunnerYieldEvent",
    "AgentRunnerResumeEvent",
    # Plugin
    "PLUGIN_EVENT_TYPES",
    "PluginEvent",
    # Session
    "SessionCheckpointRequestedEvent",
    "SessionWriteFailedEvent",
    "SessionLoadedEvent",
    "SessionSwitchedEvent",
    # Dump Manager
    "DumpManager",
]
