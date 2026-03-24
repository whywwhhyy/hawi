
"""
Hawi Event System

统一事件系统：
- Event: 只读、非阻塞，由 Model 和 Agent 产生
- Hook: 阻塞、可修改，仅由 Agent 产生

命名规范：
- Model*Event: 由 Model 产生的事件
- Agent*Event: 由 Agent 产生的事件
- Scheduler*Event: 由 Scheduler 产生的事件
"""

from .event import (
    Event,
    EventSource,
    EventType,
    ModelEventType,
    AgentEventType,
    SchedulerEventType,
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
    AgentErrorEvent,
)
from .scheduler_events import (
    SchedulerEnqueueEvent,
    SchedulerDequeueEvent,
    SchedulerInterruptEvent,
    AgentInterruptEvent,
    SchedulerYieldEvent,
    SchedulerResumeEvent,
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
    "SchedulerEventType",
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
    "AgentErrorEvent",
    # Scheduler
    "SchedulerEnqueueEvent",
    "SchedulerDequeueEvent",
    "SchedulerInterruptEvent",
    "AgentInterruptEvent",
    "SchedulerYieldEvent",
    "SchedulerResumeEvent",
    # Dump Manager
    "DumpManager",
]
