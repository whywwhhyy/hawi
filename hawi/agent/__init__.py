"""Hawi Agent - Core agent implementation with LLM API support.

This package provides:
- HawiAgent: Core agent with tool execution and plugin support
- AgentContext: Conversation state management
- Model: Abstract base class for LLM providers
- Events: Streaming event system
- Result: Execution result types
"""

from .hawi_agent import HawiAgent
from .context import AgentContext
# from ..events import (
#     Event,
#     EventBus,
#     # Model events
#     ModelStreamStartEvent,
#     ModelStreamStopEvent,
#     ModelContentBlockStartEvent,
#     ModelContentBlockDeltaEvent,
#     ModelContentBlockStopEvent,
#     ModelMetadataEvent,
#     ModelErrorEvent,
#     # Agent events
#     AgentRunStartEvent,
#     AgentRunStopEvent,
#     AgentToolCallEvent,
#     AgentToolResultEvent,
#     AgentMessageAddedEvent,
#     AgentErrorEvent,
# )
from .printers import (
    PlainPrinter,
    RichPrinter,
    BlockPrinter,
    StreamMarkdownPrinter,
)
from ..errors import (
    HawiError,
    AgentError,
    MaxIterationsError,
    ModelError,
    NetworkError,
    ThrottleError,
    DeniedError,
    UnknownModelError,
    ToolNotFoundError,
    ToolValidationError,
    ToolExecutionError,
    ConfigurationError,
)
from ..models.message import DeltaPart
from .result import AgentRunResult, ToolCallRecord

__all__ = [
    # Core
    "HawiAgent",
    "AgentContext",
    # # Events
    # "Event",
    # "EventBus",
    "PlainPrinter",
    "RichPrinter",
    "BlockPrinter",
    "StreamMarkdownPrinter",
    # # Model events
    # "ModelStreamStartEvent",
    # "ModelStreamStopEvent",
    # "ModelContentBlockStartEvent",
    # "ModelContentBlockDeltaEvent",
    # "ModelContentBlockStopEvent",
    # "ModelMetadataEvent",
    # "ModelErrorEvent",
    # # Agent events
    # "AgentRunStartEvent",
    # "AgentRunStopEvent",
    # "AgentToolCallEvent",
    # "AgentToolResultEvent",
    # "AgentMessageAddedEvent",
    # "AgentErrorEvent",
    # Results
    "AgentRunResult",
    "ToolCallRecord",
    # Errors
    "HawiError",
    "AgentError",
    "MaxIterationsError",
    "ModelError",
    "NetworkError",
    "ThrottleError",
    "DeniedError",
    "UnknownModelError",
    "ToolNotFoundError",
    "ToolValidationError",
    "ToolExecutionError",
    "ConfigurationError",
]
