"""Hawi Agent - Core agent implementation with LLM API support.

This package provides:
- HawiAgent: Core agent with tool execution and plugin support
- AgentContext: Conversation state management
- Model: Abstract base class for LLM providers
- Events: Streaming event system
- Result: Execution result types
"""

from .agent import HawiAgent
from .context import AgentContext
from .events import (
    Event,
    EventBus,
    PlainPrinter,
    RichStreamingPrinter,
    create_event_printer,
    # Model events
    model_stream_start_event,
    model_stream_stop_event,
    model_content_block_start_event,
    model_content_block_delta_event,
    model_content_block_stop_event,
    model_metadata_event,
    # Agent events
    agent_run_start_event,
    agent_run_stop_event,
    agent_tool_call_event,
    agent_tool_result_event,
    agent_message_added_event,
    agent_error_event,
)
from .model import Model, ModelErrorType, ModelFailurePolicy, BalanceInfo, StreamEvent
from .result import AgentRunResult, ToolCallRecord

__all__ = [
    # Core
    "HawiAgent",
    "AgentContext",
    "Model",
    # Events
    "Event",
    "EventBus",
    "PlainPrinter",
    "RichStreamingPrinter",
    "create_event_printer",
    # Model events
    "model_stream_start_event",
    "model_stream_stop_event",
    "model_content_block_start_event",
    "model_content_block_delta_event",
    "model_content_block_stop_event",
    "model_metadata_event",
    # Agent events
    "agent_run_start_event",
    "agent_run_stop_event",
    "agent_tool_call_event",
    "agent_tool_result_event",
    "agent_message_added_event",
    "agent_error_event",
    # Results
    "AgentRunResult",
    "ToolCallRecord",
    # Model
    "ModelErrorType",
    "ModelFailurePolicy",
    "BalanceInfo",
    "StreamEvent",
]
