"""Hawi Agent - Core agent implementation with LLM API support.

This package provides:
- HawiAgent: Core agent with tool execution and plugin support
- AgentContext: Conversation state management
- Model: Abstract base class for LLM providers
- Events: Streaming event system
- Result: Execution result types
"""

from hawi.errors import (
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
from .hawi_agent import HawiAgent
from .context import AgentContext
from .printers import (
    PlainPrinter,
    RichPrinter,
)
from .result import AgentRunResult, ToolCallRecord

__all__ = [
    # Core
    "HawiAgent",
    "AgentContext",
    "PlainPrinter",
    "RichPrinter",
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
