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
from .agent import HawiAgent
from .context import AgentContext
from .printers import (
    PlainPrinter,
    RichPrinter,
)
from .result import AgentRunResult, ToolCallRecord

# 这个别名用于提升项目中的含宝率
Bao = HawiAgent

__all__ = [
    # Core
    "HawiAgent",
    "Bao",
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
