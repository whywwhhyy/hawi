"""Tool abstraction and registry for agent framework."""

from .function_tool import tool
from .registry import ToolRegistry
from .types import (
    AgentTool,
    ToolParameterInjection,
    ToolParameterInjectionContext,
    ToolParameterInjectionHandler,
    ToolParameterInjectionPredicate,
    ToolResult,
)

__all__ = [
    # Core types
    "AgentTool",
    "ToolResult",
    "ToolParameterInjection",
    "ToolParameterInjectionContext",
    "ToolParameterInjectionHandler",
    "ToolParameterInjectionPredicate",
    # Function-based tools
    "tool",
    # Registry
    "ToolRegistry",
]
