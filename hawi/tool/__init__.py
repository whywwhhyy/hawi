"""Tool abstraction and registry for agent framework."""

from .function_tool import tool
from .registry import ToolRegistry
from .types import AgentTool, ToolResult

__all__ = [
    # Core types
    "AgentTool",
    "ToolResult",
    # Function-based tools
    "tool",
    # Registry
    "ToolRegistry",
]
