"""Tool abstraction and registry for agent framework."""

from .base import Tool, ToolResult
from .registry import ToolRegistry

__all__ = [
    # Base
    "Tool",
    "ToolResult",
    # Registry
    "ToolRegistry",
]
