"""Hawi Resources - MCP-compatible resource management.

Resources provide contextual data to agents, similar to MCP Resources.
They can be text or binary, static or dynamically generated.
"""

from __future__ import annotations

# Core types
from .resource import HawiResource, ResourceContent

# Import implementation classes for convenience
from .implementations import (
    HawiLiteralResource,
    HawiFileResource,
    HawiDynamicResource,
)

__all__ = [
    # Protocols
    "HawiResource",
    # Content
    "ResourceContent",
    # Implementations
    "HawiLiteralResource",
    "HawiFileResource",
    "HawiDynamicResource",
]
