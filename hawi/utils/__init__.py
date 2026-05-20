"""Infrastructure utilities for agent framework."""

from .lifecycle import ExitHandler, exit_scope, register_exit_handler
from .loader import (
    ModuleLoader,
    has_subclass,
    has_function,
    has_attribute,
    extract_subclass,
    extract_all_subclasses,
    extract_function,
)
from .markdown_streaming_parser import (
    BlockType,
    RenderEvent,
    BlockUpdate,
    BlockCommit,
    MarkdownStreamingParser,
)
from .workspace import find_git_root

__all__ = [
    # Lifecycle
    "ExitHandler",
    "exit_scope",
    "register_exit_handler",
    # Loader
    "ModuleLoader",
    "has_subclass",
    "has_function",
    "has_attribute",
    "extract_subclass",
    "extract_all_subclasses",
    "extract_function",
    # Markdown streaming parser
    "BlockType",
    "RenderEvent",
    "BlockUpdate",
    "BlockCommit",
    "MarkdownStreamingParser",
    # Workspace
    "find_git_root",
]
