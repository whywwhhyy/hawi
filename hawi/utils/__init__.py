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
]
