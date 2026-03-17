"""Infrastructure utilities for agent framework."""

from .lifecycle import ExitHandler, exit_scope, register_exit_handler

__all__ = [
    # Lifecycle
    "ExitHandler",
    "exit_scope",
    "register_exit_handler",
]
