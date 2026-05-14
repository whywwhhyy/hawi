"""Hawi AgentRunner - Message queue management and agent orchestration.

This module provides the AgentRunner for managing agent execution
with support for complex message processing, prioritization, and interruption.
"""

from __future__ import annotations

from .queue import QueueType, QueuedMessage, MessageQueueManager
from .interceptor import EventMode, EventInterceptor
from .executor import AgentExecutor, AgentRunnerState, ErrorAction
from .runner import (
    AgentErrorHook,
    AgentRunner,
    AgentRunnerError,
    AgentRunnerErrorHook,
    ModelErrorHook,
)

__all__ = [
    # Queue
    "QueueType",
    "QueuedMessage",
    "MessageQueueManager",
    # Interceptor
    "EventMode",
    "EventInterceptor",
    # Executor
    "AgentExecutor",
    "AgentRunnerState",
    "ErrorAction",
    # AgentRunner
    "AgentErrorHook",
    "AgentRunner",
    "AgentRunnerError",
    "AgentRunnerErrorHook",
    "ModelErrorHook",
]
