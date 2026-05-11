"""Managed sub-agent API."""

from .manager import SubAgentManager
from .prompts import ROLE_SYSTEM_PROMPTS
from .types import (
    SubAgentError,
    SubAgentHandle,
    SubAgentLifecycleState,
    SubAgentLimits,
    SubAgentMode,
    SubAgentPluginPolicy,
    SubAgentQueue,
    SubAgentResultContract,
    SubAgentRole,
    SubAgentSpec,
    SubAgentStatus,
)

__all__ = [
    "ROLE_SYSTEM_PROMPTS",
    "SubAgentError",
    "SubAgentHandle",
    "SubAgentLifecycleState",
    "SubAgentLimits",
    "SubAgentManager",
    "SubAgentMode",
    "SubAgentPluginPolicy",
    "SubAgentQueue",
    "SubAgentResultContract",
    "SubAgentRole",
    "SubAgentSpec",
    "SubAgentStatus",
]
