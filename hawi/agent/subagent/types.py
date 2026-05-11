"""Core types for managed sub-agents."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Literal

from hawi.events import Event, EventBus
from hawi.models import ContentPart, Model
from hawi.plugin import HawiPlugin

from ..result import AgentRunResult

if TYPE_CHECKING:
    from ..agent import HawiAgent
    from ..scheduler import HawiScheduler


SubAgentMode = Literal["fork", "fresh"]
SubAgentRole = Literal[
    "general",
    "planner",
    "reviewer",
    "explorer",
    "implementer",
    "critic",
    "summarizer",
]
SubAgentQueue = Literal["normal", "high_prio", "urgent"]
SubAgentTimeoutAction = Literal["status", "interrupt", "close", "raise"]
SubAgentResultContract = Literal[
    "text",
    "json",
    "plan",
    "review",
    "diff",
    "artifact",
]


class SubAgentLifecycleState(str, Enum):
    """Lifecycle state for a managed sub-agent."""

    CREATED = "CREATED"
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    INTERRUPTING = "INTERRUPTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    CLOSED = "CLOSED"


@dataclass
class SubAgentLimits:
    """Safety limits for a sub-agent.

    Only ``max_runtime_seconds`` and ``max_children`` are enforced in the first
    implementation. The rest are explicit API placeholders for the next
    scheduler/tool-call budget pass.
    """

    max_runtime_seconds: float | None = None
    max_tool_calls: int | None = None
    max_iterations: int | None = None
    max_recursion_depth: int = 1
    max_children: int | None = None


@dataclass
class SubAgentPluginPolicy:
    """Plugin inheritance and extension policy for child agents."""

    inherit: bool = True
    extra_plugins: list[HawiPlugin] = field(default_factory=list)
    extra_factories: list[Callable[[], HawiPlugin]] = field(default_factory=list)
    allowlist: list[str] | None = None
    denylist: list[str] | None = None
    tool_allowlist: list[str] | None = None
    tool_denylist: list[str] | None = None


@dataclass
class SubAgentSpec:
    """Configuration for creating a sub-agent."""

    mode: SubAgentMode = "fork"
    name: str | None = None
    role: SubAgentRole | str = "general"
    model: Model | str | None = None
    system_prompt: str | list[ContentPart] | None = None
    plugin_policy: SubAgentPluginPolicy = field(default_factory=SubAgentPluginPolicy)
    working_dir: str | None = None
    initial_prompt: str | list[ContentPart] | None = None
    initial_plan: str | dict[str, Any] | list[Any] | None = None
    limits: SubAgentLimits = field(default_factory=SubAgentLimits)
    result_contract: SubAgentResultContract | str = "text"
    ownership: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    description: str | None = None


@dataclass
class SubAgentStatus:
    """Serializable status snapshot for a sub-agent."""

    id: str
    name: str
    role: str
    state: str
    scheduler_state: str
    executor_state: str
    queue_lengths: dict[str, int]
    created_at: float
    updated_at: float
    closed_at: float | None = None
    model_id: str | None = None
    working_dir: str | None = None
    last_result_text: str | None = None
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe status dict."""
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "state": self.state,
            "scheduler_state": self.scheduler_state,
            "executor_state": self.executor_state,
            "queue_lengths": self.queue_lengths,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "closed_at": self.closed_at,
            "model_id": self.model_id,
            "working_dir": self.working_dir,
            "last_result_text": self.last_result_text,
            "last_error": self.last_error,
        }


@dataclass
class SubAgentHandle:
    """Runtime handle for a managed sub-agent."""

    id: str
    spec: SubAgentSpec
    agent: HawiAgent
    scheduler: HawiScheduler
    scheduler_task: asyncio.Task[None]
    event_bus: EventBus
    state: SubAgentLifecycleState = SubAgentLifecycleState.CREATED
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    closed_at: float | None = None
    last_result: AgentRunResult | None = None
    last_error: str | None = None
    recent_events: list[dict[str, Any]] = field(default_factory=list)
    event_handler: Callable[[Event], Any] | None = None
    monitor_task: asyncio.Task[None] | None = None

    @property
    def name(self) -> str:
        return self.spec.name or self.id

    @property
    def role(self) -> str:
        return str(self.spec.role)


class SubAgentError(RuntimeError):
    """Raised when sub-agent operations fail."""
