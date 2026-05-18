"""Runtime state types used by HawiAgent."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from hawi.errors import HawiError
from hawi.models import ContentPart

from .result import ToolCallRecord


@dataclass
class _ExecutionState:
    """Internal execution state during agent run."""

    iteration: int = 0
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    error: HawiError | str | None = None
    should_stop: bool = False
    run_id: str = ""
    stop_reason: str | None = None
    pending_reinvoke_message: str | list[ContentPart] | None = None
    last_auto_compact_iteration: int | None = None


@dataclass
class _RecentToolResult:
    """A tool result that has not yet been accepted by a model call."""

    tool_call_id: str
    tool_name: str
    content: str
    is_error: bool
    truncate_attempts: int = 0


class SteerPartMergeMode(str, Enum):
    """Preferred steer lowering strategy for the related model."""

    APPEND_TO_TOOL_RESULT = "append_to_tool_result"
    USER_MESSAGE_TEMPLATE = "user_message_template"
    TOOL_RESULT_ASSISTANT_TEMPLATE_AND_USER_MESSAGE = (
        "tool_result_assistant_template_and_user_message"
    )


@dataclass
class PendingInput:
    """Queued user input awaiting materialization into context messages."""

    id: str
    content: list[ContentPart]
    candidate_tool_call_ids: tuple[str, ...]
    created_at: float
    preferred_merge_mode: SteerPartMergeMode | None = None


@dataclass
class MaterializedSteerMessage:
    """A pending steer input that has been appended to context."""

    content: list[ContentPart]
    metadata: dict[str, Any]
    context_message_id: str


@dataclass
class AddedToolResultMessages:
    """Tool-result context message plus any steer messages materialized after it."""

    context_message_id: str
    materialized_messages: list[MaterializedSteerMessage] = field(default_factory=list)

    def __iter__(self):
        return iter(self.materialized_messages)

    def __len__(self) -> int:
        return len(self.materialized_messages)

    def __bool__(self) -> bool:
        return bool(self.materialized_messages)
