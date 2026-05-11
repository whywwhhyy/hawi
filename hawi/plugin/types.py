from __future__ import annotations
from typing import TYPE_CHECKING, TypedDict, NotRequired, TypeAlias, Any, Callable, Coroutine

from hawi.tool.types import ToolResult
from .hook_context import HookContext, HookResult

if TYPE_CHECKING:
    from hawi.agent import HawiAgent
    from hawi.models import Model
    from hawi.models import MessageResponse


# ===== Hook method types (unbound, with self as first arg) =====
# Used for type-checking @hook-decorated methods inside HawiPlugin subclasses.
# Supports both sync and async hook methods.

# Type alias for hook return value (both sync and async)
HookReturnType:TypeAlias = HookResult | None | Coroutine[Any, Any, HookResult | None]
SystemPromptVariability: TypeAlias = str
SystemPromptVariabilityInput: TypeAlias = (
    SystemPromptVariability | list[SystemPromptVariability] | tuple[SystemPromptVariability, ...]
)

SYSTEM_PROMPT_VARIABILITY_ORDER: dict[str, int] = {
    "hardcoded": 0,
    "plugin_config": 10,
    "time_year": 20,
    "time_month": 30,
    "time_day": 40,
    "time_hour": 50,
    "working_dir": 60,
    "filesystem": 70,
    "session_state": 80,
    "conversation_state": 90,
    "external_state": 100,
    "default": 1000,
}

SYSTEM_PROMPT_VARIABILITY_DEFAULT = "default"

_SYSTEM_PROMPT_VARIABILITY_ALIASES: dict[str, str] = {
    "none": "hardcoded",
    "static": "hardcoded",
    "time:year": "time_year",
    "time.year": "time_year",
    "year": "time_year",
    "time:month": "time_month",
    "time.month": "time_month",
    "month": "time_month",
    "time:day": "time_day",
    "time.day": "time_day",
    "day": "time_day",
    "time:hour": "time_hour",
    "time.hour": "time_hour",
    "hour": "time_hour",
    "cwd": "working_dir",
    "current_working_dir": "working_dir",
    "working_directory": "working_dir",
}


def normalize_system_prompt_variability(
    variability: SystemPromptVariabilityInput | None,
) -> tuple[SystemPromptVariability, ...]:
    """Normalize a hook's declared system-prompt variability factors."""
    if variability is None:
        return (SYSTEM_PROMPT_VARIABILITY_DEFAULT,)
    if isinstance(variability, str):
        factors = (variability,)
    else:
        factors = tuple(variability)
    if not factors:
        return ("hardcoded",)
    return tuple(
        _SYSTEM_PROMPT_VARIABILITY_ALIASES.get(str(factor), str(factor))
        for factor in factors
    )


def system_prompt_variability_rank(hook: Callable[..., Any]) -> int:
    """Return the cache-stability rank for a system-prompt injection hook.

    A hook with multiple factors is only as stable as its most variable factor,
    so the maximum declared rank is used. Hooks with no declaration sort last.
    """
    factors = getattr(
        hook,
        "_system_prompt_variability",
        (SYSTEM_PROMPT_VARIABILITY_DEFAULT,),
    )
    if isinstance(factors, str):
        factors = (factors,)
    if not factors:
        factors = ("hardcoded",)
    return max(
        SYSTEM_PROMPT_VARIABILITY_ORDER.get(
            _SYSTEM_PROMPT_VARIABILITY_ALIASES.get(str(factor), str(factor)),
            SYSTEM_PROMPT_VARIABILITY_ORDER[SYSTEM_PROMPT_VARIABILITY_DEFAULT],
        )
        for factor in factors
    )


def is_system_prompt_injection_hook(hook: Callable[..., Any]) -> bool:
    """Whether a hook declares that it mutates or injects system prompt text."""
    return bool(getattr(hook, "_injects_system_prompt", False))

BeforeSessionMethod:TypeAlias = Callable[[Any, "HawiAgent", HookContext], HookReturnType]
AfterSessionMethod:TypeAlias = Callable[[Any, "HawiAgent", HookContext], HookReturnType]
BeforeConversationMethod:TypeAlias = Callable[[Any, "HawiAgent", HookContext], HookReturnType]
AfterConversationMethod:TypeAlias = Callable[[Any, "HawiAgent", HookContext], HookReturnType]
BeforeModelCallMethod:TypeAlias = Callable[[Any, "HawiAgent", "Model", HookContext], HookReturnType]
AfterModelCallMethod:TypeAlias = Callable[[Any, "HawiAgent", "MessageResponse", HookContext], HookReturnType]
BeforeToolCallMethod:TypeAlias = Callable[[Any, "HawiAgent", str, dict, HookContext], HookReturnType]
AfterToolCallMethod:TypeAlias = Callable[[Any, "HawiAgent", str, dict, ToolResult, HookContext], HookReturnType]


# ===== PluginHooks TypedDict (stores bound methods after plugin initialization) =====
# After binding, `self` is absorbed — signatures match the Method types minus the first arg.
class PluginHooks(TypedDict):
    before_session: NotRequired[Callable[..., HookReturnType]]
    after_session: NotRequired[Callable[..., HookReturnType]]
    before_conversation: NotRequired[Callable[..., HookReturnType]]
    after_conversation: NotRequired[Callable[..., HookReturnType]]
    before_model_call: NotRequired[Callable[..., HookReturnType]]
    after_model_call: NotRequired[Callable[..., HookReturnType]]
    before_tool_calling: NotRequired[Callable[..., HookReturnType]]
    after_tool_calling: NotRequired[Callable[..., HookReturnType]]
