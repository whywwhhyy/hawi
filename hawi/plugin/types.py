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
