from __future__ import annotations
from typing import TYPE_CHECKING, TypedDict, NotRequired, Any, Callable

from hawi.tool.types import ToolResult
from .hook_context import HookContext, HookResult

if TYPE_CHECKING:
    from hawi.agent import HawiAgent
    from hawi.agent.context import AgentContext
    from hawi.models import Model
    from hawi.models import MessageResponse


# ===== Hook method types (unbound, with self as first arg) =====
# Used for type-checking @hook-decorated methods inside HawiPlugin subclasses.

BeforeSessionMethod = Callable[[Any, "HawiAgent", HookContext], HookResult | None]
AfterSessionMethod = Callable[[Any, "HawiAgent", HookContext], HookResult | None]
BeforeConversationMethod = Callable[[Any, "HawiAgent", HookContext], HookResult | None]
AfterConversationMethod = Callable[[Any, "HawiAgent", HookContext], HookResult | None]
BeforeModelCallMethod = Callable[[Any, "HawiAgent", "AgentContext", "Model", HookContext], HookResult | None]
AfterModelCallMethod = Callable[[Any, "HawiAgent", "AgentContext", "MessageResponse", HookContext], HookResult | None]
BeforeToolCallMethod = Callable[[Any, "HawiAgent", str, dict, HookContext], HookResult | None]
AfterToolCallMethod = Callable[[Any, "HawiAgent", str, dict, ToolResult, HookContext], HookResult | None]


# ===== PluginHooks TypedDict (stores bound methods after plugin initialization) =====
# After binding, `self` is absorbed — signatures match the Method types minus the first arg.
class PluginHooks(TypedDict):
    before_session: NotRequired[Callable[..., HookResult | None]]
    after_session: NotRequired[Callable[..., HookResult | None]]
    before_conversation: NotRequired[Callable[..., HookResult | None]]
    after_conversation: NotRequired[Callable[..., HookResult | None]]
    before_model_call: NotRequired[Callable[..., HookResult | None]]
    after_model_call: NotRequired[Callable[..., HookResult | None]]
    before_tool_calling: NotRequired[Callable[..., HookResult | None]]
    after_tool_calling: NotRequired[Callable[..., HookResult | None]]
