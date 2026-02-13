from __future__ import annotations
from typing import TYPE_CHECKING, TypedDict, Callable, NotRequired, Any

if TYPE_CHECKING:
    from hawi.agent import HawiAgent
    from hawi.agent.context import AgentContext
    from hawi.agent.model import Model
    from hawi.agent.messages import MessageResponse

BeforeSessionHook = Callable[["HawiAgent"], None]
AfterSessionHook = Callable[["HawiAgent"], None]
BeforeConversationHook = Callable[["HawiAgent"], None]
AfterConversationHook = Callable[["HawiAgent"], None]
BeforeModelCallHook = Callable[["HawiAgent", "AgentContext", "Model"], None]
AfterModelCallHook = Callable[["HawiAgent", "AgentContext", "MessageResponse"], None]
BeforeToolCallHook = Callable[["HawiAgent",str,dict[str,Any]], None]
AfterToolCallHook = Callable[["HawiAgent",str,dict[str,Any],Any], None]

class PluginHooks(TypedDict):
    before_session: NotRequired[BeforeSessionHook]
    after_session: NotRequired[AfterSessionHook]
    before_conversation: NotRequired[BeforeConversationHook]
    after_conversation: NotRequired[AfterConversationHook]
    before_model_call: NotRequired[BeforeModelCallHook]
    after_model_call: NotRequired[AfterModelCallHook]
    before_tool_calling: NotRequired[BeforeToolCallHook]
    after_tool_calling: NotRequired[AfterToolCallHook]
