from __future__ import annotations
from typing import TYPE_CHECKING, TypedDict, NotRequired, Any, Callable, Union

if TYPE_CHECKING:
    from hawi.agent import HawiAgent
    from hawi.agent.context import AgentContext
    from hawi.agent.model import Model
    from hawi.agent.message import MessageResponse

# ===== Hook types for regular functions =====
# Function: def hook(agent: HawiAgent) -> None
BeforeSessionFunc = Callable[["HawiAgent"], None]
AfterSessionFunc = Callable[["HawiAgent"], None]
BeforeConversationFunc = Callable[["HawiAgent"], None]
AfterConversationFunc = Callable[["HawiAgent"], None]
BeforeModelCallFunc = Callable[["HawiAgent", "AgentContext", "Model"], None]
AfterModelCallFunc = Callable[["HawiAgent", "AgentContext", "MessageResponse"], None]
BeforeToolCallFunc = Callable[["HawiAgent", str, dict[str, Any]], None]
AfterToolCallFunc = Callable[["HawiAgent", str, dict[str, Any], Any], None]


# ===== Hook types for methods (with self) =====
# Method: def method(self, agent: HawiAgent) -> None
BeforeSessionMethod = Callable[[Any, "HawiAgent"], None]
AfterSessionMethod = Callable[[Any, "HawiAgent"], None]
BeforeConversationMethod = Callable[[Any, "HawiAgent"], None]
AfterConversationMethod = Callable[[Any, "HawiAgent"], None]
BeforeModelCallMethod = Callable[[Any, "HawiAgent", "AgentContext", "Model"], None]
AfterModelCallMethod = Callable[[Any, "HawiAgent", "AgentContext", "MessageResponse"], None]
BeforeToolCallMethod = Callable[[Any, "HawiAgent", str, dict[str, Any]], None]
AfterToolCallMethod = Callable[[Any, "HawiAgent", str, dict[str, Any], Any], None]


# ===== Union types for decorators (accept both functions and methods) =====
BeforeSessionHook = Union[BeforeSessionFunc, BeforeSessionMethod]
AfterSessionHook = Union[AfterSessionFunc, AfterSessionMethod]
BeforeConversationHook = Union[BeforeConversationFunc, BeforeConversationMethod]
AfterConversationHook = Union[AfterConversationFunc, AfterConversationMethod]
BeforeModelCallHook = Union[BeforeModelCallFunc, BeforeModelCallMethod]
AfterModelCallHook = Union[AfterModelCallFunc, AfterModelCallMethod]
BeforeToolCallHook = Union[BeforeToolCallFunc, BeforeToolCallMethod]
AfterToolCallHook = Union[AfterToolCallFunc, AfterToolCallMethod]


# ===== PluginHooks TypedDict (stores bound methods after plugin initialization) =====
class PluginHooks(TypedDict):
    before_session: NotRequired[BeforeSessionFunc]
    after_session: NotRequired[AfterSessionFunc]
    before_conversation: NotRequired[BeforeConversationFunc]
    after_conversation: NotRequired[AfterConversationFunc]
    before_model_call: NotRequired[BeforeModelCallFunc]
    after_model_call: NotRequired[AfterModelCallFunc]
    before_tool_calling: NotRequired[BeforeToolCallFunc]
    after_tool_calling: NotRequired[AfterToolCallFunc]
