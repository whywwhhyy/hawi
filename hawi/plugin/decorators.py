from __future__ import annotations
from typing import Callable, Any, TYPE_CHECKING, overload, ParamSpec, TypeVar

from .types import (
    BeforeSessionHook,
    AfterSessionHook,
    BeforeConversationHook,
    AfterConversationHook,
    BeforeModelCallHook,
    AfterModelCallHook,
    BeforeToolCallHook,
    AfterToolCallHook,
)

# Type variables for preserving function signature
P = ParamSpec("P")
R = TypeVar("R")


def before_session(func: BeforeSessionHook):
    setattr(func, "_is_hook", True)
    setattr(func, "_hook_type", "before_session")
    return func


def after_session(func: AfterSessionHook):
    setattr(func, "_is_hook", True)
    setattr(func, "_hook_type", "after_session")
    return func


def before_conversation(func: BeforeConversationHook):
    setattr(func, "_is_hook", True)
    setattr(func, "_hook_type", "before_conversation")
    return func


def after_conversation(func: AfterConversationHook):
    setattr(func, "_is_hook", True)
    setattr(func, "_hook_type", "after_conversation")
    return func


def before_model_call(func: BeforeModelCallHook):
    """Hook called before model invocation.

    Can be used to modify context or replace model.

    Args:
        agent: The HawiAgent instance
        context: The AgentContext
        model: The Model to be called (can be replaced by assigning to event)
    """
    setattr(func, "_is_hook", True)
    setattr(func, "_hook_type", "before_model_call")
    return func


def after_model_call(func: AfterModelCallHook):
    """Hook called after model invocation.

    Can be used to modify the response.

    Args:
        agent: The HawiAgent instance
        context: The AgentContext
        response: The MessageResponse from the model
    """
    setattr(func, "_is_hook", True)
    setattr(func, "_hook_type", "after_model_call")
    return func


def before_tool_calling(func: BeforeToolCallHook):
    setattr(func, "_is_hook", True)
    setattr(func, "_hook_type", "before_tool_calling")
    return func


def after_tool_calling(func: AfterToolCallHook):
    setattr(func, "_is_hook", True)
    setattr(func, "_hook_type", "after_tool_calling")
    return func


@overload
def tool(func: Callable[P, R], /) -> Callable[P, R]:
    """Decorator usage without parentheses: @tool"""
    ...


@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    parameters_schema: dict[str, Any] | None = None,
    audit: bool = False,
    context: str = "",
    timeout: float | None = None,
    tags: list[str] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator usage with parentheses: @tool() or @tool(name=...)"""
    ...


def tool(
    func: Callable[P, R] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    parameters_schema: dict[str, Any] | None = None,
    audit: bool | None = None,
    context: str | None = None,
    timeout: float | None = None,
    tags: list[str] | None = None,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Mark a function as a tool.

    Can be used with or without parentheses:

        @tool
        def search(query: str) -> str:
            '''Search for information.'''
            return f"Results for {query}"

        @tool(name="my_search")
        async def search_async(query: str) -> str:
            return f"Results for {query}"

        @tool(description="Custom description")
        def another_tool(x: int) -> int:
            return x * 2

    Args:
        func: The function to decorate (when used without parentheses).
        name: Optional override for tool name.
        description: Optional override for tool description.
        parameters_schema: Optional override for parameter schema.
        audit: When True, tool calls require human approval.
        context: Parameter name to inject from runtime context.
        timeout: Execution timeout in seconds (None = no timeout).
        tags: Tags for categorization and filtering.

    Returns:
        function instance, or a decorator function.
    """
    def decorator(f: Callable[P, R]) -> Callable[P, R]:
        # Mark function for plugin discovery
        setattr(f, "_is_agent_tool", True)
        setattr(f, "_agent_tool_parameters", {k: v for k, v in {
            'name':name,
            'description':description,
            'parameters_schema':parameters_schema,
            'audit':audit,
            'context':context,
            'timeout':timeout,
            'tags':tags,
        }.items() if v is not None})
        return f

    if func is not None:
        # Used as @tool (without parentheses)
        return decorator(func)
    else:
        # Used as @tool() or @tool(name="...")
        return decorator
