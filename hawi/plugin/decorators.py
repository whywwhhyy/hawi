from __future__ import annotations
from typing import Any, overload, ParamSpec, TypeVar, Callable

from .types import (
    BeforeSessionMethod,
    AfterSessionMethod,
    BeforeConversationMethod,
    AfterConversationMethod,
    BeforeModelCallMethod,
    AfterModelCallMethod,
    BeforeToolCallMethod,
    AfterToolCallMethod,
)

# Type variables for preserving function signature
P = ParamSpec("P")
R = TypeVar("R")


def before_session(func: BeforeSessionMethod) -> BeforeSessionMethod:
    setattr(func, "_is_hook", True)
    setattr(func, "_hook_type", "before_session")
    return func


def after_session(func: AfterSessionMethod) -> AfterSessionMethod:
    setattr(func, "_is_hook", True)
    setattr(func, "_hook_type", "after_session")
    return func


def before_conversation(func: BeforeConversationMethod) -> BeforeConversationMethod:
    setattr(func, "_is_hook", True)
    setattr(func, "_hook_type", "before_conversation")
    return func


def after_conversation(func: AfterConversationMethod) -> AfterConversationMethod:
    setattr(func, "_is_hook", True)
    setattr(func, "_hook_type", "after_conversation")
    return func


def before_model_call(func: BeforeModelCallMethod) -> BeforeModelCallMethod:
    """Hook called before model invocation.

    Can be used to modify context or add per-turn system prompt instructions.

    Args:
        agent: The HawiAgent instance
        context: The AgentContext
        model: The Model to be called
        ctx: HookContext with run_id, iteration, cumulative usage
    """
    setattr(func, "_is_hook", True)
    setattr(func, "_hook_type", "before_model_call")
    return func


def after_model_call(func: AfterModelCallMethod) -> AfterModelCallMethod:
    """Hook called after model invocation.

    Can be used to modify the response, track latency, or enforce budgets.

    Args:
        agent: The HawiAgent instance
        context: The AgentContext
        response: The MessageResponse from the model
        ctx: HookContext with run_id, iteration, duration_ms, usage, stop_reason
    """
    setattr(func, "_is_hook", True)
    setattr(func, "_hook_type", "after_model_call")
    return func


def before_tool_calling(func: BeforeToolCallMethod) -> BeforeToolCallMethod:
    """Hook called before tool execution.

    Return ``HookResult.skip(result)`` to bypass the tool and use a synthetic result.

    Args:
        agent: The HawiAgent instance
        tool_name: Name of the tool being called
        arguments: Dict of arguments (mutable — changes are reflected in the call)
        ctx: HookContext with run_id, iteration, tool_call_id, tool object
    """
    setattr(func, "_is_hook", True)
    setattr(func, "_hook_type", "before_tool_calling")
    return func


def after_tool_calling(func: AfterToolCallMethod) -> AfterToolCallMethod:
    """Hook called after tool execution.

    Can be used to transform the result, cache it, or collect statistics.

    Args:
        agent: The HawiAgent instance
        tool_name: Name of the tool that was called
        arguments: Dict of arguments used
        result: ToolResult (mutable — changes are reflected in the conversation)
        ctx: HookContext with run_id, iteration, tool_call_id, tool, duration_ms
    """
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
            'name':name or f.__qualname__.replace('.','__'),
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
