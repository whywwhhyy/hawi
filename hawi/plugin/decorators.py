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
    """Hook called once at the start of an agent session (before any run).

    Args:
        agent: The HawiAgent instance.
        ctx: HookContext with run_id and iteration=0.

    Context operations (safe at this point):
        Modifications to ``agent.context`` take effect before the first
        model call of the session.

    Returns:
        - ``None`` to continue normally.
        - ``HookResult.abort(reason)`` to terminate the agent run.
    """
    setattr(func, "_is_hook", True)
    setattr(func, "_hook_type", "before_session")
    return func


def after_session(func: AfterSessionMethod) -> AfterSessionMethod:
    setattr(func, "_is_hook", True)
    setattr(func, "_hook_type", "after_session")
    return func


def before_conversation(func: BeforeConversationMethod) -> BeforeConversationMethod:
    """Hook called at the start of each conversation turn.

    Args:
        agent: The HawiAgent instance.
        ctx: HookContext with run_id and iteration=0.

    Context operations (safe at this point):
        Modifications to ``agent.context`` take effect before the first
        model call of this conversation.

    Returns:
        - ``None`` to continue normally.
        - ``HookResult.abort(reason)`` to terminate the agent run.
    """
    setattr(func, "_is_hook", True)
    setattr(func, "_hook_type", "before_conversation")
    return func


def after_conversation(func: AfterConversationMethod) -> AfterConversationMethod:
    setattr(func, "_is_hook", True)
    setattr(func, "_hook_type", "after_conversation")
    return func


def before_model_call(func: BeforeModelCallMethod) -> BeforeModelCallMethod:
    """Hook called before each model invocation.

    Args:
        agent: The HawiAgent instance. Access context via ``agent.context``.
        model: The Model about to be called.
        ctx: HookContext with run_id and iteration.

    Context operations (safe at this point):
        Modifications to ``agent.context`` (inject, collapse, truncate,
        add_user_message, etc.) take effect in the upcoming model call.

    Returns:
        - ``None`` to continue normally.
        - ``HookResult.replace_model(model)`` to use a different model for
          this call only.
        - ``HookResult.restart_turn()`` to skip this model call and continue
          to the next loop iteration.
        - ``HookResult.abort(reason)`` to terminate the agent run.
    """
    setattr(func, "_is_hook", True)
    setattr(func, "_hook_type", "before_model_call")
    return func


def after_model_call(func: AfterModelCallMethod) -> AfterModelCallMethod:
    """Hook called after each model invocation, before the assistant message
    is written to context.

    Args:
        agent: The HawiAgent instance. Access context via ``agent.context``.
        response: The MessageResponse from the model (contains stop_reason,
            usage, content).
        ctx: HookContext with run_id, iteration, and duration_ms.

    Timing note:
        The assistant message has NOT yet been added to ``agent.context``
        at this point. It will be added immediately after this hook returns.

    Context operations (safe at this point):
        You may call ``agent.context`` operations here; they will take effect
        before the assistant message is written.

    Returns:
        - ``None`` to continue normally.
        - ``HookResult.reinvoke(message)`` to append a message to context,
          stop the current run (stop_reason ``"hook_reinvoke"``), and
          re-invoke the agent with the new message.
        - ``HookResult.abort(reason)`` to terminate the agent run.
    """
    setattr(func, "_is_hook", True)
    setattr(func, "_hook_type", "after_model_call")
    return func


def before_tool_calling(func: BeforeToolCallMethod) -> BeforeToolCallMethod:
    """Hook called before tool execution.

    Args:
        agent: The HawiAgent instance.
        tool_name: Name of the tool being called (always present, even if
            the tool object is not found).
        arguments: Mutable dict of arguments. Changes made here are
            reflected in the actual tool call.
        ctx: HookContext with run_id, iteration, tool_call_id, and tool
            (the AgentTool object, or None if the tool was not found).

    Note:
        ``tool_name`` is always a string. ``ctx.tool`` is the resolved
        AgentTool object and may be None if the tool name is unrecognised.

    Context operations (safe at this point):
        Modifications to ``agent.context`` take effect before the tool runs.

    Returns:
        - ``None`` to continue normally.
        - ``HookResult.skip(result)`` to bypass tool execution and use
          ``result`` as the synthetic tool output.
        - ``HookResult.abort(reason)`` to terminate the agent run.
    """
    setattr(func, "_is_hook", True)
    setattr(func, "_hook_type", "before_tool_calling")
    return func


def after_tool_calling(func: AfterToolCallMethod) -> AfterToolCallMethod:
    """Hook called after tool execution, before the tool result is written
    to context.

    Args:
        agent: The HawiAgent instance.
        tool_name: Name of the tool that was called.
        arguments: Dict of arguments that were used.
        result: Mutable ToolResult. Changes made here are reflected in
            the tool result written to context.
        ctx: HookContext with run_id, iteration, tool_call_id, tool, and
            duration_ms.

    Timing note:
        The tool result has NOT yet been added to ``agent.context`` at this
        point. It will be added immediately after this hook returns.

    Context operations (safe at this point):
        Modifications to ``agent.context`` take effect before the tool
        result is written.

    Returns:
        - ``None`` to continue normally.
        - ``HookResult.abort(reason)`` to terminate the agent run.
        - ``HookResult.reinvoke(message)`` to append a message to context
          and re-enter the model loop after the current tool result is
          written.
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
