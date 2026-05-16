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
    SystemPromptVariabilityInput,
    normalize_system_prompt_variability,
)

# Type variables for preserving function signature
P = ParamSpec("P")
R = TypeVar("R")


def _mark_hook(
    func: Callable[..., Any],
    hook_type: str,
    *,
    system_prompt_variability: SystemPromptVariabilityInput | None = None,
) -> Callable[..., Any]:
    setattr(func, "_is_hook", True)
    setattr(func, "_hook_type", hook_type)
    if system_prompt_variability is not None:
        setattr(func, "_injects_system_prompt", True)
        setattr(
            func,
            "_system_prompt_variability",
            normalize_system_prompt_variability(system_prompt_variability),
        )
    return func


@overload
def before_session(func: BeforeSessionMethod, /) -> BeforeSessionMethod:
    ...


@overload
def before_session(
    *,
    system_prompt_variability: SystemPromptVariabilityInput | None = None,
) -> Callable[[BeforeSessionMethod], BeforeSessionMethod]:
    ...


def before_session(
    func: BeforeSessionMethod | None = None,
    *,
    system_prompt_variability: SystemPromptVariabilityInput | None = None,
) -> BeforeSessionMethod | Callable[[BeforeSessionMethod], BeforeSessionMethod]:
    """Hook called once at the start of an agent session (before any run).

    Args:
        agent: The HawiAgent instance.
        ctx: HookContext with run_id and iteration=0.
        system_prompt_variability: Optional declaration for hooks that inject
            stable, non-compressible system prompt content. Declared system
            prompt hooks run once per agent session; put changing context in a
            user-role message from ``before_conversation`` instead.

    Context operations (safe at this point):
        Modifications to ``agent.context`` take effect before the first
        model call of the session.

    Returns:
        - ``None`` to continue normally.
        - ``HookResult.abort(reason)`` to terminate the agent run.
    """
    def decorate(hook: BeforeSessionMethod) -> BeforeSessionMethod:
        return _mark_hook(
            hook,
            "before_session",
            system_prompt_variability=system_prompt_variability,
        )  # type: ignore[return-value]

    if func is None:
        return decorate
    return decorate(func)


def after_session(func: AfterSessionMethod) -> AfterSessionMethod:
    return _mark_hook(func, "after_session")  # type: ignore[return-value]


@overload
def before_conversation(func: BeforeConversationMethod, /) -> BeforeConversationMethod:
    ...


@overload
def before_conversation(
    *,
    system_prompt_variability: SystemPromptVariabilityInput | None = None,
) -> Callable[[BeforeConversationMethod], BeforeConversationMethod]:
    ...


def before_conversation(
    func: BeforeConversationMethod | None = None,
    *,
    system_prompt_variability: SystemPromptVariabilityInput | None = None,
) -> BeforeConversationMethod | Callable[[BeforeConversationMethod], BeforeConversationMethod]:
    """Hook called at the start of each conversation turn.

    Args:
        agent: The HawiAgent instance.
        ctx: HookContext with run_id and iteration=0.
        system_prompt_variability: Backward-compatible declaration for hooks
            that inject stable, non-compressible system prompt content. Declared
            system prompt hooks run once per agent session; new per-turn or
            changing context should be injected as a user-role message before
            the current user prompt.

    Context operations (safe at this point):
        Modifications to ``agent.context`` take effect before the first
        model call of this conversation.

    Returns:
        - ``None`` to continue normally.
        - ``HookResult.abort(reason)`` to terminate the agent run.
    """
    def decorate(hook: BeforeConversationMethod) -> BeforeConversationMethod:
        return _mark_hook(
            hook,
            "before_conversation",
            system_prompt_variability=system_prompt_variability,
        )  # type: ignore[return-value]

    if func is None:
        return decorate
    return decorate(func)


def after_conversation(func: AfterConversationMethod) -> AfterConversationMethod:
    return _mark_hook(func, "after_conversation")  # type: ignore[return-value]


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
    return _mark_hook(func, "before_model_call")  # type: ignore[return-value]


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
    return _mark_hook(func, "after_model_call")  # type: ignore[return-value]


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
    return _mark_hook(func, "before_tool_calling")  # type: ignore[return-value]


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
    return _mark_hook(func, "after_tool_calling")  # type: ignore[return-value]


def system_prompt_variability(
    variability: SystemPromptVariabilityInput = "default",
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Mark a hook as injecting stable system prompt content once per session."""
    def decorate(func: Callable[P, R]) -> Callable[P, R]:
        setattr(func, "_injects_system_prompt", True)
        setattr(
            func,
            "_system_prompt_variability",
            normalize_system_prompt_variability(variability),
        )
        return func

    return decorate


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
