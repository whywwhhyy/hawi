"""Function-based tool implementation and decorator."""

from __future__ import annotations

import inspect
from typing import Any, Callable, ParamSpec, TypeVar, overload

from .types import AgentTool, ToolResult


# Type variables for preserving function signature
P = ParamSpec("P")
R = TypeVar("R")


class FunctionAgentTool(AgentTool):
    """AgentTool implementation backed by a Python function.

    This class wraps a Python function (sync or async) into an AgentTool,
    automatically extracting parameters and documentation.

    Automatically detects function type and implements run() or arun() accordingly.

    Usage:
        def search(query: str, limit: int = 10) -> str:
            return f"Results for {query}"

        tool = FunctionAgentTool(search)
        result = await tool.ainvoke({"query": "test"})
    """

    # Mark as tool for plugin discovery
    _is_tool = True

    def __init__(
        self,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        parameters_schema: dict[str, Any] | None = None,
        audit: bool = False,
        context: str = "",
        timeout: float | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Initialize from a function.

        Args:
            func: The function to wrap (sync or async).
            name: Optional override for tool name.
            description: Optional override for tool description.
            parameters_schema: Optional override for parameter schema.
            audit: When True, tool calls require human approval.
            context: Parameter name to inject from runtime context.
            timeout: Execution timeout in seconds (None = no timeout).
            tags: Tags for categorization and filtering.
        """
        self._func = func
        self._name = name
        self._description = description
        self._parameters_schema = parameters_schema
        self._is_async = inspect.iscoroutinefunction(func)
        self.audit = audit
        self.context = context
        self.timeout = timeout
        self.tags = tags or []

    @property
    def name(self) -> str:
        """The unique name of the tool."""
        return self._name or self._func.__name__

    @property
    def description(self) -> str:
        """Human-readable description of what the tool does."""
        if self._description is not None:
            return self._description
        return (self._func.__doc__ or "").strip()

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema defining the expected input parameters."""
        if self._parameters_schema is not None:
            return self._parameters_schema

        from ._utils import build_parameters_schema

        schema = build_parameters_schema(self._func)
        if schema is None:
            return {}
        return schema

    def _execute_function(self, *args: Any, **kwargs: Any) -> ToolResult:
        """Execute the wrapped function and convert result to ToolResult."""
        try:
            result = self._func(*args, **kwargs)

            # Convert result to ToolResult
            if isinstance(result, ToolResult):
                return result
            else:
                return ToolResult(success=True, output=result)

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"{type(e).__name__}: {e}",
            )

    async def _execute_function_async(self, **kwargs: Any) -> ToolResult:
        """Execute the wrapped async function and convert result to ToolResult."""
        try:
            result = await self._func(**kwargs)

            # Convert result to ToolResult
            if isinstance(result, ToolResult):
                return result
            else:
                return ToolResult(success=True, output=result)

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"{type(e).__name__}: {e}",
            )

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the wrapped function synchronously.

        Only implemented for sync functions. For async functions,
        base class will route to arun().
        """
        if self._is_async:
            # Let base class handle the async case
            return super().run(**kwargs)
        return self._execute_function(**kwargs)

    async def arun(self, **kwargs: Any) -> ToolResult:
        """Execute the wrapped function asynchronously.

        Only implemented for async functions. For sync functions,
        base class will route to run() in thread pool.
        """
        if not self._is_async:
            # Let base class handle the sync case
            return await super().arun(**kwargs)
        return await self._execute_function_async(**kwargs)


@overload
def tool(func: Callable[P, R], /) -> FunctionAgentTool:
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
) -> Callable[[Callable[P, R]], FunctionAgentTool]:
    """Decorator usage with parentheses: @tool() or @tool(name=...)"""
    ...


def tool(
    func: Callable[P, R] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    parameters_schema: dict[str, Any] | None = None,
    audit: bool = False,
    context: str = "",
    timeout: float | None = None,
    tags: list[str] | None = None,
) -> FunctionAgentTool | Callable[[Callable[P, R]], FunctionAgentTool]:
    """Decorator to convert a Python function into a FunctionAgentTool.

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
        FunctionAgentTool instance, or a decorator function.
    """

    def decorator(f: Callable[P, R]) -> FunctionAgentTool:
        return FunctionAgentTool(
            f,
            name=name,
            description=description,
            parameters_schema=parameters_schema,
            audit=audit,
            context=context,
            timeout=timeout,
            tags=tags,
        )

    if func is not None:
        # Used as @tool (without parentheses)
        return decorator(func)
    else:
        # Used as @tool() or @tool(name="...")
        return decorator
