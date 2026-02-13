"""Tool type definitions and AgentTool base class."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, TypeAlias, Union

# Type for tool output values - basic JSON-serializable types (including None)
ToolOutput: TypeAlias = Union[bool, str, int, float, list, dict, None]


class ToolResult:
    """Standard result format for tool execution.

    Attributes:
        success: Whether the tool execution succeeded.
        output: The result of the tool execution. Must be a basic JSON-serializable type.
                On failure, may contain error information (typically str).
        error: The error message (if any)
    """

    def __init__(self, success: bool, output: ToolOutput | None = None, error: str = "") -> None:
        self.success = success
        self.output = output
        self.error = error

    def __getitem__(self, key: str) -> Any:
        """Support dict-like access for backward compatibility."""
        if key == "success":
            return self.success
        if key == "output":
            return self.output
        if key == "error":
            return self.error
        raise KeyError(key)

    def __repr__(self) -> str:
        items = [f"success={self.success}"]
        if self.output:
            items.append(f"output={self.output!r}")
        if self.error:
            items.append(f"error={self.error}")
        items = ', '.join(items)
        return f"ToolResult({items})"


@dataclass
class PendingToolCall:
    """A tool call pending audit/approval.

    Attributes:
        tool_call_id: Unique identifier for this tool call
        tool_name: Name of the tool
        arguments: Arguments passed to the tool
        requested_at: Timestamp when the call was requested
    """

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    requested_at: float = field(default_factory=time.time)


# Type alias for tool functions
ToolFunc = Callable[..., ToolResult]
AsyncToolFunc = Callable[..., Coroutine[Any, Any, ToolResult]]


class AgentTool(ABC):
    """Abstract base class for agent tools.

    Subclasses must implement at least one of:
    - run(self, **kwargs) -> ToolResult: Synchronous execution
    - arun(self, **kwargs) -> ToolResult: Asynchronous execution

    Subclasses must also implement:
    - name (property): Tool name
    - description (property): Tool description
    - parameters_schema (property): JSON Schema for parameters

    Auto-detected properties:
    - supports_sync: Whether this tool supports synchronous execution
    - supports_async: Whether this tool supports asynchronous execution

    Optional configuration attributes:
    - audit: bool = False - When True, tool calls require human approval.
      The call is cached and the Agent is notified. After review, the call
      is either executed or rejected. Implemented by HawiAgent.
    - context: str = "" - Parameter name to inject from runtime context.
      When set, this parameter is hidden from ToolDefinition and automatically
      injected during invocation. Implemented by HawiAgent.
    - timeout: float | None = None - Execution timeout in seconds
    - tags: list[str] = [] - Tags for categorization and filtering
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The unique name of the tool used for identification and invocation."""
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what the tool does."""
        raise NotImplementedError

    @property
    @abstractmethod
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema defining the expected input parameters."""
        raise NotImplementedError

    @property
    def supports_sync(self) -> bool:
        """Whether this tool supports synchronous execution.

        Auto-detected based on whether run() is overridden.
        """
        return self._is_method_overridden("run")

    @property
    def supports_async(self) -> bool:
        """Whether this tool supports asynchronous execution.

        Auto-detected based on whether arun() is overridden.
        """
        return self._is_method_overridden("arun")

    # Audit mode: When True, tool calls require human approval before execution
    audit: bool = False

    # Context injection: Parameter name to inject from runtime context
    context: str | None = None

    # Timeout in seconds for tool execution (None = no timeout)
    timeout: float | None = None

    # Tags for categorization and filtering
    tags: list[str] = []

    def validate_parameters(self, parameters: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate parameters against JSON Schema."""
        from ._utils import validate_parameters
        return validate_parameters(parameters, self.parameters_schema)

    def _is_method_overridden(self, method_name: str) -> bool:
        """Check if a method has been overridden by a subclass."""
        return getattr(type(self), method_name) is not getattr(AgentTool, method_name)

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool synchronously with validated parameters."""
        if self._is_method_overridden("arun"):
            try:
                return asyncio.run(self.arun(**kwargs))
            except RuntimeError as e:
                if "cannot be called from a running event loop" in str(e):
                    raise RuntimeError(
                        "Cannot call sync run() from async context when only arun() is implemented. "
                        "Use ainvoke() instead."
                    ) from e
                raise
        raise NotImplementedError(f"{type(self).__name__} must implement run() or arun()")

    async def arun(self, **kwargs: Any) -> ToolResult:
        """Execute the tool asynchronously with validated parameters."""
        if self._is_method_overridden("run"):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: self.run(**kwargs))
        raise NotImplementedError(f"{type(self).__name__} must implement run() or arun()")

    def invoke(self, parameters: dict[str, Any]) -> ToolResult:
        """Invoke tool synchronously with parameter validation."""
        is_valid, errors = self.validate_parameters(parameters)
        if not is_valid:
            return ToolResult(success=False, error=f"Parameter validation failed: {'; '.join(errors)}")
        try:
            return self.run(**parameters)
        except Exception as e:
            return ToolResult(success=False, error=f"Tool execution failed: {type(e).__name__}: {e}")

    async def ainvoke(self, parameters: dict[str, Any]) -> ToolResult:
        """Invoke tool asynchronously with parameter validation."""
        is_valid, errors = self.validate_parameters(parameters)
        if not is_valid:
            return ToolResult(success=False, error=f"Parameter validation failed: {'; '.join(errors)}")
        try:
            return await self.arun(**parameters)
        except Exception as e:
            return ToolResult(success=False, error=f"Tool execution failed: {type(e).__name__}: {e}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
