"""Base Tool abstraction."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine, Union
from pydantic import BaseModel


class ToolResult(BaseModel):
    """Standard result format for tool execution."""
    success: bool
    output: str = ""
    error: str | None = None
    data: dict[str, Any] | None = None


class Tool(ABC):
    """Abstract base class for tools.

    Tools can implement either sync (execute) or async (execute_async) method.
    If both are implemented, async takes precedence in async contexts.
    """

    name: str
    description: str
    parameters: type[BaseModel] | None = None

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Synchronous execution. Implement this OR execute_async."""
        raise NotImplementedError("Implement execute or execute_async")

    async def execute_async(self, **kwargs) -> ToolResult:
        """Asynchronous execution. Override for async tools."""
        # Default: run sync version in thread
        return self.execute(**kwargs)

    def get_schema(self) -> dict[str, Any]:
        """Get JSON schema for LLM tool calling."""
        schema = {
            "name": self.name,
            "description": self.description,
            "parameters": {},
        }
        if self.parameters:
            schema["parameters"] = self.parameters.model_json_schema()
        return schema


# Type alias for tool functions
ToolFunc = Callable[..., ToolResult]
AsyncToolFunc = Callable[..., Coroutine[Any, Any, ToolResult]]
