"""Global tool registry with two-layer architecture."""

import threading
from typing import Any, Callable
from pydantic import BaseModel

from .types import AgentTool

class ToolRegistry:
    """Global tool registry supporting two-layer architecture:

    1. Base Access Layer: Factory-based registration with config-based creation
    2. Agent Tool Layer: Instance-based registration for LLM-facing tools
    """

    _instance: "ToolRegistry | None" = None
    _lock: threading.Lock = threading.Lock()

    # Base layer: name -> (factory, default_config)
    _base_factories: dict[str, tuple[Callable[[dict], Any], dict]] = {}
    _base_cache: dict[str, Any] = {}  # name -> cached instance

    # Agent layer: name -> Tool instance
    _agent_tools: dict[str, AgentTool] = {}

    def __new__(cls) -> "ToolRegistry":
        raise RuntimeError("Use ToolRegistry.instance()")

    @classmethod
    def instance(cls) -> "ToolRegistry":
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = object.__new__(cls)
        return cls._instance

    # ===== Base Layer: Factory-based =====

    @classmethod
    def register_base(
        cls,
        name: str,
        factory: Callable[[dict], Any],
        default_config: dict | None = None,
    ) -> None:
        """Register a base access tool (e.g., Jira, Confluence client).

        Args:
            name: Tool identifier
            factory: Function that creates the tool instance from config
            default_config: Default configuration for creating instances
        """
        with cls._lock:
            cls._base_factories[name] = (factory, default_config or {})

    @classmethod
    def get_base(cls, name: str, config: dict | None = None) -> Any:
        """Get or create base tool instance.

        Uses cache if no custom config provided.
        """
        with cls._lock:
            if name not in cls._base_factories:
                raise KeyError(f"Base tool '{name}' not registered")

            factory, default_config = cls._base_factories[name]
            merged_config = {**default_config, **(config or {})}

            # Use cached version if using default config
            if not config and name in cls._base_cache:
                return cls._base_cache[name]

            instance = factory(merged_config)

            if not config:
                cls._base_cache[name] = instance

            return instance

    @classmethod
    def clear_base_cache(cls, name: str | None = None) -> None:
        """Clear cached base tool instance(s)."""
        with cls._lock:
            if name:
                cls._base_cache.pop(name, None)
            else:
                cls._base_cache.clear()

    # ===== Agent Layer: Instance-based =====

    @classmethod
    def register_agent_tool(cls, tool: AgentTool) -> None:
        """Register an agent-facing tool instance."""
        with cls._lock:
            cls._agent_tools[tool.name] = tool

    @classmethod
    def get_agent_tool(cls, name: str) -> AgentTool:
        """Get agent tool by name."""
        with cls._lock:
            if name not in cls._agent_tools:
                raise KeyError(f"Agent tool '{name}' not registered")
            return cls._agent_tools[name]

    @classmethod
    def get_agent_tools(cls, names: list[str] | None = None) -> list[AgentTool]:
        """Get agent tools, optionally filtered by names."""
        with cls._lock:
            if names is None:
                return list(cls._agent_tools.values())
            return [cls._agent_tools[n] for n in names if n in cls._agent_tools]

    @classmethod
    def list_agent_tools(cls) -> list[str]:
        """List all registered agent tool names."""
        with cls._lock:
            return list(cls._agent_tools.keys())

    @classmethod
    def unregister_agent_tool(cls, name: str) -> None:
        """Unregister an agent tool."""
        with cls._lock:
            cls._agent_tools.pop(name, None)

    # ===== Utilities =====

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations (for testing)."""
        with cls._lock:
            cls._base_factories.clear()
            cls._base_cache.clear()
            cls._agent_tools.clear()


# Convenience function for context-based testing
def get_registry() -> type[ToolRegistry]:
    """Get ToolRegistry class (for dependency injection in tests)."""
    return ToolRegistry
