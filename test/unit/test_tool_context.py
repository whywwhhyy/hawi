"""Unit tests for tool call context injection functionality.

Tests the context injection mechanism that allows tools to receive runtime
context information hidden from the LLM.
"""

import pytest
from hawi.agent.context import AgentContext, ToolCallContext
from hawi.tool.types import AgentTool, ToolResult


class ContextTool(AgentTool):
    """Tool that uses context injection."""

    context = "user_id"  # Parameter to inject from context

    @property
    def name(self) -> str:
        return "context_tool"

    @property
    def description(self) -> str:
        return "A tool that requires user_id from context"

    @property
    def parameters_schema(self) -> dict:
        # Note: user_id is NOT in schema - it's injected
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "resource": {"type": "string"}
            },
            "required": ["action", "resource"]
        }

    def run(self, action: str, resource: str, user_id: str = None) -> ToolResult:
        # user_id comes from context injection
        if user_id is None:
            return ToolResult(success=False, error="No user_id in context")
        return ToolResult(
            success=True,
            output={"action": action, "resource": resource, "user_id": user_id}
        )


class MultiContextTool(AgentTool):
    """Tool with multiple context parameters."""

    # Note: Can only inject one parameter via 'context' attribute
    # Additional context must be passed through context parameter

    @property
    def name(self) -> str:
        return "multi_context_tool"

    @property
    def description(self) -> str:
        return "A tool that uses context"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"data": {"type": "string"}}
        }

    def run(self, data: str) -> ToolResult:
        return ToolResult(success=True, output=data)


class NoContextTool(AgentTool):
    """Tool without context injection."""

    context = None  # Explicitly no context

    @property
    def name(self) -> str:
        return "no_context_tool"

    @property
    def description(self) -> str:
        return "A tool without context"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"input": {"type": "string"}}
        }

    def run(self, input: str) -> ToolResult:
        return ToolResult(success=True, output=input)


class MockAgent:
    """Mock agent for testing."""
    pass


class TestToolCallContext:
    """Tests for ToolCallContext dataclass."""

    def test_tool_call_context_creation(self):
        """Test ToolCallContext creation."""
        agent = MockAgent()
        ctx = ToolCallContext(agent=agent)
        assert ctx.agent is agent


class TestToolContextProperty:
    """Tests for tool context property."""

    def test_context_tool_has_context(self):
        """Test that context tool has context set."""
        tool = ContextTool()
        assert tool.context == "user_id"

    def test_no_context_tool_has_none(self):
        """Test that no-context tool has context=None."""
        tool = NoContextTool()
        assert tool.context is None

    def test_default_context_is_none(self):
        """Test that default context is None."""
        class DefaultTool(AgentTool):
            @property
            def name(self): return "default"

            @property
            def description(self): return "default"

            @property
            def parameters_schema(self): return {}

            def run(self, **kwargs): return ToolResult(True)

        tool = DefaultTool()
        assert tool.context is None


class TestAgentContextToolCallContext:
    """Tests for AgentContext tool_call_context."""

    def test_tool_call_context_initially_none(self):
        """Test that tool_call_context is initially None."""
        context = AgentContext()
        assert context.tool_call_context is None

    def test_tool_call_context_settable(self):
        """Test that tool_call_context can be set."""
        context = AgentContext()
        agent = MockAgent()

        context.tool_call_context = ToolCallContext(agent=agent)

        assert context.tool_call_context is not None
        assert context.tool_call_context.agent is agent

    def test_tool_call_context_in_copy(self):
        """Test that tool_call_context is not copied."""
        context = AgentContext()
        agent = MockAgent()
        context.tool_call_context = ToolCallContext(agent=agent)

        copied = context.copy()

        # tool_call_context should not be in the copy (it's marked as
        # compare=False in the dataclass)
        assert copied.tool_call_context is None


class TestContextInjectionScenario:
    """Tests for context injection scenarios."""

    def test_context_parameter_hidden_from_schema(self):
        """Test that context parameter is hidden from ToolDefinition."""
        from hawi.agent.messages import ToolDefinition

        tool = ContextTool()

        # The schema should NOT contain user_id
        schema = tool.parameters_schema
        assert "user_id" not in schema["properties"]
        assert "action" in schema["properties"]
        assert "resource" in schema["properties"]

    def test_context_injection_simulation(self):
        """Simulate context injection flow."""
        tool = ContextTool()

        # LLM provides these parameters
        llm_params = {"action": "read", "resource": "document.txt"}

        # Context provides this parameter (injected by HawiAgent)
        context_params = {"user_id": "user-123"}

        # Merge parameters
        merged_params = {**llm_params, **context_params}

        result = tool.run(**merged_params)

        assert result.success is True
        assert result.output["user_id"] == "user-123"
        assert result.output["action"] == "read"

    def test_context_injection_missing_context(self):
        """Test tool behavior when context is missing."""
        tool = ContextTool()

        # Only LLM params, no context injection
        llm_params = {"action": "read", "resource": "document.txt"}

        result = tool.run(**llm_params)

        assert result.success is False
        assert "No user_id" in result.error

    def test_context_with_default_value(self):
        """Test tool with context parameter having default value."""
        class ContextWithDefaultTool(AgentTool):
            context = "tenant_id"

            @property
            def name(self): return "tenant_tool"

            @property
            def description(self): return "Tool with tenant context"

            @property
            def parameters_schema(self):
                return {
                    "type": "object",
                    "properties": {"data": {"type": "string"}}
                }

            def run(self, data: str, tenant_id: str = "default_tenant") -> ToolResult:
                return ToolResult(success=True, output={"data": data, "tenant": tenant_id})

        tool = ContextWithDefaultTool()

        # Without context injection, uses default
        result = tool.run(data="test")
        assert result.output["tenant"] == "default_tenant"

        # With context injection, uses injected value
        result = tool.run(data="test", tenant_id="injected_tenant")
        assert result.output["tenant"] == "injected_tenant"


class TestToolTimeout:
    """Tests for tool timeout property."""

    def test_default_timeout_is_none(self):
        """Test that default timeout is None."""
        tool = ContextTool()
        assert tool.timeout is None

    def test_custom_timeout(self):
        """Test setting custom timeout."""
        class TimeoutTool(AgentTool):
            timeout = 30.0

            @property
            def name(self): return "timeout_tool"

            @property
            def description(self): return "Tool with timeout"

            @property
            def parameters_schema(self): return {}

            def run(self, **kwargs): return ToolResult(True)

        tool = TimeoutTool()
        assert tool.timeout == 30.0


class TestToolTags:
    """Tests for tool tags property."""

    def test_default_tags_is_empty(self):
        """Test that default tags is empty list."""
        tool = ContextTool()
        assert tool.tags == []

    def test_custom_tags(self):
        """Test setting custom tags."""
        class TaggedTool(AgentTool):
            tags = ["database", "write", "dangerous"]

            @property
            def name(self): return "tagged_tool"

            @property
            def description(self): return "Tool with tags"

            @property
            def parameters_schema(self): return {}

            def run(self, **kwargs): return ToolResult(True)

        tool = TaggedTool()
        assert tool.tags == ["database", "write", "dangerous"]

    def test_tags_filtering(self):
        """Test filtering tools by tags."""
        class Tool1(AgentTool):
            tags = ["read", "safe"]
            @property
            def name(self): return "tool1"
            @property
            def description(self): return ""
            @property
            def parameters_schema(self): return {}
            def run(self, **kwargs): return ToolResult(True)

        class Tool2(AgentTool):
            tags = ["write", "dangerous"]
            @property
            def name(self): return "tool2"
            @property
            def description(self): return ""
            @property
            def parameters_schema(self): return {}
            def run(self, **kwargs): return ToolResult(True)

        tools = [Tool1(), Tool2()]

        # Filter by tag
        dangerous_tools = [t for t in tools if "dangerous" in t.tags]
        assert len(dangerous_tools) == 1
        assert dangerous_tools[0].name == "tool2"

        safe_tools = [t for t in tools if "safe" in t.tags]
        assert len(safe_tools) == 1
        assert safe_tools[0].name == "tool1"
