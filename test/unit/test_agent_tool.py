"""Unit tests for AgentTool and ToolResult."""

import asyncio
import pytest
from hawi.tool import AgentTool, ToolResult, tool


class TestToolResult:
    """Test cases for ToolResult class."""

    def test_tool_result_basic(self):
        """Test basic ToolResult creation."""
        result = ToolResult(success=True, output="test output")
        assert result.success is True
        assert result.output == "test output"

    def test_tool_result_with_none_output(self):
        """Test ToolResult with None output."""
        result = ToolResult(success=True)
        assert result.success is True
        assert result.output is None

    def test_tool_result_with_various_types(self):
        """Test ToolResult with different output types."""
        # String
        assert ToolResult(True, "string").output == "string"
        # Integer
        assert ToolResult(True, 42).output == 42
        # Float
        assert ToolResult(True, 3.14).output == 3.14
        # Boolean
        assert ToolResult(True, True).output is True
        # List
        assert ToolResult(True, [1, 2, 3]).output == [1, 2, 3]
        # Dict
        assert ToolResult(True, {"key": "value"}).output == {"key": "value"}
        # None
        assert ToolResult(True, None).output is None

    def test_tool_result_error(self):
        """Test ToolResult for error case."""
        result = ToolResult(success=False, error="error message")
        assert result.success is False
        assert result.error == "error message"

    def test_tool_result_repr(self):
        """Test ToolResult string representation."""
        result = ToolResult(success=True, output="test")
        assert repr(result) == "ToolResult(success=True, output='test')"


class TestAgentToolBasic:
    """Test cases for basic AgentTool functionality."""

    def test_abstract_class_cannot_instantiate(self):
        """Test that AgentTool is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            AgentTool()

    def test_required_properties(self):
        """Test that subclasses must implement required properties."""

        class IncompleteTool(AgentTool):
            @property
            def name(self):
                return "test"

        with pytest.raises(TypeError):
            IncompleteTool()

    def test_tool_with_only_sync(self):
        """Test tool that only implements sync execution."""

        class SyncTool(AgentTool):
            @property
            def name(self):
                return "sync_tool"

            @property
            def description(self):
                return "A sync tool"

            @property
            def parameters_schema(self):
                return {
                    "type": "object",
                    "properties": {
                        "value": {"type": "integer"}
                    },
                    "required": ["value"]
                }

            def run(self, value: int):
                return ToolResult(success=True, output=value * 2)

        tool = SyncTool()
        assert tool.name == "sync_tool"
        assert tool.description == "A sync tool"
        assert tool.supports_sync is True
        assert tool.supports_async is False  # Not natively implemented

    def test_tool_with_only_async(self):
        """Test tool that only implements async execution."""

        class AsyncTool(AgentTool):
            @property
            def name(self):
                return "async_tool"

            @property
            def description(self):
                return "An async tool"

            @property
            def parameters_schema(self):
                return {"type": "object", "properties": {}}

            async def arun(self, **kwargs):
                await asyncio.sleep(0)
                return ToolResult(success=True, output="async result")

        tool = AsyncTool()
        assert tool.name == "async_tool"
        assert tool.supports_sync is False  # Not natively implemented
        assert tool.supports_async is True

    def test_tool_with_both(self):
        """Test tool that implements both sync and async execution."""

        class BothTool(AgentTool):
            @property
            def name(self):
                return "both_tool"

            @property
            def description(self):
                return "Both sync and async tool"

            @property
            def parameters_schema(self):
                return {"type": "object", "properties": {}}

            def run(self, **kwargs):
                return ToolResult(success=True, output="sync")

            async def arun(self, **kwargs):
                return ToolResult(success=True, output="async")

        tool = BothTool()
        assert tool.supports_sync is True
        assert tool.supports_async is True


class TestAgentToolInvoke:
    """Test cases for AgentTool invoke methods."""

    def test_sync_invoke(self):
        """Test synchronous invoke."""

        class AddTool(AgentTool):
            @property
            def name(self):
                return "add"

            @property
            def description(self):
                return "Add two numbers"

            @property
            def parameters_schema(self):
                return {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"}
                    },
                    "required": ["a", "b"]
                }

            def run(self, a: int, b: int):
                return ToolResult(success=True, output=a + b)

        tool = AddTool()
        result = tool.invoke({"a": 2, "b": 3})
        assert result.success is True
        assert result.output == 5

    def test_async_invoke(self):
        """Test asynchronous invoke."""

        class MultiplyTool(AgentTool):
            @property
            def name(self):
                return "multiply"

            @property
            def description(self):
                return "Multiply two numbers"

            @property
            def parameters_schema(self):
                return {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"}
                    },
                    "required": ["x", "y"]
                }

            async def arun(self, x: float, y: float):
                await asyncio.sleep(0)
                return ToolResult(success=True, output=x * y)

        tool = MultiplyTool()

        async def test():
            result = await tool.ainvoke({"x": 2.5, "y": 4})
            assert result.success is True
            assert result.output == 10.0

        asyncio.run(test())

    def test_invoke_validation_failure(self):
        """Test that invoke returns error on validation failure."""

        class StrictTool(AgentTool):
            @property
            def name(self):
                return "strict"

            @property
            def description(self):
                return "Strict validation tool"

            @property
            def parameters_schema(self):
                return {
                    "type": "object",
                    "properties": {
                        "required_param": {"type": "string"}
                    },
                    "required": ["required_param"]
                }

            def run(self, required_param: str):
                return ToolResult(success=True, output=required_param)

        tool = StrictTool()
        result = tool.invoke({})  # Missing required parameter
        assert result.success is False
        assert "Parameter validation failed" in result.error

    def test_invoke_execution_error(self):
        """Test that invoke handles execution errors."""

        class ErrorTool(AgentTool):
            @property
            def name(self):
                return "error_tool"

            @property
            def description(self):
                return "Always errors"

            @property
            def parameters_schema(self):
                return {"type": "object", "properties": {}}

            def run(self, **kwargs):
                raise ValueError("Something went wrong")

        tool = ErrorTool()
        result = tool.invoke({})
        assert result.success is False
        assert "ValueError: Something went wrong" in result.error

    def test_callable_via_invoke(self):
        """Test that tool can be invoked via invoke method."""

        class EchoTool(AgentTool):
            @property
            def name(self):
                return "echo"

            @property
            def description(self):
                return "Echo input"

            @property
            def parameters_schema(self):
                return {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"}
                    },
                    "required": ["message"]
                }

            def run(self, message: str):
                return ToolResult(success=True, output=message)

        tool = EchoTool()
        result = tool.invoke({"message": "hello"})
        assert result.success is True
        assert result.output == "hello"


class TestAgentToolProperties:
    """Test cases for AgentTool optional properties."""

    def test_default_properties(self):
        """Test default values for optional properties."""

        class DefaultTool(AgentTool):
            @property
            def name(self):
                return "default"

            @property
            def description(self):
                return "Default tool"

            @property
            def parameters_schema(self):
                return {"type": "object", "properties": {}}

            def run(self, **kwargs):
                return ToolResult(True)

        tool = DefaultTool()
        assert tool.audit is False
        assert tool.context is None
        assert tool.timeout is None
        assert tool.tags == []

    def test_custom_properties(self):
        """Test custom values for optional properties."""

        class ConfiguredTool(AgentTool):
            audit = True
            context = "user_id"
            timeout = 30.0
            tags = ["sensitive", "write"]

            @property
            def name(self):
                return "configured"

            @property
            def description(self):
                return "Configured tool"

            @property
            def parameters_schema(self):
                return {"type": "object", "properties": {}}

            def run(self, **kwargs):
                return ToolResult(True)

        tool = ConfiguredTool()
        assert tool.audit is True
        assert tool.context == "user_id"
        assert tool.timeout == 30.0
        assert tool.tags == ["sensitive", "write"]


class TestToolDecorator:
    """Test cases for @tool decorator."""

    def test_tool_decorator_sync(self):
        """Test @tool decorator with sync function."""

        @tool
        def calculate(x: int, y: int) -> int:
            """Calculate sum."""
            return x + y

        assert calculate.name == "calculate"
        assert calculate.description == "Calculate sum."
        assert "x" in calculate.parameters_schema["properties"]
        assert "y" in calculate.parameters_schema["properties"]

        result = calculate.invoke({"x": 1, "y": 2})
        assert result.success is True
        assert result.output == 3

    def test_tool_decorator_async(self):
        """Test @tool decorator with async function."""

        @tool
        async def async_multiply(a: float, b: float) -> float:
            """Multiply numbers."""
            return a * b

        assert async_multiply.name == "async_multiply"
        assert async_multiply.supports_async is True

        async def test():
            result = await async_multiply.ainvoke({"a": 2.5, "b": 4})
            assert result.success is True
            assert result.output == 10.0

        asyncio.run(test())

    def test_tool_decorator_with_params(self):
        """Test @tool decorator with custom parameters."""

        @tool(name="custom_name", description="Custom description")
        def my_func():
            """This docstring is overridden."""
            return "ok"

        assert my_func.name == "custom_name"
        assert my_func.description == "Custom description"

    def test_tool_decorator_with_no_docstring(self):
        """Test @tool decorator with function that has no docstring."""

        @tool
        def no_doc(x: int) -> int:
            return x

        assert no_doc.description == ""

    def test_tool_decorator_return_dict(self):
        """Test @tool decorator with function returning dict."""

        @tool
        def get_user(user_id: int):
            return {"id": user_id, "name": "Test User"}

        result = get_user.invoke({"user_id": 123})
        assert result.success is True
        assert result.output == {"id": 123, "name": "Test User"}

    def test_tool_decorator_with_default_params(self):
        """Test @tool decorator with default parameters."""

        @tool
        def search(query: str, limit: int = 10):
            return f"Search {query} with limit {limit}"

        result = search.invoke({"query": "test"})
        assert result.success is True
        assert result.output == "Search test with limit 10"

        result = search.invoke({"query": "test", "limit": 5})
        assert result.output == "Search test with limit 5"


class TestAgentToolValidation:
    """Test cases for parameter validation."""

    def test_validate_parameters_success(self):
        """Test successful parameter validation."""

        class ValidatedTool(AgentTool):
            @property
            def name(self):
                return "validated"

            @property
            def description(self):
                return "Validated tool"

            @property
            def parameters_schema(self):
                return {
                    "type": "object",
                    "properties": {
                        "count": {"type": "integer", "minimum": 0},
                        "name": {"type": "string", "minLength": 1}
                    },
                    "required": ["count", "name"]
                }

            def run(self, count: int, name: str):
                return ToolResult(True, f"{name}: {count}")

        tool = ValidatedTool()
        is_valid, errors = tool.validate_parameters({"count": 5, "name": "test"})
        assert is_valid is True
        assert errors == []

    def test_validate_parameters_failure(self):
        """Test failed parameter validation."""

        class ValidatedTool(AgentTool):
            @property
            def name(self):
                return "validated"

            @property
            def description(self):
                return "Validated tool"

            @property
            def parameters_schema(self):
                return {
                    "type": "object",
                    "properties": {
                        "count": {"type": "integer", "minimum": 0}
                    },
                    "required": ["count"]
                }

            def run(self, count: int):
                return ToolResult(True, str(count))

        tool = ValidatedTool()
        is_valid, errors = tool.validate_parameters({"count": -1})
        assert is_valid is False
        assert len(errors) > 0

    def test_validate_parameters_missing_required(self):
        """Test validation with missing required parameter."""

        class RequiredTool(AgentTool):
            @property
            def name(self):
                return "required"

            @property
            def description(self):
                return "Required tool"

            @property
            def parameters_schema(self):
                return {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"}
                    },
                    "required": ["value"]
                }

            def run(self, value: str):
                return ToolResult(True, value)

        tool = RequiredTool()
        is_valid, errors = tool.validate_parameters({})
        assert is_valid is False
        assert any("required" in error.lower() for error in errors)


class TestAgentToolRepr:
    """Test cases for string representation."""

    def test_repr(self):
        """Test __repr__ method."""

        class TestTool(AgentTool):
            @property
            def name(self):
                return "test_tool"

            @property
            def description(self):
                return "Test"

            @property
            def parameters_schema(self):
                return {"type": "object", "properties": {}}

            def run(self, **kwargs):
                return ToolResult(True)

        tool = TestTool()
        assert repr(tool) == "TestTool(name='test_tool')"
