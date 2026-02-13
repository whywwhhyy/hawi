"""Unit tests for tool call audit functionality.

Tests the audit mechanism for tool calls including pending tool call tracking,
approval/rejection workflow, and integration with AgentContext.
"""

import pytest
from hawi.agent.context import AgentContext
from hawi.tool.types import AgentTool, ToolResult, PendingToolCall


class AuditTool(AgentTool):
    """Tool with audit enabled."""

    audit = True

    @property
    def name(self) -> str:
        return "audit_tool"

    @property
    def description(self) -> str:
        return "A tool that requires audit"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "target": {"type": "string"}
            },
            "required": ["command"]
        }

    def run(self, command: str, target: str = "") -> ToolResult:
        return ToolResult(success=True, output=f"Executed: {command} on {target}")


class NonAuditTool(AgentTool):
    """Tool without audit."""

    audit = False

    @property
    def name(self) -> str:
        return "non_audit_tool"

    @property
    def description(self) -> str:
        return "A tool that does not require audit"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"input": {"type": "string"}}
        }

    def run(self, input: str) -> ToolResult:
        return ToolResult(success=True, output=f"Result: {input}")


class TestPendingToolCall:
    """Tests for PendingToolCall dataclass."""

    def test_pending_tool_call_creation(self):
        """Test basic PendingToolCall creation."""
        pending = PendingToolCall(
            tool_call_id="tc-123",
            tool_name="test_tool",
            arguments={"key": "value"},
            requested_at=1234567890.0
        )
        assert pending.tool_call_id == "tc-123"
        assert pending.tool_name == "test_tool"
        assert pending.arguments == {"key": "value"}
        assert pending.requested_at == 1234567890.0

    def test_pending_tool_call_default_timestamp(self):
        """Test that requested_at defaults to current time."""
        import time
        before = time.time()
        pending = PendingToolCall(
            tool_call_id="tc-456",
            tool_name="test_tool",
            arguments={}
        )
        after = time.time()
        assert before <= pending.requested_at <= after


class TestAgentContextAudit:
    """Tests for audit functionality in AgentContext."""

    def test_add_pending_tool_call(self):
        """Test adding a pending tool call."""
        context = AgentContext()

        pending = context._add_pending_tool_call(
            tool_call_id="tc-001",
            tool_name="dangerous_command",
            arguments={"command": "rm -rf /"}
        )

        assert isinstance(pending, PendingToolCall)
        assert pending.tool_call_id == "tc-001"
        assert pending.tool_name == "dangerous_command"
        assert pending.arguments["command"] == "rm -rf /"

    def test_get_pending_tool_calls(self):
        """Test getting all pending tool calls."""
        context = AgentContext()

        # Initially empty
        assert context.get_pending_tool_calls() == []

        # Add some pending calls
        context._add_pending_tool_call("tc-001", "tool1", {})
        context._add_pending_tool_call("tc-002", "tool2", {})

        pending = context.get_pending_tool_calls()
        assert len(pending) == 2
        assert all(isinstance(p, PendingToolCall) for p in pending)

    def test_audit_pending_tool_calls_approve(self):
        """Test approving pending tool calls."""
        context = AgentContext()

        context._add_pending_tool_call("tc-001", "tool1", {})
        context._add_pending_tool_call("tc-002", "tool2", {})
        context._add_pending_tool_call("tc-003", "tool3", {})

        approved, rejected = context.audit_pending_tool_calls(
            approve=["tc-001", "tc-003"]
        )

        assert len(approved) == 2
        assert len(rejected) == 0
        assert {p.tool_call_id for p in approved} == {"tc-001", "tc-003"}

        # tc-002 should still be pending
        pending = context.get_pending_tool_calls()
        assert len(pending) == 1
        assert pending[0].tool_call_id == "tc-002"

    def test_audit_pending_tool_calls_reject(self):
        """Test rejecting pending tool calls."""
        context = AgentContext()

        context._add_pending_tool_call("tc-001", "tool1", {})
        context._add_pending_tool_call("tc-002", "tool2", {})

        approved, rejected = context.audit_pending_tool_calls(
            reject=["tc-001"]
        )

        assert len(approved) == 0
        assert len(rejected) == 1
        assert rejected[0].tool_call_id == "tc-001"

        # tc-002 should still be pending
        pending = context.get_pending_tool_calls()
        assert len(pending) == 1
        assert pending[0].tool_call_id == "tc-002"

    def test_audit_pending_tool_calls_both(self):
        """Test approving and rejecting simultaneously."""
        context = AgentContext()

        context._add_pending_tool_call("tc-001", "tool1", {})
        context._add_pending_tool_call("tc-002", "tool2", {})
        context._add_pending_tool_call("tc-003", "tool3", {})

        approved, rejected = context.audit_pending_tool_calls(
            approve=["tc-001"],
            reject=["tc-002"]
        )

        assert len(approved) == 1
        assert len(rejected) == 1
        assert approved[0].tool_call_id == "tc-001"
        assert rejected[0].tool_call_id == "tc-002"

        # tc-003 should still be pending
        pending = context.get_pending_tool_calls()
        assert len(pending) == 1
        assert pending[0].tool_call_id == "tc-003"

    def test_audit_nonexistent_tool_call(self):
        """Test auditing a tool call that doesn't exist."""
        context = AgentContext()

        context._add_pending_tool_call("tc-001", "tool1", {})

        approved, rejected = context.audit_pending_tool_calls(
            approve=["nonexistent"]
        )

        assert len(approved) == 0
        assert len(rejected) == 0
        # Original should still be there
        assert len(context.get_pending_tool_calls()) == 1

    def test_clear_pending_tool_calls(self):
        """Test clearing all pending tool calls."""
        context = AgentContext()

        context._add_pending_tool_call("tc-001", "tool1", {})
        context._add_pending_tool_call("tc-002", "tool2", {})

        context.clear_pending_tool_calls()

        assert context.get_pending_tool_calls() == []


class TestToolAuditProperty:
    """Tests for tool audit property."""

    def test_audit_tool_has_audit_true(self):
        """Test that audit tool has audit=True."""
        tool = AuditTool()
        assert tool.audit is True

    def test_non_audit_tool_has_audit_false(self):
        """Test that non-audit tool has audit=False."""
        tool = NonAuditTool()
        assert tool.audit is False

    def test_default_audit_is_false(self):
        """Test that default audit is False."""
        class DefaultTool(AgentTool):
            @property
            def name(self): return "default"

            @property
            def description(self): return "default"

            @property
            def parameters_schema(self): return {}

            def run(self, **kwargs): return ToolResult(True)

        tool = DefaultTool()
        assert tool.audit is False


class TestAuditWorkflow:
    """Tests for complete audit workflow."""

    def test_audit_workflow_scenario(self):
        """Test a complete audit workflow scenario."""
        context = AgentContext(tools=[AuditTool()])

        # Simulate tool call requests
        tool_calls = [
            ("tc-001", "audit_tool", {"command": "read", "target": "file1.txt"}),
            ("tc-002", "audit_tool", {"command": "write", "target": "file2.txt"}),
            ("tc-003", "audit_tool", {"command": "delete", "target": "file3.txt"}),
        ]

        # Add all to pending
        for tc_id, tool_name, args in tool_calls:
            context._add_pending_tool_call(tc_id, tool_name, args)

        assert len(context.get_pending_tool_calls()) == 3

        # Review and decide: approve read, reject delete, leave write pending
        approved, rejected = context.audit_pending_tool_calls(
            approve=["tc-001"],
            reject=["tc-003"]
        )

        # Verify approved
        assert len(approved) == 1
        assert approved[0].tool_call_id == "tc-001"
        assert approved[0].arguments["command"] == "read"

        # Verify rejected
        assert len(rejected) == 1
        assert rejected[0].tool_call_id == "tc-003"
        assert rejected[0].arguments["command"] == "delete"

        # Verify pending
        pending = context.get_pending_tool_calls()
        assert len(pending) == 1
        assert pending[0].tool_call_id == "tc-002"
        assert pending[0].arguments["command"] == "write"

    def test_audit_with_complex_arguments(self):
        """Test audit with complex nested arguments."""
        context = AgentContext()

        complex_args = {
            "command": "update",
            "config": {
                "database": "production",
                "tables": ["users", "orders"],
                "options": {"backup": True, "verify": False}
            },
            "timestamp": 1234567890
        }

        context._add_pending_tool_call("tc-001", "config_tool", complex_args)

        pending = context.get_pending_tool_calls()[0]
        assert pending.arguments["config"]["database"] == "production"
        assert pending.arguments["config"]["tables"] == ["users", "orders"]
