"""Tests for framework-level tool parameter injection."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from hawi.agent import HawiAgent
from hawi.agent.agent import _ExecutionState
from hawi.tool import AgentTool, ToolParameterInjection, ToolResult


class StrictEchoTool(AgentTool):
    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    def name(self) -> str:
        return "strict_echo"

    @property
    def description(self) -> str:
        return "Echo text with strict parameters"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        }

    def run(self, text: str) -> ToolResult:  # type: ignore[override]
        self.calls.append(text)
        return ToolResult(success=True, output=text)


class AuditCommandTool(AgentTool):
    audit = True

    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    def name(self) -> str:
        return "audit_command"

    @property
    def description(self) -> str:
        return "Run a command after approval"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
            "additionalProperties": False,
        }

    def run(self, command: str) -> ToolResult:  # type: ignore[override]
        self.calls.append(command)
        return ToolResult(success=True, output=f"ran {command}")


@pytest.fixture
def agent() -> HawiAgent:
    return HawiAgent(model=MagicMock())


def approval_reason_injection(**kwargs: Any) -> ToolParameterInjection:
    return ToolParameterInjection(
        name="approval_reason",
        schema={
            "type": "string",
            "description": "Explain why this tool call needs human approval.",
            "minLength": 1,
        },
        required=True,
        **kwargs,
    )


def test_tool_definition_schema_is_augmented_without_mutating_tool_schema(agent: HawiAgent):
    tool = StrictEchoTool()
    agent.plugins.add_tool(tool)
    agent.plugins.add_tool_parameter_injection(approval_reason_injection())

    definitions = agent.plugins.get_tool_definitions()

    assert definitions[0]["schema"]["properties"]["approval_reason"]["type"] == "string"
    assert definitions[0]["schema"]["required"] == ["text", "approval_reason"]
    assert definitions[0]["schema"]["additionalProperties"] is False
    assert "approval_reason" not in tool.parameters_schema["properties"]


async def test_injected_parameter_is_stripped_before_tool_execution(agent: HawiAgent):
    tool = StrictEchoTool()
    seen: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = []

    def handler(ctx, value):
        seen.append(
            (
                ctx.tool_name,
                value,
                dict(ctx.arguments),
                dict(ctx.injected_arguments),
            )
        )
        return None

    agent.plugins.add_tool(tool)
    agent.plugins.add_tool_parameter_injection(
        approval_reason_injection(handler=handler)
    )

    record = await agent._execute_tool(
        {
            "type": "tool_call",
            "id": "tc-1",
            "name": "strict_echo",
            "arguments": {
                "text": "hello",
                "approval_reason": "Need to show the user the result.",
            },
        },
        _ExecutionState(run_id="run-1", iteration=1),
    )

    assert record.result.success is True
    assert record.result.output == "hello"
    assert tool.calls == ["hello"]
    assert seen == [
        (
            "strict_echo",
            "Need to show the user the result.",
            {
                "text": "hello",
                "approval_reason": "Need to show the user the result.",
            },
            {"approval_reason": "Need to show the user the result."},
        )
    ]


async def test_missing_required_injected_parameter_fails_before_tool(agent: HawiAgent):
    tool = StrictEchoTool()
    agent.plugins.add_tool(tool)
    agent.plugins.add_tool_parameter_injection(approval_reason_injection())

    record = await agent._execute_tool(
        {
            "type": "tool_call",
            "id": "tc-1",
            "name": "strict_echo",
            "arguments": {"text": "hello"},
        },
        _ExecutionState(run_id="run-1", iteration=1),
    )

    assert record.result.success is False
    assert "Injected parameter validation failed" in record.result.error
    assert tool.calls == []


async def test_injected_parameter_handler_failure_returns_tool_result(agent: HawiAgent):
    tool = StrictEchoTool()

    def handler(ctx, value):
        raise RuntimeError("audit service unavailable")

    agent.plugins.add_tool(tool)
    agent.plugins.add_tool_parameter_injection(
        approval_reason_injection(handler=handler)
    )

    record = await agent._execute_tool(
        {
            "type": "tool_call",
            "id": "tc-1",
            "name": "strict_echo",
            "arguments": {
                "text": "hello",
                "approval_reason": "Need to show the user the result.",
            },
        },
        _ExecutionState(run_id="run-1", iteration=1),
    )

    assert record.result.success is False
    assert "Injected parameter handler failed" in record.result.error
    assert "audit service unavailable" in record.result.error
    assert tool.calls == []


async def test_audit_review_keeps_injected_parameter_but_approval_strips_it(agent: HawiAgent):
    tool = AuditCommandTool()
    agent.plugins.add_tool(tool)
    agent.plugins.add_tool_parameter_injection(approval_reason_injection())

    pending_record = await agent._execute_tool(
        {
            "type": "tool_call",
            "id": "tc-approve",
            "name": "audit_command",
            "arguments": {
                "command": "deploy",
                "approval_reason": "This changes production state.",
            },
        },
        _ExecutionState(run_id="run-1", iteration=1),
    )

    assert pending_record.result.success is True
    assert tool.calls == []
    pending = agent.review_pending_tools()
    assert pending[0]["arguments"]["approval_reason"] == "This changes production state."

    approved = await agent.approve_pending_tools(["tc-approve"])

    assert approved[0].result.success is True
    assert approved[0].result.output == "ran deploy"
    assert tool.calls == ["deploy"]


def test_injected_parameter_conflict_is_reported(agent: HawiAgent):
    tool = StrictEchoTool()
    agent.plugins.add_tool(tool)
    agent.plugins.add_tool_parameter_injection(
        ToolParameterInjection(
            name="text",
            schema={"type": "string"},
        )
    )

    with pytest.raises(ValueError, match="conflicts"):
        agent.plugins.get_tool_definitions()


def test_injected_parameter_can_be_scoped_to_matching_tools(agent: HawiAgent):
    tool = StrictEchoTool()
    audit_tool = AuditCommandTool()
    agent.plugins.add_tool(tool)
    agent.plugins.add_tool(audit_tool)
    agent.plugins.add_tool_parameter_injection(
        approval_reason_injection(applies_to=lambda t: bool(getattr(t, "audit", False)))
    )

    definitions = {
        definition["name"]: definition["schema"]
        for definition in agent.plugins.get_tool_definitions()
    }

    assert "approval_reason" not in definitions["strict_echo"]["properties"]
    assert "approval_reason" in definitions["audit_command"]["properties"]
