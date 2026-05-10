"""Agent tools for Hawi core sub-agent management."""

from __future__ import annotations

from typing import Any, Literal

from hawi.agent import (
    SubAgentLimits,
    SubAgentPluginPolicy,
    SubAgentSpec,
    ToolCallContext,
)
from hawi.plugin import HawiPlugin, tool


class SubAgentPlugin(HawiPlugin):
    """Expose a small lifecycle tool set for managed sub-agents."""

    @classmethod
    def gui_config_schema(cls) -> dict:
        return {"type": "object", "properties": {}, "additionalProperties": False}

    @classmethod
    def gui_default_config(cls) -> dict:
        return {}

    @tool(
        name="create_subagent",
        context="ctx",
        tags=["subagent", "orchestration"],
        parameters_schema={
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["fork", "fresh"],
                    "default": "fork",
                    "description": "fork copies parent context; fresh starts with no parent messages.",
                },
                "name": {"type": "string"},
                "role": {
                    "type": "string",
                    "default": "general",
                    "description": "general, planner, reviewer, explorer, implementer, critic, or summarizer.",
                },
                "model": {"type": "string"},
                "system_prompt": {"type": "string"},
                "working_dir": {"type": "string"},
                "initial_prompt": {"type": "string"},
                "initial_plan": {},
                "inherit_plugins": {"type": "boolean", "default": True},
                "max_runtime_seconds": {"type": "number"},
                "max_iterations": {"type": "integer"},
                "result_contract": {
                    "type": "string",
                    "default": "text",
                    "description": "text, json, plan, review, diff, or artifact.",
                },
                "ownership": {"type": "object"},
                "metadata": {"type": "object"},
            },
        },
    )
    async def create_subagent(
        self,
        mode: Literal["fork", "fresh"] = "fork",
        name: str | None = None,
        role: str = "general",
        model: str | None = None,
        system_prompt: str | None = None,
        working_dir: str | None = None,
        initial_prompt: str | None = None,
        initial_plan: Any | None = None,
        inherit_plugins: bool = True,
        max_runtime_seconds: float | None = None,
        max_iterations: int | None = None,
        result_contract: str = "text",
        ownership: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        ctx: ToolCallContext | None = None,
    ) -> dict[str, Any]:
        """Create a background sub-agent and optionally enqueue its first task."""
        if ctx is None:
            raise RuntimeError("create_subagent requires Hawi tool context")
        handle = await ctx.agent.subagents.spawn(
            SubAgentSpec(
                mode=mode,
                name=name,
                role=role,
                model=model,
                system_prompt=system_prompt,
                plugin_policy=SubAgentPluginPolicy(inherit=inherit_plugins),
                working_dir=working_dir,
                initial_prompt=initial_prompt,
                initial_plan=initial_plan,
                limits=SubAgentLimits(
                    max_runtime_seconds=max_runtime_seconds,
                    max_iterations=max_iterations,
                ),
                result_contract=result_contract,
                ownership=ownership or {},
                metadata=metadata or {},
            )
        )
        return {
            "subagent_id": handle.id,
            "status": ctx.agent.subagents.status(handle.id).to_dict(),
        }

    @tool(
        name="send_subagent_message",
        context="ctx",
        tags=["subagent", "orchestration"],
        parameters_schema={
            "type": "object",
            "properties": {
                "subagent_id": {"type": "string"},
                "message": {"type": "string"},
                "queue": {
                    "type": "string",
                    "enum": ["normal", "high_prio", "urgent"],
                    "default": "normal",
                },
                "metadata": {"type": "object"},
            },
            "required": ["subagent_id", "message"],
        },
    )
    async def send_subagent_message(
        self,
        subagent_id: str,
        message: str,
        queue: Literal["normal", "high_prio", "urgent"] = "normal",
        metadata: dict[str, Any] | None = None,
        ctx: ToolCallContext | None = None,
    ) -> dict[str, Any]:
        """Send follow-up guidance or material to a sub-agent."""
        if ctx is None:
            raise RuntimeError("send_subagent_message requires Hawi tool context")
        message_id = ctx.agent.subagents.send(
            subagent_id,
            message,
            queue=queue,
            metadata=metadata,
        )
        return {
            "message_id": message_id,
            "status": ctx.agent.subagents.status(subagent_id).to_dict(),
        }

    @tool(
        name="read_subagent",
        context="ctx",
        tags=["subagent", "orchestration"],
        parameters_schema={
            "type": "object",
            "properties": {
                "subagent_id": {"type": "string"},
                "view": {
                    "type": "string",
                    "enum": ["status", "summary", "events", "context_tail"],
                    "default": "summary",
                },
                "limit": {"type": "integer", "default": 20},
            },
            "required": ["subagent_id"],
        },
    )
    async def read_subagent(
        self,
        subagent_id: str,
        view: Literal["status", "summary", "events", "context_tail"] = "summary",
        limit: int = 20,
        ctx: ToolCallContext | None = None,
    ) -> dict[str, Any]:
        """Read sub-agent status, events, or recent context."""
        if ctx is None:
            raise RuntimeError("read_subagent requires Hawi tool context")
        return ctx.agent.subagents.read(subagent_id, view=view, limit=limit)

    @tool(
        name="close_subagent",
        context="ctx",
        tags=["subagent", "orchestration"],
        parameters_schema={
            "type": "object",
            "properties": {
                "subagent_id": {"type": "string"},
                "reason": {"type": "string", "default": "done"},
                "interrupt": {"type": "boolean", "default": True},
            },
            "required": ["subagent_id"],
        },
    )
    async def close_subagent(
        self,
        subagent_id: str,
        reason: str = "done",
        interrupt: bool = True,
        ctx: ToolCallContext | None = None,
    ) -> dict[str, Any]:
        """Interrupt/cancel and close a sub-agent."""
        if ctx is None:
            raise RuntimeError("close_subagent requires Hawi tool context")
        status = await ctx.agent.subagents.close(
            subagent_id,
            reason=reason,
            interrupt=interrupt,
        )
        return {"status": status.to_dict()}
