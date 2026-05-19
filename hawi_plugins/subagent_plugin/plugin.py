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

    name = "hawi/subagent"
    display_name = "Subagent"
    description = "创建和管理后台子 agent，支持分工、等待和状态查询。"
    dependencies = ()

    @property
    def permissions(self):
        from hawi.permission import PermissionDeclared, WELL_KNOWN_PERMISSIONS as WKP
        return [
            PermissionDeclared(
                permission=WKP["subagent:spawn"],
                tool_names=["create_subagent"],
            ),
            PermissionDeclared(
                permission=WKP["subagent:send"],
                tool_names=["send_subagent_message", "wait_subagent", "read_subagent"],
            ),
            PermissionDeclared(
                permission=WKP["subagent:close"],
                tool_names=["close_subagent"],
            ),
        ]

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
                "notify_timeout": {
                    "type": "number",
                    "default": 0,
                    "description": "Seconds to wait for the initial task before returning. 0 returns immediately.",
                },
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
        notify_timeout: float = 0,
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
            "wait": (
                await ctx.agent.subagents.wait_report(handle.id, timeout=notify_timeout)
                if notify_timeout > 0
                else None
            ),
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
                "notify_timeout": {
                    "type": "number",
                    "default": 0,
                    "description": "Seconds to wait for this queued task before returning. 0 returns immediately.",
                },
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
        notify_timeout: float = 0,
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
            "wait": (
                await ctx.agent.subagents.wait_report(subagent_id, timeout=notify_timeout)
                if notify_timeout > 0
                else None
            ),
        }

    @tool(
        name="wait_subagent",
        context="ctx",
        tags=["subagent", "orchestration"],
        parameters_schema={
            "type": "object",
            "properties": {
                "subagent_id": {"type": "string"},
                "notify_timeout": {
                    "type": "number",
                    "default": 30,
                    "description": "Maximum seconds to wait before returning a running status.",
                },
                "timeout_action": {
                    "type": "string",
                    "enum": ["status", "interrupt", "close"],
                    "default": "status",
                    "description": "status returns a running report; interrupt/close act on timeout.",
                },
            },
            "required": ["subagent_id"],
        },
    )
    async def wait_subagent(
        self,
        subagent_id: str,
        notify_timeout: float = 30,
        timeout_action: Literal["status", "interrupt", "close"] = "status",
        ctx: ToolCallContext | None = None,
    ) -> dict[str, Any]:
        """Wait for a sub-agent like waiting on a shell job."""
        if ctx is None:
            raise RuntimeError("wait_subagent requires Hawi tool context")
        return await ctx.agent.subagents.wait_report(
            subagent_id,
            timeout=notify_timeout,
            timeout_action=timeout_action,
        )

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
                    "enum": [
                        "status",
                        "summary",
                        "events",
                        "context_tail",
                        "markdown",
                        "export",
                        "ref",
                    ],
                    "default": "summary",
                },
                "limit": {"type": "integer", "default": 20},
                "ref_path": {
                    "type": "string",
                    "description": "Reference filename returned by view=export or view=markdown.",
                },
            },
            "required": ["subagent_id"],
        },
    )
    async def read_subagent(
        self,
        subagent_id: str,
        view: Literal[
            "status",
            "summary",
            "events",
            "context_tail",
            "markdown",
            "export",
            "ref",
        ] = "summary",
        limit: int = 20,
        ref_path: str | None = None,
        ctx: ToolCallContext | None = None,
    ) -> dict[str, Any]:
        """Read sub-agent status, events, or recent context."""
        if ctx is None:
            raise RuntimeError("read_subagent requires Hawi tool context")
        return ctx.agent.subagents.read(
            subagent_id,
            view=view,
            limit=limit,
            ref_path=ref_path,
        )

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
