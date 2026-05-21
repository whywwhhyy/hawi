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
        description=(
            "Create a managed background sub-agent and optionally enqueue its "
            "first task. Always provide a clear initial_prompt that tells the "
            "child it is a sub-agent and states the exact assigned task. The "
            "plugins argument controls the child's tools: omit it or set it to "
            "null/None to inherit the parent agent's current plugin setup; set "
            "it to [] to give the child no tools; set it to a list of active "
            "parent plugin ids to choose tools for the task type. When "
            "share_context is true, plugins must be omitted or null/None "
            "because shared-context sub-agents inherit the parent plugin setup."
        ),
        context="ctx",
        tags=["subagent", "orchestration"],
        parameters_schema={
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["fork", "fresh"],
                    "default": "fresh",
                    "description": (
                        "fresh starts isolated; fork copies parent context. "
                        "Prefer share_context for the common toggle."
                    ),
                },
                "share_context": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "When true, copy the parent conversation context into "
                        "the sub-agent. Defaults to false."
                    ),
                },
                "name": {"type": "string"},
                "role": {
                    "type": "string",
                    "default": "general",
                    "description": "general, planner, reviewer, explorer, implementer, critic, or summarizer.",
                },
                "model": {"type": "string"},
                "system_prompt": {
                    "type": "string",
                    "description": (
                        "Explicit system prompt controlled by the parent agent. "
                        "If omitted, Hawi uses the role prompt."
                    ),
                },
                "working_dir": {"type": "string"},
                "initial_prompt": {
                    "type": "string",
                    "description": (
                        "Required first user prompt for the sub-agent. State "
                        "that it is a sub-agent task and describe the exact "
                        "work to perform."
                    ),
                },
                "initial_plan": {},
                "plugins": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                    "default": None,
                    "description": (
                        "Child plugin policy. Omit this field or set it to "
                        "null/None to inherit the parent agent's current plugin "
                        "setup. Set it to [] to create a child with no tools. "
                        "Set it to a list of active parent plugin ids (for "
                        "example hawi/filesystem, hawi/shell, "
                        "hawi/python_interpreter, hawi/web, or hawi/mcp) to "
                        "choose tools for the task type. When share_context is "
                        "true, plugin settings cannot be changed; omit this "
                        "field or use null/None."
                    ),
                },
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
            "required": ["initial_prompt"],
        },
    )
    async def create_subagent(
        self,
        mode: Literal["fork", "fresh"] = "fresh",
        share_context: bool | None = None,
        name: str | None = None,
        role: str = "general",
        model: str | None = None,
        system_prompt: str | None = None,
        working_dir: str | None = None,
        initial_prompt: str | None = None,
        initial_plan: Any | None = None,
        plugins: list[str] | None = None,
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
        if not isinstance(initial_prompt, str) or not initial_prompt.strip():
            raise ValueError(
                "create_subagent requires initial_prompt: give the child a clear "
                "user prompt that explains it is a sub-agent and states its task."
            )
        effective_mode: Literal["fork", "fresh"] = (
            "fork"
            if share_context is True
            else "fresh"
            if share_context is False
            else mode
        )
        plugin_policy = self._plugin_policy_for_request(
            ctx,
            mode=effective_mode,
            plugins=plugins,
        )
        handle = await ctx.agent.subagents.spawn(
            SubAgentSpec(
                mode=effective_mode,
                name=name,
                role=role,
                model=model,
                system_prompt=system_prompt,
                plugin_policy=plugin_policy,
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

    def _plugin_policy_for_request(
        self,
        ctx: ToolCallContext,
        *,
        mode: Literal["fork", "fresh"],
        plugins: list[str] | None,
    ) -> SubAgentPluginPolicy:
        requested = self._normalize_plugin_names(plugins)
        if mode == "fork":
            if plugins is not None:
                raise ValueError(
                    "shared-context sub-agents must inherit the parent plugin "
                    "setup; set plugins to null/None when share_context is true."
                )
            return SubAgentPluginPolicy(inherit=True)
        if plugins is None:
            return SubAgentPluginPolicy(inherit=True)
        if not requested:
            return SubAgentPluginPolicy(inherit=False)
        return SubAgentPluginPolicy(
            inherit=False,
            extra_plugins=self._clone_selected_parent_plugins(ctx, requested),
        )

    @staticmethod
    def _normalize_plugin_names(plugins: list[str] | None) -> list[str]:
        if plugins is None:
            return []
        if not isinstance(plugins, list):
            raise TypeError("plugins must be null/None or a list of plugin ids")
        names: list[str] = []
        for item in plugins:
            if not isinstance(item, str):
                raise TypeError("plugins entries must be strings")
            normalized = item.strip()
            if normalized:
                names.append(normalized)
        return names

    def _clone_selected_parent_plugins(
        self,
        ctx: ToolCallContext,
        requested: list[str],
    ) -> list[HawiPlugin]:
        active_plugins = ctx.agent.plugins.get_plugins()
        selected: list[HawiPlugin] = []
        seen: set[str] = set()
        available = {
            alias.casefold(): plugin
            for plugin in active_plugins
            for alias in self._plugin_aliases(plugin)
        }
        for name in requested:
            plugin = available.get(name.casefold())
            if plugin is None:
                choices = ", ".join(
                    sorted({
                        alias
                        for item in active_plugins
                        for alias in self._plugin_aliases(item)
                        if "/" in alias or alias.startswith("hawi/")
                    })
                )
                raise ValueError(
                    f"Unknown sub-agent plugin: {name}. Active plugin ids: {choices}"
                )
            plugin_key = plugin.plugin_id
            if plugin_key in seen:
                continue
            seen.add(plugin_key)
            clone = plugin.clone()
            clone.bind_plugin_identity(
                plugin_id=getattr(plugin, "_plugin_id", None),
                plugin_name=getattr(plugin, "_plugin_name", None),
            )
            selected.append(clone)
        return selected

    @staticmethod
    def _plugin_aliases(plugin: HawiPlugin) -> set[str]:
        return {
            str(value)
            for value in (
                plugin.plugin_id,
                plugin.name,
                plugin.plugin_name,
                plugin.display_name,
                plugin.__class__.__name__,
            )
            if value
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
