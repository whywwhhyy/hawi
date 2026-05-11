"""Plugin hook dispatch component for HawiAgent."""

from __future__ import annotations

import inspect
from typing import Protocol

from hawi.models import Model
from hawi.models.message import MessageResponse
from hawi.plugin import PluginManager
from hawi.plugin.hook_context import HookContext, HookResult
from hawi.tool.types import ToolResult


class HookOwner(Protocol):
    _plugin_manager: PluginManager


class HookDispatcher:
    """Dispatch plugin hooks while passing the owning agent to hook callables."""

    def __init__(self, agent: object, owner: HookOwner) -> None:
        self._agent = agent
        self._owner = owner

    async def invoke_session(
        self,
        hook_type: str,
        ctx: HookContext,
    ) -> HookResult | None:
        """Invoke before/after_session and before/after_conversation hooks."""
        for hook in self._owner._plugin_manager.get_hooks(hook_type):
            result = hook(self._agent, ctx)
            if inspect.isawaitable(result):
                result = await result
            if result is not None:
                return result
        return None

    async def invoke_before_model_call(
        self,
        model: Model,
        ctx: HookContext,
    ) -> HookResult | None:
        """Invoke before_model_call hook: (agent, model, ctx)."""
        for hook in self._owner._plugin_manager.get_hooks("before_model_call"):
            result = hook(self._agent, model, ctx)
            if inspect.isawaitable(result):
                result = await result
            if result is not None:
                return result
        return None

    async def invoke_after_model_call(
        self,
        response: MessageResponse,
        ctx: HookContext,
    ) -> HookResult | None:
        """Invoke after_model_call hook: (agent, response, ctx)."""
        for hook in self._owner._plugin_manager.get_hooks("after_model_call"):
            result = hook(self._agent, response, ctx)
            if inspect.isawaitable(result):
                result = await result
            if result is not None:
                return result
        return None

    async def invoke_before_tool_calling(
        self,
        tool_name: str,
        arguments: dict,
        ctx: HookContext,
    ) -> HookResult | None:
        """Invoke before_tool_calling hook: (agent, tool_name, arguments, ctx)."""
        for hook in self._owner._plugin_manager.get_hooks("before_tool_calling"):
            result = hook(self._agent, tool_name, arguments, ctx)
            if inspect.isawaitable(result):
                result = await result
            if result is not None:
                return result
        return None

    async def invoke_after_tool_calling(
        self,
        tool_name: str,
        arguments: dict,
        tool_result: ToolResult,
        ctx: HookContext,
    ) -> HookResult | None:
        """Invoke after_tool_calling hook: (agent, tool_name, arguments, result, ctx)."""
        for hook in self._owner._plugin_manager.get_hooks("after_tool_calling"):
            result = hook(self._agent, tool_name, arguments, tool_result, ctx)
            if inspect.isawaitable(result):
                result = await result
            if result is not None:
                return result
        return None
