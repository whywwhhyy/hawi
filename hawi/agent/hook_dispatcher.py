"""Plugin hook dispatch component for HawiAgent."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Protocol

from hawi.models import Model
from hawi.models.message import MessageResponse
from hawi.plugin import PluginManager
from hawi.plugin.hook_context import HookContext, HookResult
from hawi.plugin.types import (
    is_system_prompt_injection_hook,
    system_prompt_variability_rank,
)
from hawi.tool.types import ToolResult


class HookOwner(Protocol):
    _plugin_manager: PluginManager


HookObserver = Callable[[str, Callable[..., Any], HookResult | None], Any]
HookStartObserver = Callable[[str, Callable[..., Any]], Any]


class HookDispatcher:
    """Dispatch plugin hooks while passing the owning agent to hook callables."""

    def __init__(self, agent: object, owner: HookOwner) -> None:
        self._agent = agent
        self._owner = owner

    async def invoke_session(
        self,
        hook_type: str,
        ctx: HookContext,
        on_hook_start: HookStartObserver | None = None,
        on_hook_end: HookObserver | None = None,
    ) -> HookResult | None:
        """Invoke before/after_session and before/after_conversation hooks."""
        for hook in self._owner._plugin_manager.get_hooks(hook_type):
            if self._should_skip_system_prompt_hook(hook_type, hook):
                continue
            await self._notify_hook_start(on_hook_start, hook_type, hook)
            tracks_system_prompt = is_system_prompt_injection_hook(hook)
            before_part_ids = (
                self._current_system_prompt_part_ids()
                if tracks_system_prompt
                else set()
            )
            result = hook(self._agent, ctx)
            if inspect.isawaitable(result):
                result = await result
            if tracks_system_prompt:
                self._record_system_prompt_parts(
                    before_part_ids,
                    system_prompt_variability_rank(hook),
                )
                mark_run = getattr(self._agent, "_mark_system_prompt_hook_run", None)
                if callable(mark_run):
                    mark_run(hook_type, hook)
            await self._notify_hook_end(on_hook_end, hook_type, hook, result)
            if result is not None:
                return result
        return None

    def _should_skip_system_prompt_hook(self, hook_type: str, hook: Callable[..., Any]) -> bool:
        if hook_type not in {"before_session", "before_conversation"}:
            return False
        if not is_system_prompt_injection_hook(hook):
            return False
        if getattr(self._agent, "_suppress_system_prompt_hooks", False):
            return True
        has_run = getattr(self._agent, "_system_prompt_hook_has_run", None)
        return bool(callable(has_run) and has_run(hook_type, hook))

    def _current_system_prompt_part_ids(self) -> set[int]:
        context = getattr(self._agent, "context", None)
        parts = getattr(context, "system_prompt", None)
        if not isinstance(parts, list):
            return set()
        return {id(part) for part in parts}

    def _record_system_prompt_parts(
        self,
        previous_part_ids: set[int],
        rank: int,
    ) -> None:
        context = getattr(self._agent, "context", None)
        parts = getattr(context, "system_prompt", None)
        if not isinstance(parts, list):
            return
        rank_by_part_id = getattr(
            self._agent,
            "_system_prompt_part_variability_rank",
            None,
        )
        if rank_by_part_id is None:
            rank_by_part_id = {}
            setattr(
                self._agent,
                "_system_prompt_part_variability_rank",
                rank_by_part_id,
            )

        current_ids: set[int] = set()
        for part in parts:
            part_id = id(part)
            current_ids.add(part_id)
            if part_id not in previous_part_ids:
                rank_by_part_id[part_id] = rank

        for part_id in list(rank_by_part_id):
            if part_id not in current_ids:
                del rank_by_part_id[part_id]

        indexed_parts = list(enumerate(parts))
        indexed_parts.sort(
            key=lambda item: self._system_prompt_sort_key(
                item[0],
                item[1],
                rank_by_part_id,
            )
        )
        # context is guaranteed to be non-None here because parts was a list above
        assert context is not None
        context.system_prompt = [part for _, part in indexed_parts]

    @staticmethod
    def _system_prompt_sort_key(
        index: int,
        part: object,
        rank_by_part_id: dict[int, int],
    ) -> tuple[int, int, int]:
        rank = rank_by_part_id.get(id(part))
        if rank is None:
            return (0, 0, index)
        return (1, rank, index)

    async def invoke_before_model_call(
        self,
        model: Model,
        ctx: HookContext,
        on_hook_start: HookStartObserver | None = None,
        on_hook_end: HookObserver | None = None,
    ) -> HookResult | None:
        """Invoke before_model_call hook: (agent, model, ctx)."""
        for hook in self._owner._plugin_manager.get_hooks("before_model_call"):
            await self._notify_hook_start(
                on_hook_start,
                "before_model_call",
                hook,
            )
            result = hook(self._agent, model, ctx)
            if inspect.isawaitable(result):
                result = await result
            await self._notify_hook_end(
                on_hook_end,
                "before_model_call",
                hook,
                result,
            )
            if result is not None:
                return result
        return None

    async def invoke_after_model_call(
        self,
        response: MessageResponse,
        ctx: HookContext,
        on_hook_start: HookStartObserver | None = None,
        on_hook_end: HookObserver | None = None,
    ) -> HookResult | None:
        """Invoke after_model_call hook: (agent, response, ctx)."""
        for hook in self._owner._plugin_manager.get_hooks("after_model_call"):
            await self._notify_hook_start(
                on_hook_start,
                "after_model_call",
                hook,
            )
            result = hook(self._agent, response, ctx)
            if inspect.isawaitable(result):
                result = await result
            await self._notify_hook_end(
                on_hook_end,
                "after_model_call",
                hook,
                result,
            )
            if result is not None:
                return result
        return None

    @staticmethod
    async def _notify_hook_start(
        observer: HookStartObserver | None,
        hook_type: str,
        hook: Callable[..., Any],
    ) -> None:
        if observer is None:
            return
        result = observer(hook_type, hook)
        if inspect.isawaitable(result):
            await result

    @staticmethod
    async def _notify_hook_end(
        observer: HookObserver | None,
        hook_type: str,
        hook: Callable[..., Any],
        result: HookResult | None,
    ) -> None:
        if observer is None:
            return
        observed = observer(hook_type, hook, result)
        if inspect.isawaitable(observed):
            await observed

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
