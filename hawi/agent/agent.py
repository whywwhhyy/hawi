"""HawiAgent - Core agent implementation with tool execution and plugin support.

This module implements the HawiAgent class that orchestrates LLM interaction,
tool execution, and plugin hooks for agent workflows.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from copy import deepcopy
from dataclasses import replace
from typing import Any, Optional, Literal, Callable, cast


from hawi.models import (
    CachePoint,
    Model,
    ContentPart,
    DeltaPart,
    TokenUsage,
    ToolCallPart,
    model_registry,
)
from hawi.models.message import Message, MessageResponse
from hawi.models.usage import (
    merge_token_usage,
    normalize_token_usage,
    usage_context_tokens,
)
from hawi.plugin import HawiPlugin
from hawi.plugin import PluginManager
from hawi.plugin.hook_context import HookContext, HookResult
from hawi.tool.types import AgentTool, ToolResult

from hawi.errors import (
    AgentError,
    ModelError,
    ContextLengthError,
    MaxIterationsError,
)
from hawi.events import (
    Event,
    EventBus,
    EventHandler,
    SyncEventHandler,
    AgentContextInjectedEvent,
    AgentErrorEvent,
    AgentMessageAddedEvent,
    AgentRunStartEvent,
    AgentRunStopEvent,
    AgentSystemPromptEvent,
    ModelErrorEvent,
    ModelRetryEvent,
    ModelStreamStartEvent,
    ModelStreamStopEvent,
    ModelMetadataEvent,
    DumpManager,
)
from .context import (
    AgentContext,
    ContextCompactionRecord,
    ContextUsageSnapshot,
    ToolCallContext,
    estimate_content_tokens,
)
from .compaction import AgentCompactor
from .eventing import AgentEvents
from .config import (
    AutoCompactConfig,
    ModelErrorNotifyPolicy,
    ModelErrorPolicy,
    ModelErrorPolicyConfig,
    ModelErrorRetryPolicy,
    ModelErrorStopPolicy,
    default_model_error_policy,
)
from .hook_dispatcher import HookDispatcher
from .result import AgentRunResult, ToolCallRecord
from .runtime import AgentRuntime
from .state import (
    AddedToolResultMessages,
    MaterializedSteerMessage,
    PendingInput,
    SteerPartMergeMode,
    _ExecutionState,
    _RecentToolResult,
)
from .stream_accumulator import StreamBlockAccumulator
from .tool_executor import (
    PreparedToolArguments,
    ToolCallPromise,
    ToolCallRequest,
    ToolExecutor,
)


SKIP_BEFORE_CONVERSATION_HOOKS_METADATA_KEY = "skip_before_conversation_hooks"


class HawiAgent:
    """Core agent implementation for Hawi framework.

    Supports tool execution loops, plugin hooks, streaming events,
    and context management.

    Example:
        # Basic usage
        agent = HawiAgent(model=deepseek_model, plugins=[MyPlugin()])
        result = agent.run("What's the weather in Beijing?")
        print(result.text)

        # Streaming
        for event in agent.run("Hello"):
            if event.type == "message":
                print(event.content)

        # Async
        result = await agent.arun("Hello")
    """

    def __init__(
        self,
        model: Model | str,
        *,
        plugins: list[HawiPlugin] | None = None,
        plugin_factories: list[Callable[[], HawiPlugin]] | None = None,
        system_prompt: str | list[ContentPart] | None = None,
        max_iterations: int | None = None,
        model_error_policy: Optional[ModelErrorPolicyConfig] = None,
        event_bus: EventBus | None = None,
        event_dump_file: str | None = None,
        streaming: bool = True,
        auto_compact: AutoCompactConfig | dict[str, Any] | bool | None = None,
        cache_point: CachePoint | dict[str, Any] | bool | None = None,
        cache_tool_definitions: CachePoint | dict[str, Any] | bool | None = None,
        auto_cache_static_prefix: CachePoint | dict[str, Any] | bool | None = True,
        permission_set: "PermissionSet | FrozenPermissionSet | dict[str, str] | None" = None,
    ):
        """Initialize HawiAgent.

        Args:
            model: Default model for agent execution. Can be:
                - Model instance (direct use)
                - str (model name from models.yaml, e.g., "deepseek-openai/deepseek-chat")
            plugins: List of plugins providing tools and hooks (default: empty list).
                On clone, `plugin.clone()` is called for each plugin.
            plugin_factories: List of factory functions that create plugins (default: empty list).
                Factories are called during init and clone to create fresh instances.
                Useful for stateful plugins that need complete isolation on fork.
            system_prompt: Default system prompt (str or list[ContentPart])
            max_iterations: Maximum tool execution iterations (None for unlimited)
            model_error_policy: Error handling policy mapping error_type to config
            event_bus: Event bus for event publishing. If None, creates a default EventBus
            event_dump_file: Path to dump all events for debugging (default: None)
            streaming: Whether to use streaming mode by default (default: True)
            auto_compact: Automatic context compaction configuration. Pass
                ``True`` to enable default threshold-based compaction.
            cache_point: Provider-neutral top-level/automatic prompt cache point.
            cache_tool_definitions: Cache the tool-definition prefix when supported.
            auto_cache_static_prefix: Agent-managed cache point for the stable
                tools/system prefix. Defaults to an ephemeral cache point.
            permission_set: Runtime permission configuration controlling which
                plugin tools are visible and executable.  Can be:

                - ``PermissionSet`` / ``FrozenPermissionSet`` instance.
                - ``dict`` mapping ``"scope:capability"`` ids to ``"allow"`` /
                  ``"deny"`` / ``"human_review"`` / ``"agent_review"``.

                When ``None`` (default), all tools are visible (backwards
                compatible).  See :mod:`hawi.permission`.

        Note:
            Both `plugins` and `plugin_factories` can be used together.
            Factories are invoked first during initialization.
        """
        # Resolve model from registry if string is provided
        if isinstance(model, str):
            model = model_registry.create_model(model)
        self._validate_model_steer_merge_mode(model, source="HawiAgent.__init__")
        self._default_model = model
        self._max_iterations = max_iterations
        self._streaming = streaming
        self._event_bus = event_bus or EventBus()
        self._auto_compact = self._normalize_auto_compact(auto_compact, model)

        # Initialize event dump manager
        self._dump_manager = DumpManager(event_dump_file) if event_dump_file else None


        if model_error_policy is None:
            self._model_error_policy = self._default_model_error_policy()
        else:
            self._model_error_policy = model_error_policy

        # Initialize PluginManager for plugin/tool/hook management
        self._plugin_manager = PluginManager(
            plugins=plugins,
            plugin_factories=plugin_factories,
        )
        self._plugin_manager.bind_event_bus(self._event_bus)

        # Initialize permission system
        from hawi.permission import PermissionSet, FrozenPermissionSet, PermissionAuditSink
        self._permission_audit_sink = PermissionAuditSink()
        if permission_set is not None:
            if isinstance(permission_set, dict):
                permission_set = PermissionSet.from_dict(permission_set)
            self._plugin_manager.set_permission_set(permission_set)
        self._events = AgentEvents(self)
        self._hooks = HookDispatcher(self, self)
        self._suppress_system_prompt_hooks = False
        self._system_prompt_injection_hook_keys_run: set[tuple[str, str]] = set()
        self._system_prompt_part_variability_rank: dict[int, int] = {}
        self._last_emitted_system_prompt: list[ContentPart] | None = None
        self._last_hook_result_injector: dict[str, str | None] | None = None
        self._compactor = AgentCompactor(self)
        self._runtime = AgentRuntime(self)
        self.review_broker: Any = None

        # Convert system_prompt to list[ContentPart] if needed
        system_prompt_parts: list[ContentPart] | None = None
        if isinstance(system_prompt, str):
            system_prompt_parts = [{"type": "text", "text": system_prompt}]
        else:
            system_prompt_parts = system_prompt

        self._system_prompt = system_prompt_parts

        # Initialize context with tool definitions
        defs = self._plugin_manager.get_tool_definitions()
        self._context = AgentContext(
            system_prompt=system_prompt_parts,
            tool_definitions=defs if defs else None,
        )
        if cache_point is not None:
            self._context.set_cache_point(cache_point)
        if cache_tool_definitions is not None:
            self._context.set_tool_cache_point(cache_tool_definitions)
        if auto_cache_static_prefix is not None:
            self._context.set_static_prefix_cache_point(auto_cache_static_prefix)

        # Set up tool call context for runtime injection
        self._context.tool_call_context = ToolCallContext(agent=self)

        # Initialize interrupt state for cooperative cancellation
        self._cancel_event = asyncio.Event()
        self._current_tool_calls: list[ToolCallPart] = []
        self._interrupted_tool_call_ids: list[str] = []
        self._last_interrupt_reason: str | None = None
        self._steer_lock = threading.RLock()
        self._pending_inputs: list[PendingInput] = []
        self._last_unsent_tool_results: list[_RecentToolResult] = []
        self._pending_model_input_started_at: float | None = None
        self._session_lock = threading.RLock()
        self._session_active = False
        self._autonomous_run_task: asyncio.Task[AgentRunResult] | None = None
        # Live reference to the in-flight _ExecutionState. Set in _execute()
        # and cleared in its finally block; SessionManager reads it via
        # snapshot_runtime() to capture run_id and iteration.
        self._active_execution_state: _ExecutionState | None = None
        self._tool_executor = self._build_tool_executor()

        # Core sub-agent lifecycle manager. Imported lazily to avoid module
        # cycles with the runner package during HawiAgent import.
        from .subagent import SubAgentManager
        self._subagents = SubAgentManager(self)

    @classmethod
    def _default_model_error_policy(cls) -> ModelErrorPolicyConfig:
        return default_model_error_policy()

    @property
    def plugins(self) -> PluginManager:
        """Get the plugin manager for accessing and modifying plugins/tools/hooks."""
        return self._plugin_manager

    @property
    def permission_set(self) -> "PermissionSet | FrozenPermissionSet | None":
        """The active permission set controlling visible and executable tools.

        Returns ``None`` when no permission set is configured (all tools
        visible — backwards compatible).

        See :mod:`hawi.permission` for the full permission model.
        """
        return self._plugin_manager.permission_set

    def set_permissions(
        self,
        permissions: "PermissionSet | FrozenPermissionSet | dict[str, str] | None",
    ) -> None:
        """Replace the active permission set and refresh tool definitions.

        After setting, the agent's context tool definitions are automatically
        refreshed to reflect the new visibility.

        Args:
            permissions: A ``PermissionSet``, ``FrozenPermissionSet``, or a
                plain ``dict`` mapping ``"scope:capability"`` ids to policy
                strings.  Pass ``None`` to disable all permission filtering.
        """
        from hawi.permission import PermissionSet, FrozenPermissionSet

        if isinstance(permissions, dict):
            permissions = PermissionSet.from_dict(permissions)
        self._plugin_manager.set_permission_set(permissions)
        # Refresh tool definitions in context
        defs = self._plugin_manager.get_tool_definitions()
        self._context.tool_definitions = defs if defs else None

    @property
    def permission_audit_sink(self):
        """The agent's permission audit sink for observability."""
        return self._permission_audit_sink

    @property
    def subagents(self):
        """Get the sub-agent lifecycle manager."""
        return self._subagents

    @property
    def context(self) -> AgentContext:
        """Get the agent's context (read-only access).

        Returns:
            The current AgentContext
        """
        return self._context

    def context_usage(self, model: Model | None = None) -> ContextUsageSnapshot:
        """Return the current estimated context-window occupancy."""
        return self._context.usage_snapshot(
            self._context_limit_for_model(model or self._default_model)
        )

    def _refresh_context_usage_snapshot(
        self,
        model: Model | None = None,
        *,
        preserve_provider: bool = True,
    ) -> ContextUsageSnapshot:
        """Persist current context usage for UI/status snapshots."""
        snapshot = self.context_usage(model)
        current = self._context.context_usage_snapshot()
        if (
            preserve_provider
            and current is not None
            and current.source == "provider_usage"
            and current.max_context_tokens == snapshot.max_context_tokens
        ):
            if current.used_tokens >= snapshot.used_tokens:
                return current
            snapshot = ContextUsageSnapshot(
                used_tokens=snapshot.used_tokens,
                max_context_tokens=snapshot.max_context_tokens,
                usage_ratio=snapshot.usage_ratio,
                remaining_tokens=snapshot.remaining_tokens,
                source="provider_usage",
            )
        self._context.set_context_usage(snapshot)
        return snapshot

    def set_context(self, context: AgentContext) -> None:
        """Replace the agent's context.

        Args:
            context: New context to use
        """
        self._context = context
        self._context.tool_call_context = ToolCallContext(agent=self)
        if hasattr(self, "_tool_executor"):
            self._tool_executor = self._build_tool_executor()

    def set_system_prompt(self, system_prompt: str | list[ContentPart] | None) -> None:
        """Replace the agent system prompt and keep clone defaults in sync."""
        if system_prompt is None:
            self._context.system_prompt = None
            self._system_prompt = None
            return
        self._context.set_system_prompt(system_prompt)
        self._system_prompt = self._context.get_system_prompt()

    def suppress_system_prompt_hooks(self, suppress: bool = True) -> None:
        """Control whether declared system-prompt injection hooks are skipped."""
        self._suppress_system_prompt_hooks = suppress

    def reset_system_prompt_injection_hooks(self) -> None:
        """Allow declared system-prompt injection hooks to run for a new session."""
        self._system_prompt_injection_hook_keys_run.clear()
        self._system_prompt_part_variability_rank.clear()
        self._last_emitted_system_prompt = None

    def _system_prompt_hook_has_run(
        self,
        hook_type: str,
        hook: Callable[..., Any],
    ) -> bool:
        return (
            hook_type,
            self._system_prompt_hook_key(hook),
        ) in self._system_prompt_injection_hook_keys_run

    def _mark_system_prompt_hook_run(
        self,
        hook_type: str,
        hook: Callable[..., Any],
    ) -> None:
        self._system_prompt_injection_hook_keys_run.add(
            (hook_type, self._system_prompt_hook_key(hook))
        )

    @staticmethod
    def _system_prompt_hook_key(hook: Callable[..., Any]) -> str:
        owner = getattr(hook, "__self__", None)
        method_name = getattr(hook, "__name__", None)
        if isinstance(owner, HawiPlugin):
            return ":".join(
                (
                    "plugin",
                    str(id(owner)),
                    owner.plugin_id,
                    owner.__class__.__module__,
                    owner.__class__.__qualname__,
                    method_name or type(hook).__name__,
                )
            )
        module = getattr(hook, "__module__", "")
        qualname = getattr(hook, "__qualname__", method_name or type(hook).__name__)
        return f"hook:{module}:{qualname}:{id(hook)}"

    def set_model(self, model: Model | str) -> None:
        """Replace the default model for this agent.

        Args:
            model: New model to use. Can be:
                - Model instance (direct use)
                - str (model name from models.yaml, e.g., "deepseek-chat")

        Example:
            # Switch to a different model
            agent.set_model("kimi-k2-5")

            # Or use a model instance directly
            from hawi.models import DeepSeekModel
            agent.set_model(DeepSeekModel(model_id="deepseek-chat"))
        """
        if isinstance(model, str):
            model = model_registry.create_model(model)
        self._validate_model_steer_merge_mode(model, source="HawiAgent.set_model")
        # Reset current model state before replacing
        self._default_model.reset()
        self._default_model = model
        self._auto_compact = self._normalize_auto_compact(
            self._auto_compact,
            model,
        )
        self._refresh_context_usage_snapshot(model, preserve_provider=False)

    @staticmethod
    def _validate_model_steer_merge_mode(model: Any, *, source: str) -> None:
        """Validate steer configuration early for real Hawi Model instances."""
        if isinstance(model, Model):
            model.validate_steer_merge_mode_config(source=source)

    @property
    def model(self) -> Model:
        """Get the current default model."""
        return self._default_model

    def clone(self) -> HawiAgent:
        """Create a clone of this agent with copied state.

        The cloned agent has:
        - Copied context (messages, tools, system_prompt)
        - Cloned plugins (each plugin's `clone()` method is called)
        - Re-invoked plugin factories to create fresh instances
        - Same default model
        - Same configuration (max_iterations, etc.)

        The clone is independent - modifications to the clone's context
        do not affect the original agent.

        Plugin handling:
        - Plugins passed via `plugins` param: `plugin.clone()` is called for each.
          Default `clone()` returns self (safe for stateless plugins).
          Stateful plugins should override `clone()` to return a fresh copy.
        - Plugins passed via `plugin_factories`: factories are re-invoked to
          create fresh instances, ensuring complete state isolation.

        Returns:
            New HawiAgent instance with copied state
        """
        new_agent = HawiAgent(
            model=self._default_model,
            system_prompt=self._system_prompt,
            max_iterations=self._max_iterations,
            model_error_policy=self._model_error_policy,
            event_bus=self._event_bus,
            streaming=self._streaming,
            event_dump_file=self._dump_manager.dump_file if self._dump_manager else None,
            auto_compact=self._auto_compact,
        )
        new_agent._plugin_manager = self._plugin_manager.clone()
        new_agent._plugin_manager.bind_event_bus(new_agent._event_bus)
        new_agent.review_broker = self.review_broker
        new_agent.set_context(self._context.copy())
        new_agent.suppress_system_prompt_hooks(self._suppress_system_prompt_hooks)
        return new_agent

    def fork(self) -> HawiAgent:
        """Alias for clone().

        Returns:
            New HawiAgent instance with copied state
        """
        return self.clone()

    async def spawn_subagent(self, *args: Any, **kwargs: Any):
        """Create a managed sub-agent via :attr:`subagents`."""
        return await self._subagents.spawn(*args, **kwargs)

    def send_subagent_input(
        self,
        subagent_id: str,
        message: str | list[ContentPart],
        queue: Literal["normal", "high_prio", "urgent"] = "normal",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Send a message to a managed sub-agent."""
        return self._subagents.send(
            subagent_id,
            message,
            queue=queue,
            metadata=metadata,
        )

    def read_subagent(
        self,
        subagent_id: str,
        *,
        view: Literal["status", "summary", "events", "context_tail"] = "summary",
        limit: int = 20,
    ) -> dict[str, Any]:
        """Read a controlled view of a managed sub-agent."""
        return self._subagents.read(subagent_id, view=view, limit=limit)

    async def close_subagent(
        self,
        subagent_id: str,
        *,
        reason: str = "closed",
        interrupt: bool = True,
    ):
        """Close a managed sub-agent."""
        return await self._subagents.close(
            subagent_id,
            reason=reason,
            interrupt=interrupt,
        )

    @staticmethod
    def _normalize_auto_compact(
        config: AutoCompactConfig | dict[str, Any] | bool | None,
        model: Model | None = None,
    ) -> AutoCompactConfig:
        """Normalize user-facing auto-compact options."""
        model_max_context_tokens = HawiAgent._model_max_context_tokens(model)
        if isinstance(config, AutoCompactConfig):
            return HawiAgent._clamp_auto_compact_config_to_model(
                config,
                model_max_context_tokens,
            )
        if isinstance(config, dict):
            values = dict(config)
            if model_max_context_tokens is not None and "max_context_tokens" not in values:
                values["max_context_tokens"] = model_max_context_tokens
            return HawiAgent._clamp_auto_compact_config_to_model(
                AutoCompactConfig(**values),
                model_max_context_tokens,
            )
        if isinstance(config, bool):
            return HawiAgent._clamp_auto_compact_config_to_model(
                AutoCompactConfig(
                    enabled=config,
                    max_context_tokens=model_max_context_tokens or 128_000,
                ),
                model_max_context_tokens,
            )
        return HawiAgent._clamp_auto_compact_config_to_model(
            AutoCompactConfig(
                enabled=True,
                max_context_tokens=model_max_context_tokens or 128_000,
            ),
            model_max_context_tokens,
        )

    @staticmethod
    def _model_max_context_tokens(model: Model | None) -> int | None:
        if model is None:
            return None
        getter = getattr(model, "get_max_context_tokens", None)
        if not callable(getter):
            return None
        value = getter()
        if isinstance(value, int) and value > 0:
            return value
        return None

    @staticmethod
    def _clamp_auto_compact_config_to_model(
        config: AutoCompactConfig,
        model_max_context_tokens: int | None,
    ) -> AutoCompactConfig:
        if (
            model_max_context_tokens is not None
            and (
                config.max_context_tokens <= 0
                or config.max_context_tokens > model_max_context_tokens
            )
        ):
            return replace(
                config,
                max_context_tokens=model_max_context_tokens,
            )
        return config

    def _apply_context_length_limit_from_error(
        self,
        model: Model,
        error: ContextLengthError,
    ) -> None:
        provider_limit = error.max_context_tokens
        if not isinstance(provider_limit, int) or provider_limit <= 0:
            return
        if (
            self._auto_compact.max_context_tokens <= 0
            or provider_limit < self._auto_compact.max_context_tokens
        ):
            self._auto_compact = replace(
                self._auto_compact,
                max_context_tokens=provider_limit,
            )
        setter = getattr(model, "configure_max_context_tokens", None)
        if callable(setter):
            current_limit = self._model_max_context_tokens(model)
            if current_limit is None or provider_limit < current_limit:
                setter(provider_limit)

    async def _force_auto_compact_for_context_length_error(
        self,
        model: Model,
        state: _ExecutionState,
        event_bus: EventBus | None,
        error: ContextLengthError,
    ) -> bool:
        cfg = self._auto_compact
        if not cfg.enabled or self.has_active_tool_calls:
            return False
        self._apply_context_length_limit_from_error(model, error)
        try:
            record = await self._compactor.acompact(
                model=model,
                config=self._auto_compact,
                event_bus=event_bus,
                run_id=state.run_id,
                mode="auto",
            )
        except Exception:
            return False
        if record is None:
            return False
        state.last_auto_compact_iteration = state.iteration
        return True

    def _context_limit_for_model(self, model: Model) -> int | None:
        getter = getattr(model, "get_max_context_tokens", None)
        if callable(getter):
            value = getter()
            if isinstance(value, int) and value > 0:
                return value
        if self._auto_compact.max_context_tokens > 0:
            return self._auto_compact.max_context_tokens
        return None

    @staticmethod
    def _is_first_output_chunk(chunk: DeltaPart) -> bool:
        """Return True when a delta contains observable model output."""
        chunk_type = chunk.get("type")
        if chunk_type in {"text_delta", "reasoning_delta"}:
            return bool(chunk.get("delta"))
        if chunk_type == "tool_call_delta":
            return bool(
                chunk.get("arguments_delta")
                or chunk.get("name")
                or chunk.get("id")
            )
        return False

    @staticmethod
    def _tokens_per_second(tokens: int | None, duration_ms: float | None) -> float | None:
        if tokens is None or tokens <= 0 or duration_ms is None or duration_ms <= 0:
            return None
        return tokens / (duration_ms / 1000)

    @staticmethod
    def _prefill_tokens_for_timing(
        usage: TokenUsage | None,
        context_tokens: int,
    ) -> int | None:
        if context_tokens <= 0:
            return None
        cache_read_tokens = 0
        if usage is not None:
            value = usage.get("cache_read_tokens")
            if isinstance(value, int) and value > 0:
                cache_read_tokens = value
        return max(0, context_tokens - cache_read_tokens)

    def _mark_model_input_started(self, at: float | None = None) -> None:
        """Remember when new model-visible input began waiting for a reply."""
        timestamp = time.time() if at is None else at
        if (
            self._pending_model_input_started_at is None
            or timestamp < self._pending_model_input_started_at
        ):
            self._pending_model_input_started_at = timestamp

    def _consume_model_input_started(self, fallback: float) -> float:
        started_at = self._pending_model_input_started_at
        self._pending_model_input_started_at = None
        return started_at if started_at is not None else fallback

    def _mark_context_growth_as_model_input(
        self,
        *,
        message_count_before: int,
        started_at: float,
    ) -> None:
        if len(self._context.messages) > message_count_before:
            self._mark_model_input_started(started_at)
            self._refresh_context_usage_snapshot()

    @classmethod
    def _model_timing_metadata(
        cls,
        *,
        started_at: float,
        first_token_at: float | None,
        completed_at: float,
        prefill_tokens: int | None,
        decode_tokens: int | None,
    ) -> dict[str, float | int | None]:
        ttft_ms = (
            max(0.0, (first_token_at - started_at) * 1000)
            if first_token_at is not None
            else None
        )
        decode_ms = (
            max(0.0, (completed_at - first_token_at) * 1000)
            if first_token_at is not None
            else None
        )
        return {
            "started_at": started_at,
            "first_token_at": first_token_at,
            "completed_at": completed_at,
            "ttft_ms": ttft_ms,
            "decode_ms": decode_ms,
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_tokens,
            "prefill_tokens_per_second": cls._tokens_per_second(
                prefill_tokens,
                ttft_ms,
            ),
            "decode_tokens_per_second": cls._tokens_per_second(
                decode_tokens,
                decode_ms,
            ),
        }

    def interrupt(self, reason: str = "user") -> list[str]:
        return self._runtime.interrupt(reason)

    def on_interrupt(self, reason: str = "user") -> None:
        return self._runtime.on_interrupt(reason)

    def clear_interrupt_state(self) -> None:
        self._runtime.clear_interrupt_state()

    def _check_interrupt(self) -> bool:
        return self._runtime.check_interrupt()

    def _interrupt_tool_result_content(self, reason: str) -> str:
        return self._runtime.interrupt_tool_result_content(reason)

    async def _recover_unanswered_tool_calls(
        self,
        *,
        run_id: str | None,
        event_bus: EventBus | None,
        reason: str,
        emit_events: bool,
    ) -> None:
        await self._runtime.recover_unanswered_tool_calls(
            run_id=run_id,
            event_bus=event_bus,
            reason=reason,
            emit_events=emit_events,
        )

    @property
    def has_active_tool_calls(self) -> bool:
        return self._runtime.has_active_tool_calls

    def snapshot_runtime(self) -> dict[str, Any]:
        return self._runtime.snapshot_runtime()

    def load_runtime(self, data: dict[str, Any]) -> None:
        self._runtime.load_runtime(data)

    def snapshot_steer(self) -> list[dict[str, Any]]:
        return self._runtime.snapshot_steer()

    def load_steer(self, data: list[dict[str, Any]]) -> None:
        self._runtime.load_steer(data)

    def steer(
        self,
        content: str | list[ContentPart],
        *,
        merge_mode: SteerPartMergeMode | None = None,
    ) -> str:
        return self._runtime.steer(content, merge_mode=merge_mode)

    def get_pending_input_messages(self) -> list[dict[str, Any]]:
        return self._runtime.get_pending_input_messages()

    def has_pending_inputs(self) -> bool:
        return self._runtime.has_pending_inputs()

    _normalize_content_parts = staticmethod(AgentRuntime.normalize_content_parts)
    _truncate_preview = staticmethod(AgentRuntime.truncate_preview)
    _serialize_content_parts = staticmethod(AgentRuntime.serialize_content_parts)
    _tool_result_content = staticmethod(AgentRuntime.tool_result_content)
    _truncate_tool_result_for_retry = staticmethod(
        AgentRuntime.truncate_tool_result_for_retry
    )
    _context_retry_tool_result_target_chars = staticmethod(
        AgentRuntime.context_retry_tool_result_target_chars
    )
    _context_retry_needed_reduction_chars = staticmethod(
        AgentRuntime.context_retry_needed_reduction_chars
    )

    def _recent_tool_result_from_record(
        self,
        record: ToolCallRecord,
    ) -> _RecentToolResult:
        return self._runtime.recent_tool_result_from_record(record)

    def _mark_tool_results_unsent(self, records: list[ToolCallRecord]) -> None:
        self._runtime.mark_tool_results_unsent(records)

    async def _truncate_last_unsent_tool_results_for_context_retry(
        self,
        error: ContextLengthError,
    ) -> bool:
        return await self._runtime.truncate_last_unsent_tool_results_for_context_retry(
            error
        )

    async def _drain_pending_inputs_to_context(
        self,
        run_id: str,
        event_bus: EventBus | None,
    ) -> bool:
        return await self._runtime.drain_pending_inputs_to_context(run_id, event_bus)

    def _clear_autonomous_run_task(self, task: asyncio.Task[AgentRunResult]) -> None:
        self._runtime.clear_autonomous_run_task(task)

    async def _run_pending_turns(self) -> AgentRunResult:
        return await self._runtime.run_pending_turns()

    def _ensure_pending_turn_loop(self) -> None:
        self._runtime.ensure_pending_turn_loop()

    def _add_tool_result_with_pending_steer(
        self,
        tool_call_id: str,
        content: str | list[ContentPart],
        *,
        is_error: bool = False,
        materialize_pending_steer: bool = True,
        cache_point: CachePoint | dict[str, Any] | bool | None = None,
        cache_point_source: str | None = None,
    ) -> AddedToolResultMessages:
        return self._runtime.add_tool_result_with_pending_steer(
            tool_call_id,
            content,
            is_error=is_error,
            materialize_pending_steer=materialize_pending_steer,
            cache_point=cache_point,
            cache_point_source=cache_point_source,
        )

    async def _emit_tool_result_message_event(
        self,
        *,
        run_id: str,
        tool_call_id: str,
        content: str | list[ContentPart],
        is_error: bool,
        context_message_id: str,
        event_bus: EventBus | None,
    ) -> None:
        await self._runtime.emit_tool_result_message_event(
            run_id=run_id,
            tool_call_id=tool_call_id,
            content=content,
            is_error=is_error,
            context_message_id=context_message_id,
            event_bus=event_bus,
        )

    def _materialize_pending_steer_for_tool_results(
        self,
        tool_call_ids: list[str],
    ) -> list[MaterializedSteerMessage]:
        return self._runtime.materialize_pending_steer_for_tool_results(tool_call_ids)

    def _steer_message_metadata(
        self,
        pending_input: PendingInput,
        matched_tool_call_id: str,
    ) -> dict[str, Any]:
        return self._runtime.steer_message_metadata(
            pending_input,
            matched_tool_call_id,
        )

    async def _emit_materialized_steer_events(
        self,
        run_id: str,
        materialized_messages: list[MaterializedSteerMessage],
        event_bus: EventBus | None,
    ) -> None:
        await self._runtime.emit_materialized_steer_events(
            run_id,
            materialized_messages,
            event_bus,
        )

    def compact(
        self,
        *,
        model: Model | None = None,
        prompt: str | None = None,
        keep_last_messages: int | None = None,
    ) -> ContextCompactionRecord | None:
        """Synchronously compact the current conversation context."""
        return self._compactor.compact(
            model=model,
            prompt=prompt,
            keep_last_messages=keep_last_messages,
        )

    async def acompact(
        self,
        *,
        model: Model | None = None,
        prompt: str | None = None,
        keep_last_messages: int | None = None,
        config: AutoCompactConfig | None = None,
        event_bus: EventBus | None = None,
        run_id: str | None = None,
        mode: Literal["manual", "auto"] = "manual",
    ) -> ContextCompactionRecord | None:
        """Compact older context into a model-generated handoff summary."""
        return await self._compactor.acompact(
            model=model,
            prompt=prompt,
            keep_last_messages=keep_last_messages,
            config=config,
            event_bus=event_bus,
            run_id=run_id,
            mode=mode,
        )

    async def _maybe_auto_compact(
        self,
        model: Model,
        state: _ExecutionState,
        event_bus: EventBus | None = None,
    ) -> bool:
        return await self._compactor._maybe_auto_compact(model, state, event_bus)

    async def _generate_compaction_summary(
        self,
        model: Model,
        *,
        prompt: str,
        max_output_tokens: int,
        max_transcript_chars: int,
        compression_budget: int = 20_000,
        max_summary_chars: int = 4_000,
    ) -> str:
        return await self._compactor._generate_compaction_summary(
            model,
            prompt=prompt,
            compression_budget=compression_budget,
            max_output_tokens=max_output_tokens,
            max_summary_chars=max_summary_chars,
            max_transcript_chars=max_transcript_chars,
        )

    async def _collect_model_text(
        self,
        model: Model,
        *,
        messages: list[Message],
        system: list[ContentPart],
        max_output_tokens: int,
        streaming: bool,
    ) -> str:
        return await self._compactor._collect_model_text(
            model,
            messages=messages,
            system=system,
            max_output_tokens=max_output_tokens,
            streaming=streaming,
        )

    def _build_compaction_transcript(
        self,
        messages: list[Message],
        *,
        max_chars: int,
    ) -> str:
        return self._compactor._build_compaction_transcript(messages, max_chars=max_chars)

    def _render_message_content_for_compaction(self, message: Message) -> str:
        return self._compactor._render_message_content_for_compaction(message)

    def _fallback_compaction_summary(self, messages: list[Message]) -> str:
        return self._compactor._fallback_compaction_summary(messages)

    @property
    def event_bus(self) -> EventBus:
        """Get the agent's EventBus for event subscriptions."""
        return self._events.event_bus

    def subscribe(
        self,
        callback: EventHandler,
        event_types: list[str] | None = None,
        maxsize: int = 100,
    ) -> None:
        """Subscribe to agent events."""
        self._events.subscribe(callback, event_types, maxsize)

    def subscribe_blocking(
        self,
        callback: SyncEventHandler,
        event_types: list[str] | None = None,
    ) -> None:
        """Subscribe to agent events with a blocking sync handler."""
        self._events.subscribe_blocking(callback, event_types)

    def unsubscribe(
        self,
        callback: Callable[[Event], None],
    ) -> bool:
        """Unsubscribe from agent events."""
        return self._events.unsubscribe(callback)

    async def _emit_event(
        self,
        event: Event,
        event_bus: EventBus | None,
    ) -> Event:
        if (
            isinstance(event, AgentMessageAddedEvent)
            and event.role in {"user", "tool"}
        ):
            self._mark_model_input_started(event.timestamp)
        return await self._events.emit(event, event_bus)

    async def _emit_system_prompt_event_if_changed(
        self,
        *,
        run_id: str,
        origin: str,
        event_bus: EventBus | None,
    ) -> None:
        current = list(self._context.get_system_prompt() or [])
        content_snapshot = deepcopy(current)
        if not content_snapshot:
            self._last_emitted_system_prompt = content_snapshot
            return
        if content_snapshot == self._last_emitted_system_prompt:
            return
        self._last_emitted_system_prompt = deepcopy(content_snapshot)
        display_content = self._system_prompt_display_content(current)
        if not display_content:
            return
        await self._emit_event(
            AgentSystemPromptEvent.create(
                run_id=run_id,
                content=display_content,
                origin=origin,
                plugin_role="framework",
                injection_name="system_prompt",
                metadata={"content_scope": "full_prompt"},
            ),
            event_bus,
        )

    def _system_prompt_display_content(
        self,
        content: list[ContentPart],
    ) -> list[ContentPart]:
        """Return only the base system prompt parts for UI display events."""
        injected_part_ids = set(self._system_prompt_part_variability_rank)
        if not injected_part_ids and self._system_prompt is not None:
            return deepcopy(self._system_prompt)
        return [
            deepcopy(part)
            for part in content
            if id(part) not in injected_part_ids
        ]

    async def _emit_system_prompt_injected_event(
        self,
        *,
        run_id: str,
        hook_type: str,
        before_part_ids: set[int],
        before_content: list[ContentPart],
        injector: dict[str, str | None],
        event_bus: EventBus | None,
    ) -> None:
        current = list(self._context.get_system_prompt() or [])
        added = [
            deepcopy(part)
            for part in current
            if id(part) not in before_part_ids
        ]
        change_type = "append"
        if not added and current != before_content:
            added = deepcopy(current)
            change_type = "replace"
        if not added:
            return
        await self._emit_event(
            AgentSystemPromptEvent.create(
                run_id=run_id,
                content=added,
                origin=hook_type,
                plugin_id=injector.get("plugin_id"),
                plugin_name=injector.get("plugin_name"),
                plugin_role=injector.get("plugin_role") or "plugin",
                injection_name=injector.get("injection_name"),
                metadata={
                    "content_scope": "injected_segment",
                    "change_type": change_type,
                },
            ),
            event_bus,
        )

    async def _emit_context_injected_events(
        self,
        *,
        run_id: str,
        hook_type: str,
        before_messages: list[Message],
        injector: dict[str, str | None] | None = None,
        event_bus: EventBus | None,
    ) -> None:
        before_ids = {id(message) for message in before_messages}
        user_target = self._last_user_message_target(before_messages)
        injector = injector or self._framework_injector("context")

        for position, message in enumerate(self._context.messages):
            if id(message) in before_ids:
                continue
            role = str(message.get("role", ""))
            if role not in {"user", "assistant", "tool", "system", "error"}:
                continue
            content = message.get("content")
            if not isinstance(content, list) or not content:
                continue
            metadata = message.get("metadata")
            merge_target: Literal["user_message"] | None = None
            merge_position: Literal["before", "after"] | None = None
            target_message_id: str | None = None
            target_message_index: int | None = None
            target_context_message_id: str | None = None
            if role == "user" and hook_type == "before_conversation":
                merge_target = "user_message"
                merge_position = "after"
                if user_target is not None:
                    user_target_message_id = user_target.get("message_id")
                    user_target_message_index = user_target.get("message_index")
                    user_target_context_message_id = user_target.get(
                        "context_message_id"
                    )
                    target_message_id = (
                        user_target_message_id
                        if isinstance(user_target_message_id, str)
                        else None
                    )
                    target_message_index = (
                        user_target_message_index
                        if isinstance(user_target_message_index, int)
                        else None
                    )
                    target_context_message_id = (
                        user_target_context_message_id
                        if isinstance(user_target_context_message_id, str)
                        else None
                    )
                    target_object_id = user_target.get("message_object_id")
                    if isinstance(target_object_id, int):
                        target_position = self._message_position_by_object_id(
                            target_object_id
                        )
                        if target_position is not None and position < target_position:
                            merge_position = "before"

            await self._emit_event(
                AgentContextInjectedEvent.create(
                    run_id=run_id,
                    role=cast(
                        Literal["user", "assistant", "tool", "system", "error"],
                        role,
                    ),
                    content=deepcopy(content),
                    hook_type=hook_type,
                    position=position,
                    plugin_id=injector.get("plugin_id"),
                    plugin_name=injector.get("plugin_name"),
                    plugin_role=injector.get("plugin_role") or "framework",
                    injection_name=injector.get("injection_name"),
                    metadata=metadata if isinstance(metadata, dict) else None,
                    context_message_id=message.get("context_message_id"),
                    merge_target=merge_target,
                    merge_position=merge_position,
                    target_message_id=target_message_id,
                    target_message_index=target_message_index,
                    target_context_message_id=target_context_message_id,
                ),
                event_bus,
            )

    async def _add_hook_injected_user_message(
        self,
        content: str | list[ContentPart],
        *,
        run_id: str,
        hook_type: str,
        injector: dict[str, str | None] | None = None,
        event_bus: EventBus | None,
    ) -> None:
        before_messages = list(self._context.messages)
        self._context.add_user_message(content)
        await self._emit_context_injected_events(
            run_id=run_id,
            hook_type=hook_type,
            before_messages=before_messages,
            injector=injector,
            event_bus=event_bus,
        )

    def _hook_observers(
        self,
        *,
        run_id: str,
        event_bus: EventBus | None,
    ) -> tuple[
        Callable[[str, Callable[..., Any]], None],
        Callable[[str, Callable[..., Any], HookResult | None], Any],
        Callable[[], None],
    ]:
        snapshots: dict[int, tuple[list[Message], set[int], list[ContentPart]]] = {}
        result_injector: dict[str, str | None] | None = None

        def on_hook_start(hook_type: str, hook: Callable[..., Any]) -> None:
            system_prompt = list(self._context.get_system_prompt() or [])
            snapshots[id(hook)] = (
                list(self._context.messages),
                {id(part) for part in system_prompt},
                deepcopy(system_prompt),
            )

        async def on_hook_end(
            hook_type: str,
            hook: Callable[..., Any],
            result: HookResult | None,
        ) -> None:
            nonlocal result_injector
            before_messages, before_part_ids, before_system = snapshots.pop(
                id(hook),
                ([], set(), []),
            )
            injector = self._hook_injector(hook)
            await self._emit_context_injected_events(
                run_id=run_id,
                hook_type=hook_type,
                before_messages=before_messages,
                injector=injector,
                event_bus=event_bus,
            )
            await self._emit_system_prompt_injected_event(
                run_id=run_id,
                hook_type=hook_type,
                before_part_ids=before_part_ids,
                before_content=before_system,
                injector=injector,
                event_bus=event_bus,
            )
            if result is not None:
                result_injector = injector

        def remember_result_injector() -> None:
            self._last_hook_result_injector = result_injector

        return on_hook_start, on_hook_end, remember_result_injector

    @staticmethod
    def _framework_injector(injection_name: str) -> dict[str, str | None]:
        return {
            "plugin_id": None,
            "plugin_name": None,
            "plugin_role": "framework",
            "injection_name": injection_name,
        }

    @staticmethod
    def _hook_injector(hook: Callable[..., Any]) -> dict[str, str | None]:
        owner = getattr(hook, "__self__", None)
        if isinstance(owner, HawiPlugin):
            return {
                "plugin_id": owner.plugin_id,
                "plugin_name": owner.plugin_name,
                "plugin_role": "plugin",
                "injection_name": getattr(hook, "__name__", owner.__class__.__name__),
            }
        return {
            "plugin_id": None,
            "plugin_name": None,
            "plugin_role": "dynamic_hook",
            "injection_name": getattr(hook, "__name__", type(hook).__name__),
        }

    def _consume_last_hook_result_injector(
        self,
        fallback_name: str,
    ) -> dict[str, str | None]:
        injector = self._last_hook_result_injector
        self._last_hook_result_injector = None
        return injector or self._framework_injector(fallback_name)

    @staticmethod
    def _last_user_message_target(
        messages: list[Message],
    ) -> dict[str, str | int | None] | None:
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if message.get("role") != "user":
                continue
            metadata = message.get("metadata")
            message_id = (
                str(metadata.get("message_id"))
                if isinstance(metadata, dict) and metadata.get("message_id")
                else None
            )
            return {
                "message_id": message_id,
                "message_index": index,
                "context_message_id": message.get("context_message_id"),
                "message_object_id": id(message),
            }
        return None

    def _message_position_by_object_id(self, message_object_id: int) -> int | None:
        for index, message in enumerate(self._context.messages):
            if id(message) == message_object_id:
                return index
        return None

    async def _invoke_session_hook(
        self,
        hook_type: str,
        ctx: HookContext,
        event_bus: EventBus | None = None,
    ) -> HookResult | None:
        on_hook_start, on_hook_end, remember_result_injector = self._hook_observers(
            run_id=ctx.run_id,
            event_bus=event_bus,
        )
        result = await self._hooks.invoke_session(
            hook_type,
            ctx,
            on_hook_start=on_hook_start,
            on_hook_end=on_hook_end,
        )
        remember_result_injector()
        return result

    async def _invoke_before_model_call(
        self,
        model: Model,
        ctx: HookContext,
        event_bus: EventBus | None = None,
    ) -> HookResult | None:
        on_hook_start, on_hook_end, remember_result_injector = self._hook_observers(
            run_id=ctx.run_id,
            event_bus=event_bus,
        )
        result = await self._hooks.invoke_before_model_call(
            model,
            ctx,
            on_hook_start=on_hook_start,
            on_hook_end=on_hook_end,
        )
        remember_result_injector()
        return result

    async def _invoke_after_model_call(
        self,
        response: MessageResponse,
        ctx: HookContext,
        event_bus: EventBus | None = None,
    ) -> HookResult | None:
        on_hook_start, on_hook_end, remember_result_injector = self._hook_observers(
            run_id=ctx.run_id,
            event_bus=event_bus,
        )
        result = await self._hooks.invoke_after_model_call(
            response,
            ctx,
            on_hook_start=on_hook_start,
            on_hook_end=on_hook_end,
        )
        remember_result_injector()
        return result

    async def _invoke_before_tool_calling(
        self,
        tool_name: str,
        arguments: dict,
        ctx: HookContext,
    ) -> HookResult | None:
        return await self._hooks.invoke_before_tool_calling(tool_name, arguments, ctx)

    async def _invoke_after_tool_calling(
        self,
        tool_name: str,
        arguments: dict,
        tool_result: ToolResult,
        ctx: HookContext,
    ) -> HookResult | None:
        return await self._hooks.invoke_after_tool_calling(
            tool_name,
            arguments,
            tool_result,
            ctx,
        )

    def run(
        self,
        message: str | list[ContentPart] | None = None,
    ) -> AgentRunResult:
        """Execute agent with a message (synchronous).

        Args:
            message: User message (str or content parts)

        Returns:
            AgentRunResult containing the execution result
        """
        return asyncio.run(self.arun(message))

    async def arun(
        self,
        message: str | list[ContentPart] | None = None,
    ) -> AgentRunResult:
        """Execute agent asynchronously.

        Args:
            message: User message (str or content parts)

        Returns:
            AgentRunResult containing the execution result
        """
        return await self._arun_internal(message=message)

    async def _arun_internal(
        self,
        message: str | list[ContentPart] | None,
        *,
        model: Model | None = None,
        model_error_policy: Optional[ModelErrorPolicyConfig] = None,
        event_bus: EventBus | None = None,
        streaming: bool | None = None,
        message_metadata: dict[str, Any] | None = None,
    ) -> AgentRunResult:
        """Internal async run entry that supports runtime overrides."""
        m = model or self._default_model
        policy = model_error_policy or self._model_error_policy
        effective_event_bus = event_bus or self._event_bus
        effective_streaming = streaming if streaming is not None else self._streaming

        with self._session_lock:
            self._session_active = True
        try:
            return await self._execute(
                message,
                m,
                policy,
                effective_event_bus,
                effective_streaming,
                message_metadata,
            )
        finally:
            with self._session_lock:
                self._session_active = False

    def _create_tool_executor(self) -> ToolExecutor:
        """Return the agent-level tool executor."""
        return self._tool_executor

    @property
    def tool_executor(self) -> ToolExecutor:
        """Agent-level manager for queued tool call requests."""
        return self._tool_executor

    def _build_tool_executor(self) -> ToolExecutor:
        """Build a tool executor bound to the agent's current runtime state."""
        return ToolExecutor(
            agent=self,
            plugin_manager=self._plugin_manager,
            context=self._context,
            emit_event=self._emit_event,
            render_tool_result=self._tool_result_content,
            add_tool_result=self._add_tool_result_with_pending_steer,
            emit_tool_result_message=self._emit_tool_result_message_event,
            emit_materialized_steer_events=self._emit_materialized_steer_events,
            current_tool_calls=self._current_tool_calls,
        )

    async def _persist_interrupted_assistant_message(
        self,
        *,
        run_id: str,
        event_bus: EventBus | None,
        content_parts: list[ContentPart],
        text_handler: StreamBlockAccumulator | None,
        thinking_handler: StreamBlockAccumulator | None,
        reason: str,
    ) -> bool:
        """Persist assistant content that streamed before an interruption."""
        interrupted_content = list(content_parts)
        partials = [
            partial
            for partial in (
                thinking_handler.partial_content() if thinking_handler else None,
                text_handler.partial_content() if text_handler else None,
            )
            if partial is not None
        ]
        for _, part in sorted(partials, key=lambda item: item[0]):
            interrupted_content.append(part)

        if not self._has_persistable_interrupted_content(interrupted_content):
            return False

        metadata = {
            "partial": True,
            "interrupted": True,
            "interrupt_reason": reason,
        }
        context_message_id = self._context.add_assistant_message(
            content=interrupted_content,
            metadata=metadata,
        )
        await self._emit_event(
            AgentMessageAddedEvent.create(
                run_id=run_id,
                role="assistant",
                content=interrupted_content,
                metadata=metadata,
                context_message_id=context_message_id,
            ),
            event_bus,
        )
        return True

    @staticmethod
    def _has_persistable_interrupted_content(content: list[ContentPart]) -> bool:
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "text" and str(part.get("text") or "").strip():
                return True
            if part_type == "reasoning" and (
                str(part.get("reasoning") or "").strip()
                or part.get("signature")
                or part.get("redacted_content")
            ):
                return True
            if part_type == "tool_call" and (part.get("id") or part.get("name")):
                return True
        return False

    async def _execute(
        self,
        message: str | list[ContentPart] | None,
        model: Model,
        policy: ModelErrorPolicyConfig,
        event_bus: EventBus | None,
        streaming: bool,
        message_metadata: dict[str, Any] | None = None,
    ) -> AgentRunResult:
        """Execute agent and return result (pure EventBus-driven)."""

        policy = policy
        state = _ExecutionState()
        run_id = str(uuid.uuid4())[:8]
        state.run_id = run_id
        start_time = time.time()
        self._active_execution_state = state

        if self._last_interrupt_reason:
            await self._recover_unanswered_tool_calls(
                run_id=None,
                event_bus=None,
                reason=self._last_interrupt_reason,
                emit_events=False,
            )

        # Clear any previous interrupt state for fresh execution
        self.clear_interrupt_state()

        # Update tool definitions from PluginManager before execution
        defs = self._plugin_manager.get_tool_definitions()
        self._context.tool_definitions = defs if defs else None

        # Record initial message count to track delta for this invocation
        initial_message_count = len(self._context.messages)

        # Track cumulative usage across all model calls (for multi-turn conversations)
        cumulative_usage: TokenUsage | None = None
        post_conversation_reinvoke_message: str | list[ContentPart] | None = None
        inflight_content_parts: list[ContentPart] = []
        inflight_text_handler: StreamBlockAccumulator | None = None
        inflight_thinking_handler: StreamBlockAccumulator | None = None
        inflight_assistant_message_added = False

        user_content: list[ContentPart] | None = None
        message_metadata = dict(message_metadata) if message_metadata else None
        if message is not None:
            self._last_unsent_tool_results = []
            if isinstance(message, str):
                user_content = [{"type": "text", "text": message}]
            else:
                user_content = message

        # Agent run start
        await self._emit_event(
            AgentRunStartEvent.create(run_id=run_id),
            event_bus,
        )

        # before_session hook
        hook_message_count = len(self._context.messages)
        _hr = await self._invoke_session_hook(
            "before_session",
            HookContext(run_id=run_id, iteration=0),
            event_bus=event_bus,
        )
        self._mark_context_growth_as_model_input(
            message_count_before=hook_message_count,
            started_at=time.time(),
        )
        if _hr and _hr.action == "abort":
            state.should_stop = True
            state.stop_reason = "hook_abort"

        await self._emit_system_prompt_event_if_changed(
            run_id=run_id,
            origin="session_start",
            event_bus=event_bus,
        )

        # Add user message only after session-level system prompt material exists.
        if not state.should_stop and message is not None and user_content is not None:
            context_message_id = self._context.add_user_message(
                message,
                metadata=message_metadata,
            )
            self._refresh_context_usage_snapshot(model)
            await self._emit_event(
                AgentMessageAddedEvent.create(
                    run_id=run_id,
                    role="user",
                    content=user_content,
                    metadata=message_metadata,
                    context_message_id=context_message_id,
                ),
                event_bus,
            )

        skip_before_conversation_hooks = (
            isinstance(message_metadata, dict)
            and message_metadata.get(SKIP_BEFORE_CONVERSATION_HOOKS_METADATA_KEY)
            is True
        )

        # before_conversation hook
        if not state.should_stop and not skip_before_conversation_hooks:
            hook_message_count = len(self._context.messages)
            _hr = await self._invoke_session_hook(
                "before_conversation",
                HookContext(run_id=run_id, iteration=0),
                event_bus=event_bus,
            )
            self._mark_context_growth_as_model_input(
                message_count_before=hook_message_count,
                started_at=time.time(),
            )
            if _hr and _hr.action == "abort":
                state.should_stop = True
                state.stop_reason = "hook_abort"

        try:
            if message is None:
                await self._drain_pending_inputs_to_context(run_id, event_bus)

            while not state.should_stop:
                m = model  # reset per-iteration model (replace_model only affects one call)

                # Check max iterations
                if self._max_iterations is not None and state.iteration >= self._max_iterations:
                    err = MaxIterationsError(f"Maximum iterations ({self._max_iterations}) reached")
                    state.error = err
                    await self._emit_event(
                        AgentErrorEvent.create(run_id=run_id, error=err),
                        event_bus,
                    )
                    break

                state.iteration += 1

                # before_model_call hook
                hook_message_count = len(self._context.messages)
                _hr = await self._invoke_before_model_call(
                    m,
                    HookContext(run_id=run_id, iteration=state.iteration),
                    event_bus=event_bus,
                )
                self._mark_context_growth_as_model_input(
                    message_count_before=hook_message_count,
                    started_at=time.time(),
                )
                if _hr:
                    if _hr.action == "abort":
                        state.should_stop = True
                        state.stop_reason = "hook_abort"
                        break
                    elif _hr.action == "replace_model" and _hr.model is not None:
                        m = _hr.model  # use replacement model for this iteration only
                    elif _hr.action == "restart_turn":
                        continue  # skip model call, go to next loop iteration
                    elif _hr.action == "reinvoke" and _hr.message is not None:
                        self._mark_model_input_started()
                        await self._add_hook_injected_user_message(
                            _hr.message,
                            run_id=run_id,
                            hook_type="before_model_call.reinvoke",
                            injector=self._consume_last_hook_result_injector(
                                "before_model_call.reinvoke"
                            ),
                            event_bus=event_bus,
                        )
                        await self._emit_event(
                            AgentRunStopEvent.create(
                                run_id=run_id,
                                stop_reason="hook_reinvoke",
                                duration_ms=(time.time() - start_time) * 1000,
                                usage=cumulative_usage,
                            ),
                            event_bus,
                        )
                        return await self._arun_internal(
                            message=None,
                            model=model,
                            event_bus=event_bus,
                            streaming=streaming,
                        )

                if await self._maybe_auto_compact(m, state, event_bus):
                    # Compaction rewrites history, so the old absolute index may
                    # no longer be meaningful for this invocation's delta view.
                    initial_message_count = min(
                        initial_message_count,
                        len(self._context.messages),
                    )

                await self._emit_system_prompt_event_if_changed(
                    run_id=run_id,
                    origin="model_input",
                    event_bus=event_bus,
                )

                # Model stream start
                request_id = f"{run_id}-{state.iteration}"
                context_usage = self.context_usage(m)
                await self._emit_event(
                    ModelStreamStartEvent.create(request_id=request_id),
                    event_bus,
                )

                # Call model with streaming
                content_parts: list[ContentPart] = []
                tool_calls: list[ToolCallPart] = []
                tool_executor = self._create_tool_executor()
                early_tool_promises: list[ToolCallPromise] = []
                last_tool_request_blocker: str | None = None
                stop_reason = "end_turn"
                usage: TokenUsage | None = None
                model_call_start = time.time()
                model_input_started_at = self._consume_model_input_started(
                    model_call_start
                )
                first_token_at: float | None = None
                response_content: list[ContentPart] = []

                # Content block handlers for processing different chunk types
                text_handler = StreamBlockAccumulator.create_text_handler()
                thinking_handler = StreamBlockAccumulator.create_thinking_handler()
                tool_handler = StreamBlockAccumulator.create_tool_handler()
                inflight_content_parts = content_parts
                inflight_text_handler = text_handler
                inflight_thinking_handler = thinking_handler
                inflight_assistant_message_added = False

                # Get model response stream (streaming or non-streaming unified)
                model_stream_gen = self._call_model_with_retry(
                    m, policy, state, request_id, event_bus, streaming
                )
                try:
                    async for chunk in model_stream_gen:
                        chunk_received_at = time.time()
                        if state.error:
                            if isinstance(state.error, AgentError):
                                await self._emit_event(
                                    AgentErrorEvent.create(run_id=run_id, error=state.error),
                                    event_bus,
                                )
                            await self._emit_event(
                                AgentRunStopEvent.create(
                                    run_id=run_id,
                                    stop_reason="error",
                                    duration_ms=(time.time() - start_time) * 1000,
                                    usage=cumulative_usage,
                                ),
                                event_bus,
                            )
                            break

                        # Handle both DeltaPart (dict) and Event (Pydantic model)
                        if isinstance(chunk, Event):
                            # Skip ModelEvent objects - they are for observation only
                            continue

                        chunk_type = chunk.get("type", "")
                        if first_token_at is None and self._is_first_output_chunk(chunk):
                            first_token_at = chunk_received_at
                        idx = chunk.get("index", 0)

                        # Get or create handler for this chunk type
                        if chunk_type == "text_delta":
                            handler = text_handler
                        elif chunk_type in {"reasoning_delta", "signature_delta"}:
                            handler = thinking_handler
                        elif chunk_type == "tool_call_delta":
                            handler = tool_handler
                        elif chunk_type == "finish":
                            stop_reason = chunk.get("stop_reason") or "end_turn"
                            usage_data = chunk.get("usage")
                            if usage_data:
                                usage = normalize_token_usage(usage_data)
                                cumulative_usage = merge_token_usage(cumulative_usage, usage)
                            continue
                        else:
                            continue  # Unknown chunk type

                        # Handle the chunk with appropriate handler
                        for part, events in handler.handle(chunk, request_id, is_streaming=streaming):
                            # Emit all events through _emit_event for consistent handling
                            for event in events:
                                await self._emit_event(event, event_bus)

                            if part is not None:
                                content_parts.append(part)
                                if part["type"] == "tool_call":
                                    tool_call_part = cast(ToolCallPart, part)
                                    tool_calls.append(tool_call_part)
                                    promise = tool_executor.enqueue_call(
                                        ToolCallRequest(
                                            tool_call=tool_call_part,
                                            run_id=run_id,
                                            iteration=state.iteration,
                                            blocked_by=last_tool_request_blocker,
                                            event_bus=event_bus,
                                            materialize_pending_steer=False,
                                            add_to_context=False,
                                            emit_final_event=False,
                                            run_injection_handlers=True,
                                            audit_action="queue",
                                        )
                                    )
                                    early_tool_promises.append(promise)
                                    last_tool_request_blocker = (
                                        tool_call_part.get("id")
                                        or promise.request_id
                                    )
                finally:
                    # Ensure the model stream generator is properly closed
                    await model_stream_gen.aclose()

                if state.error:
                    tool_executor.clear()
                    if not inflight_assistant_message_added:
                        inflight_assistant_message_added = await self._persist_interrupted_assistant_message(
                            run_id=run_id,
                            event_bus=event_bus,
                            content_parts=inflight_content_parts,
                            text_handler=inflight_text_handler,
                            thinking_handler=inflight_thinking_handler,
                            reason="error",
                        )
                    # Send error event before breaking (error occurred during streaming)
                    if isinstance(state.error, AgentError):
                        await self._emit_event(
                            AgentErrorEvent.create(run_id=run_id, error=state.error),
                            event_bus,
                        )
                    break

                model_call_end = time.time()
                # Model stream stop
                await self._emit_event(
                    ModelStreamStopEvent.create(
                        request_id=request_id,
                        stop_reason=stop_reason,
                    ),
                    event_bus,
                )

                # Model metadata (usage + per-call latency)
                usage_output_tokens = (
                    usage.get("output_tokens")
                    if usage is not None and isinstance(usage.get("output_tokens"), int)
                    else None
                )
                estimated_output_tokens = estimate_content_tokens(content_parts)
                context_output_tokens = (
                    usage_output_tokens
                    if usage_output_tokens is not None and usage_output_tokens > 0
                    else estimated_output_tokens
                )
                prompt_context_tokens = context_usage.used_tokens
                metadata_context_tokens = prompt_context_tokens + context_output_tokens
                metadata_context_ratio = context_usage.usage_ratio
                metadata_context_source = context_usage.source
                provider_context_tokens = usage_context_tokens(usage)
                if provider_context_tokens is not None and provider_context_tokens > 0:
                    prompt_context_tokens = provider_context_tokens
                    metadata_context_tokens = provider_context_tokens + context_output_tokens
                    metadata_context_source = "provider_usage"
                if context_usage.max_context_tokens:
                    metadata_context_ratio = min(
                        1.0,
                        metadata_context_tokens / context_usage.max_context_tokens,
                    )
                else:
                    metadata_context_ratio = None
                metadata_remaining_tokens = (
                    max(0, context_usage.max_context_tokens - metadata_context_tokens)
                    if context_usage.max_context_tokens is not None
                    else None
                )
                self._context.set_context_usage(
                    ContextUsageSnapshot(
                        used_tokens=metadata_context_tokens,
                        max_context_tokens=context_usage.max_context_tokens,
                        usage_ratio=metadata_context_ratio,
                        remaining_tokens=metadata_remaining_tokens,
                        source=metadata_context_source,
                    )
                )
                decode_tokens = (
                    usage_output_tokens
                    if usage_output_tokens is not None and usage_output_tokens > 0
                    else estimated_output_tokens or None
                )
                timing_metadata = self._model_timing_metadata(
                    started_at=model_input_started_at,
                    first_token_at=first_token_at,
                    completed_at=model_call_end,
                    prefill_tokens=self._prefill_tokens_for_timing(
                        usage,
                        prompt_context_tokens,
                    ),
                    decode_tokens=decode_tokens,
                )
                await self._emit_event(
                    ModelMetadataEvent.create(
                        request_id=request_id,
                        usage=usage,
                        latency_ms=(model_call_end - model_call_start) * 1000,
                        **timing_metadata,
                        context_tokens=metadata_context_tokens,
                        max_context_tokens=context_usage.max_context_tokens,
                        context_ratio=metadata_context_ratio,
                        context_source=metadata_context_source,
                    ),
                    event_bus,
                )

                # Build content parts for the assistant message
                # Content parts include text/reasoning, but NOT tool_calls (they go in separate field)
                response_content: list[ContentPart] = content_parts

                # Build response object for after_model_call hook
                response = MessageResponse(
                    id=request_id,
                    role="assistant",
                    content=response_content,
                    stop_reason=stop_reason,
                    usage=usage,
                )
                
                # after_model_call hook
                hook_message_count = len(self._context.messages)
                _hr = await self._invoke_after_model_call(
                    response,
                    HookContext(
                        run_id=run_id,
                        iteration=state.iteration,
                        duration_ms=(time.time() - model_call_start) * 1000,
                    ),
                    event_bus=event_bus,
                )
                self._mark_context_growth_as_model_input(
                    message_count_before=hook_message_count,
                    started_at=time.time(),
                )
                if _hr:
                    if _hr.action == "abort":
                        tool_executor.clear()
                        state.should_stop = True
                        state.stop_reason = "hook_abort"
                    elif _hr.action == "reinvoke" and _hr.message is not None:
                        tool_executor.clear()
                        self._mark_model_input_started()
                        await self._add_hook_injected_user_message(
                            _hr.message,
                            run_id=run_id,
                            hook_type="after_model_call.reinvoke",
                            injector=self._consume_last_hook_result_injector(
                                "after_model_call.reinvoke"
                            ),
                            event_bus=event_bus,
                        )
                        await self._emit_event(
                            AgentRunStopEvent.create(
                                run_id=run_id,
                                stop_reason="hook_reinvoke",
                                duration_ms=(time.time() - start_time) * 1000,
                                usage=cumulative_usage,
                            ),
                            event_bus,
                        )
                        return await self._arun_internal(
                            message=None,
                            model=model,
                            event_bus=event_bus,
                            streaming=streaming,
                        )

                # Add assistant message to context
                # tool_calls are now included in content as ToolCallPart items
                context_message_id = self._context.add_assistant_message(
                    content=response_content
                )
                self._refresh_context_usage_snapshot(m)
                inflight_assistant_message_added = True

                # Emit event for assistant message added
                await self._emit_event(
                    AgentMessageAddedEvent.create(
                        run_id=run_id,
                        role="assistant",
                        content=response_content,
                        context_message_id=context_message_id,
                    ),
                    event_bus,
                )
                inflight_content_parts = []
                inflight_text_handler = None
                inflight_thinking_handler = None

                if state.should_stop:
                    await self._emit_event(
                        AgentRunStopEvent.create(
                            run_id=run_id,
                            stop_reason=state.stop_reason or "hook_abort",
                            duration_ms=(time.time() - start_time) * 1000,
                            usage=cumulative_usage,
                        ),
                        event_bus,
                    )
                    break

                # Check if tool calls need to be executed
                if not tool_calls:
                    if await self._drain_pending_inputs_to_context(run_id, event_bus):
                        continue
                    if (
                        state.last_auto_compact_iteration != state.iteration
                        and await self._maybe_auto_compact(m, state, event_bus)
                    ):
                        initial_message_count = min(
                            initial_message_count,
                            len(self._context.messages),
                        )
                    # No tool calls, we're done
                    duration_ms = (time.time() - start_time) * 1000
                    await self._emit_event(
                        AgentRunStopEvent.create(
                            run_id=run_id,
                            stop_reason=stop_reason or "end_turn",
                            duration_ms=duration_ms,
                            usage=cumulative_usage,
                        ),
                        event_bus,
                    )
                    break

                # Execute tool calls through the executor in model order.
                # Calls in one assistant turn may depend on earlier calls.
                if early_tool_promises:
                    tool_batch = await tool_executor.resolve_batch(
                        early_tool_promises,
                        run_id=run_id,
                        iteration=state.iteration,
                        event_bus=event_bus,
                        is_interrupted=self._check_interrupt,
                        materialize_pending_steer=True,
                    )
                else:
                    tool_batch = await tool_executor.execute_batch(
                        tool_calls,
                        run_id=run_id,
                        iteration=state.iteration,
                        event_bus=event_bus,
                        is_interrupted=self._check_interrupt,
                        materialize_pending_steer=True,
                    )
                state.tool_calls.extend(tool_batch.records)

                if tool_batch.records:
                    self._mark_tool_results_unsent(tool_batch.records)

                if tool_batch.control is not None:
                    if tool_batch.control.action == "abort":
                        state.should_stop = True
                        state.stop_reason = "hook_abort"
                        await self._emit_event(
                            AgentRunStopEvent.create(
                                run_id=run_id,
                                stop_reason="hook_abort",
                                duration_ms=(time.time() - start_time) * 1000,
                                usage=cumulative_usage,
                            ),
                            event_bus,
                        )
                        break
                    if (
                        tool_batch.control.action == "reinvoke"
                        and tool_batch.control.message is not None
                    ):
                        state.pending_reinvoke_message = tool_batch.control.message
                        state.stop_reason = "hook_reinvoke"
                        self._mark_model_input_started()
                        await self._add_hook_injected_user_message(
                            state.pending_reinvoke_message,
                            run_id=run_id,
                            hook_type="after_tool_calling.reinvoke",
                            injector=self._framework_injector(
                                "after_tool_calling.reinvoke"
                            ),
                            event_bus=event_bus,
                        )
                        await self._emit_event(
                            AgentRunStopEvent.create(
                                run_id=run_id,
                                stop_reason="hook_reinvoke",
                                duration_ms=(time.time() - start_time) * 1000,
                                usage=cumulative_usage,
                            ),
                            event_bus,
                        )
                        return await self._arun_internal(
                            message=None,
                            model=model,
                            event_bus=event_bus,
                            streaming=streaming,
                        )

                # Check if execution was interrupted
                if self._check_interrupt():
                    stop_reason = "interrupted"
                    await self._emit_event(
                        AgentRunStopEvent.create(
                            run_id=run_id,
                            stop_reason=stop_reason,
                            duration_ms=(time.time() - start_time) * 1000,
                            usage=cumulative_usage,
                        ),
                        event_bus,
                    )
                    break

                # Continue loop for next iteration

        except asyncio.CancelledError:
            reason = self._last_interrupt_reason or "cancelled"
            if not inflight_assistant_message_added:
                inflight_assistant_message_added = await self._persist_interrupted_assistant_message(
                    run_id=run_id,
                    event_bus=event_bus,
                    content_parts=inflight_content_parts,
                    text_handler=inflight_text_handler,
                    thinking_handler=inflight_thinking_handler,
                    reason=reason,
                )
            await self._recover_unanswered_tool_calls(
                run_id=run_id,
                event_bus=event_bus,
                reason=reason,
                emit_events=True,
            )
            await self._emit_event(
                AgentRunStopEvent.create(
                    run_id=run_id,
                    stop_reason="interrupted",
                    duration_ms=(time.time() - start_time) * 1000,
                    usage=cumulative_usage,
                ),
                event_bus,
            )
            raise
        except AgentError as e:
            # AgentError 已经被包装过了，发送 error event 然后重新抛出
            await self._emit_event(
                AgentErrorEvent.create(run_id=run_id, error=e),
                event_bus,
            )
            # Send run_stop event for errors
            await self._emit_event(
                AgentRunStopEvent.create(
                    run_id=run_id,
                    stop_reason="error",
                    duration_ms=(time.time() - start_time) * 1000,
                    usage=cumulative_usage,
                ),
                event_bus,
            )
            raise
        except Exception as e:
            # 包装为 AgentError，保留原始异常
            err = AgentError("tool_execution", f"{type(e).__name__}: {e}")
            state.error = err
            await self._emit_event(
                AgentErrorEvent.create(run_id=run_id, error=err),
                event_bus,
            )
            # Send run_stop event for errors
            await self._emit_event(
                AgentRunStopEvent.create(
                    run_id=run_id,
                    stop_reason="error",
                    duration_ms=(time.time() - start_time) * 1000,
                    usage=cumulative_usage,
                ),
                event_bus,
            )
            # 使用 raise from 保留原始异常的调用栈
            raise err from e

        finally:
            _final_ctx = HookContext(
                run_id=run_id,
                iteration=state.iteration,
                duration_ms=(time.time() - start_time) * 1000,
                error=state.error if isinstance(state.error, Exception) else None,
            )
            # after_conversation hook
            _hr = await self._invoke_session_hook(
                "after_conversation",
                _final_ctx,
                event_bus=event_bus,
            )
            if (
                _hr is not None
                and _hr.action == "reinvoke"
                and _hr.message is not None
                and state.error is None
            ):
                post_conversation_reinvoke_message = _hr.message

            # after_session hook
            await self._invoke_session_hook(
                "after_session",
                _final_ctx,
                event_bus=event_bus,
            )

            # Clear the live reference so SessionManager.snapshot_runtime()
            # reports an idle agent.
            self._active_execution_state = None
            self._tool_executor.clear()

        if post_conversation_reinvoke_message is not None:
            self._mark_model_input_started()
            await self._add_hook_injected_user_message(
                post_conversation_reinvoke_message,
                run_id=run_id,
                hook_type="after_conversation.reinvoke",
                injector=self._consume_last_hook_result_injector(
                    "after_conversation.reinvoke"
                ),
                event_bus=event_bus,
            )
            return await self._arun_internal(
                message=None,
                model=model,
                event_bus=event_bus,
                streaming=streaming,
            )

        self._pending_model_input_started_at = None

        # Build and return result
        duration_ms = (time.time() - start_time) * 1000
        if state.error:
            stop_reason = "error"
        elif state.stop_reason is not None:
            stop_reason = state.stop_reason
        elif state.iteration > 0 and not state.tool_calls:
            stop_reason = "end_turn"
        else:
            stop_reason = "tool_use"

        # Get the last assistant message as response
        all_messages = self._context.messages
        response = None
        for msg in reversed(all_messages):
            if msg["role"] == "assistant":
                response = msg
                break

        # Return only the delta messages added in this invocation
        delta_messages = all_messages[initial_message_count:]

        return AgentRunResult(
            stop_reason=stop_reason,
            messages=delta_messages,
            response=response,
            usage=cumulative_usage,
            tool_calls=state.tool_calls,
            error=str(state.error) if state.error else None,
        )

    async def _call_model_with_retry(
        self,
        model: Model,
        policy: ModelErrorPolicyConfig,
        state: _ExecutionState,
        request_id: str,
        event_bus: EventBus | None,
        streaming: bool,
    ) -> AsyncGenerator[DeltaPart | Event, None]:
        """Call model with retry logic (streaming and non-streaming unified).

        Yields DeltaPart or Event for each chunk of content from the model.
        Accumulates content to build complete response for tool call handling.
        """
        last_error = None
        max_retries = 0

        # Calculate max retries from policy
        for p in policy.values():
            if p.action == "retry" and isinstance(p, ModelErrorRetryPolicy) and p.retry_count > max_retries:
                max_retries = p.retry_count

        attempt = 0
        context_retry_attempt = 0
        context_compact_attempted = False
        max_context_retries = 3
        while attempt <= max_retries:
            try:
                request = self._context.prepare_request()

                # Unified event consumption: Model always returns AsyncGenerator
                async for event in model.ainvoke(
                    messages=request.messages,
                    streaming=streaming,
                    system=request.system,
                    tools=request.tools,
                    cache_point=request.cache_point,
                    cache_tool_definitions=request.cache_tool_definitions,
                ):
                    yield event
                self._last_unsent_tool_results = []
                return

            except ModelError as e:
                last_error = e

                if (
                    isinstance(e, ContextLengthError)
                    and context_retry_attempt < max_context_retries
                    and await self._truncate_last_unsent_tool_results_for_context_retry(e)
                ):
                    context_retry_attempt += 1
                    if event_bus:
                        await event_bus.publish_async(
                            ModelRetryEvent.create(
                                request_id=request_id,
                                error_type=e.error_type,
                                attempt=context_retry_attempt,
                                max_retries=max_context_retries,
                                error_message=str(e),
                            )
                        )
                    continue

                if isinstance(e, ContextLengthError) and not context_compact_attempted:
                    context_compact_attempted = True
                    if await self._force_auto_compact_for_context_length_error(
                        model,
                        state,
                        event_bus,
                        e,
                    ):
                        if event_bus:
                            await event_bus.publish_async(
                                ModelRetryEvent.create(
                                    request_id=request_id,
                                    error_type=e.error_type,
                                    attempt=1,
                                    max_retries=max_context_retries,
                                    error_message=str(e),
                                )
                            )
                        continue

                # 直接使用 ModelError 的 error_type
                policy_for_error = policy[e.error_type]

                if policy_for_error.action == "stop":
                    # Emit error event and gracefully stop
                    state.error = e
                    if event_bus:
                        await event_bus.publish_async(ModelErrorEvent.create(error=e))
                    return

                if attempt < max_retries:
                    # Emit retry event before sleeping
                    if event_bus:
                        await event_bus.publish_async(
                            ModelRetryEvent.create(
                                request_id=request_id,
                                error_type=e.error_type,
                                attempt=attempt + 1,
                                max_retries=max_retries,
                                error_message=str(e),
                            )
                        )
                    await asyncio.sleep(min(2 ** attempt, 60))
                attempt += 1

        if last_error:
            # All retries exhausted for retryable errors
            err = ModelError("network", f"Model call failed after {attempt} attempts: {last_error}")
            state.error = err
            if event_bus:
                await event_bus.publish_async(ModelErrorEvent.create(error=err))

    def _convert_event_to_delta_parts(
        self,
        event_type: str,
        data: dict[str, Any],
    ) -> list[DeltaPart]:
        """Convert model event to DeltaPart sequence.

        This ensures non-streaming mode produces the same DeltaPart sequence
        as streaming mode, allowing unified processing.
        """
        parts: list[DeltaPart] = []
        idx = data.get("block_index", 0)

        if event_type == "model.content_block_start":
            block_type = data.get("block_type", "text")
            if block_type == "text":
                parts.append({
                    "type": "text_delta",
                    "index": idx,
                    "delta": "",
                    "is_start": True,
                    "is_end": False,
                })
            elif block_type == "reasoning":
                parts.append({
                    "type": "reasoning_delta",
                    "index": idx,
                    "delta": "",
                    "is_start": True,
                    "is_end": False,
                })

        elif event_type == "model.content_block_delta":
            part = data.get("part", {})
            delta_type = part.get("type", "text_delta")
            parts.append({
                "type": delta_type,
                "index": idx,
                "delta": part.get("delta", ""),
                "is_start": part.get("is_start", False),
                "is_end": part.get("is_end", False),
            })

        elif event_type == "model.content_block_stop":
            # Determine type from content
            content = data.get("content", [])
            if content:
                content_type = content[0].get("type", "text")
                if content_type == "text":
                    parts.append({
                        "type": "text_delta",
                        "index": idx,
                        "delta": "",
                        "is_start": False,
                        "is_end": True,
                    })
                elif content_type == "reasoning":
                    parts.append({
                        "type": "reasoning_delta",
                        "index": idx,
                        "delta": "",
                        "is_start": False,
                        "is_end": True,
                    })

        elif event_type == "model.tool_call_block_start":
            parts.append({
                "type": "tool_call_delta",
                "index": idx,
                "id": data.get("tool_call_id", ""),
                "name": data.get("tool_name", ""),
                "arguments_delta": "",
                "is_start": True,
                "is_end": False,
            })

        elif event_type == "model.tool_call_block_delta":
            parts.append({
                "type": "tool_call_delta",
                "index": idx,
                "id": None,
                "name": None,
                "arguments_delta": data.get("arguments_delta", ""),
                "is_start": False,
                "is_end": False,
            })

        elif event_type == "model.tool_call_block_stop":
            parts.append({
                "type": "tool_call_delta",
                "index": idx,
                "id": data.get("tool_call_id", ""),
                "name": data.get("tool_name", ""),
                "arguments_delta": "",
                "is_start": False,
                "is_end": True,
            })

        elif event_type == "model.stream_stop":
            parts.append({
                "type": "finish",
                "stop_reason": data.get("stop_reason", "end_turn"),
                "usage": None,  # usage is carried by the subsequent model.metadata event
            })

        return parts

    async def _prepare_tool_arguments(
        self,
        tool: AgentTool,
        arguments: dict[str, Any],
        *,
        tool_call_id: str,
        state: _ExecutionState,
        run_injection_handlers: bool,
    ) -> PreparedToolArguments:
        """Compatibility wrapper around :class:`ToolExecutor` argument prep."""
        return await self._create_tool_executor().prepare_tool_arguments(
            tool,
            arguments,
            tool_call_id=tool_call_id,
            run_id=state.run_id,
            iteration=state.iteration,
            run_injection_handlers=run_injection_handlers,
        )

    def _inject_tool_runtime_context(
        self,
        tool: AgentTool,
        tool_arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Compatibility wrapper for runtime context injection."""
        return self._create_tool_executor().inject_tool_runtime_context(
            tool,
            tool_arguments,
        )

    async def _execute_tool(
        self,
        tool_call: ToolCallPart,
        state: _ExecutionState,
        *,
        event_bus: EventBus | None = None,
        materialize_pending_steer: bool = True,
    ) -> ToolCallRecord:
        """Compatibility wrapper around :class:`ToolExecutor` single-call run."""
        return await self._create_tool_executor().execute_call(
            tool_call,
            run_id=state.run_id,
            iteration=state.iteration,
            event_bus=event_bus,
            materialize_pending_steer=materialize_pending_steer,
        )

    def review_pending_tools(self) -> list[dict[str, Any]]:
        """Get list of pending tool calls awaiting audit.

        Returns:
            List of pending tool call info dicts with keys:
            - tool_call_id: str
            - tool_name: str
            - arguments: dict
            - requested_at: float
        """
        return [
            {
                "tool_call_id": p.tool_call_id,
                "tool_name": p.tool_name,
                "arguments": p.arguments,
                "requested_at": p.requested_at,
            }
            for p in self._context.get_pending_tool_calls()
        ]

    async def approve_pending_tools(
        self,
        tool_call_ids: list[str],
        event_bus: EventBus | None = None,
    ) -> list[ToolCallRecord]:
        """Approve and execute pending tool calls.

        Args:
            tool_call_ids: List of tool_call_ids to approve
            event_bus: Optional event bus for publishing events

        Returns:
            List of ToolCallRecord for executed tools
        """
        approved, _ = self._context.audit_pending_tool_calls(approve=tool_call_ids, reject=[])
        records: list[ToolCallRecord] = []
        executor = self._create_tool_executor()

        for pending in approved:
            tool_call: ToolCallPart = {
                "type": "tool_call",
                "id": pending.tool_call_id,
                "name": pending.tool_name,
                "arguments": pending.arguments,
            }
            record = await executor.execute_call(
                tool_call,
                run_id="audit",
                iteration=0,
                event_bus=event_bus,
                run_injection_handlers=False,
                audit_action="execute",
            )
            records.append(record)

        if records:
            self._mark_tool_results_unsent(records)

        return records

    def reject_pending_tools(self, tool_call_ids: list[str]) -> list[str]:
        """Reject pending tool calls.

        Args:
            tool_call_ids: List of tool_call_ids to reject

        Returns:
            List of rejected tool_call_ids
        """
        _, rejected = self._context.audit_pending_tool_calls(approve=[], reject=tool_call_ids)
        return [r.tool_call_id for r in rejected]
