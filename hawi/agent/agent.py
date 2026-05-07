"""HawiAgent - Core agent implementation with tool execution and plugin support.

This module implements the HawiAgent class that orchestrates LLM interaction,
tool execution, and plugin hooks for agent workflows.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import os
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Coroutine, Literal, Mapping, Callable, cast


from hawi.models import (
    Model,
    ContentPart,
    DeltaPart,
    TokenUsage,
    ToolCallPart,
    ToolDefinition,
    model_registry,
)
from hawi.models.message import Message, MessageResponse
from hawi.models.usage import merge_token_usage, normalize_token_usage
from hawi.plugin import HawiPlugin
from hawi.plugin import PluginManager
from hawi.plugin.hook_context import HookContext, HookResult
from hawi.tool.types import AgentTool, ToolParameterInjectionContext, ToolResult

from hawi.errors import (
    AgentErrorType,
    ModelErrorType,
    HawiError,
    AgentError,
    ModelError,
    MaxIterationsError,
    ToolNotFoundError,
    ToolExecutionError,
)
from hawi.events import (
    Event,
    EventBus,
    EventHandler,
    SyncEventHandler,
    AgentErrorEvent,
    AgentMessageAddedEvent,
    AgentRunStartEvent,
    AgentRunStopEvent,
    AgentToolCallEvent,
    AgentToolResultPartEvent,
    AgentToolResultEvent,
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
    CONTEXT_COMPACTION_PROMPT,
    CONTEXT_COMPACTION_SUMMARY_PREFIX,
    ToolCallContext,
)
from .result import AgentRunResult, ToolCallRecord
from .stream_accumulator import StreamBlockAccumulator




@dataclass
class ModelErrorPolicy:
    """模型失败处理策略"""
    action: Literal[
        'retry',
        'notify_agent',
        'stop',
    ]

class ModelErrorRetryPolicy(ModelErrorPolicy):
    def __init__(self, retry_count:int):
        super().__init__('retry')
        self.retry_count:int = retry_count

class ModelErrorNotifyPolicy(ModelErrorPolicy):
    def __init__(self):
        super().__init__('notify_agent')

class ModelErrorStopPolicy(ModelErrorPolicy):
    def __init__(self):
        super().__init__('stop')

ModelErrorPolicyConfig = Mapping[ModelErrorType, ModelErrorPolicy]


@dataclass
class AutoCompactConfig:
    """Configuration for automatic context compaction."""

    enabled: bool = True
    max_context_tokens: int = 128_000
    trigger_tokens: int | None = None
    trigger_ratio: float = 0.8
    keep_last_messages: int = 8
    min_messages: int = 12
    summary_max_output_tokens: int = 2048
    max_transcript_chars: int = 120_000
    prompt: str = CONTEXT_COMPACTION_PROMPT
    summary_prefix: str = CONTEXT_COMPACTION_SUMMARY_PREFIX

    def token_limit(self) -> int:
        """Return the estimated-token threshold that triggers compaction."""
        if self.trigger_tokens is not None:
            return self.trigger_tokens
        return max(1, int(self.max_context_tokens * self.trigger_ratio))


@dataclass
class _ExecutionState:
    """Internal execution state during agent run."""

    iteration: int = 0
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    error: HawiError | str | None = None
    should_stop: bool = False
    run_id: str = ""


@dataclass
class _PreparedToolArguments:
    """Arguments split into tool-visible and framework-visible parts."""

    tool_arguments: dict[str, Any]
    injected_arguments: dict[str, Any] = field(default_factory=dict)
    short_circuit_result: ToolResult | None = None


class SteerPartMergeMode(str, Enum):
    """Preferred steer lowering strategy for the related model."""

    APPEND_TO_TOOL_RESULT = "append_to_tool_result"
    USER_MESSAGE_TEMPLATE = "user_message_template"
    TOOL_RESULT_ASSISTANT_TEMPLATE_AND_USER_MESSAGE = (
        "tool_result_assistant_template_and_user_message"
    )

@dataclass
class PendingInput:
    """Queued user input awaiting materialization into context messages."""

    id: str
    content: list[ContentPart]
    candidate_tool_call_ids: tuple[str, ...]
    preferred_merge_mode: SteerPartMergeMode | None = None


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

        Note:
            Both `plugins` and `plugin_factories` can be used together.
            Factories are invoked first during initialization.
        """
        # Resolve model from registry if string is provided
        if isinstance(model, str):
            model = model_registry.create_model(model)
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

        # Set up tool call context for runtime injection
        self._context.tool_call_context = ToolCallContext(agent=self)

        # Initialize interrupt state for cooperative cancellation
        self._cancel_event = asyncio.Event()
        self._current_tool_calls: list[ToolCallPart] = []
        self._interrupted_tool_call_ids: list[str] = []
        self._last_interrupt_reason: str | None = None
        self._steer_lock = threading.RLock()
        self._pending_inputs: list[PendingInput] = []
        self._session_lock = threading.RLock()
        self._session_active = False
        self._autonomous_run_task: asyncio.Task[AgentRunResult] | None = None

    @classmethod
    def _default_model_error_policy(cls) -> ModelErrorPolicyConfig:
        return defaultdict(ModelErrorStopPolicy, {
            'network': ModelErrorRetryPolicy(retry_count=10),
            'throttle': ModelErrorRetryPolicy(retry_count=3),
        })

    @property
    def plugins(self) -> PluginManager:
        """Get the plugin manager for accessing and modifying plugins/tools/hooks."""
        return self._plugin_manager

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

    def set_context(self, context: AgentContext) -> None:
        """Replace the agent's context.

        Args:
            context: New context to use
        """
        self._context = context

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
        # Reset current model state before replacing
        self._default_model.reset()
        self._default_model = model

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
        new_agent.set_context(self._context.copy())
        return new_agent

    def fork(self) -> HawiAgent:
        """Alias for clone().

        Returns:
            New HawiAgent instance with copied state
        """
        return self.clone()

    @staticmethod
    def _normalize_auto_compact(
        config: AutoCompactConfig | dict[str, Any] | bool | None,
        model: Model | None = None,
    ) -> AutoCompactConfig:
        """Normalize user-facing auto-compact options."""
        model_max_context_tokens = None
        if model is not None:
            getter = getattr(model, "get_max_context_tokens", None)
            if callable(getter):
                value = getter()
                if isinstance(value, int) and value > 0:
                    model_max_context_tokens = value
        if isinstance(config, AutoCompactConfig):
            return config
        if isinstance(config, dict):
            values = dict(config)
            if model_max_context_tokens is not None and "max_context_tokens" not in values:
                values["max_context_tokens"] = model_max_context_tokens
            return AutoCompactConfig(**values)
        if isinstance(config, bool):
            return AutoCompactConfig(
                enabled=config,
                max_context_tokens=model_max_context_tokens or 128_000,
            )
        return AutoCompactConfig(
            enabled=True,
            max_context_tokens=model_max_context_tokens or 128_000,
        )

    def _context_limit_for_model(self, model: Model) -> int | None:
        getter = getattr(model, "get_max_context_tokens", None)
        if callable(getter):
            value = getter()
            if isinstance(value, int) and value > 0:
                return value
        if self._auto_compact.max_context_tokens > 0:
            return self._auto_compact.max_context_tokens
        return None

    def interrupt(self, reason: str = "user") -> list[str]:
        """Interrupt current agent execution.

        Signals the agent to stop after the current operation completes.
        This is a cooperative cancellation mechanism - the agent will check
        the interrupt flag at safe points and stop gracefully.

        Args:
            reason: Reason for interruption (e.g., "user", "scheduler", "timeout")

        Returns:
            List of tool_call_ids that were currently executing when interrupted
        """
        self.on_interrupt(reason)
        self._cancel_event.set()
        self._last_interrupt_reason = reason
        interrupted_ids = [tc.get("id", "") for tc in self._current_tool_calls]
        self._interrupted_tool_call_ids.extend(interrupted_ids)
        return interrupted_ids

    def on_interrupt(self, reason: str = "user") -> None:
        """Interrupt hook (no-op by default).

        Subclasses or integrations can override this to react to interrupt
        requests without changing the default cooperative cancel behavior.
        """
        return None

    def clear_interrupt_state(self) -> None:
        """Clear interrupt state for a fresh execution.

        Should be called before starting a new agent run.
        """
        self._cancel_event.clear()
        self._interrupted_tool_call_ids.clear()
        self._current_tool_calls.clear()
        with self._steer_lock:
            self._pending_inputs.clear()

    def _check_interrupt(self) -> bool:
        """Check if an interrupt has been requested.

        Returns:
            True if interrupted, False otherwise
        """
        return self._cancel_event.is_set()

    def _interrupt_tool_result_content(self, reason: str) -> str:
        return f"Tool call interrupted before completion (reason: {reason})."

    async def _recover_unanswered_tool_calls(
        self,
        *,
        run_id: str | None,
        event_bus: EventBus | None,
        reason: str,
        emit_events: bool,
    ) -> None:
        content = self._interrupt_tool_result_content(reason)
        recovered = self._context.add_missing_tool_results(content)
        self._last_interrupt_reason = None
        if not emit_events or not run_id:
            return
        for item in recovered:
            await self._emit_event(
                AgentToolResultEvent.create(
                    run_id=run_id,
                    tool_call_id=item.tool_call_id,
                    success=False,
                    result_preview=content,
                    duration_ms=0.0,
                    result_obj=ToolResult(success=False, error=content),
                ),
                event_bus,
            )

    @property
    def has_active_tool_calls(self) -> bool:
        """Whether the agent is currently waiting on one or more tool calls."""
        return len(self._current_tool_calls) > 0

    def steer(
        self,
        content: str | list[ContentPart],
        *,
        merge_mode: SteerPartMergeMode = (
            SteerPartMergeMode.TOOL_RESULT_ASSISTANT_TEMPLATE_AND_USER_MESSAGE
        ),
    ) -> str:
        """Queue steer content for later materialization.

        Args:
            content: Steering message content.
            merge_mode: Preferred lowering strategy when this input is consumed
                during a tool-result path.

        Returns:
            A steer identifier for tracing/debugging.
        """
        steer_content = self._normalize_content_parts(content)
        steer_id = str(uuid.uuid4())[:8]
        should_start_new_loop = False
        with self._steer_lock:
            candidate_tool_call_ids = tuple(
                tc.get("id", "")
                for tc in self._current_tool_calls
                if tc.get("id")
            )
            self._pending_inputs.append(
                PendingInput(
                    id=steer_id,
                    content=steer_content,
                    candidate_tool_call_ids=candidate_tool_call_ids,
                    preferred_merge_mode=merge_mode,
                )
            )
            if self.has_active_tool_calls:
                return steer_id
            with self._session_lock:
                should_start_new_loop = not self._session_active

        if should_start_new_loop:
            self._ensure_pending_turn_loop()
        return steer_id

    def _normalize_content_parts(
        self,
        content: str | list[ContentPart],
    ) -> list[ContentPart]:
        """Normalize content input into a list of ContentPart."""
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        return list(content)

    def _serialize_content_parts(self, content: list[ContentPart]) -> str:
        """Serialize content parts into readable plain text."""
        chunks: list[str] = []
        for part in content:
            part_type = part.get("type")
            if part_type == "text":
                chunks.append(part.get("text", ""))
            elif part_type == "reasoning":
                chunks.append(part.get("reasoning") or "")
            elif part_type == "steer":
                nested_content = part.get("content", [])
                if isinstance(nested_content, list):
                    nested_text = self._serialize_content_parts(
                        cast(list[ContentPart], nested_content)
                    )
                else:
                    nested_text = str(nested_content)
                if nested_text:
                    chunks.append(nested_text)
            elif part_type == "tool_result":
                nested_content = part.get("content", [])
                if isinstance(nested_content, str):
                    chunks.append(nested_content)
                elif isinstance(nested_content, list):
                    nested_text = self._serialize_content_parts(
                        cast(list[ContentPart], nested_content)
                    )
                    if nested_text:
                        chunks.append(nested_text)
                else:
                    chunks.append(str(nested_content))
            else:
                chunks.append(str(part))
        return "\n".join(chunk for chunk in chunks if chunk.strip())

    def compact(
        self,
        *,
        model: Model | None = None,
        prompt: str | None = None,
        keep_last_messages: int | None = None,
    ) -> ContextCompactionRecord | None:
        """Synchronously compact the current conversation context."""
        return asyncio.run(
            self.acompact(
                model=model,
                prompt=prompt,
                keep_last_messages=keep_last_messages,
            )
        )

    async def acompact(
        self,
        *,
        model: Model | None = None,
        prompt: str | None = None,
        keep_last_messages: int | None = None,
        config: AutoCompactConfig | None = None,
    ) -> ContextCompactionRecord | None:
        """Compact older context into a model-generated handoff summary."""
        cfg = config or self._auto_compact
        keep_last = (
            keep_last_messages
            if keep_last_messages is not None
            else cfg.keep_last_messages
        )
        if self._context.compaction_tail_start(keep_last) <= 0:
            return None

        summary = await self._generate_compaction_summary(
            model or self._default_model,
            prompt=prompt or cfg.prompt,
            max_output_tokens=cfg.summary_max_output_tokens,
            max_transcript_chars=cfg.max_transcript_chars,
        )
        return self._context.compact_with_summary(
            summary,
            keep_last=keep_last,
            summary_prefix=cfg.summary_prefix,
        )

    async def _maybe_auto_compact(
        self,
        model: Model,
        state: _ExecutionState,
    ) -> bool:
        """Run automatic compaction if the configured threshold is crossed."""
        cfg = self._auto_compact
        if not cfg.enabled:
            return False
        if self.has_active_tool_calls:
            return False
        if len(self._context.messages) < cfg.min_messages:
            return False
        if self._context.estimate_tokens() < cfg.token_limit():
            return False

        record = await self.acompact(model=model, config=cfg)
        if record is not None:
            state.iteration = max(state.iteration, 0)
        return record is not None

    async def _generate_compaction_summary(
        self,
        model: Model,
        *,
        prompt: str,
        max_output_tokens: int,
        max_transcript_chars: int,
    ) -> str:
        """Ask the model to summarize the current context for compaction."""
        transcript = self._build_compaction_transcript(
            self._context.messages,
            max_chars=max_transcript_chars,
        )
        summary_request: Message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Summarize the following Hawi conversation transcript "
                        "for continuation after context compaction.\n\n"
                        f"{transcript}"
                    ),
                }
            ],
            "name": None,
            "metadata": None,
        }
        system_prompt: list[ContentPart] = [{"type": "text", "text": prompt}]

        try:
            summary = await self._collect_model_text(
                model,
                messages=[summary_request],
                system=system_prompt,
                max_output_tokens=max_output_tokens,
                streaming=False,
            )
        except NotImplementedError:
            summary = await self._collect_model_text(
                model,
                messages=[summary_request],
                system=system_prompt,
                max_output_tokens=max_output_tokens,
                streaming=True,
            )

        summary = summary.strip()
        if summary:
            return summary
        return self._fallback_compaction_summary(self._context.messages)

    async def _collect_model_text(
        self,
        model: Model,
        *,
        messages: list[Message],
        system: list[ContentPart],
        max_output_tokens: int,
        streaming: bool,
    ) -> str:
        """Collect text deltas from one direct model call."""
        chunks: list[str] = []
        async for delta in model.ainvoke(
            messages=messages,
            streaming=streaming,
            system=system,
            tools=None,
            max_output_tokens=max_output_tokens,
        ):
            if isinstance(delta, Event):
                continue
            if delta.get("type") == "text_delta":
                chunks.append(str(delta.get("delta", "")))
        return "".join(chunks)

    def _build_compaction_transcript(
        self,
        messages: list[Message],
        *,
        max_chars: int,
    ) -> str:
        """Render Hawi messages into compact plain text for summarization."""
        rendered: list[str] = ["<conversation>"]
        for index, message in enumerate(messages, 1):
            rendered.append(f"\n## Message {index}: {message['role']}")
            metadata = message.get("metadata") or {}
            source = metadata.get("source")
            if source:
                rendered.append(f"source: {source}")
            rendered.append(self._render_message_content_for_compaction(message))
        rendered.append("\n</conversation>")

        text = "\n".join(part for part in rendered if part)
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        head_chars = max(0, max_chars // 4)
        tail_chars = max(0, max_chars - head_chars)
        return (
            text[:head_chars]
            + "\n\n...[transcript truncated for compaction prompt budget]...\n\n"
            + text[-tail_chars:]
        )

    def _render_message_content_for_compaction(self, message: Message) -> str:
        """Render one message's content in a summary-friendly format."""
        lines: list[str] = []
        for part in message.get("content", []):
            if not isinstance(part, dict):
                lines.append(str(part))
                continue
            part_type = part.get("type")
            if part_type == "tool_call":
                lines.append(
                    "tool_call "
                    f"{part.get('name', 'unknown')}({part.get('id', '')}): "
                    f"{json.dumps(part.get('arguments', {}), ensure_ascii=False)}"
                )
            elif part_type == "tool_result":
                nested = part.get("content", "")
                if isinstance(nested, list):
                    nested_text = self._serialize_content_parts(
                        cast(list[ContentPart], nested)
                    )
                else:
                    nested_text = str(nested)
                lines.append(
                    "tool_result "
                    f"{part.get('tool_call_id', '')}"
                    f"{' error' if part.get('is_error') else ''}: "
                    f"{nested_text}"
                )
            elif part_type == "steer":
                nested = part.get("content", [])
                if isinstance(nested, list):
                    steer_text = self._serialize_content_parts(
                        cast(list[ContentPart], nested)
                    )
                else:
                    steer_text = str(nested)
                lines.append(
                    "steer: "
                    + steer_text
                )
            else:
                lines.append(self._serialize_content_parts([cast(ContentPart, part)]))
        return "\n".join(line for line in lines if line.strip())

    def _fallback_compaction_summary(self, messages: list[Message]) -> str:
        """Build a deterministic fallback if the summarizer returns no text."""
        recent_user_messages: list[str] = []
        for message in reversed(messages):
            if message["role"] != "user":
                continue
            text = self._serialize_content_parts(list(message.get("content", [])))
            if text:
                recent_user_messages.append(text)
            if len(recent_user_messages) >= 3:
                break
        recent_user_messages.reverse()
        recent = "\n".join(f"- {text}" for text in recent_user_messages)
        return (
            "The previous conversation was compacted automatically, but the "
            "summary model returned no text. Continue from the recent preserved "
            "messages. Recent user requests:\n"
            f"{recent or '- No user request text available.'}"
        )

    async def _drain_pending_inputs_to_context(
        self,
        run_id: str,
        event_bus: EventBus | None,
    ) -> bool:
        """Move queued pending inputs into the conversation as plain user messages."""
        with self._steer_lock:
            pending_inputs = self._pending_inputs[:]
            self._pending_inputs.clear()

        if not pending_inputs:
            return False

        for pending in pending_inputs:
            self._context.add_user_message(pending.content)
            await self._emit_event(
                AgentMessageAddedEvent.create(
                    run_id=run_id,
                    role="user",
                    content=pending.content,
                ),
                event_bus,
            )
        return True

    def _clear_autonomous_run_task(self, task: asyncio.Task[AgentRunResult]) -> None:
        """Drop the autonomous run task reference after completion."""
        with self._session_lock:
            if self._autonomous_run_task is task:
                self._autonomous_run_task = None

    async def _run_pending_turns(self) -> AgentRunResult:
        """Execute queued turns using the agent's current configuration."""
        return await self._arun_internal(message=None)

    def _ensure_pending_turn_loop(self) -> None:
        """Start a new loop to process queued pending turns when idle."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._run_pending_turns())
            return

        with self._session_lock:
            if self._session_active:
                return
            if self._autonomous_run_task is not None and not self._autonomous_run_task.done():
                return
            task = loop.create_task(self._run_pending_turns())
            task.add_done_callback(self._clear_autonomous_run_task)
            self._autonomous_run_task = task

    def _add_tool_result_with_pending_steer(
        self,
        tool_call_id: str,
        content: str | list[ContentPart],
        *,
        is_error: bool = False,
        materialize_pending_steer: bool = True,
    ) -> None:
        """Add a tool result and materialize one matching pending input as steer."""
        tool_result_content = self._normalize_content_parts(content)
        self._context.add_tool_result(
            tool_call_id=tool_call_id,
            content=tool_result_content,
            is_error=is_error,
        )

        if materialize_pending_steer:
            self._materialize_pending_steer_for_tool_results([tool_call_id])

    def _materialize_pending_steer_for_tool_results(
        self,
        tool_call_ids: list[str],
    ) -> None:
        """Append pending steer messages after a completed tool-result batch."""
        if not tool_call_ids:
            return

        tool_call_id_set = set(tool_call_ids)
        materialized: list[tuple[PendingInput, str]] = []
        with self._steer_lock:
            remaining: list[PendingInput] = []
            for item in self._pending_inputs:
                matched_tool_call_id = next(
                    (
                        candidate
                        for candidate in item.candidate_tool_call_ids
                        if candidate in tool_call_id_set
                    ),
                    None,
                )
                if matched_tool_call_id is None:
                    remaining.append(item)
                    continue
                materialized.append((item, matched_tool_call_id))
            self._pending_inputs = remaining

        for pending_input, matched_tool_call_id in materialized:
            steer_part: ContentPart = {
                "type": "steer",
                "content": list(pending_input.content),
                "tool_call_id": matched_tool_call_id,
                "preferred_merge_mode": (
                    pending_input.preferred_merge_mode.value
                    if pending_input.preferred_merge_mode is not None
                    else None
                ),
            }
            self._context.add_user_message([steer_part])

    async def _invoke_session_hook(self, hook_type: str, ctx: HookContext) -> HookResult | None:
        """Invoke before/after_session and before/after_conversation hooks: (agent, ctx)."""
        for hook in self._plugin_manager.get_hooks(hook_type):
            result = hook(self, ctx)
            if inspect.isawaitable(result):
                result = await result
            if result is not None:
                return result
        return None

    async def _invoke_before_model_call(self, model: Model, ctx: HookContext) -> HookResult | None:
        """Invoke before_model_call hook: (agent, model, ctx)."""
        for hook in self._plugin_manager.get_hooks("before_model_call"):
            result = hook(self, model, ctx)
            if inspect.isawaitable(result):
                result = await result
            if result is not None:
                return result
        return None

    async def _invoke_after_model_call(self, response: "MessageResponse", ctx: HookContext) -> HookResult | None:
        """Invoke after_model_call hook: (agent, response, ctx)."""
        for hook in self._plugin_manager.get_hooks("after_model_call"):
            result = hook(self, response, ctx)
            if inspect.isawaitable(result):
                result = await result
            if result is not None:
                return result
        return None

    async def _invoke_before_tool_calling(self, tool_name: str, arguments: dict, ctx: HookContext) -> HookResult | None:
        """Invoke before_tool_calling hook: (agent, tool_name, arguments, ctx)."""
        for hook in self._plugin_manager.get_hooks("before_tool_calling"):
            result = hook(self, tool_name, arguments, ctx)
            if inspect.isawaitable(result):
                result = await result
            if result is not None:
                return result
        return None

    async def _invoke_after_tool_calling(self, tool_name: str, arguments: dict, tool_result: ToolResult, ctx: HookContext) -> HookResult | None:
        """Invoke after_tool_calling hook: (agent, tool_name, arguments, result, ctx)."""
        for hook in self._plugin_manager.get_hooks("after_tool_calling"):
            result = hook(self, tool_name, arguments, tool_result, ctx)
            if inspect.isawaitable(result):
                result = await result
            if result is not None:
                return result
        return None

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
    ) -> AgentRunResult:
        """Internal async run entry that supports runtime overrides."""
        m = model or self._default_model
        policy = model_error_policy or self._model_error_policy
        effective_event_bus = event_bus or self._event_bus
        effective_streaming = streaming if streaming is not None else self._streaming

        with self._session_lock:
            self._session_active = True
        try:
            return await self._execute(message, m, policy, effective_event_bus, effective_streaming)
        finally:
            with self._session_lock:
                self._session_active = False

    @property
    def event_bus(self) -> EventBus:
        """Get the agent's EventBus for event subscriptions."""
        return self._event_bus

    def subscribe(
        self,
        callback: EventHandler,
        event_types: list[str] | None = None,
        maxsize: int = 100,
    ) -> None:
        """Subscribe to agent events (non-blocking, supports sync/async handlers).

        Args:
            callback: Callback function to handle events (sync or async)
            event_types: List of event types to subscribe to, None for all
            maxsize: Queue size for the handler
        """
        self._event_bus.subscribe(callback, event_types, maxsize)

    def subscribe_blocking(
        self,
        callback: SyncEventHandler,
        event_types: list[str] | None = None,
    ) -> None:
        """Subscribe to agent events (blocking, sync handler only).

        The handler executes synchronously in the publisher's thread.

        Args:
            callback: Sync callback function to handle events
            event_types: List of event types to subscribe to, None for all

        Raises:
            ValueError: If callback is an async function
        """
        self._event_bus.subscribe_blocking(callback, event_types)

    def unsubscribe(
        self,
        callback: Callable[[Event], None],
    ) -> bool:
        """Unsubscribe from agent events (delegates to EventBus).

        Args:
            callback: Callback function to remove
            wait: Whether to wait for queued events to be processed
            timeout: Timeout for waiting (seconds)

        Returns:
            True if successfully unsubscribed
        """
        return self._event_bus.unsubscribe(callback)

    async def _emit_event(
        self,
        event: Event,
        event_bus: EventBus | None,
    ) -> Event:
        """Emit event to event bus(es).
        
        Always publishes to self._event_bus to ensure events reach
        subscribers registered via agent.subscribe(). Additionally publishes
        to the provided event_bus if different from self._event_bus.
        """
        # Always publish to self._event_bus
        await self._event_bus.publish_async(event)
        
        # Also publish to external event_bus if provided and different
        if event_bus is not None and event_bus is not self._event_bus:
            await event_bus.publish_async(event)

        # Dump event to file if configured
        if self._dump_manager is not None:
            self._dump_manager.dump(event)

        return event

    async def _execute(
        self,
        message: str | list[ContentPart] | None,
        model: Model,
        policy: ModelErrorPolicyConfig,
        event_bus: EventBus | None,
        streaming: bool,
    ) -> AgentRunResult:
        """Execute agent and return result (pure EventBus-driven)."""

        policy = policy
        state = _ExecutionState()
        run_id = str(uuid.uuid4())[:8]
        state.run_id = run_id
        start_time = time.time()

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

        # Add user message if provided
        if message is not None:
            # Normalize message to list[ContentPart]
            if isinstance(message, str):
                user_content: list[ContentPart] = [{"type": "text", "text": message}]
            else:
                user_content = message
            
            self._context.add_user_message(message)
            await self._emit_event(
                AgentMessageAddedEvent.create(
                    run_id=run_id,
                    role="user",
                    content=user_content,
                ),
                event_bus,
            )

        # Agent run start
        await self._emit_event(
            AgentRunStartEvent.create(run_id=run_id),
            event_bus,
        )

        # before_session hook
        _hr = await self._invoke_session_hook("before_session", HookContext(run_id=run_id, iteration=0))
        if _hr and _hr.action == "abort":
            state.should_stop = True

        # before_conversation hook
        if not state.should_stop:
            _hr = await self._invoke_session_hook("before_conversation", HookContext(run_id=run_id, iteration=0))
            if _hr and _hr.action == "abort":
                state.should_stop = True

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
                _hr = await self._invoke_before_model_call(
                    m,
                    HookContext(run_id=run_id, iteration=state.iteration),
                )
                if _hr:
                    if _hr.action == "abort":
                        state.should_stop = True
                        break
                    elif _hr.action == "replace_model" and _hr.model is not None:
                        m = _hr.model  # use replacement model for this iteration only
                    elif _hr.action == "restart_turn":
                        continue  # skip model call, go to next loop iteration
                    elif _hr.action == "reinvoke" and _hr.message is not None:
                        self._context.add_user_message(_hr.message)
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

                if await self._maybe_auto_compact(m, state):
                    # Compaction rewrites history, so the old absolute index may
                    # no longer be meaningful for this invocation's delta view.
                    initial_message_count = min(
                        initial_message_count,
                        len(self._context.messages),
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
                stop_reason = "end_turn"
                usage: TokenUsage | None = None
                model_call_start = time.time()
                response_content: list[ContentPart] = []

                # Content block handlers for processing different chunk types
                text_handler = StreamBlockAccumulator.create_text_handler()
                thinking_handler = StreamBlockAccumulator.create_thinking_handler()
                tool_handler = StreamBlockAccumulator.create_tool_handler()

                # Get model response stream (streaming or non-streaming unified)
                model_stream_gen = self._call_model_with_retry(
                    m, policy, state, request_id, event_bus, streaming
                )
                try:
                    async for chunk in model_stream_gen:
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
                        idx = chunk.get("index", 0)

                        # Get or create handler for this chunk type
                        if chunk_type == "text_delta":
                            handler = text_handler
                        elif chunk_type == "reasoning_delta":
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
                                    tool_calls.append(part)
                finally:
                    # Ensure the model stream generator is properly closed
                    await model_stream_gen.aclose()

                if state.error:
                    # Send error event before breaking (error occurred during streaming)
                    if isinstance(state.error, AgentError):
                        await self._emit_event(
                            AgentErrorEvent.create(run_id=run_id, error=state.error),
                            event_bus,
                        )
                    break

                # Model stream stop
                await self._emit_event(
                    ModelStreamStopEvent.create(
                        request_id=request_id,
                        stop_reason=stop_reason,
                    ),
                    event_bus,
                )

                # Model metadata (usage + per-call latency)
                await self._emit_event(
                    ModelMetadataEvent.create(
                        request_id=request_id,
                        usage=usage,
                        latency_ms=(time.time() - model_call_start) * 1000,
                        context_tokens=context_usage.used_tokens,
                        max_context_tokens=context_usage.max_context_tokens,
                        context_ratio=context_usage.usage_ratio,
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
                _hr = await self._invoke_after_model_call(
                    response,
                    HookContext(
                        run_id=run_id,
                        iteration=state.iteration,
                        duration_ms=(time.time() - model_call_start) * 1000,
                    ),
                )
                if _hr:
                    if _hr.action == "abort":
                        state.should_stop = True
                    elif _hr.action == "reinvoke" and _hr.message is not None:
                        self._context.add_user_message(_hr.message)
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
                self._context.add_assistant_message(content=response_content)

                # Emit event for assistant message added
                await self._emit_event(
                    AgentMessageAddedEvent.create(
                        run_id=run_id,
                        role="assistant",
                        content=response_content,
                    ),
                    event_bus,
                )

                # Check if tool calls need to be executed
                if not tool_calls:
                    if await self._drain_pending_inputs_to_context(run_id, event_bus):
                        continue
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

                # Execute tool calls
                active_batch_tool_calls = [
                    tc for tc in tool_calls if tc not in self._current_tool_calls
                ]
                completed_tool_call_ids: list[str] = []
                self._current_tool_calls.extend(active_batch_tool_calls)
                try:
                    for tc in tool_calls:
                        # Check for interrupt before executing tool
                        if self._check_interrupt():
                            # Interrupted - stop processing remaining tools
                            break

                        if tc in self._current_tool_calls:
                            self._current_tool_calls.remove(tc)
                        self._current_tool_calls.insert(0, tc)
                        record = await self._execute_tool(
                            tc,
                            state,
                            event_bus=event_bus,
                            materialize_pending_steer=False,
                        )
                        state.tool_calls.append(record)
                        completed_tool_call_ids.append(record.tool_call_id)
                        await self._emit_event(
                            AgentToolResultEvent.create(
                                run_id=run_id,
                                tool_call_id=record.tool_call_id,
                                success=record.result.success,
                                result_preview=str(record.result.output),
                                duration_ms=record.duration_ms,
                                result_obj=record.result,
                            ),
                            event_bus,
                        )
                        if tc in self._current_tool_calls:
                            self._current_tool_calls.remove(tc)
                finally:
                    for tc in active_batch_tool_calls:
                        if tc in self._current_tool_calls:
                            self._current_tool_calls.remove(tc)

                if completed_tool_call_ids and not self._check_interrupt():
                    self._materialize_pending_steer_for_tool_results(completed_tool_call_ids)

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
            _hr = await self._invoke_session_hook("after_conversation", _final_ctx)
            if (
                _hr is not None
                and _hr.action == "reinvoke"
                and _hr.message is not None
                and state.error is None
            ):
                post_conversation_reinvoke_message = _hr.message

            # after_session hook
            await self._invoke_session_hook("after_session", _final_ctx)

        if post_conversation_reinvoke_message is not None:
            self._context.add_user_message(post_conversation_reinvoke_message)
            return await self._arun_internal(
                message=None,
                model=model,
                event_bus=event_bus,
                streaming=streaming,
            )

        # Build and return result
        duration_ms = (time.time() - start_time) * 1000
        if state.error:
            stop_reason = "error"
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
        for attempt in range(max_retries + 1):
            try:
                request = self._context.prepare_request()

                # Unified event consumption: Model always returns AsyncGenerator
                async for event in model.ainvoke(
                    messages=request.messages,
                    streaming=streaming,
                    system=[part for part in (request.system or ()) if part['type'] == 'text'],
                    tools=request.tools,
                ):
                    yield event
                return

            except ModelError as e:
                last_error = e

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

        if last_error:
            # All retries exhausted for retryable errors
            err = ModelError("network", f"Model call failed after {attempt + 1} attempts: {last_error}")
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
    ) -> _PreparedToolArguments:
        """Validate and strip framework-injected parameters before tool calls."""
        tool_arguments = dict(arguments)
        injections = self._plugin_manager.get_tool_parameter_injections(tool)
        if not injections:
            return _PreparedToolArguments(tool_arguments=tool_arguments)

        from hawi.tool._utils import validate_parameters

        injected_schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                injection.name: injection.schema_copy()
                for injection in injections
            },
        }
        required = [injection.name for injection in injections if injection.required]
        if required:
            injected_schema["required"] = required

        is_valid, errors = validate_parameters(arguments, injected_schema)
        if not is_valid:
            return _PreparedToolArguments(
                tool_arguments=tool_arguments,
                short_circuit_result=ToolResult(
                    success=False,
                    error=f"Injected parameter validation failed: {'; '.join(errors)}",
                ),
            )

        injected_arguments: dict[str, Any] = {}
        for injection in injections:
            if injection.name in tool_arguments:
                injected_arguments[injection.name] = tool_arguments.pop(injection.name)

        prepared = _PreparedToolArguments(
            tool_arguments=tool_arguments,
            injected_arguments=injected_arguments,
        )
        if not run_injection_handlers:
            return prepared

        handler_context = ToolParameterInjectionContext(
            agent=self,
            tool=tool,
            tool_name=tool.name,
            tool_call_id=tool_call_id,
            run_id=state.run_id,
            iteration=state.iteration,
            arguments=dict(arguments),
            injected_arguments=dict(injected_arguments),
        )

        for injection in injections:
            if injection.name not in injected_arguments or injection.handler is None:
                continue
            try:
                handler_result = injection.handler(
                    handler_context,
                    injected_arguments[injection.name],
                )
                if inspect.isawaitable(handler_result):
                    handler_result = await handler_result
            except Exception as e:
                prepared.short_circuit_result = ToolResult(
                    success=False,
                    error=(
                        "Injected parameter handler failed: "
                        f"{type(e).__name__}: {e}"
                    ),
                )
                break
            if isinstance(handler_result, ToolResult):
                prepared.short_circuit_result = handler_result
                break

        return prepared

    def _inject_tool_runtime_context(
        self,
        tool: AgentTool,
        tool_arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Inject Hawi's runtime tool-call context into tool-visible arguments."""
        prepared = dict(tool_arguments)
        context_param = getattr(tool, "context", None)
        if context_param and self._context.tool_call_context:
            prepared[context_param] = self._context.tool_call_context
        return prepared

    async def _execute_tool(
        self,
        tool_call: ToolCallPart,
        state: _ExecutionState,
        *,
        event_bus: EventBus | None = None,
        materialize_pending_steer: bool = True,
    ) -> ToolCallRecord:
        """Execute a single tool call."""
        tool_name = tool_call["name"]
        arguments = tool_call["arguments"]
        tool_call_id = tool_call["id"]

        start_time = time.time()

        await self._emit_event(
            AgentToolCallEvent.create(
                run_id=state.run_id,
                tool_name=tool_name,
                arguments=arguments,
                tool_call_id=tool_call_id,
            ),
            event_bus,
        )

        # Find tool early so hook context can include the tool object
        tool = self._plugin_manager.get_tool(tool_name)
        audit_pending = False

        # before_tool_calling hook
        _before_ctx = HookContext(
            run_id=state.run_id,
            iteration=state.iteration,
            tool_call_id=tool_call_id,
            tool=tool,
        )
        _hr = await self._invoke_before_tool_calling(tool_name, arguments, _before_ctx)
        if _hr and _hr.action == "skip":
            result = _hr.tool_result or ToolResult(success=False, error="Hook skipped tool without providing a result")
        elif tool is None:
            err = ToolNotFoundError(f"Tool '{tool_name}' not found")
            result = ToolResult(success=False, error=f"{err.__class__.__name__}: {err.message}")
        else:
            prepared = await self._prepare_tool_arguments(
                tool,
                arguments,
                tool_call_id=tool_call_id,
                state=state,
                run_injection_handlers=True,
            )
            if prepared.short_circuit_result is not None:
                result = prepared.short_circuit_result
            elif getattr(tool, "audit", False):
                # Audit mode: cache the raw tool call for review. Injected
                # parameters stay visible to the reviewer but are stripped on
                # approval before the real tool runs.
                self._context._add_pending_tool_call(tool_call_id, tool_name, arguments)
                audit_pending = True
                result = ToolResult(
                    success=True,
                    output=f"[AUDIT PENDING] Tool '{tool_name}' has been submitted for review. "
                           f"Use review_pending_tools() to check status and approve/reject."
                )
            else:
                # Prepare arguments with runtime context injection if needed
                tool_arguments = self._inject_tool_runtime_context(
                    tool,
                    prepared.tool_arguments,
                )

                try:
                    # Validate parameters before execution
                    is_valid, errors = tool.validate_parameters(tool_arguments)
                    if not is_valid:
                        result = ToolResult(
                            success=False,
                            error=f"Parameter validation failed: {'; '.join(errors)}"
                        )
                    else:
                        # Call arun and check if result is async generator
                        owner = self._plugin_manager.get_tool_owner(tool_name)
                        event_scope = (
                            owner.plugin_event_context(
                                run_id=state.run_id,
                                tool_call_id=tool_call_id,
                                tool_name=tool_name,
                                iteration=state.iteration,
                            )
                            if owner is not None
                            else contextlib.nullcontext()
                        )
                        with event_scope:
                            raw_result = await tool.arun(**tool_arguments)

                            # Check if result is async generator (async tool streaming)
                            if inspect.isasyncgen(raw_result):
                                # Async generator: stream results part by part
                                parts: list[str] = []
                                # Type cast to AsyncGenerator for iteration
                                from typing import cast
                                async_gen = cast(AsyncGenerator[Any, None], raw_result)
                                async for part in async_gen:
                                    parts.append(str(part))
                                    # Emit partial result event
                                    await self._emit_event(
                                        AgentToolResultPartEvent.create(
                                            run_id=state.run_id,
                                            tool_call_id=tool_call_id,
                                            part=str(part),
                                            part_index=len(parts) - 1,
                                            is_final=False,
                                        ),
                                        self._event_bus,
                                    )
                                # Final part event
                                await self._emit_event(
                                    AgentToolResultPartEvent.create(
                                        run_id=state.run_id,
                                        tool_call_id=tool_call_id,
                                        part="",
                                        part_index=len(parts),
                                        is_final=True,
                                    ),
                                    self._event_bus,
                                )
                                # Combine all parts as final result
                                full_output = "".join(parts)
                                result = ToolResult(success=True, output=full_output)
                            else:
                                # Normal result: wrap in ToolResult
                                if isinstance(raw_result, ToolResult):
                                    result = raw_result
                                else:
                                    result = ToolResult(success=True, output=raw_result)
                except Exception as e:
                    # 包装为 ToolExecutionError，保留原始异常
                    err = ToolExecutionError(f"Tool '{tool_name}' execution failed: {e}", details={"original": e})
                    # All errors return to model as string (per design requirement)
                    result = ToolResult(success=False, error=f"{err.__class__.__name__}: {err.message}")

        duration_ms = (time.time() - start_time) * 1000

        # after_tool_calling hook
        await self._invoke_after_tool_calling(
            tool_name, arguments, result,
            HookContext(
                run_id=state.run_id,
                iteration=state.iteration,
                tool_call_id=tool_call_id,
                tool=tool,
                duration_ms=duration_ms,
            ),
        )

        # Add tool result to context (unless audit pending - will be added after approval)
        if not audit_pending:
            # Build result content: include both output and error
            output_str = result.output if isinstance(result.output, str) else str(result.output) if result.output else ""
            if not result.success and result.error:
                # On failure, include error information
                result_content = f"Error: {result.error}"
                if output_str:
                    result_content = f"Output before error:\n{output_str}\n\n{result_content}"
            else:
                result_content = output_str
            self._add_tool_result_with_pending_steer(
                tool_call_id=tool_call_id,
                content=result_content,
                is_error=not result.success,
                materialize_pending_steer=materialize_pending_steer,
            )

        return ToolCallRecord(
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            duration_ms=duration_ms,
            tool_call_id=tool_call_id,
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

        for pending in approved:
            # Execute the approved tool
            tool = self._plugin_manager.get_tool(pending.tool_name)
            if tool is None:
                result = ToolResult(
                    success=False,
                    error=f"Tool '{pending.tool_name}' not found during approval execution"
                )
            else:
                prepared = await self._prepare_tool_arguments(
                    tool,
                    pending.arguments,
                    tool_call_id=pending.tool_call_id,
                    state=_ExecutionState(run_id="audit", iteration=0),
                    run_injection_handlers=False,
                )
                if prepared.short_circuit_result is not None:
                    result = prepared.short_circuit_result
                else:
                    # Prepare arguments with runtime context injection if needed
                    tool_arguments = self._inject_tool_runtime_context(
                        tool,
                        prepared.tool_arguments,
                    )

                    try:
                        result = await tool.ainvoke(tool_arguments)
                    except Exception as e:
                        result = ToolResult(success=False, error=f"{type(e).__name__}: {e}")

            # Create record
            record = ToolCallRecord(
                tool_name=pending.tool_name,
                arguments=pending.arguments,
                result=result,
                duration_ms=0.0,  # Could track actual execution time if needed
                tool_call_id=pending.tool_call_id,
            )
            records.append(record)

            # Add tool result to context
            output_str = result.output if isinstance(result.output, str) else str(result.output) if result.output else ""
            if not result.success and result.error:
                result_content = f"Error: {result.error}"
                if output_str:
                    result_content = f"Output before error:\n{output_str}\n\n{result_content}"
            else:
                result_content = output_str
            self._add_tool_result_with_pending_steer(
                tool_call_id=pending.tool_call_id,
                content=result_content,
                is_error=not result.success,
            )

            # Emit event if event bus provided
            if event_bus is not None:
                await self._emit_event(
                    AgentToolResultEvent.create(
                        run_id="audit",
                        tool_call_id=record.tool_call_id,
                        success=record.result.success,
                        result_preview=str(record.result.output)[:100],
                        duration_ms=record.duration_ms,
                        result_obj=record.result,
                    ),
                    event_bus,
                )

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
