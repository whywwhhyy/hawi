"""HawiAgent - Core agent implementation with tool execution and plugin support.

This module implements the HawiAgent class that orchestrates LLM interaction,
tool execution, and plugin hooks for agent workflows.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import time
import uuid
from collections import defaultdict
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any, Optional, Coroutine, Literal, Mapping, Callable, cast


from hawi.models import (
    Model,
    ContentPart,
    DeltaPart,
    TextPart,
    TokenUsage,
    ToolCallPart,
    ToolDefinition,
)
from hawi.plugin import HawiPlugin
from hawi.tool.types import AgentTool, ToolResult

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
from .events import (
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
    ModelContentBlockDeltaEvent,
    ModelContentBlockStartEvent,
    ModelContentBlockStopEvent,
    ModelErrorEvent,
    ModelRetryEvent,
    ModelToolCallBlockDeltaEvent,
    ModelToolCallBlockStartEvent,
    ModelToolCallBlockStopEvent,
    ModelStreamStartEvent,
    ModelStreamStopEvent,
    DumpManager,
)
from .context import AgentContext, ToolCallContext
from .result import AgentRunResult, ToolCallRecord

@dataclass
class ContentBlockHandler:
    """内容块处理器，统一处理 text/thinking/tool_call 等块类型。

    通过实例方法处理不同类型的内容块，避免代码重复。

    事件通过 EventBus 异步发布，方法返回完成的 ContentPart（块结束时）或 None。

    Example:
        # 文本块处理器
        text_handler = ContentBlockHandler.create_text_handler()

        # 工具调用处理器
        tool_handler = ContentBlockHandler.create_tool_handler()

        # 使用
        part = await handler.handle(chunk, request_id, event_bus)
        if part is not None:
            content_parts.append(part)
    """

    # 块类型配置
    stream_part_type: str  # "text_delta", "thinking_delta", "tool_call_delta"
    block_type: Literal["text", "thinking", "tool_use", "redacted_thinking"]

    # 当前块状态
    _current_block_index: int = field(default=-1, repr=False)
    _accumulator: Any = field(default=None, repr=False)

    @classmethod
    def create_text_handler(cls) -> ContentBlockHandler:
        """创建文本块处理器"""
        return cls(
            stream_part_type="text_delta",
            block_type="text",
        )

    @classmethod
    def create_thinking_handler(cls) -> ContentBlockHandler:
        """创建推理块处理器"""
        return cls(
            stream_part_type="thinking_delta",
            block_type="thinking",
        )

    @classmethod
    def create_tool_handler(cls) -> ContentBlockHandler:
        """创建工具调用块处理器"""
        return cls(
            stream_part_type="tool_call_delta",
            block_type="tool_use",
        )

    def _create_accumulator(self) -> Any:
        """创建累积器"""
        if self.block_type == "text":
            return []
        elif self.block_type == "thinking":
            return []
        elif self.block_type == "tool_use":
            return {"id": "", "name": "", "arguments": ""}
        return None

    def _add_delta(self, chunk: DeltaPart) -> None:
        """添加 delta 到累积器"""
        if self._accumulator is None:
            return

        if self.block_type in ("text", "thinking"):
            # list[str] accumulator
            delta = chunk.get("delta", "")
            if delta:
                self._accumulator.append(delta)
        elif self.block_type == "tool_use":
            # dict accumulator
            acc = self._accumulator
            chunk_id = chunk.get("id")
            chunk_name = chunk.get("name")
            chunk_args = chunk.get("arguments_delta")
            if chunk_id:
                acc["id"] = chunk_id
            if chunk_name:
                acc["name"] = chunk_name
            if chunk_args:
                acc["arguments"] += chunk_args

    def _build_part(self, idx: int) -> ContentPart:
        """从累积器构建 ContentPart"""
        if self._accumulator is None:
            raise ValueError("No accumulator to build part from")

        if self.block_type == "text":
            return TextPart(type="text", text="".join(self._accumulator))
        elif self.block_type == "thinking":
            from hawi.models.message import ReasoningPart
            return ReasoningPart(
                type="reasoning",
                reasoning="".join(self._accumulator),
                signature=None,
                redacted_content=None,
            )
        elif self.block_type == "tool_use":
            acc = self._accumulator
            return ToolCallPart(
                type="tool_call",
                id=acc["id"],
                name=acc["name"],
                arguments=self._parse_tool_arguments(acc["arguments"]),
            )
        raise ValueError(f"Unknown block type: {self.block_type}")

    def _is_empty(self) -> bool:
        """检查累积器是否为空"""
        if self._accumulator is None:
            return True
        if self.block_type in ("text", "thinking"):
            return not "".join(self._accumulator).strip()
        elif self.block_type == "tool_use":
            return not self._accumulator.get("name")
        return False

    @staticmethod
    def _parse_tool_arguments(args_str: str) -> dict[str, Any]:
        """解析工具参数 JSON"""
        import json
        try:
            return json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            return {}

    async def handle(
        self,
        chunk: DeltaPart,
        request_id: str,
        event_bus: EventBus | None,
        is_streaming: bool = True,
    ) -> ContentPart | None:
        """处理单个 chunk，返回完成的 Part（块结束时）或 None。

        事件通过 event_bus 异步发布，不阻塞处理流程。

        Args:
            chunk: DeltaPart（必须是匹配的 type）
            request_id: 请求 ID
            event_bus: 事件总线（可为 None）
            is_streaming: 是否来自流式接口（默认 True）

        Returns:
            块完成时返回 ContentPart，否则返回 None
        """
        idx = chunk.get("index", 0)

        # is_start: 初始化新块，发送 StartEvent
        if chunk.get("is_start"):
            self._current_block_index = idx
            self._accumulator = self._create_accumulator()

            if self.block_type == "tool_use":
                event = ModelToolCallBlockStartEvent.create(
                    request_id=request_id,
                    block_index=idx,
                    tool_call_id=chunk.get("id") or "",
                    tool_name=chunk.get("name") or "",
                )
            else:
                event = ModelContentBlockStartEvent.create(
                    request_id=request_id,
                    block_index=idx,
                    block_type=self.block_type,
                )

            if event_bus is not None:
                await event_bus.publish_async(event)

        # 发送 DeltaEvent
        if self.block_type == "tool_use":
            event = ModelToolCallBlockDeltaEvent.create(
                request_id=request_id,
                block_index=idx,
                tool_call_id=chunk.get("id") or "",
                arguments_delta=chunk.get("arguments_delta", ""),
                is_streaming=is_streaming,
            )
        else:
            event = ModelContentBlockDeltaEvent.create(
                request_id=request_id,
                part=chunk,
                is_streaming=is_streaming,
            )

        if event_bus is not None:
            await event_bus.publish_async(event)

        # 累积内容
        self._add_delta(chunk)

        # is_end: 构建 Part，发送 StopEvent，返回 Part
        if chunk.get("is_end") and self._accumulator is not None:
            part = self._build_part(idx)

            if self.block_type == "tool_use":
                acc = self._accumulator
                # ModelToolCallBlockStopEvent 包含完整参数
                event = ModelToolCallBlockStopEvent.create(
                    request_id=request_id,
                    block_index=idx,
                    tool_call_id=acc.get("id") or "",
                    tool_name=acc.get("name") or "",
                    arguments=self._parse_tool_arguments(acc.get("arguments", "")),
                )
            else:
                event = ModelContentBlockStopEvent.create(
                    request_id=request_id,
                    block_index=idx,
                    content=[part],
                )

            if event_bus is not None:
                await event_bus.publish_async(event)

            # 在重置前检查是否为空
            is_empty = self._is_empty()

            # 重置状态
            self._current_block_index = -1
            self._accumulator = None

            if not is_empty:
                return part

        return None


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
class _ExecutionState:
    """Internal execution state during agent run."""

    iteration: int = 0
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    error: HawiError | str | None = None
    should_stop: bool = False


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
        model: Model,
        *,
        plugins: list[HawiPlugin] | None = None,
        plugin_factories: list[Callable[[], HawiPlugin]] | None = None,
        system_prompt: str | list[ContentPart] | None = None,
        max_iterations: int | None = None,
        model_error_policy: Optional[ModelErrorPolicyConfig] = None,
        event_bus: EventBus | None = None,
        event_dump_file: str | None = None,
        streaming: bool = True,
    ):
        """Initialize HawiAgent.

        Args:
            model: Default model for agent execution
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

        Note:
            Both `plugins` and `plugin_factories` can be used together.
            Factories are invoked first during initialization.
        """
        self._default_model = model
        self._max_iterations = max_iterations
        self._streaming = streaming
        self._event_bus = event_bus or EventBus()

        # 检查模型是否适合异步使用
        if getattr(model, '_async_only', False) is False:
            import warnings
            warnings.warn(
                "The model was obtained with async_only=False (sync-only). "
                "HawiAgent uses async calls internally, so the model will work but cannot be reused by other agents. "
                "Consider using model_registry.obtain_model(..., async_only=True) for better resource utilization.",
                UserWarning,
                stacklevel=2
            )

        # Initialize event dump manager
        self._dump_manager = DumpManager(event_dump_file) if event_dump_file else None


        if model_error_policy is None:
            self._model_error_policy = self._default_model_error_policy()
        else:
            self._model_error_policy = model_error_policy

        # Store factory functions for clone/fork operations
        self._plugin_factories: list[Callable[[], HawiPlugin]] = plugin_factories or []
        # Store original plugins param for clone (these will have clone() called on them)
        self._original_plugins: list[HawiPlugin] = plugins or []

        # Initialize plugins: factories are called to create instances, plus any directly passed plugins
        # Use empty list if None to avoid mutable default argument issue
        self._plugins: list[HawiPlugin] = [f() for f in self._plugin_factories] + self._original_plugins
        self._hooks: dict[str, Any] = {}

        # Convert system_prompt to list[ContentPart] if needed
        system_prompt_parts: list[ContentPart] | None = None
        if isinstance(system_prompt, str):
            system_prompt_parts = [{"type": "text", "text": system_prompt}]
        else:
            system_prompt_parts = system_prompt

        self._system_prompt = system_prompt_parts

        # Collect tools from plugins and store them in the agent
        self._tools = self._collect_tools_from_plugins()

        # Initialize context with tool definitions
        self._context = AgentContext(
            system_prompt=system_prompt_parts,
            tool_definitions=self._convert_tools_to_definitions() if self._tools else None,
        )

        # Set up tool call context for runtime injection
        self._context.tool_call_context = ToolCallContext(agent=self)

    def _collect_tools_from_plugins(self) -> list[AgentTool]:
        """Collect tools from all plugins.

        Returns:
            List of unique tools (later plugins override earlier ones)
        """
        tools_by_name: dict[str, AgentTool] = {}
        for plugin in self._plugins:
            for tool in plugin.tools:
                if tool.name in tools_by_name:
                    import warnings
                    warnings.warn(
                        f"Tool '{tool.name}' is being overwritten by {plugin.__class__.__name__}",
                        UserWarning,
                        stacklevel=3,
                    )
                tools_by_name[tool.name] = tool

            # Collect hooks from plugin
            plugin_hooks = plugin.hooks
            for hook_type, hook_fn in plugin_hooks.items():
                self._hooks[hook_type] = hook_fn

        return list(tools_by_name.values())

    def _convert_tools_to_definitions(self) -> list[ToolDefinition]:
        """Convert AgentTool instances to ToolDefinition format.

        Returns:
            List of ToolDefinition for model consumption
        """
        return [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "schema": tool.parameters_schema,
            }
            for tool in self._tools
        ]

    def get_tool(self, name: str) -> AgentTool | None:
        """Get a tool by name.

        Args:
            name: Tool name

        Returns:
            AgentTool if found, None otherwise
        """
        for tool in self._tools:
            if tool.name == name:
                return tool
        return None

    @classmethod
    def _default_model_error_policy(cls) -> ModelErrorPolicyConfig:
        return defaultdict(ModelErrorStopPolicy, {
            'network': ModelErrorRetryPolicy(retry_count=10),
            'throttle': ModelErrorRetryPolicy(retry_count=3),
        })

    @property
    def context(self) -> AgentContext:
        """Get the agent's context (read-only access).

        Returns:
            The current AgentContext
        """
        return self._context

    def set_context(self, context: AgentContext) -> None:
        """Replace the agent's context.

        Args:
            context: New context to use
        """
        self._context = context

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
        # Copy configuration
        new_agent = HawiAgent(
            model=self._default_model,
            plugins=[plugin.clone() for plugin in self._original_plugins],
            plugin_factories=self._plugin_factories.copy(),  # Copy factories list
            system_prompt=self._system_prompt,
            max_iterations=self._max_iterations,
            event_bus=self._event_bus,
            model_error_policy=self._model_error_policy,
        )

        # Copy context (deep copy)
        new_agent.set_context(self._context.copy())

        return new_agent

    def fork(self) -> HawiAgent:
        """Alias for clone().

        Returns:
            New HawiAgent instance with copied state
        """
        return self.clone()

    def _invoke_hook(self, hook_type: str, *args, **kwargs) -> None:
        """Invoke a hook if registered."""
        hook = self._hooks.get(hook_type)
        if hook:
            try:
                hook(*args, **kwargs)
            except Exception as e:
                import warnings

                warnings.warn(
                    f"Hook '{hook_type}' failed: {e}",
                    RuntimeWarning,
                    stacklevel=3,
                )

    def run(
        self,
        message: str | list[ContentPart] | None = None,
        *,
        model: Model | None = None,
        model_error_policy: Optional[ModelErrorPolicyConfig] = None,
        event_bus: EventBus | None = None,
        streaming: bool | None = None,
    ) -> AgentRunResult:
        """Execute agent with a message (synchronous).

        Args:
            message: User message (str, content parts, or None to use existing context)
            model: Override model for this run
            model_error_policy: Override failure policy for this run
            event_bus: Optional event bus for publishing events (defaults to self.event_bus)
            streaming: Whether to use streaming mode (defaults to self._streaming)

        Returns:
            AgentRunResult containing the execution result
        """
        # Run async execution in sync context
        return asyncio.run(self.arun(
            message=message,
            model=model,
            model_error_policy=model_error_policy,
            event_bus=event_bus,
            streaming=streaming,
        ))

    async def arun(
        self,
        message: str | list[ContentPart] | None = None,
        *,
        model: Model | None = None,
        model_error_policy: Optional[ModelErrorPolicyConfig] = None,
        event_bus: EventBus | None = None,
        streaming: bool | None = None,
    ) -> AgentRunResult:
        """Execute agent asynchronously.

        Args:
            message: User message (str, content parts, or None to use existing context)
            model: Override model for this run
            model_error_policy: Override failure policy for this run
            event_bus: Optional event bus for publishing events (defaults to self.event_bus)
            streaming: Whether to use streaming mode (defaults to self._streaming)

        Returns:
            AgentRunResult containing the execution result
        """
        # Normalize parameters
        m = model or self._default_model
        policy = model_error_policy or self._model_error_policy
        effective_event_bus = event_bus or self._event_bus
        effective_streaming = streaming if streaming is not None else self._streaming

        return await self._execute(message, m, policy, effective_event_bus, effective_streaming)

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

        m = model
        policy = policy
        state = _ExecutionState()
        run_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        # Record initial message count to track delta for this invocation
        initial_message_count = len(self._context.messages)

        # Track cumulative usage across all model calls (for multi-turn conversations)
        cumulative_usage: TokenUsage | None = None

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
                    message_preview=str(message)[:100],
                ),
                event_bus,
            )

        # Agent run start
        await self._emit_event(
            AgentRunStartEvent.create(run_id=run_id, message_preview=str(message)[:100] if message else None),
            event_bus,
        )

        # before_conversation hook
        self._invoke_hook("before_conversation", self)

        try:
            while not state.should_stop:
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
                self._invoke_hook("before_model_call", self, self._context, m)

                # Model stream start
                request_id = f"{run_id}-{state.iteration}"
                await self._emit_event(
                    ModelStreamStartEvent.create(request_id=request_id),
                    event_bus,
                )

                # Call model with streaming
                content_parts: list[ContentPart] = []
                tool_calls: list[ToolCallPart] = []
                stop_reason = "end_turn"
                usage: TokenUsage | None = None

                # Content block handlers for processing different chunk types
                text_handler = ContentBlockHandler.create_text_handler()
                thinking_handler = ContentBlockHandler.create_thinking_handler()
                tool_handler = ContentBlockHandler.create_tool_handler()

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
                        elif chunk_type == "thinking_delta":
                            handler = thinking_handler
                        elif chunk_type == "tool_call_delta":
                            handler = tool_handler
                        elif chunk_type == "finish":
                            stop_reason = chunk.get("stop_reason") or "end_turn"
                            usage_data = chunk.get("usage")
                            if usage_data:
                                # usage_data can be TokenUsage (TypedDict) or dict[str, int]
                                usage = cast(TokenUsage, usage_data)
                                # Accumulate usage for multi-turn conversations
                                if cumulative_usage is None:
                                    cumulative_usage = usage
                                else:
                                    # Use a temp variable to help type checker
                                    current = cumulative_usage

                                    def add_optional_tokens(a: int | None, b: int | None) -> int | None:
                                        """Add two optional token counts, returning None if both are None."""
                                        if a is None and b is None:
                                            return None
                                        return (a or 0) + (b or 0)
                                    cumulative_usage = {
                                        "input_tokens": current["input_tokens"] + usage["input_tokens"],
                                        "output_tokens": current["output_tokens"] + usage["output_tokens"],
                                        "cache_write_tokens": add_optional_tokens(
                                            current.get("cache_write_tokens"),
                                            usage.get("cache_write_tokens"),
                                        ),
                                        "cache_read_tokens": add_optional_tokens(
                                            current.get("cache_read_tokens"),
                                            usage.get("cache_read_tokens"),
                                        ),
                                    }
                            continue
                        else:
                            continue  # Unknown chunk type

                        # Handle the chunk with appropriate handler
                        part = await handler.handle(chunk, request_id, event_bus, is_streaming=streaming)
                        if part is not None:
                            content_parts.append(part)
                            if part["type"] == "tool_call":
                                tool_calls.append(part)
                                # For tool calls, also send AgentToolCallEvent
                                await self._emit_event(
                                    AgentToolCallEvent.create(
                                        run_id=run_id,
                                        tool_name=part["name"],
                                        arguments=part["arguments"],
                                        tool_call_id=part["id"],
                                    ),
                                    event_bus,
                                )
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
                        usage=usage,
                    ),
                    event_bus,
                )

                # after_model_call hook
                self._invoke_hook("after_model_call", self, self._context, None)

                # Build content parts for the assistant message
                # Content parts include text/reasoning, but NOT tool_calls (they go in separate field)
                response_content: list[ContentPart] = content_parts

                # Add assistant message to context
                # tool_calls are now included in content as ToolCallPart items
                self._context.add_assistant_message(content=response_content)

                # Emit event for assistant message added
                await self._emit_event(
                    AgentMessageAddedEvent.create(
                        run_id=run_id,
                        role="assistant",
                        content=response_content,
                        message_preview=str(response_content)[:100] if response_content else "",
                    ),
                    event_bus,
                )

                # Check if tool calls need to be executed
                if not tool_calls:
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
                for tc in tool_calls:
                    record = await self._execute_tool(tc, state)
                    state.tool_calls.append(record)
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

                # Continue loop for next iteration

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
            # after_conversation hook
            self._invoke_hook("after_conversation", self)

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
            elif block_type == "thinking":
                parts.append({
                    "type": "thinking_delta",
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
                        "type": "thinking_delta",
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
                "usage": data.get("usage"),
            })

        return parts

    async def _execute_tool(
        self,
        tool_call: ToolCallPart,
        state: _ExecutionState,
    ) -> ToolCallRecord:
        """Execute a single tool call."""
        tool_name = tool_call["name"]
        arguments = tool_call["arguments"]
        tool_call_id = tool_call["id"]

        start_time = time.time()

        # before_tool_calling hook
        self._invoke_hook("before_tool_calling", self, tool_name, arguments)

        # Find tool
        tool = self.get_tool(tool_name)
        if tool is None:
            err = ToolNotFoundError(f"Tool '{tool_name}' not found")
            result = ToolResult(success=False, error=f"{err.__class__.__name__}: {err.message}")
        elif getattr(tool, "audit", False):
            # Audit mode: cache the tool call and return pending status
            self._context._add_pending_tool_call(tool_call_id, tool_name, arguments)
            result = ToolResult(
                success=True,
                output=f"[AUDIT PENDING] Tool '{tool_name}' has been submitted for review. "
                       f"Use review_pending_tools() to check status and approve/reject."
            )
        else:
            # Prepare arguments with context injection if needed
            tool_arguments = dict(arguments)
            context_param = getattr(tool, "context", None)
            if context_param and self._context.tool_call_context:
                # Inject the tool call context (currently just the agent reference)
                tool_arguments[context_param] = self._context.tool_call_context.agent

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
                                    run_id=getattr(self, '_current_run_id', ''),
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
                                run_id=getattr(self, '_current_run_id', ''),
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
        self._invoke_hook("after_tool_calling", self, tool_name, arguments, result)

        # Add tool result to context (unless audit pending - will be added after approval)
        if not (tool and getattr(tool, "audit", False)):
            # Build result content: include both output and error
            output_str = result.output if isinstance(result.output, str) else str(result.output) if result.output else ""
            if not result.success and result.error:
                # On failure, include error information
                result_content = f"Error: {result.error}"
                if output_str:
                    result_content = f"Output before error:\n{output_str}\n\n{result_content}"
            else:
                result_content = output_str
            self._context.add_tool_result(
                tool_call_id=tool_call_id,
                content=result_content,
                is_error=not result.success,
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
            tool = self.get_tool(pending.tool_name)
            if tool is None:
                result = ToolResult(
                    success=False,
                    error=f"Tool '{pending.tool_name}' not found during approval execution"
                )
            else:
                # Prepare arguments with context injection if needed
                tool_arguments = dict(pending.arguments)
                context_param = getattr(tool, "context", None)
                if context_param and self._context.tool_call_context:
                    tool_arguments[context_param] = self._context.tool_call_context.agent

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
            self._context.add_tool_result(
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
