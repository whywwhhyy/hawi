"""HawiAgent - Core agent implementation with tool execution and plugin support.

This module implements the HawiAgent class that orchestrates LLM interaction,
tool execution, and plugin hooks for agent workflows.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections import defaultdict
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any, Optional, Coroutine, Literal, Mapping, Callable


from hawi.model import Model
from hawi.model.message import (
    ContentPart,
    StreamPart,
    TextPart,
    TokenUsage,
    ToolCallPart,
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
from hawi.events import (
    Event,
    EventBus,
    AgentErrorEvent,
    AgentMessageAddedEvent,
    AgentRunStartEvent,
    AgentRunStopEvent,
    AgentToolCallEvent,
    AgentToolResultEvent,
    ModelContentBlockDeltaEvent,
    ModelContentBlockStartEvent,
    ModelContentBlockStopEvent,
    ModelErrorEvent,
    ModelToolUseBlockDeltaEvent,
    ModelToolUseBlockStartEvent,
    ModelToolUseBlockStopEvent,
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

    def _add_delta(self, chunk: StreamPart) -> None:
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
            from hawi.model.message import ReasoningPart
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
        chunk: StreamPart,
        request_id: str,
        event_bus: EventBus | None,
    ) -> ContentPart | None:
        """处理单个 chunk，返回完成的 Part（块结束时）或 None。

        事件通过 event_bus 异步发布，不阻塞处理流程。

        Args:
            chunk: StreamPart（必须是匹配的 type）
            request_id: 请求 ID
            event_bus: 事件总线（可为 None）

        Returns:
            块完成时返回 ContentPart，否则返回 None
        """
        idx = chunk.get("index", 0)

        # is_start: 初始化新块，发送 StartEvent
        if chunk.get("is_start"):
            self._current_block_index = idx
            self._accumulator = self._create_accumulator()

            if self.block_type == "tool_use":
                event = ModelToolUseBlockStartEvent.create(
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
                await event_bus.publish(event)

        # 发送 DeltaEvent
        if self.block_type == "tool_use":
            event = ModelToolUseBlockDeltaEvent.create(
                request_id=request_id,
                block_index=idx,
                tool_call_id=chunk.get("id") or "",
                arguments_delta=chunk.get("arguments_delta", ""),
            )
        else:
            event = ModelContentBlockDeltaEvent.create(
                request_id=request_id,
                part=chunk,
            )

        if event_bus is not None:
            await event_bus.publish(event)

        # 累积内容
        self._add_delta(chunk)

        # is_end: 构建 Part，发送 StopEvent，返回 Part
        if chunk.get("is_end") and self._accumulator is not None:
            part = self._build_part(idx)

            if self.block_type == "tool_use":
                acc = self._accumulator
                event = ModelToolUseBlockStopEvent.create(
                    request_id=request_id,
                    block_index=idx,
                    tool_call_id=acc.get("id") or "",
                    arguments=acc.get("arguments", ""),
                )
            else:
                event = ModelContentBlockStopEvent.create(
                    request_id=request_id,
                    block_index=idx,
                    content=[part],
                )

            if event_bus is not None:
                await event_bus.publish(event)

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
        for event in agent.run("Hello", stream=True):
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
        system_prompt: str | list[ContentPart] | None = None,
        max_iterations: int | None = None,
        model_error_policy: Optional[ModelErrorPolicyConfig] = None,
        event_bus: EventBus | None = None,
        event_dump_file: str | None = None,
    ):
        """Initialize HawiAgent.

        Args:
            model: Default model for agent execution
            plugins: List of plugins providing tools and hooks (default: empty list)
            system_prompt: Default system prompt (str or list[ContentPart])
            max_iterations: Maximum tool execution iterations (None for unlimited)
            model_error_policy: Error handling policy mapping error_type to config
            event_bus: Event bus for event publishing. If None, creates a default EventBus
            event_dump_file: Path to dump all events for debugging (default: None)
        """
        self._default_model = model
        self._max_iterations = max_iterations
        self._event_bus = event_bus or EventBus()

        # Initialize event dump manager
        self._dump_manager = DumpManager(event_dump_file) if event_dump_file else None

        if model_error_policy is None:
            self._model_model_error_policy_config = self._default_model_error_policy()
        else:
            self._model_model_error_policy_config = model_error_policy

        # Initialize plugins and collect tools/hooks
        # Use empty list if None to avoid mutable default argument issue
        self._plugins: list[HawiPlugin] = plugins or []
        self._hooks: dict[str, Any] = {}

        # Convert system_prompt to list[ContentPart] if needed
        system_prompt_parts: list[ContentPart] | None = None
        if isinstance(system_prompt, str):
            system_prompt_parts = [{"type": "text", "text": system_prompt}]
        else:
            system_prompt_parts = system_prompt

        self._system_prompt = system_prompt_parts

        # Initialize context with tools from plugins
        self._context = AgentContext(
            system_prompt=system_prompt_parts,
            tools=self._collect_tools_from_plugins(),
            cache_tool_definitions=True,
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
        - Same plugins (shared reference)
        - Same default model
        - Same configuration (max_iterations, etc.)

        The clone is independent - modifications to the clone's context
        do not affect the original agent.

        Returns:
            New HawiAgent instance with copied state
        """
        # Copy configuration
        new_agent = HawiAgent(
            model=self._default_model,
            plugins=self._plugins,  # Shared - plugins are typically stateless
            system_prompt=self._system_prompt,
            max_iterations=self._max_iterations,
            event_bus=self._event_bus,
            model_error_policy=self._model_model_error_policy_config,
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
    ) -> AgentRunResult:
        """Execute agent with a message (synchronous).

        Args:
            message: User message (str, content parts, or None to use existing context)
            model: Override model for this run
            model_error_policy: Override failure policy for this run
            event_bus: Optional event bus for publishing events (defaults to self.event_bus)

        Returns:
            AgentRunResult containing the execution result
        """
        # Normalize model_error_policy to empty dict if None
        policy = model_error_policy or {}

        # Use provided event_bus or default
        effective_event_bus = event_bus or self._event_bus

        # Run async execution in sync context
        return asyncio.run(self._execute(message, model, policy, effective_event_bus))

    async def arun(
        self,
        message: str | list[ContentPart] | None = None,
        *,
        model: Model | None = None,
        model_error_policy: Optional[ModelErrorPolicyConfig] = None,
        event_bus: EventBus | None = None,
    ) -> AgentRunResult:
        """Execute agent asynchronously.

        Args:
            message: User message (str, content parts, or None to use existing context)
            model: Override model for this run
            model_error_policy: Override failure policy for this run
            event_bus: Optional event bus for publishing events (defaults to self.event_bus)

        Returns:
            AgentRunResult containing the execution result
        """
        # Normalize model_error_policy to empty dict if None
        policy = model_error_policy or {}

        # Use provided event_bus or default
        effective_event_bus = event_bus or self._event_bus

        return await self._execute(message, model, policy, effective_event_bus)

    @property
    def event_bus(self) -> EventBus:
        """Get the agent's EventBus for event subscriptions."""
        return self._event_bus

    def subscribe(
        self,
        callback: Callable[[Event], Coroutine[Any, Any, None]],
        event_types: list[str] | None = None,
        blocking: bool = False,
        maxsize: int = 100,
    ) -> None:
        """Subscribe to agent events (delegates to EventBus).

        Args:
            callback: Async callback function to handle events
            event_types: List of event types to subscribe to, None for all
            blocking: If True, agent waits for this handler to complete
            maxsize: Queue size for non-blocking handlers
        """
        self._event_bus.subscribe(callback, event_types, blocking, maxsize)

    async def unsubscribe(
        self,
        callback: Callable[[Event], Coroutine[Any, Any, None]],
        wait: bool = False,
        timeout: float | None = None,
    ) -> bool:
        """Unsubscribe from agent events (delegates to EventBus).

        Args:
            callback: Callback function to remove
            wait: Whether to wait for queued events to be processed
            timeout: Timeout for waiting (seconds)

        Returns:
            True if successfully unsubscribed
        """
        return await self._event_bus.unsubscribe(callback, wait, timeout)

    async def _emit_event(
        self,
        event: Event,
        event_bus: EventBus | None,
    ) -> Event:
        """Emit event to both generator and event bus."""
        if event_bus is not None:
            await event_bus.publish(event)

        # Dump event to file if configured
        if self._dump_manager is not None:
            self._dump_manager.dump(event)

        return event

    async def _execute(
        self,
        message: str | list[ContentPart] | None,
        model: Model | None,
        model_error_policy_config: Optional[ModelErrorPolicyConfig],
        event_bus: EventBus | None = None,
    ) -> AgentRunResult:
        """Execute agent and return result (pure EventBus-driven)."""

        m = model or self._default_model
        policy = model_error_policy_config or self._model_model_error_policy_config
        state = _ExecutionState()
        run_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        # Track cumulative usage across all model calls (for multi-turn conversations)
        cumulative_usage: TokenUsage | None = None

        # Track all events for building result
        events: list[Event] = []

        # Add user message if provided
        if message is not None:
            self._context.add_user_message(message)
            await self._emit_event(
                AgentMessageAddedEvent.create(
                    run_id=run_id,
                    role="user",
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

                # Track current handlers by block index
                handlers: dict[int, ContentBlockHandler] = {}

                # Use try/finally to ensure proper cleanup of the async generator
                model_stream_gen = self._call_model_with_retry_streaming(
                    m, policy, state, request_id, event_bus
                )
                try:
                    async for chunk in model_stream_gen:
                        if state.error:
                            # state.error 现在应该是异常对象
                            if isinstance(state.error, AgentError):
                                await self._emit_event(
                                    AgentErrorEvent.create(run_id=run_id, error=state.error),
                                    event_bus,
                                )
                            break

                        chunk_type = chunk["type"]
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
                            usage_dict = chunk.get("usage")
                            if usage_dict:
                                usage = TokenUsage(
                                    input_tokens=usage_dict.get("input_tokens", 0),
                                    output_tokens=usage_dict.get("output_tokens", 0),
                                    cache_write_tokens=usage_dict.get("cache_write_tokens"),
                                    cache_read_tokens=usage_dict.get("cache_read_tokens"),
                                )
                                # Accumulate usage for multi-turn conversations
                                if cumulative_usage is None:
                                    cumulative_usage = usage
                                else:
                                    cumulative_usage = TokenUsage(
                                        input_tokens=cumulative_usage.input_tokens + usage.input_tokens,
                                        output_tokens=cumulative_usage.output_tokens + usage.output_tokens,
                                        cache_write_tokens=self._add_optional_tokens(
                                            cumulative_usage.cache_write_tokens,
                                            usage.cache_write_tokens,
                                        ),
                                        cache_read_tokens=self._add_optional_tokens(
                                            cumulative_usage.cache_read_tokens,
                                            usage.cache_read_tokens,
                                        ),
                                    )
                            continue
                        else:
                            continue  # Unknown chunk type

                        # Handle the chunk with appropriate handler
                        part = await handler.handle(chunk, request_id, event_bus)
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
                            tool_name=record.tool_name,
                            tool_call_id=record.tool_call_id,
                            success=record.result.success,
                            result_preview=str(record.result.output)[:100],
                            duration_ms=record.duration_ms,
                            arguments=record.arguments,
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
            raise
        except Exception as e:
            # 包装为 AgentError，保留原始异常
            err = AgentError("tool_execution", f"{type(e).__name__}: {e}")
            state.error = err
            await self._emit_event(
                AgentErrorEvent.create(run_id=run_id, error=err),
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
        messages = self._context.messages
        response = None
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                response = msg
                break

        return AgentRunResult(
            stop_reason=stop_reason,
            messages=messages,
            response=response,
            usage=cumulative_usage,
            tool_calls=state.tool_calls,
            error=str(state.error) if state.error else None,
        )

    async def _call_model_with_retry_streaming(
        self,
        model: Model,
        policy: ModelErrorPolicyConfig,
        state: _ExecutionState,
        request_id: str,
        event_bus: EventBus | None,
    ) -> AsyncGenerator[StreamPart, None]:
        """Call model with streaming and retry logic.

        Yields StreamPart for each chunk of content from the model.
        Accumulates content to build complete response for tool call handling.
        """
        last_error = None
        max_retries = 0

        # Calculate max retries from policy
        for p in policy.values():
            if p.action == "retry" and isinstance(p, ModelErrorRetryPolicy) and p.retry_count > max_retries:
                max_retries = p.retry_count

        attempt = 0
        stream_gen = None
        for attempt in range(max_retries + 1):
            try:
                request = self._context.prepare_request()

                # Use astream() for streaming output
                # Store generator reference for proper cleanup
                async with model.astream(
                    messages=request.messages,
                    system=[part for part in (request.system or ()) if part['type'] == 'text'],
                    tools=request.tools,
                ) as stream_gen:
                    async for chunk in stream_gen:
                        yield chunk

                return  # Success, exit retry loop

            except ModelError as e:
                stream_gen = None
                last_error = e

                # 直接使用 ModelError 的 model_error_type
                policy_for_error = policy[e.error_type]

                if policy_for_error.action == "stop":
                    # Emit error event and gracefully stop
                    state.error = e
                    if event_bus:
                        await event_bus.publish(ModelErrorEvent.create(error=e))
                    return

                if attempt < max_retries:
                    await asyncio.sleep(min(2 ** attempt, 60))

        if last_error:
            # All retries exhausted for retryable errors
            err = ModelError("network", f"Model call failed after {attempt + 1} attempts: {last_error}")
            state.error = err
            if event_bus:
                await event_bus.publish(ModelErrorEvent.create(error=err))
            return

    async def _call_model_with_retry(
        self,
        model: Model,
        policy: ModelErrorPolicyConfig,
        state: _ExecutionState,
    ) -> Any:
        """Call model with retry logic based on failure policy."""
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
                return await model.ainvoke(
                    messages=request.messages,
                    system=[part for part in (request.system or ()) if part['type'] == 'text'],
                    tools=request.tools,
                )
            except ModelError as e:
                last_error = e

                # 直接使用 ModelError 的 error_type (policy is defaultdict)
                policy_for_error = policy[e.error_type]

                if policy_for_error.action == "stop":
                    # Set error in state and return gracefully
                    state.error = e
                    return None

                if attempt < max_retries:
                    # Exponential backoff
                    wait_time = min(2**attempt, 60)
                    await asyncio.sleep(wait_time)
                    continue

        if last_error:
            # All retries exhausted for retryable errors
            err = ModelError("network", f"Model call failed after {attempt + 1} attempts: {last_error}")
            state.error = err
            return None

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
        tool = self._context.get_tool(tool_name)
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
                result = await tool.ainvoke(tool_arguments)
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
            tool = self._context.get_tool(pending.tool_name)
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
                        tool_name=record.tool_name,
                        tool_call_id=record.tool_call_id,
                        success=record.result.success,
                        result_preview=str(record.result.output)[:100],
                        duration_ms=record.duration_ms,
                        arguments=record.arguments,
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

    def _build_result_from_events(self, events: list[Event]) -> AgentRunResult:
        """Build AgentRunResult from collected events."""
        tool_calls: list[ToolCallRecord] = []
        stop_reason = "unknown"
        error = None
        total_usage: TokenUsage | None = None

        for event in events:
            if isinstance(event, AgentToolResultEvent):
                tool_calls.append(
                    ToolCallRecord(
                        tool_name=event.tool_name,
                        arguments=event.arguments or {},
                        result=ToolResult(
                            success=event.success,
                            output=event.result_preview,
                        ),
                        duration_ms=event.duration_ms,
                        tool_call_id=event.tool_call_id,
                    )
                )
            elif isinstance(event, AgentRunStopEvent):
                stop_reason = event.stop_reason
            elif isinstance(event, AgentErrorEvent):
                error = str(event.error) if event.error else None
                stop_reason = "error"
            elif isinstance(event, ModelStreamStopEvent):
                # Accumulate usage from each model call (for multi-turn conversations)
                usage = event.usage
                if usage:
                    if total_usage is None:
                        total_usage = usage
                    else:
                        # Accumulate token counts
                        total_usage = TokenUsage(
                            input_tokens=total_usage.input_tokens + usage.input_tokens,
                            output_tokens=total_usage.output_tokens + usage.output_tokens,
                            cache_write_tokens=self._add_optional_tokens(
                                total_usage.cache_write_tokens,
                                usage.cache_write_tokens,
                            ),
                            cache_read_tokens=self._add_optional_tokens(
                                total_usage.cache_read_tokens,
                                usage.cache_read_tokens,
                            ),
                        )

        # Get final response (last assistant message)
        response = None
        for msg in reversed(self._context.messages):
            if msg["role"] == "assistant":
                response = msg
                break

        return AgentRunResult(
            stop_reason=stop_reason,
            messages=self._context.messages.copy(),
            response=response,
            usage=total_usage,
            tool_calls=tool_calls,
            error=error,
        )

    @staticmethod
    def _add_optional_tokens(a: int | None, b: int | None) -> int | None:
        """Add two optional token counts, returning None if both are None."""
        if a is None and b is None:
            return None
        return (a or 0) + (b or 0)
