"""HawiAgent - Core agent implementation with tool execution and plugin support.

This module implements the HawiAgent class that orchestrates LLM interaction,
tool execution, and plugin hooks for agent workflows.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any, Coroutine, Literal, TypedDict, overload

from hawi.agent.model import Model, ModelFailurePolicy
from hawi.agent.message import (
    ContentPart,
    StreamPart,
    TextPart,
    TokenUsage,
    ToolCallPart,
)
from hawi.plugin import HawiPlugin
from hawi.tool.types import AgentTool, ToolResult

from .context import AgentContext, ToolCallContext
from .events import (
    Event,
    EventBus,
    agent_error_event,
    agent_message_added_event,
    agent_run_start_event,
    agent_run_stop_event,
    agent_tool_call_event,
    agent_tool_result_event,
    model_content_block_delta_event,
    model_content_block_start_event,
    model_content_block_stop_event,
    model_stream_start_event,
    model_stream_stop_event,
)
from .result import AgentRunResult, ToolCallRecord


class ModelFailurePolicyConfig(TypedDict, total=False):
    """Configuration for model failure policy.

    Example:
        {
            "network": {"action": "retry", "retry_count": 3},
            "throttle": {"action": "retry", "retry_count": 5},
            "denied": {"action": "stop"},
        }
    """
    action: Literal["retry", "stop"]
    retry_count: int


@dataclass
class _ExecutionState:
    """Internal execution state during agent run."""

    iteration: int = 0
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    error: str | None = None
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
        enable_streaming: bool = True,
        model_failure_policy: dict[str, ModelFailurePolicyConfig] | None = None,
        event_bus: EventBus | None = None,
    ):
        """Initialize HawiAgent.

        Args:
            model: Default model for agent execution
            plugins: List of plugins providing tools and hooks (default: empty list)
            system_prompt: Default system prompt (str or list[ContentPart])
            max_iterations: Maximum tool execution iterations (None for unlimited)
            enable_streaming: Whether streaming is enabled by default
            model_failure_policy: Error handling policy mapping error_type to config
            event_bus: Event bus for multi-agent coordination (not implemented)

        Raises:
            NotImplementedError: If event_bus is provided (not yet supported)
        """
        if event_bus is not None:
            raise NotImplementedError("event_bus is not yet supported")

        self._default_model = model
        self._max_iterations = max_iterations
        self._enable_streaming = enable_streaming
        # Use empty dict if None to avoid mutable default argument issue
        self._model_failure_policy = self._parse_failure_policy(model_failure_policy)

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

    def _parse_failure_policy(
        self, policy: dict[str, ModelFailurePolicyConfig] | None
    ) -> dict[str, ModelFailurePolicy]:
        """Parse failure policy from dict to ModelFailurePolicy objects."""
        if policy is None:
            return {}
        result: dict[str, ModelFailurePolicy] = {}
        for error_type, config in policy.items():
            result[error_type] = ModelFailurePolicy(
                error_type=error_type,
                action=config.get("action", "stop"),
                retry_count=config.get("retry_count", 0),
            )
        return result

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
        - Same configuration (max_iterations, enable_streaming, etc.)

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
            enable_streaming=self._enable_streaming,
            model_failure_policy={
                k: {"action": v.action, "retry_count": v.retry_count}
                for k, v in self._model_failure_policy.items()
            } if self._model_failure_policy else {},
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

    @overload
    def run(
        self,
        message: str | list[ContentPart] | None = None,
        *,
        model: Model | None = None,
        stream: Literal[True],
        model_failure_policy: dict[str, ModelFailurePolicyConfig] | None = None,
        event_bus: EventBus | None = None,
    ) -> Iterator[Event]: ...

    @overload
    def run(
        self,
        message: str | list[ContentPart] | None = None,
        *,
        model: Model | None = None,
        stream: Literal[False] = False,
        model_failure_policy: dict[str, ModelFailurePolicyConfig] | None = None,
        event_bus: EventBus | None = None,
    ) -> AgentRunResult: ...

    def run(
        self,
        message: str | list[ContentPart] | None = None,
        *,
        model: Model | None = None,
        stream: bool | None = None,
        model_failure_policy: dict[str, ModelFailurePolicyConfig] | None = None,
        event_bus: EventBus | None = None,
    ) -> AgentRunResult | Iterator[Event]:
        """Execute agent with a message.

        Args:
            message: User message (str, content parts, or None to use existing context)
            model: Override model for this run
            stream: Whether to stream events (default: self._enable_streaming)
            model_failure_policy: Override failure policy for this run
            event_bus: Optional event bus for publishing events

        Returns:
            AgentRunResult if stream=False, Iterator[Event] if stream=True
        """
        use_stream = self._enable_streaming if stream is None else stream

        # Normalize model_failure_policy to empty dict if None
        policy = model_failure_policy or {}

        if use_stream:
            return self._run_stream(message, model, policy, event_bus)
        else:
            # Collect all events and return result
            events = list(self._run_stream(message, model, policy, event_bus))
            return self._build_result_from_events(events)

    @overload
    def arun(
        self,
        message: str | list[ContentPart] | None = None,
        *,
        model: Model | None = None,
        stream: Literal[True],
        model_failure_policy: dict[str, ModelFailurePolicyConfig] | None = None,
        event_bus: EventBus | None = None,
    ) -> AsyncIterator[Event]: ...

    @overload
    def arun(
        self,
        message: str | list[ContentPart] | None = None,
        *,
        model: Model | None = None,
        stream: Literal[False] = False,
        model_failure_policy: dict[str, ModelFailurePolicyConfig] | None = None,
        event_bus: EventBus | None = None,
    ) -> "Coroutine[Any, Any, AgentRunResult]": ...

    def arun(
        self,
        message: str | list[ContentPart] | None = None,
        *,
        model: Model | None = None,
        stream: bool | None = None,
        model_failure_policy: dict[str, ModelFailurePolicyConfig] | None = None,
        event_bus: EventBus | None = None,
    ) -> AsyncIterator[Event] | Coroutine[Any, Any, AgentRunResult]:
        """Execute agent asynchronously.

        Args:
            message: User message (str, content parts, or None to use existing context)
            model: Override model for this run
            stream: Whether to stream events (default: self._enable_streaming)
            model_failure_policy: Override failure policy for this run
            event_bus: Optional event bus for publishing events

        Returns:
            AgentRunResult if stream=False, AsyncIterator[Event] if stream=True
        """
        use_stream = self._enable_streaming if stream is None else stream

        # Normalize model_failure_policy to empty dict if None
        policy = model_failure_policy or {}

        if use_stream:
            # Directly return async iterator (not a coroutine)
            return self._arun_stream(message, model, policy, event_bus)
        else:
            # Return coroutine that resolves to AgentRunResult
            return self._arun_non_stream(message, model, policy, event_bus)

    async def _arun_non_stream(
        self,
        message: str | list[ContentPart] | None,
        model: Model | None,
        failure_policy: dict[str, ModelFailurePolicyConfig] | None,
        event_bus: EventBus | None = None,
    ) -> AgentRunResult:
        """Non-streaming async execution."""
        events = []
        async for event in self._arun_stream(message, model, failure_policy, event_bus):
            events.append(event)
        return self._build_result_from_events(events)

    def _run_stream(
        self,
        message: str | list[ContentPart] | None,
        model: Model | None,
        failure_policy: dict[str, ModelFailurePolicyConfig] | None,
        event_bus: EventBus | None = None,
    ) -> Iterator[Event]:
        """Synchronous streaming execution."""
        # Run async generator through asyncio
        loop = asyncio.new_event_loop()
        async_gen = None
        try:
            asyncio.set_event_loop(loop)
            async_gen = self._arun_stream(message, model, failure_policy, event_bus)

            while True:
                try:
                    event = loop.run_until_complete(async_gen.__anext__())
                    yield event
                except StopAsyncIteration:
                    break
        finally:
            # Properly close the async generator to prevent 'aclose' warnings
            if async_gen is not None:
                try:
                    loop.run_until_complete(async_gen.aclose())
                except Exception:
                    pass  # Ignore errors during cleanup
            
            # Cancel any remaining pending tasks to prevent warnings
            try:
                pending = asyncio.all_tasks(loop)
                if pending:
                    for task in pending:
                        task.cancel()
                    # Run the event loop briefly to let cancellations complete
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass  # Ignore errors during cleanup
            
            loop.close()
            asyncio.set_event_loop(None)

    async def _emit_event(
        self,
        event: Event,
        event_bus: EventBus | None,
    ) -> Event:
        """Emit event to both generator and event bus."""
        if event_bus is not None:
            await event_bus.publish(event)
        return event

    async def _arun_stream(
        self,
        message: str | list[ContentPart] | None,
        model: Model | None,
        failure_policy: dict[str, ModelFailurePolicyConfig] | None,
        event_bus: EventBus | None = None,
    ) -> AsyncGenerator[Event, None]:
        """Asynchronous streaming execution."""

        m = model or self._default_model
        policy = self._parse_failure_policy(failure_policy) if failure_policy else self._model_failure_policy
        state = _ExecutionState()
        run_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        # Track cumulative usage across all model calls (for multi-turn conversations)
        cumulative_usage: TokenUsage | None = None

        # Add user message if provided
        if message is not None:
            self._context.add_user_message(message)
            await self._emit_event(
                agent_message_added_event(
                    run_id=run_id,
                    role="user",
                    message_preview=str(message)[:100],
                ),
                event_bus,
            )

        # Agent run start
        yield await self._emit_event(
            agent_run_start_event(run_id=run_id, message_preview=str(message)[:100] if message else None),
            event_bus,
        )

        # before_conversation hook
        self._invoke_hook("before_conversation", self)

        try:
            while not state.should_stop:
                # Check max iterations
                if self._max_iterations is not None and state.iteration >= self._max_iterations:
                    state.error = f"Maximum iterations ({self._max_iterations}) reached"
                    yield await self._emit_event(
                        agent_error_event(run_id=run_id, error_type="max_iterations", error_message=state.error),
                        event_bus,
                    )
                    break

                state.iteration += 1

                # before_model_call hook
                self._invoke_hook("before_model_call", self, self._context, m)

                # Model stream start
                request_id = f"{run_id}-{state.iteration}"
                yield await self._emit_event(
                    model_stream_start_event(request_id=request_id),
                    event_bus,
                )

                # Call model with streaming
                content_parts: list[ContentPart] = []
                tool_calls: list[ToolCallPart] = []
                stop_reason = "end_turn"
                usage: TokenUsage | None = None

                # Accumulators for building complete content from deltas
                block_accumulators: dict[int, list[str]] = {}
                tool_call_accumulators: dict[int, dict[str, Any]] = {}

                # Use try/finally to ensure proper cleanup of the async generator
                model_stream_gen = self._call_model_with_retry_streaming(
                    m, policy, state, request_id, event_bus
                )
                try:
                    async for chunk in model_stream_gen:
                        if state.error:
                            yield await self._emit_event(
                                agent_error_event(run_id=run_id, error_type="model_error", error_message=state.error),
                                event_bus,
                            )
                            break

                        # Handle text_delta chunk
                        if chunk["type"] == "text_delta":
                            idx = chunk["index"]

                            if chunk["is_start"]:
                                yield await self._emit_event(
                                    model_content_block_start_event(
                                        request_id=request_id,
                                        block_index=idx,
                                        block_type="text",
                                    ),
                                    event_bus,
                                )

                            if chunk["delta"]:
                                yield await self._emit_event(
                                    model_content_block_delta_event(
                                        request_id=request_id,
                                        block_index=idx,
                                        delta_type="text",
                                        delta=chunk["delta"],
                                    ),
                                    event_bus,
                                )
                                block_accumulators.setdefault(idx, []).append(chunk["delta"])

                            if chunk["is_end"]:
                                full_text = "".join(block_accumulators.get(idx, []))
                                yield await self._emit_event(
                                    model_content_block_stop_event(
                                        request_id=request_id,
                                        block_index=idx,
                                        block_type="text",
                                        full_content=full_text,
                                    ),
                                    event_bus,
                                )
                                if full_text:
                                    content_parts.append(TextPart(type="text", text=full_text))

                        # Handle thinking_delta chunk
                        elif chunk["type"] == "thinking_delta":
                            idx = chunk["index"]

                            if chunk["is_start"]:
                                yield await self._emit_event(
                                    model_content_block_start_event(
                                        request_id=request_id,
                                        block_index=idx,
                                        block_type="thinking",
                                    ),
                                    event_bus,
                                )

                            if chunk["delta"]:
                                yield await self._emit_event(
                                    model_content_block_delta_event(
                                        request_id=request_id,
                                        block_index=idx,
                                        delta_type="thinking",
                                        delta=chunk["delta"],
                                    ),
                                    event_bus,
                                )
                                block_accumulators.setdefault(idx, []).append(chunk["delta"])

                            if chunk["is_end"]:
                                full_thinking = "".join(block_accumulators.get(idx, []))
                                yield await self._emit_event(
                                    model_content_block_stop_event(
                                        request_id=request_id,
                                        block_index=idx,
                                        block_type="thinking",
                                        full_content=full_thinking,
                                    ),
                                    event_bus,
                                )
                                if full_thinking:
                                    from hawi.agent.message import ReasoningPart
                                    content_parts.append(ReasoningPart(
                                        type="reasoning",
                                        reasoning=full_thinking,
                                        signature=None,
                                        redacted_content=None
                                    ))

                        # Handle tool_call_delta chunk
                        elif chunk["type"] == "tool_call_delta":
                            idx = chunk["index"]

                            if chunk["is_start"]:
                                yield await self._emit_event(
                                    model_content_block_start_event(
                                        request_id=request_id,
                                        block_index=idx,
                                        block_type="tool_use",
                                        tool_call_id=chunk.get("id"),
                                        tool_name=chunk.get("name"),
                                    ),
                                    event_bus,
                                )
                                # Initialize accumulator for this tool call
                                tool_call_accumulators[idx] = {
                                    "id": chunk.get("id") or "",
                                    "name": chunk.get("name") or "",
                                    "arguments": "",
                                }

                            if chunk.get("arguments_delta"):
                                args_delta = chunk["arguments_delta"]
                                yield await self._emit_event(
                                    model_content_block_delta_event(
                                        request_id=request_id,
                                        block_index=idx,
                                        delta_type="tool_input",
                                        delta=args_delta,
                                    ),
                                    event_bus,
                                )
                                if idx in tool_call_accumulators:
                                    tool_call_accumulators[idx]["arguments"] += args_delta

                            if chunk["is_end"]:
                                tool_info = tool_call_accumulators.get(idx, {"id": "", "name": "", "arguments": ""})
                                # Parse arguments JSON
                                import json
                                try:
                                    parsed_args = json.loads(tool_info["arguments"]) if tool_info["arguments"] else {}
                                except json.JSONDecodeError:
                                    parsed_args = {}

                                yield await self._emit_event(
                                    model_content_block_stop_event(
                                        request_id=request_id,
                                        block_index=idx,
                                        block_type="tool_use",
                                        tool_call_id=tool_info["id"],
                                        tool_name=tool_info["name"],
                                        tool_arguments=parsed_args,
                                    ),
                                    event_bus,
                                )

                                if tool_info["id"] and tool_info["name"]:
                                    tool_calls.append(ToolCallPart(
                                        type="tool_call",
                                        id=tool_info["id"],
                                        name=tool_info["name"],
                                        arguments=parsed_args,
                                    ))
                                    # Send agent_tool_call event
                                    yield await self._emit_event(
                                        agent_tool_call_event(
                                            run_id=run_id,
                                            tool_name=tool_info["name"],
                                            arguments=parsed_args,
                                            tool_call_id=tool_info["id"],
                                        ),
                                        event_bus,
                                    )

                        # Handle finish chunk
                        elif chunk["type"] == "finish":
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
                finally:
                    # Ensure the model stream generator is properly closed
                    await model_stream_gen.aclose()

                if state.error:
                    # Send error event before breaking (error occurred during streaming)
                    yield await self._emit_event(
                        agent_error_event(run_id=run_id, error_type="model_error", error_message=state.error),
                        event_bus,
                    )
                    break

                # Model stream stop
                yield await self._emit_event(
                    model_stream_stop_event(
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
                # tool_calls is already list[ToolCallPart], pass directly
                self._context.add_assistant_message(
                    content=response_content,
                    tool_calls=tool_calls if tool_calls else None,
                )

                # Check if tool calls need to be executed
                if not tool_calls:
                    # No tool calls, we're done
                    duration_ms = (time.time() - start_time) * 1000
                    yield await self._emit_event(
                        agent_run_stop_event(
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
                    yield await self._emit_event(
                        agent_tool_result_event(
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

        except Exception as e:
            state.error = f"{type(e).__name__}: {e}"
            yield await self._emit_event(
                agent_error_event(
                    run_id=run_id,
                    error_type="exception",
                    error_message=state.error,
                ),
                event_bus,
            )
            # Re-raise the exception to preserve traceback for debugging
            raise

        finally:
            # after_conversation hook
            self._invoke_hook("after_conversation", self)

    async def _call_model_with_retry_streaming(
        self,
        model: Model,
        policy: dict[str, ModelFailurePolicy],
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
            if p.action == "retry" and p.retry_count > max_retries:
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

            except Exception as e:
                stream_gen = None

                last_error = e
                error_type = model.classify_error(e)
                policy_for_error = policy.get(error_type, ModelFailurePolicy(error_type, "stop"))

                if policy_for_error.action == "stop":
                    # Non-retryable error - raise immediately
                    raise

                if attempt < max_retries:
                    import asyncio
                    await asyncio.sleep(min(2 ** attempt, 60))

        if last_error:
            # All retries exhausted for retryable errors
            state.error = f"Model call failed after {attempt + 1} attempts: {last_error}"
            raise last_error

    async def _call_model_with_retry(
        self,
        model: Model,
        policy: dict[str, ModelFailurePolicy],
        state: _ExecutionState,
    ) -> Any:
        """Call model with retry logic based on failure policy."""
        last_error = None
        max_retries = 0

        # Calculate max retries from policy
        for p in policy.values():
            if p.action == "retry" and p.retry_count > max_retries:
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
            except Exception as e:
                last_error = e
                error_type = model.classify_error(e)
                policy_for_error = policy.get(error_type, ModelFailurePolicy(error_type, "stop"))

                if policy_for_error.action == "stop":
                    # Non-retryable error - raise immediately
                    raise

                if attempt < max_retries:
                    # Exponential backoff
                    wait_time = min(2**attempt, 60)
                    await asyncio.sleep(wait_time)
                    continue

        if last_error:
            # All retries exhausted for retryable errors
            state.error = f"Model call failed after {attempt + 1} attempts: {last_error}"
            raise last_error

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
            result = ToolResult(success=False, error=f"Tool '{tool_name}' not found")
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
                # All errors return to model as string (per design requirement)
                result = ToolResult(success=False, error=f"{type(e).__name__}: {e}")

        duration_ms = (time.time() - start_time) * 1000

        # after_tool_calling hook
        self._invoke_hook("after_tool_calling", self, tool_name, arguments, result)

        # Add tool result to context (unless audit pending - will be added after approval)
        if not (tool and getattr(tool, "audit", False)):
            result_content = result.output if isinstance(result.output, str) else str(result.output)
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
            result_content = result.output if isinstance(result.output, str) else str(result.output)
            self._context.add_tool_result(
                tool_call_id=pending.tool_call_id,
                content=result_content,
                is_error=not result.success,
            )

            # Emit event if event bus provided
            if event_bus is not None:
                await self._emit_event(
                    agent_tool_result_event(
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
            meta = event.metadata
            if event.type == "agent.tool_result":
                tool_calls.append(
                    ToolCallRecord(
                        tool_name=meta["tool_name"],
                        arguments=meta.get("arguments", {}),
                        result=ToolResult(
                            success=meta["success"],
                            output=meta["result_preview"],
                        ),
                        duration_ms=meta["duration_ms"],
                        tool_call_id=meta["tool_call_id"],
                    )
                )
            elif event.type == "agent.run_stop":
                stop_reason = meta.get("stop_reason", "unknown")
            elif event.type == "agent.error":
                error = meta.get("error_message")
                stop_reason = "error"
            elif event.type == "model.stream_stop":
                # Accumulate usage from each model call (for multi-turn conversations)
                usage = meta.get("usage")
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
