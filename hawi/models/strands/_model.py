"""
Strands model adapter for Hawi.

This module provides an adapter for using Strands framework models within the Hawi ecosystem.
"""

import json
import logging
from typing import Any, AsyncGenerator, Iterator, Sequence, cast

from hawi.models import BalanceInfo, Model
from hawi.models import (
    AudioPart,
    ContentPart,
    DocumentPart,
    ImagePart,
    Message,
    MessageRequest,
    MessageResponse,
    ReasoningPart,
    DeltaPart,
    TextPart,
    ToolCallPart,
    ToolChoice,
    ToolDefinition,
    ToolResultPart,
    VideoPart,
    DeltaTextPart,
    DeltaThinkingPart,
    DeltaToolCallPart,
    DeltaFinishPart,
    TokenEstimate,
)
from hawi.models.usage import normalize_strands_usage

from ._converters import (
    _convert_content_to_strands,
    _convert_messages_to_strands,
    _convert_single_message_to_strands,
    _convert_part_to_strands_block,
    _convert_strands_block_to_part,
    _convert_strands_tool_use_to_part,
    _convert_tool_choice_to_strands,
    _convert_tool_definition_to_strands,
)
from ._streaming import _convert_strands_stream, _convert_strands_event_to_stream_part
from ._utils import _map_strands_stop_reason


logger = logging.getLogger(__name__)


class StrandsModel(Model):
    """
    Strands Model to Hawi Model adapter.

    Wraps a Strands Model instance and implements Hawi's Model interface,
    providing automatic conversion between
    the two message formats.

    Attributes:
        strands_model: The underlying Strands Model instance
        model_id: Model identifier (automatically extracted from strands model)

    Example:
        >>> from strands_models import DeepSeekOpenAIModel
        >>> from hawi.models.strands import StrandsModel
        >>>
        >>> strands_model = DeepSeekOpenAIModel(...)
        >>> hawi_model = StrandsModel(strands_model)
        >>> response = hawi_model.invoke(messages=[{"role": "user", "content": [{"type": "text", "text": "Hello"}], "name": None, "tool_calls": None, "tool_call_id": None, "metadata": None}])
    """

    default_steer_merge_mode = "tool_result_assistant_template_and_user_message"

    def __init__(self, strands_model: Any) -> None:
        """
        Initialize adapter.

        Args:
            strands_model: Strands framework Model instance
                (e.g., DeepSeekOpenAIModel, KimiOpenAIModel, etc.)
        """
        self.strands_model = strands_model
        self._model_id = self._extract_model_id()

    def _extract_model_id(self) -> str:
        """Extract model_id from strands model."""
        # Strands models usually store model_id in config
        if hasattr(self.strands_model, "config"):
            config = self.strands_model.config
            if isinstance(config, dict):
                return config.get("model_id", "unknown")
        # Or have model_id attribute directly
        if hasattr(self.strands_model, "model_id"):
            return self.strands_model.model_id
        return "unknown"

    @property
    def model_id(self) -> str:
        """Model identifier."""
        return self._model_id

    # ==========================================================================
    # Request/Response Conversion
    # ==========================================================================

    def _prepare_request_impl(self, request: MessageRequest) -> dict[str, Any]:
        """
        Convert Hawi MessageRequest to Strands format.

        Converts Hawi's generic message format to Strands API request format.
        """
        # Convert messages
        strands_messages = _convert_messages_to_strands(request.messages)

        # Build Strands request
        strands_request: dict[str, Any] = {
            "messages": strands_messages,
        }

        # Convert system prompt
        if request.system:
            # list[ContentPart] format, extract text content
            system_texts = []
            for part in request.system:
                if part.get("type") == "text":
                    system_texts.append(cast(TextPart, part)["text"])
            if system_texts:
                strands_request["system_prompt"] = "\n".join(system_texts)

        # Convert tool definitions
        if request.tools:
            strands_request["tool_specs"] = [
                _convert_tool_definition_to_strands(tool)
                for tool in request.tools
            ]

        # Convert tool_choice
        if request.tool_choice:
            strands_request["tool_choice"] = _convert_tool_choice_to_strands(
                request.tool_choice
            )

        # Convert other parameters
        if request.max_output_tokens is not None:
            strands_request["max_tokens"] = request.max_output_tokens
        if request.temperature is not None:
            strands_request["temperature"] = request.temperature
        if request.top_p is not None:
            strands_request["top_p"] = request.top_p

        return strands_request

    def _parse_response_impl(self, response: dict[str, Any]) -> MessageResponse:
        """
        Convert Strands response to Hawi MessageResponse.

        Converts Strands API response to Hawi's generic response format.
        """
        # Extract content
        content: list[ContentPart] = []

        # Strands response may contain multiple content blocks
        if "content" in response:
            for block in response["content"]:
                part = _convert_strands_block_to_part(block)
                if part:
                    content.append(part)

        # Handle tool calls (toolUse is part of content blocks in Strands)
        # Note: _convert_strands_block_to_part already handles toolUse,
        # so no need to handle separately here

        # Extract usage (Strands uses camelCase field names)
        usage = normalize_strands_usage(response.get("usage"))

        # Extract stop_reason
        stop_reason = response.get("stop_reason")
        if stop_reason:
            stop_reason = _map_strands_stop_reason(stop_reason)

        # Extract reasoning_content (DeepSeek Reasoner, etc.)
        reasoning_content = response.get("reasoning_content")

        return MessageResponse(
            id=response.get("id", ""),
            role="assistant",
            content=content,
            stop_reason=stop_reason,
            usage=usage,
            reasoning_content=reasoning_content,
        )

    # ==========================================================================
    # Invocation Implementation
    # ==========================================================================

    def _invoke_impl(
        self,
        request: MessageRequest,
    ) -> MessageResponse:
        """Sync invocation implementation."""
        # Prepare Strands format request
        strands_request = self._prepare_request_impl(request)

        # Call strands model
        # Strands models usually have run_sync or similar method
        if hasattr(self.strands_model, "run_sync"):
            strands_response = self.strands_model.run_sync(**strands_request)
        elif hasattr(self.strands_model, "invoke"):
            strands_response = self.strands_model.invoke(**strands_request)
        else:
            raise NotImplementedError(
                f"Strands model {type(self.strands_model)} does not support sync invocation"
            )

        # Convert response
        return self._parse_response_impl(strands_response)

    def _stream_impl(self, request: MessageRequest) -> Iterator[DeltaPart]:
        """Sync streaming implementation - bridges async stream using asyncio."""
        import asyncio

        async_gen = self._astream_impl(request)

        # Try to get current running event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is None:
            # No event loop, create new to run async generator
            async def collect():
                result = []
                async for item in async_gen:
                    result.append(item)
                return result

            items = asyncio.run(collect())
            yield from items
        else:
            # Has event loop, use nest-asyncio style bridging
            # Create queue to buffer events
            queue: list[DeltaPart | None] = []
            exhausted = False

            async def pump():
                nonlocal exhausted
                try:
                    async for item in async_gen:
                        queue.append(item)
                finally:
                    queue.append(None)
                    exhausted = True

            # Schedule task in event loop
            _ = asyncio.ensure_future(pump(), loop=loop)

            # Sync iterate, get data from queue via busy wait
            import time
            while True:
                if queue:
                    item = queue.pop(0)
                    if item is None:
                        break
                    yield item
                elif exhausted:
                    break
                else:
                    # Brief yield, allow other tasks to run
                    time.sleep(0.001)

    def _estimate_tokens_impl(
        self,
        request: MessageRequest,
    ) -> TokenEstimate:
        strands_request = self._prepare_request_impl(request)

        for method_name in ("estimate_tokens", "count_tokens"):
            method = getattr(self.strands_model, method_name, None)
            if callable(method):
                result = method(**strands_request)
                return self._coerce_token_estimate(result, method_name)

        estimate = self._heuristic_token_estimate(
            request,
            details={"strands_model": type(self.strands_model).__name__},
        )
        estimate.provider = "strands"
        estimate.details["provider_count_endpoint"] = "depends_on_underlying_strands_model"
        return estimate

    async def _aestimate_tokens_impl(
        self,
        request: MessageRequest,
    ) -> TokenEstimate:
        strands_request = self._prepare_request_impl(request)

        for method_name in ("aestimate_tokens", "acount_tokens"):
            method = getattr(self.strands_model, method_name, None)
            if callable(method):
                result = await method(**strands_request)
                return self._coerce_token_estimate(result, method_name)

        return self._estimate_tokens_impl(request)

    def _coerce_token_estimate(
        self,
        result: Any,
        method_name: str,
    ) -> TokenEstimate:
        if isinstance(result, TokenEstimate):
            if result.provider is None:
                result.provider = "strands"
            if result.model_id is None:
                result.model_id = self.model_id
            result.details.setdefault("strands_method", method_name)
            return result

        if isinstance(result, int):
            tokens = result
            details: dict[str, Any] = {"raw_result": result}
        elif isinstance(result, dict):
            tokens = (
                result.get("input_tokens")
                or result.get("context_tokens")
                or result.get("total_tokens")
                or result.get("tokens")
                or 0
            )
            details = dict(result)
        else:
            tokens = (
                getattr(result, "input_tokens", None)
                or getattr(result, "context_tokens", None)
                or getattr(result, "total_tokens", None)
                or getattr(result, "tokens", None)
                or 0
            )
            details = {
                "raw_result_type": type(result).__name__,
                "strands_method": method_name,
            }

        tokens = int(tokens or 0)
        return TokenEstimate(
            input_tokens=tokens,
            context_tokens=tokens,
            total_tokens=tokens,
            method="provider_count",
            confidence="exact" if tokens > 0 else "approximate",
            provider="strands",
            model_id=self.model_id,
            details={"strands_method": method_name, **details},
        )

    def list_models(self) -> list[str]:
        """Delegate model-list queries to the wrapped Strands model when possible."""
        for method_name in ("list_models", "get_models"):
            method = getattr(self.strands_model, method_name, None)
            if callable(method):
                return self._coerce_model_id_list(method())

        models = getattr(self.strands_model, "models", None)
        if models is not None:
            return self._coerce_model_id_list(models)

        raise NotImplementedError(
            f"{self.__class__.__name__} underlying model does not support model list query"
        )

    async def alist_models(self) -> list[str]:
        """Async model-list query delegated to the wrapped Strands model."""
        for method_name in ("alist_models", "aget_models"):
            method = getattr(self.strands_model, method_name, None)
            if callable(method):
                return await self._acoerce_model_id_list(await method())
        return self.list_models()

    async def _ainvoke_impl(
        self,
        request: MessageRequest,
    ) -> AsyncGenerator[DeltaPart, None]:
        """Async non-streaming implementation - splits full response into DeltaPart sequence

        Args:
            request: Message request

        Yields:
            DeltaPart: Delta block sequence
        """
        strands_request = self._prepare_request_impl(request)

        if hasattr(self.strands_model, "run_async"):
            strands_response = await self.strands_model.run_async(**strands_request)
        elif hasattr(self.strands_model, "ainvoke"):
            strands_response = await self.strands_model.ainvoke(**strands_request)
        else:
            # Fallback: use streaming API and collect response
            full_text_parts: list[str] = []
            full_thinking_parts: list[str] = []
            tool_calls: list[DeltaToolCallPart] = []  # Only collect tool_call_delta type
            final_stop_reason = "end_turn"
            final_usage = None

            async for chunk in self._astream_impl(request):
                chunk_type = chunk["type"]
                if chunk_type == "text_delta":
                    part = cast(DeltaTextPart, chunk)
                    full_text_parts.append(part["delta"])
                elif chunk_type == "reasoning_delta":
                    part = cast(DeltaThinkingPart, chunk)
                    full_thinking_parts.append(part["delta"])
                elif chunk_type == "tool_call_delta":
                    part = cast(DeltaToolCallPart, chunk)
                    tool_calls.append(part)
                elif chunk_type == "signature_delta":
                    # Signatures usually don't need collection, ignore
                    pass
                elif chunk_type == "metadata_delta":
                    # Metadata doesn't need collection
                    pass
                elif chunk_type == "finish":
                    part = cast(DeltaFinishPart, chunk)
                    final_stop_reason = part["stop_reason"]
                    final_usage = part["usage"]

            # Yield collected content as complete parts
            if full_text_parts:
                yield DeltaTextPart(
                    type="text_delta",
                    index=0,
                    delta="".join(full_text_parts),
                    is_start=True,
                    is_end=True,
                )
            if full_thinking_parts:
                yield DeltaThinkingPart(
                    type="reasoning_delta",
                    index=1 if full_text_parts else 0,
                    delta="".join(full_thinking_parts),
                    is_start=True,
                    is_end=True,
                )
            # TODO: Aggregate tool calls properly
            for tc in tool_calls:
                yield tc

            yield DeltaFinishPart(
                type="finish",
                stop_reason=final_stop_reason,
                usage=final_usage,
            )
            return

        result = self._parse_response_impl(strands_response)

        # Yield content blocks as DeltaPart
        for idx, part in enumerate(result.content):
            part_type = part["type"]

            if part_type == "text":
                text_part = cast(TextPart, part)
                yield DeltaTextPart(
                    type="text_delta",
                    index=idx,
                    delta=text_part["text"],
                    is_start=True,
                    is_end=True,
                )
            elif part_type == "reasoning":
                reasoning_part = cast(ReasoningPart, part)
                yield DeltaThinkingPart(
                    type="reasoning_delta",
                    index=idx,
                    delta=reasoning_part.get("reasoning") or "",
                    is_start=True,
                    is_end=True,
                )
            elif part_type == "tool_call":
                tool_part = cast(ToolCallPart, part)
                yield DeltaToolCallPart(
                    type="tool_call_delta",
                    index=idx,
                    id=tool_part["id"],
                    name=tool_part["name"],
                    arguments_delta=json.dumps(tool_part["arguments"]),
                    is_start=True,
                    is_end=True,
                )

        # Yield finish part
        yield DeltaFinishPart(
            type="finish",
            stop_reason=result.stop_reason or "end_turn",
            usage=result.usage,
        )

    async def _astream_impl(self, request: MessageRequest) -> AsyncGenerator[DeltaPart, None]:
        """Async streaming implementation."""
        strands_request = self._prepare_request_impl(request)

        if hasattr(self.strands_model, "run_stream_async"):
            strands_stream = self.strands_model.run_stream_async(**strands_request)
        elif hasattr(self.strands_model, "astream"):
            strands_stream = self.strands_model.astream(**strands_request)
        else:
            # Fallback: call sync stream() and wrap in async generator
            sync_stream = self.strands_model.stream(**strands_request)
            # Handle both sync and async generators (strands may return either)
            if hasattr(sync_stream, "__aiter__"):
                # Already an async generator
                strands_stream = sync_stream
            else:
                # Sync generator, wrap it
                async def async_wrapper():
                    for event in sync_stream:
                        yield event

                strands_stream = async_wrapper()

        state = {"index": 0, "block_started": False, "pending_usage": None}
        async for event in strands_stream:
            for chunk in _convert_strands_event_to_stream_part(event, state):
                yield chunk

    # ==========================================================================
    # Internal Conversion Methods (for testing and backward compatibility)
    # ==========================================================================

    def _convert_messages_to_strands(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert Hawi messages to Strands format."""
        return _convert_messages_to_strands(messages)

    def _convert_single_message_to_strands(self, msg: Message) -> dict[str, Any]:
        """Convert single Hawi message to Strands format."""
        return _convert_single_message_to_strands(msg)

    def _convert_tool_definition_to_strands(self, tool: ToolDefinition) -> dict[str, Any]:
        """Convert Hawi tool definition to Strands format."""
        return _convert_tool_definition_to_strands(tool)

    def _convert_tool_choice_to_strands(self, tool_choice: ToolChoice) -> dict[str, Any]:
        """Convert Hawi tool choice to Strands format."""
        return _convert_tool_choice_to_strands(tool_choice)

    def _convert_strands_event_to_stream_part(
        self, event: Any, state: dict[str, Any]
    ) -> Iterator[DeltaPart]:
        """Convert Strands streaming event to DeltaPart."""
        return _convert_strands_event_to_stream_part(event, state)

    def _map_strands_stop_reason(self, reason: str) -> str:
        """Map Strands stop reason to Hawi format."""
        return _map_strands_stop_reason(reason)

    # ==========================================================================
    # Balance Query (Optional)
    # ==========================================================================

    def get_balance(self) -> list[BalanceInfo]:
        """
        Query account balance.

        If underlying strands model supports balance query, delegate to it.
        Otherwise raise NotImplementedError.
        """
        # Check if strands
        if hasattr(self.strands_model, "get_balance"):
            return self.strands_model.get_balance()

        # Check if has balance attribute
        if hasattr(self.strands_model, "balance"):
            balance = self.strands_model.balance
            if isinstance(balance, list):
                return balance
            elif isinstance(balance, BalanceInfo):
                return [balance]

        raise NotImplementedError(
            f"{self.__class__.__name__} does not support balance query"
        )
