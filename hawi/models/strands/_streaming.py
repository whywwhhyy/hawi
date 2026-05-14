"""
Stream conversion utilities for Strands model adapter.
"""

import json
import logging
from typing import Any, AsyncGenerator, Iterator

from hawi.models import (
    DeltaPart,
    DeltaTextPart,
    DeltaThinkingPart,
    DeltaToolCallPart,
    DeltaFinishPart,
)
from hawi.models.usage import normalize_strands_usage

from ._utils import _map_strands_stop_reason


logger = logging.getLogger(__name__)


def _convert_strands_stream(
    strands_stream: Iterator[Any],
) -> Iterator[DeltaPart]:
    """
    Convert Strands stream to DeltaPart stream.

    Args:
        strands_stream: Strands streaming event iterator

    Yields:
        DeltaPart: Hawi delta parts
    """
    state = {"index": 0, "block_started": False, "pending_usage": None}

    for event in strands_stream:
        yield from _convert_strands_event_to_stream_part(event, state)


def _convert_strands_event_to_stream_part(
    event: Any,
    state: dict[str, Any],
) -> Iterator[DeltaPart]:
    """
    Convert single Strands event to stream part.

    Args:
        event: Strands streaming event (dict or object)
        state: State dict containing index, block_started, block_type, pending_usage

    Yields:
        DeltaPart: Hawi delta parts

    Handles:
        - contentBlockDelta: Text/thinking/tool argument increments
        - contentBlockStart: Block initialization (especially for tool calls)
        - contentBlockStop: Block completion
        - messageStop: Message completion with finish event
        - metadata: Token usage information
        - finish: Legacy custom finish event

    Supports multiple event formats:
        - Nested: {event_type: event_data}
        - Flat: {type: event_type, ...}
        - Object: event.type, event.delta, etc.
    """
    index = state["index"]
    block_started = state["block_started"]
    pending_usage = state["pending_usage"]

    # Strands event formats:
    # 1. Nested format: {event_type: event_data} e.g., {'contentBlockDelta': {'delta': {'text': '...'}}}
    # 2. Flat format: {'type': event_type, ...} e.g., {'type': 'contentBlockDelta', 'delta': {'text': '...'}}
    # 3. Object format: event.type, event.delta, etc.
    event_type = ""
    event_data = {}

    if isinstance(event, dict):
        # Check for flat format (type field)
        if "type" in event:
            event_type = event["type"]
            # event_data contains all fields except type
            event_data = {k: v for k, v in event.items() if k != "type"}
        else:
            # Nested format: find event type key (not 'delta', 'start' etc.)
            event_type_keys = [
                "contentBlockDelta",
                "contentBlockStart",
                "contentBlockStop",
                "messageStart",
                "messageStop",
                "metadata",
                "internalServerException",
                "modelStreamErrorException",
                "serviceUnavailableException",
                "throttlingException",
                "validationException",
                "finish",  # Backward compatibility: old custom event format
            ]
            for key in event_type_keys:
                if key in event:
                    event_type = key
                    event_data = event[key]
                    break
    else:
        # Handle object format (access attributes directly)
        event_type = getattr(event, "type", "")
        event_data = {}
        # Try to get standard Strands event fields from object
        if hasattr(event, "delta"):
            event_data["delta"] = event.delta
        if hasattr(event, "start"):
            event_data["start"] = event.start
        if hasattr(event, "stopReason"):
            event_data["stopReason"] = event.stopReason
        if hasattr(event, "metadata"):
            event_data["metadata"] = event.metadata

    if event_type == "contentBlockDelta":
        # Strands standard event: contentBlockDelta
        delta = event_data.get("delta", {})
        if isinstance(delta, dict):
            # Text increment
            if "text" in delta:
                text = delta["text"]
                if text:
                    # If it's a new block, send start
                    if not block_started:
                        yield DeltaTextPart(
                            type="text_delta",
                            index=index,
                            delta="",
                            is_start=True,
                            is_end=False,
                        )
                        state["block_started"] = True
                        state["block_type"] = "text"

                    yield DeltaTextPart(
                        type="text_delta",
                        index=index,
                        delta=text,
                        is_start=False,
                        is_end=False,
                    )
            # reasoningContent increment
            elif "reasoningContent" in delta:
                reasoning = delta["reasoningContent"]
                reasoning_text = ""
                if isinstance(reasoning, dict):
                    reasoning_text = reasoning.get("text") or ""
                if reasoning_text:
                    if not block_started:
                        yield DeltaThinkingPart(
                            type="reasoning_delta",
                            index=index,
                            delta="",
                            is_start=True,
                            is_end=False,
                        )
                        state["block_started"] = True
                        state["block_type"] = "reasoning"

                    yield DeltaThinkingPart(
                        type="reasoning_delta",
                        index=index,
                        delta=reasoning_text,
                        is_start=False,
                        is_end=False,
                    )
            # Tool input increment
            elif "toolUse" in delta:
                tool_input = delta["toolUse"].get("input", "")
                # Tool block already initialized in contentBlockStart
                # Just ensure block_type is set
                if block_started:
                    state["block_type"] = "tool_use"
                if tool_input:
                    yield DeltaToolCallPart(
                        type="tool_call_delta",
                        index=index,
                        id=None,
                        name=None,
                        arguments_delta=tool_input
                        if isinstance(tool_input, str)
                        else json.dumps(tool_input),
                        is_start=False,
                        is_end=False,
                    )

    elif event_type == "contentBlockStart":
        # Strands block start event
        start = event_data.get("start", {})
        if isinstance(start, dict):
            if "toolUse" in start:
                tool = start["toolUse"]
                yield DeltaToolCallPart(
                    type="tool_call_delta",
                    index=index,
                    id=tool.get("toolUseId"),
                    name=tool.get("name"),
                    arguments_delta="",
                    is_start=True,
                    is_end=False,
                )
                state["block_started"] = True
                state["block_type"] = "tool_use"

    elif event_type == "contentBlockStop":
        # Strands block end event - send is_end marker
        if block_started:
            # Send appropriate end event based on block type
            block_type = state.get("block_type", "text")
            if block_type == "tool_use":
                yield DeltaToolCallPart(
                    type="tool_call_delta",
                    index=index,
                    id=None,
                    name=None,
                    arguments_delta="",
                    is_start=False,
                    is_end=True,
                )
            elif block_type == "reasoning":
                yield DeltaThinkingPart(
                    type="reasoning_delta",
                    index=index,
                    delta="",
                    is_start=False,
                    is_end=True,
                )
            else:  # text or default
                yield DeltaTextPart(
                    type="text_delta",
                    index=index,
                    delta="",
                    is_start=False,
                    is_end=True,
                )
            state["block_started"] = False
            state["block_type"] = None
            state["index"] = index + 1

    elif event_type == "messageStop":
        # Strands message end event
        stop_reason = event_data.get("stopReason", "end_turn")
        if isinstance(stop_reason, dict):
            stop_reason = stop_reason.get("stopReason", "end_turn")
        mapped_stop_reason = (
            _map_strands_stop_reason(stop_reason) if stop_reason else "end_turn"
        )
        yield DeltaFinishPart(
            type="finish",
            stop_reason=mapped_stop_reason,
            usage=pending_usage,
        )
        state["pending_usage"] = None

    elif event_type == "metadata":
        # Strands returns usage in metadata event
        # event_data is already metadata content when using new format
        # { "metadata": { "usage": {...} } } -> event_data = { "usage": {...} }
        usage = event_data.get("usage") if isinstance(event_data, dict) else None
        if usage:
            # Save usage to pending, wait for finish event
            new_usage = normalize_strands_usage(usage)
            if new_usage is not None:
                state["pending_usage"] = new_usage

    # Retain backward compatibility for old custom event format
    elif event_type == "finish":
        stop_reason = event_data.get("stop_reason", "end_turn")
        mapped_stop_reason = (
            _map_strands_stop_reason(stop_reason) if stop_reason else "end_turn"
        )
        yield DeltaFinishPart(
            type="finish",
            stop_reason=mapped_stop_reason,
            usage=pending_usage,
        )
        state["pending_usage"] = None

    else:
        # Unknown event type, try generic handling
        logger.debug(f"Unknown strands event type: {event_type}")
