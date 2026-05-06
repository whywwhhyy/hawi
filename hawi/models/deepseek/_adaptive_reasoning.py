"""Helpers for DeepSeek adaptive thinking response normalization."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import cast

from hawi.models.message import DeltaPart, MessageResponse, ReasoningPart


def is_reasoning_model(model_id: str) -> bool:
    """Return whether this DeepSeek model should always expose reasoning."""
    return "reasoner" in model_id.lower()


def ensure_reasoning_part(
    response: MessageResponse,
    reasoning: str | None,
) -> MessageResponse:
    """Ensure response.content starts with a reasoning part, preserving empties."""
    reasoning_text = reasoning or ""
    content = list(response.content)

    for part in content:
        if part.get("type") == "reasoning":
            part["reasoning"] = part.get("reasoning") or ""
            response.reasoning_content = part["reasoning"]
            response.content = content
            return response

    reasoning_part: ReasoningPart = {
        "type": "reasoning",
        "reasoning": reasoning_text,
        "signature": None,
    }
    response.reasoning_content = reasoning_text
    response.content = [reasoning_part] + content
    return response


def should_ensure_reasoning_part(
    model_id: str,
    *,
    server_reasoning_present: bool,
) -> bool:
    """DeepSeek reasoners need an empty block when adaptive thinking is silent."""
    return is_reasoning_model(model_id) or server_reasoning_present


def with_empty_reasoning_delta_if_missing(
    parts: Iterator[DeltaPart],
    *,
    enabled: bool,
) -> Iterator[DeltaPart]:
    """Prepend an empty reasoning delta block if a stream starts without one."""
    if not enabled:
        yield from parts
        return

    saw_reasoning = False
    injected = False

    for part in parts:
        part_type = part.get("type")
        if part_type == "reasoning_delta":
            saw_reasoning = True

        if not injected and not saw_reasoning and part_type in {
            "text_delta",
            "tool_call_delta",
            "finish",
        }:
            yield _empty_reasoning_delta()
            injected = True

        yield _shift_delta_index(part, 1) if injected else part


async def awith_empty_reasoning_delta_if_missing(
    parts: AsyncIterator[DeltaPart],
    *,
    enabled: bool,
) -> AsyncIterator[DeltaPart]:
    """Async variant of with_empty_reasoning_delta_if_missing."""
    if not enabled:
        async for part in parts:
            yield part
        return

    saw_reasoning = False
    injected = False

    async for part in parts:
        part_type = part.get("type")
        if part_type == "reasoning_delta":
            saw_reasoning = True

        if not injected and not saw_reasoning and part_type in {
            "text_delta",
            "tool_call_delta",
            "finish",
        }:
            yield _empty_reasoning_delta()
            injected = True

        yield _shift_delta_index(part, 1) if injected else part


def _empty_reasoning_delta() -> DeltaPart:
    return {
        "type": "reasoning_delta",
        "index": 0,
        "delta": "",
        "is_start": True,
        "is_end": True,
    }


def _shift_delta_index(part: DeltaPart, offset: int) -> DeltaPart:
    if part.get("type") == "finish" or "index" not in part:
        return part

    shifted = dict(part)
    shifted["index"] = cast(int, shifted.get("index", 0)) + offset
    return cast(DeltaPart, shifted)
