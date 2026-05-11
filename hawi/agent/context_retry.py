"""Helpers for context-length retries after oversized tool results."""

from __future__ import annotations

from hawi.errors import ContextLengthError


def truncate_tool_result_for_retry(
    content: str,
    error: ContextLengthError,
    *,
    attempt: int,
) -> str:
    if len(content) <= 1:
        return content

    target_chars = context_retry_tool_result_target_chars(
        len(content),
        error,
        attempt=attempt,
    )
    if target_chars >= len(content):
        target_chars = max(1, len(content) // 2)

    omitted_chars = max(0, len(content) - target_chars)
    token_detail = ""
    if error.requested_tokens is not None and error.max_context_tokens is not None:
        token_detail = (
            f" requested_tokens={error.requested_tokens}, "
            f"max_context_tokens={error.max_context_tokens},"
        )
    marker = (
        "\n\n[Hawi truncated this tool result after a model context-length "
        f"error;{token_detail} omitted_chars={omitted_chars}.]\n\n"
    )
    budget = max(0, target_chars - len(marker))
    if budget <= 0:
        return marker.strip()

    head_chars = max(1, int(budget * 0.75))
    tail_chars = max(0, budget - head_chars)
    head = content[:head_chars].rstrip()
    if tail_chars <= 0:
        return head + marker.rstrip()
    tail = content[-tail_chars:].lstrip()
    return head + marker + tail


def context_retry_tool_result_target_chars(
    content_chars: int,
    error: ContextLengthError,
    *,
    attempt: int,
) -> int:
    ratio: float | None = None
    if (
        error.max_context_tokens is not None
        and error.requested_tokens is not None
        and error.max_context_tokens > 0
        and error.requested_tokens > error.max_context_tokens
    ):
        ratio = error.max_context_tokens / error.requested_tokens

    if ratio is not None:
        target = int(content_chars * ratio * 0.8)
    else:
        target = content_chars // 2

    if attempt > 0:
        target = target // (2 ** attempt)

    min_chars = min(2_000, max(1, content_chars // 2))
    return max(min_chars, min(target, content_chars - 1))


def context_retry_needed_reduction_chars(
    error: ContextLengthError,
) -> int | None:
    if (
        error.max_context_tokens is None
        or error.requested_tokens is None
        or error.max_context_tokens <= 0
        or error.requested_tokens <= error.max_context_tokens
    ):
        return None
    overflow_tokens = error.requested_tokens - error.max_context_tokens
    return max(1, int(overflow_tokens * 4 * 1.2))
