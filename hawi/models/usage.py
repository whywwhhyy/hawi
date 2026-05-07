"""Token usage normalization helpers.

Provider APIs expose token accounting with similar high-level concepts but
different field names and cache semantics. These helpers keep adapter code
small and make Hawi's public ``TokenUsage`` shape consistent.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from .message import TokenUsage


TOKEN_USAGE_INT_FIELDS = (
    "input_tokens",
    "output_tokens",
    "context_tokens",
    "total_tokens",
    "cache_write_tokens",
    "cache_read_tokens",
    "cache_miss_tokens",
    "reasoning_tokens",
    "input_audio_tokens",
    "output_audio_tokens",
    "accepted_prediction_tokens",
    "rejected_prediction_tokens",
)


def normalize_openai_usage(usage: Any) -> TokenUsage | None:
    """Normalize OpenAI-compatible Chat Completions usage data.

    This also covers DeepSeek's OpenAI-compatible endpoint, including its
    ``prompt_cache_hit_tokens`` / ``prompt_cache_miss_tokens`` fields.
    """
    data = _as_mapping(usage)
    if not data:
        return None

    prompt_details = _as_mapping(
        data.get("prompt_tokens_details") or data.get("input_tokens_details")
    )
    completion_details = _as_mapping(
        data.get("completion_tokens_details") or data.get("output_tokens_details")
    )

    input_tokens = _int_or_zero(data.get("prompt_tokens") or data.get("input_tokens"))
    output_tokens = _int_or_zero(
        data.get("completion_tokens") or data.get("output_tokens")
    )

    result: dict[str, int | None] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "context_tokens": input_tokens,
        "total_tokens": _int_or_none(data.get("total_tokens")),
    }

    cache_read_tokens = _first_int(
        data.get("prompt_cache_hit_tokens"),
        prompt_details.get("cached_tokens"),
    )
    cache_miss_tokens = _first_int(data.get("prompt_cache_miss_tokens"))
    cache_write_tokens = _first_int(
        data.get("prompt_cache_creation_tokens"),
        prompt_details.get("cache_creation_tokens"),
    )
    _put_optional(result, "cache_read_tokens", cache_read_tokens)
    _put_optional(result, "cache_miss_tokens", cache_miss_tokens)
    _put_optional(result, "cache_write_tokens", cache_write_tokens)
    _put_optional(
        result,
        "reasoning_tokens",
        _first_int(completion_details.get("reasoning_tokens")),
    )
    _put_optional(
        result,
        "input_audio_tokens",
        _first_int(prompt_details.get("audio_tokens")),
    )
    _put_optional(
        result,
        "output_audio_tokens",
        _first_int(completion_details.get("audio_tokens")),
    )
    _put_optional(
        result,
        "accepted_prediction_tokens",
        _first_int(completion_details.get("accepted_prediction_tokens")),
    )
    _put_optional(
        result,
        "rejected_prediction_tokens",
        _first_int(completion_details.get("rejected_prediction_tokens")),
    )

    if result["total_tokens"] is None:
        result["total_tokens"] = input_tokens + output_tokens
    return cast(TokenUsage, result)


def normalize_anthropic_usage(usage: Any) -> TokenUsage | None:
    """Normalize Anthropic Messages usage data.

    Anthropic reports cache-created and cache-read input tokens separately; the
    total input billed/rate-limited tokens are the sum of all input categories.
    """
    data = _as_mapping(usage)
    if not data:
        return None

    input_tokens = _int_or_zero(data.get("input_tokens"))
    output_tokens = _int_or_zero(data.get("output_tokens"))
    cache_write_tokens = _first_int(
        data.get("cache_creation_input_tokens"),
        _sum_cache_creation(_as_mapping(data.get("cache_creation"))),
    )
    cache_read_tokens = _first_int(data.get("cache_read_input_tokens"))

    result: dict[str, int | None] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "context_tokens": input_tokens + (cache_write_tokens or 0) + (cache_read_tokens or 0),
        "total_tokens": _first_int(data.get("total_tokens")),
    }
    _put_optional(result, "cache_write_tokens", cache_write_tokens)
    _put_optional(result, "cache_read_tokens", cache_read_tokens)

    if result["total_tokens"] is None:
        result["total_tokens"] = (
            input_tokens + output_tokens + (cache_write_tokens or 0) + (cache_read_tokens or 0)
        )
    return cast(TokenUsage, result)


def normalize_strands_usage(usage: Any) -> TokenUsage | None:
    """Normalize Strands/Bedrock-style usage data."""
    data = _as_mapping(usage)
    if not data:
        return None

    input_tokens = _int_or_zero(data.get("inputTokens") or data.get("input_tokens"))
    output_tokens = _int_or_zero(data.get("outputTokens") or data.get("output_tokens"))
    cache_write_tokens = _first_int(
        data.get("cacheWriteInputTokens"),
        data.get("cache_write_tokens"),
        data.get("cache_write_input_tokens"),
    )
    cache_read_tokens = _first_int(
        data.get("cacheReadInputTokens"),
        data.get("cache_read_tokens"),
        data.get("cache_read_input_tokens"),
    )

    result: dict[str, int | None] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "context_tokens": input_tokens + (cache_write_tokens or 0) + (cache_read_tokens or 0),
        "total_tokens": _first_int(data.get("totalTokens"), data.get("total_tokens")),
    }
    _put_optional(result, "cache_write_tokens", cache_write_tokens)
    _put_optional(result, "cache_read_tokens", cache_read_tokens)
    if result["total_tokens"] is None:
        result["total_tokens"] = (
            input_tokens + output_tokens + (cache_write_tokens or 0) + (cache_read_tokens or 0)
        )
    return cast(TokenUsage, result)


def normalize_token_usage(usage: Any) -> TokenUsage | None:
    """Normalize already-Hawi-shaped usage and ensure ``total_tokens`` exists."""
    data = _as_mapping(usage)
    if not data:
        return None
    result: dict[str, int | None] = {}
    for field in TOKEN_USAGE_INT_FIELDS:
        value = _int_or_none(data.get(field))
        if value is not None or field in {"input_tokens", "output_tokens"}:
            result[field] = value or 0
    if "input_tokens" not in result:
        result["input_tokens"] = 0
    if "output_tokens" not in result:
        result["output_tokens"] = 0
    if result.get("context_tokens") is None:
        result["context_tokens"] = result["input_tokens"]
    if result.get("total_tokens") is None:
        result["total_tokens"] = usage_total(cast(TokenUsage, result))
    return cast(TokenUsage, result)


def merge_token_usage(
    current: TokenUsage | None,
    usage: TokenUsage | Mapping[str, Any] | None,
) -> TokenUsage | None:
    """Add one per-call usage record into a cumulative usage record."""
    normalized = normalize_token_usage(usage)
    if normalized is None:
        return current
    if current is None:
        return normalized

    merged: dict[str, int | None] = {}
    current_normalized = normalize_token_usage(current) or current
    for field in TOKEN_USAGE_INT_FIELDS:
        a = _int_or_none(current_normalized.get(field))
        b = _int_or_none(normalized.get(field))
        if a is None and b is None:
            continue
        merged[field] = (a or 0) + (b or 0)

    if "input_tokens" not in merged:
        merged["input_tokens"] = 0
    if "output_tokens" not in merged:
        merged["output_tokens"] = 0
    if "context_tokens" not in merged:
        merged["context_tokens"] = merged["input_tokens"]
    if "total_tokens" not in merged:
        merged["total_tokens"] = usage_total(cast(TokenUsage, merged))
    return cast(TokenUsage, merged)


def usage_context_tokens(usage: Mapping[str, Any] | None) -> int | None:
    """Return normalized prompt/context occupancy for one model request."""
    data = normalize_token_usage(usage)
    if data is None:
        return None
    return _int_or_none(data.get("context_tokens"))


def usage_total(usage: Mapping[str, Any] | None) -> int:
    """Return a robust total token count for display."""
    data = _as_mapping(usage)
    if not data:
        return 0
    total = _int_or_none(data.get("total_tokens"))
    if total is not None:
        return total
    return (
        _int_or_zero(data.get("input_tokens"))
        + _int_or_zero(data.get("output_tokens"))
        + _int_or_zero(data.get("cache_write_tokens"))
        + _int_or_zero(data.get("cache_read_tokens"))
    )


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    fields = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "context_tokens",
        "prompt_tokens_details",
        "completion_tokens_details",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "cache_creation",
        "inputTokens",
        "outputTokens",
        "totalTokens",
        "cacheWriteInputTokens",
        "cacheReadInputTokens",
    )
    attrs = {
        field: getattr(value, field)
        for field in fields
        if hasattr(value, field)
    }
    if attrs:
        return attrs
    return {}


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _int_or_zero(value: Any) -> int:
    return _int_or_none(value) or 0


def _first_int(*values: Any) -> int | None:
    for value in values:
        converted = _int_or_none(value)
        if converted is not None:
            return converted
    return None


def _put_optional(target: dict[str, int | None], key: str, value: int | None) -> None:
    if value is not None:
        target[key] = value


def _sum_cache_creation(cache_creation: Mapping[str, Any]) -> int | None:
    if not cache_creation:
        return None
    total = 0
    found = False
    for value in cache_creation.values():
        converted = _int_or_none(value)
        if converted is None:
            continue
        total += converted
        found = True
    return total if found else None
