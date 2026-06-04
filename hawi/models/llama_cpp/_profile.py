"""llama.cpp server profile normalization helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

from hawi.models.message import ModelProfileInfo, TokenUsage
from hawi.models.usage import normalize_token_usage


TIMING_FIELDS = (
    "cache_n",
    "prompt_n",
    "prompt_ms",
    "prompt_per_token_ms",
    "prompt_per_second",
    "predicted_n",
    "predicted_ms",
    "predicted_per_token_ms",
    "predicted_per_second",
    "draft_n",
    "draft_n_accepted",
)

PROMPT_PROGRESS_FIELDS = ("total", "cache", "processed", "time_ms")


def normalize_llama_cpp_timings(value: Any) -> dict[str, float | int] | None:
    """Return the llama.cpp timings fields Hawi consumes."""
    data = _as_mapping(value)
    if not data:
        return None

    result: dict[str, float | int] = {}
    for field in TIMING_FIELDS:
        number = _number_or_none(data.get(field))
        if number is not None:
            result[field] = number
    return result or None


def normalize_prompt_progress(value: Any) -> dict[str, float | int] | None:
    """Normalize server ``prompt_progress`` fields used for profile fallback."""
    data = _as_mapping(value)
    if not data:
        return None

    result: dict[str, float | int] = {}
    for field in PROMPT_PROGRESS_FIELDS:
        number = _number_or_none(data.get(field))
        if number is not None:
            result[field] = number
    return result or None


def augment_llama_cpp_usage(
    usage: TokenUsage | Mapping[str, Any] | None,
    timings: Any,
) -> TokenUsage | None:
    """Fill usage cache/token fallbacks from llama.cpp final timings."""
    normalized = normalize_token_usage(usage)
    timing_data = normalize_llama_cpp_timings(timings)
    if timing_data is None:
        return normalized

    cache_tokens = _int_or_none(timing_data.get("cache_n"))
    prefill_tokens = _int_or_none(timing_data.get("prompt_n"))
    decode_tokens = _int_or_none(timing_data.get("predicted_n"))

    if normalized is None:
        input_tokens = (cache_tokens or 0) + (prefill_tokens or 0)
        output_tokens = decode_tokens or 0
        if input_tokens == 0 and output_tokens == 0:
            return None
        normalized = cast(
            TokenUsage,
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "context_tokens": input_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        )
    else:
        normalized = cast(TokenUsage, dict(normalized))

    if cache_tokens is not None and normalized.get("cache_read_tokens") is None:
        normalized["cache_read_tokens"] = cache_tokens

    if normalized.get("context_tokens") is None:
        normalized["context_tokens"] = normalized["input_tokens"]
    if normalized.get("total_tokens") is None:
        normalized["total_tokens"] = (
            normalized["input_tokens"] + normalized["output_tokens"]
        )
    return normalized


def llama_cpp_profile_info(
    *,
    timings: Any,
    prompt_progress: Any = None,
    peak_decode_tokens_per_second: float | int | None = None,
) -> ModelProfileInfo | None:
    """Map llama.cpp profile fields to Hawi's provider-neutral profile shape."""
    timing_data = normalize_llama_cpp_timings(timings)
    if timing_data:
        result: ModelProfileInfo = {}
        _put_number(result, "ttft_ms", timing_data.get("prompt_ms"))
        _put_number(result, "prefill_ms", timing_data.get("prompt_ms"))
        _put_number(result, "decode_ms", timing_data.get("predicted_ms"))
        _put_number(result, "cache_tokens", timing_data.get("cache_n"))
        _put_number(result, "prefill_tokens", timing_data.get("prompt_n"))
        _put_number(result, "decode_tokens", timing_data.get("predicted_n"))
        _put_number(
            result,
            "prefill_tokens_per_second",
            timing_data.get("prompt_per_second"),
        )
        _put_number(
            result,
            "decode_tokens_per_second",
            timing_data.get("predicted_per_second"),
        )
        _put_number(
            result,
            "peak_decode_tokens_per_second",
            peak_decode_tokens_per_second,
        )
        return result or None

    progress = normalize_prompt_progress(prompt_progress)
    if not progress:
        return None

    cache = _number_or_none(progress.get("cache")) or 0
    processed = _number_or_none(progress.get("processed")) or 0
    time_ms = _number_or_none(progress.get("time_ms")) or 0
    prefill_tokens = max(0, processed - cache)

    result: ModelProfileInfo = {"prefill_tokens": prefill_tokens}
    _put_number(result, "cache_tokens", cache)
    _put_number(result, "prefill_ms", time_ms)
    _put_number(result, "ttft_ms", time_ms)
    if prefill_tokens > 0 and time_ms > 0:
        result["prefill_tokens_per_second"] = prefill_tokens / time_ms * 1000
    return result


def llama_cpp_profile_metadata(
    *,
    timings: Any,
    prompt_progress: Any = None,
    peak_decode_tokens_per_second: float | int | None = None,
) -> ModelProfileInfo | None:
    """Backward-compatible alias for llama.cpp profile conversion."""
    return llama_cpp_profile_info(
        timings=timings,
        prompt_progress=prompt_progress,
        peak_decode_tokens_per_second=peak_decode_tokens_per_second,
    )


def _put_number(
    target: Any,
    key: str,
    value: Any,
) -> None:
    number = _number_or_none(value)
    if number is not None:
        target[key] = number


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    return {}


def _number_or_none(value: Any) -> float | int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    if parsed.is_integer():
        return int(parsed)
    return parsed


def _int_or_none(value: Any) -> int | None:
    number = _number_or_none(value)
    if number is None:
        return None
    return int(number)
