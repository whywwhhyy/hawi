"""llama.cpp OpenAI-compatible stream processing."""

from __future__ import annotations

from typing import Any, Iterator, cast

from hawi.models.message import DeltaFinishPart, DeltaPart, DeltaProfilePart
from hawi.models.openai._streaming import StreamProcessor

from ._profile import (
    augment_llama_cpp_usage,
    llama_cpp_profile_info,
    normalize_llama_cpp_timings,
    normalize_prompt_progress,
)


class LlamaCppStreamProcessor(StreamProcessor):
    """OpenAI stream processor with llama.cpp profile field support."""

    def __init__(
        self,
        *,
        expect_usage: bool = False,
        profiling_enabled: bool = True,
    ) -> None:
        super().__init__(expect_usage=expect_usage)
        self._profiling_enabled = profiling_enabled
        self._timings: dict[str, float | int] | None = None
        self._prompt_progress: dict[str, float | int] | None = None
        self._peak_decode_tokens_per_second: float | int | None = None
        self._last_decode_sample: tuple[float | int, float | int] | None = None
        self._last_emitted_profile: dict[str, float | int | None] | None = None

    @property
    def timings(self) -> dict[str, float | int] | None:
        return self._timings

    @property
    def prompt_progress(self) -> dict[str, float | int] | None:
        return self._prompt_progress

    def process_chunk(self, chunk_dict: dict[str, Any]) -> Iterator[DeltaPart]:
        timings = normalize_llama_cpp_timings(chunk_dict.get("timings"))
        if timings is not None:
            self._timings = timings
            self._update_peak_decode_tokens_per_second(timings)

        prompt_progress = normalize_prompt_progress(
            chunk_dict.get("prompt_progress")
        )
        if prompt_progress is not None:
            self._prompt_progress = prompt_progress

        profile_delta = self._profile_delta()
        if profile_delta is not None:
            yield profile_delta

        for part in super().process_chunk(chunk_dict):
            yield self._attach_profile(part)

    def finalize(self) -> Iterator[DeltaPart]:
        for part in super().finalize():
            yield self._attach_profile(part)

    def _attach_profile(self, part: DeltaPart) -> DeltaPart:
        if part.get("type") != "finish":
            return part

        finish_part = cast(DeltaFinishPart, part)
        finish_part["usage"] = augment_llama_cpp_usage(
            finish_part.get("usage"),
            self._timings,
        )
        if not self._profiling_enabled:
            return finish_part

        profile = llama_cpp_profile_info(
            timings=self._timings,
            prompt_progress=self._prompt_progress,
            peak_decode_tokens_per_second=self._peak_decode_tokens_per_second,
        )
        if profile is not None:
            finish_part["profile"] = profile
        return finish_part

    def _profile_delta(self) -> DeltaProfilePart | None:
        if not self._profiling_enabled:
            return None

        profile = llama_cpp_profile_info(
            timings=self._timings,
            prompt_progress=self._prompt_progress,
            peak_decode_tokens_per_second=self._peak_decode_tokens_per_second,
        )
        if profile is None:
            return None

        current = cast(dict[str, float | int | None], dict(profile))
        if current == self._last_emitted_profile:
            return None
        self._last_emitted_profile = current
        return DeltaProfilePart(type="profile_delta", profile=profile)

    def _update_peak_decode_tokens_per_second(
        self,
        timings: dict[str, float | int],
    ) -> None:
        decoded = timings.get("predicted_n")
        decode_ms = timings.get("predicted_ms")
        if not isinstance(decoded, (int, float)) or not isinstance(decode_ms, (int, float)):
            return
        if decoded <= 0 or decode_ms <= 0:
            return

        current = (decoded, decode_ms)
        previous = self._last_decode_sample
        self._last_decode_sample = current
        if previous is None:
            return

        previous_decoded, previous_decode_ms = previous
        delta_tokens = decoded - previous_decoded
        delta_ms = decode_ms - previous_decode_ms
        if delta_tokens <= 0 or delta_ms <= 0:
            return

        tokens_per_second = delta_tokens / delta_ms * 1000
        if (
            self._peak_decode_tokens_per_second is None
            or tokens_per_second > self._peak_decode_tokens_per_second
        ):
            self._peak_decode_tokens_per_second = tokens_per_second
