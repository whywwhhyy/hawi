from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, cast

import pytest

from hawi.agent import HawiAgent
from hawi.events import EventBus
from hawi.models import Model, ModelRegistry
from hawi.models.llama_cpp import (
    LlamaCppModel,
    LlamaCppStreamProcessor,
    normalize_llama_cpp_timings,
)
from hawi.models.message import (
    DeltaPart,
    Message,
    MessageRequest,
    MessageResponse,
    TextPart,
)


def _user_message(text: str = "hi") -> Message:
    return {
        "role": "user",
        "content": [{"type": "text", "text": text}],
        "name": None,
        "metadata": None,
    }


def test_llama_cpp_prepare_stream_request_enables_profile_fields() -> None:
    model = LlamaCppModel(
        model_id="local-model",
        base_url="http://127.0.0.1:8080/v1",
    )
    request = MessageRequest(
        messages=[_user_message()],
        max_output_tokens=32,
    )

    payload = model._prepare_stream_request(request)

    assert payload["stream"] is True
    assert payload["extra_body"]["return_progress"] is True
    assert payload["extra_body"]["timings_per_token"] is True
    assert "return_progress" not in payload
    assert "timings_per_token" not in payload
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["max_tokens"] == 32
    assert "max_completion_tokens" not in payload


def test_llama_cpp_prepare_request_respects_profiling_override() -> None:
    model = LlamaCppModel(
        model_id="local-model",
        base_url="http://127.0.0.1:8080/v1",
    )
    request = MessageRequest(
        messages=[_user_message()],
        profiling=False,
    )

    non_stream_payload = model._prepare_request_impl(request)
    stream_payload = model._prepare_stream_request(request)

    assert model.supports_profiling() is True
    assert "extra_body" not in non_stream_payload
    assert "extra_body" not in stream_payload


def test_llama_cpp_timings_normalizes_consumed_fields() -> None:
    timings = normalize_llama_cpp_timings(
        {
            "cache_n": 123,
            "prompt_n": 456,
            "prompt_ms": 1234.0,
            "predicted_n": 100,
            "predicted_ms": 2000.0,
        }
    )

    assert timings is not None
    assert timings["cache_n"] == 123
    assert timings["prompt_n"] == 456
    assert timings["predicted_n"] == 100
    assert timings["predicted_ms"] == 2000.0


def test_llama_cpp_stream_processor_attaches_profile_and_usage_fallbacks() -> None:
    processor = LlamaCppStreamProcessor(expect_usage=True)
    progress_parts = list(
        processor.process_chunk(
            {
                "choices": [{"index": 0, "delta": {}}],
                "prompt_progress": {
                    "total": 579,
                    "cache": 123,
                    "processed": 456,
                    "time_ms": 1000,
                },
            }
        )
    )
    assert len(progress_parts) == 1
    assert progress_parts[0]["type"] == "profile_delta"
    assert progress_parts[0]["profile"].get("cache_tokens") == 123
    assert progress_parts[0]["profile"].get("prefill_tokens") == 333

    parts = list(
        processor.process_chunk(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 579,
                    "completion_tokens": 100,
                    "total_tokens": 679,
                },
                "timings": {
                    "cache_n": 123,
                    "prompt_n": 456,
                    "prompt_ms": 1234.0,
                    "prompt_per_token_ms": 2.706,
                    "prompt_per_second": 369.5,
                    "predicted_n": 100,
                    "predicted_ms": 2000.0,
                    "predicted_per_token_ms": 20.0,
                    "predicted_per_second": 50.0,
                },
            }
        )
    )

    assert parts[0]["type"] == "profile_delta"
    assert parts[0]["profile"].get("peak_decode_tokens_per_second") == 50.0
    finish = parts[-1]
    assert finish["type"] == "finish"
    usage = finish["usage"]
    assert usage is not None
    assert usage["input_tokens"] == 579
    assert usage["output_tokens"] == 100
    assert usage.get("cache_read_tokens") == 123

    profile = finish.get("profile")
    assert profile is not None
    assert profile == {
        "ttft_ms": 1234.0,
        "prefill_ms": 1234.0,
        "decode_ms": 2000.0,
        "cache_tokens": 123,
        "prefill_tokens": 456,
        "decode_tokens": 100,
        "prefill_tokens_per_second": 369.5,
        "decode_tokens_per_second": 50.0,
        "peak_decode_tokens_per_second": 50.0,
    }


def test_llama_cpp_model_is_registered_builtin_adapter() -> None:
    registry = ModelRegistry()
    registry.clear()

    assert registry.get_model_adapter("LlamaCppModel") is LlamaCppModel

    registry.register_provider(
        "llama",
        "LlamaCppModel",
        ["local"],
        {"base_url": "http://127.0.0.1:8080/v1"},
        quiet=True,
    )

    model = registry.create_model("llama/local")

    assert isinstance(model, LlamaCppModel)
    assert model.model_id == "local"


class ProfileFinishModel(Model):
    default_steer_merge_mode = "tool_result_assistant_template_and_user_message"

    def __init__(self) -> None:
        super().__init__()
        self._model_id = "profile-finish"

    @property
    def model_id(self) -> str:
        return self._model_id

    def _prepare_request_impl(self, request: MessageRequest) -> dict[str, Any]:
        return {}

    def _parse_response_impl(self, response: dict[str, Any]) -> MessageResponse:
        return MessageResponse(
            id="profile-finish",
            content=[TextPart(type="text", text="ok")],
            stop_reason="end_turn",
            usage={"input_tokens": 579, "output_tokens": 100},
        )

    def _invoke_impl(self, request: MessageRequest) -> MessageResponse:
        return self._parse_response_impl({})

    async def _ainvoke_impl(
        self,
        request: MessageRequest,
    ) -> AsyncGenerator[DeltaPart, None]:
        yield cast(
            DeltaPart,
            {
                "type": "profile_delta",
                "profile": {
                    "cache_tokens": 123,
                    "prefill_ms": 246.0,
                    "prefill_tokens": 456,
                    "prefill_tokens_per_second": 369.5,
                },
            },
        )
        yield cast(
            DeltaPart,
            {
                "type": "text_delta",
                "index": 0,
                "delta": "ok",
                "is_start": True,
                "is_end": True,
            },
        )
        yield cast(
            DeltaPart,
            {
                "type": "finish",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 579, "output_tokens": 100},
                "profile": {
                    "ttft_ms": 1234.0,
                    "prefill_ms": 1234.0,
                    "decode_ms": 2000.0,
                    "prefill_tokens": 456,
                    "decode_tokens": 100,
                    "prefill_tokens_per_second": 369.5,
                    "decode_tokens_per_second": 50.0,
                    "peak_decode_tokens_per_second": 55.0,
                },
            },
        )


@pytest.mark.asyncio
async def test_agent_metadata_uses_finish_profile_timing_fields() -> None:
    events: list[Any] = []
    bus = EventBus()
    bus.subscribe_blocking(events.append, event_types=["model.profile", "model.metadata"])
    agent = HawiAgent(model=ProfileFinishModel(), event_bus=bus, streaming=False)

    try:
        await agent.arun("hi")
    finally:
        bus.close()

    profile_event = events[0]
    assert profile_event.type == "model.profile"
    assert profile_event.cache_tokens == 123
    assert profile_event.prefill_ms == 246.0
    assert profile_event.prefill_tokens == 456
    assert profile_event.prefill_tokens_per_second == 369.5

    metadata = events[-1]
    assert metadata.type == "model.metadata"
    assert metadata.ttft_ms == 1234.0
    assert metadata.prefill_ms == 1234.0
    assert metadata.decode_ms == 2000.0
    assert metadata.prefill_tokens == 456
    assert metadata.decode_tokens == 100
    assert metadata.prefill_tokens_per_second == 369.5
    assert metadata.decode_tokens_per_second == 50.0
    assert metadata.peak_decode_tokens_per_second == 55.0
