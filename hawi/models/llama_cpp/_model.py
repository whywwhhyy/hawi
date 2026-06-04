"""llama.cpp server OpenAI-compatible model adapter."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Iterator
from typing import Any, cast

from hawi.errors import (
    ContextLengthError,
    NetworkError,
    RemoteError,
)
from hawi.models import (
    DeltaFinishPart,
    DeltaPart,
    DeltaTextPart,
    DeltaThinkingPart,
    DeltaToolCallPart,
    MessageRequest,
    MessageResponse,
    ReasoningPart,
    TextPart,
    TokenEstimate,
    ToolCallPart,
)
from hawi.models.openai import OpenAIModel
from hawi.models.openai._model import _convert_openai_error

from ._profile import augment_llama_cpp_usage, llama_cpp_profile_metadata
from ._streaming import LlamaCppStreamProcessor


DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"


def _convert_llama_cpp_error(error: Exception) -> Exception:
    """Convert common llama.cpp server failures to Hawi model errors."""
    message = str(error)
    lower_message = message.lower()
    if "n_prompt_tokens" in lower_message and "n_ctx" in lower_message:
        return ContextLengthError(f"Context length exceeded: {message}")

    converted = _convert_openai_error(error)
    if converted is not error:
        return converted

    if "slot" in lower_message and "unavailable" in lower_message:
        return RemoteError(f"llama.cpp server slot unavailable: {message}")
    if "connection refused" in lower_message or "failed to connect" in lower_message:
        return NetworkError(f"llama.cpp server connection failed: {message}")
    return error


class LlamaCppModel(OpenAIModel):
    """llama.cpp server adapter using its OpenAI-compatible Chat Completions API."""

    default_steer_merge_mode = "tool_result_assistant_template_and_user_message"

    def __init__(
        self,
        *,
        model_id: str = "local",
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
        max_retries: int = 0,
        require_usage: bool = True,
        return_progress: bool = True,
        timings_per_token: bool = True,
        profiling_enabled: bool = True,
        **params: Any,
    ) -> None:
        if not isinstance(model_id, str):
            raise TypeError(
                f"model_id must be a string, got {type(model_id).__name__}. "
                "Did you pass a list instead of a single model ID?"
            )

        self.return_progress = return_progress
        self.timings_per_token = timings_per_token
        self.profiling_enabled = profiling_enabled
        super().__init__(
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            require_usage=require_usage,
            **params,
        )

    def supports_profiling(self) -> bool:
        return True

    def _prepare_request_impl(self, request: MessageRequest) -> dict[str, Any]:
        req = super()._prepare_request_impl(request)
        self._normalize_generation_token_param(req)
        if self._profiling_enabled_for(request) and self.timings_per_token:
            self._set_extra_body_default(req, "timings_per_token", True)
        return req

    def _prepare_stream_request(self, request: MessageRequest) -> dict[str, Any]:
        req = super()._prepare_stream_request(request)
        self._normalize_generation_token_param(req)
        profiling_enabled = self._profiling_enabled_for(request)
        if profiling_enabled and self.return_progress:
            self._set_extra_body_default(req, "return_progress", True)
        if profiling_enabled and self.timings_per_token:
            self._set_extra_body_default(req, "timings_per_token", True)
        if self.require_usage:
            stream_options = dict(req.get("stream_options") or {})
            stream_options["include_usage"] = True
            req["stream_options"] = stream_options
        return req

    def _profiling_enabled_for(self, request: MessageRequest) -> bool:
        return self.profiling_enabled if request.profiling is None else request.profiling

    @staticmethod
    def _normalize_generation_token_param(req: dict[str, Any]) -> None:
        if "max_completion_tokens" in req and "max_tokens" not in req:
            req["max_tokens"] = req.pop("max_completion_tokens")

    @staticmethod
    def _set_extra_body_default(
        req: dict[str, Any],
        key: str,
        value: Any,
    ) -> None:
        extra_body = req.get("extra_body")
        extra_body = dict(extra_body) if isinstance(extra_body, dict) else {}
        extra_body.setdefault(key, value)
        req["extra_body"] = extra_body

    def _parse_response_impl(self, response: dict[str, Any]) -> MessageResponse:
        result = super()._parse_response_impl(response)
        result.usage = augment_llama_cpp_usage(
            result.usage,
            response.get("timings"),
        )
        return result

    def _invoke_impl(self, request: MessageRequest) -> MessageResponse:
        try:
            return super()._invoke_impl(request)
        except Exception as error:
            converted = _convert_llama_cpp_error(error)
            if converted is not error:
                raise converted from error
            raise

    def _stream_impl(self, request: MessageRequest) -> Iterator[DeltaPart]:
        req = self._prepare_stream_request(request)
        processor = LlamaCppStreamProcessor(
            expect_usage=self.require_usage,
            profiling_enabled=self._profiling_enabled_for(request),
        )

        try:
            stream = self.client.chat.completions.create(**req)
        except Exception as error:
            converted = _convert_llama_cpp_error(error)
            if converted is not error:
                raise converted from error
            raise

        for chunk in stream:
            chunk_dict = chunk.model_dump()
            yield from processor.process_chunk(chunk_dict)
        yield from processor.finalize()

    async def _ainvoke_impl(
        self,
        request: MessageRequest,
    ) -> AsyncGenerator[DeltaPart, None]:
        loop = asyncio.get_event_loop()
        req = self._prepare_request_impl(request)

        try:
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(**req),
            )
        except Exception as error:
            converted = _convert_llama_cpp_error(error)
            if converted is not error:
                raise converted from error
            raise

        response_dict = response.model_dump()
        result = self._parse_response_impl(response_dict)

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

        yield self._finish_part(result, response_dict, request)

    async def _astream_impl(
        self,
        request: MessageRequest,
    ) -> AsyncGenerator[DeltaPart, None]:
        req = self._prepare_stream_request(request)
        processor = LlamaCppStreamProcessor(
            expect_usage=self.require_usage,
            profiling_enabled=self._profiling_enabled_for(request),
        )

        try:
            stream = await self.async_client.chat.completions.create(**req)
            async with stream:
                async for chunk in stream:
                    chunk_dict = chunk.model_dump()
                    for delta_part in processor.process_chunk(chunk_dict):
                        yield delta_part
                for delta_part in processor.finalize():
                    yield delta_part
        except Exception as error:
            converted = _convert_llama_cpp_error(error)
            if converted is not error:
                raise converted from error
            raise

    def _finish_part(
        self,
        result: MessageResponse,
        response: dict[str, Any],
        request: MessageRequest,
    ) -> DeltaFinishPart:
        finish_part = DeltaFinishPart(
            type="finish",
            stop_reason=result.stop_reason or "end_turn",
            usage=result.usage,
        )
        if not self._profiling_enabled_for(request):
            return finish_part

        profile = llama_cpp_profile_metadata(timings=response.get("timings"))
        if profile is not None:
            finish_part["profile"] = profile
        return finish_part

    def _estimate_tokens_impl(
        self,
        request: MessageRequest,
    ) -> TokenEstimate:
        estimate = super()._estimate_tokens_impl(request)
        estimate.provider = "llama_cpp"
        estimate.details["recommended_exact_source"] = "response.usage"
        estimate.details["profile_source"] = "response.timings"
        return estimate
