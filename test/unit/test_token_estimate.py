from __future__ import annotations

from types import SimpleNamespace
from typing import AsyncGenerator, Iterator

import httpx
import pytest

from hawi.models import Model, TokenEstimate
from hawi.models.anthropic import AnthropicModel
from hawi.models.deepseek.deepseek_anthropic import DeepSeekAnthropicModel
from hawi.models.deepseek import DeepSeekModel
from hawi.models.kimi.kimi_anthropic import KimiAnthropicModel
from hawi.models.kimi.kimi_openai import KimiOpenAIModel
from hawi.models.minimax.minimax_anthropic import MiniMaxAnthropicModel
from hawi.models.minimax.minimax_openai import MiniMaxOpenAIModel
from hawi.models.openai import OpenAIModel
from hawi.models.strands import StrandsModel
from hawi.models.message import DeltaPart, Message, MessageRequest, MessageResponse, TextPart


def create_user_message(text: str) -> Message:
    return {
        "role": "user",
        "content": [{"type": "text", "text": text}],
        "name": None,
        "metadata": None,
    }


class HeuristicModel(Model):
    default_steer_merge_mode = "tool_result_assistant_template_and_user_message"

    def __init__(self) -> None:
        super().__init__()
        self._model_id = "heuristic-model"

    @property
    def model_id(self) -> str:
        return self._model_id

    def _prepare_request_impl(self, request: MessageRequest) -> dict:
        return {"model": self.model_id, "messages": request.messages}

    def _parse_response_impl(self, response: dict) -> MessageResponse:
        return MessageResponse(
            id="resp",
            content=[TextPart(type="text", text="ok")],
            stop_reason="end_turn",
        )

    def _invoke_impl(self, request: MessageRequest) -> MessageResponse:
        return self._parse_response_impl({})

    def _stream_impl(self, request: MessageRequest) -> Iterator[DeltaPart]:
        return iter(())

    async def _ainvoke_impl(
        self,
        request: MessageRequest,
    ) -> AsyncGenerator[DeltaPart, None]:
        if False:
            yield {}  # pragma: no cover


def test_default_estimate_tokens_returns_heuristic() -> None:
    estimate = HeuristicModel().estimate_tokens([create_user_message("hello")])

    assert estimate.method == "heuristic"
    assert estimate.confidence == "approximate"
    assert estimate.input_tokens is not None
    assert estimate.input_tokens > 0
    assert estimate.context_tokens == estimate.input_tokens


@pytest.mark.asyncio
async def test_default_aestimate_tokens_returns_heuristic() -> None:
    estimate = await HeuristicModel().aestimate_tokens([create_user_message("hello")])

    assert estimate.method == "heuristic"
    assert estimate.confidence == "approximate"


def test_deepseek_delegate_uses_provider_marked_heuristic() -> None:
    model = DeepSeekModel(model_id="deepseek-chat", api_key="dummy")

    estimate = model.estimate_tokens([create_user_message("hello")])

    assert estimate.method == "heuristic"
    assert estimate.provider == "deepseek"
    assert estimate.details["provider_count_endpoint"] == "not_available_in_official_docs"


def test_openai_compatible_estimate_is_marked_heuristic() -> None:
    model = OpenAIModel(model_id="gpt-test", api_key="dummy")

    estimate = model.estimate_tokens([create_user_message("hello")])

    assert estimate.method == "heuristic"
    assert estimate.provider == "openai_compatible"
    assert estimate.details["recommended_exact_source"] == "response.usage"


def test_minimax_variants_use_provider_marked_heuristic() -> None:
    for model in [
        MiniMaxOpenAIModel(model_id="MiniMax-M2.5", api_key="dummy"),
        MiniMaxAnthropicModel(
            model_id="MiniMax-M2.7",
            api_key="dummy",
            thinking_budget=0,
        ),
    ]:
        estimate = model.estimate_tokens([create_user_message("hello")])

        assert estimate.method == "heuristic"
        assert estimate.provider == "minimax"
        assert estimate.details["recommended_exact_source"] == "response.usage"


def test_deepseek_anthropic_uses_same_marked_heuristic() -> None:
    model = DeepSeekAnthropicModel(
        model_id="deepseek-chat",
        api_key="dummy",
        thinking_budget=0,
    )

    estimate = model.estimate_tokens([create_user_message("hello")])

    assert estimate.method == "heuristic"
    assert estimate.provider == "deepseek"
    assert estimate.details["provider_count_endpoint"] == "not_available_in_official_docs"


def test_kimi_estimate_tokens_uses_official_endpoint(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_post(
        url: str,
        *,
        headers: dict[str, str],
        json: dict,
        timeout: float,
    ) -> httpx.Response:
        captured.update({
            "url": url,
            "headers": headers,
            "json": json,
            "timeout": timeout,
        })
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            json={"data": {"total_tokens": 42}},
            request=request,
        )

    monkeypatch.setattr("hawi.models.kimi._token_estimate.httpx.post", fake_post)
    model = KimiOpenAIModel(
        model_id="kimi-k2",
        api_key="sk-test",
        base_url="https://api.moonshot.cn/v1",
    )

    estimate = model.estimate_tokens(
        [create_user_message("hello")],
        system="system",
    )

    assert estimate == TokenEstimate(
        input_tokens=42,
        context_tokens=42,
        total_tokens=42,
        method="provider_count",
        confidence="exact",
        provider="kimi",
        model_id="kimi-k2",
        details={"data": {"total_tokens": 42}},
    )
    assert captured["url"] == "https://api.moonshot.cn/v1/tokenizers/estimate-token-count"
    assert captured["headers"] == {"Authorization": "Bearer sk-test"}
    assert captured["timeout"] == 60.0
    assert captured["json"]["model"] == "kimi-k2"  # type: ignore[index]
    assert captured["json"]["messages"]  # type: ignore[index]


def test_kimi_anthropic_uses_same_official_endpoint(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_post(
        url: str,
        *,
        headers: dict[str, str],
        json: dict,
        timeout: float,
    ) -> httpx.Response:
        captured.update({"url": url, "headers": headers, "json": json})
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            json={"data": {"total_tokens": 33}},
            request=request,
        )

    monkeypatch.setattr("hawi.models.kimi._token_estimate.httpx.post", fake_post)
    model = KimiAnthropicModel(
        model_id="kimi-k2.5",
        api_key="sk-test",
        thinking_budget=0,
    )

    estimate = model.estimate_tokens([create_user_message("hello")])

    assert estimate.method == "provider_count"
    assert estimate.input_tokens == 33
    assert estimate.provider == "kimi"
    assert captured["url"] == "https://api.moonshot.cn/v1/tokenizers/estimate-token-count"
    assert captured["json"]["model"] == "kimi-k2.5"  # type: ignore[index]
    assert captured["json"]["messages"]  # type: ignore[index]


def test_anthropic_estimate_tokens_uses_count_tokens() -> None:
    class FakeMessages:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] | None = None

        def count_tokens(self, **kwargs) -> object:
            self.kwargs = kwargs
            return SimpleNamespace(
                input_tokens=17,
                model_dump=lambda: {"input_tokens": 17},
            )

    class FakeClient:
        def __init__(self) -> None:
            self.messages = FakeMessages()

    fake_client = FakeClient()
    model = AnthropicModel(
        model_id="claude-test",
        api_key="sk-test",
        max_output_tokens=4096,
        thinking_budget=0,
    )
    model._client = fake_client  # type: ignore[assignment]

    estimate = model.estimate_tokens(
        [create_user_message("hello")],
        system="system",
    )

    assert estimate.input_tokens == 17
    assert estimate.method == "provider_count"
    assert estimate.confidence == "exact"
    assert estimate.provider == "anthropic"
    assert fake_client.messages.kwargs is not None
    assert fake_client.messages.kwargs["model"] == "claude-test"
    assert "max_tokens" not in fake_client.messages.kwargs


def test_strands_delegates_estimate_tokens_when_available() -> None:
    class FakeStrands:
        model_id = "strands-test"

        def estimate_tokens(self, **kwargs) -> dict[str, int]:
            return {"input_tokens": 29}

    estimate = StrandsModel(FakeStrands()).estimate_tokens(
        [create_user_message("hello")]
    )

    assert estimate.input_tokens == 29
    assert estimate.method == "provider_count"
    assert estimate.provider == "strands"


def test_strands_falls_back_to_marked_heuristic() -> None:
    class FakeStrands:
        model_id = "strands-test"

    estimate = StrandsModel(FakeStrands()).estimate_tokens(
        [create_user_message("hello")]
    )

    assert estimate.method == "heuristic"
    assert estimate.provider == "strands"
    assert estimate.details["provider_count_endpoint"] == "depends_on_underlying_strands_model"
