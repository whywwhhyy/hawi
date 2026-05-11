"""Tests for provider-specific model-list adapters."""

from __future__ import annotations

import httpx
import pytest

from hawi.models.deepseek import DeepSeekAnthropicModel
from hawi.models.kimi import KimiAnthropicModel
from hawi.models.minimax import MiniMaxAnthropicModel


def _json_response(url: str, payload: dict) -> httpx.Response:
    return httpx.Response(
        200,
        request=httpx.Request("GET", url),
        json=payload,
    )


def test_deepseek_anthropic_lists_models_from_root_endpoint(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return _json_response(
            url,
            {
                "object": "list",
                "data": [
                    {"id": "deepseek-chat"},
                    {"id": "deepseek-reasoner"},
                ],
            },
        )

    monkeypatch.setattr("hawi.models._model_listing.httpx.get", fake_get)

    model = DeepSeekAnthropicModel(
        api_key="test-key",
        base_url="https://api.deepseek.com/anthropic",
    )

    assert model.list_models() == ["deepseek-chat", "deepseek-reasoner"]
    assert calls[0][0] == "https://api.deepseek.com/models"
    assert calls[0][1]["headers"] == {"Authorization": "Bearer test-key"}


def test_minimax_anthropic_lists_models_from_configured_domain(monkeypatch):
    responses = [
        {
            "data": [{"id": "MiniMax-M2.7"}],
            "has_more": True,
            "last_id": "MiniMax-M2.7",
        },
        {
            "data": [{"id": "MiniMax-M2.5"}],
            "has_more": False,
            "last_id": "MiniMax-M2.5",
        },
    ]
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return _json_response(url, responses.pop(0))

    monkeypatch.setattr("hawi.models._model_listing.httpx.get", fake_get)

    model = MiniMaxAnthropicModel(
        api_key="test-key",
        base_url="https://api.minimaxi.com/anthropic",
    )

    assert model.list_models() == ["MiniMax-M2.7", "MiniMax-M2.5"]
    assert calls[0][0] == "https://api.minimaxi.com/anthropic/v1/models"
    assert calls[0][1]["headers"] == {"Authorization": "Bearer test-key"}
    assert calls[0][1]["params"] == {"limit": 100}
    assert calls[1][1]["params"] == {
        "limit": 100,
        "after_id": "MiniMax-M2.7",
    }


def test_kimi_anthropic_returns_kimi_code_model_id():
    model = KimiAnthropicModel(
        model_id="kimi-k2.5",
        api_key="test-key",
    )

    assert model.list_models() == ["kimi-for-coding"]


@pytest.mark.asyncio
async def test_kimi_anthropic_async_returns_kimi_code_model_id():
    model = KimiAnthropicModel(
        model_id="kimi-k2.5",
        api_key="test-key",
    )

    assert await model.alist_models() == ["kimi-for-coding"]
