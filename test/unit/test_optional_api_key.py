from __future__ import annotations

from hawi.models._auth import DUMMY_API_KEY
from hawi.models.anthropic import AnthropicModel
from hawi.models.minimax import MiniMaxModel
from hawi.models.openai import OpenAIModel


def test_openai_model_uses_dummy_key_for_local_endpoint_without_api_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    model = OpenAIModel(model_id="local-model", base_url="http://localhost:1234/v1")

    assert model.client.api_key == DUMMY_API_KEY
    assert model.client.auth_headers == {"Authorization": f"Bearer {DUMMY_API_KEY}"}


def test_openai_model_treats_blank_api_key_as_dummy(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    model = OpenAIModel(
        model_id="local-model",
        base_url="http://localhost:1234/v1",
        api_key="",
    )

    assert model.client.api_key == DUMMY_API_KEY


def test_anthropic_model_uses_dummy_key_for_local_endpoint_without_api_key(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "env-token")

    model = AnthropicModel(model_id="local-model", base_url="http://localhost:1234")

    assert model.client.api_key == DUMMY_API_KEY
    assert model.client.auth_headers == {"X-Api-Key": DUMMY_API_KEY}
    assert "Authorization" not in model.client.auth_headers


def test_minimax_factory_accepts_missing_api_key_for_local_endpoint() -> None:
    model = MiniMaxModel(
        model_id="local-model",
        base_url="http://localhost:1234/v1",
        api="openai",
    )

    assert model._delegate.client.api_key == DUMMY_API_KEY
