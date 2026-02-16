import pytest
from unittest.mock import MagicMock

from hawi.agent.message import MessageRequest
from hawi.agent.models.kimi.kimi_openai import KimiOpenAIModel
from hawi.agent.models.kimi.kimi_anthropic import KimiAnthropicModel


def _user_message():
    return {
        "role": "user",
        "content": [{"type": "text", "text": "Hello"}],
        "name": None,
        "tool_calls": None,
        "tool_call_id": None,
        "metadata": None,
    }


def test_temperature_range_enforced():
    model = KimiOpenAIModel(
        model_id="kimi-k2",
        api_key="test-key",
        temperature=1.2,
    )
    request = MessageRequest(messages=[_user_message()])

    with pytest.raises(ValueError):
        model._prepare_request_impl(request)


def test_temperature_zero_rejects_multiple_choices():
    model = KimiOpenAIModel(
        model_id="kimi-k2",
        api_key="test-key",
        temperature=0.0,
        n=2,
    )
    request = MessageRequest(messages=[_user_message()])

    with pytest.raises(ValueError):
        model._prepare_request_impl(request)


def test_tool_choice_required_downgraded_to_auto():
    model = KimiOpenAIModel(
        model_id="kimi-k2",
        api_key="test-key",
    )
    request = MessageRequest(
        messages=[_user_message()],
        tool_choice={"type": "any", "name": None},
    )

    req = model._prepare_request_impl(request)

    assert req.get("tool_choice") == "auto"


def test_stream_includes_usage():
    model = KimiOpenAIModel(
        model_id="kimi-k2",
        api_key="test-key",
    )
    request = MessageRequest(messages=[_user_message()])

    model._client = MagicMock()
    model._client.chat.completions.create.return_value = []

    list(model._stream_impl(request))

    _, kwargs = model._client.chat.completions.create.call_args
    assert kwargs.get("stream") is True
    assert kwargs.get("stream_options") == {"include_usage": True}


def test_anthropic_citations_serialized():
    model = KimiAnthropicModel(
        model_id="kimi-k2.5",
        api_key="test-key",
    )

    block = {
        "type": "text",
        "text": "Hello",
        "citations": [{"source": "a"}],
    }

    result = model._serialize_content_block(block)

    assert result["citations"] == [{"source": "a"}]
