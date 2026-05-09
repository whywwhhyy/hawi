from hawi.models.message import Message, MessageRequest
from hawi.models.openai import OpenAIModel


def _assistant_message(content, metadata=None) -> Message:
    return {
        "role": "assistant",
        "content": content,
        "name": None,
        "metadata": metadata,
    }


def test_openai_model_does_not_pass_reasoning_by_default():
    model = OpenAIModel(model_id="test-model", api_key="test-key")
    message = _assistant_message([
        {"type": "reasoning", "reasoning": "Think first.", "signature": None},
        {"type": "text", "text": "Final answer."},
    ])

    result = model._convert_message_to_openai(message)[0]

    assert result["content"] == "Final answer."
    assert "reasoning_content" not in result


def test_openai_model_can_pass_reasoning_in_context():
    model = OpenAIModel(
        model_id="test-model",
        api_key="test-key",
        include_reasoning_in_context=True,
    )
    message = _assistant_message([
        {"type": "reasoning", "reasoning": "Think first.", "signature": None},
        {"type": "text", "text": "Final answer."},
    ])

    result = model._convert_message_to_openai(message)[0]

    assert result["content"] == "Final answer."
    assert result["reasoning_content"] == "Think first."


def test_openai_model_passes_reasoning_on_tool_call_messages():
    model = OpenAIModel(
        model_id="test-model",
        api_key="test-key",
        include_reasoning_in_tool_calls=True,
    )
    message = _assistant_message([
        {"type": "reasoning", "reasoning": "Need the tool.", "signature": None},
        {
            "type": "tool_call",
            "id": "call_123",
            "name": "calculate",
            "arguments": {"expression": "1+1"},
        },
    ])

    results = model._convert_message_to_openai(message)

    assert len(results) == 1
    assert results[0]["content"] is None
    assert results[0]["tool_calls"]
    assert results[0]["reasoning_content"] == "Need the tool."


def test_openai_model_uses_default_reasoning_for_tool_calls():
    model = OpenAIModel(
        model_id="test-model",
        api_key="test-key",
        include_reasoning_in_tool_calls=True,
        default_tool_call_reasoning_content="Using a tool.",
    )
    message = _assistant_message([
        {
            "type": "tool_call",
            "id": "call_123",
            "name": "calculate",
            "arguments": {"expression": "1+1"},
        },
    ])

    result = model._convert_message_to_openai(message)[0]

    assert result["reasoning_content"] == "Using a tool."


def test_openai_model_parses_reasoning_content_as_part():
    model = OpenAIModel(model_id="test-model", api_key="test-key")
    response = {
        "id": "resp_123",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Final answer.",
                "reasoning_content": "Think first.",
            },
            "finish_reason": "stop",
        }],
    }

    result = model._parse_response_impl(response)
    content = list(result.content)

    assert result.reasoning_content == "Think first."
    assert content[0]["type"] == "reasoning"
    assert content[0]["reasoning"] == "Think first."
    assert content[1] == {"type": "text", "text": "Final answer."}


def test_openai_prepare_request_passes_metadata_reasoning():
    model = OpenAIModel(
        model_id="test-model",
        api_key="test-key",
        include_reasoning_in_context=True,
    )
    message = _assistant_message(
        [{"type": "text", "text": "Final answer."}],
        metadata={"reasoning_content": "Stored in metadata."},
    )

    req = model._prepare_request_impl(MessageRequest(messages=[message]))

    assert req["messages"][0]["reasoning_content"] == "Stored in metadata."
