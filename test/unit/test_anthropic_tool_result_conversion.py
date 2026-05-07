from __future__ import annotations

from hawi.models.anthropic import AnthropicModel
from hawi.models.message import Message, MessageRequest


def test_anthropic_merges_consecutive_tool_results_into_next_user_message() -> None:
    model = AnthropicModel(model_id="claude-test", thinking_budget=None)

    request = MessageRequest(messages=[
        _user("run both tools"),
        _assistant_tool_calls(["call-1", "call-2"]),
        _tool_result("call-1", "first ok"),
        _tool_result("call-2", "second ok"),
    ])

    prepared = model._prepare_request_sync(request)

    assert len(prepared["messages"]) == 3
    tool_result_message = prepared["messages"][2]
    assert tool_result_message["role"] == "user"
    assert [part["type"] for part in tool_result_message["content"]] == [
        "tool_result",
        "tool_result",
    ]
    assert [part["tool_use_id"] for part in tool_result_message["content"]] == [
        "call-1",
        "call-2",
    ]


def test_anthropic_keeps_tool_results_before_interleaved_user_text() -> None:
    model = AnthropicModel(model_id="claude-test", thinking_budget=None)

    request = MessageRequest(messages=[
        _user("run both tools"),
        _assistant_tool_calls(["call-1", "call-2"]),
        _tool_result("call-1", "first ok"),
        _user("please prioritize the follow-up"),
        _tool_result("call-2", "second ok"),
    ])

    prepared = model._prepare_request_sync(request)

    assert len(prepared["messages"]) == 3
    merged_user_message = prepared["messages"][2]
    assert merged_user_message["role"] == "user"
    assert [part["type"] for part in merged_user_message["content"]] == [
        "tool_result",
        "tool_result",
        "text",
    ]
    assert [part["tool_use_id"] for part in merged_user_message["content"][:2]] == [
        "call-1",
        "call-2",
    ]
    assert merged_user_message["content"][2]["text"] == "please prioritize the follow-up"


def test_anthropic_adaptive_thinking_uses_output_config_effort() -> None:
    model = AnthropicModel(
        model_id="claude-opus-4-7",
        thinking_type="adaptive",
        thinking_effort="high",
    )

    prepared = model._prepare_request_sync(MessageRequest(messages=[_user("hi")]))

    assert prepared["thinking"] == {"type": "adaptive"}
    assert prepared["output_config"]["effort"] == "high"


def test_anthropic_legacy_thinking_keeps_budget_tokens() -> None:
    model = AnthropicModel(model_id="claude-sonnet-4-5", thinking_budget=2048)

    prepared = model._prepare_request_sync(MessageRequest(messages=[_user("hi")]))

    assert prepared["thinking"] == {
        "type": "enabled",
        "budget_tokens": 2048,
    }
    assert "output_config" not in prepared


def _user(text: str) -> Message:
    return {
        "role": "user",
        "content": [{"type": "text", "text": text}],
        "name": None,
        "metadata": None,
    }


def _assistant_tool_calls(tool_call_ids: list[str]) -> Message:
    return {
        "role": "assistant",
        "content": [
            {
                "type": "tool_call",
                "id": tool_call_id,
                "name": "example_tool",
                "arguments": {},
            }
            for tool_call_id in tool_call_ids
        ],
        "name": None,
        "metadata": None,
    }


def _tool_result(tool_call_id: str, text: str) -> Message:
    return {
        "role": "tool",
        "content": [
            {
                "type": "tool_result",
                "tool_call_id": tool_call_id,
                "content": [{"type": "text", "text": text}],
                "is_error": False,
            }
        ],
        "name": None,
        "metadata": None,
    }
