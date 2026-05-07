from __future__ import annotations

import pytest

from hawi.errors import ValidationError
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


def test_anthropic_default_max_tokens_grows_for_default_thinking_budget() -> None:
    model = AnthropicModel(model_id="claude-sonnet-4-5")

    prepared = model._prepare_request_sync(MessageRequest(messages=[_user("hi")]))

    assert prepared["thinking"] == {
        "type": "enabled",
        "budget_tokens": 8000,
    }
    assert prepared["max_tokens"] == 12096


def test_anthropic_explicit_too_small_max_tokens_rejected_locally() -> None:
    model = AnthropicModel(
        model_id="claude-sonnet-4-5",
        thinking_budget=8000,
        max_output_tokens=4096,
    )

    with pytest.raises(ValidationError, match="thinking_budget must be less"):
        model._prepare_request_sync(MessageRequest(messages=[_user("hi")]))


def test_anthropic_explicit_large_enough_max_tokens_kept() -> None:
    model = AnthropicModel(
        model_id="claude-sonnet-4-5",
        thinking_budget=8000,
        max_output_tokens=9000,
    )

    prepared = model._prepare_request_sync(MessageRequest(messages=[_user("hi")]))

    assert prepared["max_tokens"] == 9000
    assert prepared["thinking"] == {
        "type": "enabled",
        "budget_tokens": 8000,
    }


def test_anthropic_cache_point_attaches_to_previous_message_block() -> None:
    model = AnthropicModel(model_id="claude-test", thinking_budget=None)
    message = _user("Large reusable content")
    message["content"] = [
        {"type": "text", "text": "Large reusable content"},
        {"type": "cache_point", "cache_point": {"type": "ephemeral"}},
    ]

    prepared = model._prepare_request_sync(MessageRequest(messages=[message]))

    block = prepared["messages"][0]["content"][0]
    assert block["cache_control"] == {"type": "ephemeral"}


def test_anthropic_legacy_cache_control_attaches_to_previous_message_block() -> None:
    model = AnthropicModel(model_id="claude-test", thinking_budget=None)
    message = _user("Large reusable content")
    message["content"] = [
        {"type": "text", "text": "Large reusable content"},
        {"type": "cache_control", "cache_control": {"type": "ephemeral"}},
    ]

    prepared = model._prepare_request_sync(MessageRequest(messages=[message]))

    block = prepared["messages"][0]["content"][0]
    assert block["cache_control"] == {"type": "ephemeral"}


def test_anthropic_top_level_cache_point_uses_extra_body() -> None:
    model = AnthropicModel(model_id="claude-test", thinking_budget=None)

    prepared = model._prepare_request_sync(
        MessageRequest(
            messages=[_user("hi")],
            cache_point={"type": "ephemeral", "ttl": "1h"},
        )
    )

    assert prepared["extra_body"]["cache_control"] == {
        "type": "ephemeral",
        "ttl": "1h",
    }


def test_anthropic_tool_definitions_cache_point_marks_last_tool() -> None:
    model = AnthropicModel(model_id="claude-test", thinking_budget=None)

    prepared = model._prepare_request_sync(
        MessageRequest(
            messages=[_user("hi")],
            tools=[
                {
                    "type": "function",
                    "name": "first",
                    "description": "First tool",
                    "schema": {"type": "object", "properties": {}},
                },
                {
                    "type": "function",
                    "name": "second",
                    "description": "Second tool",
                    "schema": {"type": "object", "properties": {}},
                },
            ],
            cache_tool_definitions={"type": "ephemeral"},
        )
    )

    assert "cache_control" not in prepared["tools"][0]
    assert prepared["tools"][1]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_does_not_invent_cache_points_without_ir() -> None:
    model = AnthropicModel(model_id="claude-test", thinking_budget=None)

    prepared = model._prepare_request_sync(
        MessageRequest(
            messages=[_user("hi")],
            system=[{"type": "text", "text": "You are Hawi."}],
            tools=[
                {
                    "type": "function",
                    "name": "example",
                    "description": "Example tool",
                    "schema": {"type": "object", "properties": {}},
                }
            ],
        )
    )

    assert prepared["system"] == "You are Hawi."
    assert "cache_control" not in prepared["tools"][0]


def test_anthropic_cache_point_attaches_to_tool_result_block() -> None:
    model = AnthropicModel(model_id="claude-test", thinking_budget=None)
    message = _tool_result("call-1", "tool output")
    message["content"].append(
        {"type": "cache_point", "cache_point": {"type": "ephemeral"}}
    )

    prepared = model._prepare_request_sync(
        MessageRequest(messages=[_user("use tool"), _assistant_tool_calls(["call-1"]), message])
    )

    block = prepared["messages"][2]["content"][0]
    assert block["type"] == "tool_result"
    assert block["cache_control"] == {"type": "ephemeral"}


def test_anthropic_skips_unsigned_reasoning_blocks() -> None:
    model = AnthropicModel(model_id="claude-test", thinking_budget=None)

    prepared = model._prepare_request_sync(
        MessageRequest(messages=[_assistant_with_reasoning(signature=None)])
    )

    content = prepared["messages"][0]["content"]
    assert [block["type"] for block in content] == ["text"]
    assert "signature" not in content[0]


def test_anthropic_keeps_signed_reasoning_blocks() -> None:
    model = AnthropicModel(model_id="claude-test", thinking_budget=None)

    prepared = model._prepare_request_sync(
        MessageRequest(messages=[_assistant_with_reasoning(signature="sig-123")])
    )

    content = prepared["messages"][0]["content"]
    assert content[0] == {
        "type": "thinking",
        "thinking": "Need to reason.",
        "signature": "sig-123",
    }
    assert content[1] == {"type": "text", "text": "Done."}


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


def _assistant_with_reasoning(signature: str | None) -> Message:
    return {
        "role": "assistant",
        "content": [
            {
                "type": "reasoning",
                "reasoning": "Need to reason.",
                "signature": signature,
                "redacted_content": None,
            },
            {"type": "text", "text": "Done."},
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
