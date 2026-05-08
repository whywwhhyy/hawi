import httpx
from openai import BadRequestError

from hawi.errors import ContextLengthError
from hawi.models.openai._model import _convert_openai_error
from hawi.models.openai._streaming import StreamProcessor


def test_bad_request_context_length_error_is_structured():
    message = (
        "This model's maximum context length is 1048576 tokens. "
        "However, you requested 1916559 tokens "
        "(1916559 in the messages, 0 in the completion). "
        "Please reduce the length of the messages or completion."
    )
    body = {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": None,
            "code": "invalid_request_error",
        }
    }
    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    response = httpx.Response(400, request=request, json=body)
    error = BadRequestError(
        f"Error code: 400 - {body}",
        response=response,
        body=body,
    )

    converted = _convert_openai_error(error)

    assert isinstance(converted, ContextLengthError)
    assert converted.error_type == "context_length"
    assert converted.max_context_tokens == 1_048_576
    assert converted.requested_tokens == 1_916_559
    assert converted.message_tokens == 1_916_559
    assert converted.completion_tokens == 0


def test_stream_processor_keeps_parallel_tool_calls_separate():
    processor = StreamProcessor()

    first_parts = list(processor.process_chunk({
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-a",
                            "function": {
                                "name": "WebPlugin__fetch",
                                "arguments": '{"url":"https://example.com"}',
                            },
                        },
                        {
                            "index": 1,
                            "id": "call-b",
                            "function": {
                                "name": "FileSystemPlugin__read_file",
                                "arguments": '{"file_path":"README.md"}',
                            },
                        },
                    ]
                },
                "finish_reason": None,
            }
        ]
    }))

    starts = [part for part in first_parts if part["type"] == "tool_call_delta" and part["is_start"]]
    deltas = [
        part
        for part in first_parts
        if part["type"] == "tool_call_delta" and part["arguments_delta"]
    ]
    assert [(part["index"], part["id"], part["name"]) for part in starts] == [
        (0, "call-a", "WebPlugin__fetch"),
        (1, "call-b", "FileSystemPlugin__read_file"),
    ]
    assert [(part["index"], part["arguments_delta"]) for part in deltas] == [
        (0, '{"url":"https://example.com"}'),
        (1, '{"file_path":"README.md"}'),
    ]

    final_parts = list(processor.process_chunk({
        "choices": [
            {
                "delta": {},
                "finish_reason": "tool_calls",
            }
        ]
    }))
    stops = [part for part in final_parts if part["type"] == "tool_call_delta" and part["is_end"]]

    assert [(part["index"], part["id"], part["name"]) for part in stops] == [
        (0, "call-a", "WebPlugin__fetch"),
        (1, "call-b", "FileSystemPlugin__read_file"),
    ]
    assert final_parts[-1]["type"] == "finish"
    assert final_parts[-1]["stop_reason"] == "tool_use"


def test_stream_processor_routes_fragmented_parallel_tool_arguments_by_index():
    processor = StreamProcessor()

    list(processor.process_chunk({
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {"index": 0, "id": "call-a", "function": {"name": "fetch"}},
                        {"index": 1, "id": "call-b", "function": {"name": "read"}},
                    ]
                },
                "finish_reason": None,
            }
        ]
    }))
    parts = list(processor.process_chunk({
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {"index": 0, "function": {"arguments": '{"url":'}},
                        {"index": 1, "function": {"arguments": '{"path":'}},
                    ]
                },
                "finish_reason": None,
            }
        ]
    }))

    deltas = [
        part
        for part in parts
        if part["type"] == "tool_call_delta" and part["arguments_delta"]
    ]
    assert [(part["index"], part["arguments_delta"]) for part in deltas] == [
        (0, '{"url":'),
        (1, '{"path":'),
    ]
