from hawi.models.openai._streaming import StreamProcessor


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
