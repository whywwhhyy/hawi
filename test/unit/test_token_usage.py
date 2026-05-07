from hawi.models.openai._streaming import StreamProcessor
from hawi.models.usage import (
    merge_token_usage,
    normalize_anthropic_usage,
    normalize_openai_usage,
)


def test_openai_usage_normalizes_cache_and_reasoning_details() -> None:
    usage = normalize_openai_usage(
        {
            "prompt_tokens": 100,
            "completion_tokens": 30,
            "total_tokens": 130,
            "prompt_tokens_details": {"cached_tokens": 40, "audio_tokens": 3},
            "completion_tokens_details": {
                "reasoning_tokens": 12,
                "audio_tokens": 4,
                "accepted_prediction_tokens": 5,
                "rejected_prediction_tokens": 6,
            },
        }
    )

    assert usage is not None
    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 30
    assert usage.get("total_tokens") == 130
    assert usage.get("cache_read_tokens") == 40
    assert usage.get("reasoning_tokens") == 12
    assert usage.get("input_audio_tokens") == 3
    assert usage.get("output_audio_tokens") == 4
    assert usage.get("accepted_prediction_tokens") == 5
    assert usage.get("rejected_prediction_tokens") == 6


def test_deepseek_openai_usage_normalizes_cache_hit_and_miss_tokens() -> None:
    usage = normalize_openai_usage(
        {
            "prompt_tokens": 100,
            "completion_tokens": 30,
            "total_tokens": 130,
            "prompt_cache_hit_tokens": 80,
            "prompt_cache_miss_tokens": 20,
            "completion_tokens_details": {"reasoning_tokens": 11},
        }
    )

    assert usage is not None
    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 30
    assert usage.get("cache_read_tokens") == 80
    assert usage.get("cache_miss_tokens") == 20
    assert usage.get("reasoning_tokens") == 11


def test_openai_stream_processor_reads_usage_from_deepseek_final_choice_chunk() -> None:
    processor = StreamProcessor(expect_usage=True)

    parts = list(
        processor.process_chunk(
            {
                "choices": [
                    {
                        "delta": {},
                        "finish_reason": "stop",
                        "index": 0,
                    }
                ],
                "usage": {
                    "prompt_tokens": 17,
                    "completion_tokens": 9,
                    "total_tokens": 26,
                    "completion_tokens_details": {"reasoning_tokens": 2},
                },
            }
        )
    )

    assert parts[-1]["type"] == "finish"
    usage = parts[-1]["usage"]
    assert usage is not None
    assert usage["input_tokens"] == 17
    assert usage["output_tokens"] == 9
    assert usage.get("total_tokens") == 26
    assert usage.get("reasoning_tokens") == 2


def test_openai_stream_processor_waits_for_usage_only_chunk_when_expected() -> None:
    processor = StreamProcessor(expect_usage=True)

    finish_parts = list(
        processor.process_chunk(
            {
                "choices": [
                    {
                        "delta": {},
                        "finish_reason": "stop",
                        "index": 0,
                    }
                ],
            }
        )
    )
    assert finish_parts == []

    usage_parts = list(
        processor.process_chunk(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 7,
                    "total_tokens": 12,
                },
            }
        )
    )

    assert usage_parts[-1]["type"] == "finish"
    assert usage_parts[-1]["usage"] is not None
    assert usage_parts[-1]["usage"].get("total_tokens") == 12


def test_anthropic_total_includes_cache_input_categories() -> None:
    usage = normalize_anthropic_usage(
        {
            "input_tokens": 10,
            "cache_creation_input_tokens": 20,
            "cache_read_input_tokens": 30,
            "output_tokens": 40,
        }
    )

    assert usage is not None
    assert usage["input_tokens"] == 10
    assert usage.get("cache_write_tokens") == 20
    assert usage.get("cache_read_tokens") == 30
    assert usage["output_tokens"] == 40
    assert usage.get("total_tokens") == 100


def test_merge_token_usage_sums_all_known_detail_fields() -> None:
    merged = merge_token_usage(
        {
            "input_tokens": 1,
            "output_tokens": 2,
            "total_tokens": 3,
            "cache_read_tokens": 4,
            "reasoning_tokens": 5,
        },
        {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
            "cache_read_tokens": 40,
            "reasoning_tokens": 50,
        },
    )

    assert merged is not None
    assert merged["input_tokens"] == 11
    assert merged["output_tokens"] == 22
    assert merged.get("total_tokens") == 33
    assert merged.get("cache_read_tokens") == 44
    assert merged.get("reasoning_tokens") == 55
