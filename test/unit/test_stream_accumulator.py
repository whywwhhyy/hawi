"""Unit tests for StreamBlockAccumulator.

Tests cover:
- Basic block lifecycle (text / tool_use)
- Validation: duplicate start raises ValueError; late chunks silently dropped
- Sorting: out-of-order blocks buffered and flushed in idx order
"""

import pytest

from hawi.agent.stream_accumulator import StreamBlockAccumulator
from hawi.models.message import DeltaTextPart, DeltaThinkingPart, DeltaToolCallPart

REQUEST_ID = "req-test"


# ---------------------------------------------------------------------------
# Chunk builders
# ---------------------------------------------------------------------------

def text_chunk(
    idx: int,
    delta: str = "",
    *,
    is_start: bool = False,
    is_end: bool = False,
) -> DeltaTextPart:
    return DeltaTextPart(
        type="text_delta",
        index=idx,
        delta=delta,
        is_start=is_start,
        is_end=is_end,
    )


def thinking_chunk(
    idx: int,
    delta: str = "",
    *,
    is_start: bool = False,
    is_end: bool = False,
) -> DeltaThinkingPart:
    return DeltaThinkingPart(
        type="reasoning_delta",
        index=idx,
        delta=delta,
        is_start=is_start,
        is_end=is_end,
    )


def tool_chunk(
    idx: int,
    *,
    tool_id: str = "",
    name: str = "",
    args: str = "",
    is_start: bool = False,
    is_end: bool = False,
) -> DeltaToolCallPart:
    return DeltaToolCallPart(
        type="tool_call_delta",
        index=idx,
        id=tool_id,
        name=name,
        arguments_delta=args,
        is_start=is_start,
        is_end=is_end,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parts(results: list) -> list:
    """Extract ContentPart from each result tuple, skipping None."""
    return [r[0] for r in results if r[0] is not None]


def all_event_types(results: list) -> list[str]:
    """Flatten all event types across all result tuples."""
    return [e.type for _, events in results for e in events]


# ---------------------------------------------------------------------------
# Basic lifecycle
# ---------------------------------------------------------------------------

class TestBasicLifecycle:

    def test_text_block_returns_text_part(self):
        acc = StreamBlockAccumulator.create_text_handler()

        r1 = acc.handle(text_chunk(0, "Hello", is_start=True), REQUEST_ID)
        r2 = acc.handle(text_chunk(0, " world"), REQUEST_ID)
        r3 = acc.handle(text_chunk(0, "!", is_end=True), REQUEST_ID)

        completed = parts(r1 + r2 + r3)
        assert len(completed) == 1
        assert completed[0]["type"] == "text"
        assert completed[0]["text"] == "Hello world!"

    def test_tool_block_returns_tool_call_part(self):
        acc = StreamBlockAccumulator.create_tool_handler()

        r1 = acc.handle(tool_chunk(0, tool_id="tc1", name="search", is_start=True), REQUEST_ID)
        r2 = acc.handle(tool_chunk(0, args='{"q": "py'), REQUEST_ID)
        r3 = acc.handle(tool_chunk(0, args='thon"}', is_end=True), REQUEST_ID)

        completed = parts(r1 + r2 + r3)
        assert len(completed) == 1
        p = completed[0]
        assert p["type"] == "tool_call"
        assert p["name"] == "search"
        assert p["arguments"] == {"q": "python"}

    def test_reasoning_block_returns_reasoning_part(self):
        acc = StreamBlockAccumulator.create_thinking_handler()

        r1 = acc.handle(thinking_chunk(0, "Let me think", is_start=True), REQUEST_ID)
        r2 = acc.handle(thinking_chunk(0, "...", is_end=True), REQUEST_ID)

        completed = parts(r1 + r2)
        assert len(completed) == 1
        assert completed[0]["type"] == "reasoning"

    def test_empty_reasoning_block_returns_reasoning_part(self):
        acc = StreamBlockAccumulator.create_thinking_handler()

        r1 = acc.handle(thinking_chunk(0, "", is_start=True), REQUEST_ID)
        r2 = acc.handle(thinking_chunk(0, "", is_end=True), REQUEST_ID)

        completed = parts(r1 + r2)
        assert len(completed) == 1
        assert completed[0]["type"] == "reasoning"
        assert completed[0].get("reasoning") == ""

    def test_empty_text_block_returns_none(self):
        acc = StreamBlockAccumulator.create_text_handler()

        r1 = acc.handle(text_chunk(0, "", is_start=True), REQUEST_ID)
        r2 = acc.handle(text_chunk(0, "   ", is_end=True), REQUEST_ID)  # whitespace only

        assert parts(r1 + r2) == []

    def test_handle_returns_list(self):
        acc = StreamBlockAccumulator.create_text_handler()
        result = acc.handle(text_chunk(0, "hi", is_start=True), REQUEST_ID)
        assert isinstance(result, list)

    def test_events_emitted_for_full_block(self):
        acc = StreamBlockAccumulator.create_text_handler()

        r1 = acc.handle(text_chunk(0, "x", is_start=True), REQUEST_ID)
        r2 = acc.handle(text_chunk(0, "y", is_end=True), REQUEST_ID)

        types = all_event_types(r1 + r2)
        assert "model.content_block_start" in types
        assert "model.content_block_delta" in types
        assert "model.content_block_stop" in types

    def test_sequential_blocks_processed_correctly(self):
        """Two text blocks processed one after the other."""
        acc = StreamBlockAccumulator.create_text_handler()

        # Block 0
        acc.handle(text_chunk(0, "foo", is_start=True), REQUEST_ID)
        r_end0 = acc.handle(text_chunk(0, is_end=True), REQUEST_ID)

        # Block 1 (different idx)
        acc.handle(text_chunk(1, "bar", is_start=True), REQUEST_ID)
        r_end1 = acc.handle(text_chunk(1, is_end=True), REQUEST_ID)

        p0 = parts(r_end0)
        p1 = parts(r_end1)
        assert p0[0]["text"] == "foo"
        assert p1[0]["text"] == "bar"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:

    def test_duplicate_start_raises(self):
        """Starting the same block idx again after it finished is a protocol error."""
        acc = StreamBlockAccumulator.create_text_handler()

        acc.handle(text_chunk(0, "hello", is_start=True), REQUEST_ID)
        acc.handle(text_chunk(0, is_end=True), REQUEST_ID)

        with pytest.raises(ValueError, match="already completed"):
            acc.handle(text_chunk(0, "oops", is_start=True), REQUEST_ID)

    def test_late_delta_silently_dropped(self):
        """Delta arriving after is_end for the same block (乱序) is discarded."""
        acc = StreamBlockAccumulator.create_text_handler()

        acc.handle(text_chunk(0, "hello", is_start=True), REQUEST_ID)
        acc.handle(text_chunk(0, is_end=True), REQUEST_ID)

        # Late delta — block already finished
        result = acc.handle(text_chunk(0, " world"), REQUEST_ID)
        assert result == []

    def test_late_end_silently_dropped(self):
        """is_end arriving again after block finished is discarded."""
        acc = StreamBlockAccumulator.create_text_handler()

        acc.handle(text_chunk(0, "hi", is_start=True), REQUEST_ID)
        acc.handle(text_chunk(0, is_end=True), REQUEST_ID)

        result = acc.handle(text_chunk(0, is_end=True), REQUEST_ID)
        assert result == []

    def test_duplicate_start_does_not_affect_next_block(self):
        """After a duplicate-start error, a legitimately new block (new idx) still works."""
        acc = StreamBlockAccumulator.create_text_handler()

        acc.handle(text_chunk(0, "a", is_start=True), REQUEST_ID)
        acc.handle(text_chunk(0, is_end=True), REQUEST_ID)

        with pytest.raises(ValueError):
            acc.handle(text_chunk(0, is_start=True), REQUEST_ID)

        # Block with a new idx should work fine
        acc.handle(text_chunk(1, "b", is_start=True), REQUEST_ID)
        r = acc.handle(text_chunk(1, is_end=True), REQUEST_ID)
        assert parts(r)[0]["text"] == "b"


# ---------------------------------------------------------------------------
# Sorting (cross-block out-of-order)
# ---------------------------------------------------------------------------

class TestSorting:

    def test_chunk_for_other_idx_buffered_while_block_open(self):
        """Chunk belonging to a different idx is buffered, not processed immediately."""
        acc = StreamBlockAccumulator.create_text_handler()

        acc.handle(text_chunk(0, "A", is_start=True), REQUEST_ID)

        # Chunk for idx=1 arrives while idx=0 is still open
        result = acc.handle(text_chunk(1, "B", is_start=True), REQUEST_ID)
        assert result == []
        assert 1 in acc._pending

    def test_buffered_block_flushed_after_current_ends(self):
        """When block 0 ends, buffered block 1 is processed and returned."""
        acc = StreamBlockAccumulator.create_text_handler()

        acc.handle(text_chunk(0, "A", is_start=True), REQUEST_ID)
        acc.handle(text_chunk(1, "B", is_start=True), REQUEST_ID)   # buffered
        acc.handle(text_chunk(1, is_end=True), REQUEST_ID)           # buffered

        # Ending block 0 should flush block 1
        results = acc.handle(text_chunk(0, is_end=True), REQUEST_ID)
        completed = parts(results)

        # results[0] is the part for block 0 ("A"), results[1] for block 1 ("B")
        assert len(completed) == 2
        assert completed[0]["text"] == "A"
        assert completed[1]["text"] == "B"

    def test_multiple_buffered_blocks_flushed_in_idx_order(self):
        """Blocks 2 and 1 both buffered; flushed as 1 then 2 (sorted)."""
        acc = StreamBlockAccumulator.create_text_handler()

        acc.handle(text_chunk(0, "A", is_start=True), REQUEST_ID)

        # idx=2 arrives before idx=1
        acc.handle(text_chunk(2, "C", is_start=True), REQUEST_ID)
        acc.handle(text_chunk(2, is_end=True), REQUEST_ID)
        acc.handle(text_chunk(1, "B", is_start=True), REQUEST_ID)
        acc.handle(text_chunk(1, is_end=True), REQUEST_ID)

        results = acc.handle(text_chunk(0, is_end=True), REQUEST_ID)
        completed = parts(results)

        assert len(completed) == 3
        assert completed[0]["text"] == "A"   # block 0
        assert completed[1]["text"] == "B"   # block 1 (sorted before 2)
        assert completed[2]["text"] == "C"   # block 2

    def test_pending_cleared_after_flush(self):
        """After flush, _pending is empty."""
        acc = StreamBlockAccumulator.create_text_handler()

        acc.handle(text_chunk(0, "A", is_start=True), REQUEST_ID)
        acc.handle(text_chunk(1, "B", is_start=True), REQUEST_ID)
        acc.handle(text_chunk(1, is_end=True), REQUEST_ID)
        acc.handle(text_chunk(0, is_end=True), REQUEST_ID)

        assert acc._pending == {}

    def test_is_end_early_then_late_deltas_dropped(self):
        """Within a block, if is_end arrives before some deltas (out-of-order),
        subsequent late deltas for that finished idx are silently discarded."""
        acc = StreamBlockAccumulator.create_text_handler()

        acc.handle(text_chunk(0, "hello", is_start=True), REQUEST_ID)
        # is_end arrives "early" (out-of-order — before a stray delta)
        acc.handle(text_chunk(0, is_end=True), REQUEST_ID)

        # Stray delta that was delayed in transit
        result = acc.handle(text_chunk(0, " world"), REQUEST_ID)
        assert result == []

    def test_tool_blocks_out_of_order(self):
        """Two tool_use blocks arrive with their chunks interleaved."""
        acc = StreamBlockAccumulator.create_tool_handler()

        # Block 0 starts
        acc.handle(tool_chunk(0, tool_id="t0", name="foo", is_start=True), REQUEST_ID)

        # Block 1 starts and ends while block 0 is still open
        acc.handle(tool_chunk(1, tool_id="t1", name="bar", is_start=True), REQUEST_ID)
        acc.handle(tool_chunk(1, args='{"x":1}', is_end=True), REQUEST_ID)

        # Block 0 ends — should also return flushed block 1
        results = acc.handle(tool_chunk(0, args='{"y":2}', is_end=True), REQUEST_ID)
        completed = parts(results)

        assert len(completed) == 2
        assert completed[0]["name"] == "foo"
        assert completed[0]["arguments"] == {"y": 2}
        assert completed[1]["name"] == "bar"
        assert completed[1]["arguments"] == {"x": 1}
