"""Tests for Model unified streaming interface.

Tests that Model.invoke() and Model.ainvoke() work correctly
with streaming=True and streaming=False parameters.
"""

import pytest
from typing import Iterator, cast, AsyncGenerator

from hawi.models import Model
from hawi.models.message import (
    Message,
    MessageResponse,
    DeltaPart,
    TextPart,
    DeltaFinishPart,
)
from hawi.errors import ModelError
from hawi.models.message import TokenUsage
from hawi.events.event import Event
from hawi.events.model_events import (
    ModelStreamStartEvent,
    ModelStreamStopEvent,
    ModelContentBlockStartEvent,
    ModelContentBlockStopEvent,
)


# Helper to create valid Message objects
def create_user_message(text: str) -> Message:
    return {
        "role": "user",
        "content": [{"type": "text", "text": text}],
        "name": None,
        "metadata": None,
    }


class MockModel(Model):
    """Mock model for testing the unified interface."""

    default_steer_merge_mode = "tool_result_assistant_template_and_user_message"

    def __init__(self, model_id: str = "test-model"):
        self._model_id = model_id
        self._params = {}

    @property
    def model_id(self) -> str:
        return self._model_id

    def _get_params(self) -> dict:
        return self._params

    def _prepare_request_impl(self, request) -> dict:
        return {"model": self._model_id, "messages": []}

    def _parse_response_impl(self, response: dict) -> MessageResponse:
        return MessageResponse(
            id="resp-123",
            content=[TextPart(type="text", text=response.get("text", "Hello"))],
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=10, output_tokens=5, cache_write_tokens=None, cache_read_tokens=None),
        )

    def _invoke_impl(self, request) -> MessageResponse:
        """Mock non-streaming implementation."""
        return MessageResponse(
            id="resp-123",
            content=[TextPart(type="text", text="Hello, world!")],
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=10, output_tokens=5, cache_write_tokens=None, cache_read_tokens=None),
        )

    async def _ainvoke_impl(self, request) -> AsyncGenerator[DeltaPart | Event, None]:  # type: ignore[override]
        """Mock async non-streaming implementation."""
        import time
        request_id = f"test-{int(time.time() * 1000)}"

        # Yield stream_start
        yield ModelStreamStartEvent.create(request_id=request_id)

        # Yield content block
        yield ModelContentBlockStartEvent.create(
            request_id=request_id,
            block_index=0,
            block_type="text",
        )

        yield cast(DeltaPart, {
            "type": "text_delta",
            "index": 0,
            "delta": "Hello, world!",
            "is_start": True,
            "is_end": True,
        })

        yield ModelContentBlockStopEvent.create(
            request_id=request_id,
            block_index=0,
            content=[{"type": "text", "text": "Hello, world!"}],
        )

        yield ModelStreamStopEvent.create(
            request_id=request_id,
            stop_reason="end_turn",
        )

    async def _astream_impl(self, request) -> AsyncGenerator[DeltaPart | Event, None]:  # type: ignore[override]
        """Mock async streaming implementation."""
        import time
        request_id = f"test-{int(time.time() * 1000)}"

        # Yield stream_start
        yield ModelStreamStartEvent.create(request_id=request_id)

        # Stream in small chunks
        chunks = ["Hello", ", ", "world", "!"]
        for i, chunk in enumerate(chunks):
            is_start = i == 0
            is_end = i == len(chunks) - 1

            if is_start:
                yield ModelContentBlockStartEvent.create(
                    request_id=request_id,
                    block_index=0,
                    block_type="text",
                )

            yield cast(DeltaPart, {
                "type": "text_delta",
                "delta": chunk,
                "index": 0,
                "is_start": is_start,
                "is_end": is_end,
            })

            if is_end:
                yield ModelContentBlockStopEvent.create(
                    request_id=request_id,
                    block_index=0,
                    content=[{"type": "text", "text": "Hello, world!"}],
                )

        yield ModelStreamStopEvent.create(
            request_id=request_id,
            stop_reason="end_turn",
        )

    def _stream_impl(self, request) -> Iterator[DeltaPart]:
        """Mock streaming implementation."""
        # Stream in small chunks
        chunks = ["Hello", ", ", "world", "!"]
        for i, chunk in enumerate(chunks):
            is_start = i == 0
            is_end = i == len(chunks) - 1
            yield cast(DeltaPart, {
                "type": "text_delta",
                "delta": chunk,
                "index": 0,
                "is_start": is_start,
                "is_end": is_end,
            })
        yield cast(DeltaPart, {"type": "finish", "stop_reason": "end_turn", "usage": None})


class TestInvokeStreaming:
    """Tests for invoke() with streaming parameter."""

    def test_invoke_streaming_false_returns_message_response(self):
        """Test that invoke(streaming=False) returns MessageResponse."""
        model = MockModel()
        messages = [create_user_message("Hi")]

        result = model.invoke(messages=messages, streaming=False)

        assert isinstance(result, MessageResponse)
        assert result.stop_reason == "end_turn"
        assert len(result.content) == 1
        assert result.content[0]["type"] == "text"
        assert result.content[0]["text"] == "Hello, world!"

    def test_invoke_streaming_true_returns_iterator(self):
        """Test that invoke(streaming=True) returns Iterator[DeltaPart]."""
        model = MockModel()
        messages = [create_user_message("Hi")]

        result = model.invoke(messages=messages, streaming=True)

        # Should be an iterator
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")

        # Collect all chunks
        chunks = list(result)

        # Should have text deltas + finish delta
        assert len(chunks) == 5  # 4 text chunks + 1 finish

        # Check text deltas
        text_chunks = [c for c in chunks if c["type"] == "text_delta"]
        assert len(text_chunks) == 4
        assert text_chunks[0]["delta"] == "Hello"
        assert text_chunks[1]["delta"] == ", "
        assert text_chunks[2]["delta"] == "world"
        assert text_chunks[3]["delta"] == "!"

        # Check streaming flags
        assert text_chunks[0]["is_start"] is True
        assert text_chunks[0]["is_end"] is False
        assert text_chunks[3]["is_start"] is False
        assert text_chunks[3]["is_end"] is True

        # Check finish delta
        finish_chunks = [c for c in chunks if c["type"] == "finish"]
        assert len(finish_chunks) == 1
        assert finish_chunks[0]["stop_reason"] == "end_turn"

    def test_invoke_default_is_non_streaming(self):
        """Test that invoke() without streaming parameter defaults to non-streaming."""
        model = MockModel()
        messages = [create_user_message("Hi")]

        result = model.invoke(messages=messages)

        assert isinstance(result, MessageResponse)

    def test_invoke_with_event_callback(self):
        """Test that invoke() works without event_callback in non-streaming mode.

        Note: event_callback parameter has been removed in the refactor.
        Now Model always returns events through the generator interface.
        """
        model = MockModel()
        messages = [create_user_message("Hi")]

        result = model.invoke(messages=messages, streaming=False)

        assert isinstance(result, MessageResponse)
        assert result.content[0]["type"] == "text"
        assert result.content[0]["text"] == "Hello, world!"


class TestAinvokeStreaming:
    """Tests for ainvoke() with streaming parameter.

    Note: After refactor, ainvoke() always returns AsyncGenerator[DeltaPart | Event, None].
    Both streaming=True and streaming=False return async generators.
    """

    @pytest.mark.asyncio
    async def test_ainvoke_streaming_false_returns_async_generator(self):
        """Test that ainvoke(streaming=False) returns AsyncGenerator with events."""
        model = MockModel()
        messages = [create_user_message("Hi")]

        result = model.ainvoke(messages=messages, streaming=False)

        # Should be an async generator (has __aiter__ and __anext__)
        assert hasattr(result, "__aiter__")
        assert hasattr(result, "__anext__")

        # Collect all events using async for
        events = []
        async for event in result:
            events.append(event)

        # Should have events: stream_start, block_start, delta, block_stop, stream_stop
        assert len(events) == 5

        # Check event types (handle both DeltaPart dict access and ModelEvent attribute access)
        def get_type(e):
            return e["type"] if isinstance(e, dict) else e.type

        assert get_type(events[0]) == "model.stream_start"
        assert get_type(events[1]) == "model.content_block_start"
        assert get_type(events[2]) == "text_delta"
        assert get_type(events[3]) == "model.content_block_stop"
        assert get_type(events[4]) == "model.stream_stop"

    @pytest.mark.asyncio
    async def test_ainvoke_streaming_true_returns_async_generator(self):
        """Test that ainvoke(streaming=True) returns AsyncGenerator."""
        model = MockModel()
        messages = [create_user_message("Hi")]

        result = model.ainvoke(messages=messages, streaming=True)

        # Should be an async generator (has __aiter__ and __anext__)
        assert hasattr(result, "__aiter__")
        assert hasattr(result, "__anext__")

        # Collect all chunks using async for
        chunks = []
        async for chunk in result:
            chunks.append(chunk)

        # Should have events: stream_start, block_start, 4 text deltas, block_stop, stream_stop = 8
        assert len(chunks) == 8

        # Check text deltas (filter out ModelEvent objects)
        text_chunks = [c for c in chunks if isinstance(c, dict) and c.get("type") == "text_delta"]
        assert len(text_chunks) == 4

        # Check streaming flags
        assert text_chunks[0]["is_start"] is True
        assert text_chunks[3]["is_end"] is True

    @pytest.mark.asyncio
    async def test_ainvoke_default_is_async_generator(self):
        """Test that ainvoke() without streaming parameter defaults to async generator."""
        model = MockModel()
        messages = [create_user_message("Hi")]

        result = model.ainvoke(messages=messages)

        # Should be an async generator
        assert hasattr(result, "__aiter__")

        # Collect events
        events = []
        async for event in result:
            events.append(event)

        # Should have events
        assert len(events) > 0

        def get_type(e):
            return e["type"] if isinstance(e, dict) else e.type
        assert get_type(events[0]) == "model.stream_start"


class TestStreamingConsistency:
    """Tests for consistency between streaming and non-streaming modes."""

    def test_event_sequence_consistency(self):
        """Test that both modes produce compatible event sequences.

        Note: After refactor, invoke() still returns MessageResponse for non-streaming,
        but the internal implementation has changed.
        """
        model = MockModel()
        messages = [create_user_message("Hi")]

        # Non-streaming returns MessageResponse
        result = model.invoke(messages=messages, streaming=False)
        assert isinstance(result, MessageResponse)
        assert result.content[0]["type"] == "text"
        assert result.content[0]["text"] == "Hello, world!"

        # Collect streaming events (from iterator)
        streaming_events = []
        for chunk in model.invoke(messages=messages, streaming=True):
            if isinstance(chunk, dict):
                if chunk.get("type") == "text_delta":
                    from hawi.models.message import DeltaTextPart
                    streaming_events.append(cast(DeltaTextPart, chunk)["delta"])

        # Both should produce complete content
        streaming_content = "".join(streaming_events)
        assert streaming_content == "Hello, world!"

    @pytest.mark.asyncio
    async def test_async_event_sequence_consistency(self):
        """Test that async streaming produces correct sequences."""
        model = MockModel()
        messages = [create_user_message("Hi")]

        # Non-streaming async (now also returns async generator)
        non_streaming_events = []
        async for event in model.ainvoke(messages=messages, streaming=False):
            non_streaming_events.append(event)

        # Streaming async
        streaming_events = []
        async for chunk in model.ainvoke(messages=messages, streaming=True):
            streaming_events.append(chunk)

        # Verify we get events from both modes
        assert len(non_streaming_events) > 0
        assert len(streaming_events) > 0

        # Helper to get type
        def get_type(e):
            return e["type"] if isinstance(e, dict) else e.type

        # Non-streaming should have ModelEvent types
        assert get_type(non_streaming_events[0]) == "model.stream_start"

        # Streaming should have DeltaPart types
        text_deltas = [e for e in streaming_events if isinstance(e, dict) and e.get("type") == "text_delta"]
        assert len(text_deltas) > 0


class TestModelNotImplemented:
    """Tests for models that don't support streaming."""

    class NoStreamingModel(Model):
        """Model that doesn't implement streaming."""

        default_steer_merge_mode = "tool_result_assistant_template_and_user_message"

        def __init__(self):
            self._model_id = "no-stream"

        @property
        def model_id(self) -> str:
            return self._model_id

        def _prepare_request_impl(self, request) -> dict:
            return {}

        def _parse_response_impl(self, response: dict) -> MessageResponse:
            return MessageResponse(id="no-stream-1", content=[], stop_reason="end_turn")

        def _invoke_impl(self, request) -> MessageResponse:
            return MessageResponse(id="no-stream-1", content=[], stop_reason="end_turn")

        # _stream_impl not overridden - should raise NotImplementedError

    def test_streaming_raises_not_implemented(self):
        """Test that streaming raises NotImplementedError when not supported."""
        model = self.NoStreamingModel()
        messages = [create_user_message("Hi")]

        with pytest.raises(NotImplementedError):
            list(model.invoke(messages=messages, streaming=True))

    @pytest.mark.asyncio
    async def test_async_streaming_not_supported(self):
        """Test that async streaming raises NotImplementedError when not supported."""
        model = self.NoStreamingModel()
        messages = [create_user_message("Hi")]

        # When streaming is not supported, the generator should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            async for chunk in model.ainvoke(messages=messages, streaming=True):
                pass


class TestTypeAnnotations:
    """Tests that type annotations are correct."""

    def test_invoke_overload_resolution(self):
        """Test that type overloads resolve correctly for invoke."""
        model = MockModel()
        messages = [create_user_message("Hi")]

        # streaming=False should return MessageResponse
        result_false = model.invoke(messages=messages, streaming=False)
        assert isinstance(result_false, MessageResponse)

        # streaming=True should return Iterator
        result_true = model.invoke(messages=messages, streaming=True)
        assert isinstance(result_true, Iterator)

    @pytest.mark.asyncio
    async def test_ainvoke_returns_async_generator(self):
        """Test that ainvoke always returns AsyncGenerator.

        Note: After refactor, ainvoke() always returns AsyncGenerator[DeltaPart | Event, None]
        regardless of streaming parameter.
        """
        model = MockModel()
        messages = [create_user_message("Hi")]

        # Both streaming=True and streaming=False return AsyncGenerator
        result_false = model.ainvoke(messages=messages, streaming=False)
        result_true = model.ainvoke(messages=messages, streaming=True)

        # Both should be async generators (have __aiter__)
        assert hasattr(result_false, "__aiter__")
        assert hasattr(result_true, "__aiter__")

        # Both should produce events when iterated
        events_false = []
        async for event in result_false:
            events_false.append(event)

        events_true = []
        async for event in result_true:
            events_true.append(event)

        assert len(events_false) > 0
        assert len(events_true) > 0
