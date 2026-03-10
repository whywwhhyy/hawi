"""Tests for Model unified streaming interface.

Tests that Model.invoke() and Model.ainvoke() work correctly
with streaming=True and streaming=False parameters.
"""

import pytest
from typing import Iterator, cast

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
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )

    def _invoke_impl(self, request, event_callback=None) -> MessageResponse:
        """Mock non-streaming implementation."""
        if event_callback:
            # Simulate event sequence for non-streaming
            event_callback("model.stream_start", {"request_id": "test-123"})
            event_callback("model.content_block_start", {"request_id": "test-123", "block_index": 0, "type": "text"})
            event_callback("model.content_block_delta", {
                "request_id": "test-123",
                "block_index": 0,
                "delta": {
                    "type": "text_delta",
                    "text": "Hello, world!",
                    "index": 0,
                    "is_start": True,
                    "is_end": True,
                },
            })
            event_callback("model.content_block_stop", {
                "request_id": "test-123",
                "block_index": 0,
                "content": [{"type": "text", "text": "Hello, world!"}],
            })
            event_callback("model.stream_stop", {"request_id": "test-123", "stop_reason": "end_turn"})

        return MessageResponse(
            id="resp-123",
            content=[TextPart(type="text", text="Hello, world!")],
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
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
        """Test that invoke() calls event_callback in non-streaming mode."""
        model = MockModel()
        messages = [create_user_message("Hi")]
        events = []

        def event_callback(event_type: str, data: dict):
            events.append((event_type, data))

        result = model.invoke(messages=messages, streaming=False, event_callback=event_callback)

        assert isinstance(result, MessageResponse)
        assert len(events) > 0

        # Check event sequence
        event_types = [e[0] for e in events]
        assert "model.stream_start" in event_types
        assert "model.content_block_start" in event_types
        assert "model.content_block_delta" in event_types
        assert "model.content_block_stop" in event_types
        assert "model.stream_stop" in event_types

        # Check that delta has correct flags for non-streaming
        delta_event = next(e for e in events if e[0] == "model.content_block_delta")
        delta = delta_event[1]["delta"]
        assert delta["is_start"] is True
        assert delta["is_end"] is True


class TestAinvokeStreaming:
    """Tests for ainvoke() with streaming parameter."""

    @pytest.mark.asyncio
    async def test_ainvoke_streaming_false_returns_message_response(self):
        """Test that ainvoke(streaming=False) returns MessageResponse."""
        model = MockModel()
        messages = [create_user_message("Hi")]

        result = await model.ainvoke(messages=messages, streaming=False)

        assert isinstance(result, MessageResponse)
        assert result.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_ainvoke_streaming_true_returns_async_generator(self):
        """Test that ainvoke(streaming=True) returns AsyncGenerator."""
        model = MockModel()
        messages = [create_user_message("Hi")]

        result = await model.ainvoke(messages=messages, streaming=True)

        # Should be an async generator (has __aiter__ and __anext__)
        assert hasattr(result, "__aiter__")
        assert hasattr(result, "__anext__")

        # Collect all chunks using async for
        chunks = []
        async for chunk in result:
            chunks.append(chunk)

        # Should have text deltas + finish delta
        assert len(chunks) == 5  # 4 text chunks + 1 finish

        # Check text deltas
        text_chunks = [c for c in chunks if c["type"] == "text_delta"]
        assert len(text_chunks) == 4

        # Check streaming flags
        assert text_chunks[0]["is_start"] is True
        assert text_chunks[3]["is_end"] is True

    @pytest.mark.asyncio
    async def test_ainvoke_default_is_non_streaming(self):
        """Test that ainvoke() without streaming parameter defaults to non-streaming."""
        model = MockModel()
        messages = [create_user_message("Hi")]

        result = await model.ainvoke(messages=messages)

        assert isinstance(result, MessageResponse)

    @pytest.mark.asyncio
    async def test_ainvoke_with_event_callback(self):
        """Test that ainvoke() calls event_callback in non-streaming mode."""
        model = MockModel()
        messages = [create_user_message("Hi")]
        events = []

        def event_callback(event_type: str, data: dict):
            events.append((event_type, data))

        result = await model.ainvoke(messages=messages, streaming=False, event_callback=event_callback)

        assert isinstance(result, MessageResponse)
        assert len(events) > 0
        assert any(e[0] == "model.stream_start" for e in events)
        assert any(e[0] == "model.stream_stop" for e in events)


class TestStreamingConsistency:
    """Tests for consistency between streaming and non-streaming modes."""

    def test_event_sequence_consistency(self):
        """Test that both modes produce compatible event sequences."""
        model = MockModel()
        messages = [create_user_message("Hi")]

        # Collect non-streaming events
        non_streaming_events = []
        def non_streaming_callback(event_type: str, data: dict):
            non_streaming_events.append((event_type, data))

        model.invoke(messages=messages, streaming=False, event_callback=non_streaming_callback)

        # Collect streaming events (from iterator)
        streaming_events = []
        for chunk in model.invoke(messages=messages, streaming=True):
            if chunk["type"] == "text_delta":
                streaming_events.append(("delta", chunk["delta"]))
            elif chunk["type"] == "finish":
                streaming_events.append(("finish", chunk["stop_reason"]))

        # Both should produce complete content
        non_streaming_content = "".join(
            e[1]["delta"]["text"] for e in non_streaming_events
            if e[0] == "model.content_block_delta"
        )
        streaming_content = "".join(e[1] for e in streaming_events if e[0] == "delta")

        assert non_streaming_content == streaming_content == "Hello, world!"

    @pytest.mark.asyncio
    async def test_async_event_sequence_consistency(self):
        """Test that async streaming produces correct sequences."""
        model = MockModel()
        messages = [create_user_message("Hi")]

        # Non-streaming async
        non_streaming_events = []
        def non_streaming_callback(event_type: str, data: dict):
            non_streaming_events.append((event_type, data))

        await model.ainvoke(messages=messages, streaming=False, event_callback=non_streaming_callback)

        # Streaming async
        streaming_events = []
        async for chunk in await model.ainvoke(messages=messages, streaming=True):
            if chunk["type"] == "text_delta":
                streaming_events.append(("delta", chunk["delta"]))
            elif chunk["type"] == "finish":
                streaming_events.append(("finish", chunk["stop_reason"]))

        # Verify consistency
        non_streaming_content = "".join(
            e[1]["delta"]["text"] for e in non_streaming_events
            if e[0] == "model.content_block_delta"
        )
        streaming_content = "".join(e[1] for e in streaming_events if e[0] == "delta")

        assert non_streaming_content == streaming_content


class TestModelNotImplemented:
    """Tests for models that don't support streaming."""

    class NoStreamingModel(Model):
        """Model that doesn't implement streaming."""

        def __init__(self):
            self._model_id = "no-stream"

        @property
        def model_id(self) -> str:
            return self._model_id

        def _prepare_request_impl(self, request) -> dict:
            return {}

        def _parse_response_impl(self, response: dict) -> MessageResponse:
            return MessageResponse(id="no-stream-1", content=[], stop_reason="end_turn")

        def _invoke_impl(self, request, event_callback=None) -> MessageResponse:
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
        """Test that async streaming handles unsupported models gracefully."""
        model = self.NoStreamingModel()
        messages = [create_user_message("Hi")]

        result = await model.ainvoke(messages=messages, streaming=True)

        # When streaming is not supported, the generator yields no chunks
        # (the base class _astream_impl swallows thread errors)
        chunks = []
        async for chunk in result:
            chunks.append(chunk)
        assert len(chunks) == 0


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
    async def test_ainvoke_overload_resolution(self):
        """Test that type overloads resolve correctly for ainvoke."""
        model = MockModel()
        messages = [create_user_message("Hi")]

        # streaming=False should return MessageResponse
        result_false = await model.ainvoke(messages=messages, streaming=False)
        assert isinstance(result_false, MessageResponse)

        # streaming=True should return AsyncGenerator
        result_true = await model.ainvoke(messages=messages, streaming=True)
        # AsyncGenerator is both an async iterator and an async iterable
        assert hasattr(result_true, "__aiter__")
