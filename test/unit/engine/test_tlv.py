"""Tests for hawi.engine.tlv: TLV framing primitives."""

from __future__ import annotations

import asyncio

import pytest

from hawi.engine.tlv import (
    DEFAULT_MAX_FRAME_SIZE,
    TYPE_JSON_FRAME,
    FrameTooLargeError,
    UnexpectedEOFError,
    encode_frame,
    read_frame,
)


def test_encode_frame_json_layout():
    body = b'{"hello":1}'
    framed = encode_frame(TYPE_JSON_FRAME, body)
    assert framed[0:1] == b"\x01"
    assert framed[1:5] == len(body).to_bytes(4, "big")
    assert framed[5:] == body


def test_encode_frame_empty_body_is_legal():
    framed = encode_frame(TYPE_JSON_FRAME, b"")
    assert framed == b"\x01\x00\x00\x00\x00"


async def _bytes_reader(data: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    reader.feed_eof()
    return reader


async def test_read_frame_json_roundtrip():
    body = b'{"hello":1}'
    framed = encode_frame(TYPE_JSON_FRAME, body)
    reader = await _bytes_reader(framed)
    type_byte, value = await read_frame(reader)
    assert type_byte == TYPE_JSON_FRAME
    assert value == body


async def test_read_frame_handles_two_back_to_back():
    body1 = b'{"a":1}'
    body2 = b'{"b":2}'
    reader = await _bytes_reader(encode_frame(TYPE_JSON_FRAME, body1) + encode_frame(TYPE_JSON_FRAME, body2))
    t1, v1 = await read_frame(reader)
    t2, v2 = await read_frame(reader)
    assert (t1, v1) == (TYPE_JSON_FRAME, body1)
    assert (t2, v2) == (TYPE_JSON_FRAME, body2)


async def test_read_frame_returns_none_on_clean_eof():
    reader = await _bytes_reader(b"")
    result = await read_frame(reader)
    assert result is None


async def test_read_frame_raises_on_truncated_header():
    reader = await _bytes_reader(b"\x01\x00\x00")  # only 3 of 5 header bytes
    with pytest.raises(UnexpectedEOFError):
        await read_frame(reader)


async def test_read_frame_raises_on_truncated_body():
    reader = await _bytes_reader(b"\x01\x00\x00\x00\x05hi")  # claims 5 bytes, has 2
    with pytest.raises(UnexpectedEOFError):
        await read_frame(reader)


async def test_read_frame_rejects_oversized():
    # length = 1024, but max = 100
    framed = b"\x01\x00\x00\x04\x00" + b"x" * 1024
    reader = await _bytes_reader(framed)
    with pytest.raises(FrameTooLargeError):
        await read_frame(reader, max_size=100)


async def test_read_frame_default_max_size():
    assert DEFAULT_MAX_FRAME_SIZE == 16 * 1024 * 1024


async def test_read_frame_skips_unknown_type():
    # Unknown type 0x42 between two JSON frames; reader returns the unknown frame
    # and the caller decides what to do (route or discard).
    body1 = b'{"a":1}'
    unknown = encode_frame(0x42, b"opaque")
    body2 = b'{"b":2}'
    reader = await _bytes_reader(encode_frame(TYPE_JSON_FRAME, body1) + unknown + encode_frame(TYPE_JSON_FRAME, body2))
    t1, _ = await read_frame(reader)
    t2, _ = await read_frame(reader)
    t3, _ = await read_frame(reader)
    assert (t1, t2, t3) == (TYPE_JSON_FRAME, 0x42, TYPE_JSON_FRAME)
