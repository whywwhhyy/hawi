"""Tests for the tlv2ndjson debug CLI."""

from __future__ import annotations

import asyncio
import io
import json
import subprocess
import sys

from hawi_engine.tlv import TYPE_JSON_FRAME, encode_frame
from hawi_engine.tlv2ndjson import translate_stream


def _make_stream(frames: list[dict]) -> bytes:
    out = b""
    for frame in frames:
        out += encode_frame(TYPE_JSON_FRAME, json.dumps(frame).encode("utf-8"))
    return out


async def test_translate_stream_emits_one_line_per_frame():
    incoming = _make_stream([{"a": 1}, {"b": 2}])
    sink = io.StringIO()
    reader = asyncio.StreamReader()
    reader.feed_data(incoming)
    reader.feed_eof()
    await translate_stream(reader, sink)
    lines = sink.getvalue().splitlines()
    assert lines == ['{"a": 1}', '{"b": 2}']


async def test_translate_stream_drops_unknown_types():
    incoming = (
        encode_frame(TYPE_JSON_FRAME, b'{"keep":1}')
        + encode_frame(0x42, b"opaque")
        + encode_frame(TYPE_JSON_FRAME, b'{"keep":2}')
    )
    sink = io.StringIO()
    reader = asyncio.StreamReader()
    reader.feed_data(incoming)
    reader.feed_eof()
    await translate_stream(reader, sink)
    lines = sink.getvalue().splitlines()
    assert lines == ['{"keep":1}', '{"keep":2}']


def test_cli_entry_point_exists():
    """`tlv2ndjson` should be installed as a console script."""
    result = subprocess.run(
        [sys.executable, "-m", "hawi_engine.tlv2ndjson", "--help"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0
    assert "tlv2ndjson" in result.stdout.lower()
