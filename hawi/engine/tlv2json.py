"""tlv2json — translate a TLV byte stream into newline-delimited JSON.

Usage:
    cat <tlv_capture> | tlv2json | jq

Reads from stdin, writes to stdout. Only TYPE_JSON_FRAME (0x01) frames are
emitted; other types are silently discarded so the JSON sink stays clean.
The output is newline-delimited JSON because a TLV stream can contain many
JSON frames.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import TextIO

from .tlv import (
    DEFAULT_MAX_FRAME_SIZE,
    TYPE_JSON_FRAME,
    FrameTooLargeError,
    UnexpectedEOFError,
    read_frame,
)


async def translate_stream(
    reader: asyncio.StreamReader,
    sink: TextIO,
    *,
    max_size: int = DEFAULT_MAX_FRAME_SIZE,
    diagnostic_name: str = "tlv2json",
) -> None:
    """Read TLV frames from `reader` and write JSON-frame bodies as lines to `sink`."""
    while True:
        try:
            result = await read_frame(reader, max_size=max_size)
        except (FrameTooLargeError, UnexpectedEOFError) as exc:
            print(f"{diagnostic_name}: {exc}", file=sys.stderr)
            return
        if result is None:
            return
        type_byte, value = result
        if type_byte != TYPE_JSON_FRAME:
            continue
        sink.write(value.decode("utf-8") + "\n")
        sink.flush()


async def _amain(argv: list[str], *, prog: str = "tlv2json") -> int:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Convert a TLV byte stream on stdin into newline-delimited JSON on stdout.",
    )
    parser.add_argument(
        "--max-frame-size",
        type=int,
        default=DEFAULT_MAX_FRAME_SIZE,
        help="Max frame body size in bytes (default 16 MiB).",
    )
    args = parser.parse_args(argv)

    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    try:
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)
    except (NotImplementedError, OSError):
        # Fallback: read all of stdin synchronously (good enough for offline debug)
        data = sys.stdin.buffer.read()
        reader.feed_data(data)
        reader.feed_eof()

    await translate_stream(
        reader,
        sys.stdout,
        max_size=args.max_frame_size,
        diagnostic_name=prog,
    )
    return 0


def main() -> None:
    sys.exit(asyncio.run(_amain(sys.argv[1:])))


if __name__ == "__main__":
    main()
