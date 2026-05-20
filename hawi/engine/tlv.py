"""TLV (Type-Length-Value) framing for the Hawi engine wire protocol.

Wire layout:
    +--------+----------------+--------------------+
    | type   | length (4B BE) | value (length B)   |
    | 1 byte |                |                    |
    +--------+----------------+--------------------+

Type byte values:
    0x01 JSON_FRAME    — UTF-8-encoded JSON object (the protocol envelope)
    0x02 BINARY_BLOB   — reserved for blob fast-path (Plan 5)
    0x03-0xFE          — reserved
    0xFF               — reserved sentinel (never emitted)

Callers compose JSON frames via encode_frame(TYPE_JSON_FRAME, json_bytes) and
parse incoming frames via `read_frame(reader)`. The reader returns
(type_byte, value_bytes) for each successful frame, None on clean EOF, and
raises UnexpectedEOFError on partial reads.
"""

from __future__ import annotations

import asyncio
from typing import Final, Protocol


class _AsyncByteReader(Protocol):
    """Minimum surface area `read_frame` needs from a reader.

    Both `asyncio.StreamReader` and the threaded stdin fallback in
    `builtin_gateways._ThreadedStdinReader` satisfy this — they expose an
    awaitable `readexactly(n)` that returns up to `n` bytes and raises
    `asyncio.IncompleteReadError` on short reads.
    """

    async def readexactly(self, n: int, /) -> bytes: ...


TYPE_JSON_FRAME: Final[int] = 0x01
TYPE_BINARY_BLOB: Final[int] = 0x02

DEFAULT_MAX_FRAME_SIZE: Final[int] = 16 * 1024 * 1024  # 16 MiB

_HEADER_LEN: Final[int] = 5  # 1 byte type + 4 bytes BE length


class FrameTooLargeError(ValueError):
    """Raised when an incoming frame's length exceeds the allowed maximum."""


class UnexpectedEOFError(ConnectionError):
    """Raised when the stream ends mid-frame (truncated header or body)."""


def encode_frame(type_byte: int, value: bytes) -> bytes:
    """Encode a single TLV frame.

    type_byte must be in 0..255. Value length must fit in 4 unsigned bytes
    (i.e., <= 4 GiB). Callers are responsible for keeping value sizes within
    their negotiated max_size.
    """
    if not 0 <= type_byte <= 0xFF:
        raise ValueError(f"type_byte must be 0..255, got {type_byte}")
    length = len(value)
    if length > 0xFFFFFFFF:
        raise ValueError(f"frame value length {length} exceeds 4 GiB")
    return bytes([type_byte]) + length.to_bytes(4, "big") + value


async def read_frame(
    reader: _AsyncByteReader,
    *,
    max_size: int = DEFAULT_MAX_FRAME_SIZE,
) -> tuple[int, bytes] | None:
    """Read one TLV frame from the reader.

    Returns:
        (type_byte, value) on success.
        None on clean end-of-stream (no bytes were read before EOF).

    Raises:
        UnexpectedEOFError: stream ended mid-header or mid-body.
        FrameTooLargeError: declared length exceeds max_size.
    """
    try:
        header = await reader.readexactly(_HEADER_LEN)
    except asyncio.IncompleteReadError as exc:
        if not exc.partial:
            return None
        raise UnexpectedEOFError(
            f"Stream ended mid-header after {len(exc.partial)} of {_HEADER_LEN} bytes"
        ) from exc

    type_byte = header[0]
    length = int.from_bytes(header[1:5], "big")

    if length > max_size:
        raise FrameTooLargeError(
            f"Frame length {length} exceeds max_size {max_size}"
        )

    if length == 0:
        return type_byte, b""

    try:
        value = await reader.readexactly(length)
    except asyncio.IncompleteReadError as exc:
        raise UnexpectedEOFError(
            f"Stream ended mid-body after {len(exc.partial)} of {length} bytes"
        ) from exc

    return type_byte, value
