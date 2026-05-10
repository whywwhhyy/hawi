"""Path-safety primitives for the blob store.

The blob store is content-addressed: callers never name files. This module
provides the validation/derivation utilities so every path that touches the
filesystem is provably inside the sandbox root.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

Direction = Literal["inbound", "outbound"]
"""Allowed blob directions. Each direction maps to its own subdirectory."""

_VALID_DIRECTIONS: frozenset[str] = frozenset({"inbound", "outbound"})
_BLOB_ID_RE = re.compile(r"^[0-9a-f]{64}$")


class SandboxViolation(ValueError):
    """Raised when a path or identifier would escape the sandbox."""


def validate_blob_id(blob_id: str) -> str:
    """Reject anything that isn't a 64-character lowercase-hex string.

    We use lowercase-hex because that's what `hashlib.sha256().hexdigest()`
    returns. Any other shape (mixed case, slashes, dots, .., etc.) is rejected.
    """
    if not isinstance(blob_id, str):
        raise SandboxViolation(f"blob_id must be a string, got {type(blob_id).__name__}")
    if not _BLOB_ID_RE.match(blob_id):
        raise SandboxViolation(
            "blob_id must be 64 lowercase-hex characters (got %r)" % blob_id[:80]
        )
    return blob_id


def blob_id_to_relpath(blob_id: str) -> Path:
    """Map a validated blob_id to a sharded relative path: <first2>/<remaining62>."""
    bid = validate_blob_id(blob_id)
    return Path(bid[:2]) / bid[2:]


def resolve_blob_path(sandbox_root: Path, direction: str, blob_id: str) -> Path:
    """Return the absolute on-disk path for a blob, asserting sandbox containment.

    Raises:
        SandboxViolation if direction is unknown, blob_id malformed, or the
        resolved path escapes sandbox_root/direction.
    """
    if direction not in _VALID_DIRECTIONS:
        raise SandboxViolation(f"unknown direction: {direction!r}")
    relpath = blob_id_to_relpath(blob_id)
    direction_root = (sandbox_root / direction).resolve()
    candidate = (direction_root / relpath).resolve()
    if not candidate.is_relative_to(direction_root):
        raise SandboxViolation(
            f"resolved path escapes sandbox: {candidate} not under {direction_root}"
        )
    return candidate
