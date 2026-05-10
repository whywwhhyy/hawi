"""Blob protocol: content-addressed binary storage with sandboxing and LRU.

Public surface:
    Direction               — Literal["inbound", "outbound"]
    BlobStore               — SQLite-backed LRU store
    SandboxViolation        — raised on any path-safety failure
    QuotaExceeded           — raised when a write would exceed the per-direction quota
    BlobNotFound            — raised when a referenced blob_id is unknown
    Sha256Mismatch          — raised when finalize sees a body whose hash doesn't match the declared one

Internals (sandbox.py, store.py, commands.py) are not part of the public API.
"""

from .sandbox import Direction, SandboxViolation
from .store import BlobNotFound, BlobStore, QuotaExceeded, Sha256Mismatch

__all__ = [
    "BlobStore",
    "BlobNotFound",
    "Direction",
    "QuotaExceeded",
    "SandboxViolation",
    "Sha256Mismatch",
]
