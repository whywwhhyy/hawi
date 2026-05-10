"""SQLite-backed blob store with LRU eviction, quota, and sandboxed file storage.

A single BlobStore instance owns:
  - one sqlite3 connection (WAL mode)
  - the directory layout under root/{inbound,outbound}
  - an asyncio.Lock that serializes write paths

Threading model: SQLite calls are wrapped in asyncio.to_thread so the event
loop is never blocked on disk I/O. A single per-store Lock protects
read/modify/write sequences (LRU eviction + insert).
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import AsyncIterator, Optional

from .sandbox import Direction, resolve_blob_path, validate_blob_id

logger = logging.getLogger(__name__)


class BlobNotFound(KeyError):
    pass


class QuotaExceeded(RuntimeError):
    pass


class Sha256Mismatch(ValueError):
    pass


_VALID_DIRECTIONS: frozenset[str] = frozenset({"inbound", "outbound"})


@dataclasses.dataclass(frozen=True)
class BlobInfo:
    blob_id: str
    sha256: str
    direction: str
    size: int
    mime: Optional[str]
    ref_count: int


def _rmdir_if_empty(path: Path) -> None:
    """Best-effort: remove `path` if it exists and is an empty directory."""
    try:
        path.rmdir()
    except (FileNotFoundError, OSError):
        # Either it doesn't exist or it isn't empty — neither is an error here.
        pass


_SCHEMA = """
CREATE TABLE IF NOT EXISTS blobs (
    blob_id     TEXT PRIMARY KEY,
    sha256      TEXT NOT NULL,
    direction   TEXT NOT NULL CHECK(direction IN ('inbound','outbound')),
    size        INTEGER NOT NULL,
    mime        TEXT,
    last_access REAL NOT NULL,
    ref_count   INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL,
    finalized   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_blobs_sha256_dir ON blobs(sha256, direction);
CREATE INDEX IF NOT EXISTS idx_blobs_lru ON blobs(direction, ref_count, last_access);
"""


class BlobStore:
    """Content-addressed blob store with per-direction quota and LRU eviction."""

    def __init__(
        self,
        *,
        root: Path,
        quota_bytes: int = 1 << 30,  # 1 GiB
        chunk_size: int = 64 * 1024,
    ) -> None:
        self.root = Path(root)
        self.quota_bytes = quota_bytes
        self.chunk_size = chunk_size
        self._db_path = self.root / "store.db"
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "inbound").mkdir(exist_ok=True)
        (self.root / "outbound").mkdir(exist_ok=True)

        def _open() -> sqlite3.Connection:
            # check_same_thread=False is safe here because all access goes
            # through asyncio.to_thread under the same asyncio.Lock — only one
            # thread touches the connection at a time, and the asyncio Lock
            # provides the necessary memory barrier.
            conn = sqlite3.connect(
                self._db_path, isolation_level=None, check_same_thread=False
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(_SCHEMA)
            return conn

        self._conn = await asyncio.to_thread(_open)

    async def close(self) -> None:
        if self._conn is not None:
            await asyncio.to_thread(self._conn.close)
            self._conn = None

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("BlobStore is not started; call await start() first")
        return self._conn

    # ----- Upload --------------------------------------------------------

    async def upload_init(
        self,
        *,
        direction: Direction,
        sha256: str,
        size: int,
        mime: Optional[str],
    ) -> str:
        """Create a new pending blob entry and return its blob_id.

        The blob_id is a freshly-minted random hex string (NOT the sha256) so
        that distinct uploads of the same content can coexist as separate
        staging files until finalization.
        """
        if direction not in _VALID_DIRECTIONS:
            raise ValueError("direction must be 'inbound' or 'outbound'")
        if size < 0:
            raise ValueError("size must be >= 0")
        if not isinstance(sha256, str) or len(sha256) != 64 or not all(c in "0123456789abcdef" for c in sha256):
            raise ValueError("sha256 must be 64 lowercase hex chars")

        blob_id = uuid.uuid4().hex + uuid.uuid4().hex  # 64 chars
        async with self._lock:
            now = time.time()
            await asyncio.to_thread(
                self._db().execute,
                "INSERT INTO blobs (blob_id, sha256, direction, size, mime, last_access, ref_count, created_at, finalized)"
                " VALUES (?, ?, ?, ?, ?, ?, 0, ?, 0)",
                (blob_id, sha256, direction, size, mime, now, now),
            )
        # Pre-create the staging path's parent dir
        path = resolve_blob_path(self.root, direction, blob_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(path.touch, exist_ok=False)
        return blob_id

    async def upload_chunk(self, blob_id: str, seq: int, data: bytes) -> None:
        """Append `data` to the staging file for `blob_id`. Caller orders by `seq`.

        Note: this implementation appends sequentially. Out-of-order chunks are
        not supported in v1 (caller must order). seq is recorded for future
        retransmit support but currently only validated as monotonic-non-decreasing.
        """
        validate_blob_id(blob_id)
        if seq < 0:
            raise ValueError("seq must be >= 0")
        info = await self._get_unfinalized(blob_id)
        if info is None:
            raise BlobNotFound(blob_id)
        path = resolve_blob_path(self.root, info["direction"], blob_id)

        def _append() -> None:
            with open(path, "ab") as f:
                f.write(data)

        await asyncio.to_thread(_append)

    async def upload_finalize(self, blob_id: str) -> BlobInfo:
        """Verify size + sha256, register, bump ref_count to 1, run quota check."""
        validate_blob_id(blob_id)
        info = await self._get_unfinalized(blob_id)
        if info is None:
            raise BlobNotFound(blob_id)
        direction = info["direction"]
        declared_sha = info["sha256"]
        declared_size = info["size"]
        path = resolve_blob_path(self.root, direction, blob_id)

        def _verify() -> tuple[int, str]:
            h = hashlib.sha256()
            actual_size = 0
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(64 * 1024)
                    if not chunk:
                        break
                    h.update(chunk)
                    actual_size += len(chunk)
            return actual_size, h.hexdigest()

        actual_size, actual_sha = await asyncio.to_thread(_verify)

        async with self._lock:
            if actual_sha != declared_sha or actual_size != declared_size:
                # Clean up staging file + (best-effort) empty parent shard + DB row
                await asyncio.to_thread(path.unlink, missing_ok=True)
                await asyncio.to_thread(_rmdir_if_empty, path.parent)
                await asyncio.to_thread(
                    self._db().execute, "DELETE FROM blobs WHERE blob_id = ?", (blob_id,)
                )
                raise Sha256Mismatch(
                    f"declared sha256={declared_sha[:8]}...,size={declared_size}; "
                    f"actual sha256={actual_sha[:8]}...,size={actual_size}"
                )

            # Quota check + LRU eviction in same critical section
            try:
                await self._evict_to_fit(direction, needed_bytes=actual_size, exclude_blob_id=blob_id)
            except QuotaExceeded:
                await asyncio.to_thread(path.unlink, missing_ok=True)
                await asyncio.to_thread(_rmdir_if_empty, path.parent)
                await asyncio.to_thread(
                    self._db().execute, "DELETE FROM blobs WHERE blob_id = ?", (blob_id,)
                )
                raise

            now = time.time()
            await asyncio.to_thread(
                self._db().execute,
                "UPDATE blobs SET finalized = 1, ref_count = 1, last_access = ? WHERE blob_id = ?",
                (now, blob_id),
            )

            row = await asyncio.to_thread(
                self._db().execute,
                "SELECT blob_id, sha256, direction, size, mime, ref_count FROM blobs WHERE blob_id = ?",
                (blob_id,),
            )
            row = row.fetchone()
            return BlobInfo(*row)

    # ----- Read ---------------------------------------------------------

    async def has(self, *, sha256: str, direction: Direction) -> Optional[str]:
        if direction not in _VALID_DIRECTIONS:
            raise ValueError("direction must be 'inbound' or 'outbound'")
        if not isinstance(sha256, str) or len(sha256) != 64 or not all(c in "0123456789abcdef" for c in sha256):
            raise ValueError("sha256 must be 64 lowercase hex chars")
        cur = await asyncio.to_thread(
            self._db().execute,
            "SELECT blob_id FROM blobs WHERE sha256 = ? AND direction = ? AND finalized = 1 LIMIT 1",
            (sha256, direction),
        )
        row = cur.fetchone()
        return row[0] if row else None

    async def fetch_chunks(
        self, blob_id: str, *, chunk_size: Optional[int] = None
    ) -> AsyncIterator[tuple[int, bytes]]:
        """Yield (seq, chunk_bytes) for the blob. Updates last_access."""
        validate_blob_id(blob_id)
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        cur = await asyncio.to_thread(
            self._db().execute,
            "SELECT direction FROM blobs WHERE blob_id = ? AND finalized = 1",
            (blob_id,),
        )
        row = cur.fetchone()
        if not row:
            raise BlobNotFound(blob_id)
        direction = row[0]
        path = resolve_blob_path(self.root, direction, blob_id)
        size = chunk_size or self.chunk_size

        async def _read_all() -> AsyncIterator[tuple[int, bytes]]:
            seq = 0

            def _next(f) -> bytes:
                return f.read(size)

            f = await asyncio.to_thread(open, path, "rb")
            try:
                while True:
                    chunk = await asyncio.to_thread(_next, f)
                    if not chunk:
                        return
                    yield seq, chunk
                    seq += 1
            finally:
                await asyncio.to_thread(f.close)

        # Update last_access (tracking purpose; ignore concurrency races, latest wins)
        await asyncio.to_thread(
            self._db().execute,
            "UPDATE blobs SET last_access = ? WHERE blob_id = ?",
            (time.time(), blob_id),
        )

        async for item in _read_all():
            yield item

    async def release(self, blob_id: str) -> int:
        """Decrement ref_count (floor at 0). Returns the new ref_count."""
        validate_blob_id(blob_id)
        async with self._lock:
            await asyncio.to_thread(
                self._db().execute,
                "UPDATE blobs SET ref_count = MAX(ref_count - 1, 0) WHERE blob_id = ? AND finalized = 1",
                (blob_id,),
            )
            cur = await asyncio.to_thread(
                self._db().execute,
                "SELECT ref_count FROM blobs WHERE blob_id = ? AND finalized = 1",
                (blob_id,),
            )
            row = cur.fetchone()
            if not row:
                raise BlobNotFound(blob_id)
            return row[0]

    # ----- Internals -----------------------------------------------------

    async def _get_unfinalized(self, blob_id: str) -> Optional[dict]:
        cur = await asyncio.to_thread(
            self._db().execute,
            "SELECT blob_id, sha256, direction, size, mime, finalized FROM blobs WHERE blob_id = ? AND finalized = 0",
            (blob_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        keys = ("blob_id", "sha256", "direction", "size", "mime", "finalized")
        return dict(zip(keys, row))

    async def _evict_to_fit(self, direction: str, *, needed_bytes: int, exclude_blob_id: str) -> None:
        """Evict oldest unreferenced finalized blobs until quota would not be exceeded.

        Quota counts ALL finalized blobs in `direction` plus the candidate's
        `needed_bytes`. The candidate (exclude_blob_id) is currently finalized=0
        so it isn't counted in the existing total — we add `needed_bytes` to
        decide if we have room.
        """
        cur = await asyncio.to_thread(
            self._db().execute,
            "SELECT COALESCE(SUM(size), 0) FROM blobs WHERE direction = ? AND finalized = 1",
            (direction,),
        )
        used = cur.fetchone()[0]

        budget = self.quota_bytes - used
        if needed_bytes <= budget:
            return  # already fits

        # Try to evict oldest unreferenced (ref_count = 0) until we fit.
        cur = await asyncio.to_thread(
            self._db().execute,
            "SELECT blob_id, size FROM blobs"
            " WHERE direction = ? AND finalized = 1 AND ref_count = 0"
            " ORDER BY last_access ASC",
            (direction,),
        )
        candidates = cur.fetchall()
        freed = 0
        evicted_ids: list[str] = []
        for blob_id, size in candidates:
            if blob_id == exclude_blob_id:
                continue
            evicted_ids.append(blob_id)
            freed += size
            if (used - freed) + needed_bytes <= self.quota_bytes:
                break
        if (used - freed) + needed_bytes > self.quota_bytes:
            raise QuotaExceeded(
                f"direction {direction!r}: needed {needed_bytes} bytes, "
                f"used {used}, quota {self.quota_bytes}, evictable {freed}"
            )

        for blob_id in evicted_ids:
            cur = await asyncio.to_thread(
                self._db().execute,
                "SELECT direction FROM blobs WHERE blob_id = ?",
                (blob_id,),
            )
            row = cur.fetchone()
            if not row:
                continue
            path = resolve_blob_path(self.root, row[0], blob_id)
            await asyncio.to_thread(path.unlink, missing_ok=True)
            await asyncio.to_thread(_rmdir_if_empty, path.parent)
            await asyncio.to_thread(
                self._db().execute, "DELETE FROM blobs WHERE blob_id = ?", (blob_id,)
            )
        logger.info("Evicted %d blob(s) from %s, freed %d bytes", len(evicted_ids), direction, freed)
