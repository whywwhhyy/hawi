"""Tests for BlobStore: round-trip, LRU, quota, ref counting, sha256 verify."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest

from hawi_engine.blob.store import (
    BlobNotFound,
    BlobStore,
    QuotaExceeded,
    Sha256Mismatch,
)
from hawi_engine.blob.sandbox import resolve_blob_path


@pytest.fixture
async def store(tmp_path: Path) -> BlobStore:
    s = BlobStore(root=tmp_path / ".hawi" / "blobs", quota_bytes=1024, chunk_size=128)
    await s.start()
    yield s
    await s.close()


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def _put_blob(
    store: BlobStore,
    body: bytes,
    *,
    direction: str = "inbound",
    release: bool = False,
) -> str:
    bid = await store.upload_init(
        direction=direction,  # type: ignore[arg-type]
        sha256=_hash(body),
        size=len(body),
        mime=None,
    )
    await store.upload_chunk(bid, seq=0, data=body)
    await store.upload_finalize(bid)
    if release:
        await store.release(bid)
    return bid


async def test_upload_finalize_fetch_roundtrip(store: BlobStore):
    body = b"hello world"
    bid = await store.upload_init(direction="inbound", sha256=_hash(body), size=len(body), mime="text/plain")
    await store.upload_chunk(bid, seq=0, data=body)
    info = await store.upload_finalize(bid)
    assert info.size == len(body)
    assert info.sha256 == _hash(body)
    assert info.ref_count == 1

    chunks = []
    async for seq, chunk in store.fetch_chunks(bid, chunk_size=4):
        chunks.append((seq, chunk))
    assembled = b"".join(c for _, c in sorted(chunks))
    assert assembled == body


async def test_zero_byte_blob_roundtrip(store: BlobStore):
    bid = await store.upload_init(direction="inbound", sha256=_hash(b""), size=0, mime=None)
    info = await store.upload_finalize(bid)

    chunks = []
    async for item in store.fetch_chunks(bid):
        chunks.append(item)

    assert info.size == 0
    assert info.sha256 == _hash(b"")
    assert chunks == []


async def test_finalize_rejects_sha256_mismatch(store: BlobStore):
    body = b"abc"
    declared = _hash(b"different")
    bid = await store.upload_init(direction="inbound", sha256=declared, size=len(body), mime=None)
    await store.upload_chunk(bid, seq=0, data=body)
    with pytest.raises(Sha256Mismatch):
        await store.upload_finalize(bid)
    # Staging file should be cleaned up
    assert not list((store.root / "inbound").rglob("*"))


async def test_has_finds_by_sha256(store: BlobStore):
    body = b"shared"
    bid = await store.upload_init(direction="inbound", sha256=_hash(body), size=len(body), mime=None)
    await store.upload_chunk(bid, seq=0, data=body)
    await store.upload_finalize(bid)

    found = await store.has(sha256=_hash(body), direction="inbound")
    assert found == bid

    missing = await store.has(sha256=_hash(b"absent"), direction="inbound")
    assert missing is None


async def test_has_ignores_unfinalized_blob(store: BlobStore):
    body = b"pending"
    await store.upload_init(direction="inbound", sha256=_hash(body), size=len(body), mime=None)

    assert await store.has(sha256=_hash(body), direction="inbound") is None


async def test_release_decrements_ref_count(store: BlobStore):
    body = b"x"
    bid = await store.upload_init(direction="inbound", sha256=_hash(body), size=1, mime=None)
    await store.upload_chunk(bid, 0, body)
    info = await store.upload_finalize(bid)
    assert info.ref_count == 1
    new_rc = await store.release(bid)
    assert new_rc == 0


async def test_release_is_floored_at_zero(store: BlobStore):
    bid = await _put_blob(store, b"x")

    assert await store.release(bid) == 0
    assert await store.release(bid) == 0


async def test_release_unknown_blob_raises(store: BlobStore):
    with pytest.raises(BlobNotFound):
        await store.release("0" * 64)


async def test_release_unfinalized_blob_raises(store: BlobStore):
    bid = await store.upload_init(direction="inbound", sha256=_hash(b"x"), size=1, mime=None)

    with pytest.raises(BlobNotFound):
        await store.release(bid)


async def test_upload_chunk_rejects_after_finalize(store: BlobStore):
    bid = await _put_blob(store, b"stable")
    path = resolve_blob_path(store.root, "inbound", bid)
    before = path.read_bytes()

    with pytest.raises(BlobNotFound):
        await store.upload_chunk(bid, seq=1, data=b"mutation")

    assert path.read_bytes() == before


async def test_lru_evicts_unreferenced_blobs(store: BlobStore):
    """quota=1024; upload 8 blobs of 200 bytes each = 1600. With ref_count=0
    on all but the last, the older ones get evicted to fit."""
    bids: list[str] = []
    for i in range(8):
        body = bytes([i]) * 200
        bid = await store.upload_init(direction="inbound", sha256=_hash(body), size=200, mime=None)
        await store.upload_chunk(bid, 0, body)
        await store.upload_finalize(bid)
        await store.release(bid)  # ref_count=0 after release
        bids.append(bid)
        # Tiny sleep so last_access timestamps order naturally
        await asyncio.sleep(0.005)

    # Now upload one more; it should fit because LRU evicts the oldest unreferenced.
    body = b"y" * 200
    bid_new = await store.upload_init(direction="inbound", sha256=_hash(body), size=200, mime=None)
    await store.upload_chunk(bid_new, 0, body)
    await store.upload_finalize(bid_new)

    # Total size on disk should be <= quota
    total = sum(p.stat().st_size for p in (store.root / "inbound").rglob("*") if p.is_file())
    assert total <= store.quota_bytes
    # The most recently accessed must still be present
    assert (await store.has(sha256=_hash(body), direction="inbound")) == bid_new


async def test_lru_uses_last_access_not_creation_order(store: BlobStore):
    bid_a = await _put_blob(store, b"a" * 300, release=True)
    await asyncio.sleep(0.005)
    bid_b = await _put_blob(store, b"b" * 300, release=True)
    await asyncio.sleep(0.005)
    bid_c = await _put_blob(store, b"c" * 300, release=True)

    # Promote A so quota pressure evicts B, the oldest unreferenced blob.
    async for _ in store.fetch_chunks(bid_a):
        pass
    await asyncio.sleep(0.005)
    bid_d = await _put_blob(store, b"d" * 300, release=True)

    assert await store.has(sha256=_hash(b"a" * 300), direction="inbound") == bid_a
    assert await store.has(sha256=_hash(b"b" * 300), direction="inbound") is None
    assert await store.has(sha256=_hash(b"c" * 300), direction="inbound") == bid_c
    assert await store.has(sha256=_hash(b"d" * 300), direction="inbound") == bid_d


async def test_quota_is_per_direction(store: BlobStore):
    for i in range(5):
        await _put_blob(store, bytes([i]) * 200, direction="inbound")

    outbound_body = b"out" * 100
    outbound_id = await _put_blob(store, outbound_body, direction="outbound")

    assert await store.has(sha256=_hash(outbound_body), direction="outbound") == outbound_id


async def test_quota_exceeded_when_referenced(store: BlobStore):
    """If all blobs are referenced (ref_count > 0), eviction can't free space; raise."""
    bids = []
    for i in range(5):
        body = bytes([i]) * 200
        bid = await store.upload_init(direction="inbound", sha256=_hash(body), size=200, mime=None)
        await store.upload_chunk(bid, 0, body)
        await store.upload_finalize(bid)  # ref_count=1, NOT released
        bids.append(bid)

    # 6th upload at 200 bytes would need 1200 total > 1024 quota; can't evict any.
    body = b"z" * 200
    bid6 = await store.upload_init(direction="inbound", sha256=_hash(body), size=200, mime=None)
    await store.upload_chunk(bid6, 0, body)
    with pytest.raises(QuotaExceeded):
        await store.upload_finalize(bid6)
    assert not resolve_blob_path(store.root, "inbound", bid6).exists()
    assert await store.has(sha256=_hash(body), direction="inbound") is None
    with pytest.raises(BlobNotFound):
        async for _ in store.fetch_chunks(bid6):
            pass


async def test_fetch_unknown_blob_raises(store: BlobStore):
    with pytest.raises(BlobNotFound):
        async for _ in store.fetch_chunks("0" * 64):
            pass


async def test_fetch_rejects_invalid_chunk_size(store: BlobStore):
    bid = await _put_blob(store, b"chunk-size")

    with pytest.raises(ValueError):
        async for _ in store.fetch_chunks(bid, chunk_size=0):
            pass


async def test_upload_init_rejects_invalid_direction_without_files(store: BlobStore):
    with pytest.raises(ValueError):
        await store.upload_init(
            direction="sideways",  # type: ignore[arg-type]
            sha256=_hash(b"x"),
            size=1,
            mime=None,
        )
    assert not (store.root / "sideways").exists()
