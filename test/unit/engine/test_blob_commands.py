"""Tests for blob.* command handlers."""

from __future__ import annotations

import asyncio
import base64
import hashlib
from pathlib import Path

import pytest

from hawi_engine.blob.commands import dispatch_blob_command
from hawi_engine.blob.store import BlobStore
from hawi_engine.protocol import CoreCommand, VERSION


@pytest.fixture
async def store(tmp_path: Path):
    s = BlobStore(root=tmp_path / ".hawi" / "blobs", quota_bytes=4096)
    await s.start()
    yield s
    await s.close()


class _CapturingClient:
    def __init__(self) -> None:
        self.id = "test"
        self.authenticated = True
        self.negotiated_caps: set[str] = {"blob_v1"}
        self.sent: list[dict] = []

    async def send(self, frame: dict) -> None:
        self.sent.append(frame)

    async def close(self) -> None:
        return None


async def test_request_retransmit_returns_not_implemented(store):
    client = _CapturingClient()
    cmd = CoreCommand(
        type="blob.request_retransmit",
        id="r1",
        payload={"blob_id": "a" * 64, "missing_seqs": [1, 2]},
    )
    await dispatch_blob_command(client, cmd, store=store)
    err = next(f for f in client.sent if f["type"] == "error")
    assert err["payload"]["code"] == "not_implemented"


async def test_full_upload_then_has_then_fetch(store):
    client = _CapturingClient()
    body = b"some bytes for the blob"
    sha = hashlib.sha256(body).hexdigest()

    # 1. upload_init
    init_cmd = CoreCommand(
        type="blob.upload_init", id="i1",
        payload={"direction": "inbound", "sha256": sha, "size": len(body), "mime": "application/octet-stream"},
    )
    await dispatch_blob_command(client, init_cmd, store=store)
    init_ack = next(f for f in client.sent if f["id"] == "i1" and f["type"] == "ack")
    bid = init_ack["payload"]["blob_id"]

    # 2. upload_chunk (single chunk)
    chunk_cmd = CoreCommand(
        type="blob.upload_chunk", id="c1",
        payload={"blob_id": bid, "seq": 0, "data_b64": base64.b64encode(body).decode("ascii"), "final": True},
    )
    await dispatch_blob_command(client, chunk_cmd, store=store)
    assert any(f for f in client.sent if f["id"] == "c1" and f["type"] == "ack")

    # 3. upload_finalize
    fin_cmd = CoreCommand(type="blob.upload_finalize", id="f1", payload={"blob_id": bid})
    await dispatch_blob_command(client, fin_cmd, store=store)
    fin_ack = next(f for f in client.sent if f["id"] == "f1" and f["type"] == "ack")
    assert fin_ack["payload"]["sha256"] == sha
    assert fin_ack["payload"]["size"] == len(body)

    # 4. has finds it
    has_cmd = CoreCommand(type="blob.has", id="h1",
                          payload={"sha256": sha, "direction": "inbound"})
    await dispatch_blob_command(client, has_cmd, store=store)
    has_ack = next(f for f in client.sent if f["id"] == "h1" and f["type"] == "ack")
    assert has_ack["payload"]["blob_id"] == bid

    # 5. fetch streams blob.chunk events + blob.complete
    fetch_cmd = CoreCommand(type="blob.fetch", id="g1",
                            payload={"blob_id": bid, "chunk_size": 8})
    await dispatch_blob_command(client, fetch_cmd, store=store)
    chunks = [f for f in client.sent if f["type"] == "blob.chunk"]
    assert b"".join(base64.b64decode(c["payload"]["data_b64"]) for c in chunks) == body
    assert any(f for f in client.sent if f["type"] == "blob.complete")


async def test_finalize_mismatch_returns_error(store):
    client = _CapturingClient()
    body = b"abc"
    declared = hashlib.sha256(b"different").hexdigest()
    init_cmd = CoreCommand(
        type="blob.upload_init", id="i1",
        payload={"direction": "inbound", "sha256": declared, "size": len(body), "mime": None},
    )
    await dispatch_blob_command(client, init_cmd, store=store)
    bid = next(f for f in client.sent if f["id"] == "i1")["payload"]["blob_id"]

    chunk_cmd = CoreCommand(
        type="blob.upload_chunk", id="c1",
        payload={"blob_id": bid, "seq": 0, "data_b64": base64.b64encode(body).decode("ascii")},
    )
    await dispatch_blob_command(client, chunk_cmd, store=store)

    fin_cmd = CoreCommand(type="blob.upload_finalize", id="f1", payload={"blob_id": bid})
    await dispatch_blob_command(client, fin_cmd, store=store)
    err = next(f for f in client.sent if f["id"] == "f1" and f["type"] == "error")
    assert err["payload"]["code"] == "sha256_mismatch"


async def test_release_decrements(store):
    client = _CapturingClient()
    body = b"r"
    sha = hashlib.sha256(body).hexdigest()
    init = CoreCommand(type="blob.upload_init", id="i", payload={
        "direction": "inbound", "sha256": sha, "size": 1, "mime": None})
    await dispatch_blob_command(client, init, store=store)
    bid = next(f for f in client.sent if f["id"] == "i")["payload"]["blob_id"]
    await dispatch_blob_command(client, CoreCommand(type="blob.upload_chunk", id="c", payload={
        "blob_id": bid, "seq": 0, "data_b64": base64.b64encode(body).decode("ascii")}), store=store)
    await dispatch_blob_command(client, CoreCommand(type="blob.upload_finalize", id="f", payload={
        "blob_id": bid}), store=store)
    await dispatch_blob_command(client, CoreCommand(type="blob.release", id="r", payload={
        "blob_id": bid}), store=store)
    rel_ack = next(f for f in client.sent if f["id"] == "r" and f["type"] == "ack")
    assert rel_ack["payload"]["ref_count"] == 0
