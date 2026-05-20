"""Local integration tests for engine TCP/TLV and blob protocol wiring."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import hawi.engine.builtin_gateways as builtin_gateways
from hawi.engine.blob.store import BlobStore
from hawi.engine.protocol import VERSION
from hawi.engine.runtime import CoreRuntime
from hawi.engine.tlv import TYPE_JSON_FRAME, encode_frame, read_frame


async def _connect_tcp_with_retry(port: int) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    last_exc: Exception | None = None
    for _ in range(50):
        try:
            return await asyncio.open_connection("127.0.0.1", port)
        except OSError as exc:
            last_exc = exc
            await asyncio.sleep(0.02)
    assert last_exc is not None
    raise last_exc


async def _send_command(
    writer: asyncio.StreamWriter,
    command_type: str,
    request_id: str,
    payload: dict[str, Any],
) -> None:
    body = json.dumps(
        {
            "version": VERSION,
            "type": command_type,
            "id": request_id,
            "payload": payload,
        }
    ).encode("utf-8")
    writer.write(encode_frame(TYPE_JSON_FRAME, body))
    await writer.drain()


async def _recv_frame(reader: asyncio.StreamReader, *, timeout: float = 2) -> dict[str, Any]:
    result = await asyncio.wait_for(read_frame(reader), timeout=timeout)
    assert result is not None, "stream closed"
    type_byte, value = result
    assert type_byte == TYPE_JSON_FRAME
    return json.loads(value.decode("utf-8"))


async def _recv_until(
    reader: asyncio.StreamReader,
    predicate,
    *,
    timeout: float = 2,
) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        assert remaining > 0, "timed out waiting for expected frame"
        frame = await _recv_frame(reader, timeout=remaining)
        if predicate(frame):
            return frame


@pytest.mark.asyncio
async def test_blob_upload_fetch_roundtrip_over_tcp_tlv(
    tmp_path: Path,
    unused_tcp_port: int,
) -> None:
    store = BlobStore(root=tmp_path / ".hawi" / "blobs", quota_bytes=4096, chunk_size=4)
    await store.start()
    runtime = CoreRuntime(
        model_name="test-model",
        token=None,
        status_interval=60.0,
        blob_store=store,
    )
    args = argparse.Namespace(
        host="127.0.0.1",
        port=unused_tcp_port,
        outbound_queue_size=20,
        max_frame_size=16 * 1024 * 1024,
    )
    server_task = asyncio.create_task(
        builtin_gateways.TcpGateway().serve(runtime, args)
    )
    reader, writer = await _connect_tcp_with_retry(unused_tcp_port)

    try:
        ready = await _recv_frame(reader)
        assert ready["type"] == "core.ready"

        await _send_command(
            writer,
            "hello",
            "hello-1",
            {"client_caps": ["blob_v1", "tlv_v1"]},
        )
        hello = await _recv_until(reader, lambda f: f["id"] == "hello-1")
        assert hello["type"] == "ack"
        assert hello["payload"]["negotiated"] == ["blob_v1", "tlv_v1"]

        body = b"integration blob bytes"
        sha = hashlib.sha256(body).hexdigest()
        await _send_command(
            writer,
            "blob.upload_init",
            "init-1",
            {"direction": "inbound", "sha256": sha, "size": len(body), "mime": None},
        )
        init_ack = await _recv_until(reader, lambda f: f["id"] == "init-1")
        blob_id = init_ack["payload"]["blob_id"]

        await _send_command(
            writer,
            "blob.upload_chunk",
            "chunk-1",
            {
                "blob_id": blob_id,
                "seq": 0,
                "data_b64": base64.b64encode(body).decode("ascii"),
            },
        )
        chunk_ack = await _recv_until(reader, lambda f: f["id"] == "chunk-1")
        assert chunk_ack["type"] == "ack"

        await _send_command(writer, "blob.upload_finalize", "final-1", {"blob_id": blob_id})
        final_ack = await _recv_until(reader, lambda f: f["id"] == "final-1")
        assert final_ack["payload"]["sha256"] == sha

        await _send_command(
            writer,
            "blob.fetch",
            "fetch-1",
            {"blob_id": blob_id, "chunk_size": 6},
        )
        fetch_ack = await _recv_until(reader, lambda f: f["id"] == "fetch-1")
        assert fetch_ack["type"] == "ack"

        chunks: list[bytes] = []
        while True:
            frame = await _recv_frame(reader)
            if frame["type"] == "blob.complete":
                break
            assert frame["type"] == "blob.chunk"
            chunks.append(base64.b64decode(frame["payload"]["data_b64"]))

        assert b"".join(chunks) == body

        await _send_command(writer, "shutdown", "shutdown-1", {})
        shutdown_ack = await _recv_until(reader, lambda f: f["id"] == "shutdown-1")
        assert shutdown_ack["type"] == "ack"
        await asyncio.wait_for(server_task, timeout=2)
    finally:
        if not runtime.is_shutdown_requested:
            await runtime.stop()
        writer.close()
        await writer.wait_closed()
        if not server_task.done():
            server_task.cancel()
            await asyncio.gather(server_task, return_exceptions=True)
        await store.close()
