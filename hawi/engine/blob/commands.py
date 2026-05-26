"""Per-command handlers for the blob.* command family.

Each handler takes a runtime client + parsed CoreCommand and a BlobStore;
it sends back ack/error frames via client.send().
"""

from __future__ import annotations

import base64
from typing import Any

from ..protocol import CoreCommand, make_ack, make_error, make_frame
from .store import BlobNotFound, BlobStore, QuotaExceeded, Sha256Mismatch


async def dispatch_blob_command(client, command: CoreCommand, *, store: BlobStore) -> None:
    """Route a blob.* command to its handler."""
    handler = _HANDLERS.get(command.type)
    if handler is None:
        await client.send(
            make_error(
                f"Unknown blob command: {command.type}",
                request_id=command.id,
                code="unknown_command",
            )
        )
        return
    try:
        await handler(client, command, store)
    except (ValueError, BlobNotFound, QuotaExceeded, Sha256Mismatch) as exc:
        code = _exception_code(exc)
        await client.send(
            make_error(str(exc), request_id=command.id, code=code)
        )


def _exception_code(exc: Exception) -> str:
    if isinstance(exc, BlobNotFound):
        return "blob_not_found"
    if isinstance(exc, QuotaExceeded):
        return "quota_exceeded"
    if isinstance(exc, Sha256Mismatch):
        return "sha256_mismatch"
    return "bad_request"


async def _handle_upload_init(client, command: CoreCommand, store: BlobStore) -> None:
    p = command.payload
    direction = p.get("direction")
    sha256 = p.get("sha256")
    size = p.get("size")
    mime = p.get("mime")
    if direction not in ("inbound", "outbound"):
        raise ValueError("payload.direction must be 'inbound' or 'outbound'")
    if not isinstance(sha256, str):
        raise ValueError("payload.sha256 must be a string")
    if not isinstance(size, int):
        raise ValueError("payload.size must be an integer")
    if mime is not None and not isinstance(mime, str):
        raise ValueError("payload.mime must be a string or null")

    blob_id = await store.upload_init(direction=direction, sha256=sha256, size=size, mime=mime)
    await client.send(
        make_ack("blob.upload_init", request_id=command.id, payload={"blob_id": blob_id})
    )


async def _handle_upload_chunk(client, command: CoreCommand, store: BlobStore) -> None:
    p = command.payload
    blob_id = p.get("blob_id")
    seq = p.get("seq")
    data_b64 = p.get("data_b64")
    if not isinstance(blob_id, str):
        raise ValueError("payload.blob_id must be a string")
    if not isinstance(seq, int):
        raise ValueError("payload.seq must be an integer")
    if seq < 0:
        raise ValueError("payload.seq must be >= 0")
    if not isinstance(data_b64, str):
        raise ValueError("payload.data_b64 must be a base64 string")
    try:
        data = base64.b64decode(data_b64, validate=True)
    except Exception as exc:
        raise ValueError(f"payload.data_b64 not valid base64: {exc}")

    await store.upload_chunk(blob_id, seq, data)
    await client.send(
        make_ack("blob.upload_chunk", request_id=command.id, payload={"seq": seq})
    )


async def _handle_upload_finalize(client, command: CoreCommand, store: BlobStore) -> None:
    blob_id = command.payload.get("blob_id")
    if not isinstance(blob_id, str):
        raise ValueError("payload.blob_id must be a string")
    info = await store.upload_finalize(blob_id)
    await client.send(
        make_ack(
            "blob.upload_finalize",
            request_id=command.id,
            payload={
                "blob_id": info.blob_id,
                "uri": f"hawi-blob://{info.blob_id}",
                "sha256": info.sha256,
                "direction": info.direction,
                "size": info.size,
                "mime": info.mime,
                "ref_count": info.ref_count,
            },
        )
    )


async def _handle_info(client, command: CoreCommand, store: BlobStore) -> None:
    blob_id = command.payload.get("blob_id")
    if not isinstance(blob_id, str):
        raise ValueError("payload.blob_id must be a string")
    info = await store.info(blob_id)
    await client.send(
        make_ack(
            "blob.info",
            request_id=command.id,
            payload={
                "blob_id": info.blob_id,
                "uri": f"hawi-blob://{info.blob_id}",
                "sha256": info.sha256,
                "direction": info.direction,
                "size": info.size,
                "mime": info.mime,
                "ref_count": info.ref_count,
            },
        )
    )


async def _handle_has(client, command: CoreCommand, store: BlobStore) -> None:
    p = command.payload
    sha256 = p.get("sha256")
    direction = p.get("direction", "inbound")
    if not isinstance(sha256, str):
        raise ValueError("payload.sha256 must be a string")
    if direction not in ("inbound", "outbound"):
        raise ValueError("payload.direction must be 'inbound' or 'outbound'")
    blob_id = await store.has(sha256=sha256, direction=direction)
    await client.send(
        make_ack(
            "blob.has",
            request_id=command.id,
            payload={"exists": blob_id is not None, "blob_id": blob_id},
        )
    )


async def _handle_fetch(client, command: CoreCommand, store: BlobStore) -> None:
    p = command.payload
    blob_id = p.get("blob_id")
    chunk_size = p.get("chunk_size")
    if not isinstance(blob_id, str):
        raise ValueError("payload.blob_id must be a string")
    if chunk_size is not None and not isinstance(chunk_size, int):
        raise ValueError("payload.chunk_size must be an integer or null")
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError("payload.chunk_size must be > 0")

    chunks = store.fetch_chunks(blob_id, chunk_size=chunk_size)
    try:
        first_chunk = await anext(chunks)
    except StopAsyncIteration:
        first_chunk = None

    # Send an initial ack only after validating that the blob exists and the
    # chunking parameters are usable.
    await client.send(
        make_ack("blob.fetch", request_id=command.id, payload={"blob_id": blob_id})
    )

    if first_chunk is not None:
        seq, chunk = first_chunk
        await client.send(
            make_frame(
                "blob.chunk",
                {
                    "blob_id": blob_id,
                    "seq": seq,
                    "data_b64": base64.b64encode(chunk).decode("ascii"),
                },
            )
        )

    async for seq, chunk in chunks:
        await client.send(
            make_frame(
                "blob.chunk",
                {
                    "blob_id": blob_id,
                    "seq": seq,
                    "data_b64": base64.b64encode(chunk).decode("ascii"),
                },
            )
        )

    await client.send(
        make_frame("blob.complete", {"blob_id": blob_id})
    )


async def _handle_release(client, command: CoreCommand, store: BlobStore) -> None:
    blob_id = command.payload.get("blob_id")
    if not isinstance(blob_id, str):
        raise ValueError("payload.blob_id must be a string")
    new_rc = await store.release(blob_id)
    await client.send(
        make_ack("blob.release", request_id=command.id, payload={"ref_count": new_rc})
    )


async def _handle_request_retransmit(client, command: CoreCommand, store: BlobStore) -> None:
    """Reserved for future use; v1 returns not_implemented."""
    await client.send(
        make_error(
            "blob.request_retransmit is not implemented in this server.",
            request_id=command.id,
            code="not_implemented",
        )
    )


_HANDLERS = {
    "blob.upload_init": _handle_upload_init,
    "blob.upload_chunk": _handle_upload_chunk,
    "blob.upload_finalize": _handle_upload_finalize,
    "blob.info": _handle_info,
    "blob.has": _handle_has,
    "blob.fetch": _handle_fetch,
    "blob.release": _handle_release,
    "blob.request_retransmit": _handle_request_retransmit,
}
