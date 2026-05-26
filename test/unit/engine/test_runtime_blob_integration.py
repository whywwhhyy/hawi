"""Runtime-level coverage for the blob protocol wiring."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from hawi.engine.blob.store import BlobStore
from hawi.engine.blob.resolver import resolve_blob_references_for_model
from hawi.engine.protocol import VERSION
from hawi.engine.runtime import CoreRuntime
from hawi.models.message import MessageRequest, blob_source


@dataclass(eq=False)
class _Client:
    id: str = "client"
    authenticated: bool = True
    negotiated_caps: set[str] = field(default_factory=set)
    sent: list[dict[str, Any]] = field(default_factory=list)

    async def send(self, frame: dict[str, Any]) -> None:
        self.sent.append(frame)

    async def close(self) -> None:
        return None


@pytest.fixture
async def blob_store(tmp_path: Path) -> BlobStore:
    store = BlobStore(root=tmp_path / ".hawi" / "blobs", quota_bytes=4096, chunk_size=8)
    await store.start()
    yield store
    await store.close()


def _frame(command_type: str, request_id: str, payload: dict[str, Any]) -> str:
    return json.dumps(
        {
            "version": VERSION,
            "type": command_type,
            "id": request_id,
            "payload": payload,
        }
    )


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def _store_blob(
    store: BlobStore,
    body: bytes,
    *,
    mime: str | None = None,
) -> str:
    blob_id = await store.upload_init(
        direction="inbound",
        sha256=_sha(body),
        size=len(body),
        mime=mime,
    )
    await store.upload_chunk(blob_id, 0, body)
    await store.upload_finalize(blob_id)
    return blob_id


async def test_hello_advertises_blob_v1_only_when_store_enabled(blob_store: BlobStore) -> None:
    disabled = CoreRuntime(model_name="test-model", token=None)
    disabled_client = _Client(authenticated=False)

    await disabled.handle_frame(
        disabled_client,
        _frame("hello", "h-disabled", {"client_caps": ["blob_v1", "tlv_v1"]}),
    )

    disabled_ack = next(f for f in disabled_client.sent if f["type"] == "ack")
    assert "blob_v1" not in disabled_ack["payload"]["server_caps"]
    assert disabled_ack["payload"]["negotiated"] == ["tlv_v1"]

    enabled = CoreRuntime(model_name="test-model", token=None, blob_store=blob_store)
    enabled_client = _Client(authenticated=False)

    await enabled.handle_frame(
        enabled_client,
        _frame("hello", "h-enabled", {"client_caps": ["blob_v1", "tlv_v1"]}),
    )

    enabled_ack = next(f for f in enabled_client.sent if f["type"] == "ack")
    assert "blob_v1" in enabled_ack["payload"]["server_caps"]
    assert enabled_ack["payload"]["negotiated"] == ["blob_v1", "tlv_v1"]
    assert enabled_client.negotiated_caps == {"blob_v1", "tlv_v1"}


async def test_runtime_rejects_blob_command_when_store_disabled() -> None:
    runtime = CoreRuntime(model_name="test-model", token=None)
    client = _Client(authenticated=True)

    await runtime.handle_frame(
        client,
        _frame("blob.has", "has-1", {"direction": "inbound", "sha256": "0" * 64}),
    )

    assert client.sent[-1]["type"] == "error"
    assert client.sent[-1]["id"] == "has-1"
    assert client.sent[-1]["payload"]["code"] == "blob_disabled"


async def test_runtime_dispatches_blob_upload_fetch_roundtrip(blob_store: BlobStore) -> None:
    runtime = CoreRuntime(model_name="test-model", token=None, blob_store=blob_store)
    client = _Client(authenticated=True)
    body = b"runtime blob bytes"
    sha = _sha(body)

    await runtime.handle_frame(
        client,
        _frame(
            "blob.upload_init",
            "init-1",
            {"direction": "inbound", "sha256": sha, "size": len(body), "mime": None},
        ),
    )
    init_ack = next(f for f in client.sent if f["id"] == "init-1" and f["type"] == "ack")
    blob_id = init_ack["payload"]["blob_id"]

    await runtime.handle_frame(
        client,
        _frame(
            "blob.upload_chunk",
            "chunk-1",
            {
                "blob_id": blob_id,
                "seq": 0,
                "data_b64": base64.b64encode(body).decode("ascii"),
            },
        ),
    )
    await runtime.handle_frame(
        client,
        _frame("blob.upload_finalize", "finalize-1", {"blob_id": blob_id}),
    )
    await runtime.handle_frame(
        client,
        _frame("blob.fetch", "fetch-1", {"blob_id": blob_id, "chunk_size": 5}),
    )

    fetch_ack = next(f for f in client.sent if f["id"] == "fetch-1" and f["type"] == "ack")
    chunks = [f for f in client.sent if f["type"] == "blob.chunk"]
    complete = next(f for f in client.sent if f["type"] == "blob.complete")

    assert fetch_ack["payload"]["blob_id"] == blob_id
    assert b"".join(base64.b64decode(c["payload"]["data_b64"]) for c in chunks) == body
    assert complete["payload"]["blob_id"] == blob_id


async def test_runtime_blob_command_requires_authentication(blob_store: BlobStore) -> None:
    runtime = CoreRuntime(model_name="test-model", token="secret", blob_store=blob_store)
    client = _Client(authenticated=False)

    await runtime.handle_frame(
        client,
        _frame("blob.has", "has-1", {"direction": "inbound", "sha256": "0" * 64}),
    )

    assert client.sent[-1]["type"] == "error"
    assert client.sent[-1]["payload"]["code"] == "unauthenticated"


async def test_blob_resolver_lowers_blob_source_without_mutating_request(
    blob_store: BlobStore,
) -> None:
    blob_id = await _store_blob(blob_store, b"image bytes", mime="image/png")
    request = MessageRequest(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": blob_source(
                            blob_id,
                            mime_type="image/png",
                            filename="screen.png",
                        ),
                    }
                ],
                "name": None,
                "metadata": None,
            }
        ]
    )

    resolved = await resolve_blob_references_for_model(request, blob_store)

    original_source = request.messages[0]["content"][0]["source"]
    resolved_source = resolved.messages[0]["content"][0]["source"]
    assert original_source["blob_id"] == blob_id
    assert resolved_source["url"].startswith("data:image/png;base64,")
    assert resolved_source["filename"] == "screen.png"
    assert "blob_id" not in resolved_source
    MessageRequest(messages=resolved.messages)


async def test_blob_resolver_downgrades_non_image_file_to_placeholder(
    blob_store: BlobStore,
) -> None:
    blob_id = await _store_blob(blob_store, b"%PDF-1.7", mime="application/pdf")
    request = MessageRequest(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "please inspect"},
                    {
                        "type": "file",
                        "source": blob_source(
                            blob_id,
                            mime_type="application/pdf",
                            filename="paper.pdf",
                        ),
                    },
                ],
                "name": None,
                "metadata": None,
            }
        ]
    )

    resolved = await resolve_blob_references_for_model(request, blob_store)

    assert request.messages[0]["content"][1]["source"]["blob_id"] == blob_id
    assert resolved.messages[0]["content"] == [
        {"type": "text", "text": "please inspect"},
        {"type": "text", "text": "[file attachment: paper.pdf; application/pdf]"},
    ]
    MessageRequest(messages=resolved.messages)
