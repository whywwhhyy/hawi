"""Resolve Hawi blob media references into provider-ready model inputs."""

from __future__ import annotations

import base64
from copy import deepcopy
from typing import Any, Mapping, cast

from hawi.models.message import BLOB_URI_SCHEME, ContentPart, Message, MessageRequest

from .store import BlobInfo, BlobStore


_MEDIA_PART_TYPES = {"image", "document", "audio", "video", "file"}
_REFERENCE_SOURCE_KEYS = {
    "kind",
    "uri",
    "blob_id",
    "sha256",
    "direction",
    "size",
}


async def resolve_blob_references_for_model(
    request: MessageRequest,
    store: BlobStore,
) -> MessageRequest:
    """Return a request copy with blob/data_uri media lowered for model adapters.

    Agent context and UI events keep compact ``hawi-blob://`` references. This
    resolver runs only at the model boundary, where adapters currently expect
    URL/data/base64 fields.
    """
    messages: list[Message] = []
    changed = False
    for message in request.messages:
        resolved_message, message_changed = await _resolve_message(message, store)
        messages.append(resolved_message)
        changed = changed or message_changed

    system = request.system
    if system is not None:
        system, system_changed = await _resolve_content(system, store)
        changed = changed or system_changed

    if not changed:
        return request
    return request.model_copy(
        deep=True,
        update={"messages": messages, "system": system},
    )


async def _resolve_message(
    message: Message,
    store: BlobStore,
) -> tuple[Message, bool]:
    content = message.get("content")
    if not isinstance(content, list):
        return deepcopy(message), False

    resolved_content, changed = await _resolve_content(content, store)
    if not changed:
        return deepcopy(message), False

    resolved = deepcopy(message)
    resolved["content"] = resolved_content
    return resolved, True


async def _resolve_content(
    content: list[ContentPart],
    store: BlobStore,
) -> tuple[list[ContentPart], bool]:
    resolved: list[ContentPart] = []
    changed = False
    for part in content:
        resolved_part, part_changed = await _resolve_part(part, store)
        resolved.append(resolved_part)
        changed = changed or part_changed
    return resolved, changed


async def _resolve_part(
    part: ContentPart,
    store: BlobStore,
) -> tuple[ContentPart, bool]:
    if not isinstance(part, dict):
        return part, False

    part_type = str(part.get("type") or "")
    if part_type in _MEDIA_PART_TYPES and isinstance(part.get("source"), Mapping):
        return await _resolve_media_part(part, part_type, store)

    if part_type in {"steer", "tool_result"}:
        nested = part.get("content")
        if isinstance(nested, list):
            resolved_nested, changed = await _resolve_content(
                cast(list[ContentPart], nested),
                store,
            )
            if changed:
                resolved = deepcopy(part)
                resolved["content"] = resolved_nested
                return cast(ContentPart, resolved), True

    return deepcopy(part), False


async def _resolve_media_part(
    part: Mapping[str, Any],
    part_type: str,
    store: BlobStore,
) -> tuple[ContentPart, bool]:
    source = dict(cast(Mapping[str, Any], part["source"]))
    blob_id = _blob_id_from_source(source)
    data_uri = source.get("data_uri")

    if blob_id:
        info = await store.info(blob_id)
        data = await _read_blob(store, blob_id)
        mime = _mime_for(part_type, source, info)
        b64 = base64.b64encode(data).decode("ascii")
        data_uri = f"data:{mime};base64,{b64}"
    elif isinstance(data_uri, str) and data_uri.startswith("data:"):
        info = None
        mime = _mime_for(part_type, source, None, data_uri=data_uri)
        b64 = _base64_from_data_uri(data_uri)
    elif not source.get("url") and isinstance(source.get("uri"), str):
        resolved = deepcopy(part)
        resolved_source = dict(source)
        resolved_source["url"] = source["uri"]
        resolved["source"] = resolved_source
        return cast(ContentPart, resolved), True
    else:
        return cast(ContentPart, deepcopy(part)), False

    if part_type == "audio":
        resolved = deepcopy(part)
        resolved_source = _provider_source(source)
        resolved_source["data"] = b64
        resolved_source["url"] = data_uri
        resolved_source["mime_type"] = mime
        resolved_source.setdefault("format", _format_from_mime(mime, "wav"))
        resolved["source"] = resolved_source
        return cast(ContentPart, resolved), True

    resolved_source = _provider_source(source)
    resolved_source["url"] = data_uri
    resolved_source["mime_type"] = mime
    if part_type == "video":
        resolved_source["data"] = b64
        resolved_source.setdefault("format", _format_from_mime(mime, "mp4"))

    if part_type == "file":
        title = part.get("title") or source.get("filename") or "File"
        return cast(
            ContentPart,
            {
                "type": "document",
                "source": resolved_source,
                "title": title,
                "context": part.get("context"),
            },
        ), True

    resolved = deepcopy(part)
    resolved["source"] = resolved_source
    return cast(ContentPart, resolved), True


async def _read_blob(store: BlobStore, blob_id: str) -> bytes:
    chunks: list[bytes] = []
    async for _, chunk in store.fetch_chunks(blob_id):
        chunks.append(chunk)
    return b"".join(chunks)


def _blob_id_from_source(source: Mapping[str, Any]) -> str | None:
    blob_id = source.get("blob_id")
    if isinstance(blob_id, str) and blob_id:
        return blob_id

    for key in ("uri", "url"):
        value = source.get(key)
        if isinstance(value, str) and value.startswith(BLOB_URI_SCHEME):
            return value.removeprefix(BLOB_URI_SCHEME)
    return None


def _provider_source(source: Mapping[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in source.items() if key not in _REFERENCE_SOURCE_KEYS}


def _mime_for(
    part_type: str,
    source: Mapping[str, Any],
    info: BlobInfo | None,
    *,
    data_uri: str | None = None,
) -> str:
    explicit = source.get("mime_type") or source.get("mime") or (info.mime if info else None)
    if isinstance(explicit, str) and explicit:
        return explicit
    if data_uri:
        prefix = data_uri.split(",", 1)[0]
        if prefix.startswith("data:"):
            media_type = prefix[5:].split(";", 1)[0]
            if media_type:
                return media_type
    return {
        "image": "image/png",
        "document": "application/octet-stream",
        "audio": "audio/wav",
        "video": "video/mp4",
        "file": "application/octet-stream",
    }.get(part_type, "application/octet-stream")


def _base64_from_data_uri(data_uri: str) -> str:
    if "," not in data_uri:
        return data_uri
    return data_uri.split(",", 1)[1]


def _format_from_mime(mime: str, default: str) -> str:
    if "/" not in mime:
        return default
    suffix = mime.split("/", 1)[1].split(";", 1)[0].lower()
    return {
        "mpeg": "mp3",
        "x-wav": "wav",
        "quicktime": "mov",
        "3gpp": "three_gp",
    }.get(suffix, suffix or default)
