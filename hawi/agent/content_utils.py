"""Content rendering helpers for agent internals."""

from __future__ import annotations

from typing import Iterable, cast

from hawi.models import ContentPart
from hawi.tool.types import ToolResult


def normalize_content_parts(content: str | list[ContentPart]) -> list[ContentPart]:
    """Normalize content input into a list of ContentPart."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return list(content)


def merge_content_parts(
    contents: Iterable[str | list[ContentPart]],
    *,
    separator: str = "\n\n",
) -> list[ContentPart]:
    """Merge multiple user content payloads into one content-part list."""
    merged: list[ContentPart] = []
    for content in contents:
        if merged and separator:
            merged.append({"type": "text", "text": separator})
        merged.extend(normalize_content_parts(content))
    return merged


def truncate_preview(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def content_preview(content: str | list[ContentPart], max_length: int = 100) -> str:
    """Build a compact preview for text and referenced media content."""
    if isinstance(content, str):
        return truncate_preview(content, max_length)
    preview = serialize_content_parts(content)
    return truncate_preview(preview, max_length)


def serialize_content_parts(content: list[ContentPart]) -> str:
    """Serialize content parts into readable plain text."""
    chunks: list[str] = []
    for part in content:
        part_type = part.get("type")
        if part_type == "text":
            chunks.append(part.get("text", ""))
        elif part_type == "reasoning":
            chunks.append(part.get("reasoning") or "")
        elif part_type == "steer":
            nested_content = part.get("content", [])
            if isinstance(nested_content, list):
                nested_text = serialize_content_parts(
                    cast(list[ContentPart], nested_content)
                )
            else:
                nested_text = str(nested_content)
            if nested_text:
                chunks.append(nested_text)
        elif part_type == "tool_result":
            nested_content = part.get("content", [])
            if isinstance(nested_content, str):
                chunks.append(nested_content)
            elif isinstance(nested_content, list):
                nested_text = serialize_content_parts(
                    cast(list[ContentPart], nested_content)
                )
                if nested_text:
                    chunks.append(nested_text)
            else:
                chunks.append(str(nested_content))
        elif part_type in {"image", "document", "audio", "video", "file"}:
            chunks.append(_media_part_preview(part))
        else:
            chunks.append(str(part))
    return "\n".join(chunk for chunk in chunks if chunk.strip())


def _media_part_preview(part: ContentPart) -> str:
    part_type = str(part.get("type") or "media")
    source = part.get("source")
    source = source if isinstance(source, dict) else {}
    title = part.get("title") if isinstance(part, dict) else None
    filename = source.get("filename") or title
    mime_type = source.get("mime_type") or source.get("format")
    uri = (
        source.get("uri")
        or source.get("url")
        or source.get("data_uri")
        or source.get("path")
        or source.get("file_id")
        or source.get("blob_id")
        or ""
    )
    if isinstance(uri, str) and uri.startswith("data:"):
        uri = uri.split(",", 1)[0] + ",..."
    label_parts = [part_type]
    if filename:
        label_parts.append(str(filename))
    if mime_type:
        label_parts.append(str(mime_type))
    if uri:
        label_parts.append(str(uri))
    return "[" + ": ".join(label_parts) + "]"


def tool_result_content(result: ToolResult) -> str:
    """Render a ToolResult exactly as it is written into model context."""
    output_str = (
        result.output
        if isinstance(result.output, str)
        else str(result.output)
        if result.output
        else ""
    )
    if not result.success and result.error:
        result_content = f"Error: {result.error}"
        if output_str:
            result_content = f"Output before error:\n{output_str}\n\n{result_content}"
        return result_content
    return output_str
