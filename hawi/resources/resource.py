"""Core resource types and protocol definitions."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class HawiResource(Protocol):
    """Protocol for resources compatible with MCP Resource specification.

    Resources are uniquely identified by URI and can contain text or binary data.
    """

    @property
    def uri(self) -> str:
        """Unique URI identifier for this resource (e.g., 'file:///prompts/system.txt')."""
        ...

    @property
    def name(self) -> str:
        """Short name for display purposes."""
        ...

    @property
    def description(self) -> str | None:
        """Optional human-readable description."""
        ...

    @property
    def mime_type(self) -> str | None:
        """Optional MIME type (e.g., 'text/plain', 'application/json')."""
        ...

    @property
    def size(self) -> int | None:
        """Optional size in bytes."""
        ...

    def read(self) -> ResourceContent:
        """Read and return the resource content."""
        ...


class ResourceContent:
    """Content of a resource - can be text or binary."""

    def __init__(
        self,
        uri: str,
        text: str | None = None,
        blob: bytes | None = None,
        mime_type: str | None = None,
    ):
        if text is None and blob is None:
            raise ValueError("Either text or blob must be provided")
        if text is not None and blob is not None:
            raise ValueError("Cannot provide both text and blob")

        self.uri = uri
        self.text = text
        self.blob = blob
        self.mime_type = mime_type

    @property
    def is_text(self) -> bool:
        return self.text is not None

    @property
    def is_binary(self) -> bool:
        return self.blob is not None

    def get_text(self) -> str:
        if self.text is None:
            raise ValueError("Resource is binary, not text")
        return self.text

    def get_bytes(self) -> bytes:
        if self.blob is not None:
            return self.blob
        if self.text is not None:
            return self.text.encode("utf-8")
        raise ValueError("Resource has no content")

    def to_dict(self) -> dict[str, Any]:
        """Convert to MCP-compatible dict format."""
        result: dict[str, Any] = {"uri": self.uri}
        if self.mime_type:
            result["mimeType"] = self.mime_type
        if self.text is not None:
            result["text"] = self.text
        elif self.blob is not None:
            import base64
            result["blob"] = base64.b64encode(self.blob).decode("ascii")
        return result
