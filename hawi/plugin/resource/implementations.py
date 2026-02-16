"""Resource factory and implementations."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Callable

from .resource import HawiResource, ResourceContent

if TYPE_CHECKING:
    pass

class HawiLiteralResource(HawiResource):
    """In-memory text resource with literal content."""

    def __init__(
        self,
        uri: str,
        name: str,
        content: str,
        description: str | None = None,
        mime_type: str = "text/plain",
    ):
        self._uri = uri
        self._name = name
        self._content = content
        self._description = description
        self._mime_type = mime_type
        self._size = len(content.encode("utf-8"))

    @property
    def uri(self) -> str:
        return self._uri

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str | None:
        return self._description

    @property
    def mime_type(self) -> str | None:
        return self._mime_type

    @property
    def size(self) -> int | None:
        return self._size

    def read(self) -> ResourceContent:
        return ResourceContent(
            uri=self._uri,
            text=self._content,
            mime_type=self._mime_type,
        )


class HawiFileResource(HawiResource):
    """File-based resource with lazy loading."""

    def __init__(
        self,
        filepath: pathlib.Path | str,
        uri: str | None = None,
        name: str | None = None,
        mime_type: str | None = None,
    ):
        self._filepath = pathlib.Path(filepath)
        self._uri = uri or f"file://{self._filepath.resolve().as_posix()}"
        self._name = name or self._filepath.stem
        self._mime_type = mime_type
        self._description: str | None = None
        self._content_cache: ResourceContent | None = None

    @property
    def uri(self) -> str:
        return self._uri

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str | None:
        if self._description is None:
            try:
                with self._filepath.open("r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    self._description = first_line[:200]
            except (OSError, UnicodeDecodeError):
                self._description = self._filepath.name
        return self._description

    @property
    def mime_type(self) -> str | None:
        if self._mime_type is None:
            self._mime_type = self._detect_mime_type()
        return self._mime_type

    @property
    def size(self) -> int | None:
        try:
            return self._filepath.stat().st_size
        except OSError:
            return None

    def _detect_mime_type(self) -> str:
        suffix = self._filepath.suffix.lower()
        mime_map = {
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".json": "application/json",
            ".yaml": "application/yaml",
            ".yml": "application/yaml",
            ".py": "text/x-python",
            ".js": "application/javascript",
            ".ts": "application/typescript",
            ".html": "text/html",
            ".css": "text/css",
            ".xml": "application/xml",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".svg": "image/svg+xml",
            ".pdf": "application/pdf",
        }
        return mime_map.get(suffix, "application/octet-stream")

    def _is_text_file(self) -> bool:
        mime = self.mime_type or ""
        return mime.startswith("text/") or mime in (
            "application/json",
            "application/yaml",
            "application/javascript",
            "application/typescript",
            "application/xml",
        )

    def read(self) -> ResourceContent:
        if self._content_cache is not None:
            return self._content_cache

        if self._is_text_file():
            try:
                text = self._filepath.read_text(encoding="utf-8")
                self._content_cache = ResourceContent(
                    uri=self._uri,
                    text=text,
                    mime_type=self.mime_type,
                )
            except UnicodeDecodeError:
                # Fall back to binary
                blob = self._filepath.read_bytes()
                self._content_cache = ResourceContent(
                    uri=self._uri,
                    blob=blob,
                    mime_type=self.mime_type,
                )
        else:
            blob = self._filepath.read_bytes()
            self._content_cache = ResourceContent(
                uri=self._uri,
                blob=blob,
                mime_type=self.mime_type,
            )

        return self._content_cache


class HawiDynamicResource(HawiResource):
    """Dynamically generated resource via callback function."""

    def __init__(
        self,
        uri: str,
        name: str,
        description: str,
        generator: Callable[[], ResourceContent],
        mime_type: str | None = None,
    ):
        self._uri = uri
        self._name = name
        self._description = description
        self._generator = generator
        self._mime_type = mime_type

    @property
    def uri(self) -> str:
        return self._uri

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str | None:
        return self._description

    @property
    def mime_type(self) -> str | None:
        return self._mime_type

    @property
    def size(self) -> int | None:
        return None  # Dynamic, size unknown until read

    def read(self) -> ResourceContent:
        return self._generator()
