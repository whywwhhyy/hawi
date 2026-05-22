"""Authentication helpers shared by model adapters."""

from __future__ import annotations

DUMMY_API_KEY = "hawi-dummy-api-key"


def normalize_optional_api_key(api_key: str | None) -> str | None:
    """Treat blank API keys as intentionally absent."""
    if api_key is None:
        return None
    if str(api_key).strip() == "":
        return None
    return api_key


def sdk_api_key(
    api_key: str | None,
    *,
    base_url: str | None,
    explicit_api_key: bool,
) -> str | None:
    """Return the key value to pass into SDK constructors.

    OpenAI-compatible SDKs often read API keys from environment variables when
    ``api_key`` is omitted. For local/custom endpoints, a dummy key avoids that
    fallback and keeps SDK construction compatible with APIs that ignore auth.
    """
    if api_key is not None:
        return api_key
    if explicit_api_key or base_url:
        return DUMMY_API_KEY
    return None
