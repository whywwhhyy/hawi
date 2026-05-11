"""Helpers for provider model-list HTTP endpoints."""

from __future__ import annotations

from typing import Any

import httpx

from hawi.errors import (
    DeniedError,
    NetworkError,
    RemoteError,
    ThrottleError,
    ValidationError,
)
from hawi.models.model import Model


def bearer_auth_headers(api_key: str | None) -> dict[str, str]:
    """Build Bearer auth headers when an API key is configured."""
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def fetch_json_model_ids(
    url: str,
    *,
    provider: str,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    timeout: float = 60.0,
    paginate: bool = False,
) -> list[str]:
    """Fetch model IDs from a JSON endpoint with optional Anthropic pagination."""
    return _collect_paginated_ids(
        lambda request_params: _fetch_json(
            url,
            provider=provider,
            headers=headers,
            params=request_params,
            timeout=timeout,
        ),
        params=params,
        paginate=paginate,
    )


async def afetch_json_model_ids(
    url: str,
    *,
    provider: str,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    timeout: float = 60.0,
    paginate: bool = False,
) -> list[str]:
    """Async variant of :func:`fetch_json_model_ids`."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await _acollect_paginated_ids(
            lambda request_params: _afetch_json(
                client,
                url,
                provider=provider,
                headers=headers,
                params=request_params,
            ),
            params=params,
            paginate=paginate,
        )


def _collect_paginated_ids(
    fetch_page,
    *,
    params: dict[str, Any] | None,
    paginate: bool,
) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    request_params = dict(params or {})

    for _ in range(100):
        data = fetch_page(dict(request_params))
        _append_model_ids(ids, seen, data)

        if not paginate or not isinstance(data, dict) or not data.get("has_more"):
            break
        cursor = data.get("last_id")
        if not cursor or cursor == request_params.get("after_id"):
            break
        request_params["after_id"] = cursor

    return ids


async def _acollect_paginated_ids(
    fetch_page,
    *,
    params: dict[str, Any] | None,
    paginate: bool,
) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    request_params = dict(params or {})

    for _ in range(100):
        data = await fetch_page(dict(request_params))
        _append_model_ids(ids, seen, data)

        if not paginate or not isinstance(data, dict) or not data.get("has_more"):
            break
        cursor = data.get("last_id")
        if not cursor or cursor == request_params.get("after_id"):
            break
        request_params["after_id"] = cursor

    return ids


def _append_model_ids(ids: list[str], seen: set[str], data: Any) -> None:
    for model_id in Model._coerce_model_id_list(data):
        if model_id not in seen:
            ids.append(model_id)
            seen.add(model_id)


def _fetch_json(
    url: str,
    *,
    provider: str,
    headers: dict[str, str] | None,
    params: dict[str, Any] | None,
    timeout: float,
) -> Any:
    try:
        response = httpx.get(
            url,
            headers=headers,
            params=params,
            timeout=timeout,
        )
        return _json_or_raise(response, provider=provider)
    except httpx.HTTPStatusError as exc:
        raise _convert_status_error(exc, provider=provider) from exc
    except httpx.RequestError as exc:
        raise NetworkError(f"{provider} model list network error: {exc}") from exc


async def _afetch_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    provider: str,
    headers: dict[str, str] | None,
    params: dict[str, Any] | None,
) -> Any:
    try:
        response = await client.get(url, headers=headers, params=params)
        return _json_or_raise(response, provider=provider)
    except httpx.HTTPStatusError as exc:
        raise _convert_status_error(exc, provider=provider) from exc
    except httpx.RequestError as exc:
        raise NetworkError(f"{provider} model list network error: {exc}") from exc


def _json_or_raise(response: httpx.Response, *, provider: str) -> Any:
    response.raise_for_status()
    try:
        return response.json()
    except ValueError as exc:
        raise ValidationError(f"{provider} model list returned invalid JSON") from exc


def _convert_status_error(
    exc: httpx.HTTPStatusError,
    *,
    provider: str,
) -> Exception:
    status_code = exc.response.status_code
    message = _response_error_message(exc.response)
    prefix = f"{provider} model list failed: HTTP {status_code}"
    if message:
        prefix = f"{prefix}: {message}"

    if status_code in (401, 403):
        return DeniedError(prefix)
    if status_code == 429:
        return ThrottleError(prefix)
    if 500 <= status_code < 600:
        return RemoteError(prefix)
    return ValidationError(prefix)


def _response_error_message(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text[:500]

    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error.get("code") or error)[:500]
        if error:
            return str(error)[:500]
        message = data.get("message")
        if message:
            return str(message)[:500]
    return str(data)[:500]
