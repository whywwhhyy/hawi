"""Provider-level token estimation helpers for Kimi/Moonshot models."""

from __future__ import annotations

from typing import Any

import httpx

from hawi.models.message import MessageRequest
from hawi.models.model import TokenEstimate


class KimiTokenEstimateMixin:
    """Use Kimi/Moonshot's official token estimate endpoint."""

    api_key: str | None
    base_url: str
    timeout: float
    model_id: str
    token_estimate_base_url: str | None

    def _estimate_tokens_impl(
        self,
        request: MessageRequest,
    ) -> TokenEstimate:
        if not self.api_key:
            raise RuntimeError("API key is required for Kimi token estimation")

        req = self._prepare_kimi_token_estimate_request(request)
        url = f"{self._kimi_token_estimate_base_url().rstrip('/')}/tokenizers/estimate-token-count"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            response = httpx.post(
                url,
                headers=headers,
                json=req,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Kimi token estimate failed: HTTP {e.response.status_code}"
            ) from e
        except httpx.RequestError as e:
            raise RuntimeError(
                f"Kimi token estimate failed: network error - {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Kimi token estimate failed: {e}") from e

        return self._parse_kimi_token_estimate_response(data)

    async def _aestimate_tokens_impl(
        self,
        request: MessageRequest,
    ) -> TokenEstimate:
        if not self.api_key:
            raise RuntimeError("API key is required for Kimi token estimation")

        req = self._prepare_kimi_token_estimate_request(request)
        url = f"{self._kimi_token_estimate_base_url().rstrip('/')}/tokenizers/estimate-token-count"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, headers=headers, json=req)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Kimi token estimate failed: HTTP {e.response.status_code}"
            ) from e
        except httpx.RequestError as e:
            raise RuntimeError(
                f"Kimi token estimate failed: network error - {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Kimi token estimate failed: {e}") from e

        return self._parse_kimi_token_estimate_response(data)

    def _prepare_kimi_token_estimate_request(
        self,
        request: MessageRequest,
    ) -> dict[str, Any]:
        req = self._prepare_request_impl(request)
        extra_body = req.pop("extra_body", None)
        if isinstance(extra_body, dict):
            req.update(extra_body)
        req.pop("stream", None)
        req.pop("stream_options", None)
        return req

    def _kimi_token_estimate_base_url(self) -> str:
        configured = getattr(self, "token_estimate_base_url", None)
        return configured or self.base_url

    def _parse_kimi_token_estimate_response(
        self,
        response: dict[str, Any],
    ) -> TokenEstimate:
        code = response.get("code")
        if code not in (None, 0):
            raise RuntimeError(f"Kimi token estimate failed: API error code {code}")
        data = response.get("data") if isinstance(response.get("data"), dict) else response
        raw_tokens = (data or {}).get("total_tokens")
        if raw_tokens is None:
            raise RuntimeError("Kimi token estimate failed: missing total_tokens")
        tokens = int(raw_tokens)
        return TokenEstimate(
            input_tokens=tokens,
            context_tokens=tokens,
            total_tokens=tokens,
            method="provider_count",
            confidence="exact",
            provider="kimi",
            model_id=self.model_id,
            details=response,
        )

