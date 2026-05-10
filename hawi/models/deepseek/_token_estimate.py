"""Provider-level token estimation helpers for DeepSeek models."""

from __future__ import annotations

from hawi.models.message import MessageRequest
from hawi.models.model import TokenEstimate


class DeepSeekTokenEstimateMixin:
    """DeepSeek has no public online count endpoint; use marked heuristic."""

    def _estimate_tokens_impl(
        self,
        request: MessageRequest,
    ) -> TokenEstimate:
        estimate = self._heuristic_token_estimate(request)
        estimate.provider = "deepseek"
        estimate.details["provider_count_endpoint"] = "not_available_in_official_docs"
        estimate.details["recommended_exact_source"] = "response.usage"
        return estimate

    async def _aestimate_tokens_impl(
        self,
        request: MessageRequest,
    ) -> TokenEstimate:
        return self._estimate_tokens_impl(request)

