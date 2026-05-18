"""Runtime review primitives shared by engine, agent, and plugins."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RuntimeReviewDecision:
    """A generic approve/reject decision produced by a human or reviewer."""

    approved: bool
    feedback: str = ""
    modified_output: str | None = None
    next_step_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeReviewRequest:
    """One pending review wait point."""

    review_id: str
    plugin_id: str
    review_type: str
    payload: dict[str, Any]
    future: asyncio.Future[Any]
    created_at: float = field(default_factory=time.time)


class RuntimeReviewBroker:
    """Coordinate blocking review requests without blocking the engine loop."""

    def __init__(self) -> None:
        self._pending: dict[str, RuntimeReviewRequest] = {}

    def create(
        self,
        review_id: str,
        *,
        plugin_id: str,
        review_type: str,
        payload: dict[str, Any] | None = None,
    ) -> RuntimeReviewRequest:
        existing = self._pending.get(review_id)
        if existing is not None and not existing.future.done():
            return existing

        loop = asyncio.get_running_loop()
        request = RuntimeReviewRequest(
            review_id=review_id,
            plugin_id=plugin_id,
            review_type=review_type,
            payload=dict(payload or {}),
            future=loop.create_future(),
        )
        self._pending[review_id] = request
        return request

    def get(self, review_id: str) -> RuntimeReviewRequest | None:
        request = self._pending.get(review_id)
        if request is not None and request.future.done():
            self._pending.pop(review_id, None)
            return None
        return request

    def has(self, review_id: str) -> bool:
        return self.get(review_id) is not None

    def pending_ids(self) -> list[str]:
        return list(self._pending)

    async def wait(self, review_id: str) -> Any:
        request = self.get(review_id)
        if request is None:
            raise KeyError(f"No pending review {review_id!r}.")
        return await request.future

    def resolve(self, review_id: str, decision: RuntimeReviewDecision) -> bool:
        request = self.get(review_id)
        if request is None or request.future.done():
            return False
        request.future.set_result(decision)
        return True

    def cancel(self, review_id: str) -> None:
        request = self._pending.pop(review_id, None)
        if request is not None and not request.future.done():
            request.future.cancel()

    def discard(self, review_id: str) -> None:
        self._pending.pop(review_id, None)
