"""Development-time invariant checks.

In ``--debug`` / ``HAWI_DEBUG=1`` mode these behave like Python ``assert``
and raise ``AssertionError`` so failures are loud and fail-fast.

In normal (production) mode they log an ERROR and silently return, allowing
the agent to attempt a graceful degradation rather than crashing.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_DEBUG_FLAG: bool | None = None


def _is_debug() -> bool:
    """Check whether the runtime is in debug mode (lazily cached)."""
    global _DEBUG_FLAG
    if _DEBUG_FLAG is None:
        env_val = os.environ.get("HAWI_DEBUG", "").strip().lower()
        _DEBUG_FLAG = env_val in {"1", "true", "yes", "on"}
    return _DEBUG_FLAG


def debug_assert(condition: bool, message: str) -> None:
    """Raise AssertionError in debug mode; log and suppress otherwise.

    Use this for invariants that *should* always hold but whose violation
    does not warrant a hard crash in production.  The ERROR log records the
    violation so operators can investigate.
    """
    if condition:
        return
    if _is_debug():
        raise AssertionError(message)
    logger.error("Invariant check failed (suppressed): %s", message)
