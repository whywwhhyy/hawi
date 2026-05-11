"""Agent configuration types."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, Mapping

from hawi.errors import ModelErrorType

from .context import CONTEXT_COMPACTION_PROMPT, CONTEXT_COMPACTION_SUMMARY_PREFIX


@dataclass
class ModelErrorPolicy:
    """Model failure handling strategy."""

    action: Literal[
        "retry",
        "notify_agent",
        "stop",
    ]


class ModelErrorRetryPolicy(ModelErrorPolicy):
    def __init__(self, retry_count: int):
        super().__init__("retry")
        self.retry_count: int = retry_count


class ModelErrorNotifyPolicy(ModelErrorPolicy):
    def __init__(self):
        super().__init__("notify_agent")


class ModelErrorStopPolicy(ModelErrorPolicy):
    def __init__(self):
        super().__init__("stop")


ModelErrorPolicyConfig = Mapping[ModelErrorType, ModelErrorPolicy]


def default_model_error_policy() -> ModelErrorPolicyConfig:
    return defaultdict(
        ModelErrorStopPolicy,
        {
            "network": ModelErrorRetryPolicy(retry_count=10),
            "throttle": ModelErrorRetryPolicy(retry_count=3),
        },
    )


@dataclass
class AutoCompactConfig:
    """Configuration for automatic context compaction."""

    enabled: bool = True
    max_context_tokens: int = 128_000
    trigger_tokens: int | None = None
    trigger_ratio: float = 0.8
    keep_last_messages: int = 8
    min_messages: int = 12
    summary_max_output_tokens: int = 2048
    max_transcript_chars: int = 120_000
    prompt: str = CONTEXT_COMPACTION_PROMPT
    summary_prefix: str = CONTEXT_COMPACTION_SUMMARY_PREFIX

    def token_limit(self) -> int:
        """Return the estimated-token threshold that triggers compaction."""
        if self.trigger_tokens is not None:
            return self.trigger_tokens
        return max(1, int(self.max_context_tokens * self.trigger_ratio))
