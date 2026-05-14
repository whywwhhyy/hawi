"""Context compaction behavior for HawiAgent."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import Literal, Protocol, cast

from hawi.events import (
    AgentCompactStartEvent,
    AgentCompactStopEvent,
    Event,
    EventBus,
)
from hawi.models import ContentPart, Model
from hawi.models.message import Message

from .config import AutoCompactConfig
from .context import (
    AgentContext,
    ContextCompactionRecord,
    ContextUsageSnapshot,
    estimate_text_tokens,
)
from .state import _ExecutionState


COMPACTION_TEXT_PART_MAX_CHARS = 2_000
COMPACTION_TOOL_RESULT_MAX_CHARS = 3_000


class CompactionOwner(Protocol):
    _auto_compact: AutoCompactConfig
    _context: AgentContext
    _default_model: Model

    @property
    def has_active_tool_calls(self) -> bool: ...

    def _serialize_content_parts(self, content: list[ContentPart]) -> str: ...

    def _refresh_context_usage_snapshot(
        self,
        model: Model | None = None,
        *,
        preserve_provider: bool = True,
    ) -> ContextUsageSnapshot: ...

    async def _emit_event(
        self,
        event: Event,
        event_bus: EventBus | None,
    ) -> Event: ...


class AgentCompactor:
    """Explicit context compaction component owned by HawiAgent."""

    def __init__(self, owner: CompactionOwner) -> None:
        self._owner = owner

    def compact(
        self,
        *,
        model: Model | None = None,
        prompt: str | None = None,
        keep_last_messages: int | None = None,
    ) -> ContextCompactionRecord | None:
        """Synchronously compact the current conversation context."""
        return asyncio.run(
            self.acompact(
                model=model,
                prompt=prompt,
                keep_last_messages=keep_last_messages,
            )
        )

    async def acompact(
        self,
        *,
        model: Model | None = None,
        prompt: str | None = None,
        keep_last_messages: int | None = None,
        config: AutoCompactConfig | None = None,
        event_bus: EventBus | None = None,
        run_id: str | None = None,
        mode: Literal["manual", "auto"] = "manual",
    ) -> ContextCompactionRecord | None:
        """Compact older context into a model-generated handoff summary."""
        owner = self._owner
        cfg = config or owner._auto_compact
        keep_last = (
            keep_last_messages
            if keep_last_messages is not None
            else cfg.keep_last_messages
        )
        tail_start = owner._context.compaction_tail_start(keep_last)
        if tail_start <= 0:
            return None
        messages_to_compact = list(owner._context.messages[:tail_start])

        tokens_before = owner._context.estimate_tokens()
        message_count_before = len(owner._context.messages)
        started_at = time.time()
        await owner._emit_event(
            AgentCompactStartEvent.create(
                run_id=run_id,
                mode=mode,
                keep_last_messages=keep_last,
                tokens_before=tokens_before,
                message_count_before=message_count_before,
            ),
            event_bus,
        )

        try:
            summary = await self._generate_compaction_summary(
                model or owner._default_model,
                messages=messages_to_compact,
                prompt=prompt or cfg.prompt,
                compression_budget=cfg.compression_budget,
                max_output_tokens=cfg.summary_max_output_tokens,
                max_summary_chars=cfg.summary_max_chars,
                max_transcript_chars=self._effective_max_transcript_chars(
                    cfg,
                    prompt=prompt or cfg.prompt,
                ),
            )
            record = owner._context.compact_with_summary(
                summary,
                keep_last=keep_last,
                summary_prefix=cfg.summary_prefix,
            )
        except asyncio.CancelledError as exc:
            await self._emit_compact_stop_error(
                run_id=run_id,
                mode=mode,
                started_at=started_at,
                tokens_before=tokens_before,
                message_count_before=message_count_before,
                error=str(exc) or "cancelled",
                event_bus=event_bus,
            )
            raise
        except Exception as exc:
            await self._emit_compact_stop_error(
                run_id=run_id,
                mode=mode,
                started_at=started_at,
                tokens_before=tokens_before,
                message_count_before=message_count_before,
                error=str(exc),
                event_bus=event_bus,
            )
            raise

        if record is None:
            await owner._emit_event(
                AgentCompactStopEvent.create(
                    run_id=run_id,
                    mode=mode,
                    status="skipped",
                    duration_ms=(time.time() - started_at) * 1000,
                    tokens_before=tokens_before,
                    tokens_after=owner._context.estimate_tokens(),
                    message_count_before=message_count_before,
                    message_count_after=len(owner._context.messages),
                ),
                event_bus,
            )
            return None

        owner._refresh_context_usage_snapshot(
            model or owner._default_model,
            preserve_provider=False,
        )
        await owner._emit_event(
            AgentCompactStopEvent.create(
                run_id=run_id,
                mode=mode,
                status="success",
                duration_ms=(time.time() - started_at) * 1000,
                tokens_before=record.tokens_before,
                tokens_after=record.tokens_after,
                message_count_before=message_count_before,
                message_count_after=len(owner._context.messages),
                replaced_message_count=len(record.replaced_messages),
                kept_message_count=record.kept_messages,
            ),
            event_bus,
        )
        return record

    async def _maybe_auto_compact(
        self,
        model: Model,
        state: _ExecutionState,
        event_bus: EventBus | None = None,
    ) -> bool:
        """Run automatic compaction if the configured threshold is crossed."""
        owner = self._owner
        cfg = owner._auto_compact
        if not cfg.enabled:
            return False
        if owner.has_active_tool_calls:
            return False
        context_tokens = self._context_tokens_for_auto_compact(cfg)
        if context_tokens < cfg.token_limit():
            return False

        did_compact = False
        keep_last: int | None = cfg.keep_last_messages
        attempted_keep_last: set[int] = set()

        while keep_last is not None and keep_last not in attempted_keep_last:
            attempted_keep_last.add(keep_last)
            record = await self.acompact(
                model=model,
                config=cfg,
                event_bus=event_bus,
                run_id=state.run_id,
                mode="auto",
                keep_last_messages=keep_last,
            )
            if record is None:
                keep_last = self._next_smaller_keep_last(keep_last)
                continue

            did_compact = True
            state.iteration = max(state.iteration, 0)
            state.last_auto_compact_iteration = state.iteration
            if record.tokens_after < record.tokens_before:
                break
            keep_last = self._next_smaller_keep_last(keep_last)

        return did_compact

    async def _emit_compact_stop_error(
        self,
        *,
        run_id: str | None,
        mode: Literal["manual", "auto"],
        started_at: float,
        tokens_before: int,
        message_count_before: int,
        error: str,
        event_bus: EventBus | None = None,
    ) -> None:
        owner = self._owner
        with contextlib.suppress(Exception):
            await owner._emit_event(
                AgentCompactStopEvent.create(
                    run_id=run_id,
                    mode=mode,
                    status="error",
                    duration_ms=(time.time() - started_at) * 1000,
                    tokens_before=tokens_before,
                    tokens_after=owner._context.estimate_tokens(),
                    message_count_before=message_count_before,
                    message_count_after=len(owner._context.messages),
                    error=error,
                ),
                event_bus,
            )

    @staticmethod
    def _next_smaller_keep_last(keep_last: int) -> int | None:
        if keep_last <= 0:
            return None
        next_keep = max(0, keep_last // 2)
        if next_keep == keep_last:
            next_keep = keep_last - 1
        return next_keep

    def _context_tokens_for_auto_compact(self, cfg: AutoCompactConfig) -> int:
        """Return the best available token count for auto-compact gating."""
        context = self._owner._context
        estimated_tokens = context.estimate_tokens()
        saved_snapshot = context.context_usage_snapshot()
        if saved_snapshot is None:
            return estimated_tokens
        saved_tokens = saved_snapshot.used_tokens
        if saved_tokens <= 0:
            return estimated_tokens
        if (
            saved_snapshot.max_context_tokens is not None
            and saved_snapshot.max_context_tokens != cfg.max_context_tokens
        ):
            return estimated_tokens
        return max(estimated_tokens, saved_tokens)

    async def _generate_compaction_summary(
        self,
        model: Model,
        *,
        messages: list[Message] | None = None,
        prompt: str,
        compression_budget: int,
        max_output_tokens: int,
        max_summary_chars: int,
        max_transcript_chars: int,
    ) -> str:
        """Ask the model to summarize the current context for compaction."""
        source_messages = messages if messages is not None else self._owner._context.messages
        transcript = self._build_compaction_transcript(
            source_messages,
            max_chars=max_transcript_chars,
        )
        summary_request: Message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Summarize the following Hawi conversation transcript "
                        "for continuation after context compaction. Do not copy "
                        "large tool outputs or source files; preserve only durable "
                        "facts, decisions, current state, and next steps.\n\n"
                        f"{transcript}"
                    ),
                }
            ],
            "name": None,
            "metadata": None,
        }
        system_prompt: list[ContentPart] = [
            {
                "type": "text",
                "text": self._compaction_prompt_with_budget(
                    prompt,
                    compression_budget=compression_budget,
                    max_output_tokens=max_output_tokens,
                    max_summary_chars=max_summary_chars,
                ),
            }
        ]

        try:
            summary = await self._collect_model_text(
                model,
                messages=[summary_request],
                system=system_prompt,
                max_output_tokens=max_output_tokens,
                streaming=False,
            )
        except NotImplementedError:
            summary = await self._collect_model_text(
                model,
                messages=[summary_request],
                system=system_prompt,
                max_output_tokens=max_output_tokens,
                streaming=True,
            )

        summary = summary.strip()
        if summary:
            return self._clamp_summary(summary, max_summary_chars)
        return self._clamp_summary(
            self._fallback_compaction_summary(source_messages),
            max_summary_chars,
        )

    def _effective_max_transcript_chars(
        self,
        cfg: AutoCompactConfig,
        *,
        prompt: str,
    ) -> int:
        """Cap compaction transcript size to fit the summary model call."""
        if cfg.max_context_tokens <= 0:
            return cfg.max_transcript_chars

        system_prompt = self._compaction_prompt_with_budget(
            prompt,
            compression_budget=cfg.compression_budget,
            max_output_tokens=cfg.summary_max_output_tokens,
            max_summary_chars=cfg.summary_max_chars,
        )
        prompt_overhead_tokens = (
            estimate_text_tokens(system_prompt)
            + estimate_text_tokens(
                "Summarize the following Hawi conversation transcript "
                "for continuation after context compaction."
            )
            + 512
        )
        available_tokens = (
            cfg.max_context_tokens
            - cfg.summary_max_output_tokens
            - prompt_overhead_tokens
        )
        dynamic_chars = max(1024, available_tokens * 4)
        if cfg.max_transcript_chars <= 0:
            return dynamic_chars
        return min(cfg.max_transcript_chars, dynamic_chars)

    async def _collect_model_text(
        self,
        model: Model,
        *,
        messages: list[Message],
        system: list[ContentPart],
        max_output_tokens: int,
        streaming: bool,
    ) -> str:
        """Collect text deltas from one direct model call."""
        chunks: list[str] = []
        async for delta in model.ainvoke(
            messages=messages,
            streaming=streaming,
            system=system,
            tools=None,
            max_output_tokens=max_output_tokens,
        ):
            if isinstance(delta, Event):
                continue
            if delta.get("type") == "text_delta":
                chunks.append(str(delta.get("delta", "")))
        return "".join(chunks)

    def _build_compaction_transcript(
        self,
        messages: list[Message],
        *,
        max_chars: int,
    ) -> str:
        """Render Hawi messages into compact plain text for summarization."""
        rendered: list[str] = ["<conversation>"]
        for index, message in enumerate(messages, 1):
            rendered.append(f"\n## Message {index}: {message['role']}")
            metadata = message.get("metadata") or {}
            source = metadata.get("source")
            if source:
                rendered.append(f"source: {source}")
            rendered.append(self._render_message_content_for_compaction(message))
        rendered.append("\n</conversation>")

        text = "\n".join(part for part in rendered if part)
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        head_chars = max(0, max_chars // 4)
        tail_chars = max(0, max_chars - head_chars)
        return (
            text[:head_chars]
            + "\n\n...[transcript truncated for compaction prompt budget]...\n\n"
            + text[-tail_chars:]
        )

    @staticmethod
    def _compaction_prompt_with_budget(
        prompt: str,
        *,
        compression_budget: int,
        max_output_tokens: int,
        max_summary_chars: int,
    ) -> str:
        budget = max(0, compression_budget)
        output_limit = max(1, max_output_tokens)
        char_limit = max(1, max_summary_chars)
        return (
            f"{prompt}\n\n"
            "Compression budget guidance:\n"
            f"- The configured compression budget is {budget:,} tokens of "
            "context-window headroom reserved after compaction.\n"
            f"- Keep the handoff summary comfortably within the summary output "
            f"limit of {output_limit:,} tokens.\n"
            f"- Hard target: the final summary must be under {char_limit:,} "
            "characters.\n"
            "- Do not quote or reproduce large tool outputs, logs, source files, "
            "or directory listings; mention the file/path and the conclusion.\n"
            "- Prefer durable facts, decisions, constraints, and next steps; "
            "drop transcript detail that is not needed to resume accurately."
        )

    def _render_message_content_for_compaction(self, message: Message) -> str:
        """Render one message's content in a summary-friendly format."""
        lines: list[str] = []
        for part in message.get("content", []):
            if not isinstance(part, dict):
                lines.append(str(part))
                continue
            part_type = part.get("type")
            if part_type == "text":
                lines.append(
                    self._abbreviate_text(
                        str(part.get("text", "")),
                        max_chars=COMPACTION_TEXT_PART_MAX_CHARS,
                        label="text",
                    )
                )
            elif part_type == "tool_call":
                arguments = json.dumps(part.get("arguments", {}), ensure_ascii=False)
                lines.append(
                    "tool_call "
                    f"{part.get('name', 'unknown')}({part.get('id', '')}): "
                    f"{self._abbreviate_text(arguments, max_chars=1200, label='arguments')}"
                )
            elif part_type == "tool_result":
                nested = part.get("content", "")
                if isinstance(nested, list):
                    nested_text = self._owner._serialize_content_parts(
                        cast(list[ContentPart], nested)
                    )
                else:
                    nested_text = str(nested)
                lines.append(
                    "tool_result "
                    f"{part.get('tool_call_id', '')}"
                    f"{' error' if part.get('is_error') else ''}: "
                    f"{self._abbreviate_text(nested_text, max_chars=COMPACTION_TOOL_RESULT_MAX_CHARS, label='tool result')}"
                )
            elif part_type == "steer":
                nested = part.get("content", [])
                if isinstance(nested, list):
                    steer_text = self._owner._serialize_content_parts(
                        cast(list[ContentPart], nested)
                    )
                else:
                    steer_text = str(nested)
                lines.append(
                    "steer: "
                    + self._abbreviate_text(
                        steer_text,
                        max_chars=COMPACTION_TEXT_PART_MAX_CHARS,
                        label="steer",
                    )
                )
            else:
                rendered = self._owner._serialize_content_parts(
                    [cast(ContentPart, part)]
                )
                lines.append(
                    self._abbreviate_text(
                        rendered,
                        max_chars=COMPACTION_TEXT_PART_MAX_CHARS,
                        label=str(part_type or "content"),
                    )
                )
        return "\n".join(line for line in lines if line.strip())

    @staticmethod
    def _abbreviate_text(text: str, *, max_chars: int, label: str) -> str:
        text = text.strip()
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        marker = f"\n...[{label} truncated for compaction]...\n"
        head_chars = max(0, (max_chars - len(marker)) * 3 // 4)
        tail_chars = max(0, max_chars - len(marker) - head_chars)
        if tail_chars == 0:
            return text[:max_chars].rstrip()
        return text[:head_chars].rstrip() + marker + text[-tail_chars:].lstrip()

    @staticmethod
    def _clamp_summary(summary: str, max_chars: int) -> str:
        text = summary.strip()
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        marker = "\n\n[Compaction summary truncated to configured size]\n\n"
        head_chars = max(0, (max_chars - len(marker)) * 3 // 4)
        tail_chars = max(0, max_chars - len(marker) - head_chars)
        if tail_chars == 0:
            return text[:max_chars].rstrip()
        return text[:head_chars].rstrip() + marker + text[-tail_chars:].lstrip()

    def _fallback_compaction_summary(self, messages: list[Message]) -> str:
        """Build a deterministic fallback if the summarizer returns no text."""
        recent_user_messages: list[str] = []
        for message in reversed(messages):
            if message["role"] != "user":
                continue
            text = self._owner._serialize_content_parts(
                list(message.get("content", []))
            )
            if text:
                recent_user_messages.append(text)
            if len(recent_user_messages) >= 3:
                break
        recent_user_messages.reverse()
        recent = "\n".join(f"- {text}" for text in recent_user_messages)
        return (
            "The previous conversation was compacted automatically, but the "
            "summary model returned no text. Continue from the recent preserved "
            "messages. Recent user requests:\n"
            f"{recent or '- No user request text available.'}"
        )
