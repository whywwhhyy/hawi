"""Context compaction behavior for HawiAgent."""

from __future__ import annotations

import asyncio
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
from .context import AgentContext, ContextCompactionRecord
from .state import _ExecutionState


class CompactionOwner(Protocol):
    _auto_compact: AutoCompactConfig
    _context: AgentContext
    _default_model: Model

    @property
    def has_active_tool_calls(self) -> bool: ...

    def _serialize_content_parts(self, content: list[ContentPart]) -> str: ...

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
        if owner._context.compaction_tail_start(keep_last) <= 0:
            return None

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
                prompt=prompt or cfg.prompt,
                max_output_tokens=cfg.summary_max_output_tokens,
                max_transcript_chars=cfg.max_transcript_chars,
            )
            record = owner._context.compact_with_summary(
                summary,
                keep_last=keep_last,
                summary_prefix=cfg.summary_prefix,
            )
        except Exception as exc:
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
                    error=str(exc),
                ),
                event_bus,
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
        if len(owner._context.messages) < cfg.min_messages:
            return False
        if owner._context.estimate_tokens() < cfg.token_limit():
            return False

        record = await self.acompact(
            model=model,
            config=cfg,
            event_bus=event_bus,
            run_id=state.run_id,
            mode="auto",
        )
        if record is not None:
            state.iteration = max(state.iteration, 0)
        return record is not None

    async def _generate_compaction_summary(
        self,
        model: Model,
        *,
        prompt: str,
        max_output_tokens: int,
        max_transcript_chars: int,
    ) -> str:
        """Ask the model to summarize the current context for compaction."""
        transcript = self._build_compaction_transcript(
            self._owner._context.messages,
            max_chars=max_transcript_chars,
        )
        summary_request: Message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Summarize the following Hawi conversation transcript "
                        "for continuation after context compaction.\n\n"
                        f"{transcript}"
                    ),
                }
            ],
            "name": None,
            "metadata": None,
        }
        system_prompt: list[ContentPart] = [{"type": "text", "text": prompt}]

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
            return summary
        return self._fallback_compaction_summary(self._owner._context.messages)

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

    def _render_message_content_for_compaction(self, message: Message) -> str:
        """Render one message's content in a summary-friendly format."""
        lines: list[str] = []
        for part in message.get("content", []):
            if not isinstance(part, dict):
                lines.append(str(part))
                continue
            part_type = part.get("type")
            if part_type == "tool_call":
                lines.append(
                    "tool_call "
                    f"{part.get('name', 'unknown')}({part.get('id', '')}): "
                    f"{json.dumps(part.get('arguments', {}), ensure_ascii=False)}"
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
                    f"{nested_text}"
                )
            elif part_type == "steer":
                nested = part.get("content", [])
                if isinstance(nested, list):
                    steer_text = self._owner._serialize_content_parts(
                        cast(list[ContentPart], nested)
                    )
                else:
                    steer_text = str(nested)
                lines.append("steer: " + steer_text)
            else:
                lines.append(
                    self._owner._serialize_content_parts([cast(ContentPart, part)])
                )
        return "\n".join(line for line in lines if line.strip())

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
