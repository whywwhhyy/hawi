"""Map internal Hawi events to the stable core-cli semantic event surface."""

from __future__ import annotations

from typing import Any

import hawi.events
from hawi.events import Event
from hawi.models.usage import usage_total

from .protocol import make_error, make_frame, to_json_safe

TOOL_CALL_PURPOSE_PARAMETER = "tool_call_purpose"


class SemanticEventMapper:
    """Stateful mapper from Hawi EventBus events to core protocol events."""

    def __init__(self) -> None:
        self._active_run_id: str | None = None
        self._current_queue_kind = "normal"
        self._run_queue: dict[str, str] = {}
        self._active_tool_calls: dict[str, dict[str, Any]] = {}
        self._pending_model_input_started_at: float | None = None
        self._active_model_request_id: str | None = None
        self._active_model_stream_started_at: float | None = None
        self._reported_ttft_request_ids: set[str] = set()

    def map(self, event: Event) -> list[dict[str, Any]]:
        """Return zero or more semantic protocol events for one Hawi event."""
        etype = event.type

        if etype.startswith("plugin."):
            return [
                make_frame(
                    etype,
                    self._plugin_payload(event),
                )
            ]

        if etype == "runner.enqueue":
            event = event  # type: ignore[assignment]
            queue_type = str(getattr(event, "queue_type", ""))
            message_id = str(getattr(event, "message_id", ""))
            content_preview = str(getattr(event, "content_preview", ""))
            return [
                make_frame(
                    "debug.info",
                    {
                        "message": (
                            f"Enqueue to {queue_type}: "
                            f"{content_preview}"
                        ),
                        "event_type": etype,
                        "message_id": message_id,
                    },
                )
            ]

        if etype == "runner.dequeue":
            event = event  # type: ignore[assignment]
            queue_type = str(getattr(event, "queue_type", "normal"))
            message_id = str(getattr(event, "message_id", ""))
            if queue_type in {"normal", "high_prio", "urgent"}:
                self._current_queue_kind = queue_type
            return [
                make_frame(
                    "debug.info",
                    {
                        "message": f"Dequeue from {queue_type}",
                        "event_type": etype,
                        "message_id": message_id,
                        "queue": queue_type,
                    },
                )
            ]

        if etype == "runner.interrupt":
            return [
                make_frame(
                    "runner.interrupt",
                    {
                        "reason": getattr(event, "reason", ""),
                        "interrupted_tool_calls": getattr(event, "interrupted_tool_calls", []),
                    },
                )
            ]

        if etype == "runner.paused":
            return [
                make_frame(
                    "runner.paused",
                    {
                        "reason": getattr(event, "reason", getattr(event, "pause_reason", "")),
                        "resumable": getattr(event, "resumable", True),
                        "last_error_message": getattr(event, "last_error_message", None),
                    },
                )
            ]

        if etype == "runner.resumed":
            return [
                make_frame(
                    "runner.resumed",
                    {
                        "source": getattr(event, "source", "resume"),
                    },
                )
            ]

        if etype == "agent.run_start":
            run_id = str(getattr(event, "run_id", ""))
            self._active_run_id = run_id
            self._run_queue[run_id] = self._current_queue_kind
            return []

        if etype == "agent.compact_start":
            return [
                make_frame(
                    "agent.compact_start",
                    {
                        "run_id": getattr(event, "run_id", None),
                        "mode": getattr(event, "mode", ""),
                        "keep_last_messages": getattr(
                            event,
                            "keep_last_messages",
                            0,
                        ),
                        "tokens_before": getattr(event, "tokens_before", None),
                        "message_count_before": getattr(
                            event,
                            "message_count_before",
                            None,
                        ),
                    },
                )
            ]

        if etype == "agent.compact_stop":
            return [
                make_frame(
                    "agent.compact_stop",
                    {
                        "run_id": getattr(event, "run_id", None),
                        "mode": getattr(event, "mode", ""),
                        "status": getattr(event, "status", ""),
                        "duration_ms": getattr(event, "duration_ms", 0.0),
                        "tokens_before": getattr(event, "tokens_before", None),
                        "tokens_after": getattr(event, "tokens_after", None),
                        "message_count_before": getattr(
                            event,
                            "message_count_before",
                            None,
                        ),
                        "message_count_after": getattr(
                            event,
                            "message_count_after",
                            None,
                        ),
                        "replaced_message_count": getattr(
                            event,
                            "replaced_message_count",
                            None,
                        ),
                        "kept_message_count": getattr(
                            event,
                            "kept_message_count",
                            None,
                        ),
                        "error": getattr(event, "error", None),
                    },
                )
            ]

        if etype == "agent.message_added":
            role = getattr(event, "role", "")
            if role in {"user", "tool"}:
                self._mark_model_wait_start(getattr(event, "timestamp", None))
            if role != "user":
                return []
            run_id = str(getattr(event, "run_id", self._active_run_id or ""))
            text = self._extract_text(getattr(event, "content", []))
            if not text:
                return []
            queue = self._queue_for_user_message(event, run_id)
            message_id = self._message_id_for_user_message(event, run_id)
            display_message_type = self._display_message_type_for_user_message(
                event,
                queue,
            )
            return [
                make_frame(
                    "run.start",
                    {
                        "run_id": run_id,
                        "message_id": message_id,
                        "user_content": text,
                        "queue": queue,
                        "display_message_type": display_message_type,
                    },
                )
            ]

        if etype == "model.content_block_delta":
            run_id = self._active_run_id or ""
            request_id = str(getattr(event, "request_id", self._active_model_request_id or ""))
            delta_type = getattr(event, "delta_type", "")
            delta = getattr(event, "delta", "")
            if delta_type == "text" and delta:
                frames = self._ttft_debug_frames(event, request_id=request_id)
                frames.append(make_frame("run.text_delta", {"run_id": run_id, "delta": delta}))
                return frames
            if delta_type == "reasoning" and delta:
                frames = self._ttft_debug_frames(event, request_id=request_id)
                frames.append(make_frame("run.thinking_delta", {"run_id": run_id, "delta": delta}))
                return frames
            return []

        if etype == "model.metadata":
            usage = dict(getattr(event, "usage", None) or {})
            input_tokens = int(usage.get("input_tokens", 0) or 0)
            output_tokens = int(usage.get("output_tokens", 0) or 0)
            total_tokens = usage_total(usage)
            run_id = self._active_run_id or ""
            request_id = getattr(event, "request_id", "")
            ttft_ms = getattr(event, "ttft_ms", None)
            return [
                make_frame(
                    "model.metadata",
                    {
                        "run_id": run_id,
                        "request_id": request_id,
                        "usage": usage,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                        "cache_write_tokens": usage.get("cache_write_tokens"),
                        "cache_read_tokens": usage.get("cache_read_tokens"),
                        "cache_miss_tokens": usage.get("cache_miss_tokens"),
                        "reasoning_tokens": usage.get("reasoning_tokens"),
                        "input_audio_tokens": usage.get("input_audio_tokens"),
                        "output_audio_tokens": usage.get("output_audio_tokens"),
                        "accepted_prediction_tokens": usage.get("accepted_prediction_tokens"),
                        "rejected_prediction_tokens": usage.get("rejected_prediction_tokens"),
                        "latency_ms": getattr(event, "latency_ms", None),
                        "started_at": getattr(event, "started_at", None),
                        "first_token_at": getattr(event, "first_token_at", None),
                        "completed_at": getattr(event, "completed_at", None),
                        "ttft_ms": ttft_ms,
                        "decode_ms": getattr(event, "decode_ms", None),
                        "prefill_tokens": getattr(event, "prefill_tokens", None),
                        "decode_tokens": getattr(event, "decode_tokens", None),
                        "prefill_tokens_per_second": getattr(
                            event,
                            "prefill_tokens_per_second",
                            None,
                        ),
                        "decode_tokens_per_second": getattr(
                            event,
                            "decode_tokens_per_second",
                            None,
                        ),
                        "context_tokens": getattr(event, "context_tokens", None),
                        "max_context_tokens": getattr(event, "max_context_tokens", None),
                        "context_ratio": getattr(event, "context_ratio", None),
                        "context_source": getattr(event, "context_source", None),
                    },
                )
            ]

        if etype == "model.retry":
            return [
                make_frame(
                    "model.retry",
                    {
                        "run_id": self._active_run_id or "",
                        "request_id": getattr(event, "request_id", ""),
                        "attempt": getattr(event, "attempt", 0),
                        "max_retries": getattr(event, "max_retries", 0),
                        "error_type": getattr(event, "error_type", ""),
                        "error_message": getattr(event, "error_message", ""),
                    },
                )
            ]

        if etype == "model.error":
            error = getattr(event, "error", None)
            frames = self._ttft_debug_frames(
                event,
                request_id=self._active_model_request_id or "",
                failed=True,
            )
            frames.append(
                make_error(
                    self._error_message(error, "Model error"),
                    code="model_error",
                    details=self._error_details(error),
                )
            )
            return frames

        if etype == "agent.error":
            error = getattr(event, "error", None)
            return [
                make_error(
                    self._error_message(error, "Agent error"),
                    code="agent_error",
                    details=self._error_details(error),
                )
            ]

        if etype == "model.tool_call_block_start":
            tool_call_id = str(getattr(event, "tool_call_id", ""))
            if not tool_call_id:
                # The stream accumulator defers the StartEvent until the id
                # is known, so receiving one with an empty id at this layer
                # would be a bug upstream. Defensively drop it rather than
                # forward an ambiguous frame to the GUI.
                return []
            tool_name = str(getattr(event, "tool_name", ""))
            self._active_tool_calls[tool_call_id] = {
                "tool_name": tool_name,
                "arguments": {},
                "run_id": self._active_run_id or "",
                "tool_call_purpose": "",
            }
            frames = self._ttft_debug_frames(
                event,
                request_id=str(getattr(event, "request_id", self._active_model_request_id or "")),
            )
            frames.append(
                make_frame(
                    "tool.call_start",
                    {
                        "run_id": self._active_run_id or "",
                        "tool_call_id": tool_call_id,
                        "tool_name": tool_name,
                        "status": "pending",
                        "tool_call_purpose": "",
                    },
                )
            )
            return frames

        if etype == "model.tool_call_block_delta":
            tool_call_id = str(getattr(event, "tool_call_id", ""))
            if not tool_call_id:
                return []
            return [
                make_frame(
                    "tool.call_delta",
                    {
                        "run_id": self._active_run_id or "",
                        "tool_call_id": tool_call_id,
                        "delta": getattr(event, "arguments_delta", ""),
                        "is_streaming": getattr(event, "is_streaming", True),
                    },
                )
            ]

        if etype == "model.tool_call_block_stop":
            tool_call_id = str(getattr(event, "tool_call_id", ""))
            if not tool_call_id:
                return []
            tool_name = str(getattr(event, "tool_name", ""))
            raw_arguments = getattr(event, "arguments", {})
            tool_call_purpose = self._tool_call_purpose(raw_arguments)
            arguments = self._visible_tool_arguments(raw_arguments)
            self._active_tool_calls.setdefault(
                tool_call_id,
                {"run_id": self._active_run_id or ""},
            ).update(
                {
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "tool_call_purpose": tool_call_purpose,
                }
            )
            return [
                make_frame(
                    "tool.call_stop",
                    {
                        "run_id": self._active_run_id or "",
                        "tool_call_id": tool_call_id,
                        "tool_name": tool_name,
                        "tool_call_purpose": tool_call_purpose,
                        "arguments": arguments,
                    },
                )
            ]

        if etype == "agent.tool_call":
            tool_call_id = str(getattr(event, "tool_call_id", ""))
            self._active_tool_calls.setdefault(
                tool_call_id,
                {
                    "run_id": getattr(event, "run_id", self._active_run_id or ""),
                    "tool_name": getattr(event, "tool_name", ""),
                    "arguments": self._visible_tool_arguments(getattr(event, "arguments", {})),
                    "tool_call_purpose": self._tool_call_purpose(
                        getattr(event, "arguments", {})
                    ),
                },
            ).update(
                {
                    "run_id": getattr(event, "run_id", self._active_run_id or ""),
                    "tool_name": getattr(event, "tool_name", ""),
                    "arguments": self._visible_tool_arguments(getattr(event, "arguments", {})),
                    "tool_call_purpose": self._tool_call_purpose(
                        getattr(event, "arguments", {})
                    ),
                }
            )
            call_info = self._active_tool_calls.get(tool_call_id, {})
            return [
                make_frame(
                    "tool.call_start",
                    {
                        "run_id": call_info.get("run_id", getattr(event, "run_id", "")),
                        "tool_call_id": tool_call_id,
                        "tool_name": call_info.get("tool_name", ""),
                        "status": "running",
                        "tool_call_purpose": call_info.get(
                            "tool_call_purpose",
                            "",
                        ),
                        "arguments": call_info.get("arguments", {}),
                    },
                )
            ]

        if etype == "agent.tool_result_part":
            tool_call_id = str(getattr(event, "tool_call_id", ""))
            call_info = self._active_tool_calls.get(tool_call_id, {})
            return [
                make_frame(
                    "tool.result",
                    {
                        "run_id": getattr(event, "run_id", self._active_run_id or ""),
                        "tool_call_id": tool_call_id,
                        "tool_call_purpose": call_info.get("tool_call_purpose", ""),
                        "part": getattr(event, "part", ""),
                        "part_index": getattr(event, "part_index", 0),
                        "is_final": getattr(event, "is_final", False),
                        "is_part": True,
                    },
                )
            ]

        if etype == "agent.tool_result":
            tool_call_id = str(getattr(event, "tool_call_id", ""))
            call_info = self._active_tool_calls.pop(tool_call_id, {})
            result = getattr(event, "result", None)
            output = None
            error = ""
            if result is not None:
                output = getattr(result, "output", None)
                error = getattr(result, "error", "") or ""
            if output is None:
                output = error or getattr(event, "result_preview", "")
            return [
                make_frame(
                    "tool.result",
                    {
                        "run_id": call_info.get("run_id", getattr(event, "run_id", "")),
                        "tool_call_id": tool_call_id,
                        "tool_name": call_info.get("tool_name", ""),
                        "tool_call_purpose": call_info.get("tool_call_purpose", ""),
                        "success": getattr(event, "success", False),
                        "output": to_json_safe(output),
                        "error": error,
                        "duration_ms": getattr(event, "duration_ms", 0.0),
                        "is_part": False,
                    },
                )
            ]

        if etype == "agent.interrupt":
            return [
                make_frame(
                    "agent.interrupt",
                    {
                        "run_id": getattr(event, "run_id", ""),
                        "interrupt_type": getattr(event, "interrupt_type", ""),
                    },
                )
            ]

        if etype == "agent.run_stop":
            run_id = str(getattr(event, "run_id", ""))
            self._run_queue.pop(run_id, None)
            if self._active_run_id == run_id:
                self._active_run_id = None
            self._pending_model_input_started_at = None
            self._active_model_request_id = None
            self._active_model_stream_started_at = None
            return [
                make_frame(
                    "run.stop",
                    {
                        "run_id": run_id,
                        "stop_reason": getattr(event, "stop_reason", ""),
                        "duration_ms": getattr(event, "duration_ms", 0.0),
                        "usage": to_json_safe(getattr(event, "usage", None)),
                    },
                )
            ]

        if etype in {
            "model.stream_start",
            "model.stream_stop",
            "model.content_block_start",
            "model.content_block_stop",
            "model.content_metadata",
        }:
            if etype == "model.stream_start":
                self._active_model_request_id = str(getattr(event, "request_id", ""))
                self._active_model_stream_started_at = self._float_timestamp(
                    getattr(event, "timestamp", None)
                )
            return [
                make_frame(
                    "debug.info",
                    {
                        "message": self._debug_message(event),
                        "event_type": etype,
                    },
                )
            ]

        return []

    @staticmethod
    def _float_timestamp(value: Any) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _mark_model_wait_start(self, timestamp: Any) -> None:
        started_at = self._float_timestamp(timestamp)
        if started_at is None:
            return
        if (
            self._pending_model_input_started_at is None
            or started_at < self._pending_model_input_started_at
        ):
            self._pending_model_input_started_at = started_at

    def _ttft_debug_frames(
        self,
        event: Event,
        *,
        request_id: str,
        failed: bool = False,
    ) -> list[dict[str, Any]]:
        if request_id and request_id in self._reported_ttft_request_ids:
            return []

        started_at = (
            self._pending_model_input_started_at
            if self._pending_model_input_started_at is not None
            else self._active_model_stream_started_at
        )
        event_at = self._float_timestamp(getattr(event, "timestamp", None))
        if started_at is None or event_at is None:
            return []

        elapsed_ms = max(0.0, (event_at - started_at) * 1000)
        if request_id:
            self._reported_ttft_request_ids.add(request_id)
        self._pending_model_input_started_at = None

        message = (
            f"TTFT unavailable after {elapsed_ms:.0f}ms"
            if failed
            else f"TTFT {elapsed_ms:.0f}ms"
        )
        return [
            make_frame(
                "debug.info",
                {
                    "run_id": self._active_run_id or "",
                    "request_id": request_id,
                    "event_type": "model.error" if failed else event.type,
                    "ttft_ms": None if failed else elapsed_ms,
                    "elapsed_ms": elapsed_ms,
                    "message": message,
                },
            )
        ]

    @classmethod
    def _extract_text(cls, content: Any) -> str:
        chunks: list[str] = []
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    chunks.append(str(part.get("text", "")))
                elif isinstance(part, dict) and part.get("type") == "steer":
                    chunks.append(cls._extract_text(part.get("content", [])))
        return "".join(chunk for chunk in chunks if chunk)

    def _queue_for_user_message(self, event: Event, run_id: str) -> str:
        metadata = getattr(event, "metadata", None)
        if isinstance(metadata, dict):
            queue = str(metadata.get("queue", ""))
            if queue in {"normal", "high_prio", "urgent"}:
                return queue

        queue = self._run_queue.get(run_id, self._current_queue_kind)
        if queue in {"normal", "high_prio", "urgent"}:
            return queue
        return "normal"

    def _message_id_for_user_message(self, event: Event, _run_id: str) -> str:
        metadata = getattr(event, "metadata", None)
        if isinstance(metadata, dict):
            message_id = str(metadata.get("message_id", "") or "")
            if message_id:
                return message_id
        return ""

    @staticmethod
    def _display_message_type_for_user_message(event: Event, queue: str) -> str:
        metadata = getattr(event, "metadata", None)
        if isinstance(metadata, dict):
            display_message_type = str(metadata.get("display_message_type", "") or "")
            if display_message_type in {"normal", "steer", "urgent"}:
                return display_message_type
        if queue == "urgent":
            return "urgent"
        return "normal"

    @staticmethod
    def _error_message(error: Any, fallback: str) -> str:
        if error is None:
            return fallback
        message = getattr(error, "message", None)
        if message:
            return str(message)
        return str(error)

    @staticmethod
    def _error_details(error: Any) -> dict[str, Any] | None:
        if error is None:
            return None
        return {
            "type": getattr(error, "error_type", "unknown"),
            "class": error.__class__.__name__,
        }

    @staticmethod
    def _tool_call_purpose(arguments: Any) -> str:
        if not isinstance(arguments, dict):
            return ""
        value = arguments.get(TOOL_CALL_PURPOSE_PARAMETER)
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return str(value).strip()

    @staticmethod
    def _visible_tool_arguments(arguments: Any) -> Any:
        if not isinstance(arguments, dict):
            return to_json_safe(arguments)
        if TOOL_CALL_PURPOSE_PARAMETER not in arguments:
            return to_json_safe(arguments)
        visible = dict(arguments)
        visible.pop(TOOL_CALL_PURPOSE_PARAMETER, None)
        return to_json_safe(visible)

    @staticmethod
    def _debug_message(event: Event) -> str:
        if event.type == "model.stream_start":
            return "Model stream started"
        if event.type == "model.stream_stop":
            return f"Model stream stopped: {getattr(event, 'stop_reason', '')}"
        if event.type == "model.content_block_start":
            return f"Content block start: {getattr(event, 'block_type', '')}"
        if event.type == "model.content_block_stop":
            block_type = None
            if isinstance(event, hawi.events.ModelContentBlockStopEvent):
                block_type = event.block_type
            return f"Content block stop: {block_type or 'unknown'}"
        if event.type == "model.content_metadata":
            return "Content metadata"
        return event.type

    def _plugin_payload(self, event: Event) -> dict[str, Any]:
        raw_payload = getattr(event, "payload", {})
        payload = dict(raw_payload) if isinstance(raw_payload, dict) else {"data": raw_payload}
        plugin_name = str(getattr(event, "plugin_name", "") or "")
        plugin_id = str(getattr(event, "plugin_id", "") or plugin_name)
        run_id = str(getattr(event, "run_id", "") or self._active_run_id or "")
        tool_call_id = str(getattr(event, "tool_call_id", "") or "")
        message_id = str(getattr(event, "message_id", "") or "")
        payload.update(
            {
                "plugin_id": plugin_id,
                "plugin_name": plugin_name,
                "run_id": run_id,
                "tool_call_id": tool_call_id,
            }
        )
        if message_id:
            payload["message_id"] = message_id
        return to_json_safe(payload)
