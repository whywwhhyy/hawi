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
        self._reported_model_interrupt_request_ids: set[str] = set()
        self._reported_ttft_request_ids: set[str] = set()

    def map(self, event: Event) -> list[dict[str, Any]]:
        """Return zero or more semantic protocol events for one Hawi event."""
        etype = event.type

        if etype.startswith("subagent."):
            return [
                make_frame(
                    etype,
                    self._subagent_payload(event),
                )
            ]

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
            reason = str(getattr(event, "reason", ""))
            interrupted_tool_calls = [
                str(tool_call_id)
                for tool_call_id in getattr(event, "interrupted_tool_calls", [])
                if str(tool_call_id)
            ]
            frames = [
                make_frame(
                    "runner.interrupt",
                    {
                        "reason": reason,
                        "interrupted_tool_calls": interrupted_tool_calls,
                    },
                )
            ]
            frames.extend(self._interrupted_tool_frames(reason, interrupted_tool_calls))
            if self._active_model_request_id:
                request_id = self._active_model_request_id
                frames.append(self._model_interrupted_frame(request_id, reason))
                self._reported_model_interrupt_request_ids.add(request_id)
                frames.append(
                    make_frame(
                        "debug.info",
                        {
                            "message": "Model stream stopped: interrupted",
                            "event_type": "model.stream_stop",
                            "request_id": request_id,
                            "stop_reason": "interrupted",
                        },
                    )
                )
                self._active_model_request_id = None
                self._active_model_stream_started_at = None
            return frames

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

        if etype == "agent.system_prompt":
            content = getattr(event, "content", [])
            return [
                make_frame(
                    "agent.system_prompt",
                    {
                        "run_id": getattr(event, "run_id", self._active_run_id or ""),
                        "content": to_json_safe(content),
                        "text": self._extract_text(content),
                        "origin": getattr(event, "origin", "model_input"),
                        "plugin_id": getattr(event, "plugin_id", None),
                        "plugin_name": getattr(event, "plugin_name", None),
                        "plugin_role": getattr(event, "plugin_role", "framework"),
                        "injection_name": getattr(event, "injection_name", None),
                        "metadata": to_json_safe(getattr(event, "metadata", None)),
                    },
                )
            ]

        if etype == "agent.context_injected":
            content = getattr(event, "content", [])
            return [
                make_frame(
                    "agent.context_injected",
                    {
                        "run_id": getattr(event, "run_id", self._active_run_id or ""),
                        "role": getattr(event, "role", ""),
                        "content": to_json_safe(content),
                        "text": self._extract_text(content),
                        "hook_type": getattr(event, "hook_type", None),
                        "position": getattr(event, "position", None),
                        "plugin_id": getattr(event, "plugin_id", None),
                        "plugin_name": getattr(event, "plugin_name", None),
                        "plugin_role": getattr(event, "plugin_role", "framework"),
                        "injection_name": getattr(event, "injection_name", None),
                        "metadata": to_json_safe(getattr(event, "metadata", None)),
                        "context_message_id": getattr(event, "context_message_id", None),
                        "merge_target": getattr(event, "merge_target", None),
                        "merge_position": getattr(event, "merge_position", None),
                        "target_message_id": getattr(event, "target_message_id", None),
                        "target_message_index": getattr(event, "target_message_index", None),
                        "target_context_message_id": getattr(
                            event,
                            "target_context_message_id",
                            None,
                        ),
                    },
                )
            ]

        if etype == "agent.tool_runtime_context_injected":
            return [
                make_frame(
                    "agent.tool_runtime_context_injected",
                    {
                        "run_id": getattr(event, "run_id", self._active_run_id or ""),
                        "tool_name": getattr(event, "tool_name", ""),
                        "tool_call_id": getattr(event, "tool_call_id", ""),
                        "parameter_name": getattr(event, "parameter_name", ""),
                        "plugin_id": getattr(event, "plugin_id", None),
                        "plugin_name": getattr(event, "plugin_name", None),
                        "plugin_role": getattr(event, "plugin_role", "tool_owner"),
                        "injection_name": getattr(event, "injection_name", None),
                    },
                )
            ]

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
            content = getattr(event, "content", [])
            content_list = content if isinstance(content, list) else []
            preview = self._content_preview(content_list, 240) if content_list else ""
            if role in {"user", "tool"}:
                self._mark_model_wait_start(getattr(event, "timestamp", None))
            if role == "assistant":
                return [
                    make_frame(
                        "run.message_committed",
                        {
                            "run_id": getattr(event, "run_id", self._active_run_id or ""),
                            "role": "assistant",
                            "content": to_json_safe(content_list),
                            "content_preview": preview,
                            "context_message_id": getattr(
                                event,
                                "context_message_id",
                                None,
                            ),
                        },
                    )
                ]
            if role != "user":
                return []
            run_id = str(getattr(event, "run_id", self._active_run_id or ""))
            text = self._extract_text(content_list)
            visible_content = text or preview
            if not visible_content and not content_list:
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
                        "user_content": visible_content,
                        "content": to_json_safe(content_list),
                        "content_preview": preview or visible_content,
                        "queue": queue,
                        "display_message_type": display_message_type,
                        "context_message_id": getattr(
                            event,
                            "context_message_id",
                            None,
                        ),
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
                        "prefill_ms": getattr(event, "prefill_ms", None),
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
                        "peak_decode_tokens_per_second": getattr(
                            event,
                            "peak_decode_tokens_per_second",
                            None,
                        ),
                        "context_tokens": getattr(event, "context_tokens", None),
                        "max_context_tokens": getattr(event, "max_context_tokens", None),
                        "context_ratio": getattr(event, "context_ratio", None),
                        "context_source": getattr(event, "context_source", None),
                    },
                )
            ]

        if etype == "model.profile":
            return [
                make_frame(
                    "model.profile",
                    {
                        "run_id": self._active_run_id or "",
                        "request_id": getattr(event, "request_id", ""),
                        "cache_tokens": getattr(event, "cache_tokens", None),
                        "ttft_ms": getattr(event, "ttft_ms", None),
                        "prefill_ms": getattr(event, "prefill_ms", None),
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
                        "peak_decode_tokens_per_second": getattr(
                            event,
                            "peak_decode_tokens_per_second",
                            None,
                        ),
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
                        "context_message_id": getattr(
                            event,
                            "context_message_id",
                            None,
                        ),
                        "interrupted": bool(getattr(event, "interrupted", False)),
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
            request_id = str(getattr(event, "request_id", ""))
            frames = [
                make_frame(
                    "debug.info",
                    {
                        "message": self._debug_message(event),
                        "event_type": etype,
                    },
                )
            ]
            if etype == "model.stream_start":
                self._active_model_request_id = request_id
                self._active_model_stream_started_at = self._float_timestamp(
                    getattr(event, "timestamp", None)
                )
            elif etype == "model.stream_stop":
                stop_reason = str(getattr(event, "stop_reason", ""))
                if (
                    stop_reason == "interrupted"
                    and request_id
                    and request_id not in self._reported_model_interrupt_request_ids
                ):
                    frames.append(self._model_interrupted_frame(request_id, "interrupted"))
                    self._reported_model_interrupt_request_ids.add(request_id)
                self._active_model_request_id = None
                self._active_model_stream_started_at = None
            return frames

        return []

    def _interrupted_tool_frames(
        self,
        reason: str,
        interrupted_tool_call_ids: list[str],
    ) -> list[dict[str, Any]]:
        frames: list[dict[str, Any]] = []
        emitted_tool_call_ids: set[str] = set()
        for tool_call_id, call_info in list(self._active_tool_calls.items()):
            emitted_tool_call_ids.add(tool_call_id)
            frames.append(
                make_frame(
                    "tool.interrupted",
                    {
                        "run_id": call_info.get("run_id", self._active_run_id or ""),
                        "tool_call_id": tool_call_id,
                        "tool_name": call_info.get("tool_name", ""),
                        "tool_call_purpose": call_info.get("tool_call_purpose", ""),
                        "reason": reason,
                    },
                )
            )
            self._active_tool_calls.pop(tool_call_id, None)
        for tool_call_id in interrupted_tool_call_ids:
            if tool_call_id in emitted_tool_call_ids:
                continue
            frames.append(
                make_frame(
                    "tool.interrupted",
                    {
                        "run_id": self._active_run_id or "",
                        "tool_call_id": tool_call_id,
                        "tool_name": "",
                        "tool_call_purpose": "",
                        "reason": reason,
                    },
                )
            )
        return frames

    def _model_interrupted_frame(self, request_id: str, reason: str) -> dict[str, Any]:
        return make_frame(
            "model.interrupted",
            {
                "run_id": self._active_run_id or "",
                "request_id": request_id,
                "reason": reason,
                "stop_reason": "interrupted",
            },
        )

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

    @classmethod
    def _content_preview(cls, content: Any, max_chars: int = 160) -> str:
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            chunks: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in {"text", "steer"}:
                    text = cls._extract_text([part])
                    if text:
                        chunks.append(text)
                elif part.get("type") in {"image", "document", "audio", "video", "file"}:
                    chunks.append(cls._media_preview(part))
            text = "\n".join(chunk for chunk in chunks if chunk)
        else:
            text = ""
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."

    @staticmethod
    def _media_preview(part: dict[str, Any]) -> str:
        part_type = str(part.get("type") or "media")
        source = part.get("source")
        source = source if isinstance(source, dict) else {}
        title = part.get("title")
        filename = source.get("filename") or title
        mime_type = source.get("mime_type") or source.get("mimeType") or source.get("format")
        uri = (
            source.get("uri")
            or source.get("url")
            or source.get("data_uri")
            or source.get("path")
            or source.get("file_id")
            or source.get("blob_id")
            or ""
        )
        if isinstance(uri, str) and uri.startswith("data:"):
            uri = uri.split(",", 1)[0] + ",..."
        details = [part_type]
        if filename:
            details.append(str(filename))
        if mime_type:
            details.append(str(mime_type))
        if uri:
            details.append(str(uri))
        return "[" + ": ".join(details) + "]"

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
            if display_message_type in {"normal", "steer", "urgent", "resume"}:
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

    @staticmethod
    def _subagent_payload(event: Event) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "subagent_id": getattr(event, "subagent_id", ""),
            "subagent_name": getattr(event, "subagent_name", ""),
            "subagent_role": getattr(event, "subagent_role", ""),
            "status": getattr(event, "status", {}),
        }
        child_event = getattr(event, "child_event", None)
        if child_event is not None:
            payload["child_event"] = child_event
        message_entry = getattr(event, "message_entry", None)
        if message_entry is not None:
            payload["message_entry"] = message_entry
        reason = getattr(event, "reason", None)
        if reason is not None:
            payload["reason"] = reason
        return to_json_safe(payload)
