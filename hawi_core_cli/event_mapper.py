"""Map internal Hawi events to the stable core-cli semantic event surface."""

from __future__ import annotations

from typing import Any

import hawi.events
from hawi.events import Event

from .protocol import make_error, make_frame, to_json_safe


class SemanticEventMapper:
    """Stateful mapper from Hawi EventBus events to core protocol events."""

    def __init__(self) -> None:
        self._active_run_id: str | None = None
        self._current_queue_kind = "normal"
        self._run_queue: dict[str, str] = {}
        self._active_tool_calls: dict[str, dict[str, Any]] = {}
        self._tool_call_id_by_block: dict[int, str] = {}

    def map(self, event: Event) -> list[dict[str, Any]]:
        """Return zero or more semantic protocol events for one Hawi event."""
        etype = event.type

        if etype == "scheduler.enqueue":
            event = event  # type: ignore[assignment]
            return [
                make_frame(
                    "debug.info",
                    {
                        "message": (
                            f"Enqueue to {getattr(event, 'queue_type', '')}: "
                            f"{getattr(event, 'content_preview', '')}"
                        ),
                        "event_type": etype,
                        "message_id": getattr(event, "message_id", ""),
                    },
                )
            ]

        if etype == "scheduler.dequeue":
            event = event  # type: ignore[assignment]
            queue_type = str(getattr(event, "queue_type", "normal"))
            if queue_type in {"normal", "high_prio", "urgent"}:
                self._current_queue_kind = queue_type
            return [
                make_frame(
                    "debug.info",
                    {
                        "message": f"Dequeue from {queue_type}",
                        "event_type": etype,
                        "message_id": getattr(event, "message_id", ""),
                        "queue": queue_type,
                    },
                )
            ]

        if etype == "scheduler.interrupt":
            return [
                make_frame(
                    "scheduler.interrupt",
                    {
                        "reason": getattr(event, "reason", ""),
                        "interrupted_tool_calls": getattr(event, "interrupted_tool_calls", []),
                    },
                )
            ]

        if etype == "agent.run_start":
            run_id = str(getattr(event, "run_id", ""))
            self._active_run_id = run_id
            self._run_queue[run_id] = self._current_queue_kind
            return []

        if etype == "agent.message_added":
            if getattr(event, "role", "") != "user":
                return []
            run_id = str(getattr(event, "run_id", self._active_run_id or ""))
            text = self._extract_text(getattr(event, "content", []))
            if not text:
                return []
            return [
                make_frame(
                    "run.start",
                    {
                        "run_id": run_id,
                        "user_content": text,
                        "queue": self._run_queue.get(run_id, self._current_queue_kind),
                    },
                )
            ]

        if etype == "model.content_block_delta":
            run_id = self._active_run_id or ""
            delta_type = getattr(event, "delta_type", "")
            delta = getattr(event, "delta", "")
            if delta_type == "text" and delta:
                return [make_frame("run.text_delta", {"run_id": run_id, "delta": delta})]
            if delta_type == "reasoning" and delta:
                return [make_frame("run.thinking_delta", {"run_id": run_id, "delta": delta})]
            return []

        if etype == "model.metadata":
            usage = dict(getattr(event, "usage", None) or {})
            input_tokens = int(usage.get("input_tokens", 0) or 0)
            output_tokens = int(usage.get("output_tokens", 0) or 0)
            total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or 0)
            return [
                make_frame(
                    "model.metadata",
                    {
                        "run_id": self._active_run_id or "",
                        "request_id": getattr(event, "request_id", ""),
                        "usage": usage,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                        "latency_ms": getattr(event, "latency_ms", None),
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
            return [
                make_error(
                    self._error_message(error, "Model error"),
                    code="model_error",
                    details=self._error_details(error),
                )
            ]

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
            block_index = int(getattr(event, "block_index", 0) or 0)
            if tool_call_id:
                self._tool_call_id_by_block[block_index] = tool_call_id
            tool_name = str(getattr(event, "tool_name", ""))
            self._active_tool_calls[tool_call_id] = {
                "tool_name": tool_name,
                "arguments": {},
                "run_id": self._active_run_id or "",
            }
            return [
                make_frame(
                    "tool.call_start",
                    {
                        "run_id": self._active_run_id or "",
                        "tool_call_id": tool_call_id,
                        "tool_name": tool_name,
                    },
                )
            ]

        if etype == "model.tool_call_block_delta":
            block_index = int(getattr(event, "block_index", 0) or 0)
            tool_call_id = str(
                getattr(event, "tool_call_id", "")
                or self._tool_call_id_by_block.get(block_index, "")
            )
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
            block_index = int(getattr(event, "block_index", 0) or 0)
            tool_call_id = str(
                getattr(event, "tool_call_id", "")
                or self._tool_call_id_by_block.get(block_index, "")
            )
            if not tool_call_id:
                return []
            tool_name = str(getattr(event, "tool_name", ""))
            arguments = to_json_safe(getattr(event, "arguments", {}))
            self._tool_call_id_by_block[block_index] = tool_call_id
            self._active_tool_calls.setdefault(
                tool_call_id,
                {"run_id": self._active_run_id or ""},
            ).update({"tool_name": tool_name, "arguments": arguments})
            return [
                make_frame(
                    "tool.call_stop",
                    {
                        "run_id": self._active_run_id or "",
                        "tool_call_id": tool_call_id,
                        "tool_name": tool_name,
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
                    "arguments": to_json_safe(getattr(event, "arguments", {})),
                },
            )
            return []

        if etype == "agent.tool_result_part":
            return [
                make_frame(
                    "tool.result",
                    {
                        "run_id": getattr(event, "run_id", self._active_run_id or ""),
                        "tool_call_id": getattr(event, "tool_call_id", ""),
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
            self._tool_call_id_by_block.clear()
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
    def _extract_text(content: Any) -> str:
        text = ""
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text += str(part.get("text", ""))
        return text

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
