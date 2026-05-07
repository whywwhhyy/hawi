"""Map internal Hawi events to the stable core-cli semantic event surface."""

from __future__ import annotations

from typing import Any

import hawi.events
from hawi.events import Event

from .protocol import make_error, make_frame, to_json_safe

TOOL_CALL_DESCRIPTION_PARAMETER = "tool_call_description"


class SemanticEventMapper:
    """Stateful mapper from Hawi EventBus events to core protocol events."""

    def __init__(self) -> None:
        self._active_run_id: str | None = None
        self._current_queue_kind = "normal"
        self._run_queue: dict[str, str] = {}
        self._active_tool_calls: dict[str, dict[str, Any]] = {}
        self._tool_call_display_id_by_block: dict[tuple[str, int], str] = {}
        self._display_id_by_actual_tool_call_id: dict[str, str] = {}

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
            actual_tool_call_id = str(getattr(event, "tool_call_id", ""))
            request_id = str(getattr(event, "request_id", ""))
            block_index = int(getattr(event, "block_index", 0) or 0)
            display_tool_call_id = actual_tool_call_id or self._pending_tool_call_id(
                request_id,
                block_index,
            )
            self._tool_call_display_id_by_block[(request_id, block_index)] = display_tool_call_id
            if actual_tool_call_id:
                self._display_id_by_actual_tool_call_id[actual_tool_call_id] = display_tool_call_id
            tool_name = str(getattr(event, "tool_name", ""))
            self._active_tool_calls[display_tool_call_id] = {
                "tool_name": tool_name,
                "arguments": {},
                "run_id": self._active_run_id or "",
                "actual_tool_call_id": actual_tool_call_id,
                "tool_call_description": "",
            }
            return [
                make_frame(
                    "tool.call_start",
                    {
                        "run_id": self._active_run_id or "",
                        "tool_call_id": display_tool_call_id,
                        "actual_tool_call_id": actual_tool_call_id,
                        "tool_name": tool_name,
                        "tool_call_description": "",
                    },
                )
            ]

        if etype == "model.tool_call_block_delta":
            request_id = str(getattr(event, "request_id", ""))
            block_index = int(getattr(event, "block_index", 0) or 0)
            actual_tool_call_id = str(getattr(event, "tool_call_id", ""))
            display_tool_call_id = self._display_tool_call_id(
                actual_tool_call_id=actual_tool_call_id,
                request_id=request_id,
                block_index=block_index,
            )
            if not display_tool_call_id:
                return []
            return [
                make_frame(
                    "tool.call_delta",
                    {
                        "run_id": self._active_run_id or "",
                        "tool_call_id": display_tool_call_id,
                        "actual_tool_call_id": actual_tool_call_id,
                        "delta": getattr(event, "arguments_delta", ""),
                        "is_streaming": getattr(event, "is_streaming", True),
                    },
                )
            ]

        if etype == "model.tool_call_block_stop":
            request_id = str(getattr(event, "request_id", ""))
            block_index = int(getattr(event, "block_index", 0) or 0)
            actual_tool_call_id = str(getattr(event, "tool_call_id", ""))
            display_tool_call_id = self._display_tool_call_id(
                actual_tool_call_id=actual_tool_call_id,
                request_id=request_id,
                block_index=block_index,
            )
            if not display_tool_call_id:
                return []
            tool_name = str(getattr(event, "tool_name", ""))
            raw_arguments = getattr(event, "arguments", {})
            tool_call_description = self._tool_call_description(raw_arguments)
            arguments = self._visible_tool_arguments(raw_arguments)
            self._tool_call_display_id_by_block[(request_id, block_index)] = display_tool_call_id
            self._active_tool_calls.setdefault(
                display_tool_call_id,
                {"run_id": self._active_run_id or ""},
            ).update(
                {
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "actual_tool_call_id": actual_tool_call_id,
                    "tool_call_description": tool_call_description,
                }
            )
            return [
                make_frame(
                    "tool.call_stop",
                    {
                        "run_id": self._active_run_id or "",
                        "tool_call_id": display_tool_call_id,
                        "actual_tool_call_id": actual_tool_call_id,
                        "tool_name": tool_name,
                        "tool_call_description": tool_call_description,
                        "arguments": arguments,
                    },
                )
            ]

        if etype == "agent.tool_call":
            actual_tool_call_id = str(getattr(event, "tool_call_id", ""))
            display_tool_call_id = (
                self._display_id_by_actual_tool_call_id.get(actual_tool_call_id)
                or actual_tool_call_id
            )
            self._active_tool_calls.setdefault(
                display_tool_call_id,
                {
                    "run_id": getattr(event, "run_id", self._active_run_id or ""),
                    "tool_name": getattr(event, "tool_name", ""),
                    "arguments": self._visible_tool_arguments(getattr(event, "arguments", {})),
                    "actual_tool_call_id": actual_tool_call_id,
                    "tool_call_description": self._tool_call_description(
                        getattr(event, "arguments", {})
                    ),
                },
            ).update(
                {
                    "run_id": getattr(event, "run_id", self._active_run_id or ""),
                    "tool_name": getattr(event, "tool_name", ""),
                    "arguments": self._visible_tool_arguments(getattr(event, "arguments", {})),
                    "actual_tool_call_id": actual_tool_call_id,
                    "tool_call_description": self._tool_call_description(
                        getattr(event, "arguments", {})
                    ),
                }
            )
            return []

        if etype == "agent.tool_result_part":
            actual_tool_call_id = str(getattr(event, "tool_call_id", ""))
            display_tool_call_id = (
                self._display_id_by_actual_tool_call_id.get(actual_tool_call_id)
                or actual_tool_call_id
            )
            call_info = self._active_tool_calls.get(display_tool_call_id, {})
            return [
                make_frame(
                    "tool.result",
                    {
                        "run_id": getattr(event, "run_id", self._active_run_id or ""),
                        "tool_call_id": display_tool_call_id,
                        "actual_tool_call_id": actual_tool_call_id,
                        "tool_call_description": call_info.get("tool_call_description", ""),
                        "part": getattr(event, "part", ""),
                        "part_index": getattr(event, "part_index", 0),
                        "is_final": getattr(event, "is_final", False),
                        "is_part": True,
                    },
                )
            ]

        if etype == "agent.tool_result":
            actual_tool_call_id = str(getattr(event, "tool_call_id", ""))
            display_tool_call_id = (
                self._display_id_by_actual_tool_call_id.get(actual_tool_call_id)
                or actual_tool_call_id
            )
            call_info = self._active_tool_calls.pop(display_tool_call_id, {})
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
                        "tool_call_id": display_tool_call_id,
                        "actual_tool_call_id": actual_tool_call_id,
                        "tool_name": call_info.get("tool_name", ""),
                        "tool_call_description": call_info.get("tool_call_description", ""),
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
            self._tool_call_display_id_by_block.clear()
            self._display_id_by_actual_tool_call_id.clear()
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
    def _tool_call_description(arguments: Any) -> str:
        if not isinstance(arguments, dict):
            return ""
        value = arguments.get(TOOL_CALL_DESCRIPTION_PARAMETER)
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return str(value).strip()

    @staticmethod
    def _visible_tool_arguments(arguments: Any) -> Any:
        if not isinstance(arguments, dict):
            return to_json_safe(arguments)
        if TOOL_CALL_DESCRIPTION_PARAMETER not in arguments:
            return to_json_safe(arguments)
        visible = dict(arguments)
        visible.pop(TOOL_CALL_DESCRIPTION_PARAMETER, None)
        return to_json_safe(visible)

    @staticmethod
    def _pending_tool_call_id(request_id: str, block_index: int) -> str:
        request_part = request_id or "unknown-request"
        return f"pending:{request_part}:{block_index}"

    def _display_tool_call_id(
        self,
        *,
        actual_tool_call_id: str,
        request_id: str,
        block_index: int,
    ) -> str:
        if actual_tool_call_id and actual_tool_call_id in self._display_id_by_actual_tool_call_id:
            return self._display_id_by_actual_tool_call_id[actual_tool_call_id]
        block_key = (request_id, block_index)
        display_tool_call_id = self._tool_call_display_id_by_block.get(block_key, "")
        if not display_tool_call_id and actual_tool_call_id:
            display_tool_call_id = actual_tool_call_id
            self._tool_call_display_id_by_block[block_key] = display_tool_call_id
        if actual_tool_call_id and display_tool_call_id:
            self._display_id_by_actual_tool_call_id[actual_tool_call_id] = display_tool_call_id
        return display_tool_call_id

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
