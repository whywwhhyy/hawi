"""ChatView — scrollable chat area using tk.Text with rich tag formatting."""

from __future__ import annotations

import tkinter as tk
from datetime import datetime
from tkinter import ttk

from ..protocol import QueueKind
from ..theme import COLORS, QUEUE_COLORS, QUEUE_LABELS


class ChatView(tk.Frame):
    """Scrollable chat area displaying message bubbles."""

    def __init__(self, parent: tk.Widget, **kwargs):
        kwargs.setdefault("bg", COLORS["bg_chat"])
        super().__init__(parent, **kwargs)
        self._stream_marks: dict[str, str] = {}  # run_id -> tk mark name
        self._thinking_marks: dict[str, str] = {}  # run_id -> thinking mark name
        self._thinking_started: dict[str, bool] = {}  # run_id -> whether thinking header was shown
        self._metadata_marks: dict[str, str] = {}  # run_id -> metadata mark name
        self._tool_call_marks: dict[str, str] = {}  # tool_call_id -> mark name
        self._tool_call_started: dict[str, bool] = {}  # tool_call_id -> whether header was shown
        self._build()

    def _build(self):
        self.text = tk.Text(
            self,
            state=tk.DISABLED,
            wrap=tk.WORD,
            bg=COLORS["bg_chat"],
            fg=COLORS["text_primary"],
            relief=tk.FLAT,
            borderwidth=0,
            padx=16,
            pady=8,
            spacing1=2,
            spacing3=4,
            cursor="arrow",
            insertwidth=0,
            selectbackground="#C8D8FF",
        )
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.text.yview)
        self.text.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._configure_tags()

    def _configure_tags(self):
        t = self.text
        bold = ("TkDefaultFont", 12, "bold")
        italic = ("TkDefaultFont", 12, "italic")
        small = ("TkDefaultFont", 11)
        body = ("TkDefaultFont", 13)

        # User message headers (per queue kind)
        t.tag_configure("user_header_normal",   foreground=COLORS["queue_normal"],  font=bold)
        t.tag_configure("user_header_high_prio", foreground=COLORS["queue_high"],   font=bold)
        t.tag_configure("user_header_urgent",   foreground=COLORS["queue_urgent"],  font=bold)

        # User body
        t.tag_configure("user_body", foreground=COLORS["text_primary"], font=body,
                         lmargin1=24, lmargin2=24)

        # Agent
        t.tag_configure("agent_header", foreground=COLORS["agent_header"], font=bold)
        t.tag_configure("agent_body",   foreground=COLORS["text_primary"], font=body,
                         lmargin1=24, lmargin2=24)

        # Tool
        t.tag_configure("tool_call",   foreground=COLORS["text_secondary"], font=italic,
                         lmargin1=24, lmargin2=32)
        t.tag_configure("tool_ok",     foreground=COLORS["tool_ok"],  font=small,
                         lmargin1=24, lmargin2=32)
        t.tag_configure("tool_fail",   foreground=COLORS["tool_fail"], font=small,
                         lmargin1=24, lmargin2=32)

        # Interrupt and system
        t.tag_configure("interrupt", foreground=COLORS["interrupt_fg"], font=bold, lmargin1=16)
        t.tag_configure("system",    foreground=COLORS["system_fg"],    font=italic,
                         justify=tk.CENTER)

        # Timestamp
        t.tag_configure("timestamp", foreground=COLORS["text_timestamp"], font=small)

        # Separator
        t.tag_configure("separator", foreground=COLORS["border"])

        # Thinking / Reasoning
        t.tag_configure("thinking", foreground=COLORS["text_secondary"], font=italic,
                         lmargin1=24, lmargin2=32)

        # Metadata (token usage, latency)
        t.tag_configure("metadata", foreground=COLORS["text_timestamp"], font=small,
                         lmargin1=24, lmargin2=24)

        # Retry warning
        t.tag_configure("retry", foreground=COLORS["queue_high"], font=bold,
                         lmargin1=24, lmargin2=24)

        # Error
        t.tag_configure("error", foreground=COLORS["tool_fail"], font=bold,
                         lmargin1=16, lmargin2=16)

        # Tool call arguments delta (streaming)
        t.tag_configure("tool_delta", foreground=COLORS["text_secondary"], font=italic,
                         lmargin1=32, lmargin2=40)

        # Debug info
        t.tag_configure("debug", foreground=COLORS["text_timestamp"], font=("TkDefaultFont", 9),
                         lmargin1=16, lmargin2=16)

    def _is_at_bottom(self) -> bool:
        """Check if text widget is currently scrolled to bottom."""
        return self.text.yview()[1] >= 0.999

    def _now(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _insert(self, *args):
        """Enable the text widget, insert, then disable again."""
        at_bottom = self._is_at_bottom()
        self.text.config(state=tk.NORMAL)
        self.text.insert(*args)
        self.text.config(state=tk.DISABLED)
        if at_bottom:
            self.text.see(tk.END)

    def _insert_multi(self, parts: list[tuple]):
        """Batch-insert multiple (index, text, tag?) tuples in one NORMAL window."""
        at_bottom = self._is_at_bottom()
        self.text.config(state=tk.NORMAL)
        for part in parts:
            self.text.insert(*part)
        self.text.config(state=tk.DISABLED)
        if at_bottom:
            self.text.see(tk.END)

    # ─── Public methods ───────────────────────────────────────────────────────

    def add_user_message(self, content: str, queue_kind: QueueKind):
        header_tag = f"user_header_{queue_kind}"
        label = QUEUE_LABELS.get(queue_kind, "普通")
        color_name = QUEUE_LABELS.get(queue_kind, "")
        ts = self._now()
        self._insert_multi([
            (tk.END, "\n", ()),
            (tk.END, f"▶ 你 [{label}]", header_tag),
            (tk.END, f"  {ts}\n", "timestamp"),
            (tk.END, f"  {content}\n", "user_body"),
        ])

    def start_agent_message(self, run_id: str):
        """Insert agent header and create streaming marks for content and thinking."""
        ts = self._now()
        at_bottom = self._is_at_bottom()
        self.text.config(state=tk.NORMAL)
        self.text.insert(tk.END, "\n", ())
        self.text.insert(tk.END, "◀ Agent", "agent_header")
        self.text.insert(tk.END, f"  {ts}\n", "timestamp")
        # Create mark for thinking content (before main content)
        thinking_mark = f"thinking_{run_id}"
        self.text.mark_set(thinking_mark, tk.END)
        self.text.mark_gravity(thinking_mark, tk.RIGHT)
        self._thinking_marks[run_id] = thinking_mark
        self._thinking_started[run_id] = False
        # Create right-gravity mark at current end — delta inserts will push it forward
        mark_name = f"stream_{run_id}"
        self.text.mark_set(mark_name, tk.END)
        self.text.mark_gravity(mark_name, tk.RIGHT)
        self._stream_marks[run_id] = mark_name
        self.text.config(state=tk.DISABLED)
        if at_bottom:
            self.text.see(tk.END)

    def append_delta(self, run_id: str, delta: str):
        """Append streaming text at the mark position."""
        mark = self._stream_marks.get(run_id)
        if not mark:
            return
        at_bottom = self._is_at_bottom()
        self.text.config(state=tk.NORMAL)
        self.text.insert(mark, delta, "agent_body")
        self.text.config(state=tk.DISABLED)
        if at_bottom:
            self.text.see(tk.END)

    def append_thinking(self, run_id: str, delta: str):
        """Append thinking/reasoning content at the thinking mark position."""
        mark = self._thinking_marks.get(run_id)
        if not mark:
            return
        at_bottom = self._is_at_bottom()
        self.text.config(state=tk.NORMAL)
        # Show thinking header on first delta
        if not self._thinking_started.get(run_id, False):
            self.text.insert(mark, "  💭 Thinking…\n", "thinking")
            self._thinking_started[run_id] = True
        self.text.insert(mark, delta, "thinking")
        self.text.config(state=tk.DISABLED)
        if at_bottom:
            self.text.see(tk.END)

    def add_model_metadata(self, run_id: str, input_tokens: int, output_tokens: int,
                            total_tokens: int, latency_ms: float | None):
        """Add model metadata (token usage, latency) to the message."""
        mark = self._stream_marks.get(run_id)
        if not mark:
            return
        at_bottom = self._is_at_bottom()
        latency_str = f"{latency_ms:.0f}ms" if latency_ms else "N/A"
        meta_text = f"\n  📊 Tokens: {input_tokens:,}↑ / {output_tokens:,}↓ / {total_tokens:,}∑  ·  Latency: {latency_str}\n"
        self.text.config(state=tk.NORMAL)
        self.text.insert(mark, meta_text, "metadata")
        self.text.config(state=tk.DISABLED)
        if at_bottom:
            self.text.see(tk.END)

    def add_retry_info(self, run_id: str, attempt: int, max_retries: int,
                       error_type: str, error_message: str):
        """Add retry warning information."""
        mark = self._stream_marks.get(run_id)
        at_bottom = self._is_at_bottom()
        retry_text = f"\n  ⚠️ 重试 {attempt}/{max_retries}: [{error_type}] {error_message}\n"
        self.text.config(state=tk.NORMAL)
        if mark:
            self.text.insert(mark, retry_text, "retry")
        else:
            self.text.insert(tk.END, retry_text, "retry")
        self.text.config(state=tk.DISABLED)
        if at_bottom:
            self.text.see(tk.END)

    def add_error(self, run_id: str, message: str):
        """Add error message."""
        at_bottom = self._is_at_bottom()
        error_text = f"\n  ❌ {message}\n"
        self.text.config(state=tk.NORMAL)
        self.text.insert(tk.END, error_text, "error")
        self.text.config(state=tk.DISABLED)
        if at_bottom:
            self.text.see(tk.END)

    def add_agent_interrupt(self, run_id: str, interrupt_type: str):
        """Add agent interrupt notification."""
        type_labels = {
            "user": "用户中断",
            "scheduler": "调度器中断",
            "error": "错误中断",
        }
        label = type_labels.get(interrupt_type, f"中断({interrupt_type})")
        at_bottom = self._is_at_bottom()
        interrupt_text = f"\n  ⏹️ {label}\n"
        self.text.config(state=tk.NORMAL)
        self.text.insert(tk.END, interrupt_text, "interrupt")
        self.text.config(state=tk.DISABLED)
        if at_bottom:
            self.text.see(tk.END)

    def add_debug_info(self, message: str):
        """Add debug information."""
        at_bottom = self._is_at_bottom()
        debug_text = f"\n  [Debug] {message}\n"
        self.text.config(state=tk.NORMAL)
        self.text.insert(tk.END, debug_text, "debug")
        self.text.config(state=tk.DISABLED)
        if at_bottom:
            self.text.see(tk.END)

    def finish_agent_message(self, run_id: str, stop_reason: str, duration_ms: float):
        """Insert completion footer and clean up the streaming mark."""
        mark = self._stream_marks.pop(run_id, None)
        # Clean up thinking mark
        thinking_mark = self._thinking_marks.pop(run_id, None)
        self._thinking_started.pop(run_id, None)
        # Clean up metadata mark
        self._metadata_marks.pop(run_id, None)
        secs = duration_ms / 1000
        footer = f"\n  ─ 完成 · {stop_reason} · {secs:.1f}s\n"
        at_bottom = self._is_at_bottom()
        if mark:
            self.text.config(state=tk.NORMAL)
            # Insert after the stream mark position
            self.text.insert(mark, footer, "system")
            self.text.mark_unset(mark)
            if thinking_mark:
                self.text.mark_unset(thinking_mark)
            self.text.config(state=tk.DISABLED)
        else:
            self._insert(tk.END, footer, "system")
        if at_bottom:
            self.text.see(tk.END)

    def add_tool_call(self, tool_name: str, args_preview: str, run_id: str, tool_call_id: str = ""):
        """Append a tool call line to the active agent bubble."""
        line = f"\n  ⚙ {tool_name}({args_preview})"
        mark = self._stream_marks.get(run_id)
        at_bottom = self._is_at_bottom()
        if mark:
            self.text.config(state=tk.NORMAL)
            self.text.insert(mark, line, "tool_call")
            # Create mark for tool call arguments delta
            if tool_call_id:
                tc_mark = f"toolcall_{tool_call_id}"
                self.text.mark_set(tc_mark, mark)
                self.text.mark_gravity(tc_mark, tk.RIGHT)
                self._tool_call_marks[tool_call_id] = tc_mark
                self._tool_call_started[tool_call_id] = False
            self.text.config(state=tk.DISABLED)
        else:
            self._insert(tk.END, line + "\n", "tool_call")
            if tool_call_id:
                tc_mark = f"toolcall_{tool_call_id}"
                self.text.mark_set(tc_mark, tk.END)
                self.text.mark_gravity(tc_mark, tk.RIGHT)
                self._tool_call_marks[tool_call_id] = tc_mark
                self._tool_call_started[tool_call_id] = False
        if at_bottom:
            self.text.see(tk.END)

    def append_tool_call_delta(self, run_id: str, tool_call_id: str, delta: str):
        """Append tool call arguments delta (streaming)."""
        mark = self._tool_call_marks.get(tool_call_id)
        if not mark:
            return
        at_bottom = self._is_at_bottom()
        self.text.config(state=tk.NORMAL)
        # Show header on first delta
        if not self._tool_call_started.get(tool_call_id, False):
            self.text.insert(mark, "\n    {", "tool_delta")
            self._tool_call_started[tool_call_id] = True
        self.text.insert(mark, delta, "tool_delta")
        self.text.config(state=tk.DISABLED)
        if at_bottom:
            self.text.see(tk.END)

    def add_tool_result(self, tool_name: str, success: bool, output: str,
                        duration_ms: float, run_id: str):
        """Append a tool result line."""
        icon = "✓" if success else "✗"
        tag = "tool_ok" if success else "tool_fail"
        # Truncate long output
        if len(output) > 200:
            output = output[:197] + "..."
        output_line = output.replace("\n", " ").strip()
        line = f"\n  {icon} {output_line}  [{duration_ms:.0f}ms]\n"
        mark = self._stream_marks.get(run_id)
        at_bottom = self._is_at_bottom()
        if mark:
            self.text.config(state=tk.NORMAL)
            self.text.insert(mark, line, tag)
            self.text.config(state=tk.DISABLED)
        else:
            self._insert(tk.END, line, tag)
        if at_bottom:
            self.text.see(tk.END)

    def add_interrupt(self, reason: str):
        self._insert_multi([
            (tk.END, "\n", ()),
            (tk.END, f"⚡ 中断: {reason}\n", "interrupt"),
        ])

    def add_system(self, content: str):
        self._insert_multi([
            (tk.END, f"\n{content}\n", "system"),
        ])

    def clear(self):
        self.text.config(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        self.text.config(state=tk.DISABLED)
        self._stream_marks.clear()
        self._thinking_marks.clear()
        self._thinking_started.clear()
        self._metadata_marks.clear()
        self._tool_call_marks.clear()
        self._tool_call_started.clear()
