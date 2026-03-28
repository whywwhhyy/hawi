"""ChatView — scrollable chat area using tk.Text with rich tag formatting."""

from __future__ import annotations

import tkinter as tk
from datetime import datetime
from tkinter import ttk

from ..protocol import QueueKind, UiToolCall, UiToolResult
from ..theme import COLORS, QUEUE_COLORS, QUEUE_LABELS


class ChatView(tk.Frame):
    """Scrollable chat area displaying message bubbles."""

    def __init__(self, parent: tk.Widget, **kwargs):
        kwargs.setdefault("bg", COLORS["bg_chat"])
        super().__init__(parent, **kwargs)
        self._stream_marks: dict[str, str] = {}  # run_id -> tk mark name
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

    def _now(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _insert(self, *args):
        """Enable the text widget, insert, then disable again."""
        self.text.config(state=tk.NORMAL)
        self.text.insert(*args)
        self.text.config(state=tk.DISABLED)

    def _insert_multi(self, parts: list[tuple]):
        """Batch-insert multiple (index, text, tag?) tuples in one NORMAL window."""
        self.text.config(state=tk.NORMAL)
        for part in parts:
            self.text.insert(*part)
        self.text.config(state=tk.DISABLED)
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
        """Insert agent header and create a streaming mark."""
        ts = self._now()
        self.text.config(state=tk.NORMAL)
        self.text.insert(tk.END, "\n", ())
        self.text.insert(tk.END, "◀ Agent", "agent_header")
        self.text.insert(tk.END, f"  {ts}\n", "timestamp")
        # Create right-gravity mark at current end — delta inserts will push it forward
        mark_name = f"stream_{run_id}"
        self.text.mark_set(mark_name, tk.END)
        self.text.mark_gravity(mark_name, tk.RIGHT)
        self._stream_marks[run_id] = mark_name
        self.text.config(state=tk.DISABLED)
        self.text.see(tk.END)

    def append_delta(self, run_id: str, delta: str):
        """Append streaming text at the mark position."""
        mark = self._stream_marks.get(run_id)
        if not mark:
            return
        self.text.config(state=tk.NORMAL)
        self.text.insert(mark, delta, "agent_body")
        self.text.config(state=tk.DISABLED)
        self.text.see(tk.END)

    def finish_agent_message(self, run_id: str, stop_reason: str, duration_ms: float):
        """Insert completion footer and clean up the streaming mark."""
        mark = self._stream_marks.pop(run_id, None)
        secs = duration_ms / 1000
        footer = f"\n  ─ 完成 · {stop_reason} · {secs:.1f}s\n"
        if mark:
            self.text.config(state=tk.NORMAL)
            # Insert after the stream mark position
            self.text.insert(mark, footer, "system")
            self.text.mark_unset(mark)
            self.text.config(state=tk.DISABLED)
        else:
            self._insert(tk.END, footer, "system")
        self.text.see(tk.END)

    def add_tool_call(self, tool_name: str, args_preview: str, run_id: str):
        """Append a tool call line to the active agent bubble."""
        line = f"\n  ⚙ {tool_name}({args_preview})"
        mark = self._stream_marks.get(run_id)
        if mark:
            self.text.config(state=tk.NORMAL)
            self.text.insert(mark, line, "tool_call")
            self.text.config(state=tk.DISABLED)
            self.text.see(tk.END)
        else:
            self._insert(tk.END, line + "\n", "tool_call")
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
        line = f"\n  {icon} {output_line}  [{duration_ms:.0f}ms]"
        mark = self._stream_marks.get(run_id)
        if mark:
            self.text.config(state=tk.NORMAL)
            self.text.insert(mark, line, tag)
            self.text.config(state=tk.DISABLED)
            self.text.see(tk.END)
        else:
            self._insert(tk.END, line + "\n", tag)
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
