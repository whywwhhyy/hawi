"""InputArea — text entry, queue selector, and send button."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from ..protocol import QueueKind
from ..theme import COLORS, QUEUE_COLORS, QUEUE_LABELS


class InputArea(ttk.Frame):
    """Bottom input area: queue indicator row + text entry + send button."""

    def __init__(
        self,
        parent: tk.Widget,
        on_submit: Callable[[str, QueueKind], None],
        **kwargs,
    ):
        super().__init__(parent, style="InputArea.TFrame", **kwargs)
        self.on_submit = on_submit
        self._current_queue: QueueKind = "high_prio"
        self._queue_order: list[QueueKind] = ["normal", "high_prio", "urgent"]
        self._queue_labels: dict[QueueKind, ttk.Label] = {}
        self._build()

    def _build(self):
        # Top divider
        divider = tk.Frame(self, bg=COLORS["border"], height=1)
        divider.pack(side=tk.TOP, fill=tk.X)

        # Queue indicator row
        indicator_row = ttk.Frame(self, style="InputArea.TFrame")
        indicator_row.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(4, 0))

        ttk.Label(indicator_row, text="Shift+Tab 切换:", style="InputArea.TLabel").pack(
            side=tk.LEFT, padx=(0, 4)
        )
        for kind in self._queue_order:
            lbl = ttk.Label(indicator_row, text=QUEUE_LABELS[kind], style="QueueInactive.TLabel")
            lbl.pack(side=tk.LEFT, padx=4)
            self._queue_labels[kind] = lbl

        self._refresh_indicator()

        # Input row
        input_row = ttk.Frame(self, style="InputArea.TFrame")
        input_row.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        # Text input (multi-line, Enter to send, Shift+Enter for newline)
        self.text_input = tk.Text(
            input_row,
            height=3,
            font=("TkDefaultFont", 13),
            bg=COLORS["bg_input_box"],
            fg=COLORS["text_primary"],
            relief=tk.SOLID,
            borderwidth=1,
            wrap=tk.WORD,
            undo=True,
        )
        self.text_input.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Highlight color on focus
        self.text_input.bind("<FocusIn>",  lambda e: self.text_input.config(highlightthickness=2,
                                                                              highlightcolor=COLORS["queue_normal"],
                                                                              highlightbackground=COLORS["border"]))
        self.text_input.bind("<FocusOut>", lambda e: self.text_input.config(highlightthickness=1))

        # Key bindings
        self.text_input.bind("<Return>",       self._on_return)
        self.text_input.bind("<Escape>",       self._on_escape)
        self.text_input.bind("<Shift-Tab>",    self._on_tab)

        # Send button
        self.send_btn = ttk.Button(
            input_row,
            text="发送",
            style="Send.TButton",
            command=self._send,
            width=6,
        )
        self.send_btn.pack(side=tk.LEFT, padx=(8, 0))

    def _refresh_indicator(self):
        for kind, lbl in self._queue_labels.items():
            color = QUEUE_COLORS[kind]
            if kind == self._current_queue:
                lbl.config(style="QueueActive.TLabel", foreground=color)
            else:
                lbl.config(style="QueueInactive.TLabel", foreground=COLORS["text_secondary"])

    def _on_return(self, event: tk.Event) -> str:
        """Enter sends; Shift+Enter inserts newline."""
        if event.state & 0x1:  # Shift is held
            return  # allow default (newline)
        self._send()
        return "break"  # suppress default newline

    def _on_escape(self, event: tk.Event):
        self.clear()
        return "break"

    def _on_tab(self, event: tk.Event) -> str:
        self.cycle_queue()
        return "break"  # suppress focus change

    def _send(self):
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            return
        self.clear()
        self.on_submit(text, self._current_queue)

    def cycle_queue(self):
        idx = self._queue_order.index(self._current_queue)
        self._current_queue = self._queue_order[(idx + 1) % len(self._queue_order)]
        self._refresh_indicator()

    def set_queue(self, kind: QueueKind):
        self._current_queue = kind
        self._refresh_indicator()

    @property
    def current_queue(self) -> QueueKind:
        return self._current_queue

    def clear(self):
        self.text_input.delete("1.0", tk.END)

    def focus(self):
        self.text_input.focus_set()
