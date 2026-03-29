"""StatusBarFrame — top status row showing scheduler/agent state and queue lengths."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from ..protocol import UiStatusUpdate
from ..theme import COLORS, STATE_COLORS, QUEUE_COLORS


class StatusBarFrame(ttk.Frame):
    """Top bar: Scheduler state | Agent state | Queue lengths | Model name."""

    def __init__(
        self,
        parent: tk.Widget,
        factory_name: str = "",
        on_model_click: Callable[[], None] | None = None,
        **kwargs
    ):
        super().__init__(parent, style="StatusBar.TFrame", **kwargs)
        self._factory_name = factory_name
        self._on_model_click = on_model_click
        self._build()

    def _build(self):
        pad = {"padx": 6, "pady": 4}

        # Scheduler state
        ttk.Label(self, text="Scheduler:", style="StatusBarSecondary.TLabel").pack(side=tk.LEFT, **pad)
        self._sched_label = ttk.Label(self, text="IDLE", style="StatusBar.TLabel",
                                      foreground=STATE_COLORS.get("IDLE", COLORS["state_idle"]))
        self._sched_label.pack(side=tk.LEFT, padx=(0, 12), pady=4)

        ttk.Separator(self, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, pady=4)

        # Agent state
        ttk.Label(self, text="Agent:", style="StatusBarSecondary.TLabel").pack(side=tk.LEFT, padx=(12, 6), pady=4)
        self._agent_label = ttk.Label(self, text="IDLE", style="StatusBar.TLabel",
                                      foreground=STATE_COLORS.get("IDLE", COLORS["state_idle"]))
        self._agent_label.pack(side=tk.LEFT, padx=(0, 12), pady=4)

        ttk.Separator(self, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, pady=4)

        # Queue lengths (初始为0，显示灰色)
        self._urgent_label = ttk.Label(
            self, text="紧急: 0", style="StatusBar.TLabel",
            foreground=COLORS["text_secondary"]
        )
        self._urgent_label.pack(side=tk.LEFT, padx=(12, 6), pady=4)

        self._high_label = ttk.Label(
            self, text="高优: 0", style="StatusBar.TLabel",
            foreground=COLORS["text_secondary"]
        )
        self._high_label.pack(side=tk.LEFT, padx=6, pady=4)

        self._normal_label = ttk.Label(
            self, text="普通: 0", style="StatusBar.TLabel",
            foreground=COLORS["text_secondary"]
        )
        self._normal_label.pack(side=tk.LEFT, padx=6, pady=4)

        # Model name (right side, clickable)
        self._model_label = tk.Label(
            self,
            text=self._factory_name,
            bg=COLORS["bg_window"],
            fg=COLORS["queue_normal"],
            font=("TkDefaultFont", 11, "underline"),
            cursor="hand2",
        )
        self._model_label.pack(side=tk.RIGHT, padx=8, pady=4)

        if self._on_model_click:
            self._model_label.bind("<Button-1>", lambda e: self._on_model_click())

    def update_status(self, msg: UiStatusUpdate):
        """Update all labels from a UiStatusUpdate message."""
        # Scheduler state
        s = msg.scheduler_state
        s_color = STATE_COLORS.get(s, COLORS["text_secondary"])
        self._sched_label.config(text=s, foreground=s_color)

        # Agent state
        a = msg.agent_state
        a_color = STATE_COLORS.get(a, COLORS["text_secondary"])
        self._agent_label.config(text=a, foreground=a_color)

        # Queue lengths
        ql = msg.queue_lengths
        u = ql.get("urgent", 0)
        h = ql.get("high_prio", 0)
        n = ql.get("normal", 0)

        self._urgent_label.config(
            text=f"紧急: {u}",
            foreground=COLORS["queue_urgent"] if u > 0 else COLORS["text_secondary"],
        )
        self._high_label.config(
            text=f"高优: {h}",
            foreground=COLORS["queue_high"] if h > 0 else COLORS["text_secondary"],
        )
        self._normal_label.config(
            text=f"普通: {n}",
            foreground=COLORS["queue_normal"] if n > 0 else COLORS["text_secondary"],
        )

    def set_model(self, factory_name: str):
        """Update the displayed model name."""
        self._factory_name = factory_name
        self._model_label.config(text=factory_name)
