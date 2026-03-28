"""Hawi GUI Theme — colors, fonts, ttk style configuration."""

from __future__ import annotations

import platform
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

# ─── Color palette ───────────────────────────────────────────────────────────

COLORS = {
    # Backgrounds
    "bg_window":    "#FAFAFA",
    "bg_chat":      "#FFFFFF",
    "bg_status":    "#EEEEEE",
    "bg_input":     "#F5F5F5",
    "bg_input_box": "#FFFFFF",
    "border":       "#DDDDDD",

    # Text
    "text_primary":   "#1A1A1A",
    "text_secondary": "#888888",
    "text_timestamp": "#AAAAAA",

    # Queue colors
    "queue_normal":   "#2979FF",  # blue
    "queue_high":     "#FF9100",  # orange
    "queue_urgent":   "#F44336",  # red

    # State colors
    "state_idle":         "#4CAF50",  # green
    "state_running":      "#00BCD4",  # cyan
    "state_interrupting": "#F44336",  # red
    "state_ready":        "#FFC107",  # amber

    # Message bubble colors
    "user_bg":      "#F0F7FF",
    "agent_header": "#2E7D32",  # dark green
    "tool_ok":      "#4CAF50",
    "tool_fail":    "#F44336",
    "interrupt_fg": "#F44336",
    "system_fg":    "#888888",

    # Button
    "btn_send_bg": "#2979FF",
    "btn_send_fg": "#FFFFFF",
}

QUEUE_COLORS = {
    "normal":   COLORS["queue_normal"],
    "high_prio": COLORS["queue_high"],
    "urgent":   COLORS["queue_urgent"],
}

QUEUE_LABELS = {
    "normal":    "普通",
    "high_prio": "高优",
    "urgent":    "紧急",
}

STATE_COLORS = {
    "IDLE":         COLORS["state_idle"],
    "READY":        COLORS["state_ready"],
    "RUNNING":      COLORS["state_running"],
    "INTERRUPTING": COLORS["state_interrupting"],
}


def _detect_theme(root: tk.Tk | None = None) -> str:
    """Return the best available ttk theme for the current platform."""
    sys = platform.system()
    # Create style with root if provided, otherwise use default
    style = ttk.Style(root) if root else ttk.Style()
    available = style.theme_names()
    if sys == "Darwin" and "aqua" in available:
        return "aqua"
    if sys == "Windows" and "vista" in available:
        return "vista"
    if "clam" in available:
        return "clam"
    return "default"


def configure_ttk_style(root: tk.Tk) -> ttk.Style:
    """Configure ttk styles and return the Style instance."""
    style = ttk.Style(root)
    style.theme_use(_detect_theme(root))

    # StatusBar frame
    style.configure(
        "StatusBar.TFrame",
        background=COLORS["bg_status"],
    )
    style.configure(
        "StatusBar.TLabel",
        background=COLORS["bg_status"],
        foreground=COLORS["text_primary"],
    )
    style.configure(
        "StatusBarSecondary.TLabel",
        background=COLORS["bg_status"],
        foreground=COLORS["text_secondary"],
    )

    # InputArea frame
    style.configure(
        "InputArea.TFrame",
        background=COLORS["bg_input"],
    )
    style.configure(
        "InputArea.TLabel",
        background=COLORS["bg_input"],
        foreground=COLORS["text_secondary"],
    )

    # Queue indicator labels
    style.configure(
        "QueueActive.TLabel",
        background=COLORS["bg_input"],
        foreground=COLORS["text_primary"],
        font=("TkDefaultFont", 11, "bold"),
    )
    style.configure(
        "QueueInactive.TLabel",
        background=COLORS["bg_input"],
        foreground=COLORS["text_secondary"],
        font=("TkDefaultFont", 11),
    )

    # Send button
    style.configure(
        "Send.TButton",
        font=("TkDefaultFont", 12, "bold"),
    )

    return style


def get_fonts(root: tk.Tk) -> dict[str, tuple]:
    """Return font tuples for various UI elements."""
    default_family = tkfont.nametofont("TkDefaultFont").actual()["family"]
    mono_family = tkfont.nametofont("TkFixedFont").actual()["family"]

    return {
        "chat_body":   (default_family, 13),
        "chat_header": (default_family, 12, "bold"),
        "chat_small":  (default_family, 11),
        "status":      (default_family, 12),
        "input":       (default_family, 13),
        "mono":        (mono_family, 12),
    }
