"""ModelSelectionDialog — model picker used at startup and for in-app switching."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ..theme import COLORS


class ModelSelectionDialog(tk.Toplevel):
    """Modal/non-modal dialog for selecting a model factory.

    Usage (startup, modal):
        dlg = ModelSelectionDialog(parent, factories, title="Select Model")
        parent.wait_window(dlg)
        selected = dlg.result  # None if cancelled

    Usage (in-app, non-modal):
        dlg = ModelSelectionDialog(parent, factories, modal=False,
                                   on_select=callback)
    """

    def __init__(
        self,
        parent: tk.Widget,
        factories: list[str],
        title: str = "选择模型",
        modal: bool = True,
        on_select: "Callable[[str], None] | None" = None,
    ):
        super().__init__(parent)
        self.title(title)
        self.result: str | None = None
        self._factories = factories
        self._on_select = on_select
        self._modal = modal

        self.resizable(False, False)
        self.configure(bg=COLORS["bg_window"])

        self._build()
        self._center(parent)

        if modal:
            self.transient(parent)
            self.grab_set()
            self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        else:
            self.protocol("WM_DELETE_WINDOW", self.destroy)

        self.deiconify()  # Show the dialog window
        self.lift()
        self.focus_force()

    def _build(self):
        pad = {"padx": 16, "pady": 8}

        # Title label
        ttk.Label(self, text="选择模型工厂", font=("TkDefaultFont", 14, "bold")).pack(**pad)

        # Search box
        self._search_var = tk.StringVar()
        self._search_var.trace_add("write", self._on_search)
        search_frame = ttk.Frame(self)
        search_frame.pack(fill=tk.X, padx=16, pady=(0, 4))
        ttk.Label(search_frame, text="搜索:").pack(side=tk.LEFT)
        ttk.Entry(search_frame, textvariable=self._search_var, width=30).pack(
            side=tk.LEFT, padx=(8, 0), fill=tk.X, expand=True
        )

        # Listbox
        list_frame = tk.Frame(self, bg=COLORS["bg_chat"])
        list_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=4)

        self._listbox = tk.Listbox(
            list_frame,
            selectmode=tk.SINGLE,
            bg=COLORS["bg_chat"],
            fg=COLORS["text_primary"],
            selectbackground=COLORS["queue_normal"],
            selectforeground="#FFFFFF",
            font=("TkDefaultFont", 13),
            relief=tk.FLAT,
            borderwidth=0,
            activestyle="none",
            height=min(len(self._factories), 12),
            width=40,
        )
        sb = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self._listbox.yview)
        self._listbox.configure(yscrollcommand=sb.set)
        self._listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self._populate(self._factories)

        self._listbox.bind("<Double-Button-1>", self._on_select_event)
        self._listbox.bind("<Return>", self._on_select_event)

        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=16, pady=(4, 12))

        ttk.Button(btn_frame, text="取消", command=self._on_cancel).pack(side=tk.RIGHT)
        ttk.Button(btn_frame, text="选择", command=self._on_select_event,
                   style="Send.TButton").pack(side=tk.RIGHT, padx=(0, 8))

        # Keyboard
        self.bind("<Return>", self._on_select_event)
        self.bind("<Escape>", lambda e: self._on_cancel())

        # Select first item
        if self._factories:
            self._listbox.selection_set(0)
            self._listbox.focus_set()

    def _populate(self, items: list[str]):
        self._listbox.delete(0, tk.END)
        for item in items:
            self._listbox.insert(tk.END, item)

    def _on_search(self, *_):
        query = self._search_var.get().lower()
        filtered = [f for f in self._factories if query in f.lower()]
        self._populate(filtered)
        if filtered:
            self._listbox.selection_set(0)

    def _on_select_event(self, event=None):
        sel = self._listbox.curselection()
        if not sel:
            return
        name = self._listbox.get(sel[0])
        self.result = name
        if self._on_select:
            self._on_select(name)
        self.destroy()

    def _on_cancel(self):
        self.result = None
        self.destroy()

    def _center(self, parent: tk.Widget):
        self.update_idletasks()
        w = self.winfo_reqwidth()
        h = self.winfo_reqheight()
        try:
            px = parent.winfo_rootx() + (parent.winfo_width() - w) // 2
            py = parent.winfo_rooty() + (parent.winfo_height() - h) // 2
        except Exception:
            px, py = 200, 200
        self.geometry(f"+{max(0, px)}+{max(0, py)}")
