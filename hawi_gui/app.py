"""HawiGuiApp — main window orchestrator."""

from __future__ import annotations

import os
import queue
import tkinter as tk
from tkinter import ttk

from .protocol import (
    CmdClearContext,
    CmdClearQueue,
    CmdEnqueue,
    CmdStop,
    CmdSwitchModel,
    QueueKind,
    UiAgentInterrupt,
    UiDebugInfo,
    UiError,
    UiInterrupt,
    UiModelMetadata,
    UiModelRetry,
    UiReady,
    UiRunStart,
    UiRunStop,
    UiStatusUpdate,
    UiTextDelta,
    UiThinkingDelta,
    UiToolCall,
    UiToolCallDelta,
    UiToolResult,
)
from .scheduler_bridge import SchedulerThread
from .theme import COLORS, configure_ttk_style
from .widgets.chat_view import ChatView
from .widgets.input_area import InputArea
from .widgets.model_dialog import ModelSelectionDialog
from .widgets.status_bar import StatusBarFrame

HELP_TEXT = (
    "命令: /clear(清空上下文)  /cq(清普通队列)  /chq(清高优队列)  "
    "/cuq(清紧急队列)  /ca(清所有队列)  /quit(退出)\n"
    "快捷键: Shift+Tab(切换优先级)  Ctrl+L(清空上下文)  Esc(清空输入)"
)


class HawiGuiApp:
    """Main GUI application."""

    def __init__(self, model_name: str, root: tk.Tk | None = None):
        self.model_name = model_name
        self.ui_queue: queue.Queue = queue.Queue()
        self.cmd_queue: queue.Queue = queue.Queue()
        self._active_runs: set[str] = set()
        self._sched_thread: SchedulerThread | None = None

        self.root = root if root is not None else tk.Tk()
        self.root.title(f"Hawi — {model_name}")
        self.root.geometry("960x720")
        self.root.minsize(640, 480)
        self.root.configure(bg=COLORS["bg_window"])

        configure_ttk_style(self.root)
        self._build_menu()
        self._build_widgets()
        self._bind_shortcuts()

    # ─── UI construction ─────────────────────────────────────────────────────

    def _build_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        model_menu = tk.Menu(menubar, tearoff=False)
        menubar.add_cascade(label="模型", menu=model_menu)
        model_menu.add_command(label="切换模型…", command=self._show_model_switcher)
        model_menu.add_separator()
        self._model_menu_label = tk.StringVar(value=f"当前: {self.model_name}")
        model_menu.add_command(
            label=f"当前: {self.model_name}",
            state=tk.DISABLED,
        )
        self._model_menu = model_menu

        help_menu = tk.Menu(menubar, tearoff=False)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="快捷键说明", command=self._show_help)

    def _build_widgets(self):
        # Status bar (top)
        self.status_bar = StatusBarFrame(
            self.root,
            model_name=self.model_name,
            on_model_click=self._show_model_switcher,
        )
        self.status_bar.pack(side=tk.TOP, fill=tk.X)

        # Chat view (middle, expands)
        self.chat_view = ChatView(self.root)
        self.chat_view.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Input area (bottom)
        self.input_area = InputArea(self.root, on_submit=self._on_submit)
        self.input_area.pack(side=tk.BOTTOM, fill=tk.X)

    def _bind_shortcuts(self):
        self.root.bind("<Control-l>", self._cmd_clear_context)
        self.root.bind("<Control-m>", self._show_model_switcher_event)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ─── Run ─────────────────────────────────────────────────────────────────

    def run(self):
        """Start the scheduler thread and enter the tkinter mainloop."""
        self._sched_thread = SchedulerThread(
            ui_queue=self.ui_queue,
            cmd_queue=self.cmd_queue,
            model_name=self.model_name,
        )
        self._sched_thread.start()
        self.chat_view.add_system("正在启动 Scheduler…")

        # Poll for ready signal
        self.root.after(100, self._poll_ready)
        # Start UI message polling
        self.root.after(50, self._poll_ui_queue)

        # macOS: bring Python process to foreground
        self._activate_macos_window()

        self.input_area.focus()
        self.root.mainloop()

    # ─── Polling ─────────────────────────────────────────────────────────────

    def _poll_ready(self):
        if self._sched_thread and self._sched_thread._ready.is_set():
            self.chat_view.add_system("✓ Scheduler 已就绪，开始对话吧")
        else:
            self.root.after(100, self._poll_ready)

    def _poll_ui_queue(self):
        """Drain the ui_queue and dispatch all messages in one pass."""
        batch = []
        try:
            while True:
                batch.append(self.ui_queue.get_nowait())
        except queue.Empty:
            pass

        if batch:
            # Check if at bottom before processing batch
            at_bottom = self.chat_view._is_at_bottom()
            # Enable text widget once for the whole batch
            self.chat_view.text.config(state=tk.NORMAL)
            for msg in batch:
                self._dispatch(msg)
            self.chat_view.text.config(state=tk.DISABLED)
            # Only auto-scroll if user was already at bottom
            if at_bottom:
                self.chat_view.text.see(tk.END)

        self.root.after(50, self._poll_ui_queue)

    def _dispatch(self, msg):
        """Route a UI message to the appropriate widget method."""
        if isinstance(msg, UiStatusUpdate):
            self.status_bar.update_status(msg)

        elif isinstance(msg, UiReady):
            self.model_name = msg.model_name
            self.root.title(f"Hawi — {msg.model_name}")
            self.status_bar.set_model(msg.model_name)
            self.chat_view.add_system(f"✓ 已切换到模型: {msg.model_name}")
            self._update_model_menu(msg.model_name)

        elif isinstance(msg, UiRunStart):
            if msg.user_content:
                self.chat_view.add_user_message(msg.user_content, msg.queue_kind)
            if msg.run_id not in self._active_runs:
                self._active_runs.add(msg.run_id)
                self.chat_view.start_agent_message(msg.run_id)

        elif isinstance(msg, UiTextDelta):
            if msg.run_id not in self._active_runs:
                self._active_runs.add(msg.run_id)
                self.chat_view.start_agent_message(msg.run_id)
            self.chat_view.append_delta(msg.run_id, msg.delta)

        elif isinstance(msg, UiThinkingDelta):
            if msg.run_id not in self._active_runs:
                self._active_runs.add(msg.run_id)
                self.chat_view.start_agent_message(msg.run_id)
            self.chat_view.append_thinking(msg.run_id, msg.delta)

        elif isinstance(msg, UiRunStop):
            self._active_runs.discard(msg.run_id)
            self.chat_view.finish_agent_message(msg.run_id, msg.stop_reason, msg.duration_ms)

        elif isinstance(msg, UiToolCall):
            args_preview = ", ".join(
                f"{k}={str(v)[:40]}" for k, v in msg.arguments.items()
            ) if msg.arguments else ""
            self.chat_view.add_tool_call(msg.tool_name, args_preview, msg.run_id, msg.tool_call_id)

        elif isinstance(msg, UiToolResult):
            self.chat_view.add_tool_result(
                msg.tool_name, msg.success, msg.output, msg.duration_ms, msg.run_id
            )

        elif isinstance(msg, UiInterrupt):
            self.chat_view.add_interrupt(msg.reason)

        elif isinstance(msg, UiError):
            self.chat_view.add_error("", msg.message)

        elif isinstance(msg, UiModelMetadata):
            self.chat_view.add_model_metadata(
                msg.run_id,
                msg.input_tokens,
                msg.output_tokens,
                msg.total_tokens,
                msg.latency_ms,
            )

        elif isinstance(msg, UiModelRetry):
            self.chat_view.add_retry_info(
                msg.run_id,
                msg.attempt,
                msg.max_retries,
                msg.error_type,
                msg.error_message,
            )

        elif isinstance(msg, UiToolCallDelta):
            self.chat_view.append_tool_call_delta(
                msg.run_id,
                msg.tool_call_id,
                msg.delta,
            )

        elif isinstance(msg, UiAgentInterrupt):
            self.chat_view.add_agent_interrupt(
                msg.run_id,
                msg.interrupt_type,
            )

        elif isinstance(msg, UiDebugInfo):
            self.chat_view.add_debug_info(msg.message)

    # ─── Input handling ───────────────────────────────────────────────────────

    def _on_submit(self, text: str, queue_kind: QueueKind):
        if text.startswith("/"):
            self._handle_command(text)
        else:
            self.cmd_queue.put(CmdEnqueue(content=text, queue=queue_kind))

    def _handle_command(self, text: str):
        cmd = text.lower().strip()
        if cmd in ("/help", "/h"):
            self.chat_view.add_system(HELP_TEXT)
        elif cmd == "/clear":
            self._cmd_clear_context()
        elif cmd == "/cq":
            self.cmd_queue.put(CmdClearQueue(queue="normal"))
            self.chat_view.add_system("✓ 普通队列已清空")
        elif cmd == "/chq":
            self.cmd_queue.put(CmdClearQueue(queue="high_prio"))
            self.chat_view.add_system("✓ 高优队列已清空")
        elif cmd == "/cuq":
            self.cmd_queue.put(CmdClearQueue(queue="urgent"))
            self.chat_view.add_system("✓ 紧急队列已清空")
        elif cmd == "/ca":
            self.cmd_queue.put(CmdClearQueue(queue="all"))
            self.chat_view.add_system("✓ 所有队列已清空")
        elif cmd in ("/quit", "/q", "/exit"):
            self._on_close()
        else:
            self.chat_view.add_system(f"未知命令: {text}，输入 /help 查看帮助")

    def _cmd_clear_context(self, event=None):
        self.cmd_queue.put(CmdClearContext())
        self.chat_view.add_system("✓ 对话上下文已清空")

    # ─── Model switching ──────────────────────────────────────────────────────

    def _show_model_switcher_event(self, event=None):
        """Handler for Ctrl+M keyboard shortcut."""
        self._show_model_switcher()
        return "break"

    def _show_model_switcher(self):
        from hawi.models import model_registry
        models = model_registry.list_models()
        if not models:
            self.chat_view.add_system("没有可用的模型工厂")
            return
        dlg = ModelSelectionDialog(
            self.root, models,
            title="切换模型",
            modal=False,
            on_select=self._switch_model,
        )

    def _switch_model(self, model_name: str):
        if model_name == self.model_name:
            return
        self.chat_view.add_system(f"正在切换到模型: {model_name}…")
        self.cmd_queue.put(CmdSwitchModel(model_name=model_name))

    def _update_model_menu(self, model_name: str):
        """Update the disabled menu label showing current model."""
        try:
            last_idx = self._model_menu.index(tk.END)
            if last_idx is None:
                return
            self._model_menu.entryconfig(last_idx, label=f"当前: {model_name}")
        except Exception:
            pass

    # ─── Close ───────────────────────────────────────────────────────────────

    def _on_close(self):
        if self._sched_thread:
            self.cmd_queue.put(CmdStop())
        self.root.destroy()

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _show_help(self):
        self.chat_view.add_system(HELP_TEXT)

    @staticmethod
    def _activate_macos_window():
        """On macOS, activate the Python process so its windows come to front."""
        import platform
        if platform.system() != "Darwin":
            return
        try:
            import subprocess
            subprocess.Popen([
                "osascript", "-e",
                'tell application "System Events" to set frontmost '
                'of the first process whose unix id is '
                f'{os.getpid()} to true'
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
