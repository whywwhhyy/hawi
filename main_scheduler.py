"""Hawi Scheduler TUI - Interactive terminal app using Textual.

架构：
- UI 线程：Textual app（主线程）
- Scheduler 线程：asyncio event loop（独立线程）
- 双向通讯：ui_queue（scheduler→UI）、cmd_queue（UI→scheduler）
"""

from __future__ import annotations

import asyncio
import queue
import sys
import threading
import warnings
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

warnings.filterwarnings("ignore", message="PydanticSerializationUnexpectedValue.*")

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer, Vertical
from textual.events import Key
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Footer, Input, Label, RichLog, Static

from hawi.agent import HawiAgent, HawiScheduler
from hawi.agent.scheduler.executor import SchedulerState
from hawi.events import Event
from hawi.models import model_registry

# ─────────────────────────────────────────────
# 消息类型定义（UI ↔ Scheduler 通讯协议）
# ─────────────────────────────────────────────

QueueKind = Literal["normal", "high_prio", "urgent"]


@dataclass
class CmdEnqueue:
    """UI → Scheduler: 发送消息"""
    content: str
    queue: QueueKind


@dataclass
class CmdClearContext:
    """UI → Scheduler: 清空对话上下文"""


@dataclass
class CmdClearQueue:
    """UI → Scheduler: 清空队列"""
    queue: QueueKind | Literal["all"]


@dataclass
class CmdStop:
    """UI → Scheduler: 停止"""


@dataclass
class UiStatusUpdate:
    """Scheduler → UI: 状态更新"""
    scheduler_state: str
    agent_state: str
    queue_lengths: dict[str, int]


@dataclass
class UiTextDelta:
    """Scheduler → UI: 流式文本增量"""
    delta: str
    run_id: str


@dataclass
class UiRunStart:
    """Scheduler → UI: Agent run 开始"""
    run_id: str
    user_content: str
    queue_kind: QueueKind


@dataclass
class UiRunStop:
    """Scheduler → UI: Agent run 结束"""
    run_id: str
    stop_reason: str
    duration_ms: float


@dataclass
class UiToolCall:
    """Scheduler → UI: 工具调用开始"""
    tool_name: str
    tool_call_id: str
    arguments: dict  # 完整参数
    run_id: str


@dataclass
class UiToolResult:
    """Scheduler → UI: 工具结果"""
    tool_call_id: str
    tool_name: str
    success: bool
    output: str      # 完整输出
    duration_ms: float
    run_id: str


@dataclass
class UiInterrupt:
    """Scheduler → UI: 执行被中断"""
    reason: str


@dataclass
class UiError:
    """Scheduler → UI: 错误"""
    message: str


# ─────────────────────────────────────────────
# Scheduler 线程
# ─────────────────────────────────────────────

class SchedulerThread(threading.Thread):
    """在独立线程中运行 asyncio + HawiScheduler。"""

    def __init__(
        self,
        ui_queue: queue.Queue,
        cmd_queue: queue.Queue,
        factory_name: str,
    ):
        super().__init__(daemon=True, name="SchedulerThread")
        self.ui_queue = ui_queue
        self.cmd_queue = cmd_queue
        self.factory_name = factory_name
        self.loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._scheduler: HawiScheduler | None = None
        # 当前正在处理的消息信息（run_id → queue_kind）
        self._pending_run: dict[str, QueueKind] = {}
        self._current_queue_kind: QueueKind = "normal"

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._main())
        finally:
            self.loop.close()

    def send_ui(self, msg):
        """发送消息到 UI 队列（线程安全）。"""
        self.ui_queue.put(msg)

    async def _main(self):
        # 创建 model 和 scheduler
        model = model_registry.create_model(self.factory_name)

        from hawi_plugins.skills_plugin import SkillsPlugin
        from hawi_plugins.web import WebPlugin

        agent = HawiAgent(
            model=model,
            plugins=[SkillsPlugin(skills_dir=".skills"), WebPlugin()],
            system_prompt="You are a helpful AI assistant.",
            max_iterations=None,
            streaming=True,
        )
        self._scheduler = HawiScheduler(agent)

        # 订阅 agent 事件（agent.event_bus 同时收到 model 事件和 agent 事件）
        agent.event_bus.subscribe(self._on_agent_event)
        self._scheduler.event_bus.subscribe(self._on_scheduler_event)
        # 当前活跃的 run_id（用于关联 model 层流式事件）
        self._active_run_id: str | None = None
        # 工具调用缓存 tool_call_id → {tool_name, arguments, run_id}
        self._active_tool_calls: dict[str, dict] = {}
        # msg_id → queue_kind，在 enqueue 时记录，run_start 时消费
        self._msg_queue_kind: dict[str, QueueKind] = {}
        # 最近一次入队的 queue_kind（fallback）
        self._last_enqueued_kind: QueueKind = "normal"

        self._ready.set()

        # 启动 scheduler loop + 命令处理
        scheduler_task = asyncio.create_task(
            self._scheduler.run_forever(poll_interval=0.1)
        )
        cmd_task = asyncio.create_task(self._process_commands())
        status_task = asyncio.create_task(self._status_loop())

        await asyncio.gather(scheduler_task, cmd_task, status_task,
                             return_exceptions=True)

    async def _status_loop(self):
        """定期推送状态更新到 UI。"""
        while True:
            await asyncio.sleep(0.3)
            if self._scheduler is None:
                continue
            lengths = self._scheduler.get_queue_lengths()
            sched_state = self._scheduler.state.name
            exec_state = self._scheduler._executor.state.name
            self.send_ui(UiStatusUpdate(
                scheduler_state=sched_state,
                agent_state=exec_state,
                queue_lengths=lengths,
            ))

    async def _process_commands(self):
        """从 cmd_queue 读取 UI 命令并执行。"""
        loop = asyncio.get_event_loop()
        while True:
            try:
                cmd = await loop.run_in_executor(
                    None, lambda: self.cmd_queue.get(timeout=0.1)
                )
            except queue.Empty:
                continue

            if isinstance(cmd, CmdStop):
                if self._scheduler:
                    self._scheduler.stop()
                break
            elif isinstance(cmd, CmdEnqueue):
                if self._scheduler:
                    msg_id = self._scheduler.enqueue(
                        cmd.content, cmd.queue,
                        metadata={"queue_kind": cmd.queue},
                    )
                    self._msg_queue_kind[msg_id] = cmd.queue
                    self._last_enqueued_kind = cmd.queue
            elif isinstance(cmd, CmdClearContext):
                if self._scheduler:
                    self._scheduler.agent.context.clear()
            elif isinstance(cmd, CmdClearQueue):
                if self._scheduler:
                    if cmd.queue == "all":
                        self._scheduler.clear_all_queues()
                    else:
                        self._scheduler.clear_queue(cmd.queue)

    def _on_agent_event(self, event: Event):
        """处理 agent 事件，转发到 UI。"""
        etype = event.type

        if etype == "agent.run_start":
            self._active_run_id = event.run_id
            # 用最近入队的 queue_kind 关联这个 run
            self._pending_run[event.run_id] = self._last_enqueued_kind

        elif etype == "agent.message_added":
            if event.role == "user":
                text = ""
                for part in event.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text += part.get("text", "")
                if text:
                    qk = self._pending_run.get(event.run_id, "normal")
                    self.send_ui(UiRunStart(
                        run_id=event.run_id,
                        user_content=text,
                        queue_kind=qk,
                    ))

        elif etype == "model.content_block_delta":
            if event.delta_type == "text" and event.delta and self._active_run_id:
                self.send_ui(UiTextDelta(delta=event.delta, run_id=self._active_run_id))

        elif etype == "agent.run_stop":
            self._pending_run.pop(event.run_id, None)
            if self._active_run_id == event.run_id:
                self._active_run_id = None
            self.send_ui(UiRunStop(
                run_id=event.run_id,
                stop_reason=event.stop_reason,
                duration_ms=event.duration_ms,
            ))

        elif etype == "agent.tool_call":
            run_id = self._active_run_id or ""
            self._active_tool_calls[event.tool_call_id] = {
                "tool_name": event.tool_name,
                "arguments": event.arguments,
                "run_id": run_id,
            }
            self.send_ui(UiToolCall(
                tool_name=event.tool_name,
                tool_call_id=event.tool_call_id,
                arguments=event.arguments,
                run_id=run_id,
            ))

        elif etype == "agent.tool_result":
            call_info = self._active_tool_calls.pop(event.tool_call_id, {})
            tool_name = call_info.get("tool_name", event.tool_call_id)
            run_id = call_info.get("run_id", self._active_run_id or "")
            output = (event.result.output if event.result else None) or event.result_preview
            self.send_ui(UiToolResult(
                tool_call_id=event.tool_call_id,
                tool_name=tool_name,
                success=event.success,
                output=str(output) if output else "",
                duration_ms=event.duration_ms,
                run_id=run_id,
            ))

    def _on_scheduler_event(self, event: Event):
        if event.type == "scheduler.interrupt":
            self.send_ui(UiInterrupt(reason=event.reason))
        elif event.type == "scheduler.dequeue":
            # 记录最近出队的消息类型，用于关联 run_start
            qk = event.queue_type  # "normal" | "high_prio" | "urgent"
            if qk in ("normal", "high_prio", "urgent"):
                self._last_enqueued_kind = qk

    def enqueue_from_ui(self, content: str, queue: QueueKind):
        """线程安全地从 UI 发送入队命令。"""
        self.cmd_queue.put(CmdEnqueue(content=content, queue=queue))

    def wait_ready(self, timeout: float = 10.0) -> bool:
        return self._ready.wait(timeout)


# ─────────────────────────────────────────────
# Textual Widgets
# ─────────────────────────────────────────────

QUEUE_COLORS = {
    "normal": "dodger_blue2",
    "high_prio": "dark_orange",
    "urgent": "red3",
}

QUEUE_LABELS = {
    "normal": "普通",
    "high_prio": "高优",
    "urgent": "紧急",
}


class StatusBar(Static):
    """顶部状态栏：显示 Scheduler/Agent 状态和队列长度。"""

    scheduler_state: reactive[str] = reactive("IDLE")
    agent_state: reactive[str] = reactive("IDLE")
    queue_lengths: reactive[dict] = reactive({"urgent": 0, "high_prio": 0, "normal": 0})

    def render(self) -> str:
        s = self.scheduler_state
        a = self.agent_state

        # 状态颜色
        s_color = "green" if s == "IDLE" else ("yellow" if s == "READY" else ("red" if s == "INTERRUPTING" else "cyan"))
        a_color = "green" if a == "IDLE" else "cyan"

        ql = self.queue_lengths
        u = ql.get("urgent", 0)
        h = ql.get("high_prio", 0)
        n = ql.get("normal", 0)

        u_str = f"[red]紧急:{u}[/red]" if u > 0 else f"紧急:{u}"
        h_str = f"[yellow]高优:{h}[/yellow]" if h > 0 else f"高优:{h}"
        n_str = f"普通:{n}"

        return (
            f" Scheduler:[{s_color}]{s}[/{s_color}]  "
            f"Agent:[{a_color}]{a}[/{a_color}]  │  "
            f"队列 {u_str}  {h_str}  {n_str}"
        )

    def update_status(self, msg: UiStatusUpdate):
        self.scheduler_state = msg.scheduler_state
        self.agent_state = msg.agent_state
        self.queue_lengths = msg.queue_lengths


class MessageBubble(Static):
    """单条消息气泡。"""

    def __init__(self, role: str, content: str = "", queue_kind: QueueKind = "normal",
                 is_tool: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.role = role
        self._content = content
        self.queue_kind = queue_kind
        self.is_tool = is_tool
        self._ts = datetime.now().strftime("%H:%M:%S")
        self._log: RichLog | None = None  # agent 消息专用

    def compose(self) -> ComposeResult:
        color = QUEUE_COLORS.get(self.queue_kind, "cyan")
        label = QUEUE_LABELS.get(self.queue_kind, "普通")

        if self.role == "user":
            yield Label(
                f"[bold {color}]▶ 你 [{label}][/bold {color}]  [{color}]{self._ts}[/{color}]",
                classes="bubble-header"
            )
            yield Label(self._content, classes="bubble-user-content")
        elif self.role == "assistant":
            yield Label(
                f"[bold green]◀ Agent[/bold green]  [green]{self._ts}[/green]",
                classes="bubble-header"
            )
            log = RichLog(
                highlight=False,
                markup=False,
                wrap=True,
                id=f"log-{self.id}",
                classes="bubble-agent-log",
            )
            self._log = log
            yield log
        elif self.role == "tool":
            icon = "✓" if not self.is_tool else "⚙"
            yield Label(
                f"[dim]{icon} {self._content}[/dim]",
                classes="bubble-tool"
            )
        elif self.role == "interrupt":
            yield Label(
                f"[bold red]⚡ 中断: {self._content}[/bold red]",
                classes="bubble-interrupt"
            )
        elif self.role == "system":
            yield Label(
                f"[dim italic]{self._content}[/dim italic]",
                classes="bubble-system"
            )

    def on_mount(self):
        # RichLog 挂载后写入初始内容（如果有）
        if self.role == "assistant" and self._log and self._content:
            self._log.write(self._content)
        self._line_buf = ""  # 未完成行的缓冲

    def append_text(self, delta: str):
        """流式追加文本到 agent 气泡的 RichLog。"""
        self._content += delta
        if not self._log:
            return

        # 把 delta 追加到行缓冲，按换行符分割写入
        self._line_buf += delta
        lines = self._line_buf.split("\n")
        # 最后一个元素是未完成的行，继续缓冲
        for line in lines[:-1]:
            self._log.write(line)
        self._line_buf = lines[-1]

    def flush_line_buf(self):
        """把剩余缓冲内容写入（run 结束时调用）。"""
        if self._log and self._line_buf:
            self._log.write(self._line_buf)
            self._line_buf = ""


class ChatView(ScrollableContainer):
    """中间消息区域。"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._run_bubbles: dict[str, MessageBubble] = {}  # run_id → agent bubble
        self._bubble_counter = 0

    def _next_id(self) -> str:
        self._bubble_counter += 1
        return f"bubble-{self._bubble_counter}"

    def add_user_message(self, content: str, queue_kind: QueueKind):
        bubble = MessageBubble(
            role="user", content=content, queue_kind=queue_kind,
            id=self._next_id()
        )
        self.mount(bubble)
        self.scroll_end(animate=False)

    def start_agent_message(self, run_id: str) -> MessageBubble:
        bubble = MessageBubble(
            role="assistant", content="", queue_kind="normal",
            id=self._next_id()
        )
        self._run_bubbles[run_id] = bubble
        self.mount(bubble)
        self.scroll_end(animate=False)
        return bubble

    def append_agent_delta(self, run_id: str, delta: str):
        bubble = self._run_bubbles.get(run_id)
        if bubble:
            bubble.append_text(delta)
            self.scroll_end(animate=False)

    def finish_agent_message(self, run_id: str, stop_reason: str, duration_ms: float):
        bubble = self._run_bubbles.pop(run_id, None)
        if bubble:
            bubble.flush_line_buf()  # 写入最后一行未完成的缓冲
            secs = duration_ms / 1000
            self.mount(MessageBubble(
                role="system",
                content=f"完成 · {stop_reason} · {secs:.1f}s",
                id=self._next_id()
            ))
            self.scroll_end(animate=False)

    def add_tool_event(self, content: str, is_tool: bool = True):
        bubble = MessageBubble(
            role="tool", content=content, is_tool=is_tool,
            id=self._next_id()
        )
        self.mount(bubble)
        self.scroll_end(animate=False)

    def add_interrupt(self, reason: str):
        bubble = MessageBubble(role="interrupt", content=reason, id=self._next_id())
        self.mount(bubble)
        self.scroll_end(animate=False)

    def add_system(self, content: str):
        bubble = MessageBubble(role="system", content=content, id=self._next_id())
        self.mount(bubble)
        self.scroll_end(animate=False)


class QueueTypeIndicator(Static):
    """显示当前选中的消息类型。"""

    queue_kind: reactive[QueueKind] = reactive("normal")

    def render(self) -> str:
        kinds: list[QueueKind] = ["normal", "high_prio", "urgent"]
        parts = []
        for k in kinds:
            color = QUEUE_COLORS[k]
            label = QUEUE_LABELS[k]
            if k == self.queue_kind:
                parts.append(f"[bold {color}][{label}][/bold {color}]")
            else:
                parts.append(f"[dim]{label}[/dim]")
        return "  Tab切换: " + "  ".join(parts) + "  "


# ─────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────

TUI_CSS = """
Screen {
    layout: vertical;
    background: #f5f5f5;
}

StatusBar {
    height: 1;
    background: #e0e0e0;
    color: #333333;
    padding: 0 1;
    dock: top;
}

ChatView {
    height: 1fr;
    padding: 0 1;
    background: #f5f5f5;
}

.bubble-header {
    margin-top: 1;
    color: #666666;
}

.bubble-user-content {
    padding: 0 2;
    color: #1a1a1a;
}

.bubble-agent-log {
    padding: 0 2;
    height: auto;
    max-height: 30;
    background: #f5f5f5;
    border: none;
    scrollbar-size: 0 0;
    color: #1a1a1a;
}

.bubble-agent-content {
    padding: 0 2;
    color: #1a1a1a;
}

.bubble-tool {
    padding: 0 2;
    color: #888888;
}

.bubble-interrupt {
    padding: 0 2;
}

.bubble-system {
    padding: 0 2;
    color: #888888;
}

#input-area {
    height: auto;
    dock: bottom;
    background: #e8e8e8;
    padding: 0 1;
}

QueueTypeIndicator {
    height: 1;
    color: #666666;
}

#user-input {
    height: 3;
    background: #ffffff;
    color: #1a1a1a;
    border: tall #4a9eff;
}

#user-input:focus {
    border: tall #1a7f37;
}
"""


class HawiTUI(App):
    """Hawi Scheduler 交互式终端 App。"""

    CSS = TUI_CSS
    THEME = "textual-light"

    BINDINGS = [
        Binding("shift+tab", "cycle_queue", "切换消息类型", show=True),
        Binding("ctrl+c", "quit", "退出", show=True),
        Binding("ctrl+l", "clear_context", "清空上下文", show=True),
        Binding("escape", "clear_input", "清空输入", show=False),
    ]

    current_queue: reactive[QueueKind] = reactive("normal")

    def __init__(self, factory_name: str, **kwargs):
        super().__init__(**kwargs)
        self.factory_name = factory_name
        self.ui_queue: queue.Queue = queue.Queue()
        self.cmd_queue: queue.Queue = queue.Queue()
        self._sched_thread: SchedulerThread | None = None
        self._active_runs: set[str] = set()

    def compose(self) -> ComposeResult:
        yield StatusBar(id="status-bar")
        yield ChatView(id="chat-view")
        with Vertical(id="input-area"):
            yield QueueTypeIndicator(id="queue-indicator")
            yield Input(
                placeholder="输入消息，Enter 发送，Shift+Tab 切换优先级，/help 查看命令...",
                id="user-input",
            )

    def on_mount(self):
        # 启动 scheduler 线程
        self._sched_thread = SchedulerThread(
            ui_queue=self.ui_queue,
            cmd_queue=self.cmd_queue,
            factory_name=self.factory_name,
        )
        self._sched_thread.start()

        chat = self.query_one("#chat-view", ChatView)
        chat.add_system("正在启动 Scheduler...")

        # 等待 scheduler 就绪（非阻塞，用 set_interval 轮询）
        self._ready_timer = self.set_interval(0.1, self._poll_ready)
        self.set_interval(0.05, self._poll_ui_queue)

        # 聚焦输入框
        self.query_one("#user-input", Input).focus()

    def _poll_ready(self):
        if self._sched_thread and self._sched_thread._ready.is_set():
            chat = self.query_one("#chat-view", ChatView)
            chat.add_system("✓ Scheduler 已就绪，开始对话吧")
            self._ready_timer.stop()

    def _poll_ui_queue(self):
        """从 ui_queue 读取消息并更新 UI（在 UI 线程中调用）。"""
        try:
            while True:
                msg = self.ui_queue.get_nowait()
                self._handle_ui_message(msg)
        except queue.Empty:
            pass

    def _handle_ui_message(self, msg):
        chat = self.query_one("#chat-view", ChatView)
        status = self.query_one("#status-bar", StatusBar)

        if isinstance(msg, UiStatusUpdate):
            status.update_status(msg)

        elif isinstance(msg, UiRunStart):
            if msg.user_content:
                chat.add_user_message(msg.user_content, msg.queue_kind)
            if msg.run_id not in self._active_runs:
                self._active_runs.add(msg.run_id)
                chat.start_agent_message(msg.run_id)

        elif isinstance(msg, UiTextDelta):
            if msg.run_id not in self._active_runs:
                self._active_runs.add(msg.run_id)
                chat.start_agent_message(msg.run_id)
            chat.append_agent_delta(msg.run_id, msg.delta)
        elif isinstance(msg, UiRunStop):
            self._active_runs.discard(msg.run_id)
            chat.finish_agent_message(msg.run_id, msg.stop_reason, msg.duration_ms)

        elif isinstance(msg, UiToolCall):
            # 工具调用开始：在对应的 agent bubble 里追加一行
            args_preview = ", ".join(
                f"{k}={str(v)[:40]}" for k, v in msg.arguments.items()
            ) if msg.arguments else ""
            line = f"\n⚙ {msg.tool_name}({args_preview})"
            chat.append_agent_delta(msg.run_id, line)

        elif isinstance(msg, UiToolResult):
            # 工具结果：追加到同一个 agent bubble
            icon = "✓" if msg.success else "✗"
            # 截断长输出，保留前 200 字符
            output = msg.output
            if len(output) > 200:
                output = output[:197] + "..."
            # 去掉多余换行，压成单行摘要
            output_line = output.replace("\n", " ").strip()
            line = f"\n  {icon} {output_line}  [{msg.duration_ms:.0f}ms]\n"
            chat.append_agent_delta(msg.run_id, line)

        elif isinstance(msg, UiInterrupt):
            chat.add_interrupt(msg.reason)

        elif isinstance(msg, UiError):
            chat.add_system(f"[red]错误: {msg.message}[/red]")

    def on_input_submitted(self, event: Input.Submitted):
        text = event.value.strip()
        if not text:
            return

        event.input.clear()

        # 处理 / 命令
        if text.startswith("/"):
            self._handle_command(text)
            return

        # 发送消息到 scheduler
        self.cmd_queue.put(CmdEnqueue(content=text, queue=self.current_queue))

    def on_key(self, event: Key):
        """拦截 shift+tab，防止被焦点系统消费。"""
        if event.key == "shift+tab":
            event.prevent_default()
            event.stop()
            self.action_cycle_queue()

    def _handle_command(self, text: str):
        chat = self.query_one("#chat-view", ChatView)
        cmd = text.lower().strip()

        if cmd in ("/help", "/h"):
            chat.add_system(
                "命令: /clear(清空上下文) /cq(清空普通队列) /chq(清空高优队列) "
                "/cuq(清空紧急队列) /ca(清空所有队列) /status(队列状态) /quit(退出)"
            )
        elif cmd == "/clear":
            self.cmd_queue.put(CmdClearContext())
            chat.add_system("✓ 对话上下文已清空")
        elif cmd == "/cq":
            self.cmd_queue.put(CmdClearQueue(queue="normal"))
            chat.add_system("✓ 普通队列已清空")
        elif cmd == "/chq":
            self.cmd_queue.put(CmdClearQueue(queue="high_prio"))
            chat.add_system("✓ 高优队列已清空")
        elif cmd == "/cuq":
            self.cmd_queue.put(CmdClearQueue(queue="urgent"))
            chat.add_system("✓ 紧急队列已清空")
        elif cmd == "/ca":
            self.cmd_queue.put(CmdClearQueue(queue="all"))
            chat.add_system("✓ 所有队列已清空")
        elif cmd in ("/quit", "/q", "/exit"):
            self.action_quit()
        else:
            chat.add_system(f"未知命令: {text}，输入 /help 查看帮助")

    def action_cycle_queue(self):
        """Shift+Tab 循环切换消息类型。"""
        order: list[QueueKind] = ["normal", "high_prio", "urgent"]
        idx = order.index(self.current_queue)
        self.current_queue = order[(idx + 1) % len(order)]
        indicator = self.query_one("#queue-indicator", QueueTypeIndicator)
        indicator.queue_kind = self.current_queue

    def action_clear_context(self):
        self.cmd_queue.put(CmdClearContext())
        chat = self.query_one("#chat-view", ChatView)
        chat.add_system("✓ 对话上下文已清空")

    def action_clear_input(self):
        self.query_one("#user-input", Input).clear()

    def action_quit(self):
        if self._sched_thread:
            self.cmd_queue.put(CmdStop())
        self.exit()


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

def load_models():
    from pathlib import Path
    user_config = Path.home() / ".hawi" / "models.yaml"
    project_config = Path.cwd() / "models.yaml"
    if user_config.exists():
        model_registry.load_config(user_config, quiet=True)
    if project_config.exists():
        model_registry.load_config(project_config, quiet=True)


def main():
    import sys
    from utils.terminal import user_select

    argv = sys.argv[1:]
    load_models()

    available = model_registry.list_factories()
    if not available:
        print("No model factories found. Please configure ~/.hawi/models.yaml")
        sys.exit(1)

    factory_name = None
    for arg in argv[:]:
        if arg in available:
            argv.remove(arg)
            factory_name = arg
            break

    if factory_name is None:
        factory_name = user_select(available, "Select model factory:")
        if factory_name is None:
            sys.exit(0)

    app = HawiTUI(factory_name=factory_name)
    app.run()


if __name__ == "__main__":
    main()
