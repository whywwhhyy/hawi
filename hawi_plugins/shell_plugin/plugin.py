from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import IO, Literal

from hawi.plugin import HawiPlugin, tool
from hawi.tool import ToolResult


DEFAULT_NOTIFY_TIMEOUT_SECONDS = 300.0
CONTROL_WAIT_SECONDS = 2.0


@dataclass
class ShellCommand:
    id: str
    command: str
    process: subprocess.Popen[str]
    started_at: float
    stdout_chunks: list[str] = field(default_factory=list)
    stderr_chunks: list[str] = field(default_factory=list)
    stdout_thread: threading.Thread | None = None
    stderr_thread: threading.Thread | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def stdout(self) -> str:
        with self.lock:
            return "".join(self.stdout_chunks)

    def stderr(self) -> str:
        with self.lock:
            return "".join(self.stderr_chunks)

    def is_running(self) -> bool:
        return self.process.poll() is None

    def join_readers(self, timeout: float = 0.2) -> None:
        for thread in (self.stdout_thread, self.stderr_thread):
            if thread and thread.is_alive():
                thread.join(timeout=timeout)


class ShellPlugin(HawiPlugin):
    """
    Shell 操作插件，提供运行 shell 命令的能力。
    """

    name = "hawi/shell"
    display_name = "Shell"
    description = "运行和管理后台 shell 命令，支持查看输出、状态和中断。"
    dependencies = ()

    def __init__(self) -> None:
        self._commands: dict[str, ShellCommand] = {}
        self._commands_lock = threading.Lock()

    @classmethod
    def gui_config_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    @classmethod
    def gui_default_config(cls) -> dict:
        return {}

    @tool(name="run_shell")
    def run_shell(self, command: str, notify_timeout: float = DEFAULT_NOTIFY_TIMEOUT_SECONDS) -> ToolResult:
        """
        运行 shell 命令。

        这是一个可控的长命令启动工具。

        notify_timeout 表示本工具调用愿意在前台等待命令完成并通知 agent 的秒数，
        默认 300 秒。它不是 shell 的 timeout 命令，也不是命令的最大运行时长。
        到达 notify_timeout 后：

        - 本次 run_shell 工具调用会立即返回给 agent；
        - shell 进程不会被杀掉，会继续在后台运行；
        - 返回值会包含 Shell command id、Status: running、已产生的 stdout/stderr；
        - 返回值不会包含真实 Exit code，因为进程尚未结束；
        - agent 应使用 shell_control 继续查看、打断或取消该后台命令。

        如果需要查看后台命令最新输出，调用：
            shell_control(command_id="<Shell command id>", action="status", notify_timeout=10)

        如果需要请求命令自己中断，调用：
            shell_control(command_id="<Shell command id>", action="interrupt")

        如果需要终止命令，调用：
            shell_control(command_id="<Shell command id>", action="cancel")

        Args:
            command: 要执行的 shell 命令
            notify_timeout: 前台等待命令完成并通知 agent 的秒数。默认 300 秒。
                到时只返回当前状态并保留后台进程，不会杀掉命令。
        """
        try:
            notify_timeout = _normalize_notify_timeout(notify_timeout)
            shell_command = self._start_command(command)
            try:
                returncode = shell_command.process.wait(timeout=notify_timeout)
            except subprocess.TimeoutExpired:
                return ToolResult(
                    success=True,
                    output=_format_running_command(
                        shell_command,
                        notify_timeout=notify_timeout,
                    ),
                )

            shell_command.join_readers()
            output = _format_process_result(
                command=command,
                returncode=returncode,
                stdout=shell_command.stdout(),
                stderr=shell_command.stderr(),
            )
            self._forget_command(shell_command.id)
            if returncode != 0:
                return ToolResult(
                    success=False,
                    output=output,
                    error=f"Command exited with status {returncode}",
                )
            return ToolResult(success=True, output=output)
        except Exception as e:
            return ToolResult(success=False, error=f"Error running command: {type(e).__name__}: {e}")

    @tool(name="shell_control")
    def shell_control(
        self,
        command_id: str,
        action: Literal["status", "interrupt", "cancel"] = "status",
        notify_timeout: float = 0.0,
    ) -> ToolResult:
        """
        控制或查询到达 notify_timeout 后仍在后台运行的 shell 命令。

        只有 run_shell 因前台等待时间到达而返回 Status: running 时，
        才需要使用本工具。command_id 来自 run_shell 返回值中的
        Shell command id。

        action 语义：

        - status: 查看后台命令当前状态和最新 stdout/stderr。
            如果命令已经结束，会返回最终 Exit code 并移除该 command id。
        - interrupt: 向后台命令发送 SIGINT，类似终端中的 Ctrl-C。
            适合让命令自行清理后退出。
        - cancel: 终止后台命令。会先发送 SIGTERM，若仍未退出再强制杀掉。

        注意：status 不会中断命令，只是查看；interrupt/cancel 才会改变命令状态。

        notify_timeout 表示返回本次 shell_control 结果前最多等待多少秒。
        默认 0 秒，即立即返回当前状态。常见用法：

            shell_control(command_id="<Shell command id>", action="status", notify_timeout=10)

        这表示等待最多 10 秒后再次检查状态。如果这 10 秒内命令结束，
        本工具会返回最终 Exit code 和完整 stdout/stderr；如果仍未结束，
        本工具会返回 Status: running 和当前已产生的 stdout/stderr。

        Args:
            command_id: run_shell 到达 notify_timeout 后返回的 shell command id
            action: status 查看状态；interrupt 发送 SIGINT；cancel 终止命令
            notify_timeout: 返回前最多等待多少秒后再次检查状态。默认 0 秒立即返回。
        """
        try:
            notify_timeout = _normalize_control_notify_timeout(notify_timeout)
        except Exception as e:
            return ToolResult(success=False, error=f"Error controlling command: {type(e).__name__}: {e}")

        shell_command = self._get_command(command_id)
        if shell_command is None:
            return ToolResult(
                success=False,
                error=f"Unknown shell command id: {command_id}",
            )

        if action == "status":
            return self._status_result(shell_command, notify_timeout=notify_timeout)
        if action == "interrupt":
            return self._interrupt_result(shell_command, notify_timeout=notify_timeout)
        if action == "cancel":
            return self._cancel_result(shell_command, notify_timeout=notify_timeout)
        return ToolResult(
            success=False,
            error="action must be one of: status, interrupt, cancel",
        )

    def _start_command(self, command: str) -> ShellCommand:
        command_id = f"shell-{uuid.uuid4().hex[:8]}"
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=os.name != "nt",
        )
        shell_command = ShellCommand(
            id=command_id,
            command=command,
            process=process,
            started_at=time.time(),
        )
        shell_command.stdout_thread = _start_reader(
            process.stdout,
            shell_command.stdout_chunks,
            shell_command.lock,
            f"{command_id}-stdout",
        )
        shell_command.stderr_thread = _start_reader(
            process.stderr,
            shell_command.stderr_chunks,
            shell_command.lock,
            f"{command_id}-stderr",
        )
        with self._commands_lock:
            self._commands[command_id] = shell_command
        return shell_command

    def _get_command(self, command_id: str) -> ShellCommand | None:
        with self._commands_lock:
            return self._commands.get(command_id)

    def _forget_command(self, command_id: str) -> None:
        with self._commands_lock:
            self._commands.pop(command_id, None)

    def _status_result(self, shell_command: ShellCommand, notify_timeout: float = 0.0) -> ToolResult:
        returncode = shell_command.process.poll()
        if returncode is None and notify_timeout > 0:
            try:
                returncode = shell_command.process.wait(timeout=notify_timeout)
            except subprocess.TimeoutExpired:
                return ToolResult(
                    success=True,
                    output=_format_running_command(shell_command, notify_timeout=notify_timeout),
                )

        if returncode is None:
            return ToolResult(
                success=True,
                output=_format_running_command(shell_command),
            )

        shell_command.join_readers()
        output = _format_process_result(
            command=shell_command.command,
            returncode=returncode,
            stdout=shell_command.stdout(),
            stderr=shell_command.stderr(),
            command_id=shell_command.id,
            status="completed",
        )
        self._forget_command(shell_command.id)
        if returncode != 0:
            return ToolResult(
                success=False,
                output=output,
                error=f"Command exited with status {returncode}",
            )
        return ToolResult(success=True, output=output)

    def _interrupt_result(self, shell_command: ShellCommand, notify_timeout: float = 0.0) -> ToolResult:
        if not shell_command.is_running():
            return self._status_result(shell_command)

        _send_process_signal(shell_command.process, signal.SIGINT)
        return self._wait_after_control(
            shell_command,
            action="interrupt",
            wait_timeout=_control_wait_timeout(notify_timeout),
        )

    def _cancel_result(self, shell_command: ShellCommand, notify_timeout: float = 0.0) -> ToolResult:
        if not shell_command.is_running():
            return self._status_result(shell_command)

        _send_process_signal(shell_command.process, signal.SIGTERM)
        try:
            shell_command.process.wait(timeout=_control_wait_timeout(notify_timeout))
        except subprocess.TimeoutExpired:
            _kill_process(shell_command.process)
        return self._wait_after_control(
            shell_command,
            action="cancel",
            wait_timeout=CONTROL_WAIT_SECONDS,
        )

    def _wait_after_control(
        self,
        shell_command: ShellCommand,
        action: str,
        wait_timeout: float,
    ) -> ToolResult:
        try:
            returncode = shell_command.process.wait(timeout=wait_timeout)
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=True,
                output=_format_running_command(shell_command, control_action=action),
            )

        shell_command.join_readers()
        output = _format_process_result(
            command=shell_command.command,
            returncode=returncode,
            stdout=shell_command.stdout(),
            stderr=shell_command.stderr(),
            command_id=shell_command.id,
            status=f"{action}ed",
        )
        self._forget_command(shell_command.id)
        return ToolResult(
            success=False,
            output=output,
            error=f"Command {action}ed",
        )


def _start_reader(
    pipe: IO[str] | None,
    chunks: list[str],
    lock: threading.Lock,
    name: str,
) -> threading.Thread | None:
    if pipe is None:
        return None

    def read_pipe() -> None:
        try:
            while True:
                char = pipe.read(1)
                if char == "":
                    break
                with lock:
                    chunks.append(char)
        finally:
            pipe.close()

    thread = threading.Thread(target=read_pipe, name=name, daemon=True)
    thread.start()
    return thread


def _send_process_signal(process: subprocess.Popen[str], sig: signal.Signals) -> None:
    try:
        if os.name == "nt":
            if sig == signal.SIGTERM:
                process.terminate()
            else:
                process.send_signal(sig)
        else:
            os.killpg(process.pid, sig)
    except ProcessLookupError:
        return


def _kill_process(process: subprocess.Popen[str]) -> None:
    try:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _normalize_notify_timeout(notify_timeout: float) -> float:
    notify_timeout = float(notify_timeout)
    if notify_timeout <= 0:
        raise ValueError("notify_timeout must be greater than 0")
    return notify_timeout


def _normalize_control_notify_timeout(notify_timeout: float) -> float:
    notify_timeout = float(notify_timeout)
    if notify_timeout < 0:
        raise ValueError("notify_timeout must be greater than or equal to 0")
    return notify_timeout


def _control_wait_timeout(notify_timeout: float) -> float:
    return notify_timeout if notify_timeout > 0 else CONTROL_WAIT_SECONDS


def _format_process_result(
    command: str,
    returncode: int,
    stdout: str,
    stderr: str,
    *,
    command_id: str | None = None,
    status: str | None = None,
) -> str:
    parts = []
    if command_id:
        parts.append(f"Shell command id: {command_id}")
    if status:
        parts.append(f"Status: {status}")
    parts.extend([
        f"Command: {command}",
        f"Exit code: {returncode}",
    ])
    if stdout:
        parts.append(f"Stdout:\n{stdout}")
    else:
        parts.append("Stdout: <empty>")
    if stderr:
        parts.append(f"Stderr:\n{stderr}")
    else:
        parts.append("Stderr: <empty>")
    return "\n\n".join(parts)


def _format_running_command(
    shell_command: ShellCommand,
    *,
    notify_timeout: float | None = None,
    control_action: str | None = None,
) -> str:
    elapsed = time.time() - shell_command.started_at
    parts = [
        f"Shell command id: {shell_command.id}",
        "Status: running",
        f"Command: {shell_command.command}",
        f"Elapsed: {elapsed:.1f}s",
    ]
    if notify_timeout is not None:
        parts.append(f"Notify wait elapsed: {notify_timeout:.1f}s")
        parts.append("No exit code is available yet; the process is still running in the background and was not killed.")
    if control_action:
        parts.append(f"Control action sent: {control_action}")
    parts.append(
        'Use shell_control(command_id="{0}", action="status", notify_timeout=10) to inspect latest output after waiting, '
        'action="interrupt" to send SIGINT, or action="cancel" to terminate it.'.format(
            shell_command.id
        )
    )
    stdout = shell_command.stdout()
    stderr = shell_command.stderr()
    if stdout:
        parts.append(f"Stdout so far:\n{stdout}")
    else:
        parts.append("Stdout so far: <empty>")
    if stderr:
        parts.append(f"Stderr so far:\n{stderr}")
    else:
        parts.append("Stderr so far: <empty>")
    return "\n\n".join(parts)
