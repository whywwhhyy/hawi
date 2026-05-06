from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal, TextIO

from hawi.plugin import HawiPlugin, tool
from hawi.tool import ToolResult


DEFAULT_TIMEOUT_SECONDS = 300.0
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

    @tool
    def run_shell(self, command: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> ToolResult:
        """
        运行 shell 命令。

        如果命令在 timeout 秒内未结束，会保留后台进程并返回 shell command id。
        后续可用 shell_control(command_id, action="status") 查看最新输出，
        用 action="interrupt" 发送中断信号，或 action="cancel" 终止命令。

        Args:
            command: 要执行的 shell 命令
            timeout: 等待命令完成的秒数，默认 300 秒
        """
        try:
            timeout = _normalize_timeout(timeout)
            shell_command = self._start_command(command)
            try:
                returncode = shell_command.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                return ToolResult(
                    success=True,
                    output=_format_running_command(
                        shell_command,
                        timeout=timeout,
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

    @tool
    def shell_control(
        self,
        command_id: str,
        action: Literal["status", "interrupt", "cancel"] = "status",
    ) -> ToolResult:
        """
        控制或查询超时后仍在后台运行的 shell 命令。

        Args:
            command_id: run_shell 超时返回的 shell command id
            action: status 查看状态；interrupt 发送 SIGINT；cancel 终止命令
        """
        shell_command = self._get_command(command_id)
        if shell_command is None:
            return ToolResult(
                success=False,
                error=f"Unknown shell command id: {command_id}",
            )

        if action == "status":
            return self._status_result(shell_command)
        if action == "interrupt":
            return self._interrupt_result(shell_command)
        if action == "cancel":
            return self._cancel_result(shell_command)
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

    def _status_result(self, shell_command: ShellCommand) -> ToolResult:
        returncode = shell_command.process.poll()
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

    def _interrupt_result(self, shell_command: ShellCommand) -> ToolResult:
        if not shell_command.is_running():
            return self._status_result(shell_command)

        _send_process_signal(shell_command.process, signal.SIGINT)
        return self._wait_after_control(shell_command, action="interrupt")

    def _cancel_result(self, shell_command: ShellCommand) -> ToolResult:
        if not shell_command.is_running():
            return self._status_result(shell_command)

        _send_process_signal(shell_command.process, signal.SIGTERM)
        try:
            shell_command.process.wait(timeout=CONTROL_WAIT_SECONDS)
        except subprocess.TimeoutExpired:
            _kill_process(shell_command.process)
        return self._wait_after_control(shell_command, action="cancel")

    def _wait_after_control(self, shell_command: ShellCommand, action: str) -> ToolResult:
        try:
            returncode = shell_command.process.wait(timeout=CONTROL_WAIT_SECONDS)
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
    pipe: TextIO | None,
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


def _normalize_timeout(timeout: float) -> float:
    timeout = float(timeout)
    if timeout <= 0:
        raise ValueError("timeout must be greater than 0")
    return timeout


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
    timeout: float | None = None,
    control_action: str | None = None,
) -> str:
    elapsed = time.time() - shell_command.started_at
    parts = [
        f"Shell command id: {shell_command.id}",
        "Status: running",
        f"Command: {shell_command.command}",
        f"Elapsed: {elapsed:.1f}s",
    ]
    if timeout is not None:
        parts.append(f"Timeout reached: {timeout:.1f}s")
    if control_action:
        parts.append(f"Control action sent: {control_action}")
    parts.append(
        'Use shell_control(command_id="{0}", action="status") to inspect latest output, '
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
