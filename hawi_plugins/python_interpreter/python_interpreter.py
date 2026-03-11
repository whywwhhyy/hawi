"""
持久化子进程Python解释器
使用长度前缀协议进行进程间通信，支持多次调用间保持状态
"""

import asyncio
import subprocess
import json
import struct
import select
import os
import tempfile
import shutil
from typing import Optional, AsyncGenerator
from threading import Lock

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from hawi.tool import ToolResult
from hawi.utils.lifecycle import ExitHandler


# 创建 rich console 实例用于美化输出
_console = Console()


class PythonInterpreter:
    """
    持久化子进程Python解释器

    启动一个长期运行的Python子进程，通过stdin/stdout进行通信。
    使用4字节长度前缀协议（大端序）来确保完整读取数据。
    支持在多次execute调用之间保持变量状态。
    """

    # 子进程中运行的服务器代码
    _SERVER_CODE = '''
import sys
import json
import io
import contextlib
import traceback
import struct
import threading
import queue
import time

# 初始化命名空间，保持状态
namespace = {
    "__name__": "__main__",
    "__builtins__": __builtins__,
}

# 预导入常用模块
import math, os, sys as sys_mod, json as json_mod, datetime, time, random, re
import collections, itertools, functools, pathlib, typing
namespace.update({
    'math': math, 'os': os, 'sys': sys_mod, 'json': json_mod,
    'datetime': datetime, 'time': time, 'random': random, 're': re,
    'collections': collections, 'itertools': itertools,
    'functools': functools, 'pathlib': pathlib, 'typing': typing,
})

def read_exact(n):
    """从stdin精确读取n个字节"""
    data = b''
    while len(data) < n:
        chunk = sys.stdin.buffer.read(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data

def send_response(data):
    """发送响应（带长度前缀）"""
    encoded = json.dumps(data).encode('utf-8')
    sys.stdout.buffer.write(struct.pack('>I', len(encoded)))
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()

class StreamingBuffer:
    """流式输出缓冲区，将输出实时发送到队列"""
    def __init__(self, output_queue):
        self.queue = output_queue
        self.buffer = []
        self._closed = False

    def write(self, data):
        if self._closed:
            return
        self.buffer.append(data)
        # 立即发送
        self.flush()

    def flush(self):
        if self.buffer and not self._closed:
            text = ''.join(self.buffer)
            self.buffer = []
            if text:
                self.queue.put({"chunk": text, "stream": True})

    def close(self):
        self._closed = True
        self.flush()

def execute_code_streaming(code, output_queue):
    """在线程中执行代码，通过队列流式输出"""
    stdout_buffer = StreamingBuffer(output_queue)
    stderr_buffer = StreamingBuffer(output_queue)

    result = {"output": "", "error": "", "success": False}
    full_output = []
    full_error = []

    try:
        with contextlib.redirect_stdout(stdout_buffer), \
             contextlib.redirect_stderr(stderr_buffer):

            # 先尝试作为表达式求值
            try:
                compiled = compile(code, '<string>', 'eval')
                eval_result = eval(compiled, namespace)
                if eval_result is not None:
                    print(repr(eval_result))
            except SyntaxError:
                # 不是表达式，用exec执行
                exec(code, namespace)

        # 等待缓冲区 flush
        time.sleep(0.01)

        result["success"] = True

    except Exception as e:
        result["error"] = traceback.format_exc()
        result["success"] = False

    # 关闭缓冲区
    stdout_buffer.close()
    stderr_buffer.close()

    # 发送最终结果
    result["stream"] = False
    output_queue.put(result)

while True:
    # 读取4字节长度头
    length_bytes = read_exact(4)
    if length_bytes is None:
        break

    length = struct.unpack('>I', length_bytes)[0]

    # 读取代码
    code_bytes = read_exact(length)
    if code_bytes is None:
        break

    code = code_bytes.decode('utf-8')

    # 检查是否是流式模式（代码以 # __STREAMING__\\n 开头）
    streaming_mode = code.startswith("# __STREAMING__\\n")
    if streaming_mode:
        code = code.replace("# __STREAMING__\\n", "", 1)

    if streaming_mode:
        # 流式执行模式
        output_queue = queue.Queue()

        # 在后台线程执行代码
        exec_thread = threading.Thread(
            target=execute_code_streaming,
            args=(code, output_queue)
        )
        exec_thread.start()

        # 从队列读取输出并发送
        while True:
            try:
                chunk = output_queue.get(timeout=0.001)
                send_response(chunk)
                if not chunk.get("stream", False):
                    # 最终响应
                    break
            except queue.Empty:
                # 检查线程是否结束
                if not exec_thread.is_alive():
                    # 线程结束但没有数据，发送空结果
                    send_response({"output": "", "error": "", "success": True, "stream": False})
                    break
                continue

        exec_thread.join(timeout=1.0)
    else:
        # 传统同步模式
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        result = {
            "output": "",
            "error": "",
            "success": False
        }

        try:
            with contextlib.redirect_stdout(stdout_capture), \
                 contextlib.redirect_stderr(stderr_capture):

                # 先尝试作为表达式求值
                try:
                    compiled = compile(code, '<string>', 'eval')
                    eval_result = eval(compiled, namespace)
                    if eval_result is not None:
                        print(repr(eval_result))
                except SyntaxError:
                    # 不是表达式，用exec执行
                    exec(code, namespace)

            result["output"] = stdout_capture.getvalue()
            result["success"] = True

        except Exception as e:
            result["error"] = traceback.format_exc()
            result["success"] = False

        # 添加stderr内容
        stderr_content = stderr_capture.getvalue()
        if stderr_content:
            result["error"] = stderr_content + "\\n" + result["error"]

        send_response(result)
'''

    def __init__(self, work_dir: Optional[str] = None, print_execution=False):
        """
        初始化持久化子进程解释器（使用 uv 管理环境）

        Args:
            work_dir: 工作目录，若为 None 则使用临时目录
        """
        self._proc: Optional[subprocess.Popen] = None
        self._lock = Lock()
        self._async_lock: Optional[asyncio.Lock] = None
        self._closed = False
        self._owns_temp_dir = work_dir is None
        self.print_execution = print_execution

        if work_dir is None:
            # 创建临时工作目录，用于隔离解释器环境
            self._work_dir = tempfile.mkdtemp(prefix="python_executor_")
        else:
            # 如果指定的目录不存在，创建它
            if not os.path.exists(work_dir):
                os.makedirs(work_dir, exist_ok=True)
            self._work_dir = work_dir

        if not os.path.isfile(os.path.join(self._work_dir, "pyproject.toml")):
            # 使用 uv init 初始化临时项目
            self._init_temp_project()

        self._start_server()

        # 注册退出处理函数，确保在程序退出时正确清理
        self._exit_handler = ExitHandler.get_instance()
        # 创建一个包装函数，检查是否已经关闭
        def cleanup_wrapper():
            if not self._closed:
                self.close()
        self._exit_handler.register(cleanup_wrapper, priority=10, name=f"PythonInterpreter_{id(self)}")

    def _init_temp_project(self) -> None:
        """使用 uv init 初始化临时项目"""
        try:
            # 在临时目录中运行 uv init
            result = subprocess.run(
                ["uv", "init", "--no-readme", "--name", "python-vm"],
                cwd=self._work_dir,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"Warning: uv init failed: {result.stderr}")
        except Exception as e:
            print(f"Warning: Failed to init temp project: {e}")

    def _start_server(self) -> None:
        """启动子进程服务器（使用 uv）"""
        # 使用 uv --project run python -c "..."，自动使用临时项目的虚拟环境
        cmd = ["uv", "--project", self._work_dir, "run", "python", "-c", self._SERVER_CODE]

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0  # 无缓冲
        )

    def _read_with_timeout(self, n: int, timeout: Optional[float] = None) -> Optional[bytes]:
        """
        从stdout读取n个字节，带超时

        Args:
            n: 要读取的字节数
            timeout: 超时时间（秒），None表示阻塞

        Returns:
            读取的数据，超时返回None
        """
        assert self._proc
        if self._proc.stdout is None:
            return None

        result = b''
        remaining = timeout

        while len(result) < n:
            # 使用select检查是否有数据可读
            if remaining is not None and remaining <= 0:
                return None  # 超时

            try:
                ready, _, _ = select.select([self._proc.stdout], [], [], remaining)
                if not ready:
                    return None  # 超时

                # 有数据可读
                chunk = self._proc.stdout.read(n - len(result))
                if not chunk:
                    return None  # EOF
                result += chunk

                if remaining is not None:
                    # 简单处理：不精确计算耗时
                    remaining = max(0, remaining - 0.001)
            except (OSError, ValueError):
                return None

        return result

    def _execute(self, code: str, timeout: Optional[float] = None) -> ToolResult:
        assert self._proc
        if self._closed:
            raise RuntimeError("Interpreter has been closed")

        if not code or not code.strip():
            return ToolResult(output="", error="", success=True)

        with self._lock:
            assert self._proc
            # 检查进程是否还活着
            if self._proc.poll() is not None:
                # 进程已退出，尝试重启
                error_output = ""
                if self._proc.stderr:
                    error_output = self._proc.stderr.read().decode('utf-8', errors='replace')
                self._start_server()
                return ToolResult(
                    output="",
                    error=f"Subprocess died and was restarted. Previous error: {error_output}",
                    success=False
                )

            # 发送代码（长度前缀 + 数据）
            code_bytes = code.encode('utf-8')
            length_prefix = struct.pack('>I', len(code_bytes))

            try:
                assert self._proc.stdin
                self._proc.stdin.write(length_prefix)
                self._proc.stdin.write(code_bytes)
                self._proc.stdin.flush()
            except BrokenPipeError:
                self._start_server()
                return ToolResult(
                    output="",
                    error="Subprocess connection broken, restarted",
                    success=False
                )

            # 读取响应长度（4字节），带超时
            try:
                length_bytes = self._read_with_timeout(4, timeout)
                if length_bytes is None:
                    # 超时或读取失败，杀死进程
                    self._close_proc()
                    self._start_server()
                    return ToolResult(
                        output="",
                        error=f"Timeout after {timeout}s" if timeout else "Failed to read response length",
                        success=False
                    )

                response_length = struct.unpack('>I', length_bytes)[0]

                # 读取响应数据，带超时
                response_bytes = self._read_with_timeout(response_length, timeout)
                if response_bytes is None:
                    self._close_proc()
                    self._start_server()
                    return ToolResult(
                        output="",
                        error=f"Timeout after {timeout}s" if timeout else "Failed to read complete response",
                        success=False
                    )

                response = json.loads(response_bytes.decode('utf-8'))

                return ToolResult(
                    output=response.get("output", ""),
                    error=response.get("error", ""),
                    success=response.get("success", False)
                )

            except Exception as e:
                return ToolResult(
                    output="",
                    error=f"Communication error: {str(e)}",
                    success=False
                )

    def execute(self, code: str, timeout: Optional[float] = None) -> ToolResult:
        """
        在子进程中执行Python代码

        **重要提示：解释器会保留之前运行过的结果。**
        所有变量、函数定义、导入的模块等都会在多次执行之间保持状态。
        如果需要清空状态重新开始，请调用 restart_server()。

        Args:
            code: 要执行的Python代码
            timeout: 超时时间（秒），None表示不超时

        Returns:
            ToolResult: 执行结果
        """
        result = self._execute(code, timeout)

        if self.print_execution:
            # 构建统一的内容面板
            from rich.text import Text

            # 代码部分
            syntax = Syntax(code, "python", theme="monokai", line_numbers=True)

            # 分隔线
            divider = Text("─" * 60, style="dim")

            # 结果部分
            output_text = str(result.output or "(无输出)")
            error_text = result.error or ""
            success = result.success

            status_emoji = "✅" if success else "❌"
            status_color = "green" if success else "red"
            status_text = "成功" if success else "失败"

            # 构建结果文本
            result_text = Text()
            result_text.append("状态: ", style="bold")
            result_text.append(f"{status_emoji} {status_text}\n", style=f"bold {status_color}")
            result_text.append("\n输出:\n", style="bold cyan")
            result_text.append(output_text)

            if error_text:
                result_text.append("\n\n错误:\n", style="bold red")
                result_text.append(error_text, style="red")

            # 使用 Group 组合内容
            from rich.console import Group
            content = Group(syntax, Text(""), divider, Text(""), result_text)

            _console.print(Panel(content, title="[bold blue]Python 执行[/bold blue]", border_style="blue"))

        return result

    async def execute_streaming(
        self, code: str, timeout: Optional[float] = None
    ) -> AsyncGenerator[str, None]:
        """
        异步流式执行 Python 代码，产生输出片段

        **重要提示：解释器会保留之前运行过的结果。**
        所有变量、函数定义、导入的模块等都会在多次执行之间保持状态。

        Args:
            code: 要执行的 Python 代码
            timeout: 超时时间（秒），None 表示不超时

        Yields:
            输出片段字符串
        """
        import asyncio

        if self._closed:
            raise RuntimeError("Interpreter has been closed")

        if not code or not code.strip():
            return

        # 使用异步锁避免阻塞事件循环
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()

        async with self._async_lock:
            assert self._proc
            # 检查进程是否还活着
            if self._proc.poll() is not None:
                # 进程已退出，尝试重启
                self._start_server()
                yield "Error: Subprocess died and was restarted\n"
                return

            # 添加流式标记
            streaming_code = "# __STREAMING__\n" + code
            code_bytes = streaming_code.encode("utf-8")
            length_prefix = struct.pack(">I", len(code_bytes))

            try:
                assert self._proc.stdin
                self._proc.stdin.write(length_prefix)
                self._proc.stdin.write(code_bytes)
                self._proc.stdin.flush()
            except BrokenPipeError:
                self._start_server()
                yield "Error: Subprocess connection broken, restarted\n"
                return

            # 读取流式响应
            start_time = asyncio.get_event_loop().time()
            while True:
                try:
                    # 检查超时
                    if timeout is not None:
                        elapsed = asyncio.get_event_loop().time() - start_time
                        if elapsed > timeout:
                            self._close_proc()
                            self._start_server()
                            yield f"\n[Timeout after {timeout}s]\n"
                            return

                    # 读取响应长度（4字节）
                    length_bytes = await asyncio.wait_for(
                        self._read_async(4),
                        timeout=1.0 if timeout is None else min(1.0, timeout),
                    )
                    if length_bytes is None:
                        await asyncio.sleep(0.01)
                        continue

                    response_length = struct.unpack(">I", length_bytes)[0]

                    # 读取响应数据
                    response_bytes = await asyncio.wait_for(
                        self._read_async(response_length),
                        timeout=1.0 if timeout is None else min(1.0, timeout),
                    )
                    if response_bytes is None:
                        await asyncio.sleep(0.01)
                        continue

                    response = json.loads(response_bytes.decode("utf-8"))

                    if response.get("stream", False):
                        # 流式片段
                        chunk = response.get("chunk", "")
                        if chunk:
                            yield chunk
                    else:
                        # 最终响应
                        if response.get("error"):
                            yield f"\n[Error]\n{response['error']}\n"
                        elif response.get("output"):
                            yield response["output"]
                        return

                except asyncio.TimeoutError:
                    await asyncio.sleep(0.01)
                    continue
                except Exception as e:
                    yield f"\n[Communication error: {str(e)}]\n"
                    return

    async def _read_async(self, n: int) -> Optional[bytes]:
        """异步读取 n 字节"""
        import asyncio

        assert self._proc
        if self._proc.stdout is None:
            return None

        result = b""
        start_time = asyncio.get_event_loop().time()

        while len(result) < n:
            # 检查超时（总超时 5 秒）
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > 5.0:
                return None

            # 使用 select 检查是否有数据可读
            ready, _, _ = select.select([self._proc.stdout], [], [], 0.1)
            if not ready:
                # 检查进程是否还活着
                if self._proc.poll() is not None:
                    return None
                await asyncio.sleep(0.01)
                continue

            try:
                chunk = self._proc.stdout.read(n - len(result))
                if not chunk:
                    return None
                result += chunk
            except Exception:
                return None

        return result

    def is_alive(self) -> bool:
        """检查子进程是否还活着"""
        if self._closed or self._proc is None:
            return False
        return self._proc.poll() is None

    def restart(self) -> None:
        """重启子进程（清空所有状态）"""
        print("Restarting python interpreter")
        with self._lock:
            self._close_proc()
            self._start_server()

    def restart_server(self) -> ToolResult:
        """
        重启子进程服务器，清空所有状态

        **注意：此操作会清空解释器中保留的所有变量、函数定义和导入的模块。**
        执行此函数后，解释器将回到初始状态，如同新创建时一样。
        """
        self.restart()
        return ToolResult(
            output="Server restarted successfully",
            error="",
            success=True
        )

    def install_dependency(
        self,
        package: str | list[str],
        auto_restart: bool = True
    ) -> ToolResult:
        """
        使用 uv 在临时环境中安装依赖包（隔离，不污染agent项目）

        Args:
            package: 包名或包名列表，如 "requests"、[ "requests", "numpy>=1.20" ]
            auto_restart: 安装成功后是否自动重启解释器，使新包立即可用。
                         默认为 True。设为 False 可手动控制重启时机。

        Returns:
            ToolResult: 安装结果
        """
        # 统一转换为列表
        packages = [package] if isinstance(package, str) else package
        packages_str = ", ".join(packages)
        print(f"Installing dependencies: {packages_str}")

        try:
            # 在临时项目中安装依赖（使用 --project 指定目录）
            cmd = ["uv", "--project", self._work_dir, "add", *packages]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            success = result.returncode == 0
            output = result.stdout

            # 解析 uv add 输出，判断是否有新包被安装
            # uv add 输出包含 "Installed N packages" 表示有实际安装
            # 如果只显示 "Audited" 或 "Resolved" 但没有 "Installed"，表示都已存在
            has_new_install = "Installed" in result.stdout and "packages" in result.stdout

            # 安装成功、有实际新包、且启用自动重启时，才重启解释器
            if success and has_new_install and auto_restart:
                self.restart()
                output += "\n[Interpreter restarted to apply new packages]"
            elif success and not has_new_install:
                output += "\n[All packages already installed, no restart needed]"

            return ToolResult(
                output=output,
                error=result.stderr if not success else "",
                success=success
            )
        except Exception as e:
            return ToolResult(
                output="",
                error=f"Failed to install {package}: {str(e)}",
                success=False
            )

    def _close_proc(self) -> None:
        """关闭子进程"""
        if self._proc is None:
            return

        try:
            # 尝试优雅关闭
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.wait(timeout=1)
        except:
            pass

        # 强制终止
        if self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except:
                pass

        if self._proc.poll() is None:
            try:
                self._proc.kill()
            except:
                pass

    def close(self) -> None:
        """关闭解释器，释放资源"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._close_proc()

            # 清理临时目录（仅当拥有临时目录时才清理）
            if self._owns_temp_dir and os.path.exists(self._work_dir):
                try:
                    shutil.rmtree(self._work_dir)
                except Exception:
                    pass  # 忽略清理失败的错误

            # 从退出处理器中注销清理函数（如果已注册）
            if hasattr(self, '_exit_handler'):
                try:
                    # 注意：ExitHandler 目前没有注销单个函数的方法
                    # 我们可以在 close() 中设置标志，让清理函数检查
                    pass
                except Exception:
                    pass

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """上下文管理器出口"""
        self.close()

    def __del__(self):
        """析构时确保资源释放"""
        if not self._closed:
            self.close()
