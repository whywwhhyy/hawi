import os
import tempfile
import shutil
from typing import Optional, Dict, Literal, AsyncGenerator, Any
from threading import Lock

from hawi.utils.lifecycle import ExitHandler
from .python_interpreter import PythonInterpreter

from hawi.tool import ToolResult
from hawi.plugin import HawiPlugin
import hawi.plugin as plugin


class PythonInterpreterPlugin(HawiPlugin):
    """多Python解释器管理器 - 统一接口的Python解释器Plugin

    提供多个独立的Python解释器实例管理，同时支持一个默认实例（instance=None时）。
    所有工具方法的interpreter_name参数默认为None，表示使用默认实例。

    脚本管理功能集中在此类中，不依赖于特定解释器实例，而是使用统一的工作目录。
    """

    DEFAULT_INSTANCE_NAME = "__default__"

    class Instance:
        def __init__(self, *args, **kwargs):
            self.lock = Lock()  # 同步锁，用于线程安全
            self.async_lock: Any = None  # 异步锁，用于协程安全（延迟创建）
            self.executor = PythonInterpreter(*args, **kwargs)
            self.running: bool = False

    @classmethod
    def gui_config_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "work_dir": {
                    "type": "string",
                    "title": "Work Directory",
                    "default": ".hawi/python_interpreter",
                    "description": "Working directory for interpreter scripts and project files.",
                },
                "print_execution": {
                    "type": "boolean",
                    "title": "Print Execution Logs",
                    "default": False,
                    "description": "Whether to print execution logs from the interpreter subprocess.",
                },
            },
            "additionalProperties": False,
        }

    @classmethod
    def gui_default_config(cls) -> dict:
        return {
            "work_dir": ".hawi/python_interpreter",
            "print_execution": False,
        }

    def __init__(self, work_dir: Optional[str] = None, print_execution: bool = False):
        """
        初始化多解释器管理器

        Args:
            work_dir: 脚本存储的工作目录，若为 None 则使用临时目录
            print_execution: 是否打印执行输出
        """
        self.lock = Lock()
        self.interpreters: Dict[str, PythonInterpreterPlugin.Instance] = {}
        self.print_execution = print_execution
        self._closed = False

        # 初始化脚本工作目录
        self._owns_work_dir = work_dir is None
        if work_dir is None:
            self._work_dir = tempfile.mkdtemp(prefix="python_scripts_")
        else:
            if not os.path.exists(work_dir):
                os.makedirs(work_dir, exist_ok=True)
            self._work_dir = work_dir

        self._exit_handler = ExitHandler.get_instance()

        def cleanup_wrapper():
            if not self._closed:
                self.close()
        self._exit_handler.register(cleanup_wrapper, priority=5, name=f"MultiPythonInterpreter_{id(self)}")

    def _get_script_path(self, script_name: str) -> str:
        """获取脚本的完整路径"""
        if not script_name.endswith('.py'):
            script_name += '.py'
        script_name = os.path.basename(script_name)
        scripts_dir = os.path.join(self._work_dir, "scripts")
        return os.path.join(scripts_dir, script_name)

    def _get_instance(self, interpreter_name: Optional[str]) -> 'PythonInterpreterPlugin.Instance':
        """获取解释器实例，name为None时返回默认实例

        如果默认实例不存在，会自动创建。
        """
        name = interpreter_name or self.DEFAULT_INSTANCE_NAME
        with self.lock:
            if name not in self.interpreters:
                if interpreter_name is None:
                    # 自动创建默认实例
                    self.interpreters[name] = PythonInterpreterPlugin.Instance(
                        print_execution=self.print_execution
                    )
                else:
                    raise KeyError(f"Interpreter '{interpreter_name}' not found")
            return self.interpreters[name]

    def get_interpreter(self, interpreter_name: Optional[str] = None) -> 'PythonInterpreterPlugin.Instance':
        """获取指定名称的解释器实例

        Args:
            interpreter_name: 解释器名称，若为None则返回默认实例

        Returns:
            Instance: 解释器实例包装器

        Raises:
            KeyError: 指定名称的解释器不存在（默认实例会自动创建）
        """
        return self._get_instance(interpreter_name)

    @plugin.tool
    def create_interpreter(
        self,
        interpreter_name: Optional[str] = None,
        work_dir: Optional[str] = None
    ) -> str:
        """创建一个新的 Python 解释器实例

        Args:
            interpreter_name: 解释器名称，若为None则自动生成
            work_dir: 工作目录，若为None则使用临时目录

        Returns:
            str: 创建结果信息
        """
        with self.lock:
            # 自动生成名称
            if interpreter_name is None:
                index = len(self.interpreters)
                while f"interpret_{index}" in self.interpreters:
                    index += 1
                interpreter_name = f"interpret_{index}"

            # 不能创建名为 __default__ 的解释器
            if interpreter_name == self.DEFAULT_INSTANCE_NAME:
                return f"Cannot create interpreter with reserved name '{interpreter_name}'"

            if interpreter_name in self.interpreters:
                return f"Interpreter '{interpreter_name}' already exists!"

            self.interpreters[interpreter_name] = PythonInterpreterPlugin.Instance(
                work_dir, print_execution=self.print_execution
            )
            return f"Created interpreter '{interpreter_name}'"

    @plugin.tool
    def remove_interpreters(self, interpreter_names: str | list[str]) -> str:
        """关闭并移除指定的 Python 解释器实例

        Args:
            interpreter_names: 解释器实例名称或名称列表

        Returns:
            str: 关闭结果信息
        """
        if isinstance(interpreter_names, str):
            interpreter_names = [interpreter_names]

        results = []
        for name in interpreter_names:
            # 不能移除默认实例
            if name == self.DEFAULT_INSTANCE_NAME:
                results.append(f"Cannot remove default interpreter '{name}'")
                continue

            with self.lock:
                if name not in self.interpreters:
                    raise KeyError(f"Interpreter '{name}' not found")
                instance = self.interpreters.pop(name)
            with instance.lock:
                instance.executor.close()
            results.append(f"Closed interpreter '{name}'")

        return "\n".join(results) if len(results) > 1 else results[0]

    @plugin.tool
    def install_dependency(
        self,
        package: str | list[str],
        interpreter_name: Optional[str] = None,
        auto_restart: bool = True
    ) -> ToolResult:
        """在指定解释器实例的临时环境中安装依赖包

        Args:
            package: 包名或包名列表，如 "requests"、[ "requests", "numpy>=1.20" ]
            interpreter_name: 解释器实例名称，None表示默认实例
            auto_restart: 安装成功后是否自动重启解释器使新包生效，默认为True

        Returns:
            ToolResult: 安装结果
        """
        instance = self._get_instance(interpreter_name)
        with instance.lock:
            return instance.executor.install_dependency(package, auto_restart)

    @plugin.tool
    def restart_interpreter(self, interpreter_name: Optional[str] = None) -> ToolResult:
        """重启指定的 Python 解释器实例

        **注意：此操作会清空解释器中保留的所有变量、函数定义和导入的模块。**

        Args:
            interpreter_name: 解释器实例名称，None表示默认实例

        Returns:
            ToolResult: 重启结果
        """
        instance = self._get_instance(interpreter_name)
        with instance.lock:
            return instance.executor.restart_server()

    @plugin.tool
    async def execute(
        self,
        code: str,
        interpreter_name: Optional[str] = None,
        timeout: Optional[float] = None
    ):
        """在指定解释器实例中执行 Python 代码（流式输出）

        **重要提示：解释器会保留之前运行过的结果。**
        所有变量、函数定义、导入的模块等都会在多次执行之间保持状态。

        此工具以流式方式返回执行结果，可以实时看到代码输出。

        Args:
            code: 要执行的 Python 代码
            interpreter_name: 解释器实例名称，None表示默认实例
            timeout: 超时时间（秒），None表示不超时

        Yields:
            str: 代码执行的输出片段
        """
        import asyncio

        instance = self._get_instance(interpreter_name)

        # 使用异步锁避免阻塞事件循环（允许响应 Ctrl+C）
        if instance.async_lock is None:
            instance.async_lock = asyncio.Lock()

        async with instance.async_lock:
            async for chunk in instance.executor.execute_streaming(code, timeout):
                yield chunk

    # ==================== 脚本管理方法（统一存储，不绑定特定解释器） ====================

    @plugin.tool
    def save_script(
        self,
        script_name: str,
        code: str,
        description: str = ""
    ) -> str:
        """保存脚本到脚本目录

        Args:
            script_name: 脚本文件名（会自动添加 .py 后缀）
            code: 脚本代码内容
            description: 脚本描述，会保存在脚本开头

        Returns:
            str: 保存结果信息
        """
        script_path = self._get_script_path(script_name)
        scripts_dir = os.path.dirname(script_path)
        os.makedirs(scripts_dir, exist_ok=True)

        with open(script_path, 'w', encoding='utf-8') as f:
            if description:
                for line in description.strip().split('\n'):
                    f.write(f"# {line}\n")
                f.write("\n")
            f.write(code)
        return f"Script '{os.path.basename(script_path)}' saved"

    @plugin.tool
    async def execute_script(
        self,
        script_name: str,
        interpreter_name: Optional[str] = None,
        timeout: Optional[float] = None
    ):
        """执行脚本目录中的脚本（流式输出）

        脚本内容会被读取并在指定解释器中执行。

        Args:
            script_name: 脚本文件名
            interpreter_name: 解释器实例名称，None表示默认实例
            timeout: 超时时间（秒）

        Yields:
            str: 脚本执行的输出片段
        """
        script_path = self._get_script_path(script_name)
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Script '{script_name}' not found")

        with open(script_path, 'r', encoding='utf-8') as f:
            code = f.read()

        # 在指定解释器中执行脚本内容（流式）
        async for chunk in self.execute(code, interpreter_name, timeout):
            yield chunk

    @plugin.tool
    def delete_script(self, script_name: str) -> str:
        """删除脚本目录中的脚本

        Args:
            script_name: 脚本文件名

        Returns:
            str: 删除结果信息
        """
        script_path = self._get_script_path(script_name)
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Script '{script_name}' not found")
        os.remove(script_path)
        return f"Script '{os.path.basename(script_path)}' deleted"

    @plugin.tool
    def list_scripts(self) -> list[dict]:
        """列出脚本目录中的所有脚本及其描述

        Returns:
            list[dict]: 脚本信息列表，每项包含 name 和 description
        """
        scripts_dir = os.path.join(self._work_dir, "scripts")
        if not os.path.exists(scripts_dir):
            return []

        result = []
        for filename in os.listdir(scripts_dir):
            if filename.endswith('.py'):
                script_path = os.path.join(scripts_dir, filename)
                description = ""
                with open(script_path, 'r', encoding='utf-8') as f:
                    lines = []
                    for line in f:
                        if line.startswith('# '):
                            lines.append(line[2:].strip())
                        elif line.startswith('#'):
                            lines.append(line[1:].strip())
                        else:
                            break
                    description = '\n'.join(lines)
                result.append({
                    "name": filename,
                    "description": description
                })
        return result

    @plugin.tool
    def read_script(self, script_name: str) -> dict:
        """读取脚本内容

        Args:
            script_name: 脚本文件名

        Returns:
            dict: 包含 name、description 和 code 的字典

        Raises:
            FileNotFoundError: 脚本不存在
        """
        script_path = self._get_script_path(script_name)
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Script '{script_name}' not found")

        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 解析描述（开头的注释行）
        lines = content.split('\n')
        desc_lines = []
        code_lines = []
        in_desc = True

        for line in lines:
            if in_desc:
                if line.startswith('# '):
                    desc_lines.append(line[2:])
                elif line.startswith('#'):
                    desc_lines.append(line[1:])
                elif line.strip() == '':
                    desc_lines.append('')
                else:
                    in_desc = False
                    code_lines.append(line)
            else:
                code_lines.append(line)

        # 移除描述末尾的空行
        while desc_lines and desc_lines[-1] == '':
            desc_lines.pop()

        return {
            "name": os.path.basename(script_path),
            "description": '\n'.join(desc_lines),
            "code": '\n'.join(code_lines)
        }

    def close(self) -> None:
        """关闭所有解释器，释放资源"""
        with self.lock:
            if self._closed:
                return
            self._closed = True

            # 关闭所有解释器
            for interpreter_name, instance in list(self.interpreters.items()):
                try:
                    with instance.lock:
                        instance.executor.close()
                except Exception:
                    pass  # 忽略关闭过程中的错误

            # 清空解释器字典
            self.interpreters.clear()

            # 清理脚本工作目录（仅当拥有时才清理）
            if self._owns_work_dir and os.path.exists(self._work_dir):
                try:
                    shutil.rmtree(self._work_dir)
                except Exception:
                    pass  # 忽略清理失败

    def clone(self) -> 'PythonInterpreterPlugin':
        """创建此插件的新实例用于 fork/clone 操作。

        返回一个全新的 PythonInterpreterPlugin 实例，具有：
        - 新的独立工作目录
        - 空的解释器字典（没有预创建的解释器）
        - 相同的 print_execution 配置

        这是 HawiPlugin.clone() 的实现，确保在 agent fork/clone 时
        不会共享 Python 解释器子进程状态。

        Returns:
            全新的 PythonInterpreterPlugin 实例
        """
        # 创建新实例，不传递 work_dir 以创建新的临时目录
        # 这样脚本目录也是独立的
        new_plugin = PythonInterpreterPlugin(
            work_dir=None,  # 创建新的临时工作目录
            print_execution=self.print_execution
        )
        return new_plugin

    def save_state(self) -> dict:
        """Capture state that survives a process restart.

        Subprocess Python state (defined functions, imports, in-memory
        variables) is fundamentally lost when the interpreter dies, so we
        only persist what's reproducible: the work directory and the names
        of declared interpreters. Saved scripts on disk survive naturally
        because they live under ``self._work_dir``.
        """
        with self.lock:
            interpreter_names = list(self.interpreters.keys())
        return {
            "work_dir": self._work_dir,
            "owns_work_dir": self._owns_work_dir,
            "print_execution": self.print_execution,
            "interpreter_names": interpreter_names,
        }

    def load_state(self, data: dict) -> None:
        """Re-establish interpreter slots after a restart.

        Each previously-declared interpreter is reinstantiated empty —
        callers must re-run any setup scripts that established prior in-process
        state. Callers should treat this as a clean restart.
        """
        work_dir = data.get("work_dir")
        if work_dir and os.path.isdir(work_dir):
            self._work_dir = work_dir
            self._owns_work_dir = bool(data.get("owns_work_dir", False))
        self.print_execution = bool(
            data.get("print_execution", self.print_execution)
        )

        with self.lock:
            for name in data.get("interpreter_names", []):
                if name in self.interpreters:
                    continue
                self.interpreters[name] = PythonInterpreterPlugin.Instance(
                    print_execution=self.print_execution
                )

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
