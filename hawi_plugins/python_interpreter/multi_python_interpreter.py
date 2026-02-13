from typing import Optional,TypedDict,Dict
from threading import Lock

from hawi.utils.lifecycle import ExitHandler, exit_scope
from .python_interpreter import PythonInterpreter

from hawi.tool import ToolResult
from hawi.plugin import HawiPlugin
import hawi.plugin as plugin

class MultiPythonInterpreter(HawiPlugin):
    class Instance:
        def __init__(self, *args, **kwargs):
            self.lock = Lock()
            self.executor = PythonInterpreter(*args, **kwargs)

    def __init__(self):
        self.lock = Lock()
        self.interpreters:Dict[str,MultiPythonInterpreter.Instance] = {}
        self._closed = False
        self._exit_handler = ExitHandler.get_instance()
        def cleanup_wrapper():
            if not self._closed:
                self.close()
        self._exit_handler.register(cleanup_wrapper, priority=5, name=f"MultiPythonInterpreter_{id(self)}")

    def get_interpreter(self,
               interpreter_name: Optional[str]) -> 'MultiPythonInterpreter.Instance':
        """获取指定名称的解释器实例，不存在则抛出 KeyError"""
        with self.lock:
            if interpreter_name not in self.interpreters:
                raise KeyError(f"Interpreter '{interpreter_name}' not found")
            return self.interpreters[interpreter_name]

    @plugin.tool
    def create_interpreter(self,
               interpreter_name: Optional[str] = None,
               work_dir: Optional[str] = None) -> str:
        """
        创建一个新的 Python 解释器实例

        Args:
            interpreter_name: 解释器名称，若为 None 则自动生成
            work_dir: 工作目录，若为 None 则使用临时目录；若目录不存在或为空则自动初始化

        Returns:
            str: 创建结果信息
        """
        with self.lock:
            index = len(self.interpreters)
            while interpreter_name is None:
                interpreter_name = f"interpret_{index}"
                if interpreter_name in self.interpreters:
                    index += 1
                    interpreter_name = None
            if interpreter_name in self.interpreters:
                return f"Interpreter '{interpreter_name}' already exists!"
            self.interpreters[interpreter_name] = MultiPythonInterpreter.Instance(work_dir)
            return f"Created interpreter '{interpreter_name}'"

    @plugin.tool
    def remove_interpreters(self, interpreter_names: str | list[str]) -> str:
        """
        关闭并移除指定的 Python 解释器实例

        Args:
            interpreter_names: 解释器实例名称或名称列表

        Returns:
            str: 关闭结果信息
        """
        if isinstance(interpreter_names, str):
            interpreter_names = [interpreter_names]
        
        results = []
        for name in interpreter_names:
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
        interpreter_name: str,
        package: str | list[str],
        auto_restart: bool = True
    ) -> ToolResult:
        """
        在指定解释器实例的临时环境中安装依赖包

        Args:
            interpreter_name: 解释器实例名称
            package: 包名或包名列表，如 "requests"、["requests", "numpy>=1.20"]
            auto_restart: 安装成功后是否自动重启解释器使新包生效，默认为 True

        Returns:
            ToolResult: 安装结果，包含输出、错误信息和成功状态
        """
        instance = self.get_interpreter(interpreter_name)
        with instance.lock:
            return instance.executor.install_dependency(package, auto_restart)

    @plugin.tool
    def restart_interpreter(self, interpreter_name: str) -> ToolResult:
        """
        重启指定的 Python 解释器实例

        **注意：此操作会清空解释器中保留的所有变量、函数定义和导入的模块。**
        执行此函数后，解释器将回到初始状态，如同新创建时一样。
        通常在安装新依赖后或需要清除状态时使用。

        Args:
            interpreter_name: 解释器实例名称

        Returns:
            ToolResult: 重启结果
        """
        instance = self.get_interpreter(interpreter_name)
        with instance.lock:
            return instance.executor.restart_server()

    @plugin.tool
    def execute(self, interpreter_name: str, code: str, timeout: Optional[float] = None) -> ToolResult:
        """
        在指定解释器实例中执行 Python 代码

        **重要提示：解释器会保留之前运行过的结果。**
        所有变量、函数定义、导入的模块等都会在多次执行之间保持状态。
        如果需要清空状态重新开始，请调用 restart_interpreter()。

        Args:
            interpreter_name: 解释器实例名称
            code: 要执行的 Python 代码
            timeout: 超时时间（秒），None 表示不超时

        Returns:
            ToolResult: 执行结果，包含 stdout 输出、错误信息和成功状态
        """
        instance = self.get_interpreter(interpreter_name)
        with instance.lock:
            return instance.executor.execute(code, timeout)

    @plugin.tool
    def save_script(self, interpreter_name: str, script_name: str, code: str, description: str = "") -> str:
        """
        保存脚本到指定解释器的脚本目录

        Args:
            interpreter_name: 解释器实例名称
            script_name: 脚本文件名（会自动添加 .py 后缀）
            code: 脚本代码内容
            description: 脚本描述，会保存在脚本开头

        Returns:
            str: 保存结果信息
        """
        instance = self.get_interpreter(interpreter_name)
        with instance.lock:
            return instance.executor.save_script(script_name, code, description)

    @plugin.tool
    def execute_script(self, interpreter_name: str, script_name: str, timeout: Optional[float] = None) -> ToolResult:
        """
        执行指定解释器脚本目录中的脚本

        **重要提示：解释器会保留之前运行过的结果。**
        脚本执行时可以看到之前代码执行中定义的变量和导入的模块。
        如果需要清空状态重新开始，请调用 restart_interpreter()。

        Args:
            interpreter_name: 解释器实例名称
            script_name: 脚本文件名
            timeout: 超时时间（秒），None 表示不超时

        Returns:
            ToolResult: 执行结果
        """
        instance = self.get_interpreter(interpreter_name)
        with instance.lock:
            return instance.executor.execute_script(script_name, timeout)

    @plugin.tool
    def delete_script(self, interpreter_name: str, script_name: str) -> str:
        """
        删除指定解释器脚本目录中的脚本

        Args:
            interpreter_name: 解释器实例名称
            script_name: 脚本文件名

        Returns:
            str: 删除结果信息
        """
        instance = self.get_interpreter(interpreter_name)
        with instance.lock:
            return instance.executor.delete_script(script_name)

    @plugin.tool
    def list_scripts(self, interpreter_name: str) -> list[dict]:
        """
        列出指定解释器脚本目录中的所有脚本及其描述

        Args:
            interpreter_name: 解释器实例名称

        Returns:
            list[dict]: 脚本信息列表，每项包含 name 和 description
        """
        instance = self.get_interpreter(interpreter_name)
        with instance.lock:
            return instance.executor.list_scripts()

    @plugin.tool
    def read_script(self, interpreter_name: str, script_name: str) -> dict:
        """
        读取指定解释器脚本目录中的脚本内容

        Args:
            interpreter_name: 解释器实例名称
            script_name: 脚本文件名

        Returns:
            dict: 包含 name、description 和 code 的字典
        """
        instance = self.get_interpreter(interpreter_name)
        with instance.lock:
            return instance.executor.read_script(script_name)

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
