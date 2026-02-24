"""
PythonInterpreter 的测试套件
"""

import os
import tempfile
import shutil
import pytest
from typing import Generator

from hawi_plugins.python_interpreter import PythonInterpreter


class TestPythonInterpreter:
    """PythonInterpreter 单解释器测试"""

    @pytest.fixture
    def executor(self) -> Generator[PythonInterpreter, None, None]:
        """创建临时解释器，测试后自动清理"""
        exe = PythonInterpreter()
        yield exe
        exe.close()

    @pytest.fixture
    def temp_dir(self) -> Generator[str, None, None]:
        """创建临时目录，测试后自动清理"""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    class TestBasicExecution:
        """基础代码执行测试"""

        def test_execute_simple_expression(self, executor: PythonInterpreter):
            """测试执行简单表达式"""
            result = executor.execute(code="print(1 + 1)")
            assert result["success"] is True
            assert "2" in result["output"]

        def test_execute_multiple_statements(self, executor: PythonInterpreter):
            """测试执行多行代码"""
            code = """
x = 10
y = 20
print(x + y)
"""
            result = executor.execute(code)
            assert result["success"] is True
            assert "30" in result["output"]

        def test_execute_with_variables(self, executor: PythonInterpreter):
            """测试变量状态保持"""
            executor.execute(code="x = 10")
            result = executor.execute(code="print(x * 10)")
            assert result["success"] is True
            assert "100" in result["output"]

        def test_execute_empty_code(self, executor: PythonInterpreter):
            """测试执行空代码"""
            result = executor.execute(code="")
            assert result["success"] is True
            assert result["output"] == ""

        def test_execute_syntax_error(self, executor: PythonInterpreter):
            """测试语法错误处理"""
            result = executor.execute(code="\1")
            assert result["success"] is False
            assert "SyntaxError" in result["error"] or "error" in result["error"].lower()

        def test_execute_runtime_error(self, executor: PythonInterpreter):
            """测试运行时错误处理"""
            result = executor.execute(code="1/0")
            assert result["success"] is False
            assert "ZeroDivisionError" in result["error"]

        def test_execute_timeout(self, executor: PythonInterpreter):
            """测试超时功能"""
            code = "import time; time.sleep(2)"
            result = executor.execute(code, timeout=0.1)
            assert result["success"] is False
            assert "Timeout" in result["error"]

    class TestStatePersistence:
        """状态持久化测试"""

        def test_imports_persist(self, executor: PythonInterpreter):
            """测试导入在多次执行间保持"""
            executor.execute(code="import math")
            result = executor.execute(code="print(math.pi)")
            assert result["success"] is True
            assert "3.14" in result["output"]

        def test_functions_persist(self, executor: PythonInterpreter):
            """测试函数定义在多次执行间保持"""
            executor.execute("""
def add(a, b):
    return a + b
""")
            result = executor.execute(code="print(add(3, 4))")
            assert result["success"] is True
            assert "7" in result["output"]

        def test_classes_persist(self, executor: PythonInterpreter):
            """测试类定义在多次执行间保持"""
            executor.execute("""
class Person:
    def __init__(self, name):
        self.name = name
    def greet(self):
        return f"Hello, {self.name}"
""")
            executor.execute(code="p = Person('Alice')")
            result = executor.execute(code="print(p.greet())")
            assert result["success"] is True
            assert "Hello, Alice" in result["output"]

    class TestRestart:
        """重启功能测试"""

        def test_restart_clears_state(self, executor: PythonInterpreter):
            """测试重启后状态被清空"""
            executor.execute(code="x = 10")
            executor.restart()
            result = executor.execute(code="print(x)")
            assert result["success"] is False  # NameError

        def test_restart_server_method(self, executor: PythonInterpreter):
            """测试 restart_server 方法"""
            executor.execute(code="x = 10")
            result = executor.restart_server()
            assert result["success"] is True
            result2 = executor.execute(code="print(x)")
            assert result2["success"] is False

        def test_is_alive(self, executor: PythonInterpreter):
            """测试 is_alive 方法"""
            assert executor.is_alive() is True
            executor.close()
            assert executor.is_alive() is False

    class TestDependencyInstallation:
        """依赖安装测试（需要 uv）"""

        @pytest.mark.skipif(shutil.which("uv") is None, reason="uv not installed")
        def test_install_single_package(self, executor: PythonInterpreter):
            """测试安装单个包"""
            result = executor.install_dependency("requests")
            assert result["success"] is True
            # 验证包可用
            test_result = executor.execute(code="import requests; print('requests imported')")
            assert test_result["success"] is True

        @pytest.mark.skipif(shutil.which("uv") is None, reason="uv not installed")
        def test_install_multiple_packages(self, executor: PythonInterpreter):
            """测试安装多个包"""
            result = executor.install_dependency(["requests", "urllib3"])
            assert result["success"] is True

        @pytest.mark.skipif(shutil.which("uv") is None, reason="uv not installed")
        def test_install_already_installed_package(self, executor: PythonInterpreter):
            """测试安装已存在的包（不应重启）"""
            executor.install_dependency("requests")
            # 第二次安装应该检测到已存在
            result = executor.install_dependency("requests")
            assert result["success"] is True
            assert "already installed" in result["output"].lower() or "no restart" in result["output"].lower()

        @pytest.mark.skipif(shutil.which("uv") is None, reason="uv not installed")
        def test_install_with_version_spec(self, executor: PythonInterpreter):
            """测试安装指定版本的包"""
            result = executor.install_dependency("requests>=2.25.0")
            assert result["success"] is True

    class TestContextManager:
        """上下文管理器测试"""

        def test_context_manager(self):
            """测试 with 语句自动清理"""
            with PythonInterpreter() as exe:
                result = exe.execute("print('test')")
                assert result["success"] is True
            # 退出后应该已关闭
            assert not exe.is_alive()

    class TestWorkDir:
        """工作目录测试"""

        def test_custom_work_dir(self, temp_dir: str):
            """测试使用自定义工作目录"""
            executor = PythonInterpreter(work_dir=temp_dir)
            try:
                result = executor.execute(code="print('hello')")
                assert result["success"] is True
            finally:
                executor.close()

        def test_temp_dir_auto_cleanup(self):
            """测试临时目录自动清理"""
            executor = PythonInterpreter()
            work_dir = executor._work_dir
            assert os.path.exists(work_dir)
            executor.close()
            assert not os.path.exists(work_dir)

        def test_custom_work_dir_not_cleaned(self, temp_dir: str):
            """测试自定义工作目录不会被清理"""
            executor = PythonInterpreter(work_dir=temp_dir)
            executor.close()
            assert os.path.exists(temp_dir)  # 应该仍然存在


class TestEdgeCases:
    """边界情况测试"""

    def test_large_code_execution(self):
        """测试执行大量代码"""
        with PythonInterpreter() as exe:
            # 生成大量代码 - 使用 globals() 动态访问变量
            code = "\n".join([f"x{i} = {i}" for i in range(100)])
            code += "\nprint(sum([globals()[f'x{i}'] for i in range(100)]))"
            result = exe.execute(code)
            assert result["success"] is True
            assert "4950" in result["output"]  # 0+1+2+...+99 = 4950

    def test_unicode_in_code(self):
        """测试 Unicode 字符处理"""
        with PythonInterpreter() as exe:
            result = exe.execute('print("你好世界 🎉")')
            assert result["success"] is True
            assert "你好世界" in result["output"]

    def test_multiline_string(self):
        """测试多行字符串"""
        with PythonInterpreter() as exe:
            code = '''
text = """
Line 1
Line 2
Line 3
"""
print(text)
'''
            result = exe.execute(code)
            assert result["success"] is True
            assert "Line 2" in result["output"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
