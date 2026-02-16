"""
PythonInterpreter 和 MultiPythonInterpreter 的测试套件
"""

import os
import tempfile
import shutil
import pytest
from typing import Generator

from hawi_plugins.python_interpreter import PythonInterpreter,MultiPythonInterpreter

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
            executor.execute(code="\1")
            executor.restart()
            result = executor.execute(code="\1")
            assert result["success"] is False  # NameError

        def test_restart_server_method(self, executor: PythonInterpreter):
            """测试 restart_server 方法"""
            executor.execute(code="\1")
            result = executor.restart_server()
            assert result["success"] is True
            result2 = executor.execute(code="\1")
            assert result2["success"] is False

        def test_is_alive(self, executor: PythonInterpreter):
            """测试 is_alive 方法"""
            assert executor.is_alive() is True
            executor.close()
            assert executor.is_alive() is False

    class TestScriptManagement:
        """脚本管理功能测试"""

        def test_save_script(self, executor: PythonInterpreter):
            """测试保存脚本"""
            result = executor.save_script("test_script", "print('hello')", "Test description")
            assert "saved" in result.lower() or "success" in result.lower()

        def test_save_script_with_description(self, executor: PythonInterpreter):
            """测试保存带描述的脚本"""
            executor.save_script("desc_test", "x = 1", "This is a test script")
            scripts = executor.list_scripts()
            assert len(scripts) == 1
            assert scripts[0]["name"] == "desc_test.py"
            assert "test script" in scripts[0]["description"]

        def test_execute_script(self, executor: PythonInterpreter):
            """测试执行脚本"""
            executor.save_script("calc", "print(2 + 3)")
            result = executor.execute_script("calc")
            assert result["success"] is True
            assert "5" in result["output"]

        def test_delete_script(self, executor: PythonInterpreter):
            """测试删除脚本"""
            executor.save_script("to_delete", "pass")
            result = executor.delete_script("to_delete")
            assert "deleted" in result.lower() or "success" in result.lower()
            scripts = executor.list_scripts()
            assert len(scripts) == 0

        def test_read_script(self, executor: PythonInterpreter):
            """测试读取脚本"""
            executor.save_script("readable", "print('hello')", "A test script")
            result = executor.read_script("readable")
            assert result["name"] == "readable.py"
            assert "test script" in result["description"]
            assert "print('hello')" in result["code"]

        def test_read_script_not_found(self, executor: PythonInterpreter):
            """测试读取不存在的脚本"""
            with pytest.raises(FileNotFoundError):
                executor.read_script("nonexistent")

        def test_list_scripts_empty(self, executor: PythonInterpreter):
            """测试空脚本列表"""
            scripts = executor.list_scripts()
            assert scripts == []

        def test_list_scripts_multiple(self, executor: PythonInterpreter):
            """测试列出多个脚本"""
            executor.save_script("script1", "x = 1", "First script")
            executor.save_script("script2", "y = 2", "Second script")
            scripts = executor.list_scripts()
            assert len(scripts) == 2
            names = [s["name"] for s in scripts]
            assert "script1.py" in names
            assert "script2.py" in names

        def test_execute_nonexistent_script(self, executor: PythonInterpreter):
            """测试执行不存在的脚本"""
            with pytest.raises(FileNotFoundError):
                executor.execute_script("nonexistent")

        def test_delete_nonexistent_script(self, executor: PythonInterpreter):
            """测试删除不存在的脚本"""
            with pytest.raises(FileNotFoundError):
                executor.delete_script("nonexistent")

        def test_script_auto_adds_py_extension(self, executor: PythonInterpreter):
            """测试自动添加 .py 后缀"""
            executor.save_script("myscript", "print(1)")  # 无后缀
            result = executor.execute_script("myscript.py")  # 有后缀
            assert result["success"] is True

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
                # 检查脚本目录是否创建在正确位置
                scripts_dir = os.path.join(temp_dir, "scripts")
                executor.save_script("test", "pass")
                assert os.path.exists(os.path.join(scripts_dir, "test.py"))
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


class TestMultiPythonInterpreter:
    """MultiPythonInterpreter 多解释器测试"""

    @pytest.fixture
    def multi(self) -> Generator[MultiPythonInterpreter, None, None]:
        """创建多解释器管理器"""
        m = MultiPythonInterpreter()
        yield m
        # 清理所有解释器
        for name in list(m.interpreters.keys()):
            m.remove_interpreters(name)

    class TestCreateAndRemove:
        """创建和移除解释器测试"""

        def test_create_interpreter(self, multi: MultiPythonInterpreter):
            """测试创建解释器"""
            result = multi.create_interpreter("test1")
            assert "created" in result.lower() or "success" in result.lower()
            assert "test1" in multi.interpreters

        def test_create_interpreter_auto_name(self, multi: MultiPythonInterpreter):
            """测试自动生成解释器名称"""
            result = multi.create_interpreter()
            assert "interpret_0" in result
            result2 = multi.create_interpreter()
            assert "interpret_1" in result2

        def test_create_duplicate_name(self, multi: MultiPythonInterpreter):
            """测试创建重复名称的解释器"""
            multi.create_interpreter("dup")
            result = multi.create_interpreter("dup")
            assert "already exists" in result.lower()

        def test_remove_interpreter(self, multi: MultiPythonInterpreter):
            """测试移除解释器"""
            multi.create_interpreter("to_remove")
            result = multi.remove_interpreters("to_remove")
            assert "closed" in result.lower() or "removed" in result.lower()
            assert "to_remove" not in multi.interpreters

        def test_remove_nonexistent_interpreter(self, multi: MultiPythonInterpreter):
            """测试移除不存在的解释器"""
            with pytest.raises(KeyError):
                multi.remove_interpreters("nonexistent")

        def test_get_interpreter_not_found(self, multi: MultiPythonInterpreter):
            """测试获取不存在的解释器"""
            with pytest.raises(KeyError):
                multi.get_interpreter("nonexistent")

    class TestMultiExecution:
        """多解释器执行测试"""

        def test_multiple_interpreters_isolated(self, multi: MultiPythonInterpreter):
            """测试多个解释器之间状态隔离"""
            multi.create_interpreter("exe1")
            multi.create_interpreter("exe2")

            multi.execute("exe1", "x = 100")
            multi.execute("exe2", "x = 200")

            result1 = multi.execute("exe1", "print(x)")
            result2 = multi.execute("exe2", "print(x)")

            assert "100" in result1["output"]
            assert "200" in result2["output"]

        def test_execute_nonexistent_interpreter(self, multi: MultiPythonInterpreter):
            """测试在不存在解释器上执行"""
            with pytest.raises(KeyError):
                multi.execute("nonexistent", "print(1)")

    class TestMultiScripts:
        """多解释器脚本测试"""

        def test_save_script_per_interpreter(self, multi: MultiPythonInterpreter):
            """测试每个解释器有独立的脚本"""
            multi.create_interpreter("exe1")
            multi.create_interpreter("exe2")

            multi.save_script("exe1", "script", "print('exe1')")
            multi.save_script("exe2", "script", "print('exe2')")

            result1 = multi.execute_script("exe1", "script")
            result2 = multi.execute_script("exe2", "script")

            assert "exe1" in result1["output"]
            assert "exe2" in result2["output"]

        def test_list_scripts_per_interpreter(self, multi: MultiPythonInterpreter):
            """测试列出各解释器的脚本"""
            multi.create_interpreter("exe1")
            multi.create_interpreter("exe2")

            multi.save_script("exe1", "s1", "pass")
            multi.save_script("exe2", "s2", "pass")

            scripts1 = multi.list_scripts("exe1")
            scripts2 = multi.list_scripts("exe2")

            assert len(scripts1) == 1
            assert len(scripts2) == 1
            assert scripts1[0]["name"] == "s1.py"
            assert scripts2[0]["name"] == "s2.py"

    class TestGetTools:
        """工具列表测试"""

        def test_tools_property_returns_tools(self, multi: MultiPythonInterpreter):
            """测试 tools 属性返回工具列表"""
            tools = multi.tools
            assert len(tools) >= 5
            for tool in tools:
                # tools are FunctionAgentTool instances with invoke method
                assert hasattr(tool, 'invoke')

        def test_tool_names(self, multi: MultiPythonInterpreter):
            """测试工具名称"""
            tools = multi.tools
            tool_names = [t.name for t in tools]
            assert "MultiPythonInterpreter__create_interpreter" in tool_names
            assert "MultiPythonInterpreter__remove_interpreters" in tool_names
            assert "MultiPythonInterpreter__save_script" in tool_names
            assert "MultiPythonInterpreter__execute_script" in tool_names
            assert "MultiPythonInterpreter__delete_script" in tool_names
            assert "MultiPythonInterpreter__list_scripts" in tool_names
            assert "MultiPythonInterpreter__install_dependency" in tool_names
            assert "MultiPythonInterpreter__restart_interpreter" in tool_names
            assert "MultiPythonInterpreter__execute" in tool_names


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
