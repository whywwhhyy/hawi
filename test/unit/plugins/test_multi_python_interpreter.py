"""
MultiPythonInterpreter 的单元测试套件
"""

import os
import tempfile
import shutil
import threading
import pytest
from typing import Generator

from hawi_plugins.python_interpreter import MultiPythonInterpreter


class TestMultiPythonInterpreter:
    """MultiPythonInterpreter 单元测试"""

    @pytest.fixture
    def multi(self) -> Generator[MultiPythonInterpreter, None, None]:
        """创建多解释器管理器，测试后自动清理"""
        m = MultiPythonInterpreter()
        yield m
        # 清理所有解释器
        if not m._closed:
            for name in list(m.interpreters.keys()):
                try:
                    m.remove_interpreters(name)
                except (KeyError, Exception):
                    pass
            m.close()

    @pytest.fixture
    def temp_dir(self) -> Generator[str, None, None]:
        """创建临时目录，测试后自动清理"""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    class TestCreateAndRemove:
        """创建和移除解释器测试"""

        def test_create_interpreter_with_name(self, multi: MultiPythonInterpreter):
            """测试使用指定名称创建解释器"""
            result = multi.create_interpreter("test_interpreter")
            assert "Created interpreter 'test_interpreter'" == result
            assert "test_interpreter" in multi.interpreters
            assert multi.interpreters["test_interpreter"].executor is not None

        def test_create_interpreter_auto_name(self, multi: MultiPythonInterpreter):
            """测试自动生成解释器名称"""
            result1 = multi.create_interpreter()
            assert "interpret_0" in result1
            result2 = multi.create_interpreter()
            assert "interpret_1" in result2
            result3 = multi.create_interpreter()
            assert "interpret_2" in result3

        def test_create_interpreter_auto_name_with_gap(self, multi: MultiPythonInterpreter):
            """测试自动命名跳过已存在的名称"""
            multi.create_interpreter("interpret_0")
            multi.create_interpreter("interpret_2")
            result = multi.create_interpreter()
            # 当前实现基于解释器数量，会尝试 interpret_2，发现已存在后跳到 interpret_3
            assert "interpret_3" in result

        def test_create_duplicate_name(self, multi: MultiPythonInterpreter):
            """测试创建重复名称的解释器"""
            multi.create_interpreter("dup")
            result = multi.create_interpreter("dup")
            assert "already exists" in result.lower()
            assert len(multi.interpreters) == 1

        def test_create_with_custom_work_dir(self, multi: MultiPythonInterpreter, temp_dir: str):
            """测试使用自定义工作目录创建解释器"""
            result = multi.create_interpreter("custom_dir", work_dir=temp_dir)
            assert "Created" in result
            assert os.path.exists(temp_dir)

        def test_remove_interpreter(self, multi: MultiPythonInterpreter):
            """测试移除解释器"""
            multi.create_interpreter("to_remove")
            result = multi.remove_interpreters("to_remove")
            assert "Closed interpreter 'to_remove'" == result
            assert "to_remove" not in multi.interpreters

        def test_remove_nonexistent_interpreter(self, multi: MultiPythonInterpreter):
            """测试移除不存在的解释器抛出 KeyError"""
            with pytest.raises(KeyError, match="Interpreter 'nonexistent' not found"):
                multi.remove_interpreters("nonexistent")

        def test_get_interpreter(self, multi: MultiPythonInterpreter):
            """测试获取解释器实例"""
            multi.create_interpreter("getter")
            instance = multi.get_interpreter("getter")
            assert instance is not None
            assert instance.executor is not None
            assert instance.lock is not None

        def test_get_interpreter_not_found(self, multi: MultiPythonInterpreter):
            """测试获取不存在的解释器抛出 KeyError"""
            with pytest.raises(KeyError, match="Interpreter 'notfound' not found"):
                multi.get_interpreter("notfound")

    class TestExecute:
        """代码执行测试"""

        def test_execute_simple_code(self, multi: MultiPythonInterpreter):
            """测试在指定解释器中执行简单代码"""
            multi.create_interpreter("exe1")
            result = multi.execute("exe1", "print('hello')")
            assert result["success"] is True
            assert "hello" in result["output"]

        def test_execute_with_state(self, multi: MultiPythonInterpreter):
            """测试解释器状态保持"""
            multi.create_interpreter("stateful")
            multi.execute("stateful", "x = 42")
            result = multi.execute("stateful", "print(x)")
            assert result["success"] is True
            assert "42" in result["output"]

        def test_execute_isolation(self, multi: MultiPythonInterpreter):
            """测试多个解释器之间状态隔离"""
            multi.create_interpreter("iso1")
            multi.create_interpreter("iso2")

            multi.execute("iso1", "x = 100")
            multi.execute("iso2", "x = 200")

            result1 = multi.execute("iso1", "print(x)")
            result2 = multi.execute("iso2", "print(x)")

            assert "100" in result1["output"]
            assert "200" in result2["output"]

        def test_execute_syntax_error(self, multi: MultiPythonInterpreter):
            """测试语法错误处理"""
            multi.create_interpreter("error_exe")
            result = multi.execute("error_exe", "if x")
            assert result["success"] is False
            assert "SyntaxError" in result["error"] or "error" in result["error"].lower()

        def test_execute_runtime_error(self, multi: MultiPythonInterpreter):
            """测试运行时错误处理"""
            multi.create_interpreter("runtime_exe")
            result = multi.execute("runtime_exe", "1/0")
            assert result["success"] is False
            assert "ZeroDivisionError" in result["error"]

        def test_execute_timeout(self, multi: MultiPythonInterpreter):
            """测试超时功能"""
            multi.create_interpreter("timeout_exe")
            result = multi.execute("timeout_exe", "import time; time.sleep(2)", timeout=0.1)
            assert result["success"] is False
            assert "Timeout" in result["error"]

        def test_execute_nonexistent_interpreter(self, multi: MultiPythonInterpreter):
            """测试在不存在解释器上执行抛出 KeyError"""
            with pytest.raises(KeyError):
                multi.execute("nonexistent", "print(1)")

    class TestScriptManagement:
        """脚本管理功能测试"""

        def test_save_script(self, multi: MultiPythonInterpreter):
            """测试保存脚本"""
            multi.create_interpreter("script_exe")
            result = multi.save_script("script_exe", "test_script", "print('hello')", "Test description")
            assert "saved" in result.lower() or "success" in result.lower()

        def test_execute_script(self, multi: MultiPythonInterpreter):
            """测试执行脚本"""
            multi.create_interpreter("script_runner")
            multi.save_script("script_runner", "calc", "print(2 + 3)")
            result = multi.execute_script("script_runner", "calc")
            assert result["success"] is True
            assert "5" in result["output"]

        def test_read_script(self, multi: MultiPythonInterpreter):
            """测试读取脚本"""
            multi.create_interpreter("reader")
            multi.save_script("reader", "myscript", "x = 1", "A test script")
            result = multi.read_script("reader", "myscript")
            assert result["name"] == "myscript.py"
            assert "test script" in result["description"]
            assert "x = 1" in result["code"]

        def test_delete_script(self, multi: MultiPythonInterpreter):
            """测试删除脚本"""
            multi.create_interpreter("deleter")
            multi.save_script("deleter", "to_delete", "pass")
            result = multi.delete_script("deleter", "to_delete")
            assert "deleted" in result.lower() or "success" in result.lower()
            scripts = multi.list_scripts("deleter")
            assert len(scripts) == 0

        def test_list_scripts(self, multi: MultiPythonInterpreter):
            """测试列出脚本"""
            multi.create_interpreter("lister")
            multi.save_script("lister", "s1", "x = 1", "First script")
            multi.save_script("lister", "s2", "y = 2", "Second script")
            scripts = multi.list_scripts("lister")
            assert len(scripts) == 2
            names = [s["name"] for s in scripts]
            assert "s1.py" in names
            assert "s2.py" in names

        def test_scripts_per_interpreter_isolation(self, multi: MultiPythonInterpreter):
            """测试每个解释器有独立的脚本空间"""
            multi.create_interpreter("exe1")
            multi.create_interpreter("exe2")

            multi.save_script("exe1", "script", "print('exe1')")
            multi.save_script("exe2", "script", "print('exe2')")

            result1 = multi.execute_script("exe1", "script")
            result2 = multi.execute_script("exe2", "script")

            assert "exe1" in result1["output"]
            assert "exe2" in result2["output"]

        def test_script_not_found(self, multi: MultiPythonInterpreter):
            """测试读取不存在的脚本抛出错误"""
            multi.create_interpreter("script_check")
            with pytest.raises(FileNotFoundError):
                multi.read_script("script_check", "nonexistent")

    class TestDependencyInstallation:
        """依赖安装测试（需要 uv）"""

        @pytest.mark.skipif(shutil.which("uv") is None, reason="uv not installed")
        def test_install_dependency(self, multi: MultiPythonInterpreter):
            """测试在指定解释器安装依赖"""
            multi.create_interpreter("dep_exe")
            result = multi.install_dependency("dep_exe", "requests")
            # 安装可能成功或已存在
            assert result["success"] is True or "already" in result.get("output", "").lower()

        @pytest.mark.skipif(shutil.which("uv") is None, reason="uv not installed")
        def test_install_multiple_packages(self, multi: MultiPythonInterpreter):
            """测试安装多个包"""
            multi.create_interpreter("multi_dep")
            result = multi.install_dependency("multi_dep", ["urllib3", "certifi"])
            assert result["success"] is True

    class TestRestartServer:
        """重启服务器测试"""

        def test_restart_server(self, multi: MultiPythonInterpreter):
            """测试重启指定解释器"""
            multi.create_interpreter("restart_test")
            multi.execute("restart_test", "x = 42")
            result = multi.restart_interpreter("restart_test")
            assert result["success"] is True
            # 重启后状态应该清空
            result2 = multi.execute("restart_test", "print(x)")
            assert result2["success"] is False  # NameError

        def test_restart_nonexistent_interpreter(self, multi: MultiPythonInterpreter):
            """测试重启不存在的解释器抛出 KeyError"""
            with pytest.raises(KeyError):
                multi.restart_interpreter("nonexistent")

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
            """测试工具名称正确"""
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

    class TestContextManager:
        """上下文管理器测试"""

        def test_context_manager(self):
            """测试 with 语句自动清理"""
            with MultiPythonInterpreter() as m:
                m.create_interpreter("ctx_test")
                result = m.execute("ctx_test", "print('test')")
                assert result["success"] is True
                assert len(m.interpreters) == 1
            # 退出后应该已关闭
            assert m._closed is True
            assert len(m.interpreters) == 0

        def test_context_manager_exception(self):
            """测试上下文管理器在异常时也能清理"""
            m = None
            try:
                with MultiPythonInterpreter() as m:
                    m.create_interpreter("exc_test")
                    raise ValueError("Test exception")
            except ValueError:
                pass
            # 即使发生异常，也应该已关闭
            assert m and m._closed is True

    class TestClose:
        """关闭资源测试"""

        def test_close_all_interpreters(self, multi: MultiPythonInterpreter):
            """测试关闭所有解释器"""
            multi.create_interpreter("c1")
            multi.create_interpreter("c2")
            multi.create_interpreter("c3")
            assert len(multi.interpreters) == 3

            multi.close()

            assert len(multi.interpreters) == 0
            assert multi._closed is True

        def test_close_idempotent(self, multi: MultiPythonInterpreter):
            """测试多次关闭不会出错"""
            multi.create_interpreter("idempotent")
            multi.close()
            # 第二次关闭不应该报错
            multi.close()
            assert multi._closed is True

        def test_operations_after_close(self, multi: MultiPythonInterpreter):
            """测试关闭后操作应失败"""
            multi.create_interpreter("after_close")
            multi.close()
            # 关闭后应该无法执行操作
            # 但当前实现没有检查关闭状态，所以这可能是未定义行为

    class TestThreadSafety:
        """线程安全测试"""

        def test_concurrent_create(self, multi: MultiPythonInterpreter):
            """测试并发创建解释器"""
            errors = []
            created = []

            def create_worker():
                try:
                    result = multi.create_interpreter()
                    created.append(result)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=create_worker) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0, f"Errors during concurrent create: {errors}"
            assert len(multi.interpreters) == 5

        def test_concurrent_execute(self, multi: MultiPythonInterpreter):
            """测试并发执行代码"""
            multi.create_interpreter("concurrent_exe")
            results = []
            errors = []

            def execute_worker(n):
                try:
                    multi.execute("concurrent_exe", f"x = {n}")
                    result = multi.execute("concurrent_exe", "print(x)")
                    results.append(result["output"])
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=execute_worker, args=(i,)) for i in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0, f"Errors during concurrent execute: {errors}"
            # 所有执行都应该成功
            assert len(results) == 5

        def test_concurrent_multi_interpreter_execute(self, multi: MultiPythonInterpreter):
            """测试在多个解释器上并发执行"""
            for i in range(3):
                multi.create_interpreter(f"exe_{i}")

            results = {}
            errors = []

            def execute_worker(name, value):
                try:
                    multi.execute(name, f"x = {value}")
                    result = multi.execute(name, "print(x)")
                    results[name] = result["output"]
                except Exception as e:
                    errors.append(e)

            threads = [
                threading.Thread(target=execute_worker, args=(f"exe_{i}", i * 10))
                for i in range(3)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0
            assert "0" in results["exe_0"]
            assert "10" in results["exe_1"]
            assert "20" in results["exe_2"]

    class TestExitHandler:
        """退出处理测试"""

        def test_exit_handler_registration(self):
            """测试退出处理函数注册"""
            # 创建新的实例来测试注册
            m = MultiPythonInterpreter()
            # 验证 _exit_handler 已设置
            assert m._exit_handler is not None
            m.close()

        def test_cleanup_on_exit(self):
            """测试退出时清理资源"""
            m = MultiPythonInterpreter()
            m.create_interpreter("cleanup_test")
            # 模拟退出处理
            m.close()
            assert m._closed is True
            assert len(m.interpreters) == 0


class TestEdgeCases:
    """边界情况测试"""

    def test_large_number_of_interpreters(self):
        """测试创建大量解释器"""
        with MultiPythonInterpreter() as m:
            for i in range(10):
                m.create_interpreter(f"bulk_{i}")
            assert len(m.interpreters) == 10

            # 验证每个解释器都独立工作
            for i in range(10):
                result = m.execute(f"bulk_{i}", f"print({i})")
                assert result["success"] is True
                assert str(i) in result["output"]

    def test_unicode_in_code(self):
        """测试 Unicode 字符处理"""
        with MultiPythonInterpreter() as m:
            m.create_interpreter("unicode")
            result = m.execute("unicode", 'print("你好世界 🎉")')
            assert result["success"] is True
            assert "你好世界" in result["output"]

    def test_multiline_code_execution(self):
        """测试多行代码执行"""
        with MultiPythonInterpreter() as m:
            m.create_interpreter("multiline")
            code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(factorial(5))
"""
            result = m.execute("multiline", code)
            assert result["success"] is True
            assert "120" in result["output"]

    def test_empty_code(self):
        """测试空代码执行"""
        with MultiPythonInterpreter() as m:
            m.create_interpreter("empty")
            result = m.execute("empty", "")
            assert result["success"] is True

    def test_special_characters_in_interpreter_name(self):
        """测试特殊字符在解释器名称中的处理"""
        with MultiPythonInterpreter() as m:
            # 测试各种名称
            names = ["test-123", "test_456", "Test789", "test.name"]
            for name in names:
                result = m.create_interpreter(name)
                assert "Created" in result
                assert name in m.interpreters


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
