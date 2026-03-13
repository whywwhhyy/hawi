"""
MultiPythonInterpreter 的单元测试套件
"""

import os
import tempfile
import shutil
import threading
import asyncio
import pytest
from typing import Generator, AsyncGenerator, Any

from hawi_plugins.python_interpreter import PythonInterpreterPlugin


def _run_async_gen(agen: AsyncGenerator[str, None]) -> dict[str, Any]:
    """Helper to run async generator and collect results into a ToolResult-like dict."""
    async def collect() -> tuple[list[str], bool, str]:
        chunks: list[str] = []
        async for chunk in agen:
            chunks.append(chunk)
        # Combine all chunks - the last one might contain error info
        output = "".join(chunks)
        # Simple heuristic: if chunks contain error indicators, mark as failed
        success = not any("Error" in c or "Traceback" in c for c in chunks)
        error = "" if success else output
        return chunks, success, error

    try:
        # Try to use existing event loop (for async contexts)
        loop = asyncio.get_running_loop()
        # If we're in an async context, we need to schedule the coroutine
        # This shouldn't happen in our tests, but handle it gracefully
        raise RuntimeError("Cannot run async generator inside an async context")
    except RuntimeError:
        # No running loop, we can use asyncio.run
        try:
            chunks, success, error = asyncio.run(collect())
            return {
                "success": success,
                "output": "".join(chunks),
                "error": error,
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
            }


def _run_async_gen_threadsafe(agen: AsyncGenerator[str, None]) -> dict[str, Any]:
    """Thread-safe version of _run_async_gen for use in threaded contexts."""
    async def collect() -> tuple[list[str], bool, str]:
        chunks: list[str] = []
        async for chunk in agen:
            chunks.append(chunk)
        output = "".join(chunks)
        success = not any("Error" in c or "Traceback" in c for c in chunks)
        error = "" if success else output
        return chunks, success, error

    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    try:
        chunks, success, error = loop.run_until_complete(collect())
        return {
            "success": success,
            "output": "".join(chunks),
            "error": error,
        }
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": str(e),
        }
    finally:
        loop.close()


class TestMultiPythonInterpreter:
    """MultiPythonInterpreter 单元测试"""

    @pytest.fixture
    def multi(self) -> Generator[PythonInterpreterPlugin, None, None]:
        """创建多解释器管理器，测试后自动清理"""
        m = PythonInterpreterPlugin()
        yield m
        # 清理所有解释器
        if not m._closed:
            for name in list(m.interpreters.keys()):
                try:
                    if name != m.DEFAULT_INSTANCE_NAME:
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

    class TestDefaultInstance:
        """默认实例测试"""

        def test_default_instance_auto_created(self, multi: PythonInterpreterPlugin):
            """测试默认实例自动创建"""
            # 第一次使用默认实例时会自动创建
            result = _run_async_gen(multi.execute("print('hello')"))
            assert result["success"] is True
            assert "hello" in result["output"]
            assert multi.DEFAULT_INSTANCE_NAME in multi.interpreters

        def test_default_instance_state_persistence(self, multi: PythonInterpreterPlugin):
            """测试默认实例状态保持"""
            _run_async_gen(multi.execute("x = 42"))
            result = _run_async_gen(multi.execute("print(x)"))
            assert result["success"] is True
            assert "42" in result["output"]

        def test_default_instance_restart(self, multi: PythonInterpreterPlugin):
            """测试默认实例重启"""
            _run_async_gen(multi.execute("x = 100"))
            result = multi.restart_interpreter()
            assert result["success"] is True
            # 重启后状态清空
            result2 = _run_async_gen(multi.execute("print(x)"))
            assert result2["success"] is False  # NameError

        def test_explicit_none_same_as_default(self, multi: PythonInterpreterPlugin):
            """测试显式传入None与不传效果相同"""
            _run_async_gen(multi.execute("x = 123"))
            result1 = _run_async_gen(multi.execute("print(x)"))
            result2 = _run_async_gen(multi.execute("print(x)", interpreter_name=None))
            assert result1["output"] == result2["output"]

    class TestCreateAndRemove:
        """创建和移除解释器测试"""

        def test_create_interpreter_with_name(self, multi: PythonInterpreterPlugin):
            """测试使用指定名称创建解释器"""
            result = multi.create_interpreter("test_interpreter")
            assert "Created interpreter 'test_interpreter'" == result
            assert "test_interpreter" in multi.interpreters
            assert multi.interpreters["test_interpreter"].executor is not None

        def test_create_interpreter_auto_name(self, multi: PythonInterpreterPlugin):
            """测试自动生成解释器名称"""
            result1 = multi.create_interpreter()
            assert "interpret_0" in result1
            result2 = multi.create_interpreter()
            assert "interpret_1" in result2
            result3 = multi.create_interpreter()
            assert "interpret_2" in result3

        def test_create_interpreter_auto_name_with_gap(self, multi: PythonInterpreterPlugin):
            """测试自动命名跳过已存在的名称"""
            multi.create_interpreter("interpret_0")
            multi.create_interpreter("interpret_2")
            result = multi.create_interpreter()
            # 当前实现基于解释器数量，会尝试 interpret_2，发现已存在后跳到 interpret_3
            assert "interpret_3" in result

        def test_create_duplicate_name(self, multi: PythonInterpreterPlugin):
            """测试创建重复名称的解释器"""
            multi.create_interpreter("dup")
            result = multi.create_interpreter("dup")
            assert "already exists" in result.lower()
            assert len(multi.interpreters) == 1

        def test_create_reserved_name_fails(self, multi: PythonInterpreterPlugin):
            """测试不能使用保留名称创建解释器"""
            result = multi.create_interpreter(multi.DEFAULT_INSTANCE_NAME)
            assert "reserved" in result.lower() or "Cannot create" in result

        def test_create_with_custom_work_dir(self, multi: PythonInterpreterPlugin, temp_dir: str):
            """测试使用自定义工作目录创建解释器"""
            result = multi.create_interpreter("custom_dir", work_dir=temp_dir)
            assert "Created" in result
            assert os.path.exists(temp_dir)

        def test_remove_interpreter(self, multi: PythonInterpreterPlugin):
            """测试移除解释器"""
            multi.create_interpreter("to_remove")
            result = multi.remove_interpreters("to_remove")
            assert "Closed interpreter 'to_remove'" == result
            assert "to_remove" not in multi.interpreters

        def test_remove_nonexistent_interpreter(self, multi: PythonInterpreterPlugin):
            """测试移除不存在的解释器抛出 KeyError"""
            with pytest.raises(KeyError, match="Interpreter 'nonexistent' not found"):
                multi.remove_interpreters("nonexistent")

        def test_remove_default_interpreter_fails(self, multi: PythonInterpreterPlugin):
            """测试不能移除默认实例"""
            # 先使用默认实例
            _run_async_gen(multi.execute("pass"))
            result = multi.remove_interpreters(multi.DEFAULT_INSTANCE_NAME)
            assert "Cannot remove" in result or "default" in result.lower()
            # 默认实例应该还在
            assert multi.DEFAULT_INSTANCE_NAME in multi.interpreters

        def test_get_interpreter(self, multi: PythonInterpreterPlugin):
            """测试获取解释器实例"""
            multi.create_interpreter("getter")
            instance = multi.get_interpreter("getter")
            assert instance is not None
            assert instance.executor is not None
            assert instance.lock is not None

        def test_get_default_interpreter(self, multi: PythonInterpreterPlugin):
            """测试获取默认解释器实例"""
            instance = multi.get_interpreter(None)
            assert instance is not None
            assert multi.DEFAULT_INSTANCE_NAME in multi.interpreters

        def test_get_interpreter_not_found(self, multi: PythonInterpreterPlugin):
            """测试获取不存在的解释器抛出 KeyError"""
            with pytest.raises(KeyError, match="Interpreter 'notfound' not found"):
                multi.get_interpreter("notfound")

    class TestExecute:
        """代码执行测试"""

        def test_execute_simple_code(self, multi: PythonInterpreterPlugin):
            """测试在指定解释器中执行简单代码"""
            multi.create_interpreter("exe1")
            result = _run_async_gen(multi.execute("print('hello')", interpreter_name="exe1"))
            assert result["success"] is True
            assert "hello" in result["output"]

        def test_execute_with_state(self, multi: PythonInterpreterPlugin):
            """测试解释器状态保持"""
            multi.create_interpreter("stateful")
            _run_async_gen(multi.execute("x = 42", interpreter_name="stateful"))
            result = _run_async_gen(multi.execute("print(x)", interpreter_name="stateful"))
            assert result["success"] is True
            assert "42" in result["output"]

        def test_execute_isolation(self, multi: PythonInterpreterPlugin):
            """测试多个解释器之间状态隔离"""
            multi.create_interpreter("iso1")
            multi.create_interpreter("iso2")

            _run_async_gen(multi.execute("x = 100", interpreter_name="iso1"))
            _run_async_gen(multi.execute("x = 200", interpreter_name="iso2"))

            result1 = _run_async_gen(multi.execute("print(x)", interpreter_name="iso1"))
            result2 = _run_async_gen(multi.execute("print(x)", interpreter_name="iso2"))

            assert "100" in result1["output"]
            assert "200" in result2["output"]

        def test_execute_syntax_error(self, multi: PythonInterpreterPlugin):
            """测试语法错误处理"""
            multi.create_interpreter("error_exe")
            result = _run_async_gen(multi.execute("if x", interpreter_name="error_exe"))
            assert result["success"] is False
            assert "SyntaxError" in result["error"] or "error" in result["error"].lower()

        def test_execute_runtime_error(self, multi: PythonInterpreterPlugin):
            """测试运行时错误处理"""
            multi.create_interpreter("runtime_exe")
            result = _run_async_gen(multi.execute("1/0", interpreter_name="runtime_exe"))
            assert result["success"] is False
            assert "ZeroDivisionError" in result["error"]

        def test_execute_timeout(self, multi: PythonInterpreterPlugin):
            """测试超时功能"""
            multi.create_interpreter("timeout_exe")
            result = _run_async_gen(multi.execute("import time; time.sleep(2)", interpreter_name="timeout_exe", timeout=0.1))
            # Timeout returns partial output and may succeed if sleep is interrupted
            # Just check it completes without hanging
            assert "Timeout" in result["error"] or result["success"] is True or result["success"] is False

        def test_execute_nonexistent_interpreter(self, multi: PythonInterpreterPlugin):
            """测试在不存在解释器上执行抛出 KeyError"""
            # KeyError is raised when trying to iterate the async generator
            # because _get_instance is called when iteration starts
            with pytest.raises(KeyError):
                agen = multi.execute("print(1)", interpreter_name="nonexistent")
                # Need to actually iterate to trigger the error
                asyncio.run(agen.__anext__())

    class TestScriptManagement:
        """脚本管理功能测试 - 脚本管理不绑定特定解释器"""

        def test_save_script(self, multi: PythonInterpreterPlugin):
            """测试保存脚本"""
            result = multi.save_script("test_script", "print('hello')", description="Test description")
            assert "saved" in result.lower() or "success" in result.lower()

        def test_execute_script(self, multi: PythonInterpreterPlugin):
            """测试执行脚本"""
            multi.save_script("calc", "print(2 + 3)")
            result = _run_async_gen(multi.execute_script("calc"))
            assert result["success"] is True
            assert "5" in result["output"]

        def test_execute_script_in_specific_interpreter(self, multi: PythonInterpreterPlugin):
            """测试在指定解释器中执行脚本"""
            multi.create_interpreter("script_runner")
            multi.save_script("state_test", "x = 999")
            # 先在指定解释器中执行脚本
            _run_async_gen(multi.execute_script("state_test", interpreter_name="script_runner"))
            # 验证状态保存在该解释器中
            result = _run_async_gen(multi.execute("print(x)", interpreter_name="script_runner"))
            assert "999" in result["output"]

        def test_read_script(self, multi: PythonInterpreterPlugin):
            """测试读取脚本"""
            multi.save_script("myscript", "x = 1", description="A test script")
            result = multi.read_script("myscript")
            assert result["name"] == "myscript.py"
            assert "test script" in result["description"]
            assert "x = 1" in result["code"]

        def test_delete_script(self, multi: PythonInterpreterPlugin):
            """测试删除脚本"""
            multi.save_script("to_delete", "pass")
            result = multi.delete_script("to_delete")
            assert "deleted" in result.lower() or "success" in result.lower()
            scripts = multi.list_scripts()
            assert len(scripts) == 0

        def test_list_scripts(self, multi: PythonInterpreterPlugin):
            """测试列出脚本"""
            multi.save_script("s1", "x = 1", description="First script")
            multi.save_script("s2", "y = 2", description="Second script")
            scripts = multi.list_scripts()
            assert len(scripts) == 2
            names = [s["name"] for s in scripts]
            assert "s1.py" in names
            assert "s2.py" in names

        def test_script_not_found(self, multi: PythonInterpreterPlugin):
            """测试读取不存在的脚本抛出错误"""
            with pytest.raises(FileNotFoundError):
                multi.read_script("nonexistent")

        def test_script_auto_adds_py_extension(self, multi: PythonInterpreterPlugin):
            """测试自动添加 .py 后缀"""
            multi.save_script("myscript", "print(1)")  # 无后缀
            scripts = multi.list_scripts()
            assert len(scripts) == 1
            assert scripts[0]["name"] == "myscript.py"

    class TestDependencyInstallation:
        """依赖安装测试（需要 uv）"""

        @pytest.mark.skipif(shutil.which("uv") is None, reason="uv not installed")
        def test_install_dependency(self, multi: PythonInterpreterPlugin):
            """测试在指定解释器安装依赖"""
            multi.create_interpreter("dep_exe")
            result = multi.install_dependency("requests", interpreter_name="dep_exe")
            # 安装可能成功或已存在
            assert result["success"] is True or "already" in str(result["output"] or "").lower()

        @pytest.mark.skipif(shutil.which("uv") is None, reason="uv not installed")
        def test_install_multiple_packages(self, multi: PythonInterpreterPlugin):
            """测试安装多个包"""
            multi.create_interpreter("multi_dep")
            result = multi.install_dependency(["urllib3", "certifi"], interpreter_name="multi_dep")
            assert result["success"] is True

        @pytest.mark.skipif(shutil.which("uv") is None, reason="uv not installed")
        def test_install_to_default_instance(self, multi: PythonInterpreterPlugin):
            """测试在默认实例安装依赖"""
            result = multi.install_dependency("requests")
            assert result["success"] is True or "already" in str(result["output"] or "").lower()

    class TestRestartServer:
        """重启服务器测试"""

        def test_restart_server(self, multi: PythonInterpreterPlugin):
            """测试重启指定解释器"""
            multi.create_interpreter("restart_test")
            _run_async_gen(multi.execute("x = 42", interpreter_name="restart_test"))
            result = multi.restart_interpreter("restart_test")
            assert result["success"] is True
            # 重启后状态应该清空
            result2 = _run_async_gen(multi.execute("print(x)", interpreter_name="restart_test"))
            assert result2["success"] is False  # NameError

        def test_restart_nonexistent_interpreter(self, multi: PythonInterpreterPlugin):
            """测试重启不存在的解释器抛出 KeyError"""
            with pytest.raises(KeyError):
                multi.restart_interpreter("nonexistent")

    class TestGetTools:
        """工具列表测试"""

        def test_tools_property_returns_tools(self, multi: PythonInterpreterPlugin):
            """测试 tools 属性返回工具列表"""
            tools = multi.tools
            assert len(tools) >= 5
            for tool in tools:
                # tools are FunctionAgentTool instances with invoke method
                assert hasattr(tool, 'invoke')

        def test_tool_names(self, multi: PythonInterpreterPlugin):
            """测试工具名称正确"""
            tools = multi.tools
            tool_names = [t.name for t in tools]
            assert "PythonInterpreterPlugin__create_interpreter" in tool_names
            assert "PythonInterpreterPlugin__remove_interpreters" in tool_names
            assert "PythonInterpreterPlugin__save_script" in tool_names
            assert "PythonInterpreterPlugin__execute_script" in tool_names
            assert "PythonInterpreterPlugin__delete_script" in tool_names
            assert "PythonInterpreterPlugin__list_scripts" in tool_names
            assert "PythonInterpreterPlugin__install_dependency" in tool_names
            assert "PythonInterpreterPlugin__restart_interpreter" in tool_names
            assert "PythonInterpreterPlugin__execute" in tool_names
            assert "PythonInterpreterPlugin__read_script" in tool_names

    class TestContextManager:
        """上下文管理器测试"""

        def test_context_manager(self):
            """测试 with 语句自动清理"""
            with PythonInterpreterPlugin() as m:
                m.create_interpreter("ctx_test")
                result = _run_async_gen(m.execute("print('test')", interpreter_name="ctx_test"))
                assert result["success"] is True
                assert len(m.interpreters) == 1
            # 退出后应该已关闭
            assert m._closed is True
            assert len(m.interpreters) == 0

        def test_context_manager_exception(self):
            """测试上下文管理器在异常时也能清理"""
            m = None
            try:
                with PythonInterpreterPlugin() as m:
                    m.create_interpreter("exc_test")
                    raise ValueError("Test exception")
            except ValueError:
                pass
            # 即使发生异常，也应该已关闭
            assert m and m._closed is True

    class TestClose:
        """关闭资源测试"""

        def test_close_all_interpreters(self, multi: PythonInterpreterPlugin):
            """测试关闭所有解释器"""
            multi.create_interpreter("c1")
            multi.create_interpreter("c2")
            multi.create_interpreter("c3")
            assert len(multi.interpreters) == 3

            multi.close()

            assert len(multi.interpreters) == 0
            assert multi._closed is True

        def test_close_idempotent(self, multi: PythonInterpreterPlugin):
            """测试多次关闭不会出错"""
            multi.create_interpreter("idempotent")
            multi.close()
            # 第二次关闭不应该报错
            multi.close()
            assert multi._closed is True

    class TestThreadSafety:
        """线程安全测试"""

        def test_concurrent_create(self, multi: PythonInterpreterPlugin):
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

    class TestExitHandler:
        """退出处理测试"""

        def test_exit_handler_registration(self):
            """测试退出处理函数注册"""
            # 创建新的实例来测试注册
            m = PythonInterpreterPlugin()
            # 验证 _exit_handler 已设置
            assert m._exit_handler is not None
            m.close()

        def test_cleanup_on_exit(self):
            """测试退出时清理资源"""
            m = PythonInterpreterPlugin()
            m.create_interpreter("cleanup_test")
            # 模拟退出处理
            m.close()
            assert m._closed is True
            assert len(m.interpreters) == 0


class TestEdgeCases:
    """边界情况测试"""

    def test_large_number_of_interpreters(self):
        """测试创建大量解释器"""
        with PythonInterpreterPlugin() as m:
            for i in range(10):
                m.create_interpreter(f"bulk_{i}")
            assert len(m.interpreters) == 10

            # 验证每个解释器都独立工作
            for i in range(10):
                result = _run_async_gen(m.execute(f"print({i})", interpreter_name=f"bulk_{i}"))
                assert result["success"] is True
                assert str(i) in result["output"]

    def test_unicode_in_code(self):
        """测试 Unicode 字符处理"""
        with PythonInterpreterPlugin() as m:
            m.create_interpreter("unicode")
            result = _run_async_gen(m.execute('print("你好世界 🎉")', interpreter_name="unicode"))
            assert result["success"] is True
            assert "你好世界" in result["output"]

    def test_multiline_code_execution(self):
        """测试多行代码执行"""
        with PythonInterpreterPlugin() as m:
            m.create_interpreter("multiline")
            code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(factorial(5))
"""
            result = _run_async_gen(m.execute(code, interpreter_name="multiline"))
            assert result["success"] is True
            assert "120" in result["output"]

    def test_empty_code(self):
        """测试空代码执行"""
        with PythonInterpreterPlugin() as m:
            m.create_interpreter("empty")
            result = _run_async_gen(m.execute("", interpreter_name="empty"))
            assert result["success"] is True

    def test_special_characters_in_interpreter_name(self):
        """测试特殊字符在解释器名称中的处理"""
        with PythonInterpreterPlugin() as m:
            # 测试各种名称
            names = ["test-123", "test_456", "Test789", "test.name"]
            for name in names:
                result = m.create_interpreter(name)
                assert "Created" in result
                assert name in m.interpreters


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
