# 生命周期管理

Hawi 提供强大的生命周期管理功能，确保资源在任何情况下都能正确清理。

## ExitHandler

`ExitHandler` 是一个全局单例，提供多层退出处理保证。

### 特性

- **多层保证**：使用 atexit、信号处理、异常钩子等多种机制
- **优先级执行**：按优先级顺序执行清理函数（数字越小越早执行）
- **线程安全**：所有操作都是线程安全的
- **上下文管理器支持**：支持 `with` 语句进行作用域清理

### 基本用法

```python
from hawi.utils import ExitHandler

# 获取全局实例
handler = ExitHandler.get_instance()

# 注册清理函数
handler.register(lambda: print("清理资源..."))

# 带优先级的注册（数字越小越早执行）
handler.register(cleanup_db, priority=1)
handler.register(cleanup_cache, priority=2)
```

### 装饰器用法

```python
from hawi.utils import ExitHandler

handler = ExitHandler.get_instance()

@handler.register(priority=1)
def cleanup_temp_files():
    """清理临时文件（高优先级）"""
    import shutil
    shutil.rmtree("/tmp/myapp", ignore_errors=True)

@handler.register(priority=100)
def log_shutdown():
    """记录关闭日志（低优先级）"""
    logging.info("应用程序已关闭")
```

### 上下文管理器

```python
from hawi.utils import exit_scope

with exit_scope():
    # 在此作用域内工作
    do_something()
    # 退出作用域时自动执行所有注册的清理函数
```

### 模块级便捷函数

```python
from hawi.utils import register_exit_handler, execute_early_and_clear

# 直接注册（无需获取实例）
@register_exit_handler(priority=1)
def my_cleanup():
    pass

# 提前执行并清除（用于测试或手动触发）
execute_early_and_clear()
```

## 应用场景

### 1. 清理临时资源

```python
from hawi.utils import register_exit_handler
import tempfile
import shutil

# 创建临时目录
temp_dir = tempfile.mkdtemp()

@register_exit_handler(priority=1)
def cleanup_temp():
    """确保临时目录被删除"""
    shutil.rmtree(temp_dir, ignore_errors=True)
```

### 2. 关闭数据库连接

```python
from hawi.utils import ExitHandler

class DatabaseManager:
    def __init__(self):
        self.connection = None
        self._register_cleanup()
    
    def connect(self):
        self.connection = create_connection()
    
    def _register_cleanup(self):
        handler = ExitHandler.get_instance()
        # 使用 weakref 确保对象被回收时也能清理
        handler.register_weakref(
            self,
            lambda: self.close(),
            name="db_cleanup"
        )
    
    def close(self):
        if self.connection:
            self.connection.close()
            self.connection = None
```

### 3. 插件资源清理

```python
from hawi.plugin import HawiPlugin
from hawi.plugin.decorators import hook
from hawi.utils import ExitHandler

class ResourcePlugin(HawiPlugin):
    def __init__(self):
        self.resources = []
        handler = ExitHandler.get_instance()
        handler.register(self.cleanup, priority=10)
    
    def acquire_resource(self, resource):
        self.resources.append(resource)
    
    def cleanup(self):
        """插件退出时清理资源"""
        for resource in self.resources:
            resource.release()
        self.resources.clear()
    
    @hook("after_conversation")
    async def on_conversation_end(self, agent):
        """会话结束时的清理"""
        # 可以在这里进行会话级别的清理
        pass
```

## 执行机制

ExitHandler 使用以下机制确保清理函数被执行：

1. **atexit**：标准 Python 退出处理
2. **信号处理**：捕获 SIGTERM、SIGINT 进行优雅关闭
3. **异常钩子**：未捕获异常时执行清理
4. **上下文管理器**：显式作用域控制
5. **weakref.finalize**：对象回收时的后备机制

## 最佳实践

### 1. 优先级设置

```python
# 关键资源优先清理（低优先级数字）
handler.register(critical_cleanup, priority=1)

# 非关键资源后清理
handler.register(metrics_flush, priority=100)
```

### 2. 错误处理

```python
def safe_cleanup():
    """清理函数应该捕获异常，避免影响其他清理"""
    try:
        risky_operation()
    except Exception as e:
        logging.error(f"清理失败: {e}")

handler.register(safe_cleanup)
```

### 3. 测试中的使用

```python
import pytest
from hawi.utils import ExitHandler, execute_early_and_clear

@pytest.fixture(autouse=True)
def clean_exit_handlers():
    """测试后清理所有 exit handler"""
    yield
    execute_early_and_clear()
```

## API 参考

### ExitHandler

```python
class ExitHandler:
    @classmethod
    def get_instance(cls) -> ExitHandler:
        """获取全局单例实例"""
    
    def register(
        self,
        func: Callable | None = None,
        *,
        priority: int = 100,
        name: str | None = None,
    ) -> Callable:
        """注册清理函数，可用作装饰器"""
    
    def register_at_exit(
        self,
        func: Callable,
        priority: int = 100,
        name: str | None = None,
    ) -> None:
        """显式注册（非装饰器方式）"""
    
    def register_weakref(
        self,
        obj: Any,
        func: Callable,
        name: str | None = None,
    ) -> weakref.ref:
        """绑定到对象生命周期的清理"""
    
    def execute_early_and_clear(self) -> list[Any]:
        """立即执行所有清理并清除"""
    
    def execute_and_keep(self) -> list[Any]:
        """执行但不清除（仍可触发）"""
    
    def clear(self) -> int:
        """清除所有清理函数（不执行）"""
```

### 便捷函数

```python
def register_exit_handler(
    func: Callable | None = None,
    *,
    priority: int = 100,
    name: str | None = None,
) -> Callable:
    """模块级注册函数"""

def execute_early_and_clear() -> list[Any]:
    """模块级提前执行"""

def clear_exit_handlers() -> int:
    """模块级清除"""

@contextmanager
def exit_scope():
    """作用域上下文管理器"""
```
