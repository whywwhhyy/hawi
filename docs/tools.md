# 工具系统使用指南

Hawi 提供灵活的工具系统，支持类继承和函数装饰器两种方式定义工具。

## 概述

工具系统采用两层架构：

- **Agent Tool 层**: 实例化、LLM 可调用的工具
- **Base Access 层**: 基础设施客户端（Jira、Confluence 等）

## 定义工具

### 方式一：继承 AgentTool

```python
from hawi.tool import AgentTool, ToolResult

class CalculatorTool(AgentTool):
    """数学计算器工具"""

    name = "calculator"
    description = "计算数学表达式，支持 +, -, *, /, ** 等运算符"
    parameters_schema = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "数学表达式，如 '1 + 2 * 3'"
            }
        },
        "required": ["expression"]
    }

    # 可选：配置属性
    audit = False              # 是否需要人工审批
    timeout = 30.0             # 执行超时（秒）
    tags = ["math", "utility"] # 标签

    def run(self, expression: str) -> ToolResult:
        """同步执行"""
        try:
            result = eval(expression)
            return ToolResult(success=True, output={"result": result})
        except Exception as e:
            return ToolResult(success=False, error=str(e))
```

### 方式二：使用 @tool 装饰器

```python
from hawi.tool import tool

@tool()
def calculator(expression: str) -> dict:
    """
    计算数学表达式。

    Args:
        expression: 数学表达式，如 '1 + 2 * 3'

    Returns:
        包含计算结果的字典
    """
    return {"result": eval(expression)}

# 获取工具实例
calc_tool = calculator.to_tool()

# 或转换为插件
plugin = calculator.to_plugin()
```

### 异步工具

```python
from hawi.tool import AgentTool, ToolResult
import aiohttp

class WeatherTool(AgentTool):
    name = "weather"
    description = "获取城市天气"
    parameters_schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名称"}
        },
        "required": ["city"]
    }

    async def arun(self, city: str) -> ToolResult:
        """异步执行"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"https://api.weather.com/{city}") as resp:
                data = await resp.json()
                return ToolResult(success=True, output=data)
```

## 工具配置

### 需要审批的工具

```python
class DangerousTool(AgentTool):
    name = "execute_command"
    description = "执行系统命令"
    audit = True  # 需要人工审批

    def run(self, command: str) -> ToolResult:
        import subprocess
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return ToolResult(success=True, output={"stdout": result.stdout})
```

### 带超时的工具

```python
class SlowTool(AgentTool):
    name = "long_running_task"
    description = "耗时任务"
    timeout = 60.0  # 60秒超时

    def run(self, data: str) -> ToolResult:
        # 长时间运行的任务
        ...
```

## 使用工具

### 直接调用

```python
tool = CalculatorTool()

# 同步调用
result = tool.invoke({"expression": "1 + 2"})
if result.success:
    print(result.output)
else:
    print(f"错误: {result.error}")

# 异步调用
result = await tool.ainvoke({"expression": "1 + 2"})
```

### 与 Agent 集成

```python
from hawi.agent import HawiAgent

# 方式一：传入工具实例列表
tools = [CalculatorTool(), WeatherTool()]
agent = HawiAgent(model=model, tools=tools)

# 方式二：使用插件
from hawi_plugins.python_interpreter import PythonInterpreter

agent = HawiAgent(
    model=model,
    plugins=[PythonInterpreter()]
)

result = agent.run("计算 15 的平方根")
```

## 工具注册表

### 全局注册

```python
from hawi.tool import ToolRegistry

# 注册 Agent 工具
ToolRegistry.register_agent_tool(CalculatorTool())

# 获取工具
all_tools = ToolRegistry.get_agent_tools()  # 获取所有
calc = ToolRegistry.get_agent_tool("calculator")  # 获取特定
```

### Base 层注册

```python
# 注册基础设施客户端
ToolRegistry.register_base(
    name="jira",
    factory=lambda config: JiraClient(**config),
    default_config={"url": "https://jira.example.com"}
)

# 获取客户端
jira = ToolRegistry.get_base("jira", {"token": "xxx"})
```

## 工具结果

```python
from hawi.tool import ToolResult

# 成功结果
success_result = ToolResult(
    success=True,
    output={"result": 42, "details": "..."}
)

# 失败结果
error_result = ToolResult(
    success=False,
    error="参数格式错误"
)

# 包含图片的结果
image_result = ToolResult(
    success=True,
    output={
        "image_url": "https://example.com/image.png",
        "description": "生成的图片"
    }
)
```

## 高级示例

### 文件处理工具

```python
import os
from pathlib import Path

class FileTool(AgentTool):
    name = "file_operations"
    description = "文件读写操作"
    parameters_schema = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["read", "write", "list"],
                "description": "操作类型"
            },
            "path": {"type": "string", "description": "文件路径"},
            "content": {"type": "string", "description": "写入内容（write 操作）"}
        },
        "required": ["operation", "path"]
    }
    audit = True  # 文件操作需要审批

    def run(self, operation: str, path: str, content: str = None) -> ToolResult:
        try:
            if operation == "read":
                with open(path, 'r') as f:
                    return ToolResult(success=True, output={"content": f.read()})

            elif operation == "write":
                with open(path, 'w') as f:
                    f.write(content)
                return ToolResult(success=True, output={"message": "写入成功"})

            elif operation == "list":
                files = os.listdir(path)
                return ToolResult(success=True, output={"files": files})

        except Exception as e:
            return ToolResult(success=False, error=str(e))
```

### HTTP API 工具

```python
import requests
from typing import Literal

class APITool(AgentTool):
    name = "http_request"
    description = "发送 HTTP 请求"
    parameters_schema = {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "PUT", "DELETE"]
            },
            "url": {"type": "string"},
            "headers": {"type": "object"},
            "body": {"type": "object"}
        },
        "required": ["method", "url"]
    }

    def run(
        self,
        method: Literal["GET", "POST", "PUT", "DELETE"],
        url: str,
        headers: dict = None,
        body: dict = None
    ) -> ToolResult:
        try:
            resp = requests.request(method, url, headers=headers, json=body)
            return ToolResult(
                success=True,
                output={
                    "status": resp.status_code,
                    "headers": dict(resp.headers),
                    "body": resp.json() if resp.headers.get('content-type', '').startswith('application/json') else resp.text
                }
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))
```

### 数据库工具

```python
import sqlite3

class DatabaseTool(AgentTool):
    name = "database"
    description = "SQLite 数据库操作"
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "SQL 查询"},
            "params": {"type": "array", "description": "查询参数"}
        },
        "required": ["query"]
    }
    audit = True

    def __init__(self, db_path: str):
        self.db_path = db_path

    def run(self, query: str, params: list = None) -> ToolResult:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                if query.strip().upper().startswith("SELECT"):
                    rows = cursor.execute(query, params or []).fetchall()
                    return ToolResult(
                        success=True,
                        output={"rows": [dict(row) for row in rows]}
                    )
                else:
                    cursor.execute(query, params or [])
                    conn.commit()
                    return ToolResult(
                        success=True,
                        output={"affected_rows": cursor.rowcount}
                    )
        except Exception as e:
            return ToolResult(success=False, error=str(e))
```

## 最佳实践

1. **明确的描述**: 工具描述要清晰说明用途和参数
2. **参数验证**: 使用 JSON Schema 定义参数类型和约束
3. **错误处理**: 始终返回 ToolResult，包含详细的错误信息
4. **超时设置**: 为可能耗时的操作设置合理的超时
5. **敏感操作**: 对危险操作（文件、网络、执行）设置 audit=True
6. **文档完善**: 使用 docstring 说明工具功能和参数含义
