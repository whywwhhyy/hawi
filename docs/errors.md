# 错误处理指南

Hawi 提供结构化的异常体系，便于诊断和错误处理。

## 异常层次

```
HawiError (基类)
├── ConfigurationError      # 配置错误
├── ModelError             # 模型相关错误
│   ├── NetworkError       # 网络连接错误
│   ├── ThrottleError      # 速率限制 (429)
│   ├── DeniedError        # 访问被拒绝 (401/403)
│   ├── ValidationError    # 数据格式错误
│   └── UnknownModelError  # 未知模型
└── AgentError             # Agent 相关错误
    ├── MaxIterationsError # 超过最大迭代次数
    ├── ToolNotFoundError  # 工具未注册
    ├── ToolValidationError    # 工具参数验证失败
    └── ToolExecutionError     # 工具执行失败

Note: ValidationError (模型验证错误) 也是 ModelError 的子类
```

## 核心类

### HawiError

基础异常类，自动捕获调用栈：

```python
from hawi.errors import HawiError

try:
    # 操作
except HawiError as e:
    print(f"错误类型: {e.error_type}")
    print(f"错误信息: {e.message}")
    print(f"调用栈:\n{e.stack_trace}")
```

**属性：**

| 属性 | 类型 | 说明 |
|------|------|------|
| `error_type` | `ErrorType` | 错误类型标识 |
| `message` | `str \| None` | 错误消息 |
| `stack_trace` | `str` | 完整调用栈 |

### ErrorType 类型别名

```python
from hawi.errors import ErrorType, ModelErrorType, AgentErrorType

# ErrorType = ModelErrorType | AgentErrorType | 'configuration' | 'unknown'
error_type: ErrorType = "network"
```

## 使用示例

### 模型错误处理

```python
from hawi.errors import (
    ModelError, NetworkError, ThrottleError, DeniedError, ValidationError
)
from hawi.models import model_registry

model = model_registry.obtain_model("deepseek-openai", {"model_id": "deepseek-chat"})

try:
    response = await model.ainvoke(request)
except NetworkError as e:
    print(f"网络错误: {e.message}")
except ThrottleError as e:
    print(f"请求被限流，等待后重试")
except DeniedError as e:
    print(f"认证失败: {e.message}")
except ValidationError as e:
    print(f"请求格式错误: {e.message}")
except ModelError as e:
    print(f"模型错误 [{e.error_type}]: {e.message}")
```

### Agent 错误处理

```python
from hawi.errors import (
    AgentError, MaxIterationsError, ToolNotFoundError,
    ToolValidationError, ToolExecutionError
)
from hawi import HawiAgent

agent = HawiAgent(model=model)

try:
    result = agent.run("复杂的任务")
except MaxIterationsError as e:
    print(f"Agent 执行超过最大迭代次数限制")
except ToolNotFoundError as e:
    print(f"工具不存在: {e.message}")
except ToolValidationError as e:
    print(f"工具参数验证失败: {e.message}")
except ToolExecutionError as e:
    print(f"工具执行失败: {e.message}")
except AgentError as e:
    print(f"Agent 错误 [{e.error_type}]: {e.message}")
```

### 配置错误

```python
from hawi.errors import ConfigurationError

try:
    # 加载配置
    from hawi.models import load_config
    load_config("models.yaml")
except ConfigurationError as e:
    print(f"配置错误: {e.message}")
```

## 错误栈追踪

### get_error_stack 函数

```python
from hawi.errors import get_error_stack

try:
    # 操作
except Exception as e:
    stack = get_error_stack(e)
    print(f"调用栈:\n{stack}")
```

## 最佳实践

### 1. 按类型捕获

```python
# ✅ 推荐：按具体类型捕获
try:
    result = agent.run(prompt)
except MaxIterationsError:
    print("任务太复杂，考虑简化")
except ToolExecutionError as e:
    print(f"工具执行失败: {e.message}")

# ❌ 不推荐：只捕获基类
try:
    result = agent.run(prompt)
except Exception as e:
    print("出错了")
```

### 2. 保留上下文信息

```python
try:
    result = agent.run(prompt)
except AgentError as e:
    # 添加上下文信息
    logger.error(f"Agent 执行失败 [prompt={prompt[:50]}]: {e.message}")
    raise
```

### 3. 转换为用户友好的错误

```python
from hawi.errors import HawiError

def run_with_user_friendly_errors(agent, prompt):
    try:
        return agent.run(prompt)
    except MaxIterationsError:
        return "任务太复杂，我需要更多步骤来完成。"
    except ToolNotFoundError as e:
        return f"我没有找到 '{e.message}' 这个工具。"
    except NetworkError:
        return "网络连接失败，请检查网络后重试。"
    except HawiError as e:
        return f"发生错误: {e.message}"
