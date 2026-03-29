# 模型适配器使用指南

Hawi 提供统一的模型接口，支持多种 LLM 提供商和 API 格式。

## 支持的模型

| 提供商 | 模型 | OpenAI API | Anthropic API |
|--------|------|------------|---------------|
| DeepSeek | deepseek-chat | ✅ | ✅ |
| DeepSeek | deepseek-reasoner | ✅ | ✅ |
| Kimi | kimi-k2-5 | ✅ | ✅ |
| Kimi | kimi-latest | ✅ | ✅ |
| MiniMax | MiniMax-M2.7 | ✅ | ✅ |

## 快速开始

### DeepSeek

```python
from hawi.models import DeepSeekModel

# 自动检测 API 类型（默认 OpenAI）
model = DeepSeekModel(
    model_id="deepseek-chat",
    api_key="your-api-key"
)

# 强制使用 Anthropic API 格式
model = DeepSeekModel(
    model_id="deepseek-reasoner",
    api_key="your-api-key",
    api="anthropic"  # 或 "openai"
)
```

### Kimi

```python
from hawi.models import KimiModel

# OpenAI API 格式
model = KimiModel(
    model_id="kimi-k2-5",
    api_key="your-api-key",
    api="openai"
)

# Anthropic API 格式
model = KimiModel(
    model_id="kimi-latest",
    api_key="your-api-key",
    api="anthropic"
)
```

### MiniMax

```python
from hawi.models import MiniMaxModel

# 自动检测 API 类型（默认 OpenAI）
model = MiniMaxModel(
    model_id="MiniMax-M2.5",
    api_key="your-api-key"
)

# 强制使用 Anthropic API 格式
model = MiniMaxModel(
    model_id="MiniMax-M2.5",
    api_key="your-api-key",
    api="anthropic"
)
```

### 直接使用适配器类

```python
from hawi.models import (
    DeepSeekOpenAIModel,
    DeepSeekAnthropicModel,
    KimiOpenAIModel,
    KimiAnthropicModel,
    MiniMaxOpenAIModel,
    MiniMaxAnthropicModel,
)

# 直接使用具体实现
model = DeepSeekOpenAIModel(
    model_id="deepseek-chat",
    api_key="your-api-key"
)
```

## 高级配置

### 自定义 Base URL

```python
model = DeepSeekModel(
    model_id="deepseek-chat",
    api_key="your-api-key",
    base_url="https://custom-proxy.example.com/v1"
)
```

### 超时和重试

```python
model = DeepSeekModel(
    model_id="deepseek-chat",
    api_key="your-api-key",
    timeout=60.0,       # 请求超时（秒）
    max_retries=3,      # 最大重试次数
)
```

### 生成参数

```python
model = DeepSeekModel(
    model_id="deepseek-chat",
    api_key="your-api-key",
    temperature=0.7,        # 温度
    max_output_tokens=4096, # 最大输出 token 数
    top_p=0.9,             # 核采样
)
```

### Thinking 模式 (Anthropic 兼容 API)

部分模型支持 Thinking 模式（深度思考），可以通过 `thinking_budget` 参数控制 token 预算：

```python
# Anthropic 格式的模型
model = KimiModel(
    model_id="kimi-k2.5",
    api_key="your-api-key",
    api="anthropic",
    thinking_budget=8000,  # Thinking token 预算，0 或 None 表示禁用
    max_output_tokens=8192,
)

# 请求级覆盖
request = MessageRequest(
    messages=[...],
    thinking_budget=16000,  # 覆盖实例级设置
)
```

## 使用模型

### 同步调用

```python
from hawi.models import MessageRequest, Message

request = MessageRequest(
    messages=[
        Message(role="user", content=[{"type": "text", "text": "Hello!"}])
    ],
    tools=None,
    system=None
)

response = model.invoke(request)
print(response.message["content"][0]["text"])
```

### 流式调用

```python
for event in model.stream(request):
    if event.type == "content_block_delta":
        print(event.delta, end="", flush=True)
```

### 异步调用

```python
response = await model.ainvoke(request)

# 异步流式
async for event in model.astream(request):
    print(event)
```

## 特殊功能

### Reasoning 内容（DeepSeek）

```python
model = DeepSeekModel(model_id="deepseek-reasoner")
response = model.invoke(request)

# 获取 reasoning 内容
for part in response.message["content"]:
    if part["type"] == "reasoning":
        print(f"思考过程: {part['reasoning']}")
```

### 余额查询（DeepSeek）

```python
balance = model.get_balance()
for info in balance:
    print(f"{info.currency}: {info.available}")
```

### 工具调用

```python
from hawi.models import ToolDefinition

tools = [
    ToolDefinition(
        name="calculator",
        description="计算数学表达式",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string"}
            },
            "required": ["expression"]
        }
    )
]

request = MessageRequest(
    messages=[...],
    tools=tools,
    system=None
)

response = model.invoke(request)
# 检查是否有工具调用
for part in response.message["content"]:
    if part["type"] == "tool_call":
        print(f"调用工具: {part['name']}")
```

## 错误处理

```python
from hawi.errors import ModelErrorType

try:
    response = model.invoke(request)
except Exception as e:
    error_type = model.classify_error(e)

    if error_type == ModelErrorType.NETWORK:
        print("网络错误，请检查连接")
    elif error_type == ModelErrorType.THROTTLE:
        print("请求过于频繁，请稍后重试")
    elif error_type == ModelErrorType.DENIED:
        print("请求被拒绝，请检查 API Key")
    else:
        print(f"未知错误: {e}")
```

## 与 Agent 集成

```python
from hawi.agent import HawiAgent

model = DeepSeekModel(model_id="deepseek-chat")
agent = HawiAgent(model=model)

result = agent.run("Hello!")
```

## Strands 适配器

使用 Strands 框架的模型：

```python
from hawi.agent.models import StrandsModel
from strands import Agent

strands_agent = Agent(model="claude-3-5-sonnet")
model = StrandsModel(strands_agent)

agent = HawiAgent(model=model)
```

## 模型注册表 (ModelRegistry)

Hawi 提供模型注册表功能，支持通过名字和参数字典动态创建模型实例。

### 基本用法

```python
from hawi.models import model_registry, get_global_registry

# 使用模块级全局单例（预置所有内置模型）
model = model_registry.create("DeepSeekModel", {"model_id": "deepseek-chat", "api_key": "..."})

# 或获取全局注册表实例
registry = get_global_registry()
model = registry.create("OpenAIModel", {"model_id": "gpt-4o", "api_key": "..."})
```

### 创建独立注册表

```python
from hawi.models import ModelRegistry

# 创建空的独立注册表
registry = ModelRegistry()
registry.register(DeepSeekModel)
registry.register(OpenAIModel, alias="gpt")
model = registry.create("gpt", {"model_id": "gpt-4o", "api_key": "..."})
```

### 别名管理

```python
# 注册时指定别名
registry.register(DeepSeekModel, aliases=["deepseek", "ds"])

# 动态添加别名
registry.alias("OpenAIModel", "openai")
registry.alias("OpenAIModel", "gpt")

# 检查是否已注册
if "deepseek" in registry:
    print("DeepSeek is registered")
```

### 默认参数

```python
# 为特定模型设置全局默认参数
registry.set_defaults("DeepSeekModel", {"temperature": 0.7, "max_tokens": 2048})

# 创建实例时会自动合并默认参数（传入参数优先级更高）
model = registry.create("DeepSeekModel", {"model_id": "deepseek-chat"})
# 等同于: DeepSeekModel(temperature=0.7, max_tokens=2048, model_id="deepseek-chat", ...)

# 获取默认参数
defaults = registry.get_defaults("DeepSeekModel")

# 清除默认参数
registry.clear_defaults("DeepSeekModel")  # 清除特定模型
registry.clear_defaults()  # 清除所有默认参数
```

### 列出已注册的模型

```python
# 列出所有已注册的模型类
models = registry.list_models()
print(list(models.keys()))  # ['OpenAIModel', 'AnthropicModel', 'DeepSeekModel', ...]

# 列出所有别名
aliases = registry.list_aliases()
print(aliases)  # {'gpt': 'OpenAIModel', 'deepseek': 'DeepSeekModel', ...}
```

### 获取 Factory 参数

使用 `get_factory_arguments()` 方法可以获取已注册 Factory 的配置参数：

```python
from hawi.models import model_registry

# 获取原始参数（不包含继承的参数）
args = model_registry.get_factory_arguments("my-model")
print(args)  # {'model_id': 'my-model'}

# 获取展开后的完整参数（包含从 parent/template 继承的参数）
full_args = model_registry.get_factory_arguments("my-model", expanded=True)
print(full_args)  # {'model_id': 'my-model', 'api_key': '...', 'temperature': 0.7}
```

这对于调试配置或查看继承链展开后的最终参数非常有用。

### 对象池模式 (obtain_model)
+++++++


`obtain_model()` 是 `create()` 的增强版本，支持实例复用（对象池模式）。推荐使用此方法获取模型实例。

#### 设计原理

- **异步调用可复用**: 所有 Model 实现都支持单线程异步并发（`ainvoke`/`astream`），因此可以安全复用同一实例
- **同步调用需独占**: 同步调用（`invoke`/`stream`）会阻塞事件循环，如果复用实例会导致其他任务等待
- **性能与安全权衡**: 异步复用减少连接开销，同步隔离避免阻塞问题

#### 基本用法

```python
# 获取异步实例（默认，可复用）
model = registry.obtain_model("deepseek-openai", {"model_id": "deepseek-chat"})

# 再次调用返回同一实例（节省连接资源）
model2 = registry.obtain_model("deepseek-openai", {"model_id": "deepseek-chat"})
assert model is model2  # True

# 获取同步实例（独占新实例）
model3 = registry.obtain_model("deepseek-openai", {"model_id": "deepseek-chat"}, async_only=False)
assert model is not model3  # True
```

#### 异步调用场景（推荐）

```python
# 获取可复用实例
model = registry.obtain_model("deepseek-openai", {"model_id": "deepseek-chat"})

# 多个异步调用共享同一实例
response1 = await model.ainvoke(request1)
response2 = await model.ainvoke(request2)  # 并发执行，不会互相阻塞
```

#### 同步调用场景

```python
# 获取独占实例
model = registry.obtain_model("deepseek-openai", {"model_id": "deepseek-chat"}, async_only=False)

# 同步调用不会阻塞其他异步任务
response = model.invoke(request)
```

#### 对象池管理

```python
# 查看对象池信息
pool_info = registry.get_pool_info()
print(pool_info)  # {'size': 2, 'keys': [...]}

# 释放特定实例（从对象池移除）
registry.release_model("deepseek-openai", {"model_id": "deepseek-chat"})

# 清空整个对象池
registry.clear_pool()
```

#### 完整示例

```python
from hawi.models import model_registry

# 配置已在 apikey.yaml 中设置，只需指定 model_id
model = model_registry.obtain_model("deepseek-openai", {"model_id": "deepseek-chat"})

# 使用模型
from hawi.models import MessageRequest, Message

request = MessageRequest(
    messages=[Message(role="user", content=[{"type": "text", "text": "Hello!"}])]
)

# 异步调用
response = await model.ainvoke(request)
print(response.message["content"][0]["text"])

# 查看对象池状态
print(f"Pool size: {model_registry.get_pool_info()['size']}")  # 1
```
