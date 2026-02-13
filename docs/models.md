# 模型适配器使用指南

Hawi 提供统一的模型接口，支持多种 LLM 提供商和 API 格式。

## 支持的模型

| 提供商 | 模型 | OpenAI API | Anthropic API |
|--------|------|------------|---------------|
| DeepSeek | deepseek-chat | ✅ | ✅ |
| DeepSeek | deepseek-reasoner | ✅ | ✅ |
| Kimi | kimi-k2-5 | ✅ | ✅ |
| Kimi | kimi-latest | ✅ | ✅ |

## 快速开始

### DeepSeek

```python
from hawi.agent.models import DeepSeekModel

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
from hawi.agent.models import KimiModel

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

### 直接使用适配器类

```python
from hawi.agent.models import (
    DeepSeekOpenAIModel,
    DeepSeekAnthropicModel,
    KimiOpenAIModel,
    KimiAnthropicModel,
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
    temperature=0.7,    # 温度
    max_tokens=4096,    # 最大生成 token 数
    top_p=0.9,          # 核采样
)
```

## 使用模型

### 同步调用

```python
from hawi.agent.messages import MessageRequest, Message

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
from hawi.agent.messages import ToolDefinition

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
from hawi.agent.model import ModelErrorType

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
