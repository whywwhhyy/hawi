# Agent 执行结果

Hawi 的执行结果系统提供了完整的对话历史、工具调用记录和 Token 使用统计。

## AgentRunResult

`AgentRunResult` 是 `agent.run()` 和 `agent.arun()` 的返回值，包含完整的执行状态。

### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `stop_reason` | `str` | 停止原因: `"end_turn"`, `"tool_use"`, `"max_iterations"`, `"error"` |
| `messages` | `list[Message]` | 完整的对话历史 |
| `response` | `Message \| None` | 最后一条助手消息 |
| `usage` | `TokenUsage \| None` | Token 使用统计 |
| `tool_calls` | `list[ToolCallRecord]` | 工具调用记录 |
| `error` | `str \| None` | 错误信息（如果 stop_reason 是 error） |

### 使用示例

```python
from hawi import HawiAgent

agent = HawiAgent(model=model)
result = agent.run("计算 1+1")

# 获取文本回复
text = result.text
print(text)

# 获取推理内容（如果是推理模型）
reasoning = result.reasoning_text
print(f"推理过程: {reasoning}")

# 检查停止原因
if result.stop_reason == "max_iterations":
    print("达到最大迭代次数")
elif result.stop_reason == "error":
    print(f"执行错误: {result.error}")

# 获取 Token 使用
if result.usage:
    print(f"输入: {result.usage['input_tokens']}")
    print(f"输出: {result.usage['output_tokens']}")
```

### 便捷属性

```python
# 直接获取文本内容（__str__ 方法的简写）
text = result.text

# 等同于:
text = str(result)

# 获取推理/思考内容
reasoning = result.reasoning_text
```

### 转换为字典

```python
# 序列化为字典（用于日志或存储）
data = result.to_dict()

# 包含的信息:
# - stop_reason: 停止原因
# - messages: 消息列表
# - response: 响应消息
# - usage: Token 使用统计
# - tool_calls: 工具调用记录
# - tool_calls_from_content: 从内容中提取的工具调用
# - error: 错误信息
```

## ToolCallRecord

`ToolCallRecord` 记录单次工具调用的完整信息。

### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `tool_name` | `str` | 工具名称 |
| `arguments` | `dict[str, Any]` | 调用参数 |
| `result` | `ToolResult` | 执行结果 |
| `duration_ms` | `float` | 执行耗时（毫秒） |
| `tool_call_id` | `str` | 工具调用唯一标识 |

### 访问工具调用记录

```python
result = agent.run("执行多个任务")

for record in result.tool_calls:
    print(f"工具: {record.tool_name}")
    print(f"参数: {record.arguments}")
    print(f"耗时: {record.duration_ms:.2f}ms")
    print(f"成功: {record.result.success}")
    print(f"输出: {record.result.output}")
```

## TokenUsage

Token 使用统计，包含输入和输出 Token 数。

### 结构

```python
{
    "input_tokens": int,        # 输入 Token 数
    "output_tokens": int,       # 输出 Token 数
    "cache_write_tokens": int | None,  # 缓存写入（可选）
    "cache_read_tokens": int | None,   # 缓存读取（可选）
}
```

### 使用示例

```python
result = agent.run("Hello")

if result.usage:
    usage = result.usage
    total = usage["input_tokens"] + usage["output_tokens"]
    print(f"总 Token 使用: {total}")
    
    # 缓存统计（如果支持）
    if usage.get("cache_read_tokens"):
        print(f"缓存命中: {usage['cache_read_tokens']} tokens")
```

## 完整示例

```python
import asyncio
from hawi import HawiAgent
from hawi.models import DeepSeekModel

async def main():
    model = DeepSeekModel(model_id="deepseek-chat")
    agent = HawiAgent(model=model, max_iterations=10)
    
    result = await agent.arun(
        "搜索 Python 最新版本信息，并总结主要特性"
    )
    
    # 检查结果
    print(f"停止原因: {result.stop_reason}")
    print(f"迭代次数: {len(result.tool_calls)}")
    
    # 输出回复
    print(f"\n回复:\n{result.text}")
    
    # 工具调用详情
    if result.tool_calls:
        print("\n工具调用记录:")
        for record in result.tool_calls:
            print(f"  - {record.tool_name}: {record.duration_ms:.0f}ms")
    
    # Token 使用
    if result.usage:
        usage = result.usage
        print(f"\nToken 使用:")
        print(f"  输入: {usage['input_tokens']}")
        print(f"  输出: {usage['output_tokens']}")
    
    # 保存完整结果
    import json
    with open("result.json", "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

asyncio.run(main())
```

## 错误处理

当执行出错时，结果会包含错误信息：

```python
result = agent.run("可能失败的任务")

if result.stop_reason == "error":
    print(f"执行失败: {result.error}")
    # 可以查看之前的消息了解上下文
    for msg in result.messages:
        print(f"{msg['role']}: {msg['content']}")
else:
    print(f"成功: {result.text}")
```

## 与上下文结合

执行结果可以用于继续对话：

```python
# 第一轮对话
result1 = agent.run("你好")

# 继续对话（使用同一 agent 的上下文）
result2 = agent.run("请继续")

# result2 包含了完整的对话历史
print(f"总消息数: {len(result2.messages)}")
```
