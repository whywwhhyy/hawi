# AgentContext 使用指南

`AgentContext` 负责管理 Agent 的对话状态，包括消息历史、工具定义和系统提示词。

## 概述

```python
from hawi import AgentContext

context = AgentContext(
    messages=[],                    # 对话历史
    tool_definitions=None,          # 工具定义列表
    system_prompt=None,             # 系统提示词
)
```

## 消息管理

### 添加消息

```python
# 添加用户消息
context.add_user_message("你好，请帮我计算 1+1")

# 添加助手消息
context.add_assistant_message([
    {"type": "text", "text": "结果是 2"}
])

# 添加工具结果
context.add_tool_result(
    tool_call_id="call_123",
    content="2",
    is_error=False
)

# 直接添加消息
context.add_message({
    "role": "user",
    "content": [{"type": "text", "text": "Hello"}],
    "name": None,
    "metadata": None
})
```

### 访问消息

```python
# 获取所有消息
messages = context.messages

# 获取最后一条消息
last_message = context.messages[-1]

# 获取助手回复的文本
for msg in context.messages:
    if msg["role"] == "assistant":
        for part in msg["content"]:
            if part["type"] == "text":
                print(part["text"])
```

## 系统提示词

```python
# 设置系统提示词（字符串）
context.set_system_prompt("你是一个有帮助的助手")

# 设置系统提示词（ContentPart 列表）
context.set_system_prompt([
    {"type": "text", "text": "你是一个编程助手"},
    {"type": "text", "text": "用中文回答"}
])

# 获取系统提示词
system_prompt = context.get_system_prompt()
```

## 上下文操作

### truncate - 截断消息

保留最后 N 条消息：

```python
# 只保留最后 10 条消息
context.truncate(keep_last=10)
```

### collapse - 折叠消息

将一段消息替换为摘要：

```python
# 将索引 0-5 的消息折叠为摘要
context.collapse(start=0, end=6, summary="用户询问了天气和时间")
```

### inject - 插入消息

在指定位置插入消息：

```python
# 在末尾追加（position=-1）
context.inject({"role": "user", "content": [...], ...})

# 在开头插入
context.inject({"role": "system", "content": [...], ...}, position=0)

# 在特定位置插入
context.inject(message, position=3)
```

### clear - 清空消息

```python
# 只清空消息，保留工具定义和系统提示词
context.clear()
```

### copy - 复制上下文

```python
# 创建深拷贝
backup = context.copy()
```

## 工具定义管理

```python
from hawi.models import ToolDefinition

# 设置工具定义
context.tool_definitions = [
    ToolDefinition(
        name="calculator",
        description="计算数学表达式",
        parameters={...}
    )
]

# 获取工具定义
tools = context.tool_definitions
```

## 构建请求

### prepare_request

从当前上下文构建 `MessageRequest`：

```python
request = context.prepare_request()
# 等同于:
# MessageRequest(
#     messages=context.messages.copy(),
#     system=context.system_prompt,
#     tools=context.tool_definitions,
# )
```

## 持久化

### save - 保存上下文

支持 Markdown 和 JSON 两种格式：

```python
# 保存为 Markdown（人类可读）
context.save("conversation.md", format="markdown")

# 保存为 JSON（完整状态，可恢复）
context.save("conversation.json", format="json")
```

**Markdown 格式示例：**

```markdown
# Agent Context History

*Saved at: 2024-01-15 10:30:00*

---

## System Prompt

你是一个有帮助的助手

---

## Available Tools

**Total tools:** 1

### `calculator`

**Description:** 计算数学表达式

**Parameters Schema:**
```json
{
  "type": "object",
  "properties": {
    "expression": {"type": "string"}
  }
}
```

---

## Conversation History

**Total messages:** 3

### Message 1: **USER**

Hello

---

### Message 2: **ASSISTANT**

**Tool Call:** `calculator`
- ID: `call_123`
- Arguments:
  - **expression**: `1+1`

**Tool Result** (ID: `call_123`):
```
2
```

**Tool Call:** `calculator`
- ID: `call_456`
- Arguments:
  - **expression**: `2+2`

**Tool Result** (ID: `call_456`):
```
4
```

### Message 3: **ASSISTANT**

结果是 4
```

### load - 加载上下文

从 JSON 文件恢复上下文状态：

```python
# 加载上下文
context.load("conversation.json")

# 会覆盖现有的 messages 和 system_prompt
# 工具定义和 cache_tool_definitions 保持不变
```

**注意：**
- 工具定义不会被加载，需要重新设置
- 仅支持 JSON 格式加载

## 工具调用审计

### PendingToolCall

待审批的工具调用：

```python
from hawi.tool.types import PendingToolCall

# 添加待审批的工具调用
pending = context._add_pending_tool_call(
    tool_call_id="call_789",
    tool_name="execute",
    arguments={"code": "rm -rf /"}
)

# 获取所有待审批的工具调用
pending_calls = context.get_pending_tool_calls()
```

### audit_pending_tool_calls - 审批工具调用

```python
# 审批通过
approved, rejected = context.audit_pending_tool_calls(
    approve=["call_123", "call_456"],
    reject=["call_789"]
)

# 清除所有待审批
context.clear_pending_tool_calls()
```

## ToolCallContext

工具执行时的运行时上下文。提供有界 API，通过属性访问 agent 内部能力，接口清晰、意图明确。

```python
from hawi.agent.context import ToolCallContext

# 在工具执行时注入
context.tool_call_context = ToolCallContext(agent=agent)

# 工具中可以访问
class MyTool(AgentTool):
    context = "ctx"  # 声明需要注入的参数名

    def run(self, **kwargs):
        ctx: ToolCallContext = kwargs["ctx"]
        
        # 访问对话上下文（消息历史、system prompt）
        messages = ctx.context.messages
        
        # 访问完整 agent（sub-agent 编排、动态工具注册等）
        agent = ctx.agent
```

### 属性说明

| 属性 | 类型 | 说明 |
|------|------|------|
| `context` | `AgentContext` | 对话上下文，包含消息历史、system prompt、历史操作等 |
| `agent` | `HawiAgent` | 完整 agent 实例，支持 sub-agent 编排、动态工具注册等 |

### 使用场景

```python
# 动态工具注册示例
class DynamicToolTool(AgentTool):
    name = "register_tool"
    description = "动态注册新工具"
    context = "ctx"  # 注入 ToolCallContext

    parameters_schema = {
        "type": "object",
        "properties": {
            "tool_name": {"type": "string"},
            "tool_code": {"type": "string"}
        },
        "required": ["tool_name", "tool_code"]
    }

    def run(self, tool_name: str, tool_code: str, ctx: ToolCallContext) -> ToolResult:
        # 访问 agent 动态注册工具
        new_tool = create_tool_from_code(tool_code)
        ctx.agent.add_tool(new_tool)
        return ToolResult(success=True, output=f"Registered {tool_name}")
```
