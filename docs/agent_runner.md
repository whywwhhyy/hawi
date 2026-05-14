# AgentRunner 运行器

## 概述

AgentRunner 是 HawiAgent 的运行驱动层，支持复杂消息处理和编排。主要面向两个场景：

1. **Always-on Background Agent** - 持续运行、等待消息的守护 Agent
2. **Multi-agent System** - 多 Agent 协作、消息路由

### 解决的问题

- HawiAgent 的 `run()`/`arun()` 是完整生命周期，无法自然地"持续运行"
- 没有消息队列机制
- 没有工具调用中断能力
- 事件系统无法被拦截/转换

## 快速开始

```python
from hawi.agent import HawiAgent, AgentRunner, QueueType
from hawi.models import model_registry

# 创建 Agent
model = model_registry.create_model("deepseek-chat")
agent = HawiAgent(model=model)

# 创建运行器
runner = AgentRunner(agent)

# 消息入队（默认普通队列）
runner.enqueue("请介绍一下 Python")
runner.enqueue("[重要] 这个问题更优先", "high_prio")
runner.enqueue("[紧急] 立即停止当前任务", "urgent")

# 启动守护循环
import asyncio
await runner.run_forever(poll_interval=0.5)
```

## 三种消息队列

### QueueType 枚举

| 队列 | 优先级 | 处理时机 | 合并规则 |
|------|--------|----------|----------|
| `URGENT` | 3 (最高) | 立即打断当前执行 | 单槽位设计，只保留最新 |
| `HIGH_PRIO` | 2 | Tool Call 结束后检查 | 智能合并：tool_result 合并 / 插队 |
| `NORMAL` | 1 (最低) | 仅 Agent Idle 时 | 不合并 |

### 队列语义详解

**URGENT（紧急队列）**：
- 单槽位设计，不使用队列，只保留最新的一条
- 入队时立即触发中断，清除所有正在执行的工具调用
- 适用场景："取消当前操作"、"切换到新任务"

```python
# URGENT 会立即打断当前执行
runner.enqueue("停止当前任务，立即回答这个问题", "urgent")
```

**HIGH_PRIO（高优先级队列）**：
- Tool Call 结束时分两种情况处理：
  - 上一条消息是 tool_result → 合并到 tool_result
  - 上一条消息是用户消息 → 插队到 NORMAL 队首

```python
# HIGH_PRIO 会智能合并
runner.enqueue("请先回答这个问题", "high_prio")
```

**NORMAL（普通队列）**：
- 仅在 Agent 完全空闲时执行
- 按入队顺序处理

```python
# 普通消息入队
runner.enqueue("帮我查一下天气")
```

## 原子操作 API

```python
# 入队消息，返回 message_id
msg_id = runner.enqueue("content", "normal")

# 移除指定消息
runner.remove_message(msg_id)

# 清空指定队列
runner.clear_queue("normal")

# 清空所有队列
runner.clear_all_queues()

# 获取队列长度
lengths = runner.get_queue_lengths()
# {'normal': 0, 'high_prio': 0, 'urgent': 0}
```

## 事件系统

AgentRunner 产生以下事件类型：

```python
from hawi.events import (
    AgentRunnerEnqueueEvent,   # 消息入队
    AgentRunnerDequeueEvent,  # 消息出队
    AgentRunnerInterruptEvent, # 运行器打断
    AgentInterruptEvent,    # Agent 被请求打断
)
```

### 订阅 AgentRunner 事件

```python
from hawi.events import EventBus, AgentRunnerEnqueueEvent

bus = EventBus()

@bus.subscribe
async def on_enqueue(event: AgentRunnerEnqueueEvent):
    print(f"消息入队: {event.message_id}, 队列: {event.queue_type}")

runner.subscribe(bus)
```

## 错误处理

AgentRunner 提供三种错误处理 Hook：

```python
from hawi.agent.runner import (
    ModelErrorHook,
    AgentErrorHook,
    AgentRunnerErrorHook,
    ErrorAction,
)

# Model 错误处理
class MyModelHook:
    async def on_model_error(self, error, context):
        print(f"Model 错误: {error}")
        return ErrorAction.CONTINUE  # 继续处理下一条消息

# Agent 错误处理
class MyAgentHook:
    async def on_agent_error(self, error, message, context):
        return ErrorAction.RETRY  # 重试当前消息

runner.set_model_error_hook(MyModelHook())
runner.set_agent_error_hook(MyAgentHook())
```

### ErrorAction 选项

| 选项 | 说明 |
|------|------|
| `RETRY` | 重试当前消息 |
| `ABORT` | 终止 AgentRunner |
| `CONTINUE` | 记录错误，继续处理下一条消息（默认） |

## EventInterceptor - 事件拦截

EventInterceptor 是 AgentRunner 内部使用的事件处理组件，可以拦截、转换或阻止事件传播。当前公开 API 主要通过 `AgentRunner` 的事件输出消费这些事件；如需开放外部注册入口，应先补一层稳定的 runner 方法，而不是直接依赖内部字段。

```python
from hawi.agent.runner import EventMode

# EventMode 定义内部处理结果：透传、拦截、转换、丢弃。
```

### EventMode

| 模式 | 说明 |
|------|------|
| `PASS_THROUGH` | 透传（默认） |
| `INTERCEPT` | 拦截（不转发） |
| `REPROCESS` | 转换后转发 |
| `SUPPRESS` | 丢弃 |

## 状态机

AgentRunner 内部维护以下状态：

```
IDLE ──(enqueue)──▶ READY ──(execute)──▶ RUNNING
  ▲                                  │
  │                                  ▼
  │                            INTERRUPTING
  │                                  │
  └──────────────────────────────────┘
```

| 状态 | 说明 |
|------|------|
| `IDLE` | 等待消息 |
| `READY` | 有消息，检查队列优先级 |
| `RUNNING` | 正常执行 |
| `INTERRUPTING` | 主动触发中断 |

## 与 Plugin Manager 集成

AgentRunner 不直接定义额外的 plugin hook；插件仍通过 `HawiAgent` 的 hook 与 tool 机制影响单次 run。Runner 层只负责把队列消息驱动成 agent run，并把 `runner.*` 事件暴露给观察者和持久化层。

## 使用示例

### Always-on Agent

```python
import asyncio
from hawi.agent import HawiAgent, AgentRunner

async def main():
    agent = HawiAgent(model=model)
    runner = AgentRunner(agent)
    
    # 启动守护循环
    await runner.run_forever(poll_interval=0.5)

# 后台运行
asyncio.create_task(main())

# 从外部入队消息
runner.enqueue("处理这个任务", "normal")
```

将 `main()` 放入任意脚本后，可用 `uv run python <script>.py` 运行。

## API 参考

### AgentRunner

```python
class AgentRunner:
    def __init__(
        self,
        agent: HawiAgent,
    ): ...

    # 消息队列操作
    def enqueue(
        self,
        content: str | list[ContentPart],
        queue: Literal["normal", "high_prio", "urgent"] = "normal",
        metadata: dict | None = None,
    ) -> str: ...  # 返回 message_id

    def remove_message(self, message_id: str) -> bool: ...
    def clear_queue(self, queue: QueueType) -> int: ...
    def clear_all_queues(self) -> dict[QueueType, int]: ...
    def get_queue_lengths(self) -> dict[str, int]: ...

    # 生命周期
    async def run_forever(self, poll_interval: float = 0.1): ...
    def stop(): ...

    # 错误处理
    def set_model_error_hook(self, hook: ModelErrorHook): ...
    def set_agent_error_hook(self, hook: AgentErrorHook): ...
    def set_runner_error_hook(self, hook: AgentRunnerErrorHook): ...
```

### QueuedMessage

```python
@dataclass
class QueuedMessage:
    id: str                    # UUID[:8]
    content: str | list[ContentPart]
    queue_type: QueueType
    created_at: float
    metadata: dict[str, Any]
    merged_tool_call_ids: list[str] = field(default_factory=list)
```

## 相关文档

- [Event 系统](./event_system.md) - 事件系统使用指南
- [Hook 系统](./hook_system.md) - 钩子系统使用指南
