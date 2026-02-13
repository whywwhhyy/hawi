# Hawi 架构文档

## 架构概览

Hawi 采用分层架构设计，确保关注点分离和单向依赖。

```
┌─────────────────────────────────────────────────────────────────┐
│                        Builder 层                                │
│              配置驱动的 Agent 和工作流构建                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       Workflow 层                                │
│              工作流编排 - 流程控制和节点执行                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                         Agent 层                                 │
│         执行层 - LLM 交互、模型适配、工具调用                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  HawiAgent  │  │    Model    │  │    Event/Hook 系统       │  │
│  │             │  │   Adapters  │  │                         │  │
│  │ - 执行循环   │  │ - DeepSeek  │  │ - EventBus              │  │
│  │ - 工具调用   │  │ - Kimi      │  │ - Plugin Hooks          │  │
│  │ - 状态管理   │  │ - Anthropic │  │ - Streaming             │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                          Tool 层                                 │
│              工具抽象、注册表、函数工具包装                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  AgentTool  │  │ ToolRegistry│  │  FunctionAgentTool      │  │
│  │             │  │             │  │                         │  │
│  │ - 抽象接口   │  │ - 两层架构   │  │ - 函数包装器             │  │
│  │ - 执行方法   │  │ - 全局注册   │  │ - 自动 Schema           │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                         Utils 层                                 │
│         基础设施层 - 上下文、生命周期、终端 UI                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Context   │  │   Exit      │  │      Terminal           │  │
│  │   Manager   │  │   Handler   │  │                         │  │
│  │             │  │             │  │ - ConversationPrinter   │  │
│  │ - 线程安全   │  │ - 多层清理   │  │ - Rich 格式化           │  │
│  │ - 引用计数   │  │ - 信号处理   │  │ - 进度显示              │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 核心设计原则

### 1. 单向依赖

```
builder → workflow → agent → tool → utils
```

- 上层可以依赖下层
- 下层不能依赖上层
- 同层之间可以相互依赖

### 2. 依赖注入

通过构造函数注入依赖，便于测试和替换：

```python
class HawiAgent:
    def __init__(
        self,
        model: Model,                    # 必需依赖
        plugins: list[HawiPlugin] = None, # 可选依赖
        event_bus: EventBus = None,       # 可选依赖
    ):
        ...
```

### 3. 插件化设计

核心功能通过插件扩展：

```python
# 内置插件
PythonInterpreter        # Python 代码执行
MultiPythonInterpreter   # 多实例 Python

# 自定义插件
class MyPlugin(HawiPlugin):
    @hook("before_tool_calling")
    async def on_tool(self, agent, tool_name, arguments):
        # 干预工具调用
        ...
```

## 数据流

### Agent 执行流

```
┌─────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────┐
│  User   │────▶│   Agent     │────▶│    Model    │────▶│  LLM    │
│  Input  │     │   Start     │     │    Call     │     │  API    │
└─────────┘     └─────────────┘     └─────────────┘     └─────────┘
                                                        │
                              ┌─────────────────────────┘
                              ▼
┌─────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────┐
│  Final  │◀────│   Agent     │◀────│   Tool      │◀────│  Tool   │
│ Output  │     │   End       │     │   Results   │     │  Calls  │
└─────────┘     └─────────────┘     └─────────────┘     └─────────┘
```

### 事件流

```
Model Provider
      │
      ├──▶ Model Events ──▶ EventBus ──▶ Consumers (non-blocking)
      │
      ▼
   Agent
      │
      ├──▶ Agent Events ──▶ EventBus ──▶ Consumers (non-blocking)
      │
      └──▶ Hooks ─────────▶ Plugins ───▶ (blocking, mutable)
```

## 模块详解

### Agent 模块

```python
# 核心类
HawiAgent              # Agent 主类
AgentContext           # 对话上下文
Model (ABC)            # 模型抽象基类

# 事件系统
Event                  # 事件基类
EventBus               # 事件总线
ConversationPrinter    # 对话打印机

# 消息类型
Message                # 消息
ContentPart            # 内容片段 (TextPart, ToolCallPart, etc.)
MessageRequest         # 模型请求
MessageResponse        # 模型响应
```

### Model 适配器

```python
# 基类
Model                  # 抽象基类
OpenAIModel            # OpenAI API 基类
AnthropicModel         # Anthropic API 基类

# DeepSeek 实现
DeepSeekModel          # 工厂类
DeepSeekOpenAIModel    # OpenAI 格式
DeepSeekAnthropicModel # Anthropic 格式

# Kimi 实现
KimiModel              # 工厂类
KimiOpenAIModel        # OpenAI 格式
KimiAnthropicModel     # Anthropic 格式

# 适配器
StrandsModel           # Strands 框架适配器
```

### Tool 模块

```python
# 核心类
AgentTool (ABC)        # 工具抽象基类
ToolResult             # 工具结果
ToolRegistry           # 工具注册表

# 实现类
FunctionAgentTool      # 函数工具包装器

# 装饰器
@tool()                # 函数转工具装饰器
```

### Plugin 模块

```python
# 核心类
HawiPlugin             # 插件基类
PluginHooks            # Hook 类型定义

# 装饰器
@hook(name)            # Hook 装饰器
@before_session        # 会话前
@after_session         # 会话后
@before_model_call     # 模型调用前
@after_model_call      # 模型调用后
@before_tool_calling   # 工具调用前
@after_tool_calling    # 工具调用后
```

### Utils 模块

```python
# 上下文管理
ContextManager         # 线程安全上下文

# 生命周期
ExitHandler            # 退出处理器

# 终端 UI
ConversationPrinter    # 对话打印机
```

## 扩展点

### 1. 添加新的模型适配器

```python
from hawi.agent.model import Model

class MyModel(Model):
    @property
    def model_id(self) -> str:
        return "my-model"

    def _invoke_impl(self, request: MessageRequest) -> MessageResponse:
        # 实现同步调用
        ...

    def _stream_impl(self, request: MessageRequest) -> Iterator[StreamEvent]:
        # 实现流式调用
        ...
```

### 2. 添加新的工具

```python
from hawi.tool import AgentTool, ToolResult

class MyTool(AgentTool):
    name = "my_tool"
    description = "工具描述"
    parameters_schema = {...}

    def run(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output={...})
```

### 3. 添加新的插件

```python
from hawi.plugin import HawiPlugin
from hawi.plugin.decorators import hook

class MyPlugin(HawiPlugin):
    @hook("before_tool_calling")
    async def on_tool(self, agent, tool_name, arguments):
        # 干预工具调用
        ...
```

## 性能考虑

### 上下文隔离

- `ContextManager` 使用 `contextvars` 实现协程级别的上下文隔离
- 避免全局状态污染
- 支持上下文 fork 用于并发任务

### 流式处理

- 模型层支持原生流式 API
- 事件系统使用异步广播避免阻塞
- 支持增量内容更新

### 资源管理

- `ExitHandler` 提供多层清理保证
- `PythonInterpreter` 使用持久化子进程避免重复启动开销
- 工具调用支持超时控制

## 安全考虑

### 工具执行

- 工具支持 `audit` 标记需要人工审批
- 超时控制防止长时间运行
- 错误隔离防止单工具失败影响整个 Agent

### 上下文安全

- 上下文数据通过引用计数管理生命周期
- 敏感数据（API Key）不存储在上下文中

### 代码执行

- `PythonInterpreter` 在独立子进程中运行
- 支持脚本保存和复用
- 可以重启子进程清理状态
