# Printer 系统

Printer 负责将 Agent 执行过程中的事件流格式化为可读的输出。

## 概述

Printer 系统采用策略模式，根据终端环境自动选择最佳输出方式：

| Printer | 说明 | 适用场景 |
|---------|------|---------|
| `RichPrinter` | 智能流式 Markdown 渲染器，支持代码块自定义样式 | 终端环境（默认） |
| `PlainPrinter` | 纯文本输出 | 非终端环境 |
| `BasePrinter` | 基类 | 自定义实现 |
| `auto` | 自动检测 | 通用（默认） |

## RichPrinter 工作模式

`RichPrinter` 支持两种工作模式，自动检测或手动指定：

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| **Streaming** | 实时动态更新当前块 | 标准终端、支持 ANSI |
| **Non-streaming** | 块确定后才打印 | 管道、文件、dumb 终端、CI |

**自动检测逻辑：**
- TTY 终端 → Streaming 模式
- dumb/unknown 终端 → Non-streaming 模式
- CI 环境 → Non-streaming 模式
- Jupyter/Notebook → Non-streaming 模式
- 管道/重定向 → PlainPrinter

## 代码块渲染

`RichPrinter` 使用 Rich 原生的 Markdown 渲染器处理代码块，支持语法高亮：

```python
from hawi.agent.printers import RichPrinter

printer = RichPrinter()
```

**效果示例：**

```python
def hello():
    print("Hello, World!")
```

## 工厂函数

### create_printer

```python
from hawi.agent.printers import create_printer

# 自动检测
printer = create_printer()

# 强制指定模式
printer = create_printer(streaming=True)   # 强制 streaming
printer = create_printer(streaming=False)  # 强制 non-streaming

# 完整控制
printer = create_printer(
    printer_type="auto",        # "auto" | "rich" | "plain"
    streaming=None,             # None=自动, True=streaming, False=non-streaming
    show_reasoning=True,
    show_tools=True,
    show_errors=True,
    show_error_stack=True,
    max_arg_length=80,
    max_result_length=200,
    show_full_tool_content=True,
)
```

**环境变量控制：**
```bash
# 强制 non-streaming 模式
HAWI_STREAMING=0 python my_agent.py

# 强制 streaming 模式
HAWI_STREAMING=1 python my_agent.py
```

## RichPrinter

智能流式 Markdown 渲染器，支持自定义代码块样式。

```python
from hawi.agent.printers import RichPrinter

# 自动检测模式
printer = RichPrinter()

# 完整参数
printer = RichPrinter(
    show_reasoning=True,            # 显示 thinking 内容
    show_tools=True,                # 显示工具调用
    show_errors=True,               # 显示错误
    show_error_stack=True,          # 显示错误堆栈
    max_arg_length=80,              # 参数最大显示长度
    max_result_length=200,          # 结果最大显示长度
    show_full_tool_content=True,    # 显示完整工具内容
    streaming=None,                 # None=自动, True=streaming, False=non-streaming
    console=None,                   # 自定义 Console
    refresh_per_second=12.5,        # Live 刷新频率
)
```

**Streaming 模式特性：**
- 块级分割：识别 Markdown 块边界（空行分隔）
- 增量输出：完成的块立即输出，不等待流结束
- 动态更新：当前未完成块使用 Live 实时更新
- 高性能：长文档不卡顿

**Non-streaming 模式特性：**
- 块确定后（双换行）才输出
- 无动态更新，适合管道/文件重定向
- 与 CI/CD 系统兼容

**Thinking 模式：**
当模型输出 thinking 内容时，会显示带边框的面板：

```
┌─────────────────────────────────┐
│ 🤔 Thinking                     │
├─────────────────────────────────┤
│ 让我逐步分析这个问题...          │
│                                 │
│ 第一步：理解问题                 │
└─────────────────────────────────┘
```

## PlainPrinter

纯文本输出，适用于非终端环境：

```python
from hawi.agent.printers import PlainPrinter

printer = PlainPrinter(
    show_reasoning=True,
    show_tools=True,
    show_errors=True,
)
```

## 在 Agent 中使用

### 自动模式（推荐）

```python
from hawi import HawiAgent
from hawi.agent.printers import create_printer

# 自动检测最佳 printer
printer = create_printer()
agent = HawiAgent(model=model)

for event in agent.run("讲个故事", stream=True):
    # 事件自动打印
    pass
```

### 强制指定模式

```python
# 强制 streaming 模式
printer = create_printer(streaming=True)

# 强制 non-streaming 模式
printer = create_printer(streaming=False)
```


### 非终端环境

```python
# 管道/文件重定向时自动使用 PlainPrinter
printer = create_printer()  # 自动检测为非 TTY，使用 PlainPrinter
```

## 事件处理

Printer 自动处理以下事件类型：

| 事件 | 处理 |
|------|------|
| `model.stream_start` | 流式响应开始 |
| `model.stream_stop` | 流式响应结束（含 usage） |
| `model.metadata` | 模型元数据（usage、latency 等） |
| `model.content_block_start` | 内容块开始 |
| `model.content_block_delta` | 内容块增量更新 |
| `model.content_block_stop` | 内容块结束 |
| `model.tool_use_block_start` | 工具调用块开始 |
| `model.tool_use_block_delta` | 工具调用块增量更新 |
| `model.tool_use_block_stop` | 工具调用块结束 |
| `agent.run_start` | Agent 执行开始 |
| `agent.run_stop` | Agent 执行结束 |
| `agent.tool_call` | 工具调用 |
| `agent.tool_result_part` | 工具结果分片（流式） |
| `agent.tool_result` | 工具执行完成 |
| `agent.error` | 错误信息 |

## 自定义 Printer

继承 `BasePrinter` 实现自定义格式化：

```python
from hawi.agent.printers.base import BasePrinter

class MyPrinter(BasePrinter):
    async def _on_content_block_start(self, event):
        print(f"[CONTENT] ", end="")

    async def _on_content_block_delta(self, event):
        delta = event.delta if hasattr(event, 'delta') else ""
        print(delta, end="", flush=True)

    async def _on_content_block_stop(self, event):
        print()

    async def _on_tool_use_block_start(self, event):
        tool_name = event.tool_name if hasattr(event, 'tool_name') else "unknown"
        print(f"\n>>> Calling {tool_name}")

    async def _on_tool_use_block_delta(self, event):
        delta = event.arguments_delta if hasattr(event, 'arguments_delta') else ""
        print(delta, end="", flush=True)

    async def _on_tool_use_block_stop(self, event):
        pass

    def _print_tool_result(self, tool_name, success, result_preview, duration, arguments=None):
        status = "✓" if success else "✗"
        print(f"\n<<< {status} {tool_name} ({duration:.0f}ms): {result_preview}")

    def _print_error(self, error):
        print(f"\n!!! Error: {error}")
```
