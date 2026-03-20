# Printer 系统

Printer 负责将 Agent 执行过程中的事件流格式化为可读的输出。

## 概述

Printer 系统采用策略模式，支持多种输出格式：

| Printer | 说明 | 适用场景 |
|---------|------|---------|
| `StreamingMarkdownPrinter` | 流式 Markdown 渲染，增量块级输出 | 终端 TTY 环境（默认） |
| `RichPrinter` | Rich 库格式化，支持颜色和布局 | 终端 TTY 环境 |
| `BlockPrinter` | 分块输出，流水线友好 | 非 TTY + 非流式 |
| `PlainPrinter` | 纯文本逐行输出 | 非 TTY + 流式 |
| `auto` | 自动检测环境选择 | 通用（默认） |

## 工厂函数

### create_printer

```python
from hawi.agent.printers import create_printer

printer = create_printer(
    printer_type="auto",     # "auto" | "rich" | "block" | "plain" | "streaming"
    streaming=False,
    show_reasoning=True,
    show_tools=True,
    show_errors=True,
    show_error_stack=True,
    max_arg_length=80,
    max_result_length=200,
    show_full_tool_content=True,
)
```

**自动检测逻辑：**

```python
is_tty = sys.stdout.isatty()

if printer_type == "auto":
    if is_tty:
        # 终端环境 → StreamingMarkdownPrinter（默认）
        actual = "streaming"
    elif streaming:
        # 非终端 + 流式 → PlainPrinter
        actual = "plain"
    else:
        # 非终端 + 非流式 → BlockPrinter
        actual = "block"
```

## StreamingMarkdownPrinter

流式 Markdown 渲染器，基于块级增量解析策略，大幅提升长文档流式渲染性能。

**核心特性：**
- 块级分割：识别 Markdown 块边界（空行分隔）
- 增量输出：完成的块立即输出，不等待流结束
- 动态更新：当前未完成块使用 Live 实时更新
- 自动清理：流结束时自动处理剩余内容

```python
from hawi.agent.printers import StreamingMarkdownPrinter

printer = StreamingMarkdownPrinter(
    show_reasoning=True,       # 显示 thinking 内容
    show_tools=True,           # 显示工具调用
    show_errors=True,          # 显示错误
    show_error_stack=True,     # 显示错误堆栈
    max_arg_length=80,         # 参数最大显示长度
    max_result_length=200,     # 结果最大显示长度
    show_full_tool_content=True,  # 显示完整工具内容
    console=None,              # 自定义 Console
    refresh_per_second=12.5,   # 刷新频率
)
```

**工作原理：**
1. 接收文本片段，累积到缓冲区
2. 识别块边界（双换行符 `\n\n` 分隔）
3. 已完成的块立即输出到终端
4. 当前未完成块使用 Rich Live 动态更新
5. 流结束时，停止 Live 并输出剩余内容

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

## RichPrinter

支持颜色、进度条等富文本格式，需要终端支持：

```python
from hawi.agent.printers import RichPrinter

printer = RichPrinter(
    show_reasoning=True,       # 显示 reasoning 内容
    show_tools=True,          # 显示工具调用
    show_errors=True,         # 显示错误
    reasoning_prefix="🤔 ",    # reasoning 前缀
    tool_call_prefix="🔧 ",   # 工具调用前缀
    tool_result_prefix="",    # 工具结果前缀
    error_prefix="❌ ",
    max_arg_length=80,        # 参数最大显示长度
    max_result_length=200,    # 结果最大显示长度
)
```

**输出示例：**

```
🤔 让我计算一下 1+1 的结果
🔧 execute({'code': '1+1'})
✓ execute (45ms): 2
答案是 2
```

## BlockPrinter

分块输出，适合日志和管道：

```python
from hawi.agent.printers import BlockPrinter

printer = BlockPrinter(
    show_reasoning=True,
    show_tools=True,
    show_errors=True,
    max_arg_length=80,
    max_result_length=200,
)
```

**输出示例：**

```
[reasoning]
让我计算一下...
[/reasoning]

[tool_call]
execute({'code': '1+1'})
[/tool_call]

[tool_result]
2
[/tool_result]

[response]
答案是 2
[/response]
```

## PlainPrinter

纯文本逐行输出，适合流式处理：

```python
from hawi.agent.printers import PlainPrinter

printer = PlainPrinter(
    show_reasoning=True,
    show_tools=True,
    show_errors=True,
)
```

## 在 Agent 中使用

### 方式一：直接使用

```python
import asyncio
from hawi import HawiAgent
from hawi.agent.printers import create_printer

async def main():
    printer = create_printer(streaming=True)
    agent = HawiAgent(model=model)

    async for event in agent.arun("讲个故事", stream=True):
        await printer.handle(event)

asyncio.run(main())
```

### 方式二：同步调用

```python
printer = create_printer(streaming=True)
agent = HawiAgent(model=model)

async def process(prompt):
    async for event in agent.arun(prompt, stream=True):
        await printer.handle(event)

asyncio.run(process("Hello"))
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
        # 从事件中获取 delta 文本
        delta = event.delta if hasattr(event, 'delta') else ""
        print(delta, end="", flush=True)

    async def _on_content_block_stop(self, event):
        print()  # 换行

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
