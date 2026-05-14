# 快速入门指南

本指南帮助你在几分钟内上手 Hawi Agent 框架。

## 安装

```bash
# 使用 uv（推荐）
uv sync

# 或使用 pip
pip install -e .
```

## 配置 API Key

创建 `apikey.yaml` 文件（已添加到 `.gitignore`）：

```yaml
- name: deepseek
  apikey: sk-your-deepseek-key

- name: kimi-openai
  apikey: sk-your-kimi-key
```

或使用环境变量：

```bash
export DEEPSEEK_API_KEY="sk-your-key"
export KIMI_API_KEY="sk-your-key"
```

## 第一个 Agent

### 基础对话

```python
from hawi.agent import HawiAgent
from hawi.models import DeepSeekModel

# 创建模型
model = DeepSeekModel(model_id="deepseek-chat")

# 创建 Agent
agent = HawiAgent(model=model)

# 运行对话
result = agent.run("你好，请介绍一下自己")

# 获取最后一条消息的内容
last_message = result.messages[-1]
content = last_message["content"][0]["text"]
print(content)
```

### 流式输出

```python
from hawi.agent.printers import create_printer

agent = HawiAgent(model=model)
printer = create_printer(streaming=True)

# 流式执行
async for event in agent.arun("讲一个短故事", stream=True):
    await printer.handle(event)
```

## 添加工具

### 使用内置工具

```python
from hawi_plugins.python_interpreter import PythonInterpreter

# 创建带 Python 解释器的 Agent
interpreter = PythonInterpreter()
agent = HawiAgent(
    model=model,
    plugins=[interpreter]
)

# Agent 现在可以执行 Python 代码
result = agent.run("计算 15 的阶乘")
```

### 创建自定义工具

```python
from hawi.tool import tool

@tool()
def calculator(expression: str) -> float:
    """计算数学表达式。"""
    return eval(expression)

# 转换为插件
plugin = calculator.to_plugin()
agent = HawiAgent(model=model, plugins=[plugin])

result = agent.run("15 * 23 等于多少？")
```

## 使用插件

### 日志插件示例

```python
from hawi.plugin import HawiPlugin
from hawi.plugin.decorators import hook

class LoggingPlugin(HawiPlugin):
    @hook("before_tool_calling")
    async def on_tool(self, agent, tool_name, arguments):
        print(f"🔧 调用工具: {tool_name}")

    @hook("after_tool_calling")
    async def on_result(self, agent, tool_name, arguments, result):
        status = "✅" if result.success else "❌"
        print(f"{status} 工具执行完成")

# 使用插件
agent = HawiAgent(
    model=model,
    plugins=[LoggingPlugin(), interpreter]
)
```

## 切换模型

```python
from hawi.models import DeepSeekModel, KimiModel

# DeepSeek
deepseek = DeepSeekModel(model_id="deepseek-chat")
agent = HawiAgent(model=deepseek)

# Kimi
kimi = KimiModel(model_id="kimi-k2-5", api="openai")
agent = HawiAgent(model=kimi)

# DeepSeek Reasoner（支持推理）
reasoner = DeepSeekModel(model_id="deepseek-reasoner")
agent = HawiAgent(model=reasoner)
```

## 完整示例

```python
import asyncio
from hawi.agent import HawiAgent
from hawi.models import DeepSeekModel
from hawi.agent.printers import create_printer
from hawi.tool import tool
from hawi.plugin import HawiPlugin
from hawi.plugin.decorators import hook

# 1. 创建工具
@tool()
def search_web(query: str) -> dict:
    """搜索网络（模拟）"""
    return {"results": f"关于 '{query}' 的搜索结果..."}

# 2. 创建插件
class MyPlugin(HawiPlugin):
    @hook("before_conversation")
    async def on_start(self, agent):
        print("🚀 开始对话\n")

    @hook("after_conversation")
    async def on_end(self, agent):
        print("\n🏁 对话结束")

async def main():
    # 3. 创建模型
    model = DeepSeekModel(model_id="deepseek-chat")

    # 4. 创建 Agent
    agent = HawiAgent(
        model=model,
        plugins=[MyPlugin(), search_web.to_plugin()]
    )

    # 5. 流式对话
    printer = create_printer(streaming=True)

    async for event in agent.arun(
        "搜索一下 Python 的最新版本",
        stream=True
    ):
        await printer.handle(event)

if __name__ == "__main__":
    asyncio.run(main())
```

## 使用图形界面

Hawi 提供了基于 Electron 的图形界面，支持流式对话、工具调用可视化、优先级队列和模型切换。

### 启动 GUI

```bash
# 一键启动（首次会自动安装 GUI 依赖，并弹出模型选择对话框）
./hawi_gui/start.sh

# 指定模型工厂启动
./hawi_gui/start.sh kimi-k2-5
```

### GUI 功能

- **三级优先级队列**：Tab 键切换 普通/高优/紧急
- **流式输出**：实时显示模型回复
- **工具调用**：可视化展示工具调用和结果
- **模型切换**：菜单栏随时切换不同模型
- **斜杠命令**：
  - `/clear` - 清空对话上下文
  - `/cq` - 清空普通队列
  - `/chq` - 清空高优队列
  - `/cuq` - 清空紧急队列
  - `/ca` - 清空所有队列
  - `/quit` - 退出

### GUI 架构

GUI 通过 `AgentRunnerThread` 在后台运行异步运行器，与 tkinter 主线程通过队列通信：

```
┌─────────────┐     queues      ┌─────────────────┐
│  tkinter    │ ◄────────────► │ AgentRunnerThread │
│    UI       │   (thread-safe) │  (asyncio loop) │
└─────────────┘                 └─────────────────┘
                                        │
                                        ▼
                                ┌───────────────┐
                                │ AgentRunner │
                                │   + Agent     │
                                └───────────────┘
```

## 下一步

- 阅读 [架构文档](./architecture.md) 了解整体设计
- 查看 [Event 系统](./event_system.md) 学习流式处理
- 了解 [Hook 系统](./hook_system.md) 实现插件扩展
- 探索 [模型适配器](./models.md) 支持更多 LLM
- 学习 [工具系统](./tools.md) 创建自定义工具
