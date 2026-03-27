# Hook System 钩子系统

## 概述

Hook 是 Hawi 提供的**阻塞式、可修改**的扩展机制。与 Event 不同，Hook 可以：

- 阻塞 Agent 执行直到处理完成
- 读取和修改 Agent 状态
- 干预工具调用参数和结果
- **返回 `HookResult` 控制 Agent 流程**

## 核心概念

```mermaid
flowchart TD
    Start(["Agent 执行流程"]) --> BS["Hook: before_session"]
    BS --> BC["Hook: before_conversation"]
    BC --> BMC["Hook: before_model_call"]
    BMC -->|"replace_model / restart_turn / reinvoke / abort"| Model["Model 调用"]
    Model --> AMC["Hook: after_model_call"]
    AMC -->|"reinvoke / abort"| BTC["Hook: before_tool_calling"]
    BTC -->|"skip / abort"| Tool["工具执行"]
    Tool --> ATC["Hook: after_tool_calling"]
    ATC --> AEC["Hook: after_conversation"]
    AEC --> AS["Hook: after_session"]
    AS --> End(["结束"])
```

## 使用方法

Hook 必须定义在 `HawiPlugin` 子类中。每种 Hook 通过装饰器标记，并接受 `HookContext` 作为最后一个参数。

```python
from hawi.plugin import HawiPlugin, HookContext, HookResult
from hawi.plugin.decorators import before_conversation, after_conversation

class MyPlugin(HawiPlugin):
    @before_conversation
    async def on_start(self, agent, ctx: HookContext):
        print(f"[{ctx.run_id}] 会话开始，iteration={ctx.iteration}")

    @after_conversation
    async def on_end(self, agent, ctx: HookContext):
        if ctx.error:
            print(f"执行出错: {ctx.error}")
```

## HookContext

每次 Hook 调用都会收到一个 `HookContext` 对象（最后一个参数）。

```python
@dataclass(frozen=True)
class HookContext:
    run_id: str               # 本次 arun() 的唯一 ID
    iteration: int            # 当前循环轮次（session/conversation 级钩子 = 0）
    tool_call_id: str | None  # 工具调用 ID（tool 类钩子专用）
    tool: AgentTool | None    # 工具对象本体（tool 类钩子专用）
    duration_ms: float | None # 执行耗时（after 类钩子）
    error: Exception | None   # 错误（after_conversation / after_session）
```

各 Hook 的有效 `ctx` 字段：

| Hook | 有效字段 |
|------|---------|
| `before_session` | `run_id` |
| `after_session` | `run_id`, `duration_ms`, `error` |
| `before_conversation` | `run_id` |
| `after_conversation` | `run_id`, `duration_ms`, `error` |
| `before_model_call` | `run_id`, `iteration` |
| `after_model_call` | `run_id`, `iteration`, `duration_ms` |
| `before_tool_calling` | `run_id`, `iteration`, `tool_call_id`, `tool` |
| `after_tool_calling` | `run_id`, `iteration`, `tool_call_id`, `tool`, `duration_ms` |

> **注意**：`after_model_call` 的 `stop_reason` 和 `usage` 已从 `HookContext` 中移除。
> 请通过 `response.stop_reason` 和 `response.usage` 访问（`response` 是该 hook 的第二个参数）。

## HookResult — 流程控制

Hook 可以通过返回 `HookResult` 来控制 Agent 流程。返回 `None` 表示继续正常执行。

```python
from hawi.plugin import HookResult
from hawi.tool.types import ToolResult
```

### 跳过工具执行（skip）

在 `before_tool_calling` 中返回 `HookResult.skip(result)`，用 synthetic result 替代实际执行：

```python
@before_tool_calling
def cache_get(self, agent, tool_name, arguments, ctx: HookContext):
    key = f"{tool_name}:{arguments}"
    if key in self._cache:
        return HookResult.skip(self._cache[key])
```

### 提前终止 Agent 循环（abort）

在任意 Hook 中返回 `HookResult.abort(reason)`，提前结束 Agent run：

```python
@before_model_call
def budget_guard(self, agent, model, ctx: HookContext):
    # 通过 agent.context 访问上下文信息
    return HookResult.abort("token budget exceeded")
```

终止后，`AgentRunResult.stop_reason == "hook_abort"`。

### 替换本次 Model（replace_model）

在 `before_model_call` 中返回 `HookResult.replace_model(model)`，仅替换**本次**调用所用的 model：

```python
@before_model_call
def use_fallback(self, agent, model, ctx: HookContext):
    if ctx.iteration > 3:
        return HookResult.replace_model(self.fallback_model)
```

### 跳过本次 Model 调用（restart_turn）

在 `before_model_call` 中返回 `HookResult.restart_turn()`，跳过本次 model call 并直接进入下一次循环迭代（不终止整个 run）：

```python
@before_model_call
def maybe_skip(self, agent, model, ctx: HookContext):
    if self.should_skip_this_turn():
        return HookResult.restart_turn()
```

### 注入消息并重新驱动（reinvoke）

在 `before_model_call` 或 `after_model_call` 中返回 `HookResult.reinvoke(message)`，将 message 追加到 context，终止当前 run（`stop_reason == "hook_reinvoke"`），并以新消息重新调用 `arun()`：

```python
@after_model_call
def inject_followup(self, agent, response, ctx: HookContext):
    if self.needs_followup(response):
        return HookResult.reinvoke("请继续完成上面的任务。")
```

### Hook Chain 执行规则

多个插件注册同一 Hook 类型时，按注册顺序依次执行：

- 返回 `None` → 继续执行链中的下一个 Hook。
- 返回 `HookResult` → **链立即停止**，后续 Hook 不再执行。

```python
agent = HawiAgent(
    plugins=[PluginA(), PluginB(), PluginC()]
)
# before_model_call 执行顺序：
# 1. PluginA.before_model_call → None → 继续
# 2. PluginB.before_model_call → HookResult.abort() → 链停止
# 3. PluginC.before_model_call → 不执行
```

## Hook 类型完整签名

```python
from hawi.plugin.decorators import (
    before_session,       # (self, agent, ctx)
    after_session,        # (self, agent, ctx)
    before_conversation,  # (self, agent, ctx)
    after_conversation,   # (self, agent, ctx)
    before_model_call,    # (self, agent, model, ctx)
    after_model_call,     # (self, agent, response, ctx)
    before_tool_calling,  # (self, agent, tool_name, arguments, ctx)
    after_tool_calling,   # (self, agent, tool_name, arguments, result, ctx)
)
```

所有 Hook 均可返回 `HookResult | None`（同步或异步均可）。

## 上下文操作时序

各 Hook 时机下操作 `agent.context` 的注意事项：

| Hook | 时序说明 |
|------|---------|
| `before_session` / `before_conversation` | 修改在整个 run 开始前生效 |
| `before_model_call` | 修改在本次 model call 中生效 |
| `after_model_call` | assistant message **尚未**写入 context（hook 返回后才写入） |
| `before_tool_calling` | 修改在工具执行前生效；`arguments` 可直接修改 |
| `after_tool_calling` | tool result **尚未**写入 context（hook 返回后才写入）；`result` 可直接修改 |

## 异常处理

- Hook 中的**普通异常直接中断 Agent 执行**（透传到调用方，不会被静默捕获）。
- 可预期的"拒绝"请使用 `HookResult.skip()`；"终止运行"请使用 `HookResult.abort()`。

```python
@before_tool_calling
def permission_check(self, agent, tool_name, arguments, ctx: HookContext):
    if not self.has_permission(tool_name):
        # ✅ 拒绝工具调用，给 LLM 一个错误结果
        return HookResult.skip(ToolResult(success=False, output={"error": f"Permission denied: {tool_name}"}))

    if tool_name == "shutdown":
        # ✅ 立即终止整个 Agent run
        return HookResult.abort("shutdown tool called")

    # ❌ 不要用异常来"拒绝"——异常会中断整个 Agent，而不是返回错误给 LLM
```

## 实用示例

### Token 预算守卫

```python
class BudgetPlugin(HawiPlugin):
    def __init__(self, max_output_tokens: int):
        self.max_output_tokens = max_output_tokens
        self._total_output = 0

    @after_model_call
    def check_budget(self, agent, response, ctx: HookContext):
        if response.usage:
            self._total_output += response.usage["output_tokens"]
        if self._total_output > self.max_output_tokens:
            return HookResult.abort("output token budget exceeded")
```

### 工具缓存

```python
class CachePlugin(HawiPlugin):
    def __init__(self):
        self._cache: dict = {}

    @before_tool_calling
    def cache_get(self, agent, tool_name, arguments, ctx: HookContext):
        key = f"{tool_name}:{sorted(arguments.items())}"
        if key in self._cache:
            return HookResult.skip(self._cache[key])

    @after_tool_calling
    def cache_set(self, agent, tool_name, arguments, result, ctx: HookContext):
        if result.success:
            key = f"{tool_name}:{sorted(arguments.items())}"
            self._cache[key] = result
```

### 工具执行统计

```python
class StatsPlugin(HawiPlugin):
    def __init__(self):
        self.tool_durations: dict[str, list[float]] = {}

    @after_tool_calling
    def record_duration(self, agent, tool_name, arguments, result, ctx: HookContext):
        if ctx.duration_ms is not None:
            self.tool_durations.setdefault(tool_name, []).append(ctx.duration_ms)

    def report(self):
        for name, durations in self.tool_durations.items():
            avg = sum(durations) / len(durations)
            print(f"{name}: avg={avg:.1f}ms, calls={len(durations)}")
```

### 动态 System Prompt 注入

```python
class ContextPlugin(HawiPlugin):
    def __init__(self, user_id: str):
        self.user_id = user_id

    @before_conversation
    async def inject(self, agent, ctx: HookContext):
        prefs = await self.load_user_prefs(self.user_id)
        agent.context.system_prompt.append({
            "type": "text",
            "text": f"\n用户偏好: {prefs}"
        })
```

### 动态切换备用 Model

```python
class FallbackModelPlugin(HawiPlugin):
    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback
        self._fail_count = 0

    @before_model_call
    def maybe_fallback(self, agent, model, ctx: HookContext):
        if self._fail_count >= 2:
            return HookResult.replace_model(self.fallback)

    @after_model_call
    def track_failures(self, agent, response, ctx: HookContext):
        if response.stop_reason == "error":
            self._fail_count += 1
        else:
            self._fail_count = 0
```

## Hook 与 Event 的区别

| 特性 | Hook | Event |
|------|------|-------|
| 阻塞性 | ✅ 阻塞执行 | ❌ 非阻塞（worker 线程） |
| 可修改性 | ✅ 可修改 Agent 状态 | ❌ 只读（frozen Pydantic 模型） |
| 执行顺序 | 链式，可中途停止 | 广播，所有订阅者都执行 |
| 流程控制 | ✅ skip / abort / replace_model / restart_turn / reinvoke | ❌ 无 |
| 消费者数量 | 多个（chain） | 多个（broadcast） |
| 错误处理 | 透传，中断执行 | 捕获，不影响主流程 |
| 适用场景 | 干预、扩展 | 观察、记录 |
