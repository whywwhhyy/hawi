# Hook System 扩展实施 Plan

本文档是对现有 [hook_system.md](hook_system.md) 的能力扩展计划，覆盖
"插件完整定制 agent 流程" 所需的新增 hook、事件增强与现存 bug 修复。

设计经过两路独立 review（内部 Plan + codex CLI）后收口。每个 Phase 的设计
应足够详细以支持直接实施——文件路径、行号、签名、diff、边界条件、测试
都在文档中明确。

> 维护说明：本文档中的 `agent.py` 行号来自重构前的大单文件版本，只能作为
> 历史定位参考。当前 hook dispatch 逻辑在
> `hawi/agent/hook_dispatcher.py`，`HawiAgent` 仍保留 `_invoke_*()` facade；
> 主循环调用点仍在 `hawi/agent/agent.py`。

---

## 目录

1. [Scope](#scope)
2. [全局约定](#全局约定)
3. [Phase 0：修复现存 bug](#phase-0修复现存-bug)
4. [Phase 1：基建（HookContext + events + before_user_message）](#phase-1基建)
5. [Phase 2：HookResult.strip](#phase-2hookresultstrip)
6. [Phase 3：Compact 事务化](#phase-3compact-事务化)
7. [Phase 4：错误 / 中断观察](#phase-4错误--中断观察)
8. [既有插件迁移](#既有插件迁移)
9. [测试策略](#测试策略)
10. [Phase 之间的依赖与发布](#phase-之间的依赖与发布)
11. [已拒绝提案（决策记录）](#已拒绝提案决策记录)
12. [悬而未决](#悬而未决)

---

## Scope

### In scope

- 修复 `before_tool_calling` / `after_tool_calling` 返回值消费 bug
- 新增 hook：`before_user_message`、`before_compact` / `after_compact`、
  `on_interrupt`
- 新增 HookResult action：`strip`
- HookContext 增字段；`AgentMessageAddedEvent` 增 metadata
- 新增观察事件：`AgentModelErrorEvent` / `AgentToolErrorEvent`
- 显式化 `before_tool_calling` 中 `arguments` 原地修改的契约

### Out of scope（review 后拒绝或推迟）

见末尾 [已拒绝提案](#已拒绝提案决策记录) 节。

---

## 全局约定

本节是所有 Phase 共享的契约，实施时严格遵守。

### 文件路径

所有路径相对仓库根 `/Users/hayden/Projects/Python/Hawi/`。

主要文件：
- `hawi/plugin/hook_context.py` — `HookContext` / `HookResult` / 新增 draft 类
- `hawi/plugin/decorators.py` — hook decorator
- `hawi/plugin/types.py` — method 类型 + `PluginHooks` TypedDict
- `hawi/plugin/manager.py` — `PluginManager`（hook 收集与 dispatch；不需改 hook 接入逻辑）
- `hawi/plugin/plugin.py` — `HawiPlugin` 基类（自动收集机制不需改）
- `hawi/agent/agent.py` — 主循环、hook facade 调用点
- `hawi/agent/hook_dispatcher.py` — hook dispatch 实现
- `hawi/agent/runtime.py` — interrupt、runtime snapshot、steer 路径
- `hawi/agent/compaction.py` — compact 路径
- `hawi/agent/context.py` — `AgentContext` 与 `ContextCompactionRecord`
- `hawi/events/agent_events.py` — agent 级事件
- `hawi_plugins/workflow_plugin/plugin.py` — 受 Phase 0 影响
- `hawi_plugins/python_interpreter/plugin.py` — 受 Phase 4 影响

### Hook 调用语义（不变）

由 `PluginManager.get_hooks(hook_type)` 返回的列表按注册顺序串行调用
（[manager.py:_collect_plugin_hooks](../hawi/plugin/manager.py)）。任一 hook 返回非
`None` 的 `HookResult` 立即终止链路，结果交由 agent 处理。

支持 sync 与 async 两种 hook 实现：
```python
result = hook(...)
if inspect.isawaitable(result):
    result = await result
```
此约定已存在于 [agent.py:_invoke_*](../hawi/agent/agent.py#L1393-L1441)，所有新 hook 沿用。

### Hook 内部异常的处理

当前实现：hook 内部 raise 直接冒泡到主循环，被外层 `except Exception` 包成
`AgentError("tool_execution", ...)`（[agent.py:2105-2124](../hawi/agent/agent.py#L2105-L2124)）。

**决策：保留此行为。** Hook 异常视为插件 bug，不静默吞掉。新增 hook 沿用相同
策略。事件 / 监控由 `AgentErrorEvent` 提供。

### Async vs sync

`before_user_message` / `before_compact` / `on_interrupt` 等新 hook 全部支持
sync 与 async 两种实现，与现有 hook 一致。

### 类型 import 规范

新增的 dataclass 与 method 类型在 `hook_context.py` / `types.py` 中定义。
TYPE_CHECKING 守卫现有约定保持。

### `__match_args__` 锁定

`HookContext` 是 frozen dataclass，新增字段会污染 `__match_args__`，破坏
`match ctx:` 模式。**所有 dataclass 的 `__match_args__` 必须显式锁定到当前
"语义稳定字段"集合**，不随新增字段自动增长。

---

## Phase 0：修复现存 bug

> 必须最先合入；与所有新功能解耦；建议作为独立 patch release。

### 0.1 问题

#### Bug 1：`before_tool_calling` 不消费 `abort`

[agent.py:2525-2527](../hawi/agent/agent.py#L2525-L2527)：

```python
_hr = await self._invoke_before_tool_calling(tool_name, arguments, _before_ctx)
if _hr and _hr.action == "skip":
    result = _hr.tool_result or ToolResult(success=False, error="Hook skipped tool without providing a result")
elif tool is None:
    ...
```

只识别 `skip`，但 [decorators.py:127-154](../hawi/plugin/decorators.py#L127-L154)
docstring 承诺 "Returns ... `HookResult.abort(reason)` to terminate the agent run"。

#### Bug 2：`after_tool_calling` 完全不读返回值

[agent.py:2632-2641](../hawi/agent/agent.py#L2632-L2641)：

```python
await self._invoke_after_tool_calling(
    tool_name, arguments, result,
    HookContext(...),
)
```

直接 `await ...` 丢弃。但
[decorators.py:178-180](../hawi/plugin/decorators.py#L178-L180) docstring 承诺
"Returns ... `HookResult.abort(reason)`"。

#### 实际后果

[workflow_plugin/plugin.py:215-250](../hawi_plugins/workflow_plugin/plugin.py#L215-L250)
的 `gate_guard`（@after_tool_calling）通过 `_on_approved` / `_on_rejected`
返回 `HookResult.reinvoke(...)` 推进工作流——**当前根本不生效**。

### 0.2 决策：每个 hook 接受哪些 action

为了避免再出现 "docstring 承诺但实现没接" 的偏差，**每个 hook 显式声明
合法 action 集合**，运行时校验：

| Hook | 合法 action |
|---|---|
| `before_session` | `abort` |
| `after_session` | （无返回值意义；运行时忽略） |
| `before_conversation` | `abort` |
| `after_conversation` | `abort`、`reinvoke` |
| `before_model_call` | `abort`、`replace_model`、`restart_turn`、`reinvoke` |
| `after_model_call` | `abort`、`reinvoke` |
| `before_tool_calling` | `abort`、`skip`、`strip`（Phase 2 加） |
| `after_tool_calling` | `abort`、`reinvoke` |

不在表中的 action 由框架记录 warning 并视为 `None`（不阻断 run）。

实现时新增辅助：

```python
# hawi/plugin/hook_context.py
_ALLOWED_ACTIONS: dict[str, frozenset[str]] = {
    "before_session": frozenset({"abort"}),
    "after_session": frozenset(),
    "before_conversation": frozenset({"abort"}),
    "after_conversation": frozenset({"abort", "reinvoke"}),
    "before_model_call": frozenset({"abort", "replace_model", "restart_turn", "reinvoke"}),
    "after_model_call": frozenset({"abort", "reinvoke"}),
    "before_tool_calling": frozenset({"abort", "skip", "strip"}),  # strip from Phase 2
    "after_tool_calling": frozenset({"abort", "reinvoke"}),
    "before_user_message": frozenset({"abort"}),       # Phase 1
    "before_compact": frozenset({"abort"}),            # Phase 3
    "after_compact": frozenset(),                       # Phase 3
    "on_interrupt": frozenset(),                        # Phase 4
}

def validate_hook_result(hook_type: str, result: HookResult | None) -> HookResult | None:
    if result is None:
        return None
    allowed = _ALLOWED_ACTIONS.get(hook_type)
    if allowed is None or result.action not in allowed:
        warnings.warn(
            f"Hook '{hook_type}' returned action '{result.action}' which is not "
            f"valid for this hook type (allowed: {sorted(allowed) if allowed else '<none>'}). "
            f"Result will be ignored.",
            stacklevel=3,
        )
        return None
    return result
```

### 0.3 修复 `_invoke_*` 链路

`hawi/agent/agent.py` [Line 1393-1441](../hawi/agent/agent.py#L1393-L1441) 区域，
每个 `_invoke_*` 在收到 `result` 后过 `validate_hook_result(hook_type, result)`：

```python
async def _invoke_before_tool_calling(self, tool_name, arguments, ctx):
    for hook in self._plugin_manager.get_hooks("before_tool_calling"):
        result = hook(self, tool_name, arguments, ctx)
        if inspect.isawaitable(result):
            result = await result
        result = validate_hook_result("before_tool_calling", result)
        if result is not None:
            return result
    return None
```

其它 `_invoke_*` 同样改造。

### 0.4 `before_tool_calling` 接 `abort`

`hawi/agent/agent.py` [Line 2525-2530](../hawi/agent/agent.py#L2525-L2530) 改为：

```python
_hr = await self._invoke_before_tool_calling(tool_name, arguments, _before_ctx)
if _hr is not None:
    if _hr.action == "skip":
        result = _hr.tool_result or ToolResult(
            success=False,
            error="Hook skipped tool without providing a result",
        )
    elif _hr.action == "abort":
        # 立即终止 run；当前 tool_call 合成 error 结果以保协议
        result = ToolResult(
            success=False,
            error=f"Aborted by before_tool_calling hook: {_hr.reason}",
        )
        state.should_stop = True
    # action == "strip" 在 Phase 2 处理；本 phase 暂不出现
elif tool is None:
    err = ToolNotFoundError(f"Tool '{tool_name}' not found")
    result = ToolResult(success=False, error=f"{err.__class__.__name__}: {err.message}")
else:
    # 原 prepared / audit / 实际执行分支
    ...
```

注意 `state.should_stop = True` 让外层 `while not state.should_stop` 在
本轮 tool batch 结束后退出。本 tool_call 的 result 仍写入 context（保
tool_call ↔ tool_result 配对）。

### 0.5 `after_tool_calling` 接 `abort` / `reinvoke`

`hawi/agent/agent.py` [Line 2631-2641](../hawi/agent/agent.py#L2631-L2641)：

```python
duration_ms = (time.time() - start_time) * 1000

# after_tool_calling hook
_hr_after = await self._invoke_after_tool_calling(
    tool_name, arguments, result,
    HookContext(
        run_id=state.run_id,
        iteration=state.iteration,
        tool_call_id=tool_call_id,
        tool=tool,
        duration_ms=duration_ms,
    ),
)

# 即便 abort / reinvoke，本 tool_result 仍要先写入 context（保协议）
if not audit_pending:
    result_content = self._tool_result_content(result)
    materialized_messages = self._add_tool_result_with_pending_steer(...)
    ...

# 落库后再处理 hook 决策
if _hr_after is not None:
    if _hr_after.action == "abort":
        state.should_stop = True
    elif _hr_after.action == "reinvoke" and _hr_after.message is not None:
        # 与 after_model_call.reinvoke 一致的处理路径：
        # 1) 缓存 reinvoke message
        # 2) 设置 should_stop 让 batch 完成后退出 while
        # 3) finally 块 / 外层捕获后递归 _arun_internal
        state.pending_reinvoke_message = _hr_after.message
        state.should_stop = True
```

新增 `_ExecutionState.pending_reinvoke_message: str | list[ContentPart] | None = None`
字段。

外层 while 退出后（[agent.py:2007 之后](../hawi/agent/agent.py#L2007)），在
`finally` 块之前检查 `state.pending_reinvoke_message`：

```python
# 在 break 退出 while 后、进入 finally 之前
if state.pending_reinvoke_message is not None:
    self._context.add_user_message(state.pending_reinvoke_message)
    await self._emit_event(
        AgentRunStopEvent.create(
            run_id=run_id, stop_reason="hook_reinvoke",
            duration_ms=(time.time() - start_time) * 1000,
            usage=cumulative_usage,
        ),
        event_bus,
    )
    return await self._arun_internal(
        message=None, model=model,
        event_bus=event_bus, streaming=streaming,
    )
```

注意：**reinvoke 必须在 batch 内的所有 tool_result 都写完之后才能触发**，
否则下一轮 model 看到 partial tool_result 协议会爆。这是现有
`after_model_call.reinvoke` 不存在的新约束，因为 after_model_call 时还没
进入 tool batch。

### 0.6 `_check_interrupt` 与 `should_stop` 交互

[agent.py:2055-2070](../hawi/agent/agent.py#L2055-L2070) 区域已有
`_check_interrupt` 检查；新增 `should_stop` 判定不与之冲突，但要确保
`should_stop=True` 后 batch 内剩余的 tool_call 仍执行（避免 partial batch
的协议问题）。当前 [agent.py:2003-2005](../hawi/agent/agent.py#L2003-L2005) 是
`if self._check_interrupt(): break` 在 batch 内提前退出——hook abort 路径
**不应**这样退出，因为 abort 是有序终止，要把 batch 跑完再 break。

修正：`should_stop` 检查只放在 `while not state.should_stop` 那一处
（[agent.py:1740](../hawi/agent/agent.py#L1740)），不在 tool batch 循环内
break。

### 0.7 改动文件清单

| 文件 | 改动 |
|---|---|
| `hawi/plugin/hook_context.py` | 增 `_ALLOWED_ACTIONS` + `validate_hook_result` |
| `hawi/agent/agent.py` | 8 个 `_invoke_*` 改用 validator；2525 处接 abort；2631 处接 abort/reinvoke；新增 `pending_reinvoke_message` 字段 |
| `hawi/plugin/decorators.py` | 在每个 hook docstring "Returns" 段下方追加 "**Allowed actions:** ..." 行，与 `_ALLOWED_ACTIONS` 对齐 |

### 0.8 测试

新建 `test/unit/test_hook_action_consumption.py`：

```python
import pytest
from hawi.agent import HawiAgent
from hawi.plugin import HawiPlugin
from hawi.plugin.decorators import before_tool_calling, after_tool_calling
from hawi.plugin.hook_context import HookResult, HookContext
from hawi.tool import tool, ToolResult


class _AbortBeforePlugin(HawiPlugin):
    @before_tool_calling
    def hook(self, agent, tool_name, arguments, ctx):
        return HookResult.abort(reason="test abort")


class _AbortAfterPlugin(HawiPlugin):
    @after_tool_calling
    def hook(self, agent, tool_name, arguments, result, ctx):
        return HookResult.abort(reason="test abort")


class _ReinvokeAfterPlugin(HawiPlugin):
    @after_tool_calling
    def hook(self, agent, tool_name, arguments, result, ctx):
        return HookResult.reinvoke("please continue with X")


@tool()
def echo(text: str) -> str:
    return text


@pytest.mark.asyncio
async def test_before_tool_calling_abort_terminates_run(stub_model):
    """before_tool_calling 返回 abort 后 run 立即停止，且 tool 未执行。"""
    plugin = _AbortBeforePlugin()
    agent = HawiAgent(model=stub_model, plugins=[plugin])
    # stub_model 第一次 call 返回 echo(text="x") 的 tool_call
    result = await agent.arun("hello")
    assert result.stop_reason == "tool_use"  # batch 完成后 should_stop=True 退出
    # echo 这次没有真实执行（tool_result 是 hook 合成的 error）
    last_tool_record = result.tool_calls[-1]
    assert "Aborted by before_tool_calling" in last_tool_record.result.error


@pytest.mark.asyncio
async def test_after_tool_calling_abort_writes_result_then_stops(stub_model):
    """after_tool_calling abort 后 tool_result 已写入，下一轮 model 不再被调用。"""
    plugin = _AbortAfterPlugin()
    agent = HawiAgent(model=stub_model, plugins=[plugin])
    result = await agent.arun("hello")
    assert result.stop_reason == "tool_use"
    # 上下文中既有 tool_call 也有 tool_result（协议正确）
    msgs = agent.context.messages
    assert any(m["role"] == "tool" for m in msgs)


@pytest.mark.asyncio
async def test_after_tool_calling_reinvoke_appends_and_recurses(stub_model_two_turn):
    """after_tool_calling reinvoke 触发新一轮 _arun_internal。"""
    plugin = _ReinvokeAfterPlugin()
    agent = HawiAgent(model=stub_model_two_turn, plugins=[plugin])
    result = await agent.arun("hello")
    # stub 第二轮直接 end_turn；最终 stop_reason 是 end_turn
    assert result.stop_reason == "end_turn"
    # reinvoke 注入的 user message 在历史中
    user_msgs = [m for m in agent.context.messages if m["role"] == "user"]
    assert any("please continue with X" in str(m["content"]) for m in user_msgs)


@pytest.mark.asyncio
async def test_invalid_action_warns_and_ignored(stub_model):
    """before_session 返回 skip 不合法，warn 并被忽略。"""
    class P(HawiPlugin):
        @before_session
        def hook(self, agent, ctx):
            return HookResult.skip(ToolResult(success=True, output="x"))

    agent = HawiAgent(model=stub_model, plugins=[P()])
    with pytest.warns(UserWarning, match="not valid for this hook type"):
        await agent.arun("hello")
```

新建 `test/integration/test_workflow_gate_guard.py`：

```python
@pytest.mark.asyncio
async def test_workflow_gate_approved_path_advances():
    """Phase 0 修复后 gate_guard approve 路径真正推进工作流。"""
    # 完整 workflow run：approve → reinvoke → 下一节点
    ...

@pytest.mark.asyncio
async def test_workflow_gate_rejected_path_retries():
    """reject 路径触发 reinvoke 让 agent 修正。"""
    ...
```

具体 fixture 复用 `test/integration/test_workflow_*.py` 中已有的 stub model。

---

## Phase 1：基建

> 三个独立子项可并行。无功能影响，仅扩字段 + 加新 hook 入口。

### 1.1 HookContext 字段增补

#### 完整新签名

```python
# hawi/plugin/hook_context.py

@dataclass(frozen=True)
class HookContext:
    """Runtime context passed to every hook call.

    Existing fields are kept for backward compat. New fields default to None.
    Any field that is not meaningful for a given hook type is None — see the
    decorator docstring for which fields are populated.
    """

    # === Existing fields (signature unchanged) ===
    run_id: str
    iteration: int
    tool_call_id: str | None = None
    tool: AgentTool | None = None
    duration_ms: float | None = None
    error: Exception | None = None

    # === New in Phase 1 ===
    model: Model | None = None
    """The Model in use for the current call. Populated for:
    - before_model_call / after_model_call (the model about to run / that ran)
    - before_tool_calling / after_tool_calling (the model owning this turn)
    Otherwise None.
    """

    tool_call_index: int | None = None
    """0-based index of this tool_call within the current iteration's batch.
    Populated for before_tool_calling / after_tool_calling only.
    """

    tool_batch_size: int | None = None
    """Total number of tool_calls in the current iteration's batch.
    Populated for before_tool_calling / after_tool_calling only.
    """

    message_delta_start: int | None = None
    """Index into agent.context.messages marking the first message added in
    the current arun() invocation. Populated for all hooks during a run.
    Note: invalidated by compaction; see compact hooks for safe usage.
    """

    # __match_args__ locked to the original 6 fields so that pattern matching
    # written before this change keeps working unchanged.
    __match_args__ = (
        "run_id", "iteration", "tool_call_id",
        "tool", "duration_ms", "error",
    )
```

#### 字段填充矩阵

每个 `HookContext(...)` 构造点必须按下表填充：

| 构造位置（agent.py 行号） | hook_type | 必填字段 |
|---|---|---|
| ~1741 before_session | before_session | run_id, iteration=0, message_delta_start |
| ~1747 before_conversation | before_conversation | run_id, iteration=0, message_delta_start |
| ~1772 before_model_call | before_model_call | run_id, iteration, model, message_delta_start |
| ~1968 after_model_call | after_model_call | run_id, iteration, duration_ms, model, message_delta_start |
| ~2517 before_tool_calling | before_tool_calling | run_id, iteration, tool_call_id, tool, model, tool_call_index, tool_batch_size, message_delta_start |
| ~2634 after_tool_calling | after_tool_calling | + duration_ms |
| ~2160 finally(after_conversation) | after_conversation | run_id, iteration, duration_ms, error, message_delta_start |
| ~2167 finally(after_session) | after_session | 同上 |

**`message_delta_start`** 在 `_arun_internal` 入口处计算一次（已有
`initial_message_count` 变量，[agent.py:1714](../hawi/agent/agent.py#L1714)），
作为 closure 传给所有 HookContext 构造点。compact 后值会过期；compact 路径
有自己的 hook 不依赖此字段。

**`tool_call_index` / `tool_batch_size`** 在 tool batch 循环
（[agent.py:1995-2030](../hawi/agent/agent.py#L1995-L2030)）中按
`enumerate(tool_calls)` 计算，传入 `_execute_tool`。

**`model`** 在 tool hook 中传 `state.model`（如有 replace_model 则是替换值）。
新增 `_ExecutionState.model: Model | None = None`，每轮 `before_model_call`
后写入。

### 1.2 `AgentMessageAddedEvent` 增 metadata

#### 新签名

```python
# hawi/events/agent_events.py

MessageSource = Literal[
    "user",          # add_user_message from user input
    "assistant",     # add_assistant_message from model response
    "tool_result",   # add_tool_result from tool execution
    "injected",      # context.inject() (e.g. environ_prompt_plugin)
    "compacted",     # compact_with_summary() summary message
    "reinvoke",      # add_user_message from HookResult.reinvoke
    "steer",         # pending_steer materialized into user message
]


class AgentMessageAddedEvent(Event):
    run_id: str
    role: Literal["user", "assistant", "tool"]
    content: list[ContentPart]
    metadata: dict[str, Any] | None = None
    # === New in Phase 1 ===
    source: MessageSource | None = None
    message_id: str | None = None
    message_index: int | None = None

    @classmethod
    def create(
        cls,
        run_id: str,
        role: Literal["user", "assistant", "tool"],
        content: list[ContentPart],
        metadata: dict[str, Any] | None = None,
        *,
        source: MessageSource | None = None,
        message_id: str | None = None,
        message_index: int | None = None,
    ) -> AgentMessageAddedEvent:
        return cls(
            type="agent.message_added",
            source="agent",
            run_id=run_id,
            role=role,
            content=content,
            metadata=metadata,
            # NB: event already has a top-level `source` field meaning event
            # source ("agent"). The new `source` field below describes the
            # origin of the message. Renamed to message_source on the event
            # to disambiguate.
            message_source=source,
            message_id=message_id,
            message_index=message_index,
        )
```

> **重要：字段命名冲突**
>
> `Event` 基类已有 `source` 字段，含义是 "事件发布方"（这里始终为 `"agent"`）。
> 不能用 `source` 表示消息来源——会冲突。
>
> 新字段命名为 **`message_source`**，类型 `MessageSource | None`。
> 文中其它地方提到 "source enum" 时指的是 `message_source`。

#### 所有 `AgentMessageAddedEvent.create(...)` 调用点

| agent.py 行号 | 现有调用 | 应填 message_source |
|---|---|---|
| 1271 | `_drain_pending_inputs_to_context` 中 user pending input | `"steer"` |
| 1351 | `_add_tool_result_with_pending_steer` 中 tool result | `"tool_result"` |
| 1447 | `_emit_tool_result_message_event` 中 tool result | `"tool_result"` |
| 1695 | `_execute()` 中 user message 写入 | `"user"` |
| 1971 | 主循环中 assistant message 写入 | `"assistant"` |

`environ_prompt_plugin/plugin.py` 的 `inject()` 调用如果未来发事件，对应
`"injected"`；当前实现不发事件，本 phase 不动。

`context.py` 中 `compact_with_summary` 不直接发事件；compact 完成后由
`acompact` 调用方发事件，对应 `"compacted"`（Phase 3 处理）。

#### `message_id` 与 `message_index`

`message_id`：从 `metadata["message_id"]` 透传。如果 metadata 没有，则 fallback
为 `None`。**Phase 1 不主动生成 id**——这是 todo.md 中独立项目（Message
一等公民 id）的工作；本 phase 仅把 metadata 中已有的 id 提升到事件字段。

`message_index`：写入 context 后取 `len(self._context.messages) - 1`。这是
**写入时的瞬时 index**；compaction 会让它失效。文档明确 "用于实时观察，
不应作长期 ref"。

#### 改动点

`hawi/agent/agent.py` 五处 `AgentMessageAddedEvent.create(...)`：

```python
# 例：行 1695 区域
self._context.add_user_message(message, metadata=message_metadata)
await self._emit_event(
    AgentMessageAddedEvent.create(
        run_id=run_id,
        role="user",
        content=user_content,
        metadata=message_metadata,
        message_source="user",
        message_id=(message_metadata or {}).get("message_id"),
        message_index=len(self._context.messages) - 1,
    ),
    event_bus,
)
```

### 1.3 `before_user_message` + `UserMessageDraft`

#### 新类型

```python
# hawi/plugin/hook_context.py

@dataclass
class UserMessageDraft:
    """Mutable draft for a user message about to be added to context.

    Hooks may modify ``content`` and ``metadata`` in place. Setting
    ``content`` to an empty list aborts the message write — equivalent to
    calling arun() without a message argument.

    The ``original_*`` fields capture the input as the agent received it,
    for trace/audit. They are read-only by convention; the framework does
    not enforce immutability but plugins should treat them as such.
    """
    content: list[ContentPart]
    metadata: dict[str, Any] | None
    original_content: list[ContentPart] = field(default_factory=list)
    original_metadata: dict[str, Any] | None = None
```

#### 新 method 类型 + decorator

```python
# hawi/plugin/types.py
BeforeUserMessageMethod: TypeAlias = Callable[
    [Any, "HawiAgent", "UserMessageDraft", HookContext],
    HookReturnType,
]

class PluginHooks(TypedDict):
    # existing fields...
    before_user_message: NotRequired[Callable[..., HookReturnType]]
```

```python
# hawi/plugin/decorators.py

def before_user_message(func: BeforeUserMessageMethod) -> BeforeUserMessageMethod:
    """Hook called when a user message is provided to arun()/run(),
    BEFORE it is written to context.

    Args:
        agent: The HawiAgent instance.
        draft: UserMessageDraft. Mutate ``draft.content`` / ``draft.metadata``
            to rewrite the message. Set ``draft.content = []`` to drop the
            message (run will proceed as if arun() was called with message=None).
        ctx: HookContext with run_id, iteration=0, message_delta_start.

    Not fired when:
        - arun(message=None) — there is no message to intercept.

    Allowed actions:
        - None to continue normally with current draft state.
        - HookResult.abort(reason) to terminate the agent run before any
          message is written.

    Hook chain:
        Multiple plugins' hooks run in registration order with the SAME
        draft object. Each may further mutate; first non-None HookResult
        stops the chain.
    """
    setattr(func, "_is_hook", True)
    setattr(func, "_hook_type", "before_user_message")
    return func
```

#### 集成位点

`hawi/agent/agent.py` `_execute()` 中 [Line 1684-1701](../hawi/agent/agent.py#L1684-L1701)
区域：

```python
# 现有：
if message is not None:
    self._last_unsent_tool_results = []
    if isinstance(message, str):
        user_content: list[ContentPart] = [{"type": "text", "text": message}]
    else:
        user_content = message
    message_metadata = dict(message_metadata) if message_metadata else None
    self._context.add_user_message(message, metadata=message_metadata)
    await self._emit_event(
        AgentMessageAddedEvent.create(
            run_id=run_id, role="user", content=user_content, metadata=message_metadata,
        ),
        event_bus,
    )
```

改造为：

```python
if message is not None:
    self._last_unsent_tool_results = []
    if isinstance(message, str):
        user_content: list[ContentPart] = [{"type": "text", "text": message}]
    else:
        user_content = list(message)  # 拷贝，防 hook mutation 影响调用方
    metadata_copy: dict[str, Any] | None = (
        dict(message_metadata) if message_metadata else None
    )

    draft = UserMessageDraft(
        content=user_content,
        metadata=metadata_copy,
        original_content=list(user_content),
        original_metadata=dict(metadata_copy) if metadata_copy else None,
    )
    _hr_user = await self._invoke_user_message_hook(
        draft,
        HookContext(run_id=run_id, iteration=0, message_delta_start=initial_message_count),
    )
    if _hr_user is not None and _hr_user.action == "abort":
        # 不写 message；走与 abort run 同路径
        state.should_stop = True
        # 仍要发 run start / run stop 事件保观察一致性
        await self._emit_event(AgentRunStartEvent.create(run_id=run_id), event_bus)
        await self._emit_event(
            AgentRunStopEvent.create(
                run_id=run_id, stop_reason="hook_abort",
                duration_ms=(time.time() - start_time) * 1000,
                usage=None,
            ),
            event_bus,
        )
        # 直接构造空 result 返回
        return AgentRunResult(
            stop_reason="hook_abort", messages=[], response=None,
            usage=None, tool_calls=[], error=f"Aborted by before_user_message: {_hr_user.reason}",
        )

    if not draft.content:
        # 消息被插件清空 = 等价于 arun(message=None)
        message = None
    else:
        self._context.add_user_message(draft.content, metadata=draft.metadata)
        await self._emit_event(
            AgentMessageAddedEvent.create(
                run_id=run_id,
                role="user",
                content=draft.content,
                metadata=draft.metadata,
                message_source="user",
                message_id=(draft.metadata or {}).get("message_id"),
                message_index=len(self._context.messages) - 1,
            ),
            event_bus,
        )
        message = draft.content  # 让后续逻辑看到改写后的值
```

`_invoke_user_message_hook` 实现紧挨 `_invoke_session_hook`：

```python
async def _invoke_user_message_hook(
    self, draft: UserMessageDraft, ctx: HookContext,
) -> HookResult | None:
    for hook in self._plugin_manager.get_hooks("before_user_message"):
        result = hook(self, draft, ctx)
        if inspect.isawaitable(result):
            result = await result
        result = validate_hook_result("before_user_message", result)
        if result is not None:
            return result
    return None
```

#### 边界条件

1. **`arun(message=None)` 不触发 hook**：当用户调 `arun()` 不带 message
   （比如恢复运行、runner 驱动），`message is None` 分支跳过 hook。
   插件需要拦截这种入口请用 `before_session`。
2. **`before_session` 与本 hook 时序**：当前 `before_session` 在 add_user_message
   之后。本 phase 不动 `before_session` 时序。新顺序：
   `arun → before_user_message → add_user_message → emit event → before_session → ...`
3. **draft 可被多 hook 串行修改**：每个插件看到的 draft 是上一个插件改完的状态。
   first non-None HookResult 终止链路（与现有约定一致）。
4. **`draft.content = []`**：表示 "drop message"，run 继续按 message=None 走。
   想终止 run 用 `HookResult.abort(reason)`。
5. **改 metadata 中的 message_id**：允许；事件 `message_id` 字段从最终 metadata
   读取。

#### 改动文件

- `hawi/plugin/hook_context.py` — `HookContext` 新字段、`UserMessageDraft`、
  `_ALLOWED_ACTIONS`、`validate_hook_result`
- `hawi/plugin/types.py` — `BeforeUserMessageMethod`、`PluginHooks`
- `hawi/plugin/decorators.py` — `before_user_message` decorator + 全部
  decorator docstring 增 "Allowed actions" 段
- `hawi/agent/agent.py` — 集成、`_invoke_user_message_hook`、所有
  `HookContext(...)` 调用点新字段、所有 `AgentMessageAddedEvent.create(...)`
  调用点新字段、`_ExecutionState.model`
- `hawi/events/agent_events.py` — `AgentMessageAddedEvent` 新字段
- `docs/hook_system.md` — 更新流程图、列表（追加 before_user_message）

---

## Phase 2：HookResult.strip

> 强依赖 Phase 0；与异步 tool executor 工作有交互，落地前协调。

### 2.1 设计

新 action `strip`：在 `before_tool_calling` 中返回，**既不执行工具，也不在
assistant message 中保留对应 `tool_call`，也不写 `tool_result`**。

与现有 `skip` 对比：

| Action | 执行 tool | 写 assistant tool_call | 写 tool_result |
|---|---|---|---|
| `skip(synthetic)` | 否 | 是 | 是（synthetic） |
| **`strip()`** | **否** | **否（裁剪）** | **否** |

### 2.2 协议安全

`strip` 必须在 assistant message **写入 context 之前**生效。否则 tool_call_id
已进 context，下一轮 model call 会因为缺 tool_result 报错（Anthropic 协议
甚至直接 reject 请求）。

当前流程（[agent.py:1959-1979](../hawi/agent/agent.py#L1959-L1979)）：

```
1. after_model_call hook
2. self._context.add_assistant_message(content=response_content)
3. emit AgentMessageAddedEvent
4. tool batch 循环：
     for tc in tool_calls:
         before_tool_calling(tc)  ← 这里返回 skip 决定不执行
         execute or synthesize
         after_tool_calling
```

新流程（Phase 2）：

```
1. after_model_call hook
2. PRESCAN：对每个 tool_call 跑 before_tool_calling，收集决策到
   per_call_decision: dict[tool_call_id, HookResult | None]
3. 根据 prescan 决策从 response_content 裁剪 strip 的 ToolCallPart
4. self._context.add_assistant_message(content=cleaned_content)
5. emit AgentMessageAddedEvent（content 是 cleaned_content）
6. tool batch 循环：
     for tc in cleaned_tool_calls:    # strip 的已被剔除
         decision = per_call_decision[tc.id]
         if decision is skip:        # 用 prescan 的 synthetic
             result = decision.tool_result
         else:                       # decision is None (run normally)
             result = await execute(tc)
         after_tool_calling
```

**关键：`before_tool_calling` 在每个 tool 上只跑一次（prescan 阶段）**。
tool batch 循环里直接查 cache，不重跑 hook，避免：
1. side effect 重复（hook 内打日志、改 context）
2. 不一致（prescan 决定 skip，循环里再跑一次返回 None，行为漂移）

### 2.3 签名

```python
# hawi/plugin/hook_context.py

@dataclass(frozen=True)
class HookResult:
    action: Literal[
        "skip", "abort", "replace_model",
        "reinvoke", "restart_turn", "strip",
    ]
    # ... existing fields ...

    @staticmethod
    def strip() -> HookResult:
        """Return from before_tool_calling to drop the tool_call entirely.

        Effects:
            - The tool is NOT executed.
            - The tool_call is REMOVED from the assistant message before
              it is written to context (model never sees it on next turn).
            - NO tool_result is written.

        Use case:
            Security policy enforcement where you want the model to be
            unaware that a particular tool_call was attempted. Most "block
            this tool" cases should prefer ``skip(synthetic_result)`` so
            the model can adapt with explicit feedback.

        Risks:
            The model may repeat the request on next turn since it has no
            record of the strip. Combine with prompt-level guidance or with
            a dedicated tool that the model is told to use instead.
        """
        return HookResult(action="strip")
```

### 2.4 主循环改造

[agent.py:1959-2030](../hawi/agent/agent.py#L1959-L2030) 区域大改。完整新版：

```python
# 现有 line 1959 区域起，after after_model_call hook 之后
response = MessageResponse(
    id=request_id, role="assistant", content=response_content,
    stop_reason=stop_reason, usage=usage,
)

_hr = await self._invoke_after_model_call(response, ctx_after)
# ... 现有 after_model_call 处理 ...

# === Phase 2 NEW: prescan tool_calls before writing assistant message ===
# 仅当 stop_reason 是 tool_use 且有 tool_calls 时才需要 prescan。
per_call_decision: dict[str, HookResult | None] = {}
stripped_ids: set[str] = set()

if tool_calls:
    for idx, tc in enumerate(tool_calls):
        tool_obj = self._plugin_manager.get_tool(tc["name"])
        ctx_before = HookContext(
            run_id=run_id,
            iteration=state.iteration,
            tool_call_id=tc["id"],
            tool=tool_obj,
            model=m,
            tool_call_index=idx,
            tool_batch_size=len(tool_calls),
            message_delta_start=initial_message_count,
        )
        decision = await self._invoke_before_tool_calling(
            tc["name"], tc["arguments"], ctx_before,
        )
        per_call_decision[tc["id"]] = decision
        if decision is not None and decision.action == "strip":
            stripped_ids.add(tc["id"])

# 裁剪 response_content：移除被 strip 的 tool_call parts
if stripped_ids:
    cleaned_content = [
        part for part in response_content
        if not (
            isinstance(part, dict)
            and part.get("type") == "tool_call"
            and part.get("id") in stripped_ids
        )
    ]
    cleaned_tool_calls = [tc for tc in tool_calls if tc["id"] not in stripped_ids]
else:
    cleaned_content = response_content
    cleaned_tool_calls = tool_calls

# Add assistant message (cleaned)
self._context.add_assistant_message(content=cleaned_content)
await self._emit_event(
    AgentMessageAddedEvent.create(
        run_id=run_id, role="assistant", content=cleaned_content,
        message_source="assistant",
        message_index=len(self._context.messages) - 1,
    ),
    event_bus,
)

# 没有剩余 tool_calls → 与原"无 tool_call"分支一致
if not cleaned_tool_calls:
    if await self._drain_pending_inputs_to_context(run_id, event_bus):
        continue
    duration_ms = (time.time() - start_time) * 1000
    await self._emit_event(
        AgentRunStopEvent.create(
            run_id=run_id, stop_reason=stop_reason or "end_turn",
            duration_ms=duration_ms, usage=cumulative_usage,
        ),
        event_bus,
    )
    break

# Tool batch 循环（与现有结构基本一致，只是 hook 决策来自 cache）
active_batch_tool_calls = [
    tc for tc in cleaned_tool_calls if tc not in self._current_tool_calls
]
completed_tool_call_ids: list[str] = []
completed_tool_records: list[ToolCallRecord] = []
self._current_tool_calls.extend(active_batch_tool_calls)
try:
    for tc in cleaned_tool_calls:
        if self._check_interrupt():
            break
        if tc in self._current_tool_calls:
            self._current_tool_calls.remove(tc)
        self._current_tool_calls.insert(0, tc)
        record = await self._execute_tool(
            tc, state,
            event_bus=event_bus,
            materialize_pending_steer=False,
            prescanned_decision=per_call_decision.get(tc["id"]),  # NEW
        )
        # ... 现有处理 ...
finally:
    ...
```

### 2.5 `_execute_tool` 改造

`_execute_tool` 增可选参数 `prescanned_decision: HookResult | None = None`。
非 None 时跳过 `_invoke_before_tool_calling` 调用，直接用 prescan 决策：

```python
async def _execute_tool(
    self,
    tool_call: ToolCallPart,
    state: _ExecutionState,
    *,
    event_bus: EventBus | None = None,
    materialize_pending_steer: bool = True,
    prescanned_decision: HookResult | None = None,
) -> ToolCallRecord:
    tool_name = tool_call["name"]
    arguments = tool_call["arguments"]
    tool_call_id = tool_call["id"]
    start_time = time.time()

    await self._emit_event(
        AgentToolCallEvent.create(
            run_id=state.run_id, tool_name=tool_name,
            arguments=arguments, tool_call_id=tool_call_id,
        ),
        event_bus,
    )

    tool = self._plugin_manager.get_tool(tool_name)
    audit_pending = False

    # === Hook decision: from prescan if provided, else fresh invoke ===
    if prescanned_decision is not None:
        _hr = prescanned_decision
    else:
        _before_ctx = HookContext(
            run_id=state.run_id,
            iteration=state.iteration,
            tool_call_id=tool_call_id,
            tool=tool,
            model=state.model,
            # tool_call_index/tool_batch_size 在直调 _execute_tool 时未知
            message_delta_start=state.message_delta_start,
        )
        _hr = await self._invoke_before_tool_calling(tool_name, arguments, _before_ctx)

    # strip 在 _execute_tool 不应出现（主循环已裁剪）；防御性 assert
    if _hr is not None and _hr.action == "strip":
        raise AgentError(
            "internal", "strip decision leaked into _execute_tool — "
            "this should be filtered out by the main loop prescan",
        )

    if _hr is not None and _hr.action == "skip":
        result = _hr.tool_result or ToolResult(
            success=False, error="Hook skipped tool without providing a result",
        )
    elif _hr is not None and _hr.action == "abort":
        result = ToolResult(
            success=False,
            error=f"Aborted by before_tool_calling hook: {_hr.reason}",
        )
        state.should_stop = True
    elif tool is None:
        # ... existing not-found path ...
    else:
        # ... existing audit / prepared / execute path ...
```

### 2.6 边界条件

1. **`audit=True` 工具与 strip**：
   - prescan 阶段返回 strip → tool_call 被裁，不进 audit pending list
   - prescan 返回 skip → audit pending 不被触发（synthetic 直接当结果）
   - prescan 返回 None → audit 正常走 pending 路径
   - 决策：`audit=True` + strip 是允许的；插件清楚自己在做什么。

2. **多 plugin hook 链中 `strip` 的优先级**：
   first non-None HookResult 终止链路。如果 plugin A 返回 strip 而 plugin B
   想看到这个 tool_call 也无法看到——这是 hook chain 既有行为。

3. **prescan 阶段 hook 的副作用问题**：
   prescan 把 hook 的"调用一次"提前到 assistant message 写入前。如果 hook
   原本写日志 "tool X about to run"，行为不变（只是时间点稍早）。如果 hook
   读取 `agent.context.messages` 想看到 assistant message，**会看不到**——
   assistant message 此时尚未写入。
   - 决策：文档明确 "before_tool_calling 中 `agent.context.messages` 不包含
     正在处理的 assistant message"。这其实在 Phase 0 之前已经如此（before_tool_calling
     在 add_assistant_message 之前），只是 strip 的引入让这个时序更加明显。

4. **`AgentToolCallEvent` 时序**：
   当前 [agent.py:2502-2510](../hawi/agent/agent.py#L2502-L2510) 在
   `_execute_tool` 一开始就 emit。`strip` 的 tool_call 不进入 `_execute_tool`，
   所以**不会发 `AgentToolCallEvent`**——这是符合预期的：模型层面也不知道这次
   调用发生过。

5. **`current_tool_calls` 跟踪**：
   `self._current_tool_calls` 是用于 interrupt 时合成 tool_result 的列表
   ([agent.py:_recover_unanswered_tool_calls](../hawi/agent/agent.py))。strip
   后的 tool_call 不应进入此列表（main loop 用 `cleaned_tool_calls` 而不是
   `tool_calls` 来 extend）。

6. **`_drain_pending_inputs_to_context`**：
   当 cleaned_tool_calls 为空（全部 strip）时，走"无 tool_call"分支，与
   model 直接 `end_turn` 一致。

### 2.7 改动文件

- `hawi/plugin/hook_context.py` — `HookResult.strip` 静态方法 + action enum 加 strip
- `hawi/agent/agent.py` — main loop 重组（prescan + clean + batch loop）；
  `_execute_tool` 增 `prescanned_decision` 参数；`_ExecutionState` 增
  `model` / `message_delta_start` 字段
- `hawi/plugin/decorators.py` — `before_tool_calling` docstring 加 strip
  说明、显式化"原地修改 arguments 是合法操作"
- `docs/hook_system.md` — 流程图加 strip 分支

### 2.8 测试

新建 `test/unit/test_hook_strip.py`：

```python
class _StripPlugin(HawiPlugin):
    @before_tool_calling
    def hook(self, agent, tool_name, arguments, ctx):
        if tool_name == "danger":
            return HookResult.strip()
        return None


class _SkipPlugin(HawiPlugin):
    @before_tool_calling
    def hook(self, agent, tool_name, arguments, ctx):
        if tool_name == "block":
            return HookResult.skip(ToolResult(success=False, output="blocked"))
        return None


@pytest.mark.asyncio
async def test_strip_removes_tool_call_from_assistant_message(stub_model_calls_one_tool):
    """单 tool batch 的 strip 让 assistant message 没有任何 tool_call。"""
    agent = HawiAgent(model=stub_model_calls_one_tool("danger"), plugins=[_StripPlugin()])
    await agent.arun("hello")
    assistant_msgs = [m for m in agent.context.messages if m["role"] == "assistant"]
    last = assistant_msgs[-1]
    tool_calls = [p for p in last["content"] if p.get("type") == "tool_call"]
    assert tool_calls == []


@pytest.mark.asyncio
async def test_strip_does_not_emit_tool_call_event(stub_model_calls_one_tool, capture_events):
    agent = HawiAgent(model=stub_model_calls_one_tool("danger"), plugins=[_StripPlugin()])
    capture_events(agent, types=["agent.tool_call", "agent.tool_result"])
    await agent.arun("hello")
    assert not capture_events.collected


@pytest.mark.asyncio
async def test_strip_no_tool_result_in_context(stub_model_calls_one_tool):
    agent = HawiAgent(model=stub_model_calls_one_tool("danger"), plugins=[_StripPlugin()])
    await agent.arun("hello")
    tool_msgs = [m for m in agent.context.messages if m["role"] == "tool"]
    assert tool_msgs == []


@pytest.mark.asyncio
async def test_strip_one_keep_others(stub_model_calls_three_tools):
    """三个 tool batch：strip 第一个、skip 第二个、正常执行第三个。"""
    class P(HawiPlugin):
        @before_tool_calling
        def hook(self, agent, tool_name, arguments, ctx):
            if tool_name == "danger": return HookResult.strip()
            if tool_name == "block": return HookResult.skip(ToolResult(success=True, output="blocked"))
            return None

    agent = HawiAgent(model=stub_model_calls_three_tools(["danger", "block", "ok"]), plugins=[P()])
    await agent.arun("hello")
    assistant = next(m for m in reversed(agent.context.messages) if m["role"] == "assistant")
    tcs = [p for p in assistant["content"] if p.get("type") == "tool_call"]
    assert sorted(p["name"] for p in tcs) == ["block", "ok"]  # danger gone
    tool_msgs = [m for m in agent.context.messages if m["role"] == "tool"]
    assert len(tool_msgs) == 2  # block + ok


@pytest.mark.asyncio
async def test_before_tool_calling_invoked_once_per_tool(stub_model_calls_one_tool):
    """prescan + 主循环中 hook 在每个 tool 上只调用一次。"""
    call_log = []

    class P(HawiPlugin):
        @before_tool_calling
        def hook(self, agent, tool_name, arguments, ctx):
            call_log.append(tool_name)
            return None

    agent = HawiAgent(model=stub_model_calls_one_tool("ok"), plugins=[P()])
    await agent.arun("hello")
    assert call_log.count("ok") == 1


@pytest.mark.asyncio
async def test_strip_all_tools_in_batch_yields_end_turn(stub_model_calls_one_tool):
    """全部 tool 都 strip 后下一轮直接走 no-tool 分支。"""
    class P(HawiPlugin):
        @before_tool_calling
        def hook(self, agent, tool_name, arguments, ctx):
            return HookResult.strip()

    # stub 第二轮直接 end_turn
    agent = HawiAgent(model=stub_model_calls_one_tool("ok", end_turn_next=True), plugins=[P()])
    result = await agent.arun("hello")
    assert result.stop_reason == "end_turn"
```

---

## Phase 3：Compact 事务化

> 强依赖 Phase 1（HookContext / events）+ todo.md 中 "Message id 一等公民" 项目。

### 3.1 设计原则

- `before_compact` 通过 **`CompactDraft`** 表达"想保留哪些消息"，**不**直接
  改 `agent.context.messages`
- 框架在 hook 后重算预算、原子提交（snapshot/rollback 包裹
  `compact_with_summary`）
- `after_compact` 仅观察，不接受返回值
- pinned 消息总 token 上限：keep_last 区域 token 量的 **30%**；超出 raise
  `CompactBudgetExceededError`，原 messages 完全不变

### 3.2 新类型

```python
# hawi/agent/context.py

@dataclass(frozen=True)
class _CompactDraftReadOnly:
    """只读视图，避免插件意外破坏 plan input。"""
    compactable_messages: tuple[Message, ...]
    kept_tail: tuple[Message, ...]
    keep_last: int
    target_tokens: int
    estimated_compactable_tokens: int


@dataclass
class CompactDraft:
    """Mutable plan for a compaction about to happen.

    Read-only inputs (do not modify):
        compactable_messages: messages the framework plans to compact.
        kept_tail: messages the framework plans to keep verbatim.
        keep_last, target_tokens: framework decisions.
        estimated_compactable_tokens: tokens that will be removed.

    Writable fields (modify these to influence compaction):
        pinned_message_ids: set of message ids to PIN — they will move from
            compactable_messages to kept_tail. Each pinned message's tokens
            count against a budget cap (30% of kept_tail tokens). Exceeding
            the cap raises CompactBudgetExceededError; original messages
            remain unchanged.
        rescue_notes: list of strings to append to the generated summary
            (after framework summary text).
    """
    compactable_messages: tuple[Message, ...]
    kept_tail: tuple[Message, ...]
    keep_last: int
    target_tokens: int
    estimated_compactable_tokens: int
    pinned_message_ids: set[str] = field(default_factory=set)
    rescue_notes: list[str] = field(default_factory=list)


class CompactBudgetExceededError(Exception):
    """pinned messages 超出预算上限时抛出。"""
    def __init__(self, pinned_tokens: int, budget: int):
        self.pinned_tokens = pinned_tokens
        self.budget = budget
        super().__init__(
            f"Pinned messages total {pinned_tokens} tokens, "
            f"exceeds budget {budget} (30% of kept_tail).",
        )
```

### 3.3 新 hook

```python
# hawi/plugin/decorators.py

def before_compact(func):
    """Hook called before automatic context compaction.

    Args:
        agent: HawiAgent instance.
        draft: CompactDraft. Modify ``draft.pinned_message_ids`` to keep
            specific messages, or ``draft.rescue_notes`` to append notes
            to the generated summary.
        ctx: HookContext with run_id and iteration.

    Allowed actions:
        - None to proceed with (possibly modified) draft.
        - HookResult.abort(reason) to cancel this compaction. Original
          messages remain unchanged. Note that the agent loop will continue
          with the over-budget context, which may then fail the next model
          call. Use only when the plugin has its own remediation plan.

    Hook chain:
        Multiple plugins' hooks share the same draft. pinned_message_ids
        and rescue_notes are unioned/concatenated naturally via mutation.
    """
    setattr(func, "_is_hook", True)
    setattr(func, "_hook_type", "before_compact")
    return func


def after_compact(func):
    """Hook called after automatic context compaction succeeds.

    Args:
        agent: HawiAgent instance.
        record: ContextCompactionRecord describing what was compacted.
        ctx: HookContext with run_id, iteration, duration_ms.

    Returns:
        None (return value is ignored).

    Note:
        Not called when before_compact aborts or the compaction fails.
    """
    setattr(func, "_is_hook", True)
    setattr(func, "_hook_type", "after_compact")
    return func
```

### 3.4 集成与事务

`hawi/agent/context.py` `compact_with_summary` 改造为支持 pinned:

```python
def compact_with_summary(
    self,
    summary: str,
    *,
    keep_last: int = 8,
    summary_prefix: str = CONTEXT_COMPACTION_SUMMARY_PREFIX,
    pinned_message_ids: set[str] | None = None,
    rescue_notes: list[str] | None = None,
) -> ContextCompactionRecord | None:
    """Replace older history with a summary while preserving pinned messages.

    Pinned messages (by id) are moved from the compactable region to be
    inserted right after the summary message, in original order.
    """
    tail_start = self.compaction_tail_start(keep_last)
    if tail_start <= 0:
        return None

    tokens_before = self.estimate_tokens()
    older = self.messages[:tail_start]
    kept_tail = list(self.messages[tail_start:])
    pinned_ids = pinned_message_ids or set()

    # Extract pinned (preserve original order)
    pinned_messages: list[Message] = []
    truly_replaced: list[Message] = []
    for msg in older:
        msg_id = (msg.get("metadata") or {}).get("message_id")
        if msg_id is not None and msg_id in pinned_ids:
            pinned_messages.append(deepcopy(msg))
        else:
            truly_replaced.append(deepcopy(msg))

    summary_message = self._make_compaction_summary_message(
        summary if not rescue_notes else (
            summary + "\n\n--- rescue notes ---\n" + "\n".join(rescue_notes)
        ),
        summary_prefix=summary_prefix,
    )

    # Atomic: build new list, then assign
    new_messages = [summary_message, *pinned_messages, *kept_tail]
    self.messages = new_messages
    tokens_after = self.estimate_tokens()

    record = ContextCompactionRecord(
        summary=summary,
        replaced_messages=truly_replaced,
        kept_messages=len(kept_tail) + len(pinned_messages),
        tokens_before=tokens_before,
        tokens_after=tokens_after,
    )
    self.compaction_records.append(record)
    return record
```

### 3.5 `_maybe_auto_compact` 改造

`hawi/agent/agent.py` [Line 996-1015](../hawi/agent/agent.py#L996-L1015)：

```python
async def _maybe_auto_compact(self, model: Model, state: _ExecutionState) -> bool:
    cfg = self._auto_compact
    if not cfg.enabled:
        return False
    if self.has_active_tool_calls:
        return False
    if len(self._context.messages) < cfg.min_messages:
        return False
    if self._context.estimate_tokens() < cfg.token_limit():
        return False

    record = await self.acompact(model=model, config=cfg)
    if record is not None:
        state.iteration = max(state.iteration, 0)
    return record is not None


async def acompact(self, *, model, config) -> ContextCompactionRecord | None:
    keep_last = config.keep_last
    target_tokens = config.token_limit()

    # === Phase 3 NEW: build draft + run before_compact hook ===
    tail_start = self._context.compaction_tail_start(keep_last)
    if tail_start <= 0:
        return None

    compactable = tuple(self._context.messages[:tail_start])
    kept_tail = tuple(self._context.messages[tail_start:])
    estimated_compactable_tokens = sum(
        estimate_message_tokens(m) for m in compactable
    )

    draft = CompactDraft(
        compactable_messages=compactable,
        kept_tail=kept_tail,
        keep_last=keep_last,
        target_tokens=target_tokens,
        estimated_compactable_tokens=estimated_compactable_tokens,
    )

    ctx = HookContext(
        run_id=self._active_execution_state.run_id if self._active_execution_state else "unknown",
        iteration=self._active_execution_state.iteration if self._active_execution_state else 0,
        message_delta_start=self._active_execution_state.message_delta_start if self._active_execution_state else 0,
    )

    _hr = await self._invoke_compact_hook("before_compact", draft, ctx)
    if _hr is not None and _hr.action == "abort":
        return None  # 取消 compact，messages 不变

    # === Budget check ===
    if draft.pinned_message_ids:
        pinned_tokens = self._estimate_pinned_tokens(draft)
        kept_tail_tokens = sum(estimate_message_tokens(m) for m in kept_tail)
        budget = int(kept_tail_tokens * 0.30)
        if pinned_tokens > budget:
            raise CompactBudgetExceededError(pinned_tokens, budget)

    # === Generate summary ===
    summary = await self._generate_compaction_summary(model, ...)

    # === Atomic: snapshot before mutation ===
    snapshot = list(self._context.messages)
    snapshot_records = list(self._context.compaction_records)
    try:
        record = self._context.compact_with_summary(
            summary,
            keep_last=keep_last,
            pinned_message_ids=draft.pinned_message_ids,
            rescue_notes=draft.rescue_notes,
        )
    except Exception:
        # Rollback
        self._context.messages = snapshot
        self._context.compaction_records = snapshot_records
        raise

    if record is None:
        return None

    # === Phase 3 NEW: emit message_added(compacted) + after_compact ===
    await self._emit_event(
        AgentMessageAddedEvent.create(
            run_id=ctx.run_id, role="user",  # summary 实际放在 messages[0]
            content=self._context.messages[0]["content"],
            message_source="compacted",
            message_index=0,
        ),
        None,
    )

    duration_ms = 0.0  # acompact 内可累计 timing
    after_ctx = HookContext(
        run_id=ctx.run_id, iteration=ctx.iteration,
        duration_ms=duration_ms, message_delta_start=ctx.message_delta_start,
    )
    await self._invoke_compact_hook("after_compact", record, after_ctx)

    return record


async def _invoke_compact_hook(self, hook_type: str, payload, ctx: HookContext):
    for hook in self._plugin_manager.get_hooks(hook_type):
        result = hook(self, payload, ctx)
        if inspect.isawaitable(result):
            result = await result
        result = validate_hook_result(hook_type, result)
        if result is not None:
            return result
    return None


def _estimate_pinned_tokens(self, draft: CompactDraft) -> int:
    total = 0
    for msg in draft.compactable_messages:
        msg_id = (msg.get("metadata") or {}).get("message_id")
        if msg_id and msg_id in draft.pinned_message_ids:
            total += estimate_message_tokens(msg)
    return total
```

### 3.6 `message_id` 一等公民依赖

Phase 3 强依赖 todo.md "Message 类型增加一等公民 id" 项目。该项目落地前：
- `pinned_message_ids` 仍可用，但只能钉住 **metadata 中已有 `message_id`**
  的消息（runner/steer 路径已在传，但普通 user/assistant message 暂时
  没有）
- 普通插件钉不住"之前模型生成的 assistant 消息"
- 文档明示这一限制

待 message_id 一等公民项目落地后，`Message.metadata` 自动带 `message_id`，
本 phase 无需再改。

### 3.7 `_ExecutionState` 增字段

```python
@dataclass
class _ExecutionState:
    # existing fields...
    model: Model | None = None              # Phase 1 already added
    message_delta_start: int = 0            # Phase 1 already added
    pending_reinvoke_message: str | list[ContentPart] | None = None  # Phase 0
```

### 3.8 改动文件

- `hawi/agent/context.py` — `CompactDraft`、`CompactBudgetExceededError`、
  `compact_with_summary` 增 pinned/rescue 参数
- `hawi/agent/agent.py` — `_maybe_auto_compact` / `acompact` 改造、
  `_invoke_compact_hook`、`_estimate_pinned_tokens`
- `hawi/plugin/decorators.py` — `before_compact` / `after_compact`
- `hawi/plugin/types.py` — `BeforeCompactMethod` / `AfterCompactMethod`
- `hawi/plugin/hook_context.py` — `_ALLOWED_ACTIONS` 加两条
- `docs/hook_system.md` / `docs/context.md` — 文档

### 3.9 测试

新建 `test/unit/test_hook_compact.py`：

```python
@pytest.mark.asyncio
async def test_before_compact_pin_keeps_message():
    """钉住的消息不被压缩，留在 kept_tail。"""
    ...

@pytest.mark.asyncio
async def test_pinned_over_budget_raises():
    """钉住消息超过 30% 预算 raise CompactBudgetExceededError、messages 不变。"""
    ...

@pytest.mark.asyncio
async def test_before_compact_abort_cancels():
    """abort 时 messages、compaction_records 都不变。"""
    ...

@pytest.mark.asyncio
async def test_compact_failure_rollback():
    """compact_with_summary raise 时 messages 完整回滚。"""
    ...

@pytest.mark.asyncio
async def test_after_compact_receives_record():
    """after_compact 收到正确的 ContextCompactionRecord。"""
    ...

@pytest.mark.asyncio
async def test_rescue_notes_appended_to_summary():
    ...

@pytest.mark.asyncio
async def test_message_compacted_event_fired():
    """compact 完成后 AgentMessageAddedEvent(message_source='compacted') 发布。"""
    ...

@pytest.mark.asyncio
async def test_two_plugins_pin_set_unioned():
    """两个插件各自 pin，最终 pinned_message_ids 取并集。"""
    ...
```

---

## Phase 4：错误 / 中断观察

> 与其它 phase 独立。可与 Phase 1 并行。

### 4.1 `on_interrupt` hook

#### 签名

```python
# hawi/plugin/decorators.py

def on_interrupt(func):
    """Hook called when the agent run is interrupted.

    Args:
        agent: HawiAgent instance.
        reason: str describing why the interrupt fired (from
            agent.interrupt(reason=...) or the framework default).
        ctx: HookContext with run_id, iteration.

    Triggered by:
        - agent.interrupt() / runner.interrupt()
        - asyncio.CancelledError during the agent loop

    Returns:
        None (return value ignored).

    Use case:
        Cleanup external resources (subprocesses, sockets, file handles).
        Plugins must not block here — keep the work bounded.
    """
    setattr(func, "_is_hook", True)
    setattr(func, "_hook_type", "on_interrupt")
    return func
```

#### 集成位点

[agent.py:2065-2080](../hawi/agent/agent.py#L2065-L2080) 的 `CancelledError`
分支 + interrupt API。每个触发点构造 HookContext 并调
`_invoke_interrupt_hook`：

```python
except asyncio.CancelledError:
    reason = self._last_interrupt_reason or "cancelled"
    interrupt_ctx = HookContext(
        run_id=run_id, iteration=state.iteration,
        message_delta_start=state.message_delta_start,
    )
    # 先跑 hook（cleanup），再做 unanswered tool recovery
    try:
        await self._invoke_interrupt_hook(reason, interrupt_ctx)
    except Exception as e:
        # hook 内部异常：log 但不阻断 cancel 路径
        logger.exception("on_interrupt hook raised: %s", e)
    await self._recover_unanswered_tool_calls(...)
    ...
    raise
```

`_invoke_interrupt_hook`：

```python
async def _invoke_interrupt_hook(self, reason: str, ctx: HookContext):
    for hook in self._plugin_manager.get_hooks("on_interrupt"):
        result = hook(self, reason, ctx)
        if inspect.isawaitable(result):
            result = await result
        # on_interrupt 不接受返回值，忽略
```

### 4.2 `AgentModelErrorEvent`（事件，非 hook）

```python
# hawi/events/agent_events.py

class AgentModelErrorEvent(Event):
    run_id: str
    request_id: str
    error_type: str          # exception class name
    error_message: str
    attempt: int             # 0-based attempt index in retry sequence
    will_retry: bool         # framework decision

    @classmethod
    def create(cls, run_id, request_id, error, attempt, will_retry):
        return cls(
            type="agent.model_error", source="agent",
            run_id=run_id, request_id=request_id,
            error_type=type(error).__name__,
            error_message=str(error),
            attempt=attempt, will_retry=will_retry,
        )
```

#### 发布位点

`hawi/agent/agent.py` `_call_model_with_retry`
（[Line 2188+](../hawi/agent/agent.py#L2188)），每次 exception 处理分支：

```python
except SomeModelError as e:
    will_retry = ...  # 由 policy 决定
    await self._emit_event(
        AgentModelErrorEvent.create(
            run_id=state.run_id, request_id=request_id,
            error=e, attempt=attempt, will_retry=will_retry,
        ),
        event_bus,
    )
    if will_retry:
        attempt += 1
        continue
    raise
```

### 4.3 `AgentToolErrorEvent`

```python
class AgentToolErrorEvent(Event):
    run_id: str
    tool_call_id: str
    tool_name: str
    error_type: str
    error_message: str
    duration_ms: float
```

发布位点：[agent.py:2620-2624](../hawi/agent/agent.py#L2620-L2624) 的
`ToolExecutionError` 分支：

```python
except Exception as e:
    err = ToolExecutionError(...)
    await self._emit_event(
        AgentToolErrorEvent.create(
            run_id=state.run_id, tool_call_id=tool_call_id,
            tool_name=tool_name, error=e,
            duration_ms=(time.time() - start_time) * 1000,
        ),
        event_bus,
    )
    result = ToolResult(success=False, error=...)
```

### 4.4 改动文件

- `hawi/plugin/decorators.py` — `on_interrupt`
- `hawi/plugin/types.py` — `OnInterruptMethod`
- `hawi/plugin/hook_context.py` — `_ALLOWED_ACTIONS["on_interrupt"] = frozenset()`
- `hawi/agent/agent.py` — interrupt 路径 + 错误事件发布
- `hawi/events/agent_events.py` — 两个新事件
- `hawi/agent/runner/interceptor.py`（如有事件类型注册）

### 4.5 测试

```python
# test/unit/test_hook_interrupt.py
@pytest.mark.asyncio
async def test_on_interrupt_fires_with_reason():
    captured = []
    class P(HawiPlugin):
        @on_interrupt
        def hook(self, agent, reason, ctx): captured.append(reason)

    agent = HawiAgent(model=long_running_stub_model, plugins=[P()])
    task = asyncio.create_task(agent.arun("hello"))
    await asyncio.sleep(0.1)
    agent.interrupt(reason="user requested")
    with pytest.raises(asyncio.CancelledError):
        await task
    assert captured == ["user requested"]


@pytest.mark.asyncio
async def test_python_interpreter_close_on_interrupt():
    plugin = PythonInterpreterPlugin()
    agent = HawiAgent(model=long_running_stub_model, plugins=[plugin])
    task = asyncio.create_task(agent.arun("hello"))
    await asyncio.sleep(0.1)
    agent.interrupt()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert plugin._closed  # close() called via on_interrupt


# test/unit/test_error_events.py
@pytest.mark.asyncio
async def test_model_error_event_published_on_retry():
    ...

@pytest.mark.asyncio
async def test_tool_error_event_published_on_exception():
    ...
```

---

## 既有插件迁移

`hawi_plugins/` 下 10 个插件的影响详表。Phase 0 修复 bug 让某些插件
"从无效变为按设计运行"，需要回归测试覆盖。

### `workflow_plugin`

**影响**：Phase 0 让 [workflow_plugin/plugin.py:215-250](../hawi_plugins/workflow_plugin/plugin.py#L215-L250)
的 `gate_guard`（@after_tool_calling）真正生效。

**改动**：无代码改动。

**测试**：新增 `test/integration/test_workflow_gate_guard.py` 覆盖：
- approve 路径：gate 通过 → reinvoke → 下一节点
- reject 路径：gate 驳回 → reinvoke 让 agent 修正
- STATUS: FAILED 重试机制（[workflow_plugin/plugin.py:670-715](../hawi_plugins/workflow_plugin/plugin.py#L670-L715)）

### `plan_plugin`

**影响**：[plan_plugin/plugin.py:259+](../hawi_plugins/plan_plugin/plugin.py#L259) 的
`@after_tool_calling` 同样在 Phase 0 之前不生效（如有 `HookResult.reinvoke`
返回路径）。

**改动**：无代码改动；增回归测试。

### `python_interpreter`

**影响**：Phase 4 上线后改用 `on_interrupt` 取代部分 close 路径，
更细粒度地保证 interrupt 时子进程被回收。

**改动**：

```python
# hawi_plugins/python_interpreter/plugin.py

from hawi.plugin import HawiPlugin
from hawi.plugin.decorators import on_interrupt

class PythonInterpreterPlugin(HawiPlugin):
    # existing __init__ etc.

    @on_interrupt
    def cleanup_on_interrupt(self, agent, reason, ctx):
        """Close all sub-interpreters when run is interrupted."""
        # close() is idempotent (line 426-429: if self._closed return)
        self.close()
```

**保留** [plugin.py:85-88](../hawi_plugins/python_interpreter/plugin.py#L85-L88) 的
`exit_handler.register(cleanup_wrapper, ...)`：那是进程退出时的兜底，
on_interrupt 是 run 中断时的更早清理。两者不冲突，因为 close 是幂等的。

### 其它插件

| 插件 | 受影响项 | 改动 |
|---|---|---|
| `environ_prompt_plugin` | 仅 before_session/before_conversation 注入 prompt | 零改动 |
| `mcp_plugin` | 仅作 tool 提供方 | 零改动 |
| `skills_plugin` | 仅 before_session 注入 prompt | 零改动 |
| `filesystem_plugin` | 仅 tools | 零改动 |
| `shell_plugin` | 仅 tools | 零改动 |
| `web` | 仅 tools | 零改动 |
| `subagent_plugin` | 仅 tools | 零改动 |

### `HookContext` 字段增加是否破坏插件

`HookContext` 是 frozen dataclass，新增字段全部 default=None。下列用法
**不破坏**：

- 关键字访问 `ctx.run_id` ✓
- 位置参数构造 `HookContext(run_id, iteration)` ✓（旧字段在前，新字段全 default）
- pattern matching `match ctx: case HookContext(run_id=r): ...` ✓（`__match_args__` 锁定到原 6 字段）

下列用法 **会破坏**（grep 全仓库未发现）：

- `dataclasses.replace(ctx, ...)` 之后期望字段集不变
- 直接用 `dataclasses.fields(ctx)` 序列化所有字段

如果将来发现破坏点，在 PR 中单独处理。

---

## 测试策略

### 文件组织

```
test/
├── unit/
│   ├── test_hook_action_consumption.py    # Phase 0
│   ├── test_hook_user_message.py           # Phase 1.3
│   ├── test_event_message_metadata.py      # Phase 1.2
│   ├── test_hook_context_fields.py         # Phase 1.1
│   ├── test_hook_strip.py                  # Phase 2
│   ├── test_hook_compact.py                # Phase 3
│   ├── test_hook_interrupt.py              # Phase 4
│   ├── test_error_events.py                # Phase 4
│   └── test_hook_chain_semantics.py        # 跨 phase：链路、validation
└── integration/
    ├── test_workflow_gate_guard.py         # Phase 0 回归
    └── test_python_interpreter_interrupt.py # Phase 4 集成
```

### 共享 fixture

新增 `test/conftest.py`（如果不存在）：

```python
@pytest.fixture
def stub_model():
    """简单 stub：第一次 call 直接 end_turn。"""
    ...

@pytest.fixture
def stub_model_calls_one_tool():
    """第一次返回单 tool_call，第二次根据 end_turn_next 决定。"""
    def _factory(name: str, *, end_turn_next: bool = True): ...
    return _factory

@pytest.fixture
def stub_model_calls_three_tools():
    """第一次返回 3 个 tool_call。"""
    def _factory(names: list[str]): ...
    return _factory

@pytest.fixture
def long_running_stub_model():
    """模拟可中断的长 stream。"""
    ...

@pytest.fixture
def capture_events():
    """注册临时 EventBus 订阅，把指定 type 的事件收集到 list。"""
    ...
```

### 覆盖率目标

每 Phase 落地后：

- 新 hook：100% 路径（fire / abort / 无插件）
- 新事件：发布点全覆盖
- 边界条件：每个 "边界条件" 小节至少一个 test
- 既有插件回归：至少 smoke test

---

## Phase 之间的依赖与发布

```
Phase 0 ──┬── Phase 1 ──┬── Phase 2
          │             │
          │             └── Phase 3（额外依赖 message_id 一等公民）
          │
          └── Phase 4
```

**发布建议**：
- Phase 0：独立 patch release（修 bug，可立即合并）
- Phase 1：minor release（新 hook + 事件字段；向后兼容）
- Phase 2：minor release；落地前与异步 tool executor worktree 同步
- Phase 3：minor release；落地前确认 message_id 一等公民已合并
- Phase 4：minor release；与 Phase 1 并行开发可能

---

## 已拒绝提案（决策记录）

| 提案 | 拒绝理由 |
|---|---|
| `before_tool_batch` / `after_tool_batch` | 与异步 tool executor 工作冲突；当前没有跨 tool 全局分析的强需求 |
| `route(tool_name)` / `ToolCallDraft.name` 改写 | 污染 audit/trace；所有用例都能通过 tool 内 dispatch 或 plugin-owned wrapper tool 实现 |
| `replace_request(MessageRequest)` | 鼓励插件每次 model call 重写 system prompt，破坏 prompt cache |
| `inject_tool_call(name, args)` | 让插件伪造 model 输出，污染 conversation history、绕过审计 |
| `continue_with` 与 `reinvoke` 合并 | 语义差距过大（同 run continue vs 重启 run），flag 控制会让插件作者搞混 run_id / events / session hooks |
| `on_message_added` 作为 hook | 这是观察点，应通过 `AgentMessageAddedEvent.message_source` 字段满足，避免阻塞写入路径 |
| streaming token-level hook | `ModelStream*Event` 已覆盖；除非有"阻断流"需求否则不引入 |
| `before_iteration` / `after_iteration` | YAGNI；`before_model_call` + `HookContext.iteration` 已足够 |
| `ToolBatchDraft.execution_order` | 工具调用可能有前后依赖，不假设可重排；并行执行另开提案 |
| `HookContext.is_streaming` | agent 上有 `_streaming` 属性可读，不重复暴露 |

---

## 悬而未决

- **`continue_with`**（轻量 reinvoke 替代品，不重启 run）：单独评估，不在
  本计划内。`reinvoke` 当前递归 `_arun_internal` 的实现细节泄漏（栈深、
  iteration 重置、cumulative_usage 截断）值得单独重构。
- **异步 tool executor 落地后**是否再开 batch hook 提案：取决于异步 executor
  暴露的执行编排点，届时再评估。
- **Phase 3 强依赖**的 `message_id` 一等公民项目（todo.md 已列）：先行落地。
  Phase 3 的 PR 必须显式声明 message_id 已合入。
- **hook 优先级 / 排序**：当前 hook 链按注册顺序执行。如果将来出现"插件
  A 必须先于 B 跑"的需求，再设计 priority 字段。本计划范围内不做。
- **hook timeout**：插件长 hook 可能阻塞主循环，目前无超时。本计划范围内
  不做；如需，独立提案。
