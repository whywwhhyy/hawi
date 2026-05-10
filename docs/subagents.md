# SubAgent 设计草案

## 目标

SubAgent 是 Hawi 的 core 级编排原语：主 agent、插件和工作流都可以直接创建、驱动、查询和关闭子 agent；模型侧工具只是这层 API 的轻量包装。

第一版目标保持克制：

- 支持从父 agent fork 上下文创建子 agent。
- 支持创建全新上下文的子 agent。
- 创建时可配置模型、插件、system prompt、工作目录、初始任务、预算和输出协议。
- 子 agent 默认后台运行，由 scheduler 驱动，并能查询状态、继续对话、读取结果和关闭。
- 事件、审计和持久化预留明确落点，但不把 multi-agent 变成默认执行架构。

## 已有基础

现有代码已经提供了几块可复用能力：

- `HawiAgent.clone()` / `fork()` 会复制上下文、克隆插件、复用模型配置。
- `HawiScheduler` 已经能后台消费队列、支持普通/高优先级/紧急消息、支持中断。
- `PluginManager.clone()` 和 `plugin_factories` 已经能让插件在 fork 时获得隔离实例。
- `AgentContext.snapshot/load_snapshot` 和 `SessionManager` 已经建立了持久化模式。
- `WorkflowPlugin` 当前的 `SubAgentReviewer` 已经用 `agent.clone()` 做过最小 sub-agent reviewer，但缺少统一生命周期和状态管理。

## 核心对象

建议新增 `hawi/agent/subagent.py`，由 `HawiAgent` 暴露 `agent.subagents`。

```python
handle = await agent.subagents.spawn(
    mode="fork",
    name="reviewer",
    role="reviewer",
    initial_prompt="Review this plan and return JSON.",
)

await agent.subagents.send(handle.id, "Please focus on API stability.")
status = agent.subagents.status(handle.id)
result = await agent.subagents.wait(handle.id, timeout=60)
await agent.subagents.close(handle.id)
```

### `SubAgentSpec`

`spawn()` 接受一个 dataclass 或等价关键字参数：

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `mode` | `"fork"` | `"fork"` 复制父上下文；`"fresh"` 创建空上下文 |
| `name` | 自动生成 | 人类可读名称，便于日志和 GUI 展示 |
| `role` | `"general"` | 角色预设：`planner`、`reviewer`、`explorer`、`implementer`、`critic`、`summarizer` |
| `model` | 父 agent 模型 | 可传模型名或 `Model` 实例 |
| `system_prompt` | 角色默认值 | 显式覆盖子 agent system prompt |
| `plugins` | 继承父插件 | 插件策略，见下文 |
| `working_dir` | `None` | 子 agent 的逻辑工作目录；不使用进程级 `os.chdir()` |
| `initial_prompt` | `None` | 创建后立即入队的首条任务 |
| `initial_plan` | `None` | 结构化计划，可渲染进首条任务或作为 metadata |
| `limits` | 保守默认值 | 最大轮数、最长运行时间、最大 tool call 数、最大递归深度 |
| `result_contract` | `"text"` | 期望结果：`text`、`json`、`plan`、`review`、`diff`、`artifact` |
| `ownership` | 空 | 可读/可写文件或模块范围，用于并行修改约束 |
| `metadata` | `{}` | 调用方自定义审计信息 |

### 上下文模式

`fork`：

- 基于 `parent.clone()`。
- 深拷贝消息历史、system prompt、cache 配置和工具定义。
- 插件按现有 `clone()` / factory 规则隔离。
- 适合 reviewer、critic、summarizer 这类需要理解父上下文的任务。

`fresh`：

- 新建 `HawiAgent(model=..., plugins=..., system_prompt=...)`。
- 默认没有父对话历史，只注入角色 prompt、初始任务和显式材料。
- 适合 explorer、implementer、实验性分支和低污染探索。

后续可加两个受控模式：

- `summary`：先对父上下文做 handoff summary，再给子 agent。
- `messages`：调用方显式传入一组消息或 artifact 引用。

### 插件策略

插件配置不要只做布尔继承，建议用小型策略对象：

```python
SubAgentPluginPolicy(
    inherit=True,
    allowlist=None,
    denylist=["shell"],
    extra_plugins=[],
    extra_factories=[],
    tool_allowlist=None,
    tool_denylist=None,
)
```

第一版可以实现最小集：

- `inherit=True/False`
- `extra_plugins`
- `extra_factories`

`allowlist/denylist` 和 tool 级过滤作为下一步落地，避免第一版卡在权限系统重构上。

## 角色默认 System Prompt

角色预设只提供薄默认值，调用方可以完全覆盖：

- `general`：独立完成指定任务，必要时说明假设和不确定性。
- `planner`：输出可执行计划，标明依赖、风险、验收条件。
- `reviewer`：优先发现缺陷、回归风险和缺失测试。
- `explorer`：只读探索代码或资料，输出路径、证据和结论。
- `implementer`：在声明 ownership 内执行修改，记录变更文件。
- `critic`：寻找反例、边界条件和错误假设。
- `summarizer`：压缩上下文，保留决策、约束和下一步。

这些预设应在 core 中维护，而不是散落在各插件里。插件可以注册额外 role preset，但 core 内置值必须稳定。

## 后台引擎

`SubAgentManager` 管理每个子 agent 的运行实体：

```text
SubAgentHandle
  id
  spec
  agent: HawiAgent
  scheduler: HawiScheduler
  scheduler_task: asyncio.Task
  state
  created_at / updated_at / closed_at
  last_result
  last_error
```

生命周期建议：

```text
CREATED -> IDLE -> RUNNING -> IDLE
                 -> COMPLETED
                 -> FAILED
                 -> CANCELLED
                 -> CLOSED
```

`spawn(initial_prompt=...)` 的默认行为：

1. 创建 child agent。
2. 创建 child scheduler。
3. 启动 `scheduler.run_forever()` 后台 task。
4. 如果有 `initial_prompt` 或 `initial_plan`，入 `normal` 队列。
5. 返回 `SubAgentHandle`，不阻塞等待最终结果。

`run_subagent()` 是 Python 便利 API，可内部 `spawn + wait + close`，但不建议第一版作为模型默认工具暴露。

## Python API

建议 core API：

```python
handle = await agent.subagents.spawn(
    mode="fork",
    role="reviewer",
    system_prompt=None,
    working_dir="/repo",
    initial_prompt="Review the current design.",
)

await agent.subagents.send(
    handle.id,
    "Please also check persistence risks.",
    queue="high_prio",
)

status = agent.subagents.status(handle.id)
result = await agent.subagents.wait(handle.id, timeout=120)
events = agent.subagents.recent_events(handle.id, limit=50)
await agent.subagents.interrupt(handle.id, reason="parent_changed_direction")
await agent.subagents.close(handle.id, reason="done")
```

同步环境可提供薄包装：

```python
handle = agent.spawn_subagent(...)
agent.send_subagent_input(handle.id, "...")
agent.close_subagent(handle.id)
```

这些方法只是代理到 `agent.subagents`，让插件和旧代码更容易迁移。

## Agent 工具面

模型可见工具要少而清楚。第一版建议暴露 4 个工具：

| 工具 | 作用 |
|------|------|
| `create_subagent` | 创建后台子 agent，可带初始任务 |
| `send_subagent_message` | 向已有子 agent 发送后续指导或材料 |
| `read_subagent` | 查询状态、最近输出、最终结果和错误 |
| `close_subagent` | 中断、取消或关闭子 agent |

不建议默认暴露完整 scheduler 队列操作，也不建议把所有生命周期动作塞进一个 `subagent_control(action=...)` 万能工具。四个工具的 schema 更短，模型误用成本也低。

`read_subagent` 可用 `view` 参数控制返回量：

- `status`：只返回状态和队列长度。
- `summary`：返回状态、最近结果摘要、错误。
- `events`：返回最近 N 条 mapped events。
- `context_tail`：返回最近 N 条子 agent 消息，默认不开放或需要显式权限。

## 事件与审计

子 agent 事件需要能关联回父任务。第一版不要修改所有事件模型，可以先通过转发器生成父侧 plugin event：

```json
{
  "type": "plugin.event",
  "plugin_id": "subagent",
  "event_name": "subagent.event",
  "payload": {
    "subagent_id": "...",
    "subagent_role": "reviewer",
    "parent_run_id": "...",
    "parent_tool_call_id": "...",
    "child_event_type": "agent.run_stop",
    "child_run_id": "..."
  }
}
```

后续如果需要更强类型，再增加 `subagent.*` 事件族。

审计记录至少包含：

- 创建参数：mode、role、model、plugins policy、working_dir、limits。
- 输入材料：initial prompt、显式消息、artifact 引用。
- 父侧关联：parent run id、tool call id、创建工具名。
- 输出：最终结果、错误、中断原因。
- 权限：ownership、工具 allow/deny、需要用户确认的动作。

## 持久化

第一版可以先支持“运行中子 agent 重启后标记为 interrupted/unknown”，避免恢复半执行模型流。

完整持久化需要：

- manager registry snapshot。
- 每个 child 的 `AgentContext.snapshot()`。
- 每个 child scheduler 的 queue snapshot。
- child runtime snapshot。
- 插件状态 snapshot。
- 未完成子 agent 的恢复策略：默认补合成 error tool result，并让父 agent 看到该子任务需要重启或关闭。

这和当前 `SessionManager` 的组件化 snapshot 模式一致，建议新增 `subagents.json` 和 `subagents/<id>/...` 目录，而不是把所有内容塞进父 context。

## 工作目录

`working_dir` 是子 agent 的逻辑目录，不应该调用进程级 `os.chdir()`。落地策略：

- core 在 `SubAgentSpec` 里记录 `working_dir`。
- 支持工作目录的插件读取子 agent runtime metadata。
- shell/filesystem 类插件后续增加 per-plugin cwd/root 配置或 tool context 注入。
- 不支持 workdir 的插件仍按现有行为运行，但 `read_subagent` 要暴露 warning。

## 失败语义

默认策略：

- 子 agent 失败不自动让父 agent 失败。
- `read_subagent` 返回 `FAILED` 和错误摘要。
- `wait()` 可通过 `raise_on_error=True` 让 Python 调用方选择抛出。
- `run_subagent()` 默认返回结构化失败结果，适合工具调用。
- 超时默认 interrupt，再按配置决定 cancel 或保留后台。

## 额外可配置项脑暴

这些不必第一版全做，但设计上要留位置：

- `description`：说明子 agent 为什么存在，便于 GUI 展示。
- `priority`：初始任务进入 normal/high_prio。
- `handoff_format`：结果以 Markdown、JSON、diff、artifact manifest 返回。
- `artifact_sink`：结果写入父 agent artifact store、文件或 workflow artifact。
- `visibility`：子 agent 消息是否进入 GUI 主时间线。
- `approval_policy`：子 agent 危险工具是否沿用父 agent 审批。
- `secrets_policy`：是否继承环境变量、API key 或外部凭据。
- `env`：子 agent 工具运行时环境变量白名单。
- `max_children`：限制同一 parent 下子 agent 数量。
- `parent_notification`：完成、失败、需要输入时是否给父 agent 发送 high_prio steer。
- `cleanup_policy`：完成后立即关闭、保留结果、保留完整上下文。

## 实施顺序

1. Core types：`SubAgentSpec`、`SubAgentHandle`、`SubAgentStatus`、`SubAgentManager`。
2. `HawiAgent` 持有 `subagents` manager，并提供兼容代理方法。
3. 实现 `fork` / `fresh` 创建路径、后台 scheduler task、send/status/wait/interrupt/close。
4. 增加最小事件转发和审计 payload。
5. 新增 `SubAgentPlugin`，暴露 4 个 agent tools。
6. 将 `WorkflowPlugin` 的 `SubAgentReviewer` 改为复用 core API。
7. 增加 snapshot skeleton，再接入完整持久化和 GUI 展示。

## 暂不做

- 不让 subagent 默认共享父 agent 的可变上下文。
- 不让模型直接创建无限递归子 agent。
- 不把 multi-agent workflow 做成主循环默认行为。
- 不在第一版实现复杂权限语言；先用插件继承和 allow/deny 预留口。
- 不用 `os.chdir()` 改变全局进程目录。
