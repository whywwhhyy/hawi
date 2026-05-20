# 单发送、队列任务、停止/继续：端到端方案设计

本文档描述一次产品语义和实现语义的收敛：GUI 不再让用户显式选择
`normal` / `high_prio` / `urgent`，而是把用户可理解的动作收敛为：

1. 主输入框只有一个“发送”动作。
2. 队列面板负责添加、编辑、排序、删除“稍后任务”。
3. “停止”是急停语义，可选附带一条新消息；无消息时就是纯停止，有消息时就是旧 `urgent` 的产品化表达。
4. 停止后 runner 进入暂停态，不自动执行后续队列任务。
5. 暂停后“停止”按钮变成“继续”，点击后由系统自动发送一条继续消息。
6. 因网络/模型错误导致停止时，也进入可继续状态，用户可点“继续”恢复对话。

目标是让 GUI 表达“用户意图”，而不是暴露底层队列优先级。

## 设计原则

- **产品层隐藏优先级**：用户不需要理解 `high_prio`、`normal`、`urgent`。
- **底层保留兼容语义**：Hawi 库和 engine 暂时保留 `urgent` 类型、旧 API 和持久化兼容，但新路径不再把 `urgent` 当作可消费队列。
- **发送代表“现在说话”**：主输入框发送的消息默认进入当前对话流；运行中就是 steer，空闲时就是普通 user message。
- **队列代表“稍后任务”**：队列面板里的任务只进入 `normal` 队列，按顺序自动执行。
- **停止代表“暂停自动推进”**：用户停止后，当前任务被中断，队列不再继续消费，直到用户主动发送新消息或点击继续。
- **Urgent 收敛为 Stop with Message**：不要把 `urgent` 暴露成第三种发送方式。它只是“停止当前运行，并可选附带一条马上执行的新消息”的底层兼容路径。
- **继续是 runtime 动作，不是裸文本协议**：GUI 按钮叫“继续”，底层应优先走 `resume` 命令；MVP 可降级为发送一条明确的继续提示。

## 现状摘要

当前关键实现分布：

- Hawi 库：
  - `hawi/agent/runner/queue.py`
    - `MessageQueueManager` 管理 `normal` / `high_prio` / `urgent`。
    - 当前只支持入队、出队、删除、清空、snapshot/load，不支持编辑和排序。
  - `hawi/agent/runner/runner.py`
    - `enqueue(queue="urgent")` 会触发 interrupt。
    - `enqueue(queue="high_prio")` 在 runner 忙时调用 `agent.steer()`，在空闲时入高优先级队列。
    - `run_forever()` 当前会在 idle 后继续消费 pending steer、`high_prio`、`normal`。
  - `hawi/agent/runner/executor.py`
    - `interrupt()` 取消当前 task，但结束后状态回到 `IDLE`。
    - 当前没有“暂停自动消费队列”的状态。
  - `hawi/agent/runtime.py`
    - pending steer inputs 存在 agent runtime 内，当前可能在中断后被继续 drain。
- Engine：
  - `hawi/engine/protocol.py`
    - 已有命令：`enqueue`、`interrupt`、`clear_queue`、`get_status`。
    - 没有 `resume`、队列编辑、队列排序命令。
  - `hawi/engine/runtime.py`
    - `_handle_enqueue()` 直接把 payload 里的 queue 传给 runner。
    - `_status_payload()` 已返回 `queue_lengths` 和 `queue_messages`。
  - `hawi/engine/event_mapper.py`
    - `agent.message_added` 映射为 GUI 的 `run.start`。
    - `display_message_type` 目前只有 `normal` / `steer` / `urgent`。
- GUI：
  - `hawi_gui/src/renderer/App.tsx`
    - 当前有“优先级：普通 / 优先 / 紧急”分段按钮。
    - 默认 state 里 `queue` 是 `"high_prio"`。
    - Enter 和“发送”按钮使用当前 queue 调用 `enqueue`。
    - “停止”按钮调用 `interrupt`。
  - `hawi_gui/src/renderer/state.ts`
    - 状态中有 `queueLengths` / `queueMessages`。
    - 没有 paused/resumable 状态。

## 目标产品语义

### 主输入框：发送

主输入框只保留一个发送动作。

用户点击“发送”或按 Enter：

- Agent 正在运行：
  - 作为插话发送。
  - 底层使用 `high_prio` / steer。
- Agent 空闲：
  - 作为普通对话消息执行。
  - 底层仍可使用 `high_prio`，由 runner 在空闲时立即消费，并以普通 user message 落入上下文。
- Runner 处于暂停态：
  - 先解除暂停，再发送这条用户消息。
  - 该消息优先于队列任务执行。

推荐 GUI 文案：

- 按钮：`发送`
- 输入框 placeholder：`输入消息`
- 不出现“优先级”文案。

### 队列面板：稍后任务

队列面板承担“普通队列”的全部用户入口。

用户可以：

- 添加新任务。
- 编辑未执行任务。
- 删除任务。
- 拖拽或按钮调整顺序。
- 清空队列。

队列任务底层对应 `normal` 队列。它们只有在 runner 未暂停且当前执行空闲时才自动执行。

推荐 GUI 文案：

- 顶部状态：`插话 1 · 排队 3`
- 面板标题：`待处理`
- 队列区标题：`稍后任务`
- 添加按钮：`加入队列`
- 暂停态提示：`已暂停，队列任务不会自动执行。`

### 停止

“停止”是一个原子控制动作，可以不带消息，也可以带消息。

点击 GUI 主按钮“停止”时默认不带消息：

- 当前执行被 interrupt。
- 未完成 tool call 仍按现有机制补齐 synthetic tool result，避免 provider context 损坏。
- Runner 进入暂停态。
- Runner 不再自动消费：
  - pending steer inputs
  - `high_prio` 队列
  - `normal` 队列
- GUI 的“停止”按钮变成“继续”。

旧 `urgent` 的产品语义不再是“紧急队列”，而是：

```text
stop(message?)
```

也就是：

- `stop()`：停止当前任务并暂停。
- `stop(message="...")`：停止当前任务，临时阻止普通队列抢跑，然后立刻把附带消息作为新的用户消息执行。

`stop(message)` 必须是原子动作，不能在 GUI 侧拆成“先停止、再发送”两个互相竞态的命令。它适合未来暴露为发送菜单里的低频动作，比如“停止并用这条消息继续”，但不进入默认主发送流。

### 继续

暂停态下，“继续”按钮触发 `resume` 动作。

建议底层不要永久等价于裸文本 `"continue"`，而是新增 engine command：

```json
{
  "type": "resume",
  "payload": {
    "message": null
  }
}
```

如果 `message` 为空，engine 使用默认继续提示：

```text
请从刚才中断或停止的位置继续。如果无法可靠继续，请说明当前状态并等待我的下一步指示。
```

这样 GUI 上是一个稳定的“继续”按钮，底层未来可以升级为更精确的 resume/retry 机制。

MVP 降级方案：

- GUI 点击继续时调用 `enqueue`：

```json
{
  "content": "请从刚才中断或停止的位置继续。如果无法可靠继续，请说明当前状态并等待我的下一步指示。",
  "queue": "high_prio",
  "metadata": {
    "intent": "resume",
    "display_message_type": "resume",
    "auto_generated": true
  }
}
```

但推荐实现正式 `resume` 命令，避免 GUI 硬编码恢复协议。

### 网络/模型错误后的继续

当模型请求、网络连接或 provider adapter 抛出最终错误，并且重试策略已经耗尽时：

- Runner 不应继续自动消费队列。
- Runner 进入 `paused_by_error`。
- GUI 显示错误，同时“停止”位置变成“继续”。
- 用户点击“继续”后，engine 发送默认继续提示，或未来执行更精确的 retry/resume。

## 状态模型

不要把“执行状态”和“控制状态”混在一个 enum 里。

现有 `AgentRunnerState` 仍表达执行状态：

- `IDLE`
- `READY`
- `RUNNING`
- `INTERRUPTING`

新增控制状态，建议命名为 `RunnerControlState` 或简单字段：

```python
@dataclass
class RunnerControlSnapshot:
    paused: bool = False
    pause_reason: str | None = None
    resumable: bool = False
    paused_at: float | None = None
    last_error_message: str | None = None
    resume_message: str | None = None
```

推荐 pause reason：

- `user_interrupt`
- `model_error`
- `network_error`
- `runtime_error`
- `session_restored`

状态转换：

| 事件 | 执行状态变化 | 控制状态变化 | 队列消费 |
| --- | --- | --- | --- |
| 主输入发送，未暂停 | idle/running 按现有逻辑 | 不变 | running 时 steer，idle 时立即执行 |
| 主输入发送，已暂停 | `IDLE -> RUNNING` | `paused=false` | 先执行用户消息，后续恢复正常消费 |
| 队列面板加入任务 | 不变 | 不变 | 未暂停时稍后自动消费，暂停时只保存 |
| 点击停止 | `RUNNING -> INTERRUPTING -> IDLE` | `paused=true, reason=user_interrupt` | 停止消费 |
| 停止并附带消息 | `RUNNING -> INTERRUPTING -> RUNNING` | `paused=false` | 先停止当前 run，再执行附带消息；normal 队列仍等待 |
| 点击继续 | `IDLE -> RUNNING` | `paused=false` | 发送 resume message |
| 最终模型错误 | `RUNNING -> IDLE` | `paused=true, reason=model_error/network_error` | 停止消费 |
| 清空队列 | 不变 | 不变 | 删除 normal queue |

关键约束：

- 暂停态不是 executor busy。
- 暂停态下 runner 可以接收队列编辑命令。
- 暂停态下 runner 不主动执行任何 pending work。
- 用户主输入或 `resume` 是解除暂停的明确动作。

## Hawi 库层设计

### 1. 保留 QueueType，新增 MessageIntent

不要第一步删除 `urgent`。保留底层三队列，新增语义 metadata。

建议新增类型：

```python
MessageIntent = Literal[
    "user_send",
    "queue_task",
    "resume",
    "stop",
    "legacy",
    "stop_with_message",
]
```

metadata 约定：

```python
{
    "intent": "user_send",
    "source": "gui_main_input",
    "display_message_type": "normal" | "steer" | "resume",
    "queue_kind": "high_prio" | "normal" | "urgent"
}
```

说明：

- GUI 主输入使用 `intent=user_send`、`queue=high_prio`。
- 队列面板添加任务使用 `intent=queue_task`、`queue=normal`。
- 继续按钮使用 `intent=resume`、`queue=high_prio`。
- 纯停止使用 `intent=stop`，不产生用户消息。
- 停止并附带消息使用 `intent=stop_with_message`，语义上替代旧 `urgent`。
- 旧客户端不传 intent 时按 `legacy` 处理。

### 2. MessageQueueManager 增加编辑和排序 API

目标文件：`hawi/agent/runner/queue.py`

新增方法：

```python
def update_message(
    self,
    message_id: str,
    *,
    content: str | list[ContentPart] | None = None,
    metadata: dict[str, Any] | None = None,
) -> bool: ...

def reorder_queue(
    self,
    queue_type: QueueType,
    message_ids: list[str],
) -> list[str]: ...

def move_message(
    self,
    message_id: str,
    *,
    before_id: str | None = None,
    after_id: str | None = None,
    index: int | None = None,
) -> bool: ...
```

MVP 最小集合：

- `update_message()` 只允许更新未执行的 queued message。
- `reorder_queue(QueueType.NORMAL, ids)` 只支持 normal queue。
- `remove_message()` 已存在，可复用。

排序规则：

- `message_ids` 必须覆盖 normal queue 当前全部消息 id。
- 如果传入缺失或未知 id，抛 `ValueError`。
- 保持每个 message 的 `created_at` 不变。

队列 snapshot 建议增加完整 content，供 GUI 编辑：

```python
{
    "id": "...",
    "queue": "normal",
    "content_preview": "...",
    "content": "...",
    "created_at": 123.4,
    "metadata": {...}
}
```

如果 content 是 content part list，GUI 第一版可以只允许纯文本任务；非纯文本任务显示为只读。

### 3. AgentRunner 增加暂停/继续控制

目标文件：`hawi/agent/runner/runner.py`

新增字段：

```python
self._paused = False
self._pause_reason: str | None = None
self._paused_at: float | None = None
self._last_pause_error: str | None = None
```

新增方法：

```python
def pause(
    self,
    reason: str,
    *,
    error_message: str | None = None,
) -> None: ...

def resume(self) -> None: ...

def is_paused(self) -> bool: ...

def control_snapshot(self) -> dict[str, Any]: ...

def submit_immediate_message(
    self,
    content: str | list[ContentPart],
    *,
    intent: str = "user_send",
    event_bus: EventBus | None = None,
    metadata: dict[str, Any] | None = None,
) -> str: ...

async def stop(
    self,
    reason: str = "user",
    *,
    message: str | list[ContentPart] | None = None,
    pause: bool | None = None,
    event_bus: EventBus | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]: ...
```

`submit_immediate_message()` 行为：

1. 清除 pause。
2. 使用 `queue="high_prio"` 入队。
3. metadata 默认包含传入的 `intent`。
4. 返回 message id。

`stop()` 行为：

1. 调用 executor interrupt，完成当前 run 的中断和 tool result recovery。
2. 如果 `message is None`：
   - 设置 `paused=true`。
   - 返回 interrupted tool call ids 和 control snapshot。
3. 如果 `message is not None`：
   - 不进入长期 paused，或先短暂 pause 后立即 resume。
   - 将附带消息作为 `intent=stop_with_message` 的 immediate message 执行。
   - 返回 interrupted tool call ids、message id 和 control snapshot。

### 4. interrupt/stop 行为增加 pause 和 message 参数

现有：

```python
async def interrupt(self, reason: str = "user") -> list[str]:
    return await self._executor.interrupt(reason)
```

建议改为：

```python
async def interrupt(
    self,
    reason: str = "user",
    *,
    pause: bool = False,
    message: str | list[ContentPart] | None = None,
) -> list[str]:
    interrupted = await self._executor.interrupt(reason)
    if message is not None:
        self.resume()
        self.submit_immediate_message(
            message,
            intent="stop_with_message",
            metadata={"intent": "stop_with_message", "source": "stop"},
        )
    elif pause:
        self.pause("user_interrupt")
    return interrupted
```

兼容策略：

- 库层默认 `pause=False`，避免破坏旧 API。
- legacy GUI 的 interrupt 命令传 `pause=true` 时，engine 转为纯停止。
- 如果 reason 是 `"user"` 且来自 GUI，也可以由 engine 负责转为 pause。
- 旧 `enqueue(queue="urgent")` 内部可以降级为 `interrupt(message=content, pause=False)` 或新 `stop(message=content)`，但不应再作为新 GUI 的入口。

### 5. run_forever 尊重暂停态

当前 `run_forever()` idle 后会继续消费 pending inputs、`high_prio`、`normal`。

新增逻辑：

```python
if self._paused:
    await asyncio.sleep(poll_interval)
    continue
```

推荐规则：

- `paused=True` 时不消费任何 queued message。
- engine 的主输入发送、resume 命令、stop with message 在调用 enqueue 前先 `runner.resume()`。
- 这样主输入可以解除暂停并执行，普通队列不会偷偷启动。
- `urgent` 不再作为可被 run loop 消费的队列建模；旧 urgent 入队应在 enqueue 入口立即转成 stop with message。

### 6. pending steer 的暂停处理

停止时最容易出问题的是 pending steer inputs。

现有行为：

- running 中发送 `high_prio` 会进入 `agent._pending_inputs`。
- 中断后 runner idle 时可能调用 `execute_pending_inputs()`，把 pending inputs 作为普通消息 drain 到上下文。

新规则：

- `paused=True` 时禁止 `_start_pending_input_execution()`。
- pending inputs 留在内存和 snapshot 中，并在 `core.status.queue_messages.high_prio` 里继续可见。
- GUI 第一版把它显示为“待送达插话”，只读。

可选增强：

- 停止时提供 `clear_pending_inputs=True` 选项。
- GUI 后续可允许用户删除 pending steer，或转为普通队列任务。

MVP 不强制清理 pending steer，只要不在暂停态自动执行即可。

### 7. 错误后暂停

目标文件：`hawi/agent/runner/executor.py` 和 `hawi/agent/runner/runner.py`

当 `_execute_with_error_handling()` 捕获最终异常，并且 runner error hook 返回 `CONTINUE` 时：

- 现在行为：吞掉异常，executor 回到 idle，runner 继续消费队列。
- 新行为：通知 runner 进入错误暂停态。

建议在 runner 增加：

```python
async def _on_execution_error(self, error: Exception, message: QueuedMessage) -> ErrorAction:
    action = await self._on_agent_error(error, message)
    if action == ErrorAction.CONTINUE:
        reason = classify_pause_reason(error)
        self.pause(reason, error_message=str(error))
    return action
```

或者在 executor 里调用：

```python
self._runner.pause("runtime_error", error_message=str(e))
```

错误分类 MVP：

- 如果异常类型名或 message 包含 connection、timeout、network，则 `network_error`。
- 否则 `model_error` 或 `runtime_error`。

更长期可以把 provider adapter 的错误类型标准化。

### 8. 持久化 pause state

目标文件：

- `hawi/session/manager.py`
- `hawi/session/layout.py` 如需 bump version

当前 `queues.json` 持久化：

- runner queue snapshot
- pending steer inputs
- pending audit tool calls

建议在 `queues.json` 增加：

```json
{
  "version": 2,
  "runner": {...},
  "runner_control": {
    "paused": true,
    "pause_reason": "user_interrupt",
    "resumable": true,
    "paused_at": 123.4,
    "last_error_message": null
  },
  "pending_steer_inputs": [...]
}
```

load 时：

- 如果没有 `runner_control`，按未暂停处理。
- 如果 session restored 时存在未完成 runtime/tool state，可设置 `pause_reason=session_restored`。

## Engine 协议设计

### 1. 新增 command types

目标文件：

- `hawi/engine/protocol.py`
- `hawi_gui/src/shared/protocol.ts`

新增：

```python
COMMAND_TYPES |= {
    "resume",
    "stop",
    "queue_task_add",
    "queue_task_update",
    "queue_task_remove",
    "queue_task_reorder",
}
```

命令定义：

#### resume

```json
{
  "type": "resume",
  "payload": {
    "message": null
  }
}
```

响应：

```json
{
  "type": "ack",
  "payload": {
    "command": "resume",
    "message_id": "abcd1234",
    "queue": "high_prio"
  }
}
```

#### stop

推荐新增 `stop` 作为产品语义命令，同时保留旧 `interrupt` 兼容。

纯停止：

```json
{
  "type": "stop",
  "payload": {
    "reason": "user",
    "message": null
  }
}
```

停止并附带消息：

```json
{
  "type": "stop",
  "payload": {
    "reason": "user",
    "message": "别继续现在这个方向了，改成先写测试计划",
    "metadata": {
      "source": "gui_stop_with_message"
    }
  }
}
```

响应：

```json
{
  "type": "ack",
  "payload": {
    "command": "stop",
    "interrupted_tool_calls": ["tool-1"],
    "message_id": "abcd1234",
    "control": {
      "paused": false,
      "pause_reason": null,
      "resumable": false
    }
  }
}
```

如果 `message` 是 `null`，响应里的 `message_id` 也是 `null`，control 为 paused：

```json
{
  "type": "ack",
  "payload": {
    "command": "stop",
    "interrupted_tool_calls": [],
    "message_id": null,
    "control": {
      "paused": true,
      "pause_reason": "user_interrupt",
      "resumable": true
    }
  }
}
```

#### queue_task_add

```json
{
  "type": "queue_task_add",
  "payload": {
    "content": "写完后补测试",
    "index": null
  }
}
```

行为：

- 添加到 `normal` 队列。
- metadata 包含 `intent=queue_task`。
- 如果 `index` 存在，添加后移动到指定位置。

#### queue_task_update

```json
{
  "type": "queue_task_update",
  "payload": {
    "message_id": "abcd1234",
    "content": "更新后的任务"
  }
}
```

#### queue_task_remove

```json
{
  "type": "queue_task_remove",
  "payload": {
    "message_id": "abcd1234"
  }
}
```

#### queue_task_reorder

```json
{
  "type": "queue_task_reorder",
  "payload": {
    "message_ids": ["id2", "id1", "id3"]
  }
}
```

### 2. 扩展 enqueue metadata

保留 `enqueue` 兼容旧客户端。

GUI 主输入调用：

```json
{
  "type": "enqueue",
  "payload": {
    "content": "用户输入",
    "queue": "high_prio",
    "metadata": {
      "intent": "user_send",
      "source": "gui_main_input"
    }
  }
}
```

engine `_handle_enqueue()` 行为：

- 如果 `metadata.intent in {"user_send", "resume"}`：
  - 先 `runner.resume()`。
  - queue 强制或默认为 `high_prio`。
- 如果 `metadata.intent == "queue_task"`：
  - 建议拒绝从主 enqueue 走，提示使用 `queue_task_add`。
  - 也可兼容地转为 `normal`。
- 旧客户端不传 intent：
  - 维持现有行为。

### 3. interrupt 命令兼容，stop 命令承载新语义

现有 GUI 停止按钮调用：

```json
{
  "type": "interrupt",
  "payload": {
    "reason": "user"
  }
}
```

建议新 GUI 改为：

```json
{
  "type": "stop",
  "payload": {
    "reason": "user"
  }
}
```

engine 兼容策略：

- `pause` 缺省为 `false`，避免影响非 GUI 使用者。
- 旧 `interrupt { pause:true }` 仍可用，内部转发到 `runner.stop(message=None)`。
- 旧 `enqueue(queue="urgent", content=...)` 仍可用，内部转发到 `runner.stop(message=content)`。
- 新 GUI 只使用 `stop`，不再发送 `urgent`。

响应 payload 增加：

```json
{
  "interrupted_tool_calls": [],
  "control": {
    "paused": true,
    "pause_reason": "user_interrupt",
    "resumable": true
  }
}
```

### 4. core.status 增加 control 字段

目标文件：

- `hawi/engine/runtime.py`
- `hawi_gui/src/renderer/state.ts`

新增 payload：

```json
{
  "runner_state": "IDLE",
  "agent_state": "IDLE",
  "queue_lengths": {
    "urgent": 0,
    "high_prio": 0,
    "normal": 2
  },
  "queue_messages": {
    "urgent": [],
    "high_prio": [],
    "normal": []
  },
  "control": {
    "paused": true,
    "pause_reason": "user_interrupt",
    "resumable": true,
    "paused_at": 123.4,
    "last_error_message": null
  }
}
```

旧 GUI 忽略该字段，不受影响。

### 5. 新增事件，可选但推荐

可以只靠 `core.status`，但推荐新增事件让 GUI 即时更新：

```python
EVENT_TYPES |= {
    "runner.paused",
    "runner.resumed",
}
```

事件 payload：

```json
{
  "reason": "user_interrupt",
  "resumable": true,
  "last_error_message": null
}
```

```json
{
  "message_id": "abcd1234",
  "source": "resume"
}
```

如果不想增加事件类型，`interrupt` ack 后主动发一次 `core.status` 也可以。

### 6. capability 标识

如果担心 GUI 与旧 core 混跑，新增 server caps：

- `message_intent_v1`
- `runner_pause_v1`
- `resume_v1`
- `queue_edit_v1`

GUI 可根据 caps 决定：

- 新 core：使用新 UI 和新命令。
- 旧 core：隐藏队列编辑，继续用旧 enqueue/interrupt。

## GUI 设计

### 1. 删除优先级选择

目标文件：

- `hawi_gui/src/renderer/App.tsx`
- `hawi_gui/src/renderer/styles.css`
- `hawi_gui/src/renderer/App.test.ts`

删除当前 control row：

```text
优先级: 普通 / 优先 / 紧急
```

删除或废弃：

- `const [queue, setQueue] = useState<QueueKind>("high_prio")`
- Tab+Shift 切换 queue 的快捷键
- `queueLabels` 在主输入区域的用途

保留 queueLabels 可用于队列面板展示，但不要叫“优先级”。

### 2. 主输入发送逻辑

`submitInput()` 改为：

```ts
await sendCommand("enqueue", {
  content: text,
  queue: "high_prio",
  metadata: {
    intent: "user_send",
    source: "gui_main_input"
  }
});
```

注意：

- 不再读取本地 queue state。
- 如果当前 paused，engine 会解除暂停并执行这条消息。
- Slash command 保持原有逻辑。

### 3. 停止/继续按钮

新增 app state：

```ts
interface RuntimeControlState {
  paused: boolean;
  pauseReason?: string;
  resumable: boolean;
  pausedAt?: number;
  lastErrorMessage?: string;
}
```

`AppState` 增加：

```ts
control: RuntimeControlState;
```

`core.status` reducer 解析 payload.control。

按钮规则：

- `agentState === "RUNNING"` 或 `runnerState === "RUNNING"`：
  - 显示 `停止`
  - 点击：

```ts
sendCommand("stop", { reason: "user" })
```

如果未来在发送菜单里增加“停止并发送当前输入”，调用：

```ts
sendCommand("stop", {
  reason: "user",
  message: input.trim(),
  metadata: { source: "gui_stop_with_message" }
})
```

- `state.control.paused && state.control.resumable`：
  - 显示 `继续`
  - 点击：

```ts
sendCommand("resume", {})
```

- idle 且未 paused：
  - 可以隐藏停止按钮，或置灰。

推荐文案：

- running：`停止`
- paused by user：`继续`
- paused by error：`继续`
- tooltip：
  - 停止：`停止当前任务并暂停队列`
  - 继续：`从刚才停止的位置继续`

### 4. 队列状态与面板

现有 `PriorityStatus` 改名为 `QueueStatus`。

原文案：

```text
优先 1 · 普通 3
```

改为：

```text
插话 1 · 排队 3
```

含义：

- `插话` = pending steer/high_prio preview，通常只读。
- `排队` = normal queue tasks，可编辑排序。

面板结构：

```text
待处理
已暂停，队列任务不会自动执行。  # 仅 paused 时显示

待送达插话
  - 用户刚刚发送但尚未落入上下文的 steer（只读）

稍后任务
  [ 添加一个稍后任务...          ] [加入队列]
  [拖拽] 任务 A       [编辑] [删除]
  [拖拽] 任务 B       [编辑] [删除]
```

交互：

- 添加任务：`queue_task_add`
- 编辑任务：`queue_task_update`
- 删除任务：`queue_task_remove`
- 拖拽排序：`queue_task_reorder`
- 清空：复用 `clear_queue { queue: "normal" }`

MVP 如果不做拖拽，可先做上移/下移按钮：

- `ArrowUp`
- `ArrowDown`

但数据协议仍建议用 `queue_task_reorder`，这样后续拖拽不用改 backend。

### 5. 暂停态提示

暂停态需要在两个地方可见：

1. 输入区按钮从停止变继续。
2. 队列面板显示“已暂停，队列任务不会自动执行。”

如果是错误暂停，再显示错误摘要：

```text
上次运行因网络错误停止。点击继续会尝试接着对话。
```

不要把错误暂停表现成 fatal，需要给用户一个顺手恢复的出口。

### 6. 消息气泡标签

`display_message_type` 建议扩展：

```ts
export type DisplayMessageType =
  | "normal"
  | "steer"
  | "urgent"
  | "resume";
```

展示：

- `normal`：默认不显示标签。
- `steer`：显示 `插话`。
- `resume`：显示 `继续`。
- `urgent`：旧会话兼容，显示 `紧急消息`。
  - 新事件不应再产生该展示类型；停止并附带消息应显示为普通新用户消息，metadata 可保留 `intent=stop_with_message`。

主输入 idle 时即使底层 queue 是 `high_prio`，最终 materialized metadata 可以仍是 `normal`，不显示“插话”。

## 行为细节和边界情况

### 停止后已有 normal 队列怎么办

保留队列，不清空。

停止只暂停自动执行，不删除用户排好的任务。

用户有三个选择：

- 点击继续：恢复当前对话；完成后队列可以继续自动执行。
- 主输入发送新消息：恢复并先执行这条消息；完成后队列可以继续自动执行。
- 在队列面板删除/重排任务。

如果希望“继续当前任务后仍不跑队列”，可以后续增加一个独立开关 `暂停队列`。MVP 不做。

### 停止后 pending steer 怎么办

MVP：

- 保留。
- 暂停态不执行。
- 在队列面板显示为只读“待送达插话”。

后续增强：

- 支持删除 pending steer。
- 支持转为 normal queue task。

### 用户暂停后又发送新消息

主输入新消息解除暂停，并优先执行。

如果此时还有 pending steer，推荐顺序：

1. 新用户消息先执行。
2. pending steer 暂不自动 drain，避免旧插话污染新任务。

这需要一个小调整：

- `submit_immediate_message(intent=user_send)` 可选择不 drain pending inputs。
- 或在 user stop 时把 pending inputs 标记为 suspended。

MVP 可接受保留现有 pending inputs 行为，但必须保证它们不会在暂停期间自动执行。

### 继续按钮和上下文完整性

点击继续不应该恢复一个 provider-invalid context。

继续前必须确保：

- 未回答 tool call 已补 synthetic tool result。
- interrupted assistant partial 已按现有机制持久化。
- context 中没有 dangling assistant tool_call。

现有 Hawi 已有相关机制，新的 resume 只要复用现有 interrupt recovery 即可。

### 网络错误后的继续

网络错误时可能没有 assistant partial，也可能模型请求完全失败。

继续提示应该让模型基于现有上下文继续：

```text
请继续刚才未完成的回答。如果上一轮没有产生可继续内容，请根据当前对话重新回答最近的问题。
```

如果后续 provider adapter 能暴露 request retry token，再升级为真正 retry。

## 测试计划

### Hawi unit tests

新增或修改测试位置：

- `test/unit/agent/runner/test_queue.py`
- `test/unit/agent/runner/test_runner.py`
- `test/unit/session/test_snapshot_round_trip.py`

覆盖：

1. `MessageQueueManager.update_message()` 只更新未执行 normal message。
2. `MessageQueueManager.reorder_queue()` 能按 id 重排 normal queue。
3. `reorder_queue()` 对缺失/未知 id 抛错。
4. `runner.stop(message=None)` 后 `control_snapshot.paused == True`。
5. paused runner 不消费 normal queue。
6. paused runner 不 drain pending steer inputs。
7. `submit_immediate_message(intent="resume")` 清除 paused 并入 high_prio。
8. `runner.stop(message="...")` 会中断当前 run，并把附带消息作为 `stop_with_message` 执行。
9. `enqueue(queue="urgent")` 兼容降级为 `stop(message=content)`。
10. session snapshot/load 保留 pause state 和 queued task 顺序。

### Engine tests

目标文件：

- `hawi_gui/src/main/session-engine-manager.test.ts`
- 或 engine runtime 对应 Python tests，如果已有。

覆盖：

1. `enqueue` with `intent=user_send` 强制走 high_prio，并解除 pause。
2. `stop { message:null }` ack 返回 paused control。
3. `stop { message:"..." }` 返回 `message_id`，并不留下 paused control。
4. 旧 `interrupt { pause:true }` 仍兼容为纯停止。
5. 旧 `enqueue queue=urgent` 兼容为 stop with message。
6. `resume` 发送默认继续消息。
7. `queue_task_add/update/remove/reorder` 正确调用 runner queue API。
8. `core.status` 包含 control 和可编辑 queue message content。

### GUI reducer tests

目标文件：

- `hawi_gui/src/renderer/state.test.ts`
- `hawi_gui/src/renderer/App.test.ts`

覆盖：

1. `core.status.payload.control` 更新 `state.control`。
2. paused 状态下按钮文案应为“继续”。
3. running 状态下按钮文案应为“停止”。
4. `renderQueueStatusText()` 输出 `插话 1 · 排队 3`。
5. normal queue preview 计为排队任务。
6. high_prio/pending input preview 计为插话。

### 手工验收

1. 启动 GUI，主输入区没有“优先级”控件。
2. Agent 空闲时发送消息，正常得到回复。
3. Agent 运行中发送消息，消息作为插话进入当前回合。
4. 添加两个稍后任务，队列面板可见且可排序。
5. 当前任务运行中点击停止：
   - 当前 run 停止。
   - 按钮变为继续。
   - 队列任务不自动执行。
6. 如果使用“停止并发送”入口：
   - 当前 run 停止。
   - 附带消息作为新用户消息执行。
   - 普通队列任务不抢在附带消息前执行。
7. 点击继续：
   - 自动发送继续提示。
   - Agent 继续对话。
8. 模拟网络错误：
   - GUI 显示错误。
   - 按钮变为继续。
   - 点击继续可重新进入对话。

## 实施步骤

### Phase 1：Hawi 库层

文件：

- `hawi/agent/runner/queue.py`
- `hawi/agent/runner/runner.py`
- `hawi/agent/runner/executor.py`
- `hawi/session/manager.py`

任务：

1. 给 queue manager 加 update/reorder/move 能力。
2. 给 runner 加 paused control state。
3. `run_forever()` 尊重 paused。
4. 增加 `runner.stop(message?)`，用它承载纯停止和 stop with message。
5. `interrupt(pause=True)` 保持兼容并转到纯停止语义。
6. 旧 urgent enqueue 降级为 stop with message。
7. 错误最终失败后设置 paused_by_error。
8. snapshot/load 保存 control state。
9. 增加单元测试。

### Phase 2：Engine 协议

文件：

- `hawi/engine/protocol.py`
- `hawi/engine/runtime.py`
- `hawi/engine/event_mapper.py`
- `hawi_gui/src/shared/protocol.ts`

任务：

1. 新增 `stop` 命令，payload 支持可选 `message`。
2. 新增 `resume` 命令。
3. 新增 queue task CRUD/reorder 命令。
4. 扩展 `interrupt` payload 支持 `pause`，作为旧协议兼容。
5. 扩展 `enqueue` metadata intent 处理，并将旧 `urgent` 转为 `stop(message)`。
6. `core.status` 增加 `control`。
7. 可选新增 `runner.paused` / `runner.resumed` 事件。
8. 增加协议和 runtime tests。

### Phase 3：GUI

文件：

- `hawi_gui/src/renderer/App.tsx`
- `hawi_gui/src/renderer/state.ts`
- `hawi_gui/src/renderer/styles.css`
- `hawi_gui/src/renderer/App.test.ts`
- `hawi_gui/src/renderer/state.test.ts`

任务：

1. 删除优先级选择控件和相关 local queue state。
2. 主输入固定发送 `queue=high_prio, intent=user_send`。
3. 增加 control state reducer。
4. 停止/继续按钮根据 running/paused 状态切换。
5. PriorityStatus 改为 QueueStatus，文案改为 `插话 / 排队`。
6. 队列面板支持添加、编辑、删除、排序 normal queue tasks。
7. paused/error paused 提示。
8. 更新 CSS 和测试。

### Phase 4：文档和清理

文件：

- `docs/agent_runner.md`
- `docs/todo.md`

任务：

1. 更新 AgentRunner 文档，说明 GUI 产品语义与底层队列语义不同。
2. `docs/todo.md` 勾选或新增对应任务。
3. 保留旧 `urgent` 文档，但标注 GUI 不再直接暴露。

## 推荐最终 UI 草图

主输入区：

```text
[ 输入消息 .................................................. ] [发送] [停止]
```

暂停态：

```text
[ 输入消息 .................................................. ] [发送] [继续]
```

队列弹窗：

```text
待处理                                          排队 3

已暂停，队列任务不会自动执行。                  # paused only

待送达插话
  请先按这个约束改                              # read-only

稍后任务
  [ 写完后补测试                              ] [加入队列]

  ↕  1. 生成 changelog                         [编辑] [删除]
  ↕  2. 补充 README                            [编辑] [删除]
  ↕  3. 跑完整测试                             [编辑] [删除]
```

## 最小可行实现建议

如果需要尽快落地，按这个最小集合做：

1. GUI 删除优先级选择，主输入固定 `high_prio`。
2. runner 增加 paused，GUI stop 调用新 `stop` 命令；旧 GUI 的 interrupt `pause=true` 保持兼容。
3. paused 时 runner 不消费 normal queue 和 pending inputs。
4. GUI 停止/继续按钮切换。
5. 继续按钮先用 `enqueue(default_resume_prompt, high_prio)` 实现。
6. 队列面板第一版只支持添加、删除、上移、下移，不做拖拽。

正式协议 `resume` 和队列编辑命令可以在同一轮补齐，避免 GUI 长期硬编码协议。

## 风险和注意事项

- 不要在 GUI 里继续暴露 `urgent`，否则用户会重新回到“优先级选择”的心智。
- 不要让 stop 后 runner 自动 drain pending steer，否则会违反“停止后等待用户”的核心语义。
- 不要把 `continue` 作为永久字符串协议；它应该是默认 prompt 或 resume command 的 fallback。
- 不要清空 normal queue，停止不是删除计划。
- 不要在 paused 状态下把 runner_state 伪装成 RUNNING；paused 是控制状态，不是执行状态。
- 注意 session restore：如果恢复后存在未完成 runtime 状态，宁可进入 paused/resumable，也不要自动继续跑队列。

## 开放问题

1. 点击“继续”后，当前 continue run 结束时是否自动恢复 normal queue 消费？
   - 本文建议 MVP 恢复正常消费。
   - 如果用户反馈意外，可增加“暂停队列”独立开关。
2. 停止时 pending steer 是否应该自动转入 normal queue？
   - 本文建议 MVP 保留为只读待送达插话。
3. 网络错误后的继续是否应该优先 retry 原 provider request？
   - 本文建议先发送恢复 prompt，后续等 provider adapter 有更明确 retry 能力再升级。
4. 队列任务是否需要独立标题、正文、标签？
   - MVP 只做纯文本 content。
   - 后续可扩展 metadata：`title`、`tags`、`created_by`、`scheduled_at`。
