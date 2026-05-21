# Session History Paging Design

## 背景

当前 GUI 加载 session 时，engine 会把 `message_history.jsonl` 全量读入内存，解析成数组，并一次性通过 IPC 返回给 renderer。renderer 随后用这份全量历史构建所有聊天节点、插件状态、artifact 状态和 subagent 状态。

这对小 session 很直接，但大 session 会出现几个瓶颈：

- `layout.read_jsonl()` 当前使用 `read_text().splitlines()`，会先把整个文件读成字符串。
- 每一行都会立即 `json.loads`，形成完整 Python list。
- 完整 `message_history` 会被一次性序列化并传给 GUI。
- GUI 需要一次性 replay 全量历史，生成所有节点和派生状态。

因此问题不在于 JSONL 不能存几十 MB 或几百 MB，而是当前把 JSONL 当成“每次全量读取的数组文件”使用。

## 目标

- 首次打开 session 时只加载末尾一小段历史，让主界面快速可交互。
- 支持用户向上滚动时按需查询更早历史，并插入到当前时间线顶部。
- 保持 agent 恢复语义不变：`context.json`、`runtime.json`、`queues.json` 仍然完整恢复给 engine。
- 保留 `message_history.jsonl` 的可读性和 append-only 写入模型。
- 为后续 SQLite 本地数据库迁移留下清晰边界。

## 非目标

- 第一阶段不重写整个 session 存储为数据库。
- 第一阶段不做全文搜索或复杂筛选。
- 第一阶段不改变模型上下文恢复方式。
- 第一阶段不把巨大 tool 参数/结果直接塞进 timeline；超大内容应配合 blob/reference 机制单独处理。

## 推荐路线

第一阶段继续使用 JSONL，但增加旁路索引和分页接口。

新增 sidecar index 文件，例如：

```text
message_history.jsonl
message_history.idx
```

`message_history.idx` 记录每条 history record 的轻量定位信息：

```json
{"seq":0,"offset":0,"length":312,"timestamp":1760000000.0,"role":"user","run_id":"run-1","context_message_id":"..."}
```

索引用途：

- 通过 `offset + length` 直接读取指定 JSONL 行。
- 支持尾部 N 条读取，而不扫描和解析全文件。
- 支持 `before` / `after` cursor 分页。
- 后续可追加 role、run_id、timestamp 等筛选字段。

索引可以在写入 message history 时同步 append；旧 session 没有 index 时，首次查询懒构建。

## Core API

扩展现有 `session_history` 命令，不破坏旧语义：

```json
{
  "type": "session_history",
  "payload": {
    "session_id": "session-...",
    "mode": "tail",
    "limit": 80
  }
}
```

建议返回：

```json
{
  "session_id": "session-...",
  "message_history": [],
  "page": {
    "total": 1234,
    "start_seq": 1154,
    "end_seq": 1233,
    "has_before": true,
    "has_after": false,
    "before_cursor": 1154,
    "after_cursor": 1233,
    "mode": "tail",
    "limit": 80
  },
  "context_usage": {}
}
```

分页模式：

- `tail`: 读取末尾 `limit` 条。
- `before`: 读取 `before_cursor` 之前的 `limit` 条。
- `after`: 读取 `after_cursor` 之后的 `limit` 条。
- `range`: 按 seq 范围读取，用于调试和未来跳转。
- 兼容模式：如果 payload 没有分页参数，可以暂时保留全量读取，但 GUI 应迁到分页模式。

`session_load` / `session_switch` / `session_fork` / `session_rewind` 的 ack 不再需要总是携带全量 `message_history`。第一阶段可以让这些命令默认返回 tail page，以减少改动；后续可以只返回 `session_id`，由 GUI 再显式调用 `session_history(mode="tail")`。

## SessionManager API

新增分页读取接口：

```python
read_message_history_page(
    session_id: str | None = None,
    *,
    mode: Literal["tail", "before", "after", "range"] = "tail",
    limit: int = 80,
    cursor: int | None = None,
    start_seq: int | None = None,
    end_seq: int | None = None,
) -> SessionHistoryPage
```

`SessionHistoryPage` 应包含：

- `entries`
- `total`
- `start_seq`
- `end_seq`
- `has_before`
- `has_after`
- `before_cursor`
- `after_cursor`

注意：现有 `read_message_history()` 仍可保留给 markdown export、测试和兼容路径。

## GUI 设计

当前 `gui.load_session_history` 会替换整个 `AppState.nodes`。分页后需要拆成两类动作：

- `gui.load_session_history_tail`: 替换当前 session 的可见历史窗口。
- `gui.prepend_session_history_page`: 把更早历史插入顶部。

前端状态需要记录：

- 当前已加载 seq 范围。
- 是否还有更早历史。
- 是否正在加载更早历史。
- prepend 后保持滚动位置稳定。

建议第一阶段只支持从顶部向前加载，不做中间跳转。

## 大记录策略

分页只能减少“记录数量”的加载压力，不能解决“单条记录巨大”的问题。对于 `write_file` / `edit_file` / 大 tool result：

- message history 中保存 preview、size、hash、blob_id / ref path。
- GUI 默认渲染 preview。
- 用户展开时再按需读取完整 blob。

这部分可以和已有 blob store / markdown ref 导出策略合流。

## 为什么不第一步使用数据库

SQLite 对以下需求很适合：

- 跨 session 搜索。
- 按 role / run_id / timestamp / context_message_id 查询。
- 大量 session catalog 统计。
- 子 agent 历史统一查询。
- 后续全文索引。

但它不会自动解决：

- 单条消息巨大。
- GUI 一次性构建所有节点。
- IPC 一次性传输大 payload。

数据库迁移还会带来额外成本：

- schema / migration / backfill。
- JSONL 兼容和导入策略。
- 测试事务、损坏恢复、并发读写。
- 导出、fork、rewind 等现有文件级操作改造。

所以推荐顺序是：

1. JSONL + offset index + 分页查询。
2. 大记录 blob/reference 化。
3. 当需要搜索、复杂筛选和跨 session 查询时，再引入 SQLite。

## 粗略工作量

- JSONL + offset index + 分页接口：1-3 天。
- GUI 尾部加载 + 顶部按需加载：1-3 天。
- 大记录 blob/reference 化：2-5 天，取决于覆盖 tool 参数、tool result、markdown export 的范围。
- SQLite 最小替换 message history：3-5 天。
- 完整数据库化 session / subagent / catalog / search：1-2 周以上。

## 实施顺序

1. 增加 `message_history.idx` 读写和懒构建。
2. 增加 `SessionManager.read_message_history_page()`。
3. 扩展 core protocol `session_history` 支持分页 payload 和 page metadata。
4. 让 `session_load` / `session_switch` 返回 tail page 或让 GUI 显式查询 tail。
5. GUI state 增加 history window metadata。
6. ChatTranscript 顶部触发 `before` 查询并 prepend。
7. 对超大 tool 参数/结果引入 preview + blob/reference。
8. 评估 SQLite 是否进入下一阶段。
