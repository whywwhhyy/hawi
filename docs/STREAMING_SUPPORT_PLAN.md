# Streaming 支持设计方案

## 1. 当前架构分析

### 1.1 数据流现状

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Model     │────▶│    Agent    │────▶│   Event     │────▶│  Printer    │
│  (astream)  │     │ (_arun_stream)│    │   System    │     │ (_on_content_│
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
  StreamEvent         完整 MessageResponse   Content Block     缓冲区刷新
  (逐字/逐块)          (一次性收集)           (整块输出)        (80字/换行)
```

### 1.2 关键问题

**问题 1：Agent 层丢失流式特性**

```python
# hawi/agent/agent.py:453
response = await self._call_model_with_retry(m, policy, state)
# 这里使用了 ainvoke() 而不是 astream()，导致完整响应一次性返回
```

**问题 2：Content Block 一次性输出**

```python
# hawi/agent/agent.py:478-496
for part in response.content:
    # 一次性生成 start/delta/stop 三个事件
    yield model_content_block_delta_event(delta=text)  # 整块文本
```

**问题 3：Printer 缓冲区策略**

```python
# hawi/agent/events.py:531-539
if "\n" in self._text_buffer or len(self._text_buffer) > 80:
    # 只在换行或80字时才刷新，不是逐字
```

## 2. 目标设计

### 2.1 期望数据流

```
Model StreamEvent          Agent Event                 Printer Output
       │                         │                            │
       ▼                         ▼                            ▼
  ┌─────────┐              ┌──────────┐               ┌──────────────┐
  │ content │─────────────▶│ content_ │──────────────▶│ print(char,  │
  │  "H"    │              │ block_   │               │ end="",      │
  │         │              │ delta    │               │ flush=True)  │
  ├─────────┤              └──────────┘               └──────────────┘
  │ content │                                              │
  │  "e"    │─────────────▶ ...                            ▼
  ├─────────┤                                         逐字实时显示
  │ content │
  │  "l"    │─────────────▶ ...
  ├─────────┤
  │ content │
  │  "l"    │─────────────▶ ...
  ├─────────┤
  │ content │
  │  "o"    │─────────────▶ ...
  └─────────┘
```

### 2.2 设计要求

1. **逐字显示**：每个字符实时输出，无缓冲延迟
2. **平滑滚动**：支持打字机效果（可选延迟）
3. **换行处理**：正确处理 `\n`，自动换行
4. **性能优化**：高频 flush 不阻塞主线程
5. **兼容现有**：保持 reasoning/tool 面板显示

## 3. 详细设计方案

### 3.1 Agent 层修改

#### 3.1.1 使用 astream() 替代 ainvoke()

```python
# hawi/agent/agent.py

async def _call_model_with_retry_streaming(
    self,
    model: Model,
    policy: dict[str, ModelFailurePolicy],
    state: _ExecutionState,
    request_id: str,
    event_bus: EventBus | None,
) -> AsyncIterator[Event]:
    """流式调用模型，逐字生成事件"""

    max_retries = max((p.retry_count for p in policy.values() if p.action == "retry"), default=0)

    for attempt in range(max_retries + 1):
        try:
            request = self._context.prepare_request()
            block_index = 0

            # 使用 astream() 而不是 ainvoke()
            async for stream_event in model.astream(
                messages=request.messages,
                system=request.system,
                tools=request.tools,
            ):
                # 实时转换 StreamEvent 为 Agent Event
                if stream_event.type == "content":
                    yield await self._emit_event(
                        model_content_block_delta_event(
                            request_id=request_id,
                            block_index=block_index,
                            delta_type="text",
                            delta=stream_event.content.get("text", ""),
                        ),
                        event_bus,
                    )
                elif stream_event.type == "reasoning":
                    yield await self._emit_event(
                        model_content_block_delta_event(
                            request_id=request_id,
                            block_index=block_index,
                            delta_type="reasoning",
                            delta=stream_event.reasoning or "",
                        ),
                        event_bus,
                    )
                elif stream_event.type == "finish":
                    block_index += 1

            return  # 成功完成

        except Exception as e:
            error_type = model.classify_error(e)
            policy_for_error = policy.get(error_type, ModelFailurePolicy(error_type, "stop"))

            if policy_for_error.action == "stop" or attempt >= max_retries:
                raise

            # 重试延迟
            await asyncio.sleep(min(2**attempt, 60))
```

#### 3.1.2 修改 _arun_stream 使用新的流式调用

```python
async def _arun_stream(...) -> AsyncIterator[Event]:
    # ...

    # Model stream start
    yield await self._emit_event(
        model_stream_start_event(request_id=request_id),
        event_bus,
    )

    # 使用流式调用替代一次性调用
    content_parts = []
    async for event in self._call_model_with_retry_streaming(
        m, policy, state, request_id, event_bus
    ):
        yield event
        # 收集内容用于后续 tool call 处理
        if event.type == "model.content_block_delta":
            content_parts.append(event.metadata.get("delta", ""))

    # 完整响应用于 tool call 解析
    full_response = "".join(content_parts)
    # ... 解析 tool calls 并执行
```

### 3.2 ConversationPrinter 修改

#### 3.2.1 逐字实时输出

```python
# hawi/agent/events.py

class StreamingPrinter(ConversationPrinter):
    """支持逐字流式显示的打印机"""

    def __init__(self, *args, typing_delay: float = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.typing_delay = typing_delay  # 打字机效果延迟（秒）
        self._current_line = ""

    async def _on_content_block_delta(self, event: Event) -> None:
        """逐字实时输出"""
        meta = event.metadata
        delta_type = meta.get("delta_type")
        delta = meta.get("delta", "")

        if not delta:
            return

        if delta_type == "text":
            # 逐字符输出
            for char in delta:
                self._console.print(char, end="", flush=True)

                if self.typing_delay > 0:
                    await asyncio.sleep(self.typing_delay)

                # 处理换行
                if char == "\n":
                    self._current_line = ""
                else:
                    self._current_line += char

        elif delta_type == "reasoning" and self.show_reasoning:
            # reasoning 仍然累积到缓冲区
            self._reasoning_buffer += delta
```

#### 3.2.2 行缓冲模式（可选）

```python
class LineBufferedPrinter(ConversationPrinter):
    """行缓冲模式，平衡实时性和性能"""

    def __init__(self, *args, buffer_size: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer_size = buffer_size  # 缓冲字符数
        self._char_buffer = []

    async def _flush_char_buffer(self):
        """刷新字符缓冲区"""
        if self._char_buffer:
            text = "".join(self._char_buffer)
            self._console.print(text, end="", flush=True)
            self._char_buffer = []

    async def _on_content_block_delta(self, event: Event) -> None:
        meta = event.metadata
        delta = meta.get("delta", "")

        if meta.get("delta_type") == "text":
            for char in delta:
                self._char_buffer.append(char)

                # 遇到换行立即刷新
                if char == "\n":
                    await self._flush_char_buffer()
                # 缓冲区满也刷新
                elif len(self._char_buffer) >= self.buffer_size:
                    await self._flush_char_buffer()

        # 其他类型用父类处理
        else:
            await self._flush_char_buffer()
            await super()._on_content_block_delta(event)
```

### 3.3 main.py 修改

#### 3.3.1 使用 StreamingPrinter

```python
# main.py

from hawi.agent.events import StreamingPrinter

async def process_events():
    # 使用逐字打印模式
    printer = StreamingPrinter(
        show_reasoning=True,
        show_tools=True,
        typing_delay=0.01,  # 10ms 打字机效果
    )

    async for event in agent.arun(prompt, stream=True):
        await printer.handle(event)
```

#### 3.3.2 支持用户配置

```python
# 命令行参数支持
parser.add_argument("--typing-delay", type=float, default=0,
                    help="打字机效果延迟（秒）")
parser.add_argument("--buffer-mode", choices=["char", "line", "block"],
                    default="char", help="流式缓冲模式")
```

## 4. 实现步骤

### Phase 1: Model 层流式调用
- [ ] 修改 `_call_model_with_retry` 支持 `astream()`
- [ ] 添加 `_call_model_with_retry_streaming()` 方法
- [ ] 测试各模型 provider 的流式输出

### Phase 2: Agent 层事件转换
- [ ] 修改 `_arun_stream` 使用流式调用
- [ ] 确保 StreamEvent 正确转换为 Agent Event
- [ ] 处理 tool call 的流式解析

### Phase 3: Printer 逐字显示
- [ ] 创建 `StreamingPrinter` 类
- [ ] 实现 `_on_content_block_delta` 逐字输出
- [ ] 添加打字机效果选项

### Phase 4: 集成与测试
- [ ] 修改 `main.py` 使用新 Printer
- [ ] 添加命令行参数
- [ ] 性能测试（高频 flush 影响）
- [ ] 多模型兼容性测试

## 5. 性能考虑

### 5.1 高频 flush 优化

```python
# 方案 A：自适应缓冲
import time

class AdaptivePrinter:
    def __init__(self):
        self._last_flush = time.time()
        self._flush_interval = 0.016  # 60fps

    async def _on_content_block_delta(self, event):
        for char in delta:
            self._buffer.append(char)

            now = time.time()
            if now - self._last_flush >= self._flush_interval:
                self._flush()
                self._last_flush = now
```

### 5.2 异步输出

```python
# 使用 asyncio.Queue 避免阻塞
class AsyncPrinter:
    def __init__(self):
        self._queue = asyncio.Queue()
        self._task = asyncio.create_task(self._output_loop())

    async def _output_loop(self):
        while True:
            char = await self._queue.get()
            if char is None:  # 结束信号
                break
            self._console.print(char, end="", flush=True)

    async def _on_content_block_delta(self, event):
        for char in event.metadata.get("delta", ""):
            await self._queue.put(char)
```

## 6. 兼容性处理

### 6.1 模型不支持 streaming

```python
async def astream(self, ...):
    if not self._supports_streaming:
        # Fallback 到非流式
        response = await self.ainvoke(...)
        for char in response.content[0].get("text", ""):
            yield StreamEvent("content", content={"type": "text", "text": char})
            await asyncio.sleep(0)  # 让出控制权
```

### 6.2 终端不支持实时刷新

```python
import sys

class AutoDetectPrinter:
    def __init__(self):
        self._is_tty = sys.stdout.isatty()

    async def _on_content_block_delta(self, event):
        if self._is_tty:
            # 逐字显示
            for char in delta:
                print(char, end="", flush=True)
        else:
            # 管道/文件输出，使用缓冲
            self._buffer += delta
```

## 7. 配置选项

```python
@dataclass
class StreamingConfig:
    """流式显示配置"""

    # 模式选择
    mode: Literal["char", "line", "block"] = "char"

    # 打字机效果
    typing_delay: float = 0.0  # 每个字符延迟（秒）

    # 缓冲设置
    buffer_size: int = 1  # 字符缓冲大小
    flush_interval: float = 0.0  # 强制刷新间隔

    # 显示选项
    show_cursor: bool = False  # 显示闪烁光标
    cursor_char: str = "▌"     # 光标字符

    # 性能优化
    adaptive_rate: bool = True  # 根据终端速度自适应
```

## 8. 预期效果

### 8.1 逐字输出示例

```
用户：计算 2+3

🤔 我# 逐个字符出现
🤔 我需
🤔 我需要
🤔 我需要计
🤔 我需要计算
...
╭─────────────────────────── 🔧 Tool Call & Result ───────────────────────────╮
...
```

### 8.2 性能指标

- 延迟：< 50ms（字符到达到显示）
- 吞吐量：支持 1000+ tokens/second
- CPU：额外开销 < 5%

## 9. 风险与回退

| 风险 | 影响 | 回退方案 |
|------|------|----------|
| 高频 flush 卡顿 | 高 | 使用行缓冲模式 |
| 网络抖动显示错乱 | 中 | 客户端缓冲 100ms |
| 旧终端不支持 | 低 | 自动检测并禁用 |
| Tool call 解析失败 | 高 | 保留完整响应模式 |

## 10. 相关文件

- `hawi/agent/agent.py` - Agent 流式执行逻辑
- `hawi/agent/model.py` - Model 流式接口
- `hawi/agent/events.py` - 事件定义和 Printer
- `hawi/agent/models/openai/_streaming.py` - OpenAI 流式处理
- `main.py` - CLI 入口和配置
