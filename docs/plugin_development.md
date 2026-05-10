# Hawi 插件开发指南

## 目录

- [插件系统概述](#插件系统概述)
- [快速开始](#快速开始)
- [插件注册与发现](#插件注册与发现)
- [HawiPlugin 基类](#hawiplugin-基类)
- [Hook 系统](#hook-系统)
- [Tools（工具）](#tools工具)
- [Resources（资源）](#resources资源)
- [Plugin Events（插件事件）](#plugin-events插件事件)
- [GUI 配置 Schema](#gui-配置-schema)
- [Clone 与 Fork 支持](#clone-与-fork-支持)
- [配置文件加载](#配置文件加载)
- [最佳实践](#最佳实践)
- [完整示例：EnvironPromptPlugin](#完整示例environpromptplugin)
- [参考与调试](#参考与调试)

---

## 插件系统概述

Hawi 的插件系统提供**三类扩展能力**：

| 能力 | 说明 | 适用场景 |
|------|------|----------|
| **Hook** | 阻塞式、可修改的执行拦截器 | 注入上下文、拦截工具调用、控制流程 |
| **Tool** | Agent 可调用的函数工具 | 文件操作、Shell 执行、网络请求 |
| **Resource** | MCP 兼容的上下文资源 | 提供结构化数据给模型 |

插件通过 **`PluginManager`** 统一管理，在 Agent 初始化时注册。

---

## 快速开始

### 最小的插件

```python
# my_plugin/plugin.py
from hawi.plugin import HawiPlugin, before_conversation

class MyPlugin(HawiPlugin):
    @before_conversation
    def greet(self, agent, ctx):
        """每次 conversation 开始前打个招呼"""
        print(f"Hello from plugin! run_id={ctx.run_id}")
```

### 包结构

```
hawi_plugins/
  my_plugin/
    __init__.py          # 导出插件类
    plugin.py            # 核心实现
    environ_prompt.yaml.example  # 示例配置文件（可选）
```

`__init__.py`:
```python
from .plugin import MyPlugin

__all__ = ["MyPlugin"]
```

---

## 插件注册与发现

要让插件被 Hawi 框架发现和使用，需要完成 **3 个注册点**：

### 1. 定义插件 Key 常量

在 `hawi_engine/runtime.py` 中添加：

```python
# === 第 1 步：定义 key 常量 ===
PLUGIN_MY_PLUGIN = "my_plugin"

# === 第 2 步：加入 KNOWN_PLUGINS 集合 ===
KNOWN_PLUGINS = {
    # ... 已有插件 ...
    PLUGIN_MY_PLUGIN,
}

# === 第 3 步：加入 PLUGIN_LABELS 映射 ===
PLUGIN_LABELS = {
    # ... 已有插件 ...
    PLUGIN_MY_PLUGIN: "MyPlugin",
}
```

### 2. 添加创建逻辑

在 `runtime.py` 的 `_create_plugins` 方法中添加：

```python
async def _create_plugins(self, selected_plugins, plugin_configs):
    for plugin_key in selected_plugins:
        cfg = dict(plugin_configs.get(plugin_key, {}))
        # ...
        elif plugin_key == PLUGIN_MY_PLUGIN:
            from hawi_plugins.my_plugin import MyPlugin
            plugin = MyPlugin(**cfg)  # 传递配置参数
        # ...
```

### 3. 注册到 GUI 目录

在 `hawi_engine/inspect.py` 中添加：

```python
from .runtime import (
    # ... 已有导入 ...
    PLUGIN_MY_PLUGIN,
)

def _plugin_entries():
    # ...
    from hawi_plugins.my_plugin import MyPlugin

    return [
        # ... 已有条目 ...
        (PLUGIN_MY_PLUGIN, "MyPlugin", MyPlugin),
    ]
```

完成这 3 步后，插件将在 GUI 的"插件配置"对话框中可见，用户可以通过勾选启用/停用。

---

## HawiPlugin 基类

所有插件继承自 `HawiPlugin`，核心接口：

```python
class HawiPlugin:
    @property
    def hooks(self) -> PluginHooks: ...
    @property
    def tools(self) -> Sequence[AgentTool]: ...
    @property
    def resources(self) -> Sequence[HawiResource]: ...

    def clone(self) -> HawiPlugin: ...           # 用于 agent fork
    def bind_event_bus(self, event_bus): ...      # 绑定事件总线
    def bind_plugin_identity(self, *, plugin_id, plugin_name): ...
```

### 身份绑定

```python
plugin = MyPlugin()
plugin.bind_plugin_identity(
    plugin_id="my_plugin",           # 稳定标识符（GUI 用）
    plugin_name="MyPlugin",          # 显示名称（事件中用）
)
```

### 事件辅助方法

插件提供了若干便捷的 emit 方法：

```python
# 消息
self.emit_message("处理完成", level="info", title="MyPlugin")

# 状态更新
self.emit_status("running", label="处理中", progress=0.5)

# 工具进度
self.emit_tool_progress(progress=0.8, status="parsing", message="解析第 3 页")

# Artifacts（产物）
self.upsert_artifact(
    "artifact-key",
    artifact_type="document",
    title="分析报告",
    content="# 报告内容",
    language="markdown",
)
self.append_artifact("artifact-key", "追加内容")
self.remove_artifact("artifact-key")
self.clear_artifacts(scope="plugin")
```

---

## Hook 系统

### 可用 Hook 一览

| Hook 装饰器 | 方法签名 | 触发时机 |
|-------------|---------|----------|
| `@before_session` | `(self, agent, ctx)` | Session 开始时（一次） |
| `@after_session` | `(self, agent, ctx)` | Session 结束时（一次） |
| `@before_conversation` | `(self, agent, ctx)` | 每次 conversation 开始时 |
| `@after_conversation` | `(self, agent, ctx)` | 每次 conversation 结束时 |
| `@before_model_call` | `(self, agent, model, ctx)` | 每次模型调用前 |
| `@after_model_call` | `(self, agent, response, ctx)` | 每次模型调用后 |
| `@before_tool_calling` | `(self, agent, tool_name, arguments, ctx)` | 每次工具调用前 |
| `@after_tool_calling` | `(self, agent, tool_name, arguments, result, ctx)` | 每次工具调用后 |

### HookContext 字段

| 字段 | before_session | after_session | before_conversation | after_conversation | before_model_call | after_model_call | before_tool_calling | after_tool_calling |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `run_id` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `iteration` | | ✅ | | ✅ | ✅ | ✅ | ✅ | ✅ |
| `tool_call_id` | | | | | | | ✅ | ✅ |
| `tool` | | | | | | | ✅ | ✅ |
| `duration_ms` | | ✅ | | ✅ | | ✅ | | ✅ |
| `error` | | ✅ | | ✅ | | | | |

### HookResult 流程控制

Hook 可以通过返回值控制 Agent 流程：

| 结果 | 可用 Hook | 效果 |
|------|-----------|------|
| `None` | 所有 | 正常继续 |
| `HookResult.abort(reason)` | 所有 | 终止当前 run |
| `HookResult.skip(result)` | `before_tool_calling` | 跳过工具执行 |
| `HookResult.replace_model(model)` | `before_model_call` | 替换本次模型 |
| `HookResult.restart_turn()` | `before_model_call` | 跳过本轮模型调用 |
| `HookResult.reinvoke(message)` | `before_model_call`, `after_model_call`, `after_conversation` | 注入消息并重新驱动 |

### Hook Chain 执行规则

多个插件注册同一 Hook 时，按注册顺序执行：

```python
agent = HawiAgent(plugins=[PluginA(), PluginB()])
# 1. PluginA.hook → None → 继续
# 2. PluginB.hook → HookResult → 链停止
```

### 上下文操作时序

| Hook | 安全操作 |
|------|---------|
| `before_session` | 修改 `agent.context.system_prompt`（首次生效） |
| `before_conversation` | 修改 `agent.context`、添加/注入消息 |
| `before_model_call` | 修改 `agent.context`（本轮调用的 model 请求中生效） |
| `after_model_call` | `response` 已生成，assistant message **尚未**写入 context |
| `before_tool_calling` | 可直接修改 `arguments` 字典；访问 `agent.context` |
| `after_tool_calling` | tool result **尚未**写入 context；可直接修改 `result` |

---

## Tools（工具）

插件可以暴露工具供 Agent 调用：

```python
from hawi.plugin import HawiPlugin, tool
from hawi.tool import ToolResult

class SearchPlugin(HawiPlugin):
    @tool(
        name="search",
        description="搜索信息",
        parameters_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
                "limit": {"type": "integer", "description": "返回数量", "default": 10},
            },
            "required": ["query"],
        },
    )
    def search(self, query: str, limit: int = 10) -> ToolResult:
        results = self._do_search(query, limit)
        return ToolResult(success=True, output=results)

    @tool(
        name="async_search",
        description="异步搜索",
        audit=True,  # 需要人工审核
        timeout=30.0,  # 超时控制
        tags=["search", "web"],  # 分类标签
    )
    async def async_search(self, query: str) -> ToolResult:
        results = await self._async_search(query)
        return ToolResult(success=True, output=results)
```

### 工具参数注入

框架支持在工具 schema 中注入框架参数（如 `tool_call_purpose`）：

```python
# 在 runtime.py 中注册
agent.plugins.add_tool_parameter_injection(
    ToolParameterInjection(
        name="tool_call_purpose",
        schema={"type": "string", "description": "说明本次工具调用的目的；允许与其他调用重复"},
        required=True,
    )
)
```

---

## Resources（资源）

资源提供 MCP 兼容的结构化数据：

```python
from hawi.plugin import HawiPlugin, HawiResource, ResourceContent

class DataPlugin(HawiPlugin):
    @property
    def resources(self):
        return [
            HawiResource(
                uri="data://config/app",
                name="App Configuration",
                description="应用配置",
                mime_type="application/json",
                handler=self._read_config,
            ),
        ]

    async def _read_config(self, uri: str) -> ResourceContent:
        config = {"debug": True, "version": "2.0"}
        return ResourceContent(
            uri=uri,
            mime_type="application/json",
            text=json.dumps(config),
        )
```

---

## Plugin Events（插件事件）

插件可以通过事件总线发送自定义事件：

```python
class MonitorPlugin(HawiPlugin):
    @after_tool_calling
    def report(self, agent, tool_name, arguments, result, ctx):
        self.emit_plugin_event(
            "plugin.event",
            {
                "event_name": "tool.executed",
                "tool_name": tool_name,
                "duration_ms": ctx.duration_ms,
                "success": result.success,
            },
            run_id=ctx.run_id,
            tool_call_id=ctx.tool_call_id,
        )
```

---

## GUI 配置 Schema

插件可以提供 UI 配置界面：

```python
class MyPlugin(HawiPlugin):
    @classmethod
    def gui_config_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "title": "Enabled",
                    "description": "启用插件",
                    "default": True,
                },
                "api_key": {
                    "type": "string",
                    "title": "API Key",
                    "description": "服务 API 密钥",
                },
                "max_results": {
                    "type": "integer",
                    "title": "最大结果数",
                    "default": 10,
                },
                "mode": {
                    "type": "string",
                    "title": "运行模式",
                    "enum": ["fast", "balanced", "accurate"],
                    "default": "balanced",
                },
            },
            "required": ["api_key"],
            "additionalProperties": False,
        }

    @classmethod
    def gui_default_config(cls) -> dict:
        return {
            "enabled": True,
            "max_results": 10,
            "mode": "balanced",
        }
```

---

## Clone 与 Fork 支持

当 Agent 被 clone/fork 时，插件的 `clone()` 方法被调用：

```python
class StatefulPlugin(HawiPlugin):
    def __init__(self):
        self._cache: dict = {}
        self._counter = 0

    def clone(self) -> "StatefulPlugin":
        new = StatefulPlugin()
        # 深度复制状态（避免共享 mutable 对象）
        new._cache = dict(self._cache)  # 浅拷贝，视需求决定
        new._counter = self._counter
        return new
```

**默认行为**：返回 `self`（适用于无状态插件，如 EnvironPromptPlugin 的默认配置加载器）。

**工厂模式**：对于需要完全隔离的有状态插件，使用 `plugin_factories`：

```python
agent = HawiAgent(
    plugin_factories=[lambda: MyStatefulPlugin()],
)
```

---

## 配置文件加载

插件常用模式：从 `.hawi/` 读取 YAML 配置文件，不存在则使用默认值。

推荐实现：

```python
from pathlib import Path

CONFIG_FILENAME = "my_plugin.yaml"
CONFIG_CANDIDATES = [
    Path(".hawi") / CONFIG_FILENAME,          # 项目本地
    Path.home() / ".hawi" / CONFIG_FILENAME,   # 用户全局
]

class MyPlugin(HawiPlugin):
    def __init__(self):
        self._config = self._load_config()

    @staticmethod
    def _load_config() -> dict:
        for candidate in CONFIG_CANDIDATES:
            resolved = candidate.resolve()
            if resolved.is_file():
                try:
                    import yaml
                    with open(resolved) as f:
                        return dict(yaml.safe_load(f) or {})
                except Exception:
                    break
        return {"enabled": True, ...}  # 默认值
```

---

## 最佳实践

### 1. Hook 设计原则

- **`before_session`** — 适合做**一次性初始化**：修改 system prompt、建立连接、加载配置
- **`before_conversation`** — 适合**回合级注入**：注入用户环境信息、添加上下文提示
- **`before_model_call`** — 适合**动态干预**：替换模型、注入临时上下文
- **`after_model_call`** — 适合**分析响应**：审计输出、触发 reinvoke
- **`before_tool_calling`** — 适合**安全检查**：权限验证、参数修改、缓存
- **`after_tool_calling`** — 适合**记录结果**：统计、缓存写入

### 2. 注入内容规范

注入到 context 的内容应**清晰标记**为框架自动注入，与用户输入区分：

```python
ENVIRON_TAG_BEGIN = "<hawi-environ>"
ENVIRON_TAG_END = "</hawi-environ>"

def _stamp_environ_block(body: str) -> str:
    header = (
        "[Environment Information — auto-injected by "
        "MyPlugin. This is NOT user input.]"
    )
    return (
        f"\n\n{ENVIRON_TAG_BEGIN}\n"
        f"{header}\n\n{body}\n"
        f"{ENVIRON_TAG_END}\n"
    )
```

### 3. 避免重复注入

使用 sentinel 标记或状态字段防止多次注入：

```python
# 方案 A：标记检查
def _strip_existing_blocks(parts):
    return [p for p in parts if TAG_BEGIN not in str(p.get("text", ""))]

# 方案 B：状态检查
if self._already_injected:
    return
self._already_injected = True
```

### 4. 错误处理

```python
@before_tool_calling
def safe_check(self, agent, tool_name, arguments, ctx):
    try:
        if not self._has_permission(tool_name):
            return HookResult.skip(
                ToolResult(success=False, error="无权限")
            )
    except Exception as e:
        # 可预期的失败用 HookResult，不要抛异常
        return HookResult.skip(
            ToolResult(success=False, error=f"检查失败: {e}")
        )
```

### 5. 文件扫描性能

扫描 CWD 时注意：
- 跳过 `.git/`, `node_modules/`, `__pycache__/` 等
- 限制扫描深度（通常 ≤ 4 层）
- 限制返回数量（通常 ≤ 30 条）
- 使用 `os.walk` 的 `topdown=True` + 修改 `_dirs` 来剪枝

---

## 完整示例：EnvironPromptPlugin

这是插件系统的完整参考实现，展示了：

1. ✅ **配置文件读取** — 从 `.hawi/environ_prompt.yaml` 读取配置
2. ✅ **System prompt 注入** — 使用 `@before_session` 一次性注入 session 级环境信息和项目记忆文件
3. ✅ **User prompt 注入** — 使用 `@before_conversation` 在用户消息前插入动态环境信息
4. ✅ **内容标记** — 使用 `<hawi-environ>` 标记和声明文字区分框架注入
5. ✅ **文件变更追踪** — 追踪上一次注入时间戳，报告修改的文件
6. ✅ **配置灵活性** — 每个特性可独立开关，支持自定义文本和文件引用
7. ✅ **GUI 集成** — 已注册到 GUI 插件目录
8. ✅ **Clone 支持** — 正确复制运行时状态
9. ✅ **Project steering** — 按配置文件名顺序选择 `AGENTS.md` / `CLAUDE.md`，并按目录 scope 从外到内生效

### 关键代码片段

```python
class EnvironPromptPlugin(HawiPlugin):
    def __init__(self):
        self._config = self._load_config()
        self._last_prompt_ts: float = 0.0
        self._session_started: bool = False

    @before_session
    def inject_system_prompt_env(self, agent, ctx):
        """在 system prompt 末尾追加静态环境信息"""
        if self._session_started:
            return
        self._session_started = True

        parts = []
        if config.get("include_session_info", True):
            parts.append(self._format_session_info())
        if config.get("include_project_steering", True):
            parts.append(self._format_project_steering())
        # ... 追加到 system_prompt

    @before_conversation
    def inject_user_prompt_env(self, agent, ctx):
        """在用户消息前插入动态环境信息"""
        parts = []
        if config.get("include_cwd", True):
            parts.append(f"Current directory: {Path.cwd()}")
        if config.get("include_modified_files", True):
            parts.append(self._format_modified_files(self._last_prompt_ts))
        # ... 注入到用户消息前
        self._last_prompt_ts = time.time()
```

完整源码见 `hawi_plugins/environ_prompt_plugin/plugin.py`。

---

## 参考与调试

### 查看已注册的插件

```python
agent = HawiAgent(model=model, plugins=[MyPlugin()])
print("Hooks:", agent.plugins.get_hooks("before_conversation"))
print("Tools:", agent.plugins.get_tools())
print("Plugin count:", len(agent.plugins.get_plugins()))
```

### 动态添加/移除

```python
# 在运行时添加 Hook
agent.plugins.add_hook("before_model_call", my_hook_fn)

# 动态添加 Tool
from hawi.tool import AgentTool
agent.plugins.add_tool(my_tool)

# 隐藏工具（不让模型看到）
agent.plugins.mask_tool("dangerous_tool")
agent.plugins.unmask_tool("dangerous_tool")
```

### 关键源文件索引

| 文件 | 用途 |
|------|------|
| `hawi/plugin/plugin.py` | `HawiPlugin` 基类 + 事件辅助方法 |
| `hawi/plugin/hook_context.py` | `HookContext` + `HookResult` |
| `hawi/plugin/decorators.py` | Hook 装饰器定义 |
| `hawi/plugin/types.py` | 类型别名 |
| `hawi/plugin/manager.py` | `PluginManager` + 动态管理 |
| `hawi_engine/runtime.py` | 插件创建 + 注册 |
| `hawi_engine/inspect.py` | GUI 插件目录 |
| `hawi/agent/agent.py` | Agent 执行流程（Hook 调用位置） |
| `hawi/agent/context.py` | `AgentContext`（消息管理） |
