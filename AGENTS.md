# AGENTS.md

## 项目简介

Hawi 是一个轻量级 AI Agent 框架，核心目标是把多模型适配、工具/插件调用、事件流、会话持久化和桌面 GUI 组合成一个可运行的 Agent 系统。

## 运行原理

Agent 侧由 `HawiAgent` 维护对话上下文，调用模型产生回复或工具调用；工具由 plugin 注册并通过 `ToolExecutor` 执行；运行中的状态、事件、队列和会话历史会通过 `events/`、`runner/`、`session/` 持续流转与落盘。

GUI 侧启动 `hawi-engine` 作为常驻核心进程，通过 stdio/tcp/http 协议发送命令、接收事件，再由 React renderer 渲染聊天、工具、队列、插件状态和历史会话。

## 组织结构

- `hawi/`：Python 核心包，包含 `agent/`、`models/`、`tool/`、`plugin/`、`events/`、`session/`、`engine/` 和内置插件。
- `hawi_gui/`：当前桌面 GUI，采用 shadcn 风格工作台界面，包含 React renderer、Electron main/preload、Tauri shell 和共享协议类型；以后 GUI 开发默认只改这里。
- `hawi_legacy_gui/`：旧版桌面 GUI 封存目录。除非用户明确要求维护旧版，否则不要修改、重构、格式化或迁移这里的文件。
- `hawi_engine/`：兼容入口包。
- `test/`：Python 单元与集成测试。
- `docs/`、`assets/`、`packaging/`：文档、资源和打包辅助文件。
- `main.py`：本地交互式 Agent CLI 示例入口。

## 构建方法

Python 核心：

```bash
uv sync
uv run python main.py [model] [prompt]
uv run hawi-engine --inspect
```

GUI：

```bash
cd hawi_gui
npm install
npm run build
npm run dev
```

桌面打包：

```bash
cd hawi_gui
npm run build:core
npm run package
```

## 测试方法

Python：

```bash
uv run pytest
uv run pytest test/unit
uv run pytest test/integration
```

GUI / Tauri：

```bash
cd hawi_gui
npm test
npm run build

cd src-tauri
cargo check
```

部分集成测试需要 `.hawi/models.yaml` 或 `~/.hawi/models.yaml` 中的模型/API 配置。
