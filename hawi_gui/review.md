# Hawi GUI Code Review

> 审查日期: 2025-07-21
> 审查范围: `hawi_gui/` 目录下的 Electron + React + TypeScript 前端代码
> 修复日期: 2025-07-21
> 修复状态: **14/18 已完成**, 3 项暂缓/保留, 1 项未处理

---

## 总览

整体观感：这是一个**干净、结构清晰**的 GUI 应用。三层架构（main → preload → renderer）职责分明，状态管理采用可预测的 reducer 模式，测试覆盖了核心逻辑，安全性实践到位。

以下按**优先级**（P0=必须修复, P1=建议修改, P2=锦上添花）列出可改进的点。

---

## P0 — 必须修复

### 1. ✅ package.json 依赖分类错误

**文件**: `package.json` (line 13-29)

**问题**: `electron`、`@vitejs/plugin-react`、`vite` 被放在了 `dependencies` 里。

**修复**: 已移到 `devDependencies`。`dependencies` 现仅保留运行时依赖（`react`、`react-dom`、`highlight.js`、`lucide-react`、`markdown-it`）。

---

## P1 — 建议修改

### 2. ⏸️ App.tsx 过大（874行），应拆分组件

**文件**: `src/renderer/App.tsx` (全文件)

**状态**: 暂缓。需要独立 PR + 测试验证。建议拆分方案已记录在下方。

<details>
<summary>建议拆分方案</summary>

| 新文件 | 职责 |
|---|---|
| `App.tsx` | 顶层布局 + 状态注入（大幅精简） |
| `ChatPanel.tsx` | 聊天区 + 自动滚动逻辑 |
| `MessageInput.tsx` | 输入框 + 优先级选择 + IME 处理 |
| `ChatBubble.tsx` | 通用气泡（user/agent/system/error/meta/debug） |
| `ThinkingBubble.tsx` | 思考气泡 + 折叠动画 |
| `ToolBubble.tsx` | 工具调用气泡 + 参数展示 + 折叠 |
| `ModelDialog.tsx` | 模型选择对话框 + 搜索 |
| `PluginDialog.tsx` | 插件配置对话框 + 表单 |
| `SchemaField.tsx` | JSON Schema 字段渲染器 |
| `Modal.tsx` | 通用 Modal 组件 |

</details>

### 3. ✅ 构建脚本缺少热重载开发模式

**文件**: `package.json` (line 8-11)

**修复**: 添加了 `dev:fast` 脚本 (`tsc -p tsconfig.node.json && vite build && electron .`)。完整 HMR 方案（如 `electron-vite`）作为后续优化项。

### 4. ✅ Plugin SchemaField 不支持复杂 JSON Schema 类型

**文件**: `src/renderer/App.tsx` (line 706-728, `SchemaField` 函数)
**关联文件**: `src/shared/protocol.ts` (`JsonSchemaObject`)

**修复**:
- `JsonSchemaObject` 新增 `enum?: unknown[]` 字段
- `SchemaField` 新增 `enum` 支持 → 渲染为 `<select>` 下拉框
- `SchemaField` 新增 `error` prop → 内联红色错误提示（配合 #8）
- CSS 新增 `.schema-field select` 和 `.schema-error` 样式

> `array`、`object`、`file` 类型暂由 text input 处理，后续可按需扩展。

### 5. ✅ 多处硬编码 Magic Number

**修复**: 已提取为命名常量

| 位置 | 硬编码值 | 新常量 |
|---|---|---|
| `main.ts` line 244 | `800` | `GRACEFUL_SHUTDOWN_TIMEOUT_MS` |
| `main.ts` line 269 | `15_000` | `DEFAULT_COMMAND_TIMEOUT_MS` |
| `state.ts` line 236 | `-199` | `MAX_DEBUG_LINES` |
| `state.ts` line 425 | `1200` | `MAX_RESULT_PREVIEW_LENGTH` |

### 6. ⏸️ `reduceCoreEvent` switch-case 过大（~200行）

**文件**: `src/renderer/state.ts` (line 61-252)

**状态**: 暂缓。state.ts 在修复期间收到了外部更新（新增 `tool.call_start` 去重逻辑等），策略模式重构需在稳定版本上重新执行。

<details>
<summary>建议重构方案</summary>

```typescript
const handlers: Record<string, (state: AppState, payload: Record<string, unknown>) => AppState> = {
  "run.start": handleRunStart,
  "run.text_delta": handleRunTextDelta,
  "run.thinking_delta": handleRunThinkingDelta,
  "tool.call_start": handleToolCallStart,
  "tool.call_delta": handleToolCallDelta,
  "tool.call_stop": handleToolCallStop,
  "tool.result": handleToolResult,
  // ...
};

export function reduceCoreEvent(state: AppState, frame: CoreFrame): AppState {
  const handler = handlers[frame.type];
  if (!handler) return state;
  return handler(state, (frame.payload ?? {}) as Record<string, unknown>);
}
```

</details>

### 7. ✅ `escapeHtml` 和 `escapeText` 存在重复代码

**文件**: `src/renderer/App.tsx` (line 850-873)

**修复**: `escapeText` 现在内部调用 `escapeHtml` 后追加 `\n` 替换。

### 8. ✅ PluginDialog 错误信息未内联展示

**文件**: `src/renderer/App.tsx` (line 697)

**修复**:
- 错误状态从 `string[]` 改为 `Record<string, string>`（按 `pluginKey.field` 索引）
- `SchemaField` 接收 `error` prop，在字段下方显示红色内联错误
- 全局错误摘要保留在 footer 中作为兜底展示

### 9. ✅ `writeFrame` 内联类型应复用 `CoreCommand`

**文件**: `src/main/main.ts` (line 278-280)

**修复**: `writeFrame` 参数类型改为 `CoreCommand`；`sendCommand` 中的 frame 对象也标注为 `CoreCommand`。

---

## P2 — 锦上添花

### 10. ✅ 缺少格式化/Lint 配置

**修复**: 添加了 `.eslintrc.cjs`（TypeScript + React Hooks 规则）和 `.prettierrc`。

### 11. ✅ `.json-*` 相关 CSS 样式未被使用

**文件**: `src/renderer/styles.css` (line 567-634)

**修复**: 移除了约 70 行未使用的 `.json-tree`、`.json-branch`、`.json-children`、`.json-row`、`.json-key`、`.json-value` 等样式。

### 12. ✅ `.argument-*` 和 `.json-*` CSS 高度重复

**修复**: 清理 `.json-*` 样式后问题自然消除（#11）。

### 13. ✅ System Prompt "应用" 按钮缺少视觉反馈

**文件**: `src/renderer/App.tsx` (line 232-236)

**修复**: 添加 `systemPromptApplied` 状态。点击应用后按钮短暂显示"已应用"文字并禁用 1.5 秒，提供即时视觉反馈。

### 14. ⚠️ 测试覆盖率不足

**状态**: 未处理。4 个测试文件共 ~425 行，覆盖了 reducer + 工具函数，但缺少：
- `CoreProcess` 类（主进程逻辑）的测试
- 组件渲染测试（ChatBubble、ThinkingBubble、ToolBubble、ModelDialog、PluginDialog）
- 用户交互流程测试
- IPC 通信流程测试

> 作为独立的测试增强任务处理。

### 15. ↩️ `AppState` 中存在未使用字段

**状态**: 已保留。`metadataLines`、`debugLines`、`errors` 仍被现有测试和调试面板逻辑使用，本轮未移除。

### 16. ✅ 代码块样式层级复杂

**文件**: `src/renderer/styles.css`

**修复**: 移除了 `.markdown pre.code-block` 和 `.markdown pre.code-block code` 的覆盖层，padding 直接在 `.markdown pre, .bubble pre` 上设置，减少 CSS 覆盖层级。

### 17. ✅ sendCommand 超时不因命令类型而异

**文件**: `src/main/main.ts`

**修复**: `sendCommand` 新增可选 `timeoutMs` 参数（默认 `DEFAULT_COMMAND_TIMEOUT_MS`），调用方可根据命令类型传入不同超时。

### 18. ✅ 持续集成 / 自动化

**修复**: 添加了 `.github/workflows/gui-ci.yml`，PR 自动执行 npm ci → build → test。

---

## 做得好的（值得保持）

1. **安全性**：`contextIsolation: true`、`nodeIntegration: false`、CSP 配置、外部链接通过 `shell.openExternal` 打开
2. **IME 输入法处理**：`composing` + `keyCode 229` + `isComposing` 三重判断，中文输入体验好
3. **自动滚动**：用户选中文本时暂停自动滚动，细节到位
4. **流式 JSON 参数解析**：`parseToolArguments` + `closePartialTopLevelObject` + `lastTopLevelComma` 处理不完整 JSON 流，实现精妙
5. **折叠动画**：CSSTransition 配合 `grid-template-rows: 1fr ↔ 0fr` 实现平滑折叠
6. **`prefers-reduced-motion`**：尊重用户系统设置
7. **Thinking 自动折叠**：完成思考后自动折叠，并保留 excerpt 预览
8. **工具描述分离**：`tool_call_description` 注入参数与业务参数分离展示
9. **测试**：reducer 纯函数测试充分，覆盖了流式参数、thinking 折叠、工具结果等边界情况
10. **NDJSON 解析**：`parseNdjsonChunk` 跨 chunk 拼包处理正确

---

## 总结

| 等级 | 总数 | 已完成 | 暂缓 | 说明 |
|---|---|---|---|---|
| P0（必须修） | 1 | 1 | 0 | 依赖分类 ✅ |
| P1（建议修） | 8 | 6 | 2 | 组件拆分、策略模式暂缓 |
| P2（锦上添花） | 9 | 7 | 2 | 测试覆盖率未处理，AppState 字段保留 |

**暂缓项**：App.tsx 组件拆分、reduceCoreEvent 策略模式重构、AppState 字段清理需要更多时间和测试支持，建议作为独立 PR 处理。
