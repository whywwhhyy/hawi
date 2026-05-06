import { memo, useEffect, useMemo, useReducer, useRef, useState, type ReactNode } from "react";
import MarkdownIt from "markdown-it";
import { Bot, Brain, Check, ChevronDown, ChevronRight, Plug, RotateCcw, Send, Square, Trash2, Wrench, X } from "lucide-react";
import type { CoreCommandType, CoreFrame, GuiMetadata, JsonSchemaObject, PersistedConfig, PluginCatalogItem, QueueKind } from "../shared/protocol";
import { VERSION } from "../shared/protocol";
import { coerceSchemaValue, mergePluginDefaults, validatePluginConfig } from "./pluginConfig";
import { createInitialState, reduceCoreEvent, type ChatNode } from "./state";

const markdown = new MarkdownIt({ html: false, linkify: true, breaks: true });
const AUTO_SCROLL_BOTTOM_THRESHOLD_PX = 5;

const queueLabels: Record<QueueKind, string> = {
  normal: "普通",
  high_prio: "高优",
  urgent: "紧急"
};

export default function App() {
  const [metadata, setMetadata] = useState<GuiMetadata | null>(null);
  const [config, setConfig] = useState<PersistedConfig | null>(null);
  const [state, dispatch] = useReducer(reduceCoreEvent, undefined, createInitialState);
  const [input, setInput] = useState("");
  const [queue, setQueue] = useState<QueueKind>("high_prio");
  const [modelDialogOpen, setModelDialogOpen] = useState(false);
  const [pluginDialogOpen, setPluginDialogOpen] = useState(false);
  const chatRef = useRef<HTMLDivElement | null>(null);
  const followTailRef = useRef(true);
  const selectingChatRef = useRef(false);

  useEffect(() => {
    window.hawi.getMetadata().then((meta) => {
      setMetadata(meta);
      setConfig(meta.config);
      if (!meta.config.modelName) {
        setModelDialogOpen(true);
      }
    }).catch((error) => {
      dispatch(errorFrame(error));
    });
  }, []);

  useEffect(() => {
    const offEvent = window.hawi.onCoreEvent((frame) => dispatch(frame));
    const offLog = window.hawi.onCoreLog((message) => {
      dispatch({ version: VERSION, type: "debug.info", payload: { message } });
    });
    return () => {
      offEvent();
      offLog();
    };
  }, []);

  useEffect(() => {
    const element = chatRef.current;
    if (!element || selectingChatRef.current || hasChatSelection()) return;
    if (!followTailRef.current && !isNearChatBottom(element)) return;
    element.scrollTo({ top: element.scrollHeight });
    followTailRef.current = true;
  }, [state.nodes]);

  useEffect(() => {
    function syncSelection() {
      selectingChatRef.current = hasChatSelection();
      updateFollowTail();
    }
    window.addEventListener("mouseup", syncSelection);
    document.addEventListener("selectionchange", syncSelection);
    return () => {
      window.removeEventListener("mouseup", syncSelection);
      document.removeEventListener("selectionchange", syncSelection);
    };
  }, []);

  const showDebug = config?.showDebug ?? true;
  const selectedModel = config?.modelName || "-";
  const systemPromptLocked = state.nodes.some(isConversationNode);

  function updateFollowTail() {
    const element = chatRef.current;
    if (!element) return;
    followTailRef.current = isNearChatBottom(element);
  }

  function hasChatSelection() {
    const element = chatRef.current;
    const selection = window.getSelection();
    if (!element || !selection || selection.isCollapsed || selection.rangeCount === 0) {
      return false;
    }
    const anchor = selection.anchorNode;
    const focus = selection.focusNode;
    return Boolean((anchor && element.contains(anchor)) || (focus && element.contains(focus)));
  }

  async function sendCommand(type: CoreCommandType, payload: Record<string, unknown>) {
    try {
      await window.hawi.sendCommand(type, payload);
    } catch (error) {
      dispatch(errorFrame(error));
    }
  }

  async function saveAndSet(nextConfig: PersistedConfig) {
    const saved = await window.hawi.saveConfig(nextConfig);
    setConfig(saved);
    return saved;
  }

  async function restartWith(nextConfig: PersistedConfig) {
    const saved = await saveAndSet(nextConfig);
    try {
      await window.hawi.restartCore(saved);
    } catch (error) {
      dispatch(errorFrame(error));
    }
  }

  async function submitInput() {
    const text = input.trim();
    if (!text) return;
    if (text.startsWith("/")) {
      await runSlashCommand(text);
      setInput("");
      return;
    }
    setInput("");
    await sendCommand("enqueue", { content: text, queue, metadata: {} });
  }

  async function runSlashCommand(text: string) {
    const command = text.slice(1).toLowerCase();
    if (command === "clear") await clearConversation();
    else if (command === "cq") await sendCommand("clear_queue", { queue: "normal" });
    else if (command === "chq") await sendCommand("clear_queue", { queue: "high_prio" });
    else if (command === "cuq") await sendCommand("clear_queue", { queue: "urgent" });
    else if (command === "ca") await sendCommand("clear_queue", { queue: "all" });
    else dispatch({ version: VERSION, type: "debug.info", payload: { message: `未知命令: ${text}` } });
  }

  async function clearConversation() {
    window.getSelection()?.removeAllRanges();
    followTailRef.current = true;
    selectingChatRef.current = false;
    dispatch({ version: VERSION, type: "gui.clear_chat", payload: {} });
    await sendCommand("clear_context", {});
  }

  async function applySystemPrompt() {
    if (!config) return;
    await sendCommand("set_system_prompt", { system_prompt: config.systemPrompt });
    await saveAndSet(config);
  }

  async function selectModel(modelName: string) {
    if (!config || !metadata) return;
    const nextConfig = { ...config, modelName };
    setModelDialogOpen(false);
    setConfig(nextConfig);
    try {
      await window.hawi.sendCommand("switch_model", { model_name: modelName });
      await saveAndSet(nextConfig);
    } catch {
      await restartWith(nextConfig);
    }
  }

  async function applyPlugins(selectedPlugins: string[], pluginConfigs: Record<string, Record<string, unknown>>) {
    if (!config) return;
    const nextConfig = { ...config, selectedPlugins, pluginConfigs };
    setConfig(nextConfig);
    setPluginDialogOpen(false);
    await saveAndSet(nextConfig);
    await sendCommand("apply_plugins", { selected_plugins: selectedPlugins, plugin_configs: pluginConfigs });
  }

  if (!metadata || !config) {
    return <div className="boot">Loading Hawi metadata...</div>;
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="status-strip">
          <StatusCell label="Scheduler" value={state.schedulerState} />
          <StatusCell label="Agent" value={state.agentState} />
          <div className="queue-status">Queue U/H/N: {state.queueLengths.urgent}/{state.queueLengths.high_prio}/{state.queueLengths.normal}</div>
        </div>
        <button className="tool-button" title="插件配置" onClick={() => setPluginDialogOpen(true)}>
          <Plug size={17} /> 插件配置
        </button>
        <button className="tool-button" title="切换模型" onClick={() => setModelDialogOpen(true)}>
          <Bot size={17} /> Model: {selectedModel}
        </button>
      </header>

      <section className={`prompt-row ${systemPromptLocked ? "locked" : ""}`}>
        <label>System Prompt:</label>
        <textarea
          value={config.systemPrompt}
          disabled={systemPromptLocked}
          title={systemPromptLocked ? "清空消息后可编辑" : "System Prompt"}
          onChange={(event) => setConfig({ ...config, systemPrompt: event.target.value })}
        />
        <button
          className="tool-button"
          title="应用系统提示"
          disabled={systemPromptLocked}
          onClick={applySystemPrompt}
        >
          <Check size={17} /> 应用
        </button>
      </section>

      <main
        className="chat-panel"
        ref={chatRef}
        onScroll={updateFollowTail}
        onMouseDown={() => {
          selectingChatRef.current = true;
        }}
      >
        {state.nodes
          .filter((node) => showDebug || node.kind !== "debug")
          .map((node) => (
            <ChatBubble node={node} key={node.id} />
          ))}
      </main>

      <section className="control-row">
        <span className="label">优先级:</span>
        {(["normal", "high_prio", "urgent"] as QueueKind[]).map((key) => (
          <button
            key={key}
            className={`segment ${queue === key ? "active" : ""}`}
            onClick={() => setQueue(key)}
          >
            {queueLabels[key]}
          </button>
        ))}
        <button className="icon-button" title="清空上下文和屏幕" onClick={clearConversation}>
          <Trash2 size={17} />
        </button>
        <label className="debug-toggle">
          <input
            type="checkbox"
            checked={showDebug}
            onChange={(event) => {
              const next = { ...config, showDebug: event.target.checked };
              setConfig(next);
              void saveAndSet(next);
            }}
          />
          Debug
        </label>
        <button className="icon-button" title="重启 core" onClick={() => restartWith(config)}>
          <RotateCcw size={17} />
        </button>
      </section>

      <footer className="input-row">
        <textarea
          value={input}
          placeholder="输入消息"
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              void submitInput();
            }
            if (event.key === "Tab" && event.shiftKey) {
              event.preventDefault();
              setQueue(queue === "normal" ? "high_prio" : queue === "high_prio" ? "urgent" : "normal");
            }
          }}
        />
        <button className="primary-button" onClick={submitInput}>
          <Send size={18} /> 发送
        </button>
        <button className="danger-button" onClick={() => sendCommand("interrupt", { reason: "user" })}>
          <Square size={16} /> 停止
        </button>
      </footer>

      {modelDialogOpen && (
        <ModelDialog
          models={metadata.inspect.models}
          current={config.modelName}
          onClose={() => setModelDialogOpen(false)}
          onSelect={selectModel}
        />
      )}
      {pluginDialogOpen && (
        <PluginDialog
          catalog={metadata.inspect.plugin_catalog}
          selectedPlugins={config.selectedPlugins}
          pluginConfigs={config.pluginConfigs}
          onClose={() => setPluginDialogOpen(false)}
          onApply={applyPlugins}
        />
      )}
    </div>
  );
}

function StatusCell({ label, value }: { label: string; value: string }) {
  return (
    <div className={`status-cell state-${value.toLowerCase()}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

const ChatBubble = memo(function ChatBubble({ node }: { node: ChatNode }) {
  if (node.kind === "divider") {
    return (
      <div className="run-divider">
        <span>{node.content}</span>
      </div>
    );
  }
  if (node.kind === "meta" || node.kind === "system" || node.kind === "debug") {
    return (
      <div className={`note-line ${node.kind}`}>
        {node.content}
      </div>
    );
  }
  if (node.kind === "tool" && node.tool) {
    return <ToolBubble node={node} />;
  }
  if (node.kind === "thinking") {
    return <ThinkingBubble node={node} />;
  }
  const html = node.kind === "agent" ? markdown.render(node.content) : escapeText(node.content);
  return (
    <article className={`bubble ${node.kind}`}>
      <div className="bubble-head">
        <span>{node.kind === "user" ? queueLabels[node.queue ?? "normal"] : labelForKind(node.kind)}</span>
      </div>
      <div className="markdown" dangerouslySetInnerHTML={{ __html: html }} />
    </article>
  );
});

const ThinkingBubble = memo(function ThinkingBubble({ node }: { node: ChatNode }) {
  const [collapsed, setCollapsed] = useState(() => node.complete === true);
  const autoCollapsedRef = useRef(node.complete === true);
  const html = markdown.render(node.content);

  useEffect(() => {
    if (node.complete === true && !autoCollapsedRef.current) {
      setCollapsed(true);
      autoCollapsedRef.current = true;
    }
  }, [node.complete]);

  return (
    <article className={`bubble thinking ${collapsed ? "collapsed" : ""}`}>
      <div className="bubble-head">
        <span><Brain size={15} /> Thinking</span>
        <button
          className="thinking-toggle"
          title={collapsed ? "展开思考内容" : "折叠思考内容"}
          aria-expanded={!collapsed}
          onClick={() => setCollapsed(!collapsed)}
        >
          {collapsed ? <ChevronRight size={15} /> : <ChevronDown size={15} />}
        </button>
      </div>
      {collapsed
        ? <div className="thinking-summary">{thinkingExcerpt(node.content)}</div>
        : <div className="markdown" dangerouslySetInnerHTML={{ __html: html }} />}
    </article>
  );
});

const ToolBubble = memo(function ToolBubble({ node }: { node: ChatNode }) {
  const tool = node.tool!;
  const hasStructuredArguments = tool.arguments !== undefined;
  return (
    <article className={`bubble tool ${tool.status}`}>
      <div className="bubble-head">
        <span><Wrench size={15} /> {tool.name}</span>
        <strong>{tool.status}</strong>
      </div>
      <details open>
        <summary>
          Arguments
          {tool.argsState !== "complete" && <span className="detail-state">receiving</span>}
        </summary>
        {hasStructuredArguments
          ? <ToolArguments value={tool.arguments} />
          : <div className="tool-hint">{tool.argsState === "complete" ? "No arguments." : "Receiving arguments..."}</div>}
      </details>
      {(tool.resultPreview || tool.status === "fail") && (
        <details open>
          <summary>Result {tool.durationMs ? `· ${tool.durationMs.toFixed(0)}ms` : ""}</summary>
          <pre>{tool.resultPreview || "Tool failed without an error message."}</pre>
        </details>
      )}
    </article>
  );
});

function ToolArguments({ value }: { value: unknown }) {
  if (isRecord(value)) {
    const entries = Object.entries(value);
    if (entries.length === 0) {
      return <div className="tool-hint">No arguments.</div>;
    }
    return (
      <div className="argument-list">
        {entries.map(([key, item]) => (
          <ArgumentRow key={key} name={key} value={item} />
        ))}
      </div>
    );
  }
  return (
    <div className="argument-list">
      <ArgumentRow name="value" value={value} />
    </div>
  );
}

function ArgumentRow({ name, value }: { name: string; value: unknown }) {
  const multiline = isMultilineArgumentValue(value);
  return (
    <div className={`argument-row ${multiline ? "multiline" : ""}`}>
      <div className="argument-name"><strong>{name}</strong>:</div>
      <div className="argument-value">
        {renderArgumentValue(value)}
      </div>
    </div>
  );
}

function renderArgumentValue(value: unknown): ReactNode {
  if (value === null) {
    return <span className="argument-primitive null">null</span>;
  }
  if (typeof value === "string") {
    return <span className="argument-primitive string">{value === "" ? "\"\"" : value}</span>;
  }
  if (typeof value === "number") {
    return <span className="argument-primitive number">{String(value)}</span>;
  }
  if (typeof value === "boolean") {
    return <span className="argument-primitive boolean">{String(value)}</span>;
  }
  if (Array.isArray(value) || isRecord(value)) {
    return <pre className="argument-code">{JSON.stringify(value, null, 2)}</pre>;
  }
  return <span className="argument-primitive unknown">{String(value)}</span>;
}

function isMultilineArgumentValue(value: unknown): boolean {
  if (typeof value === "string") {
    return value.includes("\n");
  }
  return Array.isArray(value) || isRecord(value);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

function ModelDialog({ models, current, onClose, onSelect }: { models: string[]; current: string; onClose: () => void; onSelect: (model: string) => void }) {
  const [filter, setFilter] = useState("");
  const grouped = useMemo(() => {
    const groups = new Map<string, string[]>();
    for (const model of models.filter((item) => item.toLowerCase().includes(filter.toLowerCase()))) {
      const [provider] = model.split("/", 1);
      groups.set(provider, [...(groups.get(provider) ?? []), model]);
    }
    return [...groups.entries()];
  }, [models, filter]);
  return (
    <Modal title="切换模型" className="model-modal" onClose={onClose}>
      <input className="search" value={filter} onChange={(event) => setFilter(event.target.value)} placeholder="搜索模型" />
      <div className="model-grid">
        {grouped.map(([provider, entries]) => (
          <section className="model-provider" key={provider}>
            <h3>{provider}</h3>
            {entries.map((model) => (
              <button className={model === current ? "model active" : "model"} key={model} onClick={() => onSelect(model)}>
                {model.split("/").slice(1).join("/")}
              </button>
            ))}
          </section>
        ))}
      </div>
    </Modal>
  );
}

function PluginDialog({ catalog, selectedPlugins, pluginConfigs, onClose, onApply }: { catalog: PluginCatalogItem[]; selectedPlugins: string[]; pluginConfigs: Record<string, Record<string, unknown>>; onClose: () => void; onApply: (selected: string[], configs: Record<string, Record<string, unknown>>) => void }) {
  const initial = mergePluginDefaults(catalog, selectedPlugins, pluginConfigs);
  const [selected, setSelected] = useState(new Set(initial.selectedPlugins));
  const [configs, setConfigs] = useState(initial.pluginConfigs);
  const [errors, setErrors] = useState<string[]>([]);

  function apply() {
    const selectedList = catalog.filter((item) => selected.has(item.key)).map((item) => item.key);
    const nextConfigs: Record<string, Record<string, unknown>> = {};
    const validation: string[] = [];
    for (const item of catalog) {
      if (!selected.has(item.key)) continue;
      nextConfigs[item.key] = configs[item.key] ?? {};
      validation.push(...validatePluginConfig(item, nextConfigs[item.key]));
    }
    if (validation.length) {
      setErrors(validation);
      return;
    }
    onApply(selectedList, nextConfigs);
  }

  return (
    <Modal title="插件配置" onClose={onClose}>
      <div className="plugin-list">
        {catalog.map((item) => (
          <section className="plugin-item" key={item.key}>
            <label className="plugin-enable">
              <input
                type="checkbox"
                checked={selected.has(item.key)}
                onChange={(event) => {
                  const next = new Set(selected);
                  if (event.target.checked) next.add(item.key);
                  else next.delete(item.key);
                  setSelected(next);
                }}
              />
              <strong>{item.label}</strong>
            </label>
            {Object.entries(item.schema.properties ?? {}).map(([field, schema]) => (
              <SchemaField
                key={field}
                field={field}
                schema={schema}
                disabled={!selected.has(item.key)}
                value={(configs[item.key] ?? item.defaults)[field] ?? schema.default ?? ""}
                onChange={(value) => {
                  setConfigs({
                    ...configs,
                    [item.key]: {
                      ...(configs[item.key] ?? item.defaults),
                      [field]: value
                    }
                  });
                }}
              />
            ))}
          </section>
        ))}
      </div>
      {errors.length > 0 && <div className="form-errors">{errors.join("\n")}</div>}
      <div className="modal-actions">
        <button className="tool-button" onClick={onClose}>取消</button>
        <button className="primary-button" onClick={apply}>应用</button>
      </div>
    </Modal>
  );
}

function SchemaField({ field, schema, disabled, value, onChange }: { field: string; schema: JsonSchemaObject; disabled: boolean; value: unknown; onChange: (value: unknown) => void }) {
  const label = schema.title ?? field;
  if (schema.type === "boolean") {
    return (
      <label className="schema-field inline">
        <span>{label}</span>
        <input type="checkbox" disabled={disabled} checked={Boolean(value)} onChange={(event) => onChange(event.target.checked)} />
      </label>
    );
  }
  return (
    <label className="schema-field">
      <span>{label}</span>
      <input
        disabled={disabled}
        type={schema.type === "integer" || schema.type === "number" ? "number" : "text"}
        value={String(value ?? "")}
        onChange={(event) => onChange(coerceSchemaValue(schema, event.target.value))}
      />
      {schema.description && <small>{schema.description}</small>}
    </label>
  );
}

function Modal({ title, children, className = "", onClose }: { title: string; children: ReactNode; className?: string; onClose: () => void }) {
  return (
    <div className="modal-backdrop">
      <section className={`modal ${className}`.trim()}>
        <header>
          <h2>{title}</h2>
          <button className="icon-button" onClick={onClose} title="关闭"><X size={18} /></button>
        </header>
        {children}
      </section>
    </div>
  );
}

function errorFrame(error: unknown): CoreFrame {
  return {
    version: VERSION,
    type: "error",
    payload: { message: error instanceof Error ? error.message : String(error) }
  };
}

function labelForKind(kind: string): string {
  if (kind === "agent") return "Hawi";
  if (kind === "thinking") return "Thinking";
  if (kind === "system") return "System";
  if (kind === "meta") return "Meta";
  if (kind === "debug") return "Debug";
  if (kind === "error") return "Error";
  return kind;
}

function isConversationNode(node: ChatNode): boolean {
  return node.kind === "user"
    || node.kind === "agent"
    || node.kind === "thinking"
    || node.kind === "tool"
    || node.kind === "divider";
}

export function thinkingExcerpt(value: string, maxChars = 120): string {
  const normalized = value.replace(/\s+/g, " ").trim();
  if (!normalized) return "";
  const chars = Array.from(normalized);
  if (chars.length <= maxChars) return normalized;
  return `${chars.slice(0, maxChars).join("")}...`;
}

export function isNearChatBottom(element: Pick<HTMLElement, "scrollHeight" | "scrollTop" | "clientHeight">): boolean {
  return element.scrollHeight - element.scrollTop - element.clientHeight < AUTO_SCROLL_BOTTOM_THRESHOLD_PX;
}

function escapeText(value: string): string {
  return value.replace(/[&<>"']/g, (char) => {
    switch (char) {
      case "&": return "&amp;";
      case "<": return "&lt;";
      case ">": return "&gt;";
      case "\"": return "&quot;";
      case "'": return "&#039;";
      default: return char;
    }
  }).replace(/\n/g, "<br />");
}
