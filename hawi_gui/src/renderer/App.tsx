import { memo, useEffect, useLayoutEffect, useMemo, useReducer, useRef, useState, type MouseEvent as ReactMouseEvent, type ReactNode, type WheelEvent } from "react";
import MarkdownIt from "markdown-it";
import hljs from "highlight.js/lib/core";
import bash from "highlight.js/lib/languages/bash";
import css from "highlight.js/lib/languages/css";
import javascript from "highlight.js/lib/languages/javascript";
import json from "highlight.js/lib/languages/json";
import markdownLanguage from "highlight.js/lib/languages/markdown";
import python from "highlight.js/lib/languages/python";
import typescript from "highlight.js/lib/languages/typescript";
import xml from "highlight.js/lib/languages/xml";
import yaml from "highlight.js/lib/languages/yaml";
import { Activity, Bot, Brain, Check, CheckCircle2, ChevronDown, ChevronRight, Circle, Copy, FileText, LoaderCircle, Plug, Plus, RotateCcw, Send, Square, Trash2, Wrench, X } from "lucide-react";
import type { CoreCommandType, CoreFrame, GuiMetadata, JsonSchemaObject, PersistedConfig, PluginCatalogItem, QueueKind, SessionMetaPayload } from "../shared/protocol";
import { VERSION } from "../shared/protocol";
import { coerceSchemaValue, mergePluginDefaults, validatePluginConfig } from "./pluginConfig";
import { createInitialState, reduceCoreEvent, type ChatNode, type ContextUsageState, type PluginArtifactState, type PluginMessageState, type PluginStatusState, type QueueMessageState, type ToolProgressState } from "./state";

hljs.registerLanguage("bash", bash);
hljs.registerLanguage("css", css);
hljs.registerLanguage("javascript", javascript);
hljs.registerLanguage("json", json);
hljs.registerLanguage("markdown", markdownLanguage);
hljs.registerLanguage("python", python);
hljs.registerLanguage("typescript", typescript);
hljs.registerLanguage("xml", xml);
hljs.registerLanguage("yaml", yaml);

const LANGUAGE_ALIASES: Record<string, string> = {
  html: "xml",
  js: "javascript",
  jsx: "javascript",
  md: "markdown",
  py: "python",
  sh: "bash",
  shell: "bash",
  ts: "typescript",
  tsx: "typescript",
  yml: "yaml",
  zsh: "bash"
};

const markdown = new MarkdownIt({
  html: false,
  linkify: true,
  breaks: true,
  highlight: highlightCode
});
const defaultLinkOpen = markdown.renderer.rules.link_open
  ?? ((tokens, idx, options, _env, self) => self.renderToken(tokens, idx, options));
markdown.renderer.rules.link_open = (tokens, idx, options, env, self) => {
  tokens[idx].attrSet("target", "_blank");
  tokens[idx].attrSet("rel", "noopener noreferrer");
  return defaultLinkOpen(tokens, idx, options, env, self);
};
const AUTO_SCROLL_BOTTOM_THRESHOLD_PX = 5;
const AUTO_SCROLL_SETTLE_FRAMES = 2;
const COPY_FEEDBACK_MS = 1200;
const SYSTEM_PROMPT_MAX_ROWS = 3;
const MESSAGE_INPUT_MAX_ROWS = 5;
const useBrowserLayoutEffect = typeof window === "undefined" ? useEffect : useLayoutEffect;

const queueLabels: Record<QueueKind, string> = {
  normal: "普通",
  high_prio: "优先",
  urgent: "紧急"
};

const userMessageTypeLabels = {
  normal: "普通消息",
  steer: "Steer",
  urgent: "紧急消息"
} as const;

export function renderPriorityStatusText(
  queueLengths: Record<QueueKind, number>,
  queueMessages?: Record<QueueKind, QueueMessageState[]>
): string {
  const highPriorityCount = hasHighPriorityWork(queueLengths, queueMessages) ? 1 : 0;
  return `优先 ${highPriorityCount} · 普通 ${normalQueueCount(queueLengths, queueMessages)}`;
}

export function shouldInitializeSessionState(metadata: GuiMetadata | null): boolean {
  return Boolean(metadata?.coreRunning);
}

function hasHighPriorityWork(
  queueLengths: Record<QueueKind, number>,
  queueMessages?: Record<QueueKind, QueueMessageState[]>
): boolean {
  return queueLengths.high_prio > 0 || (queueMessages?.high_prio.length ?? 0) > 0;
}

function normalQueueCount(
  queueLengths: Record<QueueKind, number>,
  queueMessages?: Record<QueueKind, QueueMessageState[]>
): number {
  return Math.max(queueLengths.normal, queueMessages?.normal.length ?? 0);
}

export default function App() {
  const [metadata, setMetadata] = useState<GuiMetadata | null>(null);
  const [config, setConfig] = useState<PersistedConfig | null>(null);
  const [state, dispatch] = useReducer(reduceCoreEvent, undefined, createInitialState);
  const [input, setInput] = useState("");
  const [queue, setQueue] = useState<QueueKind>("high_prio");
  const [modelDialogOpen, setModelDialogOpen] = useState(false);
  const [pluginDialogOpen, setPluginDialogOpen] = useState(false);
  const [queuePopoverOpen, setQueuePopoverOpen] = useState(false);
  const [sessionDialogOpen, setSessionDialogOpen] = useState(false);
  const [sessions, setSessions] = useState<SessionMetaPayload[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [sessionBusy, setSessionBusy] = useState(false);
  const chatRef = useRef<HTMLDivElement | null>(null);
  const systemPromptRef = useRef<HTMLTextAreaElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const configRef = useRef<PersistedConfig | null>(null);
  const pendingSystemPromptConfigRef = useRef<PersistedConfig | null>(null);
  const initializeSessionStateRef = useRef<() => Promise<void>>(async () => undefined);
  const applyingSystemPromptRef = useRef(false);
  const followTailRef = useRef(true);
  const selectingChatRef = useRef(false);
  const userScrollIntentRef = useRef(false);
  const isAutoScrollingRef = useRef(false);
  const autoScrollFrameRef = useRef<number | null>(null);
  const inputComposingRef = useRef(false);
  const inputCompositionEndTimerRef = useRef<number | null>(null);
  const coreRunning = shouldInitializeSessionState(metadata);

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
    if (!coreRunning) return;
    void initializeSessionStateRef.current();
  }, [coreRunning]);

  useEffect(() => {
    configRef.current = config;
  }, [config]);

  useBrowserLayoutEffect(() => {
    keepChatTailVisible();
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

  useEffect(() => () => {
    cancelPendingAutoScroll();
    if (inputCompositionEndTimerRef.current !== null) {
      window.clearTimeout(inputCompositionEndTimerRef.current);
    }
  }, []);

  const showDebug = config?.showDebug ?? true;
  const selectedModel = config?.modelName || "-";
  const systemPromptLocked = state.nodes.some(isConversationNode);

  useEffect(() => {
    resizeTextareaToRows(systemPromptRef.current, SYSTEM_PROMPT_MAX_ROWS);
  }, [config?.systemPrompt, systemPromptLocked]);

  useEffect(() => {
    resizeTextareaToRows(inputRef.current, MESSAGE_INPUT_MAX_ROWS);
  }, [input]);

  function updateFollowTail() {
    const element = chatRef.current;
    if (!element) return;
    followTailRef.current = resolveFollowTailOnScroll(
      followTailRef.current,
      isNearChatBottom(element),
      userScrollIntentRef.current,
      selectingChatRef.current,
      isAutoScrollingRef.current
    );
    userScrollIntentRef.current = false;
  }

  function markChatUserScrollIntent() {
    userScrollIntentRef.current = true;
    isAutoScrollingRef.current = false;
    cancelPendingAutoScroll();
  }

  function handleChatWheel(event: WheelEvent<HTMLElement>) {
    markChatUserScrollIntent();
    if (event.deltaY < 0) {
      followTailRef.current = false;
    }
  }

  function keepChatTailVisible(frame = 0) {
    const element = chatRef.current;
    if (!element || selectingChatRef.current || hasChatSelection()) return;
    if (!followTailRef.current && !isNearChatBottom(element)) return;

    followTailRef.current = true;
    isAutoScrollingRef.current = true;
    element.scrollTop = element.scrollHeight;

    if (frame >= AUTO_SCROLL_SETTLE_FRAMES || typeof window.requestAnimationFrame !== "function") {
      finishAutoScrolling();
      return;
    }
    cancelPendingAutoScroll();
    autoScrollFrameRef.current = window.requestAnimationFrame(() => {
      autoScrollFrameRef.current = null;
      keepChatTailVisible(frame + 1);
    });
  }

  function finishAutoScrolling() {
    if (typeof window.requestAnimationFrame !== "function") {
      isAutoScrollingRef.current = false;
      return;
    }
    cancelPendingAutoScroll();
    autoScrollFrameRef.current = window.requestAnimationFrame(() => {
      autoScrollFrameRef.current = null;
      isAutoScrollingRef.current = false;
    });
  }

  function cancelPendingAutoScroll() {
    if (autoScrollFrameRef.current === null || typeof window.cancelAnimationFrame !== "function") return;
    window.cancelAnimationFrame(autoScrollFrameRef.current);
    autoScrollFrameRef.current = null;
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

  async function sendCommand(type: CoreCommandType, payload: Record<string, unknown>): Promise<CoreFrame | null> {
    if (!coreRunning) {
      setModelDialogOpen(true);
      return null;
    }
    try {
      return await window.hawi.sendCommand(type, payload);
    } catch (error) {
      dispatch(errorFrame(error));
      return null;
    }
  }

  async function initializeSessionState() {
    const listFrame = await sendCommand("session_list", {});
    updateSessionsFromFrame(listFrame);
    const historyFrame = await sendCommand("session_history", {});
    applySessionHistoryFromFrame(historyFrame);
  }

  initializeSessionStateRef.current = initializeSessionState;

  async function refreshSessions() {
    const frame = await sendCommand("session_list", {});
    updateSessionsFromFrame(frame);
  }

  function updateSessionsFromFrame(frame: CoreFrame | null) {
    const payload = framePayload(frame);
    if (!payload) return;
    const nextSessions = normalizeSessionList(payload.sessions);
    setSessions(nextSessions);
    const nextCurrent = optionalPayloadString(payload.current_session_id);
    if (nextCurrent) {
      setCurrentSessionId(nextCurrent);
    }
  }

  function applySessionHistoryFromFrame(frame: CoreFrame | null) {
    const payload = framePayload(frame);
    if (!payload) return;
    const nextCurrent = optionalPayloadString(payload.session_id);
    if (nextCurrent) {
      setCurrentSessionId(nextCurrent);
    }
    dispatch({
      version: VERSION,
      type: "gui.load_session_history",
      payload: {
        message_history: Array.isArray(payload.message_history) ? payload.message_history : [],
        context_usage: payload.context_usage
      }
    });
    followTailRef.current = true;
  }

  async function openSessionDialog() {
    setSessionDialogOpen((value) => !value);
    setSessionBusy(true);
    try {
      await refreshSessions();
    } finally {
      setSessionBusy(false);
    }
  }

  async function loadSession(sessionId: string) {
    if (!sessionId || sessionId === currentSessionId) {
      setSessionDialogOpen(false);
      return;
    }
    setSessionBusy(true);
    try {
      const frame = await sendCommand("session_switch", { session_id: sessionId });
      applySessionHistoryFromFrame(frame);
      await refreshSessions();
      setSessionDialogOpen(false);
    } finally {
      setSessionBusy(false);
    }
  }

  async function deleteSession(sessionId: string) {
    if (!sessionId || sessionId === currentSessionId) {
      return;
    }
    if (!window.confirm(`删除 Session ${shortSessionId(sessionId)}？`)) {
      return;
    }
    setSessionBusy(true);
    try {
      const frame = await sendCommand("session_delete", { session_id: sessionId });
      if (!frame) return;
      setSessions((items) => items.filter((item) => item.session_id !== sessionId));
      await refreshSessions();
    } finally {
      setSessionBusy(false);
    }
  }

  async function newSession() {
    setSessionBusy(true);
    try {
      if (state.sessionMessageCount > 0) {
        await sendCommand("session_save_now", {});
      }
      const frame = await sendCommand("session_new", {});
      const payload = framePayload(frame);
      const sessionId = optionalPayloadString(payload?.session_id);
      if (sessionId) {
        setCurrentSessionId(sessionId);
      }
      dispatch({
        version: VERSION,
        type: "gui.load_session_history",
        payload: { message_history: [] }
      });
      followTailRef.current = true;
      await refreshSessions();
      setSessionDialogOpen(false);
    } finally {
      setSessionBusy(false);
    }
  }

  async function saveAndSet(nextConfig: PersistedConfig) {
    const saved = await window.hawi.saveConfig(nextConfig);
    setConfig(saved);
    setMetadata((current) => current ? { ...current, config: saved } : current);
    return saved;
  }

  async function restartWith(nextConfig: PersistedConfig) {
    const saved = await saveAndSet(nextConfig);
    try {
      await window.hawi.restartCore(saved);
      setMetadata((current) => current ? { ...current, config: saved, coreRunning: true } : current);
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

  function startInputComposition() {
    if (inputCompositionEndTimerRef.current !== null) {
      window.clearTimeout(inputCompositionEndTimerRef.current);
      inputCompositionEndTimerRef.current = null;
    }
    inputComposingRef.current = true;
  }

  function endInputComposition() {
    if (inputCompositionEndTimerRef.current !== null) {
      window.clearTimeout(inputCompositionEndTimerRef.current);
    }
    inputCompositionEndTimerRef.current = window.setTimeout(() => {
      inputComposingRef.current = false;
      inputCompositionEndTimerRef.current = null;
    }, 0);
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

  function applySystemPrompt(nextConfig: PersistedConfig) {
    pendingSystemPromptConfigRef.current = nextConfig;
    if (applyingSystemPromptRef.current) return;
    applyingSystemPromptRef.current = true;

    void (async () => {
      try {
        while (pendingSystemPromptConfigRef.current) {
          const pending = pendingSystemPromptConfigRef.current;
          pendingSystemPromptConfigRef.current = null;
          await sendCommand("set_system_prompt", { system_prompt: pending.systemPrompt });
          try {
            const saved = await window.hawi.saveConfig(pending);
            if (configRef.current?.systemPrompt === pending.systemPrompt) {
              setConfig(saved);
            }
          } catch (error) {
            dispatch(errorFrame(error));
          }
        }
      } finally {
        applyingSystemPromptRef.current = false;
      }
    })();
  }

  async function selectModel(modelName: string) {
    if (!config || !metadata) return;
    const nextConfig = { ...config, modelName };
    setModelDialogOpen(false);
    setConfig(nextConfig);
    try {
      if (metadata.coreRunning) {
        await window.hawi.sendCommand("switch_model", { model_name: modelName });
        await saveAndSet(nextConfig);
      } else {
        await restartWith(nextConfig);
      }
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
          <StatusCell
            value={combinedRunState(state.schedulerState, state.agentState)}
            title={`Scheduler: ${state.schedulerState} · Executor: ${state.agentState}`}
          />
          <PriorityStatusCell
            queueLengths={state.queueLengths}
            queueMessages={state.queueMessages}
            open={queuePopoverOpen}
            onToggle={() => setQueuePopoverOpen((value) => !value)}
          />
          <ContextUsageCell usage={state.contextUsage} />
          <SessionStatusCell
            messageCount={state.sessionMessageCount}
            sessions={sessions}
            currentSessionId={currentSessionId}
            open={sessionDialogOpen}
            busy={sessionBusy}
            onToggle={openSessionDialog}
            onSelect={loadSession}
            onDelete={deleteSession}
            onNew={newSession}
          />
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
          ref={systemPromptRef}
          rows={1}
          value={config.systemPrompt}
          disabled={systemPromptLocked}
          title={systemPromptLocked ? "当前会话已有消息，System Prompt 已锁定" : "System Prompt"}
          onChange={(event) => {
            resizeTextareaToRows(event.currentTarget, SYSTEM_PROMPT_MAX_ROWS);
            const nextConfig = { ...config, systemPrompt: event.target.value };
            setConfig(nextConfig);
            applySystemPrompt(nextConfig);
          }}
        />
      </section>

      <section className="workspace-row">
        <main
          className="chat-panel"
          ref={chatRef}
          onScroll={updateFollowTail}
          onWheel={handleChatWheel}
          onTouchStart={markChatUserScrollIntent}
          onMouseDown={() => {
            selectingChatRef.current = true;
            markChatUserScrollIntent();
          }}
        >
          {state.nodes
            .filter((node) => showDebug || node.kind !== "debug")
            .map((node) => (
              <ChatBubble node={node} key={node.id} />
            ))}
        </main>
        <PluginPreviewPanel
          artifacts={state.artifacts}
          artifactOrder={state.artifactOrder}
          selectedArtifactId={state.selectedArtifactId}
          messages={state.pluginMessages}
          statuses={state.pluginStatuses}
          toolProgress={state.toolProgress}
          onSelectArtifact={(artifactKey) => {
            dispatch({ version: VERSION, type: "gui.select_artifact", payload: { artifact_key: artifactKey } });
          }}
        />
      </section>

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
        <button className="icon-button" title="重启 Core 进程并应用当前配置" onClick={() => restartWith(config)}>
          <RotateCcw size={17} />
        </button>
      </section>

      <footer className="input-row">
        <textarea
          ref={inputRef}
          rows={1}
          value={input}
          placeholder="输入消息"
          onChange={(event) => {
            resizeTextareaToRows(event.currentTarget, MESSAGE_INPUT_MAX_ROWS);
            setInput(event.target.value);
          }}
          onCompositionStart={startInputComposition}
          onCompositionEnd={endInputComposition}
          onKeyDown={(event) => {
            if (shouldSubmitInputFromKeyEvent(event, inputComposingRef.current)) {
              event.preventDefault();
              void submitInput();
              return;
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

function StatusCell({ value, title }: { value: string; title?: string }) {
  const icon = statusIconForRunState(value);
  const showLabel = value !== "RUNNING";
  return (
    <div className={`status-cell state-${value.toLowerCase()}`} title={title}>
      <span className="status-icon" aria-hidden="true">{icon}</span>
      {showLabel && <strong>{value}</strong>}
    </div>
  );
}

function statusIconForRunState(value: string): ReactNode {
  if (value === "RUNNING") return <LoaderCircle size={15} />;
  if (value === "INTERRUPTING") return <RotateCcw size={15} />;
  if (value === "STOPPED") return <Circle size={15} />;
  return <CheckCircle2 size={15} />;
}

function combinedRunState(schedulerState: string, agentState: string): string {
  if (agentState === "INTERRUPTING" || schedulerState === "INTERRUPTING") {
    return "INTERRUPTING";
  }
  if (agentState === "RUNNING" || schedulerState === "RUNNING") {
    return "RUNNING";
  }
  if (agentState === "READY" || schedulerState === "READY") {
    return "READY";
  }
  if (agentState === "STOPPED" || schedulerState === "STOPPED") {
    return "STOPPED";
  }
  return "IDLE";
}

function PriorityStatusCell({
  queueLengths,
  queueMessages,
  open,
  onToggle
}: {
  queueLengths: Record<QueueKind, number>;
  queueMessages: Record<QueueKind, QueueMessageState[]>;
  open: boolean;
  onToggle: () => void;
}) {
  const highPriorityCount = hasHighPriorityWork(queueLengths, queueMessages) ? 1 : 0;
  const normalCount = normalQueueCount(queueLengths, queueMessages);

  return (
    <div className={`priority-status ${open ? "active" : ""}`}>
      <button
        type="button"
        className="priority-status-trigger"
        title="优先消息会作为 steer 处理；普通消息进入队列。"
        aria-label={renderPriorityStatusText(queueLengths, queueMessages)}
        aria-pressed={open}
        onClick={onToggle}
      >
        <span>优先 <strong>{highPriorityCount}</strong></span>
        <span>普通 <strong>{normalCount}</strong></span>
      </button>
      {open && (
        <QueuePopover
          queueLengths={queueLengths}
          queueMessages={queueMessages}
        />
      )}
    </div>
  );
}

function ContextUsageCell({ usage }: { usage?: ContextUsageState }) {
  const ratio = usage?.ratio ?? 0;
  const percent = usage?.ratio === undefined ? "n/a" : `${Math.round(ratio * 100)}%`;
  const used = usage ? compactNumber(usage.usedTokens) : "-";
  const max = usage?.maxContextTokens ? compactNumber(usage.maxContextTokens) : "-";
  const usageLabel = `${usage?.source === "estimate" ? "~" : ""}${used}/${max}`;
  return (
    <div className="context-status" title={`Context ${usageLabel}`}>
      <span>Context</span>
      <strong>{percent}</strong>
      <div className="context-meter" aria-hidden="true">
        <span style={{ width: `${Math.min(100, Math.max(0, ratio * 100))}%` }} />
      </div>
      <small>{usageLabel}</small>
    </div>
  );
}

function SessionStatusCell({
  messageCount,
  sessions,
  currentSessionId,
  open,
  busy,
  onToggle,
  onSelect,
  onDelete,
  onNew
}: {
  messageCount: number;
  sessions: SessionMetaPayload[];
  currentSessionId: string | null;
  open: boolean;
  busy: boolean;
  onToggle: () => void;
  onSelect: (sessionId: string) => void;
  onDelete: (sessionId: string) => void;
  onNew: () => void;
}) {
  const sortedSessions = [...sessions].sort((a, b) => {
    const left = Date.parse(a.updated_at || a.created_at || "");
    const right = Date.parse(b.updated_at || b.created_at || "");
    return (Number.isFinite(right) ? right : 0) - (Number.isFinite(left) ? left : 0);
  });

  return (
    <div className={`session-status ${open ? "active" : ""}`}>
      <button
        type="button"
        className="session-trigger"
        title="切换 Session"
        disabled={busy}
        onClick={onToggle}
      >
        <span>Session</span>
        <strong>{messageCount}</strong>
        <small>{currentSessionId ? shortSessionId(currentSessionId) : "-"}</small>
      </button>
      <button
        type="button"
        className="mini-button icon-only"
        title="新建 Session"
        aria-label="新建 Session"
        disabled={busy}
        onClick={(event) => {
          event.stopPropagation();
          onNew();
        }}
      >
        <Plus size={15} />
      </button>
      {open && (
        <div className="session-popover">
          <header>
            <span>Sessions</span>
            <strong>{sortedSessions.length}</strong>
          </header>
          <div className="session-list">
            {sortedSessions.length === 0 ? (
              <div className="session-empty">No sessions</div>
            ) : sortedSessions.map((session) => {
              const isCurrent = session.session_id === currentSessionId;
              return (
                <div
                  key={session.session_id}
                  className={`session-option ${isCurrent ? "current" : ""}`}
                >
                  <button
                    type="button"
                    className="session-select"
                    disabled={busy}
                    onClick={() => onSelect(session.session_id)}
                  >
                    <span>{session.name || shortSessionId(session.session_id)}</span>
                    <small>{formatSessionUpdatedAt(session.updated_at)}</small>
                  </button>
                  {!isCurrent && (
                    <button
                      type="button"
                      className="session-delete"
                      title="删除 Session"
                      aria-label={`删除 Session ${shortSessionId(session.session_id)}`}
                      disabled={busy}
                      onClick={() => onDelete(session.session_id)}
                    >
                      <Trash2 size={13} />
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

function QueuePopover({
  queueLengths,
  queueMessages
}: {
  queueLengths: Record<QueueKind, number>;
  queueMessages: Record<QueueKind, QueueMessageState[]>;
}) {
  const normalCount = normalQueueCount(queueLengths, queueMessages);
  const total = (hasHighPriorityWork(queueLengths, queueMessages) ? 1 : 0) + normalCount;
  return (
    <div className="queue-popover">
      <header>
        <span>待处理消息</span>
        <strong>{total}</strong>
      </header>
      <QueueMessageGroup
        kind="high_prio"
        length={Math.max(queueLengths.high_prio, queueMessages.high_prio.length)}
        messages={queueMessages.high_prio}
      />
      <QueueMessageGroup
        kind="normal"
        length={normalCount}
        messages={queueMessages.normal}
      />
    </div>
  );
}

function QueueMessageGroup({
  kind,
  length,
  messages
}: {
  kind: QueueKind;
  length: number;
  messages: QueueMessageState[];
}) {
  const displayCount = kind === "high_prio" ? (length > 0 ? 1 : 0) : length;
  const mergeNote = kind === "high_prio" && messages.length > 1
    ? `优先消息 ${messages.length} 条`
    : "";
  return (
    <section className="queue-group">
      <header>
        <span>{queueLabels[kind]}</span>
        <strong>{displayCount}</strong>
      </header>
      {mergeNote && <div className="queue-merge-note">{mergeNote}</div>}
      {messages.length === 0 ? (
        <div className="queue-empty">{displayCount > 0 ? "暂无消息预览" : "空"}</div>
      ) : (
        <div className="queue-message-list">
          {messages.map((message) => (
            <QueueMessageItem message={message} key={`${kind}-${message.id}`} />
          ))}
        </div>
      )}
    </section>
  );
}

function QueueMessageItem({ message }: { message: QueueMessageState }) {
  const timestamp = formatQueueTimestamp(message.createdAt);
  return (
    <article className="queue-message">
      <div className="queue-message-meta">
        <span>{message.id}</span>
        {timestamp && <time>{timestamp}</time>}
      </div>
      <p>{message.contentPreview || "空消息"}</p>
    </article>
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
  return <MessageBubble node={node} />;
});

const MessageBubble = memo(function MessageBubble({ node }: { node: ChatNode }) {
  const [collapsed, setCollapsed] = useState(false);
  const html = node.kind === "agent" ? renderMarkdown(node.content) : escapeText(node.content);
  const label = node.kind === "user" ? labelForUserMessage(node) : labelForKind(node.kind);
  const receiving = node.kind === "agent" && node.complete === false;
  const toggleCollapsed = () => setCollapsed((value) => !value);

  return (
    <article className={`bubble ${node.kind} message ${collapsed ? "message-collapsed" : ""}`}>
      <div className="bubble-head collapsible-head" onClick={toggleCollapsed}>
        <span>{label}</span>
        <span className="message-actions">
          {receiving && <LiveSpinner title="正在接收消息" />}
          <CopyButton text={node.content} title="复制消息" />
          <button
            className="thinking-toggle message-toggle"
            title={collapsed ? "展开消息" : "折叠消息"}
            aria-expanded={!collapsed}
            onClick={(event) => {
              event.stopPropagation();
              toggleCollapsed();
            }}
          >
            {collapsed ? <ChevronRight size={15} /> : <ChevronDown size={15} />}
          </button>
        </span>
      </div>
      <div className={`message-body ${collapsed ? "is-collapsed" : ""}`}>
        <div className="markdown" dangerouslySetInnerHTML={{ __html: html }} />
      </div>
    </article>
  );
});

function labelForUserMessage(node: ChatNode): string {
  if (node.displayMessageType) {
    return userMessageTypeLabels[node.displayMessageType];
  }
  if (node.queue === "urgent") return userMessageTypeLabels.urgent;
  if (node.queue === "high_prio") return userMessageTypeLabels.steer;
  return userMessageTypeLabels.normal;
}

const ThinkingBubble = memo(function ThinkingBubble({ node }: { node: ChatNode }) {
  const [collapsed, setCollapsed] = useState(() => node.complete === true);
  const autoCollapsedRef = useRef(node.complete === true);
  const html = renderMarkdown(node.content);
  const receiving = node.complete === false;
  const toggleCollapsed = () => setCollapsed((value) => !value);

  useEffect(() => {
    if (node.complete === true && !autoCollapsedRef.current) {
      setCollapsed(true);
      autoCollapsedRef.current = true;
    }
  }, [node.complete]);

  return (
    <article className={`bubble thinking ${collapsed ? "collapsed" : ""}`}>
      <div className="bubble-head collapsible-head" onClick={toggleCollapsed}>
        <span><Brain size={15} /> Thinking</span>
        <span className="message-actions">
          {receiving && <LiveSpinner title="正在接收思考内容" />}
          <CopyButton text={node.content} title="复制思考内容" />
          <button
            className="thinking-toggle"
            title={collapsed ? "展开思考内容" : "折叠思考内容"}
            aria-expanded={!collapsed}
            onClick={(event) => {
              event.stopPropagation();
              toggleCollapsed();
            }}
          >
            {collapsed ? <ChevronRight size={15} /> : <ChevronDown size={15} />}
          </button>
        </span>
      </div>
      <div className={`collapsible-body thinking-body ${collapsed ? "is-collapsed" : ""}`} aria-hidden={collapsed}>
        <div className="collapsible-body-inner">
          <div className="markdown" dangerouslySetInnerHTML={{ __html: html }} />
        </div>
      </div>
      <div className={`thinking-summary ${collapsed ? "is-visible" : ""}`} aria-hidden={!collapsed}>
        {thinkingExcerpt(node.content)}
      </div>
    </article>
  );
});

const ToolBubble = memo(function ToolBubble({ node }: { node: ChatNode }) {
  const tool = node.tool!;
  const completed = tool.status === "success" || tool.status === "fail";
  const running = tool.status === "running";
  const [collapsed, setCollapsed] = useState(() => completed);
  const autoCollapsedRef = useRef(completed);
  const hasStructuredArguments = tool.arguments !== undefined;
  const toggleCollapsed = () => setCollapsed((value) => !value);

  useEffect(() => {
    if (completed && !autoCollapsedRef.current) {
      setCollapsed(true);
      autoCollapsedRef.current = true;
    }
  }, [completed]);

  return (
    <article className={`bubble tool ${tool.status} ${collapsed ? "collapsed" : ""}`}>
      <div className="bubble-head collapsible-head" onClick={toggleCollapsed}>
        <span className="tool-title">
          <span className="tool-name"><Wrench size={15} /> {tool.name}</span>
          {tool.description && <span className="tool-description">{tool.description}</span>}
        </span>
        <span className="tool-actions">
          {running ? <LiveSpinner title="工具运行中" /> : <strong>{tool.status}</strong>}
          <CopyButton text={formatToolCopyText(tool)} title="复制工具调用" />
          <button
            className="thinking-toggle tool-toggle"
            title={collapsed ? "展开工具调用" : "折叠工具调用"}
            aria-expanded={!collapsed}
            onClick={(event) => {
              event.stopPropagation();
              toggleCollapsed();
            }}
          >
            {collapsed ? <ChevronRight size={15} /> : <ChevronDown size={15} />}
          </button>
        </span>
      </div>
      <div className={`collapsible-body tool-body ${collapsed ? "is-collapsed" : ""}`} aria-hidden={collapsed}>
        <div className="collapsible-body-inner">
          {tool.progress && <ToolProgress progress={tool.progress} compact={false} />}
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
        </div>
      </div>
    </article>
  );
});

function LiveSpinner({ title }: { title: string }) {
  return (
    <span className="live-spinner" title={title} aria-label={title} role="status">
      <LoaderCircle size={15} />
    </span>
  );
}

function CopyButton({ text, title }: { text: string; title: string }) {
  const [state, setState] = useState<"idle" | "copied" | "failed">("idle");
  const timerRef = useRef<number | null>(null);

  useEffect(() => () => {
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
    }
  }, []);

  async function handleCopy(event: ReactMouseEvent<HTMLButtonElement>) {
    event.stopPropagation();
    try {
      await copyTextToClipboard(text);
      setState("copied");
    } catch {
      setState("failed");
    }
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
    }
    timerRef.current = window.setTimeout(() => {
      setState("idle");
      timerRef.current = null;
    }, COPY_FEEDBACK_MS);
  }

  const copied = state === "copied";
  const failed = state === "failed";
  const label = copied ? "已复制" : failed ? "复制失败" : title;

  return (
    <button
      type="button"
      className={`copy-button ${copied ? "copied" : ""} ${failed ? "failed" : ""}`.trim()}
      title={label}
      aria-label={label}
      onClick={handleCopy}
    >
      {copied ? <Check size={14} /> : <Copy size={14} />}
    </button>
  );
}

function ToolProgress({ progress, compact }: { progress: ToolProgressState; compact: boolean }) {
  const percent = progress.progress == null ? null : Math.round(progress.progress * 100);
  const label = progress.label ?? progress.status ?? progress.pluginName;
  return (
    <div className={`tool-progress ${compact ? "compact" : ""}`}>
      <div className="tool-progress-head">
        <span>{label}</span>
        {progress.message && <strong>{progress.message}</strong>}
        {percent !== null && <em>{percent}%</em>}
      </div>
      {percent !== null && (
        <div className="progress-track" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent} role="progressbar">
          <span style={{ width: `${percent}%` }} />
        </div>
      )}
    </div>
  );
}

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

function PluginPreviewPanel({
  artifacts,
  artifactOrder,
  selectedArtifactId,
  messages,
  statuses,
  toolProgress,
  onSelectArtifact
}: {
  artifacts: Record<string, PluginArtifactState>;
  artifactOrder: string[];
  selectedArtifactId?: string;
  messages: PluginMessageState[];
  statuses: Record<string, PluginStatusState>;
  toolProgress: Record<string, ToolProgressState>;
  onSelectArtifact: (artifactKey: string) => void;
}) {
  const artifactList = artifactOrder.map((key) => artifacts[key]).filter(Boolean);
  const selected = selectedArtifactId && artifacts[selectedArtifactId]
    ? artifacts[selectedArtifactId]
    : artifactList[0];
  const statusList = Object.values(statuses).sort((a, b) => b.updatedAt - a.updatedAt);
  const progressList = Object.values(toolProgress).sort((a, b) => b.updatedAt - a.updatedAt).slice(0, 4);
  const messageList = messages.slice(-5).reverse();

  return (
    <aside className="plugin-preview">
      <header className="preview-head">
        <span><FileText size={15} /> Artifacts</span>
        <strong>{artifactList.length}</strong>
      </header>
      {artifactList.length > 0 ? (
        <>
          <div className="artifact-tabs">
            {artifactList.map((artifact) => (
              <button
                key={artifact.key}
                className={artifact.key === selected?.key ? "artifact-tab active" : "artifact-tab"}
                title={artifact.title}
                onClick={() => onSelectArtifact(artifact.key)}
              >
                <span>{artifact.title}</span>
                <small>{artifact.artifactType}</small>
              </button>
            ))}
          </div>
          {selected && <ArtifactPreview artifact={selected} />}
        </>
      ) : (
        <div className="preview-empty">No artifacts yet.</div>
      )}

      {(statusList.length > 0 || progressList.length > 0) && (
        <section className="plugin-side-section">
          <h3><Activity size={14} /> Progress</h3>
          {progressList.map((item, index) => (
            <ToolProgress progress={item} compact key={`${item.pluginId}-${item.updatedAt}-${index}`} />
          ))}
          {statusList.slice(0, 4).map((item) => (
            <div className="plugin-status" key={item.pluginId}>
              <div>
                <strong>{item.label ?? item.pluginName}</strong>
                <span>{item.message ?? item.status}</span>
              </div>
              {item.progress != null && <em>{Math.round(item.progress * 100)}%</em>}
            </div>
          ))}
        </section>
      )}

      {messageList.length > 0 && (
        <section className="plugin-side-section">
          <h3>Messages</h3>
          {messageList.map((item) => (
            <div className={`plugin-message ${item.level}`} key={item.id}>
              <strong>{item.title ?? item.pluginName}</strong>
              <span>{item.message}</span>
            </div>
          ))}
        </section>
      )}
    </aside>
  );
}

function ArtifactPreview({ artifact }: { artifact: PluginArtifactState }) {
  const content = artifact.content ?? formatArtifactData(artifact.data);
  const isMarkdown = artifact.language === "markdown" || artifact.mimeType === "text/markdown";
  return (
    <article className="artifact-preview">
      <header>
        <div>
          <h2>{artifact.title}</h2>
          <span>{artifact.pluginName} · {artifact.artifactType}{artifact.status ? ` · ${artifact.status}` : ""}</span>
        </div>
      </header>
      {artifact.description && <p className="artifact-description">{artifact.description}</p>}
      {artifact.path && <div className="artifact-ref">{artifact.path}</div>}
      {artifact.uri && <div className="artifact-ref">{artifact.uri}</div>}
      {content ? (
        isMarkdown
          ? <div className="markdown artifact-content" dangerouslySetInnerHTML={{ __html: renderMarkdown(content) }} />
          : <pre className="artifact-code">{content}</pre>
      ) : (
        <div className="preview-empty">Waiting for content.</div>
      )}
    </article>
  );
}

function formatArtifactData(value: unknown): string {
  if (value == null) return "";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
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
      <div className="modal-toolbar">
        <input className="search" value={filter} onChange={(event) => setFilter(event.target.value)} placeholder="搜索模型" />
      </div>
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
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});

  function apply() {
    const selectedList = catalog.filter((item) => selected.has(item.key)).map((item) => item.key);
    const nextConfigs: Record<string, Record<string, unknown>> = {};
    const errors: Record<string, string> = {};
    for (const item of catalog) {
      if (!selected.has(item.key)) continue;
      nextConfigs[item.key] = configs[item.key] ?? {};
      for (const err of validatePluginConfig(item, nextConfigs[item.key])) {
        const colonIdx = err.indexOf(": ");
        const fieldKey = colonIdx > 0 ? `${item.key}.${err.slice(0, colonIdx)}` : `${item.key}.unknown`;
        errors[fieldKey] = colonIdx > 0 ? err.slice(colonIdx + 2) : err;
      }
    }
    if (Object.keys(errors).length) {
      setFieldErrors(errors);
      return;
    }
    onApply(selectedList, nextConfigs);
  }

  function getFieldError(pluginKey: string, field: string): string | undefined {
    return fieldErrors[`${pluginKey}.${field}`];
  }

  return (
    <Modal
      title="插件配置"
      className="plugin-modal"
      onClose={onClose}
      footer={
        <>
          {Object.keys(fieldErrors).length > 0 && (
            <div className="form-errors">{Object.values(fieldErrors).join("\n")}</div>
          )}
          <div className="modal-action-row">
            <button className="tool-button" onClick={onClose}>取消</button>
            <button className="primary-button" onClick={apply}>应用</button>
          </div>
        </>
      }
    >
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
                error={getFieldError(item.key, field)}
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
    </Modal>
  );
}

function SchemaField({ field, schema, disabled, value, error, onChange }: { field: string; schema: JsonSchemaObject; disabled: boolean; value: unknown; error?: string; onChange: (value: unknown) => void }) {
  const label = schema.title ?? field;
  if (schema.type === "boolean") {
    return (
      <label className="schema-field inline">
        <span>{label}</span>
        <input type="checkbox" disabled={disabled} checked={Boolean(value)} onChange={(event) => onChange(event.target.checked)} />
      </label>
    );
  }
  if (schema.enum && Array.isArray(schema.enum) && schema.enum.length > 0) {
    return (
      <label className="schema-field">
        <span>{label}</span>
        <select
          disabled={disabled}
          value={String(value ?? "")}
          onChange={(event) => onChange(coerceSchemaValue(schema, event.target.value))}
        >
          <option value="">--</option>
          {schema.enum.map((opt) => (
            <option key={String(opt)} value={String(opt)}>{String(opt)}</option>
          ))}
        </select>
        {schema.description && <small>{schema.description}</small>}
        {error && <small className="schema-error">{error}</small>}
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
      {error && <small className="schema-error">{error}</small>}
    </label>
  );
}

function Modal({ title, children, className = "", footer, onClose }: { title: string; children: ReactNode; className?: string; footer?: ReactNode; onClose: () => void }) {
  return (
    <div className="modal-backdrop">
      <section className={`modal ${className}`.trim()}>
        <header>
          <h2>{title}</h2>
          <button className="icon-button" onClick={onClose} title="关闭"><X size={18} /></button>
        </header>
        <div className="modal-body">{children}</div>
        {footer && <footer className="modal-actions">{footer}</footer>}
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

function framePayload(frame: CoreFrame | null): Record<string, unknown> | null {
  if (!frame || !frame.payload || typeof frame.payload !== "object" || Array.isArray(frame.payload)) {
    return null;
  }
  return frame.payload as Record<string, unknown>;
}

function normalizeSessionList(value: unknown): SessionMetaPayload[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter((item): item is Record<string, unknown> => (
      item !== null && typeof item === "object" && !Array.isArray(item)
    ))
    .map((item) => ({
      session_id: String(item.session_id ?? ""),
      name: String(item.name ?? item.session_id ?? ""),
      created_at: String(item.created_at ?? ""),
      updated_at: String(item.updated_at ?? ""),
      last_checkpoint_event: typeof item.last_checkpoint_event === "string"
        ? item.last_checkpoint_event
        : null,
      components_present: Array.isArray(item.components_present)
        ? item.components_present.map((entry) => String(entry))
        : []
    }))
    .filter((item) => item.session_id);
}

function optionalPayloadString(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed || null;
}

function shortSessionId(value: string): string {
  if (value.length <= 8) return value;
  return value.slice(0, 8);
}

function formatSessionUpdatedAt(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "unknown";
  return date.toLocaleString([], {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  });
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

function compactNumber(value: number): string {
  if (!Number.isFinite(value)) return "-";
  if (Math.abs(value) >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (Math.abs(value) >= 1000) return `${Math.round(value / 1000)}K`;
  return `${(value / 1000).toFixed(1)}K`;
}

function formatQueueTimestamp(value?: number): string | null {
  if (value === undefined || !Number.isFinite(value)) return null;
  const millis = value < 10_000_000_000 ? value * 1000 : value;
  const date = new Date(millis);
  if (Number.isNaN(date.getTime())) return null;
  return date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  });
}

export function isNearChatBottom(element: Pick<HTMLElement, "scrollHeight" | "scrollTop" | "clientHeight">): boolean {
  return element.scrollHeight - element.scrollTop - element.clientHeight < AUTO_SCROLL_BOTTOM_THRESHOLD_PX;
}

export function resolveFollowTailOnScroll(
  currentFollowTail: boolean,
  nearBottom: boolean,
  userScrollIntent: boolean,
  selectingChat: boolean,
  isAutoScrolling: boolean,
): boolean {
  if (isAutoScrolling || nearBottom) return true;
  if (userScrollIntent || selectingChat) return false;
  return currentFollowTail;
}

function resizeTextareaToRows(textarea: HTMLTextAreaElement | null, maxRows: number, minRows = 1) {
  if (!textarea) return;

  textarea.style.height = "auto";

  const style = window.getComputedStyle(textarea);
  const lineHeight = Number.parseFloat(style.lineHeight) || 20;
  const padding = Number.parseFloat(style.paddingTop) + Number.parseFloat(style.paddingBottom);
  const border = Number.parseFloat(style.borderTopWidth) + Number.parseFloat(style.borderBottomWidth);
  const minHeight = lineHeight * minRows + padding + border;
  const maxHeight = lineHeight * maxRows + padding + border;
  const nextHeight = Math.min(Math.max(textarea.scrollHeight, minHeight), maxHeight);

  textarea.style.height = `${nextHeight}px`;
  textarea.style.overflowY = textarea.scrollHeight > maxHeight ? "auto" : "hidden";
}

type InputKeyEvent = {
  key: string;
  shiftKey: boolean;
  nativeEvent: {
    isComposing?: boolean;
    keyCode?: number;
    which?: number;
  };
};

export function shouldSubmitInputFromKeyEvent(event: InputKeyEvent, inputComposing: boolean): boolean {
  if (event.key !== "Enter" || event.shiftKey) return false;
  return !isInputComposing(event, inputComposing);
}

function isInputComposing(event: InputKeyEvent, inputComposing: boolean): boolean {
  return inputComposing
    || event.nativeEvent.isComposing === true
    || event.nativeEvent.keyCode === 229
    || event.nativeEvent.which === 229;
}

export function renderMarkdown(value: string): string {
  return markdown.render(value);
}

export function formatToolCopyText(tool: ToolState): string {
  const sections = [
    `Tool: ${tool.name}`,
    `Status: ${tool.status}`
  ];
  if (tool.description) {
    sections.push(`Description: ${tool.description}`);
  }
  if (tool.toolCallId) {
    sections.push(`Tool call id: ${tool.toolCallId}`);
  }
  if (tool.arguments !== undefined) {
    sections.push(`Arguments:\n${formatCopyValue(tool.arguments)}`);
  } else if (tool.argsRaw) {
    sections.push(`Arguments:\n${tool.argsRaw}`);
  }
  if (tool.resultPreview) {
    sections.push(`Result:\n${tool.resultPreview}`);
  }
  return sections.join("\n\n");
}

export async function copyTextToClipboard(value: string): Promise<void> {
  const clipboard = globalThis.navigator?.clipboard;
  if (clipboard?.writeText) {
    await clipboard.writeText(value);
    return;
  }
  if (typeof document === "undefined" || !document.body) {
    throw new Error("Clipboard is unavailable");
  }

  const textarea = document.createElement("textarea");
  textarea.value = value;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.top = "-1000px";
  textarea.style.left = "-1000px";
  document.body.appendChild(textarea);
  textarea.focus();
  textarea.select();
  const copied = document.execCommand("copy");
  textarea.remove();
  if (!copied) {
    throw new Error("Clipboard copy failed");
  }
}

function formatCopyValue(value: unknown): string {
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function highlightCode(value: string, language: string): string {
  const normalizedLanguage = normalizeHighlightLanguage(language);
  if (normalizedLanguage) {
    const result = hljs.highlight(value, { language: normalizedLanguage, ignoreIllegals: true });
    return codeBlock(result.value, normalizedLanguage);
  }
  return codeBlock(escapeHtml(value));
}

function normalizeHighlightLanguage(language: string): string | null {
  const candidate = language.trim().toLowerCase();
  if (!candidate) return null;
  const normalized = LANGUAGE_ALIASES[candidate] ?? candidate;
  return hljs.getLanguage(normalized) ? normalized : null;
}

function codeBlock(value: string, language?: string): string {
  const languageClass = language ? ` language-${escapeHtmlAttribute(language)}` : "";
  return `<pre class="code-block"><code class="hljs${languageClass}">${value}</code></pre>`;
}

function escapeHtmlAttribute(value: string): string {
  return value.replace(/[^A-Za-z0-9_-]/g, "");
}

function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, (char) => {
    switch (char) {
      case "&": return "&amp;";
      case "<": return "&lt;";
      case ">": return "&gt;";
      case "\"": return "&quot;";
      case "'": return "&#039;";
      default: return char;
    }
  });
}

function escapeText(value: string): string {
  return escapeHtml(value).replace(/\n/g, "<br />");
}
