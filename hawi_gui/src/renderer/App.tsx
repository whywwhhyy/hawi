import { forwardRef, memo, useEffect, useLayoutEffect, useMemo, useReducer, useRef, useState, type MouseEvent as ReactMouseEvent, type ReactNode, type UIEvent as ReactUIEvent, type WheelEvent } from "react";
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
import { Activity, ArrowDown, ArrowUp, Bot, Brain, Check, ChevronDown, ChevronRight, Copy, FileText, GitFork, LoaderCircle, Lock, Pencil, Play, Plug, Plus, RotateCcw, Send, Square, Trash2, Wrench, X } from "lucide-react";
import type { CoreCommandType, CoreFrame, GuiMetadata, JsonSchemaObject, MarkdownExportPayload, PersistedConfig, PluginCatalogItem, QueueKind, RuntimeControlState, SessionLaunchProfile, SessionLoadState, SessionMetaPayload } from "../shared/protocol";
import { VERSION } from "../shared/protocol";
import { coerceSchemaValue, invertPluginSelection, mergePluginDefaults, selectAllPluginKeys, validatePluginConfig } from "./pluginConfig";
import { createInitialState, reduceCoreEvent, type ChatNode, type ContextCompressionState, type ContextUsageState, type FrameworkInjectionState, type PluginArtifactState, type PluginMessageState, type PluginStatusState, type ProcessingState, type QueueMessageState, type ToolProgressState, type ToolState } from "./state";

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
  normal: "稍后任务",
  high_prio: "待送达插话",
  urgent: "紧急"
};

const userMessageTypeLabels = {
  normal: "普通消息",
  steer: "Steer",
  urgent: "紧急消息"
} as const;

export function renderQueueStatusText(
  queueLengths: Record<QueueKind, number>,
  queueMessages?: Record<QueueKind, QueueMessageState[]>
): string {
  const highPriorityCount = hasHighPriorityWork(queueLengths, queueMessages) ? 1 : 0;
  return `Message Insert ${highPriorityCount} · Queue ${normalQueueCount(queueLengths, queueMessages)}`;
}

export const renderPriorityStatusText = renderQueueStatusText;

export function shouldInitializeSessionState(metadata: GuiMetadata | null): boolean {
  return Boolean(metadata?.coreRunning);
}

interface SessionRuntimeStats {
  running: number;
  loaded: number;
  maxLoaded: number;
}

type EscapeDismissTarget =
  | "contextCompactDialog"
  | "pluginDialog"
  | "modelDialog"
  | "debugMenu"
  | "queueTaskEdit"
  | "queuePopover"
  | "sessionDialog";

interface EscapeDismissState {
  contextCompactDialogOpen: boolean;
  contextCompactBusy: boolean;
  pluginDialogOpen: boolean;
  modelDialogOpen: boolean;
  debugMenuOpen: boolean;
  queuePopoverOpen: boolean;
  editingQueueTaskId: string | null;
  sessionDialogOpen: boolean;
}

export function resolveEscapeDismissTarget(state: EscapeDismissState): EscapeDismissTarget | null {
  if (state.contextCompactDialogOpen) {
    return state.contextCompactBusy ? null : "contextCompactDialog";
  }
  if (state.pluginDialogOpen) return "pluginDialog";
  if (state.modelDialogOpen) return "modelDialog";
  if (state.debugMenuOpen) return "debugMenu";
  if (state.queuePopoverOpen) {
    return state.editingQueueTaskId ? "queueTaskEdit" : "queuePopover";
  }
  if (state.sessionDialogOpen) return "sessionDialog";
  return null;
}

function resolveKeyboardScopeTarget(state: EscapeDismissState): EscapeDismissTarget | null {
  if (state.contextCompactDialogOpen) return "contextCompactDialog";
  if (state.pluginDialogOpen) return "pluginDialog";
  if (state.modelDialogOpen) return "modelDialog";
  if (state.debugMenuOpen) return "debugMenu";
  if (state.queuePopoverOpen) return "queuePopover";
  if (state.sessionDialogOpen) return "sessionDialog";
  return null;
}

function dialogScopeSelector(target: EscapeDismissTarget): string | null {
  switch (target) {
    case "contextCompactDialog":
      return ".context-compact-modal";
    case "pluginDialog":
      return ".plugin-modal";
    case "modelDialog":
      return ".model-modal";
    case "debugMenu":
      return ".menu-popover";
    case "queueTaskEdit":
    case "queuePopover":
      return ".queue-popover";
    case "sessionDialog":
      return ".session-popover";
  }
}

function dialogFocusableElements(scope: HTMLElement): HTMLElement[] {
  const elements = Array.from(scope.querySelectorAll<HTMLElement>(
    [
      "button:not(:disabled)",
      "input:not(:disabled)",
      "select:not(:disabled)",
      "textarea:not(:disabled)",
      "a[href]",
      "[tabindex]:not([tabindex='-1'])"
    ].join(",")
  )).filter((element) => element.getClientRects().length > 0 || element.dataset.dialogClose === "true");
  const closeButtons = elements.filter((element) => element.dataset.dialogClose === "true");
  return elements.filter((element) => element.dataset.dialogClose !== "true").concat(closeButtons);
}

function moveDialogFocus(scope: HTMLElement, direction: 1 | -1): boolean {
  const elements = dialogFocusableElements(scope);
  if (elements.length === 0) return false;
  const active = document.activeElement;
  const currentIndex = active instanceof HTMLElement ? elements.indexOf(active) : -1;
  const nextIndex = currentIndex === -1
    ? direction > 0 ? 0 : elements.length - 1
    : (currentIndex + direction + elements.length) % elements.length;
  elements[nextIndex].focus();
  return true;
}

function shouldPreserveArrowKey(event: KeyboardEvent): boolean {
  const target = event.target;
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  if (target instanceof HTMLTextAreaElement) return true;
  if (target instanceof HTMLInputElement) {
    return !["button", "checkbox", "radio", "submit", "reset"].includes(target.type);
  }
  return target instanceof HTMLSelectElement;
}

function shouldPreserveEnterKey(event: KeyboardEvent): boolean {
  const target = event.target;
  return target instanceof HTMLTextAreaElement || (target instanceof HTMLElement && target.isContentEditable);
}

function clickActiveButton(scope: HTMLElement): boolean {
  const active = document.activeElement;
  if (!(active instanceof HTMLElement) || !scope.contains(active)) return false;
  if (!active.matches("button:not(:disabled), [role='button']:not([aria-disabled='true'])")) return false;
  active.click();
  return true;
}

function clickDialogConfirmation(target: EscapeDismissTarget, scope: HTMLElement): boolean {
  if (clickActiveButton(scope)) return true;
  const confirm = (() => {
    switch (target) {
      case "contextCompactDialog":
      case "pluginDialog":
        return scope.querySelector<HTMLElement>(".modal-actions .primary-button:not(:disabled)");
      case "modelDialog":
        return scope.querySelector<HTMLElement>(".model.active:not(:disabled)")
          ?? scope.querySelector<HTMLElement>(".model:not(:disabled)");
      case "sessionDialog":
        return scope.querySelector<HTMLElement>(".session-option.current .session-select:not(:disabled)")
          ?? scope.querySelector<HTMLElement>(".session-select:not(:disabled)");
      case "debugMenu":
      case "queueTaskEdit":
      case "queuePopover":
        return null;
    }
  })();
  if (!confirm) return false;
  confirm.click();
  return true;
}

type SessionStates = Record<string, AppState>;

interface SessionStateAction {
  sessionId: string | null;
  frame: CoreFrame;
}

export function reduceSessionStates(states: SessionStates, action: SessionStateAction): SessionStates {
  const sessionId = action.sessionId;
  if (!sessionId) {
    return states;
  }
  const previous = states[sessionId] ?? createInitialState();
  const next = reduceCoreEvent(previous, action.frame);
  if (next === previous) {
    return states;
  }
  return { ...states, [sessionId]: next };
}

export function renderSessionCounterText(runningCount: number, loadedCount: number): string {
  return `${runningCount}/${loadedCount}`;
}

export function sessionLoadStateLabel(state: SessionLoadState): string {
  if (state === "running") return "运行中";
  if (state === "loaded") return "已加载待命";
  return "未加载";
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
  const [statesBySession, dispatchSessionState] = useReducer(reduceSessionStates, {});
  const [input, setInput] = useState("");
  const [modelDialogOpen, setModelDialogOpen] = useState(false);
  const [pluginDialogOpen, setPluginDialogOpen] = useState(false);
  const [contextCompactDialogOpen, setContextCompactDialogOpen] = useState(false);
  const [queuePopoverOpen, setQueuePopoverOpen] = useState(false);
  const [sessionDialogOpen, setSessionDialogOpen] = useState(false);
  const [sessions, setSessions] = useState<SessionMetaPayload[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [queueTaskDraft, setQueueTaskDraft] = useState("");
  const [queueTaskBusy, setQueueTaskBusy] = useState(false);
  const [editingQueueTaskId, setEditingQueueTaskId] = useState<string | null>(null);
  const [queueTaskEditDraft, setQueueTaskEditDraft] = useState("");
  const [sessionStats, setSessionStats] = useState<SessionRuntimeStats>({
    running: 0,
    loaded: 0,
    maxLoaded: 5
  });
  const [sessionBusy, setSessionBusy] = useState(false);
  const [contextCompactBusy, setContextCompactBusy] = useState(false);
  const [exportBusy, setExportBusy] = useState(false);
  const [debugMenuOpen, setDebugMenuOpen] = useState(false);
  const chatRef = useRef<HTMLDivElement | null>(null);
  const systemPromptRef = useRef<HTMLTextAreaElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const queueTaskDraftRef = useRef<HTMLTextAreaElement | null>(null);
  const configRef = useRef<PersistedConfig | null>(null);
  const currentSessionIdRef = useRef<string | null>(null);
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
  const queueTaskComposingRef = useRef(false);
  const queueTaskCompositionEndTimerRef = useRef<number | null>(null);
  const coreRunning = shouldInitializeSessionState(metadata) || sessionStats.loaded > 0;
  const fallbackState = useMemo(createInitialState, []);
  const state = currentSessionId ? statesBySession[currentSessionId] ?? fallbackState : fallbackState;

  useEffect(() => {
    window.hawi.getMetadata().then((meta) => {
      setMetadata(meta);
      setConfig(meta.config);
      setCurrentSessionId(meta.currentSessionId ?? null);
      setSessionStats({
        running: meta.runningSessionCount ?? 0,
        loaded: meta.loadedSessionCount ?? 0,
        maxLoaded: meta.maxLoadedSessions ?? 5
      });
      if (!meta.config.modelName) {
        setModelDialogOpen(true);
      }
    }).catch((error) => {
      dispatch(errorFrame(error));
    });
  }, []);

  useEffect(() => {
    const offEvent = window.hawi.onCoreEvent((frame) => handleCoreEvent(frame));
    const offLog = window.hawi.onCoreLog((message) => {
      dispatch({ version: VERSION, type: "debug.info", payload: { message } });
    });
    return () => {
      offEvent();
      offLog();
    };
    // Core event routing reads mutable session state through refs; resubscribing
    // on every render would duplicate IPC churn without changing behavior.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    currentSessionIdRef.current = currentSessionId;
  }, [currentSessionId]);

  useEffect(() => {
    if (!coreRunning) return;
    void initializeSessionStateRef.current();
  }, [coreRunning]);

  useEffect(() => {
    configRef.current = config;
  }, [config]);

  useEffect(() => {
    function handleDialogKeyboard(event: KeyboardEvent) {
      if (event.isComposing || event.metaKey || event.ctrlKey || event.altKey) return;
      const keyboardState = {
        contextCompactDialogOpen,
        contextCompactBusy,
        pluginDialogOpen,
        modelDialogOpen,
        debugMenuOpen,
        queuePopoverOpen,
        editingQueueTaskId,
        sessionDialogOpen
      };

      if (event.key === "Escape") {
        const target = resolveEscapeDismissTarget(keyboardState);
        if (!target) return;
        event.preventDefault();
        event.stopPropagation();
        switch (target) {
          case "contextCompactDialog":
            setContextCompactDialogOpen(false);
            break;
          case "pluginDialog":
            setPluginDialogOpen(false);
            break;
          case "modelDialog":
            setModelDialogOpen(false);
            break;
          case "debugMenu":
            setDebugMenuOpen(false);
            break;
          case "queueTaskEdit":
            setEditingQueueTaskId(null);
            setQueueTaskEditDraft("");
            break;
          case "queuePopover":
            setQueuePopoverOpen(false);
            break;
          case "sessionDialog":
            setSessionDialogOpen(false);
            break;
        }
        return;
      }

      if (event.key !== "Enter" && event.key !== "ArrowDown" && event.key !== "ArrowRight" && event.key !== "ArrowUp" && event.key !== "ArrowLeft") return;
      const target = resolveKeyboardScopeTarget(keyboardState);
      if (!target) return;
      const selector = dialogScopeSelector(target);
      const scope = selector ? document.querySelector<HTMLElement>(selector) : null;
      if (!scope) return;

      if (event.key === "Enter") {
        if (shouldPreserveEnterKey(event)) return;
        if (!clickDialogConfirmation(target, scope)) return;
        event.preventDefault();
        event.stopPropagation();
        return;
      }

      if (shouldPreserveArrowKey(event)) return;
      const direction = event.key === "ArrowUp" || event.key === "ArrowLeft" ? -1 : 1;
      if (!moveDialogFocus(scope, direction)) return;
      event.preventDefault();
      event.stopPropagation();
    }

    document.addEventListener("keydown", handleDialogKeyboard);
    return () => {
      document.removeEventListener("keydown", handleDialogKeyboard);
    };
  }, [
    contextCompactDialogOpen,
    contextCompactBusy,
    pluginDialogOpen,
    modelDialogOpen,
    debugMenuOpen,
    queuePopoverOpen,
    editingQueueTaskId,
    sessionDialogOpen
  ]);

  useBrowserLayoutEffect(() => {
    keepChatTailVisible();
  }, [state.nodes, state.processing?.id]);

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
    if (queueTaskCompositionEndTimerRef.current !== null) {
      window.clearTimeout(queueTaskCompositionEndTimerRef.current);
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

  function dispatch(frame: CoreFrame, sessionId = currentSessionIdRef.current) {
    dispatchSessionState({ sessionId, frame });
  }

  function handleCoreEvent(frame: CoreFrame) {
    if (frame.type === "gui.session_status") {
      applySessionRuntimeStatus(frame);
      return;
    }
    const sessionId = frameSessionId(frame) ?? currentSessionIdRef.current;
    dispatchSessionState({ sessionId, frame });
  }

  function applySessionRuntimeStatus(frame: CoreFrame) {
    const payload = framePayload(frame);
    if (!payload) return;
    const sessionId = optionalPayloadString(payload.session_id);
    const loadState = normalizeSessionLoadState(payload.load_state);
    const currentId = optionalPayloadString(payload.current_session_id);
    if (currentId) {
      setCurrentSessionId(currentId);
    }
    setSessionStats((current) => ({
      running: optionalPayloadNumber(payload.running_session_count) ?? current.running,
      loaded: optionalPayloadNumber(payload.loaded_session_count) ?? current.loaded,
      maxLoaded: optionalPayloadNumber(payload.max_loaded_sessions) ?? current.maxLoaded
    }));
    if (!sessionId) return;
    setSessions((items) => upsertSessionRuntime(items, sessionId, {
      load_state: loadState,
      loaded_at: optionalPayloadNumber(payload.loaded_at),
      last_finished_at: optionalPayloadNumber(payload.last_finished_at)
    }, {
      createIfMissing: payload.has_visible_messages === true
    }));
  }

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

  async function sendCommand(
    type: CoreCommandType,
    payload: Record<string, unknown>,
    targetSessionId: string | null = currentSessionIdRef.current
  ): Promise<CoreFrame | null> {
    if (!coreRunning) {
      setModelDialogOpen(true);
      return null;
    }
    try {
      return await window.hawi.sendCommand(type, payload, targetSessionId);
    } catch (error) {
      dispatch(errorFrame(error));
      return null;
    }
  }

  async function initializeSessionState() {
    const listFrame = await sendCommand("session_list", {}, null);
    updateSessionsFromFrame(listFrame);
    const historyFrame = await sendCommand("session_history", {}, currentSessionIdRef.current);
    applySessionHistoryFromFrame(historyFrame);
  }

  initializeSessionStateRef.current = initializeSessionState;

  async function refreshSessions(): Promise<SessionMetaPayload[]> {
    const frame = await sendCommand("session_list", {}, null);
    return updateSessionsFromFrame(frame);
  }

  function updateSessionsFromFrame(frame: CoreFrame | null): SessionMetaPayload[] {
    const payload = framePayload(frame);
    if (!payload) return sessions;
    const nextSessions = normalizeSessionList(payload.sessions);
    setSessions(nextSessions);
    setSessionStats((current) => ({
      running: optionalPayloadNumber(payload.running_session_count) ?? countSessionsByState(nextSessions, "running"),
      loaded: optionalPayloadNumber(payload.loaded_session_count) ?? nextSessions.filter((item) => item.load_state === "loaded" || item.load_state === "running").length,
      maxLoaded: optionalPayloadNumber(payload.max_loaded_sessions) ?? current.maxLoaded
    }));
    const nextCurrent = optionalPayloadString(payload.current_session_id);
    if (nextCurrent) {
      setCurrentSessionId(nextCurrent);
      syncConfigFromSession(nextCurrent, nextSessions);
    }
    return nextSessions;
  }

  function applySessionHistoryFromFrame(frame: CoreFrame | null) {
    const payload = framePayload(frame);
    if (!payload) return;
    const nextCurrent = optionalPayloadString(payload.session_id);
    if (nextCurrent) {
      setCurrentSessionId(nextCurrent);
      syncConfigFromSession(nextCurrent);
    }
    if (!Array.isArray(payload.message_history)) {
      return;
    }
    dispatch({
      version: VERSION,
      type: "gui.load_session_history",
      payload: {
        message_history: payload.message_history,
        context_usage: payload.context_usage
      }
    }, nextCurrent ?? currentSessionIdRef.current);
    followTailRef.current = true;
  }

  function syncConfigFromSession(sessionId: string, sessionList = sessions) {
    const profile = sessionList.find((session) => session.session_id === sessionId)?.gui_launch_profile;
    if (!profile || !configRef.current) {
      return;
    }
    const nextConfig = configFromLaunchProfile(profile, configRef.current);
    setConfig(nextConfig);
    setMetadata((current) => current ? { ...current, config: nextConfig } : current);
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
      const refreshed = await refreshSessions();
      syncConfigFromSession(sessionId, refreshed);
      setSessionDialogOpen(false);
    } finally {
      setSessionBusy(false);
    }
  }

  async function deleteSession(sessionId: string) {
    if (!sessionId) {
      return;
    }
    const target = sessions.find((session) => session.session_id === sessionId);
    if (target?.load_state === "running") {
      return;
    }
    if (!window.confirm(`删除 Session ${shortSessionId(sessionId)}？`)) {
      return;
    }
    setSessionBusy(true);
    try {
      const frame = await sendCommand("session_delete", { session_id: sessionId }, null);
      if (!frame) return;
      setSessions((items) => items.filter((item) => item.session_id !== sessionId));
      const nextCurrent = optionalPayloadString(framePayload(frame)?.current_session_id);
      if (nextCurrent) {
        setCurrentSessionId(nextCurrent);
      }
      const refreshed = await refreshSessions();
      if (nextCurrent) {
        syncConfigFromSession(nextCurrent, refreshed);
      }
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
      }, sessionId ?? currentSessionIdRef.current);
      followTailRef.current = true;
      const refreshed = await refreshSessions();
      if (sessionId) {
        syncConfigFromSession(sessionId, refreshed);
      }
      setSessionDialogOpen(false);
    } finally {
      setSessionBusy(false);
    }
  }

  async function forkSession(sessionId?: string, messageIndex?: number) {
    const sourceSessionId = sessionId || currentSessionId;
    if (!sourceSessionId) return;
    setSessionBusy(true);
    try {
      const payload: Record<string, unknown> = { session_id: sourceSessionId };
      if (typeof messageIndex === "number") {
        payload.message_index = messageIndex;
      }
      const frame = await sendCommand("session_fork", payload);
      applySessionHistoryFromFrame(frame);
      followTailRef.current = true;
      const refreshed = await refreshSessions();
      const forkedSessionId = optionalPayloadString(framePayload(frame)?.session_id);
      if (forkedSessionId) {
        syncConfigFromSession(forkedSessionId, refreshed);
      }
      const poppedUserText = optionalPayloadString(framePayload(frame)?.popped_user_text);
      if (poppedUserText) {
        setInput(poppedUserText);
        requestAnimationFrame(() => {
          const inputElement = inputRef.current;
          if (!inputElement) return;
          inputElement.focus();
          inputElement.setSelectionRange(inputElement.value.length, inputElement.value.length);
        });
      }
      setSessionDialogOpen(false);
    } finally {
      setSessionBusy(false);
    }
  }

  function forkMessage(node: ChatNode) {
    if (sessionBusy || !currentSessionId) return;
    if (typeof node.contextMessageIndex !== "number") return;
    void forkSession(currentSessionId, node.contextMessageIndex);
  }

  async function saveGlobalAndSet(nextConfig: PersistedConfig) {
    const saved = await window.hawi.saveConfig(nextConfig);
    setConfig(saved);
    setMetadata((current) => current ? { ...current, config: saved } : current);
    return saved;
  }

  async function restartWith(nextConfig: PersistedConfig) {
    setConfig(nextConfig);
    try {
      await window.hawi.restartCore(nextConfig);
      setMetadata((current) => current ? { ...current, config: nextConfig, coreRunning: true } : current);
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
    await sendCommand("enqueue", { content: text, queue: "high_prio", metadata: { intent: "user_send", source: "gui_main_input" } });
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

  function startQueueTaskComposition() {
    if (queueTaskCompositionEndTimerRef.current !== null) {
      window.clearTimeout(queueTaskCompositionEndTimerRef.current);
      queueTaskCompositionEndTimerRef.current = null;
    }
    queueTaskComposingRef.current = true;
  }

  function endQueueTaskComposition() {
    if (queueTaskCompositionEndTimerRef.current !== null) {
      window.clearTimeout(queueTaskCompositionEndTimerRef.current);
    }
    queueTaskCompositionEndTimerRef.current = window.setTimeout(() => {
      queueTaskComposingRef.current = false;
      queueTaskCompositionEndTimerRef.current = null;
    }, 0);
  }

  async function refreshRuntimeStatus() {
    const frame = await sendCommand("get_status", {});
    if (frame?.type === "core.status") {
      dispatch(frame);
    }
  }

  async function addQueueTask() {
    const content = queueTaskDraft.trim();
    if (!content || queueTaskBusy) return;
    setQueueTaskBusy(true);
    try {
      const frame = await sendCommand("queue_task_add", { content });
      if (!frame) return;
      setQueueTaskDraft("");
      await refreshRuntimeStatus();
    } finally {
      setQueueTaskBusy(false);
    }
  }

  async function updateQueueTask(messageId: string, content: string) {
    const nextContent = content.trim();
    if (!messageId || !nextContent || queueTaskBusy) return;
    setQueueTaskBusy(true);
    try {
      const frame = await sendCommand("queue_task_update", { message_id: messageId, content: nextContent });
      if (!frame) return;
      setEditingQueueTaskId(null);
      setQueueTaskEditDraft("");
      await refreshRuntimeStatus();
    } finally {
      setQueueTaskBusy(false);
    }
  }

  async function removeQueueTask(messageId: string) {
    if (!messageId || queueTaskBusy) return;
    setQueueTaskBusy(true);
    try {
      const frame = await sendCommand("queue_task_remove", { message_id: messageId });
      if (!frame) return;
      if (editingQueueTaskId === messageId) {
        setEditingQueueTaskId(null);
        setQueueTaskEditDraft("");
      }
      await refreshRuntimeStatus();
    } finally {
      setQueueTaskBusy(false);
    }
  }

  async function pullBackQueueTask(message: QueueMessageState) {
    const content = (message.content ?? message.contentPreview).trim();
    if (!message.id || !content || queueTaskBusy) return;
    setQueueTaskBusy(true);
    try {
      const frame = await sendCommand("queue_task_remove", { message_id: message.id });
      if (!frame) return;
      if (editingQueueTaskId === message.id) {
        setEditingQueueTaskId(null);
        setQueueTaskEditDraft("");
      }
      setQueueTaskDraft(content);
      await refreshRuntimeStatus();
      window.requestAnimationFrame(() => {
        const draft = queueTaskDraftRef.current;
        if (!draft) return;
        draft.focus();
        draft.setSelectionRange(draft.value.length, draft.value.length);
      });
    } finally {
      setQueueTaskBusy(false);
    }
  }

  async function moveQueueTask(messageId: string, direction: -1 | 1) {
    if (!messageId || queueTaskBusy) return;
    const ids = state.queueMessages.normal.map((message) => message.id);
    const index = ids.indexOf(messageId);
    const nextIndex = index + direction;
    if (index < 0 || nextIndex < 0 || nextIndex >= ids.length) return;
    const nextIds = [...ids];
    [nextIds[index], nextIds[nextIndex]] = [nextIds[nextIndex], nextIds[index]];
    setQueueTaskBusy(true);
    try {
      const frame = await sendCommand("queue_task_reorder", { message_ids: nextIds });
      if (!frame) return;
      await refreshRuntimeStatus();
    } finally {
      setQueueTaskBusy(false);
    }
  }

  async function clearNormalQueue() {
    if (queueTaskBusy) return;
    setQueueTaskBusy(true);
    try {
      const frame = await sendCommand("clear_queue", { queue: "normal" });
      if (!frame) return;
      setEditingQueueTaskId(null);
      setQueueTaskEditDraft("");
      await refreshRuntimeStatus();
    } finally {
      setQueueTaskBusy(false);
    }
  }

  function startEditingQueueTask(message: QueueMessageState) {
    setEditingQueueTaskId(message.id);
    setQueueTaskEditDraft(message.content ?? message.contentPreview);
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

  async function compactContextManually() {
    if (contextCompactBusy) return;
    setContextCompactBusy(true);
    try {
      const frame = await sendCommand("compact_context", {});
      if (!frame) return;
      setContextCompactDialogOpen(false);
      applySessionHistoryFromFrame(frame);
      const status = optionalPayloadString(framePayload(frame)?.status);
      if (status === "skipped") {
        dispatch(metaFrame("没有可压缩的旧上下文"));
      }
      await refreshRuntimeStatus();
    } finally {
      setContextCompactBusy(false);
    }
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
          if (configRef.current?.systemPrompt === pending.systemPrompt) {
            setConfig(pending);
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
      if (coreRunning) {
        await sendCommand("switch_model", { model_name: modelName });
      } else {
        await saveGlobalAndSet(nextConfig);
        await restartWith(nextConfig);
      }
    } catch (error) {
      dispatch(errorFrame(error));
    }
  }

  async function refreshProviderModels(provider: string) {
    const nextMetadata = await window.hawi.refreshProviderModels(provider);
    setMetadata(nextMetadata);
    setConfig((current) => current
      ? {
        ...current,
        modelName: nextMetadata.inspect.models.includes(current.modelName)
          ? current.modelName
          : nextMetadata.config.modelName
      }
      : nextMetadata.config);
    dispatch(metaFrame(`已刷新 ${provider} 的模型列表`));
  }

  async function applyPlugins(selectedPlugins: string[], pluginConfigs: Record<string, Record<string, unknown>>) {
    if (!config) return;
    const nextConfig = { ...config, selectedPlugins, pluginConfigs };
    setConfig(nextConfig);
    setPluginDialogOpen(false);
    await sendCommand("apply_plugins", { selected_plugins: selectedPlugins, plugin_configs: pluginConfigs });
  }

  async function exportCurrentSession() {
    if (!currentSessionId || exportBusy) return;
    setExportBusy(true);
    try {
      const frame = await sendCommand("session_export_markdown", { session_id: currentSessionId });
      const exportPayload = normalizeMarkdownExportPayload(frame?.payload?.export);
      if (!exportPayload) {
        throw new Error("导出结果为空");
      }
      const saved = await window.hawi.saveMarkdownExport(exportPayload);
      if (!saved.canceled && saved.markdownPath) {
        dispatch(metaFrame(`Markdown 已导出：${saved.markdownPath}`));
      }
    } catch (error) {
      dispatch(errorFrame(error));
    } finally {
      setExportBusy(false);
    }
  }

  if (!metadata || !config) {
    return <div className="boot">Loading Hawi metadata...</div>;
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="status-strip">
          <SessionStatusCell
            messageCount={state.sessionMessageCount}
            runningCount={sessionStats.running}
            loadedCount={sessionStats.loaded}
            sessions={sessions}
            currentSessionId={currentSessionId}
            open={sessionDialogOpen}
            busy={sessionBusy}
            onToggle={openSessionDialog}
            onSelect={loadSession}
            onDelete={deleteSession}
            onNew={newSession}
            onFork={forkSession}
          />
          <ContextUsageCell
            usage={state.contextUsage}
            compression={state.contextCompression}
            busy={contextCompactBusy}
            disabled={!coreRunning || state.runnerState === "RUNNING" || state.runnerState === "INTERRUPTING"}
            onRequestCompact={() => setContextCompactDialogOpen(true)}
          />
          <QueueStatusCell
            queueLengths={state.queueLengths}
            queueMessages={state.queueMessages}
            control={state.control}
            open={queuePopoverOpen}
            onToggle={() => setQueuePopoverOpen((value) => !value)}
            taskDraft={queueTaskDraft}
            taskDraftRef={(element) => {
              queueTaskDraftRef.current = element;
            }}
            taskBusy={queueTaskBusy}
            editingTaskId={editingQueueTaskId}
            editDraft={queueTaskEditDraft}
            onTaskDraftChange={setQueueTaskDraft}
            onTaskAdd={addQueueTask}
            onTaskCompositionStart={startQueueTaskComposition}
            onTaskCompositionEnd={endQueueTaskComposition}
            isTaskComposing={() => queueTaskComposingRef.current}
            onEditStart={startEditingQueueTask}
            onEditCancel={() => {
              setEditingQueueTaskId(null);
              setQueueTaskEditDraft("");
            }}
            onEditDraftChange={setQueueTaskEditDraft}
            onTaskUpdate={updateQueueTask}
            onTaskRemove={removeQueueTask}
            onTaskPullBack={pullBackQueueTask}
            onTaskMove={moveQueueTask}
            onTaskClear={clearNormalQueue}
          />
        </div>
        <button
          className="tool-button"
          title="导出当前 Session Markdown"
          disabled={!currentSessionId || state.sessionMessageCount === 0 || exportBusy}
          onClick={exportCurrentSession}
        >
          <FileText size={17} /> {exportBusy ? "导出中" : "导出"}
        </button>
        <button className="tool-button" title="插件配置" onClick={() => setPluginDialogOpen(true)}>
          <Plug size={17} /> 插件配置
        </button>
        <button className="tool-button" title="切换模型" onClick={() => setModelDialogOpen(true)}>
          <Bot size={17} /> Model: {selectedModel}
        </button>
        <div className="toolbar-menu">
          <button
            className={`tool-button ${debugMenuOpen ? "active" : ""}`}
            title="菜单"
            onClick={() => setDebugMenuOpen((v) => !v)}
          >
            <span style={{ letterSpacing: "1px", fontWeight: 700 }}>···</span>
          </button>
          {debugMenuOpen && (
            <div className="menu-popover">
              <label className="menu-item">
                <input
                  type="checkbox"
                  checked={showDebug}
                  onChange={(event) => {
                    const next = { ...config, showDebug: event.target.checked };
                    setConfig(next);
                    void saveGlobalAndSet(next);
                  }}
                />
                Debug 信息
              </label>
              <button className="menu-item" onClick={() => restartWith(config)}>
                <RotateCcw size={15} /> 重启 Engine
              </button>
            </div>
          )}
        </div>
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
              <ChatBubble node={node} key={node.id} onForkMessage={forkMessage} />
            ))}
          {state.processing && <ProcessingLine processing={state.processing} />}
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
          }}
        />
        <button className="primary-button" disabled={!input.trim()} onClick={submitInput}>
          <Send size={18} /> 发送
        </button>
        {state.control.paused && state.control.resumable ? (
          <button className="primary-button" onClick={() => sendCommand("resume", {})}>
            <Play size={16} /> 继续
          </button>
        ) : (
          <button
            className="danger-button"
            disabled={state.runnerState !== "RUNNING" && state.runnerState !== "INTERRUPTING"}
            onClick={() => sendCommand("stop", { reason: "user" })}
          >
            <Square size={16} /> 停止
          </button>
        )}
      </footer>

      {modelDialogOpen && (
        <ModelDialog
          models={metadata.inspect.models}
          current={config.modelName}
          onClose={() => setModelDialogOpen(false)}
          onSelect={selectModel}
          onRefresh={refreshProviderModels}
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
      {contextCompactDialogOpen && (
        <ContextCompactDialog
          usage={state.contextUsage}
          busy={contextCompactBusy}
          onClose={() => {
            if (!contextCompactBusy) setContextCompactDialogOpen(false);
          }}
          onConfirm={compactContextManually}
        />
      )}
    </div>
  );
}

function QueueStatusCell({
  queueLengths,
  queueMessages,
  control,
  open,
  onToggle,
  taskDraft,
  taskDraftRef,
  taskBusy,
  editingTaskId,
  editDraft,
  onTaskDraftChange,
  onTaskAdd,
  onTaskCompositionStart,
  onTaskCompositionEnd,
  isTaskComposing,
  onEditStart,
  onEditCancel,
  onEditDraftChange,
  onTaskUpdate,
  onTaskRemove,
  onTaskPullBack,
  onTaskMove,
  onTaskClear
}: {
  queueLengths: Record<QueueKind, number>;
  queueMessages: Record<QueueKind, QueueMessageState[]>;
  control: RuntimeControlState;
  open: boolean;
  onToggle: () => void;
  taskDraft: string;
  taskDraftRef: (element: HTMLTextAreaElement | null) => void;
  taskBusy: boolean;
  editingTaskId: string | null;
  editDraft: string;
  onTaskDraftChange: (value: string) => void;
  onTaskAdd: () => void;
  onTaskCompositionStart: () => void;
  onTaskCompositionEnd: () => void;
  isTaskComposing: () => boolean;
  onEditStart: (message: QueueMessageState) => void;
  onEditCancel: () => void;
  onEditDraftChange: (value: string) => void;
  onTaskUpdate: (messageId: string, content: string) => void;
  onTaskRemove: (messageId: string) => void;
  onTaskPullBack: (message: QueueMessageState) => void;
  onTaskMove: (messageId: string, direction: -1 | 1) => void;
  onTaskClear: () => void;
}) {
  const highPriorityCount = hasHighPriorityWork(queueLengths, queueMessages) ? 1 : 0;
  const normalCount = normalQueueCount(queueLengths, queueMessages);

  return (
    <div className={`priority-status ${open ? "active" : ""}`}>
      <button
        type="button"
        className="priority-status-trigger"
        title="Insert messages deliver soon; queued messages run later."
        aria-label={renderQueueStatusText(queueLengths, queueMessages)}
        aria-pressed={open}
        onClick={onToggle}
      >
        <span className="status-cell-label">Message</span>
        <span className="message-status-widget" aria-hidden="true">
          <span><span>Insert</span><strong>{highPriorityCount}</strong></span>
          <span><span>Queue</span><strong>{normalCount}</strong></span>
        </span>
      </button>
      {open && (
        <QueuePopover
          queueLengths={queueLengths}
          queueMessages={queueMessages}
          control={control}
          taskDraft={taskDraft}
          taskDraftRef={taskDraftRef}
          taskBusy={taskBusy}
          editingTaskId={editingTaskId}
          editDraft={editDraft}
          onTaskDraftChange={onTaskDraftChange}
          onTaskAdd={onTaskAdd}
          onTaskCompositionStart={onTaskCompositionStart}
          onTaskCompositionEnd={onTaskCompositionEnd}
          isTaskComposing={isTaskComposing}
          onEditStart={onEditStart}
          onEditCancel={onEditCancel}
          onEditDraftChange={onEditDraftChange}
          onTaskUpdate={onTaskUpdate}
          onTaskRemove={onTaskRemove}
          onTaskPullBack={onTaskPullBack}
          onTaskMove={onTaskMove}
          onTaskClear={onTaskClear}
        />
      )}
    </div>
  );
}

function ContextUsageCell({
  usage,
  compression,
  busy,
  disabled,
  onRequestCompact
}: {
  usage?: ContextUsageState;
  compression?: ContextCompressionState;
  busy: boolean;
  disabled: boolean;
  onRequestCompact: () => void;
}) {
  const compressing = compression?.active === true;
  const ratio = usage?.ratio ?? 0;
  const percent = usage?.ratio === undefined ? "n/a" : `${Math.round(ratio * 100)}%`;
  const used = usage ? compactNumber(usage.usedTokens) : "-";
  const max = usage?.maxContextTokens ? compactNumber(usage.maxContextTokens) : "-";
  const usageLabel = `${usage?.source === "estimate" ? "~" : ""}${used}/${max}`;
  const inactive = disabled || busy || compressing;
  const title = compressing
    ? `Context compressing ${usageLabel}`
    : inactive
      ? `Context ${usageLabel}`
      : `Context ${usageLabel} · 点击手动压缩上下文`;
  return (
    <button
      type="button"
      className={`context-status ${compressing ? "compressing" : ""}`}
      title={title}
      disabled={inactive}
      onClick={onRequestCompact}
      aria-label={`Context ${percent} ${usageLabel}`}
    >
      <span className="status-cell-label">Context</span>
      <span className={`context-status-widget ${compressing ? "compressing" : ""}`}>
        {compressing ? (
          <>
            <LoaderCircle className="context-status-spinner" size={13} aria-label="Compressing context" />
            <span>Compressing</span>
          </>
        ) : (
          <>
            <span className="context-usage-line">{usageLabel}</span>
            <span className="context-meter" aria-hidden="true">
              <span className="context-meter-fill" style={{ width: `${Math.min(100, Math.max(0, ratio * 100))}%` }} />
              <strong className="context-meter-label">{percent}</strong>
            </span>
          </>
        )}
      </span>
    </button>
  );
}

function SessionStatusCell({
  messageCount,
  runningCount,
  loadedCount,
  sessions,
  currentSessionId,
  open,
  busy,
  onToggle,
  onSelect,
  onDelete,
  onNew,
  onFork
}: {
  messageCount: number;
  runningCount: number;
  loadedCount: number;
  sessions: SessionMetaPayload[];
  currentSessionId: string | null;
  open: boolean;
  busy: boolean;
  onToggle: () => void;
  onSelect: (sessionId: string) => void;
  onDelete: (sessionId: string) => void;
  onNew: () => void;
  onFork: (sessionId?: string) => void;
}) {
  const sortedSessions = sortSessionsByCreatedAt(sessions);
  const canForkCurrent = Boolean(currentSessionId) && messageCount > 0;
  const currentSession = currentSessionId
    ? sessions.find((session) => session.session_id === currentSessionId)
    : undefined;
  const currentSessionName = currentSession
    ? sessionDisplayName(currentSession)
    : currentSessionId
      ? shortSessionId(currentSessionId)
      : "-";
  const sessionRuntimeLabel = `${runningCount} Running / ${loadedCount} Active`;
  const currentSessionLabel = `Current: ${currentSessionName}`;

  return (
    <div className={`session-status ${open ? "active" : ""}`}>
      <button
        type="button"
        className="session-trigger"
        title={`切换 Session\n${sessionRuntimeLabel}\n${currentSessionLabel}`}
        disabled={busy}
        onClick={onToggle}
        aria-label={`Session ${sessionRuntimeLabel}. ${currentSessionLabel}`}
      >
        <span className="status-cell-label">Session</span>
        <span className="session-status-widget" aria-hidden="true">
          <span className="session-runtime-line">
            <strong>{runningCount}</strong>
            <span>Running /</span>
            <strong>{loadedCount}</strong>
            <span>Active</span>
          </span>
          <span className="session-current-line">
            <span>Current:</span>
            <strong>{currentSessionName}</strong>
          </span>
        </span>
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
      <button
        type="button"
        className="mini-button icon-only"
        title="Fork 当前 Session"
        aria-label="Fork 当前 Session"
        disabled={busy || !canForkCurrent}
        onClick={(event) => {
          event.stopPropagation();
          onFork(currentSessionId ?? undefined);
        }}
      >
        <GitFork size={14} />
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
              const isLoadedHere = session.load_state === "loaded" || session.load_state === "running";
              const isRunning = session.load_state === "running";
              const isLocked = session.locked === true && !isCurrent && !isLoadedHere;
              const canShowDelete = !isRunning && (!isLocked || isLoadedHere);
              return (
                <div
                  key={session.session_id}
                  className={`session-option ${isCurrent ? "current" : ""} ${isLocked ? "locked" : ""} ${isLoadedHere ? "loaded" : ""}`}
                >
                  <button
                    type="button"
                    className="session-select"
                    title={isLocked ? "Session 正被其他 Hawi engine 使用，可 Fork 后继续" : "切换 Session"}
                    disabled={busy || isLocked}
                    onClick={() => onSelect(session.session_id)}
                  >
                    <small className="session-date">
                      {formatSessionTimestamp(session.created_at || session.updated_at)}
                    </small>
                    <span className="session-title">
                      <SessionLoadIndicator state={session.load_state ?? "unloaded"} />
                      {isLocked && <Lock size={12} />}
                      {sessionDisplayName(session)}
                    </span>
                  </button>
                  {(!isCurrent || canShowDelete) && (
                    <div className="session-actions">
                      {!isCurrent && (
                        <button
                          type="button"
                          className="session-action"
                          title="Fork Session"
                          aria-label={`Fork Session ${shortSessionId(session.session_id)}`}
                          disabled={busy}
                          onClick={() => onFork(session.session_id)}
                        >
                          <GitFork size={13} />
                        </button>
                      )}
                      {canShowDelete && (
                        <button
                          type="button"
                          className="session-delete"
                          title={isLoadedHere ? "关闭并删除 Session" : "删除 Session"}
                          aria-label={`删除 Session ${shortSessionId(session.session_id)}`}
                          disabled={busy}
                          onClick={() => onDelete(session.session_id)}
                        >
                          <Trash2 size={13} />
                        </button>
                      )}
                    </div>
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

function SessionLoadIndicator({ state }: { state: SessionLoadState }) {
  if (state === "running") {
    return <LoaderCircle className="session-load-spinner" size={13} aria-label={sessionLoadStateLabel(state)} />;
  }
  return (
    <span
      className={`session-load-dot ${state}`}
      aria-label={sessionLoadStateLabel(state)}
    />
  );
}

function QueuePopover({
  queueLengths,
  queueMessages,
  control,
  taskDraft,
  taskDraftRef,
  taskBusy,
  editingTaskId,
  editDraft,
  onTaskDraftChange,
  onTaskAdd,
  onTaskCompositionStart,
  onTaskCompositionEnd,
  isTaskComposing,
  onEditStart,
  onEditCancel,
  onEditDraftChange,
  onTaskUpdate,
  onTaskRemove,
  onTaskPullBack,
  onTaskMove,
  onTaskClear
}: {
  queueLengths: Record<QueueKind, number>;
  queueMessages: Record<QueueKind, QueueMessageState[]>;
  control: RuntimeControlState;
  taskDraft: string;
  taskDraftRef: (element: HTMLTextAreaElement | null) => void;
  taskBusy: boolean;
  editingTaskId: string | null;
  editDraft: string;
  onTaskDraftChange: (value: string) => void;
  onTaskAdd: () => void;
  onTaskCompositionStart: () => void;
  onTaskCompositionEnd: () => void;
  isTaskComposing: () => boolean;
  onEditStart: (message: QueueMessageState) => void;
  onEditCancel: () => void;
  onEditDraftChange: (value: string) => void;
  onTaskUpdate: (messageId: string, content: string) => void;
  onTaskRemove: (messageId: string) => void;
  onTaskPullBack: (message: QueueMessageState) => void;
  onTaskMove: (messageId: string, direction: -1 | 1) => void;
  onTaskClear: () => void;
}) {
  const normalCount = normalQueueCount(queueLengths, queueMessages);
  const total = (hasHighPriorityWork(queueLengths, queueMessages) ? 1 : 0) + normalCount;
  return (
    <div className="queue-popover">
      <header>
        <span>待处理</span>
        <strong>{total}</strong>
      </header>
      {control.paused && (
        <div className="queue-pause-note">
          {control.last_error_message
            ? `已暂停：${control.last_error_message}`
            : "已暂停，队列任务不会自动执行。"}
        </div>
      )}
      <QueueMessageGroup
        kind="high_prio"
        length={Math.max(queueLengths.high_prio, queueMessages.high_prio.length)}
        messages={queueMessages.high_prio}
      />
      <QueueTaskGroup
        length={normalCount}
        messages={queueMessages.normal}
        draft={taskDraft}
        draftRef={taskDraftRef}
        busy={taskBusy}
        editingTaskId={editingTaskId}
        editDraft={editDraft}
        onDraftChange={onTaskDraftChange}
        onAdd={onTaskAdd}
        onCompositionStart={onTaskCompositionStart}
        onCompositionEnd={onTaskCompositionEnd}
        isComposing={isTaskComposing}
        onEditStart={onEditStart}
        onEditCancel={onEditCancel}
        onEditDraftChange={onEditDraftChange}
        onUpdate={onTaskUpdate}
        onRemove={onTaskRemove}
        onPullBack={onTaskPullBack}
        onMove={onTaskMove}
        onClear={onTaskClear}
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

function QueueTaskGroup({
  length,
  messages,
  draft,
  draftRef,
  busy,
  editingTaskId,
  editDraft,
  onDraftChange,
  onAdd,
  onCompositionStart,
  onCompositionEnd,
  isComposing,
  onEditStart,
  onEditCancel,
  onEditDraftChange,
  onUpdate,
  onRemove,
  onPullBack,
  onMove,
  onClear
}: {
  length: number;
  messages: QueueMessageState[];
  draft: string;
  draftRef: (element: HTMLTextAreaElement | null) => void;
  busy: boolean;
  editingTaskId: string | null;
  editDraft: string;
  onDraftChange: (value: string) => void;
  onAdd: () => void;
  onCompositionStart: () => void;
  onCompositionEnd: () => void;
  isComposing: () => boolean;
  onEditStart: (message: QueueMessageState) => void;
  onEditCancel: () => void;
  onEditDraftChange: (value: string) => void;
  onUpdate: (messageId: string, content: string) => void;
  onRemove: (messageId: string) => void;
  onPullBack: (message: QueueMessageState) => void;
  onMove: (messageId: string, direction: -1 | 1) => void;
  onClear: () => void;
}) {
  return (
    <section className="queue-group queue-task-group">
      <header>
        <span>{queueLabels.normal}</span>
        <div className="queue-group-actions">
          <strong>{length}</strong>
          <button
            type="button"
            className="queue-text-action"
            title="清空稍后任务"
            disabled={busy || messages.length === 0}
            onClick={onClear}
          >
            清空
          </button>
        </div>
      </header>
      <form
        className="queue-add"
        onSubmit={(event) => {
          event.preventDefault();
          onAdd();
        }}
      >
        <textarea
          ref={draftRef}
          rows={2}
          value={draft}
          placeholder="添加一个稍后任务..."
          disabled={busy}
          onChange={(event) => onDraftChange(event.target.value)}
          onCompositionStart={onCompositionStart}
          onCompositionEnd={onCompositionEnd}
          onKeyDown={(event) => {
            if (shouldSubmitInputFromKeyEvent(event, isComposing())) {
              event.preventDefault();
              onAdd();
            }
          }}
        />
        <button type="submit" className="queue-add-button" disabled={busy || !draft.trim()}>
          <Plus size={14} /> 加入队列
        </button>
      </form>
      {messages.length === 0 ? (
        <div className="queue-empty">空</div>
      ) : (
        <div className="queue-message-list queue-task-list">
          {messages.map((message, index) => (
            <QueueTaskItem
              message={message}
              key={`normal-${message.id}`}
              index={index}
              count={messages.length}
              busy={busy}
              editing={editingTaskId === message.id}
              editDraft={editDraft}
              onEditStart={onEditStart}
              onEditCancel={onEditCancel}
              onEditDraftChange={onEditDraftChange}
              onUpdate={onUpdate}
              onRemove={onRemove}
              onPullBack={onPullBack}
              onMove={onMove}
            />
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

function QueueTaskItem({
  message,
  index,
  count,
  busy,
  editing,
  editDraft,
  onEditStart,
  onEditCancel,
  onEditDraftChange,
  onUpdate,
  onRemove,
  onPullBack,
  onMove
}: {
  message: QueueMessageState;
  index: number;
  count: number;
  busy: boolean;
  editing: boolean;
  editDraft: string;
  onEditStart: (message: QueueMessageState) => void;
  onEditCancel: () => void;
  onEditDraftChange: (value: string) => void;
  onUpdate: (messageId: string, content: string) => void;
  onRemove: (messageId: string) => void;
  onPullBack: (message: QueueMessageState) => void;
  onMove: (messageId: string, direction: -1 | 1) => void;
}) {
  const timestamp = formatQueueTimestamp(message.createdAt);
  if (editing) {
    return (
      <article className="queue-message queue-task editing">
        <div className="queue-message-meta">
          <span>{message.id}</span>
          {timestamp && <time>{timestamp}</time>}
        </div>
        <textarea
          rows={3}
          value={editDraft}
          disabled={busy}
          onChange={(event) => onEditDraftChange(event.target.value)}
        />
        <div className="queue-task-actions">
          <button
            type="button"
            title="保存"
            disabled={busy || !editDraft.trim()}
            onClick={() => onUpdate(message.id, editDraft)}
          >
            <Check size={13} />
          </button>
          <button type="button" title="取消" disabled={busy} onClick={onEditCancel}>
            <X size={13} />
          </button>
        </div>
      </article>
    );
  }

  return (
    <article className="queue-message queue-task">
      <div className="queue-message-meta">
        <span>{message.id}</span>
        {timestamp && <time>{timestamp}</time>}
      </div>
      <p>{(message.content ?? message.contentPreview) || "空消息"}</p>
      <div className="queue-task-actions">
        <button
          type="button"
          title="上移"
          disabled={busy || index === 0}
          onClick={() => onMove(message.id, -1)}
        >
          <ArrowUp size={13} />
        </button>
        <button
          type="button"
          title="下移"
          disabled={busy || index >= count - 1}
          onClick={() => onMove(message.id, 1)}
        >
          <ArrowDown size={13} />
        </button>
        <button type="button" title="编辑" disabled={busy} onClick={() => onEditStart(message)}>
          <Pencil size={13} />
        </button>
        <button type="button" title="拉回编辑" disabled={busy} onClick={() => onPullBack(message)}>
          <RotateCcw size={13} />
        </button>
        <button type="button" title="删除" disabled={busy} onClick={() => onRemove(message.id)}>
          <Trash2 size={13} />
        </button>
      </div>
    </article>
  );
}

const ChatBubble = memo(function ChatBubble({
  node,
  onForkMessage
}: {
  node: ChatNode;
  onForkMessage: (node: ChatNode) => void;
}) {
  if (node.kind === "divider") {
    return (
      <div className="run-divider">
        <span>{node.content}</span>
      </div>
    );
  }
  if (node.kind === "compact") {
    return (
      <div className={`run-divider context-compact ${node.complete === false ? "active" : "complete"}`}>
        {node.complete === false && <LiveSpinner title="Compressing context" />}
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
  if (node.kind === "framework" && node.framework) {
    return <FrameworkBubble item={node.framework} />;
  }
  if (node.kind === "thinking") {
    return <ThinkingBubble node={node} />;
  }
  return <MessageBubble node={node} onForkMessage={onForkMessage} />;
});

function ProcessingLine({ processing }: { processing: ProcessingState }) {
  return (
    <div className="processing-line">
      <LiveSpinner title="处理中" />
      <span>{processing.content || "处理中..."}</span>
    </div>
  );
}

const FrameworkBubble = memo(function FrameworkBubble({
  item,
  embedded = false
}: {
  item: FrameworkInjectionState;
  embedded?: boolean;
}) {
  const [collapsed, setCollapsed] = useState(true);
  const html = renderMarkdown(item.content);
  const toggleCollapsed = () => setCollapsed((value) => !value);
  const source = frameworkInjectionSourceLabel(item);

  return (
    <article className={`bubble framework ${embedded ? "embedded" : ""} ${collapsed ? "collapsed" : "expanded"}`}>
      <div className="bubble-head collapsible-head" onClick={toggleCollapsed}>
        <span className="bubble-title framework-title">
          <Plug size={15} />
          <span className="framework-label">{item.label}</span>
          {source && <span className="framework-source">{source}</span>}
        </span>
        <span className="message-actions">
          <CopyButton text={item.content} title="复制注入内容" />
          <button
            className="thinking-toggle framework-toggle"
            title={collapsed ? "展开注入内容" : "折叠注入内容"}
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
      <div className={`collapsible-body framework-body ${collapsed ? "is-collapsed" : ""}`} aria-hidden={collapsed}>
        <div className="collapsible-body-inner">
          <div className="markdown" dangerouslySetInnerHTML={{ __html: html }} />
        </div>
      </div>
    </article>
  );
});

const MessageBubble = memo(function MessageBubble({
  node,
  onForkMessage
}: {
  node: ChatNode;
  onForkMessage: (node: ChatNode) => void;
}) {
  const [collapsed, setCollapsed] = useState(false);
  const html = node.kind === "agent" ? renderMarkdown(node.content) : escapeText(node.content);
  const label = node.kind === "user" ? labelForUserMessage(node) : labelForKind(node.kind);
  const receiving = node.kind === "agent" && node.complete === false;
  const toggleCollapsed = () => setCollapsed((value) => !value);
  const expandCollapsed = () => setCollapsed(false);
  const canFork = node.canFork === true && typeof node.contextMessageIndex === "number";
  const beforeInjections = (node.injections ?? []).filter((item) => item.mergePosition === "before");
  const afterInjections = (node.injections ?? []).filter((item) => item.mergePosition !== "before");

  return (
    <article className={`bubble ${node.kind} message ${collapsed ? "message-collapsed" : ""}`}>
      <div className="bubble-head collapsible-head" onClick={toggleCollapsed}>
        <span className="bubble-title">
          {label}
          <BlockStreamStatus
            receiving={receiving}
            durationMs={node.streamDurationMs}
            receivingTitle="正在接收消息"
          />
        </span>
        <span className="message-actions">
          {canFork && (
            <button
              className="thinking-toggle message-fork"
              title={node.kind === "user" ? "Fork 到这条用户消息前" : "Fork 到这条回复后"}
              aria-label={node.kind === "user" ? "Fork 到这条用户消息前" : "Fork 到这条回复后"}
              onClick={(event) => {
                event.stopPropagation();
                onForkMessage(node);
              }}
            >
              <GitFork size={14} />
            </button>
          )}
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
      {beforeInjections.length > 0 && (
        <div className="message-injections before">
          {beforeInjections.map((item) => (
            <FrameworkBubble item={item} embedded key={item.id} />
          ))}
        </div>
      )}
      <div className={`message-body ${collapsed ? "is-collapsed" : ""}`}>
        <div className="markdown" dangerouslySetInnerHTML={{ __html: html }} />
        {collapsed && (
          <button
            type="button"
            className="message-collapse-mask"
            title="点击展开消息"
            aria-label={`展开${label}`}
            onClick={expandCollapsed}
          />
        )}
      </div>
      {afterInjections.length > 0 && (
        <div className="message-injections after">
          {afterInjections.map((item) => (
            <FrameworkBubble item={item} embedded key={item.id} />
          ))}
        </div>
      )}
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

function frameworkInjectionSourceLabel(item: FrameworkInjectionState): string {
  const source = item.pluginName
    ?? (item.pluginRole === "framework" ? "Hawi" : item.pluginRole);
  const injection = item.injectionName && item.injectionName !== source
    ? item.injectionName
    : undefined;
  const target = frameworkInjectionTargetLabel(item);
  return [source, injection, target].filter(Boolean).join(" · ");
}

function frameworkInjectionTargetLabel(item: FrameworkInjectionState): string | undefined {
  if (item.kind === "system_prompt") {
    return "system prompt";
  }
  if (item.kind === "context_injected") {
    return [item.hookType, item.role].filter(Boolean).join("/");
  }
  if (item.kind === "tool_parameter_injected") {
    return item.toolName ? `${item.toolName} parameters` : "tool parameters";
  }
  if (item.kind === "tool_runtime_context_injected") {
    if (item.toolName && item.parameterName) {
      return `${item.toolName}.${item.parameterName}`;
    }
    return item.toolName ?? item.parameterName ?? "runtime context";
  }
  return item.toolCallId ? `tool ${item.toolCallId}` : undefined;
}

const ThinkingBubble = memo(function ThinkingBubble({ node }: { node: ChatNode }) {
  const [collapsed, setCollapsed] = useState(() => node.complete === true);
  const autoCollapsedRef = useRef(node.complete === true);
  const html = renderMarkdown(node.content);
  const receiving = node.complete === false;
  const toggleCollapsed = () => setCollapsed((value) => !value);
  const expandCollapsed = () => setCollapsed(false);

  useEffect(() => {
    if (node.complete === true && !autoCollapsedRef.current) {
      setCollapsed(true);
      autoCollapsedRef.current = true;
    }
  }, [node.complete]);

  return (
    <article className={`bubble thinking ${collapsed ? "collapsed" : ""}`}>
      <div className="bubble-head collapsible-head" onClick={toggleCollapsed}>
        <span className="bubble-title">
          <Brain size={15} /> Thinking
          <BlockStreamStatus
            receiving={receiving}
            durationMs={node.streamDurationMs}
            receivingTitle="正在接收思考内容"
          />
        </span>
        <span className="message-actions">
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
      <div
        className={`thinking-summary ${collapsed ? "is-visible" : ""}`}
        aria-hidden={!collapsed}
        role={collapsed ? "button" : undefined}
        tabIndex={collapsed ? 0 : undefined}
        title={collapsed ? "点击展开思考内容" : undefined}
        onClick={() => {
          if (collapsed) expandCollapsed();
        }}
        onKeyDown={(event) => {
          if (!collapsed || (event.key !== "Enter" && event.key !== " ")) return;
          event.preventDefault();
          expandCollapsed();
        }}
      >
        {thinkingExcerpt(node.content)}
      </div>
    </article>
  );
});

const ToolBubble = memo(function ToolBubble({ node }: { node: ChatNode }) {
  const tool = node.tool!;
  const running = tool.status === "running";
  const receivingArguments = tool.argsState !== "complete";
  const [collapsed, setCollapsed] = useState(true);
  const presentation = toolPresentation(tool);
  const toggleCollapsed = () => setCollapsed((value) => !value);

  return (
    <article className={`bubble tool ${tool.status} ${collapsed ? "collapsed" : ""}`}>
      <div className="bubble-head collapsible-head" onClick={toggleCollapsed}>
        <span className="tool-title">
          <span className="tool-name">
            <Wrench size={15} /> {presentation.label}
            <BlockStreamStatus
              receiving={receivingArguments}
              durationMs={tool.streamDurationMs}
              receivingTitle="正在接收工具调用"
            />
          </span>
          {presentation.detail && (
            <span className={`tool-subject ${presentation.detailKind}`}>
              {presentation.detail}
            </span>
          )}
        </span>
        <span className="tool-actions">
          <strong>{running ? "running" : tool.status}</strong>
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
            {renderToolArguments(tool)}
          </details>
          {(tool.resultPreview || tool.resultData !== undefined || tool.status === "fail") && (
            <details open>
              <summary>Result {tool.durationMs ? `· ${tool.durationMs.toFixed(0)}ms` : ""}</summary>
              {renderToolResult(tool)}
            </details>
          )}
        </div>
      </div>
    </article>
  );
});

/** 根据工具名称渲染参数区域 */
function renderToolArguments(tool: ToolState): ReactNode {
  if (tool.argsState !== "complete") {
    return <div className="tool-hint">Receiving arguments...</div>;
  }
  const args = tool.arguments;
  if (!isRecord(args)) {
    return args === undefined
      ? <div className="tool-hint">No arguments.</div>
      : <ToolArguments value={args} />;
  }
  const toolKind = filesystemToolKind(tool.name);

  // write_file: content 单独一个大可滚动框
  if (toolKind === "write_file" && typeof args.content === "string") {
    const { content, ...rest } = args;
    const language = detectLanguageFromPath(optionalRecordString(args.file_path));
    return (
      <>
        {Object.keys(rest).length > 0 && (
          <div className="argument-list">
            {Object.entries(rest).map(([key, item]) => (
              <ArgumentRow key={key} name={key} value={item} />
            ))}
          </div>
        )}
        <div className="tool-content-section">
          <div className="tool-content-label">Content</div>
          <CodeScrollBlock
            className="tool-content-block"
            value={content}
            language={language}
          />
        </div>
      </>
    );
  }

  // edit_file: old / new 水平分屏
  if (toolKind === "edit_file") {
    return <EditFileArguments args={args} />;
  }

  // 默认参数渲染
  return <ToolArguments value={args} />;
}

/** 根据工具名称渲染结果区域 */
function renderToolResult(tool: ToolState): ReactNode {
  const preview = tool.resultPreview || (tool.status === "fail" ? "Tool failed without an error message." : "");
  const toolKind = filesystemToolKind(tool.name);

  // read_file: 从 resultData 解析结构化输出做语法高亮
  if (toolKind === "read_file") {
    // 优先从 resultData 解析结构化数据
    const resultData = isRecord(tool.resultData) ? tool.resultData : undefined;
    const fileInfo = resultData ? (isRecord(resultData.file) ? resultData.file : undefined) : undefined;
    if (resultData?.type === "file_unchanged") {
      const filePath = fileInfo ? optionalRecordString(fileInfo.filePath) : undefined;
      return (
        <div className="tool-result-view">
          <ToolResultMeta items={[filePath, "unchanged"].filter(isNonEmptyString)} />
          <div className="tool-hint">File content is unchanged from the previous read.</div>
        </div>
      );
    }
    const rawContent = fileInfo ? String(fileInfo.content ?? "") : preview;
    const detectedLang = fileInfo ? optionalRecordString(fileInfo.language) : undefined;
    const normalizedFile = normalizeReadFileContent(rawContent, detectedLang);
    const headerLines: string[] = [];
    if (fileInfo) {
      const filePath = String(fileInfo.filePath ?? "");
      const totalLines = Number(fileInfo.totalLines ?? 0);
      const startLine = Number(fileInfo.startLine ?? 0);
      const numLines = Number(fileInfo.numLines ?? 0);
      if (filePath) headerLines.push(filePath);
      if (totalLines > 0 && numLines > 0 && numLines < totalLines) {
        headerLines.push(`Lines ${startLine + 1}-${startLine + numLines} of ${totalLines}`);
      }
      if (normalizedFile.language && normalizedFile.language !== "text") {
        headerLines.push(`language: ${normalizedFile.language}`);
      }
    }
    return (
      <div className="tool-result-view">
        <ToolResultMeta items={headerLines} />
        <CodeScrollBlock
          className="tool-result-block nowrap"
          value={normalizedFile.content}
          language={normalizedFile.language}
        />
      </div>
    );
  }

  // run_shell: 终端风格
  if (toolKind === "run_shell" && preview) {
    return (
      <pre className="tool-result-block terminal nowrap" onWheel={handleNestedVerticalScroll}>{preview}</pre>
    );
  }

  // list_dir: ls -la 终端风格
  if (toolKind === "list_dir") {
    const resultData = isRecord(tool.resultData) ? tool.resultData : undefined;
    const lsText = resultData?.type === "ls_output" ? String(resultData.text ?? "") : preview;
    const meta = resultData
      ? [
        formatCount(resultData.numEntries, "entry", "entries"),
        resultData.isTruncated === true ? "truncated" : ""
      ].filter(isNonEmptyString)
      : [];
    return (
      <div className="tool-result-view">
        <ToolResultMeta items={meta} />
        <pre className="tool-result-block terminal nowrap" onWheel={handleNestedVerticalScroll}>{lsText}</pre>
      </div>
    );
  }

  if (toolKind === "glob") {
    const resultData = isRecord(tool.resultData) ? tool.resultData : undefined;
    const matches = Array.isArray(resultData?.matches)
      ? resultData.matches.map((item) => String(item))
      : [];
    return (
      <div className="tool-result-view">
        <ToolResultMeta items={[formatCount(matches.length, "match", "matches")]} />
        <pre className="tool-result-block nowrap" onWheel={handleNestedVerticalScroll}>
          {matches.length > 0 ? matches.join("\n") : "No matches."}
        </pre>
      </div>
    );
  }

  if (toolKind === "grep") {
    const resultData = isRecord(tool.resultData) ? tool.resultData : undefined;
    const content = resultData && typeof resultData.content === "string"
      ? resultData.content
      : preview;
    const meta = resultData
      ? [
        formatCount(resultData.numMatches ?? resultData.numLines, "match", "matches"),
        formatCount(resultData.numFiles, "file", "files")
      ].filter(isNonEmptyString)
      : [];
    return (
      <div className="tool-result-view">
        <ToolResultMeta items={meta} />
        <pre className="tool-result-block search nowrap" onWheel={handleNestedVerticalScroll}>{content || "No matches."}</pre>
      </div>
    );
  }

  // 其他工具: 纯文本结果（已在 formatToolResultText 中渲染为文本）
  return (
    <pre className="tool-result-block" onWheel={handleNestedVerticalScroll}>
      {preview || "Tool failed without an error message."}
    </pre>
  );
}

function ToolResultMeta({ items }: { items: string[] }) {
  if (items.length === 0) return null;
  return <div className="tool-result-meta">{items.join(" · ")}</div>;
}

function EditFileArguments({ args }: { args: Record<string, unknown> }) {
  const oldRef = useRef<HTMLPreElement | null>(null);
  const newRef = useRef<HTMLPreElement | null>(null);
  const syncingRef = useRef(false);
  const { old_string, new_string, file_path, replace_all, ...extra } = args;
  const oldText = old_string != null ? String(old_string) : "";
  const newText = new_string != null ? String(new_string) : "";
  const language = detectLanguageFromPath(optionalRecordString(file_path));
  const metaArgs: Record<string, unknown> = {};
  if (file_path !== undefined) metaArgs.file_path = file_path;
  if (replace_all !== undefined) metaArgs.replace_all = replace_all;
  Object.assign(metaArgs, extra);

  function syncScroll(source: ReactUIEvent<HTMLPreElement>, target: HTMLPreElement | null) {
    if (!target || syncingRef.current) return;
    const sourceElement = source.currentTarget;
    syncingRef.current = true;
    target.scrollTop = clampedScrollOffset(
      sourceElement.scrollTop,
      target.scrollHeight - target.clientHeight
    );
    target.scrollLeft = clampedScrollOffset(
      sourceElement.scrollLeft,
      target.scrollWidth - target.clientWidth
    );
    window.requestAnimationFrame(() => {
      syncingRef.current = false;
    });
  }

  return (
    <>
      {Object.keys(metaArgs).length > 0 && (
        <div className="argument-list compact">
          {Object.entries(metaArgs).map(([key, item]) => (
            <ArgumentRow key={key} name={key} value={item} />
          ))}
        </div>
      )}
      <div className="tool-diff-section sync">
        <div className="tool-diff-pane old">
          <div className="tool-diff-label">
            <span>Old</span>
            <small>{formatTextStats(oldText)}</small>
          </div>
          <CodeScrollBlock
            ref={oldRef}
            className="tool-diff-block"
            value={oldText}
            language={language}
            onScroll={(event) => syncScroll(event, newRef.current)}
          />
        </div>
        <div className="tool-diff-pane new">
          <div className="tool-diff-label">
            <span>New</span>
            <small>{formatTextStats(newText)}</small>
          </div>
          <CodeScrollBlock
            ref={newRef}
            className="tool-diff-block"
            value={newText}
            language={language}
            onScroll={(event) => syncScroll(event, oldRef.current)}
          />
        </div>
      </div>
    </>
  );
}

interface CodeScrollBlockProps {
  value: string;
  language?: string;
  className?: string;
  onScroll?: (event: ReactUIEvent<HTMLPreElement>) => void;
}

const CodeScrollBlock = memo(forwardRef<HTMLPreElement, CodeScrollBlockProps>(function CodeScrollBlock({
  value,
  language,
  className = "",
  onScroll
}, ref) {
  const highlighted = highlightedCode(value, language);
  const codeClass = highlighted.language
    ? `hljs language-${highlighted.language}`
    : "hljs";
  return (
    <pre
      ref={ref}
      className={className}
      onScroll={onScroll}
      onWheel={handleNestedVerticalScroll}
    >
      <code
        className={codeClass}
        dangerouslySetInnerHTML={{ __html: highlighted.html }}
      />
    </pre>
  );
}));

function BlockStreamStatus({
  receiving,
  durationMs,
  receivingTitle,
}: {
  receiving: boolean;
  durationMs?: number;
  receivingTitle: string;
}) {
  if (receiving) {
    return (
      <span className="block-stream-status receiving">
        <LiveSpinner title={receivingTitle} />
      </span>
    );
  }
  const label = formatStreamFinishedLabel(durationMs);
  if (!label) {
    return null;
  }
  return <span className="block-stream-status">{label}</span>;
}

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

function ContextCompactDialog({
  usage,
  busy,
  onClose,
  onConfirm
}: {
  usage?: ContextUsageState;
  busy: boolean;
  onClose: () => void;
  onConfirm: () => void;
}) {
  const used = usage ? compactNumber(usage.usedTokens) : "-";
  const max = usage?.maxContextTokens ? compactNumber(usage.maxContextTokens) : "-";
  const percent = usage?.ratio === undefined ? "n/a" : `${Math.round(usage.ratio * 100)}%`;
  const estimated = usage?.source === "estimate";

  return (
    <Modal
      title="手动压缩上下文"
      className="confirm-modal context-compact-modal"
      onClose={onClose}
      footer={
        <div className="modal-action-row">
          <button className="tool-button" disabled={busy} onClick={onClose}>取消</button>
          <button className="primary-button" disabled={busy} onClick={onConfirm}>
            {busy ? (
              <>
                <LoaderCircle className="inline-spinner" size={15} /> 压缩中
              </>
            ) : (
              <>
                <Brain size={15} /> 压缩
              </>
            )}
          </button>
        </div>
      }
    >
      <div className="confirm-content">
        <p>是否手动压缩当前上下文？</p>
        <dl className="context-compact-stats">
          <div>
            <dt>Context</dt>
            <dd>{estimated ? "~" : ""}{used}/{max}</dd>
          </div>
          <div>
            <dt>占用</dt>
            <dd>{percent}</dd>
          </div>
        </dl>
      </div>
    </Modal>
  );
}

function ModelDialog({ models, current, onClose, onSelect, onRefresh }: { models: string[]; current: string; onClose: () => void; onSelect: (model: string) => void; onRefresh: (provider: string) => Promise<void> }) {
  const [filter, setFilter] = useState("");
  const [refreshingProvider, setRefreshingProvider] = useState<string | null>(null);
  const [refreshError, setRefreshError] = useState<string | null>(null);
  const grouped = useMemo(() => {
    const groups = new Map<string, string[]>();
    for (const model of models.filter((item) => item.toLowerCase().includes(filter.toLowerCase()))) {
      const [provider] = model.split("/", 1);
      groups.set(provider, [...(groups.get(provider) ?? []), model]);
    }
    return [...groups.entries()];
  }, [models, filter]);

  async function refresh(provider: string) {
    if (refreshingProvider) return;
    setRefreshingProvider(provider);
    setRefreshError(null);
    try {
      await onRefresh(provider);
    } catch (error) {
      setRefreshError(formatDialogError(error));
    } finally {
      setRefreshingProvider(null);
    }
  }

  return (
    <Modal title="切换模型" className="model-modal" onClose={onClose}>
      <div className="modal-toolbar">
        <input className="search" value={filter} onChange={(event) => setFilter(event.target.value)} placeholder="搜索模型" />
      </div>
      {refreshError && <div className="model-refresh-error" role="alert">{refreshError}</div>}
      <div className="model-grid">
        {grouped.map(([provider, entries]) => (
          <section className="model-provider" key={provider}>
            <header className="model-provider-header">
              <h3>{provider}</h3>
              <button
                className="icon-button model-refresh"
                title={`刷新 ${provider} 模型列表`}
                disabled={Boolean(refreshingProvider)}
                onClick={() => void refresh(provider)}
              >
                <RotateCcw size={15} className={refreshingProvider === provider ? "spin" : ""} />
              </button>
            </header>
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

function formatDialogError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function PluginDialog({ catalog, selectedPlugins, pluginConfigs, onClose, onApply }: { catalog: PluginCatalogItem[]; selectedPlugins: string[]; pluginConfigs: Record<string, Record<string, unknown>>; onClose: () => void; onApply: (selected: string[], configs: Record<string, Record<string, unknown>>) => void }) {
  const initial = mergePluginDefaults(catalog, selectedPlugins, pluginConfigs);
  const [selected, setSelected] = useState(new Set(initial.selectedPlugins));
  const [configs, setConfigs] = useState(initial.pluginConfigs);
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const selectedCount = catalog.filter((item) => selected.has(item.key)).length;

  function apply() {
    const selectedList = catalog.filter((item) => selected.has(item.key)).map((item) => item.key);
    const nextState = mergePluginDefaults(catalog, selectedList, configs);
    const nextConfigs = nextState.pluginConfigs;
    const errors: Record<string, string> = {};
    for (const item of catalog) {
      if (!selected.has(item.key)) continue;
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
    onApply(nextState.selectedPlugins, nextConfigs);
  }

  function getFieldError(pluginKey: string, field: string): string | undefined {
    return fieldErrors[`${pluginKey}.${field}`];
  }

  function updateSelection(next: Set<string>) {
    setSelected(next);
    if (Object.keys(fieldErrors).length > 0) setFieldErrors({});
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
          <div className="plugin-action-row">
            <div className="plugin-bulk-actions">
              <button
                className="tool-button"
                disabled={catalog.length === 0}
                onClick={() => updateSelection(new Set(selectAllPluginKeys(catalog)))}
              >
                <Check size={15} /> 全选
              </button>
              <button
                className="tool-button"
                disabled={catalog.length === 0}
                onClick={() => updateSelection(new Set(invertPluginSelection(catalog, selected)))}
              >
                <RotateCcw size={15} /> 反选
              </button>
              <span className="plugin-selection-count">{selectedCount} / {catalog.length} 已选</span>
            </div>
            <div className="modal-action-row">
              <button className="tool-button" onClick={onClose}>取消</button>
              <button className="primary-button" onClick={apply}>应用</button>
            </div>
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
                  updateSelection(next);
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
  const modalRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    modalRef.current?.focus({ preventScroll: true });
  }, []);

  return (
    <div className="modal-backdrop">
      <section
        ref={modalRef}
        className={`modal ${className}`.trim()}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        tabIndex={-1}
      >
        <header>
          <h2>{title}</h2>
          <button className="icon-button" data-dialog-close="true" onClick={onClose} title="关闭"><X size={18} /></button>
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

function metaFrame(message: string): CoreFrame {
  return {
    version: VERSION,
    type: "debug.info",
    payload: { message }
  };
}

function framePayload(frame: CoreFrame | null): Record<string, unknown> | null {
  if (!frame || !frame.payload || typeof frame.payload !== "object" || Array.isArray(frame.payload)) {
    return null;
  }
  return frame.payload as Record<string, unknown>;
}

function normalizeMarkdownExportPayload(value: unknown): MarkdownExportPayload | null {
  if (!isRecord(value) || typeof value.markdown !== "string") {
    return null;
  }
  const references = Array.isArray(value.references)
    ? value.references
      .filter(isRecord)
      .map((item) => ({
        filename: String(item.filename ?? "reference.txt"),
        content: String(item.content ?? ""),
        mime_type: typeof item.mime_type === "string" ? item.mime_type : undefined
      }))
    : [];
  return {
    suggested_filename: String(value.suggested_filename ?? "hawi-export.md"),
    reference_dir_name: typeof value.reference_dir_name === "string" ? value.reference_dir_name : undefined,
    markdown: value.markdown,
    references
  };
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
        : [],
      locked: item.locked === true,
      lock_owner: isRecord(item.lock_owner) ? item.lock_owner : null,
      load_state: normalizeSessionLoadState(item.load_state),
      loaded_at: optionalPayloadNumber(item.loaded_at),
      last_finished_at: optionalPayloadNumber(item.last_finished_at),
      gui_launch_profile: normalizeLaunchProfile(item.gui_launch_profile)
    }))
    .filter((item) => item.session_id);
}

function frameSessionId(frame: CoreFrame): string | null {
  return framePayload(frame) ? optionalPayloadString(framePayload(frame)?.session_id) : null;
}

export function upsertSessionRuntime(
  sessions: SessionMetaPayload[],
  sessionId: string,
  patch: Pick<SessionMetaPayload, "load_state" | "loaded_at" | "last_finished_at">,
  options: { createIfMissing?: boolean } = {}
): SessionMetaPayload[] {
  const index = sessions.findIndex((session) => session.session_id === sessionId);
  if (index < 0) {
    if (!options.createIfMissing) {
      return sessions;
    }
    return [
      {
        session_id: sessionId,
        name: sessionId,
        created_at: patch.loaded_at ? new Date(patch.loaded_at).toISOString() : "",
        updated_at: patch.loaded_at ? new Date(patch.loaded_at).toISOString() : "",
        last_checkpoint_event: null,
        components_present: [],
        locked: false,
        lock_owner: null,
        ...patch
      },
      ...sessions
    ];
  }
  return sessions.map((session, itemIndex) => (
    itemIndex === index ? { ...session, ...patch, locked: patch.load_state === "unloaded" ? session.locked : false } : session
  ));
}

function countSessionsByState(sessions: SessionMetaPayload[], state: SessionLoadState): number {
  return sessions.filter((session) => session.load_state === state).length;
}

function normalizeSessionLoadState(value: unknown): SessionLoadState | undefined {
  if (value === "unloaded" || value === "loaded" || value === "running") {
    return value;
  }
  return undefined;
}

function normalizeLaunchProfile(value: unknown): SessionLaunchProfile | null {
  if (!isRecord(value)) return null;
  const modelName = optionalPayloadString(value.modelName);
  const systemPrompt = typeof value.systemPrompt === "string" ? value.systemPrompt : null;
  if (!modelName || systemPrompt === null) return null;
  return {
    version: 1,
    modelName,
    systemPrompt,
    selectedPlugins: Array.isArray(value.selectedPlugins)
      ? value.selectedPlugins.filter((item): item is string => typeof item === "string")
      : [],
    pluginConfigs: normalizePluginConfigs(value.pluginConfigs),
    engineArgs: Array.isArray(value.engineArgs)
      ? value.engineArgs.filter((item): item is string => typeof item === "string")
      : undefined
  };
}

function normalizePluginConfigs(value: unknown): Record<string, Record<string, unknown>> {
  if (!isRecord(value)) return {};
  return Object.fromEntries(
    Object.entries(value).map(([key, config]) => [
      key,
      isRecord(config) ? { ...config } : {}
    ])
  );
}

function configFromLaunchProfile(
  profile: SessionLaunchProfile,
  baseConfig: PersistedConfig
): PersistedConfig {
  return {
    ...baseConfig,
    modelName: profile.modelName,
    systemPrompt: profile.systemPrompt,
    selectedPlugins: [...profile.selectedPlugins],
    pluginConfigs: normalizePluginConfigs(profile.pluginConfigs)
  };
}

export function sortSessionsByCreatedAt(sessions: SessionMetaPayload[]): SessionMetaPayload[] {
  return [...sessions].sort((a, b) => {
    const left = Date.parse(a.created_at || a.updated_at || "");
    const right = Date.parse(b.created_at || b.updated_at || "");
    return (Number.isFinite(right) ? right : 0) - (Number.isFinite(left) ? left : 0);
  });
}

function optionalPayloadString(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed || null;
}

function optionalPayloadNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function shortSessionId(value: string): string {
  const timestamped = value.match(/^session-(\d{8}-\d{6})-[0-9a-f]{6}$/);
  if (timestamped) return timestamped[1];
  if (value.length <= 8) return value;
  return value.slice(0, 8);
}

function sessionDisplayName(session: SessionMetaPayload): string {
  if (!session.name || session.name === session.session_id) {
    return shortSessionId(session.session_id);
  }
  return session.name;
}

export function formatSessionTimestamp(value: string, now = Date.now()): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "unknown";
  const diffSeconds = Math.round((date.getTime() - now) / 1000);
  const absSeconds = Math.abs(diffSeconds);
  if (absSeconds < 60) return "刚刚";
  if (absSeconds < 3600) return formatRelativeTime(Math.round(diffSeconds / 60), "minute");
  if (absSeconds < 86_400) return formatRelativeTime(Math.round(diffSeconds / 3600), "hour");
  if (absSeconds < 172_800) return diffSeconds < 0 ? "昨天" : "明天";
  if (absSeconds < 604_800) return formatRelativeTime(Math.round(diffSeconds / 86_400), "day");
  return date.toLocaleDateString([], {
    month: "numeric",
    day: "numeric"
  });
}

function formatRelativeTime(value: number, unit: Intl.RelativeTimeFormatUnit): string {
  return new Intl.RelativeTimeFormat("zh-CN", { numeric: "auto" }).format(value, unit);
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

function toolPresentation(tool: ToolState): { label: string; detail?: string; detailKind: "purpose" | "path" } {
  const kind = filesystemToolKind(tool.name);
  const label = kind ? filesystemToolLabels[kind] : tool.name;
  const args = isRecord(tool.arguments) ? tool.arguments : undefined;
  const pathDetail = args
    ? optionalRecordString(args.file_path)
      ?? optionalRecordString(args.path)
      ?? optionalRecordString(args.directory)
      ?? optionalRecordString(args.pattern)
    : undefined;
  if (tool.description) {
    return { label, detail: tool.description, detailKind: "purpose" };
  }
  return { label, detail: pathDetail, detailKind: "path" };
}

const filesystemToolLabels: Record<string, string> = {
  read_file: "Read file",
  write_file: "Write file",
  edit_file: "Edit file",
  list_dir: "List directory",
  glob: "Glob",
  grep: "Grep",
  run_shell: "Shell"
};

function filesystemToolKind(name: string): string | null {
  const canonical = name.includes("__") ? name.slice(name.lastIndexOf("__") + 2) : name;
  return Object.prototype.hasOwnProperty.call(filesystemToolLabels, canonical)
    ? canonical
    : null;
}

function optionalRecordString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value : undefined;
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}

function formatCount(value: unknown, singular: string, plural: string): string {
  const count = typeof value === "number" && Number.isFinite(value) ? value : undefined;
  if (count === undefined) return "";
  return `${count} ${count === 1 ? singular : plural}`;
}

function formatTextStats(value: string): string {
  const lineCount = value ? value.split("\n").length : 0;
  const charCount = Array.from(value).length;
  return `${lineCount} ${lineCount === 1 ? "line" : "lines"} · ${charCount} chars`;
}

function clampedScrollOffset(sourceOffset: number, targetMax: number): number {
  if (targetMax <= 0) return 0;
  return Math.min(sourceOffset, targetMax);
}

function normalizeReadFileContent(content: string, language?: string): { content: string; language?: string } {
  let normalized = stripReadFileHeader(content);
  const fenced = unwrapSingleFencedCodeBlock(normalized);
  if (fenced) {
    normalized = fenced.content;
  }
  return {
    content: normalized,
    language: fenced?.language ?? language
  };
}

function stripReadFileHeader(content: string): string {
  return content.replace(/^\[(?:Lines \d+-\d+ of \d+|language: [^\]\n]+)(?: \| (?:Lines \d+-\d+ of \d+|language: [^\]\n]+))*\]\n/, "");
}

function unwrapSingleFencedCodeBlock(content: string): { language?: string; content: string } | null {
  const match = content.match(/^```([A-Za-z0-9_+.-]*)\n([\s\S]*?)\n?```\s*$/);
  if (!match) return null;
  return {
    language: match[1] || undefined,
    content: match[2]
  };
}

function detectLanguageFromPath(filePath?: string): string | undefined {
  if (!filePath) return undefined;
  const filename = filePath.split(/[\\/]/).pop()?.toLowerCase() ?? "";
  if (!filename) return undefined;
  if (filename === "dockerfile" || filename.startsWith("dockerfile.")) return "dockerfile";
  if (filename === "makefile") return "makefile";
  if (filename.endsWith(".d.ts")) return "typescript";
  const extension = filename.includes(".") ? filename.slice(filename.lastIndexOf(".")) : "";
  return extensionLanguageMap[extension];
}

const extensionLanguageMap: Record<string, string> = {
  ".bash": "bash",
  ".css": "css",
  ".js": "javascript",
  ".json": "json",
  ".jsonc": "json",
  ".jsx": "javascript",
  ".md": "markdown",
  ".py": "python",
  ".sh": "bash",
  ".ts": "typescript",
  ".tsx": "typescript",
  ".xml": "xml",
  ".yaml": "yaml",
  ".yml": "yaml",
  ".zsh": "bash"
};

function isConversationNode(node: ChatNode): boolean {
  return node.kind === "user"
    || node.kind === "agent"
    || node.kind === "thinking"
    || node.kind === "tool"
    || node.kind === "divider"
    || node.kind === "compact";
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

export function shouldBubbleNestedVerticalScroll(
  element: Pick<HTMLElement, "scrollHeight" | "scrollTop" | "clientHeight">,
  deltaY: number
): boolean {
  if (deltaY === 0 || element.scrollHeight <= element.clientHeight) return true;
  const maxScrollTop = Math.max(0, element.scrollHeight - element.clientHeight);
  if (deltaY < 0) return element.scrollTop <= Math.abs(deltaY);
  return maxScrollTop - element.scrollTop <= deltaY + 1;
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

export function formatStreamFinishedLabel(durationMs?: number): string | null {
  if (durationMs === undefined || !Number.isFinite(durationMs) || durationMs < 0) {
    return null;
  }
  return `finished in ${(durationMs / 1000).toFixed(1)}s`;
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

function highlightedCode(value: string, language?: string): { html: string; language?: string } {
  const normalizedLanguage = language ? normalizeHighlightLanguage(language) : null;
  if (normalizedLanguage) {
    const result = hljs.highlight(value, { language: normalizedLanguage, ignoreIllegals: true });
    return { html: result.value, language: normalizedLanguage };
  }
  return { html: escapeHtml(value) };
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

function handleNestedVerticalScroll(event: WheelEvent<HTMLElement>) {
  if (!shouldBubbleNestedVerticalScroll(event.currentTarget, event.deltaY)) {
    event.stopPropagation();
  }
}
