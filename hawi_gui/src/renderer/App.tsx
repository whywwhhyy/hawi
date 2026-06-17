import { forwardRef, memo, useCallback, useEffect, useLayoutEffect, useMemo, useReducer, useRef, useState, type CSSProperties, type ChangeEvent as ReactChangeEvent, type ClipboardEvent as ReactClipboardEvent, type DependencyList, type DragEvent as ReactDragEvent, type KeyboardEvent as ReactKeyboardEvent, type MouseEvent as ReactMouseEvent, type PointerEvent as ReactPointerEvent, type ReactNode, type Ref, type UIEvent as ReactUIEvent, type WheelEvent } from "react";
import { createPortal } from "react-dom";
import MarkdownIt from "markdown-it";
import katex from "katex";
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
import { Activity, ArrowDown, ArrowLeftRight, ArrowUp, Bot, Brain, Check, ChevronDown, ChevronRight, ChevronsUp, Copy, FileText, GitFork, Image as ImageIcon, LoaderCircle, Lock, MessageSquare, Paperclip, Pencil, Play, Plug, Plus, RotateCcw, Search, Send, Settings, Square, Trash2, Wrench, X } from "lucide-react";
import type { BlobSource, ContentPart, CoreCommandType, CoreFrame, GuiMetadata, JsonSchemaObject, JsonlExportPayload, MarkdownExportPayload, MediaSource, ModelProviderConfigPreview, PersistedConfig, PluginCatalogItem, PluginToolPreviewItem, PluginToolPreviewPayload, QueueKind, RuntimeControlState, SessionLaunchProfile, SessionLoadState, SessionMetaPayload } from "../shared/protocol";
import { VERSION } from "../shared/protocol";
import { MIN_CONTENT_SIZE, normalizeMinimumContentSize, type LayoutSize } from "../shared/layout";
import { StatusCell, StatusCellDisplay, StatusCellTrigger, StatusPopoverHeader } from "./StatusCell";
import { coerceSchemaValue, mergePluginDefaults, resolvePluginSelectionChange, selectAllPluginKeys, validatePluginConfig } from "./pluginConfig";
import { chatNodesFromMessageHistory, createInitialState, reduceCoreEvent, type AppState, type ChatNode, type ContextAutoCompactState, type ContextCompressionState, type ContextUsageState, type FrameworkInjectionState, type ModelProfileState, type ModelUsageState, type PluginArtifactState, type PluginMessageState, type PluginStatusState, type ProcessingState, type QueueMessageState, type SideThreadState, type SubAgentRuntimeState, type ToolProgressState, type ToolState } from "./state";

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
  svg: "xml",
  sh: "bash",
  shell: "bash",
  ts: "typescript",
  tsx: "typescript",
  yml: "yaml",
  zsh: "bash"
};

const markdown = new MarkdownIt({
  html: true,
  linkify: true,
  breaks: true,
  highlight: highlightCode
});

interface MarkdownRenderEnv {
  latexHtml?: string[];
}

function inlineLatexRule(state: MarkdownIt.StateInline, silent: boolean): boolean {
  if (state.src.charCodeAt(state.pos) !== 0x24 /* $ */) return false;
  if (state.src.charCodeAt(state.pos + 1) === 0x24 /* $ */) return false;

  const close = findInlineLatexCloseDelimiter(state.src, state.pos + 1);
  if (close < 0) return false;

  const content = state.src.slice(state.pos + 1, close);
  if (!isInlineLatexContent(content)) return false;

  if (!silent) {
    const token = state.push("math_inline", "math", 0);
    token.content = content.trim();
    token.markup = "$";
  }
  state.pos = close + 1;
  return true;
}

function findInlineLatexCloseDelimiter(value: string, from: number): number {
  for (let index = from; index < value.length; index += 1) {
    if (value.charCodeAt(index) !== 0x24 /* $ */) continue;
    if (isEscapedMarkdownDelimiter(value, index)) continue;
    return index;
  }
  return -1;
}

function isInlineLatexContent(value: string): boolean {
  if (!value.trim()) return false;
  if (/^\s|\s$/.test(value)) return false;
  return !/[\r\n]/.test(value);
}

function isEscapedMarkdownDelimiter(value: string, index: number): boolean {
  let slashCount = 0;
  for (let pos = index - 1; pos >= 0 && value.charCodeAt(pos) === 0x5c /* \ */; pos -= 1) {
    slashCount += 1;
  }
  return slashCount % 2 === 1;
}

function blockLatexRule(state: MarkdownIt.StateBlock, startLine: number, endLine: number, silent: boolean): boolean {
  let startPos = state.bMarks[startLine] + state.tShift[startLine];
  const startMax = state.eMarks[startLine];
  if (startPos + 2 > startMax) return false;
  if (state.src.slice(startPos, startPos + 2) !== "$$") return false;

  startPos += 2;
  const firstLine = state.src.slice(startPos, startMax);
  const firstLineClose = findBlockLatexCloseDelimiter(firstLine);
  let content = "";
  let nextLine = startLine;

  if (firstLineClose >= 0) {
    content = firstLine.slice(0, firstLineClose);
  } else {
    content = `${firstLine}\n`;
    let found = false;
    for (nextLine = startLine + 1; nextLine < endLine; nextLine += 1) {
      const line = state.getLines(nextLine, nextLine + 1, state.blkIndent, false);
      const close = findBlockLatexCloseDelimiter(line);
      if (close >= 0) {
        content += line.slice(0, close);
        found = true;
        break;
      }
      content += `${line}\n`;
    }
    if (!found) return false;
  }

  if (!content.trim()) return false;
  if (silent) return true;

  const token = state.push("math_block", "math", 0);
  token.block = true;
  token.content = content.trim();
  token.markup = "$$";
  token.map = [startLine, nextLine + 1];
  state.line = nextLine + 1;
  return true;
}

function findBlockLatexCloseDelimiter(value: string): number {
  for (let index = 0; index < value.length - 1; index += 1) {
    if (value.slice(index, index + 2) !== "$$") continue;
    if (isEscapedMarkdownDelimiter(value, index)) continue;
    return index;
  }
  return -1;
}

function renderLatexPlaceholder(value: string, displayMode: boolean, env: unknown): string {
  const renderEnv = env as MarkdownRenderEnv;
  if (!renderEnv.latexHtml) {
    renderEnv.latexHtml = [];
  }
  const index = renderEnv.latexHtml.length;
  renderEnv.latexHtml.push(renderLatex(value, displayMode));
  const tag = displayMode ? "div" : "span";
  return `<${tag} data-hawi-latex-placeholder="${index}"></${tag}>`;
}

function renderLatex(value: string, displayMode: boolean): string {
  try {
    return katex.renderToString(value, {
      displayMode,
      throwOnError: false,
      trust: false,
      strict: "ignore"
    });
  } catch {
    return `<code>${escapeHtml(value)}</code>`;
  }
}

function restoreLatexPlaceholders(value: string, latexHtml: string[]): string {
  if (latexHtml.length === 0) return value;
  return value.replace(
    /<(span|div)\b[^>]*\sdata-hawi-latex-placeholder="(\d+)"[^>]*>\s*<\/\1>/g,
    (_match: string, _tag: string, indexText: string) => latexHtml[Number(indexText)] ?? ""
  );
}

markdown.inline.ruler.before("escape", "hawi_inline_latex", inlineLatexRule);
markdown.block.ruler.before("fence", "hawi_block_latex", blockLatexRule, {
  alt: ["paragraph", "reference", "blockquote", "list"]
});
markdown.renderer.rules.math_inline = (tokens, idx, _options, env) => renderLatexPlaceholder(tokens[idx].content, false, env);
markdown.renderer.rules.math_block = (tokens, idx, _options, env) => renderLatexPlaceholder(tokens[idx].content, true, env);
const defaultLinkOpen = markdown.renderer.rules.link_open
  ?? ((tokens, idx, options, _env, self) => self.renderToken(tokens, idx, options));
markdown.renderer.rules.link_open = (tokens, idx, options, env, self) => {
  tokens[idx].attrSet("target", "_blank");
  tokens[idx].attrSet("rel", "noopener noreferrer");
  return defaultLinkOpen(tokens, idx, options, env, self);
};
markdown.renderer.rules.fence = (tokens, idx) => {
  const token = tokens[idx];
  const language = token.info.trim().split(/\s+/)[0]?.toLowerCase();
  if (language === "mermaid") {
    return renderMermaidFence(token.content);
  }
  if (language === "svg") {
    return renderSvgFence(token.content);
  }
  const highlighted = highlightedCode(token.content, language);
  return `${codeBlock(highlighted.html, highlighted.language)}\n`;
};
const AUTO_SCROLL_BOTTOM_THRESHOLD_PX = 5;
const AUTO_SCROLL_SETTLE_FRAMES = 2;
const COPY_FEEDBACK_MS = 1200;
const RAIL_SETTINGS_MENU_GAP_PX = 10;
const RAIL_SETTINGS_MENU_MARGIN_PX = 10;
const RAIL_SETTINGS_MENU_MAX_WIDTH_PX = 520;
const RAIL_SETTINGS_MENU_MIN_WIDTH_PX = 220;
const RAIL_EXPORT_MENU_MAX_WIDTH_PX = 240;
const RAIL_EXPORT_MENU_MIN_WIDTH_PX = 180;
const SYSTEM_PROMPT_MAX_ROWS = 8;
const MESSAGE_INPUT_MAX_ROWS = 5;
const MAX_INPUT_HISTORY = 100;
const UNLOADED_INPUT_HISTORY_KEY = "__unloaded__";
const BLOB_CHUNK_SIZE = 256 * 1024;
const MAX_ATTACHMENTS = 8;
const MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024;
const BLOB_PREVIEW_FETCH_TIMEOUT_MS = 20_000;
const STATUS_OVERLAY_SELECTOR = [
  ".context-popover",
  ".project-popover",
  ".session-popover",
  ".queue-popover",
  ".menu-popover"
].join(",");
const STATUS_PROJECT_MIN_WIDTH_PX = 250;
const STATUS_SESSION_MIN_WIDTH_PX = 292;
const STATUS_CONTEXT_MIN_WIDTH_PX = 180;
const STATUS_MESSAGE_WIDTH_PX = 168;
const STATUS_POPOVER_VIEWPORT_MARGIN = 8;
const STATUS_POPOVER_GAP_PX = 6;
const PAGE_FIND_MATCH_HIGHLIGHT = "hawi-page-find-match";
const PAGE_FIND_ACTIVE_HIGHLIGHT = "hawi-page-find-active";
const HISTORY_FIND_MATCH_HIGHLIGHT = "hawi-history-find-match";
const HISTORY_FIND_ACTIVE_HIGHLIGHT = "hawi-history-find-active";
const PAGE_FIND_EXCLUDE_SELECTOR = [
  ".page-find-overlay",
  "script",
  "style",
  "noscript",
  "input",
  "textarea",
  "select",
  "option",
  "button",
  "svg",
  "[aria-hidden='true']",
  ".message-body.is-collapsed",
  ".collapsible-body.is-collapsed"
].join(",");
const imageAttachmentExtensions = new Set(["avif", "bmp", "gif", "jpg", "jpeg", "png", "svg", "webp"]);
const audioAttachmentExtensions = new Set(["aac", "flac", "m4a", "mp3", "oga", "ogg", "opus", "wav", "weba"]);
const videoAttachmentExtensions = new Set(["avi", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ogv", "webm"]);
const documentAttachmentExtensions = new Set([
  "csv",
  "doc",
  "docx",
  "html",
  "json",
  "md",
  "pdf",
  "ppt",
  "pptx",
  "rtf",
  "toml",
  "tsv",
  "txt",
  "xls",
  "xlsx",
  "xml",
  "yaml",
  "yml",
]);
const documentAttachmentMimeTypes = new Set([
  "application/json",
  "application/msword",
  "application/pdf",
  "application/rtf",
  "application/vnd.ms-excel",
  "application/vnd.ms-powerpoint",
  "application/vnd.openxmlformats-officedocument.presentationml.presentation",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "application/xml",
]);

function waitForNextPaint(): Promise<void> {
  if (typeof window === "undefined" || typeof window.requestAnimationFrame !== "function") {
    return Promise.resolve();
  }
  return new Promise((resolve) => window.requestAnimationFrame(() => resolve()));
}
const extensionMimeTypes: Record<string, string> = {
  aac: "audio/aac",
  avif: "image/avif",
  bmp: "image/bmp",
  csv: "text/csv",
  flac: "audio/flac",
  gif: "image/gif",
  html: "text/html",
  jpeg: "image/jpeg",
  jpg: "image/jpeg",
  json: "application/json",
  m4a: "audio/mp4",
  m4v: "video/mp4",
  md: "text/markdown",
  mov: "video/quicktime",
  mp3: "audio/mpeg",
  mp4: "video/mp4",
  ogg: "audio/ogg",
  pdf: "application/pdf",
  png: "image/png",
  rtf: "application/rtf",
  svg: "image/svg+xml",
  toml: "application/toml",
  tsv: "text/tab-separated-values",
  txt: "text/plain",
  wav: "audio/wav",
  webm: "video/webm",
  webp: "image/webp",
  xml: "application/xml",
  yaml: "application/yaml",
  yml: "application/yaml",
  zip: "application/zip",
};
const useBrowserLayoutEffect = typeof window === "undefined" ? useEffect : useLayoutEffect;
const markdownCodeCopyTimers = new WeakMap<HTMLButtonElement, number>();
let mermaidRenderSequence = 0;
let mermaidModulePromise: Promise<typeof import("mermaid")> | null = null;
let mermaidRenderQueue: Promise<void> = Promise.resolve();
const MERMAID_RENDER_TIMEOUT_MS = 10_000;
const MERMAID_RENDER_RETRY_DELAYS_MS = [80, 320, 1200];
export const MERMAID_RENDER_CONFIG = {
  startOnLoad: false,
  securityLevel: "strict",
  theme: "default",
  flowchart: {
    htmlLabels: false,
  },
} as const;

const queueLabels: Record<QueueKind, string> = {
  normal: "稍后任务",
  high_prio: "待送达插话",
  urgent: "紧急"
};

const userMessageTypeLabels = {
  normal: "普通消息",
  steer: "Steer",
  urgent: "紧急消息",
  resume: "Resume"
} as const;

interface HistorySearchResult {
  sessionId: string;
  sessionName: string;
  sessionCreatedAt: string;
  sessionUpdatedAt: string;
  messageIndex: number;
  contextMessageId?: string;
  contextMessageIndex?: number;
  runId?: string;
  role: string;
  timestamp?: number | string;
  text: string;
  snippet: string;
  lastCwd?: string | null;
}

export interface HistoryLocateTarget {
  sessionId: string;
  messageIndex?: number;
  contextMessageId?: string;
  contextMessageIndex?: number;
}

export interface SideThreadQuotedRange {
  start: number;
  end: number;
}

export interface SideThreadWindowRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

interface SideSelectionAnchor {
  text: string;
  quotedRange?: SideThreadQuotedRange;
  rect: DOMRect;
  target: HistoryLocateTarget;
}

interface SideThreadPopoverState {
  mode: "compose" | "thread";
  threadId?: string;
  anchor: SideSelectionAnchor;
  windowRect: SideThreadWindowRect;
}

type PendingAttachmentStatus = "ready" | "uploading" | "uploaded" | "error";
type AttachmentPartType = "image" | "document" | "audio" | "video" | "file";

interface PendingAttachment {
  id: string;
  file: File;
  previewUrl: string;
  dataUrl?: string;
  filename: string;
  mimeType: string;
  partType: AttachmentPartType;
  size: number;
  status: PendingAttachmentStatus;
  progress: number;
  error?: string;
  blobSource?: BlobSource;
}

interface ExistingEditableAttachment {
  id: string;
  kind: "existing";
  part: ContentPart;
  source?: MediaSource;
  filename: string;
  mimeType: string;
  partType: AttachmentPartType;
  size?: number;
}

interface PendingEditableAttachment {
  id: string;
  kind: "pending";
  attachment: PendingAttachment;
}

type EditableAttachment = ExistingEditableAttachment | PendingEditableAttachment;

interface MessageEditState {
  nodeId: string;
  role: "user" | "agent";
  contextMessageId?: string;
  contextMessageIndex?: number;
  draft: string;
  attachments: EditableAttachment[];
  busy: boolean;
  error: string | null;
}

interface MessageEditController {
  active: MessageEditState | null;
  canEdit: boolean;
  maxHeight?: number;
  onStart: (node: ChatNode) => void;
  onCancel: () => void;
  onSave: () => void;
  onResend: () => void;
  onDraftChange: (value: string) => void;
  onAddFiles: (files: Iterable<File>) => void;
  onRemoveAttachment: (id: string) => void;
}

interface BlobFinalizePayload {
  blob_id: string;
  uri?: string;
  sha256?: string;
  direction?: "inbound" | "outbound";
  size?: number;
  mime?: string | null;
  ref_count?: number;
}

type BlobPreviewUrls = Record<string, string>;

interface MediaPreviewState {
  src: string;
  label: string;
  meta?: string;
  kind: string;
}

interface BlobPreviewRequest {
  blobId: string;
  uri?: string;
  mimeType?: string;
  size?: number;
}

interface RailSettingsMenuLayoutInput {
  anchorTop: number;
  anchorRight: number;
  menuHeight: number;
  viewportWidth: number;
  viewportHeight: number;
}

interface RailPopoverLayoutInput extends RailSettingsMenuLayoutInput {
  minWidth: number;
  maxWidth: number;
}

interface StatusPopoverRect {
  top: number;
  right: number;
  bottom: number;
  left: number;
  width: number;
  height: number;
}

interface StatusPopoverLayoutInput {
  anchorRect: StatusPopoverRect;
  popoverSize: {
    width: number;
    height: number;
  };
  viewportWidth: number;
  viewportHeight: number;
  margin?: number;
  gap?: number;
}

interface StatusPopoverLayout {
  top: number;
  left: number;
  maxWidth: number;
  maxHeight: number;
}

interface StatusMainColumnLayoutInput {
  containerWidth: number;
  projectMinWidth?: number;
  sessionMinWidth?: number;
  contextMinWidth?: number;
  messageWidth?: number;
}

interface StatusMainColumnLayout {
  project: number;
  session: number;
  context: number;
}

type StatusPopoverAnchorRef = {
  current: HTMLElement | null;
};

interface PendingBlobPreviewFetch {
  chunks: string[];
  timeoutId: number;
  resolve: (dataB64: string) => void;
  reject: (error: Error) => void;
}

export function renderQueueStatusText(
  queueLengths: Record<QueueKind, number>,
  queueMessages?: Record<QueueKind, QueueMessageState[]>
): string {
  const highPriorityCount = hasHighPriorityWork(queueLengths, queueMessages) ? 1 : 0;
  return `Message Insert ${highPriorityCount} · Queue ${normalQueueCount(queueLengths, queueMessages)}`;
}

export const renderPriorityStatusText = renderQueueStatusText;

export function canStopRunnerState(runnerState: string): boolean {
  return runnerState === "RUNNING" || runnerState === "INTERRUPTING";
}

export function renderUsageStatusText(usage?: ModelUsageState): string {
  const total = formatUsageTokenCount(usage?.totalTokens ?? 0);
  const input = formatUsageTokenCount(usage?.inputTokens ?? 0);
  const output = formatUsageTokenCount(usage?.outputTokens ?? 0);
  const cacheRead = formatUsageTokenCount(usage?.cacheReadTokens ?? 0);
  const cacheWrite = formatUsageTokenCount(usage?.cacheWriteTokens ?? 0);
  return `Usage Total ${total} · Input ${input} · Output ${output} · Cache Write ${cacheWrite} · Cache Read ${cacheRead}`;
}

export function shouldInitializeSessionState(metadata: GuiMetadata | null): boolean {
  return Boolean(metadata?.coreRunning);
}

interface SessionRuntimeStats {
  running: number;
  loaded: number;
  maxLoaded: number;
}

interface CacheSensitiveSettingAcknowledgement {
  sessionId: string | null;
  messageCount: number;
}

interface PendingPluginApply {
  selectedPlugins: string[];
  pluginConfigs: Record<string, Record<string, unknown>>;
}

type EscapeDismissTarget =
  | "mediaPreview"
  | "sideThreadPopover"
  | "subagentObserver"
  | "projectPopover"
  | "contextPopover"
  | "messageEdit"
  | "pluginDialog"
  | "modelDialog"
  | "systemPromptConfirm"
  | "pluginSettingsConfirm"
  | "toolCallPurposeConfirm"
  | "exportMenu"
  | "settingsMenu"
  | "queueTaskEdit"
  | "queuePopover"
  | "sessionDialog";

interface EscapeDismissState {
  mediaPreviewOpen: boolean;
  sideThreadPopoverOpen: boolean;
  contextPopoverOpen: boolean;
  projectPopoverOpen: boolean;
  pluginDialogOpen: boolean;
  modelDialogOpen: boolean;
  systemPromptConfirmOpen: boolean;
  pluginSettingsConfirmOpen: boolean;
  toolCallPurposeConfirmOpen: boolean;
  exportMenuOpen: boolean;
  subagentObserverOpen: boolean;
  settingsMenuOpen: boolean;
  queuePopoverOpen: boolean;
  editingQueueTaskId: string | null;
  sessionDialogOpen: boolean;
  editingMessageId: string | null;
}

export function resolveEscapeDismissTarget(state: EscapeDismissState): EscapeDismissTarget | null {
  if (state.mediaPreviewOpen) return "mediaPreview";
  if (state.sideThreadPopoverOpen) return "sideThreadPopover";
  if (state.subagentObserverOpen) return "subagentObserver";
  if (state.pluginDialogOpen) return "pluginDialog";
  if (state.modelDialogOpen) return "modelDialog";
  if (state.systemPromptConfirmOpen) return "systemPromptConfirm";
  if (state.pluginSettingsConfirmOpen) return "pluginSettingsConfirm";
  if (state.toolCallPurposeConfirmOpen) return "toolCallPurposeConfirm";
  if (state.exportMenuOpen) return "exportMenu";
  if (state.settingsMenuOpen) return "settingsMenu";
  if (state.projectPopoverOpen) return "projectPopover";
  if (state.contextPopoverOpen) return "contextPopover";
  if (state.queuePopoverOpen) {
    return state.editingQueueTaskId ? "queueTaskEdit" : "queuePopover";
  }
  if (state.sessionDialogOpen) return "sessionDialog";
  if (state.editingMessageId) return "messageEdit";
  return null;
}

function resolveKeyboardScopeTarget(state: EscapeDismissState): EscapeDismissTarget | null {
  if (state.sideThreadPopoverOpen) return "sideThreadPopover";
  if (state.subagentObserverOpen) return "subagentObserver";
  if (state.pluginDialogOpen) return "pluginDialog";
  if (state.modelDialogOpen) return "modelDialog";
  if (state.systemPromptConfirmOpen) return "systemPromptConfirm";
  if (state.pluginSettingsConfirmOpen) return "pluginSettingsConfirm";
  if (state.toolCallPurposeConfirmOpen) return "toolCallPurposeConfirm";
  if (state.exportMenuOpen) return "exportMenu";
  if (state.settingsMenuOpen) return "settingsMenu";
  if (state.projectPopoverOpen) return "projectPopover";
  if (state.contextPopoverOpen) return "contextPopover";
  if (state.queuePopoverOpen) return "queuePopover";
  if (state.sessionDialogOpen) return "sessionDialog";
  return null;
}

function dialogScopeSelector(target: EscapeDismissTarget): string | null {
  switch (target) {
    case "contextPopover":
      return ".context-popover";
    case "mediaPreview":
      return ".media-preview-lightbox";
    case "sideThreadPopover":
      return ".side-thread-popover";
    case "projectPopover":
      return ".project-popover";
    case "subagentObserver":
      return ".subagent-modal";
    case "pluginDialog":
      return ".plugin-modal";
    case "modelDialog":
      return ".model-modal";
    case "systemPromptConfirm":
      return ".system-prompt-confirm-modal";
    case "pluginSettingsConfirm":
      return ".plugin-settings-confirm-modal";
    case "toolCallPurposeConfirm":
      return ".tool-call-purpose-confirm-modal";
    case "exportMenu":
      return ".rail-export-menu";
    case "settingsMenu":
      return ".rail-settings-menu";
    case "queueTaskEdit":
    case "queuePopover":
      return ".queue-popover";
    case "sessionDialog":
      return ".session-popover";
    case "messageEdit":
      return null;
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
  return target instanceof HTMLTextAreaElement ||
    target instanceof HTMLInputElement ||
    (target instanceof HTMLElement && target.isContentEditable);
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
      case "contextPopover":
      case "projectPopover":
      case "sideThreadPopover":
        return scope.querySelector<HTMLElement>(".primary-button:not(:disabled)");
      case "pluginDialog":
      case "systemPromptConfirm":
      case "pluginSettingsConfirm":
      case "toolCallPurposeConfirm":
        return scope.querySelector<HTMLElement>(".modal-actions .primary-button:not(:disabled)");
      case "modelDialog":
        return scope.querySelector<HTMLElement>(".model.active:not(:disabled)")
          ?? scope.querySelector<HTMLElement>(".model:not(:disabled)");
      case "sessionDialog":
        return scope.querySelector<HTMLElement>(".session-option.current .session-title-button:not(:disabled)")
          ?? scope.querySelector<HTMLElement>(".session-title-button:not(:disabled)");
      case "exportMenu":
      case "settingsMenu":
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

interface ArtifactTypeGroup {
  type: string;
  label: string;
  artifacts: PluginArtifactState[];
}

type WorkspaceSidebarTab = "side_threads" | "subagents" | `artifact:${string}`;

function artifactSidebarTab(type: string): WorkspaceSidebarTab {
  return `artifact:${type}`;
}

function artifactTypeFromSidebarTab(tab: WorkspaceSidebarTab | null): string | null {
  if (!tab?.startsWith("artifact:")) return null;
  return tab.slice("artifact:".length);
}

export function artifactTypeLabel(type: string): string {
  const normalized = type.trim();
  if (!normalized) return "Artifact";
  return normalized
    .replace(/[-_]+/g, " ")
    .replace(/\w\S*/g, (part) => part.charAt(0).toUpperCase() + part.slice(1));
}

export function groupArtifactsByType(artifacts: PluginArtifactState[]): ArtifactTypeGroup[] {
  const groups = new Map<string, ArtifactTypeGroup>();
  for (const artifact of artifacts) {
    const type = artifact.artifactType || "artifact";
    const existing = groups.get(type);
    if (existing) {
      existing.artifacts.push(artifact);
      continue;
    }
    groups.set(type, {
      type,
      label: artifactTypeLabel(type),
      artifacts: [artifact]
    });
  }
  return [...groups.values()];
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

function measureMinimumContentSize(appShell: HTMLElement, brandBar: HTMLElement): LayoutSize {
  const brandWidth = measureInlineGroupWidth(brandBar);
  return normalizeMinimumContentSize({
    width: brandWidth + horizontalChromeWidth(appShell),
    height: MIN_CONTENT_SIZE.height
  });
}

function measureInlineGroupWidth(element: HTMLElement): number {
  const children = visibleElementChildren(element);
  if (children.length === 0) {
    return elementWidth(element);
  }
  const gap = flexColumnGap(element);
  const childrenWidth = children.reduce((sum, child) => sum + elementWidth(child), 0);
  return Math.ceil(Math.max(
    elementWidth(element),
    childrenWidth + gap * Math.max(0, children.length - 1) + horizontalChromeWidth(element)
  ));
}

function visibleElementChildren(element: HTMLElement): HTMLElement[] {
  return Array.from(element.children).filter((child): child is HTMLElement => {
    if (!(child instanceof HTMLElement)) return false;
    const style = getComputedStyle(child);
    return style.display !== "none" && style.visibility !== "collapse";
  });
}

function elementWidth(element: HTMLElement): number {
  const rectWidth = element.getBoundingClientRect().width;
  const scrollWidth = containsStatusOverlay(element)
    ? 0
    : element.scrollWidth + horizontalBorderWidth(element);
  const offsetWidth = element.offsetWidth;
  return Math.max(
    Number.isFinite(rectWidth) ? rectWidth : 0,
    Number.isFinite(scrollWidth) ? scrollWidth : 0,
    Number.isFinite(offsetWidth) ? offsetWidth : 0
  );
}

function containsStatusOverlay(element: HTMLElement): boolean {
  return element.querySelector(STATUS_OVERLAY_SELECTOR) !== null;
}

function flexColumnGap(element: HTMLElement): number {
  const style = getComputedStyle(element);
  return cssPixelValue(style.columnGap) || cssPixelValue(style.gap);
}

function horizontalChromeWidth(element: HTMLElement): number {
  const style = getComputedStyle(element);
  return cssPixelValue(style.paddingLeft)
    + cssPixelValue(style.paddingRight)
    + cssPixelValue(style.borderLeftWidth)
    + cssPixelValue(style.borderRightWidth);
}

function horizontalBorderWidth(element: HTMLElement): number {
  const style = getComputedStyle(element);
  return cssPixelValue(style.borderLeftWidth) + cssPixelValue(style.borderRightWidth);
}

function cssPixelValue(value: string): number {
  if (!value || value === "normal") return 0;
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function useStatusMainColumnLayout(stripRef: { current: HTMLElement | null }) {
  useBrowserLayoutEffect(() => {
    if (typeof window === "undefined") return;
    const strip = stripRef.current;
    if (!strip) return;

    let animationFrame: number | null = null;
    const updateLayout = () => {
      const contentWidth = strip.clientWidth || Math.max(0, strip.getBoundingClientRect().width - horizontalBorderWidth(strip));
      const columns = resolveStatusMainColumnLayout({ containerWidth: contentWidth });
      strip.style.setProperty("--status-project-width", `${columns.project}px`);
      strip.style.setProperty("--status-session-width", `${columns.session}px`);
      strip.style.setProperty("--status-context-width", `${columns.context}px`);
    };
    const scheduleLayout = () => {
      if (typeof window.requestAnimationFrame !== "function") {
        updateLayout();
        return;
      }
      if (animationFrame !== null) return;
      animationFrame = window.requestAnimationFrame(() => {
        animationFrame = null;
        updateLayout();
      });
    };

    scheduleLayout();
    const observer = typeof ResizeObserver === "undefined"
      ? null
      : new ResizeObserver(scheduleLayout);
    observer?.observe(strip);
    window.addEventListener("resize", scheduleLayout);
    return () => {
      if (animationFrame !== null && typeof window.cancelAnimationFrame === "function") {
        window.cancelAnimationFrame(animationFrame);
      }
      observer?.disconnect();
      window.removeEventListener("resize", scheduleLayout);
    };
  }, [stripRef]);
}

export function resolveStatusMainColumnLayout({
  containerWidth,
  projectMinWidth = STATUS_PROJECT_MIN_WIDTH_PX,
  sessionMinWidth = STATUS_SESSION_MIN_WIDTH_PX,
  contextMinWidth = STATUS_CONTEXT_MIN_WIDTH_PX,
  messageWidth = STATUS_MESSAGE_WIDTH_PX
}: StatusMainColumnLayoutInput): StatusMainColumnLayout {
  const minimums = [
    nonNegative(projectMinWidth),
    nonNegative(sessionMinWidth),
    nonNegative(contextMinWidth)
  ];
  const mainMinimumTotal = minimums.reduce((sum, width) => sum + width, 0);
  const availableMainWidth = Math.max(
    mainMinimumTotal,
    nonNegative(containerWidth) - nonNegative(messageWidth)
  );
  const widths = distributeExtraToShortestColumns(
    minimums,
    availableMainWidth - mainMinimumTotal
  );

  return {
    project: roundLayoutPixel(widths[0]),
    session: roundLayoutPixel(widths[1]),
    context: roundLayoutPixel(widths[2])
  };
}

function distributeExtraToShortestColumns(minWidths: number[], extraWidth: number): number[] {
  const columns = minWidths.map((width, index) => ({ index, width }));
  let remaining = Math.max(0, extraWidth);
  const epsilon = 0.001;

  while (remaining > epsilon) {
    columns.sort((a, b) => a.width === b.width ? a.index - b.index : a.width - b.width);
    const shortestWidth = columns[0].width;
    const shortestColumns = columns.filter((column) => Math.abs(column.width - shortestWidth) <= epsilon);
    const nextColumn = columns.find((column) => column.width > shortestWidth + epsilon);

    if (!nextColumn) {
      const delta = remaining / columns.length;
      for (const column of columns) {
        column.width += delta;
      }
      break;
    }

    const costToNext = (nextColumn.width - shortestWidth) * shortestColumns.length;
    if (remaining >= costToNext) {
      for (const column of shortestColumns) {
        column.width = nextColumn.width;
      }
      remaining -= costToNext;
      continue;
    }

    const delta = remaining / shortestColumns.length;
    for (const column of shortestColumns) {
      column.width += delta;
    }
    break;
  }

  return columns
    .sort((a, b) => a.index - b.index)
    .map((column) => column.width);
}

function roundLayoutPixel(value: number): number {
  return Math.round(value * 1000) / 1000;
}

const hiddenStatusPopoverStyle = {
  position: "fixed",
  top: 0,
  left: 0,
  maxWidth: `calc(100vw - ${STATUS_POPOVER_VIEWPORT_MARGIN * 2}px)`,
  maxHeight: `calc(100vh - ${STATUS_POPOVER_VIEWPORT_MARGIN * 2}px)`,
  "--status-popover-max-width": `calc(100vw - ${STATUS_POPOVER_VIEWPORT_MARGIN * 2}px)`,
  "--status-popover-max-height": `calc(100vh - ${STATUS_POPOVER_VIEWPORT_MARGIN * 2}px)`,
  visibility: "hidden"
} as CSSProperties;

function useAnchoredStatusPopover(open: boolean, anchorRef: StatusPopoverAnchorRef) {
  const popoverRef = useRef<HTMLDivElement | null>(null);
  const [style, setStyle] = useState<CSSProperties>(hiddenStatusPopoverStyle);

  useBrowserLayoutEffect(() => {
    if (!open || typeof window === "undefined") {
      setStyle(hiddenStatusPopoverStyle);
      return;
    }

    let animationFrame: number | null = null;
    const updateLayout = () => {
      const anchor = anchorRef.current;
      const popover = popoverRef.current;
      if (!anchor || !popover) return;

      const anchorRect = anchor.getBoundingClientRect();
      const popoverRect = popover.getBoundingClientRect();
      const next = resolveStatusPopoverLayout({
        anchorRect,
        popoverSize: {
          width: popoverRect.width,
          height: popoverRect.height
        },
        viewportWidth: window.innerWidth,
        viewportHeight: window.innerHeight
      });

      setStyle({
        position: "fixed",
        top: next.top,
        left: next.left,
        maxWidth: next.maxWidth,
        maxHeight: next.maxHeight,
        "--status-popover-max-width": `${next.maxWidth}px`,
        "--status-popover-max-height": `${next.maxHeight}px`,
        visibility: "visible"
      } as CSSProperties);
    };
    const scheduleLayout = () => {
      if (typeof window.requestAnimationFrame !== "function") {
        updateLayout();
        return;
      }
      if (animationFrame !== null) return;
      animationFrame = window.requestAnimationFrame(() => {
        animationFrame = null;
        updateLayout();
      });
    };

    scheduleLayout();
    const resizeObserver = typeof ResizeObserver === "undefined"
      ? null
      : new ResizeObserver(scheduleLayout);
    if (resizeObserver) {
      const anchor = anchorRef.current;
      const popover = popoverRef.current;
      if (anchor) resizeObserver.observe(anchor);
      if (popover) resizeObserver.observe(popover);
    }
    window.addEventListener("resize", scheduleLayout);
    window.addEventListener("scroll", scheduleLayout, true);
    return () => {
      if (animationFrame !== null && typeof window.cancelAnimationFrame === "function") {
        window.cancelAnimationFrame(animationFrame);
      }
      resizeObserver?.disconnect();
      window.removeEventListener("resize", scheduleLayout);
      window.removeEventListener("scroll", scheduleLayout, true);
    };
  }, [anchorRef, open]);

  return { popoverRef, style };
}

function StatusPopoverLayer({
  open,
  anchorRef,
  children
}: {
  open: boolean;
  anchorRef: StatusPopoverAnchorRef;
  children: ReactNode;
}) {
  const { popoverRef, style } = useAnchoredStatusPopover(open, anchorRef);
  if (!open || typeof document === "undefined") return null;
  return createPortal(
    <div className="status-popover-layer" ref={popoverRef} style={style}>
      {children}
    </div>,
    document.body
  );
}

export function resolveStatusPopoverLayout({
  anchorRect,
  popoverSize,
  viewportWidth,
  viewportHeight,
  margin = STATUS_POPOVER_VIEWPORT_MARGIN,
  gap = STATUS_POPOVER_GAP_PX
}: StatusPopoverLayoutInput): StatusPopoverLayout {
  const maxWidth = Math.max(0, viewportWidth - margin * 2);
  const maxHeight = Math.max(0, viewportHeight - margin * 2);
  const popoverWidth = Math.min(nonNegative(popoverSize.width), maxWidth);
  const popoverHeight = Math.min(nonNegative(popoverSize.height), maxHeight);
  const maxLeft = Math.max(margin, viewportWidth - margin - popoverWidth);
  const maxTop = Math.max(margin, viewportHeight - margin - popoverHeight);
  const preferredLeft = anchorRect.left;
  const belowTop = anchorRect.bottom + gap;
  const aboveTop = anchorRect.top - gap - popoverHeight;
  const preferredTop = belowTop + popoverHeight <= viewportHeight - margin
    ? belowTop
    : aboveTop >= margin
      ? aboveTop
      : belowTop;

  return {
    top: Math.round(clampNumber(preferredTop, margin, maxTop)),
    left: Math.round(clampNumber(preferredLeft, margin, maxLeft)),
    maxWidth: Math.round(maxWidth),
    maxHeight: Math.round(maxHeight)
  };
}

function nonNegative(value: number): number {
  return Number.isFinite(value) ? Math.max(0, value) : 0;
}

function clampNumber(value: number, min: number, max: number): number {
  if (max < min) return min;
  return Math.max(min, Math.min(value, max));
}

function resolveRailPopoverLayout({
  anchorTop,
  anchorRight,
  menuHeight,
  viewportWidth,
  viewportHeight,
  minWidth,
  maxWidth
}: RailPopoverLayoutInput): CSSProperties {
  const maxHeight = Math.max(0, viewportHeight - RAIL_SETTINGS_MENU_MARGIN_PX * 2);
  const preferredLeft = anchorRight + RAIL_SETTINGS_MENU_GAP_PX;
  const preferredAvailableWidth = viewportWidth - preferredLeft - RAIL_SETTINGS_MENU_MARGIN_PX;
  const left = preferredAvailableWidth >= minWidth
    ? preferredLeft
    : Math.max(
      RAIL_SETTINGS_MENU_MARGIN_PX,
      viewportWidth - minWidth - RAIL_SETTINGS_MENU_MARGIN_PX
    );
  const width = Math.max(
    0,
    Math.min(
      maxWidth,
      viewportWidth - left - RAIL_SETTINGS_MENU_MARGIN_PX
    )
  );
  const measuredHeight = menuHeight > 0 ? Math.min(menuHeight, maxHeight) : maxHeight;
  const maxTop = Math.max(
    RAIL_SETTINGS_MENU_MARGIN_PX,
    viewportHeight - measuredHeight - RAIL_SETTINGS_MENU_MARGIN_PX
  );
  const top = Math.max(
    RAIL_SETTINGS_MENU_MARGIN_PX,
    Math.min(anchorTop, maxTop)
  );

  return {
    top: Math.round(top),
    left: Math.round(left),
    width: Math.round(width),
    maxHeight: Math.round(maxHeight),
  };
}

export function resolveRailSettingsMenuLayout(input: RailSettingsMenuLayoutInput): CSSProperties {
  return resolveRailPopoverLayout({
    ...input,
    minWidth: RAIL_SETTINGS_MENU_MIN_WIDTH_PX,
    maxWidth: RAIL_SETTINGS_MENU_MAX_WIDTH_PX
  });
}

function resolveRailExportMenuLayout(input: RailSettingsMenuLayoutInput): CSSProperties {
  return resolveRailPopoverLayout({
    ...input,
    minWidth: RAIL_EXPORT_MENU_MIN_WIDTH_PX,
    maxWidth: RAIL_EXPORT_MENU_MAX_WIDTH_PX
  });
}

export default function App() {
  const [metadata, setMetadata] = useState<GuiMetadata | null>(null);
  const [config, setConfig] = useState<PersistedConfig | null>(null);
  const [statesBySession, dispatchSessionState] = useReducer(reduceSessionStates, {});
  const [input, setInput] = useState("");
  const [modelDialogOpen, setModelDialogOpen] = useState(false);
  const [pluginDialogOpen, setPluginDialogOpen] = useState(false);
  const [projectPopoverOpen, setProjectPopoverOpen] = useState(false);
  const [contextPopoverOpen, setContextPopoverOpen] = useState(false);
  const [queuePopoverOpen, setQueuePopoverOpen] = useState(false);
  const [sessionDialogOpen, setSessionDialogOpen] = useState(false);
  const [rightSidebarTab, setRightSidebarTab] = useState<WorkspaceSidebarTab | null>(null);
  const [subagentObserverId, setSubagentObserverId] = useState<string | null>(null);
  const [sessions, setSessions] = useState<SessionMetaPayload[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [queueTaskDraft, setQueueTaskDraft] = useState("");
  const [queueTaskBusy, setQueueTaskBusy] = useState(false);
  const [editingQueueTaskId, setEditingQueueTaskId] = useState<string | null>(null);
  const [queueTaskEditDraft, setQueueTaskEditDraft] = useState("");
  const [inputAttachments, setInputAttachments] = useState<PendingAttachment[]>([]);
  const [inputAttachmentError, setInputAttachmentError] = useState<string | null>(null);
  const [inputUploading, setInputUploading] = useState(false);
  const [editingMessage, setEditingMessage] = useState<MessageEditState | null>(null);
  const [inputDropActive, setInputDropActive] = useState(false);
  const [blobPreviewUrls, setBlobPreviewUrls] = useState<BlobPreviewUrls>({});
  const [chatPanelHeight, setChatPanelHeight] = useState<number | undefined>(undefined);
  const [mediaPreview, setMediaPreview] = useState<MediaPreviewState | null>(null);
  const [sessionStats, setSessionStats] = useState<SessionRuntimeStats>({
    running: 0,
    loaded: 0,
    maxLoaded: 5
  });
  const [sessionBusy, setSessionBusy] = useState(false);
  const [cwdBusy, setCwdBusy] = useState(false);
  const [contextCompactBusy, setContextCompactBusy] = useState(false);
  const [contextSettingsBusy, setContextSettingsBusy] = useState(false);
  const [exportBusy, setExportBusy] = useState(false);
  const [exportMenuOpen, setExportMenuOpen] = useState(false);
  const [exportMenuStyle, setExportMenuStyle] = useState<CSSProperties>({});
  const [settingsMenuOpen, setSettingsMenuOpen] = useState(false);
  const [settingsMenuStyle, setSettingsMenuStyle] = useState<CSSProperties>({});
  const [systemPromptConfirmOpen, setSystemPromptConfirmOpen] = useState(false);
  const [systemPromptEditAcknowledgement, setSystemPromptEditAcknowledgement] = useState<CacheSensitiveSettingAcknowledgement | null>(null);
  const [pluginSettingsConfirmOpen, setPluginSettingsConfirmOpen] = useState(false);
  const [pluginSettingsApplyAcknowledgement, setPluginSettingsApplyAcknowledgement] = useState<CacheSensitiveSettingAcknowledgement | null>(null);
  const [pendingPluginSettingsApply, setPendingPluginSettingsApply] = useState<PendingPluginApply | null>(null);
  const [pendingToolCallPurposeEnabled, setPendingToolCallPurposeEnabled] = useState<boolean | null>(null);
  const [historySearchOpen, setHistorySearchOpen] = useState(false);
  const [historySearchQuery, setHistorySearchQuery] = useState("");
  const [historySearchCaseSensitive, setHistorySearchCaseSensitive] = useState(false);
  const [historySearchWholeWord, setHistorySearchWholeWord] = useState(false);
  const [historySearchResults, setHistorySearchResults] = useState<HistorySearchResult[]>([]);
  const [historySearchBusy, setHistorySearchBusy] = useState(false);
  const [historySearchError, setHistorySearchError] = useState<string | null>(null);
  const [selectedHistoryResult, setSelectedHistoryResult] = useState<HistorySearchResult | null>(null);
  const [historyPreviewNodes, setHistoryPreviewNodes] = useState<ChatNode[]>([]);
  const [historyPreviewBusy, setHistoryPreviewBusy] = useState(false);
  const [historyPreviewSession, setHistoryPreviewSession] = useState<SessionMetaPayload | null>(null);
  const [historyPreviewLocateTarget, setHistoryPreviewLocateTarget] = useState<HistoryLocateTarget | null>(null);
  const [mainLocateTarget, setMainLocateTarget] = useState<HistoryLocateTarget | null>(null);
  const [pendingMainLocateScrollTarget, setPendingMainLocateScrollTarget] = useState<HistoryLocateTarget | null>(null);
  const [sideSelectionAnchor, setSideSelectionAnchor] = useState<SideSelectionAnchor | null>(null);
  const [sideThreadPopover, setSideThreadPopover] = useState<SideThreadPopoverState | null>(null);
  const [sideThreadDraft, setSideThreadDraft] = useState("");
  const [sideThreadBusy, setSideThreadBusy] = useState(false);
  const appShellRef = useRef<HTMLDivElement | null>(null);
  const brandBarRef = useRef<HTMLDivElement | null>(null);
  const statusRowRef = useRef<HTMLDivElement | null>(null);
  const statusStripRef = useRef<HTMLDivElement | null>(null);
  const minimumContentSizeRef = useRef<LayoutSize | null>(null);
  const chatRef = useRef<HTMLDivElement | null>(null);
  const historyPreviewRef = useRef<HTMLDivElement | null>(null);
  const systemPromptRef = useRef<HTMLTextAreaElement | null>(null);
  const exportMenuAnchorRef = useRef<HTMLDivElement | null>(null);
  const exportMenuRef = useRef<HTMLDivElement | null>(null);
  const settingsMenuAnchorRef = useRef<HTMLDivElement | null>(null);
  const settingsMenuRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const attachmentFileInputRef = useRef<HTMLInputElement | null>(null);
  const queueTaskDraftRef = useRef<HTMLTextAreaElement | null>(null);
  const configRef = useRef<PersistedConfig | null>(null);
  const currentSessionIdRef = useRef<string | null>(null);
  const pendingSystemPromptConfigRef = useRef<PersistedConfig | null>(null);
  const initializeSessionStateRef = useRef<() => Promise<void>>(async () => undefined);
  const startupSessionStateLoadedRef = useRef(false);
  const applyingSystemPromptRef = useRef(false);
  const sessionBusyRef = useRef(false);
  const followTailRef = useRef(true);
  const selectingChatRef = useRef(false);
  const forkSessionRef = useRef<(
    sessionId?: string,
    messageIndex?: number,
    contextMessageId?: string,
  ) => Promise<void>>(async () => undefined);
  const userScrollIntentRef = useRef(false);
  const isAutoScrollingRef = useRef(false);
  const autoScrollFrameRef = useRef<number | null>(null);
  const inputComposingRef = useRef(false);
  const inputCompositionEndTimerRef = useRef<number | null>(null);
  const inputHistoryBySessionRef = useRef<Record<string, string[]>>({});
  const inputHistoryNavigationRef = useRef<{
    sessionKey: string;
    index: number | null;
    draft: string;
  }>({ sessionKey: UNLOADED_INPUT_HISTORY_KEY, index: null, draft: "" });
  const inputAttachmentsRef = useRef<PendingAttachment[]>([]);
  const editingMessageRef = useRef<MessageEditState | null>(null);
  const blobPreviewUrlsRef = useRef<BlobPreviewUrls>({});
  const pendingBlobPreviewFetchesRef = useRef<Map<string, PendingBlobPreviewFetch>>(new Map());
  const fetchingBlobPreviewIdsRef = useRef<Set<string>>(new Set());
  const failedBlobPreviewIdsRef = useRef<Set<string>>(new Set());
  const queueTaskComposingRef = useRef(false);
  const queueTaskCompositionEndTimerRef = useRef<number | null>(null);
  const artifactPanelSessionRef = useRef<string | null>(null);
  const previousArtifactCountRef = useRef(0);
  const subagentPanelSessionRef = useRef<string | null>(null);
  const previousSubagentCountRef = useRef(0);
  const shouldLoadStartupSessionState = shouldInitializeSessionState(metadata);
  const coreRunning = shouldLoadStartupSessionState || sessionStats.loaded > 0 || Boolean(currentSessionId);
  const fallbackState = useMemo(createInitialState, []);
  const state = currentSessionId ? statesBySession[currentSessionId] ?? fallbackState : fallbackState;
  const artifactList = useMemo(
    () => state.artifactOrder.map((key) => state.artifacts[key]).filter(Boolean),
    [state.artifactOrder, state.artifacts]
  );
  const artifactGroups = useMemo(() => groupArtifactsByType(artifactList), [artifactList]);
  const selectedArtifactForSidebar = state.selectedArtifactId && state.artifacts[state.selectedArtifactId]
    ? state.artifacts[state.selectedArtifactId]
    : artifactList[0];
  const selectedArtifactTab = selectedArtifactForSidebar
    ? artifactSidebarTab(selectedArtifactForSidebar.artifactType)
    : artifactGroups[0]
      ? artifactSidebarTab(artifactGroups[0].type)
      : null;
  const hasArtifacts = artifactList.length > 0;
  const subagentList = state.subagentOrder.map((id) => state.subagents[id]).filter(Boolean);
  const hasSubagents = subagentList.length > 0;
  const sideThreadList = state.sideThreadOrder.map((id) => state.sideThreads[id]).filter(Boolean);
  const hasSideThreads = sideThreadList.length > 0;
  const hasRightSidebar = hasArtifacts || hasSubagents || hasSideThreads;
  const observedSubagent = subagentObserverId ? state.subagents[subagentObserverId] : undefined;
  const popoverSideThread = sideThreadPopover?.threadId ? state.sideThreads[sideThreadPopover.threadId] : undefined;
  const canStopConversation = canStopRunnerState(state.runnerState);
  const canEditMessages = Boolean(currentSessionId) && !canStopConversation;
  const showDebug = config?.showDebug ?? true;
  const focusModeEnabled = config?.focusModeEnabled ?? true;
  const profilingEnabled = config?.profilingEnabled ?? true;
  const visibleChatNodes = useMemo(
    () => state.nodes.filter((node) => showDebug || node.kind !== "debug"),
    [showDebug, state.nodes]
  );
  const chatTailKey = useMemo(
    () => transcriptTailKey(visibleChatNodes, state.processing),
    [state.processing, visibleChatNodes]
  );
  const syncMinimumContentSize = useCallback(() => {
    const appShell = appShellRef.current;
    const brandBar = brandBarRef.current;
    const statusRow = statusRowRef.current;
    if (
      !appShell
      || !brandBar
      || !statusRow
      || typeof window === "undefined"
      || typeof window.hawi?.setMinimumContentSize !== "function"
    ) return;
    const nextSize = measureMinimumContentSize(appShell, brandBar);
    const previousSize = minimumContentSizeRef.current;
    if (previousSize?.width === nextSize.width && previousSize.height === nextSize.height) return;
    minimumContentSizeRef.current = nextSize;
    void window.hawi.setMinimumContentSize(nextSize).catch((error) => {
      console.warn("failed to update minimum content size", error);
    });
  }, []);
  useStatusMainColumnLayout(statusStripRef);

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

  useBrowserLayoutEffect(() => {
    syncMinimumContentSize();
  });

  useBrowserLayoutEffect(() => {
    if (typeof window === "undefined") return;
    const appShell = appShellRef.current;
    const brandBar = brandBarRef.current;
    const statusRow = statusRowRef.current;
    if (!appShell || !brandBar || !statusRow) return;
    let animationFrame: number | null = null;
    const scheduleSync = () => {
      if (typeof window.requestAnimationFrame !== "function") {
        syncMinimumContentSize();
        return;
      }
      if (animationFrame !== null) return;
      animationFrame = window.requestAnimationFrame(() => {
        animationFrame = null;
        syncMinimumContentSize();
      });
    };
    const observer = typeof ResizeObserver === "undefined"
      ? null
      : new ResizeObserver(scheduleSync);
    if (observer) {
      observer.observe(appShell);
      observer.observe(brandBar);
      observer.observe(statusRow);
      for (const child of Array.from(brandBar.children)) {
        if (child instanceof HTMLElement) {
          observer.observe(child);
          for (const grandchild of Array.from(child.children)) {
            if (grandchild instanceof HTMLElement) {
              observer.observe(grandchild);
            }
          }
        }
      }
      for (const child of Array.from(statusRow.children)) {
        if (child instanceof HTMLElement) {
          observer.observe(child);
          for (const grandchild of Array.from(child.children)) {
            if (grandchild instanceof HTMLElement) {
              observer.observe(grandchild);
            }
          }
        }
      }
    }
    window.addEventListener("resize", scheduleSync);
    scheduleSync();
    return () => {
      window.removeEventListener("resize", scheduleSync);
      observer?.disconnect();
      if (animationFrame !== null && typeof window.cancelAnimationFrame === "function") {
        window.cancelAnimationFrame(animationFrame);
      }
    };
  }, [syncMinimumContentSize]);

  useEffect(() => {
    currentSessionIdRef.current = currentSessionId;
    failedBlobPreviewIdsRef.current.clear();
    resetInputHistoryNavigation();
    // Input-history navigation is reset only when the active session changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentSessionId]);

  useEffect(() => {
    if (!currentSessionId || !coreRunning) {
      dispatch({
        version: VERSION,
        type: "gui.load_side_threads",
        payload: { side_threads: [] }
      }, currentSessionId);
      return;
    }
    let cancelled = false;
    void sendSessionCommand(
      "side_thread_list",
      { session_id: currentSessionId },
      currentSessionId
    ).then((frame) => {
      if (!cancelled) {
        applySideThreadsFromFrame(frame, currentSessionId);
      }
    });
    return () => {
      cancelled = true;
    };
    // The command helpers use refs for current config/session; this effect is keyed to session identity.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentSessionId, coreRunning]);

  useEffect(() => {
    function updateSelectionAnchor() {
      setSideSelectionAnchor(resolveSideSelectionAnchor(chatRef.current, currentSessionIdRef.current));
    }
    document.addEventListener("selectionchange", updateSelectionAnchor);
    document.addEventListener("mouseup", updateSelectionAnchor);
    document.addEventListener("keyup", updateSelectionAnchor);
    return () => {
      document.removeEventListener("selectionchange", updateSelectionAnchor);
      document.removeEventListener("mouseup", updateSelectionAnchor);
      document.removeEventListener("keyup", updateSelectionAnchor);
    };
  }, []);

  useEffect(() => {
    sessionBusyRef.current = sessionBusy;
  }, [sessionBusy]);

  useEffect(() => {
    if (!shouldLoadStartupSessionState || startupSessionStateLoadedRef.current) return;
    startupSessionStateLoadedRef.current = true;
    void initializeSessionStateRef.current();
  }, [shouldLoadStartupSessionState]);

  useEffect(() => {
    configRef.current = config;
  }, [config]);

  useEffect(() => {
    inputAttachmentsRef.current = inputAttachments;
  }, [inputAttachments]);

  useEffect(() => {
    editingMessageRef.current = editingMessage;
  }, [editingMessage]);

  useEffect(() => {
    const active = editingMessageRef.current;
    if (canStopConversation && active && !active.busy) {
      cancelEditingMessage();
    }
  }, [canStopConversation]);

  useEffect(() => {
    blobPreviewUrlsRef.current = blobPreviewUrls;
  }, [blobPreviewUrls]);

  useBrowserLayoutEffect(() => {
    const element = chatRef.current;
    if (!element || typeof window === "undefined") return;
    let animationFrame: number | null = null;
    const syncHeight = () => {
      animationFrame = null;
      const nextHeight = Math.max(240, Math.floor(element.clientHeight || element.getBoundingClientRect().height));
      setChatPanelHeight((current) => current === nextHeight ? current : nextHeight);
    };
    const scheduleSync = () => {
      if (typeof window.requestAnimationFrame !== "function") {
        syncHeight();
        return;
      }
      if (animationFrame !== null) return;
      animationFrame = window.requestAnimationFrame(syncHeight);
    };
    const observer = typeof ResizeObserver === "undefined"
      ? null
      : new ResizeObserver(scheduleSync);
    observer?.observe(element);
    window.addEventListener("resize", scheduleSync);
    scheduleSync();
    return () => {
      observer?.disconnect();
      window.removeEventListener("resize", scheduleSync);
      if (animationFrame !== null && typeof window.cancelAnimationFrame === "function") {
        window.cancelAnimationFrame(animationFrame);
      }
    };
  }, []);

  useEffect(() => {
    if (!settingsMenuOpen && !exportMenuOpen) return;
    function handlePointerDown(event: PointerEvent) {
      const target = event.target;
      if (settingsMenuOpen && !(target instanceof Element && target.closest(".rail-settings-anchor"))) {
        setSettingsMenuOpen(false);
      }
      if (exportMenuOpen && !(target instanceof Element && target.closest(".rail-export-anchor"))) {
        setExportMenuOpen(false);
      }
    }
    document.addEventListener("pointerdown", handlePointerDown);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
    };
  }, [exportMenuOpen, settingsMenuOpen]);

  useBrowserLayoutEffect(() => {
    if (!exportMenuOpen) {
      setExportMenuStyle({});
      return;
    }
    const anchor = exportMenuAnchorRef.current;
    const menu = exportMenuRef.current;
    if (!anchor || !menu) return;

    let frameId: number | null = null;
    const updateMenuLayout = () => {
      const anchorRect = anchor.getBoundingClientRect();
      const menuRect = menu.getBoundingClientRect();
      const nextStyle = resolveRailExportMenuLayout({
        anchorTop: anchorRect.top,
        anchorRight: anchorRect.right,
        menuHeight: menuRect.height,
        viewportWidth: window.innerWidth,
        viewportHeight: window.innerHeight,
      });
      setExportMenuStyle((current) => (
        current.top === nextStyle.top
          && current.left === nextStyle.left
          && current.width === nextStyle.width
          && current.maxHeight === nextStyle.maxHeight
          ? current
          : nextStyle
      ));
    };
    const scheduleMenuLayout = () => {
      if (frameId !== null) return;
      frameId = window.requestAnimationFrame(() => {
        frameId = null;
        updateMenuLayout();
      });
    };

    updateMenuLayout();
    scheduleMenuLayout();
    window.addEventListener("resize", scheduleMenuLayout);
    window.addEventListener("scroll", scheduleMenuLayout, true);
    const resizeObserver = typeof ResizeObserver === "undefined"
      ? null
      : new ResizeObserver(scheduleMenuLayout);
    resizeObserver?.observe(anchor);
    resizeObserver?.observe(menu);

    return () => {
      if (frameId !== null) {
        window.cancelAnimationFrame(frameId);
      }
      window.removeEventListener("resize", scheduleMenuLayout);
      window.removeEventListener("scroll", scheduleMenuLayout, true);
      resizeObserver?.disconnect();
    };
  }, [exportMenuOpen]);

  useBrowserLayoutEffect(() => {
    if (!settingsMenuOpen) {
      setSettingsMenuStyle({});
      return;
    }
    const anchor = settingsMenuAnchorRef.current;
    const menu = settingsMenuRef.current;
    if (!anchor || !menu) return;

    let frameId: number | null = null;
    const updateMenuLayout = () => {
      const anchorRect = anchor.getBoundingClientRect();
      const menuRect = menu.getBoundingClientRect();
      const nextStyle = resolveRailSettingsMenuLayout({
        anchorTop: anchorRect.top,
        anchorRight: anchorRect.right,
        menuHeight: menuRect.height,
        viewportWidth: window.innerWidth,
        viewportHeight: window.innerHeight,
      });
      setSettingsMenuStyle((current) => (
        current.top === nextStyle.top
          && current.left === nextStyle.left
          && current.width === nextStyle.width
          && current.maxHeight === nextStyle.maxHeight
          ? current
          : nextStyle
      ));
    };
    const scheduleMenuLayout = () => {
      if (frameId !== null) return;
      frameId = window.requestAnimationFrame(() => {
        frameId = null;
        updateMenuLayout();
      });
    };

    updateMenuLayout();
    scheduleMenuLayout();
    window.addEventListener("resize", scheduleMenuLayout);
    window.addEventListener("scroll", scheduleMenuLayout, true);
    const resizeObserver = typeof ResizeObserver === "undefined"
      ? null
      : new ResizeObserver(scheduleMenuLayout);
    resizeObserver?.observe(anchor);
    resizeObserver?.observe(menu);

    return () => {
      if (frameId !== null) {
        window.cancelAnimationFrame(frameId);
      }
      window.removeEventListener("resize", scheduleMenuLayout);
      window.removeEventListener("scroll", scheduleMenuLayout, true);
      resizeObserver?.disconnect();
    };
  }, [settingsMenuOpen]);

  useEffect(() => () => {
    for (const pending of pendingBlobPreviewFetchesRef.current.values()) {
      window.clearTimeout(pending.timeoutId);
    }
    pendingBlobPreviewFetchesRef.current.clear();
    fetchingBlobPreviewIdsRef.current.clear();
    const retained = new Set(Object.values(blobPreviewUrlsRef.current));
    for (const attachment of inputAttachmentsRef.current) {
      if (!retained.has(attachment.previewUrl)) {
        revokeObjectUrl(attachment.previewUrl);
      }
    }
    for (const item of editingMessageRef.current?.attachments ?? []) {
      if (item.kind !== "pending") continue;
      if (!retained.has(item.attachment.previewUrl)) {
        revokeObjectUrl(item.attachment.previewUrl);
      }
    }
    for (const previewUrl of retained) {
      revokeObjectUrl(previewUrl);
    }
  }, []);

  useEffect(() => {
    if (!currentSessionId || !coreRunning) return;
    const requests = collectMissingBlobPreviewRequests(state, blobPreviewUrlsRef.current);
    for (const request of requests) {
      void ensureBlobPreview(request);
    }
  }, [blobPreviewUrls, coreRunning, currentSessionId, state.nodes, state.queueMessages, state.subagents]);

  useEffect(() => {
    function handleDialogKeyboard(event: KeyboardEvent) {
      if (event.isComposing || event.metaKey || event.ctrlKey || event.altKey) return;
      const keyboardState = {
        mediaPreviewOpen: mediaPreview !== null,
        sideThreadPopoverOpen: sideThreadPopover !== null,
        contextPopoverOpen,
        projectPopoverOpen,
        pluginDialogOpen,
        modelDialogOpen,
        systemPromptConfirmOpen,
        pluginSettingsConfirmOpen,
        toolCallPurposeConfirmOpen: pendingToolCallPurposeEnabled !== null,
        exportMenuOpen,
        subagentObserverOpen: subagentObserverId !== null,
        settingsMenuOpen,
        queuePopoverOpen,
        editingQueueTaskId,
        sessionDialogOpen,
        editingMessageId: editingMessageRef.current?.nodeId ?? null
      };

      if (event.key === "Escape") {
        const target = resolveEscapeDismissTarget(keyboardState);
        if (!target) {
          if (!canStopConversation) return;
          window.setTimeout(() => {
            if (event.defaultPrevented) return;
            void sendCommand("stop", { reason: "user" });
          }, 0);
          return;
        }
        event.preventDefault();
        event.stopPropagation();
        switch (target) {
          case "mediaPreview":
            setMediaPreview(null);
            break;
          case "sideThreadPopover":
            setSideThreadPopover(null);
            break;
          case "subagentObserver":
            setSubagentObserverId(null);
            break;
          case "contextPopover":
            setContextPopoverOpen(false);
            break;
          case "projectPopover":
            setProjectPopoverOpen(false);
            break;
          case "pluginDialog":
            setPluginDialogOpen(false);
            break;
          case "modelDialog":
            setModelDialogOpen(false);
            break;
          case "systemPromptConfirm":
            setSystemPromptConfirmOpen(false);
            break;
          case "pluginSettingsConfirm":
            setPluginSettingsConfirmOpen(false);
            setPendingPluginSettingsApply(null);
            break;
          case "toolCallPurposeConfirm":
            setPendingToolCallPurposeEnabled(null);
            break;
          case "exportMenu":
            setExportMenuOpen(false);
            break;
          case "settingsMenu":
            setSettingsMenuOpen(false);
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
          case "messageEdit":
            cancelEditingMessage();
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
    contextPopoverOpen,
    projectPopoverOpen,
    pluginDialogOpen,
    modelDialogOpen,
    systemPromptConfirmOpen,
    pluginSettingsConfirmOpen,
    pendingPluginSettingsApply,
    pendingToolCallPurposeEnabled,
    exportMenuOpen,
    mediaPreview,
    sideThreadPopover,
    subagentObserverId,
    settingsMenuOpen,
    queuePopoverOpen,
    editingQueueTaskId,
    sessionDialogOpen,
    canStopConversation
  ]);

  useBrowserLayoutEffect(() => {
    keepChatTailVisible();
  }, [chatTailKey]);

  useEffect(() => {
    if (artifactPanelSessionRef.current !== currentSessionId) {
      artifactPanelSessionRef.current = currentSessionId;
      previousArtifactCountRef.current = 0;
    }
    const artifactCount = state.artifactOrder.filter((key) => Boolean(state.artifacts[key])).length;
    if (artifactCount === 0) {
      previousArtifactCountRef.current = 0;
      setRightSidebarTab((current) => (
        artifactTypeFromSidebarTab(current) ? (hasSubagents ? "subagents" : null) : current
      ));
      return;
    }
    if (previousArtifactCountRef.current === 0 && selectedArtifactTab) {
      setRightSidebarTab(selectedArtifactTab);
    }
    previousArtifactCountRef.current = artifactCount;
  }, [currentSessionId, hasSubagents, selectedArtifactTab, state.artifactOrder, state.artifacts]);

  useEffect(() => {
    if (subagentPanelSessionRef.current !== currentSessionId) {
      subagentPanelSessionRef.current = currentSessionId;
      previousSubagentCountRef.current = 0;
    }
    const subagentCount = state.subagentOrder.length;
    if (subagentCount === 0) {
      previousSubagentCountRef.current = 0;
      setRightSidebarTab((current) => (
        current === "subagents" ? selectedArtifactTab : current
      ));
      setSubagentObserverId(null);
      return;
    }
    if (previousSubagentCountRef.current === 0) {
      setRightSidebarTab("subagents");
    }
    if (subagentObserverId && !state.subagents[subagentObserverId]) {
      setSubagentObserverId(null);
    }
    previousSubagentCountRef.current = subagentCount;
  }, [currentSessionId, selectedArtifactTab, state.subagentOrder, state.subagents, subagentObserverId]);

  useEffect(() => {
    if (!hasSideThreads && rightSidebarTab === "side_threads") {
      setRightSidebarTab(selectedArtifactTab ?? (hasSubagents ? "subagents" : null));
    }
    if (
      sideThreadPopover?.threadId
      && !state.sideThreads[sideThreadPopover.threadId]
    ) {
      setSideThreadPopover(null);
    }
  }, [hasSideThreads, hasSubagents, rightSidebarTab, selectedArtifactTab, sideThreadPopover, state.sideThreads]);

  useEffect(() => {
    function syncSelection() {
      selectingChatRef.current = hasChatSelection();
      if (selectingChatRef.current) {
        followTailRef.current = false;
        cancelPendingAutoScroll();
      }
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

  const currentSessionHasConversation = state.nodes.some(isConversationNode);
  const systemPromptRequiresConfirmation = shouldConfirmSystemPromptEdit({
    hasConversation: currentSessionHasConversation,
    sessionId: currentSessionId,
    messageCount: state.sessionMessageCount,
    acknowledgedSessionId: systemPromptEditAcknowledgement?.sessionId ?? null,
    acknowledgedMessageCount: systemPromptEditAcknowledgement?.messageCount ?? null
  });
  const pluginSettingsApplyRequiresConfirmation = shouldConfirmPluginSettingsApply({
    hasConversation: currentSessionHasConversation,
    sessionId: currentSessionId,
    messageCount: state.sessionMessageCount,
    acknowledgedSessionId: pluginSettingsApplyAcknowledgement?.sessionId ?? null,
    acknowledgedMessageCount: pluginSettingsApplyAcknowledgement?.messageCount ?? null
  });
  const toolCallPurposeControl = resolveToolCallPurposeControlState();
  const contextRunnerBusy = state.runnerState === "RUNNING" || state.runnerState === "INTERRUPTING";
  const canCompactContextManually = coreRunning && !contextRunnerBusy && state.contextCompression?.active !== true;

  useEffect(() => {
    resizeTextareaToRows(systemPromptRef.current, SYSTEM_PROMPT_MAX_ROWS);
  }, [config?.systemPrompt, settingsMenuOpen, systemPromptRequiresConfirmation]);

  useEffect(() => {
    resizeTextareaToRows(inputRef.current, MESSAGE_INPUT_MAX_ROWS);
  }, [input]);

  function dispatch(frame: CoreFrame, sessionId = currentSessionIdRef.current) {
    dispatchSessionState({ sessionId, frame });
  }

  function handleCoreEvent(frame: CoreFrame) {
    if (handleBlobPreviewFetchFrame(frame)) {
      return;
    }
    if (frame.type === "gui.session_status") {
      applySessionRuntimeStatus(frame);
      return;
    }
    if (frame.type === "session.title_updated") {
      applySessionTitleUpdated(frame);
      return;
    }
    const sessionId = frameSessionId(frame) ?? currentSessionIdRef.current;
    dispatchSessionState({ sessionId, frame });
  }

  function handleBlobPreviewFetchFrame(frame: CoreFrame): boolean {
    if (frame.type !== "blob.chunk" && frame.type !== "blob.complete") {
      return false;
    }
    const payload = framePayload(frame);
    const blobId = optionalPayloadString(payload?.blob_id);
    if (!blobId) return true;
    const pending = pendingBlobPreviewFetchesRef.current.get(blobId);
    if (!pending) return true;

    if (frame.type === "blob.chunk") {
      const dataB64 = optionalPayloadString(payload?.data_b64);
      if (dataB64) {
        pending.chunks.push(dataB64);
      }
      return true;
    }

    window.clearTimeout(pending.timeoutId);
    pendingBlobPreviewFetchesRef.current.delete(blobId);
    pending.resolve(pending.chunks.join(""));
    return true;
  }

  function applySessionRuntimeStatus(frame: CoreFrame) {
    const payload = framePayload(frame);
    if (!payload) return;
    const sessionId = optionalPayloadString(payload.session_id);
    const loadState = normalizeSessionLoadState(payload.load_state);
    const hasCurrentSessionId = hasPayloadKey(payload, "current_session_id");
    const currentId = optionalPayloadString(payload.current_session_id);
    if (hasCurrentSessionId) {
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
      last_finished_at: optionalPayloadNumber(payload.last_finished_at),
      last_cwd: optionalPayloadString(payload.last_cwd)
    }, {
      createIfMissing: payload.has_visible_messages === true
    }));
  }

  function applySessionTitleUpdated(frame: CoreFrame) {
    const payload = framePayload(frame);
    if (!payload) return;
    const sessionId = optionalPayloadString(payload.session_id);
    const name = optionalPayloadString(payload.name);
    if (!sessionId || !name) return;
    const updatedAt = new Date().toISOString();
    setSessions((items) => {
      const existing = items.find((item) => item.session_id === sessionId);
      if (!existing) {
        const isCurrent = sessionId === currentSessionIdRef.current;
        return sortSessionsByCreatedAt([
          {
            session_id: sessionId,
            name,
            created_at: updatedAt,
            updated_at: updatedAt,
            last_checkpoint_event: "session_auto_title",
            components_present: [],
            locked: false,
            lock_owner: null,
            load_state: isCurrent ? "loaded" : undefined,
          },
          ...items,
        ]);
      }
      return items.map((item) => (
        item.session_id === sessionId
          ? {
              ...item,
              name,
              updated_at: updatedAt,
              last_checkpoint_event: "session_auto_title",
            }
          : item
      ));
    });
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

  function pauseChatFollowForSelection(event: ReactMouseEvent<HTMLElement>) {
    if (event.button !== 0 || isInteractiveTranscriptTarget(event.target)) return;
    selectingChatRef.current = true;
    followTailRef.current = false;
    markChatUserScrollIntent();
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
    if (!(configRef.current ?? config)?.modelName) {
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

  async function sendSessionCommand(
    type: CoreCommandType,
    payload: Record<string, unknown>,
    targetSessionId: string | null = currentSessionIdRef.current
  ): Promise<CoreFrame | null> {
    try {
      return await window.hawi.sendCommand(type, payload, targetSessionId);
    } catch (error) {
      dispatch(errorFrame(error));
      return null;
    }
  }

  async function sendReadOnlyCommand(
    type: CoreCommandType,
    payload: Record<string, unknown>,
  ): Promise<CoreFrame | null> {
    try {
      return await window.hawi.sendCommand(type, payload, null);
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

  useEffect(() => {
    if (!historySearchOpen) {
      return;
    }
    const query = historySearchQuery.trim();
    if (!query) {
      setHistorySearchResults([]);
      setSelectedHistoryResult(null);
      setHistoryPreviewNodes([]);
      setHistoryPreviewSession(null);
      setHistoryPreviewLocateTarget(null);
      setHistorySearchBusy(false);
      setHistorySearchError(null);
      return;
    }
    let cancelled = false;
    setHistorySearchBusy(true);
    const timer = window.setTimeout(() => {
      void sendReadOnlyCommand("session_search", {
        query,
        limit: 100,
        case_sensitive: historySearchCaseSensitive,
        whole_word: historySearchWholeWord
      }).then((frame) => {
        if (cancelled) return;
        const payload = framePayload(frame);
        const results = normalizeHistorySearchResults(payload?.results);
        setHistorySearchResults(results);
        setHistorySearchError(null);
        const selectedKey = selectedHistoryResult ? historySearchResultKey(selectedHistoryResult) : null;
        const nextSelection = selectedKey
          ? results.find((item) => historySearchResultKey(item) === selectedKey) ?? results[0]
          : results[0];
        if (nextSelection) {
          void selectHistorySearchResult(nextSelection);
        } else {
          setSelectedHistoryResult(null);
          setHistoryPreviewNodes([]);
          setHistoryPreviewSession(null);
          setHistoryPreviewLocateTarget(null);
        }
      }).catch((error) => {
        if (!cancelled) {
          setHistorySearchError(formatDialogError(error));
        }
      }).finally(() => {
        if (!cancelled) {
          setHistorySearchBusy(false);
        }
      });
    }, 220);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [historySearchCaseSensitive, historySearchOpen, historySearchQuery, historySearchWholeWord]);

  useEffect(() => {
    if (!historyPreviewLocateTarget) return;
    if (historySearchQuery.trim()) return;
    const frame = window.requestAnimationFrame(() => {
      scrollToHistoryTarget(historyPreviewRef.current, historyPreviewLocateTarget);
    });
    return () => window.cancelAnimationFrame(frame);
  }, [historyPreviewLocateTarget, historyPreviewNodes, historySearchQuery]);

  useEffect(() => {
    if (!pendingMainLocateScrollTarget || pendingMainLocateScrollTarget.sessionId !== currentSessionId) return;
    const frame = window.requestAnimationFrame(() => {
      const found = scrollToHistoryTarget(chatRef.current, pendingMainLocateScrollTarget);
      setPendingMainLocateScrollTarget((current) => (
        reducePendingHistoryLocateScroll(current, pendingMainLocateScrollTarget, found)
      ));
    });
    return () => window.cancelAnimationFrame(frame);
  }, [pendingMainLocateScrollTarget, currentSessionId, state.nodes]);

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
    const hasCurrentSessionId = hasPayloadKey(payload, "current_session_id");
    const nextCurrent = optionalPayloadString(payload.current_session_id);
    if (hasCurrentSessionId) {
      setCurrentSessionId(nextCurrent);
    }
    if (nextCurrent) {
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

  function applySideThreadsFromFrame(frame: CoreFrame | null, fallbackSessionId = currentSessionIdRef.current) {
    const payload = framePayload(frame);
    const sessionId = optionalPayloadString(payload?.session_id) ?? fallbackSessionId;
    if (!Array.isArray(payload?.side_threads)) {
      return;
    }
    dispatch({
      version: VERSION,
      type: "gui.load_side_threads",
      payload: { side_threads: payload.side_threads }
    }, sessionId);
  }

  function requestMainLocateScroll(target: HistoryLocateTarget) {
    setMainLocateTarget(target);
    setPendingMainLocateScrollTarget(target);
  }

  function syncConfigFromSession(sessionId: string, sessionList = sessions) {
    const profile = sessionList.find((session) => session.session_id === sessionId)?.gui_launch_profile;
    if (!profile || !configRef.current) {
      return;
    }
    const nextConfig = configFromLaunchProfile(profile, configRef.current);
    configRef.current = nextConfig;
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
    const isCurrentRunning = sessionId === currentSessionIdRef.current && canStopRunnerState(state.runnerState);
    if (target?.load_state === "running" || isCurrentRunning) {
      dispatch(errorFrame("运行中的 Session 不能删除，请先停止当前任务。"));
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
      const payload = framePayload(frame);
      const hasCurrentSessionId = hasPayloadKey(payload, "current_session_id");
      const nextCurrent = optionalPayloadString(payload?.current_session_id);
      if (hasCurrentSessionId) {
        currentSessionIdRef.current = nextCurrent;
        setCurrentSessionId(nextCurrent);
      }
      const refreshed = await refreshSessions();
      if (nextCurrent) {
        syncConfigFromSession(nextCurrent, refreshed);
      } else if (hasCurrentSessionId) {
        setMainLocateTarget(null);
        setPendingMainLocateScrollTarget(null);
        followTailRef.current = true;
      }
    } finally {
      setSessionBusy(false);
    }
  }

  async function renameSession(sessionId: string, name: string) {
    const nextName = name.trim();
    if (!sessionId || !nextName) {
      return;
    }
    setSessionBusy(true);
    try {
      const frame = await sendCommand("session_rename", { session_id: sessionId, name: nextName }, null);
      if (!frame) return;
      setSessions((items) => items.map((item) => (
        item.session_id === sessionId
          ? { ...item, name: nextName, updated_at: new Date().toISOString() }
          : item
      )));
      await refreshSessions();
    } finally {
      setSessionBusy(false);
    }
  }

  async function newSession() {
    setSessionBusy(true);
    try {
      const profile = configRef.current ? launchProfileFromConfig(configRef.current) : undefined;
      const frame = await sendCommand("session_new", profile ? { gui_launch_profile: profile } : {});
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

  async function ensureActiveSession(): Promise<string | null> {
    if (currentSessionIdRef.current) {
      return currentSessionIdRef.current;
    }
    const profile = configRef.current ? launchProfileFromConfig(configRef.current) : undefined;
    const frame = await sendCommand("session_new", profile ? { gui_launch_profile: profile } : {}, null);
    const sessionId = optionalPayloadString(framePayload(frame)?.session_id);
    if (!sessionId) {
      return null;
    }
    setCurrentSessionId(sessionId);
    dispatch({
      version: VERSION,
      type: "gui.load_session_history",
      payload: { message_history: [] }
    }, sessionId);
    followTailRef.current = true;
    void refreshSessions().then((refreshed) => syncConfigFromSession(sessionId, refreshed));
    return sessionId;
  }

  async function forkSession(sessionId?: string, messageIndex?: number, contextMessageId?: string) {
    const sourceSessionId = sessionId || currentSessionId;
    if (!sourceSessionId) return;
    setSessionBusy(true);
    try {
      const payload: Record<string, unknown> = { session_id: sourceSessionId };
      if (contextMessageId) {
        payload.context_message_id = contextMessageId;
      } else if (typeof messageIndex === "number") {
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
  forkSessionRef.current = forkSession;

  async function selectHistorySearchResult(result: HistorySearchResult) {
    setSelectedHistoryResult(result);
    setHistoryPreviewBusy(true);
    setHistoryPreviewLocateTarget(historyResultToLocateTarget(result));
    const sessionMeta = sessions.find((session) => session.session_id === result.sessionId) ?? {
      session_id: result.sessionId,
      name: result.sessionName || result.sessionId,
      created_at: result.sessionCreatedAt,
      updated_at: result.sessionUpdatedAt,
      last_checkpoint_event: null,
      components_present: [],
      last_cwd: result.lastCwd ?? null,
    };
    setHistoryPreviewSession(sessionMeta);
    try {
      const frame = await sendReadOnlyCommand("session_history", {
        session_id: result.sessionId,
        read_only: true,
      });
      const payload = framePayload(frame);
      if (!Array.isArray(payload?.message_history)) {
        return;
      }
      setHistoryPreviewNodes(chatNodesFromMessageHistory(payload.message_history));
    } finally {
      setHistoryPreviewBusy(false);
    }
  }

  async function openHistorySearchResult(result: HistorySearchResult | null = selectedHistoryResult) {
    if (!result) return;
    setSessionBusy(true);
    try {
      const frame = await sendCommand("session_switch", { session_id: result.sessionId });
      applySessionHistoryFromFrame(frame);
      const refreshed = await refreshSessions();
      syncConfigFromSession(result.sessionId, refreshed);
      requestMainLocateScroll(historyResultToLocateTarget(result));
      setHistorySearchOpen(false);
      setSessionDialogOpen(false);
    } finally {
      setSessionBusy(false);
    }
  }

  const forkMessage = useCallback((node: ChatNode) => {
    const sessionId = currentSessionIdRef.current;
    if (sessionBusyRef.current || !sessionId) return;
    if (!node.contextMessageId && typeof node.contextMessageIndex !== "number") return;
    void forkSessionRef.current(sessionId, node.contextMessageIndex, node.contextMessageId);
  }, []);

  function openSideComposerFromSelection(anchor = sideSelectionAnchor) {
    if (!anchor || !currentSessionIdRef.current) return;
    setSideThreadDraft("");
    setSideThreadPopover({
      mode: "compose",
      anchor,
      windowRect: resolveSideThreadInitialWindowRect(anchor.rect, currentViewportSize()),
    });
    window.getSelection()?.removeAllRanges();
  }

  async function submitSideThreadQuestion() {
    const popover = sideThreadPopover;
    const question = sideThreadDraft.trim();
    const sessionId = currentSessionIdRef.current;
    if (!popover || !question || !sessionId) return;
    setSideThreadBusy(true);
    try {
      const payload: Record<string, unknown> = {
        session_id: sessionId,
        question,
      };
      if (popover.mode === "compose") {
        payload.quoted_text = popover.anchor.text;
        if (popover.anchor.quotedRange) {
          payload.quoted_range = popover.anchor.quotedRange;
        }
        if (popover.anchor.target.contextMessageId) {
          payload.context_message_id = popover.anchor.target.contextMessageId;
        } else if (typeof popover.anchor.target.contextMessageIndex === "number") {
          payload.message_index = popover.anchor.target.contextMessageIndex;
        } else if (typeof popover.anchor.target.messageIndex === "number") {
          payload.message_index = popover.anchor.target.messageIndex;
        }
      } else if (popover.threadId) {
        payload.side_thread_id = popover.threadId;
      }
      const command = popover.mode === "compose" ? "side_thread_start" : "side_thread_message";
      const frame = await sendSessionCommand(command, payload, sessionId);
      const thread = framePayload(frame)?.side_thread;
      const threadId = isRecord(thread) ? optionalPayloadString(thread.id) : undefined;
      if (thread) {
        dispatch({
          version: VERSION,
          type: "side_thread.update",
          payload: {
            session_id: sessionId,
            side_thread_id: threadId,
            side_thread: thread,
          }
        }, sessionId);
      }
      setSideThreadDraft("");
      if (threadId) {
        setRightSidebarTab("side_threads");
        setSideThreadPopover({ ...popover, mode: "thread", threadId });
      }
    } finally {
      setSideThreadBusy(false);
    }
  }

  function openSideThread(thread: SideThreadState) {
    const sessionId = currentSessionIdRef.current;
    if (!sessionId) return;
    const target = sideThreadLocateTarget(thread, sessionId);
    setMainLocateTarget(target);
    setRightSidebarTab("side_threads");
    window.requestAnimationFrame(() => {
      scrollToHistoryTarget(chatRef.current, target);
      window.requestAnimationFrame(() => {
        const element = findHistoryTargetElement(chatRef.current, target);
        const rect = element?.getBoundingClientRect() ?? fallbackSidePopoverRect();
        setSideThreadPopover({
          mode: "thread",
          threadId: thread.id,
          anchor: {
            text: thread.quotedText,
            quotedRange: thread.quotedRange,
            rect,
            target,
          },
          windowRect: resolveSideThreadInitialWindowRect(rect, currentViewportSize()),
        });
      });
    });
  }

  async function deleteSideThread(thread: SideThreadState) {
    const sessionId = currentSessionIdRef.current;
    if (!sessionId) return;
    if (!window.confirm(`删除旁路对话「${sideThreadTitle(thread)}」？`)) {
      return;
    }
    const frame = await sendSessionCommand(
      "side_thread_delete",
      { session_id: sessionId, side_thread_id: thread.id },
      sessionId,
    );
    if (!frame) return;
    if (framePayload(frame)?.deleted === false) return;
    dispatch({
      version: VERSION,
      type: "side_thread.update",
      payload: {
        session_id: sessionId,
        side_thread_id: thread.id,
        deleted: true,
      }
    }, sessionId);
    setSideThreadPopover((current) => (
      current?.threadId === thread.id ? null : current
    ));
  }

  function setEditingMessageAndRef(
    updater: MessageEditState | null | ((previous: MessageEditState | null) => MessageEditState | null),
  ) {
    setEditingMessage((previous) => {
      const next = typeof updater === "function" ? updater(previous) : updater;
      editingMessageRef.current = next;
      return next;
    });
  }

  function patchEditingMessage(
    nodeId: string,
    patch: Partial<MessageEditState> | ((previous: MessageEditState) => MessageEditState),
  ) {
    setEditingMessageAndRef((previous) => {
      if (!previous || previous.nodeId !== nodeId) return previous;
      return typeof patch === "function" ? patch(previous) : { ...previous, ...patch };
    });
  }

  function cancelEditingMessage() {
    const active = editingMessageRef.current;
    if (active?.busy) return;
    if (active) {
      cleanupEditablePendingPreviews(active.attachments);
    }
    setEditingMessageAndRef(null);
  }

  async function startEditingMessage(node: ChatNode) {
    if (!canEditMessages || !isEditableTextNode(node)) return;
    if (!node.contextMessageId && typeof node.contextMessageIndex !== "number") return;
    const current = editingMessageRef.current;
    if (current?.nodeId === node.id) return;
    if (current) {
      const saved = await saveEditingMessage({ close: false });
      if (!saved) return;
    }
    setEditingMessageAndRef(messageEditStateFromNode(node));
  }

  async function saveEditingMessage({ close = true }: { close?: boolean } = {}): Promise<boolean> {
    const active = editingMessageRef.current;
    if (!active) return true;
    if (active.busy) return false;
    const sessionId = currentSessionIdRef.current;
    if (!sessionId) {
      patchEditingMessage(active.nodeId, { error: "当前没有可编辑的 Session" });
      return false;
    }
    const payload = messageEditTargetPayload(active);
    if (!payload) {
      patchEditingMessage(active.nodeId, { error: "这条消息缺少历史定位信息" });
      return false;
    }

    patchEditingMessage(active.nodeId, { busy: true, error: null });
    try {
      const text = active.draft.trim();
      if (active.role === "agent") {
        if (!text) {
          patchEditingMessage(active.nodeId, { busy: false, error: "Agent 消息不能为空" });
          return false;
        }
        const frame = await sendCommand("session_message_edit", {
          ...payload,
          role: "assistant",
          text,
        }, sessionId);
        if (!frame) {
          patchEditingMessage(active.nodeId, { busy: false, error: "保存失败" });
          return false;
        }
        applySessionHistoryFromFrame(frame);
      } else {
        const contentParts = await buildEditableUserContent(active, sessionId);
        if (contentParts.length === 0) {
          patchEditingMessage(active.nodeId, { busy: false, error: "用户消息需要文本或附件" });
          return false;
        }
        const frame = await sendCommand("session_message_edit", {
          ...payload,
          role: "user",
          text,
          content_parts: contentParts,
        }, sessionId);
        if (!frame) {
          patchEditingMessage(active.nodeId, { busy: false, error: "保存失败" });
          return false;
        }
        applySessionHistoryFromFrame(frame);
      }
      cleanupEditablePendingPreviews(active.attachments);
      if (close) {
        setEditingMessageAndRef(null);
      } else {
        patchEditingMessage(active.nodeId, { busy: false, error: null });
      }
      return true;
    } catch (error) {
      patchEditingMessage(active.nodeId, {
        busy: false,
        error: error instanceof Error ? error.message : String(error),
      });
      return false;
    }
  }

  async function resendEditingMessage(): Promise<boolean> {
    const active = editingMessageRef.current;
    if (!active || active.role !== "user" || active.busy) return false;
    const sessionId = currentSessionIdRef.current;
    if (!sessionId) {
      patchEditingMessage(active.nodeId, { error: "当前没有可重新发送的 Session" });
      return false;
    }
    const payload = messageEditTargetPayload(active);
    if (!payload) {
      patchEditingMessage(active.nodeId, { error: "这条消息缺少历史定位信息" });
      return false;
    }

    patchEditingMessage(active.nodeId, { busy: true, error: null });
    try {
      const text = active.draft.trim();
      const contentParts = await buildEditableUserContent(active, sessionId);
      if (contentParts.length === 0) {
        patchEditingMessage(active.nodeId, { busy: false, error: "重新发送需要文本或附件" });
        return false;
      }

      // Rewind restores persisted conversation history, but plugin runtime state is not rewound.
      // File-change detection plugins may keep their current watchers/state after this point.
      const rewindFrame = await sendCommand("session_rewind", payload, sessionId);
      if (!rewindFrame) {
        patchEditingMessage(active.nodeId, { busy: false, error: "回退历史失败" });
        return false;
      }
      applySessionHistoryFromFrame(rewindFrame);

      const content: string | ContentPart[] = active.attachments.length === 0 ? text : contentParts;
      const enqueueFrame = await sendCommand(
        "enqueue",
        { content, queue: "high_prio", metadata: { intent: "user_resend", source: "gui_message_edit" } },
        sessionId,
      );
      if (!enqueueFrame) {
        patchEditingMessage(active.nodeId, { busy: false, error: "重新发送失败" });
        return false;
      }
      cleanupEditablePendingPreviews(active.attachments);
      setEditingMessageAndRef(null);
      if (text) {
        rememberInputHistory(text);
      }
      return true;
    } catch (error) {
      patchEditingMessage(active.nodeId, {
        busy: false,
        error: error instanceof Error ? error.message : String(error),
      });
      return false;
    }
  }

  function updateEditingMessageDraft(value: string) {
    const active = editingMessageRef.current;
    if (!active || active.busy) return;
    patchEditingMessage(active.nodeId, { draft: value, error: null });
  }

  function addEditingAttachments(files: Iterable<File>) {
    const active = editingMessageRef.current;
    if (!active || active.role !== "user" || active.busy) return;
    const incoming = Array.from(files);
    if (incoming.length === 0) return;

    const rejected: string[] = [];
    const accepted = incoming.filter((file) => {
      if (file.size > MAX_ATTACHMENT_BYTES) {
        rejected.push(file.name || "large file");
        return false;
      }
      return true;
    });
    if (accepted.length === 0) {
      patchEditingMessage(active.nodeId, { error: rejected.length ? "附件大小超过限制" : null });
      return;
    }

    const remaining = MAX_ATTACHMENTS - active.attachments.length;
    if (remaining <= 0) {
      patchEditingMessage(active.nodeId, { error: `最多添加 ${MAX_ATTACHMENTS} 个附件` });
      return;
    }
    const nextFiles = accepted.slice(0, remaining);
    const nextAttachments = nextFiles.map((file): EditableAttachment => {
      const attachment = createPendingAttachment(file);
      return {
        id: `edit-${attachment.id}`,
        kind: "pending",
        attachment,
      };
    });
    patchEditingMessage(active.nodeId, (previous) => ({
      ...previous,
      attachments: [...previous.attachments, ...nextAttachments],
      error: accepted.length > remaining || rejected.length > 0
        ? `已添加 ${nextFiles.length} 个附件，其余未添加`
        : null,
    }));
    for (const item of nextAttachments) {
      if (item.kind === "pending" && item.attachment.partType === "image") {
        void hydrateEditingAttachmentDataUrl(active.nodeId, item.attachment.id, item.attachment.file);
      }
    }
  }

  async function hydrateEditingAttachmentDataUrl(nodeId: string, id: string, file: File) {
    try {
      const bytes = new Uint8Array(await file.arrayBuffer());
      updateEditingPendingAttachment(nodeId, id, {
        dataUrl: bytesToDataUrl(bytes, attachmentMimeType(file)),
      });
    } catch {
      // Object URLs remain as a fallback for preview-only failures.
    }
  }

  function removeEditingAttachment(id: string) {
    const active = editingMessageRef.current;
    if (!active || active.busy) return;
    const removed = active.attachments.find((item) => item.id === id);
    if (removed?.kind === "pending") {
      cleanupEditablePendingPreviews([removed]);
    }
    patchEditingMessage(active.nodeId, (previous) => ({
      ...previous,
      attachments: previous.attachments.filter((item) => item.id !== id),
      error: null,
    }));
  }

  function cleanupEditablePendingPreviews(items: EditableAttachment[]) {
    const retainedPreviews = new Set(Object.values(blobPreviewUrlsRef.current));
    for (const item of items) {
      if (item.kind !== "pending") continue;
      const previewUrl = item.attachment.previewUrl;
      if (!retainedPreviews.has(previewUrl)) {
        revokeObjectUrl(previewUrl);
      }
    }
  }

  function updateEditingPendingAttachment(
    nodeId: string,
    attachmentId: string,
    patch: Partial<PendingAttachment>,
  ) {
    patchEditingMessage(nodeId, (previous) => ({
      ...previous,
      attachments: previous.attachments.map((item) => (
        item.kind === "pending" && item.attachment.id === attachmentId
          ? { ...item, attachment: { ...item.attachment, ...patch } }
          : item
      )),
    }));
  }

  async function buildEditableUserContent(
    edit: MessageEditState,
    sessionId: string,
  ): Promise<ContentPart[]> {
    const parts: ContentPart[] = [];
    const text = edit.draft.trim();
    if (text) {
      parts.push({ type: "text", text });
    }
    for (const item of edit.attachments) {
      if (item.kind === "existing") {
        parts.push(cloneContentPart(item.part));
        continue;
      }
      const source = await uploadAttachment(
        item.attachment,
        sessionId,
        (attachmentId, patch) => updateEditingPendingAttachment(edit.nodeId, attachmentId, patch),
      );
      parts.push(contentPartFromAttachment(item.attachment, source));
    }
    return parts;
  }

  const messageEditController: MessageEditController = {
    active: editingMessage,
    canEdit: canEditMessages,
    maxHeight: chatPanelHeight,
    onStart: (node) => {
      void startEditingMessage(node);
    },
    onCancel: cancelEditingMessage,
    onSave: () => {
      void saveEditingMessage();
    },
    onResend: () => {
      void resendEditingMessage();
    },
    onDraftChange: updateEditingMessageDraft,
    onAddFiles: addEditingAttachments,
    onRemoveAttachment: removeEditingAttachment,
  };

  async function saveGlobalAndSet(nextConfig: PersistedConfig) {
    const saved = await window.hawi.saveConfig(nextConfig);
    configRef.current = saved;
    setConfig(saved);
    setMetadata((current) => current ? { ...current, config: saved } : current);
    return saved;
  }

  async function restartWith(nextConfig: PersistedConfig) {
    setConfig(nextConfig);
    try {
      await restartCoreAndSet(nextConfig);
    } catch (error) {
      dispatch(errorFrame(error));
    }
  }

  async function restartCoreAndSet(nextConfig: PersistedConfig) {
    await window.hawi.restartCore(nextConfig);
    setMetadata((current) => current ? { ...current, config: nextConfig, coreRunning: true } : current);
  }

  function inputHistorySessionKey(sessionId = currentSessionIdRef.current) {
    return sessionId ?? UNLOADED_INPUT_HISTORY_KEY;
  }

  function resetInputHistoryNavigation(draft = "") {
    inputHistoryNavigationRef.current = {
      sessionKey: inputHistorySessionKey(),
      index: null,
      draft
    };
  }

  function rememberInputHistory(text: string) {
    const entry = text.trim();
    if (!entry) return;
    const key = inputHistorySessionKey();
    const existing = inputHistoryBySessionRef.current[key] ?? [];
    inputHistoryBySessionRef.current[key] = mergeInputHistory(existing, [entry]);
  }

  function currentInputHistory() {
    const key = inputHistorySessionKey();
    return mergeInputHistory(
      inputHistoryFromChatNodes(state.nodes),
      inputHistoryBySessionRef.current[key] ?? []
    );
  }

  function focusInputAtEnd() {
    window.requestAnimationFrame(() => {
      const inputElement = inputRef.current;
      if (!inputElement) return;
      inputElement.focus();
      inputElement.setSelectionRange(inputElement.value.length, inputElement.value.length);
    });
  }

  function browseInputHistory(direction: "previous" | "next") {
    const history = currentInputHistory();
    if (history.length === 0) return false;
    const sessionKey = inputHistorySessionKey();
    const navigation = inputHistoryNavigationRef.current;
    const active = navigation.sessionKey === sessionKey && navigation.index !== null;
    const draft = active ? navigation.draft : input;
    const currentIndex = active
      ? Math.min(Math.max(navigation.index ?? history.length - 1, 0), history.length - 1)
      : null;
    const nextIndex = direction === "previous"
      ? (currentIndex === null ? history.length - 1 : Math.max(currentIndex - 1, 0))
      : (currentIndex === null ? null : currentIndex + 1);

    if (nextIndex === null) {
      return false;
    }
    if (nextIndex >= history.length) {
      inputHistoryNavigationRef.current = { sessionKey, index: null, draft: "" };
      setInput(draft);
      focusInputAtEnd();
      return true;
    }

    inputHistoryNavigationRef.current = { sessionKey, index: nextIndex, draft };
    setInput(history[nextIndex]);
    focusInputAtEnd();
    return true;
  }

  function handleMainInputKeyDown(event: ReactKeyboardEvent<HTMLTextAreaElement>) {
    if (shouldSubmitInputFromKeyEvent(event, inputComposingRef.current)) {
      event.preventDefault();
      void submitInput();
      return;
    }
    const historyDirection = shouldNavigateInputHistoryFromKeyEvent(
      event,
      event.currentTarget.value,
      event.currentTarget.selectionStart,
      event.currentTarget.selectionEnd,
      inputComposingRef.current,
      inputHistoryNavigationRef.current.index !== null
    );
    if (historyDirection && browseInputHistory(historyDirection)) {
      event.preventDefault();
    }
  }

  function addAttachments(files: Iterable<File>) {
    const incoming = Array.from(files);
    if (incoming.length === 0) return;

    const rejected: string[] = [];
    const accepted = incoming.filter((file) => {
      if (file.size > MAX_ATTACHMENT_BYTES) {
        rejected.push(file.name || "large file");
        return false;
      }
      return true;
    });

    if (accepted.length === 0) {
      setInputAttachmentError(rejected.length ? "附件大小超过限制" : null);
      return;
    }

    const previous = inputAttachmentsRef.current;
    const remaining = MAX_ATTACHMENTS - previous.length;
    if (remaining <= 0) {
      setInputAttachmentError(`最多添加 ${MAX_ATTACHMENTS} 个附件`);
      return;
    }

    const nextFiles = accepted.slice(0, remaining);
    const nextAttachments = nextFiles.map(createPendingAttachment);
    setInputAttachmentsAndRef([...previous, ...nextAttachments]);
    for (const attachment of nextAttachments) {
      if (attachment.partType === "image") {
        void hydrateAttachmentDataUrl(attachment.id, attachment.file);
      }
    }
    if (accepted.length > remaining || rejected.length > 0) {
      setInputAttachmentError(`已添加 ${nextFiles.length} 个附件，其余未添加`);
    } else {
      setInputAttachmentError(null);
    }
  }

  async function hydrateAttachmentDataUrl(id: string, file: File) {
    try {
      const bytes = new Uint8Array(await file.arrayBuffer());
      updateInputAttachment(id, {
        dataUrl: bytesToDataUrl(bytes, attachmentMimeType(file)),
      });
    } catch {
      // Object URLs remain as a fallback for preview-only failures.
    }
  }

  function rememberBlobPreview(source: BlobSource, previewUrl: string, dataUrl?: string) {
    const preview = dataUrl || previewUrl;
    const keys = [
      source.blob_id,
      source.uri,
      `hawi-blob://${source.blob_id}`,
    ].filter((key): key is string => typeof key === "string" && key.length > 0);
    const nextBlobPreviewUrls = { ...blobPreviewUrlsRef.current };
    for (const key of keys) {
      nextBlobPreviewUrls[key] = preview;
    }
    blobPreviewUrlsRef.current = nextBlobPreviewUrls;
    setBlobPreviewUrls(nextBlobPreviewUrls);
  }

  async function ensureBlobPreview(request: BlobPreviewRequest) {
    const blobId = request.blobId;
    if (
      blobPreviewUrlsRef.current[blobId]
      || fetchingBlobPreviewIdsRef.current.has(blobId)
      || failedBlobPreviewIdsRef.current.has(blobId)
    ) {
      return;
    }

    fetchingBlobPreviewIdsRef.current.add(blobId);
    const dataB64Promise = createBlobPreviewFetch(blobId);
    dataB64Promise.catch(() => undefined);

    try {
      const frame = await sendCommand(
        "blob.fetch",
        { blob_id: blobId, chunk_size: BLOB_CHUNK_SIZE },
        currentSessionIdRef.current,
      );
      if (!frame) {
        throw new Error("blob.fetch failed");
      }
      const dataB64 = await dataB64Promise;
      const previewUrl = `data:${request.mimeType || "image/png"};base64,${dataB64}`;
      const nextBlobPreviewUrls = {
        ...blobPreviewUrlsRef.current,
        [blobId]: previewUrl,
        [`hawi-blob://${blobId}`]: previewUrl,
      };
      if (request.uri) {
        nextBlobPreviewUrls[request.uri] = previewUrl;
      }
      blobPreviewUrlsRef.current = nextBlobPreviewUrls;
      setBlobPreviewUrls(nextBlobPreviewUrls);
    } catch {
      dropPendingBlobPreviewFetch(blobId);
      failedBlobPreviewIdsRef.current.add(blobId);
    } finally {
      fetchingBlobPreviewIdsRef.current.delete(blobId);
    }
  }

  function createBlobPreviewFetch(blobId: string): Promise<string> {
    return new Promise((resolve, reject) => {
      const timeoutId = window.setTimeout(() => {
        pendingBlobPreviewFetchesRef.current.delete(blobId);
        fetchingBlobPreviewIdsRef.current.delete(blobId);
        failedBlobPreviewIdsRef.current.add(blobId);
        reject(new Error(`Timed out fetching blob preview: ${blobId}`));
      }, BLOB_PREVIEW_FETCH_TIMEOUT_MS);
      pendingBlobPreviewFetchesRef.current.set(blobId, {
        chunks: [],
        timeoutId,
        resolve,
        reject,
      });
    });
  }

  function dropPendingBlobPreviewFetch(blobId: string) {
    const pending = pendingBlobPreviewFetchesRef.current.get(blobId);
    if (!pending) return;
    window.clearTimeout(pending.timeoutId);
    pendingBlobPreviewFetchesRef.current.delete(blobId);
  }

  function removeInputAttachment(id: string) {
    if (inputUploading) return;
    const previous = inputAttachmentsRef.current;
    const removed = previous.find((attachment) => attachment.id === id);
    if (removed) {
      if (removed.blobSource) {
        const nextPreviewUrls = { ...blobPreviewUrlsRef.current };
        delete nextPreviewUrls[removed.blobSource.blob_id];
        delete nextPreviewUrls[removed.blobSource.uri];
        delete nextPreviewUrls[`hawi-blob://${removed.blobSource.blob_id}`];
        blobPreviewUrlsRef.current = nextPreviewUrls;
        setBlobPreviewUrls(nextPreviewUrls);
      }
      revokeObjectUrl(removed.previewUrl);
    }
    setInputAttachmentsAndRef(previous.filter((attachment) => attachment.id !== id));
  }

  function clearInputAttachments({ preserveUploaded = false }: { preserveUploaded?: boolean } = {}) {
    const retainedPreviews = new Set(Object.values(blobPreviewUrlsRef.current));
    for (const attachment of inputAttachmentsRef.current) {
      if (preserveUploaded && attachment.blobSource) {
        continue;
      }
      if (!retainedPreviews.has(attachment.previewUrl)) {
        revokeObjectUrl(attachment.previewUrl);
      }
    }
    setInputAttachmentsAndRef([]);
    setInputAttachmentError(null);
  }

  function setInputAttachmentsAndRef(
    updater: PendingAttachment[] | ((previous: PendingAttachment[]) => PendingAttachment[]),
  ) {
    setInputAttachments((previous) => {
      const next = typeof updater === "function" ? updater(previous) : updater;
      inputAttachmentsRef.current = next;
      return next;
    });
  }

  function handleMainInputPaste(event: ReactClipboardEvent<HTMLTextAreaElement>) {
    const files = attachmentFilesFromFileList(event.clipboardData.files);
    if (files.length === 0) return;
    event.preventDefault();
    addAttachments(files);
  }

  function handleMainInputDrop(event: ReactDragEvent<HTMLTextAreaElement>) {
    const files = attachmentFilesFromFileList(event.dataTransfer.files);
    if (files.length === 0) return;
    event.preventDefault();
    setInputDropActive(false);
    addAttachments(files);
  }

  function handleMainInputDragOver(event: ReactDragEvent<HTMLTextAreaElement>) {
    if (!hasAttachmentFile(event.dataTransfer)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
    setInputDropActive(true);
  }

  function handleMainInputDragLeave(event: ReactDragEvent<HTMLTextAreaElement>) {
    const nextTarget = event.relatedTarget;
    if (nextTarget instanceof Node && event.currentTarget.contains(nextTarget)) return;
    setInputDropActive(false);
  }

  function handleAttachmentFileInputChange(event: ReactChangeEvent<HTMLInputElement>) {
    addAttachments(attachmentFilesFromFileList(event.target.files));
    event.target.value = "";
  }

  async function buildComposerContent(
    text: string,
    sessionId: string,
  ): Promise<string | ContentPart[]> {
    const attachments = inputAttachmentsRef.current;
    if (attachments.length === 0) {
      return text;
    }

    const parts: ContentPart[] = [];
    if (text) {
      parts.push({ type: "text", text });
    }
    for (const attachment of attachments) {
      const source = await uploadAttachment(attachment, sessionId);
      parts.push(contentPartFromAttachment(attachment, source));
    }
    return parts;
  }

  async function uploadAttachment(
    attachment: PendingAttachment,
    sessionId: string,
    updateAttachment: (id: string, patch: Partial<PendingAttachment>) => void = updateInputAttachment,
  ): Promise<BlobSource> {
    if (attachment.blobSource) {
      return attachment.blobSource;
    }

    updateAttachment(attachment.id, { status: "uploading", progress: 0.02, error: undefined });
    try {
      const bytes = new Uint8Array(await attachment.file.arrayBuffer());
      const sha256 = await sha256Hex(bytes);
      const dataUrl = attachment.partType === "image"
        ? attachment.dataUrl ?? bytesToDataUrl(bytes, attachment.mimeType || "image/png")
        : undefined;
      if (dataUrl) {
        updateAttachment(attachment.id, { dataUrl });
      }
      updateAttachment(attachment.id, { progress: 0.08 });

      const initFrame = await sendCommand("blob.upload_init", {
        direction: "inbound",
        sha256,
        size: bytes.byteLength,
        mime: attachment.mimeType,
      }, sessionId);
      const blobId = isRecord(initFrame?.payload)
        ? optionalRecordString(initFrame.payload.blob_id)
        : undefined;
      if (!blobId) {
        throw new Error("blob.upload_init did not return blob_id");
      }

      const totalChunks = Math.max(1, Math.ceil(bytes.byteLength / BLOB_CHUNK_SIZE));
      for (let seq = 0; seq < totalChunks; seq += 1) {
        const start = seq * BLOB_CHUNK_SIZE;
        const end = Math.min(bytes.byteLength, start + BLOB_CHUNK_SIZE);
        const chunkFrame = await sendCommand("blob.upload_chunk", {
          blob_id: blobId,
          seq,
          data_b64: bytesToBase64(bytes.subarray(start, end)),
        }, sessionId);
        if (!chunkFrame) {
          throw new Error("blob.upload_chunk failed");
        }
        updateAttachment(attachment.id, {
          progress: 0.08 + ((seq + 1) / totalChunks) * 0.84,
        });
      }

      const finalizeFrame = await sendCommand("blob.upload_finalize", { blob_id: blobId }, sessionId);
      if (!finalizeFrame) {
        throw new Error("blob.upload_finalize failed");
      }
      const payload = normalizeBlobFinalizePayload(finalizeFrame?.payload, blobId, attachment, sha256);
      const source: BlobSource = {
        kind: "blob",
        blob_id: payload.blob_id,
        uri: payload.uri ?? `hawi-blob://${payload.blob_id}`,
        mime_type: payload.mime ?? attachment.mimeType,
        filename: attachment.filename,
        size: payload.size ?? attachment.size,
        sha256: payload.sha256 ?? sha256,
        direction: payload.direction ?? "inbound",
      };

      updateAttachment(attachment.id, {
        status: "uploaded",
        progress: 1,
        blobSource: source,
        error: undefined,
      });
      rememberBlobPreview(source, attachment.previewUrl, dataUrl);
      return source;
    } catch (error) {
      updateAttachment(attachment.id, {
        status: "error",
        progress: 0,
        error: error instanceof Error ? error.message : String(error),
      });
      throw error;
    }
  }

  function updateInputAttachment(id: string, patch: Partial<PendingAttachment>) {
    setInputAttachmentsAndRef((previous) => (
      previous.map((attachment) => (
        attachment.id === id ? { ...attachment, ...patch } : attachment
      ))
    ));
  }

  async function submitInput() {
    const text = input.trim();
    const hasAttachments = inputAttachmentsRef.current.length > 0;
    if ((!text && !hasAttachments) || inputUploading) return;
    if (text.startsWith("/") && !hasAttachments) {
      await runSlashCommand(text);
      setInput("");
      resetInputHistoryNavigation();
      return;
    }
    const sessionId = await ensureActiveSession();
    if (!sessionId) {
      return;
    }
    setInputUploading(true);
    try {
      const content = await buildComposerContent(text, sessionId);
      const frame = await sendCommand(
        "enqueue",
        { content, queue: "high_prio", metadata: { intent: "user_send", source: "gui_main_input" } },
        sessionId
      );
      if (frame) {
        setInput("");
        resetInputHistoryNavigation();
        clearInputAttachments({ preserveUploaded: true });
        if (text) {
          rememberInputHistory(text);
        }
      }
    } finally {
      setInputUploading(false);
    }
  }

  async function resumeConversation() {
    const text = input.trim();
    const hasAttachments = inputAttachmentsRef.current.length > 0;
    if (inputUploading) return;
    setInputUploading(true);
    try {
      const sessionId = await ensureActiveSession();
      if (!sessionId) return;
      const content = hasAttachments
        ? await buildComposerContent(text, sessionId)
        : text;
      const frame = await sendCommand("resume", resumePayloadFromContent(content), sessionId);
      if (frame) {
        if (text || hasAttachments) {
          setInput("");
          resetInputHistoryNavigation();
          clearInputAttachments({ preserveUploaded: true });
        }
        if (text) {
          rememberInputHistory(text);
        }
      }
    } finally {
      setInputUploading(false);
    }
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

  function queueMessageEditorContent(message: QueueMessageState) {
    return (message.content ?? message.contentPreview).trim();
  }

  function appendWithdrawnQueueMessageToInput(message: QueueMessageState) {
    const content = queueMessageEditorContent(message);
    if (!content) return;
    const currentText = inputRef.current?.value ?? input;
    const nextInput = currentText ? `${currentText}\n${content}` : content;
    setInput(nextInput);
    resetInputHistoryNavigation(nextInput);
    focusInputAtEnd();
  }

  async function removeQueuedMessage(message: QueueMessageState) {
    if (!message.id || queueTaskBusy) return;
    setQueueTaskBusy(true);
    try {
      const frame = await sendCommand("queue_message_remove", { message_id: message.id });
      if (!frame) return;
      if (editingQueueTaskId === message.id) {
        setEditingQueueTaskId(null);
        setQueueTaskEditDraft("");
      }
      appendWithdrawnQueueMessageToInput(message);
      await refreshRuntimeStatus();
    } finally {
      setQueueTaskBusy(false);
    }
  }

  async function promoteQueueTask(message: QueueMessageState) {
    if (!message.id || queueTaskBusy) return;
    setQueueTaskBusy(true);
    try {
      const frame = await sendCommand("queue_message_promote", { message_id: message.id });
      if (!frame) return;
      if (editingQueueTaskId === message.id) {
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
      const frame = await sendCommand("queue_message_remove", { message_id: message.id });
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
    if (contextCompactBusy || !canCompactContextManually) return;
    setContextCompactBusy(true);
    try {
      const frame = await sendCommand("compact_context", {});
      if (!frame) return;
      setContextPopoverOpen(false);
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

  async function setAutoCompactThreshold(percent: number) {
    if (contextSettingsBusy) return;
    const maxContextTokens = state.contextAutoCompact?.maxContextTokens ?? state.contextUsage?.maxContextTokens;
    if (!maxContextTokens || maxContextTokens <= 0) {
      dispatch(errorFrame(new Error("当前模型缺少 context window 信息，无法设置自动压缩阈值")));
      return;
    }
    const ratio = Math.min(1, Math.max(0.01, percent / 100));
    setContextSettingsBusy(true);
    try {
      const frame = await sendCommand("set_auto_compact", {
        trigger_tokens: Math.max(1, Math.round(maxContextTokens * ratio)),
        trigger_ratio: ratio
      });
      if (!frame) return;
      await refreshRuntimeStatus();
    } finally {
      setContextSettingsBusy(false);
    }
  }

  function applySystemPrompt(nextConfig: PersistedConfig) {
    if (!currentSessionIdRef.current) {
      configRef.current = nextConfig;
      void saveGlobalAndSet(nextConfig);
      return;
    }
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

  function requestSystemPromptEditConfirmation() {
    if (!systemPromptRequiresConfirmation) return;
    setSystemPromptConfirmOpen(true);
  }

  function confirmSystemPromptEdit() {
    setSystemPromptEditAcknowledgement({
      sessionId: currentSessionId,
      messageCount: state.sessionMessageCount
    });
    setSystemPromptConfirmOpen(false);
    if (typeof window === "undefined") return;
    window.requestAnimationFrame(() => {
      systemPromptRef.current?.focus({ preventScroll: true });
    });
  }

  function confirmPluginSettingsApply() {
    const pendingApply = pendingPluginSettingsApply;
    setPluginSettingsApplyAcknowledgement({
      sessionId: currentSessionId,
      messageCount: state.sessionMessageCount
    });
    setPluginSettingsConfirmOpen(false);
    setPendingPluginSettingsApply(null);
    if (!pendingApply) return;
    void performApplyPlugins(pendingApply.selectedPlugins, pendingApply.pluginConfigs);
  }

  function handleSystemPromptPointerDown(event: ReactMouseEvent<HTMLTextAreaElement>) {
    if (!systemPromptRequiresConfirmation) return;
    event.preventDefault();
    requestSystemPromptEditConfirmation();
  }

  function handleSystemPromptFocus() {
    if (!systemPromptRequiresConfirmation) return;
    requestSystemPromptEditConfirmation();
    systemPromptRef.current?.blur();
  }

  async function selectModel(modelName: string) {
    if (!config || !metadata) return;
    const nextConfig = { ...config, modelName };
    setModelDialogOpen(false);
    setConfig(nextConfig);
    try {
      if (currentSessionIdRef.current) {
        await sendCommand("switch_model", { model_name: modelName });
      } else {
        await saveGlobalAndSet(nextConfig);
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

  async function applyModelProviderConfig(provider: string, providerConfig: Record<string, unknown>) {
    if (!config) return;
    const baseConfig = configRef.current ?? config;
    if (Object.keys(providerConfig).length > 0) {
      const frame = await window.hawi.sendCommand("save_model_provider_config", {
        provider,
        properties: providerConfig,
      }, currentSessionIdRef.current);
      const payload = frame.payload as Record<string, unknown>;
      setMetadata((current) => {
        if (!current) return current;
        return {
          ...current,
          inspect: {
            ...current.inspect,
            models: Array.isArray(payload.models)
              ? payload.models.filter((item): item is string => typeof item === "string")
              : current.inspect.models,
            model_provider_configs: isRecord(payload.model_provider_configs)
              ? payload.model_provider_configs as Record<string, ModelProviderConfigPreview>
              : current.inspect.model_provider_configs,
          },
        };
      });
    }
    const nextProviderConfigs = { ...(baseConfig.modelProviderConfigs ?? {}) };
    delete nextProviderConfigs[provider];
    const nextConfig = { ...baseConfig, modelProviderConfigs: nextProviderConfigs };
    await saveGlobalAndSet(nextConfig);
    dispatch(metaFrame(`已保存 ${provider} 的模型配置`));
  }

  async function previewPluginTools(pluginKey: string, pluginConfig: Record<string, unknown>) {
    return window.hawi.previewPluginTools(pluginKey, pluginConfig);
  }

  async function applyPlugins(selectedPlugins: string[], pluginConfigs: Record<string, Record<string, unknown>>) {
    if (pluginSettingsApplyRequiresConfirmation) {
      setPendingPluginSettingsApply({
        selectedPlugins,
        pluginConfigs
      });
      setPluginSettingsConfirmOpen(true);
      return;
    }
    await performApplyPlugins(selectedPlugins, pluginConfigs);
  }

  async function performApplyPlugins(selectedPlugins: string[], pluginConfigs: Record<string, Record<string, unknown>>) {
    if (!config) return;
    if (!currentSessionIdRef.current) {
      const nextConfig = { ...(configRef.current ?? config), selectedPlugins, pluginConfigs: normalizePluginConfigs(pluginConfigs) };
      const savedConfig = await saveGlobalAndSet(nextConfig);
      configRef.current = savedConfig;
      setPluginDialogOpen(false);
      return;
    }
    const frame = await sendCommand("apply_plugins", { selected_plugins: selectedPlugins, plugin_configs: pluginConfigs });
    if (!frame) return;
    const payload = framePayload(frame);
    if (!payload) return;
    const appliedPlugins = Array.isArray(payload.selected_plugins)
      ? payload.selected_plugins.filter((item): item is string => typeof item === "string")
      : selectedPlugins;
    const appliedConfigs = normalizePluginConfigs(payload.plugin_configs ?? pluginConfigs);
    const nextConfig = { ...(configRef.current ?? config), selectedPlugins: appliedPlugins, pluginConfigs: appliedConfigs };
    const savedConfig = await saveGlobalAndSet(nextConfig);
    configRef.current = savedConfig;
    setPluginDialogOpen(false);
  }

  async function exportCurrentSession(format: "markdown" | "jsonl") {
    if (!currentSessionId || exportBusy) return;
    setExportMenuOpen(false);
    setSettingsMenuOpen(false);
    setExportBusy(true);
    try {
      if (format === "markdown") {
        const frame = await sendSessionCommand(
          "session_export_markdown",
          { session_id: currentSessionId },
          currentSessionId
        );
        const exportPayload = normalizeMarkdownExportPayload(frame?.payload?.export);
        if (!exportPayload) {
          throw new Error("导出结果为空");
        }
        const saved = await window.hawi.saveMarkdownExport(exportPayload);
        if (!saved.canceled && saved.markdownPath) {
          dispatch(metaFrame(`Markdown 已导出：${saved.markdownPath}`));
        }
        return;
      }

      const savedFrame = await sendSessionCommand("session_save_now", { session_id: currentSessionId }, currentSessionId);
      if (!savedFrame) return;
      const frame = await sendSessionCommand("session_history", { session_id: currentSessionId }, currentSessionId);
      const payload = framePayload(frame);
      if (!Array.isArray(payload?.message_history)) {
        throw new Error("JSONL 导出结果为空");
      }
      const exportPayload: JsonlExportPayload = {
        suggested_filename: `hawi-session-${currentSessionId}.jsonl`,
        records: payload.message_history
      };
      const saved = await window.hawi.saveJsonlExport(exportPayload);
      if (!saved.canceled && saved.jsonlPath) {
        dispatch(metaFrame(`JSONL 已导出：${saved.jsonlPath}`));
      }
    } catch (error) {
      dispatch(errorFrame(error));
    } finally {
      setExportBusy(false);
    }
  }

  async function changeWorkingDirectory() {
    setCwdBusy(true);
    try {
      const selected = await window.hawi.selectWorkingDirectory();
      if (selected.canceled || !selected.path) {
        return;
      }
      const frame = await sendCommand("change_cwd", { cwd: selected.path }, null);
      const payload = framePayload(frame);
      const sessionId = optionalPayloadString(payload?.session_id);
      const nextCwd = optionalPayloadString(payload?.last_cwd) ?? selected.path;
      setMetadata((current) => current ? { ...current, currentWorkspaceRoot: nextCwd } : current);
      if (sessionId && payload?.workspace_switched === true) {
        setCurrentSessionId(sessionId);
        dispatch({
          version: VERSION,
          type: "gui.load_session_history",
          payload: { message_history: [] }
        }, sessionId);
        followTailRef.current = true;
      }
      const refreshed = await refreshSessions();
      if (sessionId && payload?.workspace_switched === true) {
        syncConfigFromSession(sessionId, refreshed);
      }
    } catch (error) {
      dispatch(errorFrame(error));
    } finally {
      setCwdBusy(false);
    }
  }

  function updateShowDebug(enabled: boolean) {
    const baseConfig = configRef.current ?? config;
    if (!baseConfig) return;
    const next = { ...baseConfig, showDebug: enabled };
    setConfig(next);
    void saveGlobalAndSet(next);
  }

  function updateFocusModeEnabled(enabled: boolean) {
    const baseConfig = configRef.current ?? config;
    if (!baseConfig) return;
    const next = { ...baseConfig, focusModeEnabled: enabled };
    setConfig(next);
    void saveGlobalAndSet(next);
  }

  function requestToolCallPurposeEnabledChange(enabled: boolean) {
    const baseConfig = configRef.current ?? config;
    if (!baseConfig) return;
    if (shouldConfirmToolCallPurposeChange({
      hasConversation: currentSessionHasConversation,
      currentEnabled: baseConfig.toolCallPurposeEnabled,
      nextEnabled: enabled
    })) {
      setPendingToolCallPurposeEnabled(enabled);
      return;
    }
    applyToolCallPurposeEnabled(enabled);
  }

  function applyToolCallPurposeEnabled(enabled: boolean) {
    const baseConfig = configRef.current ?? config;
    if (!baseConfig) return;
    const next = { ...baseConfig, toolCallPurposeEnabled: enabled };
    configRef.current = next;
    setConfig(next);
    void saveGlobalAndSet(next);
  }

  function confirmToolCallPurposeChange() {
    const enabled = pendingToolCallPurposeEnabled;
    setPendingToolCallPurposeEnabled(null);
    if (enabled === null) return;
    applyToolCallPurposeEnabled(enabled);
  }

  function updateProfilingEnabled(enabled: boolean) {
    const baseConfig = configRef.current ?? config;
    if (!baseConfig) return;
    const next = { ...baseConfig, profilingEnabled: enabled };
    configRef.current = next;
    setConfig(next);
    void (async () => {
      try {
        if (currentSessionIdRef.current) {
          await sendCommand("set_profiling", { profiling_enabled: enabled });
        }
        await saveGlobalAndSet(next);
      } catch (error) {
        configRef.current = baseConfig;
        setConfig(baseConfig);
        dispatch(errorFrame(error));
      }
    })();
  }

  if (!metadata || !config) {
    return <div className="boot">Loading Hawi metadata...</div>;
  }

  const currentProjectPath = sessions.find((session) => session.session_id === currentSessionId)?.last_cwd
    ?? metadata.currentWorkspaceRoot
    ?? null;
  const exportDisabled = !currentSessionId || state.sessionMessageCount === 0 || exportBusy;

  return (
    <div className="app-shell shadcn-workbench" ref={appShellRef}>
      <aside className="app-rail" aria-label="Hawi navigation">
        <div className="rail-brand" title="Hawi">
          <span className="rail-brand-mark" aria-hidden="true"><Bot size={19} /></span>
          <strong>Hawi</strong>
        </div>
        <nav className="rail-actions" aria-label="Primary actions">
          <button
            type="button"
            className="rail-button"
            title="切换模型"
            onClick={() => {
              setExportMenuOpen(false);
              setSettingsMenuOpen(false);
              setModelDialogOpen(true);
            }}
          >
            <Brain size={18} />
            <span>Model</span>
          </button>
          <button
            type="button"
            className="rail-button"
            title="插件配置"
            onClick={() => {
              setExportMenuOpen(false);
              setSettingsMenuOpen(false);
              setPluginDialogOpen(true);
            }}
          >
            <Plug size={18} />
            <span>Plugins</span>
          </button>
          <div className="rail-export-anchor" ref={exportMenuAnchorRef}>
            <button
              type="button"
              className={`rail-button ${exportMenuOpen ? "active" : ""}`}
              title={exportDisabled ? "当前会话没有可导出的消息记录" : "导出消息记录"}
              aria-haspopup="menu"
              aria-expanded={exportMenuOpen}
              disabled={exportDisabled}
              onClick={() => {
                setSettingsMenuOpen(false);
                setExportMenuOpen((open) => !open);
              }}
            >
              <FileText size={18} />
              <span>{exportBusy ? "Exporting" : "Export"}</span>
            </button>
            {exportMenuOpen && (
              <div
                ref={exportMenuRef}
                className="rail-export-menu menu-popover"
                role="menu"
                style={exportMenuStyle}
              >
                <button
                  type="button"
                  className="menu-item"
                  disabled={exportDisabled}
                  onClick={() => void exportCurrentSession("jsonl")}
                >
                  <FileText size={15} /> 导出 JSONL
                </button>
                <button
                  type="button"
                  className="menu-item"
                  disabled={exportDisabled}
                  onClick={() => void exportCurrentSession("markdown")}
                >
                  <FileText size={15} /> 导出 Markdown
                </button>
              </div>
            )}
          </div>
          <div className="rail-settings-anchor" ref={settingsMenuAnchorRef}>
            <button
              type="button"
              className={`rail-button ${settingsMenuOpen ? "active" : ""}`}
              title="设置"
              aria-haspopup="menu"
              aria-expanded={settingsMenuOpen}
              onClick={() => {
                setExportMenuOpen(false);
                setSettingsMenuOpen((open) => !open);
              }}
            >
              <Settings size={18} />
              <span>Settings</span>
            </button>
            {settingsMenuOpen && (
              <div
                ref={settingsMenuRef}
                className="rail-settings-menu menu-popover"
                role="menu"
                style={settingsMenuStyle}
              >
                <div className="rail-menu-section rail-system-prompt-section">
                  <strong>System Prompt</strong>
                  <textarea
                    ref={systemPromptRef}
                    className="rail-system-prompt-input"
                    rows={4}
                    value={config.systemPrompt}
                    readOnly={systemPromptRequiresConfirmation}
                    title={systemPromptRequiresConfirmation ? "当前会话已有消息，点击确认后可编辑 System Prompt" : "System Prompt"}
                    aria-label="System Prompt"
                    aria-readonly={systemPromptRequiresConfirmation}
                    onMouseDown={handleSystemPromptPointerDown}
                    onFocus={handleSystemPromptFocus}
                    onChange={(event) => {
                      if (systemPromptRequiresConfirmation) {
                        requestSystemPromptEditConfirmation();
                        return;
                      }
                      resizeTextareaToRows(event.currentTarget, SYSTEM_PROMPT_MAX_ROWS);
                      const nextConfig = { ...config, systemPrompt: event.target.value };
                      setConfig(nextConfig);
                      applySystemPrompt(nextConfig);
                    }}
                  />
                </div>
                <div className="rail-menu-section">
                  <strong>Display</strong>
                  <label className="menu-item checkbox-menu-item" title="折叠每轮中的中间过程，只保留最后一条正式回复">
                    <input
                      type="checkbox"
                      checked={focusModeEnabled}
                      onChange={(event) => updateFocusModeEnabled(event.target.checked)}
                    />
                    专注模式
                  </label>
                  <label className="menu-item checkbox-menu-item">
                    <input
                      type="checkbox"
                      checked={showDebug}
                      onChange={(event) => updateShowDebug(event.target.checked)}
                    />
                    调试信息
                  </label>
                  <label className="menu-item checkbox-menu-item">
                    <input
                      type="checkbox"
                      checked={profilingEnabled}
                      onChange={(event) => updateProfilingEnabled(event.target.checked)}
                    />
                    Profiling
                  </label>
                  <label
                    className="menu-item checkbox-menu-item"
                    title={toolCallPurposeControl.title}
                  >
                    <input
                      type="checkbox"
                      checked={config.toolCallPurposeEnabled}
                      disabled={toolCallPurposeControl.disabled}
                      onChange={(event) => requestToolCallPurposeEnabledChange(event.target.checked)}
                    />
                    工具调用目的
                  </label>
                </div>
              </div>
            )}
          </div>
        </nav>
        <div className="rail-footer">
          <button
            type="button"
            className="rail-button"
            title="重启 Engine"
            onClick={() => void restartWith(config)}
          >
            <RotateCcw size={18} />
            <span>Restart</span>
          </button>
          <div className={`rail-engine ${coreRunning ? "running" : "idle"}`} title={coreRunning ? "Engine online" : "Engine idle"}>
            <span aria-hidden="true" />
            <strong>{coreRunning ? "On" : "Idle"}</strong>
          </div>
        </div>
      </aside>

      <main className="main-shell">
      <header className="topbar">
        <div className="brand-bar" ref={brandBarRef}>
          <div className="brand-lockup" aria-label="Model">
            <span className="brand-copy">
              <strong>Model</strong>
              <button
                type="button"
                className="brand-link"
                title="切换模型"
                onClick={() => setModelDialogOpen(true)}
              >
                {config.modelName || "No model selected"}
              </button>
            </span>
          </div>
          <div className="brand-usage">
            <UsageStatusCell usage={state.modelUsage} />
          </div>
        </div>
        <div className="status-row" ref={statusRowRef}>
          <div className="status-strip" ref={statusStripRef}>
            <ProjectStatusCell
              projectPath={currentProjectPath}
              open={projectPopoverOpen}
              busy={cwdBusy}
              onToggle={() => setProjectPopoverOpen((value) => !value)}
              onSwitch={() => {
                setProjectPopoverOpen(false);
                void changeWorkingDirectory();
              }}
            />
            <SessionStatusCell
              messageCount={state.sessionMessageCount}
              runningCount={sessionStats.running}
              loadedCount={sessionStats.loaded}
              sessions={sessions}
              currentSessionId={currentSessionId}
              open={sessionDialogOpen}
              busy={sessionBusy}
              searchOpen={historySearchOpen}
              onToggle={openSessionDialog}
              onSelect={loadSession}
              onDelete={deleteSession}
              onRename={renameSession}
              onNew={newSession}
              onFork={forkSession}
              onSearch={() => setHistorySearchOpen(true)}
            />
            <ContextUsageCell
              usage={state.contextUsage}
              autoCompact={state.contextAutoCompact}
              compression={state.contextCompression}
              busy={contextCompactBusy}
              disabled={!coreRunning}
              open={contextPopoverOpen}
              settingsBusy={contextSettingsBusy}
              canManualCompact={canCompactContextManually}
              onToggle={() => setContextPopoverOpen((value) => !value)}
              onConfirm={compactContextManually}
              onThresholdChange={setAutoCompactThreshold}
            />
            <QueueStatusCell
              queueLengths={state.queueLengths}
              queueMessages={state.queueMessages}
              blobPreviewUrls={blobPreviewUrls}
              onOpenMediaPreview={setMediaPreview}
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
              onQueuedMessageRemove={removeQueuedMessage}
              onTaskRemove={removeQueuedMessage}
              onTaskPromote={promoteQueueTask}
              onTaskPullBack={pullBackQueueTask}
              onTaskMove={moveQueueTask}
              onTaskClear={clearNormalQueue}
            />
          </div>
        </div>
      </header>

      <section className={`workspace-row ${hasRightSidebar ? "has-sidebar" : ""}`}>
        <ChatTranscript
          ref={chatRef}
          nodes={visibleChatNodes}
          blobPreviewUrls={blobPreviewUrls}
          onOpenMediaPreview={setMediaPreview}
          processing={state.processing}
          focusMode={focusModeEnabled}
          profilingEnabled={profilingEnabled}
          highlightHistoryIndex={mainLocateTarget?.sessionId === currentSessionId ? mainLocateTarget.messageIndex : undefined}
          highlightContextMessageId={mainLocateTarget?.sessionId === currentSessionId ? mainLocateTarget.contextMessageId : undefined}
          highlightContextMessageIndex={mainLocateTarget?.sessionId === currentSessionId ? mainLocateTarget.contextMessageIndex : undefined}
          onForkMessage={forkMessage}
          messageEdit={messageEditController}
          onScroll={updateFollowTail}
          onWheel={handleChatWheel}
          onTouchStart={markChatUserScrollIntent}
          onMouseDown={pauseChatFollowForSelection}
        />
        {hasRightSidebar && (
          <WorkspaceSidebar
            artifacts={state.artifacts}
            artifactOrder={state.artifactOrder}
            selectedArtifactId={state.selectedArtifactId}
            artifactGroups={artifactGroups}
            sideThreads={sideThreadList}
            messages={state.pluginMessages}
            statuses={state.pluginStatuses}
            toolProgress={state.toolProgress}
            subagents={subagentList}
            activeTab={rightSidebarTab}
            onActiveTabChange={setRightSidebarTab}
            onOpenSideThread={openSideThread}
            onDeleteSideThread={(thread) => void deleteSideThread(thread)}
            onSelectArtifact={(artifactKey) => {
              dispatch({ version: VERSION, type: "gui.select_artifact", payload: { artifact_key: artifactKey } });
            }}
            onPluginAction={(payload) => sendCommand("plugin_action", payload)}
            onObserve={(id) => setSubagentObserverId(id)}
          />
        )}
      </section>

      <footer className={`input-row ${inputDropActive ? "drop-active" : ""}`}>
        <div className="composer">
          {inputAttachments.length > 0 && (
            <AttachmentStrip
              attachments={inputAttachments}
              busy={inputUploading}
              onRemove={removeInputAttachment}
            />
          )}
          {inputAttachmentError && (
            <div className="composer-error">{inputAttachmentError}</div>
          )}
          <div className="composer-input-line">
            <input
              ref={attachmentFileInputRef}
              className="hidden-file-input"
              type="file"
              multiple
              tabIndex={-1}
              onChange={handleAttachmentFileInputChange}
            />
            <button
              className="icon-button attachment-button"
              type="button"
              title="添加附件"
              aria-label="添加附件"
              disabled={inputUploading}
              onClick={() => attachmentFileInputRef.current?.click()}
            >
              <Paperclip size={17} />
            </button>
            <textarea
              ref={inputRef}
              rows={1}
              value={input}
              placeholder="输入消息"
              disabled={inputUploading}
              onChange={(event) => {
                resizeTextareaToRows(event.currentTarget, MESSAGE_INPUT_MAX_ROWS);
                setInput(event.target.value);
                resetInputHistoryNavigation(event.target.value);
              }}
              onCompositionStart={startInputComposition}
              onCompositionEnd={endInputComposition}
              onKeyDown={handleMainInputKeyDown}
              onPaste={handleMainInputPaste}
              onDrop={handleMainInputDrop}
              onDragOver={handleMainInputDragOver}
              onDragLeave={handleMainInputDragLeave}
            />
          </div>
        </div>
        <button
          className="primary-button composer-action-button"
          disabled={inputUploading || (!input.trim() && inputAttachments.length === 0)}
          onClick={submitInput}
        >
          {inputUploading ? <LoaderCircle size={18} className="spin" /> : <Send size={18} />} 发送
        </button>
        {state.control.paused && state.control.resumable ? (
          <button className="primary-button composer-action-button" disabled={inputUploading} onClick={resumeConversation}>
            <Play size={16} /> 继续
          </button>
        ) : (
          <button
            className="danger-button composer-action-button"
            disabled={!canStopConversation}
            onClick={() => sendCommand("stop", { reason: "user" })}
          >
            <Square size={16} /> 停止
          </button>
        )}
      </footer>
      </main>

      {sideSelectionAnchor && !sideThreadPopover && createPortal(
        <SideSelectionButton
          anchor={sideSelectionAnchor}
          onClick={() => openSideComposerFromSelection(sideSelectionAnchor)}
        />,
        document.body
      )}
      {sideThreadPopover && createPortal(
        <SideThreadPopover
          state={sideThreadPopover}
          thread={popoverSideThread}
          draft={sideThreadDraft}
          busy={sideThreadBusy}
          onDraftChange={setSideThreadDraft}
          onSubmit={() => void submitSideThreadQuestion()}
          onClose={() => setSideThreadPopover(null)}
          onWindowRectChange={(windowRect) => {
            setSideThreadPopover((current) => current ? { ...current, windowRect } : current);
          }}
        />,
        document.body
      )}

      {modelDialogOpen && (
        <ModelDialog
          models={metadata.inspect.models}
          providerConfigs={metadata.inspect.model_provider_configs ?? {}}
          modelProviderConfigs={config.modelProviderConfigs ?? {}}
          current={config.modelName}
          providerConfigLocked={canStopConversation}
          onClose={() => setModelDialogOpen(false)}
          onSelect={selectModel}
          onRefresh={refreshProviderModels}
          onApplyProviderConfig={applyModelProviderConfig}
        />
      )}
      {pluginDialogOpen && (
        <PluginDialog
          catalog={metadata.inspect.plugin_catalog}
          selectedPlugins={config.selectedPlugins}
          pluginConfigs={config.pluginConfigs}
          onClose={() => setPluginDialogOpen(false)}
          onPreviewPlugin={previewPluginTools}
          onApply={applyPlugins}
        />
      )}
      {historySearchOpen && (
        <HistorySearchModal
          query={historySearchQuery}
          caseSensitive={historySearchCaseSensitive}
          wholeWord={historySearchWholeWord}
          results={historySearchResults}
          busy={historySearchBusy}
          error={historySearchError}
          selected={selectedHistoryResult}
          previewNodes={historyPreviewNodes}
          previewBusy={historyPreviewBusy}
          previewSession={historyPreviewSession}
          previewRef={historyPreviewRef}
          locateTarget={historyPreviewLocateTarget}
          onQueryChange={setHistorySearchQuery}
          onCaseSensitiveChange={setHistorySearchCaseSensitive}
          onWholeWordChange={setHistorySearchWholeWord}
          onSelect={(result) => void selectHistorySearchResult(result)}
          onOpenSession={(result) => void openHistorySearchResult(result)}
          onClose={() => setHistorySearchOpen(false)}
        />
      )}
      {observedSubagent && (
        <SubAgentObserverModal
          subagent={observedSubagent}
          focusMode={focusModeEnabled}
          onClose={() => setSubagentObserverId(null)}
        />
      )}
      {mediaPreview && (
        <MediaPreviewModal
          preview={mediaPreview}
          onClose={() => setMediaPreview(null)}
        />
      )}
      {systemPromptConfirmOpen && (
        <SystemPromptConfirmModal
          onCancel={() => setSystemPromptConfirmOpen(false)}
          onConfirm={confirmSystemPromptEdit}
        />
      )}
      {pluginSettingsConfirmOpen && (
        <PluginSettingsConfirmModal
          onCancel={() => {
            setPluginSettingsConfirmOpen(false);
            setPendingPluginSettingsApply(null);
          }}
          onConfirm={confirmPluginSettingsApply}
        />
      )}
      {pendingToolCallPurposeEnabled !== null && (
        <ToolCallPurposeConfirmModal
          enabled={pendingToolCallPurposeEnabled}
          onCancel={() => setPendingToolCallPurposeEnabled(null)}
          onConfirm={confirmToolCallPurposeChange}
        />
      )}
    </div>
  );
}

function SideSelectionButton({
  anchor,
  onClick,
}: {
  anchor: SideSelectionAnchor;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className="side-selection-button"
      style={sideSelectionButtonStyle(anchor.rect)}
      title="询问 Hawi"
      aria-label="询问 Hawi"
      onMouseDown={(event) => event.preventDefault()}
      onClick={onClick}
    >
      <MessageSquare size={15} />
    </button>
  );
}

function SideThreadPopover({
  state,
  thread,
  draft,
  busy,
  onDraftChange,
  onSubmit,
  onClose,
  onWindowRectChange,
}: {
  state: SideThreadPopoverState;
  thread?: SideThreadState;
  draft: string;
  busy: boolean;
  onDraftChange: (value: string) => void;
  onSubmit: () => void;
  onClose: () => void;
  onWindowRectChange: (rect: SideThreadWindowRect) => void;
}) {
  const composing = state.mode === "compose";
  const title = composing ? "旁路提问" : "旁路对话";
  const status = thread?.status ?? (composing ? undefined : "running");
  const quotedText = state.anchor.text || thread?.quotedText || "";
  const canSubmit = draft.trim().length > 0 && !busy && status !== "running";
  const startMove = (event: ReactPointerEvent<HTMLElement>) => {
    if (event.button !== 0 || isInteractiveClickTarget(event.target, event.currentTarget)) return;
    startSideThreadWindowGesture(event, state.windowRect, (startRect, deltaX, deltaY) => (
      moveSideThreadWindowRect(startRect, deltaX, deltaY, currentViewportSize())
    ), onWindowRectChange);
  };
  const startResize = (event: ReactPointerEvent<HTMLElement>) => {
    if (event.button !== 0) return;
    startSideThreadWindowGesture(event, state.windowRect, (startRect, deltaX, deltaY) => (
      resizeSideThreadWindowRect(startRect, deltaX, deltaY, currentViewportSize())
    ), onWindowRectChange);
  };

  return (
    <section
      className="side-thread-popover"
      style={sideThreadPopoverStyle(state.windowRect)}
      role="dialog"
      aria-label={title}
      onMouseDown={(event) => event.stopPropagation()}
    >
      <header className="side-thread-popover-head" onPointerDown={startMove}>
        <span><MessageSquare size={15} /> {title}</span>
        <button type="button" className="icon-button" title="关闭" aria-label="关闭" onClick={onClose}>
          <X size={16} />
        </button>
      </header>
      <div className="side-thread-anchor-preview">
        <strong>{thread?.anchorPreview || "选中文本"}</strong>
        {quotedText && <p>{quotedText}</p>}
      </div>
      {thread && (
        <div className="side-thread-transcript">
          <ChatTranscript
            nodes={thread.nodes}
            processing={thread.processing}
            allowFork={false}
            enablePageFind={false}
            focusMode={true}
            profilingEnabled={false}
            emptyLabel="等待回复..."
          />
        </div>
      )}
      {thread?.error && <div className="side-thread-error" role="alert">{thread.error}</div>}
      <footer className="side-thread-composer">
        <textarea
          rows={2}
          value={draft}
          placeholder={composing ? "提问..." : "继续追问..."}
          disabled={busy || status === "running"}
          onChange={(event) => onDraftChange(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && (event.metaKey || event.ctrlKey) && !event.nativeEvent.isComposing) {
              event.preventDefault();
              if (canSubmit) onSubmit();
            }
          }}
        />
        <button
          type="button"
          className="primary-button side-thread-send"
          disabled={!canSubmit}
          onClick={onSubmit}
        >
          {busy || status === "running" ? <LoaderCircle className="spin" size={16} /> : <Send size={16} />}
        </button>
      </footer>
      <button
        type="button"
        className="side-thread-resize-handle"
        aria-label="调整旁路对话大小"
        title="拖动调整大小"
        onPointerDown={startResize}
      />
    </section>
  );
}

function SystemPromptConfirmModal({
  onCancel,
  onConfirm
}: {
  onCancel: () => void;
  onConfirm: () => void;
}) {
  return (
    <Modal
      title="确认编辑 System Prompt"
      className="confirm-modal system-prompt-confirm-modal"
      onClose={onCancel}
      footer={(
        <div className="modal-action-row">
          <button type="button" className="tool-button" onClick={onCancel}>取消</button>
          <button type="button" className="primary-button" onClick={onConfirm}>确认编辑</button>
        </div>
      )}
    >
      <div className="confirm-content">
        <p>当前 Session 已有对话。修改 System Prompt 会改变后续请求的系统提示，可能降低当前上下文的 KV cache 命中率。</p>
        <p>确认后，在当前消息数不变期间可以继续编辑；会话消息更新后，下次编辑会重新确认。</p>
      </div>
    </Modal>
  );
}

function PluginSettingsConfirmModal({
  onCancel,
  onConfirm
}: {
  onCancel: () => void;
  onConfirm: () => void;
}) {
  return (
    <Modal
      title="确认应用 Plugin 设置"
      className="confirm-modal plugin-settings-confirm-modal"
      onClose={onCancel}
      footer={(
        <div className="modal-action-row">
          <button type="button" className="tool-button" onClick={onCancel}>取消</button>
          <button type="button" className="primary-button" onClick={onConfirm}>确认应用</button>
        </div>
      )}
    >
      <div className="confirm-content">
        <p>当前 Session 已有对话。修改 Plugin 启用状态或配置会改变后续可用工具和工具提示内容，可能降低当前上下文的 KV cache 命中率。</p>
        <p>确认后会应用当前 Plugin 设置；会话消息更新后，下次应用会重新确认。</p>
      </div>
    </Modal>
  );
}

function ToolCallPurposeConfirmModal({
  enabled,
  onCancel,
  onConfirm
}: {
  enabled: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  return (
    <Modal
      title="确认修改工具调用目的"
      className="confirm-modal tool-call-purpose-confirm-modal"
      onClose={onCancel}
      footer={(
        <div className="modal-action-row">
          <button type="button" className="tool-button" onClick={onCancel}>取消</button>
          <button type="button" className="primary-button" onClick={onConfirm}>确认{enabled ? "开启" : "关闭"}</button>
        </div>
      )}
    >
      <div className="confirm-content">
        <p>当前 Session 已有对话。修改“工具调用目的”会改变后续工具调用的提示内容，可能降低当前上下文的 KV cache 命中率。</p>
        <p>确认后会从后续请求开始生效；如果想保持当前 cache 命中率，请在新 Session 中修改。</p>
      </div>
    </Modal>
  );
}

function QueueStatusCell({
  queueLengths,
  queueMessages,
  blobPreviewUrls,
  onOpenMediaPreview,
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
  onQueuedMessageRemove,
  onTaskRemove,
  onTaskPromote,
  onTaskPullBack,
  onTaskMove,
  onTaskClear
}: {
  queueLengths: Record<QueueKind, number>;
  queueMessages: Record<QueueKind, QueueMessageState[]>;
  blobPreviewUrls: BlobPreviewUrls;
  onOpenMediaPreview?: (preview: MediaPreviewState) => void;
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
  onQueuedMessageRemove: (message: QueueMessageState) => void;
  onTaskRemove: (message: QueueMessageState) => void;
  onTaskPromote: (message: QueueMessageState) => void;
  onTaskPullBack: (message: QueueMessageState) => void;
  onTaskMove: (messageId: string, direction: -1 | 1) => void;
  onTaskClear: () => void;
}) {
  const anchorRef = useRef<HTMLDivElement | null>(null);
  const highPriorityCount = hasHighPriorityWork(queueLengths, queueMessages) ? 1 : 0;
  const normalCount = normalQueueCount(queueLengths, queueMessages);

  return (
    <StatusCell ref={anchorRef} className="priority-status" active={open}>
      <StatusCellTrigger
        className="priority-status-trigger"
        title="Insert messages deliver soon; queued messages run later."
        aria-label={renderQueueStatusText(queueLengths, queueMessages)}
        aria-pressed={open}
        onClick={onToggle}
        label="Message"
        contentClassName="message-status-widget"
      >
        <span><span>Insert</span><strong>{highPriorityCount}</strong></span>
        <span><span>Queue</span><strong>{normalCount}</strong></span>
      </StatusCellTrigger>
      <StatusPopoverLayer open={open} anchorRef={anchorRef}>
        <QueuePopover
          queueLengths={queueLengths}
          queueMessages={queueMessages}
          blobPreviewUrls={blobPreviewUrls}
          onOpenMediaPreview={onOpenMediaPreview}
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
          onQueuedMessageRemove={onQueuedMessageRemove}
          onTaskRemove={onTaskRemove}
          onTaskPromote={onTaskPromote}
          onTaskPullBack={onTaskPullBack}
          onTaskMove={onTaskMove}
          onTaskClear={onTaskClear}
        />
      </StatusPopoverLayer>
    </StatusCell>
  );
}

function UsageStatusCell({ usage }: { usage?: ModelUsageState }) {
  const label = renderUsageStatusText(usage);
  const metrics = [
    ["Total", formatUsageTokenCount(usage?.totalTokens ?? 0)],
    ["Input", formatUsageTokenCount(usage?.inputTokens ?? 0)],
    ["Output", formatUsageTokenCount(usage?.outputTokens ?? 0)],
    ["Cache Write", formatUsageTokenCount(usage?.cacheWriteTokens ?? 0)],
    ["Cache Read", formatUsageTokenCount(usage?.cacheReadTokens ?? 0)]
  ] as const;

  return (
    <StatusCellDisplay
      className="usage-status"
      title={label}
      aria-label={label}
      label="Usage"
      contentClassName="usage-status-widget"
    >
      {metrics.map(([metricLabel, value]) => (
        <span className="usage-status-metric" key={metricLabel}>
          <span>{metricLabel}</span>
          <strong>{value}</strong>
        </span>
      ))}
    </StatusCellDisplay>
  );
}

function ContextUsageCell({
  usage,
  autoCompact,
  compression,
  busy,
  settingsBusy,
  canManualCompact,
  disabled,
  open,
  onToggle,
  onConfirm,
  onThresholdChange
}: {
  usage?: ContextUsageState;
  autoCompact?: ContextAutoCompactState;
  compression?: ContextCompressionState;
  busy: boolean;
  settingsBusy: boolean;
  canManualCompact: boolean;
  disabled: boolean;
  open: boolean;
  onToggle: () => void;
  onConfirm: () => void;
  onThresholdChange: (percent: number) => Promise<void>;
}) {
  const anchorRef = useRef<HTMLDivElement | null>(null);
  const compressing = compression?.active === true;
  const ratio = usage?.ratio ?? 0;
  const percent = usage?.ratio === undefined ? "n/a" : `${Math.round(ratio * 100)}%`;
  const used = usage ? compactNumber(usage.usedTokens) : "-";
  const max = usage?.maxContextTokens ? compactNumber(usage.maxContextTokens) : "-";
  const usageLabel = `${usage?.source === "estimate" ? "~" : ""}${used}/${max}`;
  const thresholdPercent = autoCompact ? autoCompactThresholdPercent(autoCompact, usage) : undefined;
  const thresholdTitle = thresholdPercent === undefined ? "" : ` · 自动压缩 ${thresholdPercent}%`;
  const meterLabel = thresholdPercent === undefined ? percent : `${percent}/${thresholdPercent}%`;
  const inactive = disabled;
  const title = compressing
    ? `Context compressing ${usageLabel}${thresholdTitle} · 点击查看上下文设置`
    : inactive
      ? `Context ${usageLabel}${thresholdTitle}`
      : busy
        ? `Context ${usageLabel}${thresholdTitle} · 压缩中`
        : `Context ${usageLabel}${thresholdTitle} · 点击查看上下文设置`;
  return (
    <StatusCell ref={anchorRef} className={`context-status ${compressing ? "compressing" : ""}`} active={open}>
      <StatusCellTrigger
        className="context-status-trigger"
        title={title}
        disabled={inactive}
        onClick={onToggle}
        aria-label={`Context ${percent} ${usageLabel}`}
        aria-pressed={open}
        label="Context"
        contentClassName={`context-status-widget ${compressing ? "compressing" : ""}`}
      >
        {compressing ? (
          <>
            <LoaderCircle className="context-status-spinner" size={13} aria-label="Compressing context" />
            <span>Compress</span>
          </>
        ) : (
          <>
            <span className="context-usage-line">{usageLabel}</span>
            <span className="context-meter" aria-hidden="true">
              <span className="context-meter-fill" style={{ width: `${Math.min(100, Math.max(0, ratio * 100))}%` }} />
              {thresholdPercent !== undefined && (
                <span
                  className="context-meter-threshold"
                  style={{ left: `${Math.min(100, Math.max(0, thresholdPercent))}%` }}
                />
              )}
              <strong className="context-meter-label">{meterLabel}</strong>
            </span>
          </>
        )}
      </StatusCellTrigger>
      <StatusPopoverLayer open={open} anchorRef={anchorRef}>
        <ContextPopover
          usage={usage}
          autoCompact={autoCompact}
          busy={busy}
          settingsBusy={settingsBusy}
          canManualCompact={canManualCompact}
          onConfirm={onConfirm}
          onThresholdChange={onThresholdChange}
        />
      </StatusPopoverLayer>
    </StatusCell>
  );
}

function ProjectStatusCell({
  projectPath,
  open,
  busy,
  onToggle,
  onSwitch
}: {
  projectPath: string | null;
  open: boolean;
  busy: boolean;
  onToggle: () => void;
  onSwitch: () => void;
}) {
  const anchorRef = useRef<HTMLDivElement | null>(null);
  const projectName = projectNameFromPath(projectPath);
  const fullPath = projectPath?.trim() || "-";
  const pathLabel = middleEllipsizePath(fullPath, 58);
  return (
    <StatusCell ref={anchorRef} className="project-status" active={open}>
      <StatusCellTrigger
        className="project-trigger"
        title={`Project\n${projectName}\n${fullPath}`}
        onClick={onToggle}
        aria-label={`Project ${projectName}. ${fullPath}`}
        label="Project"
        contentClassName="project-status-widget"
      >
        <strong className="project-name-line">{projectName}</strong>
        <span className="project-path-line">{pathLabel}</span>
      </StatusCellTrigger>
      <StatusPopoverLayer open={open} anchorRef={anchorRef}>
        <div className="project-popover">
          <StatusPopoverHeader title="Project" value={projectName} />
          <div className="project-popover-body">
            <div className="project-full-path">{fullPath}</div>
            <button
              type="button"
              className="primary-button project-switch-button"
              disabled={busy}
              onClick={onSwitch}
            >
              {busy ? <LoaderCircle className="inline-spinner" size={16} /> : <ArrowLeftRight size={16} />}
              切换路径
            </button>
          </div>
        </div>
      </StatusPopoverLayer>
    </StatusCell>
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
  searchOpen,
  onToggle,
  onSelect,
  onDelete,
  onRename,
  onNew,
  onFork,
  onSearch
}: {
  messageCount: number;
  runningCount: number;
  loadedCount: number;
  sessions: SessionMetaPayload[];
  currentSessionId: string | null;
  open: boolean;
  busy: boolean;
  searchOpen: boolean;
  onToggle: () => void;
  onSelect: (sessionId: string) => void;
  onDelete: (sessionId: string) => void;
  onRename: (sessionId: string, name: string) => Promise<void> | void;
  onNew: () => void;
  onFork: (sessionId?: string) => void;
  onSearch: () => void;
}) {
  const anchorRef = useRef<HTMLDivElement | null>(null);
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  const [editDraft, setEditDraft] = useState("");
  const editInputRef = useRef<HTMLInputElement | null>(null);
  const editingSessionIdRef = useRef<string | null>(null);
  const sortedSessions = sortSessionsByCreatedAt(sessions);
  const canForkCurrent = Boolean(currentSessionId) && messageCount > 0;
  const currentSession = currentSessionId
    ? sessions.find((session) => session.session_id === currentSessionId)
    : undefined;
  const currentSessionName = currentSession
    ? sessionDisplayName(currentSession)
    : currentSessionId
      ? shortSessionId(currentSessionId)
      : "New Session";
  const sessionRuntimeLabel = `${runningCount} Running / ${loadedCount} Active`;
  const currentSessionLabel = `Current: ${currentSessionName}`;

  useEffect(() => {
    if (!editingSessionId) return;
    const input = editInputRef.current;
    if (!input) return;
    input.focus();
    input.select();
  }, [editingSessionId]);

  function startEditingSession(session: SessionMetaPayload) {
    editingSessionIdRef.current = session.session_id;
    setEditingSessionId(session.session_id);
    setEditDraft(sessionDisplayName(session));
  }

  function cancelEditingSession() {
    editingSessionIdRef.current = null;
    setEditingSessionId(null);
  }

  async function commitEditingSession(session: SessionMetaPayload) {
    if (editingSessionIdRef.current !== session.session_id) return;
    const nextName = editDraft.trim();
    const displayedName = sessionDisplayName(session).trim();
    editingSessionIdRef.current = null;
    setEditingSessionId(null);
    if (!nextName || nextName === session.name.trim() || nextName === displayedName) {
      return;
    }
    await onRename(session.session_id, nextName);
  }

  return (
    <StatusCell ref={anchorRef} className="session-status" active={open}>
      <StatusCellTrigger
        className="session-trigger"
        title={`切换 Session\n${sessionRuntimeLabel}\n${currentSessionLabel}`}
        disabled={busy}
        onClick={onToggle}
        aria-label={`Session ${sessionRuntimeLabel}. ${currentSessionLabel}`}
        label="Session"
        contentClassName="session-status-widget"
      >
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
      </StatusCellTrigger>
      <button
        type="button"
        className={`mini-button icon-only ${searchOpen ? "active" : ""}`.trim()}
        title="搜索聊天记录"
        aria-label="搜索聊天记录"
        aria-pressed={searchOpen}
        onClick={(event) => {
          event.stopPropagation();
          onSearch();
        }}
      >
        <Search size={15} />
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
      <StatusPopoverLayer open={open} anchorRef={anchorRef}>
        <div className="session-popover">
          <StatusPopoverHeader title="Sessions" value={sortedSessions.length} />
          <div className="session-list">
            {sortedSessions.length === 0 ? (
              <div className="session-empty">No sessions</div>
            ) : sortedSessions.map((session) => {
              const isCurrent = session.session_id === currentSessionId;
              const isLoadedHere = session.load_state === "loaded" || session.load_state === "running";
              const isRunning = session.load_state === "running";
              const isLocked = session.locked === true && !isCurrent && !isLoadedHere;
              const canShowDelete = !isRunning && (!isLocked || isLoadedHere);
              const isEditing = editingSessionId === session.session_id;
              return (
                <div
                  key={session.session_id}
                  className={`session-option ${isCurrent ? "current" : ""} ${isLocked ? "locked" : ""} ${isLoadedHere ? "loaded" : ""}`}
                >
                  <div className="session-select-row">
                    <small className="session-date">
                      {formatSessionTimestamp(session.created_at || session.updated_at)}
                    </small>
                    <div className="session-title-row">
                      {isEditing ? (
                        <input
                          ref={editInputRef}
                          className="session-name-input"
                          value={editDraft}
                          disabled={busy}
                          onChange={(event) => setEditDraft(event.target.value)}
                          onBlur={() => void commitEditingSession(session)}
                          onKeyDown={(event) => {
                            if (event.key === "Escape") {
                              event.preventDefault();
                              cancelEditingSession();
                              return;
                            }
                            if (event.key === "Enter" && !event.nativeEvent.isComposing) {
                              event.preventDefault();
                              void commitEditingSession(session);
                            }
                          }}
                        />
                      ) : (
                        <button
                          type="button"
                          className="session-title-button"
                          title={isLocked ? "Session 正被其他 Hawi engine 使用，可 Fork 后继续" : "切换 Session"}
                          disabled={busy || isLocked}
                          onClick={() => onSelect(session.session_id)}
                        >
                          <SessionLoadIndicator state={session.load_state ?? "unloaded"} />
                          {isLocked && <Lock size={12} />}
                          <span className="session-name-text">{sessionDisplayName(session)}</span>
                        </button>
                      )}
                    </div>
                  </div>
                  <div className="session-actions">
                    <button
                      type="button"
                      className="session-action"
                      title={isEditing ? "正在重命名" : "重命名 Session"}
                      aria-label={`重命名 Session ${shortSessionId(session.session_id)}`}
                      disabled={busy || isEditing}
                      onClick={(event) => {
                        event.stopPropagation();
                        startEditingSession(session);
                      }}
                    >
                      <Pencil size={13} />
                    </button>
                    <button
                      type="button"
                      className="session-action"
                      title={isCurrent ? "当前 Session 无需 Fork" : "Fork Session"}
                      aria-label={`Fork Session ${shortSessionId(session.session_id)}`}
                      disabled={busy || isCurrent}
                      onClick={() => onFork(session.session_id)}
                    >
                      <GitFork size={13} />
                    </button>
                    <button
                      type="button"
                      className="session-delete"
                      title={canShowDelete ? (isLoadedHere ? "关闭并删除 Session" : "删除 Session") : "当前状态不可删除"}
                      aria-label={`删除 Session ${shortSessionId(session.session_id)}`}
                      disabled={busy || !canShowDelete}
                      onClick={() => onDelete(session.session_id)}
                    >
                      <Trash2 size={13} />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </StatusPopoverLayer>
    </StatusCell>
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

function HistorySearchModal({
  query,
  caseSensitive,
  wholeWord,
  results,
  busy,
  error,
  selected,
  previewNodes,
  previewBusy,
  previewSession,
  previewRef,
  locateTarget,
  onQueryChange,
  onCaseSensitiveChange,
  onWholeWordChange,
  onSelect,
  onOpenSession,
  onClose,
}: {
  query: string;
  caseSensitive: boolean;
  wholeWord: boolean;
  results: HistorySearchResult[];
  busy: boolean;
  error: string | null;
  selected: HistorySearchResult | null;
  previewNodes: ChatNode[];
  previewBusy: boolean;
  previewSession: SessionMetaPayload | null;
  previewRef: Ref<HTMLDivElement>;
  locateTarget: HistoryLocateTarget | null;
  onQueryChange: (value: string) => void;
  onCaseSensitiveChange: (value: boolean) => void;
  onWholeWordChange: (value: boolean) => void;
  onSelect: (result: HistorySearchResult) => void;
  onOpenSession: (result: HistorySearchResult | null) => void;
  onClose: () => void;
}) {
  const selectedKey = selected ? historySearchResultKey(selected) : null;
  const previewTitle = previewSession ? sessionDisplayName(previewSession) : "未选择";
  const resultCount = busy ? "搜索中" : `${results.length}`;

  return (
    <Modal title="聊天记录搜索" className="history-search-modal" onClose={onClose}>
      <div className="history-search-layout">
        <section className="history-preview-pane">
          <header className="history-preview-header">
            <div className="history-preview-title">
              <strong>{previewTitle}</strong>
              <span>{selected ? formatHistoryResultTimestamp(selected) : ""}</span>
            </div>
            <button
              type="button"
              className="primary-button"
              disabled={!selected}
              onClick={() => onOpenSession(selected)}
            >
              <Play size={15} /> 打开会话
            </button>
          </header>
          <div className="history-preview-body">
            {previewBusy && (
              <div className="history-preview-loading">
                <LoaderCircle className="inline-spinner" size={16} /> 加载中
              </div>
            )}
            <ChatTranscript
              ref={previewRef}
              nodes={previewNodes}
              allowFork={false}
              profilingEnabled={profilingEnabled}
              enablePageFind={false}
              emptyLabel={query.trim() ? "无预览" : "输入关键词"}
              searchHighlightQuery={query}
              searchHighlightCaseSensitive={caseSensitive}
              searchHighlightWholeWord={wholeWord}
              searchHighlightTarget={locateTarget}
              highlightHistoryIndex={locateTarget?.messageIndex}
              highlightContextMessageId={locateTarget?.contextMessageId}
              highlightContextMessageIndex={locateTarget?.contextMessageIndex}
            />
          </div>
        </section>
        <aside className="history-results-pane">
          <div className="history-search-controls">
            <label className="history-search-input">
              <Search size={16} />
              <input
                autoFocus
                value={query}
                placeholder="搜索聊天记录"
                onChange={(event) => onQueryChange(event.target.value)}
              />
            </label>
            <div className="history-search-options">
              <label className="history-search-option">
                <input
                  type="checkbox"
                  checked={caseSensitive}
                  onChange={(event) => onCaseSensitiveChange(event.target.checked)}
                />
                <span>大小写敏感</span>
              </label>
              <label className="history-search-option">
                <input
                  type="checkbox"
                  checked={wholeWord}
                  onChange={(event) => onWholeWordChange(event.target.checked)}
                />
                <span>完整词语</span>
              </label>
            </div>
          </div>
          <div className="history-results-head">
            <span>Results</span>
            <strong>{resultCount}</strong>
          </div>
          {error && <div className="history-search-error" role="alert">{error}</div>}
          <div className="history-result-list">
            {results.length === 0 ? (
              <div className="session-empty">{query.trim() ? "No results" : "No query"}</div>
            ) : results.map((result) => {
              const key = historySearchResultKey(result);
              return (
                <button
                  type="button"
                  className={`history-result ${key === selectedKey ? "selected" : ""}`}
                  key={key}
                  onClick={() => onSelect(result)}
                >
                  <span className="history-result-title">
                    <strong>{result.sessionName || shortSessionId(result.sessionId)}</strong>
                    <small>{formatHistoryResultTimestamp(result)}</small>
                  </span>
                  <span className="history-result-role">{historyRoleLabel(result.role)}</span>
                  <span className="history-result-snippet">
                    <HighlightedText
                      text={result.snippet || result.text}
                      query={query}
                      caseSensitive={caseSensitive}
                      wholeWord={wholeWord}
                    />
                  </span>
                </button>
              );
            })}
          </div>
        </aside>
      </div>
    </Modal>
  );
}

function HighlightedText({
  text,
  query,
  caseSensitive,
  wholeWord
}: {
  text: string;
  query: string;
  caseSensitive: boolean;
  wholeWord: boolean;
}) {
  const ranges = historyTextMatchRanges(text, query, { caseSensitive, wholeWord });
  if (ranges.length === 0) return <>{text}</>;
  const parts: ReactNode[] = [];
  let cursor = 0;
  for (const [start, end] of ranges) {
    if (start > cursor) {
      parts.push(text.slice(cursor, start));
    }
    parts.push(<mark key={`${start}-${end}`}>{text.slice(start, end)}</mark>);
    cursor = end;
  }
  if (cursor < text.length) {
    parts.push(text.slice(cursor));
  }
  return <>{parts}</>;
}

function historyTextMatchRanges(
  text: string,
  query: string,
  options: { caseSensitive: boolean; wholeWord: boolean }
): Array<[number, number]> {
  const needle = query.trim();
  if (!needle) return [];
  const haystack = options.caseSensitive ? text : text.toLowerCase();
  const target = options.caseSensitive ? needle : needle.toLowerCase();
  const ranges: Array<[number, number]> = [];
  let cursor = 0;
  while (cursor <= haystack.length) {
    const index = haystack.indexOf(target, cursor);
    if (index < 0) break;
    const end = index + target.length;
    if (!options.wholeWord || hasNonEnglishLetterBoundaries(text, index, end)) {
      ranges.push([index, end]);
    }
    cursor = index + Math.max(1, target.length);
  }
  return ranges;
}

function hasNonEnglishLetterBoundaries(text: string, start: number, end: number): boolean {
  const left = start > 0 ? text[start - 1] : "";
  const right = end < text.length ? text[end] : "";
  return !isEnglishLetter(left) && !isEnglishLetter(right);
}

function isEnglishLetter(value: string): boolean {
  if (value.length !== 1) return false;
  const code = value.charCodeAt(0);
  return (code >= 65 && code <= 90) || (code >= 97 && code <= 122);
}

function QueuePopover({
  queueLengths,
  queueMessages,
  blobPreviewUrls,
  onOpenMediaPreview,
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
  onQueuedMessageRemove,
  onTaskRemove,
  onTaskPromote,
  onTaskPullBack,
  onTaskMove,
  onTaskClear
}: {
  queueLengths: Record<QueueKind, number>;
  queueMessages: Record<QueueKind, QueueMessageState[]>;
  blobPreviewUrls: BlobPreviewUrls;
  onOpenMediaPreview?: (preview: MediaPreviewState) => void;
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
  onQueuedMessageRemove: (message: QueueMessageState) => void;
  onTaskRemove: (message: QueueMessageState) => void;
  onTaskPromote: (message: QueueMessageState) => void;
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
        blobPreviewUrls={blobPreviewUrls}
        onOpenMediaPreview={onOpenMediaPreview}
        busy={taskBusy}
        onRemove={onQueuedMessageRemove}
      />
      <QueueTaskGroup
        length={normalCount}
        messages={queueMessages.normal}
        blobPreviewUrls={blobPreviewUrls}
        onOpenMediaPreview={onOpenMediaPreview}
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
        onPromote={onTaskPromote}
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
  messages,
  blobPreviewUrls,
  onOpenMediaPreview,
  busy = false,
  onRemove
}: {
  kind: QueueKind;
  length: number;
  messages: QueueMessageState[];
  blobPreviewUrls: BlobPreviewUrls;
  onOpenMediaPreview?: (preview: MediaPreviewState) => void;
  busy?: boolean;
  onRemove?: (message: QueueMessageState) => void;
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
            <QueueMessageItem
              message={message}
              key={`${kind}-${message.id}`}
              blobPreviewUrls={blobPreviewUrls}
              onOpenMediaPreview={onOpenMediaPreview}
              busy={busy}
              onRemove={onRemove}
            />
          ))}
        </div>
      )}
    </section>
  );
}

function QueueTaskGroup({
  length,
  messages,
  blobPreviewUrls,
  onOpenMediaPreview,
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
  onPromote,
  onPullBack,
  onMove,
  onClear
}: {
  length: number;
  messages: QueueMessageState[];
  blobPreviewUrls: BlobPreviewUrls;
  onOpenMediaPreview?: (preview: MediaPreviewState) => void;
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
  onRemove: (message: QueueMessageState) => void;
  onPromote: (message: QueueMessageState) => void;
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
              blobPreviewUrls={blobPreviewUrls}
              onOpenMediaPreview={onOpenMediaPreview}
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
              onPromote={onPromote}
              onPullBack={onPullBack}
              onMove={onMove}
            />
          ))}
        </div>
      )}
    </section>
  );
}

function QueueMessageItem({
  message,
  blobPreviewUrls,
  onOpenMediaPreview,
  busy = false,
  onRemove
}: {
  message: QueueMessageState;
  blobPreviewUrls: BlobPreviewUrls;
  onOpenMediaPreview?: (preview: MediaPreviewState) => void;
  busy?: boolean;
  onRemove?: (message: QueueMessageState) => void;
}) {
  const timestamp = formatQueueTimestamp(message.createdAt);
  const canRemove = Boolean(onRemove) && message.metadata?.withdrawable !== false;
  return (
    <article className={`queue-message ${onRemove ? "with-actions" : ""}`}>
      <div className="queue-message-meta">
        <span>{message.id}</span>
        {timestamp && <time>{timestamp}</time>}
      </div>
      <QueueMessageContent
        message={message}
        blobPreviewUrls={blobPreviewUrls}
        onOpenMediaPreview={onOpenMediaPreview}
      />
      {onRemove && (
        <div className="queue-task-actions queue-message-actions">
          <button
            type="button"
            title={canRemove ? "撤回插队消息" : "已发送，无法撤回"}
            disabled={busy || !canRemove}
            onClick={() => {
              if (canRemove) onRemove(message);
            }}
          >
            <Trash2 size={13} />
          </button>
        </div>
      )}
    </article>
  );
}

function QueueTaskItem({
  message,
  blobPreviewUrls,
  onOpenMediaPreview,
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
  onPromote,
  onPullBack,
  onMove
}: {
  message: QueueMessageState;
  blobPreviewUrls: BlobPreviewUrls;
  onOpenMediaPreview?: (preview: MediaPreviewState) => void;
  index: number;
  count: number;
  busy: boolean;
  editing: boolean;
  editDraft: string;
  onEditStart: (message: QueueMessageState) => void;
  onEditCancel: () => void;
  onEditDraftChange: (value: string) => void;
  onUpdate: (messageId: string, content: string) => void;
  onRemove: (message: QueueMessageState) => void;
  onPromote: (message: QueueMessageState) => void;
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
      <QueueMessageContent
        message={message}
        blobPreviewUrls={blobPreviewUrls}
        onOpenMediaPreview={onOpenMediaPreview}
        preferFullContent
      />
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
        <button type="button" title="升级为插队消息" disabled={busy} onClick={() => onPromote(message)}>
          <ChevronsUp size={13} />
        </button>
        <button type="button" title="拉回编辑" disabled={busy} onClick={() => onPullBack(message)}>
          <RotateCcw size={13} />
        </button>
        <button type="button" title="撤回到输入框" disabled={busy} onClick={() => onRemove(message)}>
          <Trash2 size={13} />
        </button>
      </div>
    </article>
  );
}

function AttachmentStrip({
  attachments,
  busy,
  onRemove
}: {
  attachments: PendingAttachment[];
  busy: boolean;
  onRemove: (id: string) => void;
}) {
  return (
    <div className="attachment-strip">
      {attachments.map((attachment) => (
        <figure className={`attachment-chip ${attachment.status}`} key={attachment.id}>
          <AttachmentThumbnail attachment={attachment} />
          <figcaption>
            <span title={attachment.filename}>{attachment.filename}</span>
            <small>{attachmentStatusText(attachment)}</small>
          </figcaption>
          {attachment.status === "uploading" && (
            <progress value={attachment.progress} max={1} aria-label="上传进度" />
          )}
          {attachment.status === "error" && attachment.error && (
            <small className="attachment-error" title={attachment.error}>{attachment.error}</small>
          )}
          <button
            type="button"
            title="移除附件"
            aria-label="移除附件"
            disabled={busy}
            onClick={() => onRemove(attachment.id)}
          >
            <X size={12} />
          </button>
        </figure>
      ))}
    </div>
  );
}

function AttachmentThumbnail({ attachment }: { attachment: PendingAttachment }) {
  if (attachment.partType === "image") {
    return (
      <img
        className="attachment-thumbnail"
        src={attachment.dataUrl ?? attachment.previewUrl}
        alt={attachment.filename}
      />
    );
  }

  return (
    <div className={`attachment-thumbnail attachment-thumbnail-${attachment.partType}`} title={attachment.mimeType}>
      <FileText size={20} />
    </div>
  );
}

function MessageEditBubble({
  edit,
  blobPreviewUrls,
  maxHeight,
  onDraftChange,
  onAddFiles,
  onRemoveAttachment,
  onCancel,
  onSave,
  onResend,
}: {
  edit: MessageEditState;
  blobPreviewUrls: BlobPreviewUrls;
  maxHeight?: number;
  onDraftChange: (value: string) => void;
  onAddFiles: (files: Iterable<File>) => void;
  onRemoveAttachment: (id: string) => void;
  onCancel: () => void;
  onSave: () => void;
  onResend: () => void;
}) {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const composingRef = useRef(false);
  const isUser = edit.role === "user";
  const canSubmit = isUser
    ? Boolean(edit.draft.trim() || edit.attachments.length > 0)
    : Boolean(edit.draft.trim());
  const style = maxHeight
    ? ({ "--message-edit-max-height": `${maxHeight}px` } as CSSProperties)
    : undefined;

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.focus();
    textarea.setSelectionRange(textarea.value.length, textarea.value.length);
  }, [edit.nodeId]);

  useEffect(() => {
    resizeMessageEditTextarea(textareaRef.current);
  }, [edit.draft, maxHeight]);

  const handleFiles = (files: FileList | null) => {
    if (!isUser || edit.busy) return;
    onAddFiles(attachmentFilesFromFileList(files));
  };

  const triggerPrimaryAction = () => {
    if (!canSubmit || edit.busy) return;
    if (isUser) {
      onResend();
    } else {
      onSave();
    }
  };

  return (
    <div className="message-edit-body" style={style}>
      <textarea
        ref={textareaRef}
        className="message-edit-textarea"
        value={edit.draft}
        rows={Math.min(14, Math.max(4, edit.draft.split("\n").length + 1))}
        disabled={edit.busy}
        spellCheck
        onChange={(event) => {
          onDraftChange(event.target.value);
          resizeMessageEditTextarea(event.currentTarget);
        }}
        onCompositionStart={() => {
          composingRef.current = true;
        }}
        onCompositionEnd={() => {
          window.setTimeout(() => {
            composingRef.current = false;
          }, 0);
        }}
        onKeyDown={(event) => {
          if (event.key === "Escape" && !event.nativeEvent.isComposing) {
            event.preventDefault();
            event.stopPropagation();
            onCancel();
            return;
          }
          if (shouldSubmitInputFromKeyEvent(event, composingRef.current)) {
            event.preventDefault();
            event.stopPropagation();
            triggerPrimaryAction();
          }
        }}
      />
      {isUser && (
        <>
          <input
            ref={fileInputRef}
            className="hidden-file-input"
            type="file"
            multiple
            tabIndex={-1}
            onChange={(event) => {
              handleFiles(event.target.files);
              event.target.value = "";
            }}
          />
          {edit.attachments.length > 0 && (
            <EditableAttachmentStrip
              attachments={edit.attachments}
              blobPreviewUrls={blobPreviewUrls}
              busy={edit.busy}
              onRemove={onRemoveAttachment}
            />
          )}
        </>
      )}
      {edit.error && <div className="message-edit-error">{edit.error}</div>}
      <div className="message-edit-footer">
        {isUser ? (
          <button
            type="button"
            className="message-edit-attachment-button"
            title="添加附件"
            aria-label="添加附件"
            disabled={edit.busy}
            onClick={() => fileInputRef.current?.click()}
          >
            <Paperclip size={15} />
          </button>
        ) : (
          <span />
        )}
        <div className="message-edit-actions">
          <button
            type="button"
            className="message-edit-button cancel"
            disabled={edit.busy}
            onClick={onCancel}
          >
            取消
          </button>
          <button
            type="button"
            className="message-edit-button save"
            disabled={edit.busy || !canSubmit}
            onClick={onSave}
          >
            {edit.busy ? <LoaderCircle size={14} className="spin" /> : <Check size={14} />}
            保存
          </button>
          {isUser && (
            <button
              type="button"
              className="message-edit-button resend"
              disabled={edit.busy || !canSubmit}
              onClick={onResend}
            >
              <Send size={14} />
              重新发送
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function EditableAttachmentStrip({
  attachments,
  blobPreviewUrls,
  busy,
  onRemove,
}: {
  attachments: EditableAttachment[];
  blobPreviewUrls: BlobPreviewUrls;
  busy: boolean;
  onRemove: (id: string) => void;
}) {
  return (
    <div className="attachment-strip message-edit-attachments">
      {attachments.map((item) => (
        <figure className={`attachment-chip ${editableAttachmentStatusClass(item)}`} key={item.id}>
          <EditableAttachmentThumbnail item={item} blobPreviewUrls={blobPreviewUrls} />
          <figcaption>
            <span title={editableAttachmentFilename(item)}>{editableAttachmentFilename(item)}</span>
            <small>{editableAttachmentStatusText(item)}</small>
          </figcaption>
          {item.kind === "pending" && item.attachment.status === "uploading" && (
            <progress value={item.attachment.progress} max={1} aria-label="上传进度" />
          )}
          {item.kind === "pending" && item.attachment.status === "error" && item.attachment.error && (
            <small className="attachment-error" title={item.attachment.error}>{item.attachment.error}</small>
          )}
          <button
            type="button"
            title="移除附件"
            aria-label="移除附件"
            disabled={busy}
            onClick={() => onRemove(item.id)}
          >
            <X size={12} />
          </button>
        </figure>
      ))}
    </div>
  );
}

function EditableAttachmentThumbnail({
  item,
  blobPreviewUrls,
}: {
  item: EditableAttachment;
  blobPreviewUrls: BlobPreviewUrls;
}) {
  if (item.kind === "pending") {
    return <AttachmentThumbnail attachment={item.attachment} />;
  }
  if (item.partType === "image") {
    const src = mediaDisplayUrl(item.source, blobPreviewUrls);
    if (src) {
      return (
        <img
          className="attachment-thumbnail"
          src={src}
          alt={item.filename}
        />
      );
    }
  }
  return (
    <div className={`attachment-thumbnail attachment-thumbnail-${item.partType}`} title={item.mimeType}>
      {item.partType === "image" ? <ImageIcon size={20} /> : <FileText size={20} />}
    </div>
  );
}

function editableAttachmentFilename(item: EditableAttachment): string {
  return item.kind === "pending" ? item.attachment.filename : item.filename;
}

function editableAttachmentStatusClass(item: EditableAttachment): string {
  return item.kind === "pending" ? item.attachment.status : "uploaded";
}

function editableAttachmentStatusText(item: EditableAttachment): string {
  if (item.kind === "pending") {
    return attachmentStatusText(item.attachment);
  }
  const size = typeof item.size === "number" ? formatByteCount(item.size) : "";
  return ["历史附件", item.mimeType, size].filter(Boolean).join(" - ");
}

function QueueMessageContent({
  message,
  blobPreviewUrls,
  onOpenMediaPreview,
  preferFullContent = false,
}: {
  message: QueueMessageState;
  blobPreviewUrls: BlobPreviewUrls;
  onOpenMediaPreview?: (preview: MediaPreviewState) => void;
  preferFullContent?: boolean;
}) {
  if (hasRenderableContentParts(message.contentParts)) {
    return (
      <ContentPartsView
        parts={message.contentParts ?? []}
        fallbackText={message.contentPreview}
        blobPreviewUrls={blobPreviewUrls}
        onOpenMediaPreview={onOpenMediaPreview}
        compact
      />
    );
  }
  const text = (preferFullContent ? message.content : undefined) ?? message.contentPreview;
  return <p>{text || "空消息"}</p>;
}

function ContentPartsView({
  parts,
  fallbackText,
  markdown = false,
  blobPreviewUrls,
  onOpenMediaPreview,
  compact = false,
}: {
  parts: ContentPart[];
  fallbackText?: string;
  markdown?: boolean;
  blobPreviewUrls: BlobPreviewUrls;
  onOpenMediaPreview?: (preview: MediaPreviewState) => void;
  compact?: boolean;
}) {
  const visibleParts = parts.filter(isRenderableContentPart);
  if (visibleParts.length === 0) {
    return <p>{fallbackText || "空消息"}</p>;
  }

  return (
    <div className={`content-parts ${compact ? "compact" : ""}`.trim()}>
      {visibleParts.map((part, index) => (
        <ContentPartView
          part={part}
          markdown={markdown}
          blobPreviewUrls={blobPreviewUrls}
          onOpenMediaPreview={onOpenMediaPreview}
          compact={compact}
          key={`${part.type}-${index}`}
        />
      ))}
    </div>
  );
}

function ContentPartView({
  part,
  markdown,
  blobPreviewUrls,
  onOpenMediaPreview,
  compact,
}: {
  part: ContentPart;
  markdown: boolean;
  blobPreviewUrls: BlobPreviewUrls;
  onOpenMediaPreview?: (preview: MediaPreviewState) => void;
  compact: boolean;
}) {
  if (part.type === "text") {
    const text = typeof part.text === "string" ? part.text : "";
    return <MarkdownView html={markdown ? renderMarkdown(text) : escapeText(text)} />;
  }

  if (isMediaPart(part)) {
    return (
      <MediaPartView
        part={part}
        source={isRecord(part.source) ? part.source as MediaSource : undefined}
        blobPreviewUrls={blobPreviewUrls}
        onOpenMediaPreview={onOpenMediaPreview}
        compact={compact}
      />
    );
  }

  return null;
}

function MediaPartView({
  part,
  source,
  blobPreviewUrls,
  onOpenMediaPreview,
  compact,
}: {
  part: ContentPart;
  source?: MediaSource;
  blobPreviewUrls: BlobPreviewUrls;
  onOpenMediaPreview?: (preview: MediaPreviewState) => void;
  compact: boolean;
}) {
  const src = mediaDisplayUrl(source, blobPreviewUrls);
  const label = mediaDisplayName(part, source);
  const meta = mediaMetaText(source);
  const mediaKind = String(part.type);
  const openPreview = src && onOpenMediaPreview
    ? () => onOpenMediaPreview({ src, label, meta, kind: mediaKind })
    : undefined;

  if (mediaKind === "image") {
    return (
      <figure className={`media-part image ${compact ? "compact" : ""}`.trim()}>
        {src ? (
          <button
            type="button"
            className="media-preview-button"
            title="放大查看图片"
            onClick={openPreview}
          >
            <img src={src} alt={label} />
          </button>
        ) : (
          <MediaPlaceholder icon="image" label={label} />
        )}
        <figcaption>
          <span>{label}</span>
          {meta && <small>{meta}</small>}
        </figcaption>
      </figure>
    );
  }

  if (mediaKind === "audio" && src) {
    return (
      <figure className={`media-part audio ${compact ? "compact" : ""}`.trim()}>
        <audio src={src} controls />
        <figcaption>
          <span>{label}</span>
          {meta && <small>{meta}</small>}
        </figcaption>
      </figure>
    );
  }

  if (mediaKind === "video" && src) {
    return (
      <figure className={`media-part video ${compact ? "compact" : ""}`.trim()}>
        <video src={src} controls />
        <figcaption>
          <span>{label}</span>
          {meta && <small>{meta}</small>}
        </figcaption>
      </figure>
    );
  }

  return (
    <figure className={`media-part file ${compact ? "compact" : ""}`.trim()}>
      <MediaPlaceholder icon={mediaKind === "image" ? "image" : "file"} label={label} />
      <figcaption>
        <span>{label}</span>
        {meta && <small>{meta}</small>}
      </figcaption>
    </figure>
  );
}

function MediaPlaceholder({ icon, label }: { icon: "image" | "file"; label: string }) {
  return (
    <div className="media-placeholder" title={label}>
      {icon === "image" ? <ImageIcon size={20} /> : <FileText size={20} />}
    </div>
  );
}

function MediaPreviewModal({
  preview,
  onClose,
}: {
  preview: MediaPreviewState;
  onClose: () => void;
}) {
  const modalRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    modalRef.current?.focus({ preventScroll: true });
  }, []);

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key !== "Escape") return;
      event.preventDefault();
      onClose();
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  return (
    <div
      className="modal-backdrop media-preview-backdrop"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) {
          onClose();
        }
      }}
    >
      <div
        ref={modalRef}
        className="media-preview-lightbox"
        role="dialog"
        aria-modal="true"
        aria-label={preview.label}
        tabIndex={-1}
      >
        <div className="media-preview-content">
          <img className="media-preview-image" src={preview.src} alt={preview.label} />
          {(preview.label || preview.meta) && (
            <div className="media-preview-caption">
              {preview.label && <span>{preview.label}</span>}
              {preview.meta && <small>{preview.meta}</small>}
            </div>
          )}
        </div>
        <button
          type="button"
          className="media-preview-close"
          title="关闭"
          aria-label="关闭"
          onClick={onClose}
        >
          <X size={19} />
        </button>
      </div>
    </div>
  );
}

interface ChatTranscriptProps {
  nodes: ChatNode[];
  blobPreviewUrls?: BlobPreviewUrls;
  onOpenMediaPreview?: (preview: MediaPreviewState) => void;
  processing?: ProcessingState;
  focusMode?: boolean;
  profilingEnabled?: boolean;
  onForkMessage?: (node: ChatNode) => void;
  allowFork?: boolean;
  messageEdit?: MessageEditController;
  emptyLabel?: string;
  enablePageFind?: boolean;
  searchHighlightQuery?: string;
  searchHighlightCaseSensitive?: boolean;
  searchHighlightWholeWord?: boolean;
  searchHighlightTarget?: HistoryLocateTarget | null;
  highlightHistoryIndex?: number;
  highlightContextMessageId?: string;
  highlightContextMessageIndex?: number;
  onScroll?: () => void;
  onWheel?: (event: WheelEvent<HTMLElement>) => void;
  onTouchStart?: () => void;
  onMouseDown?: (event: ReactMouseEvent<HTMLElement>) => void;
}

interface PageFindMatchOffset {
  start: number;
  end: number;
}

interface PageFindMatchOptions {
  caseSensitive?: boolean;
  wholeWord?: boolean;
}

export function pageFindTextMatchOffsets(
  text: string,
  query: string,
  options: PageFindMatchOptions = {}
): PageFindMatchOffset[] {
  const needle = query.trim();
  if (!needle) return [];
  const haystack = options.caseSensitive ? text : text.toLowerCase();
  const normalizedNeedle = options.caseSensitive ? needle : needle.toLowerCase();
  const matches: PageFindMatchOffset[] = [];
  let offset = 0;
  while (offset <= haystack.length - normalizedNeedle.length) {
    const index = haystack.indexOf(normalizedNeedle, offset);
    if (index < 0) break;
    const end = index + normalizedNeedle.length;
    if (!options.wholeWord || hasNonEnglishLetterBoundaries(text, index, end)) {
      matches.push({ start: index, end });
    }
    offset = index + Math.max(1, normalizedNeedle.length);
  }
  return matches;
}

export function resolvePageFindStep(currentIndex: number, count: number, direction: -1 | 1): number {
  if (count <= 0) return 0;
  return (currentIndex + direction + count) % count;
}

const ChatTranscript = memo(forwardRef<HTMLDivElement, ChatTranscriptProps>(function ChatTranscript({
  nodes,
  blobPreviewUrls = {},
  onOpenMediaPreview,
  processing,
  focusMode = false,
  profilingEnabled = true,
  onForkMessage,
  allowFork = true,
  messageEdit,
  emptyLabel,
  enablePageFind = true,
  searchHighlightQuery = "",
  searchHighlightCaseSensitive = false,
  searchHighlightWholeWord = false,
  searchHighlightTarget = null,
  highlightHistoryIndex,
  highlightContextMessageId,
  highlightContextMessageIndex,
  onScroll,
  onWheel,
  onTouchStart,
  onMouseDown,
}, ref) {
  const panelRef = useRef<HTMLDivElement | null>(null);
  const pageFindInputRef = useRef<HTMLInputElement | null>(null);
  const [pageFindOpen, setPageFindOpen] = useState(false);
  const [pageFindQuery, setPageFindQuery] = useState("");
  const [pageFindMatchCount, setPageFindMatchCount] = useState(0);
  const [pageFindActiveIndex, setPageFindActiveIndex] = useState(0);
  const [expandedFocusRounds, setExpandedFocusRounds] = useState<Record<string, boolean>>({});
  const transcriptItems = focusMode
    ? buildFocusTranscriptItems(nodes, processing)
    : nodes.map((node): FocusTranscriptItem => ({ type: "node", node }));
  const focusGroupContainsHighlight = (group: FocusFoldGroup) => group.nodes.some((node) => isHighlightedChatNode(
    node,
    highlightHistoryIndex,
    highlightContextMessageId,
    highlightContextMessageIndex,
  ));
  const isFocusGroupExpanded = (group: FocusFoldGroup) => (
    expandedFocusRounds[group.roundId] === true || focusGroupContainsHighlight(group)
  );
  const activeCollapsedFocusFold = focusMode && transcriptItems.some((item) => (
    item.type === "focus-fold"
    && item.group.summary.active
    && !isFocusGroupExpanded(item.group)
  ));
  const processingLine = resolveTranscriptProcessingLine({
    nodes,
    processing,
    focusMode,
    activeCollapsedFocusFold
  });
  const toggleFocusGroup = (group: FocusFoldGroup) => {
    setExpandedFocusRounds((current) => ({
      ...current,
      [group.roundId]: !current[group.roundId]
    }));
  };
  const setPanelRef = useCallback((element: HTMLDivElement | null) => {
    panelRef.current = element;
    if (typeof ref === "function") {
      ref(element);
    } else if (ref) {
      ref.current = element;
    }
  }, [ref]);
  const openPageFind = useCallback(() => {
    setPageFindOpen(true);
    window.requestAnimationFrame(() => {
      pageFindInputRef.current?.focus();
      pageFindInputRef.current?.select();
    });
  }, []);
  const closePageFind = useCallback(() => {
    setPageFindOpen(false);
  }, []);
  const updatePageFindQuery = (query: string) => {
    setPageFindQuery(query);
    setPageFindActiveIndex(0);
  };
  const stepPageFind = (direction: -1 | 1) => {
    setPageFindActiveIndex((current) => resolvePageFindStep(current, pageFindMatchCount, direction));
  };

  useEffect(() => {
    if (!enablePageFind) return;
    function handlePageFindShortcut(event: KeyboardEvent) {
      if (event.isComposing || event.altKey) return;
      if (document.querySelector(".modal-backdrop")) return;
      if ((event.metaKey || event.ctrlKey) && !event.shiftKey && event.key.toLowerCase() === "f") {
        event.preventDefault();
        event.stopPropagation();
        openPageFind();
        return;
      }
      if (pageFindOpen && event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        closePageFind();
      }
    }

    document.addEventListener("keydown", handlePageFindShortcut, true);
    return () => {
      document.removeEventListener("keydown", handlePageFindShortcut, true);
    };
  }, [closePageFind, enablePageFind, openPageFind, pageFindOpen]);

  const internalPageFindActive = enablePageFind && pageFindOpen;
  const externalSearchHighlightActive = !internalPageFindActive && searchHighlightQuery.trim().length > 0;

  usePageFindHighlights(
    panelRef,
    internalPageFindActive ? pageFindQuery : searchHighlightQuery,
    pageFindActiveIndex,
    setPageFindMatchCount,
    setPageFindActiveIndex,
    {
      caseSensitive: internalPageFindActive ? false : searchHighlightCaseSensitive,
      wholeWord: internalPageFindActive ? false : searchHighlightWholeWord,
      target: internalPageFindActive ? null : searchHighlightTarget,
      matchHighlightName: externalSearchHighlightActive ? HISTORY_FIND_MATCH_HIGHLIGHT : PAGE_FIND_MATCH_HIGHLIGHT,
      activeHighlightName: externalSearchHighlightActive ? HISTORY_FIND_ACTIVE_HIGHLIGHT : PAGE_FIND_ACTIVE_HIGHLIGHT,
    },
    [nodes, processing, focusMode, expandedFocusRounds]
  );

  const renderNodeFrame = (node: ChatNode, key = node.id) => {
    const highlighted = isHighlightedChatNode(
      node,
      highlightHistoryIndex,
      highlightContextMessageId,
      highlightContextMessageIndex,
    );
    return (
      <div
        className={`chat-node-frame ${highlighted ? "history-highlight" : ""}`.trim()}
        data-history-index={typeof node.historyIndex === "number" ? node.historyIndex : undefined}
        data-context-message-id={node.contextMessageId}
        data-context-message-index={typeof node.contextMessageIndex === "number" ? node.contextMessageIndex : undefined}
        key={key}
      >
        <ChatBubble
          node={node}
          blobPreviewUrls={blobPreviewUrls}
          onOpenMediaPreview={onOpenMediaPreview}
          allowFork={allowFork}
          onForkMessage={onForkMessage}
          messageEdit={messageEdit}
          profilingEnabled={profilingEnabled}
        />
      </div>
    );
  };

  return (
    <main
      className={`chat-panel ${internalPageFindActive ? "has-page-find" : ""}`.trim()}
      ref={setPanelRef}
      onScroll={onScroll}
      onWheel={onWheel}
      onTouchStart={onTouchStart}
      onMouseDown={onMouseDown}
    >
      {internalPageFindActive && (
        <div className="page-find-overlay">
          <div className="page-find-widget" role="search" aria-label="当前页面搜索">
            <Search size={15} />
            <input
              ref={pageFindInputRef}
              value={pageFindQuery}
              placeholder="搜索当前页面"
              aria-label="搜索当前页面"
              onChange={(event) => updatePageFindQuery(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault();
                  stepPageFind(event.shiftKey ? -1 : 1);
                } else if (event.key === "Escape") {
                  event.preventDefault();
                  closePageFind();
                }
              }}
            />
            <span className={`page-find-count ${pageFindQuery.trim() && pageFindMatchCount === 0 ? "empty" : ""}`.trim()}>
              {formatPageFindCount(pageFindQuery, pageFindMatchCount, pageFindActiveIndex)}
            </span>
            <button
              type="button"
              className="page-find-button"
              title="上一个"
              aria-label="上一个搜索结果"
              disabled={pageFindMatchCount === 0}
              onClick={() => stepPageFind(-1)}
            >
              <ArrowUp size={14} />
            </button>
            <button
              type="button"
              className="page-find-button"
              title="下一个"
              aria-label="下一个搜索结果"
              disabled={pageFindMatchCount === 0}
              onClick={() => stepPageFind(1)}
            >
              <ArrowDown size={14} />
            </button>
            <button
              type="button"
              className="page-find-button"
              title="关闭"
              aria-label="关闭当前页面搜索"
              onClick={closePageFind}
            >
              <X size={14} />
            </button>
          </div>
        </div>
      )}
      {nodes.length === 0 && !processing && emptyLabel && (
        <div className="preview-empty">{emptyLabel}</div>
      )}
      {transcriptItems.map((item) => {
        if (item.type === "node") {
          return renderNodeFrame(item.node);
        }
        const expanded = isFocusGroupExpanded(item.group);
        return (
          <FocusFold
            group={item.group}
            expanded={expanded}
            key={item.group.roundId}
            renderNodeFrame={renderNodeFrame}
            onToggle={() => toggleFocusGroup(item.group)}
          />
        );
      })}
      {processingLine && <ProcessingLine processing={processingLine} />}
    </main>
  );
}));

function usePageFindHighlights(
  rootRef: { current: HTMLElement | null },
  query: string,
  activeIndex: number,
  setMatchCount: (updater: (current: number) => number) => void,
  setActiveIndex: (updater: (current: number) => number) => void,
  options: PageFindMatchOptions & {
    target?: HistoryLocateTarget | null;
    matchHighlightName: string;
    activeHighlightName: string;
  },
  dependencies: DependencyList
) {
  useBrowserLayoutEffect(() => {
    const root = rootRef.current;
    let scrollFrame: number | null = null;
    clearPageFindHighlights(options.matchHighlightName, options.activeHighlightName);

    if (!root || !query.trim()) {
      setMatchCount((current) => current === 0 ? current : 0);
      setActiveIndex((current) => current === 0 ? current : 0);
      return () => {
        clearPageFindHighlights(options.matchHighlightName, options.activeHighlightName);
      };
    }

    const ranges = collectPageFindRanges(root, query, options);
    const count = ranges.length;
    setMatchCount((current) => current === count ? current : count);

    if (count === 0) {
      setActiveIndex((current) => current === 0 ? current : 0);
      return () => {
        clearPageFindHighlights(options.matchHighlightName, options.activeHighlightName);
      };
    }

    const targetActiveIndex = options.target
      ? ranges.findIndex((range) => pageFindRangeMatchesHistoryTarget(range, options.target))
      : -1;
    const clampedActiveIndex = targetActiveIndex >= 0
      ? targetActiveIndex
      : Math.min(Math.max(0, activeIndex), count - 1);
    if (clampedActiveIndex !== activeIndex) {
      setActiveIndex(() => clampedActiveIndex);
    }

    setPageFindHighlight(options.matchHighlightName, ranges);
    const activeRange = ranges[clampedActiveIndex];
    setPageFindHighlight(options.activeHighlightName, [activeRange]);
    scrollFrame = window.requestAnimationFrame(() => {
      scrollPageFindRangeIntoView(root, activeRange);
      scrollFrame = null;
    });

    return () => {
      if (scrollFrame !== null) {
        window.cancelAnimationFrame(scrollFrame);
      }
      clearPageFindHighlights(options.matchHighlightName, options.activeHighlightName);
    };
    // The caller passes render-shape dependencies so new transcript text gets searched.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    rootRef,
    query,
    activeIndex,
    options.caseSensitive,
    options.wholeWord,
    options.target,
    options.matchHighlightName,
    options.activeHighlightName,
    ...dependencies
  ]);
}

function collectPageFindRanges(
  root: HTMLElement,
  query: string,
  options: PageFindMatchOptions = {}
): Range[] {
  const ranges: Range[] = [];
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      return isPageFindTextNodeSearchable(root, node)
        ? NodeFilter.FILTER_ACCEPT
        : NodeFilter.FILTER_REJECT;
    }
  });
  let node = walker.nextNode();
  while (node) {
    const text = node.textContent ?? "";
    for (const match of pageFindTextMatchOffsets(text, query, options)) {
      const range = document.createRange();
      range.setStart(node, match.start);
      range.setEnd(node, match.end);
      ranges.push(range);
    }
    node = walker.nextNode();
  }
  return ranges;
}

function pageFindRangeMatchesHistoryTarget(range: Range, target: HistoryLocateTarget): boolean {
  const startElement = range.startContainer.parentElement;
  const frame = startElement?.closest<HTMLElement>(".chat-node-frame");
  return frame ? historyTargetMatchesElement(frame, target) : false;
}

function isPageFindTextNodeSearchable(root: HTMLElement, node: Node): boolean {
  const text = node.textContent;
  if (!text || text.trim() === "") return false;
  const element = node.parentElement;
  if (!element || !root.contains(element)) return false;
  const excluded = element.closest(PAGE_FIND_EXCLUDE_SELECTOR);
  if (excluded && root.contains(excluded)) return false;
  const style = window.getComputedStyle(element);
  return style.display !== "none" && style.visibility !== "hidden";
}

function setPageFindHighlight(name: string, ranges: Range[]): void {
  const registry = pageFindHighlightRegistry();
  if (!registry) return;
  registry.highlights.set(name, new registry.Highlight(...ranges));
}

function clearPageFindHighlights(
  matchHighlightName = PAGE_FIND_MATCH_HIGHLIGHT,
  activeHighlightName = PAGE_FIND_ACTIVE_HIGHLIGHT
): void {
  const registry = pageFindHighlightRegistry();
  registry?.highlights.delete(matchHighlightName);
  registry?.highlights.delete(activeHighlightName);
}

function pageFindHighlightRegistry(): {
  highlights: { set: (name: string, highlight: unknown) => void; delete: (name: string) => void };
  Highlight: new (...ranges: Range[]) => unknown;
} | null {
  if (typeof window === "undefined") return null;
  const highlights = (window.CSS as unknown as {
    highlights?: { set: (name: string, highlight: unknown) => void; delete: (name: string) => void };
  }).highlights;
  const Highlight = (window as unknown as { Highlight?: new (...ranges: Range[]) => unknown }).Highlight;
  if (!highlights || !Highlight) return null;
  return { highlights, Highlight };
}

function scrollPageFindRangeIntoView(root: HTMLElement, range: Range): void {
  const rect = range.getBoundingClientRect();
  if (rect.width === 0 && rect.height === 0) return;
  const rootRect = root.getBoundingClientRect();
  const desiredTop = root.scrollTop + rect.top - rootRect.top - Math.round(root.clientHeight * 0.32);
  const desiredLeft = root.scrollLeft + rect.left - rootRect.left - 24;
  root.scrollTo({
    top: Math.max(0, desiredTop),
    left: Math.max(0, desiredLeft),
    behavior: "smooth"
  });
}

function formatPageFindCount(query: string, count: number, activeIndex: number): string {
  if (!query.trim()) return "";
  if (count === 0) return "无结果";
  return `${Math.min(activeIndex, count - 1) + 1}/${count}`;
}

interface FocusFoldGroup {
  roundId: string;
  nodes: ChatNode[];
  summary: FocusFoldSummary;
}

interface FocusFoldSummary {
  toolCount: number;
  activity: string;
  active: boolean;
  label: string;
}

type FocusTranscriptItem =
  | { type: "node"; node: ChatNode }
  | { type: "focus-fold"; group: FocusFoldGroup };

export function buildFocusTranscriptItems(nodes: ChatNode[], processing?: ProcessingState): FocusTranscriptItem[] {
  const items: FocusTranscriptItem[] = [];
  let index = 0;
  while (index < nodes.length) {
    const node = nodes[index];
    items.push({ type: "node", node });
    if (node.kind !== "user") {
      index += 1;
      continue;
    }

    const roundStart = index + 1;
    let roundEnd = roundStart;
    while (roundEnd < nodes.length && nodes[roundEnd].kind !== "user") {
      roundEnd += 1;
    }

    const round = nodes.slice(roundStart, roundEnd);
    const roundProcessing = processing && roundEnd === nodes.length ? processing : undefined;
    const roundActive = roundProcessing !== undefined || round.some(isActiveFocusNode);
    if (roundActive) {
      if (round.length > 0 || roundProcessing) {
        items.push({
          type: "focus-fold",
          group: {
            roundId: node.id,
            nodes: round,
            summary: focusFoldSummary(round, roundProcessing, round)
          }
        });
      }
      index = roundEnd;
      continue;
    }

    const finalAgentOffset = lastFormalReplyOffset(round);
    if (finalAgentOffset < 0) {
      if (round.length > 0) {
        items.push({
          type: "focus-fold",
          group: {
            roundId: node.id,
            nodes: round,
            summary: focusFoldSummary(round, undefined, round)
          }
        });
      }
      index = roundEnd;
      continue;
    }

    const finalAgent = round[finalAgentOffset];
    const foldedNodes = round.filter((_, roundIndex) => roundIndex !== finalAgentOffset);
    if (foldedNodes.length > 0) {
      items.push({
        type: "focus-fold",
        group: {
          roundId: node.id,
          nodes: foldedNodes,
          summary: focusFoldSummary(foldedNodes, undefined, round)
        }
      });
    }
    items.push({ type: "node", node: finalAgent });
    index = roundEnd;
  }
  return items;
}

function lastFormalReplyOffset(nodes: ChatNode[]): number {
  for (let index = nodes.length - 1; index >= 0; index -= 1) {
    if (nodes[index].kind === "agent") {
      return index;
    }
  }
  return -1;
}

export function latestFocusTextNode(nodes: ChatNode[]): ChatNode | undefined {
  const offset = lastFormalReplyOffset(nodes);
  return offset >= 0 ? nodes[offset] : undefined;
}

function focusFoldSummary(
  nodes: ChatNode[],
  processing?: ProcessingState,
  summaryNodes: ChatNode[] = nodes
): FocusFoldSummary {
  const toolCount = nodes.filter((node) => node.kind === "tool").length;
  const active = processing !== undefined || nodes.some(isActiveFocusNode);
  const activity = active
    ? focusFoldActiveActivity(nodes, processing)
    : `耗时${formatFocusDurationSeconds(focusFoldDurationMs(summaryNodes))}秒`;
  return {
    toolCount,
    activity,
    active,
    label: `${formatFocusToolCount(toolCount)}, ${activity}`
  };
}

function focusFoldActiveActivity(nodes: ChatNode[], processing?: ProcessingState): string {
  const activeNode = [...nodes].reverse().find(isActiveFocusNode);
  if (activeNode) {
    return activeFocusNodeActivity(activeNode);
  }
  if (processing) {
    return "思考中";
  }
  return "思考中";
}

function isActiveFocusNode(node: ChatNode): boolean {
  if (node.complete === false) return true;
  return isActiveToolCallNode(node);
}

export function resolveTranscriptProcessingLine({
  nodes,
  processing,
  focusMode,
  activeCollapsedFocusFold
}: {
  nodes: ChatNode[];
  processing?: ProcessingState;
  focusMode: boolean;
  activeCollapsedFocusFold: boolean;
}): ProcessingState | null {
  if (hasActiveToolCall(nodes)) {
    return null;
  }
  if (focusMode && activeCollapsedFocusFold) {
    return null;
  }
  if (processing) {
    return processing;
  }
  if (hasActiveNonToolTranscriptWork(nodes)) {
    return {
      id: "transcript-active-processing",
      runId: "transcript-active",
      content: "处理中..."
    };
  }
  return null;
}

function hasActiveToolCall(nodes: ChatNode[]): boolean {
  return nodes.some(isActiveToolCallNode);
}

function isActiveToolCallNode(node: ChatNode): boolean {
  const tool = node.tool;
  if (node.kind !== "tool" || !tool) return false;
  return tool.status === "pending"
    || tool.status === "running"
    || tool.argsState === "pending"
    || tool.argsState === "streaming";
}

function hasActiveNonToolTranscriptWork(nodes: ChatNode[]): boolean {
  return nodes.some((node) => node.complete === false && !isActiveToolCallNode(node));
}

function activeFocusNodeActivity(node: ChatNode): string {
  switch (node.kind) {
    case "thinking":
      return "思考中";
    case "tool":
      return `调用工具中: ${toolDisplayName(node.tool)}`;
    case "compact":
      return "压缩上下文中";
    case "framework":
    case "system":
      return "准备中";
    case "agent":
      return "回答中";
    default:
      return "思考中";
  }
}

function toolDisplayName(tool?: ToolState): string {
  const name = tool?.name.trim();
  if (!name) return "工具";
  return name.includes("__") ? name.slice(name.lastIndexOf("__") + 2) : name;
}

function focusFoldDurationMs(nodes: ChatNode[]): number {
  const dividerDuration = latestRunDividerDurationMs(nodes);
  if (dividerDuration !== undefined) {
    return dividerDuration;
  }

  const timedNodes = nodes.filter((node) => (
    Number.isFinite(node.streamStartedAt)
    && Number.isFinite(node.streamFinishedAt)
    && (node.streamFinishedAt ?? 0) >= (node.streamStartedAt ?? 0)
  ));
  if (timedNodes.length > 0) {
    const startedAt = Math.min(...timedNodes.map((node) => node.streamStartedAt ?? 0));
    const finishedAt = Math.max(...timedNodes.map((node) => node.streamFinishedAt ?? 0));
    return Math.max(0, finishedAt - startedAt);
  }

  return nodes.reduce((sum, node) => {
    const duration = node.streamDurationMs ?? node.tool?.streamDurationMs ?? node.tool?.durationMs ?? 0;
    return Number.isFinite(duration) && duration > 0 ? sum + duration : sum;
  }, 0);
}

function latestRunDividerDurationMs(nodes: ChatNode[]): number | undefined {
  for (let index = nodes.length - 1; index >= 0; index -= 1) {
    const node = nodes[index];
    if (node.kind !== "divider") continue;
    if (
      node.streamDurationMs !== undefined
      && Number.isFinite(node.streamDurationMs)
      && node.streamDurationMs >= 0
    ) {
      return node.streamDurationMs;
    }
    const match = node.content.match(/(?:^|[·\s])(\d+(?:\.\d+)?)s\b/);
    if (!match) continue;
    const duration = Number(match[1]) * 1000;
    if (Number.isFinite(duration) && duration >= 0) {
      return duration;
    }
  }
  return undefined;
}

function formatFocusDurationSeconds(durationMs: number): string {
  return (Math.max(0, durationMs) / 1000).toFixed(1);
}

function formatFocusToolCount(count: number): string {
  return `已运行${Math.max(0, count)}个工具`;
}

function FocusFold({
  group,
  expanded,
  renderNodeFrame,
  onToggle
}: {
  group: FocusFoldGroup;
  expanded: boolean;
  renderNodeFrame: (node: ChatNode, key?: string) => ReactNode;
  onToggle: () => void;
}) {
  const visibleTextNode = group.summary.active ? latestFocusTextNode(group.nodes) : undefined;
  const showVisibleTextNode = !expanded && visibleTextNode !== undefined;

  return (
    <div className={`focus-fold ${expanded ? "expanded" : "collapsed"}`}>
      <button
        type="button"
        className="focus-fold-toggle"
        aria-expanded={expanded}
        onClick={onToggle}
      >
        {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        {group.summary.active && (
          <span className="focus-fold-spinner">
            <LiveSpinner title={group.summary.activity} />
          </span>
        )}
        <span className="focus-fold-summary">
          {group.summary.label}
        </span>
      </button>
      <div className="focus-fold-content-shell" aria-hidden={!expanded}>
        <div className="focus-fold-content">
          {group.nodes.map((node) => renderNodeFrame(node, `${group.roundId}:${node.id}`))}
        </div>
      </div>
      {visibleTextNode && (
        <div
          className={`focus-fold-visible-text ${showVisibleTextNode ? "is-visible" : "is-hidden"}`}
          key={visibleTextNode.id}
          aria-hidden={!showVisibleTextNode}
        >
          <div className="focus-fold-visible-text-inner">
            {renderNodeFrame(visibleTextNode, `${group.roundId}:visible:${visibleTextNode.id}`)}
          </div>
        </div>
      )}
    </div>
  );
}

function guardedToggle(toggle: () => void) {
  if (hasActiveTextSelection()) return;
  toggle();
}

function expandCollapsedBubbleContent(event: ReactMouseEvent<HTMLElement>, expand: () => void) {
  if (hasActiveTextSelection()) return;
  if (isInteractiveClickTarget(event.target, event.currentTarget)) return;
  expand();
}

function isInteractiveClickTarget(target: EventTarget | null, container: HTMLElement): boolean {
  if (!(target instanceof Element)) return false;
  const interactive = target.closest(
    "a, button, input, textarea, select, summary, details, [role='button'], [contenteditable='true']"
  );
  return interactive !== null && container.contains(interactive);
}

function guardNativeToggleDuringSelection(event: ReactMouseEvent<HTMLElement>) {
  if (!hasActiveTextSelection()) return;
  event.preventDefault();
  event.stopPropagation();
}

function useMeasuredContentHeight() {
  const ref = useRef<HTMLDivElement | null>(null);
  const [height, setHeight] = useState<number | null>(null);

  const syncHeight = useCallback(() => {
    const element = ref.current;
    if (!element) return;
    const next = Math.ceil(element.scrollHeight);
    setHeight((current) => current === next ? current : next);
  }, []);

  useBrowserLayoutEffect(() => {
    syncHeight();
  });

  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = typeof ResizeObserver === "undefined"
      ? null
      : new ResizeObserver(syncHeight);
    observer?.observe(element);
    element.addEventListener("hawi:markdown-content-resized", syncHeight);
    window.addEventListener("resize", syncHeight);
    return () => {
      observer?.disconnect();
      element.removeEventListener("hawi:markdown-content-resized", syncHeight);
      window.removeEventListener("resize", syncHeight);
    };
  }, [syncHeight]);

  return [ref, height] as const;
}

function AnimatedMessageBody({
  collapsed,
  expandTitle,
  onExpand,
  children
}: {
  collapsed: boolean;
  expandTitle: string;
  onExpand: (event: ReactMouseEvent<HTMLElement>) => void;
  children: ReactNode;
}) {
  const [contentRef, measuredHeight] = useMeasuredContentHeight();
  const style = measuredHeight === null
    ? undefined
    : ({ "--message-expanded-height": `${measuredHeight}px` } as CSSProperties);

  return (
    <div
      className={`message-body ${collapsed ? "is-collapsed can-expand" : ""}`}
      style={style}
      onClick={collapsed ? onExpand : undefined}
      title={collapsed ? expandTitle : undefined}
    >
      <div className="message-body-inner" ref={contentRef}>
        {children}
      </div>
      {collapsed && <span className="message-collapse-mask" aria-hidden="true" />}
    </div>
  );
}

const ChatBubble = memo(function ChatBubble({
  node,
  blobPreviewUrls,
  onOpenMediaPreview,
  allowFork,
  onForkMessage,
  messageEdit,
  profilingEnabled
}: {
  node: ChatNode;
  blobPreviewUrls: BlobPreviewUrls;
  onOpenMediaPreview?: (preview: MediaPreviewState) => void;
  allowFork: boolean;
  onForkMessage?: (node: ChatNode) => void;
  messageEdit?: MessageEditController;
  profilingEnabled: boolean;
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
    return <ToolBubble node={node} profilingEnabled={profilingEnabled} />;
  }
  if (node.kind === "framework" && node.framework) {
    return <FrameworkNodeBubble node={node} />;
  }
  if (node.kind === "handoff") {
    return <HandoffBubble node={node} />;
  }
  if (node.kind === "thinking") {
    return <ThinkingBubble node={node} />;
  }
  return (
    <MessageBubble
      node={node}
      blobPreviewUrls={blobPreviewUrls}
      onOpenMediaPreview={onOpenMediaPreview}
      allowFork={allowFork}
      onForkMessage={onForkMessage}
      messageEdit={messageEdit}
      profilingEnabled={profilingEnabled}
    />
  );
});

function ProcessingLine({ processing }: { processing: ProcessingState }) {
  return (
    <div className="processing-line">
      <LiveSpinner title="处理中" />
      <span>{processing.content || "处理中..."}</span>
    </div>
  );
}

const FrameworkNodeBubble = memo(function FrameworkNodeBubble({ node }: { node: ChatNode }) {
  if (!node.framework) return null;
  if (node.framework.kind === "system_prompt") {
    return <SystemPromptBubble node={node} />;
  }
  const childInjections = node.injections ?? [];
  if (childInjections.length === 0) {
    return <FrameworkBubble item={node.framework} />;
  }
  return (
    <div className="framework-stack">
      <FrameworkBubble item={node.framework} />
      <div className="message-injections framework-child-injections">
        {childInjections.map((item) => (
          <FrameworkBubble item={item} embedded key={item.id} />
        ))}
      </div>
    </div>
  );
});

const HandoffBubble = memo(function HandoffBubble({ node }: { node: ChatNode }) {
  return (
    <article className="bubble handoff">
      <div className="handoff-title">
        <GitFork size={15} />
        <span>前文延续</span>
      </div>
      <p>{node.content}</p>
    </article>
  );
});

const SystemPromptBubble = memo(function SystemPromptBubble({ node }: { node: ChatNode }) {
  const [collapsed, setCollapsed] = useState(true);
  const framework = node.framework;
  const childInjections = node.injections ?? [];
  const html = renderMarkdown(node.content);
  const toggleCollapsed = () => setCollapsed((value) => !value);
  if (!framework) return null;

  return (
    <article className={`bubble system-prompt message ${collapsed ? "message-collapsed" : ""}`}>
      <div className="bubble-head collapsible-head" onClick={() => guardedToggle(toggleCollapsed)}>
        <span className="bubble-title">
          {framework.label}
        </span>
        <span className="message-actions">
          <CopyButton text={node.content} title="复制 System prompt" />
          <button
            className="thinking-toggle message-toggle"
            title={collapsed ? "展开 System prompt" : "折叠 System prompt"}
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
      <AnimatedMessageBody
        collapsed={collapsed}
        expandTitle="点击展开 System prompt"
        onExpand={(event) => expandCollapsedBubbleContent(event, () => setCollapsed(false))}
      >
        <MarkdownView html={html} />
      </AnimatedMessageBody>
      {childInjections.length > 0 && (
        <div className="message-injections after">
          <div className="message-injections-inner">
            {childInjections.map((item) => (
              <FrameworkBubble item={item} embedded key={item.id} />
            ))}
          </div>
        </div>
      )}
    </article>
  );
});

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
      <div className="bubble-head collapsible-head" onClick={() => guardedToggle(toggleCollapsed)}>
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
          <MarkdownView html={html} />
        </div>
      </div>
    </article>
  );
});

const MessageBubble = memo(function MessageBubble({
  node,
  blobPreviewUrls,
  onOpenMediaPreview,
  allowFork,
  onForkMessage,
  messageEdit,
  profilingEnabled
}: {
  node: ChatNode;
  blobPreviewUrls: BlobPreviewUrls;
  onOpenMediaPreview?: (preview: MediaPreviewState) => void;
  allowFork: boolean;
  onForkMessage?: (node: ChatNode) => void;
  messageEdit?: MessageEditController;
  profilingEnabled: boolean;
}) {
  const [collapsed, setCollapsed] = useState(false);
  const html = node.kind === "agent" ? renderMarkdown(node.content) : escapeText(node.content);
  const hasStructuredContent = shouldUseContentPartsView(node.contentParts);
  const label = node.kind === "user" ? labelForUserMessage(node) : labelForKind(node.kind);
  const receiving = node.kind === "agent" && node.complete === false;
  const editing = messageEdit?.active?.nodeId === node.id;
  const toggleCollapsed = () => setCollapsed((value) => !value);
  const canFork = allowFork
    && node.canFork === true
    && onForkMessage
    && (Boolean(node.contextMessageId) || typeof node.contextMessageIndex === "number");
  const canEdit = messageEdit?.canEdit === true && isEditableTextNode(node);
  const showEditButton = Boolean(messageEdit && (node.kind === "user" || node.kind === "agent"));
  const beforeInjections = (node.injections ?? []).filter((item) => item.mergePosition === "before");
  const afterInjections = (node.injections ?? []).filter((item) => item.mergePosition !== "before");
  const profileLine = profilingEnabled ? formatMessageProfileLine(node) : null;
  const prefillProgress = profilingEnabled && node.kind === "user" && !editing
    ? resolvePrefillProgressSegments(node.profile)
    : null;
  const showPrefillOverlay = prefillProgress !== null && !collapsed;
  const bubbleStyle = prefillProgressStyle(prefillProgress);

  return (
    <article
      className={`bubble ${node.kind} message ${profileLine ? "has-profile" : ""} ${prefillProgress !== null ? "prefill-progress" : ""} ${showPrefillOverlay ? "prefill-overlay" : ""} ${collapsed ? "message-collapsed" : ""} ${editing ? "message-editing" : ""}`.trim()}
      style={bubbleStyle}
    >
      <div
        className={`bubble-head ${editing ? "" : "collapsible-head"}`.trim()}
        onClick={editing ? undefined : () => guardedToggle(toggleCollapsed)}
      >
        <span className="bubble-title">
          {label}
          <BlockStreamStatus
            receiving={receiving}
            durationMs={node.streamDurationMs}
            receivingTitle="正在接收消息"
            receivedChars={receivedCharCount(node.content)}
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
                onForkMessage?.(node);
              }}
            >
              <GitFork size={14} />
            </button>
          )}
          {showEditButton && messageEdit && (
            <button
              className="thinking-toggle message-edit"
              title={canEdit ? "编辑消息" : "当前不能编辑消息"}
              aria-label="编辑消息"
              disabled={!canEdit}
              onClick={(event) => {
                event.stopPropagation();
                if (canEdit) {
                  messageEdit.onStart(node);
                }
              }}
            >
              <Pencil size={14} />
            </button>
          )}
          <CopyButton text={node.content} title="复制消息" />
          {!editing && (
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
          )}
        </span>
      </div>
      {beforeInjections.length > 0 && (
        <div className="message-injections before">
          <div className="message-injections-inner">
            {beforeInjections.map((item) => (
              <FrameworkBubble item={item} embedded key={item.id} />
            ))}
          </div>
        </div>
      )}
      {editing && messageEdit?.active ? (
        <MessageEditBubble
          edit={messageEdit.active}
          blobPreviewUrls={blobPreviewUrls}
          maxHeight={messageEdit.maxHeight}
          onDraftChange={messageEdit.onDraftChange}
          onAddFiles={messageEdit.onAddFiles}
          onRemoveAttachment={messageEdit.onRemoveAttachment}
          onCancel={messageEdit.onCancel}
          onSave={messageEdit.onSave}
          onResend={messageEdit.onResend}
        />
      ) : (
        <AnimatedMessageBody
          collapsed={collapsed}
          expandTitle="点击展开消息"
          onExpand={(event) => expandCollapsedBubbleContent(event, () => setCollapsed(false))}
        >
          {hasStructuredContent ? (
            <ContentPartsView
              parts={node.contentParts ?? []}
              fallbackText={node.content}
              markdown={node.kind === "agent"}
              blobPreviewUrls={blobPreviewUrls}
              onOpenMediaPreview={onOpenMediaPreview}
            />
          ) : (
            <MarkdownView html={html} />
          )}
        </AnimatedMessageBody>
      )}
      {afterInjections.length > 0 && (
        <div className="message-injections after">
          <div className="message-injections-inner">
            {afterInjections.map((item) => (
              <FrameworkBubble item={item} embedded key={item.id} />
            ))}
          </div>
        </div>
      )}
      {prefillProgress && (
        <div className="message-prefill-progress-bar" aria-hidden="true">
          <span className="message-prefill-progress-cache" />
          <span className="message-prefill-progress-prefill" />
        </div>
      )}
      {profileLine && <div className="message-profile-line">{profileLine}</div>}
    </article>
  );
});

export function resolvePrefillRevealProgress(profile?: ModelProfileState): number | null {
  return resolvePrefillProgressSegments(profile)?.revealProgress ?? null;
}

function prefillProgressStyle(progress: PrefillProgressSegments | null): CSSProperties | undefined {
  return progress === null
    ? undefined
    : ({
        "--prefill-reveal-progress": `${(progress.revealProgress * 100).toFixed(2)}%`,
        "--prefill-cache-progress": `${(progress.cacheProgress * 100).toFixed(2)}%`,
        "--prefill-filled-progress": `${(progress.prefillProgress * 100).toFixed(2)}%`
      } as CSSProperties);
}

export interface PrefillProgressSegments {
  revealProgress: number;
  cacheProgress: number;
  prefillProgress: number;
}

export function resolvePrefillProgressSegments(profile?: ModelProfileState): PrefillProgressSegments | null {
  if (!profile || !isProfileNumber(profile.prefillTokens) || !isPositiveProfileNumber(profile.prefillTotalTokens)) {
    return null;
  }
  const prefillTotalTokens = Math.max(0, profile.prefillTotalTokens);
  const prefillTokens = Math.min(prefillTotalTokens, Math.max(0, profile.prefillTokens));
  const revealProgress = prefillTokens / prefillTotalTokens;
  if (revealProgress >= 1) {
    return null;
  }
  const cacheTokens = isPositiveProfileNumber(profile.cacheTokens) ? profile.cacheTokens : 0;
  const totalTokens = cacheTokens + prefillTotalTokens;
  if (totalTokens <= 0) {
    return null;
  }
  return {
    revealProgress,
    cacheProgress: cacheTokens / totalTokens,
    prefillProgress: prefillTokens / totalTokens
  };
}

function formatMessageProfileLine(node: ChatNode): string | null {
  const profile = node.profile;
  if (!profile) return null;
  if (node.kind === "user") {
    return formatUserProfileLine(profile);
  }
  if (node.kind === "agent") {
    return formatAssistantProfileLine(profile);
  }
  return null;
}

function formatUserProfileLine(profile: ModelProfileState): string | null {
  const parts: string[] = [];
  if (isProfileNumber(profile.cacheTokens)) {
    parts.push(`Cache: ${formatProfileTokenCount(profile.cacheTokens)} tok`);
  }
  const prefillParts: string[] = [];
  if (isProfileNumber(profile.prefillTokens)) {
    prefillParts.push(`${formatProfileTokenCount(profile.prefillTokens)} tok`);
  }
  if (isPositiveProfileNumber(profile.prefillTokensPerSecond)) {
    prefillParts.push(`@ ${formatProfileRate(profile.prefillTokensPerSecond)}`);
  }
  if (isProfileNumber(profile.prefillMs)) {
    prefillParts.push(formatProfileMs(profile.prefillMs));
  }
  if (prefillParts.length > 0) {
    parts.push(`Prefill: ${prefillParts.join(" ")}`);
  }
  return parts.length > 0 ? parts.join("  ") : null;
}

function formatAssistantProfileLine(profile: ModelProfileState): string | null {
  const parts: string[] = [];
  if (isProfileNumber(profile.ttftMs)) {
    parts.push(`TTFT: ${formatProfileMs(profile.ttftMs)}`);
  }
  const decodeParts: string[] = [];
  if (isProfileNumber(profile.decodeTokens)) {
    decodeParts.push(`${formatProfileTokenCount(profile.decodeTokens)} tok`);
  }
  if (isPositiveProfileNumber(profile.decodeTokensPerSecond)) {
    decodeParts.push(`@ ${formatProfileRate(profile.decodeTokensPerSecond)}`);
  }
  if (decodeParts.length > 0) {
    parts.push(`Decode: ${decodeParts.join(" ")}`);
  }
  if (isPositiveProfileNumber(profile.peakDecodeTokensPerSecond)) {
    parts.push(`Peak: ${formatProfileRate(profile.peakDecodeTokensPerSecond)}`);
  }
  if (isProfileNumber(profile.decodeMs)) {
    parts.push(formatProfileMs(profile.decodeMs));
  }
  return parts.length > 0 ? parts.join("  ") : null;
}

function isProfileNumber(value: number | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isPositiveProfileNumber(value: number | undefined): value is number {
  return isProfileNumber(value) && value > 0;
}

function formatProfileMs(value: number): string {
  return `${Math.max(0, Math.round(value))}ms`;
}

function formatProfileTokenCount(value: number): string {
  return Math.max(0, Math.round(value)).toLocaleString();
}

function formatProfileRate(value: number): string {
  return `${value.toFixed(1)} tok/s`;
}

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
  if (item.kind === "tool_runtime_context_injected") {
    if (item.toolName && item.parameterName) {
      return `${item.toolName}.${item.parameterName}`;
    }
    return item.toolName ?? item.parameterName ?? "runtime context";
  }
  return item.toolCallId ? `tool ${item.toolCallId}` : undefined;
}

const ThinkingBubble = memo(function ThinkingBubble({ node }: { node: ChatNode }) {
  const [collapsed, setCollapsed] = useState(true);
  const html = renderMarkdown(node.content);
  const receiving = node.complete === false;
  const toggleCollapsed = () => setCollapsed((value) => !value);

  return (
    <article className={`bubble thinking ${collapsed ? "collapsed" : ""}`}>
      <div className="bubble-head collapsible-head" onClick={() => guardedToggle(toggleCollapsed)}>
        <span className="bubble-title">
          <Brain size={15} /> Thinking
          <BlockStreamStatus
            receiving={receiving}
            durationMs={node.streamDurationMs}
            receivingTitle="正在接收思考内容"
            receivedChars={receivedCharCount(node.content)}
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
          <MarkdownView html={html} />
        </div>
      </div>
    </article>
  );
});

const ToolBubble = memo(function ToolBubble({
  node,
  profilingEnabled
}: {
  node: ChatNode;
  profilingEnabled: boolean;
}) {
  const tool = node.tool!;
  const running = tool.status === "running";
  const receivingArguments = tool.argsState !== "complete";
  const [collapsed, setCollapsed] = useState(true);
  const presentation = toolPresentation(tool);
  const toggleCollapsed = () => setCollapsed((value) => !value);
  const prefillProgress = profilingEnabled
    ? resolvePrefillProgressSegments(node.profile)
    : null;
  const showPrefillOverlay = prefillProgress !== null && !collapsed;
  const bubbleStyle = prefillProgressStyle(prefillProgress);

  return (
    <article
      className={`bubble tool ${tool.status} ${prefillProgress !== null ? "prefill-progress" : ""} ${showPrefillOverlay ? "prefill-overlay" : ""} ${collapsed ? "collapsed" : ""}`.trim()}
      style={bubbleStyle}
    >
      <div className="bubble-head collapsible-head" onClick={() => guardedToggle(toggleCollapsed)}>
        <span className="tool-title">
          <span className="tool-name">
            <Wrench size={15} /> {presentation.label}
          </span>
          {presentation.detail && (
            <span className={`tool-subject ${presentation.detailKind}`}>
              {presentation.detail}
            </span>
          )}
          <BlockStreamStatus
            receiving={receivingArguments}
            durationMs={tool.streamDurationMs}
            receivingTitle="正在接收工具调用"
            receivedChars={receivedCharCount(tool.argsRaw)}
          />
        </span>
        <span className="tool-actions">
          <span className="tool-status">
            <strong>{running ? "running" : tool.status}</strong>
            {running && <LiveSpinner title="工具运行中" />}
          </span>
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
            <summary onClick={guardNativeToggleDuringSelection}>
              Arguments
              {tool.argsState !== "complete" && <span className="detail-state">receiving</span>}
            </summary>
            {renderToolArguments(tool)}
          </details>
          {(tool.resultPreview || tool.resultData !== undefined || tool.status === "fail") && (
            <details open>
              <summary onClick={guardNativeToggleDuringSelection}>Result {tool.durationMs ? `· ${tool.durationMs.toFixed(0)}ms` : ""}</summary>
              {renderToolResult(tool)}
            </details>
          )}
        </div>
      </div>
      {prefillProgress && (
        <div className="message-prefill-progress-bar" aria-hidden="true">
          <span className="message-prefill-progress-cache" />
          <span className="message-prefill-progress-prefill" />
        </div>
      )}
    </article>
  );
});

/** 根据工具名称渲染参数区域 */
export function renderToolArguments(tool: ToolState): ReactNode {
  if (tool.argsState !== "complete") {
    return tool.argsRaw
      ? <CopyablePreBlock className="argument-code" value={tool.argsRaw} />
      : <div className="tool-hint">Receiving arguments...</div>;
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

  // read_file: 兼容旧版 resultData 结构化输出做语法高亮；文本输出走 preview。
  if (toolKind === "read_file") {
    // 优先从旧版 resultData 解析结构化数据
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
      <CopyablePreBlock className="tool-result-block terminal nowrap" value={preview} />
    );
  }

  // list_dir: ls -la 终端风格
  if (toolKind === "list_dir") {
    const resultData = isRecord(tool.resultData) ? tool.resultData : undefined;
    const lsText = resultData?.type === "ls_output" || resultData?.type === "directory"
      ? String(resultData.text ?? "")
      : preview;
    const meta = resultData
      ? [
        formatCount(resultData.numEntries, "entry", "entries"),
        resultData.isTruncated === true ? "truncated" : ""
      ].filter(isNonEmptyString)
      : [];
    return (
      <div className="tool-result-view">
        <ToolResultMeta items={meta} />
        <CopyablePreBlock className="tool-result-block terminal nowrap" value={lsText} />
      </div>
    );
  }

  if (toolKind === "glob") {
    const resultData = isRecord(tool.resultData) ? tool.resultData : undefined;
    if (!resultData) {
      return (
        <div className="tool-result-view">
          <CopyablePreBlock
            className="tool-result-block nowrap"
            value={preview || "No matches."}
          />
        </div>
      );
    }
    const matches = Array.isArray(resultData?.matches)
      ? resultData.matches.map((item) => String(item))
      : [];
    return (
      <div className="tool-result-view">
        <ToolResultMeta items={[formatCount(matches.length, "match", "matches")]} />
        <CopyablePreBlock
          className="tool-result-block nowrap"
          value={matches.length > 0 ? matches.join("\n") : "No matches."}
        />
      </div>
    );
  }

  if (toolKind === "grep") {
    const resultData = isRecord(tool.resultData) ? tool.resultData : undefined;
    if (!resultData) {
      return (
        <div className="tool-result-view">
          <CopyablePreBlock className="tool-result-block search nowrap" value={preview || "No matches."} />
        </div>
      );
    }
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
        <CopyablePreBlock className="tool-result-block search nowrap" value={content || "No matches."} />
      </div>
    );
  }

  // 其他工具: 纯文本结果（已在 formatToolResultText 中渲染为文本）
  return (
    <CopyablePreBlock
      className="tool-result-block"
      value={preview || "Tool failed without an error message."}
    />
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
    <CopyableCodeShell text={value}>
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
    </CopyableCodeShell>
  );
}));

function BlockStreamStatus({
  receiving,
  durationMs,
  receivingTitle,
  receivedChars,
}: {
  receiving: boolean;
  durationMs?: number;
  receivingTitle: string;
  receivedChars?: number;
}) {
  if (receiving) {
    return (
      <span className="block-stream-status receiving">
        <LiveSpinner title={receivingTitle} />
        {receivedChars !== undefined && (
          <span className="block-stream-count">{formatReceivedCharsLabel(receivedChars)}</span>
        )}
      </span>
    );
  }
  const label = formatStreamFinishedLabel(durationMs);
  if (!label) {
    return null;
  }
  return <span className="block-stream-status">{label}</span>;
}

function receivedCharCount(value: string): number {
  return Array.from(value).length;
}

function formatReceivedCharsLabel(value: number): string {
  return `${Math.max(0, value)} 字符`;
}

function LiveSpinner({ title }: { title: string }) {
  return (
    <span className="live-spinner" title={title} aria-label={title} role="status">
      <LoaderCircle size={15} />
    </span>
  );
}

const MarkdownView = memo(function MarkdownView({ html, className = "" }: { html: string; className?: string }) {
  const rootRef = useRef<HTMLDivElement | null>(null);
  useCodeBlockOverflowState(rootRef, [html]);

  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    const disclosures = Array.from(root.querySelectorAll<HTMLElement>("details.source-code-disclosure"));
    const handleToggle = (event: Event) => {
      const disclosure = event.currentTarget;
      if (disclosure instanceof HTMLElement) {
        notifyMarkdownContentResized(disclosure);
      }
    };
    for (const disclosure of disclosures) {
      disclosure.addEventListener("toggle", handleToggle);
    }
    return () => {
      for (const disclosure of disclosures) {
        disclosure.removeEventListener("toggle", handleToggle);
      }
    };
  }, [html]);

  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    let cancelled = false;
    const frameIds: number[] = [];
    const timeoutIds: number[] = [];

    const run = () => {
      if (cancelled) return;
      void renderMermaidDiagrams(root, () => cancelled).finally(() => {
        if (!cancelled && hasPendingMermaidDiagrams(root)) {
          schedule(240);
        }
      });
    };

    const schedule = (delayMs: number) => {
      if (delayMs <= 0) {
        frameIds.push(window.requestAnimationFrame(run));
        return;
      }
      timeoutIds.push(window.setTimeout(run, delayMs));
    };

    schedule(0);
    for (const delayMs of MERMAID_RENDER_RETRY_DELAYS_MS) {
      schedule(delayMs);
    }

    return () => {
      cancelled = true;
      for (const frameId of frameIds) {
        window.cancelAnimationFrame(frameId);
      }
      for (const timeoutId of timeoutIds) {
        window.clearTimeout(timeoutId);
      }
    };
  }, [html]);

  async function handleClick(event: ReactMouseEvent<HTMLDivElement>) {
    if (hasActiveTextSelection()) return;
    const target = event.target instanceof Element
      ? event.target.closest<HTMLButtonElement>(".code-copy-button")
      : null;
    if (!target || !event.currentTarget.contains(target)) return;

    event.stopPropagation();
    const block = target.closest(".code-block-shell");
    const code = block?.querySelector("pre code")?.textContent ?? "";
    await copyMarkdownCodeBlock(target, code);
  }

  return (
    <div
      ref={rootRef}
      className={`markdown ${className}`.trim()}
      onClick={handleClick}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
});

function useCodeBlockOverflowState(
  rootRef: { current: HTMLElement | null },
  dependencies: DependencyList = []
) {
  useBrowserLayoutEffect(() => {
    if (typeof window === "undefined") return;
    const root = rootRef.current;
    if (!root) return;

    let animationFrame: number | null = null;
    const sync = () => {
      syncCodeBlockOverflowState(root);
    };
    const scheduleSync = () => {
      if (typeof window.requestAnimationFrame !== "function") {
        sync();
        return;
      }
      if (animationFrame !== null) return;
      animationFrame = window.requestAnimationFrame(() => {
        animationFrame = null;
        sync();
      });
    };

    scheduleSync();
    const observer = typeof ResizeObserver === "undefined"
      ? null
      : new ResizeObserver(scheduleSync);
    observer?.observe(root);
    root.querySelectorAll<HTMLElement>(".code-block-shell > pre, .copyable-code-shell > pre").forEach((pre) => {
      observer?.observe(pre);
    });
    window.addEventListener("resize", scheduleSync);
    return () => {
      if (animationFrame !== null && typeof window.cancelAnimationFrame === "function") {
        window.cancelAnimationFrame(animationFrame);
      }
      observer?.disconnect();
      window.removeEventListener("resize", scheduleSync);
    };
    // The caller passes render-shape dependencies so newly inserted <pre> nodes are observed.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, dependencies);
}

function syncCodeBlockOverflowState(root: HTMLElement): void {
  const shells = [
    ...(root.matches(".code-block-shell, .copyable-code-shell") ? [root] : []),
    ...root.querySelectorAll<HTMLElement>(".code-block-shell, .copyable-code-shell")
  ];
  shells.forEach((shell) => {
    const pre = shell.querySelector<HTMLElement>("pre");
    const hasHorizontalScroll = pre ? codeBlockHasHorizontalOverflow(pre) : false;
    shell.classList.toggle("has-horizontal-scroll", hasHorizontalScroll);
  });
}

export function codeBlockHasHorizontalOverflow({
  clientWidth,
  scrollWidth
}: {
  clientWidth: number;
  scrollWidth: number;
}): boolean {
  return scrollWidth > clientWidth + 1;
}

async function copyMarkdownCodeBlock(button: HTMLButtonElement, text: string): Promise<void> {
  let state: "copied" | "failed" = "copied";
  try {
    await copyTextToClipboard(text);
  } catch {
    state = "failed";
  }

  const existingTimer = markdownCodeCopyTimers.get(button);
  if (existingTimer !== undefined) {
    window.clearTimeout(existingTimer);
  }

  const label = state === "copied" ? "已复制" : "复制失败";
  button.dataset.copyState = state;
  button.title = label;
  button.setAttribute("aria-label", label);

  const timer = window.setTimeout(() => {
    button.dataset.copyState = "idle";
    button.title = "复制代码";
    button.setAttribute("aria-label", "复制代码");
    markdownCodeCopyTimers.delete(button);
  }, COPY_FEEDBACK_MS);
  markdownCodeCopyTimers.set(button, timer);
}

function CopyableCodeShell({ text, children }: { text: string; children: ReactNode }) {
  const shellRef = useRef<HTMLDivElement | null>(null);
  useCodeBlockOverflowState(shellRef, [text]);

  return (
    <div className="copyable-code-shell" ref={shellRef}>
      <CopyButton text={text} title="复制代码" className="code-copy-overlay" />
      {children}
    </div>
  );
}

function CopyablePreBlock({ className, value }: { className: string; value: string }) {
  return (
    <CopyableCodeShell text={value}>
      <pre className={className} onWheel={handleNestedVerticalScroll}>{value}</pre>
    </CopyableCodeShell>
  );
}

function CopyButton({ text, title, className = "" }: { text: string; title: string; className?: string }) {
  const [state, setState] = useState<"idle" | "copied" | "failed">("idle");
  const timerRef = useRef<number | null>(null);

  useEffect(() => () => {
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
    }
  }, []);

  async function handleCopy(event: ReactMouseEvent<HTMLButtonElement>) {
    event.stopPropagation();
    if (hasActiveTextSelection()) return;
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
      className={`copy-button ${className} ${copied ? "copied" : ""} ${failed ? "failed" : ""}`.trim()}
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
    return <CopyablePreBlock className="argument-code" value={JSON.stringify(value, null, 2)} />;
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

interface PluginActionPayload extends Record<string, unknown> {
  plugin_id: string;
  action: string;
  arguments: Record<string, unknown>;
}

function WorkspaceSidebar({
  artifacts,
  artifactOrder,
  selectedArtifactId,
  artifactGroups,
  sideThreads,
  messages,
  statuses,
  toolProgress,
  subagents,
  activeTab,
  onActiveTabChange,
  onOpenSideThread,
  onDeleteSideThread,
  onSelectArtifact,
  onPluginAction,
  onObserve
}: {
  artifacts: Record<string, PluginArtifactState>;
  artifactOrder: string[];
  selectedArtifactId?: string;
  artifactGroups: ArtifactTypeGroup[];
  sideThreads: SideThreadState[];
  messages: PluginMessageState[];
  statuses: Record<string, PluginStatusState>;
  toolProgress: Record<string, ToolProgressState>;
  subagents: SubAgentRuntimeState[];
  activeTab: WorkspaceSidebarTab | null;
  onActiveTabChange: (tab: WorkspaceSidebarTab | null) => void;
  onOpenSideThread: (thread: SideThreadState) => void;
  onDeleteSideThread: (thread: SideThreadState) => void;
  onSelectArtifact: (artifactKey: string) => void;
  onPluginAction: (payload: PluginActionPayload) => Promise<CoreFrame | null>;
  onObserve: (subagentId: string) => void;
}) {
  const artifactList = artifactOrder.map((key) => artifacts[key]).filter(Boolean);
  const activeArtifactType = artifactTypeFromSidebarTab(activeTab);
  const selectedCandidate = selectedArtifactId && artifacts[selectedArtifactId]
    ? artifacts[selectedArtifactId]
    : artifactList[0];
  const selectedGroup = activeArtifactType
    ? artifactGroups.find((group) => group.type === activeArtifactType)
    : undefined;
  const fallbackGroup = selectedCandidate
    ? artifactGroups.find((group) => group.type === selectedCandidate.artifactType)
    : artifactGroups[0];
  const displayedGroup = selectedGroup ?? fallbackGroup;
  const selected = selectedCandidate && selectedCandidate.artifactType === displayedGroup?.type
    ? selectedCandidate
    : displayedGroup?.artifacts[0];
  const canShowArtifact = activeArtifactType !== null && displayedGroup !== undefined && selected !== undefined;
  const canShowSideThreads = activeTab === "side_threads" && sideThreads.length > 0;
  const canShowSubagents = activeTab === "subagents" && subagents.length > 0;
  const open = canShowArtifact || canShowSideThreads || canShowSubagents;

  function toggleArtifactGroup(group: ArtifactTypeGroup) {
    const tab = artifactSidebarTab(group.type);
    if (activeTab === tab) {
      onActiveTabChange(null);
      return;
    }
    onSelectArtifact(group.artifacts[0].key);
    onActiveTabChange(tab);
  }

  function toggleSubagents() {
    onActiveTabChange(activeTab === "subagents" ? null : "subagents");
  }

  function toggleSideThreads() {
    onActiveTabChange(activeTab === "side_threads" ? null : "side_threads");
  }

  return (
    <aside className={`workspace-sidebar ${open ? "open" : "closed"}`} aria-label="Workspace sidebar">
      {canShowArtifact && displayedGroup && selected && (
        <ArtifactSidebarContent
          group={displayedGroup}
          selected={selected}
          messages={messages}
          statuses={statuses}
          toolProgress={toolProgress}
          onClose={() => onActiveTabChange(null)}
          onSelectArtifact={(artifactKey) => {
            onSelectArtifact(artifactKey);
            const artifact = artifacts[artifactKey];
            if (artifact) {
              onActiveTabChange(artifactSidebarTab(artifact.artifactType));
            }
          }}
          onPluginAction={onPluginAction}
        />
      )}
      {canShowSideThreads && (
        <SideThreadSidebarContent
          sideThreads={sideThreads}
          onClose={() => onActiveTabChange(null)}
          onOpen={onOpenSideThread}
          onDelete={onDeleteSideThread}
        />
      )}
      {canShowSubagents && (
        <SubAgentSidebarContent
          subagents={subagents}
          onClose={() => onActiveTabChange(null)}
          onObserve={onObserve}
        />
      )}
      <nav className="workspace-sidebar-rail" aria-label="Workspace side panels">
        {artifactGroups.map((group) => {
          const tab = artifactSidebarTab(group.type);
          const active = activeTab === tab;
          return (
            <button
              key={group.type}
              type="button"
              className={active ? "workspace-sidebar-tab active" : "workspace-sidebar-tab"}
              title={`${group.label} (${group.artifacts.length})`}
              aria-pressed={active}
              onClick={() => toggleArtifactGroup(group)}
            >
              <span>{group.label}</span>
              <strong>{group.artifacts.length}</strong>
            </button>
          );
        })}
        {sideThreads.length > 0 && (
          <button
            type="button"
            className={activeTab === "side_threads" ? "workspace-sidebar-tab active" : "workspace-sidebar-tab"}
            title={`Side Threads (${sideThreads.length})`}
            aria-pressed={activeTab === "side_threads"}
            onClick={toggleSideThreads}
          >
            <span>Side</span>
            <strong>{sideThreads.length}</strong>
          </button>
        )}
        {subagents.length > 0 && (
          <button
            type="button"
            className={activeTab === "subagents" ? "workspace-sidebar-tab active" : "workspace-sidebar-tab"}
            title={`SubAgents (${subagents.length})`}
            aria-pressed={activeTab === "subagents"}
            onClick={toggleSubagents}
          >
            <span>SubAgents</span>
            <strong>{subagents.length}</strong>
          </button>
        )}
      </nav>
    </aside>
  );
}

function SideThreadSidebarContent({
  sideThreads,
  onClose,
  onOpen,
  onDelete,
}: {
  sideThreads: SideThreadState[];
  onClose: () => void;
  onOpen: (thread: SideThreadState) => void;
  onDelete: (thread: SideThreadState) => void;
}) {
  const runningCount = sideThreads.filter((thread) => thread.status === "running").length;
  const sorted = [...sideThreads].sort((left, right) => (right.createdAt ?? 0) - (left.createdAt ?? 0));

  return (
    <section className="side-thread-preview workspace-sidebar-content">
      <header className="preview-head">
        <span><MessageSquare size={15} /> Side</span>
        <button
          type="button"
          className="icon-button"
          title="隐藏 Side"
          aria-label="隐藏 Side"
          onClick={onClose}
        >
          <ChevronRight size={17} />
        </button>
      </header>
      <div className="side-thread-summary">
        <strong>{runningCount}</strong>
        <span>running</span>
        <strong>{sideThreads.length}</strong>
        <span>total</span>
      </div>
      <div className="side-thread-list">
        {sorted.map((thread) => (
          <article
            className={`side-thread-item ${thread.status === "running" ? "active" : ""}`}
            key={thread.id}
            title={thread.anchorPreview || thread.quotedPreview}
          >
            <button
              type="button"
              className="side-thread-item-main"
              onClick={() => onOpen(thread)}
            >
              <span className="side-thread-item-head">
                <strong>{sideThreadTitle(thread)}</strong>
                <em>{sideThreadStatusLabel(thread.status)}</em>
              </span>
              <span className="side-thread-item-anchor">
                {thread.quotedPreview || thread.anchorPreview || "旁路对话"}
              </span>
              <span className="side-thread-item-text">{sideThreadLastText(thread)}</span>
            </button>
            <button
              type="button"
              className="side-thread-delete icon-button"
              title="删除旁路对话"
              aria-label={`删除旁路对话：${sideThreadTitle(thread)}`}
              onClick={() => onDelete(thread)}
            >
              <Trash2 size={14} />
            </button>
          </article>
        ))}
      </div>
    </section>
  );
}

function SubAgentSidebarContent({
  subagents,
  onClose,
  onObserve,
}: {
  subagents: SubAgentRuntimeState[];
  onClose: () => void;
  onObserve: (subagentId: string) => void;
}) {
  const activeCount = subagents.filter(isSubAgentActive).length;
  const sorted = sortSubAgentsByCreatedAt(subagents);

  return (
    <section className="subagent-preview workspace-sidebar-content">
      <header className="preview-head">
        <span><Bot size={15} /> SubAgents</span>
        <button
          type="button"
          className="icon-button"
          title="隐藏 SubAgents"
          aria-label="隐藏 SubAgents"
          onClick={onClose}
        >
          <ChevronRight size={17} />
        </button>
      </header>
      <div className="subagent-summary">
        <strong>{activeCount}</strong>
        <span>running</span>
        <strong>{subagents.length}</strong>
        <span>total</span>
      </div>
      <div className="subagent-list">
        {sorted.map((item) => (
          <button
            type="button"
            className={`subagent-item ${isSubAgentActive(item) ? "active" : ""}`}
            key={item.id}
            title={item.id}
            onClick={() => onObserve(item.id)}
          >
            <span className="subagent-item-head">
              <strong>{item.name}</strong>
              <em>{subAgentStateLabel(item)}</em>
            </span>
            <span className="subagent-item-meta">
              {item.role}
              {item.nodes.length > 0 ? ` · ${item.nodes.length} nodes` : ""}
            </span>
            {item.lastEventType && (
              <span className="subagent-item-event">{item.lastEventType}</span>
            )}
          </button>
        ))}
      </div>
    </section>
  );
}

function SubAgentObserverModal({
  subagent,
  focusMode,
  onClose,
}: {
  subagent: SubAgentRuntimeState;
  focusMode: boolean;
  onClose: () => void;
}) {
  const transcriptRef = useRef<HTMLDivElement | null>(null);
  const followTailRef = useRef(true);
  const userScrollIntentRef = useRef(false);
  const selectingTranscriptRef = useRef(false);
  const isAutoScrollingRef = useRef(false);
  const plugins = subagent.status?.plugins ?? [];
  const toolNames = subagent.status?.toolNames ?? [];
  const toolCount = subagent.status?.toolCount ?? toolNames.length;
  const tailKey = useMemo(
    () => transcriptTailKey(subagent.nodes, subagent.processing),
    [subagent.nodes, subagent.processing]
  );

  useEffect(() => {
    followTailRef.current = true;
    userScrollIntentRef.current = false;
    selectingTranscriptRef.current = false;
    isAutoScrollingRef.current = false;
  }, [subagent.id]);

  useBrowserLayoutEffect(() => {
    const element = transcriptRef.current;
    if (!element) return;
    if (!followTailRef.current && !isNearChatBottom(element)) return;
    followTailRef.current = true;
    isAutoScrollingRef.current = true;
    element.scrollTop = element.scrollHeight;
    isAutoScrollingRef.current = false;
  }, [tailKey]);

  function updateFollowTail() {
    const element = transcriptRef.current;
    if (!element) return;
    selectingTranscriptRef.current = hasTranscriptSelection(element);
    followTailRef.current = resolveFollowTailOnScroll(
      followTailRef.current,
      isNearChatBottom(element),
      userScrollIntentRef.current,
      selectingTranscriptRef.current,
      isAutoScrollingRef.current
    );
    userScrollIntentRef.current = false;
  }

  function markUserScrollIntent() {
    userScrollIntentRef.current = true;
    isAutoScrollingRef.current = false;
  }

  function handleWheel(event: WheelEvent<HTMLElement>) {
    markUserScrollIntent();
    if (event.deltaY < 0) {
      followTailRef.current = false;
    }
  }

  function handleTouchStart() {
    markUserScrollIntent();
  }

  function handleMouseDown(event: ReactMouseEvent<HTMLElement>) {
    if (event.button !== 0 || isInteractiveTranscriptTarget(event.target)) return;
    selectingTranscriptRef.current = true;
    followTailRef.current = false;
    markUserScrollIntent();
  }

  return (
    <div className="modal-backdrop" role="presentation">
      <section className="modal subagent-modal" role="dialog" aria-modal="true" aria-label={`SubAgent ${subagent.name}`}>
        <header>
          <div className="subagent-modal-title">
            <h2>{subagent.name}</h2>
            <span>{subagent.role} · {subAgentStateLabel(subagent)} · {subagent.id}</span>
          </div>
          <button
            type="button"
            className="icon-button"
            title="关闭"
            aria-label="关闭"
            data-dialog-close="true"
            onClick={onClose}
          >
            <X size={17} />
          </button>
        </header>
        <div className="subagent-modal-meta">
          <span><Activity size={14} /> {subagent.lastEventType ?? "waiting"}</span>
          {subagent.status?.modelId && <span>{subagent.status.modelId}</span>}
          {subagent.status?.workingDir && <span>{subagent.status.workingDir}</span>}
          <span>{plugins.length} plugins</span>
          <span>{toolCount} tools</span>
        </div>
        <div className="subagent-modal-config">
          <span className="subagent-config-label">Plugins</span>
          <span className="subagent-config-values">
            {plugins.length > 0
              ? plugins.map((plugin) => (
                <span className="subagent-config-chip" key={plugin.id} title={plugin.id}>
                  {plugin.id}
                </span>
              ))
              : <em>none</em>}
          </span>
          <span className="subagent-config-label">Tools</span>
          <span className="subagent-config-values">
            {toolNames.length > 0
              ? toolNames.map((toolName) => (
                <span className="subagent-config-chip" key={toolName} title={toolName}>
                  {toolName}
                </span>
              ))
              : <em>none</em>}
          </span>
        </div>
        <ChatTranscript
          ref={transcriptRef}
          nodes={subagent.nodes}
          processing={subagent.processing}
          focusMode={focusMode}
          allowFork={false}
          emptyLabel="等待 SubAgent 消息..."
          onScroll={updateFollowTail}
          onWheel={handleWheel}
          onTouchStart={handleTouchStart}
          onMouseDown={handleMouseDown}
        />
      </section>
    </div>
  );
}

function isSubAgentActive(item: SubAgentRuntimeState): boolean {
  return item.state === "RUNNING"
    || item.state === "INTERRUPTING"
    || item.status?.runnerState === "RUNNING"
    || item.status?.executorState === "RUNNING"
    || item.processing !== undefined
    || item.partial.text.length > 0
    || item.partial.reasoning.length > 0;
}

function subAgentStateLabel(item: SubAgentRuntimeState): string {
  if (item.status?.lastError) return "failed";
  const state = item.state || item.status?.state || "CREATED";
  return state.toLowerCase();
}

export function sortSubAgentsByCreatedAt(subagents: SubAgentRuntimeState[]): SubAgentRuntimeState[] {
  return [...subagents].sort(compareSubAgentsByCreatedAt);
}

function compareSubAgentsByCreatedAt(left: SubAgentRuntimeState, right: SubAgentRuntimeState): number {
  const leftCreatedAt = subAgentCreatedAt(left);
  const rightCreatedAt = subAgentCreatedAt(right);
  if (leftCreatedAt !== rightCreatedAt) return leftCreatedAt - rightCreatedAt;
  return left.id.localeCompare(right.id);
}

function subAgentCreatedAt(item: SubAgentRuntimeState): number {
  return item.createdAt ?? item.status?.createdAt ?? item.lastEventAt ?? 0;
}

function ArtifactSidebarContent({
  group,
  selected,
  messages,
  statuses,
  toolProgress,
  onClose,
  onSelectArtifact,
  onPluginAction
}: {
  group: ArtifactTypeGroup;
  selected: PluginArtifactState;
  messages: PluginMessageState[];
  statuses: Record<string, PluginStatusState>;
  toolProgress: Record<string, ToolProgressState>;
  onClose: () => void;
  onSelectArtifact: (artifactKey: string) => void;
  onPluginAction: (payload: PluginActionPayload) => Promise<CoreFrame | null>;
}) {
  const statusList = Object.values(statuses).sort((a, b) => b.updatedAt - a.updatedAt);
  const progressList = Object.values(toolProgress).sort((a, b) => b.updatedAt - a.updatedAt).slice(0, 4);
  const messageList = messages.slice(-5).reverse();

  function selectArtifact(artifactKey: string) {
    onSelectArtifact(artifactKey);
  }

  return (
    <section className="plugin-preview workspace-sidebar-content">
      <header className="preview-head">
        <span><FileText size={15} /> {group.label}</span>
        <button
          type="button"
          className="icon-button"
          title="隐藏 Artifacts"
          aria-label="隐藏 Artifacts"
          onClick={onClose}
        >
          <ChevronRight size={17} />
        </button>
      </header>
      {group.artifacts.length > 1 && (
        <div className="artifact-list">
          {group.artifacts.map((artifact) => (
            <button
              key={artifact.key}
              className={artifact.key === selected.key ? "artifact-item active" : "artifact-item"}
              title={artifact.title}
              onClick={() => selectArtifact(artifact.key)}
            >
              <span>{artifact.title}</span>
              <small>{artifact.pluginName}</small>
            </button>
          ))}
        </div>
      )}
      <ArtifactPreview artifact={selected} />

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
              <PluginMessageActions item={item} onPluginAction={onPluginAction} />
            </div>
          ))}
        </section>
      )}
    </section>
  );
}

function PluginMessageActions({
  item,
  onPluginAction
}: {
  item: PluginMessageState;
  onPluginAction: (payload: PluginActionPayload) => Promise<CoreFrame | null>;
}) {
  const review = humanReviewActionFromMessage(item);
  const [busy, setBusy] = useState(false);
  const [handled, setHandled] = useState(false);
  if (!review) return null;
  const activeReview = review;

  async function submitReview(payload: PluginActionPayload) {
    if (busy || handled) return;
    setBusy(true);
    try {
      const frame = await onPluginAction(payload);
      if (frame?.type === "ack") {
        setHandled(true);
      }
    } finally {
      setBusy(false);
    }
  }

  async function approve() {
    await submitReview({
      plugin_id: activeReview.pluginId,
      action: activeReview.approveAction,
      arguments: {
        review_id: activeReview.reviewId,
        feedback: "Approved in GUI."
      }
    });
  }

  async function reject() {
    if (busy || handled) return;
    const feedback = window.prompt("拒绝原因");
    if (feedback === null) return;
    const trimmed = feedback.trim();
    if (!trimmed) return;
    await submitReview({
      plugin_id: activeReview.pluginId,
      action: activeReview.rejectAction,
      arguments: {
        review_id: activeReview.reviewId,
        feedback: trimmed
      }
    });
  }

  if (handled) {
    return (
      <div className="plugin-message-actions">
        <span className="review-action-state">已提交</span>
      </div>
    );
  }

  return (
    <div className="plugin-message-actions">
      <button type="button" className="mini-button" onClick={approve} disabled={busy}>
        {busy ? <LoaderCircle size={13} /> : <Check size={13} />} 批准
      </button>
      <button type="button" className="mini-button danger" onClick={reject} disabled={busy}>
        <X size={13} /> 拒绝
      </button>
    </div>
  );
}

function humanReviewActionFromMessage(item: PluginMessageState): {
  pluginId: string;
  reviewId: string;
  approveAction: string;
  rejectAction: string;
} | null {
  const data = isRecord(item.data) ? item.data : null;
  if (!data || data.kind !== "human_review_request") return null;
  const reviewId = optionalPayloadString(data.review_id);
  const approveAction = optionalPayloadString(data.approve_action);
  const rejectAction = optionalPayloadString(data.reject_action);
  const pluginId = optionalPayloadString(data.plugin_id) ?? item.pluginId;
  if (!reviewId || !approveAction || !rejectAction || !pluginId) return null;
  return { pluginId, reviewId, approveAction, rejectAction };
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
          ? <MarkdownView html={renderMarkdown(content)} className="artifact-content" />
          : <CopyablePreBlock className="artifact-code" value={content} />
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

function ContextPopover({
  usage,
  autoCompact,
  busy,
  settingsBusy,
  canManualCompact,
  onConfirm,
  onThresholdChange
}: {
  usage?: ContextUsageState;
  autoCompact?: ContextAutoCompactState;
  busy: boolean;
  settingsBusy: boolean;
  canManualCompact: boolean;
  onConfirm: () => void;
  onThresholdChange: (percent: number) => Promise<void>;
}) {
  const used = usage ? compactNumber(usage.usedTokens) : "-";
  const max = usage?.maxContextTokens ? compactNumber(usage.maxContextTokens) : "-";
  const percent = usage?.ratio === undefined ? "n/a" : `${Math.round(usage.ratio * 100)}%`;
  const estimated = usage?.source === "estimate";
  const currentThresholdPercent = autoCompactThresholdPercent(autoCompact, usage);
  const maxThresholdPercent = autoCompactMaxThresholdPercent(autoCompact);
  const [thresholdDraft, setThresholdDraft] = useState(currentThresholdPercent);
  const thresholdTokens = autoCompactThresholdTokens(autoCompact, usage, thresholdDraft);
  const thresholdChanged = thresholdDraft !== currentThresholdPercent;
  const thresholdDisabled = settingsBusy || !(autoCompact?.maxContextTokens ?? usage?.maxContextTokens);

  useEffect(() => {
    setThresholdDraft(currentThresholdPercent);
  }, [currentThresholdPercent]);

  function updateThresholdDraft(value: number) {
    if (!Number.isFinite(value)) return;
    setThresholdDraft(Math.min(maxThresholdPercent, Math.max(10, Math.round(value))));
  }

  return (
    <div className="context-popover">
      <StatusPopoverHeader title="Context" value={percent} />
      <div className="context-popover-body">
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
        <section className="context-auto-compact">
          <div className="context-auto-compact-head">
            <strong>自动压缩</strong>
            <span>{autoCompact ? (autoCompact.enabled ? "On" : "Off") : "n/a"}</span>
          </div>
          <label className="context-threshold-control">
            <span>触发阈值</span>
            <input
              type="range"
              min={10}
              max={maxThresholdPercent}
              step={1}
              value={thresholdDraft}
              disabled={thresholdDisabled}
              onChange={(event) => updateThresholdDraft(Number(event.currentTarget.value))}
            />
            <span className="context-threshold-number">
              <input
                type="number"
                min={10}
                max={maxThresholdPercent}
                step={1}
                value={thresholdDraft}
                disabled={thresholdDisabled}
                onChange={(event) => updateThresholdDraft(Number(event.currentTarget.value))}
              />
              <span>%</span>
            </span>
          </label>
          <div className="context-threshold-actions">
            <span>
              {thresholdTokens !== undefined
                ? `${compactNumber(thresholdTokens)} tokens`
                : "tokens n/a"}
            </span>
            <button
              className="tool-button"
              disabled={thresholdDisabled || !thresholdChanged}
              onClick={() => void onThresholdChange(thresholdDraft)}
            >
              {settingsBusy ? (
                <>
                  <LoaderCircle className="inline-spinner" size={15} /> 保存中
                </>
              ) : "保存阈值"}
            </button>
          </div>
        </section>
        <div className="context-popover-actions">
          <button
            className="primary-button"
            disabled={busy || !canManualCompact}
            title={canManualCompact ? "手动压缩上下文" : "Agent idle 后可手动压缩"}
            onClick={onConfirm}
          >
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
      </div>
    </div>
  );
}

function autoCompactThresholdPercent(
  autoCompact?: ContextAutoCompactState,
  usage?: ContextUsageState,
): number {
  const maxContextTokens = autoCompact?.maxContextTokens ?? usage?.maxContextTokens;
  const ratio = autoCompact?.tokenLimitRatio
    ?? (
      autoCompact?.tokenLimit !== undefined && maxContextTokens
        ? autoCompact.tokenLimit / maxContextTokens
        : undefined
    )
    ?? autoCompact?.triggerRatio
    ?? 0.8;
  return Math.min(autoCompactMaxThresholdPercent(autoCompact), Math.max(10, Math.round(ratio * 100)));
}

function autoCompactMaxThresholdPercent(autoCompact?: ContextAutoCompactState): number {
  return Math.min(100, Math.max(10, Math.round((autoCompact?.maxTriggerRatio ?? 0.95) * 100)));
}

function autoCompactThresholdTokens(
  autoCompact: ContextAutoCompactState | undefined,
  usage: ContextUsageState | undefined,
  percent: number,
): number | undefined {
  const maxContextTokens = autoCompact?.maxContextTokens ?? usage?.maxContextTokens;
  if (!maxContextTokens) return autoCompact?.tokenLimit;
  return Math.max(1, Math.round(maxContextTokens * (percent / 100)));
}

function ModelDialog({
  models,
  providerConfigs,
  modelProviderConfigs,
  current,
  providerConfigLocked,
  onClose,
  onSelect,
  onRefresh,
  onApplyProviderConfig
}: {
  models: string[];
  providerConfigs: Record<string, ModelProviderConfigPreview>;
  modelProviderConfigs: Record<string, Record<string, unknown>>;
  current: string;
  providerConfigLocked: boolean;
  onClose: () => void;
  onSelect: (model: string) => void;
  onRefresh: (provider: string) => Promise<void>;
  onApplyProviderConfig: (provider: string, providerConfig: Record<string, unknown>) => Promise<void>;
}) {
  const [filter, setFilter] = useState("");
  const [refreshingProvider, setRefreshingProvider] = useState<string | null>(null);
  const [refreshError, setRefreshError] = useState<string | null>(null);
  const [editingProvider, setEditingProvider] = useState<string | null>(null);
  const [savingProvider, setSavingProvider] = useState<string | null>(null);
  const [providerConfigError, setProviderConfigError] = useState<string | null>(null);
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
      await waitForNextPaint();
      await onRefresh(provider);
    } catch (error) {
      setRefreshError(formatDialogError(error));
    } finally {
      setRefreshingProvider(null);
    }
  }

  async function applyProviderConfig(provider: string, providerConfig: Record<string, unknown>) {
    if (savingProvider || providerConfigLocked) return;
    setSavingProvider(provider);
    setProviderConfigError(null);
    try {
      await waitForNextPaint();
      await onApplyProviderConfig(provider, providerConfig);
      setEditingProvider(null);
    } catch (error) {
      setProviderConfigError(formatDialogError(error));
    } finally {
      setSavingProvider(null);
    }
  }

  return (
    <Modal title="切换模型" className="model-modal" onClose={onClose}>
      <div className="modal-toolbar">
        <input className="search" value={filter} onChange={(event) => setFilter(event.target.value)} placeholder="搜索模型" />
      </div>
      {refreshError && <div className="model-refresh-error" role="alert">{refreshError}</div>}
      {providerConfigError && <div className="model-refresh-error" role="alert">{providerConfigError}</div>}
      <div className="model-grid">
        {grouped.map(([provider, entries]) => (
          <section className="model-provider" key={provider}>
            <div className="model-provider-main">
              <header className="model-provider-header">
                <h3>{provider}</h3>
                <span className="model-provider-actions">
                  <button
                    className="icon-button model-refresh"
                    title={refreshingProvider === provider ? `正在刷新 ${provider} 模型列表` : `刷新 ${provider} 模型列表`}
                    aria-label={refreshingProvider === provider ? `正在刷新 ${provider} 模型列表` : `刷新 ${provider} 模型列表`}
                    disabled={Boolean(refreshingProvider)}
                    onClick={() => void refresh(provider)}
                  >
                    <RotateCcw size={15} className={refreshingProvider === provider ? "spin" : ""} />
                  </button>
                </span>
              </header>
              {entries.map((model) => (
                <button className={model === current ? "model active" : "model"} key={model} onClick={() => onSelect(model)}>
                  {model.split("/").slice(1).join("/")}
                </button>
              ))}
            </div>
            {editingProvider === provider ? (
              <ModelProviderConfigEditorPanel
                provider={provider}
                config={providerConfigs[provider]}
                overrides={modelProviderConfigs[provider] ?? {}}
                busy={savingProvider === provider}
                onCancel={() => setEditingProvider(null)}
                onSave={(nextConfig) => void applyProviderConfig(provider, nextConfig)}
              />
            ) : (
              <ModelProviderConfigPreviewPanel
                provider={provider}
                config={providerConfigs[provider]}
                overrides={modelProviderConfigs[provider] ?? {}}
                editDisabled={providerConfigLocked || Boolean(savingProvider)}
                editTitle={providerConfigLocked ? "运行中不能编辑 provider 配置" : `编辑 ${provider} provider 配置`}
                onEdit={() => {
                  setProviderConfigError(null);
                  setEditingProvider(provider);
                }}
              />
            )}
          </section>
        ))}
      </div>
    </Modal>
  );
}

function ModelProviderConfigPreviewPanel({
  provider,
  config,
  overrides = {},
  editDisabled,
  editTitle,
  onEdit
}: {
  provider: string;
  config?: ModelProviderConfigPreview;
  overrides?: Record<string, unknown>;
  editDisabled: boolean;
  editTitle: string;
  onEdit: () => void;
}) {
  const lines = modelProviderConfigPreviewLines(config, overrides);
  return (
    <aside className="model-provider-config-preview" aria-label={`${provider} 配置预览`}>
      <div className="model-provider-config-preview-head">
        <span className="model-provider-config-title">配置</span>
        <button
          className="icon-button model-provider-config-edit"
          type="button"
          title={editTitle}
          aria-label={`编辑 ${provider} provider 配置`}
          disabled={editDisabled}
          onClick={onEdit}
        >
          <Pencil size={15} />
        </button>
      </div>
      <div className="model-provider-config-lines">
        {lines.map((line, index) => (
          <div className="model-provider-config-line" key={`${line}-${index}`} title={line}>
            {line}
          </div>
        ))}
      </div>
    </aside>
  );
}

type ProviderConfigValueType = "string" | "number" | "boolean" | "json";

interface ProviderConfigDraftField {
  id: string;
  key: string;
  value: string;
  valueType: ProviderConfigValueType;
  enabled: boolean;
  fixedKey: boolean;
  previewValue?: unknown;
}

function ModelProviderConfigEditorPanel({
  provider,
  config,
  overrides,
  busy,
  onCancel,
  onSave
}: {
  provider: string;
  config?: ModelProviderConfigPreview;
  overrides: Record<string, unknown>;
  busy: boolean;
  onCancel: () => void;
  onSave: (providerConfig: Record<string, unknown>) => void;
}) {
  const [fields, setFields] = useState<ProviderConfigDraftField[]>(() => (
    createProviderConfigDraftFields(config, overrides)
  ));
  const [error, setError] = useState<string | null>(null);

  function updateField(id: string, patch: Partial<ProviderConfigDraftField>) {
    setFields((current) => current.map((field) => field.id === id ? { ...field, ...patch } : field));
    setError(null);
  }

  function removeField(id: string) {
    setFields((current) => current.filter((field) => field.id !== id));
    setError(null);
  }

  function addField() {
    setFields((current) => [
      ...current,
      {
        id: `manual-${Date.now()}-${current.length}`,
        key: "",
        value: "",
        valueType: "string",
        enabled: true,
        fixedKey: false,
      },
    ]);
    setError(null);
  }

  function save() {
    const result = providerConfigDraftToObject(fields);
    if (!result.ok) {
      setError(result.error);
      return;
    }
    onSave(result.value);
  }

  return (
    <aside className="model-provider-config-editor" aria-label={`${provider} provider 配置编辑`}>
      <div className="provider-config-editor-head">
        <strong>临时配置</strong>
        <button className="tool-button compact" type="button" onClick={addField}>
          <Plus size={14} /> 字段
        </button>
      </div>
      <div className="provider-config-fields">
        {fields.map((field) => (
          <div className="provider-config-field" key={field.id}>
            <label className="provider-config-override">
              <input
                type="checkbox"
                checked={field.enabled}
                disabled={busy}
                onChange={(event) => updateField(field.id, { enabled: event.target.checked })}
              />
            </label>
            <input
              className="provider-config-key"
              value={field.key}
              readOnly={field.fixedKey}
              disabled={busy}
              placeholder="field"
              onChange={(event) => updateField(field.id, { key: event.target.value, enabled: true })}
            />
            <select
              className="provider-config-type"
              value={field.valueType}
              disabled={busy}
              onChange={(event) => updateField(field.id, { valueType: event.target.value as ProviderConfigValueType, enabled: true })}
            >
              <option value="string">text</option>
              <option value="number">number</option>
              <option value="boolean">bool</option>
              <option value="json">json</option>
            </select>
            {field.valueType === "boolean" ? (
              <select
                className="provider-config-value"
                value={field.value || "true"}
                disabled={busy}
                onChange={(event) => updateField(field.id, { value: event.target.value, enabled: true })}
              >
                <option value="true">true</option>
                <option value="false">false</option>
              </select>
            ) : (
              <input
                className="provider-config-value"
                value={field.value}
                disabled={busy}
                placeholder={providerConfigPreviewPlaceholder(field.previewValue)}
                onChange={(event) => updateField(field.id, { value: event.target.value, enabled: true })}
              />
            )}
            <button
              className="icon-button provider-config-remove"
              type="button"
              title="移除字段"
              disabled={busy}
              onClick={() => removeField(field.id)}
            >
              <Trash2 size={14} />
            </button>
          </div>
        ))}
      </div>
      {error && <div className="provider-config-error" role="alert">{error}</div>}
      <div className="provider-config-actions">
        <button className="tool-button" type="button" disabled={busy} onClick={onCancel}>取消</button>
        <button className="primary-button" type="button" disabled={busy} onClick={save}>
          {busy ? <LoaderCircle className="inline-spinner" size={15} /> : <Check size={15} />}
          保存
        </button>
      </div>
    </aside>
  );
}

export function modelProviderConfigPreviewLines(
  config?: ModelProviderConfigPreview,
  overrides: Record<string, unknown> = {},
): string[] {
  if (!config) return ["config: not loaded"];
  const lines = [
    `adapter: ${formatProviderConfigValue(config.adapter)}`,
    `models: ${formatProviderConfigValue(config.model_count)}`,
  ];
  for (const [key, value] of Object.entries(config.properties ?? {})) {
    lines.push(`${key}: ${formatProviderConfigValue(value)}`);
  }
  for (const [key, value] of Object.entries(overrides)) {
    lines.push(`override ${key}: ${formatProviderConfigPreviewValue(key, value)}`);
  }
  return lines;
}

function createProviderConfigDraftFields(
  config: ModelProviderConfigPreview | undefined,
  overrides: Record<string, unknown>,
): ProviderConfigDraftField[] {
  const keys = new Set([
    ...Object.keys(config?.properties ?? {}),
    ...Object.keys(overrides),
  ]);
  return [...keys].map((key) => {
    const hasOverride = Object.prototype.hasOwnProperty.call(overrides, key);
    const value = hasOverride ? overrides[key] : undefined;
    const previewValue = config?.properties?.[key];
    return {
      id: `field-${key}`,
      key,
      value: providerConfigDraftValue(hasOverride ? value : previewValue),
      valueType: inferProviderConfigValueType(hasOverride ? value : previewValue),
      enabled: hasOverride,
      fixedKey: true,
      previewValue,
    };
  });
}

type ProviderConfigDraftResult =
  | { ok: true; value: Record<string, unknown> }
  | { ok: false; error: string };

function providerConfigDraftToObject(fields: ProviderConfigDraftField[]): ProviderConfigDraftResult {
  const result: Record<string, unknown> = {};
  for (const field of fields) {
    if (!field.enabled) continue;
    const key = field.key.trim();
    if (!key) {
      return { ok: false, error: "字段名不能为空" };
    }
    if (Object.prototype.hasOwnProperty.call(result, key)) {
      return { ok: false, error: `字段重复：${key}` };
    }
    if (shouldSkipMaskedSensitiveDraft(field)) {
      continue;
    }
    const converted = providerConfigDraftFieldValue(field);
    if (!converted.ok) return converted;
    result[key] = converted.value;
  }
  return { ok: true, value: result };
}

function shouldSkipMaskedSensitiveDraft(field: ProviderConfigDraftField): boolean {
  return isSensitiveProviderConfigKey(field.key)
    && isMaskedProviderConfigValue(field.previewValue)
    && field.value === providerConfigDraftValue(field.previewValue);
}

type ProviderConfigValueResult =
  | { ok: true; value: unknown }
  | { ok: false; error: string };

function providerConfigDraftFieldValue(field: ProviderConfigDraftField): ProviderConfigValueResult {
  const raw = field.value;
  if (field.valueType === "number") {
    const value = Number(raw);
    return Number.isFinite(value)
      ? { ok: true, value }
      : { ok: false, error: `${field.key || "字段"} 需要是数字` };
  }
  if (field.valueType === "boolean") {
    return { ok: true, value: raw !== "false" };
  }
  if (field.valueType === "json") {
    try {
      return { ok: true, value: JSON.parse(raw) };
    } catch {
      return { ok: false, error: `${field.key || "字段"} 不是合法 JSON` };
    }
  }
  return { ok: true, value: raw };
}

function providerConfigDraftValue(value: unknown): string {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (value === undefined) return "";
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function inferProviderConfigValueType(value: unknown): ProviderConfigValueType {
  if (typeof value === "number") return "number";
  if (typeof value === "boolean") return "boolean";
  if (value != null && typeof value === "object") return "json";
  return "string";
}

function providerConfigPreviewPlaceholder(value: unknown): string {
  if (value === undefined) return "value";
  return `当前：${formatProviderConfigValue(value)}`;
}

function formatProviderConfigPreviewValue(key: string, value: unknown): string {
  if (!isSensitiveProviderConfigKey(key)) {
    return formatProviderConfigValue(value);
  }
  const text = formatProviderConfigValue(value);
  if (!text || text === "null") return text;
  if (text.length <= 8) return "****";
  return `${text.slice(0, 3)}...${text.slice(-4)}`;
}

function isSensitiveProviderConfigKey(key: string): boolean {
  return /(?:api[_-]?key|token|secret|password|credential)/i.test(key);
}

function isMaskedProviderConfigValue(value: unknown): boolean {
  if (typeof value !== "string") return false;
  return value === "****" || /^[^\s.]{1,8}\.\.\.[^\s.]{1,8}$/.test(value);
}

function formatProviderConfigValue(value: unknown): string {
  if (value == null) return "null";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function formatDialogError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

type PluginPreviewState =
  | { status: "loading" }
  | { status: "loaded"; data: PluginToolPreviewPayload }
  | { status: "error"; error: string };

function PluginDialog({
  catalog,
  selectedPlugins,
  pluginConfigs,
  onClose,
  onPreviewPlugin,
  onApply
}: {
  catalog: PluginCatalogItem[];
  selectedPlugins: string[];
  pluginConfigs: Record<string, Record<string, unknown>>;
  onClose: () => void;
  onPreviewPlugin: (pluginKey: string, pluginConfig: Record<string, unknown>) => Promise<PluginToolPreviewPayload>;
  onApply: (selected: string[], configs: Record<string, Record<string, unknown>>) => void;
}) {
  const initial = mergePluginDefaults(catalog, selectedPlugins, pluginConfigs);
  const [selected, setSelected] = useState(new Set(initial.selectedPlugins));
  const [configs, setConfigs] = useState(initial.pluginConfigs);
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [previews, setPreviews] = useState<Record<string, PluginPreviewState>>({});
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
    setSelected(new Set(resolvePluginSelectionChange(catalog, selected, next)));
    if (Object.keys(fieldErrors).length > 0) setFieldErrors({});
  }

  async function previewPlugin(item: PluginCatalogItem) {
    const source = configs[item.key] ?? item.defaults ?? {};
    const pluginConfig = isRecord(source) ? source : {};
    setPreviews((current) => ({
      ...current,
      [item.key]: { status: "loading" },
    }));
    try {
      await waitForNextPaint();
      const data = await onPreviewPlugin(item.key, pluginConfig);
      setPreviews((current) => ({
        ...current,
        [item.key]: { status: "loaded", data },
      }));
    } catch (error) {
      setPreviews((current) => ({
        ...current,
        [item.key]: { status: "error", error: formatDialogError(error) },
      }));
    }
  }

  function updatePluginConfig(pluginKey: string, field: string, value: unknown) {
    setConfigs({
      ...configs,
      [pluginKey]: {
        ...(configs[pluginKey] ?? catalog.find((item) => item.key === pluginKey)?.defaults ?? {}),
        [field]: value
      }
    });
    setPreviews((current) => {
      if (!current[pluginKey]) return current;
      const next = { ...current };
      delete next[pluginKey];
      return next;
    });
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
                onClick={() => updateSelection(new Set())}
              >
                <Trash2 size={15} /> 清空
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
            <div className="plugin-main">
              <label
                className="plugin-enable"
                title={`勾选并应用后加载 ${item.display_name} 插件；取消勾选并应用后不加载该插件。`}
              >
                <input
                  type="checkbox"
                  title={`勾选并应用后加载 ${item.display_name} 插件；取消勾选并应用后不加载该插件。`}
                  checked={selected.has(item.key)}
                  onChange={(event) => {
                    const next = new Set(selected);
                    if (event.target.checked) next.add(item.key);
                    else next.delete(item.key);
                    updateSelection(next);
                  }}
                />
                <strong>{item.display_name}</strong>
              </label>
              {Object.entries(item.schema.properties ?? {}).map(([field, schema]) => (
                <SchemaField
                  key={field}
                  field={field}
                  schema={schema}
                  disabled={!selected.has(item.key)}
                  error={getFieldError(item.key, field)}
                  value={(configs[item.key] ?? item.defaults)[field] ?? schema.default ?? ""}
                  onChange={(value) => updatePluginConfig(item.key, field, value)}
                />
              ))}
            </div>
            <PluginInfoPreviewPanel
              item={item}
              preview={previews[item.key]}
              onPreview={() => void previewPlugin(item)}
            />
          </section>
        ))}
      </div>
    </Modal>
  );
}

function PluginInfoPreviewPanel({
  item,
  preview,
  onPreview
}: {
  item: PluginCatalogItem;
  preview?: PluginPreviewState;
  onPreview: () => void;
}) {
  const loading = preview?.status === "loading";
  return (
    <aside className="plugin-info-preview" aria-label={`${item.display_name} 插件信息`}>
      <div className="plugin-info-head">
        <div className="plugin-info-title">
          <div className="plugin-identity">{item.name}</div>
          {item.description && <div className="plugin-description">{item.description}</div>}
        </div>
        <button
          className="tool-button plugin-preview-button"
          disabled={loading}
          title={loading ? "正在读取工具列表" : "预览工具列表"}
          onClick={onPreview}
        >
          {loading ? <LoaderCircle className="inline-spinner" size={15} /> : <Wrench size={15} />}
          {loading ? "读取中" : preview?.status === "loaded" ? "刷新" : "预览工具"}
        </button>
      </div>
      {item.dependencies.length > 0 && (
        <div className="plugin-info-section">
          <div className="plugin-info-section-title">依赖</div>
          <div className="plugin-chip-row">
            {item.dependencies.map((dependency) => (
              <span className="plugin-info-chip" key={dependency}>{dependency}</span>
            ))}
          </div>
        </div>
      )}
      <PluginPermissionPreview item={item} />
      <PluginToolPreview preview={preview} />
    </aside>
  );
}

function PluginPermissionPreview({ item }: { item: PluginCatalogItem }) {
  if (!item.permissions || item.permissions.length === 0) {
    return (
      <div className="plugin-info-section">
        <div className="plugin-info-section-title">权限声明</div>
        <div className="plugin-preview-empty">无</div>
      </div>
    );
  }
  return (
    <div className="plugin-info-section">
      <div className="plugin-info-section-title">权限声明</div>
      <div className="plugin-permissions">
        {item.permissions.map((perm) => (
          <div key={perm.id} className={`permission-tag permission-risk-${perm.risk_level}`}>
            <span className="permission-id">{perm.id}</span>
            <span className="permission-policy">{perm.default_policy}</span>
            {perm.tool_names.length > 0 && (
              <span className="permission-tools">{perm.tool_names.join(", ")}</span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function PluginToolPreview({ preview }: { preview?: PluginPreviewState }) {
  if (!preview) {
    return (
      <div className="plugin-info-section">
        <div className="plugin-info-section-title">工具</div>
        <div className="plugin-preview-empty">未预览</div>
      </div>
    );
  }
  if (preview.status === "loading") {
    return (
      <div className="plugin-info-section">
        <div className="plugin-info-section-title">工具</div>
        <div className="plugin-preview-empty">读取中</div>
      </div>
    );
  }
  if (preview.status === "error") {
    return (
      <div className="plugin-info-section">
        <div className="plugin-info-section-title">工具</div>
        <div className="plugin-preview-error" role="alert">{preview.error}</div>
      </div>
    );
  }
  const tools = preview.data.tools;
  if (tools.length === 0) {
    return (
      <div className="plugin-info-section">
        <div className="plugin-info-section-title">工具</div>
        <div className="plugin-preview-empty">无工具</div>
      </div>
    );
  }
  return (
    <div className="plugin-info-section">
      <div className="plugin-info-section-title">工具 · {tools.length}</div>
      <div className="plugin-tool-list">
        {tools.map((tool) => (
          <PluginToolPreviewRow tool={tool} key={tool.name} />
        ))}
      </div>
    </div>
  );
}

function PluginToolPreviewRow({ tool }: { tool: PluginToolPreviewItem }) {
  const parameterNames = pluginToolParameterNames(tool);
  return (
    <div className="plugin-tool-row">
      <div className="plugin-tool-row-head">
        <code title={tool.name}>{tool.short_name || tool.name}</code>
        {tool.audit && <span className="plugin-tool-badge">audit</span>}
      </div>
      {tool.description && <div className="plugin-tool-description">{tool.description}</div>}
      {parameterNames.length > 0 && (
        <div className="plugin-tool-parameters">
          {parameterNames.map((name) => (
            <span className="plugin-info-chip" key={name}>{name}</span>
          ))}
        </div>
      )}
    </div>
  );
}

function pluginToolParameterNames(tool: PluginToolPreviewItem): string[] {
  const properties = tool.schema?.properties;
  return properties ? Object.keys(properties) : [];
}

function SchemaField({ field, schema, disabled, value, error, onChange }: { field: string; schema: JsonSchemaObject; disabled: boolean; value: unknown; error?: string; onChange: (value: unknown) => void }) {
  const label = schema.title ?? field;
  const description = typeof schema.description === "string" ? schema.description : undefined;
  const labelTitle = description ? `${label}\n${description}` : label;
  if (schema.type === "boolean") {
    return (
      <label className="schema-field inline" title={labelTitle}>
        <span title={labelTitle}>{label}</span>
        <input
          type="checkbox"
          title={description}
          disabled={disabled}
          checked={Boolean(value)}
          onChange={(event) => onChange(event.target.checked)}
        />
      </label>
    );
  }
  if (schema.enum && Array.isArray(schema.enum) && schema.enum.length > 0) {
    return (
      <label className="schema-field" title={labelTitle}>
        <span title={labelTitle}>{label}</span>
        <select
          title={description}
          disabled={disabled}
          value={String(value ?? "")}
          onChange={(event) => onChange(coerceSchemaValue(schema, event.target.value))}
        >
          <option value="">--</option>
          {schema.enum.map((opt) => (
            <option key={String(opt)} value={String(opt)}>{String(opt)}</option>
          ))}
        </select>
        {error && <small className="schema-error">{error}</small>}
      </label>
    );
  }
  return (
    <label className="schema-field" title={labelTitle}>
      <span title={labelTitle}>{label}</span>
      <input
        title={description}
        disabled={disabled}
        type={schema.type === "integer" || schema.type === "number" ? "number" : "text"}
        value={String(value ?? "")}
        onChange={(event) => onChange(coerceSchemaValue(schema, event.target.value))}
      />
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

function hasPayloadKey(payload: Record<string, unknown> | null, key: string): boolean {
  return Boolean(payload && Object.prototype.hasOwnProperty.call(payload, key));
}

function normalizeHistorySearchResults(value: unknown): HistorySearchResult[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter((item): item is Record<string, unknown> => isRecord(item))
    .map((item) => {
      const sessionId = optionalPayloadString(item.session_id) ?? "";
      const messageIndex = optionalPayloadNumber(item.message_index);
      return {
        sessionId,
        sessionName: optionalPayloadString(item.session_name) ?? sessionId,
        sessionCreatedAt: optionalPayloadString(item.session_created_at) ?? "",
        sessionUpdatedAt: optionalPayloadString(item.session_updated_at) ?? "",
        messageIndex: messageIndex ?? -1,
        contextMessageId: optionalPayloadString(item.context_message_id) ?? undefined,
        contextMessageIndex: optionalPayloadNumber(item.context_message_index),
        runId: optionalPayloadString(item.run_id) ?? undefined,
        role: optionalPayloadString(item.role) ?? "message",
        timestamp: typeof item.timestamp === "number" || typeof item.timestamp === "string" ? item.timestamp : undefined,
        text: typeof item.text === "string" ? item.text : "",
        snippet: typeof item.snippet === "string" ? item.snippet : "",
        lastCwd: optionalPayloadString(item.last_cwd)
      };
    })
    .filter((item) => item.sessionId && item.messageIndex >= 0);
}

function historySearchResultKey(result: HistorySearchResult): string {
  return [
    result.sessionId,
    result.messageIndex,
    result.contextMessageId ?? "",
    result.contextMessageIndex ?? "",
  ].join(":");
}

function historyResultToLocateTarget(result: HistorySearchResult): HistoryLocateTarget {
  return {
    sessionId: result.sessionId,
    messageIndex: result.messageIndex,
    contextMessageId: result.contextMessageId,
    contextMessageIndex: result.contextMessageIndex,
  };
}

function scrollToHistoryTarget(container: HTMLElement | null, target: HistoryLocateTarget): boolean {
  const element = findHistoryTargetElement(container, target);
  if (!element) {
    return false;
  }
  element.scrollIntoView({ block: "center" });
  return true;
}

export function reducePendingHistoryLocateScroll(
  current: HistoryLocateTarget | null,
  scrolledTarget: HistoryLocateTarget,
  found: boolean,
): HistoryLocateTarget | null {
  if (!found) return current;
  return historyLocateTargetsEqual(current, scrolledTarget) ? null : current;
}

function historyLocateTargetsEqual(
  left: HistoryLocateTarget | null,
  right: HistoryLocateTarget | null,
): boolean {
  if (!left || !right) return left === right;
  return left.sessionId === right.sessionId
    && left.messageIndex === right.messageIndex
    && left.contextMessageId === right.contextMessageId
    && left.contextMessageIndex === right.contextMessageIndex;
}

function findHistoryTargetElement(container: HTMLElement | null, target: HistoryLocateTarget): HTMLElement | null {
  if (!container) return null;
  for (const selector of historyTargetSelectors(target)) {
    const element = container.querySelector<HTMLElement>(selector);
    if (element) {
      return element;
    }
  }
  return null;
}

function historyTargetSelectors(target: HistoryLocateTarget): string[] {
  return [
    typeof target.messageIndex === "number" ? `[data-history-index="${target.messageIndex}"]` : "",
    target.contextMessageId ? `[data-context-message-id="${cssAttrEscape(target.contextMessageId)}"]` : "",
    typeof target.contextMessageIndex === "number" ? `[data-context-message-index="${target.contextMessageIndex}"]` : "",
  ].filter(Boolean);
}

function historyTargetMatchesElement(element: HTMLElement, target: HistoryLocateTarget): boolean {
  if (typeof target.messageIndex === "number" && element.dataset.historyIndex === String(target.messageIndex)) {
    return true;
  }
  if (target.contextMessageId && element.dataset.contextMessageId === target.contextMessageId) {
    return true;
  }
  return typeof target.contextMessageIndex === "number"
    && element.dataset.contextMessageIndex === String(target.contextMessageIndex);
}

function isHighlightedChatNode(
  node: ChatNode,
  historyIndex?: number,
  contextMessageId?: string,
  contextMessageIndex?: number,
): boolean {
  if (typeof historyIndex === "number" && node.historyIndex === historyIndex) {
    return true;
  }
  if (contextMessageId && node.contextMessageId === contextMessageId) {
    return true;
  }
  return typeof contextMessageIndex === "number" && node.contextMessageIndex === contextMessageIndex;
}

function cssAttrEscape(value: string): string {
  return value.replace(/\\/g, "\\\\").replace(/"/g, "\\\"");
}

function resolveSideSelectionAnchor(container: HTMLElement | null, sessionId: string | null): SideSelectionAnchor | null {
  if (!container || !sessionId) return null;
  const selection = window.getSelection();
  if (!selection || selection.isCollapsed || selection.rangeCount === 0) return null;
  const range = selection.getRangeAt(0);
  const anchorNode = selection.anchorNode;
  const focusNode = selection.focusNode;
  if (
    (anchorNode && !container.contains(anchorNode))
    || (focusNode && !container.contains(focusNode))
  ) {
    return null;
  }
  const frame = closestChatNodeFrame(container, range.startContainer);
  if (!frame) return null;
  const quote = sideThreadQuoteFromRange(frame, range);
  if (!quote) return null;
  const target = sideTargetFromFrame(frame, sessionId);
  if (!target.contextMessageId && typeof target.contextMessageIndex !== "number" && typeof target.messageIndex !== "number") {
    return null;
  }
  const rect = firstUsableRangeRect(range);
  if (!rect) return null;
  return { text: quote.quotedText, quotedRange: quote.quotedRange, rect, target };
}

function closestChatNodeFrame(container: HTMLElement, node: Node): HTMLElement | null {
  const element = node instanceof HTMLElement ? node : node.parentElement;
  const frame = element?.closest<HTMLElement>(".chat-node-frame");
  return frame && container.contains(frame) ? frame : null;
}

function sideTargetFromFrame(frame: HTMLElement, sessionId: string): HistoryLocateTarget {
  return {
    sessionId,
    messageIndex: optionalDatasetNumber(frame.dataset.historyIndex),
    contextMessageId: frame.dataset.contextMessageId || undefined,
    contextMessageIndex: optionalDatasetNumber(frame.dataset.contextMessageIndex),
  };
}

function firstUsableRangeRect(range: Range): DOMRect | null {
  const rect = range.getBoundingClientRect();
  if (rect.width > 0 || rect.height > 0) return rect;
  for (const candidate of Array.from(range.getClientRects())) {
    if (candidate.width > 0 || candidate.height > 0) return candidate;
  }
  return null;
}

function sideThreadQuoteFromRange(frame: HTMLElement, range: Range): { quotedText: string; quotedRange: SideThreadQuotedRange } | null {
  const root = closestSideThreadQuoteRoot(frame, range.commonAncestorContainer);
  if (!root.contains(range.startContainer) || !root.contains(range.endContainer)) {
    return null;
  }
  const fullText = root.textContent ?? "";
  const beforeStart = range.cloneRange();
  beforeStart.selectNodeContents(root);
  beforeStart.setEnd(range.startContainer, range.startOffset);
  const beforeEnd = range.cloneRange();
  beforeEnd.selectNodeContents(root);
  beforeEnd.setEnd(range.endContainer, range.endOffset);
  return buildSideThreadQuote(fullText, beforeStart.toString().length, beforeEnd.toString().length);
}

function closestSideThreadQuoteRoot(frame: HTMLElement, node: Node): HTMLElement {
  const element = node instanceof HTMLElement ? node : node.parentElement;
  const root = element?.closest<HTMLElement>(
    ".message-body-inner, .tool-bubble, .thinking, .framework, .handoff"
  );
  return root && frame.contains(root) ? root : frame;
}

export function buildSideThreadQuote(
  fullText: string,
  rawStart: number,
  rawEnd: number,
): { quotedText: string; quotedRange: SideThreadQuotedRange } | null {
  const boundedStart = clampNumber(Math.min(rawStart, rawEnd), 0, fullText.length);
  const boundedEnd = clampNumber(Math.max(rawStart, rawEnd), 0, fullText.length);
  const rawText = fullText.slice(boundedStart, boundedEnd);
  const leadingTrim = rawText.length - rawText.trimStart().length;
  const trailingTrim = rawText.length - rawText.trimEnd().length;
  const start = boundedStart + leadingTrim;
  const end = boundedEnd - trailingTrim;
  const quotedText = fullText.slice(start, end);
  if (!quotedText.trim()) return null;
  return {
    quotedText,
    quotedRange: { start, end },
  };
}

function sideSelectionButtonStyle(rect: DOMRect): CSSProperties {
  const top = clampNumber(rect.top - 38, 8, window.innerHeight - 42);
  const left = clampNumber(rect.right + 8, 8, window.innerWidth - 42);
  return {
    top: Math.round(top),
    left: Math.round(left),
  };
}

function sideThreadPopoverStyle(rect: SideThreadWindowRect): CSSProperties {
  return {
    width: Math.round(rect.width),
    height: Math.round(rect.height),
    top: Math.round(rect.top),
    left: Math.round(rect.left),
  };
}

function currentViewportSize(): { width: number; height: number } {
  return {
    width: window.innerWidth,
    height: window.innerHeight,
  };
}

export function resolveSideThreadInitialWindowRect(
  anchorRect: Pick<DOMRect, "top" | "right" | "left">,
  viewport: { width: number; height: number },
): SideThreadWindowRect {
  const width = Math.min(420, Math.max(320, viewport.width - 24));
  const height = Math.min(560, Math.max(320, viewport.height - 24));
  const rightLeft = anchorRect.right + 12;
  const left = rightLeft + width <= viewport.width - 12
    ? rightLeft
    : clampNumber(anchorRect.left - width - 12, 12, viewport.width - width - 12);
  const top = clampNumber(anchorRect.top - 28, 12, viewport.height - height - 12);
  return clampSideThreadWindowRect({ left, top, width, height }, viewport);
}

export function moveSideThreadWindowRect(
  rect: SideThreadWindowRect,
  deltaX: number,
  deltaY: number,
  viewport: { width: number; height: number },
): SideThreadWindowRect {
  return clampSideThreadWindowRect({
    ...rect,
    left: rect.left + deltaX,
    top: rect.top + deltaY,
  }, viewport);
}

export function resizeSideThreadWindowRect(
  rect: SideThreadWindowRect,
  deltaX: number,
  deltaY: number,
  viewport: { width: number; height: number },
): SideThreadWindowRect {
  return clampSideThreadWindowRect({
    ...rect,
    width: rect.width + deltaX,
    height: rect.height + deltaY,
  }, viewport);
}

function clampSideThreadWindowRect(
  rect: SideThreadWindowRect,
  viewport: { width: number; height: number },
): SideThreadWindowRect {
  const margin = 12;
  const minWidth = Math.min(320, Math.max(160, viewport.width - margin * 2));
  const minHeight = Math.min(320, Math.max(180, viewport.height - margin * 2));
  const width = Math.round(clampNumber(rect.width, minWidth, Math.max(minWidth, viewport.width - margin * 2)));
  const height = Math.round(clampNumber(rect.height, minHeight, Math.max(minHeight, viewport.height - margin * 2)));
  return {
    width,
    height,
    left: Math.round(clampNumber(rect.left, margin, viewport.width - width - margin)),
    top: Math.round(clampNumber(rect.top, margin, viewport.height - height - margin)),
  };
}

function startSideThreadWindowGesture(
  event: ReactPointerEvent<HTMLElement>,
  startRect: SideThreadWindowRect,
  resolveNext: (startRect: SideThreadWindowRect, deltaX: number, deltaY: number) => SideThreadWindowRect,
  onWindowRectChange: (rect: SideThreadWindowRect) => void,
) {
  event.preventDefault();
  event.stopPropagation();
  const startX = event.clientX;
  const startY = event.clientY;
  const move = (pointerEvent: PointerEvent) => {
    pointerEvent.preventDefault();
    onWindowRectChange(resolveNext(startRect, pointerEvent.clientX - startX, pointerEvent.clientY - startY));
  };
  const stop = () => {
    window.removeEventListener("pointermove", move);
    window.removeEventListener("pointerup", stop);
    window.removeEventListener("pointercancel", stop);
  };
  window.addEventListener("pointermove", move);
  window.addEventListener("pointerup", stop, { once: true });
  window.addEventListener("pointercancel", stop, { once: true });
}

function fallbackSidePopoverRect(): DOMRect {
  if (typeof DOMRect !== "undefined") {
    return new DOMRect(window.innerWidth - 460, 96, 1, 1);
  }
  return {
    x: window.innerWidth - 460,
    y: 96,
    width: 1,
    height: 1,
    top: 96,
    right: window.innerWidth - 459,
    bottom: 97,
    left: window.innerWidth - 460,
    toJSON: () => ({}),
  } as DOMRect;
}

function sideThreadLocateTarget(thread: SideThreadState, sessionId: string): HistoryLocateTarget {
  return {
    sessionId,
    contextMessageId: thread.anchorContextMessageId,
    contextMessageIndex: thread.anchorMessageIndex,
  };
}

function sideThreadTitle(thread: SideThreadState): string {
  const firstUser = thread.messages.find((message) => message.role === "user");
  return truncateInlineText(sessionHistoryRecordText(firstUser), 36) || "旁路对话";
}

function sideThreadLastText(thread: SideThreadState): string {
  for (let index = thread.messages.length - 1; index >= 0; index -= 1) {
    const message = thread.messages[index];
    if (message.role !== "user" && message.role !== "assistant") continue;
    const text = sessionHistoryRecordText(message);
    if (text) return truncateInlineText(text, 92);
  }
  return thread.error ?? "";
}

function sideThreadStatusLabel(status: SideThreadState["status"]): string {
  if (status === "running") return "running";
  if (status === "error") return "error";
  if (status === "cancelled") return "cancelled";
  if (status === "stale") return "stale";
  return "done";
}

function sessionHistoryRecordText(record: { content: unknown[] } | undefined): string {
  if (!record) return "";
  return contentPartsPlainText(record.content);
}

function contentPartsPlainText(content: unknown[]): string {
  const parts: string[] = [];
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === "text") {
      parts.push(String(part.text ?? ""));
    } else if (part.type === "reasoning") {
      parts.push(String(part.reasoning ?? ""));
    } else if (typeof part.type === "string") {
      parts.push(`[${part.type}]`);
    }
  }
  return parts.join(" ").replace(/\s+/g, " ").trim();
}

function truncateInlineText(value: string, maxLength: number): string {
  if (value.length <= maxLength) return value;
  return `${value.slice(0, Math.max(0, maxLength - 1)).trimEnd()}…`;
}

function optionalDatasetNumber(value: string | undefined): number | undefined {
  if (value === undefined || value === "") return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function formatHistoryResultTimestamp(result: HistorySearchResult): string {
  const fromTimestamp = dateFromHistoryTimestamp(result.timestamp);
  if (fromTimestamp) {
    return fromTimestamp.toLocaleString([], {
      month: "numeric",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  }
  return formatSessionTimestamp(result.sessionUpdatedAt || result.sessionCreatedAt);
}

function dateFromHistoryTimestamp(value: string | number | undefined): Date | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    const millis = value < 10_000_000_000 ? value * 1000 : value;
    const date = new Date(millis);
    return Number.isNaN(date.getTime()) ? null : date;
  }
  if (typeof value === "string" && value.trim()) {
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? null : date;
  }
  return null;
}

function historyRoleLabel(role: string): string {
  if (role === "assistant") return "Hawi";
  if (role === "user") return "User";
  if (role === "tool") return "Tool";
  if (role === "system") return "System";
  if (role === "error") return "Error";
  if (role === "event") return "Event";
  return role || "Message";
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

export function resumePayloadFromInput(input: string): Record<string, unknown> {
  const message = input.trim();
  return message ? { message } : {};
}

function resumePayloadFromContent(content: string | ContentPart[]): Record<string, unknown> {
  if (typeof content === "string") {
    return resumePayloadFromInput(content);
  }
  return content.length > 0 ? { message: content } : {};
}

export function isEditableTextNode(node: ChatNode): boolean {
  return (node.kind === "user" || node.kind === "agent")
    && node.complete !== false
    && (Boolean(node.contextMessageId) || typeof node.contextMessageIndex === "number");
}

export function messageEditStateFromNode(node: ChatNode): MessageEditState {
  const role = node.kind === "agent" ? "agent" : "user";
  return {
    nodeId: node.id,
    role,
    contextMessageId: node.contextMessageId,
    contextMessageIndex: node.contextMessageIndex,
    draft: node.content,
    attachments: role === "user" ? editableAttachmentsFromContentParts(node.contentParts) : [],
    busy: false,
    error: null,
  };
}

export function messageEditTargetPayload(edit: MessageEditState): Record<string, unknown> | null {
  if (edit.contextMessageId) {
    return { context_message_id: edit.contextMessageId };
  }
  if (typeof edit.contextMessageIndex === "number") {
    return { message_index: edit.contextMessageIndex };
  }
  return null;
}

export function editableAttachmentsFromContentParts(parts?: ContentPart[]): EditableAttachment[] {
  if (!Array.isArray(parts)) return [];
  return parts
    .filter(isMediaPart)
    .map((part, index): ExistingEditableAttachment => {
      const source = isRecord(part.source) ? part.source as MediaSource : undefined;
      const blobId = optionalRecordString(source?.blob_id) ?? blobIdFromUri(source?.uri) ?? blobIdFromUri(source?.url);
      const partType = attachmentPartTypeFromContentType(String(part.type));
      const filename = mediaDisplayName(part, source);
      const mimeType = optionalRecordString(source?.mime_type)
        ?? optionalRecordString(source?.mimeType)
        ?? optionalRecordString(source?.mime)
        ?? mimeTypeFromFilename(filename)
        ?? "application/octet-stream";
      return {
        id: `existing-${index}-${blobId ?? filename}`,
        kind: "existing",
        part: cloneContentPart(part),
        source,
        filename,
        mimeType,
        partType,
        size: typeof source?.size === "number" ? source.size : undefined,
      };
    });
}

function attachmentPartTypeFromContentType(type: string): AttachmentPartType {
  if (type === "image" || type === "document" || type === "audio" || type === "video" || type === "file") {
    return type;
  }
  return "file";
}

function cloneContentPart(part: ContentPart): ContentPart {
  return JSON.parse(JSON.stringify(part)) as ContentPart;
}

function createPendingAttachment(file: File): PendingAttachment {
  const mimeType = attachmentMimeType(file);
  const partType = attachmentPartType(file, mimeType);
  return {
    id: `att-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 9)}`,
    file,
    previewUrl: URL.createObjectURL(file),
    filename: file.name || defaultAttachmentFilename(partType),
    mimeType,
    partType,
    size: file.size,
    status: "ready",
    progress: 0,
  };
}

function attachmentFilesFromFileList(files: FileList | null): File[] {
  if (!files) return [];
  return Array.from(files);
}

function hasAttachmentFile(dataTransfer: DataTransfer): boolean {
  if (attachmentFilesFromFileList(dataTransfer.files).length > 0) return true;
  return Array.from(dataTransfer.items).some((item) => (
    item.kind === "file"
  ));
}

function attachmentMimeType(file: File): string {
  return file.type || mimeTypeFromFilename(file.name) || "application/octet-stream";
}

function attachmentPartType(file: File, mimeType = attachmentMimeType(file)): AttachmentPartType {
  const extension = fileExtension(file.name);
  if (mimeType.startsWith("image/") || imageAttachmentExtensions.has(extension)) {
    return "image";
  }
  if (mimeType.startsWith("audio/") || audioAttachmentExtensions.has(extension)) {
    return "audio";
  }
  if (mimeType.startsWith("video/") || videoAttachmentExtensions.has(extension)) {
    return "video";
  }
  if (
    mimeType.startsWith("text/")
    || documentAttachmentMimeTypes.has(mimeType)
    || documentAttachmentExtensions.has(extension)
  ) {
    return "document";
  }
  return "file";
}

function defaultAttachmentFilename(partType: AttachmentPartType): string {
  return partType === "image" ? "image" : "attachment";
}

function mimeTypeFromFilename(filename: string): string | undefined {
  const extension = fileExtension(filename);
  return extension ? extensionMimeTypes[extension] : undefined;
}

function fileExtension(filename: string): string {
  const match = /\.([^./\\]+)$/.exec(filename.toLowerCase());
  return match?.[1] ?? "";
}

function contentPartFromAttachment(attachment: PendingAttachment, source: BlobSource): ContentPart {
  if (attachment.partType === "document") {
    return {
      type: "document",
      source,
      title: attachment.filename,
      context: null,
    };
  }
  return {
    type: attachment.partType,
    source,
  };
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  if (!window.crypto?.subtle) {
    throw new Error("Current runtime does not support crypto.subtle");
  }
  const buffer = bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  ) as ArrayBuffer;
  const digest = await window.crypto.subtle.digest("SHA-256", buffer);
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    const chunk = bytes.subarray(offset, offset + chunkSize);
    binary += String.fromCharCode(...chunk);
  }
  return btoa(binary);
}

function bytesToDataUrl(bytes: Uint8Array, mimeType: string): string {
  return `data:${mimeType || "application/octet-stream"};base64,${bytesToBase64(bytes)}`;
}

function revokeObjectUrl(value: string) {
  if (value.startsWith("blob:")) {
    URL.revokeObjectURL(value);
  }
}

function normalizeBlobFinalizePayload(
  value: unknown,
  blobId: string,
  attachment: PendingAttachment,
  sha256: string,
): BlobFinalizePayload {
  if (!isRecord(value)) {
    return {
      blob_id: blobId,
      uri: `hawi-blob://${blobId}`,
      sha256,
      direction: "inbound",
      size: attachment.size,
      mime: attachment.mimeType,
    };
  }
  return {
    blob_id: optionalRecordString(value.blob_id) ?? blobId,
    uri: optionalRecordString(value.uri),
    sha256: optionalRecordString(value.sha256) ?? sha256,
    direction: value.direction === "outbound" ? "outbound" : "inbound",
    size: typeof value.size === "number" ? value.size : attachment.size,
    mime: typeof value.mime === "string" ? value.mime : attachment.mimeType,
    ref_count: typeof value.ref_count === "number" ? value.ref_count : undefined,
  };
}

function attachmentStatusText(attachment: PendingAttachment): string {
  if (attachment.status === "uploading") {
    return `${Math.round(attachment.progress * 100)}%`;
  }
  if (attachment.status === "uploaded") {
    return "已上传";
  }
  if (attachment.status === "error") {
    return "失败";
  }
  return `${attachmentTypeLabel(attachment)} - ${formatByteCount(attachment.size)}`;
}

function attachmentTypeLabel(attachment: PendingAttachment): string {
  if (attachment.mimeType && attachment.mimeType !== "application/octet-stream") {
    return attachment.mimeType;
  }
  return attachment.partType;
}

export function shouldUseContentPartsView(parts?: ContentPart[]): boolean {
  return Array.isArray(parts)
    && parts.some((part) => part.type !== "text" && isRenderableContentPart(part));
}

function hasRenderableContentParts(parts?: ContentPart[]): boolean {
  return Array.isArray(parts) && parts.some(isRenderableContentPart);
}

function isRenderableContentPart(part: ContentPart): boolean {
  if (part.type === "text") {
    return typeof part.text === "string" && part.text.trim().length > 0;
  }
  return isMediaPart(part);
}

function isMediaPart(part: ContentPart): boolean {
  return part.type === "image"
    || part.type === "document"
    || part.type === "audio"
    || part.type === "video"
    || part.type === "file";
}

function collectMissingBlobPreviewRequests(
  state: AppState,
  blobPreviewUrls: BlobPreviewUrls,
): BlobPreviewRequest[] {
  const requests = new Map<string, BlobPreviewRequest>();
  for (const node of state.nodes) {
    collectBlobPreviewRequestsFromContentParts(node.contentParts, blobPreviewUrls, requests);
  }
  for (const messages of Object.values(state.queueMessages)) {
    for (const message of messages) {
      collectBlobPreviewRequestsFromContentParts(message.contentParts, blobPreviewUrls, requests);
    }
  }
  for (const subagent of Object.values(state.subagents)) {
    for (const node of subagent.nodes) {
      collectBlobPreviewRequestsFromContentParts(node.contentParts, blobPreviewUrls, requests);
    }
  }
  return [...requests.values()];
}

function collectBlobPreviewRequestsFromContentParts(
  parts: ContentPart[] | undefined,
  blobPreviewUrls: BlobPreviewUrls,
  requests: Map<string, BlobPreviewRequest>,
) {
  if (!Array.isArray(parts)) return;
  for (const part of parts) {
    if (isBlobPreviewFetchablePart(part) && isRecord(part.source)) {
      const request = blobPreviewRequestFromSource(part.source as MediaSource, blobPreviewUrls);
      if (request && !requests.has(request.blobId)) {
        requests.set(request.blobId, request);
      }
    }
    if (Array.isArray(part.content)) {
      collectBlobPreviewRequestsFromContentParts(
        part.content.filter(isContentPartRecord),
        blobPreviewUrls,
        requests,
      );
    }
  }
}

function isBlobPreviewFetchablePart(part: ContentPart): boolean {
  return part.type === "image" || part.type === "audio" || part.type === "video";
}

function blobPreviewRequestFromSource(
  source: MediaSource,
  blobPreviewUrls: BlobPreviewUrls,
): BlobPreviewRequest | undefined {
  const blobId = optionalRecordString(source.blob_id)
    ?? blobIdFromUri(source.uri)
    ?? blobIdFromUri(source.url);
  if (!blobId) return undefined;

  const uri = optionalRecordString(source.uri) ?? `hawi-blob://${blobId}`;
  const keys = [
    blobId,
    uri,
    `hawi-blob://${blobId}`,
    optionalRecordString(source.url),
  ].filter((key): key is string => Boolean(key));
  if (keys.some((key) => Boolean(blobPreviewUrls[key]))) {
    return undefined;
  }

  const size = typeof source.size === "number" ? source.size : undefined;
  if (size !== undefined && size > MAX_ATTACHMENT_BYTES) {
    return undefined;
  }
  return {
    blobId,
    uri,
    size,
    mimeType: optionalRecordString(source.mime_type)
      ?? optionalRecordString(source.mimeType)
      ?? optionalRecordString(source.mime),
  };
}

function isContentPartRecord(value: unknown): value is ContentPart {
  return isRecord(value) && typeof value.type === "string";
}

function mediaDisplayUrl(source: MediaSource | undefined, blobPreviewUrls: BlobPreviewUrls): string | undefined {
  if (!source) return undefined;
  const blobId = optionalRecordString(source.blob_id);
  if (blobId && blobPreviewUrls[blobId]) {
    return blobPreviewUrls[blobId];
  }
  for (const key of [source.uri, source.url, source.data_uri]) {
    if (typeof key === "string" && blobPreviewUrls[key]) {
      return blobPreviewUrls[key];
    }
  }
  const uriBlobId = blobIdFromUri(source.uri) ?? blobIdFromUri(source.url);
  if (uriBlobId && blobPreviewUrls[uriBlobId]) {
    return blobPreviewUrls[uriBlobId];
  }
  for (const value of [source.data_uri, source.url, source.uri]) {
    if (!value) continue;
    if (value.startsWith("hawi-blob://")) continue;
    if (value.startsWith("data:") || value.startsWith("blob:") || /^https?:\/\//i.test(value)) {
      return value;
    }
  }
  if (source.data && (source.mime_type || source.mime)) {
    return `data:${source.mime_type ?? source.mime};base64,${source.data}`;
  }
  return undefined;
}

function blobIdFromUri(value: unknown): string | undefined {
  if (typeof value !== "string" || !value.startsWith("hawi-blob://")) return undefined;
  return value.slice("hawi-blob://".length) || undefined;
}

function mediaDisplayName(part: ContentPart, source: MediaSource | undefined): string {
  return optionalRecordString(source?.filename)
    ?? (typeof part.title === "string" && part.title.trim() ? part.title : undefined)
    ?? mediaKindLabel(String(part.type));
}

function mediaMetaText(source: MediaSource | undefined): string {
  if (!source) return "";
  const mime = optionalRecordString(source.mime_type)
    ?? optionalRecordString(source.mimeType)
    ?? optionalRecordString(source.mime);
  const size = typeof source.size === "number" ? formatByteCount(source.size) : "";
  return [mime, size].filter(Boolean).join(" · ");
}

function mediaKindLabel(type: string): string {
  if (type === "image") return "Image";
  if (type === "audio") return "Audio";
  if (type === "video") return "Video";
  if (type === "document") return "Document";
  return "File";
}

function formatByteCount(size: number): string {
  if (!Number.isFinite(size) || size < 0) return "";
  if (size < 1024) return `${size} B`;
  const units = ["KB", "MB", "GB"];
  let value = size / 1024;
  for (const unit of units) {
    if (value < 1024 || unit === units[units.length - 1]) {
      return `${value.toFixed(value >= 10 ? 0 : 1)} ${unit}`;
    }
    value /= 1024;
  }
  return `${size} B`;
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
      gui_launch_profile: normalizeLaunchProfile(item.gui_launch_profile),
      last_cwd: optionalPayloadString(item.last_cwd)
    }))
    .filter((item) => item.session_id);
}

function frameSessionId(frame: CoreFrame): string | null {
  return framePayload(frame) ? optionalPayloadString(framePayload(frame)?.session_id) : null;
}

export function upsertSessionRuntime(
  sessions: SessionMetaPayload[],
  sessionId: string,
  patch: Pick<SessionMetaPayload, "load_state" | "loaded_at" | "last_finished_at" | "last_cwd">,
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
    modelProviderConfigs: normalizePluginConfigs(value.modelProviderConfigs),
    systemPrompt,
    selectedPlugins: Array.isArray(value.selectedPlugins)
      ? value.selectedPlugins.filter((item): item is string => typeof item === "string")
      : [],
    pluginConfigs: normalizePluginConfigs(value.pluginConfigs),
    toolCallPurposeEnabled: value.toolCallPurposeEnabled !== false,
    profilingEnabled: value.profilingEnabled !== false,
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
    modelProviderConfigs: normalizePluginConfigs(profile.modelProviderConfigs),
    systemPrompt: profile.systemPrompt,
    selectedPlugins: [...profile.selectedPlugins],
    pluginConfigs: normalizePluginConfigs(profile.pluginConfigs),
    toolCallPurposeEnabled: profile.toolCallPurposeEnabled !== false,
    profilingEnabled: profile.profilingEnabled !== false
  };
}

function launchProfileFromConfig(config: PersistedConfig): SessionLaunchProfile {
  return {
    version: 1,
    modelName: config.modelName,
    modelProviderConfigs: normalizePluginConfigs(config.modelProviderConfigs),
    systemPrompt: config.systemPrompt,
    selectedPlugins: [...config.selectedPlugins],
    pluginConfigs: normalizePluginConfigs(config.pluginConfigs),
    toolCallPurposeEnabled: config.toolCallPurposeEnabled,
    profilingEnabled: config.profilingEnabled !== false
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

export function projectNameFromPath(value: string | null | undefined): string {
  const path = value?.trim();
  if (!path) return "-";
  const normalized = path.replace(/[\\/]+$/, "");
  if (!normalized) return path;
  const parts = normalized.split(/[\\/]+/).filter(Boolean);
  return parts.at(-1) ?? normalized;
}

export function middleEllipsizePath(value: string | null | undefined, maxLength = 58): string {
  const text = value?.trim() || "-";
  if (text.length <= maxLength) return text;
  const budget = Math.max(8, maxLength - 1);
  const headLength = Math.ceil(budget / 2);
  const tailLength = Math.floor(budget / 2);
  return `${text.slice(0, headLength)}…${text.slice(text.length - tailLength)}`;
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
  return content.replace(/^\[(?:Lines \d+-\d+ of \d+|Chars \d+-\d+ of \d+|language: [^\]\n]+)(?: \| (?:Lines \d+-\d+ of \d+|Chars \d+-\d+ of \d+|language: [^\]\n]+))*\]\n/, "");
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

export function resolveToolCallPurposeControlState(): { disabled: boolean; title: string } {
  return {
    disabled: false,
    title: "影响后续新 Session；当前 Session 需要重启后生效"
  };
}

export function shouldConfirmSystemPromptEdit(input: {
  hasConversation: boolean;
  sessionId: string | null;
  messageCount: number;
  acknowledgedSessionId: string | null;
  acknowledgedMessageCount: number | null;
}): boolean {
  if (!input.hasConversation) return false;
  return input.acknowledgedSessionId !== input.sessionId
    || input.acknowledgedMessageCount !== input.messageCount;
}

export function shouldConfirmPluginSettingsApply(input: {
  hasConversation: boolean;
  sessionId: string | null;
  messageCount: number;
  acknowledgedSessionId: string | null;
  acknowledgedMessageCount: number | null;
}): boolean {
  return shouldConfirmSystemPromptEdit(input);
}

export function shouldConfirmToolCallPurposeChange(input: {
  hasConversation: boolean;
  currentEnabled: boolean;
  nextEnabled: boolean;
}): boolean {
  return input.hasConversation && input.currentEnabled !== input.nextEnabled;
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

function formatUsageTokenCount(value: number): string {
  if (!Number.isFinite(value)) return "-";
  const absolute = Math.abs(value);
  if (absolute >= 1_000_000) return `${trimFixedOne(value / 1_000_000)}M`;
  if (absolute >= 1000) return `${trimFixedOne(value / 1000)}K`;
  return `${Math.round(value)}`;
}

function trimFixedOne(value: number): string {
  const fixed = value.toFixed(1);
  return fixed.endsWith(".0") ? fixed.slice(0, -2) : fixed;
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

function transcriptTailKey(nodes: ChatNode[], processing?: ProcessingState): string {
  const last = nodes[nodes.length - 1];
  return [
    last?.id ?? "empty",
    last?.content.length ?? 0,
    last?.complete === false ? "open" : "done",
    last?.tool?.argsRaw.length ?? 0,
    last?.tool?.resultPreview.length ?? 0,
    processing?.id ?? "idle",
    processing?.content.length ?? 0,
  ].join(":");
}

function hasActiveTextSelection(): boolean {
  if (typeof window === "undefined") return false;
  const selection = window.getSelection();
  return Boolean(selection && !selection.isCollapsed && selection.toString().trim());
}

function hasTranscriptSelection(element: HTMLElement): boolean {
  if (typeof window === "undefined") return false;
  const selection = window.getSelection();
  if (!selection || selection.isCollapsed || selection.rangeCount === 0) return false;
  const anchor = selection.anchorNode;
  const focus = selection.focusNode;
  return Boolean(
    anchor
    && focus
    && element.contains(anchor)
    && element.contains(focus)
    && selection.toString().trim()
  );
}

function isInteractiveTranscriptTarget(target: EventTarget | null): boolean {
  return target instanceof Element && Boolean(target.closest([
    "button",
    "input",
    "textarea",
    "select",
    "a",
    "summary",
    ".bubble-head",
    "[role='button']",
  ].join(",")));
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
  if (isAutoScrolling) return true;
  if (selectingChat) return false;
  if (nearBottom) return true;
  if (userScrollIntent) return false;
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

function resizeMessageEditTextarea(textarea: HTMLTextAreaElement | null) {
  if (!textarea) return;
  textarea.style.height = "auto";
  const style = window.getComputedStyle(textarea);
  const maxHeight = Number.parseFloat(style.maxHeight);
  const minHeight = Number.parseFloat(style.minHeight) || 0;
  const boundedMaxHeight = Number.isFinite(maxHeight) && maxHeight > 0 ? maxHeight : textarea.scrollHeight;
  const nextHeight = Math.min(Math.max(textarea.scrollHeight, minHeight), boundedMaxHeight);
  textarea.style.height = `${nextHeight}px`;
  textarea.style.overflowY = textarea.scrollHeight > boundedMaxHeight ? "auto" : "hidden";
}

type InputKeyEvent = {
  key: string;
  shiftKey: boolean;
  altKey?: boolean;
  ctrlKey?: boolean;
  metaKey?: boolean;
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

export function shouldNavigateInputHistoryFromKeyEvent(
  event: InputKeyEvent,
  value: string,
  selectionStart: number | null,
  selectionEnd: number | null,
  inputComposing: boolean,
  historyActive: boolean
): "previous" | "next" | null {
  if (event.shiftKey || event.altKey || event.ctrlKey || event.metaKey) return null;
  if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return null;
  if (isInputComposing(event, inputComposing)) return null;
  if (selectionStart === null || selectionEnd === null || selectionStart !== selectionEnd) {
    return null;
  }
  const caretOnFirstLine = value.slice(0, selectionStart).indexOf("\n") === -1;
  const caretOnLastLine = value.slice(selectionEnd).indexOf("\n") === -1;
  if (event.key === "ArrowUp" && caretOnFirstLine) {
    return "previous";
  }
  if (event.key === "ArrowDown" && historyActive && caretOnLastLine) {
    return "next";
  }
  return null;
}

export function inputHistoryFromChatNodes(nodes: ChatNode[]): string[] {
  return nodes
    .filter((node) => node.kind === "user")
    .map((node) => node.content.trim())
    .filter((content) => content.length > 0);
}

export function mergeInputHistory(...histories: string[][]): string[] {
  const seen = new Set<string>();
  const merged: string[] = [];
  histories.flat().forEach((entry) => {
    const text = entry.trim();
    if (!text || seen.has(text)) return;
    seen.add(text);
    merged.push(text);
  });
  return merged.slice(-MAX_INPUT_HISTORY);
}

function isInputComposing(event: InputKeyEvent, inputComposing: boolean): boolean {
  return inputComposing
    || event.nativeEvent.isComposing === true
    || event.nativeEvent.keyCode === 229
    || event.nativeEvent.which === 229;
}

export function renderMarkdown(value: string): string {
  const env: MarkdownRenderEnv = { latexHtml: [] };
  return restoreLatexPlaceholders(
    sanitizeRenderedHtml(markdown.render(value, env)),
    env.latexHtml ?? []
  );
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

  const activeElement = document.activeElement instanceof HTMLElement ? document.activeElement : null;
  const selection = typeof window === "undefined" ? null : window.getSelection();
  const ranges = selection
    ? Array.from({ length: selection.rangeCount }, (_, index) => selection.getRangeAt(index).cloneRange())
    : [];
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
  if (selection && ranges.length > 0) {
    selection.removeAllRanges();
    for (const range of ranges) {
      selection.addRange(range);
    }
  }
  activeElement?.focus({ preventScroll: true });
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

function renderMermaidFence(value: string): string {
  const source = encodeURIComponent(value);
  return [
    `<div class="mermaid-preview-shell" data-mermaid-source="${escapeHtmlAttributeValue(source)}">`,
    "<span class=\"mermaid-preview-status\">Rendering diagram...</span>",
    "</div>",
    sourceCodeDisclosure(codeBlock(escapeHtml(value), "mermaid"))
  ].join("");
}

async function renderMermaidDiagrams(
  root: HTMLElement,
  isCancelled: () => boolean,
): Promise<void> {
  const containers = pendingMermaidContainers(root)
    .filter((container) => container.dataset.mermaidRendering !== "true");
  if (containers.length === 0 || typeof window === "undefined") return;

  let mermaid: Awaited<ReturnType<typeof loadMermaid>>;
  try {
    mermaid = await loadMermaid();
  } catch (error) {
    renderMermaidError(containers, error);
    return;
  }
  if (isCancelled()) return;

  for (const container of containers) {
    if (isCancelled()) return;
    const source = container.dataset.mermaidSource ?? "";
    container.dataset.mermaidRendering = "true";
    try {
      const definition = decodeURIComponent(source);
      const renderId = `hawi-mermaid-${++mermaidRenderSequence}`;
      const rendered = await enqueueMermaidRender(mermaid, renderId, definition);
      if (isCancelled()) return;
      container.innerHTML = sanitizeRenderedMermaidHtml(rendered.svg);
      container.dataset.mermaidRendered = "true";
      delete container.dataset.mermaidSource;
      notifyMarkdownContentResized(container);
    } catch (error) {
      container.dataset.mermaidRendered = "true";
      container.classList.add("mermaid-preview-error");
      container.textContent = mermaidErrorMessage(error);
      notifyMarkdownContentResized(container);
    } finally {
      delete container.dataset.mermaidRendering;
    }
  }
}

function notifyMarkdownContentResized(container: HTMLElement) {
  const dispatch = () => {
    container.dispatchEvent(new CustomEvent("hawi:markdown-content-resized", {
      bubbles: true,
    }));
  };
  dispatch();
  window.requestAnimationFrame(dispatch);
}

function pendingMermaidContainers(root: HTMLElement): HTMLElement[] {
  return Array.from(
    root.querySelectorAll<HTMLElement>(".mermaid-preview-shell[data-mermaid-source]")
  ).filter((container) => container.dataset.mermaidRendered !== "true");
}

function hasPendingMermaidDiagrams(root: HTMLElement): boolean {
  return pendingMermaidContainers(root)
    .some((container) => container.dataset.mermaidRendering !== "true");
}

async function enqueueMermaidRender(
  mermaid: Awaited<ReturnType<typeof loadMermaid>>,
  renderId: string,
  definition: string,
): Promise<{ svg: string; bindFunctions?: (element: Element) => void }> {
  const task = mermaidRenderQueue.then(() => (
    withTimeout(
      mermaid.render(renderId, definition),
      MERMAID_RENDER_TIMEOUT_MS,
      "Mermaid render timed out",
    )
  ));
  mermaidRenderQueue = task.then(
    () => undefined,
    () => undefined,
  );
  return task;
}

function withTimeout<T>(promise: Promise<T>, timeoutMs: number, message: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const timeoutId = window.setTimeout(() => reject(new Error(message)), timeoutMs);
    promise.then(
      (value) => {
        window.clearTimeout(timeoutId);
        resolve(value);
      },
      (error) => {
        window.clearTimeout(timeoutId);
        reject(error);
      },
    );
  });
}

async function loadMermaid() {
  if (mermaidModulePromise === null) {
    mermaidModulePromise = import("mermaid");
  }
  const module = await mermaidModulePromise;
  const mermaid = module.default;
  mermaid.initialize(MERMAID_RENDER_CONFIG);
  return mermaid;
}

function renderMermaidError(containers: HTMLElement[], error: unknown) {
  for (const container of containers) {
    container.dataset.mermaidRendered = "true";
    container.classList.add("mermaid-preview-error");
    container.textContent = mermaidErrorMessage(error);
  }
}

function mermaidErrorMessage(error: unknown): string {
  return `Mermaid render failed: ${error instanceof Error ? error.message : String(error)}`;
}

function renderSvgFence(value: string): string {
  const preview = renderSvgPreview(value);
  const highlighted = highlightedCode(value, "svg");
  return `${preview}${sourceCodeDisclosure(codeBlock(highlighted.html, highlighted.language))}`;
}

function sourceCodeDisclosure(content: string): string {
  return [
    "<details class=\"source-code-disclosure\">",
    "<summary class=\"source-code-summary\">",
    "<span class=\"source-code-chevron\" aria-hidden=\"true\"></span>",
    "<span class=\"source-code-label\">代码</span>",
    "</summary>",
    content,
    "</details>",
  ].join("");
}

function sanitizeRenderedHtml(value: string): string {
  return sanitizeRenderedHtmlWithOptions(value, {
    allowStyleAttributes: false,
    allowStyleElements: false,
  });
}

export function sanitizeRenderedMermaidHtml(value: string): string {
  return value;
}

function sanitizeRenderedHtmlWithOptions(
  value: string,
  options: {
    allowStyleAttributes: boolean;
    allowStyleElements: boolean;
  },
): string {
  const blockedTags = options.allowStyleElements
    ? "script|iframe|object|embed|link|meta|base|form|input|textarea|select|option|foreignObject"
    : "script|style|iframe|object|embed|link|meta|base|form|input|textarea|select|option|foreignObject";
  const blockedTagsPattern = new RegExp(`<(${blockedTags})\\b[\\s\\S]*?<\\/\\1>`, "gi");
  const blockedSelfClosingPattern = new RegExp(`<(${blockedTags})\\b[^>]*\\/?>`, "gi");

  return value
    .replace(blockedTagsPattern, "")
    .replace(blockedSelfClosingPattern, "")
    .replace(
      /<style\b[^>]*>([\s\S]*?)<\/style>/gi,
      (_match: string, css: string) => {
        if (!options.allowStyleElements) return "";
        const sanitized = sanitizeCssText(css);
        return sanitized ? `<style>${sanitized}</style>` : "";
      }
    )
    .replace(/\s+on[a-z]+\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+)/gi, "")
    .replace(/\s+srcdoc\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+)/gi, "")
    .replace(
      /\s+style\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)/gi,
      (match: string, rawValue: string) => {
        if (!options.allowStyleAttributes) return "";
        const sanitized = sanitizeCssText(unquoteHtmlAttribute(rawValue));
        if (!sanitized) return "";
        return ` style="${escapeHtmlAttributeValue(sanitized)}"`;
      }
    )
    .replace(
      /\s+(href|src|xlink:href|formaction)\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)/gi,
      (match: string, name: string, rawValue: string) => {
        const valueText = unquoteHtmlAttribute(rawValue).trim();
        if (!isSafeHtmlUrl(valueText)) return "";
        return ` ${name}=${rawValue}`;
      }
    );
}

function sanitizeCssText(value: string): string {
  return decodeHtmlAttributeEntities(value)
    .replace(/<\/?style\b[^>]*>/gi, "")
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/@import\b[^;{}]*(?:;|$)/gi, "")
    .replace(/url\s*\([^)]*\)/gi, "")
    .replace(/expression\s*\([^)]*\)/gi, "")
    .replace(/(?:javascript|vbscript|data)\s*:/gi, "")
    .replace(/(^|[;{]\s*)(?:-moz-binding|behavior)\s*:[^;{}]*/gi, "$1")
    .trim();
}

function unquoteHtmlAttribute(value: string): string {
  if (
    (value.startsWith("\"") && value.endsWith("\""))
    || (value.startsWith("'") && value.endsWith("'"))
  ) {
    return value.slice(1, -1);
  }
  return value;
}

function isSafeHtmlUrl(value: string): boolean {
  const normalized = stripUnsafeUrlWhitespace(
    decodeHtmlAttributeEntities(value)
  ).toLowerCase();
  if (!normalized) return true;
  if (normalized.startsWith("#")) return true;
  if (
    normalized.startsWith("/")
    || normalized.startsWith("./")
    || normalized.startsWith("../")
  ) {
    return true;
  }
  if (
    normalized.startsWith("http:")
    || normalized.startsWith("https:")
    || normalized.startsWith("mailto:")
    || normalized.startsWith("tel:")
  ) {
    return true;
  }
  if (/^data:image\/(?:png|jpe?g|gif|webp|svg\+xml)[;,]/.test(normalized)) {
    return true;
  }
  return !/^[a-z][a-z0-9+.-]*:/i.test(normalized);
}

function stripUnsafeUrlWhitespace(value: string): string {
  return Array.from(value)
    .filter((char) => {
      const codePoint = char.codePointAt(0) ?? 0;
      return codePoint > 31 && codePoint !== 127 && !/\s/.test(char);
    })
    .join("");
}

function decodeHtmlAttributeEntities(value: string): string {
  return value
    .replace(/&colon;/gi, ":")
    .replace(/&tab;/gi, "")
    .replace(/&newline;/gi, "")
    .replace(/&#0*58;/gi, ":")
    .replace(/&#x0*3a;/gi, ":");
}

function renderSvgPreview(value: string): string {
  const sanitized = sanitizeSvgForPreview(value);
  if (!sanitized) return "";
  const src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(sanitized)}`;
  return [
    "<div class=\"svg-preview-shell\">",
    `<img class="svg-preview" alt="SVG preview" src="${escapeHtmlAttributeValue(src)}" />`,
    "</div>"
  ].join("");
}

function sanitizeSvgForPreview(value: string): string | null {
  let svg = value.trim();
  if (!svg) return null;
  svg = svg
    .replace(/<\?xml[\s\S]*?\?>/gi, "")
    .replace(/<!doctype[\s\S]*?>/gi, "")
    .replace(/<!--[\s\S]*?-->/g, "")
    .trim();
  if (!/^<svg[\s>]/i.test(svg) || !/<\/svg>\s*$/i.test(svg)) {
    return null;
  }
  svg = svg
    .replace(/<(script|foreignObject|iframe|object|embed|link|meta|base|audio|video|canvas|image)\b[\s\S]*?<\/\1>/gi, "")
    .replace(/<(script|foreignObject|iframe|object|embed|link|meta|base|audio|video|canvas|image)\b[^>]*\/?>/gi, "")
    .replace(/\s+on[a-z]+\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+)/gi, "")
    .replace(/\s+(?:href|xlink:href|src)\s*=\s*(?:"(?!#)[^"]*"|'(?!#)[^']*'|(?!#)[^\s>]+)/gi, "")
    .replace(/\s+style\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+)/gi, "");
  return svg.trim();
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
  return [
    "<div class=\"code-block-shell\">",
    "<button type=\"button\" class=\"code-copy-button\" data-copy-state=\"idle\" title=\"复制代码\" aria-label=\"复制代码\"><span class=\"code-copy-icon\" aria-hidden=\"true\"></span></button>",
    `<pre class="code-block"><code class="hljs${languageClass}">${value}</code></pre>`,
    "</div>"
  ].join("");
}

function escapeHtmlAttribute(value: string): string {
  return value.replace(/[^A-Za-z0-9_-]/g, "");
}

function escapeHtmlAttributeValue(value: string): string {
  return value.replace(/[&"]/g, (char) => char === "&" ? "&amp;" : "&quot;");
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
