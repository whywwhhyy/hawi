import type { CoreFrame, PluginArtifactPayload, QueueKind } from "../shared/protocol";

const TOOL_CALL_DESCRIPTION_PARAMETER = "tool_call_description";
const MAX_DEBUG_LINES = 200;
const MAX_RESULT_PREVIEW_LENGTH = 1200;

export type ChatKind = "user" | "agent" | "thinking" | "tool" | "system" | "meta" | "error" | "debug" | "divider";

export interface ChatNode {
  id: string;
  kind: ChatKind;
  content: string;
  complete?: boolean;
  queue?: QueueKind;
  tool?: ToolState;
}

export interface ToolState {
  runId: string;
  toolCallId: string;
  name: string;
  description?: string;
  status: "pending" | "running" | "success" | "fail";
  argsRaw: string;
  argsState: "pending" | "streaming" | "complete";
  arguments?: unknown;
  resultPreview: string;
  durationMs?: number;
  progress?: ToolProgressState;
}

export interface ToolProgressState {
  pluginId: string;
  pluginName: string;
  progress?: number;
  status?: string;
  label?: string;
  message?: string;
  data?: unknown;
  updatedAt: number;
}

export interface PluginArtifactState {
  key: string;
  id: string;
  pluginId: string;
  pluginName: string;
  artifactType: string;
  title: string;
  content?: string;
  data?: unknown;
  mimeType?: string;
  language?: string;
  uri?: string;
  path?: string;
  description?: string;
  status?: string;
  metadata?: Record<string, unknown>;
  updatedAt: number;
}

export interface PluginMessageState {
  id: string;
  pluginId: string;
  pluginName: string;
  level: "debug" | "info" | "warning" | "error";
  title?: string;
  message: string;
  data?: unknown;
  timestamp: number;
}

export interface PluginStatusState {
  pluginId: string;
  pluginName: string;
  status: string;
  label?: string;
  message?: string;
  progress?: number;
  data?: unknown;
  updatedAt: number;
}

interface RunState {
  agentNodeId?: string;
  thinkingNodeId?: string;
}

export interface AppState {
  nodes: ChatNode[];
  runs: Record<string, RunState>;
  toolNodeByCallId: Record<string, string>;
  activeRunId?: string;
  schedulerState: string;
  agentState: string;
  queueLengths: Record<QueueKind, number>;
  metadataLines: string[];
  debugLines: string[];
  errors: string[];
  artifacts: Record<string, PluginArtifactState>;
  artifactOrder: string[];
  selectedArtifactId?: string;
  pluginMessages: PluginMessageState[];
  pluginStatuses: Record<string, PluginStatusState>;
  toolProgress: Record<string, ToolProgressState>;
}

export function createInitialState(): AppState {
  return {
    nodes: [],
    runs: {},
    toolNodeByCallId: {},
    schedulerState: "IDLE",
    agentState: "IDLE",
    queueLengths: { normal: 0, high_prio: 0, urgent: 0 },
    metadataLines: [],
    debugLines: [],
    errors: [],
    artifacts: {},
    artifactOrder: [],
    pluginMessages: [],
    pluginStatuses: {},
    toolProgress: {}
  };
}

export function reduceCoreEvent(state: AppState, frame: CoreFrame): AppState {
  const payload = (frame.payload ?? {}) as Record<string, unknown>;
  switch (frame.type) {
    case "gui.clear_chat":
      return {
        ...state,
        nodes: [],
        runs: {},
        toolNodeByCallId: {},
        activeRunId: undefined,
        metadataLines: [],
        debugLines: [],
        errors: [],
        artifacts: {},
        artifactOrder: [],
        selectedArtifactId: undefined,
        pluginMessages: [],
        pluginStatuses: {},
        toolProgress: {}
      };

    case "gui.select_artifact": {
      const artifactKey = String(payload.artifact_key ?? "");
      if (!artifactKey || !state.artifacts[artifactKey]) {
        return state;
      }
      return { ...state, selectedArtifactId: artifactKey };
    }

    case "core.ready":
      return addSystem(state, `模型已就绪: ${String(payload.model_name ?? "")}`.trim());

    case "core.status":
      return updateStatus(state, payload);

    case "run.start": {
      const runId = String(payload.run_id ?? "");
      const queue = normalizeQueue(payload.queue);
      return {
        ...appendNode(state, {
          id: nodeId("user", runId),
          kind: "user",
          queue,
          content: String(payload.user_content ?? "")
        }),
        activeRunId: runId,
        runs: { ...state.runs, [runId]: {} }
      };
    }

    case "run.text_delta": {
      const runId = String(payload.run_id ?? state.activeRunId ?? "");
      if (!runId) return state;
      const delta = String(payload.delta ?? "");
      return appendRunDelta(completeThinkingForRun(state, runId), runId, "agent", delta);
    }

    case "run.thinking_delta": {
      const runId = String(payload.run_id ?? state.activeRunId ?? "");
      if (!runId) return state;
      return appendRunDelta(state, runId, "thinking", String(payload.delta ?? ""));
    }

    case "run.stop": {
      const runId = String(payload.run_id ?? "");
      const completedState = completeThinkingForRun(state, runId);
      const nextRuns = { ...completedState.runs };
      delete nextRuns[runId];
      return {
        ...appendNode(completedState, {
          id: nodeId("divider", `${runId}-${Date.now()}`),
          kind: "divider",
          content: formatRunStop(payload)
        }),
        runs: nextRuns,
        activeRunId: state.activeRunId === runId ? undefined : state.activeRunId
      };
    }

    case "tool.call_start": {
      const runId = String(payload.run_id ?? state.activeRunId ?? "");
      const completedState = completeThinkingForRun(state, runId);
      const toolCallId = String(payload.tool_call_id ?? "");
      const status = normalizeToolStatus(payload.status, "running");
      const hasArguments = Object.prototype.hasOwnProperty.call(payload, "arguments");
      const argumentInfo = hasArguments ? splitToolArguments(payload.arguments) : undefined;
      const existingNodeId = completedState.toolNodeByCallId[toolCallId];
      if (existingNodeId) {
        return updateNode(completedState, existingNodeId, (node) => {
          if (!node.tool) return node;
          const description = optionalString(payload.tool_call_description)
            ?? argumentInfo?.description
            ?? node.tool.description;
          return {
            ...node,
            tool: {
              ...node.tool,
              runId,
              name: String(payload.tool_name || node.tool.name),
              description,
              status,
              argsState: argumentInfo ? "complete" : node.tool.argsState,
              arguments: argumentInfo ? argumentInfo.arguments : node.tool.arguments,
              argsRaw: argumentInfo
                ? JSON.stringify(argumentInfo.arguments, null, 2)
                : node.tool.argsRaw
            }
          };
        });
      }
      const tool: ToolState = {
        runId,
        toolCallId,
        name: String(payload.tool_name ?? "pending"),
        description: optionalString(payload.tool_call_description),
        status,
        argsRaw: "",
        argsState: argumentInfo ? "complete" : "pending",
        arguments: argumentInfo?.arguments,
        resultPreview: ""
      };
      const node: ChatNode = {
        id: nodeId("tool", toolCallId),
        kind: "tool",
        content: "",
        tool
      };
      const runs = {
        ...completedState.runs,
        [runId]: {
          ...(completedState.runs[runId] ?? {}),
          agentNodeId: undefined,
          thinkingNodeId: undefined
        }
      };
      return {
        ...appendNode(completedState, node),
        runs,
        toolNodeByCallId: { ...completedState.toolNodeByCallId, [toolCallId]: node.id }
      };
    }

    case "tool.call_delta":
      return updateTool(state, String(payload.tool_call_id ?? ""), (tool) => {
        const argsRaw = payload.is_streaming === false
          ? String(payload.delta ?? "")
          : tool.argsRaw + String(payload.delta ?? "");
        const parsed = parseToolArguments(argsRaw);
        const argumentInfo = parsed.ok ? splitToolArguments(parsed.value) : undefined;
        return {
          ...tool,
          argsRaw,
          argsState: argsRaw ? "streaming" : tool.argsState,
          description: argumentInfo?.description ?? tool.description,
          arguments: argumentInfo ? argumentInfo.arguments : tool.arguments
        };
      });

    case "tool.call_stop":
      return updateTool(state, String(payload.tool_call_id ?? ""), (tool) => {
        const hasArguments = Object.prototype.hasOwnProperty.call(payload, "arguments");
        const argumentInfo = hasArguments ? splitToolArguments(payload.arguments) : undefined;
        const description = optionalString(payload.tool_call_description)
          ?? argumentInfo?.description
          ?? tool.description;
        return {
          ...tool,
          name: String(payload.tool_name ?? tool.name),
          description,
          argsState: "complete",
          arguments: argumentInfo ? argumentInfo.arguments : tool.arguments,
          argsRaw: argumentInfo ? JSON.stringify(argumentInfo.arguments, null, 2) : tool.argsRaw
        };
      });

    case "tool.result":
      return updateToolResult(state, String(payload.tool_call_id ?? ""), payload, (tool) => {
        const text = formatToolResultText(payload);
        return {
          ...tool,
          status: payload.success === false ? "fail" : "success",
          name: String(payload.tool_name || tool.name),
          description: optionalString(payload.tool_call_description) ?? tool.description,
          resultPreview: payload.is_part === true
            ? truncate(tool.resultPreview + text)
            : truncate(text || tool.resultPreview),
          durationMs: Number(payload.duration_ms ?? tool.durationMs ?? 0)
        };
      });

    case "model.metadata": {
      const latency = payload.latency_ms == null ? "n/a" : `${Number(payload.latency_ms).toFixed(0)}ms`;
      return addMeta(
        state,
        `模型统计 in=${Number(payload.input_tokens ?? 0)} out=${Number(payload.output_tokens ?? 0)} total=${Number(payload.total_tokens ?? 0)} latency=${latency}`
      );
    }

    case "model.retry":
      return addSystem(
        state,
        `模型重试 ${String(payload.attempt ?? "")}/${String(payload.max_retries ?? "")}: [${String(payload.error_type ?? "")}] ${String(payload.error_message ?? "")}`
      );

    case "scheduler.interrupt":
      return addSystem(state, `执行被中断: ${String(payload.reason ?? "")}`);

    case "agent.interrupt":
      return addSystem(state, `Agent 中断: ${String(payload.interrupt_type ?? "")}`);

    case "debug.info":
      return {
        ...appendNode(state, {
          id: nodeId("debug", `${Date.now()}-${state.debugLines.length}`),
          kind: "debug",
          content: String(payload.message ?? "")
        }),
        debugLines: [
          ...state.debugLines.slice(-(MAX_DEBUG_LINES - 1)),
          String(payload.message ?? ""),
        ]
      };

    case "error":
      return {
        ...appendNode(state, {
          id: nodeId("error", `${Date.now()}-${state.errors.length}`),
          kind: "error",
          content: String(payload.message ?? "Unknown error")
        }),
        errors: [...state.errors, String(payload.message ?? "Unknown error")]
      };

    case "plugin.message":
    case "plugin.event":
      return addPluginMessage(state, payload, frame);

    case "plugin.status":
      return updatePluginStatus(state, payload, frame);

    case "plugin.tool_progress":
      return updateToolProgress(state, payload, frame);

    case "plugin.artifact.upsert":
      return upsertPluginArtifact(state, payload, frame);

    case "plugin.artifact.delta":
      return appendPluginArtifactDelta(state, payload, frame);

    case "plugin.artifact.remove":
      return removePluginArtifact(state, payload);

    case "plugin.artifact.clear":
      return clearPluginArtifacts(state, payload);

    default:
      return state;
  }
}

function updateStatus(state: AppState, payload: Record<string, unknown>): AppState {
  const schedulerState = String(payload.scheduler_state ?? state.schedulerState);
  const agentState = String(payload.agent_state ?? state.agentState);
  const queueLengths = normalizeQueueLengths(payload.queue_lengths, state.queueLengths);
  if (
    schedulerState === state.schedulerState
    && agentState === state.agentState
    && sameQueueLengths(queueLengths, state.queueLengths)
  ) {
    return state;
  }
  return {
    ...state,
    schedulerState,
    agentState,
    queueLengths
  };
}

function appendRunDelta(state: AppState, runId: string, kind: "agent" | "thinking", delta: string): AppState {
  const run = state.runs[runId] ?? {};
  const key = kind === "agent" ? "agentNodeId" : "thinkingNodeId";
  const existingId = run[key];
  if (existingId) {
    return updateNode(state, existingId, (node) => ({ ...node, content: node.content + delta, complete: false }));
  }
  const id = nodeId(kind, `${runId}-${state.nodes.length}`);
  const next = appendNode(state, { id, kind, content: delta, complete: false });
  return {
    ...next,
    runs: {
      ...next.runs,
      [runId]: { ...(next.runs[runId] ?? {}), [key]: id }
    }
  };
}

function completeThinkingForRun(state: AppState, runId: string): AppState {
  const thinkingNodeId = state.runs[runId]?.thinkingNodeId;
  if (!thinkingNodeId) {
    return state;
  }
  const updated = updateNode(state, thinkingNodeId, (node) => (
    node.kind === "thinking" ? { ...node, complete: true } : node
  ));
  return {
    ...updated,
    runs: {
      ...updated.runs,
      [runId]: {
        ...(updated.runs[runId] ?? {}),
        thinkingNodeId: undefined
      }
    }
  };
}

function updateTool(state: AppState, toolCallId: string, updater: (tool: ToolState) => ToolState): AppState {
  const nodeIdForTool = state.toolNodeByCallId[toolCallId];
  if (!nodeIdForTool) {
    return state;
  }
  return updateNode(state, nodeIdForTool, (node) => {
    if (!node.tool) return node;
    return { ...node, tool: updater(node.tool) };
  });
}

function updateToolResult(
  state: AppState,
  toolCallId: string,
  payload: Record<string, unknown>,
  updater: (tool: ToolState) => ToolState,
): AppState {
  const nodeIdForTool = state.toolNodeByCallId[toolCallId];
  if (nodeIdForTool) {
    const updated = updateNode(state, nodeIdForTool, (node) => {
      if (!node.tool) return node;
      return { ...node, tool: updater({ ...node.tool, toolCallId: toolCallId || node.tool.toolCallId }) };
    });
    return toolCallId
      ? { ...updated, toolNodeByCallId: { ...updated.toolNodeByCallId, [toolCallId]: nodeIdForTool } }
      : updated;
  }

  const nodeIdForNewTool = nodeId("tool", toolCallId || `${Date.now()}-${state.nodes.length}`);
  const runId = String(payload.run_id ?? state.activeRunId ?? "");
  const tool: ToolState = {
    runId,
    toolCallId,
    name: String(payload.tool_name || "unknown"),
    description: optionalString(payload.tool_call_description),
    status: "running",
    argsRaw: "",
    argsState: "complete",
    resultPreview: ""
  };
  const next = appendNode(state, {
    id: nodeIdForNewTool,
    kind: "tool",
    content: "",
    tool: updater(tool)
  });
  return toolCallId
    ? { ...next, toolNodeByCallId: { ...next.toolNodeByCallId, [toolCallId]: nodeIdForNewTool } }
    : next;
}

function updateNode(state: AppState, id: string, updater: (node: ChatNode) => ChatNode): AppState {
  return {
    ...state,
    nodes: state.nodes.map((node) => (node.id === id ? updater(node) : node))
  };
}

function appendNode(state: AppState, node: ChatNode): AppState {
  return {
    ...state,
    nodes: [...state.nodes, node]
  };
}

function addSystem(state: AppState, content: string): AppState {
  return appendNode(state, {
    id: nodeId("system", `${Date.now()}-${state.nodes.length}`),
    kind: "system",
    content
  });
}

function addMeta(state: AppState, content: string): AppState {
  return {
    ...appendNode(state, {
      id: nodeId("meta", `${Date.now()}-${state.metadataLines.length}`),
      kind: "meta",
      content
    }),
    metadataLines: [...state.metadataLines, content]
  };
}

function addPluginMessage(state: AppState, payload: Record<string, unknown>, frame: CoreFrame): AppState {
  const plugin = pluginInfo(payload);
  const message = optionalString(payload.message)
    ?? optionalString(payload.title)
    ?? formatToolValue(payload.data).trim()
    ?? "";
  if (!message) {
    return state;
  }
  const level = normalizePluginLevel(payload.level);
  const item: PluginMessageState = {
    id: String(payload.message_id ?? `${frame.type}-${frame.ts ?? Date.now()}-${state.pluginMessages.length}`),
    pluginId: plugin.pluginId,
    pluginName: plugin.pluginName,
    level,
    title: optionalString(payload.title),
    message,
    data: payload.data,
    timestamp: frameTime(frame)
  };
  return {
    ...state,
    pluginMessages: [...state.pluginMessages.slice(-79), item]
  };
}

function updatePluginStatus(state: AppState, payload: Record<string, unknown>, frame: CoreFrame): AppState {
  const plugin = pluginInfo(payload);
  const status = optionalString(payload.status) ?? "active";
  return {
    ...state,
    pluginStatuses: {
      ...state.pluginStatuses,
      [plugin.pluginId]: {
        pluginId: plugin.pluginId,
        pluginName: plugin.pluginName,
        status,
        label: optionalString(payload.label),
        message: optionalString(payload.message),
        progress: normalizeProgress(payload.progress),
        data: payload.data,
        updatedAt: frameTime(frame)
      }
    }
  };
}

function updateToolProgress(state: AppState, payload: Record<string, unknown>, frame: CoreFrame): AppState {
  const plugin = pluginInfo(payload);
  const toolCallId = String(payload.tool_call_id ?? "");
  const progress: ToolProgressState = {
    pluginId: plugin.pluginId,
    pluginName: plugin.pluginName,
    progress: normalizeProgress(payload.progress),
    status: optionalString(payload.status),
    label: optionalString(payload.label),
    message: optionalString(payload.message),
    data: payload.data,
    updatedAt: frameTime(frame)
  };
  const progressKey = toolCallId || `${plugin.pluginId}:latest`;
  const next = {
    ...state,
    toolProgress: {
      ...state.toolProgress,
      [progressKey]: progress
    }
  };
  if (!toolCallId) {
    return next;
  }
  return updateTool(next, toolCallId, (tool) => ({
    ...tool,
    progress
  }));
}

function upsertPluginArtifact(state: AppState, payload: Record<string, unknown>, frame: CoreFrame): AppState {
  const artifactPayload = getArtifactPayload(payload);
  if (!artifactPayload) {
    return state;
  }
  const plugin = pluginInfo(payload);
  const artifactId = String(artifactPayload.id ?? artifactPayload.artifact_id ?? "");
  if (!artifactId) {
    return state;
  }
  const key = artifactKey(plugin.pluginId, artifactId);
  const existing = state.artifacts[key];
  const nextArtifact: PluginArtifactState = {
    ...existing,
    key,
    id: artifactId,
    pluginId: plugin.pluginId,
    pluginName: plugin.pluginName,
    artifactType: String(artifactPayload.type ?? artifactPayload.artifact_type ?? existing?.artifactType ?? "artifact"),
    title: String(artifactPayload.title ?? existing?.title ?? artifactId),
    content: stringOrExisting(artifactPayload.content, existing?.content),
    data: artifactPayload.data ?? existing?.data,
    mimeType: optionalString(artifactPayload.mime_type ?? artifactPayload.mimeType) ?? existing?.mimeType,
    language: optionalString(artifactPayload.language) ?? existing?.language,
    uri: optionalString(artifactPayload.uri) ?? existing?.uri,
    path: optionalString(artifactPayload.path) ?? existing?.path,
    description: optionalString(artifactPayload.description) ?? existing?.description,
    status: optionalString(artifactPayload.status) ?? existing?.status,
    metadata: isRecord(artifactPayload.metadata) ? artifactPayload.metadata : existing?.metadata,
    updatedAt: frameTime(frame)
  };
  const artifactOrder = existing ? state.artifactOrder : [...state.artifactOrder, key];
  return {
    ...state,
    artifacts: {
      ...state.artifacts,
      [key]: nextArtifact
    },
    artifactOrder,
    selectedArtifactId: state.selectedArtifactId ?? key
  };
}

function appendPluginArtifactDelta(state: AppState, payload: Record<string, unknown>, frame: CoreFrame): AppState {
  const plugin = pluginInfo(payload);
  const artifactId = String(payload.artifact_id ?? payload.id ?? "");
  if (!artifactId) {
    return state;
  }
  const key = artifactKey(plugin.pluginId, artifactId);
  const existing = state.artifacts[key] ?? {
    key,
    id: artifactId,
    pluginId: plugin.pluginId,
    pluginName: plugin.pluginName,
    artifactType: "artifact",
    title: artifactId,
    updatedAt: frameTime(frame)
  };
  const field = String(payload.field ?? "content");
  const delta = String(payload.delta ?? "");
  const nextArtifact: PluginArtifactState = {
    ...existing,
    updatedAt: frameTime(frame)
  };
  if (field === "content") {
    nextArtifact.content = `${existing.content ?? ""}${delta}`;
  } else if (field === "status") {
    nextArtifact.status = `${existing.status ?? ""}${delta}`;
  } else if (field === "description") {
    nextArtifact.description = `${existing.description ?? ""}${delta}`;
  } else {
    nextArtifact.metadata = {
      ...(existing.metadata ?? {}),
      [field]: `${formatToolValue((existing.metadata ?? {})[field])}${delta}`
    };
  }
  const artifactOrder = state.artifacts[key] ? state.artifactOrder : [...state.artifactOrder, key];
  return {
    ...state,
    artifacts: {
      ...state.artifacts,
      [key]: nextArtifact
    },
    artifactOrder,
    selectedArtifactId: state.selectedArtifactId ?? key
  };
}

function removePluginArtifact(state: AppState, payload: Record<string, unknown>): AppState {
  const plugin = pluginInfo(payload);
  const artifactId = String(payload.artifact_id ?? payload.id ?? "");
  if (!artifactId) {
    return state;
  }
  const key = artifactKey(plugin.pluginId, artifactId);
  if (!state.artifacts[key]) {
    return state;
  }
  const artifacts = { ...state.artifacts };
  delete artifacts[key];
  const artifactOrder = state.artifactOrder.filter((item) => item !== key);
  return {
    ...state,
    artifacts,
    artifactOrder,
    selectedArtifactId: nextSelectedArtifact(state.selectedArtifactId, key, artifactOrder)
  };
}

function clearPluginArtifacts(state: AppState, payload: Record<string, unknown>): AppState {
  const scope = String(payload.scope ?? "plugin");
  if (scope === "all") {
    return {
      ...state,
      artifacts: {},
      artifactOrder: [],
      selectedArtifactId: undefined
    };
  }
  const plugin = pluginInfo(payload);
  const artifacts = { ...state.artifacts };
  const artifactOrder = state.artifactOrder.filter((key) => {
    if (artifacts[key]?.pluginId !== plugin.pluginId) {
      return true;
    }
    delete artifacts[key];
    return false;
  });
  return {
    ...state,
    artifacts,
    artifactOrder,
    selectedArtifactId: artifactOrder.includes(state.selectedArtifactId ?? "")
      ? state.selectedArtifactId
      : artifactOrder[0]
  };
}

function pluginInfo(payload: Record<string, unknown>): { pluginId: string; pluginName: string } {
  const pluginId = optionalString(payload.plugin_id) ?? "plugin";
  return {
    pluginId,
    pluginName: optionalString(payload.plugin_name) ?? pluginId
  };
}

function artifactKey(pluginId: string, artifactId: string): string {
  return `${pluginId}:${artifactId}`;
}

function getArtifactPayload(payload: Record<string, unknown>): PluginArtifactPayload | undefined {
  if (isRecord(payload.artifact)) {
    return payload.artifact;
  }
  return payload as PluginArtifactPayload;
}

function nextSelectedArtifact(current: string | undefined, removed: string, order: string[]): string | undefined {
  if (current && current !== removed) {
    return current;
  }
  return order[0];
}

function stringOrExisting(value: unknown, existing: string | undefined): string | undefined {
  return typeof value === "string" ? value : existing;
}

function frameTime(frame: CoreFrame): number {
  const seconds = Number(frame.ts);
  return Number.isFinite(seconds) ? seconds * 1000 : Date.now();
}

function normalizePluginLevel(value: unknown): PluginMessageState["level"] {
  return value === "debug" || value === "warning" || value === "error" ? value : "info";
}

function normalizeProgress(value: unknown): number | undefined {
  const numberValue = Number(value);
  if (!Number.isFinite(numberValue)) {
    return undefined;
  }
  const normalized = numberValue > 1 && numberValue <= 100 ? numberValue / 100 : numberValue;
  return Math.min(1, Math.max(0, normalized));
}

function normalizeToolStatus(value: unknown, fallback: ToolState["status"]): ToolState["status"] {
  return value === "pending" || value === "running" || value === "success" || value === "fail"
    ? value
    : fallback;
}

function nodeId(kind: string, id: string): string {
  return `${kind}-${id}`;
}

function normalizeQueue(value: unknown): QueueKind {
  return value === "normal" || value === "urgent" ? value : "high_prio";
}

function normalizeQueueLengths(value: unknown, fallback: Record<QueueKind, number> = { normal: 0, high_prio: 0, urgent: 0 }): Record<QueueKind, number> {
  const raw = value && typeof value === "object" ? (value as Record<string, unknown>) : {};
  return {
    normal: Number(raw.normal ?? fallback.normal),
    high_prio: Number(raw.high_prio ?? fallback.high_prio),
    urgent: Number(raw.urgent ?? fallback.urgent)
  };
}

function sameQueueLengths(left: Record<QueueKind, number>, right: Record<QueueKind, number>): boolean {
  return left.normal === right.normal && left.high_prio === right.high_prio && left.urgent === right.urgent;
}

function formatRunStop(payload: Record<string, unknown>): string {
  const reason = String(payload.stop_reason ?? "end_turn");
  const durationMs = Number(payload.duration_ms ?? 0);
  if (!Number.isFinite(durationMs) || durationMs <= 0) {
    return reason;
  }
  return `${reason} · ${(durationMs / 1000).toFixed(1)}s`;
}

function truncate(value: string, max = MAX_RESULT_PREVIEW_LENGTH): string {
  if (value.length <= max) return value;
  return `${value.slice(0, max)}...`;
}

function formatToolResultText(payload: Record<string, unknown>): string {
  if (payload.is_part === true) {
    return formatToolValue(payload.part);
  }
  const output = formatToolValue(payload.output).trim();
  if (payload.success !== false) {
    return output || formatToolValue(payload.part).trim();
  }
  const error = formatToolValue(payload.error).trim();
  if (error && output && output !== error) {
    return `Error: ${error}\n\nOutput:\n${output}`;
  }
  if (error) {
    return `Error: ${error}`;
  }
  return output;
}

function formatToolValue(value: unknown): string {
  if (value == null) return "";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function splitToolArguments(value: unknown): { arguments: unknown; description?: string } {
  if (!isRecord(value) || !(TOOL_CALL_DESCRIPTION_PARAMETER in value)) {
    return { arguments: value };
  }
  const description = optionalString(value[TOOL_CALL_DESCRIPTION_PARAMETER]);
  const visible = { ...value };
  delete visible[TOOL_CALL_DESCRIPTION_PARAMETER];
  return { arguments: visible, description };
}

function optionalString(value: unknown): string | undefined {
  if (value == null) return undefined;
  const text = String(value).trim();
  return text ? text : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function parseToolArguments(value: string): { ok: true; value: unknown } | { ok: false } {
  const trimmed = value.trim();
  if (!trimmed) return { ok: false };
  const parsed = tryParseJson(trimmed);
  if (parsed.ok) return parsed;

  const closedCandidate = closePartialTopLevelObject(trimmed);
  if (closedCandidate) {
    const closed = tryParseJson(closedCandidate);
    if (closed.ok) return closed;
  }

  const commaIndex = lastTopLevelComma(trimmed);
  if (commaIndex > 0) {
    const prefix = `${trimmed.slice(0, commaIndex)}}`;
    const prefixParsed = tryParseJson(prefix);
    if (prefixParsed.ok) return prefixParsed;
  }

  return { ok: false };
}

function tryParseJson(value: string): { ok: true; value: unknown } | { ok: false } {
  try {
    return { ok: true, value: JSON.parse(value) as unknown };
  } catch {
    return { ok: false };
  }
}

function closePartialTopLevelObject(value: string): string | undefined {
  if (!value.startsWith("{") || value.endsWith("}")) return undefined;
  if (endsInsideString(value) || value.trimEnd().endsWith(",")) return undefined;
  return `${value}}`;
}

function lastTopLevelComma(value: string): number {
  let depth = 0;
  let inString = false;
  let escaped = false;
  let lastComma = -1;

  for (let index = 0; index < value.length; index += 1) {
    const char = value[index];
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (char === "\\") {
        escaped = true;
      } else if (char === "\"") {
        inString = false;
      }
      continue;
    }
    if (char === "\"") {
      inString = true;
    } else if (char === "{" || char === "[") {
      depth += 1;
    } else if (char === "}" || char === "]") {
      depth = Math.max(0, depth - 1);
    } else if (char === "," && depth === 1) {
      lastComma = index;
    }
  }

  return lastComma;
}

function endsInsideString(value: string): boolean {
  let inString = false;
  let escaped = false;
  for (const char of value) {
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (char === "\\") {
        escaped = true;
      } else if (char === "\"") {
        inString = false;
      }
      continue;
    }
    if (char === "\"") {
      inString = true;
    }
  }
  return inString;
}
