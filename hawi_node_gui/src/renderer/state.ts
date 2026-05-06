import type { CoreFrame, QueueKind } from "../shared/protocol";

export type ChatKind = "user" | "agent" | "thinking" | "tool" | "system" | "meta" | "error" | "debug" | "divider";

export interface ChatNode {
  id: string;
  kind: ChatKind;
  content: string;
  queue?: QueueKind;
  tool?: ToolState;
}

export interface ToolState {
  runId: string;
  toolCallId: string;
  name: string;
  status: "running" | "success" | "fail";
  argsRaw: string;
  arguments?: unknown;
  resultPreview: string;
  durationMs?: number;
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
    errors: []
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
        errors: []
      };

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
      return appendRunDelta(state, runId, "agent", delta);
    }

    case "run.thinking_delta": {
      const runId = String(payload.run_id ?? state.activeRunId ?? "");
      if (!runId) return state;
      return appendRunDelta(state, runId, "thinking", String(payload.delta ?? ""));
    }

    case "run.stop": {
      const runId = String(payload.run_id ?? "");
      const nextRuns = { ...state.runs };
      delete nextRuns[runId];
      return {
        ...appendNode(state, {
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
      const toolCallId = String(payload.tool_call_id ?? "");
      const tool: ToolState = {
        runId,
        toolCallId,
        name: String(payload.tool_name ?? "pending"),
        status: "running",
        argsRaw: "",
        resultPreview: ""
      };
      const node: ChatNode = {
        id: nodeId("tool", toolCallId),
        kind: "tool",
        content: "",
        tool
      };
      const runs = {
        ...state.runs,
        [runId]: { ...(state.runs[runId] ?? {}), agentNodeId: undefined }
      };
      return {
        ...appendNode(state, node),
        runs,
        toolNodeByCallId: { ...state.toolNodeByCallId, [toolCallId]: node.id }
      };
    }

    case "tool.call_delta":
      return updateTool(state, String(payload.tool_call_id ?? ""), (tool) => ({
        ...tool,
        argsRaw: payload.is_streaming === false
          ? String(payload.delta ?? "")
          : tool.argsRaw + String(payload.delta ?? "")
      }));

    case "tool.call_stop":
      return updateTool(state, String(payload.tool_call_id ?? ""), (tool) => ({
        ...tool,
        name: String(payload.tool_name ?? tool.name),
        arguments: payload.arguments,
        argsRaw: JSON.stringify(payload.arguments ?? {}, null, 2)
      }));

    case "tool.result":
      return updateTool(state, String(payload.tool_call_id ?? ""), (tool) => {
        const text = String((payload.success === false ? payload.error : undefined) ?? payload.output ?? payload.part ?? "");
        return {
          ...tool,
          status: payload.success === false ? "fail" : "success",
          name: String(payload.tool_name || tool.name),
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
        debugLines: [...state.debugLines.slice(-199), String(payload.message ?? "")]
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
    return updateNode(state, existingId, (node) => ({ ...node, content: node.content + delta }));
  }
  const id = nodeId(kind, `${runId}-${state.nodes.length}`);
  const next = appendNode(state, { id, kind, content: delta });
  return {
    ...next,
    runs: {
      ...next.runs,
      [runId]: { ...(next.runs[runId] ?? {}), [key]: id }
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

function truncate(value: string, max = 1200): string {
  if (value.length <= max) return value;
  return `${value.slice(0, max)}...`;
}
