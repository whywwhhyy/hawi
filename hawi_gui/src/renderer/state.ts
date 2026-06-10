import { VERSION, type ContentPart, type CoreFrame, type PluginArtifactPayload, type QueueKind, type RuntimeControlState } from "../shared/protocol";

const TOOL_CALL_PURPOSE_PARAMETER = "tool_call_purpose";
const MAX_DEBUG_LINES = 200;

type ViteImportMeta = ImportMeta & {
  env?: {
    MODE?: string;
    PROD?: boolean;
  };
};

type HawiTestGlobal = typeof globalThis & {
  __HAWI_TEST_RELEASE_DEDUP_FALLBACK__?: boolean;
};

function releaseDedupFallbackEnabled(): boolean {
  const env = (import.meta as ViteImportMeta).env;
  if (env?.PROD === true) {
    return true;
  }
  return env?.MODE === "test"
    && (globalThis as HawiTestGlobal).__HAWI_TEST_RELEASE_DEDUP_FALLBACK__ === true;
}

export type ChatKind = "user" | "agent" | "thinking" | "tool" | "system" | "meta" | "error" | "debug" | "divider" | "compact" | "framework" | "handoff";
export type DisplayMessageType = "normal" | "steer" | "urgent" | "resume";
export type FrameworkInjectionKind =
  | "system_prompt"
  | "context_injected"
  | "tool_runtime_context_injected"
  | "plugin_message";

export interface FrameworkInjectionState {
  id: string;
  kind: FrameworkInjectionKind;
  label: string;
  content: string;
  runId?: string;
  pluginId?: string;
  pluginName?: string;
  pluginRole?: string;
  injectionName?: string;
  eventType?: string;
  hookType?: string;
  role?: string;
  mergeTarget?: string;
  mergePosition?: "before" | "after";
  targetMessageId?: string;
  targetContextMessageId?: string;
  targetMessageIndex?: number;
  contextPosition?: number;
  toolName?: string;
  toolCallId?: string;
  parameterName?: string;
  metadata?: Record<string, unknown>;
  timestamp: number;
}

export interface ChatNode {
  id: string;
  kind: ChatKind;
  content: string;
  contentParts?: ContentPart[];
  complete?: boolean;
  contextMessageId?: string;
  contextMessageIndex?: number;
  historyIndex?: number;
  canFork?: boolean;
  streamStartedAt?: number;
  streamFinishedAt?: number;
  streamDurationMs?: number;
  profile?: ModelProfileState;
  queue?: QueueKind;
  displayMessageType?: DisplayMessageType;
  tool?: ToolState;
  framework?: FrameworkInjectionState;
  injections?: FrameworkInjectionState[];
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
  /** 原始 tool result output 数据，可能是文本或结构化对象（仅保持最近的完整结果） */
  resultData?: unknown;
  contextMessageId?: string;
  contextMessageIndex?: number;
  streamStartedAt?: number;
  streamFinishedAt?: number;
  streamDurationMs?: number;
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

export interface ContextUsageState {
  usedTokens: number;
  maxContextTokens?: number;
  ratio?: number;
  source?: "estimate" | "provider_usage";
}

export interface ModelProfileState {
  cacheTokens?: number;
  prefillTokens?: number;
  prefillTotalTokens?: number;
  prefillMs?: number;
  prefillTokensPerSecond?: number;
  ttftMs?: number;
  decodeTokens?: number;
  decodeMs?: number;
  decodeTokensPerSecond?: number;
  peakDecodeTokensPerSecond?: number;
}

export interface ModelUsageState {
  totalTokens: number;
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
  cacheWriteTokens: number;
}

export interface ContextCompressionState {
  active: boolean;
  nodeId?: string;
  mode?: string;
  status?: string;
  tokensBefore?: number;
  tokensAfter?: number;
  messageCountBefore?: number;
  messageCountAfter?: number;
  startedAt?: number;
  updatedAt?: number;
}

export interface ContextAutoCompactState {
  enabled: boolean;
  maxContextTokens?: number;
  triggerTokens?: number;
  triggerRatio?: number;
  maxTriggerRatio?: number;
  compressionBudget?: number;
  tokenLimit?: number;
  tokenLimitRatio?: number;
}

export interface QueueMessageState {
  id: string;
  queue: QueueKind;
  contentPreview: string;
  content?: string;
  contentParts?: ContentPart[];
  createdAt?: number;
  metadata?: Record<string, unknown>;
}

export interface ProcessingState {
  id: string;
  runId: string;
  content: string;
}

export interface SubAgentStatusState {
  id: string;
  name: string;
  role: string;
  state: string;
  runnerState: string;
  executorState: string;
  queueLengths: Record<string, number>;
  createdAt?: number;
  updatedAt?: number;
  closedAt?: number;
  modelId?: string;
  workingDir?: string;
  mode?: string;
  sharedContext?: boolean;
  plugins: SubAgentPluginInfoState[];
  pluginIds: string[];
  toolNames: string[];
  toolCount: number;
  lastResultText?: string;
  lastError?: string;
}

export interface SubAgentPluginInfoState {
  id: string;
  name: string;
  displayName?: string;
  className?: string;
}

interface SubAgentPartialState {
  runId?: string;
  text: string;
  reasoning: string;
  startedAt?: number;
  updatedAt?: number;
}

export interface SubAgentRuntimeState {
  id: string;
  name: string;
  role: string;
  state: string;
  createdAt?: number;
  mode?: string;
  sharedContext?: boolean;
  status?: SubAgentStatusState;
  messageHistory: SessionHistoryRecord[];
  nodes: ChatNode[];
  processing?: ProcessingState;
  partial: SubAgentPartialState;
  eventCount: number;
  lastEventType?: string;
  lastEventAt?: number;
}

interface RunState {
  userNodeId?: string;
  agentNodeId?: string;
  thinkingNodeId?: string;
  processingId?: string;
  startedAt?: number;
  assistantCommitNodeId?: string;
  assistantMessageCounted?: boolean;
  assistantContextMessageIndex?: number;
  toolCallAssistantContextIndexed?: boolean;
  profile?: ModelProfileState;
}

export interface AppState {
  nodes: ChatNode[];
  runs: Record<string, RunState>;
  toolNodeByCallId: Record<string, string>;
  activeRunId?: string;
  runnerState: string;
  agentState: string;
  queueLengths: Record<QueueKind, number>;
  queueMessages: Record<QueueKind, QueueMessageState[]>;
  control: RuntimeControlState;
  metadataLines: string[];
  modelUsage?: ModelUsageState;
  contextUsage?: ContextUsageState;
  contextCompression?: ContextCompressionState;
  contextAutoCompact?: ContextAutoCompactState;
  debugLines: string[];
  errors: string[];
  artifacts: Record<string, PluginArtifactState>;
  artifactOrder: string[];
  selectedArtifactId?: string;
  pluginMessages: PluginMessageState[];
  pluginStatuses: Record<string, PluginStatusState>;
  toolProgress: Record<string, ToolProgressState>;
  subagents: Record<string, SubAgentRuntimeState>;
  subagentOrder: string[];
  sessionMessageCount: number;
  nextContextMessageIndex: number;
  processing?: ProcessingState;
}

export function createInitialState(): AppState {
  return {
    nodes: [],
    runs: {},
    toolNodeByCallId: {},
    runnerState: "IDLE",
    agentState: "IDLE",
    queueLengths: { normal: 0, high_prio: 0, urgent: 0 },
    queueMessages: { normal: [], high_prio: [], urgent: [] },
    control: { paused: false, resumable: false, pause_reason: null, paused_at: null, last_error_message: null },
    metadataLines: [],
    modelUsage: undefined,
    contextUsage: undefined,
    contextCompression: undefined,
    contextAutoCompact: undefined,
    debugLines: [],
    errors: [],
    artifacts: {},
    artifactOrder: [],
    pluginMessages: [],
    pluginStatuses: {},
    toolProgress: {},
    subagents: {},
    subagentOrder: [],
    sessionMessageCount: 0,
    nextContextMessageIndex: 0,
    processing: undefined
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
        modelUsage: undefined,
        contextUsage: undefined,
        contextCompression: undefined,
        contextAutoCompact: undefined,
        debugLines: [],
        errors: [],
        artifacts: {},
        artifactOrder: [],
        selectedArtifactId: undefined,
        pluginMessages: [],
        pluginStatuses: {},
        toolProgress: {},
        subagents: {},
        subagentOrder: [],
        sessionMessageCount: 0,
        nextContextMessageIndex: 0,
        processing: undefined
      };

    case "gui.load_session_history": {
      const history = normalizeSessionHistory(payload.message_history);
      const contextUsage = parseStatusContextUsage(payload.context_usage);
      const replayState = sessionHistoryReplayState(history);
      return {
        ...state,
        nodes: attachHistoryToolProgress(
          sessionHistoryNodes(history),
          replayState.toolProgress
        ),
        runs: {},
        toolNodeByCallId: {},
        activeRunId: undefined,
        metadataLines: [],
        modelUsage: replayState.modelUsage,
        contextUsage: contextUsage ?? replayState.contextUsage,
        contextCompression: undefined,
        contextAutoCompact: parseStatusAutoCompact(payload.auto_compact),
        debugLines: [],
        errors: [],
        artifacts: replayState.artifacts,
        artifactOrder: replayState.artifactOrder,
        selectedArtifactId: replayState.selectedArtifactId,
        pluginMessages: replayState.pluginMessages,
        pluginStatuses: replayState.pluginStatuses,
        toolProgress: replayState.toolProgress,
        subagents: replayState.subagents,
        subagentOrder: replayState.subagentOrder,
        sessionMessageCount: history.length,
        nextContextMessageIndex: nextContextMessageIndexFromHistory(history),
        processing: undefined
      };
    }

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
      const eventAt = frameTime(frame);
      const queue = normalizeQueue(payload.queue);
      const displayMessageType = normalizeDisplayMessageType(
        payload.display_message_type,
        queue
      );
      const messageId = optionalString(payload.message_id);
      const contextMessageId = optionalString(payload.context_message_id);
      const userContent = String(payload.user_content ?? "");
      const contentParts = normalizeContentParts(payload.content);
      const userNodeId = messageId ? userMessageNodeId(messageId) : nodeId("user", runId);
      const existingUser = releaseDedupFallbackEnabled()
        ? findRunStartUserNode(
            state,
            userNodeId,
            contextMessageId
          )
        : undefined;
      const shouldHydrateContextIndex = existingUser !== undefined
        && existingUser.contextMessageIndex === undefined;
      const contextMessageIndex = existingUser?.contextMessageIndex
        ?? state.nextContextMessageIndex;
      const userNode: ChatNode = {
        id: userNodeId,
        kind: "user",
        queue,
        displayMessageType,
        contextMessageId,
        contextMessageIndex,
        canFork: true,
        content: userContent,
        contentParts
      };
      const withUser = existingUser
        ? mergeRunStartUserNode(state, existingUser.id, userNode, contextMessageIndex)
        : appendChatNode(state, userNode);
      const processing: ProcessingState = {
        id: nodeId("processing", `${runId}-${withUser.nodes.length}`),
        runId,
        content: "处理中...",
      };
      return {
        ...withUser,
        sessionMessageCount: existingUser
          ? state.sessionMessageCount
          : state.sessionMessageCount + 1,
        nextContextMessageIndex: existingUser
          ? shouldHydrateContextIndex
            ? state.nextContextMessageIndex + 1
            : state.nextContextMessageIndex
          : contextMessageIndex + 1,
        processing,
        activeRunId: runId,
        runs: {
          ...withUser.runs,
          [runId]: {
            ...(withUser.runs[runId] ?? {}),
            userNodeId,
            startedAt: withUser.runs[runId]?.startedAt ?? eventAt,
            processingId: processing.id
          }
        }
      };
    }

    case "run.message_committed": {
      const runId = String(payload.run_id ?? state.activeRunId ?? "");
      const contextMessageId = optionalString(payload.context_message_id);
      const role = optionalString(payload.role);
      if (!runId || !contextMessageId || role !== "assistant") {
        return state;
      }
      const contentParts = visibleMessageContentParts(payload.content);
      const contentText = Array.isArray(payload.content)
        ? historyAssistantText(payload.content)
        : "";
      const run = state.runs[runId] ?? {};
      const agentNodeId = run.agentNodeId;
      if (!agentNodeId) {
        if (run.assistantCommitNodeId) {
          return commitAssistantNode(
            state,
            runId,
            run.assistantCommitNodeId,
            contextMessageId,
            contentText,
            contentParts
          );
        }
        if (!contentParts) {
          return state;
        }
        return appendCommittedAssistantNode(
          state,
          runId,
          contextMessageId,
          contentText,
          contentParts,
          frameTime(frame),
        );
      }
      return commitAssistantNode(
        state,
        runId,
        agentNodeId,
        contextMessageId,
        contentText,
        contentParts
      );
    }

    case "run.text_delta": {
      const runId = String(payload.run_id ?? state.activeRunId ?? "");
      if (!runId) return state;
      const delta = String(payload.delta ?? "");
      const eventAt = frameTime(frame);
      return appendRunDelta(
        completeThinkingForRun(state, runId, eventAt),
        runId,
        "agent",
        delta,
        eventAt
      );
    }

    case "run.thinking_delta": {
      const runId = String(payload.run_id ?? state.activeRunId ?? "");
      if (!runId) return state;
      return appendRunDelta(state, runId, "thinking", String(payload.delta ?? ""), frameTime(frame));
    }

    case "run.stop": {
      const runId = String(payload.run_id ?? "");
      const eventAt = frameTime(frame);
      const runStartedAt = state.runs[runId]?.startedAt;
      const runDurationMs = resolveRunDurationMs(payload, runStartedAt, eventAt);
      const completedState = completeActiveCompressionOnRunStop(
        completeOpenRunNodesForRun(state, runId, eventAt),
        payload,
        eventAt
      );
      const withDivider = appendChatNode(completedState, {
        id: nodeId("divider", `${runId}-${Date.now()}`),
        kind: "divider",
        content: formatRunStop(payload),
        ...(runDurationMs === undefined
          ? {}
          : {
              streamStartedAt: runStartedAt ?? eventAt - runDurationMs,
              streamFinishedAt: eventAt,
              streamDurationMs: runDurationMs
            })
      });
      const nextRuns = { ...withDivider.runs };
      delete nextRuns[runId];
      return {
        ...withDivider,
        runs: nextRuns,
        activeRunId: state.activeRunId === runId ? undefined : state.activeRunId
      };
    }

    case "tool.call_start": {
      const runId = String(payload.run_id ?? state.activeRunId ?? "");
      const eventAt = frameTime(frame);
      const completedState = completeOpenRunNodesForRun(state, runId, eventAt);
      const indexedState = indexToolCallAssistantContext(completedState, runId);
      const toolCallId = String(payload.tool_call_id ?? "");
      const status = normalizeToolStatus(payload.status, "running");
      const hasArguments = Object.prototype.hasOwnProperty.call(payload, "arguments");
      const argumentInfo = hasArguments ? splitToolArguments(payload.arguments) : undefined;
      const existingNodeId = indexedState.toolNodeByCallId[toolCallId];
      if (existingNodeId) {
        return updateChatNode(indexedState, existingNodeId, (node) => {
          if (!node.tool) return node;
          const description = optionalToolPurpose(payload)
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
              streamStartedAt: node.tool.streamStartedAt ?? (argumentInfo ? undefined : eventAt),
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
        description: optionalToolPurpose(payload),
        status,
        argsRaw: "",
        argsState: argumentInfo ? "complete" : "pending",
        arguments: argumentInfo?.arguments,
        resultPreview: "",
        streamStartedAt: argumentInfo ? undefined : eventAt
      };
      const node: ChatNode = {
        id: nodeId("tool", toolCallId),
        kind: "tool",
        content: "",
        tool
      };
      const withTool = appendChatNode(indexedState, node);
      const runs = {
        ...withTool.runs,
        [runId]: {
          ...(withTool.runs[runId] ?? {}),
          startedAt: withTool.runs[runId]?.startedAt ?? eventAt,
          agentNodeId: undefined,
          thinkingNodeId: undefined,
          assistantContextMessageIndex: undefined,
          toolCallAssistantContextIndexed: true
        }
      };
      return {
        ...withTool,
        runs,
        toolNodeByCallId: { ...withTool.toolNodeByCallId, [toolCallId]: node.id }
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
        const eventAt = frameTime(frame);
        const hasArguments = Object.prototype.hasOwnProperty.call(payload, "arguments");
        const argumentInfo = hasArguments ? splitToolArguments(payload.arguments) : undefined;
        const description = optionalToolPurpose(payload)
          ?? argumentInfo?.description
          ?? tool.description;
        return completeToolStream({
          ...tool,
          name: String(payload.tool_name ?? tool.name),
          description,
          argsState: "complete",
          arguments: argumentInfo ? argumentInfo.arguments : tool.arguments,
          argsRaw: argumentInfo ? JSON.stringify(argumentInfo.arguments, null, 2) : tool.argsRaw
        }, eventAt);
      });

    case "tool.interrupted":
      return updateTool(state, String(payload.tool_call_id ?? ""), (tool) => {
        const eventAt = frameTime(frame);
        const reason = String(payload.reason ?? "interrupted");
        const text = `Tool call interrupted before completion (reason: ${reason}).`;
        return completeToolStream({
          ...tool,
          status: "fail",
          name: String(payload.tool_name || tool.name),
          description: optionalToolPurpose(payload) ?? tool.description,
          argsState: "complete",
          resultPreview: tool.resultPreview || text,
          resultData: tool.resultData ?? text,
        }, eventAt);
      });

    case "tool.result": {
      const eventAt = frameTime(frame);
      const toolCallId = String(payload.tool_call_id ?? "");
      const contextMessageIndex = toolResultNeedsContextIndex(state, toolCallId, payload)
        ? state.nextContextMessageIndex
        : undefined;
      const updated = updateToolResult(state, toolCallId, payload, (tool) => {
        const text = formatToolResultText(payload);
        const isPart = payload.is_part === true;
        const nextTool = {
          ...tool,
          status: isPart ? tool.status : payload.success === false ? "fail" : "success",
          name: String(payload.tool_name || tool.name),
          description: optionalToolPurpose(payload) ?? tool.description,
          resultPreview: isPart
            ? tool.resultPreview + text
            : text || tool.resultPreview,
          resultData: isPart ? tool.resultData : payload.output,
          contextMessageId: tool.contextMessageId ?? optionalString(payload.context_message_id),
          contextMessageIndex: tool.contextMessageIndex ?? contextMessageIndex,
          durationMs: isPart ? tool.durationMs : Number(payload.duration_ms ?? tool.durationMs ?? 0)
        };
        return isPart ? nextTool : completeToolStream(nextTool, eventAt);
      });
      const withContextIndex = contextMessageIndex === undefined
        ? updated
        : { ...updated, nextContextMessageIndex: updated.nextContextMessageIndex + 1 };
      const runId = String(payload.run_id ?? state.activeRunId ?? "");
      if (
        payload.is_part !== true
        && runId
        && withContextIndex.activeRunId === runId
        && !hasActiveToolsForRun(withContextIndex, runId)
      ) {
        return ensureProcessingForRun(withContextIndex, runId, eventAt);
      }
      return withContextIndex;
    }

    case "model.metadata": {
      const nextState = {
        ...state,
        modelUsage: addModelUsage(state.modelUsage, parseModelUsage(payload)),
        contextUsage: parseContextUsage(payload) ?? state.contextUsage
      };
      return addMeta(attachModelProfileToRun(nextState, payload), formatModelMetadata(payload));
    }

    case "model.profile":
      return attachModelProfileToRun(state, payload);

    case "agent.system_prompt":
      return mergeSystemPromptInjection(
        state,
        frameworkInjectionFromFrame(frame, payload, "system_prompt")
      );

    case "agent.context_injected": {
      const injection = frameworkInjectionFromFrame(frame, payload, "context_injected");
      const withInjection = payload.merge_target === "user_message"
        ? attachFrameworkInjectionToUserMessage(state, payload, injection)
        : appendFrameworkInjection(state, injection);
      return noteContextInsertion(withInjection, payload);
    }

    case "agent.tool_runtime_context_injected":
      return appendFrameworkInjection(
        state,
        frameworkInjectionFromFrame(frame, payload, "tool_runtime_context_injected")
      );

    case "agent.compact_start": {
      const eventAt = frameTime(frame);
      const runId = String(payload.run_id ?? state.activeRunId ?? "manual");
      const existingNodeId = state.contextCompression?.active
        ? state.contextCompression.nodeId
        : undefined;
      const nodeIdForCompression = existingNodeId
        ?? nodeId("compact", `${runId}-${state.nodes.length}`);
      const base = existingNodeId
        ? updateChatNode(state, existingNodeId, (node) => ({
            ...node,
            kind: "compact",
            content: "Compressing context...",
            complete: false,
            streamStartedAt: node.streamStartedAt ?? eventAt,
            streamFinishedAt: undefined,
            streamDurationMs: undefined
          }))
        : appendChatNode(state, {
            id: nodeIdForCompression,
            kind: "compact",
            content: "Compressing context...",
            complete: false,
            streamStartedAt: eventAt
          });
      return {
        ...base,
        sessionMessageCount: existingNodeId
          ? base.sessionMessageCount
          : base.sessionMessageCount + 1,
        contextCompression: {
          active: true,
          nodeId: nodeIdForCompression,
          mode: optionalString(payload.mode),
          tokensBefore: optionalNumber(payload.tokens_before),
          messageCountBefore: optionalNumber(payload.message_count_before),
          startedAt: eventAt,
          updatedAt: eventAt
        }
      };
    }

    case "agent.compact_stop": {
      const eventAt = frameTime(frame);
      const nodeIdForCompression = state.contextCompression?.nodeId;
      const content = formatContextCompactionStop(payload);
      const base = nodeIdForCompression
        ? updateChatNode(state, nodeIdForCompression, (node) => (
            node.kind === "compact"
              ? completeChatNodeStream({
                  ...node,
                  content,
                  complete: true
                }, eventAt)
              : node
          ))
        : appendChatNode(state, {
            id: nodeId("compact", `${String(payload.run_id ?? state.activeRunId ?? "manual")}-${state.nodes.length}`),
            kind: "compact",
            content,
            complete: true,
            streamStartedAt: eventAt,
            streamFinishedAt: eventAt,
            streamDurationMs: 0
          });
      const withCompression = {
        ...base,
        sessionMessageCount: base.sessionMessageCount + 1,
        contextCompression: {
          ...(state.contextCompression ?? { active: false }),
          active: false,
          nodeId: nodeIdForCompression,
          mode: optionalString(payload.mode) ?? state.contextCompression?.mode,
          status: optionalString(payload.status),
          tokensBefore: optionalNumber(payload.tokens_before) ?? state.contextCompression?.tokensBefore,
          tokensAfter: optionalNumber(payload.tokens_after),
          messageCountBefore: optionalNumber(payload.message_count_before) ?? state.contextCompression?.messageCountBefore,
          messageCountAfter: optionalNumber(payload.message_count_after),
          updatedAt: eventAt
        }
      };
      return reconcileContextIndexesAfterCompaction(withCompression, payload);
    }

    case "model.retry":
      return addSystem(
        state,
        `模型重试 ${String(payload.attempt ?? "")}/${String(payload.max_retries ?? "")}: [${String(payload.error_type ?? "")}] ${String(payload.error_message ?? "")}`,
        { clearProcessing: false }
      );

    case "runner.interrupt":
      return addSystem(state, `执行被中断: ${String(payload.reason ?? "")}`);

    case "agent.interrupt":
      return addSystem(state, `Agent 中断: ${String(payload.interrupt_type ?? "")}`);

    case "model.interrupted": {
      const runId = String(payload.run_id ?? state.activeRunId ?? "");
      return runId ? completeOpenRunNodesForRun(state, runId, frameTime(frame)) : state;
    }

    case "debug.info":
      return {
        ...appendChatNode(state, {
          id: nodeId("debug", `${Date.now()}-${state.debugLines.length}`),
          kind: "debug",
          content: String(payload.message ?? "")
        }),
        debugLines: [
          ...state.debugLines.slice(-(MAX_DEBUG_LINES - 1)),
          String(payload.message ?? ""),
        ]
      };

    case "gui.workspace_changed":
      return addMeta(state, String(payload.message ?? "工作目录已切换"));

    case "error":
      return {
        ...appendChatNode(state, {
          id: nodeId("error", `${Date.now()}-${state.errors.length}`),
          kind: "error",
          content: String(payload.message ?? "Unknown error")
        }),
        errors: [...state.errors, String(payload.message ?? "Unknown error")]
      };

    case "plugin.message": {
      const nextState = addPluginMessage(state, payload, frame);
      const injection = pluginMessageFrameworkInjectionFromFrame(frame, payload);
      return injection ? appendFrameworkInjection(nextState, injection) : nextState;
    }

    case "plugin.event":
      return pruneResolvedHumanReviewMessages(
        addPluginMessage(state, payload, frame),
        payload
      );

    case "subagent.created":
    case "subagent.event":
    case "subagent.closed":
      return updateSubAgentFromEvent(state, payload, frame);

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

function normalizeControlState(value: unknown): RuntimeControlState {
  if (!isRecord(value)) {
    return { paused: false, resumable: false, pause_reason: null, paused_at: null, last_error_message: null };
  }
  return {
    paused: value.paused === true,
    pause_reason: optionalString(value.pause_reason),
    resumable: value.resumable === true,
    paused_at: optionalNumber(value.paused_at),
    last_error_message: optionalString(value.last_error_message),
  };
}

function sameControlState(a: RuntimeControlState, b: RuntimeControlState): boolean {
  return a.paused === b.paused && a.resumable === b.resumable && a.pause_reason === b.pause_reason;
}

function updateStatus(state: AppState, payload: Record<string, unknown>): AppState {
  const runnerState = String(payload.runner_state ?? state.runnerState);
  const agentState = String(payload.agent_state ?? state.agentState);
  const queueLengths = normalizeQueueLengths(payload.queue_lengths, state.queueLengths);
  const queueMessages = normalizeQueueMessages(
    payload.queue_messages,
    trimQueueMessages(state.queueMessages, queueLengths)
  );
  const contextUsage = chooseContextUsage(
    state.contextUsage,
    parseStatusContextUsage(payload.context_usage)
  );
  const contextAutoCompact = parseStatusAutoCompact(payload.auto_compact) ?? state.contextAutoCompact;
  const control = normalizeControlState(payload.control);
  if (
    runnerState === state.runnerState
    && agentState === state.agentState
    && sameQueueLengths(queueLengths, state.queueLengths)
    && sameQueueMessages(queueMessages, state.queueMessages)
    && sameContextUsage(contextUsage, state.contextUsage)
    && sameContextAutoCompact(contextAutoCompact, state.contextAutoCompact)
    && sameControlState(control, state.control)
  ) {
    return state;
  }
  return {
    ...state,
    runnerState,
    agentState,
    queueLengths,
    queueMessages,
    contextUsage,
    contextAutoCompact,
    control
  };
}

export interface SessionHistoryRecord {
  runId: string;
  role: "user" | "assistant" | "tool" | "system" | "error" | "event";
  content: unknown[];
  metadata?: Record<string, unknown>;
  contextMessageId?: string;
  contextMessageIndex?: number;
  timestamp?: number;
}

function normalizeSessionHistory(value: unknown): SessionHistoryRecord[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter(isRecord)
    .map((item, index) => normalizeSessionHistoryRecord(item, index))
    .filter((item): item is SessionHistoryRecord => item !== null);
}

function normalizeSessionHistoryRecord(item: Record<string, unknown>, index: number): SessionHistoryRecord | null {
  const role = item.role;
  if (
    role !== "user"
    && role !== "assistant"
    && role !== "tool"
    && role !== "system"
    && role !== "error"
    && role !== "event"
  ) {
    return null;
  }
  const content = Array.isArray(item.content) ? item.content : [];
  if (content.length === 0) {
    return null;
  }
  return {
    runId: optionalString(item.run_id ?? item.runId) ?? `history-${index}`,
    role,
    content,
    metadata: isRecord(item.metadata) ? item.metadata : undefined,
    contextMessageId: optionalString(item.context_message_id ?? item.contextMessageId),
    contextMessageIndex: optionalNumber(item.context_message_index ?? item.contextMessageIndex),
    timestamp: optionalNumber(item.timestamp)
  };
}

export function chatNodesFromMessageHistory(value: unknown): ChatNode[] {
  return sessionHistoryNodes(normalizeSessionHistory(value));
}

function sessionHistoryNodes(history: SessionHistoryRecord[]): ChatNode[] {
  const nodes: ChatNode[] = [];
  // Track tool nodes by tool_call_id so a subsequent tool record can fill in
  // the result. Tool calls are emitted as nodes at the position of the
  // assistant record that issued them; tool results update the existing
  // node's status + resultPreview rather than producing a new node.
  const toolNodesByCallId = new Map<string, ChatNode>();
  const compactNodesByRunId = new Map<string, ChatNode[]>();
  const userNodesByRunId = new Map<string, ChatNode>();
  const agentNodesByRunId = new Map<string, ChatNode>();
  const pendingProfilesByRunId = new Map<string, ModelProfileState>();

  history.forEach((record, index) => {
    const baseId = `${record.runId}-${index}`;
    if (record.role === "event") {
      const replayFrame = sessionHistoryEventFrame(record, index);
      if (replayFrame && applyHistoryChatEvent(nodes, replayFrame, index)) {
        return;
      }
      if (replayFrame?.type === "model.metadata") {
        const profile = parseModelProfile((replayFrame.payload ?? {}) as Record<string, unknown>);
        if (profile) {
          const merged = mergeModelProfile(pendingProfilesByRunId.get(record.runId), profile);
          pendingProfilesByRunId.set(record.runId, merged);
          applyHistoryModelProfile(
            userNodesByRunId.get(record.runId),
            agentNodesByRunId.get(record.runId),
            merged,
          );
        }
        return;
      }
      if (replayFrame && isSessionHistoryReplayOnlyEvent(replayFrame.type)) {
        return;
      }
      const eventType = optionalString(record.metadata?.event_type);
      if (eventType === "agent.compact_start") {
        const node: ChatNode = {
          id: nodeId("compact-history", baseId),
          kind: "compact",
          historyIndex: index,
          content: "Compressing context...",
          complete: false
        };
        nodes.push(node);
        const stack = compactNodesByRunId.get(record.runId) ?? [];
        stack.push(node);
        compactNodesByRunId.set(record.runId, stack);
        return;
      }
      if (eventType === "agent.compact_stop") {
        const content = historyContentText(record.content) || "Context compacted";
        const stack = compactNodesByRunId.get(record.runId) ?? [];
        const existing = stack.pop();
        if (existing) {
          existing.content = content;
          existing.complete = true;
          return;
        }
        nodes.push({
          id: nodeId("compact-history", baseId),
          kind: "compact",
          historyIndex: index,
          content,
          complete: true
        });
        return;
      }
      nodes.push({
        id: nodeId("event-history", baseId),
        kind: "system",
        historyIndex: index,
        content: historyContentText(record.content)
      });
      return;
    }
    if (record.role === "system" || record.role === "error") {
      nodes.push({
        id: nodeId(`${record.role}-history`, baseId),
        kind: record.role,
        historyIndex: index,
        content: historyContentText(record.content)
      });
      return;
    }
    if (record.role === "user") {
      const queue = normalizeQueue(record.metadata?.queue ?? "normal");
      const contentParts = visibleMessageContentParts(record.content);
      const node: ChatNode = {
        id: userMessageNodeId(optionalString(record.metadata?.message_id) ?? `history-${baseId}`),
        kind: "user",
        historyIndex: index,
        queue,
        displayMessageType: normalizeDisplayMessageType(
          record.metadata?.display_message_type,
          queue
        ),
        contextMessageId: record.contextMessageId,
        contextMessageIndex: record.contextMessageIndex,
        canFork: record.contextMessageId !== undefined || record.contextMessageIndex !== undefined,
        content: historyContentText(record.content),
        contentParts
      };
      const pendingProfile = pendingProfilesByRunId.get(record.runId);
      if (pendingProfile) {
        applyHistoryModelProfile(node, undefined, pendingProfile);
      }
      nodes.push(node);
      userNodesByRunId.set(record.runId, node);
      return;
    }
    if (record.role === "assistant") {
      const reasoning = historyReasoningText(record.content);
      const answer = historyAssistantText(record.content);
      const contentParts = visibleMessageContentParts(record.content);
      if (reasoning) {
        nodes.push({
          id: nodeId("thinking-history", baseId),
          kind: "thinking",
          historyIndex: index,
          content: reasoning,
          contextMessageId: record.contextMessageId,
          contextMessageIndex: record.contextMessageIndex,
          complete: true
        });
      }
      if (answer || contentParts) {
        const node: ChatNode = {
          id: nodeId("agent-history", baseId),
          kind: "agent",
          historyIndex: index,
          content: answer,
          contentParts,
          contextMessageId: record.contextMessageId,
          contextMessageIndex: record.contextMessageIndex,
          canFork: record.contextMessageId !== undefined || record.contextMessageIndex !== undefined,
          complete: true
        };
        const pendingProfile = pendingProfilesByRunId.get(record.runId);
        if (pendingProfile) {
          applyHistoryModelProfile(undefined, node, pendingProfile);
        }
        nodes.push(node);
        agentNodesByRunId.set(record.runId, node);
      }
      // Emit one tool node per tool_call in the assistant's content.
      // Status starts as "running" — replaced by "success"/"fail" when the
      // matching tool record is encountered. Tools that never got a result
      // (e.g., session ended mid-call) keep status="running".
      const toolCalls = record.content
        .filter(isRecord)
        .filter((part) => part.type === "tool_call");
      toolCalls.forEach((part, toolIndex) => {
        const toolCallId = optionalString(part.id) ?? "";
        const name = optionalString(part.name) ?? "tool";
        const args = part.arguments ?? part.args;
        const normalizedArgs = normalizeHistoryToolArguments(args);
        const argsRaw = normalizedArgs !== undefined
          ? formatToolValue(normalizedArgs)
          : optionalString(args) ?? "";
        const node: ChatNode = {
          id: nodeId("tool-history", `${baseId}-${toolIndex}`),
          kind: "tool",
          historyIndex: index,
          content: "",
          tool: {
            runId: record.runId,
            toolCallId,
            name,
            status: "running",
            argsRaw,
            argsState: "complete",
            arguments: normalizedArgs,
            resultPreview: ""
          }
        };
        nodes.push(node);
        if (toolCallId) {
          toolNodesByCallId.set(toolCallId, node);
        }
      });
      return;
    }
    // role === "tool"
    const result = historyToolResult(record.content);
    const existing = result.toolCallId
      ? toolNodesByCallId.get(result.toolCallId)
      : undefined;
    if (existing && existing.tool) {
      // Pair with the tool_call node already emitted above.
      existing.tool = {
        ...existing.tool,
        status: result.isError ? "fail" : "success",
        resultPreview: result.text,
        resultData: result.data,
        contextMessageId: record.contextMessageId,
        contextMessageIndex: record.contextMessageIndex
      };
      return;
    }
    // Fallback: orphan tool result with no matching tool_call (shouldn't
    // happen with well-formed history, but render it standalone so the
    // user sees the data rather than silently dropping it).
    nodes.push({
      id: nodeId("tool-history", baseId),
      kind: "tool",
      historyIndex: index,
      content: "",
      tool: {
        runId: record.runId,
        toolCallId: result.toolCallId,
        name: result.name,
        status: result.isError ? "fail" : "success",
        argsRaw: "",
        argsState: "complete",
        resultPreview: result.text,
        resultData: result.data,
        contextMessageId: record.contextMessageId,
        contextMessageIndex: record.contextMessageIndex
      }
    });
  });
  return nodes;
}

function applyHistoryModelProfile(
  userNode: ChatNode | undefined,
  agentNode: ChatNode | undefined,
  profile: ModelProfileState,
): void {
  const userProfile = userModelProfile(profile);
  if (userNode && userProfile) {
    userNode.profile = mergeModelProfile(userNode.profile, userProfile);
  }
  const agentProfile = assistantModelProfile(profile);
  if (agentNode && agentProfile) {
    agentNode.profile = mergeModelProfile(agentNode.profile, agentProfile);
  }
}

function nextContextMessageIndexFromHistory(history: SessionHistoryRecord[]): number {
  const maxIndex = history.reduce((current, record) => {
    if (typeof record.contextMessageIndex !== "number") {
      return current;
    }
    return Math.max(current, record.contextMessageIndex);
  }, -1);
  return maxIndex + 1;
}

function sessionHistoryReplayState(history: SessionHistoryRecord[]): AppState {
  return history.reduce((state, record, index) => {
    const frame = sessionHistoryEventFrame(record, index);
    return frame ? reduceCoreEvent(state, frame) : state;
  }, createInitialState());
}

function sessionHistoryEventFrame(record: SessionHistoryRecord, index: number): CoreFrame | undefined {
  if (record.role !== "event") return undefined;
  const metadata = record.metadata;
  if (!metadata) return undefined;
  const eventType = optionalString(metadata.event_type);
  if (!eventType) return undefined;
  const payload = isRecord(metadata.event_payload) ? metadata.event_payload : undefined;
  if (!payload) return undefined;
  return {
    version: VERSION,
    type: eventType,
    ts: record.timestamp,
    id: `history-${record.runId}-${index}`,
    payload
  };
}

function applyHistoryChatEvent(nodes: ChatNode[], frame: CoreFrame, historyIndex: number): boolean {
  const payload = (frame.payload ?? {}) as Record<string, unknown>;
  if (frame.type === "agent.system_prompt") {
    mergeHistorySystemPromptInjection(nodes, frame, frameworkInjectionFromFrame(frame, payload, "system_prompt"), historyIndex);
    return true;
  }
  if (frame.type === "agent.context_injected") {
    const injection = frameworkInjectionFromFrame(frame, payload, "context_injected");
    if (payload.merge_target === "user_message") {
      const targetNodeId = findUserMessageNodeId(
        {
          ...createInitialState(),
          nodes
        },
        optionalString(payload.target_message_id),
        optionalString(payload.target_context_message_id),
        optionalNumber(payload.target_message_index)
      );
      if (targetNodeId) {
        const target = nodes.find((node) => node.id === targetNodeId);
        if (target) {
          target.injections = orderFrameworkInjections([...(target.injections ?? []), injection]);
          return true;
        }
      }
    }
    appendHistoryFrameworkInjection(nodes, frame, injection, historyIndex);
    return true;
  }
  if (frame.type === "agent.tool_runtime_context_injected") {
    appendHistoryFrameworkInjection(
      nodes,
      frame,
      frameworkInjectionFromFrame(frame, payload, "tool_runtime_context_injected"),
      historyIndex
    );
    return true;
  }
  if (frame.type === "plugin.message") {
    const injection = pluginMessageFrameworkInjectionFromFrame(frame, payload);
    if (injection) {
      appendHistoryFrameworkInjection(nodes, frame, injection, historyIndex);
    }
    return true;
  }
  return false;
}

function isSessionHistoryReplayOnlyEvent(eventType: string): boolean {
  return eventType === "model.metadata"
    || eventType === "plugin.event"
    || eventType === "agent.tool_parameter_injected"
    || eventType === "plugin.status"
    || eventType === "plugin.tool_progress"
    || eventType === "plugin.artifact.upsert"
    || eventType === "plugin.artifact.delta"
    || eventType === "plugin.artifact.remove"
    || eventType === "plugin.artifact.clear"
    || eventType === "subagent.created"
    || eventType === "subagent.event"
    || eventType === "subagent.closed";
}

function historyFrameworkNode(frame: CoreFrame, injection: FrameworkInjectionState, index: number, historyIndex?: number): ChatNode {
  return {
    id: nodeId("framework-history", `${frame.id ?? injection.id}-${index}`),
    kind: "framework",
    historyIndex,
    content: injection.content,
    framework: injection
  };
}

function attachHistoryToolProgress(nodes: ChatNode[], progress: Record<string, ToolProgressState>): ChatNode[] {
  if (Object.keys(progress).length === 0) return nodes;
  return nodes.map((node) => {
    if (!node.tool?.toolCallId) return node;
    const toolProgress = progress[node.tool.toolCallId];
    if (!toolProgress) return node;
    return {
      ...node,
      tool: {
        ...node.tool,
        progress: toolProgress
      }
    };
  });
}

function normalizeHistoryToolArguments(value: unknown): unknown | undefined {
  if (isRecord(value) || Array.isArray(value)) return value;
  if (typeof value !== "string") return undefined;
  const parsed = tryParseJson(value.trim());
  return parsed.ok ? parsed.value : undefined;
}

function historyAssistantText(content: unknown[]): string {
  // Only render text/steer content here. tool_call blocks are extracted
  // separately by sessionHistoryNodes and rendered as tool nodes.
  return historyContentText(content, { includeReasoning: false });
}

function historyReasoningText(content: unknown[]): string {
  return content
    .filter(isRecord)
    .filter((part) => part.type === "reasoning")
    .map((part) => optionalString(part.reasoning ?? part.text) ?? "")
    .filter(Boolean)
    .join("\n\n");
}

function historyContentText(
  content: unknown[],
  options: { includeReasoning?: boolean } = {},
): string {
  const includeReasoning = options.includeReasoning ?? true;
  return content
    .map((part) => {
      if (!isRecord(part)) return formatToolValue(part);
      if (part.type === "text") return optionalString(part.text) ?? "";
      if (part.type === "steer" && Array.isArray(part.content)) {
        return historyContentText(part.content);
      }
      if (includeReasoning && part.type === "reasoning") {
        return optionalString(part.reasoning ?? part.text) ?? "";
      }
      if (part.type === "tool_result") {
        const nested = Array.isArray(part.content)
          ? historyContentText(part.content)
          : formatToolValue(part.content);
        const label = optionalString(part.tool_call_id) ?? "tool";
        return `Tool result ${label}: ${nested}`;
      }
      return "";
    })
    .filter(Boolean)
    .join("\n\n");
}

function visibleMessageContentParts(value: unknown): ContentPart[] | undefined {
  const parts = normalizeContentParts(value);
  if (!parts) return undefined;
  const visible = parts.filter(isVisibleMessageContentPart);
  return visible.length ? visible : undefined;
}

function isVisibleMessageContentPart(part: ContentPart): boolean {
  if (part.type === "text") {
    return typeof part.text === "string" && part.text.trim().length > 0;
  }
  return part.type === "image"
    || part.type === "document"
    || part.type === "audio"
    || part.type === "video"
    || part.type === "file";
}

function historyToolResult(content: unknown[]): {
  toolCallId: string;
  name: string;
  text: string;
  isError: boolean;
  data?: unknown;
} {
  const part = content.filter(isRecord).find((item) => item.type === "tool_result");
  if (!part) {
    return {
      toolCallId: "",
      name: "tool_result",
      text: historyContentText(content),
      isError: false
    };
  }
  const nested = Array.isArray(part.content)
    ? historyContentText(part.content)
    : formatToolValue(part.content);
  const toolCallId = optionalString(part.tool_call_id) ?? "";
  const data = part.output ?? part.data;
  return {
    toolCallId,
    name: toolCallId || "tool_result",
    text: nested,
    isError: part.is_error === true,
    data
  };
}

function parseStatusContextUsage(value: unknown): ContextUsageState | undefined {
  if (!isRecord(value)) return undefined;
  const usedTokens = optionalNumber(value.used_tokens);
  if (usedTokens === undefined) return undefined;
  const maxContextTokens = optionalNumber(value.max_context_tokens);
  const ratio = optionalNumber(value.usage_ratio);
  return {
    usedTokens,
    maxContextTokens,
    ratio: ratio === undefined ? undefined : clamp(ratio, 0, 1),
    source: contextUsageSource(value.source) ?? "estimate"
  };
}

function parseStatusAutoCompact(value: unknown): ContextAutoCompactState | undefined {
  if (!isRecord(value)) return undefined;
  const enabled = value.enabled === true;
  const maxContextTokens = optionalNumber(value.max_context_tokens);
  const triggerTokens = optionalNumber(value.trigger_tokens);
  const triggerRatio = optionalNumber(value.trigger_ratio);
  const maxTriggerRatio = optionalNumber(value.max_trigger_ratio);
  const compressionBudget = optionalNumber(value.compression_budget);
  const tokenLimit = optionalNumber(value.token_limit);
  const tokenLimitRatio = optionalNumber(value.token_limit_ratio);
  return {
    enabled,
    maxContextTokens,
    triggerTokens,
    triggerRatio: triggerRatio === undefined ? undefined : clamp(triggerRatio, 0, 1),
    maxTriggerRatio: maxTriggerRatio === undefined ? undefined : clamp(maxTriggerRatio, 0, 1),
    compressionBudget,
    tokenLimit,
    tokenLimitRatio: tokenLimitRatio === undefined ? undefined : clamp(tokenLimitRatio, 0, 1),
  };
}

function sameContextUsage(a?: ContextUsageState, b?: ContextUsageState): boolean {
  if (!a && !b) return true;
  if (!a || !b) return false;
  return a.usedTokens === b.usedTokens
    && a.maxContextTokens === b.maxContextTokens
    && a.ratio === b.ratio
    && a.source === b.source;
}

function sameContextAutoCompact(a?: ContextAutoCompactState, b?: ContextAutoCompactState): boolean {
  if (!a && !b) return true;
  if (!a || !b) return false;
  return a.enabled === b.enabled
    && a.maxContextTokens === b.maxContextTokens
    && a.triggerTokens === b.triggerTokens
    && a.triggerRatio === b.triggerRatio
    && a.maxTriggerRatio === b.maxTriggerRatio
    && a.compressionBudget === b.compressionBudget
    && a.tokenLimit === b.tokenLimit
    && a.tokenLimitRatio === b.tokenLimitRatio;
}

function chooseContextUsage(current: ContextUsageState | undefined, incoming: ContextUsageState | undefined): ContextUsageState | undefined {
  if (!incoming) return current;
  if (
    current?.source === "provider_usage"
    && incoming.source === "estimate"
  ) {
    if (incoming.usedTokens <= current.usedTokens) {
      return current;
    }
    return { ...incoming, source: "provider_usage" };
  }
  return incoming;
}

function parseContextUsage(payload: Record<string, unknown>): ContextUsageState | undefined {
  const usedTokens = optionalNumber(payload.context_tokens);
  if (usedTokens === undefined) return undefined;
  const maxContextTokens = optionalNumber(payload.max_context_tokens);
  const ratio = optionalNumber(payload.context_ratio);
  return {
    usedTokens,
    maxContextTokens,
    ratio: ratio === undefined ? undefined : clamp(ratio, 0, 1),
    source: contextUsageSource(payload.context_source) ?? "provider_usage"
  };
}

function parseModelUsage(payload: Record<string, unknown>): ModelUsageState | undefined {
  const nestedUsage = isRecord(payload.usage) ? payload.usage : {};
  const readNumber = (key: string) => optionalNumber(payload[key]) ?? optionalNumber(nestedUsage[key]);
  const rawInputTokens = readNumber("input_tokens");
  const outputTokens = readNumber("output_tokens");
  const totalTokens = readNumber("total_tokens");
  const cacheReadTokens = readNumber("cache_read_tokens");
  const cacheWriteTokens = readNumber("cache_write_tokens");
  const hasUsage = [
    rawInputTokens,
    outputTokens,
    totalTokens,
    cacheReadTokens,
    cacheWriteTokens
  ].some((value) => value !== undefined);
  if (!hasUsage) return undefined;

  const input = rawInputTokens ?? 0;
  const output = outputTokens ?? 0;
  const cacheRead = cacheReadTokens ?? 0;
  const cacheWrite = cacheWriteTokens ?? 0;
  const total = totalTokens ?? input + output + cacheRead + cacheWrite;
  return {
    totalTokens: total,
    inputTokens: nonCachedInputTokens(input, output, totalTokens, cacheRead, cacheWrite),
    outputTokens: output,
    cacheReadTokens: cacheRead,
    cacheWriteTokens: cacheWrite
  };
}

function nonCachedInputTokens(
  inputTokens: number,
  outputTokens: number,
  explicitTotalTokens: number | undefined,
  cacheReadTokens: number,
  cacheWriteTokens: number,
): number {
  if (explicitTotalTokens !== undefined && explicitTotalTokens <= inputTokens + outputTokens) {
    return Math.max(0, inputTokens - cacheReadTokens - cacheWriteTokens);
  }
  return inputTokens;
}

function addModelUsage(
  current: ModelUsageState | undefined,
  incoming: ModelUsageState | undefined,
): ModelUsageState | undefined {
  if (!incoming) return current;
  if (!current) return incoming;
  return {
    totalTokens: current.totalTokens + incoming.totalTokens,
    inputTokens: current.inputTokens + incoming.inputTokens,
    outputTokens: current.outputTokens + incoming.outputTokens,
    cacheReadTokens: current.cacheReadTokens + incoming.cacheReadTokens,
    cacheWriteTokens: current.cacheWriteTokens + incoming.cacheWriteTokens
  };
}

function parseModelProfile(payload: Record<string, unknown>): ModelProfileState | undefined {
  const nestedUsage = isRecord(payload.usage) ? payload.usage : {};
  const readUsageNumber = (key: string) => optionalNumber(payload[key]) ?? optionalNumber(nestedUsage[key]);
  const profile: ModelProfileState = {};
  putProfileNumber(profile, "cacheTokens", readUsageNumber("cache_read_tokens") ?? optionalNumber(payload.cache_tokens));
  putProfileNumber(profile, "prefillTokens", optionalNumber(payload.prefill_tokens));
  putProfileNumber(profile, "prefillTotalTokens", optionalNumber(payload.prefill_total_tokens));
  putProfileNumber(profile, "prefillMs", optionalNumber(payload.prefill_ms) ?? optionalNumber(payload.ttft_ms));
  putProfileNumber(profile, "prefillTokensPerSecond", optionalNumber(payload.prefill_tokens_per_second));
  putProfileNumber(profile, "ttftMs", optionalNumber(payload.ttft_ms));
  putProfileNumber(profile, "decodeTokens", optionalNumber(payload.decode_tokens));
  putProfileNumber(profile, "decodeMs", optionalNumber(payload.decode_ms));
  putProfileNumber(profile, "decodeTokensPerSecond", optionalNumber(payload.decode_tokens_per_second));
  putProfileNumber(profile, "peakDecodeTokensPerSecond", optionalNumber(payload.peak_decode_tokens_per_second));
  return hasProfileValues(profile) ? profile : undefined;
}

function putProfileNumber<T extends keyof ModelProfileState>(
  profile: ModelProfileState,
  key: T,
  value: number | undefined,
): void {
  if (value !== undefined) {
    profile[key] = value;
  }
}

function hasProfileValues(profile: ModelProfileState): boolean {
  return Object.values(profile).some((value) => typeof value === "number" && Number.isFinite(value));
}

function userModelProfile(profile: ModelProfileState): ModelProfileState | undefined {
  const userProfile: ModelProfileState = {};
  putProfileNumber(userProfile, "cacheTokens", profile.cacheTokens);
  putProfileNumber(userProfile, "prefillTokens", profile.prefillTokens);
  putProfileNumber(userProfile, "prefillTotalTokens", profile.prefillTotalTokens);
  putProfileNumber(userProfile, "prefillMs", profile.prefillMs);
  putProfileNumber(userProfile, "prefillTokensPerSecond", profile.prefillTokensPerSecond);
  return hasProfileValues(userProfile) ? userProfile : undefined;
}

function assistantModelProfile(profile: ModelProfileState): ModelProfileState | undefined {
  const assistantProfile: ModelProfileState = {};
  putProfileNumber(assistantProfile, "ttftMs", profile.ttftMs);
  putProfileNumber(assistantProfile, "decodeTokens", profile.decodeTokens);
  putProfileNumber(assistantProfile, "decodeMs", profile.decodeMs);
  putProfileNumber(assistantProfile, "decodeTokensPerSecond", profile.decodeTokensPerSecond);
  putProfileNumber(assistantProfile, "peakDecodeTokensPerSecond", profile.peakDecodeTokensPerSecond);
  return hasProfileValues(assistantProfile) ? assistantProfile : undefined;
}

function mergeModelProfile(
  current: ModelProfileState | undefined,
  incoming: ModelProfileState,
): ModelProfileState {
  return { ...(current ?? {}), ...incoming };
}

function attachModelProfileToRun(state: AppState, payload: Record<string, unknown>): AppState {
  const profile = parseModelProfile(payload);
  if (!profile) return state;

  const runId = String(payload.run_id ?? state.activeRunId ?? "");
  const run = runId ? state.runs[runId] : undefined;
  const mergedRunProfile = runId ? mergeModelProfile(run?.profile, profile) : undefined;
  const userNodeId = run?.userNodeId;
  const prefillNodeId = runId ? latestToolNodeIdForRun(state, runId) ?? userNodeId : userNodeId;
  const assistantNodeId = run?.agentNodeId ?? run?.assistantCommitNodeId;
  const userProfile = userModelProfile(profile);
  const assistantProfile = assistantModelProfile(profile);
  if (!userProfile && !assistantProfile) return state;

  let nodesChanged = false;
  const nodes = state.nodes.map((node) => {
    if (
      userProfile
      && node.id === prefillNodeId
      && (node.kind === "user" || node.kind === "tool")
    ) {
      nodesChanged = true;
      return { ...node, profile: mergeModelProfile(node.profile, userProfile) };
    }
    if (assistantProfile && node.id === assistantNodeId && node.kind === "agent") {
      nodesChanged = true;
      return { ...node, profile: mergeModelProfile(node.profile, assistantProfile) };
    }
    return node;
  });

  if (!mergedRunProfile || !runId) {
    return nodesChanged ? { ...state, nodes } : state;
  }

  return {
    ...state,
    nodes: nodesChanged ? nodes : state.nodes,
    runs: {
      ...state.runs,
      [runId]: {
        ...(run ?? {}),
        profile: mergedRunProfile
      }
    }
  };
}

function latestToolNodeIdForRun(state: AppState, runId: string): string | undefined {
  for (let index = state.nodes.length - 1; index >= 0; index -= 1) {
    const node = state.nodes[index];
    if (node.kind === "tool" && node.tool?.runId === runId) {
      return node.id;
    }
  }
  return undefined;
}

function contextUsageSource(value: unknown): ContextUsageState["source"] | undefined {
  if (value === "estimate" || value === "provider_usage") return value;
  return undefined;
}

function formatModelMetadata(payload: Record<string, unknown>): string {
  const latency = payload.latency_ms == null ? "n/a" : `${Number(payload.latency_ms).toFixed(0)}ms`;
  const details = formatUsageDetails(payload);
  const context = formatContextUsage(parseContextUsage(payload));
  const timing = formatModelTiming(payload);
  return [
    `模型统计 in=${Number(payload.input_tokens ?? 0)}`,
    `out=${Number(payload.output_tokens ?? 0)}`,
    `total=${Number(payload.total_tokens ?? 0)}`,
    details,
    context,
    timing,
    `latency=${latency}`
  ].filter(Boolean).join(" ");
}

function formatModelTiming(payload: Record<string, unknown>): string {
  const parts: string[] = [];
  const ttft = optionalNumber(payload.ttft_ms);
  const prefillTps = optionalNumber(payload.prefill_tokens_per_second);
  const decodeTps = optionalNumber(payload.decode_tokens_per_second);
  if (ttft !== undefined) parts.push(`ttft=${ttft.toFixed(0)}ms`);
  if (prefillTps !== undefined && prefillTps > 0) {
    parts.push(`prefill≈${formatTokenRate(prefillTps)}`);
  }
  if (decodeTps !== undefined && decodeTps > 0) {
    parts.push(`decode≈${formatTokenRate(decodeTps)}`);
  }
  return parts.join(" ");
}

function formatTokenRate(value: number): string {
  const rounded = value >= 10 ? value.toFixed(0) : value.toFixed(1);
  return `${rounded} tok/s`;
}

function formatUsageDetails(payload: Record<string, unknown>): string {
  const parts: string[] = [];
  const cacheRead = optionalNumber(payload.cache_read_tokens);
  const cacheWrite = optionalNumber(payload.cache_write_tokens);
  const cacheMiss = optionalNumber(payload.cache_miss_tokens);
  const reasoning = optionalNumber(payload.reasoning_tokens);
  if (cacheRead !== undefined) parts.push(`cache_read=${cacheRead}`);
  if (cacheWrite !== undefined) parts.push(`cache_write=${cacheWrite}`);
  if (cacheMiss !== undefined) parts.push(`cache_miss=${cacheMiss}`);
  if (reasoning !== undefined) parts.push(`reasoning=${reasoning}`);
  if (contextUsageSource(payload.context_source) === "estimate") {
    parts.push("provider_usage=missing");
  }
  return parts.length ? `[${parts.join(" ")}]` : "";
}

function formatContextUsage(context?: ContextUsageState): string {
  if (!context) return "";
  const estimated = context.source === "estimate";
  const prefix = estimated ? "ctx≈" : "ctx=";
  const suffix = estimated ? " estimated" : "";
  if (!context.maxContextTokens || context.ratio === undefined) {
    return `${prefix}${context.usedTokens}${suffix}`;
  }
  return `${prefix}${context.usedTokens}/${context.maxContextTokens} (${Math.round(context.ratio * 100)}%)${suffix}`;
}

function appendRunDelta(
  state: AppState,
  runId: string,
  kind: "agent" | "thinking",
  delta: string,
  eventAt: number,
): AppState {
  const run = state.runs[runId] ?? {};
  const key = kind === "agent" ? "agentNodeId" : "thinkingNodeId";
  const existingId = run[key];
  const pendingAssistantProfile = kind === "agent"
    ? assistantModelProfile(run.profile ?? {})
    : undefined;
  if (existingId) {
    return updateChatNode(state, existingId, (node) => ({
      ...node,
      content: node.content + delta,
      complete: false,
      streamStartedAt: node.streamStartedAt ?? eventAt
    }));
  }
  const processingId = run.processingId;
  const shouldCountAssistant = run.assistantMessageCounted !== true;
  const contextMessageIndex = kind === "agent"
    ? run.assistantContextMessageIndex ?? state.nextContextMessageIndex
    : undefined;
  const shouldIndexAssistant = kind === "agent"
    && run.assistantContextMessageIndex === undefined;
  if (processingId) {
    const withoutProcessing = clearProcessingForRun(state, runId);
    const id = nodeId(kind, `${runId}-${withoutProcessing.nodes.length}`);
    const next = appendChatNode(withoutProcessing, {
      id,
      kind,
      content: delta,
      contextMessageIndex,
      canFork: contextMessageIndex !== undefined,
      complete: false,
      streamStartedAt: eventAt,
      ...(pendingAssistantProfile ? { profile: pendingAssistantProfile } : {})
    });
    return {
      ...next,
      sessionMessageCount: shouldCountAssistant
        ? next.sessionMessageCount + 1
        : next.sessionMessageCount,
      runs: {
        ...next.runs,
        [runId]: {
          ...(next.runs[runId] ?? {}),
          startedAt: (next.runs[runId] ?? run).startedAt ?? eventAt,
          [key]: id,
          assistantCommitNodeId: kind === "agent"
            ? id
            : run.assistantCommitNodeId,
          assistantMessageCounted: true,
          assistantContextMessageIndex: kind === "agent"
            ? contextMessageIndex
            : run.assistantContextMessageIndex,
          toolCallAssistantContextIndexed: kind === "agent"
            ? false
            : run.toolCallAssistantContextIndexed,
        }
      },
      nextContextMessageIndex: shouldIndexAssistant
        ? next.nextContextMessageIndex + 1
        : next.nextContextMessageIndex
    };
  }
  const id = nodeId(kind, `${runId}-${state.nodes.length}`);
  const next = appendChatNode(state, {
    id,
    kind,
    content: delta,
    contextMessageIndex,
    canFork: contextMessageIndex !== undefined,
    complete: false,
    streamStartedAt: eventAt,
    ...(pendingAssistantProfile ? { profile: pendingAssistantProfile } : {})
  });
  return {
    ...next,
    sessionMessageCount: shouldCountAssistant
      ? next.sessionMessageCount + 1
      : next.sessionMessageCount,
    runs: {
      ...next.runs,
      [runId]: {
        ...(next.runs[runId] ?? {}),
        startedAt: (next.runs[runId] ?? run).startedAt ?? eventAt,
        [key]: id,
        assistantCommitNodeId: kind === "agent"
          ? id
          : run.assistantCommitNodeId,
        assistantMessageCounted: true,
        assistantContextMessageIndex: kind === "agent"
          ? contextMessageIndex
          : run.assistantContextMessageIndex,
        toolCallAssistantContextIndexed: kind === "agent"
          ? false
          : run.toolCallAssistantContextIndexed,
      }
    },
    nextContextMessageIndex: shouldIndexAssistant
      ? next.nextContextMessageIndex + 1
      : next.nextContextMessageIndex
  };
}

function commitAssistantNode(
  state: AppState,
  runId: string,
  nodeIdForAssistant: string,
  contextMessageId: string,
  content: string,
  contentParts?: ContentPart[],
): AppState {
  const updated = updateChatNode(state, nodeIdForAssistant, (node) => ({
    ...node,
    content: node.content || content,
    contentParts: contentParts ?? node.contentParts,
    contextMessageId,
    canFork: true
  }));
  return {
    ...updated,
    runs: {
      ...updated.runs,
      [runId]: {
        ...(updated.runs[runId] ?? {}),
        assistantCommitNodeId: undefined,
        assistantMessageCounted: true
      }
    }
  };
}

function appendCommittedAssistantNode(
  state: AppState,
  runId: string,
  contextMessageId: string,
  content: string,
  contentParts: ContentPart[],
  eventAt: number,
): AppState {
  const run = state.runs[runId] ?? {};
  const withoutProcessing = clearProcessingForRun(state, runId);
  const contextMessageIndex = run.assistantContextMessageIndex ?? withoutProcessing.nextContextMessageIndex;
  const shouldIndexAssistant = run.assistantContextMessageIndex === undefined;
  const shouldCountAssistant = run.assistantMessageCounted !== true;
  const id = nodeId("agent", `${runId}-${withoutProcessing.nodes.length}`);
  const pendingAssistantProfile = assistantModelProfile(run.profile ?? {});
  const next = appendChatNode(withoutProcessing, {
    id,
    kind: "agent",
    content,
    contentParts,
    contextMessageId,
    contextMessageIndex,
    canFork: true,
    complete: true,
    streamStartedAt: eventAt,
    streamFinishedAt: eventAt,
    streamDurationMs: 0,
    ...(pendingAssistantProfile ? { profile: pendingAssistantProfile } : {})
  });
  return {
    ...next,
    sessionMessageCount: shouldCountAssistant
      ? next.sessionMessageCount + 1
      : next.sessionMessageCount,
    runs: {
      ...next.runs,
      [runId]: {
        ...(next.runs[runId] ?? {}),
        startedAt: (next.runs[runId] ?? run).startedAt ?? eventAt,
        agentNodeId: id,
        assistantCommitNodeId: undefined,
        assistantMessageCounted: true,
        assistantContextMessageIndex: contextMessageIndex,
        toolCallAssistantContextIndexed: false
      }
    },
    nextContextMessageIndex: shouldIndexAssistant
      ? next.nextContextMessageIndex + 1
      : next.nextContextMessageIndex
  };
}

function indexToolCallAssistantContext(state: AppState, runId: string): AppState {
  if (!runId) {
    return state;
  }
  const run = state.runs[runId] ?? {};
  if (
    run.assistantContextMessageIndex !== undefined
    || run.toolCallAssistantContextIndexed === true
  ) {
    return state;
  }
  return {
    ...state,
    nextContextMessageIndex: state.nextContextMessageIndex + 1,
    runs: {
      ...state.runs,
      [runId]: {
        ...run,
        toolCallAssistantContextIndexed: true
      }
    }
  };
}

function toolResultNeedsContextIndex(
  state: AppState,
  toolCallId: string,
  payload: Record<string, unknown>,
): boolean {
  if (!toolCallId || payload.is_part === true || payload.interrupted === true) {
    return false;
  }
  const nodeIdForTool = state.toolNodeByCallId[toolCallId];
  if (!nodeIdForTool) {
    return true;
  }
  const node = state.nodes.find((item) => item.id === nodeIdForTool);
  return node?.tool?.contextMessageIndex === undefined;
}

function noteContextInsertion(
  state: AppState,
  payload: Record<string, unknown>,
): AppState {
  const insertionIndex = optionalNumber(payload.position);
  const role = optionalString(payload.role);
  if (
    insertionIndex === undefined
    || insertionIndex < 0
    || !isContextMessageRole(role)
  ) {
    return state;
  }
  return mapContextIndexes(
    state,
    (index) => index >= insertionIndex ? index + 1 : index,
    Math.max(state.nextContextMessageIndex + 1, insertionIndex + 1),
  );
}

function reconcileContextIndexesAfterCompaction(
  state: AppState,
  payload: Record<string, unknown>,
): AppState {
  if (optionalString(payload.status) !== "success") {
    return state;
  }
  const replacedMessageCount = optionalNumber(payload.replaced_message_count);
  if (replacedMessageCount === undefined || replacedMessageCount <= 0) {
    return state;
  }
  const messageCountAfter = optionalNumber(payload.message_count_after);
  return mapContextIndexes(
    state,
    (index) => (
      index < replacedMessageCount
        ? undefined
        : index - replacedMessageCount + 1
    ),
    messageCountAfter ?? Math.max(1, state.nextContextMessageIndex - replacedMessageCount + 1),
  );
}

function mapContextIndexes(
  state: AppState,
  mapIndex: (index: number) => number | undefined,
  nextContextMessageIndex: number,
): AppState {
  const nodes = state.nodes.map((node) => {
    let nextNode = node;
    if (node.contextMessageIndex !== undefined) {
      const mapped = mapIndex(node.contextMessageIndex);
      if (mapped !== node.contextMessageIndex) {
        nextNode = {
          ...nextNode,
          contextMessageIndex: mapped,
          contextMessageId: mapped === undefined ? undefined : nextNode.contextMessageId,
          canFork: mapped === undefined ? false : nextNode.canFork,
        };
      }
    }
    if (nextNode.tool?.contextMessageIndex !== undefined) {
      const mapped = mapIndex(nextNode.tool.contextMessageIndex);
      if (mapped !== nextNode.tool.contextMessageIndex) {
        nextNode = {
          ...nextNode,
          tool: {
            ...nextNode.tool,
            contextMessageId: mapped === undefined ? undefined : nextNode.tool.contextMessageId,
            contextMessageIndex: mapped,
          },
        };
      }
    }
    return nextNode;
  });

  const runs = Object.fromEntries(
    Object.entries(state.runs).map(([runId, run]) => {
      if (run.assistantContextMessageIndex === undefined) {
        return [runId, run];
      }
      const mapped = mapIndex(run.assistantContextMessageIndex);
      return [
        runId,
        mapped === run.assistantContextMessageIndex
          ? run
          : { ...run, assistantContextMessageIndex: mapped },
      ];
    })
  );

  return {
    ...state,
    nodes,
    runs,
    nextContextMessageIndex,
  };
}

function isContextMessageRole(role?: string): boolean {
  return role === "user"
    || role === "assistant"
    || role === "tool"
    || role === "system"
    || role === "error";
}

function completeThinkingForRun(state: AppState, runId: string, finishedAt: number): AppState {
  return completeRunNodeForRun(state, runId, "thinkingNodeId", "thinking", finishedAt);
}

function completeOpenRunNodesForRun(state: AppState, runId: string, finishedAt: number): AppState {
  return completeRunNodeForRun(
    completeRunNodeForRun(
      clearProcessingForRun(state, runId),
      runId,
      "thinkingNodeId",
      "thinking",
      finishedAt
    ),
    runId,
    "agentNodeId",
    "agent",
    finishedAt
  );
}

function completeActiveCompressionOnRunStop(
  state: AppState,
  payload: Record<string, unknown>,
  eventAt: number,
): AppState {
  if (!state.contextCompression?.active || !state.contextCompression.nodeId) {
    return state;
  }
  const content = String(payload.stop_reason ?? "") === "interrupted"
    ? "Context compaction interrupted"
    : "Context compaction stopped";
  const updated = updateChatNode(state, state.contextCompression.nodeId, (node) => (
    node.kind === "compact"
      ? completeChatNodeStream({ ...node, content, complete: true }, eventAt)
      : node
  ));
  return {
    ...updated,
    contextCompression: {
      ...state.contextCompression,
      active: false,
      status: String(payload.stop_reason ?? "stopped"),
      updatedAt: eventAt
    }
  };
}

function completeRunNodeForRun(
  state: AppState,
  runId: string,
  key: "agentNodeId" | "thinkingNodeId",
  kind: "agent" | "thinking",
  finishedAt: number
): AppState {
  const nodeIdForRun = state.runs[runId]?.[key];
  if (!nodeIdForRun) {
    return state;
  }
  const updated = updateChatNode(state, nodeIdForRun, (node) => (
    node.kind === kind ? completeChatNodeStream({ ...node, complete: true }, finishedAt) : node
  ));
  return {
    ...updated,
    runs: {
      ...updated.runs,
      [runId]: {
        ...(updated.runs[runId] ?? {}),
        [key]: undefined
      }
    }
  };
}

function completeChatNodeStream(node: ChatNode, finishedAt: number): ChatNode {
  if (
    node.streamFinishedAt !== undefined
    || node.streamStartedAt === undefined
    || !Number.isFinite(finishedAt)
  ) {
    return node;
  }
  return {
    ...node,
    streamFinishedAt: finishedAt,
    streamDurationMs: Math.max(0, finishedAt - node.streamStartedAt)
  };
}

function completeToolStream(tool: ToolState, finishedAt: number): ToolState {
  if (
    tool.streamFinishedAt !== undefined
    || tool.streamStartedAt === undefined
    || !Number.isFinite(finishedAt)
  ) {
    return tool;
  }
  return {
    ...tool,
    streamFinishedAt: finishedAt,
    streamDurationMs: Math.max(0, finishedAt - tool.streamStartedAt)
  };
}

function clearProcessingForRun(state: AppState, runId: string): AppState {
  const processingId = state.runs[runId]?.processingId;
  if (!processingId) {
    return state;
  }
  return {
    ...state,
    processing: state.processing?.id === processingId ? undefined : state.processing,
    runs: {
      ...state.runs,
      [runId]: {
        ...(state.runs[runId] ?? {}),
        processingId: undefined
      }
    }
  };
}

function ensureProcessingForRun(state: AppState, runId: string, eventAt?: number): AppState {
  const existingId = state.runs[runId]?.processingId;
  if (existingId && state.processing?.id === existingId) {
    return state;
  }
  const base = clearProcessing(state);
  const processing: ProcessingState = {
    id: nodeId("processing", `${runId}-${base.nodes.length}`),
    runId,
    content: "处理中...",
  };
  return {
    ...base,
    processing,
    runs: {
      ...base.runs,
      [runId]: {
        ...(base.runs[runId] ?? {}),
        startedAt: base.runs[runId]?.startedAt ?? eventAt,
        processingId: processing.id
      }
    }
  };
}

function clearProcessing(state: AppState): AppState {
  const hasRunProcessing = Object.values(state.runs).some((run) => run.processingId);
  if (!state.processing && !hasRunProcessing) {
    return state;
  }
  const runs = Object.fromEntries(
    Object.entries(state.runs).map(([runId, run]) => [
      runId,
      run.processingId ? { ...run, processingId: undefined } : run
    ])
  );
  return {
    ...state,
    processing: undefined,
    runs
  };
}

function hasActiveToolsForRun(state: AppState, runId: string): boolean {
  return state.nodes.some((node) => {
    const tool = node.tool;
    return tool?.runId === runId
      && (tool.status === "pending" || tool.status === "running");
  });
}

function updateTool(state: AppState, toolCallId: string, updater: (tool: ToolState) => ToolState): AppState {
  const nodeIdForTool = state.toolNodeByCallId[toolCallId];
  if (!nodeIdForTool) {
    return state;
  }
  return updateChatNode(state, nodeIdForTool, (node) => {
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
    const updated = updateChatNode(state, nodeIdForTool, (node) => {
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
    description: optionalToolPurpose(payload),
    status: "running",
    argsRaw: "",
    argsState: "complete",
    resultPreview: ""
  };
  const next = appendChatNode(state, {
    id: nodeIdForNewTool,
    kind: "tool",
    content: "",
    tool: updater(tool)
  });
  return toolCallId
    ? { ...next, toolNodeByCallId: { ...next.toolNodeByCallId, [toolCallId]: nodeIdForNewTool } }
    : next;
}

function updateChatNode(state: AppState, id: string, updater: (node: ChatNode) => ChatNode): AppState {
  const target = state.nodes.find((node) => node.id === id);
  const base = target && shouldClearProcessingForNode(target)
    ? clearProcessing(state)
    : state;
  return {
    ...base,
    nodes: base.nodes.map((node) => (node.id === id ? updater(node) : node))
  };
}

function findRunStartUserNode(
  state: AppState,
  userNodeId: string,
  contextMessageId?: string,
): ChatNode | undefined {
  const byId = state.nodes.find((node) => (
    node.kind === "user" && node.id === userNodeId
  ));
  if (byId) {
    return byId;
  }
  if (!contextMessageId) {
    return undefined;
  }
  return state.nodes.find((node) => (
    node.kind === "user" && node.contextMessageId === contextMessageId
  ));
}

function mergeRunStartUserNode(
  state: AppState,
  targetNodeId: string,
  incoming: ChatNode,
  contextMessageIndex: number,
): AppState {
  return {
    ...state,
    nodes: state.nodes.map((node) => {
      if (node.id !== targetNodeId || node.kind !== "user") {
        return node;
      }
      return {
        ...node,
        queue: node.queue ?? incoming.queue,
        displayMessageType: node.displayMessageType ?? incoming.displayMessageType,
        contextMessageId: node.contextMessageId ?? incoming.contextMessageId,
        contextMessageIndex: node.contextMessageIndex ?? contextMessageIndex,
        canFork: true,
        content: node.content || incoming.content,
        contentParts: node.contentParts ?? incoming.contentParts
      };
    })
  };
}

function appendChatNode(
  state: AppState,
  node: ChatNode,
  options: { clearProcessing?: boolean } = {},
): AppState {
  const shouldClearProcessing = options.clearProcessing ?? shouldClearProcessingForNode(node);
  const base = shouldClearProcessing
    ? clearProcessing(state)
    : state;
  return {
    ...base,
    nodes: [...base.nodes, node]
  };
}

function shouldClearProcessingForNode(node: ChatNode): boolean {
  return node.kind !== "debug";
}

function addSystem(
  state: AppState,
  content: string,
  options: { clearProcessing?: boolean } = {},
): AppState {
  return appendChatNode(state, {
    id: nodeId("system", `${Date.now()}-${state.nodes.length}`),
    kind: "system",
    content
  }, options);
}

function addMeta(state: AppState, content: string): AppState {
  return {
    ...appendChatNode(state, {
      id: nodeId("meta", `${Date.now()}-${state.metadataLines.length}`),
      kind: "meta",
      content
    }),
    metadataLines: [...state.metadataLines, content]
  };
}

function appendFrameworkInjection(state: AppState, injection: FrameworkInjectionState): AppState {
  if (releaseDedupFallbackEnabled() && hasFrameworkInjectionNode(state.nodes, injection)) {
    return state;
  }
  return appendChatNode(state, {
    id: nodeId("framework", `${injection.id}-${state.nodes.length}`),
    kind: "framework",
    content: injection.content,
    framework: injection
  }, { clearProcessing: false });
}

function appendHistoryFrameworkInjection(
  nodes: ChatNode[],
  frame: CoreFrame,
  injection: FrameworkInjectionState,
  historyIndex?: number,
): void {
  if (releaseDedupFallbackEnabled() && hasFrameworkInjectionNode(nodes, injection)) {
    return;
  }
  nodes.push(historyFrameworkNode(frame, injection, nodes.length, historyIndex));
}

function hasFrameworkInjectionNode(
  nodes: ChatNode[],
  injection: FrameworkInjectionState,
): boolean {
  return nodes.some((node) => (
    node.kind === "framework"
    && node.framework !== undefined
    && frameworkInjectionsMatch(node.framework, injection)
  ));
}

function mergeSystemPromptInjection(
  state: AppState,
  injection: FrameworkInjectionState,
): AppState {
  if (isSystemPromptInjectedSegment(injection)) {
    return attachSystemPromptChildInjection(state, injection);
  }
  const targetNodeId = findSystemPromptNodeId(state.nodes, injection.runId);
  if (!targetNodeId) {
    return appendChatNode(state, systemPromptChatNode(injection, state.nodes.length), { clearProcessing: false });
  }
  return {
    ...state,
    nodes: state.nodes.map((node) => (
      node.id === targetNodeId
        ? { ...node, content: injection.content, framework: injection }
        : node
    ))
  };
}

function attachSystemPromptChildInjection(
  state: AppState,
  injection: FrameworkInjectionState,
): AppState {
  const targetNodeId = findSystemPromptNodeId(state.nodes, injection.runId);
  if (!targetNodeId) {
    return appendChatNode(
      state,
      {
        ...systemPromptChatNode(systemPromptParentFromChild(injection), state.nodes.length),
        injections: [injection]
      },
      { clearProcessing: false }
    );
  }
  return {
    ...state,
    nodes: state.nodes.map((node) => {
      if (node.id !== targetNodeId) return node;
      return {
        ...node,
        injections: orderFrameworkInjections([...(node.injections ?? []), injection])
      };
    })
  };
}

function mergeHistorySystemPromptInjection(
  nodes: ChatNode[],
  frame: CoreFrame,
  injection: FrameworkInjectionState,
  historyIndex?: number,
): void {
  const targetIndex = findSystemPromptNodeIndex(nodes, injection.runId);
  if (isSystemPromptInjectedSegment(injection)) {
    if (targetIndex < 0) {
      nodes.push({
        ...historyFrameworkNode(frame, systemPromptParentFromChild(injection), nodes.length, historyIndex),
        injections: [injection]
      });
      return;
    }
    const target = nodes[targetIndex];
    nodes[targetIndex] = {
      ...target,
      injections: orderFrameworkInjections([...(target.injections ?? []), injection])
    };
    return;
  }
  if (targetIndex < 0) {
    nodes.push(historyFrameworkNode(frame, injection, nodes.length, historyIndex));
    return;
  }
  const target = nodes[targetIndex];
  nodes[targetIndex] = {
    ...target,
    content: injection.content,
    framework: injection
  };
}

function systemPromptChatNode(injection: FrameworkInjectionState, index: number): ChatNode {
  return {
    id: nodeId("system-prompt", `${injection.runId ?? injection.id}-${index}`),
    kind: "framework",
    content: injection.content,
    framework: injection
  };
}

function systemPromptParentFromChild(child: FrameworkInjectionState): FrameworkInjectionState {
  return {
    ...child,
    id: `${child.id}:parent`,
    label: "System prompt",
    content: "",
    pluginId: undefined,
    pluginName: undefined,
    pluginRole: "framework",
    injectionName: "system_prompt",
    hookType: undefined,
    metadata: { content_scope: "pending_full_prompt" }
  };
}

function findSystemPromptNodeId(nodes: ChatNode[], runId?: string): string | undefined {
  const index = findSystemPromptNodeIndex(nodes, runId);
  return index < 0 ? undefined : nodes[index].id;
}

function findSystemPromptNodeIndex(nodes: ChatNode[], runId?: string): number {
  for (let index = nodes.length - 1; index >= 0; index -= 1) {
    const node = nodes[index];
    if (node.kind !== "framework" || node.framework?.kind !== "system_prompt") {
      continue;
    }
    if (!runId || node.framework.runId === runId || node.injections?.some((item) => item.runId === runId)) {
      return index;
    }
  }
  return -1;
}

function isSystemPromptInjectedSegment(injection: FrameworkInjectionState): boolean {
  return injection.kind === "system_prompt"
    && optionalString(injection.metadata?.content_scope) === "injected_segment";
}

function attachFrameworkInjectionToUserMessage(
  state: AppState,
  payload: Record<string, unknown>,
  injection: FrameworkInjectionState,
): AppState {
  const targetMessageId = optionalString(payload.target_message_id);
  const targetContextMessageId = optionalString(payload.target_context_message_id);
  const targetMessageIndex = optionalNumber(payload.target_message_index);
  const targetNodeId = findUserMessageNodeId(
    state,
    targetMessageId,
    targetContextMessageId,
    targetMessageIndex
  );
  if (!targetNodeId) {
    return appendFrameworkInjection(state, injection);
  }
  return {
    ...state,
    nodes: state.nodes.map((node) => {
      if (node.id !== targetNodeId) return node;
      return {
        ...node,
        injections: orderFrameworkInjections([...(node.injections ?? []), injection])
      };
    })
  };
}

function findUserMessageNodeId(
  state: AppState,
  targetMessageId?: string,
  targetContextMessageId?: string,
  targetMessageIndex?: number,
): string | undefined {
  if (targetMessageId) {
    const id = userMessageNodeId(targetMessageId);
    if (state.nodes.some((node) => node.id === id && node.kind === "user")) {
      return id;
    }
  }
  if (targetContextMessageId) {
    const indexed = state.nodes.find((node) => (
      node.kind === "user" && node.contextMessageId === targetContextMessageId
    ));
    if (indexed) return indexed.id;
  }
  if (targetMessageIndex !== undefined) {
    const indexed = state.nodes.find((node) => (
      node.kind === "user" && node.contextMessageIndex === targetMessageIndex
    ));
    if (indexed) return indexed.id;
  }
  return [...state.nodes].reverse().find((node) => node.kind === "user")?.id;
}

function frameworkInjectionFromFrame(
  frame: CoreFrame,
  payload: Record<string, unknown>,
  kind: Exclude<FrameworkInjectionKind, "plugin_message">,
): FrameworkInjectionState {
  const metadata = isRecord(payload.metadata) ? payload.metadata : undefined;
  const timestamp = frameTime(frame);
  const pluginRole = optionalString(payload.plugin_role);
  const toolName = optionalString(payload.tool_name);
  const toolCallId = optionalString(payload.tool_call_id);
  const parameterName = optionalString(payload.parameter_name);
  return {
    id: frameworkInjectionId(frame, payload, kind),
    kind,
    label: frameworkInjectionLabel(kind, metadata),
    content: frameworkInjectionContent(payload, kind),
    runId: optionalString(payload.run_id),
    pluginId: optionalString(payload.plugin_id),
    pluginName: optionalString(payload.plugin_name),
    pluginRole,
    injectionName: optionalString(payload.injection_name),
    eventType: frame.type,
    hookType: optionalString(payload.hook_type ?? payload.origin),
    role: optionalString(payload.role),
    mergeTarget: optionalString(payload.merge_target),
    mergePosition: normalizeMergePosition(payload.merge_position),
    targetMessageId: optionalString(payload.target_message_id),
    targetContextMessageId: optionalString(payload.target_context_message_id),
    targetMessageIndex: optionalNumber(payload.target_message_index),
    contextPosition: optionalNumber(payload.position),
    toolName,
    toolCallId,
    parameterName,
    metadata,
    timestamp
  };
}

function frameworkInjectionLabel(
  kind: Exclude<FrameworkInjectionKind, "plugin_message">,
  metadata?: Record<string, unknown>,
): string {
  if (kind === "system_prompt") {
    return optionalString(metadata?.content_scope) === "injected_segment"
      ? "System prompt 注入信息"
      : "System prompt";
  }
  return "框架注入消息";
}

function pluginMessageFrameworkInjectionFromFrame(
  frame: CoreFrame,
  payload: Record<string, unknown>,
): FrameworkInjectionState | undefined {
  const plugin = pluginInfo(payload);
  const content = pluginMessageFrameworkContent(payload);
  if (!content) return undefined;
  return {
    id: frameworkInjectionId(frame, payload, "plugin_message"),
    kind: "plugin_message",
    label: "插件消息",
    content,
    pluginId: plugin.pluginId,
    pluginName: plugin.pluginName,
    pluginRole: "plugin",
    injectionName: optionalString(payload.title) ?? "plugin.message",
    eventType: frame.type,
    toolCallId: optionalString(payload.tool_call_id),
    metadata: isRecord(payload.metadata) ? payload.metadata : undefined,
    timestamp: frameTime(frame)
  };
}

function frameworkInjectionId(
  frame: CoreFrame,
  payload: Record<string, unknown>,
  kind: FrameworkInjectionKind,
): string {
  const explicitId = optionalString(payload.injection_id ?? payload.message_id ?? frame.id);
  if (explicitId) return explicitId;
  return [
    frame.type,
    kind,
    optionalString(payload.run_id) ?? "",
    optionalString(payload.plugin_id) ?? "",
    optionalString(payload.injection_name) ?? "",
    optionalString(payload.hook_type ?? payload.origin) ?? "",
    optionalString(payload.tool_call_id) ?? "",
    optionalString(payload.parameter_name) ?? "",
    optionalString(payload.target_message_id) ?? "",
    optionalString(payload.target_context_message_id) ?? "",
    optionalString(payload.merge_position) ?? "",
    String(optionalNumber(payload.target_message_index) ?? ""),
    String(optionalNumber(payload.position) ?? ""),
    String(frame.ts ?? ""),
    simpleHash(formatFrameworkIdentityContent(payload, kind))
  ].join(":");
}

function frameworkInjectionContent(
  payload: Record<string, unknown>,
  kind: Exclude<FrameworkInjectionKind, "plugin_message">,
): string {
  if (kind === "tool_runtime_context_injected") {
    const toolName = optionalString(payload.tool_name) ?? "tool";
    const parameterName = optionalString(payload.parameter_name) ?? "context";
    return `Tool \`${toolName}\` received Hawi runtime context as \`${parameterName}\`.`;
  }
  const text = optionalString(payload.text);
  if (text) return text;
  if (Array.isArray(payload.content)) {
    const contentText = historyContentText(payload.content);
    if (kind === "system_prompt") return contentText;
    return contentText || formatToolValue(payload.content).trim();
  }
  if (kind === "system_prompt") return "";
  return formatToolValue(payload.content).trim();
}

function pluginMessageFrameworkContent(payload: Record<string, unknown>): string {
  const title = optionalString(payload.title);
  const message = optionalString(payload.message);
  const data = payload.data;
  const sections: string[] = [];
  if (title && title !== message) {
    sections.push(`**${title}**`);
  }
  if (message) {
    sections.push(message);
  }
  const dataText = formatToolValue(data).trim();
  if (dataText) {
    sections.push(markdownJsonBlock(dataText));
  }
  return sections.join("\n\n");
}

function formatFrameworkIdentityContent(
  payload: Record<string, unknown>,
  kind: FrameworkInjectionKind,
): string {
  if (kind === "plugin_message") return pluginMessageFrameworkContent(payload);
  if (kind === "tool_runtime_context_injected") return optionalString(payload.parameter_name) ?? "";
  if (Array.isArray(payload.content)) {
    const contentText = historyContentText(payload.content);
    return kind === "system_prompt" ? contentText : contentText || formatToolValue(payload.content);
  }
  return optionalString(payload.text) ?? (kind === "system_prompt" ? "" : formatToolValue(payload.content));
}

function markdownJsonBlock(value: string): string {
  return `\`\`\`json\n${value}\n\`\`\``;
}

function simpleHash(value: string): string {
  let hash = 5381;
  for (let index = 0; index < value.length; index += 1) {
    hash = ((hash << 5) + hash) ^ value.charCodeAt(index);
  }
  return (hash >>> 0).toString(36);
}

function normalizeMergePosition(value: unknown): "before" | "after" | undefined {
  return value === "before" || value === "after" ? value : undefined;
}

function orderFrameworkInjections(items: FrameworkInjectionState[]): FrameworkInjectionState[] {
  const ordered = releaseDedupFallbackEnabled()
    ? uniqueFrameworkInjections(items)
    : [...items];
  return ordered.sort((left, right) => {
    const leftPosition = left.contextPosition ?? Number.MAX_SAFE_INTEGER;
    const rightPosition = right.contextPosition ?? Number.MAX_SAFE_INTEGER;
    if (leftPosition !== rightPosition) return leftPosition - rightPosition;
    return left.timestamp - right.timestamp;
  });
}

function uniqueFrameworkInjections(items: FrameworkInjectionState[]): FrameworkInjectionState[] {
  const seen = new Set<string>();
  const result: FrameworkInjectionState[] = [];
  for (const item of items) {
    const key = frameworkInjectionIdentity(item);
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    result.push(item);
  }
  return result;
}

function frameworkInjectionsMatch(
  left: FrameworkInjectionState,
  right: FrameworkInjectionState,
): boolean {
  return frameworkInjectionIdentity(left) === frameworkInjectionIdentity(right);
}

function frameworkInjectionIdentity(injection: FrameworkInjectionState): string {
  return [
    injection.kind,
    injection.eventType ?? "",
    injection.runId ?? "",
    injection.pluginId ?? "",
    injection.pluginName ?? "",
    injection.pluginRole ?? "",
    injection.injectionName ?? "",
    injection.hookType ?? "",
    injection.role ?? "",
    injection.mergeTarget ?? "",
    injection.mergePosition ?? "",
    injection.targetMessageId ?? "",
    injection.targetContextMessageId ?? "",
    String(injection.targetMessageIndex ?? ""),
    String(injection.contextPosition ?? ""),
    injection.toolName ?? "",
    injection.toolCallId ?? "",
    injection.parameterName ?? "",
    simpleHash(injection.content),
    simpleHash(formatToolValue(injection.metadata)),
  ].join(":");
}

function updateSubAgentFromEvent(
  state: AppState,
  payload: Record<string, unknown>,
  frame: CoreFrame,
): AppState {
  const eventName = frame.type.startsWith("subagent.")
    ? frame.type
    : optionalString(payload.event_name);
  if (!eventName || !eventName.startsWith("subagent.")) {
    return state;
  }
  const status = normalizeSubAgentStatus(payload.status);
  const id = optionalString(payload.subagent_id) ?? status?.id;
  if (!id) {
    return state;
  }

  const eventAt = frameTime(frame);
  const current = state.subagents[id] ?? createSubAgentRuntime(id, payload, status, eventAt);
  const childEvent = isRecord(payload.child_event) ? payload.child_event : undefined;
  const messageEntry = normalizeSubAgentMessageEntry(
    payload.message_entry,
    current.messageHistory.length
  );
  const nextHistory = messageEntry
    ? appendSubAgentMessageEntry(current.messageHistory, messageEntry)
    : current.messageHistory;
  const nextPartial = updateSubAgentPartial(current.partial, childEvent, messageEntry, eventAt);
  const nextStatus = status ?? current.status;
  const nextStateName = nextStatus?.state ?? lifecycleStateFromEvent(eventName, current.state);
  const nextMode = nextStatus?.mode ?? current.mode;
  const nextSharedContext = nextStatus?.sharedContext
    ?? current.sharedContext
    ?? nextMode === "fork";
  const nextSubagent: SubAgentRuntimeState = {
    ...current,
    name: nextStatus?.name ?? optionalString(payload.subagent_name) ?? current.name,
    role: nextStatus?.role ?? optionalString(payload.subagent_role) ?? current.role,
    state: nextStateName,
    createdAt: current.createdAt ?? nextStatus?.createdAt ?? eventAt,
    mode: nextMode,
    sharedContext: nextSharedContext,
    status: nextStatus,
    messageHistory: nextHistory,
    nodes: subAgentNodes(nextHistory, nextPartial, id, nextSharedContext),
    processing: subAgentProcessing(
      id,
      nextStatus,
      nextStateName,
      nextPartial,
      childEvent
    ),
    partial: nextPartial,
    eventCount: current.eventCount + 1,
    lastEventType: optionalString(childEvent?.type) ?? eventName,
    lastEventAt: eventAt,
  };
  return {
    ...state,
    subagents: {
      ...state.subagents,
      [id]: nextSubagent,
    },
    subagentOrder: state.subagentOrder.includes(id)
      ? state.subagentOrder
      : [...state.subagentOrder, id],
  };
}

function createSubAgentRuntime(
  id: string,
  payload: Record<string, unknown>,
  status: SubAgentStatusState | undefined,
  eventAt: number,
): SubAgentRuntimeState {
  return {
    id,
    name: status?.name ?? optionalString(payload.subagent_name) ?? id,
    role: status?.role ?? optionalString(payload.subagent_role) ?? "general",
    state: status?.state ?? "CREATED",
    createdAt: status?.createdAt ?? eventAt,
    mode: status?.mode,
    sharedContext: status?.sharedContext ?? status?.mode === "fork",
    status,
    messageHistory: [],
    nodes: [],
    partial: emptySubAgentPartial(),
    eventCount: 0,
    lastEventAt: eventAt,
  };
}

function normalizeSubAgentStatus(value: unknown): SubAgentStatusState | undefined {
  if (!isRecord(value)) return undefined;
  const id = optionalString(value.id);
  if (!id) return undefined;
  const plugins = normalizeSubAgentPlugins(value.plugins);
  const pluginIds = normalizeStringList(value.plugin_ids ?? value.pluginIds);
  const toolNames = normalizeStringList(value.tool_names ?? value.toolNames);
  return {
    id,
    name: optionalString(value.name) ?? id,
    role: optionalString(value.role) ?? "general",
    state: optionalString(value.state) ?? "CREATED",
    runnerState: optionalString(value.runner_state ?? value.runnerState) ?? "",
    executorState: optionalString(value.executor_state ?? value.executorState) ?? "",
    queueLengths: normalizeNumberRecord(value.queue_lengths ?? value.queueLengths),
    createdAt: optionalNumber(value.created_at ?? value.createdAt),
    updatedAt: optionalNumber(value.updated_at ?? value.updatedAt),
    closedAt: optionalNumber(value.closed_at ?? value.closedAt),
    modelId: optionalString(value.model_id ?? value.modelId),
    workingDir: optionalString(value.working_dir ?? value.workingDir),
    mode: optionalString(value.mode),
    sharedContext: optionalBoolean(value.shared_context ?? value.sharedContext),
    plugins,
    pluginIds: pluginIds.length > 0 ? pluginIds : plugins.map((plugin) => plugin.id),
    toolNames,
    toolCount: optionalNumber(value.tool_count ?? value.toolCount) ?? toolNames.length,
    lastResultText: optionalString(value.last_result_text ?? value.lastResultText),
    lastError: optionalString(value.last_error ?? value.lastError),
  };
}

function normalizeSubAgentPlugins(value: unknown): SubAgentPluginInfoState[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    if (!isRecord(item)) return [];
    const id = optionalString(item.id);
    if (!id) return [];
    return [{
      id,
      name: optionalString(item.name) ?? id,
      displayName: optionalString(item.display_name ?? item.displayName),
      className: optionalString(item.class_name ?? item.className),
    }];
  });
}

function normalizeStringList(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    const text = optionalString(item);
    return text ? [text] : [];
  });
}

function normalizeNumberRecord(value: unknown): Record<string, number> {
  if (!isRecord(value)) return {};
  const result: Record<string, number> = {};
  Object.entries(value).forEach(([key, item]) => {
    const numberValue = optionalNumber(item);
    if (numberValue !== undefined) {
      result[key] = numberValue;
    }
  });
  return result;
}

function lifecycleStateFromEvent(eventName: string, current: string): string {
  if (eventName === "subagent.closed") return "CLOSED";
  if (eventName === "subagent.created") return current || "CREATED";
  return current || "RUNNING";
}

function normalizeSubAgentMessageEntry(value: unknown, index: number): SessionHistoryRecord | undefined {
  if (!isRecord(value)) return undefined;
  return normalizeSessionHistoryRecord(value, index) ?? undefined;
}

function appendSubAgentMessageEntry(
  history: SessionHistoryRecord[],
  entry: SessionHistoryRecord,
): SessionHistoryRecord[] {
  const entryKey = sessionHistoryRecordKey(entry);
  if (history.some((item) => sessionHistoryRecordKey(item) === entryKey)) {
    return history;
  }
  return [...history, entry];
}

function sessionHistoryRecordKey(record: SessionHistoryRecord): string {
  return [
    record.timestamp ?? "",
    record.runId,
    record.role,
    record.contextMessageId ?? "",
    record.contextMessageIndex ?? "",
    simpleHash(formatToolValue(record.content)),
  ].join(":");
}

function updateSubAgentPartial(
  current: SubAgentPartialState,
  childEvent: Record<string, unknown> | undefined,
  messageEntry: SessionHistoryRecord | undefined,
  eventAt: number,
): SubAgentPartialState {
  if (messageEntry?.role === "assistant") {
    return emptySubAgentPartial();
  }
  const eventType = optionalString(childEvent?.type);
  if (
    eventType === "agent.run_start"
    || eventType === "model.stream_start"
    || eventType === "agent.run_stop"
    || eventType === "agent.error"
  ) {
    return emptySubAgentPartial();
  }
  if (eventType !== "model.content_block_delta") {
    return current;
  }
  const delta = optionalString(childEvent?.delta);
  if (!delta) {
    return current;
  }
  const deltaType = optionalString(childEvent?.delta_type);
  const base: SubAgentPartialState = {
    ...current,
    runId: optionalString(childEvent?.run_id) ?? current.runId,
    startedAt: current.startedAt ?? eventAt,
    updatedAt: eventAt,
  };
  if (deltaType === "reasoning") {
    return { ...base, reasoning: base.reasoning + delta };
  }
  if (deltaType === "text") {
    return { ...base, text: base.text + delta };
  }
  return current;
}

function subAgentNodes(
  history: SessionHistoryRecord[],
  partial: SubAgentPartialState,
  subagentId: string,
  sharedContext?: boolean,
): ChatNode[] {
  const nodes = sessionHistoryNodes(history);
  if (sharedContext) {
    nodes.unshift(subAgentSharedContextNode(subagentId));
  }
  if (partial.reasoning) {
    nodes.push({
      id: nodeId("subagent-thinking", `${subagentId}-${partial.runId ?? "partial"}`),
      kind: "thinking",
      content: partial.reasoning,
      complete: false,
      streamStartedAt: partial.startedAt,
    });
  }
  if (partial.text) {
    nodes.push({
      id: nodeId("subagent-agent", `${subagentId}-${partial.runId ?? "partial"}`),
      kind: "agent",
      content: partial.text,
      complete: false,
      streamStartedAt: partial.startedAt,
    });
  }
  return nodes;
}

function subAgentSharedContextNode(subagentId: string): ChatNode {
  return {
    id: nodeId("subagent-handoff", subagentId),
    kind: "handoff",
    content: "前文延续：这个 SubAgent 从主 Agent 的上下文分叉而来。上文仅作为背景，它只负责当前被赋予的任务，并会在完成后回报结果。",
  };
}

function subAgentProcessing(
  subagentId: string,
  status: SubAgentStatusState | undefined,
  stateName: string,
  partial: SubAgentPartialState,
  childEvent: Record<string, unknown> | undefined,
): ProcessingState | undefined {
  if (partial.text || partial.reasoning || !isSubAgentBusy(status, stateName)) {
    return undefined;
  }
  return {
    id: nodeId("subagent-processing", subagentId),
    runId: optionalString(childEvent?.run_id) ?? subagentId,
    content: "SubAgent 处理中...",
  };
}

function isSubAgentBusy(status: SubAgentStatusState | undefined, stateName: string): boolean {
  return stateName === "RUNNING"
    || stateName === "INTERRUPTING"
    || status?.runnerState === "RUNNING"
    || status?.executorState === "RUNNING"
    || status?.executorState === "INTERRUPTING";
}

function emptySubAgentPartial(): SubAgentPartialState {
  return {
    text: "",
    reasoning: "",
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
  const baseMessages = removeSupersededPluginMessages(state.pluginMessages, item);
  return {
    ...state,
    pluginMessages: [...baseMessages.slice(-79), item]
  };
}

function removeSupersededPluginMessages(
  messages: PluginMessageState[],
  item: PluginMessageState,
): PluginMessageState[] {
  const review = humanReviewMessageInfo(item);
  return messages.filter((existing) => {
    if (existing.id === item.id) {
      return false;
    }
    if (!review) {
      return true;
    }
    const existingReview = humanReviewMessageInfo(existing);
    if (!existingReview || existingReview.pluginId !== review.pluginId) {
      return true;
    }
    if (existingReview.reviewId === review.reviewId) {
      return false;
    }
    return !review.stepId || existingReview.stepId !== review.stepId;
  });
}

function pruneResolvedHumanReviewMessages(
  state: AppState,
  payload: Record<string, unknown>,
): AppState {
  const reviewingStepIds = reviewingTaskflowStepIds(payload);
  if (!reviewingStepIds) {
    return state;
  }
  const plugin = pluginInfo(payload);
  const pluginMessages = state.pluginMessages.filter((message) => {
    const review = humanReviewMessageInfo(message);
    if (!review || review.pluginId !== plugin.pluginId || !review.stepId) {
      return true;
    }
    return reviewingStepIds.has(review.stepId);
  });
  if (pluginMessages.length === state.pluginMessages.length) {
    return state;
  }
  return { ...state, pluginMessages };
}

function reviewingTaskflowStepIds(payload: Record<string, unknown>): Set<string> | undefined {
  const state = isRecord(payload.state) ? payload.state : undefined;
  const steps = Array.isArray(state?.steps) ? state.steps : undefined;
  if (!steps) {
    return undefined;
  }
  const ids = new Set<string>();
  collectReviewingStepIds(steps, ids);
  return ids;
}

function collectReviewingStepIds(steps: unknown[], ids: Set<string>): void {
  steps.forEach((step) => {
    if (!isRecord(step)) {
      return;
    }
    const id = optionalString(step.id);
    if (id && optionalString(step.status) === "reviewing") {
      ids.add(id);
    }
    if (Array.isArray(step.children)) {
      collectReviewingStepIds(step.children, ids);
    }
  });
}

function humanReviewMessageInfo(
  message: PluginMessageState,
): { pluginId: string; reviewId: string; stepId?: string } | undefined {
  const data = isRecord(message.data) ? message.data : undefined;
  if (!data) return undefined;

  // Taskflow/Workflow human_review_request
  if (data.kind === "human_review_request") {
    const reviewId = optionalString(data.review_id);
    if (!reviewId) return undefined;
    return {
      pluginId: optionalString(data.plugin_id) ?? message.pluginId,
      reviewId,
      stepId: optionalString(data.step_id),
    };
  }

  // Permission system review (via plugin.event)
  // event_name === "permission.review.requested" also carries
  // kind === "human_review_request" for the GUI protocol.
  return undefined;
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

function userMessageNodeId(messageId: string): string {
  return nodeId("user-message", messageId);
}

function normalizeQueue(value: unknown): QueueKind {
  return value === "normal" || value === "urgent" ? value : "high_prio";
}

function normalizeDisplayMessageType(
  value: unknown,
  queue: QueueKind = "normal"
): DisplayMessageType {
  if (value === "normal" || value === "steer" || value === "urgent" || value === "resume") return value;
  if (queue === "urgent") return "urgent";
  if (queue === "high_prio") return "steer";
  return "normal";
}

function normalizeQueueLengths(value: unknown, fallback: Record<QueueKind, number> = { normal: 0, high_prio: 0, urgent: 0 }): Record<QueueKind, number> {
  const raw = value && typeof value === "object" ? (value as Record<string, unknown>) : {};
  return {
    normal: Number(raw.normal ?? fallback.normal),
    high_prio: Number(raw.high_prio ?? fallback.high_prio),
    urgent: Number(raw.urgent ?? fallback.urgent)
  };
}

function normalizeQueueMessages(
  value: unknown,
  fallback: Record<QueueKind, QueueMessageState[]> = { normal: [], high_prio: [], urgent: [] }
): Record<QueueKind, QueueMessageState[]> {
  if (!isRecord(value)) return fallback;
  return {
    normal: normalizeQueueMessageList(value.normal, "normal"),
    high_prio: normalizeQueueMessageList(value.high_prio, "high_prio"),
    urgent: normalizeQueueMessageList(value.urgent, "urgent")
  };
}

function normalizeQueueMessageList(value: unknown, queue: QueueKind): QueueMessageState[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter(isRecord)
    .map((item, index) => {
      const id = optionalString(item.id) ?? `${queue}-${index}`;
      return {
        id,
        queue: normalizeQueue(item.queue ?? queue),
        contentPreview: optionalString(item.content_preview)
          ?? optionalString(item.contentPreview)
          ?? "",
        content: typeof item.content === "string" ? optionalString(item.content) : undefined,
        contentParts: normalizeContentParts(item.content_parts ?? item.contentParts ?? item.content),
        createdAt: optionalNumber(item.created_at ?? item.createdAt),
        metadata: isRecord(item.metadata) ? item.metadata : undefined
      };
    });
}

function trimQueueMessages(
  messages: Record<QueueKind, QueueMessageState[]>,
  lengths: Record<QueueKind, number>
): Record<QueueKind, QueueMessageState[]> {
  return {
    normal: messages.normal.slice(0, Math.max(0, lengths.normal)),
    high_prio: messages.high_prio.slice(0, Math.max(0, lengths.high_prio)),
    urgent: messages.urgent.slice(0, Math.max(0, lengths.urgent))
  };
}

function sameQueueLengths(left: Record<QueueKind, number>, right: Record<QueueKind, number>): boolean {
  return left.normal === right.normal && left.high_prio === right.high_prio && left.urgent === right.urgent;
}

function sameQueueMessages(left: Record<QueueKind, QueueMessageState[]>, right: Record<QueueKind, QueueMessageState[]>): boolean {
  return sameQueueMessageList(left.normal, right.normal)
    && sameQueueMessageList(left.high_prio, right.high_prio)
    && sameQueueMessageList(left.urgent, right.urgent);
}

function sameQueueMessageList(left: QueueMessageState[], right: QueueMessageState[]): boolean {
  if (left.length !== right.length) return false;
  return left.every((item, index) => {
    const other = right[index];
    return item.id === other.id
      && item.queue === other.queue
      && item.contentPreview === other.contentPreview
      && item.content === other.content
      && JSON.stringify(item.contentParts ?? null) === JSON.stringify(other.contentParts ?? null)
      && item.createdAt === other.createdAt;
  });
}

function formatRunStop(payload: Record<string, unknown>): string {
  const reason = String(payload.stop_reason ?? "end_turn");
  const durationMs = Number(payload.duration_ms ?? 0);
  if (!Number.isFinite(durationMs) || durationMs <= 0) {
    return reason;
  }
  return `${reason} · ${(durationMs / 1000).toFixed(1)}s`;
}

function resolveRunDurationMs(
  payload: Record<string, unknown>,
  startedAt: number | undefined,
  stoppedAt: number
): number | undefined {
  const explicitDuration = optionalNumber(payload.duration_ms);
  if (explicitDuration !== undefined && explicitDuration >= 0) {
    return explicitDuration;
  }
  if (startedAt !== undefined && Number.isFinite(startedAt) && Number.isFinite(stoppedAt)) {
    return Math.max(0, stoppedAt - startedAt);
  }
  return undefined;
}

function formatContextCompactionStop(payload: Record<string, unknown>): string {
  const status = optionalString(payload.status);
  if (status === "success") {
    return "Context compacted";
  }
  if (status === "skipped") {
    return "Context compaction skipped";
  }
  return "Context compaction failed";
}

function formatToolResultText(payload: Record<string, unknown>): string {
  if (payload.is_part === true) {
    return formatToolValue(payload.part);
  }

  // 提取输出文本；兼容旧版结构化 tool output。
  const rawOutput = payload.output;
  const toolName = optionalString(payload.tool_name);
  const formattedOutput = formatStructuredToolOutput(rawOutput, toolName);
  const output = shouldPreserveToolOutputLeadingWhitespace(rawOutput, toolName)
    ? formattedOutput.replace(/\s+$/, "")
    : formattedOutput.trim();

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

function formatStructuredToolOutput(value: unknown, toolName?: string): string {
  const canonicalToolName = toolName?.includes("__")
    ? toolName.slice(toolName.lastIndexOf("__") + 2)
    : toolName;
  if (!isRecord(value)) {
    return formatToolValue(value);
  }
  if (value.type === "ls_output" || value.type === "directory") {
    return String(value.text ?? "");
  }
  if (value.type === "file_unchanged" && isRecord(value.file)) {
    const filePath = optionalString(value.file.filePath);
    return filePath ? `File unchanged: ${filePath}` : "File unchanged";
  }
  if (value.type === "text" && isRecord(value.file)) {
    return String(value.file.content ?? "");
  }
  if (canonicalToolName === "grep" && typeof value.content === "string") {
    return value.content || "No matches.";
  }
  if (canonicalToolName === "glob" && Array.isArray(value.matches)) {
    return value.matches.length > 0
      ? value.matches.map((item) => String(item)).join("\n")
      : "No matches.";
  }
  return formatToolValue(value);
}

function shouldPreserveToolOutputLeadingWhitespace(value: unknown, toolName?: string): boolean {
  const canonicalToolName = toolName?.includes("__")
    ? toolName.slice(toolName.lastIndexOf("__") + 2)
    : toolName;
  if (canonicalToolName === "grep" || canonicalToolName === "glob") {
    return true;
  }
  if (!isRecord(value)) {
    return false;
  }
  return value.type === "ls_output" || value.type === "directory" || value.type === "text";
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
  if (
    !isRecord(value)
    || !(TOOL_CALL_PURPOSE_PARAMETER in value)
  ) {
    return { arguments: value };
  }
  const description = optionalString(value[TOOL_CALL_PURPOSE_PARAMETER]);
  const visible = { ...value };
  delete visible[TOOL_CALL_PURPOSE_PARAMETER];
  return { arguments: visible, description };
}

function optionalToolPurpose(payload: Record<string, unknown>): string | undefined {
  return optionalString(payload.tool_call_purpose);
}

function optionalString(value: unknown): string | undefined {
  if (value == null) return undefined;
  const text = String(value).trim();
  return text ? text : undefined;
}

function optionalNumber(value: unknown): number | undefined {
  if (value == null || value === "") return undefined;
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue : undefined;
}

function optionalBoolean(value: unknown): boolean | undefined {
  if (typeof value === "boolean") return value;
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase();
    if (normalized === "true") return true;
    if (normalized === "false") return false;
  }
  return undefined;
}

function normalizeContentParts(value: unknown): ContentPart[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const parts = value
    .filter(isRecord)
    .filter((part) => typeof part.type === "string")
    .map((part) => ({ ...part } as ContentPart));
  return parts.length ? parts : undefined;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
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
