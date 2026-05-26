import path from "node:path";
import type {
  CoreCommandType,
  CoreFrame,
  InspectPayload,
  PersistedConfig,
  SessionLaunchProfile,
  SessionLoadState,
  SessionMetaPayload,
} from "../shared/protocol";
import { VERSION } from "../shared/protocol";
import { sanitizeConfig, type EngineLauncher } from "./config";
import { CoreCommandError, CoreProcess, DEFAULT_COMMAND_TIMEOUT_MS, type EmitToRenderer } from "./core-process";
import { toolCallPurposeEngineArgs } from "./tool-parameters";

export const MAX_LOADED_SESSIONS = 5;
const SESSION_COMMAND_TIMEOUT_MS = 30_000;
const FORK_COMMAND_TIMEOUT_MS = 60_000;
const COMPACT_COMMAND_TIMEOUT_MS = 120_000;

interface EngineRecord {
  sessionId: string;
  core: CoreProcess;
  launchProfile: SessionLaunchProfile;
  workspaceRoot: string;
  loadedAt: number;
  lastFinishedAt?: number;
  hasVisibleMessages: boolean;
  agentState: string;
  runnerState: string;
  suppressEvents: boolean;
  stopping: boolean;
}

export interface ManagerSnapshot {
  currentSessionId: string | null;
  currentWorkspaceRoot: string;
  runningSessionCount: number;
  loadedSessionCount: number;
  maxLoadedSessions: number;
  coreRunning: boolean;
}

export class SessionEngineManager {
  private readonly loaded = new Map<string, EngineRecord>();
  private currentSessionId: string | null = null;
  private metadata: InspectPayload | null = null;
  private defaultConfig: PersistedConfig | null = null;
  private refreshedProviders = new Set<string>();
  private enforcingLimit = false;
  private readonlyCore: CoreProcess | null = null;

  constructor(
    private readonly emitToRenderer: EmitToRenderer,
    private readonly repoRoot: string,
    private workspaceRoot: string,
    private readonly backendLogPath: string,
    private readonly engineLauncher: EngineLauncher,
  ) {}

  configure(metadata: InspectPayload, config: PersistedConfig, refreshedProviders: Iterable<string> = []): void {
    this.metadata = metadata;
    this.defaultConfig = config;
    this.refreshedProviders = new Set(refreshedProviders);
  }

  snapshot(): ManagerSnapshot {
    return {
      currentSessionId: this.currentSessionId,
      currentWorkspaceRoot: this.currentWorkspaceRoot(),
      runningSessionCount: this.runningSessionCount(),
      loadedSessionCount: this.visibleLoadedSessionCount(),
      maxLoadedSessions: MAX_LOADED_SESSIONS,
      coreRunning: this.loaded.size > 0,
    };
  }

  async startInitial(config: PersistedConfig, metadata: InspectPayload, refreshedProviders: Iterable<string> = []): Promise<void> {
    this.configure(metadata, config, refreshedProviders);
    if (!config.modelName || this.loaded.size > 0) {
      return;
    }
    const sessionId = generateSessionId();
    const profile = profileFromConfig(config);
    this.currentSessionId = sessionId;
    this.startRecord(sessionId, profile, {
      initialSessionId: sessionId,
      initialSessionName: sessionId,
    });
    this.emitSessionRuntimeStatus(sessionId);
  }

  async stopAll(reason: string): Promise<void> {
    const records = [...this.loaded.values()];
    await Promise.all(records.map((record) => this.stopRecord(record, reason)));
    const readonlyCore = this.readonlyCore;
    this.readonlyCore = null;
    await readonlyCore?.stop(reason);
  }

  getCurrentWorkspaceRoot(): string {
    return this.currentWorkspaceRoot();
  }

  async restartCurrent(config: PersistedConfig): Promise<void> {
    this.defaultConfig = config;
    const sessionId = this.currentSessionId ?? generateSessionId();
    const previous = this.loaded.get(sessionId);
    if (previous) {
      await this.stopRecord(previous, "restart");
    }
    const profile = profileFromConfig(config);
    this.currentSessionId = sessionId;
    const record = this.startRecord(sessionId, profile, {});
    try {
      await record.core.sendCommand("session_load", { session_id: sessionId }, SESSION_COMMAND_TIMEOUT_MS);
    } catch (error) {
      await this.stopRecord(record, "restart-load-failed");
      if (isMissingSessionError(error)) {
        this.startRecord(sessionId, profile, {
          initialSessionId: sessionId,
          initialSessionName: sessionId,
        });
        this.emitSessionRuntimeStatus(sessionId);
        return;
      }
      throw error;
    }
    record.hasVisibleMessages = true;
    this.emitSessionRuntimeStatus(sessionId);
    void this.enforceLoadedLimit();
  }

  async sendCommand(type: CoreCommandType, payload: Record<string, unknown>, targetSessionId?: string | null): Promise<CoreFrame> {
    switch (type) {
      case "session_list":
        return this.sessionListFrame();
      case "session_new":
        return this.createSession(payload);
      case "session_fork":
        return this.forkSession(payload);
      case "session_switch":
      case "session_load":
        return this.switchSession(payload);
      case "session_delete":
        return this.deleteSession(payload);
      case "session_rename":
        return this.renameSession(payload);
      case "change_cwd":
        return this.changeWorkspace(payload);
      default:
        if (isReadOnlyCommand(type, payload)) {
          return this.readonlyCommand(type, payload);
        }
        return this.routeCommand(type, payload, targetSessionId);
    }
  }

  async refreshModels(provider: string): Promise<CoreFrame | null> {
    const record = this.currentRecord() ?? [...this.loaded.values()][0];
    if (!record) {
      return null;
    }
    return record.core.sendCommand("refresh_models", { provider }, 60_000);
  }

  private async routeCommand(type: CoreCommandType, payload: Record<string, unknown>, targetSessionId?: string | null): Promise<CoreFrame> {
    const sessionId = targetSessionId || stringOrNull(payload.session_id) || this.currentSessionId;
    if (!sessionId) {
      throw new Error("No active session");
    }
    const record = this.loaded.get(sessionId);
    if (!record) {
      if (type === "session_history" || type === "session_export_markdown") {
        const catalog = await this.catalogRecord();
        return catalog.core.sendCommand(
          type,
          { ...payload, session_id: stringOrNull(payload.session_id) ?? sessionId },
          commandTimeout(type),
        );
      }
      throw new Error(`Session is not loaded: ${sessionId}`);
    }

    const frame = await record.core.sendCommand(type, payload, commandTimeout(type));
    if (type === "set_system_prompt") {
      const systemPrompt = stringOrNull(payload.system_prompt);
      if (systemPrompt !== null) {
        record.launchProfile = { ...record.launchProfile, systemPrompt };
        this.syncDefaultConfigFromProfile(record.launchProfile);
        await this.saveSessionProfile(record);
      }
    } else if (type === "switch_model") {
      const modelName = stringOrNull(payload.model_name);
      if (modelName !== null) {
        record.launchProfile = { ...record.launchProfile, modelName };
        this.syncDefaultConfigFromProfile(record.launchProfile);
        await this.saveSessionProfile(record);
      }
    } else if (type === "apply_plugins") {
      record.launchProfile = {
        ...record.launchProfile,
        selectedPlugins: stringList(payload.selected_plugins),
        pluginConfigs: pluginConfigRecord(payload.plugin_configs),
      };
      this.syncDefaultConfigFromProfile(record.launchProfile);
      await this.saveSessionProfile(record);
    }
    this.emitSessionRuntimeStatus(sessionId);
    return frame;
  }

  private async createSession(payload: Record<string, unknown>): Promise<CoreFrame> {
    const workspaceRoot = this.currentWorkspaceRoot();
    await this.saveCurrentSession();
    await this.discardCurrentEmptySession();
    const sessionId = generateSessionId();
    const name = stringOrNull(payload.name) ?? sessionId;
    const profile = launchProfileFromUnknown(payload.gui_launch_profile) ??
      launchProfileFromUnknown(payload.launch_profile) ??
      profileFromConfig(this.requireDefaultConfig());
    this.syncDefaultConfigFromProfile(profile);
    this.currentSessionId = sessionId;
    this.startRecord(sessionId, profile, {
      initialSessionId: sessionId,
      initialSessionName: name,
      workspaceRoot,
    });
    this.emitSessionRuntimeStatus(sessionId);
    void this.enforceLoadedLimit();
    return ackFrame("session_new", { session_id: sessionId, name });
  }

  private async changeWorkspace(payload: Record<string, unknown>): Promise<CoreFrame> {
    const target = normalizeWorkspaceRoot(payload.cwd ?? payload.path);
    if (!target) {
      throw new Error("'cwd' is required");
    }
    const previousWorkspaceRoot = this.currentWorkspaceRoot();
    if (sameWorkspaceRoot(previousWorkspaceRoot, target)) {
      return ackFrame("change_cwd", {
        session_id: this.currentSessionId,
        workspace_switched: false,
        previous_cwd: previousWorkspaceRoot,
        last_cwd: target,
      });
    }

    await this.saveCurrentSession();
    await this.discardCurrentEmptySession();
    const readonlyCore = this.readonlyCore;
    this.readonlyCore = null;
    await readonlyCore?.stop("change-cwd");

    this.workspaceRoot = target;
    const sessionId = generateSessionId();
    const profile = profileFromConfig(this.requireDefaultConfig());
    this.currentSessionId = sessionId;
    this.startRecord(sessionId, profile, {
      initialSessionId: sessionId,
      initialSessionName: sessionId,
      workspaceRoot: target,
    });
    this.emitSessionRuntimeStatus(sessionId);
    this.emitWorkspaceChanged(
      sessionId,
      previousWorkspaceRoot,
      target,
      `已切换工作目录：${previousWorkspaceRoot} -> ${target}`,
    );
    void this.enforceLoadedLimit();
    return ackFrame("change_cwd", {
      session_id: sessionId,
      workspace_switched: true,
      previous_cwd: previousWorkspaceRoot,
      last_cwd: target,
    });
  }

  private async forkSession(payload: Record<string, unknown>): Promise<CoreFrame> {
    const sourceSessionId = stringOrNull(payload.session_id) ?? this.currentSessionId;
    if (!sourceSessionId) {
      throw new Error("No session available to fork");
    }
    await this.saveCurrentSession();
    const sourceMeta = await this.findSessionMeta(sourceSessionId);
    const sourceProfile =
      launchProfileFromUnknown(sourceMeta?.gui_launch_profile) ??
      this.loaded.get(sourceSessionId)?.launchProfile ??
      profileFromConfig(this.requireDefaultConfig());
    const sourceWorkspaceRoot =
      workspaceRootFromMeta(sourceMeta) ??
      this.loaded.get(sourceSessionId)?.workspaceRoot ??
      this.currentWorkspaceRoot();
    const provisionalId = `forking-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
    const record = this.startRecord(provisionalId, sourceProfile, {
      suppressEvents: true,
      workspaceRoot: sourceWorkspaceRoot,
    });
    let frame: CoreFrame;
    try {
      frame = await record.core.sendCommand("session_fork", payload, FORK_COMMAND_TIMEOUT_MS);
    } catch (error) {
      await this.stopRecord(record, "fork-failed");
      throw error;
    }
    const forkedSessionId = stringOrNull(framePayload(frame).session_id);
    if (!forkedSessionId) {
      await this.stopRecord(record, "fork-missing-session-id");
      throw new Error("Fork did not return a session id");
    }
    this.loaded.delete(provisionalId);
    record.sessionId = forkedSessionId;
    record.suppressEvents = false;
    record.loadedAt = Date.now();
    record.hasVisibleMessages = true;
    this.loaded.set(forkedSessionId, record);
    this.currentSessionId = forkedSessionId;
    this.emitSessionRuntimeStatus(forkedSessionId);
    void this.enforceLoadedLimit();
    return frame;
  }

  private async switchSession(payload: Record<string, unknown>): Promise<CoreFrame> {
    const sessionId = stringOrNull(payload.session_id);
    if (!sessionId) {
      throw new Error("'session_id' is required");
    }
    if (this.loaded.has(sessionId)) {
      const previousWorkspaceRoot = this.currentWorkspaceRoot();
      const record = this.loaded.get(sessionId);
      await this.saveCurrentSession();
      if (sessionId !== this.currentSessionId) {
        await this.discardCurrentEmptySession();
      }
      this.currentSessionId = sessionId;
      this.emitSessionRuntimeStatus(sessionId);
      if (record && !sameWorkspaceRoot(previousWorkspaceRoot, record.workspaceRoot)) {
        this.emitWorkspaceChanged(sessionId, previousWorkspaceRoot, record.workspaceRoot);
      }
      void this.enforceLoadedLimit();
      return ackFrame("session_switch", {
        session_id: sessionId,
        already_loaded: true,
        workspace_switched: Boolean(record && !sameWorkspaceRoot(previousWorkspaceRoot, record.workspaceRoot)),
        previous_cwd: previousWorkspaceRoot,
        last_cwd: record?.workspaceRoot ?? previousWorkspaceRoot,
      });
    }

    const meta = await this.findSessionMeta(sessionId);
    const previousWorkspaceRoot = this.currentWorkspaceRoot();
    await this.saveCurrentSession();
    await this.discardCurrentEmptySession();
    const targetWorkspaceRoot = workspaceRootFromMeta(meta) ?? previousWorkspaceRoot;
    const workspaceSwitched = !sameWorkspaceRoot(previousWorkspaceRoot, targetWorkspaceRoot);
    const profile = launchProfileFromUnknown(meta?.gui_launch_profile) ?? profileFromConfig(this.requireDefaultConfig());
    const record = this.startRecord(sessionId, profile, { workspaceRoot: targetWorkspaceRoot });
    let frame: CoreFrame;
    try {
      frame = await record.core.sendCommand("session_load", { session_id: sessionId }, SESSION_COMMAND_TIMEOUT_MS);
    } catch (error) {
      await this.stopRecord(record, "load-failed");
      throw error;
    }
    record.hasVisibleMessages = true;
    this.currentSessionId = sessionId;
    this.emitSessionRuntimeStatus(sessionId);
    if (workspaceSwitched) {
      this.emitWorkspaceChanged(sessionId, previousWorkspaceRoot, targetWorkspaceRoot);
    }
    void this.enforceLoadedLimit();
    return {
      ...frame,
      payload: {
        ...framePayload(frame),
        command: "session_switch",
        session_id: sessionId,
        workspace_switched: workspaceSwitched,
        previous_cwd: previousWorkspaceRoot,
        last_cwd: targetWorkspaceRoot,
      },
    };
  }

  private async deleteSession(payload: Record<string, unknown>): Promise<CoreFrame> {
    const sessionId = stringOrNull(payload.session_id);
    if (!sessionId) {
      throw new Error("'session_id' is required");
    }
    const loaded = this.loaded.get(sessionId);
    if (loaded && isRunningAgentState(loaded.agentState)) {
      throw new Error("Cannot delete a running session.");
    }
    const wasCurrent = sessionId === this.currentSessionId;
    const nextCurrent = wasCurrent ? this.nextCurrentSessionIdAfterDelete(sessionId) : this.currentSessionId;
    if (loaded) {
      await this.stopRecord(loaded, "delete-session");
    }
    if (wasCurrent) {
      this.currentSessionId = nextCurrent;
      if (nextCurrent) {
        this.emitSessionRuntimeStatus(nextCurrent);
      }
    }
    const catalog = await this.catalogRecord();
    const frame = await catalog.core.sendCommand("session_delete", { session_id: sessionId }, SESSION_COMMAND_TIMEOUT_MS);
    return {
      ...frame,
      payload: {
        ...framePayload(frame),
        session_id: sessionId,
        current_session_id: this.currentSessionId,
        running_session_count: this.runningSessionCount(),
        loaded_session_count: this.visibleLoadedSessionCount(),
        max_loaded_sessions: MAX_LOADED_SESSIONS,
      },
    };
  }

  private async renameSession(payload: Record<string, unknown>): Promise<CoreFrame> {
    const sessionId = stringOrNull(payload.session_id);
    const name = stringOrNull(payload.name)?.trim();
    if (!sessionId) {
      throw new Error("'session_id' is required");
    }
    if (!name) {
      throw new Error("'name' is required");
    }
    const record = this.loaded.get(sessionId);
    const target = record ?? await this.catalogRecord();
    const frame = await target.core.sendCommand("session_rename", { session_id: sessionId, name }, SESSION_COMMAND_TIMEOUT_MS);
    this.emitSessionRuntimeStatus(sessionId);
    return {
      ...frame,
      payload: {
        ...framePayload(frame),
        session_id: sessionId,
        name,
        current_session_id: this.currentSessionId,
        running_session_count: this.runningSessionCount(),
        loaded_session_count: this.visibleLoadedSessionCount(),
        max_loaded_sessions: MAX_LOADED_SESSIONS,
      },
    };
  }

  private startRecord(
    sessionId: string,
    launchProfile: SessionLaunchProfile,
    options: {
      initialSessionId?: string;
      initialSessionName?: string;
      suppressEvents?: boolean;
      workspaceRoot?: string;
    },
  ): EngineRecord {
    const existing = this.loaded.get(sessionId);
    if (existing) {
      return existing;
    }
    const metadata = this.requireMetadata();
    const config = configFromProfile(launchProfile, this.requireDefaultConfig(), metadata);
    const workspaceRoot = normalizeWorkspaceRoot(options.workspaceRoot) ?? this.currentWorkspaceRoot();
    const record = {} as EngineRecord;
    const core = new CoreProcess(
      (channel, payload) => this.handleEngineEmit(record, channel, payload),
      this.repoRoot,
      workspaceRoot,
      this.logPathForSession(sessionId, workspaceRoot),
      this.engineLauncher,
    );
    Object.assign(record, {
      sessionId,
      core,
      launchProfile,
      workspaceRoot,
      loadedAt: Date.now(),
      hasVisibleMessages: false,
      agentState: "IDLE",
      runnerState: "IDLE",
      suppressEvents: Boolean(options.suppressEvents),
      stopping: false,
    });
    this.loaded.set(sessionId, record);
    record.core.start(config, metadata, this.refreshedProviders, {
      initialSessionId: options.initialSessionId,
      initialSessionName: options.initialSessionName,
      launchProfile,
    });
    return record;
  }

  private async stopRecord(record: EngineRecord, reason: string): Promise<void> {
    if (record.stopping) {
      return;
    }
    record.stopping = true;
    try {
      if (record.core.isRunning()) {
        await record.core.sendCommand("session_save_now", {}, SESSION_COMMAND_TIMEOUT_MS).catch(() => undefined);
      }
      await record.core.stop(reason);
    } finally {
      if (this.loaded.get(record.sessionId) === record) {
        this.loaded.delete(record.sessionId);
      }
      this.emitSessionRuntimeStatus(record.sessionId, "unloaded");
    }
  }

  private async saveSessionProfile(record: EngineRecord): Promise<void> {
    if (!record.core.isRunning()) {
      return;
    }
    await record.core.sendCommand("session_save_now", {}, SESSION_COMMAND_TIMEOUT_MS).catch(() => undefined);
  }

  private async saveCurrentSession(): Promise<void> {
    const current = this.currentRecord();
    if (!current?.core.isRunning()) {
      return;
    }
    await current.core.sendCommand("session_save_now", {}, SESSION_COMMAND_TIMEOUT_MS).catch(() => undefined);
  }

  private async discardCurrentEmptySession(): Promise<void> {
    const current = this.currentRecord();
    if (!current || current.hasVisibleMessages || isRunningAgentState(current.agentState)) {
      return;
    }
    this.currentSessionId = null;
    await this.stopRecord(current, "replace-empty-session");
  }

  private handleEngineEmit(record: EngineRecord, channel: string, payload: unknown): void {
    if (channel === "core:event" && isCoreFrame(payload)) {
      if (record.suppressEvents) {
        return;
      }
      const frame = injectSessionId(payload, record.sessionId);
      this.updateRecordFromFrame(record, frame);
      this.emitToRenderer("core:event", frame);
      this.emitSessionRuntimeStatus(record.sessionId);
      if (frame.type === "run.stop" || frame.type === "core.status") {
        void this.enforceLoadedLimit();
      }
      return;
    }
    if (channel === "core:stderr") {
      this.emitToRenderer("core:stderr", `[${shortSessionId(record.sessionId)}] ${String(payload)}`);
      return;
    }
    if (channel === "core:spawn" && isRecord(payload)) {
      this.emitToRenderer("core:spawn", { ...payload, session_id: record.sessionId });
      return;
    }
    if (channel === "core:exit" && isRecord(payload)) {
      if (this.loaded.get(record.sessionId) === record) {
        this.loaded.delete(record.sessionId);
      }
      this.emitToRenderer("core:exit", { ...payload, session_id: record.sessionId });
      this.emitSessionRuntimeStatus(record.sessionId, "unloaded");
      return;
    }
    this.emitToRenderer(channel, payload);
  }

  private updateRecordFromFrame(record: EngineRecord, frame: CoreFrame): void {
    const payload = framePayload(frame);
    if (frame.type === "core.ready" && isRecord(payload.status)) {
      record.agentState = String(payload.status.agent_state ?? record.agentState);
      record.runnerState = String(payload.status.runner_state ?? record.runnerState);
    } else if (frame.type === "core.status") {
      record.agentState = String(payload.agent_state ?? record.agentState);
      record.runnerState = String(payload.runner_state ?? record.runnerState);
    } else if (frame.type === "run.start") {
      record.agentState = "RUNNING";
      record.hasVisibleMessages = true;
    } else if (frame.type === "run.stop") {
      record.agentState = "IDLE";
      record.runnerState = "IDLE";
      record.lastFinishedAt = Date.now();
    }
  }

  private async sessionListFrame(): Promise<CoreFrame> {
    const baseSessions = await this.readSessionCatalog().catch(() => []);
    const byId = new Map<string, SessionMetaPayload>();
    for (const session of baseSessions) {
      byId.set(session.session_id, {
        ...session,
        load_state: "unloaded",
        gui_launch_profile: launchProfileFromUnknown(session.gui_launch_profile),
      });
    }
    for (const record of this.loaded.values()) {
      const existing = byId.get(record.sessionId);
      if (!existing && !record.hasVisibleMessages) {
        continue;
      }
      if (existing) {
        record.hasVisibleMessages = true;
      }
      byId.set(record.sessionId, {
        session_id: record.sessionId,
        name: existing?.name || record.sessionId,
        created_at: existing?.created_at || new Date(record.loadedAt).toISOString(),
        updated_at: existing?.updated_at || new Date(record.loadedAt).toISOString(),
        last_checkpoint_event: existing?.last_checkpoint_event ?? null,
        components_present: existing?.components_present ?? [],
        ...existing,
        locked: false,
        lock_owner: null,
        load_state: loadStateForRecord(record),
        loaded_at: record.loadedAt,
        last_finished_at: record.lastFinishedAt,
        gui_launch_profile: record.launchProfile,
        last_cwd: record.workspaceRoot,
      });
    }
    return ackFrame("session_list", {
      sessions: [...byId.values()].sort(compareSessionsByCreatedAt),
      current_session_id: this.currentSessionId,
      running_session_count: this.runningSessionCount(),
      loaded_session_count: this.visibleLoadedSessionCount(),
      max_loaded_sessions: MAX_LOADED_SESSIONS,
    });
  }

  private async readSessionCatalog(): Promise<SessionMetaPayload[]> {
    const frame = await this.readonlyCommand("session_list", {});
    const payload = framePayload(frame);
    return Array.isArray(payload.sessions)
      ? payload.sessions.map(normalizeSessionMeta).filter((item): item is SessionMetaPayload => item !== null)
      : [];
  }

  private async findSessionMeta(sessionId: string): Promise<SessionMetaPayload | null> {
    const sessions = await this.readSessionCatalog().catch(() => []);
    return sessions.find((session) => session.session_id === sessionId) ?? null;
  }

  private async catalogRecord(): Promise<EngineRecord> {
    const current = this.currentRecord();
    if (current) {
      return current;
    }
    const first = [...this.loaded.values()][0];
    if (first) {
      return first;
    }
    const sessionId = `catalog-${generateSessionId()}`;
    return this.startRecord(sessionId, profileFromConfig(this.requireDefaultConfig()), {
      suppressEvents: true,
      workspaceRoot: this.currentWorkspaceRoot(),
    });
  }

  private async readonlyCommand(type: CoreCommandType, payload: Record<string, unknown>): Promise<CoreFrame> {
    const core = this.ensureReadonlyCore();
    const nextPayload = { ...payload };
    delete nextPayload.read_only;
    return core.sendCommand(type, nextPayload, commandTimeout(type));
  }

  private ensureReadonlyCore(): CoreProcess {
    if (this.readonlyCore?.isRunning()) {
      return this.readonlyCore;
    }
    const core = new CoreProcess(
      (channel, payload) => this.handleReadonlyEmit(channel, payload),
      this.repoRoot,
      this.currentWorkspaceRoot(),
      this.readonlyLogPath(),
      this.engineLauncher,
    );
    core.startReadonly();
    this.readonlyCore = core;
    return core;
  }

  private handleReadonlyEmit(channel: string, payload: unknown): void {
    if (channel === "core:stderr") {
      this.emitToRenderer("core:stderr", `[readonly] ${String(payload)}`);
      return;
    }
    if (channel === "core:spawn" && isRecord(payload)) {
      this.emitToRenderer("core:spawn", { ...payload, mode: "readonly" });
      return;
    }
    if (channel === "core:exit" && isRecord(payload)) {
      this.readonlyCore = null;
      this.emitToRenderer("core:exit", { ...payload, mode: "readonly" });
      return;
    }
    if (channel === "core:event" && isCoreFrame(payload)) {
      if (payload.type === "error") {
        this.emitToRenderer("core:event", payload);
      }
      return;
    }
    this.emitToRenderer(channel, payload);
  }

  private currentRecord(): EngineRecord | null {
    return this.currentSessionId ? (this.loaded.get(this.currentSessionId) ?? null) : null;
  }

  private nextCurrentSessionIdAfterDelete(deletedSessionId: string): string | null {
    return (
      [...this.loaded.values()]
        .filter((record) => record.sessionId !== deletedSessionId && record.hasVisibleMessages)
        .sort((a, b) => b.loadedAt - a.loadedAt)[0]?.sessionId ?? null
    );
  }

  private async enforceLoadedLimit(): Promise<void> {
    if (this.enforcingLimit) {
      return;
    }
    this.enforcingLimit = true;
    try {
      while (this.loaded.size > MAX_LOADED_SESSIONS) {
        const candidate = [...this.loaded.values()]
          .filter(
            (record) => record.sessionId !== this.currentSessionId && !isRunningAgentState(record.agentState) && record.core.isRunning(),
          )
          .sort((a, b) => (a.lastFinishedAt ?? a.loadedAt) - (b.lastFinishedAt ?? b.loadedAt))[0];
        if (!candidate) {
          return;
        }
        await this.stopRecord(candidate, "max-loaded-sessions");
      }
    } finally {
      this.enforcingLimit = false;
    }
  }

  private emitSessionRuntimeStatus(sessionId: string, overrideState?: SessionLoadState): void {
    const record = this.loaded.get(sessionId);
    this.emitToRenderer("core:event", {
      version: VERSION,
      type: "gui.session_status",
      payload: {
        session_id: sessionId,
        load_state: overrideState ?? (record ? loadStateForRecord(record) : "unloaded"),
        loaded_at: record?.loadedAt,
        last_finished_at: record?.lastFinishedAt,
        has_visible_messages: record?.hasVisibleMessages ?? false,
        last_cwd: record?.workspaceRoot,
        current_session_id: this.currentSessionId,
        running_session_count: this.runningSessionCount(),
        loaded_session_count: this.visibleLoadedSessionCount(),
        max_loaded_sessions: MAX_LOADED_SESSIONS,
      },
    });
  }

  private runningSessionCount(): number {
    return [...this.loaded.values()].filter((record) => isRunningAgentState(record.agentState)).length;
  }

  private visibleLoadedSessionCount(): number {
    return [...this.loaded.values()].filter((record) => record.hasVisibleMessages).length;
  }

  private currentWorkspaceRoot(): string {
    return this.currentRecord()?.workspaceRoot ?? this.workspaceRoot;
  }

  private emitWorkspaceChanged(sessionId: string, previousWorkspaceRoot: string, nextWorkspaceRoot: string, message?: string): void {
    this.emitToRenderer("core:event", {
      version: VERSION,
      type: "gui.workspace_changed",
      payload: {
        session_id: sessionId,
        previous_cwd: previousWorkspaceRoot,
        last_cwd: nextWorkspaceRoot,
        message: message ?? `已根据 Session 记录切换工作目录：${previousWorkspaceRoot} -> ${nextWorkspaceRoot}`,
      },
    });
  }

  private logPathForSession(sessionId: string, workspaceRoot = this.workspaceRoot): string {
    const parsed = path.parse(this.backendLogPath);
    return path.join(
      workspaceRoot,
      ".hawi",
      `${parsed.name}-${safeFilename(sessionId)}${parsed.ext || ".log"}`,
    );
  }

  private readonlyLogPath(): string {
    const parsed = path.parse(this.backendLogPath);
    return path.join(
      this.currentWorkspaceRoot(),
      ".hawi",
      `${parsed.name}-readonly${parsed.ext || ".log"}`,
    );
  }

  private requireMetadata(): InspectPayload {
    if (!this.metadata) {
      throw new Error("GUI metadata is not ready");
    }
    return this.metadata;
  }

  private requireDefaultConfig(): PersistedConfig {
    if (!this.defaultConfig) {
      throw new Error("GUI config is not ready");
    }
    return this.defaultConfig;
  }

  private syncDefaultConfigFromProfile(profile: SessionLaunchProfile): void {
    this.defaultConfig = configFromProfile(profile, this.requireDefaultConfig(), this.requireMetadata());
  }
}

export function profileFromConfig(config: PersistedConfig): SessionLaunchProfile {
  const profile = {
    version: 1,
    modelName: config.modelName,
    systemPrompt: config.systemPrompt,
    selectedPlugins: [...config.selectedPlugins],
    pluginConfigs: clonePluginConfigs(config.pluginConfigs),
    toolCallPurposeEnabled: config.toolCallPurposeEnabled,
    engineArgs: stableEngineArgs(config),
  } satisfies SessionLaunchProfile;
  return profile;
}

export function configFromProfile(
  profile: SessionLaunchProfile,
  defaultConfig: PersistedConfig,
  metadata: InspectPayload,
): PersistedConfig {
  return sanitizeConfig(
    {
      ...defaultConfig,
      modelName: profile.modelName || defaultConfig.modelName,
      systemPrompt: profile.systemPrompt || defaultConfig.systemPrompt,
      selectedPlugins: [...profile.selectedPlugins],
      pluginConfigs: clonePluginConfigs(profile.pluginConfigs),
      toolCallPurposeEnabled: profile.toolCallPurposeEnabled !== false,
      showDebug: defaultConfig.showDebug,
    },
    metadata,
  );
}

export function launchProfileFromUnknown(value: unknown): SessionLaunchProfile | null {
  if (!isRecord(value)) {
    return null;
  }
  const modelName = stringOrNull(value.modelName);
  const systemPrompt = stringOrNull(value.systemPrompt);
  if (!modelName || systemPrompt === null) {
    return null;
  }
  return {
    version: 1,
    modelName,
    systemPrompt,
    selectedPlugins: stringList(value.selectedPlugins),
    pluginConfigs: pluginConfigRecord(value.pluginConfigs),
    toolCallPurposeEnabled: value.toolCallPurposeEnabled !== false,
    engineArgs: Array.isArray(value.engineArgs) ? value.engineArgs.filter((item): item is string => typeof item === "string") : undefined,
  };
}

function stableEngineArgs(config: PersistedConfig): string[] {
  return [
    "--model",
    config.modelName,
    "--transport",
    "stdio",
    "--system-prompt",
    config.systemPrompt,
    "--plugins",
    config.selectedPlugins.join(","),
    ...toolCallPurposeEngineArgs(config.toolCallPurposeEnabled),
  ];
}

function normalizeSessionMeta(value: unknown): SessionMetaPayload | null {
  if (!isRecord(value)) {
    return null;
  }
  const sessionId = stringOrNull(value.session_id);
  if (!sessionId) {
    return null;
  }
  return {
    session_id: sessionId,
    name: stringOrNull(value.name) ?? sessionId,
    created_at: stringOrNull(value.created_at) ?? "",
    updated_at: stringOrNull(value.updated_at) ?? "",
    last_checkpoint_event: stringOrNull(value.last_checkpoint_event),
    components_present: Array.isArray(value.components_present) ? value.components_present.map((item) => String(item)) : [],
    locked: value.locked === true,
    lock_owner: isRecord(value.lock_owner) ? value.lock_owner : null,
    load_state: normalizeLoadState(value.load_state),
    loaded_at: optionalNumber(value.loaded_at),
    last_finished_at: optionalNumber(value.last_finished_at),
    gui_launch_profile: launchProfileFromUnknown(value.gui_launch_profile),
    last_cwd: stringOrNull(value.last_cwd),
  };
}

function workspaceRootFromMeta(meta: SessionMetaPayload | null | undefined): string | null {
  return normalizeWorkspaceRoot(meta?.last_cwd);
}

function normalizeWorkspaceRoot(value: unknown): string | null {
  const raw = stringOrNull(value);
  if (!raw) {
    return null;
  }
  return path.resolve(raw);
}

function sameWorkspaceRoot(left: string, right: string): boolean {
  return path.resolve(left) === path.resolve(right);
}

function ackFrame(command: string, payload: Record<string, unknown>): CoreFrame {
  return {
    version: VERSION,
    type: "ack",
    payload: { command, ok: true, ...payload },
  };
}

function injectSessionId(frame: CoreFrame, sessionId: string): CoreFrame {
  const payload = isRecord(frame.payload) ? frame.payload : {};
  return {
    ...frame,
    payload: {
      ...payload,
      session_id: payload.session_id ?? sessionId,
    },
  };
}

function framePayload(frame: CoreFrame): Record<string, unknown> {
  return isRecord(frame.payload) ? frame.payload : {};
}

function loadStateForRecord(record: EngineRecord): SessionLoadState {
  return isRunningAgentState(record.agentState) ? "running" : "loaded";
}

function isRunningAgentState(value: string): boolean {
  return value === "RUNNING" || value === "INTERRUPTING";
}

function normalizeLoadState(value: unknown): SessionLoadState | undefined {
  return value === "loaded" || value === "running" || value === "unloaded" ? value : undefined;
}

function commandTimeout(type: CoreCommandType): number {
  if (type === "compact_context") {
    return COMPACT_COMMAND_TIMEOUT_MS;
  }
  if (type === "session_export_markdown" || type === "session_history" || type === "session_list" || type === "session_search") {
    return SESSION_COMMAND_TIMEOUT_MS;
  }
  return DEFAULT_COMMAND_TIMEOUT_MS;
}

function isReadOnlyCommand(type: CoreCommandType, payload: Record<string, unknown>): boolean {
  return type === "session_search" || (type === "session_history" && payload.read_only === true);
}

function isMissingSessionError(error: unknown): boolean {
  if (error instanceof CoreCommandError) {
    const message = error.message.toLowerCase();
    const details = isRecord(error.details) ? error.details : {};
    return message.includes("session not found") || details.class === "FileNotFoundError";
  }
  return false;
}

function compareSessionsByCreatedAt(a: SessionMetaPayload, b: SessionMetaPayload): number {
  const left = Date.parse(a.created_at || a.updated_at || "");
  const right = Date.parse(b.created_at || b.updated_at || "");
  return (Number.isFinite(right) ? right : 0) - (Number.isFinite(left) ? left : 0);
}

function clonePluginConfigs(value: Record<string, Record<string, unknown>>): Record<string, Record<string, unknown>> {
  return Object.fromEntries(Object.entries(value).map(([key, config]) => [key, { ...config }]));
}

function pluginConfigRecord(value: unknown): Record<string, Record<string, unknown>> {
  if (!isRecord(value)) {
    return {};
  }
  return Object.fromEntries(Object.entries(value).map(([key, config]) => [key, isRecord(config) ? { ...config } : {}]));
}

function stringList(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function stringOrNull(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value : null;
}

function optionalNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function isCoreFrame(value: unknown): value is CoreFrame {
  return isRecord(value) && value.version === VERSION && typeof value.type === "string";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function generateSessionId(date = new Date()): string {
  const pad = (value: number) => String(value).padStart(2, "0");
  const timestamp =
    [date.getFullYear(), pad(date.getMonth() + 1), pad(date.getDate())].join("") +
    `-${pad(date.getHours())}${pad(date.getMinutes())}${pad(date.getSeconds())}`;
  const suffix = Math.random().toString(16).slice(2, 8).padEnd(6, "0");
  return `session-${timestamp}-${suffix}`;
}

function shortSessionId(value: string): string {
  const timestamped = value.match(/^session-(\d{8}-\d{6})-[0-9a-f]{6}$/);
  if (timestamped) return timestamped[1];
  return value.length <= 8 ? value : value.slice(0, 8);
}

function safeFilename(value: string): string {
  return value.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^[.-]+/, "") || "session";
}
