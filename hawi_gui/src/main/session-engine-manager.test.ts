import { describe, expect, it } from "vitest";
import type { CoreCommandType, CoreFrame, InspectPayload, PersistedConfig, SessionLaunchProfile } from "../shared/protocol";
import { VERSION } from "../shared/protocol";
import {
  MAX_LOADED_SESSIONS,
  SessionEngineManager,
  configFromProfile,
  launchProfileFromUnknown,
  profileFromConfig
} from "./session-engine-manager";

class FakeCore {
  readonly commands: Array<{ type: CoreCommandType; payload: Record<string, unknown> }> = [];
  running = true;
  stoppedReason: string | null = null;

  constructor(private readonly responses: Partial<Record<CoreCommandType, CoreFrame>> = {}) {}

  isRunning(): boolean {
    return this.running;
  }

  async sendCommand(type: CoreCommandType, payload: Record<string, unknown>): Promise<CoreFrame> {
    this.commands.push({ type, payload });
    return this.responses[type] ?? ackFrame(type, {});
  }

  async stop(reason: string): Promise<void> {
    this.stoppedReason = reason;
    this.running = false;
  }
}

interface FakeRecord {
  sessionId: string;
  core: FakeCore;
  launchProfile: SessionLaunchProfile;
  loadedAt: number;
  lastFinishedAt?: number;
  hasVisibleMessages: boolean;
  agentState: string;
  runnerState: string;
  suppressEvents: boolean;
  stopping: boolean;
}

interface ManagerInternals {
  loaded: Map<string, FakeRecord>;
  currentSessionId: string | null;
  handleEngineEmit(record: FakeRecord, channel: string, payload: unknown): void;
  emitSessionRuntimeStatus(sessionId: string): void;
  discardCurrentEmptySession(): Promise<void>;
  enforceLoadedLimit(): Promise<void>;
  sessionListFrame(): Promise<CoreFrame>;
}

const baseConfig: PersistedConfig = {
  version: 1,
  modelName: "deepseek-chat",
  systemPrompt: "system",
  selectedPlugins: ["filesystem"],
  pluginConfigs: { filesystem: { root: "." } },
  showDebug: true
};

const inspect: InspectPayload = {
  version: VERSION,
  models: ["deepseek-chat", "kimi"],
  plugin_catalog: [
    { key: "filesystem", label: "Filesystem", schema: {}, defaults: {} },
    { key: "shell", label: "Shell", schema: {}, defaults: {} }
  ],
  default_system_prompt: "default"
};

describe("session launch profiles", () => {
  it("round-trips config fields without persisting showDebug", () => {
    const profile = profileFromConfig(baseConfig);

    expect(profile).toMatchObject({
      modelName: "deepseek-chat",
      systemPrompt: "system",
      selectedPlugins: ["filesystem"],
      pluginConfigs: { filesystem: { root: "." } }
    });
    expect(profile).not.toHaveProperty("showDebug");
    expect(configFromProfile(profile, { ...baseConfig, showDebug: false }, inspect).showDebug).toBe(false);
  });

  it("normalizes unknown persisted profile data", () => {
    expect(launchProfileFromUnknown({
      version: 1,
      modelName: "kimi",
      systemPrompt: "saved",
      selectedPlugins: ["shell", 42],
      pluginConfigs: { shell: { cwd: "." }, bad: "x" },
      engineArgs: ["--model", "kimi", 7]
    })).toMatchObject({
      modelName: "kimi",
      systemPrompt: "saved",
      selectedPlugins: ["shell"],
      pluginConfigs: { shell: { cwd: "." }, bad: {} },
      engineArgs: ["--model", "kimi"]
    });
  });
});

describe("SessionEngineManager", () => {
  it("injects session ids into forwarded engine events", () => {
    const events: Array<{ channel: string; payload: unknown }> = [];
    const manager = makeManager(events);
    const internals = manager as unknown as ManagerInternals;
    const record = fakeRecord("session-a", 1);
    internals.loaded.set(record.sessionId, record);

    internals.handleEngineEmit(record, "core:event", {
      version: VERSION,
      type: "debug.info",
      payload: { message: "hello" }
    });

    expect(events[0]).toEqual({
      channel: "core:event",
      payload: {
        version: VERSION,
        type: "debug.info",
        payload: { message: "hello", session_id: "session-a" }
      }
    });
  });

  it("reports empty loaded sessions as not visibly materialized", () => {
    const events: Array<{ channel: string; payload: unknown }> = [];
    const manager = makeManager(events);
    const internals = manager as unknown as ManagerInternals;
    const record = fakeRecord("session-empty", 1);
    internals.loaded.set(record.sessionId, record);
    internals.currentSessionId = record.sessionId;

    internals.emitSessionRuntimeStatus(record.sessionId);

    const status = events[0].payload as CoreFrame;
    expect(status.type).toBe("gui.session_status");
    expect((status.payload as Record<string, unknown>).has_visible_messages).toBe(false);
    expect((status.payload as Record<string, unknown>).loaded_session_count).toBe(0);
  });

  it("reports sessions as visible after the first run starts", () => {
    const events: Array<{ channel: string; payload: unknown }> = [];
    const manager = makeManager(events);
    const internals = manager as unknown as ManagerInternals;
    const record = fakeRecord("session-new", 1);
    internals.loaded.set(record.sessionId, record);

    internals.handleEngineEmit(record, "core:event", {
      version: VERSION,
      type: "run.start",
      payload: { run_id: "run-1", user_content: "hello", queue: "normal" }
    });

    const status = events.find((event) => (
      event.channel === "core:event"
      && (event.payload as CoreFrame).type === "gui.session_status"
    ))?.payload as CoreFrame | undefined;
    expect(status?.type).toBe("gui.session_status");
    expect((status?.payload as Record<string, unknown> | undefined)?.has_visible_messages).toBe(true);
    expect((status?.payload as Record<string, unknown> | undefined)?.loaded_session_count).toBe(1);
  });

  it("does not count hidden empty sessions in the manager snapshot", () => {
    const manager = makeManager([]);
    const internals = manager as unknown as ManagerInternals;
    const empty = fakeRecord("session-empty", 1);
    const active = fakeRecord("session-active", 2);
    active.hasVisibleMessages = true;
    internals.currentSessionId = empty.sessionId;
    internals.loaded.set(empty.sessionId, empty);
    internals.loaded.set(active.sessionId, active);

    expect(manager.snapshot()).toMatchObject({
      currentSessionId: "session-empty",
      loadedSessionCount: 1,
      coreRunning: true
    });
  });

  it("discards the current empty session before creating another empty one", async () => {
    const manager = makeManager([]);
    const internals = manager as unknown as ManagerInternals;
    const empty = fakeRecord("session-empty", 1);
    internals.currentSessionId = empty.sessionId;
    internals.loaded.set(empty.sessionId, empty);

    await internals.discardCurrentEmptySession();

    expect(empty.core.stoppedReason).toBe("replace-empty-session");
    expect(internals.loaded.has(empty.sessionId)).toBe(false);
    expect(internals.currentSessionId).toBeNull();
  });

  it("routes commands to the targeted loaded session", async () => {
    const manager = makeManager([]);
    const internals = manager as unknown as ManagerInternals;
    const record = fakeRecord("session-target", 1);
    internals.loaded.set(record.sessionId, record);
    internals.currentSessionId = "session-other";

    await manager.sendCommand("enqueue", { content: "hi", queue: "normal" }, "session-target");

    expect(record.core.commands).toEqual([
      { type: "enqueue", payload: { content: "hi", queue: "normal" } }
    ]);
  });

  it("evicts the earliest finished idle non-current session over the loaded limit", async () => {
    const manager = makeManager([]);
    const internals = manager as unknown as ManagerInternals;
    internals.currentSessionId = "session-6";
    for (let index = 1; index <= MAX_LOADED_SESSIONS + 1; index += 1) {
      const record = fakeRecord(`session-${index}`, index);
      record.loadedAt = index * 1000;
      record.lastFinishedAt = index * 1000;
      internals.loaded.set(record.sessionId, record);
    }

    await internals.enforceLoadedLimit();

    expect(internals.loaded.has("session-1")).toBe(false);
    expect(internals.loaded.has("session-6")).toBe(true);
  });

  it("does not evict current or running sessions", async () => {
    const manager = makeManager([]);
    const internals = manager as unknown as ManagerInternals;
    internals.currentSessionId = "session-1";
    for (let index = 1; index <= MAX_LOADED_SESSIONS + 1; index += 1) {
      const record = fakeRecord(`session-${index}`, index);
      record.agentState = index === 2 ? "RUNNING" : "IDLE";
      record.loadedAt = index * 1000;
      record.lastFinishedAt = index * 1000;
      internals.loaded.set(record.sessionId, record);
    }

    await internals.enforceLoadedLimit();

    expect(internals.loaded.has("session-1")).toBe(true);
    expect(internals.loaded.has("session-2")).toBe(true);
    expect(internals.loaded.has("session-3")).toBe(false);
  });

  it("closes a loaded current session before deleting it and selects another loaded session", async () => {
    const manager = makeManager([]);
    const internals = manager as unknown as ManagerInternals;
    const current = fakeRecord("session-current", 1);
    const next = fakeRecord("session-next", 2);
    current.hasVisibleMessages = true;
    next.hasVisibleMessages = true;
    internals.currentSessionId = current.sessionId;
    internals.loaded.set(current.sessionId, current);
    internals.loaded.set(next.sessionId, next);

    const frame = await manager.sendCommand("session_delete", { session_id: current.sessionId });

    expect(current.core.stoppedReason).toBe("delete-session");
    expect(next.core.commands.some((command) => command.type === "session_delete")).toBe(true);
    expect(internals.loaded.has(current.sessionId)).toBe(false);
    expect(internals.currentSessionId).toBe(next.sessionId);
    expect((frame.payload as Record<string, unknown>).current_session_id).toBe(next.sessionId);
  });

  it("rejects direct deletion of running loaded sessions", async () => {
    const manager = makeManager([]);
    const internals = manager as unknown as ManagerInternals;
    const running = fakeRecord("session-running", 1);
    running.agentState = "RUNNING";
    internals.currentSessionId = running.sessionId;
    internals.loaded.set(running.sessionId, running);

    await expect(manager.sendCommand("session_delete", { session_id: running.sessionId }))
      .rejects.toThrow("Cannot delete a running session.");
  });

  it("uses persisted session profiles when listing loaded sessions", async () => {
    const persistedProfile = profileFromConfig({ ...baseConfig, modelName: "kimi" });
    const catalog = new FakeCore({
      session_list: ackFrame("session_list", {
        sessions: [
          {
            session_id: "session-saved",
            name: "Saved",
            created_at: "2025-01-01T00:00:00",
            updated_at: "2025-01-01T00:00:00",
            last_checkpoint_event: "save_now",
            components_present: ["context"],
            gui_launch_profile: persistedProfile
          }
        ],
        current_session_id: "session-catalog"
      })
    });
    const manager = makeManager([]);
    const internals = manager as unknown as ManagerInternals;
    internals.currentSessionId = "session-catalog";
    internals.loaded.set("session-catalog", fakeRecord("session-catalog", 1, catalog));
    internals.loaded.set("session-saved", fakeRecord("session-saved", 2, new FakeCore(), persistedProfile));

    const frame = await internals.sessionListFrame();
    const sessions = (frame.payload as Record<string, unknown>).sessions as Array<Record<string, unknown>>;
    const saved = sessions.find((session) => session.session_id === "session-saved");

    expect(saved?.load_state).toBe("loaded");
    expect(saved?.gui_launch_profile).toMatchObject({ modelName: "kimi" });
    expect((frame.payload as Record<string, unknown>).loaded_session_count).toBe(1);
  });

  it("omits loaded sessions that have not started a conversation", async () => {
    const manager = makeManager([]);
    const internals = manager as unknown as ManagerInternals;
    const empty = fakeRecord("session-empty", 1);
    const active = fakeRecord("session-active", 2);
    active.hasVisibleMessages = true;
    internals.currentSessionId = empty.sessionId;
    internals.loaded.set(empty.sessionId, empty);
    internals.loaded.set(active.sessionId, active);

    const frame = await internals.sessionListFrame();
    const sessions = (frame.payload as Record<string, unknown>).sessions as Array<Record<string, unknown>>;

    expect(sessions.map((session) => session.session_id)).toEqual(["session-active"]);
    expect((frame.payload as Record<string, unknown>).current_session_id).toBe("session-empty");
    expect((frame.payload as Record<string, unknown>).loaded_session_count).toBe(1);
  });

  it("shows a new loaded session after the first user message starts", async () => {
    const manager = makeManager([]);
    const internals = manager as unknown as ManagerInternals;
    const record = fakeRecord("session-new", 1);
    internals.currentSessionId = record.sessionId;
    internals.loaded.set(record.sessionId, record);

    internals.handleEngineEmit(record, "core:event", {
      version: VERSION,
      type: "run.start",
      payload: { run_id: "run-1", user_content: "hello", queue: "normal" }
    });

    const frame = await internals.sessionListFrame();
    const sessions = (frame.payload as Record<string, unknown>).sessions as Array<Record<string, unknown>>;

    expect(sessions.map((session) => session.session_id)).toContain("session-new");
    expect((frame.payload as Record<string, unknown>).loaded_session_count).toBe(1);
  });
});

function makeManager(events: Array<{ channel: string; payload: unknown }>): SessionEngineManager {
  const manager = new SessionEngineManager(
    (channel, payload) => events.push({ channel, payload }),
    "/repo",
    "/workspace",
    "/workspace/.hawi/hawi-engine.log",
    "uv"
  );
  manager.configure(inspect, baseConfig);
  return manager;
}

function fakeRecord(
  sessionId: string,
  index: number,
  core = new FakeCore(),
  launchProfile = profileFromConfig(baseConfig)
): FakeRecord {
  return {
    sessionId,
    core,
    launchProfile,
    loadedAt: index * 1000,
    lastFinishedAt: undefined,
    hasVisibleMessages: false,
    agentState: "IDLE",
    runnerState: "IDLE",
    suppressEvents: false,
    stopping: false
  };
}

function ackFrame(command: string, payload: Record<string, unknown>): CoreFrame {
  return {
    version: VERSION,
    type: "ack",
    payload: { command, ok: true, ...payload }
  };
}
