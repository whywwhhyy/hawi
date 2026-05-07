import { app, BrowserWindow, ipcMain, shell } from "electron";
import { spawn, spawnSync, ChildProcessWithoutNullStreams } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { parseNdjsonChunk, type CoreCommand, type CoreCommandType, type CoreFrame, type GuiMetadata, type InspectPayload, type PersistedConfig } from "../shared/protocol";

const repoRoot = path.resolve(__dirname, "..", "..", "..");
const guiRoot = path.join(repoRoot, "hawi_gui");
const appIndexPath = path.join(guiRoot, "dist", "index.html");
const appIndexUrl = pathToFileURL(appIndexPath);
const workspaceRoot = resolveWorkspaceRoot();
const configPath = path.join(workspaceRoot, ".hawi", "node_gui.json");
const backendLogPath = path.join(workspaceRoot, ".hawi", "hawi-core.log");
const uvCommand = process.platform === "win32" ? "uv.cmd" : "uv";
const GRACEFUL_SHUTDOWN_TIMEOUT_MS = 800;
const DEFAULT_COMMAND_TIMEOUT_MS = 15_000;

let mainWindow: BrowserWindow | null = null;
let inspectPayload: InspectPayload | null = null;
let config: PersistedConfig | null = null;
let core: CoreProcess | null = null;

function createWindow(): void {
  const window = new BrowserWindow({
    width: 1160,
    height: 780,
    minWidth: 920,
    minHeight: 640,
    title: "Hawi",
    backgroundColor: "#f7f8f8",
    webPreferences: {
      preload: path.join(guiRoot, "dist-electron", "preload", "preload.js"),
      contextIsolation: true,
      nodeIntegration: false
    }
  });
  mainWindow = window;

  installNavigationGuards(window);

  window.on("closed", () => {
    mainWindow = null;
  });

  window.loadFile(appIndexPath);
}

app.whenReady().then(() => {
  inspectPayload = loadInspectPayload();
  config = loadConfig(inspectPayload);
  const argvModel = parseArgValue("--model");
  if (argvModel && inspectPayload.models.includes(argvModel)) {
    config = { ...config, modelName: argvModel };
    saveConfig(config);
  }
  core = new CoreProcess();
  registerIpc();
  createWindow();
  if (config.modelName) {
    core.start(config, inspectPayload);
  }
});

app.on("window-all-closed", () => {
  app.quit();
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

app.on("before-quit", () => {
  core?.stop();
});

function registerIpc(): void {
  ipcMain.handle("gui:get-metadata", () => {
    const ready = getReady();
    return {
      inspect: ready.inspect,
      config: ready.config,
      coreRunning: core?.isRunning() ?? false
    } satisfies GuiMetadata;
  });

  ipcMain.handle("gui:save-config", (_event, nextConfig: PersistedConfig) => {
    config = sanitizeConfig(nextConfig, inspectPayload);
    saveConfig(config);
    return config;
  });

  ipcMain.handle("core:restart", (_event, nextConfig: PersistedConfig) => {
    const ready = getReady();
    config = sanitizeConfig(nextConfig, ready.inspect);
    saveConfig(config);
    core?.restart(config, ready.inspect);
    return { ok: true };
  });

  ipcMain.handle("core:command", async (_event, type: CoreCommandType, payload: Record<string, unknown>) => {
    if (!core) {
      throw new Error("hawi-core is not initialized");
    }
    return core.sendCommand(type, payload);
  });
}

function installNavigationGuards(window: BrowserWindow): void {
  window.webContents.setWindowOpenHandler(({ url }) => {
    openExternalUrl(url);
    return { action: "deny" };
  });

  window.webContents.on("will-navigate", (event, url) => {
    if (isAppDocumentUrl(url)) {
      return;
    }
    event.preventDefault();
    openExternalUrl(url);
  });
}

function openExternalUrl(url: string): void {
  if (isOpenableExternalUrl(url)) {
    void shell.openExternal(url);
  }
}

function isOpenableExternalUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    return parsed.protocol === "http:" || parsed.protocol === "https:" || parsed.protocol === "mailto:";
  } catch {
    return false;
  }
}

function isAppDocumentUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    return parsed.protocol === "file:" && parsed.pathname === appIndexUrl.pathname;
  } catch {
    return false;
  }
}

function getReady(): { inspect: InspectPayload; config: PersistedConfig } {
  if (!inspectPayload || !config) {
    throw new Error("GUI metadata is not ready");
  }
  return { inspect: inspectPayload, config };
}

class CoreProcess {
  private child: ChildProcessWithoutNullStreams | null = null;
  private stdoutBuffer = "";
  private pending = new Map<string, { resolve: (frame: CoreFrame) => void; reject: (error: Error) => void }>();
  private sequence = 0;

  isRunning(): boolean {
    return this.child !== null && !this.child.killed;
  }

  start(nextConfig: PersistedConfig, metadata: InspectPayload): void {
    if (!nextConfig.modelName) {
      return;
    }
    this.stop();
    mkdirSync(path.dirname(backendLogPath), { recursive: true });
    const pluginConfigPath = path.join(tmpdir(), `hawi-gui-plugins-${process.pid}.json`);
    writeFileSync(pluginConfigPath, JSON.stringify(nextConfig.pluginConfigs, null, 2), "utf-8");

    const args = [
      "run",
      "--project",
      repoRoot,
      "hawi-core",
      "--model",
      nextConfig.modelName,
      "--transport",
      "stdio",
      "--system-prompt",
      nextConfig.systemPrompt || metadata.default_system_prompt,
      "--plugins",
      nextConfig.selectedPlugins.join(","),
      "--plugin-config",
      pluginConfigPath,
      "--extra-tool-parameter",
      "tool_call_description",
      "str",
      "用一句话说明为什么要调用这个工具，显示在工具标题旁边。",
      "--log-file",
      backendLogPath
    ];
    const child = spawn(uvCommand, args, {
      cwd: workspaceRoot,
      stdio: ["pipe", "pipe", "pipe"],
      env: process.env
    });
    this.child = child;
    child.stdout.setEncoding("utf-8");
    child.stdout.on("data", (chunk: string) => this.handleStdout(chunk));
    child.stderr.setEncoding("utf-8");
    child.stderr.on("data", (chunk: string) => {
      emitToRenderer("core:stderr", chunk);
    });
    child.on("exit", (code, signal) => {
      const wasCurrent = this.child === child;
      if (wasCurrent) {
        this.child = null;
        const error = new Error(`hawi-core exited (${code ?? "null"} ${signal ?? ""})`);
        for (const pending of this.pending.values()) {
          pending.reject(error);
        }
        this.pending.clear();
      }
      emitToRenderer("core:exit", { code, signal });
    });
    emitToRenderer("core:spawn", { args: args.slice(1), cwd: workspaceRoot, logFile: backendLogPath });
  }

  restart(nextConfig: PersistedConfig, metadata: InspectPayload): void {
    this.start(nextConfig, metadata);
  }

  stop(): void {
    const child = this.child;
    if (!child) {
      return;
    }
    this.child = null;
    const error = new Error("hawi-core was stopped");
    for (const pending of this.pending.values()) {
      pending.reject(error);
    }
    this.pending.clear();
    try {
      this.writeFrame(child, { version: "hawi.core.v1", type: "shutdown", id: this.nextId(), payload: {} });
    } catch {
      // Process may already be closing.
    }
    setTimeout(() => {
      if (!child.killed) {
        child.kill();
      }
    }, GRACEFUL_SHUTDOWN_TIMEOUT_MS).unref();
  }

  sendCommand(type: CoreCommandType, payload: Record<string, unknown>, timeoutMs = DEFAULT_COMMAND_TIMEOUT_MS): Promise<CoreFrame> {
    if (!this.child || !this.child.stdin.writable) {
      return Promise.reject(new Error("hawi-core is not running"));
    }
    const id = this.nextId();
    const frame: CoreCommand = {
      version: "hawi.core.v1",
      type,
      id,
      payload
    };
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.writeFrame(this.child, frame);
      setTimeout(() => {
        if (this.pending.delete(id)) {
          reject(new Error(`Core command timed out: ${type}`));
        }
      }, timeoutMs).unref();
    });
  }

  private nextId(): string {
    this.sequence += 1;
    return `gui-${Date.now().toString(36)}-${this.sequence}`;
  }

  private writeFrame(child: ChildProcessWithoutNullStreams | null, frame: CoreCommand): void {
    child?.stdin.write(`${JSON.stringify(frame)}\n`, "utf-8");
  }

  private handleStdout(chunk: string): void {
    const result = parseNdjsonChunk(this.stdoutBuffer, chunk);
    this.stdoutBuffer = result.buffer;
    for (const error of result.errors) {
      emitToRenderer("core:event", {
        version: "hawi.core.v1",
        type: "error",
        payload: { ok: false, code: "bad_frame", message: error }
      });
    }
    for (const frame of result.frames) {
      if (frame.id && this.pending.has(frame.id) && (frame.type === "ack" || frame.type === "error" || frame.type === "pong" || frame.type === "core.status")) {
        const pending = this.pending.get(frame.id);
        this.pending.delete(frame.id);
        if (frame.type === "error") {
          pending?.reject(new Error(String((frame.payload as Record<string, unknown>).message ?? "Core error")));
        } else {
          pending?.resolve(frame);
        }
      }
      emitToRenderer("core:event", frame);
    }
  }
}

function emitToRenderer(channel: string, payload: unknown): void {
  for (const win of BrowserWindow.getAllWindows()) {
    win.webContents.send(channel, payload);
  }
}

function loadInspectPayload(): InspectPayload {
  const result = spawnSync(uvCommand, ["run", "--project", repoRoot, "hawi-core", "--inspect"], {
    cwd: workspaceRoot,
    encoding: "utf-8",
    env: process.env
  });
  if (result.status !== 0) {
    throw new Error(result.stderr || "Failed to inspect hawi-core metadata");
  }
  return JSON.parse(result.stdout) as InspectPayload;
}

function loadConfig(metadata: InspectPayload): PersistedConfig {
  if (!existsSync(configPath)) {
    return defaultConfig(metadata);
  }
  try {
    const parsed = JSON.parse(readFileSync(configPath, "utf-8")) as PersistedConfig;
    return sanitizeConfig(parsed, metadata);
  } catch {
    return defaultConfig(metadata);
  }
}

function defaultConfig(metadata: InspectPayload): PersistedConfig {
  return {
    version: 1,
    modelName: metadata.models[0] ?? "",
    systemPrompt: metadata.default_system_prompt,
    selectedPlugins: [],
    pluginConfigs: {},
    showDebug: true
  };
}

function sanitizeConfig(raw: PersistedConfig, metadata: InspectPayload | null): PersistedConfig {
  const modelName = metadata?.models.includes(raw.modelName) ? raw.modelName : metadata?.models[0] ?? "";
  const pluginKeys = new Set(metadata?.plugin_catalog.map((item) => item.key) ?? []);
  const selectedPlugins = Array.isArray(raw.selectedPlugins)
    ? raw.selectedPlugins.filter((key) => pluginKeys.has(key))
    : [];
  const pluginConfigs = raw.pluginConfigs && typeof raw.pluginConfigs === "object" ? raw.pluginConfigs : {};
  return {
    version: 1,
    modelName,
    systemPrompt: typeof raw.systemPrompt === "string" && raw.systemPrompt.trim() ? raw.systemPrompt : metadata?.default_system_prompt ?? "",
    selectedPlugins,
    pluginConfigs,
    showDebug: Boolean(raw.showDebug)
  };
}

function saveConfig(nextConfig: PersistedConfig): void {
  mkdirSync(path.dirname(configPath), { recursive: true });
  writeFileSync(configPath, JSON.stringify(nextConfig, null, 2), "utf-8");
}

function resolveWorkspaceRoot(): string {
  const raw = parseArgValue("--cwd") || process.env.HAWI_GUI_CWD || process.env.INIT_CWD || process.cwd();
  return path.resolve(raw);
}

function parseArgValue(name: string): string | null {
  const inlinePrefix = `${name}=`;
  const inlineValue = process.argv.find((arg) => arg.startsWith(inlinePrefix));
  if (inlineValue) {
    return inlineValue.slice(inlinePrefix.length);
  }
  const index = process.argv.indexOf(name);
  if (index >= 0 && process.argv[index + 1]) {
    return process.argv[index + 1];
  }
  return null;
}
