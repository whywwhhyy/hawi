import { app, BrowserWindow, dialog, ipcMain, shell } from "electron";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import type { CoreCommandType, GuiMetadata, InspectPayload, MarkdownExportPayload, PersistedConfig, SaveMarkdownExportResult } from "../shared/protocol";
import {
  type EnvPaths,
  resolveEnvPaths,
  loadInspectPayload,
  loadConfig,
  saveConfig,
  sanitizeConfig,
  preserveProviderOrder
} from "./config";
import type { EmitToRenderer } from "./core-process";
import { SessionEngineManager } from "./session-engine-manager";

const env: EnvPaths = resolveEnvPaths();
const appIndexUrl = pathToFileURL(path.join(env.guiRoot, "dist", "index.html"));

let mainWindow: BrowserWindow | null = null;
let inspectPayload: InspectPayload | null = null;
let config: PersistedConfig | null = null;
let engineManager: SessionEngineManager | null = null;
const refreshedProviders = new Set<string>();

const MIN_CONTENT_WIDTH = 1080;
const MIN_CONTENT_HEIGHT = 660;
const FILENAME_TIMESTAMP_RE = /\b\d{8}-\d{6}\b/;

function createWindow(): void {
  const window = new BrowserWindow({
    width: 1160,
    height: 780,
    minWidth: MIN_CONTENT_WIDTH,
    minHeight: MIN_CONTENT_HEIGHT,
    useContentSize: true,
    title: "Hawi",
    backgroundColor: "#f7f8f8",
    webPreferences: {
      preload: path.join(env.guiRoot, "dist-electron", "preload", "preload.js"),
      contextIsolation: true,
      nodeIntegration: false
    }
  });
  mainWindow = window;

  installNavigationGuards(window);

  window.on("closed", () => {
    mainWindow = null;
  });

  window.loadFile(path.join(env.guiRoot, "dist", "index.html"));
}

app.whenReady().then(async () => {
  inspectPayload = loadInspectPayload(env.repoRoot, env.workspaceRoot, env.uvCommand);
  config = loadConfig(env.configPath, inspectPayload);
  const argvModel = parseArgValue("--model");
  if (argvModel && inspectPayload.models.includes(argvModel)) {
    config = { ...config, modelName: argvModel };
    saveConfig(env.configPath, config);
  }
  engineManager = new SessionEngineManager(emitToRenderer, env.repoRoot, env.workspaceRoot, env.backendLogPath, env.uvCommand);
  engineManager.configure(inspectPayload, config, refreshedProviders);
  registerIpc();
  createWindow();
  if (config.modelName) {
    await engineManager.startInitial(config, inspectPayload, refreshedProviders);
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
  void engineManager?.stopAll("before-quit");
});

function registerIpc(): void {
  ipcMain.handle("gui:get-metadata", () => {
    const ready = getReady();
    return {
      inspect: ready.inspect,
      config: ready.config,
      ...currentManagerSnapshot()
    } satisfies GuiMetadata;
  });

  ipcMain.handle("gui:save-config", (_event, nextConfig: PersistedConfig) => {
    config = sanitizeConfig(nextConfig, inspectPayload);
    saveConfig(env.configPath, config);
    if (inspectPayload && engineManager) {
      engineManager.configure(inspectPayload, config, refreshedProviders);
    }
    return config;
  });

  ipcMain.handle("core:restart", async (_event, nextConfig: PersistedConfig) => {
    const ready = getReady();
    config = sanitizeConfig(nextConfig, ready.inspect);
    saveConfig(env.configPath, config);
    engineManager?.configure(ready.inspect, config, refreshedProviders);
    await engineManager?.restartCurrent(config);
    return { ok: true };
  });

  ipcMain.handle("core:command", async (_event, type: CoreCommandType, payload: Record<string, unknown>, sessionId?: string | null) => {
    if (!engineManager) {
      throw new Error("hawi-engine is not initialized");
    }
    return engineManager.sendCommand(type, payload, sessionId);
  });

  ipcMain.handle("gui:save-markdown-export", async (_event, payload: MarkdownExportPayload): Promise<SaveMarkdownExportResult> => {
    return saveMarkdownExport(payload);
  });

  ipcMain.handle("gui:refresh-provider-models", async (_event, provider: string): Promise<GuiMetadata> => {
    return refreshProviderModels(provider);
  });
}

async function refreshProviderModels(provider: string): Promise<GuiMetadata> {
  const providerName = provider.trim();
  if (!providerName) {
    throw new Error("provider is required");
  }
  const ready = getReady();
  let allModels: string[] | null = null;
  let nextInspect = ready.inspect;

  const refreshFrame = await engineManager?.refreshModels(providerName);
  if (refreshFrame) {
    const frame = refreshFrame;
    const payload = frame.payload as Record<string, unknown>;
    if (Array.isArray(payload.all_models)) {
      allModels = payload.all_models.filter((item): item is string => typeof item === "string");
    }
  } else {
    const refreshed = loadInspectPayload(env.repoRoot, env.workspaceRoot, env.uvCommand, [
      "--refresh-provider",
      providerName
    ]);
    nextInspect = refreshed;
    allModels = refreshed.models;
  }

  if (!allModels) {
    throw new Error(`refresh for provider '${providerName}' returned no models`);
  }

  refreshedProviders.add(providerName);
  inspectPayload = {
    ...nextInspect,
    models: preserveProviderOrder(ready.inspect.models, allModels)
  };
  config = sanitizeConfig(ready.config, inspectPayload);
  saveConfig(env.configPath, config);
  engineManager?.configure(inspectPayload, config, refreshedProviders);
  return {
    inspect: inspectPayload,
    config,
    ...currentManagerSnapshot()
  };
}

async function saveMarkdownExport(payload: MarkdownExportPayload): Promise<SaveMarkdownExportResult> {
  if (!payload || typeof payload.markdown !== "string") {
    throw new Error("invalid markdown export payload");
  }
  const suggested = safeMarkdownFilename(payload.suggested_filename);
  const options = {
    title: "导出 Markdown",
    defaultPath: suggested,
    filters: [{ name: "Markdown", extensions: ["md"] }]
  };
  const result = mainWindow
    ? await dialog.showSaveDialog(mainWindow, options)
    : await dialog.showSaveDialog(options);
  if (result.canceled || !result.filePath) {
    return { canceled: true };
  }

  const markdownPath = ensureMarkdownExtension(result.filePath);
  const parsed = path.parse(markdownPath);
  const referenceDirName = `${parsed.name}-ref`;
  const originalRefDir = payload.reference_dir_name;
  const markdown = originalRefDir && originalRefDir !== referenceDirName
    ? payload.markdown.split(originalRefDir).join(referenceDirName)
    : payload.markdown;

  mkdirSync(parsed.dir, { recursive: true });
  writeFileSync(markdownPath, markdown, "utf8");

  const references = Array.isArray(payload.references) ? payload.references : [];
  let referenceDir: string | undefined;
  if (references.length > 0) {
    referenceDir = path.join(parsed.dir, referenceDirName);
    mkdirSync(referenceDir, { recursive: true });
    for (const ref of references) {
      if (!ref || typeof ref.content !== "string") {
        continue;
      }
      const filename = safeReferenceFilename(ref.filename);
      writeFileSync(path.join(referenceDir, filename), ref.content, "utf8");
    }
  }

  return {
    canceled: false,
    markdownPath,
    referenceDir
  };
}

function ensureMarkdownExtension(filePath: string): string {
  return path.extname(filePath) ? filePath : `${filePath}.md`;
}

function safeMarkdownFilename(value: string | undefined): string {
  const base = value && value.trim() ? path.basename(value.trim()) : "hawi-export.md";
  const filename = base.toLowerCase().endsWith(".md") ? base : `${base}.md`;
  return filenameWithTimestamp(filename);
}

function safeReferenceFilename(value: string | undefined): string {
  const base = value && value.trim() ? path.basename(value.trim()) : "reference.txt";
  return base.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^[.-]+/, "") || "reference.txt";
}

function filenameWithTimestamp(filename: string): string {
  const parsed = path.parse(filename);
  if (FILENAME_TIMESTAMP_RE.test(parsed.name)) {
    return filename;
  }
  return `${parsed.name}-${timestampToSeconds()}${parsed.ext || ".md"}`;
}

function timestampToSeconds(date = new Date()): string {
  const pad = (value: number) => String(value).padStart(2, "0");
  return [
    date.getFullYear(),
    pad(date.getMonth() + 1),
    pad(date.getDate())
  ].join("") + `-${pad(date.getHours())}${pad(date.getMinutes())}${pad(date.getSeconds())}`;
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

function currentManagerSnapshot() {
  return engineManager?.snapshot() ?? {
    currentSessionId: null,
    runningSessionCount: 0,
    loadedSessionCount: 0,
    maxLoadedSessions: 5,
    coreRunning: false
  };
}

const emitToRenderer: EmitToRenderer = (channel, payload) => {
  for (const win of BrowserWindow.getAllWindows()) {
    win.webContents.send(channel, payload);
  }
};

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
