import { app, BrowserWindow, dialog, ipcMain, shell } from "electron";
import { mkdirSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import type {
  CoreCommandType,
  GuiMetadata,
  InspectPayload,
  JsonlExportPayload,
  MarkdownExportPayload,
  PersistedConfig,
  PluginToolPreviewPayload,
  SaveJsonlExportResult,
  SaveMarkdownExportResult,
  SelectWorkingDirectoryResult,
} from "../shared/protocol";
import { MIN_CONTENT_SIZE, minimumWindowSizeForContent, normalizeMinimumContentSize, type LayoutSize } from "../shared/layout";
import {
  type EnvPaths,
  ensureEngineWorkspace,
  resolveEnvPaths,
  loadInspectPayload,
  loadInspectPayloadAsync,
  loadPluginToolPreviewPayloadAsync,
  loadConfig,
  saveConfig,
  sanitizeConfig,
  preserveProviderOrder,
} from "./config";
import type { EmitToRenderer } from "./core-process";
import { SessionEngineManager } from "./session-engine-manager";

let env: EnvPaths | null = null;
let appIndexUrl: URL | null = null;
let mainWindow: BrowserWindow | null = null;
let inspectPayload: InspectPayload | null = null;
let config: PersistedConfig | null = null;
let engineManager: SessionEngineManager | null = null;
const refreshedProviders = new Set<string>();

const FILENAME_TIMESTAMP_RE = /\b\d{8}-\d{6}\b/;

function createWindow(): void {
  const readyEnv = getEnv();
  const window = new BrowserWindow({
    width: 1160,
    height: 780,
    minWidth: MIN_CONTENT_SIZE.width,
    minHeight: MIN_CONTENT_SIZE.height,
    useContentSize: true,
    title: "Hawi",
    backgroundColor: "#f7f8f8",
    webPreferences: {
      preload: path.join(readyEnv.guiRoot, "dist-electron", "preload", "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  mainWindow = window;

  installNavigationGuards(window);
  applyMinimumContentSize(window);
  window.webContents.on("did-finish-load", () => applyMinimumContentSize(window));

  window.on("closed", () => {
    mainWindow = null;
  });

  window.loadFile(path.join(readyEnv.guiRoot, "dist", "index.html"));
}

app.whenReady().then(async () => {
  const readyEnv = resolveEnvPaths({
    isPackaged: app.isPackaged,
    resourcesPath: process.resourcesPath,
  });
  env = readyEnv;
  appIndexUrl = pathToFileURL(path.join(readyEnv.guiRoot, "dist", "index.html"));
  ensureEngineWorkspace(readyEnv);
  inspectPayload = loadInspectPayload(readyEnv.repoRoot, readyEnv.workspaceRoot, readyEnv.engineLauncher);
  config = loadConfig(readyEnv.configPath, inspectPayload);
  inspectPayload = loadInspectPayloadWithModelProviderConfig(readyEnv, config) ?? inspectPayload;
  config = sanitizeConfig(config, inspectPayload);
  const argvModel = parseArgValue("--model");
  if (argvModel && inspectPayload.models.includes(argvModel)) {
    config = { ...config, modelName: argvModel };
    saveConfig(readyEnv.configPath, config);
  }
  engineManager = new SessionEngineManager(
    emitToRenderer,
    readyEnv.repoRoot,
    readyEnv.workspaceRoot,
    readyEnv.backendLogPath,
    readyEnv.engineLauncher,
  );
  engineManager.configure(inspectPayload, config, refreshedProviders);
  registerIpc();
  createWindow();
  if (config.modelName) {
    void engineManager.startInitial(config, inspectPayload, refreshedProviders).catch((error) => {
      console.error("Failed to prewarm hawi-engine:", error);
    });
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
      ...currentManagerSnapshot(),
    } satisfies GuiMetadata;
  });

  ipcMain.handle("gui:save-config", (_event, nextConfig: PersistedConfig) => {
    config = sanitizeConfig(nextConfig, inspectPayload);
    saveConfig(getEnv().configPath, config);
    if (inspectPayload && engineManager) {
      engineManager.configure(inspectPayload, config, refreshedProviders);
    }
    return config;
  });

  ipcMain.handle("core:restart", async (_event, nextConfig: PersistedConfig) => {
    const ready = getReady();
    config = sanitizeConfig(nextConfig, ready.inspect);
    saveConfig(getEnv().configPath, config);
    engineManager?.configure(ready.inspect, config, refreshedProviders);
    await engineManager?.restartCurrent(config);
    return { ok: true };
  });

  ipcMain.handle("core:command", async (_event, type: CoreCommandType, payload: Record<string, unknown>, sessionId?: string | null) => {
    if (!engineManager) {
      throw new Error("hawi-engine is not initialized");
    }
    const frame = await engineManager.sendCommand(type, payload, sessionId);
    if (type === "save_model_provider_config") {
      applyProviderConfigInspectPayload(frame.payload as Record<string, unknown>);
    }
    return frame;
  });

  ipcMain.handle("gui:save-markdown-export", async (_event, payload: MarkdownExportPayload): Promise<SaveMarkdownExportResult> => {
    return saveMarkdownExport(payload);
  });

  ipcMain.handle("gui:save-jsonl-export", async (_event, payload: JsonlExportPayload): Promise<SaveJsonlExportResult> => {
    return saveJsonlExport(payload);
  });

  ipcMain.handle("gui:select-working-directory", async (): Promise<SelectWorkingDirectoryResult> => {
    return selectWorkingDirectory();
  });

  ipcMain.handle("gui:set-minimum-content-size", (event, size: Partial<LayoutSize>) => {
    const window = BrowserWindow.fromWebContents(event.sender) ?? mainWindow;
    if (window) {
      applyMinimumContentSize(window, size);
    }
    return { ok: true };
  });

  ipcMain.handle("gui:refresh-provider-models", async (_event, provider: string): Promise<GuiMetadata> => {
    return refreshProviderModels(provider);
  });

  ipcMain.handle(
    "gui:preview-plugin-tools",
    async (_event, pluginKey: string, pluginConfig: Record<string, unknown> | null): Promise<PluginToolPreviewPayload> => {
      return previewPluginTools(pluginKey, pluginConfig);
    },
  );
}

function applyMinimumContentSize(window: BrowserWindow, contentSize?: Partial<LayoutSize> | null): void {
  const minContentSize = normalizeMinimumContentSize(contentSize);
  const bounds = window.getBounds();
  const contentBounds = window.getContentBounds();
  const minWindowSize = minimumWindowSizeForContent(minContentSize, {
    width: bounds.width - contentBounds.width,
    height: bounds.height - contentBounds.height,
  });
  window.setMinimumSize(minWindowSize.width, minWindowSize.height);
  const currentSize = window.getSize();
  if (currentSize[0] < minWindowSize.width || currentSize[1] < minWindowSize.height) {
    window.setSize(
      Math.max(currentSize[0], minWindowSize.width),
      Math.max(currentSize[1], minWindowSize.height),
    );
  }
}

function applyProviderConfigInspectPayload(payload: Record<string, unknown>): void {
  if (!inspectPayload || !config) return;
  const currentInspect = inspectPayload;
  const currentConfig = config;
  const nextInspect = {
    ...currentInspect,
    models: Array.isArray(payload.models)
      ? payload.models.filter((item): item is string => typeof item === "string")
      : currentInspect.models,
    model_provider_configs:
      payload.model_provider_configs && typeof payload.model_provider_configs === "object" && !Array.isArray(payload.model_provider_configs)
        ? payload.model_provider_configs as InspectPayload["model_provider_configs"]
        : currentInspect.model_provider_configs,
  } satisfies InspectPayload;
  inspectPayload = nextInspect;
  if (engineManager) {
    engineManager.configure(nextInspect, currentConfig, refreshedProviders);
  }
}

async function refreshProviderModels(provider: string): Promise<GuiMetadata> {
  const providerName = provider.trim();
  if (!providerName) {
    throw new Error("provider is required");
  }
  const ready = getReady();
  const readyEnv = getEnv();
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
    const modelProviderConfigPath = writeTemporaryModelProviderConfig(ready.config);
    try {
      const modelProviderConfigArgs = modelProviderConfigPath
        ? ["--model-provider-config", modelProviderConfigPath]
        : [];
      const refreshed = await loadInspectPayloadAsync(readyEnv.repoRoot, readyEnv.workspaceRoot, readyEnv.engineLauncher, [
        ...modelProviderConfigArgs,
        "--refresh-provider",
        providerName,
      ]);
      nextInspect = refreshed;
      allModels = refreshed.models;
    } finally {
      cleanupTemporaryConfig(modelProviderConfigPath);
    }
  }

  if (!allModels) {
    throw new Error(`refresh for provider '${providerName}' returned no models`);
  }

  refreshedProviders.add(providerName);
  inspectPayload = {
    ...nextInspect,
    models: preserveProviderOrder(ready.inspect.models, allModels),
  };
  config = sanitizeConfig(ready.config, inspectPayload);
  saveConfig(readyEnv.configPath, config);
  engineManager?.configure(inspectPayload, config, refreshedProviders);
  return {
    inspect: inspectPayload,
    config,
    ...currentManagerSnapshot(),
  };
}

function loadInspectPayloadWithModelProviderConfig(
  readyEnv: EnvPaths,
  readyConfig: PersistedConfig,
): InspectPayload | null {
  const modelProviderConfigPath = writeTemporaryModelProviderConfig(readyConfig);
  if (!modelProviderConfigPath) return null;
  try {
    return loadInspectPayload(readyEnv.repoRoot, readyEnv.workspaceRoot, readyEnv.engineLauncher, [
      "--model-provider-config",
      modelProviderConfigPath,
    ]);
  } finally {
    cleanupTemporaryConfig(modelProviderConfigPath);
  }
}

function writeTemporaryModelProviderConfig(configValue: PersistedConfig): string | null {
  const configs = configValue.modelProviderConfigs ?? {};
  if (Object.keys(configs).length === 0) return null;
  const configPath = path.join(
    tmpdir(),
    `hawi-gui-model-provider-inspect-${process.pid}-${Date.now()}.json`,
  );
  writeFileSync(configPath, JSON.stringify(configs, null, 2), "utf-8");
  return configPath;
}

function cleanupTemporaryConfig(configPath: string | null): void {
  if (!configPath) return;
  try {
    unlinkSync(configPath);
  } catch {
    // Temporary config cleanup is best-effort.
  }
}

async function previewPluginTools(
  pluginKey: string,
  pluginConfig: Record<string, unknown> | null = null,
): Promise<PluginToolPreviewPayload> {
  const key = pluginKey.trim();
  if (!key) {
    throw new Error("plugin key is required");
  }
  const readyEnv = getEnv();
  const pluginConfigPath = path.join(
    tmpdir(),
    `hawi-gui-plugin-preview-${process.pid}-${Date.now()}.json`,
  );
  writeFileSync(pluginConfigPath, JSON.stringify({ [key]: pluginConfig ?? {} }, null, 2), "utf-8");
  try {
    return await loadPluginToolPreviewPayloadAsync(
      readyEnv.repoRoot,
      readyEnv.workspaceRoot,
      readyEnv.engineLauncher,
      key,
      pluginConfigPath,
    );
  } finally {
    try {
      unlinkSync(pluginConfigPath);
    } catch {
      // Temporary preview config cleanup is best-effort.
    }
  }
}

async function saveMarkdownExport(payload: MarkdownExportPayload): Promise<SaveMarkdownExportResult> {
  if (!payload || typeof payload.markdown !== "string") {
    throw new Error("invalid markdown export payload");
  }
  const suggested = safeMarkdownFilename(payload.suggested_filename);
  const options = {
    title: "导出 Markdown",
    defaultPath: suggested,
    filters: [{ name: "Markdown", extensions: ["md"] }],
  };
  const result = mainWindow ? await dialog.showSaveDialog(mainWindow, options) : await dialog.showSaveDialog(options);
  if (result.canceled || !result.filePath) {
    return { canceled: true };
  }

  const markdownPath = ensureMarkdownExtension(result.filePath);
  const parsed = path.parse(markdownPath);
  const referenceDirName = `${parsed.name}-ref`;
  const originalRefDir = payload.reference_dir_name;
  const markdown =
    originalRefDir && originalRefDir !== referenceDirName
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
    referenceDir,
  };
}

async function saveJsonlExport(payload: JsonlExportPayload): Promise<SaveJsonlExportResult> {
  if (!payload || !Array.isArray(payload.records)) {
    throw new Error("invalid JSONL export payload");
  }
  const suggested = safeJsonlFilename(payload.suggested_filename);
  const options = {
    title: "导出 JSONL",
    defaultPath: suggested,
    filters: [{ name: "JSON Lines", extensions: ["jsonl"] }],
  };
  const result = mainWindow ? await dialog.showSaveDialog(mainWindow, options) : await dialog.showSaveDialog(options);
  if (result.canceled || !result.filePath) {
    return { canceled: true };
  }

  const jsonlPath = ensureJsonlExtension(result.filePath);
  const parsed = path.parse(jsonlPath);
  const contents = payload.records.map((record) => JSON.stringify(record)).join("\n");
  mkdirSync(parsed.dir, { recursive: true });
  writeFileSync(jsonlPath, contents ? `${contents}\n` : "", "utf8");

  return {
    canceled: false,
    jsonlPath,
  };
}

async function selectWorkingDirectory(): Promise<SelectWorkingDirectoryResult> {
  const defaultPath = engineManager?.getCurrentWorkspaceRoot() ?? env?.workspaceRoot;
  const options = {
    title: "切换工作目录",
    defaultPath,
    properties: ["openDirectory", "createDirectory"] as Electron.OpenDialogOptions["properties"],
  };
  const result = mainWindow ? await dialog.showOpenDialog(mainWindow, options) : await dialog.showOpenDialog(options);
  if (result.canceled || !result.filePaths[0]) {
    return { canceled: true };
  }
  return {
    canceled: false,
    path: path.resolve(result.filePaths[0]),
  };
}

function ensureMarkdownExtension(filePath: string): string {
  return path.extname(filePath) ? filePath : `${filePath}.md`;
}

function ensureJsonlExtension(filePath: string): string {
  return path.extname(filePath) ? filePath : `${filePath}.jsonl`;
}

function safeMarkdownFilename(value: string | undefined): string {
  const base = value && value.trim() ? path.basename(value.trim()) : "hawi-export.md";
  const filename = base.toLowerCase().endsWith(".md") ? base : `${base}.md`;
  return filenameWithTimestamp(filename);
}

function safeJsonlFilename(value: string | undefined): string {
  const base = value && value.trim() ? path.basename(value.trim()) : "hawi-export.jsonl";
  const filename = base.toLowerCase().endsWith(".jsonl") ? base : `${base}.jsonl`;
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
  return (
    [date.getFullYear(), pad(date.getMonth() + 1), pad(date.getDate())].join("") +
    `-${pad(date.getHours())}${pad(date.getMinutes())}${pad(date.getSeconds())}`
  );
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
  if (!appIndexUrl) {
    return false;
  }
  try {
    const parsed = new URL(url);
    return parsed.protocol === "file:" && parsed.pathname === appIndexUrl.pathname;
  } catch {
    return false;
  }
}

function getEnv(): EnvPaths {
  if (!env) {
    throw new Error("GUI environment is not ready");
  }
  return env;
}

function getReady(): { inspect: InspectPayload; config: PersistedConfig } {
  if (!inspectPayload || !config) {
    throw new Error("GUI metadata is not ready");
  }
  return { inspect: inspectPayload, config };
}

function currentManagerSnapshot() {
  return (
    engineManager?.snapshot() ?? {
      currentSessionId: null,
      runningSessionCount: 0,
      loadedSessionCount: 0,
      maxLoadedSessions: 5,
      coreRunning: false,
    }
  );
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
