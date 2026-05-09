import { app, BrowserWindow, ipcMain, shell } from "electron";
import path from "node:path";
import { pathToFileURL } from "node:url";
import type { CoreCommandType, GuiMetadata, InspectPayload, PersistedConfig } from "../shared/protocol";
import {
  type EnvPaths,
  resolveEnvPaths,
  loadInspectPayload,
  loadConfig,
  saveConfig,
  sanitizeConfig
} from "./config";
import { CoreProcess, type EmitToRenderer } from "./core-process";

const env: EnvPaths = resolveEnvPaths();
const appIndexUrl = pathToFileURL(path.join(env.guiRoot, "dist", "index.html"));

let mainWindow: BrowserWindow | null = null;
let inspectPayload: InspectPayload | null = null;
let config: PersistedConfig | null = null;
let core: CoreProcess | null = null;

const MIN_CONTENT_WIDTH = 1080;
const MIN_CONTENT_HEIGHT = 660;

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

app.whenReady().then(() => {
  inspectPayload = loadInspectPayload(env.repoRoot, env.workspaceRoot, env.uvCommand);
  config = loadConfig(env.configPath, inspectPayload);
  const argvModel = parseArgValue("--model");
  if (argvModel && inspectPayload.models.includes(argvModel)) {
    config = { ...config, modelName: argvModel };
    saveConfig(env.configPath, config);
  }
  core = new CoreProcess(emitToRenderer, env.repoRoot, env.workspaceRoot, env.backendLogPath, env.uvCommand);
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
  core?.stop("before-quit");
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
    saveConfig(env.configPath, config);
    return config;
  });

  ipcMain.handle("core:restart", (_event, nextConfig: PersistedConfig) => {
    const ready = getReady();
    config = sanitizeConfig(nextConfig, ready.inspect);
    saveConfig(env.configPath, config);
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
