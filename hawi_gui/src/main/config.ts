import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import type { InspectPayload, PersistedConfig } from "../shared/protocol";

export interface EnvPaths {
  repoRoot: string;
  guiRoot: string;
  workspaceRoot: string;
  configPath: string;
  backendLogPath: string;
  uvCommand: string;
}

export function resolveEnvPaths(): EnvPaths {
  const repoRoot = path.resolve(__dirname, "..", "..", "..");
  const guiRoot = path.join(repoRoot, "hawi_gui");
  const workspaceRoot = resolveWorkspaceRoot();
  const configPath = path.join(workspaceRoot, ".hawi", "node_gui.json");
  const backendLogPath = path.join(workspaceRoot, ".hawi", "hawi-core.log");
  const uvCommand = process.platform === "win32" ? "uv.cmd" : "uv";
  return { repoRoot, guiRoot, workspaceRoot, configPath, backendLogPath, uvCommand };
}

export function resolveWorkspaceRoot(): string {
  const raw = parseArgValue("--cwd") || process.env.HAWI_GUI_CWD || process.env.INIT_CWD || process.cwd();
  return path.resolve(raw);
}

export function parseArgValue(name: string): string | null {
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

export function loadInspectPayload(repoRoot: string, workspaceRoot: string, uvCommand: string): InspectPayload {
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

export function loadConfig(configPath: string, metadata: InspectPayload): PersistedConfig {
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

export function defaultConfig(metadata: InspectPayload): PersistedConfig {
  return {
    version: 1,
    modelName: metadata.models[0] ?? "",
    systemPrompt: metadata.default_system_prompt,
    selectedPlugins: [],
    pluginConfigs: {},
    showDebug: true
  };
}

export function sanitizeConfig(raw: PersistedConfig, metadata: InspectPayload | null): PersistedConfig {
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

export function saveConfig(configPath: string, nextConfig: PersistedConfig): void {
  mkdirSync(path.dirname(configPath), { recursive: true });
  writeFileSync(configPath, JSON.stringify(nextConfig, null, 2), "utf-8");
}
