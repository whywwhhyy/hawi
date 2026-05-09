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
  const uvCommand = resolveUvCommand();
  return { repoRoot, guiRoot, workspaceRoot, configPath, backendLogPath, uvCommand };
}

export function resolveUvCommand(): string {
  const override = process.env.HAWI_GUI_UV_COMMAND?.trim();
  if (override) {
    return override;
  }
  return resolveCommandOnPath("uv") ?? "uv";
}

export function resolveCommandOnPath(command: string, pathValue = process.env.PATH ?? process.env.Path ?? ""): string | null {
  const pathEntries = pathValue.split(path.delimiter).filter(Boolean);
  const extensions = process.platform === "win32" ? ["", ".exe", ".cmd", ".bat"] : [""];
  for (const entry of pathEntries) {
    for (const extension of extensions) {
      const candidate = path.join(entry, `${command}${extension}`);
      if (existsSync(candidate)) {
        return candidate;
      }
    }
  }
  return null;
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
  if (result.error) {
    throw new Error(`Failed to launch ${uvCommand}: ${result.error.message}`);
  }
  if (result.status !== 0) {
    throw new Error(formatInspectError(result.status, result.stderr, result.stdout));
  }
  return JSON.parse(result.stdout) as InspectPayload;
}

function formatInspectError(status: number | null, stderr: string | Buffer | null, stdout: string | Buffer | null): string {
  const details = [`Failed to inspect hawi-core metadata (exit ${status ?? "unknown"}).`];
  const stderrText = stderr?.toString().trim();
  const stdoutText = stdout?.toString().trim();
  if (stderrText) {
    details.push(`stderr: ${stderrText}`);
  }
  if (stdoutText) {
    details.push(`stdout: ${stdoutText}`);
  }
  return details.join("\n");
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
