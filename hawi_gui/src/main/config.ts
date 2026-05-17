import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";
import type { InspectPayload, PersistedConfig } from "../shared/protocol";

export type EngineLauncherSource = "bundled" | "uv";

export interface EngineLauncher {
  command: string;
  argsPrefix: string[];
  source: EngineLauncherSource;
}

export interface ResolveEnvPathsOptions {
  isPackaged?: boolean;
  resourcesPath?: string;
  cwd?: string;
}

export interface EnvPaths {
  repoRoot: string;
  guiRoot: string;
  workspaceRoot: string;
  configPath: string;
  backendLogPath: string;
  engineLauncher: EngineLauncher;
  uvCommand: string;
}

export function resolveEnvPaths(options: ResolveEnvPathsOptions = {}): EnvPaths {
  const guiRoot = resolveGuiRoot();
  const repoRoot = options.isPackaged ? guiRoot : path.resolve(guiRoot, "..");
  const workspaceRoot = resolveWorkspaceRoot(options);
  const configPath = path.join(workspaceRoot, ".hawi", "node_gui.json");
  const backendLogPath = path.join(workspaceRoot, ".hawi", "hawi-engine.log");
  const uvCommand = resolveUvCommand();
  const engineLauncher = resolveEngineLauncher(repoRoot, options, uvCommand);
  return { repoRoot, guiRoot, workspaceRoot, configPath, backendLogPath, engineLauncher, uvCommand };
}

export function resolveGuiRoot(): string {
  return path.resolve(__dirname, "..", "..");
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

export function resolveWorkspaceRoot(options: ResolveEnvPathsOptions = {}): string {
  const explicit = parseArgValue("--cwd") || process.env.HAWI_GUI_CWD || process.env.INIT_CWD;
  if (explicit) {
    return path.resolve(explicit);
  }
  if (options.isPackaged) {
    const cwd = path.resolve(options.cwd ?? process.cwd());
    return isUsablePackagedWorkspaceCwd(cwd, options.resourcesPath) ? cwd : path.resolve(homedir());
  }
  return path.resolve(options.cwd ?? process.cwd());
}

export function isUsablePackagedWorkspaceCwd(cwd: string, resourcesPath?: string): boolean {
  const normalizedCwd = path.resolve(cwd);
  if (normalizedCwd === path.parse(normalizedCwd).root) {
    return false;
  }
  if (!resourcesPath) {
    return true;
  }

  const normalizedResources = path.resolve(resourcesPath);
  if (isSamePathOrChild(normalizedCwd, normalizedResources)) {
    return false;
  }

  const appBundleRoot = findAppBundleRoot(normalizedResources);
  if (appBundleRoot && isSamePathOrChild(normalizedCwd, appBundleRoot)) {
    return false;
  }

  const installRoot = path.dirname(normalizedResources);
  return !isSamePathOrChild(normalizedCwd, installRoot);
}

function findAppBundleRoot(candidate: string): string | null {
  let current = path.resolve(candidate);
  while (true) {
    if (path.basename(current).toLowerCase().endsWith(".app")) {
      return current;
    }
    const parent = path.dirname(current);
    if (parent === current) {
      return null;
    }
    current = parent;
  }
}

function isSamePathOrChild(candidate: string, parent: string): boolean {
  const normalizedCandidate = path.resolve(candidate);
  const normalizedParent = path.resolve(parent);
  const relative = path.relative(normalizedParent, normalizedCandidate);
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}

export function resolveEngineLauncher(
  repoRoot: string,
  options: ResolveEnvPathsOptions = {},
  uvCommand = resolveUvCommand(),
): EngineLauncher {
  const override = process.env.HAWI_GUI_ENGINE_COMMAND?.trim();
  if (override) {
    return { command: override, argsPrefix: [], source: "bundled" };
  }

  const bundledCommand = resolveBundledEngineCommand(options.resourcesPath);
  if (bundledCommand) {
    return { command: bundledCommand, argsPrefix: [], source: "bundled" };
  }
  if (options.isPackaged) {
    throw new Error("Bundled hawi-engine executable was not found in the application resources.");
  }

  return {
    command: uvCommand,
    argsPrefix: buildUvEngineArgsPrefix(repoRoot),
    source: "uv",
  };
}

export function resolveBundledEngineCommand(resourcesPath = process.resourcesPath): string | null {
  if (!resourcesPath) {
    return null;
  }
  const executable = process.platform === "win32" ? "hawi-engine.exe" : "hawi-engine";
  const candidates = [
    path.join(resourcesPath, "bin", executable),
    path.join(resourcesPath, "app.asar.unpacked", "build", "bin", executable),
  ];
  return candidates.find((candidate) => existsSync(candidate)) ?? null;
}

export function buildUvEngineArgsPrefix(repoRoot: string): string[] {
  return ["run", "--project", repoRoot, "python", "-m", "hawi_engine"];
}

export function buildEngineRunArgs(
  repoRoot: string,
  engineArgs: string[],
  launcher: EngineLauncher = {
    command: resolveUvCommand(),
    argsPrefix: buildUvEngineArgsPrefix(repoRoot),
    source: "uv",
  },
): string[] {
  return [...launcher.argsPrefix, ...engineArgs];
}

export function buildEngineEnv(repoRoot: string, baseEnv: NodeJS.ProcessEnv = process.env, launcher?: EngineLauncher): NodeJS.ProcessEnv {
  if (launcher?.source === "bundled") {
    return {
      ...baseEnv,
      HAWI_GUI_ENGINE_SOURCE: "bundled",
    };
  }
  const currentPythonPath = baseEnv.PYTHONPATH?.trim();
  return {
    ...baseEnv,
    HAWI_GUI_ENGINE_SOURCE: "uv",
    PYTHONPATH: currentPythonPath ? `${repoRoot}${path.delimiter}${currentPythonPath}` : repoRoot,
  };
}

export function ensureEngineWorkspace(env: EnvPaths): void {
  mkdirSync(env.workspaceRoot, { recursive: true });
  if (existsSync(path.join(env.workspaceRoot, ".hawi", "models.yaml"))) {
    return;
  }
  const result = spawnSync(env.engineLauncher.command, buildEngineRunArgs(env.repoRoot, ["init"], env.engineLauncher), {
    cwd: env.workspaceRoot,
    encoding: "utf-8",
    env: buildEngineEnv(env.repoRoot, process.env, env.engineLauncher),
  });
  if (result.error) {
    throw new Error(`Failed to initialize Hawi workspace with ${env.engineLauncher.command}: ${result.error.message}`);
  }
  if (result.status !== 0) {
    throw new Error(formatEngineCommandError("Failed to initialize Hawi workspace", result.status, result.stderr, result.stdout));
  }
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

export function loadInspectPayload(
  repoRoot: string,
  workspaceRoot: string,
  engineLauncher: EngineLauncher,
  inspectArgs: string[] = [],
): InspectPayload {
  const result = spawnSync(engineLauncher.command, buildEngineRunArgs(repoRoot, ["--inspect", ...inspectArgs], engineLauncher), {
    cwd: workspaceRoot,
    encoding: "utf-8",
    env: buildEngineEnv(repoRoot, process.env, engineLauncher),
  });
  if (result.error) {
    throw new Error(`Failed to launch ${engineLauncher.command}: ${result.error.message}`);
  }
  if (result.status !== 0) {
    throw new Error(formatEngineCommandError("Failed to inspect hawi-engine metadata", result.status, result.stderr, result.stdout));
  }
  return JSON.parse(result.stdout) as InspectPayload;
}

function formatEngineCommandError(
  message: string,
  status: number | null,
  stderr: string | Buffer | null,
  stdout: string | Buffer | null,
): string {
  const details = [`${message} (exit ${status ?? "unknown"}).`];
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
  const pluginCatalog = metadata.plugin_catalog ?? [];
  const environPromptKey = "hawi/environ-prompt";
  const defaultPlugins = pluginCatalog.some((item) => item.key === environPromptKey) ? [environPromptKey] : [];
  return {
    version: 1,
    modelName: metadata.models[0] ?? "",
    systemPrompt: metadata.default_system_prompt,
    selectedPlugins: defaultPlugins,
    pluginConfigs: {},
    showDebug: true,
  };
}

export function sanitizeConfig(raw: PersistedConfig, metadata: InspectPayload | null): PersistedConfig {
  const modelName = metadata?.models.includes(raw.modelName) ? raw.modelName : (metadata?.models[0] ?? "");
  const pluginKeys = new Set(metadata?.plugin_catalog.map((item) => item.key) ?? []);
  const selectedPlugins = Array.isArray(raw.selectedPlugins) ? raw.selectedPlugins.filter((key) => pluginKeys.has(key)) : [];
  const rawPluginConfigs = raw.pluginConfigs && typeof raw.pluginConfigs === "object" ? raw.pluginConfigs : {};
  const pluginConfigs = Object.fromEntries(
    Object.entries(rawPluginConfigs).filter(([key, value]) => (
      pluginKeys.has(key)
      && value != null
      && typeof value === "object"
      && !Array.isArray(value)
    ))
  ) as Record<string, Record<string, unknown>>;
  return {
    version: 1,
    modelName,
    systemPrompt:
      typeof raw.systemPrompt === "string" && raw.systemPrompt.trim() ? raw.systemPrompt : (metadata?.default_system_prompt ?? ""),
    selectedPlugins,
    pluginConfigs,
    showDebug: Boolean(raw.showDebug),
  };
}

export function saveConfig(configPath: string, nextConfig: PersistedConfig): void {
  mkdirSync(path.dirname(configPath), { recursive: true });
  writeFileSync(configPath, JSON.stringify(nextConfig, null, 2), "utf-8");
}

export function preserveProviderOrder(previousModels: string[], nextModels: string[]): string[] {
  const previousProviders = providerOrder(previousModels);
  const groupedNext = groupModelsByProvider(nextModels);
  const ordered: string[] = [];

  for (const provider of previousProviders) {
    const models = groupedNext.get(provider);
    if (!models) continue;
    ordered.push(...models);
    groupedNext.delete(provider);
  }

  for (const models of groupedNext.values()) {
    ordered.push(...models);
  }
  return ordered;
}

function providerOrder(models: string[]): string[] {
  const providers: string[] = [];
  const seen = new Set<string>();
  for (const model of models) {
    const provider = modelProvider(model);
    if (!provider || seen.has(provider)) continue;
    providers.push(provider);
    seen.add(provider);
  }
  return providers;
}

function groupModelsByProvider(models: string[]): Map<string, string[]> {
  const groups = new Map<string, string[]>();
  const seen = new Set<string>();
  for (const model of models) {
    if (seen.has(model)) continue;
    seen.add(model);
    const provider = modelProvider(model);
    if (!provider) continue;
    groups.set(provider, [...(groups.get(provider) ?? []), model]);
  }
  return groups;
}

function modelProvider(model: string): string {
  return model.split("/", 1)[0] ?? "";
}
