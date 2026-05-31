import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import type { CoreCommand, CoreCommandType, CoreFrame, InspectPayload, PersistedConfig, SessionLaunchProfile } from "../shared/protocol";
import { VERSION } from "../shared/protocol";
import { buildEngineEnv, buildEngineRunArgs, type EngineLauncher } from "./config";
import { toolCallPurposeEngineArgs } from "./tool-parameters";
import { TLVDecoder, TYPE_JSON_FRAME, encodeJsonFrame } from "./tlv";

export const GRACEFUL_SHUTDOWN_TIMEOUT_MS = 800;
export const DEFAULT_COMMAND_TIMEOUT_MS = 15_000;

export type EmitToRenderer = (channel: string, payload: unknown) => void;

export interface CoreStartOptions {
  initialSessionId?: string;
  initialSessionName?: string;
  launchProfile?: SessionLaunchProfile;
}

export class CoreCommandError extends Error {
  readonly code: string;
  readonly details: unknown;
  readonly frame: CoreFrame;

  constructor(frame: CoreFrame) {
    const payload = frame.payload as Record<string, unknown>;
    super(String(payload.message ?? "Core error"));
    this.name = "CoreCommandError";
    this.code = typeof payload.code === "string" ? payload.code : "error";
    this.details = payload.details;
    this.frame = frame;
    Object.setPrototypeOf(this, CoreCommandError.prototype);
  }
}

export class CoreProcess {
  private child: ChildProcessWithoutNullStreams | null = null;
  private decoder = new TLVDecoder();
  private pending = new Map<string, { resolve: (frame: CoreFrame) => void; reject: (error: Error) => void }>();
  private sequence = 0;
  private stopWaiters: Array<() => void> = [];

  private static instanceSequence = 0;
  private readonly instanceId: number;

  constructor(
    private readonly emitToRenderer: EmitToRenderer,
    private readonly repoRoot: string,
    private readonly workspaceRoot: string,
    private readonly backendLogPath: string,
    private readonly engineLauncher: EngineLauncher,
  ) {
    CoreProcess.instanceSequence += 1;
    this.instanceId = CoreProcess.instanceSequence;
  }

  isRunning(): boolean {
    return this.child !== null && !this.child.killed;
  }

  start(
    nextConfig: PersistedConfig,
    metadata: InspectPayload,
    refreshedProviders: Iterable<string> = [],
    options: CoreStartOptions = {},
  ): void {
    if (!nextConfig.modelName) {
      return;
    }
    this.stop("start-replace-existing");
    mkdirSync(path.dirname(this.backendLogPath), { recursive: true });
    writeFileSync(this.backendLogPath, "", "utf-8"); // 每次启动清空日志
    const pluginConfigPath = path.join(tmpdir(), `hawi-gui-plugins-${process.pid}-${this.instanceId}.json`);
    writeFileSync(pluginConfigPath, JSON.stringify(nextConfig.pluginConfigs, null, 2), "utf-8");

    const refreshArgs = [...refreshedProviders].flatMap((provider) => ["--refresh-provider", provider]);
    const launchProfileArgs = options.launchProfile ? ["--gui-launch-profile", JSON.stringify(options.launchProfile)] : [];
    const initialSessionArgs = [
      ...(options.initialSessionId ? ["--initial-session-id", options.initialSessionId] : []),
      ...(options.initialSessionName ? ["--initial-session-name", options.initialSessionName] : []),
    ];
    const args = buildEngineRunArgs(
      this.repoRoot,
      [
        "--model",
        nextConfig.modelName,
        ...refreshArgs,
        "--transport",
        "stdio",
        "--system-prompt",
        nextConfig.systemPrompt || metadata.default_system_prompt,
        "--plugins",
        nextConfig.selectedPlugins.join(","),
        "--plugin-config",
        pluginConfigPath,
        ...launchProfileArgs,
        ...initialSessionArgs,
        ...toolCallPurposeEngineArgs(nextConfig.toolCallPurposeEnabled),
        "--log-file",
        this.backendLogPath,
      ],
      this.engineLauncher,
    );
    const child = spawn(this.engineLauncher.command, args, {
      cwd: this.workspaceRoot,
      stdio: ["pipe", "pipe", "pipe"],
      env: buildEngineEnv(this.repoRoot, process.env, this.engineLauncher),
    });
    this.child = child;
    this.decoder = new TLVDecoder();
    child.stdout.on("data", (chunk: Buffer) => this.handleStdout(chunk));
    child.stderr.setEncoding("utf-8");
    child.stderr.on("data", (chunk: string) => {
      this.emitToRenderer("core:stderr", chunk);
    });
    child.on("exit", (code, signal) => {
      const wasCurrent = this.child === child;
      if (wasCurrent) {
        this.child = null;
        const error = new Error(`hawi-engine exited (${code ?? "null"} ${signal ?? ""})`);
        for (const pending of this.pending.values()) {
          pending.reject(error);
        }
        this.pending.clear();
      }
      this.emitToRenderer("core:exit", { code, signal });
      this.resolveStopWaiters();
    });
    this.emitToRenderer("core:spawn", {
      command: this.engineLauncher.command,
      args: this.engineLauncher.source === "uv" ? args.slice(1) : args,
      cwd: this.workspaceRoot,
      engineSource: this.engineLauncher.source,
      logFile: this.backendLogPath,
    });
  }

  restart(nextConfig: PersistedConfig, metadata: InspectPayload, refreshedProviders: Iterable<string> = []): void {
    this.stop("restart");
    this.start(nextConfig, metadata, refreshedProviders);
  }

  startReadonly(): void {
    this.stop("readonly-start-replace-existing");
    mkdirSync(path.dirname(this.backendLogPath), { recursive: true });
    writeFileSync(this.backendLogPath, "", "utf-8");
    const args = buildEngineRunArgs(
      this.repoRoot,
      [
        "--readonly",
        "--transport",
        "stdio",
        "--log-file",
        this.backendLogPath,
      ],
      this.engineLauncher,
    );
    const child = spawn(this.engineLauncher.command, args, {
      cwd: this.workspaceRoot,
      stdio: ["pipe", "pipe", "pipe"],
      env: buildEngineEnv(this.repoRoot, process.env, this.engineLauncher),
    });
    this.child = child;
    this.decoder = new TLVDecoder();
    child.stdout.on("data", (chunk: Buffer) => this.handleStdout(chunk));
    child.stderr.setEncoding("utf-8");
    child.stderr.on("data", (chunk: string) => {
      this.emitToRenderer("core:stderr", chunk);
    });
    child.on("exit", (code, signal) => {
      const wasCurrent = this.child === child;
      if (wasCurrent) {
        this.child = null;
        const error = new Error(`hawi-engine readonly exited (${code ?? "null"} ${signal ?? ""})`);
        for (const pending of this.pending.values()) {
          pending.reject(error);
        }
        this.pending.clear();
      }
      this.emitToRenderer("core:exit", { code, signal });
      this.resolveStopWaiters();
    });
    this.emitToRenderer("core:spawn", {
      command: this.engineLauncher.command,
      args: this.engineLauncher.source === "uv" ? args.slice(1) : args,
      cwd: this.workspaceRoot,
      engineSource: this.engineLauncher.source,
      logFile: this.backendLogPath,
      mode: "readonly",
    });
  }

  stop(reason: string): Promise<void> {
    const child = this.child;
    if (!child) {
      return Promise.resolve();
    }
    this.child = null;
    const error = new Error("hawi-engine was stopped");
    for (const pending of this.pending.values()) {
      pending.reject(error);
    }
    this.pending.clear();
    try {
      this.writeFrame(child, {
        version: "hawi.core.v1",
        type: "shutdown",
        id: this.nextId(),
        payload: { reason },
      });
    } catch {
      // Process may already be closing.
    }
    return new Promise((resolve) => {
      this.stopWaiters.push(resolve);
      setTimeout(() => {
        if (!child.killed) {
          child.kill();
        }
        this.resolveStopWaiters();
      }, GRACEFUL_SHUTDOWN_TIMEOUT_MS).unref();
    });
  }

  sendCommand(type: CoreCommandType, payload: Record<string, unknown>, timeoutMs = DEFAULT_COMMAND_TIMEOUT_MS): Promise<CoreFrame> {
    if (!this.child || !this.child.stdin.writable) {
      return Promise.reject(new Error("hawi-engine is not running"));
    }
    const id = this.nextId();
    const frame: CoreCommand = {
      version: "hawi.core.v1",
      type,
      id,
      payload,
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
    child?.stdin.write(encodeJsonFrame(frame));
  }

  private handleStdout(chunk: Buffer): void {
    for (const { typeByte, value } of this.decoder.push(chunk)) {
      if (typeByte !== TYPE_JSON_FRAME) {
        // Reserved (binary blob, etc.) — ignore in Plan 3.
        continue;
      }
      let frame: CoreFrame;
      try {
        const parsed = JSON.parse(value.toString("utf-8")) as CoreFrame;
        if (parsed.version !== VERSION || typeof parsed.type !== "string") {
          this.emitToRenderer("core:event", {
            version: VERSION,
            type: "error",
            payload: { ok: false, code: "bad_frame", message: `Invalid core frame: ${value.toString("utf-8")}` },
          });
          continue;
        }
        frame = parsed;
      } catch (error) {
        this.emitToRenderer("core:event", {
          version: VERSION,
          type: "error",
          payload: {
            ok: false,
            code: "bad_frame",
            message: error instanceof Error ? error.message : String(error),
          },
        });
        continue;
      }
      if (frame.id && isCommandResponseFrame(frame)) {
        const pending = this.pending.get(frame.id);
        if (pending) {
          this.pending.delete(frame.id);
          if (frame.type === "error") {
            pending.reject(new CoreCommandError(frame));
          } else {
            pending.resolve(frame);
          }
        }
        continue;
      }
      this.emitToRenderer("core:event", frame);
    }
  }

  private resolveStopWaiters(): void {
    const waiters = this.stopWaiters.splice(0);
    for (const resolve of waiters) {
      resolve();
    }
  }
}

function isCommandResponseFrame(frame: CoreFrame): boolean {
  return frame.type === "ack" || frame.type === "error" || frame.type === "pong" || frame.type === "core.status";
}
