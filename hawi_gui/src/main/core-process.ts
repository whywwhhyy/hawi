import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import type { CoreCommand, CoreCommandType, CoreFrame, InspectPayload, PersistedConfig } from "../shared/protocol";
import { VERSION } from "../shared/protocol";
import { TLVDecoder, TYPE_JSON_FRAME, encodeJsonFrame } from "./tlv";

export const GRACEFUL_SHUTDOWN_TIMEOUT_MS = 800;
export const DEFAULT_COMMAND_TIMEOUT_MS = 15_000;

export type EmitToRenderer = (channel: string, payload: unknown) => void;

export class CoreProcess {
  private child: ChildProcessWithoutNullStreams | null = null;
  private decoder = new TLVDecoder();
  private pending = new Map<string, { resolve: (frame: CoreFrame) => void; reject: (error: Error) => void }>();
  private sequence = 0;

  constructor(
    private readonly emitToRenderer: EmitToRenderer,
    private readonly repoRoot: string,
    private readonly workspaceRoot: string,
    private readonly backendLogPath: string,
    private readonly uvCommand: string
  ) {}

  isRunning(): boolean {
    return this.child !== null && !this.child.killed;
  }

  start(nextConfig: PersistedConfig, metadata: InspectPayload): void {
    if (!nextConfig.modelName) {
      return;
    }
    this.stop("start-replace-existing");
    mkdirSync(path.dirname(this.backendLogPath), { recursive: true });
    writeFileSync(this.backendLogPath, "", "utf-8"); // 每次启动清空日志
    const pluginConfigPath = path.join(tmpdir(), `hawi-gui-plugins-${process.pid}.json`);
    writeFileSync(pluginConfigPath, JSON.stringify(nextConfig.pluginConfigs, null, 2), "utf-8");

    const args = [
      "run",
      "--project",
      this.repoRoot,
      "hawi-engine",
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
      "tool_call_purpose",
      "str",
      "用一句话说明本次工具调用的目的；允许与其他调用重复，会显示在工具标题旁边。",
      "--log-file",
      this.backendLogPath
    ];
    const child = spawn(this.uvCommand, args, {
      cwd: this.workspaceRoot,
      stdio: ["pipe", "pipe", "pipe"],
      env: process.env
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
    });
    this.emitToRenderer("core:spawn", { args: args.slice(1), cwd: this.workspaceRoot, logFile: this.backendLogPath });
  }

  restart(nextConfig: PersistedConfig, metadata: InspectPayload): void {
    this.stop("restart");
    this.start(nextConfig, metadata);
  }

  stop(reason: string): void {
    const child = this.child;
    if (!child) {
      return;
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
        payload: { reason }
      });
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
      return Promise.reject(new Error("hawi-engine is not running"));
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
            payload: { ok: false, code: "bad_frame", message: `Invalid core frame: ${value.toString("utf-8")}` }
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
            message: error instanceof Error ? error.message : String(error)
          }
        });
        continue;
      }
      if (frame.id && this.pending.has(frame.id) && (frame.type === "ack" || frame.type === "error" || frame.type === "pong" || frame.type === "core.status")) {
        const pending = this.pending.get(frame.id);
        this.pending.delete(frame.id);
        if (frame.type === "error") {
          pending?.reject(new Error(String((frame.payload as Record<string, unknown>).message ?? "Core error")));
        } else {
          pending?.resolve(frame);
        }
      }
      this.emitToRenderer("core:event", frame);
    }
  }
}
