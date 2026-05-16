import { describe, expect, it } from "vitest";
import { VERSION, type CoreFrame } from "../shared/protocol";
import { CoreCommandError, CoreProcess } from "./core-process";
import type { EngineLauncher } from "./config";
import { encodeJsonFrame } from "./tlv";

type RoutableCoreProcess = {
  pending: Map<string, { resolve: (frame: CoreFrame) => void; reject: (error: Error) => void }>;
  handleStdout(chunk: Buffer): void;
};

function makeRoutableCore(events: Array<{ channel: string; payload: unknown }>): RoutableCoreProcess {
  const launcher: EngineLauncher = {
    command: "uv",
    argsPrefix: ["run", "--project", "/repo", "python", "-m", "hawi_engine"],
    source: "uv",
  };
  const core = new CoreProcess(
    (channel, payload) => events.push({ channel, payload }),
    "/repo",
    "/workspace",
    "/workspace/.hawi/hawi-engine.log",
    launcher,
  );
  return core as unknown as RoutableCoreProcess;
}

describe("CoreProcess response routing", () => {
  it("resolves pending command responses without broadcasting them as global events", async () => {
    const events: Array<{ channel: string; payload: unknown }> = [];
    const core = makeRoutableCore(events);
    const pending = new Promise<CoreFrame>((resolve, reject) => {
      core.pending.set("req-1", { resolve, reject });
    });

    core.handleStdout(
      encodeJsonFrame({
        version: VERSION,
        type: "ack",
        id: "req-1",
        payload: { command: "refresh_models", ok: true },
      }),
    );

    await expect(pending).resolves.toMatchObject({
      id: "req-1",
      type: "ack",
      payload: { command: "refresh_models", ok: true },
    });
    expect(events).toEqual([]);
  });

  it("rejects pending command errors with the structured error frame", async () => {
    const events: Array<{ channel: string; payload: unknown }> = [];
    const core = makeRoutableCore(events);
    const pending = new Promise<CoreFrame>((resolve, reject) => {
      core.pending.set("req-err", { resolve, reject });
    });

    core.handleStdout(
      encodeJsonFrame({
        version: VERSION,
        type: "error",
        id: "req-err",
        payload: {
          ok: false,
          code: "refresh_failed",
          message: "provider refresh failed",
          details: { provider: "local" },
        },
      }),
    );

    await expect(pending).rejects.toMatchObject({
      name: "CoreCommandError",
      code: "refresh_failed",
      message: "provider refresh failed",
      details: { provider: "local" },
    });
    await pending.catch((error) => {
      expect(error).toBeInstanceOf(CoreCommandError);
      expect(error.frame).toMatchObject({ id: "req-err", type: "error" });
    });
    expect(events).toEqual([]);
  });

  it("continues broadcasting engine events that are not command responses", () => {
    const events: Array<{ channel: string; payload: unknown }> = [];
    const core = makeRoutableCore(events);

    core.handleStdout(
      encodeJsonFrame({
        version: VERSION,
        type: "debug.info",
        payload: { message: "hello" },
      }),
    );

    expect(events).toEqual([
      {
        channel: "core:event",
        payload: {
          version: VERSION,
          type: "debug.info",
          payload: { message: "hello" },
        },
      },
    ]);
  });
});
