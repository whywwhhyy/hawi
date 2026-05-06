import { describe, expect, it } from "vitest";
import { VERSION, makeCommand, parseNdjsonChunk } from "./protocol";

describe("protocol", () => {
  it("splits NDJSON chunks and preserves partial lines", () => {
    const first = JSON.stringify({ version: VERSION, type: "core.ready", payload: {} });
    const second = JSON.stringify({ version: VERSION, type: "pong", payload: { ok: true } });

    const result = parseNdjsonChunk("", `${first}\n${second.slice(0, 12)}`);
    expect(result.frames).toHaveLength(1);
    expect(result.buffer).toBe(second.slice(0, 12));

    const done = parseNdjsonChunk(result.buffer, `${second.slice(12)}\n`);
    expect(done.frames.map((frame) => frame.type)).toEqual(["pong"]);
    expect(done.errors).toEqual([]);
  });

  it("creates request commands with protocol version", () => {
    const command = makeCommand("enqueue", { content: "hi" }, "req-1");
    expect(command.version).toBe(VERSION);
    expect(command.id).toBe("req-1");
    expect(command.payload.content).toBe("hi");
  });
});
