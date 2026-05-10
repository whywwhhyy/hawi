import { describe, expect, it } from "vitest";
import { VERSION, makeCommand } from "./protocol";

describe("protocol", () => {
  it("creates request commands with protocol version", () => {
    const command = makeCommand("enqueue", { content: "hi" }, "req-1");
    expect(command.version).toBe(VERSION);
    expect(command.id).toBe("req-1");
    expect(command.payload.content).toBe("hi");
  });
});
