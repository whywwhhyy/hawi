import { describe, expect, it } from "vitest";
import { VERSION, makeCommand } from "./protocol";

describe("protocol", () => {
  it("creates request commands with protocol version", () => {
    const command = makeCommand("enqueue", { content: "hi" }, "req-1");
    expect(command.version).toBe(VERSION);
    expect(command.id).toBe("req-1");
    expect(command.payload.content).toBe("hi");
  });

  it("accepts blob commands and content-part payloads", () => {
    const command = makeCommand("blob.info", { blob_id: "a".repeat(64) }, "blob-1");
    const enqueue = makeCommand("enqueue", {
      content: [{
        type: "image",
        source: {
          kind: "blob",
          blob_id: "a".repeat(64),
          uri: `hawi-blob://${"a".repeat(64)}`,
          mime_type: "image/png",
          filename: "screen.png"
        }
      }]
    }, "req-media");

    expect(command.type).toBe("blob.info");
    expect(Array.isArray(enqueue.payload.content)).toBe(true);
  });
});
