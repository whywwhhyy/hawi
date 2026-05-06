import { describe, expect, it } from "vitest";
import { VERSION, type CoreFrame } from "../shared/protocol";
import { createInitialState, reduceCoreEvent } from "./state";

function frame(type: string, payload: Record<string, unknown>): CoreFrame {
  return { version: VERSION, type, payload };
}

describe("core event reducer", () => {
  it("splits agent content around tool calls", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("run.start", { run_id: "run-1", user_content: "hi", queue: "normal" }));
    state = reduceCoreEvent(state, frame("run.text_delta", { run_id: "run-1", delta: "before" }));
    state = reduceCoreEvent(state, frame("tool.call_start", { run_id: "run-1", tool_call_id: "tc-1", tool_name: "web.search" }));
    state = reduceCoreEvent(state, frame("tool.result", { run_id: "run-1", tool_call_id: "tc-1", tool_name: "web.search", success: true, output: "ok" }));
    state = reduceCoreEvent(state, frame("run.text_delta", { run_id: "run-1", delta: "after" }));

    expect(state.nodes.map((node) => node.kind)).toEqual(["user", "agent", "tool", "agent"]);
    expect(state.nodes[1].content).toBe("before");
    expect(state.nodes[3].content).toBe("after");
  });

  it("does not create an empty agent node when a tool appears first", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("run.start", { run_id: "run-2", user_content: "hi", queue: "normal" }));
    state = reduceCoreEvent(state, frame("tool.call_start", { run_id: "run-2", tool_call_id: "tc-2", tool_name: "calc" }));
    state = reduceCoreEvent(state, frame("tool.result", { run_id: "run-2", tool_call_id: "tc-2", tool_name: "calc", success: true, output: "4" }));
    state = reduceCoreEvent(state, frame("run.text_delta", { run_id: "run-2", delta: "done" }));

    expect(state.nodes.map((node) => node.kind)).toEqual(["user", "tool", "agent"]);
    expect(state.nodes[2].content).toBe("done");
  });

  it("replaces full tool argument snapshots instead of appending them", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("tool.call_start", { run_id: "run-args", tool_call_id: "tc-args", tool_name: "calc" }));
    state = reduceCoreEvent(state, frame("tool.call_delta", { tool_call_id: "tc-args", delta: "{\"expression\":", is_streaming: true }));
    state = reduceCoreEvent(state, frame("tool.call_delta", { tool_call_id: "tc-args", delta: "\"1+1\"}", is_streaming: true }));
    state = reduceCoreEvent(state, frame("tool.call_delta", { tool_call_id: "tc-args", delta: "{\"expression\":\"1+1\"}", is_streaming: false }));

    expect(state.nodes[0].tool?.argsRaw).toBe("{\"expression\":\"1+1\"}");
  });

  it("shows failed tool error messages", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("tool.call_start", { run_id: "run-fail", tool_call_id: "tc-fail", tool_name: "calc" }));
    state = reduceCoreEvent(state, frame("tool.result", { tool_call_id: "tc-fail", tool_name: "calc", success: false, error: "Parameter validation failed" }));

    expect(state.nodes[0].tool?.status).toBe("fail");
    expect(state.nodes[0].tool?.resultPreview).toBe("Parameter validation failed");
  });

  it("updates status, metadata, retry, and error nodes", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("core.status", { scheduler_state: "RUNNING", agent_state: "RUNNING", queue_lengths: { urgent: 1, high_prio: 2, normal: 3 } }));
    state = reduceCoreEvent(state, frame("model.metadata", { input_tokens: 2, output_tokens: 3, total_tokens: 5, latency_ms: 44 }));
    state = reduceCoreEvent(state, frame("model.retry", { attempt: 1, max_retries: 3, error_type: "network", error_message: "retrying" }));
    state = reduceCoreEvent(state, frame("error", { message: "boom" }));

    expect(state.schedulerState).toBe("RUNNING");
    expect(state.queueLengths).toEqual({ urgent: 1, high_prio: 2, normal: 3 });
    expect(state.metadataLines[0]).toContain("total=5");
    expect(state.nodes.map((node) => node.kind)).toContain("error");
  });

  it("adds debug output as low-emphasis chat notes", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("debug.info", { message: "stderr line" }));

    expect(state.debugLines).toEqual(["stderr line"]);
    expect(state.nodes.map((node) => node.kind)).toEqual(["debug"]);
    expect(state.nodes[0].content).toBe("stderr line");
  });

  it("renders run stops as divider nodes", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("run.start", { run_id: "run-3", user_content: "hi", queue: "normal" }));
    state = reduceCoreEvent(state, frame("run.stop", { run_id: "run-3", stop_reason: "end_turn", duration_ms: 1234 }));

    expect(state.nodes.map((node) => node.kind)).toEqual(["user", "divider"]);
    expect(state.nodes[1].content).toBe("end_turn · 1.2s");
  });

  it("clears visible chat while preserving status", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("core.status", { scheduler_state: "RUNNING", agent_state: "RUNNING", queue_lengths: { urgent: 1, high_prio: 0, normal: 0 } }));
    state = reduceCoreEvent(state, frame("run.start", { run_id: "run-4", user_content: "hi", queue: "normal" }));
    state = reduceCoreEvent(state, frame("debug.info", { message: "debug" }));

    state = reduceCoreEvent(state, frame("gui.clear_chat", {}));

    expect(state.nodes).toEqual([]);
    expect(state.debugLines).toEqual([]);
    expect(state.schedulerState).toBe("RUNNING");
    expect(state.queueLengths).toEqual({ urgent: 1, high_prio: 0, normal: 0 });
  });
});
