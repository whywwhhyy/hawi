import { describe, expect, it } from "vitest";
import { VERSION, type CoreFrame } from "../shared/protocol";
import { createInitialState, reduceCoreEvent } from "./state";

function frame(type: string, payload: Record<string, unknown>): CoreFrame {
  return { version: VERSION, type, payload };
}

describe("core event reducer", () => {
  it("shows materialized high priority messages as normal chat history", () => {
    let state = createInitialState();

    state = reduceCoreEvent(state, frame("run.start", {
      run_id: "run-1",
      message_id: "steer-1",
      user_content: "new priority",
      queue: "normal",
      display_message_type: "normal"
    }));

    expect(state.nodes).toHaveLength(1);
    expect(state.nodes[0]).toMatchObject({
      id: "user-message-steer-1",
      kind: "user",
      queue: "normal",
      displayMessageType: "normal",
      content: "new priority"
    });
    expect(state.activeRunId).toBe("run-1");
    expect(state.sessionMessageCount).toBe(1);
  });

  it("loads session message history into chat nodes", () => {
    const state = reduceCoreEvent(createInitialState(), frame("gui.load_session_history", {
      message_history: [
        {
          run_id: "run-1",
          role: "user",
          content: [{ type: "text", text: "hello" }],
          metadata: { queue: "normal", display_message_type: "normal" }
        },
        {
          run_id: "run-1",
          role: "assistant",
          content: [
            { type: "reasoning", reasoning: "thinking" },
            { type: "text", text: "answer" }
          ],
          metadata: null
        }
      ],
      context_usage: {
        used_tokens: 42,
        max_context_tokens: 1000,
        usage_ratio: 0.042,
        source: "provider_usage"
      }
    }));

    expect(state.nodes.map((node) => node.kind)).toEqual(["user", "thinking", "agent"]);
    expect(state.nodes[0].content).toBe("hello");
    expect(state.nodes[1]).toMatchObject({ content: "thinking", complete: true });
    expect(state.nodes[2]).toMatchObject({ content: "answer", complete: true });
    expect(state.sessionMessageCount).toBe(2);
    expect(state.contextUsage).toEqual({
      usedTokens: 42,
      maxContextTokens: 1000,
      ratio: 0.042,
      source: "provider_usage"
    });
  });

  it("clears stale context usage when loading a session without usage", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("model.metadata", {
      context_tokens: 900,
      max_context_tokens: 1000,
      context_ratio: 0.9,
      context_source: "provider_usage"
    }));

    state = reduceCoreEvent(state, frame("gui.load_session_history", { message_history: [] }));

    expect(state.contextUsage).toBeUndefined();
  });

  it("counts one assistant message across thinking and answer deltas", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("run.start", { run_id: "run-count", user_content: "hi", queue: "normal" }));
    state = reduceCoreEvent(state, frame("run.thinking_delta", { run_id: "run-count", delta: "thinking" }));
    state = reduceCoreEvent(state, frame("run.text_delta", { run_id: "run-count", delta: "answer" }));

    expect(state.nodes.map((node) => node.kind)).toEqual(["user", "thinking", "agent"]);
    expect(state.sessionMessageCount).toBe(2);
  });

  it("shows materialized steer messages with steer display type", () => {
    let state = createInitialState();

    state = reduceCoreEvent(state, frame("run.start", {
      run_id: "run-1",
      message_id: "steer-1",
      user_content: "new priority",
      queue: "high_prio",
      display_message_type: "steer"
    }));

    expect(state.nodes).toHaveLength(1);
    expect(state.nodes[0]).toMatchObject({
      id: "user-message-steer-1",
      kind: "user",
      queue: "high_prio",
      displayMessageType: "steer",
      content: "new priority"
    });
    expect(state.activeRunId).toBe("run-1");
    expect(state.runs["run-1"]).toEqual({});
  });

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

  it("splits thinking content around tool calls", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("run.start", { run_id: "run-thinking", user_content: "hi", queue: "normal" }));
    state = reduceCoreEvent(state, frame("run.thinking_delta", { run_id: "run-thinking", delta: "before tool" }));
    state = reduceCoreEvent(state, frame("tool.call_start", { run_id: "run-thinking", tool_call_id: "tc-thinking", tool_name: "calc" }));
    state = reduceCoreEvent(state, frame("tool.result", { run_id: "run-thinking", tool_call_id: "tc-thinking", tool_name: "calc", success: true, output: "4" }));
    state = reduceCoreEvent(state, frame("run.thinking_delta", { run_id: "run-thinking", delta: "after tool" }));

    expect(state.nodes.map((node) => node.kind)).toEqual(["user", "thinking", "tool", "thinking"]);
    expect(state.nodes[1].content).toBe("before tool");
    expect(state.nodes[1].complete).toBe(true);
    expect(state.nodes[3].content).toBe("after tool");
    expect(state.nodes[3].complete).toBe(false);
  });

  it("marks thinking complete when answer text starts", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("run.start", { run_id: "run-text-after-thinking", user_content: "hi", queue: "normal" }));
    state = reduceCoreEvent(state, frame("run.thinking_delta", { run_id: "run-text-after-thinking", delta: "thinking" }));
    state = reduceCoreEvent(state, frame("run.text_delta", { run_id: "run-text-after-thinking", delta: "answer" }));

    expect(state.nodes.map((node) => node.kind)).toEqual(["user", "thinking", "agent"]);
    expect(state.nodes[1].complete).toBe(true);
  });

  it("marks thinking complete when run stops", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("run.start", { run_id: "run-stop-thinking", user_content: "hi", queue: "normal" }));
    state = reduceCoreEvent(state, frame("run.thinking_delta", { run_id: "run-stop-thinking", delta: "thinking" }));
    state = reduceCoreEvent(state, frame("run.stop", { run_id: "run-stop-thinking", stop_reason: "end_turn" }));

    expect(state.nodes.map((node) => node.kind)).toEqual(["user", "thinking", "divider"]);
    expect(state.nodes[1].complete).toBe(true);
  });

  it("replaces full tool argument snapshots instead of appending them", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("tool.call_start", { run_id: "run-args", tool_call_id: "tc-args", tool_name: "calc" }));
    state = reduceCoreEvent(state, frame("tool.call_delta", { tool_call_id: "tc-args", delta: "{\"expression\":", is_streaming: true }));
    expect(state.nodes[0].tool?.arguments).toBeUndefined();
    state = reduceCoreEvent(state, frame("tool.call_delta", { tool_call_id: "tc-args", delta: "\"1+1\"}", is_streaming: true }));
    expect(state.nodes[0].tool?.arguments).toEqual({ expression: "1+1" });
    state = reduceCoreEvent(state, frame("tool.call_delta", { tool_call_id: "tc-args", delta: "{\"expression\":\"1+1\"}", is_streaming: false }));

    expect(state.nodes[0].tool?.argsRaw).toBe("{\"expression\":\"1+1\"}");
    expect(state.nodes[0].tool?.arguments).toEqual({ expression: "1+1" });
    expect(state.nodes[0].tool?.argsState).toBe("streaming");
  });

  it("keeps the last parsed tool arguments while receiving incomplete JSON", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("tool.call_start", { run_id: "run-stream", tool_call_id: "tc-stream", tool_name: "read" }));
    state = reduceCoreEvent(state, frame("tool.call_delta", { tool_call_id: "tc-stream", delta: "{\"path\":\"a.txt\"}", is_streaming: false }));
    state = reduceCoreEvent(state, frame("tool.call_delta", { tool_call_id: "tc-stream", delta: "{\"path\":\"a.txt\",\"limit\":", is_streaming: false }));

    expect(state.nodes[0].tool?.argsRaw).toBe("{\"path\":\"a.txt\",\"limit\":");
    expect(state.nodes[0].tool?.arguments).toEqual({ path: "a.txt" });
  });

  it("parses complete top-level arguments before the whole object closes", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("tool.call_start", { run_id: "run-partial-object", tool_call_id: "tc-partial-object", tool_name: "fetch" }));
    state = reduceCoreEvent(state, frame("tool.call_delta", {
      tool_call_id: "tc-partial-object",
      delta: "{\"url\":\"https://example.com\"",
      is_streaming: false
    }));

    expect(state.nodes[0].tool?.arguments).toEqual({ url: "https://example.com" });
  });

  it("marks tool arguments complete on call stop", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("tool.call_start", { run_id: "run-stop", tool_call_id: "tc-stop", tool_name: "read" }));
    state = reduceCoreEvent(state, frame("tool.call_stop", { tool_call_id: "tc-stop", tool_name: "read", arguments: { path: "a.txt", limit: 5 } }));

    expect(state.nodes[0].tool?.arguments).toEqual({ path: "a.txt", limit: 5 });
    expect(state.nodes[0].tool?.argsState).toBe("complete");
  });

  it("keeps discovered tools pending until execution starts", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("tool.call_start", {
      run_id: "run-pending-tool",
      tool_call_id: "tc-pending",
      tool_name: "read",
      status: "pending"
    }));
    state = reduceCoreEvent(state, frame("tool.call_start", {
      run_id: "run-pending-tool",
      tool_call_id: "tc-pending",
      tool_name: "read",
      status: "running",
      arguments: { path: "a.txt" }
    }));

    expect(state.nodes).toHaveLength(1);
    expect(state.nodes[0].tool?.status).toBe("running");
    expect(state.nodes[0].tool?.arguments).toEqual({ path: "a.txt" });
    expect(state.nodes[0].tool?.argsState).toBe("complete");
  });

  it("extracts tool call purposes from injected arguments", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("tool.call_start", { run_id: "run-desc", tool_call_id: "tc-desc", tool_name: "read" }));
    state = reduceCoreEvent(state, frame("tool.call_delta", {
      tool_call_id: "tc-desc",
      delta: "{\"path\":\"notes.md\",\"tool_call_purpose\":\"Read design notes\"}",
      is_streaming: false
    }));

    expect(state.nodes[0].tool?.description).toBe("Read design notes");
    expect(state.nodes[0].tool?.arguments).toEqual({ path: "notes.md" });

    state = reduceCoreEvent(state, frame("tool.call_stop", {
      tool_call_id: "tc-desc",
      tool_name: "read",
      arguments: {
        path: "notes.md",
        tool_call_purpose: "Read final notes"
      }
    }));

    expect(state.nodes[0].tool?.description).toBe("Read final notes");
    expect(state.nodes[0].tool?.arguments).toEqual({ path: "notes.md" });
    expect(state.nodes[0].tool?.argsRaw).not.toContain("tool_call_purpose");
  });

  it("shows failed tool error messages", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("tool.call_start", { run_id: "run-fail", tool_call_id: "tc-fail", tool_name: "calc" }));
    state = reduceCoreEvent(state, frame("tool.result", { tool_call_id: "tc-fail", tool_name: "calc", success: false, error: "Parameter validation failed" }));

    expect(state.nodes[0].tool?.status).toBe("fail");
    expect(state.nodes[0].tool?.resultPreview).toBe("Error: Parameter validation failed");
  });

  it("shows failed tool output when error is empty", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("tool.call_start", { run_id: "run-output-fail", tool_call_id: "tc-output-fail", tool_name: "shell" }));
    state = reduceCoreEvent(state, frame("tool.result", {
      tool_call_id: "tc-output-fail",
      tool_name: "shell",
      success: false,
      output: "Exit code: 7\n\nStderr:\noops",
      error: ""
    }));

    expect(state.nodes[0].tool?.status).toBe("fail");
    expect(state.nodes[0].tool?.resultPreview).toContain("Exit code: 7");
    expect(state.nodes[0].tool?.resultPreview).toContain("oops");
  });

  it("shows both failed tool error and output when both are present", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("tool.call_start", { run_id: "run-error-output", tool_call_id: "tc-error-output", tool_name: "shell" }));
    state = reduceCoreEvent(state, frame("tool.result", {
      tool_call_id: "tc-error-output",
      tool_name: "shell",
      success: false,
      output: "Exit code: 7\n\nStdout:\nbefore\n\nStderr:\noops",
      error: "Command exited with status 7"
    }));

    expect(state.nodes[0].tool?.resultPreview).toContain("Error: Command exited with status 7");
    expect(state.nodes[0].tool?.resultPreview).toContain("Output:");
    expect(state.nodes[0].tool?.resultPreview).toContain("before");
    expect(state.nodes[0].tool?.resultPreview).toContain("oops");
  });

  it("keeps unmatched tool results separate instead of guessing by name", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("tool.call_start", { run_id: "run-pending-id", tool_call_id: "pending:req:0", tool_name: "WebPlugin__fetch" }));
    state = reduceCoreEvent(state, frame("tool.result", {
      run_id: "run-pending-id",
      tool_call_id: "tc-real",
      tool_name: "WebPlugin__fetch",
      success: false,
      error: "抓取失败: DNS lookup failed"
    }));

    expect(state.nodes).toHaveLength(2);
    expect(state.nodes[0].tool?.toolCallId).toBe("pending:req:0");
    expect(state.nodes[0].tool?.status).toBe("running");
    expect(state.nodes[1].tool?.toolCallId).toBe("tc-real");
    expect(state.nodes[1].tool?.status).toBe("fail");
    expect(state.nodes[1].tool?.resultPreview).toContain("DNS lookup failed");
  });

  it("creates a visible failed tool node for orphan tool results", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("tool.result", {
      run_id: "run-orphan",
      tool_call_id: "tc-orphan",
      tool_name: "WebPlugin__fetch",
      success: false,
      error: "抓取失败"
    }));

    expect(state.nodes.map((node) => node.kind)).toEqual(["tool"]);
    expect(state.nodes[0].tool?.status).toBe("fail");
    expect(state.nodes[0].tool?.resultPreview).toContain("抓取失败");
  });

  it("formats object tool outputs as JSON", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("tool.call_start", { run_id: "run-json", tool_call_id: "tc-json", tool_name: "search" }));
    state = reduceCoreEvent(state, frame("tool.result", {
      tool_call_id: "tc-json",
      tool_name: "search",
      success: true,
      output: { results: [{ title: "hello" }] }
    }));

    expect(state.nodes[0].tool?.resultPreview).toContain('"results"');
    expect(state.nodes[0].tool?.resultPreview).toContain('"title": "hello"');
  });

  it("updates status, metadata, retry, and error nodes", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("core.status", {
      scheduler_state: "RUNNING",
      agent_state: "RUNNING",
      queue_lengths: { urgent: 1, high_prio: 2, normal: 3 },
      queue_messages: {
        urgent: [{ id: "u1", queue: "urgent", content_preview: "stop now", created_at: 100 }],
        high_prio: [{ id: "h1", queue: "high_prio", content_preview: "merge this", created_at: 101 }],
        normal: [{ id: "n1", queue: "normal", content_preview: "first", created_at: 102 }]
      }
    }));
    state = reduceCoreEvent(state, frame("model.metadata", {
      input_tokens: 2,
      output_tokens: 3,
      total_tokens: 7,
      cache_read_tokens: 1,
      reasoning_tokens: 2,
      context_tokens: 128,
      max_context_tokens: 1024,
      context_ratio: 0.125,
      context_source: "provider_usage",
      latency_ms: 44
    }));
    state = reduceCoreEvent(state, frame("model.retry", { attempt: 1, max_retries: 3, error_type: "network", error_message: "retrying" }));
    state = reduceCoreEvent(state, frame("error", { message: "boom" }));

    expect(state.schedulerState).toBe("RUNNING");
    expect(state.queueLengths).toEqual({ urgent: 1, high_prio: 2, normal: 3 });
    expect(state.queueMessages.urgent[0].contentPreview).toBe("stop now");
    expect(state.queueMessages.high_prio[0].contentPreview).toBe("merge this");
    expect(state.queueMessages.normal[0].contentPreview).toBe("first");
    expect(state.metadataLines[0]).toContain("total=7");
    expect(state.metadataLines[0]).toContain("cache_read=1");
    expect(state.metadataLines[0]).toContain("reasoning=2");
    expect(state.metadataLines[0]).toContain("ctx=128/1024 (13%)");
    expect(state.metadataLines[0]).not.toContain("estimated");
    expect(state.contextUsage).toEqual({ usedTokens: 128, maxContextTokens: 1024, ratio: 0.125, source: "provider_usage" });
    expect(state.nodes.map((node) => node.kind)).toContain("error");
  });

  it("marks estimated context when provider usage is missing", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("model.metadata", {
      context_tokens: 64,
      max_context_tokens: 1000,
      context_ratio: 0.064,
      context_source: "estimate",
      latency_ms: 10
    }));

    expect(state.metadataLines[0]).toContain("provider_usage=missing");
    expect(state.metadataLines[0]).toContain("ctx≈64/1000 (6%) estimated");
    expect(state.contextUsage).toEqual({ usedTokens: 64, maxContextTokens: 1000, ratio: 0.064, source: "estimate" });
  });

  it("keeps provider context usage over periodic estimated status", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("model.metadata", {
      context_tokens: 900,
      max_context_tokens: 1000,
      context_ratio: 0.9,
      context_source: "provider_usage"
    }));
    state = reduceCoreEvent(state, frame("core.status", {
      scheduler_state: "IDLE",
      agent_state: "IDLE",
      queue_lengths: { urgent: 0, high_prio: 0, normal: 0 },
      context_usage: {
        used_tokens: 100,
        max_context_tokens: 1000,
        usage_ratio: 0.1,
        source: "estimate"
      }
    }));

    expect(state.contextUsage).toEqual({ usedTokens: 900, maxContextTokens: 1000, ratio: 0.9, source: "provider_usage" });
  });

  it("adds debug output as low-emphasis chat notes", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("debug.info", { message: "stderr line" }));

    expect(state.debugLines).toEqual(["stderr line"]);
    expect(state.nodes.map((node) => node.kind)).toEqual(["debug"]);
    expect(state.nodes[0].content).toBe("stderr line");
  });

  it("tracks plugin artifacts and streamed artifact deltas", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("plugin.artifact.upsert", {
      plugin_id: "plan",
      plugin_name: "PlanPlugin",
      artifact: {
        id: "current",
        type: "plan",
        title: "Current Plan",
        content: "# Plan\n",
        language: "markdown"
      }
    }));
    state = reduceCoreEvent(state, frame("plugin.artifact.delta", {
      plugin_id: "plan",
      artifact_id: "current",
      delta: "- Step one\n"
    }));

    expect(state.artifactOrder).toEqual(["plan:current"]);
    expect(state.selectedArtifactId).toBe("plan:current");
    expect(state.artifacts["plan:current"].content).toContain("- Step one");
    expect(state.artifacts["plan:current"].artifactType).toBe("plan");
  });

  it("tracks plugin tool progress and attaches it to the matching tool", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("tool.call_start", { run_id: "run-progress", tool_call_id: "tc-progress", tool_name: "write" }));
    state = reduceCoreEvent(state, frame("plugin.tool_progress", {
      plugin_id: "filesystem",
      plugin_name: "FileSystemPlugin",
      tool_call_id: "tc-progress",
      progress: 25,
      message: "Writing"
    }));

    expect(state.toolProgress["tc-progress"].progress).toBe(0.25);
    expect(state.nodes[0].tool?.progress?.message).toBe("Writing");
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
