import { afterEach, describe, expect, it } from "vitest";
import { VERSION, type CoreFrame } from "../shared/protocol";
import { createInitialState, reduceCoreEvent } from "./state";

function frame(type: string, payload: Record<string, unknown>, ts?: number): CoreFrame {
  return ts === undefined
    ? { version: VERSION, type, payload }
    : { version: VERSION, type, payload, ts };
}

function enableReleaseDedupFallbackForTest() {
  (globalThis as typeof globalThis & {
    __HAWI_TEST_RELEASE_DEDUP_FALLBACK__?: boolean;
  }).__HAWI_TEST_RELEASE_DEDUP_FALLBACK__ = true;
}

describe("core event reducer", () => {
  afterEach(() => {
    delete (globalThis as typeof globalThis & {
      __HAWI_TEST_RELEASE_DEDUP_FALLBACK__?: boolean;
    }).__HAWI_TEST_RELEASE_DEDUP_FALLBACK__;
  });

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
    expect(state.processing).toMatchObject({
      content: "处理中...",
      runId: "run-1"
    });
    expect(state.activeRunId).toBe("run-1");
    expect(state.sessionMessageCount).toBe(1);
  });

  it("keeps repeated live user message events visible outside release fallback", () => {
    let state = createInitialState();
    const payload = {
      run_id: "run-dupe",
      message_id: "msg-dupe",
      user_content: "hello",
      queue: "normal"
    };

    state = reduceCoreEvent(state, frame("run.start", payload, 10));
    state = reduceCoreEvent(state, frame("run.start", payload, 11));

    expect(state.nodes.filter((node) => node.kind === "user")).toHaveLength(2);
    expect(state.sessionMessageCount).toBe(2);
  });

  it("dedupes repeated live user message events in release fallback", () => {
    enableReleaseDedupFallbackForTest();
    let state = createInitialState();
    const payload = {
      run_id: "run-dupe",
      message_id: "msg-dupe",
      user_content: "hello",
      queue: "normal"
    };

    state = reduceCoreEvent(state, frame("run.start", payload, 10));
    state = reduceCoreEvent(state, frame("run.start", payload, 11));

    expect(state.nodes.filter((node) => node.kind === "user")).toHaveLength(1);
    expect(state.nodes[0]).toMatchObject({
      id: "user-message-msg-dupe",
      content: "hello",
      contextMessageIndex: 0
    });
    expect(state.sessionMessageCount).toBe(1);
    expect(state.nextContextMessageIndex).toBe(1);
    expect(state.processing).toMatchObject({ runId: "run-dupe" });
  });

  it("dedupes a live run.start that overlaps restored history in release fallback", () => {
    enableReleaseDedupFallbackForTest();
    let state = reduceCoreEvent(createInitialState(), frame("gui.load_session_history", {
      message_history: [
        {
          run_id: "run-history-dupe",
          role: "user",
          content: [{ type: "text", text: "hello from history" }],
          metadata: { message_id: "msg-history-dupe", queue: "normal" },
          context_message_id: "ctxmsg-history-dupe"
        }
      ]
    }));

    state = reduceCoreEvent(state, frame("run.start", {
      run_id: "run-history-dupe",
      message_id: "msg-history-dupe",
      user_content: "hello from history",
      queue: "normal",
      context_message_id: "ctxmsg-history-dupe"
    }));

    expect(state.nodes.filter((node) => node.kind === "user")).toHaveLength(1);
    expect(state.nodes[0]).toMatchObject({
      id: "user-message-msg-history-dupe",
      content: "hello from history",
      contextMessageId: "ctxmsg-history-dupe",
      contextMessageIndex: 0
    });
    expect(state.sessionMessageCount).toBe(1);
    expect(state.nextContextMessageIndex).toBe(1);
  });

  it("preserves structured content parts on live user messages", () => {
    const imagePart = {
      type: "image",
      source: {
        kind: "blob",
        blob_id: "a".repeat(64),
        uri: `hawi-blob://${"a".repeat(64)}`,
        mime_type: "image/png",
        filename: "screen.png"
      }
    };

    const state = reduceCoreEvent(createInitialState(), frame("run.start", {
      run_id: "run-media",
      message_id: "msg-media",
      user_content: "[image: screen.png]",
      content: [imagePart],
      queue: "high_prio"
    }));

    expect(state.nodes[0]).toMatchObject({
      id: "user-message-msg-media",
      kind: "user",
      content: "[image: screen.png]",
      contentParts: [imagePart]
    });
  });

  it("renders committed assistant image content even without text deltas", () => {
    const imagePart = {
      type: "image",
      source: {
        kind: "blob",
        blob_id: "b".repeat(64),
        uri: `hawi-blob://${"b".repeat(64)}`,
        mime_type: "image/png",
        filename: "result.png"
      }
    };
    let state = createInitialState();

    state = reduceCoreEvent(state, frame("run.start", {
      run_id: "run-image-answer",
      message_id: "msg-image-answer",
      user_content: "draw this",
      queue: "normal"
    }));
    state = reduceCoreEvent(state, frame("run.message_committed", {
      run_id: "run-image-answer",
      role: "assistant",
      context_message_id: "ctxmsg-image-answer",
      content: [imagePart]
    }));

    expect(state.nodes.map((node) => node.kind)).toEqual(["user", "agent"]);
    expect(state.nodes[1]).toMatchObject({
      kind: "agent",
      content: "",
      contentParts: [imagePart],
      contextMessageId: "ctxmsg-image-answer",
      contextMessageIndex: 1,
      complete: true
    });
    expect(state.processing).toBeUndefined();
    expect(state.nextContextMessageIndex).toBe(2);
  });

  it("renders committed assistant text even when there were no text deltas", () => {
    let state = createInitialState();

    state = reduceCoreEvent(state, frame("run.start", {
      run_id: "run-committed-text",
      message_id: "msg-committed-text",
      user_content: "answer directly",
      queue: "normal"
    }, 10));
    state = reduceCoreEvent(state, frame("run.message_committed", {
      run_id: "run-committed-text",
      role: "assistant",
      context_message_id: "ctxmsg-committed-text",
      content: [{ type: "text", text: "direct answer" }]
    }, 10));

    expect(state.nodes.map((node) => node.kind)).toEqual(["user", "agent"]);
    expect(state.nodes[1]).toMatchObject({
      kind: "agent",
      content: "direct answer",
      contextMessageId: "ctxmsg-committed-text",
      streamDurationMs: 0
    });
    expect(state.sessionMessageCount).toBe(2);
    expect(state.nextContextMessageIndex).toBe(2);
  });

  it("enables fork controls for live user and assistant messages", () => {
    let state = createInitialState();

    state = reduceCoreEvent(state, frame("run.start", {
      run_id: "run-live",
      message_id: "msg-live",
      user_content: "hello",
      context_message_id: "ctxmsg-user-live",
      queue: "normal"
    }));
    state = reduceCoreEvent(state, frame("run.text_delta", {
      run_id: "run-live",
      delta: "answer"
    }));

    expect(state.nodes[0]).toMatchObject({
      kind: "user",
      canFork: true,
      contextMessageId: "ctxmsg-user-live",
      contextMessageIndex: 0
    });
    state = reduceCoreEvent(state, frame("run.message_committed", {
      run_id: "run-live",
      role: "assistant",
      context_message_id: "ctxmsg-assistant-live"
    }));

    expect(state.nodes[1]).toMatchObject({
      kind: "agent",
      canFork: true,
      contextMessageId: "ctxmsg-assistant-live",
      contextMessageIndex: 1
    });
    expect(state.nextContextMessageIndex).toBe(2);
  });

  it("hydrates a streamed assistant message committed after a tool call without duplicating it", () => {
    let state = createInitialState();

    state = reduceCoreEvent(state, frame("run.start", {
      run_id: "run-tool-commit",
      user_content: "lookup",
      queue: "normal"
    }, 10));
    state = reduceCoreEvent(state, frame("run.text_delta", {
      run_id: "run-tool-commit",
      delta: "I will read the file first."
    }, 11));
    state = reduceCoreEvent(state, frame("tool.call_start", {
      run_id: "run-tool-commit",
      tool_call_id: "tc-read",
      tool_name: "read_file"
    }, 12));
    state = reduceCoreEvent(state, frame("run.message_committed", {
      run_id: "run-tool-commit",
      role: "assistant",
      context_message_id: "ctxmsg-tool-assistant",
      content: [
        { type: "text", text: "I will read the file first." },
        {
          type: "tool_call",
          id: "tc-read",
          name: "read_file",
          arguments: { file_path: "notes.md" }
        }
      ]
    }, 12.5));

    expect(state.nodes.map((node) => node.kind)).toEqual(["user", "agent", "tool"]);
    expect(state.nodes[1]).toMatchObject({
      kind: "agent",
      content: "I will read the file first.",
      contextMessageId: "ctxmsg-tool-assistant",
      contextMessageIndex: 1,
      streamDurationMs: 1000
    });
    expect(state.nodes.filter((node) => node.kind === "agent")).toHaveLength(1);
    expect(state.sessionMessageCount).toBe(2);
    expect(state.nextContextMessageIndex).toBe(2);

    state = reduceCoreEvent(state, frame("tool.result", {
      run_id: "run-tool-commit",
      tool_call_id: "tc-read",
      tool_name: "read_file",
      success: true,
      output: "contents"
    }, 13));
    state = reduceCoreEvent(state, frame("run.text_delta", {
      run_id: "run-tool-commit",
      delta: "Now I can continue."
    }, 14));

    expect(state.nodes.map((node) => node.kind)).toEqual(["user", "agent", "tool", "agent"]);
    expect(state.nodes[2].tool).toMatchObject({ contextMessageIndex: 2 });
    expect(state.nodes[3]).toMatchObject({
      content: "Now I can continue.",
      contextMessageIndex: 3
    });
    expect(state.nextContextMessageIndex).toBe(4);
  });

  it("keeps live fork indices aligned across tool calls", () => {
    let state = createInitialState();

    state = reduceCoreEvent(state, frame("run.start", {
      run_id: "run-tool",
      user_content: "lookup",
      queue: "normal"
    }));
    state = reduceCoreEvent(state, frame("tool.call_start", {
      run_id: "run-tool",
      tool_call_id: "tc-tool",
      tool_name: "search"
    }));
    state = reduceCoreEvent(state, frame("tool.result", {
      run_id: "run-tool",
      tool_call_id: "tc-tool",
      tool_name: "search",
      success: true,
      output: "found"
    }));
    state = reduceCoreEvent(state, frame("run.text_delta", {
      run_id: "run-tool",
      delta: "done"
    }));

    expect(state.nodes.map((node) => node.kind)).toEqual(["user", "tool", "agent"]);
    expect(state.nodes[0]).toMatchObject({ canFork: true, contextMessageIndex: 0 });
    expect(state.nodes[1].tool).toMatchObject({ contextMessageIndex: 2 });
    expect(state.nodes[2]).toMatchObject({ canFork: true, contextMessageIndex: 3 });
    expect(state.nextContextMessageIndex).toBe(4);
  });

  it("keeps user fork indices aligned when context is injected before the prompt", () => {
    let state = createInitialState();

    state = reduceCoreEvent(state, frame("run.start", {
      run_id: "run-injected",
      user_content: "hello",
      queue: "normal"
    }));
    state = reduceCoreEvent(state, frame("agent.context_injected", {
      run_id: "run-injected",
      role: "user",
      text: "Injected before",
      hook_type: "before_conversation",
      merge_target: "user_message",
      merge_position: "before",
      target_message_index: 0,
      position: 0
    }));
    state = reduceCoreEvent(state, frame("run.text_delta", {
      run_id: "run-injected",
      delta: "answer"
    }));

    expect(state.nodes[0]).toMatchObject({
      kind: "user",
      canFork: true,
      contextMessageIndex: 1
    });
    expect(state.nodes[0].injections?.[0]).toMatchObject({
      content: "Injected before",
      contextPosition: 0
    });
    expect(state.nodes[1]).toMatchObject({
      kind: "agent",
      canFork: true,
      contextMessageIndex: 2
    });
    expect(state.nextContextMessageIndex).toBe(3);
  });

  it("remaps fork indices after context compaction", () => {
    let state = createInitialState();

    state = reduceCoreEvent(state, frame("run.start", {
      run_id: "run-1",
      user_content: "first",
      queue: "normal"
    }));
    state = reduceCoreEvent(state, frame("run.text_delta", {
      run_id: "run-1",
      delta: "reply"
    }));
    state = reduceCoreEvent(state, frame("run.start", {
      run_id: "run-2",
      user_content: "second",
      queue: "normal"
    }));
    state = reduceCoreEvent(state, frame("run.text_delta", {
      run_id: "run-2",
      delta: "later"
    }));

    state = reduceCoreEvent(state, frame("agent.compact_stop", {
      run_id: "manual",
      status: "success",
      replaced_message_count: 2,
      kept_message_count: 2,
      message_count_before: 4,
      message_count_after: 3
    }));

    expect(state.nodes[0]).toMatchObject({
      kind: "user",
      canFork: false,
      contextMessageIndex: undefined
    });
    expect(state.nodes[1]).toMatchObject({
      kind: "agent",
      canFork: false,
      contextMessageIndex: undefined
    });
    expect(state.nodes[2]).toMatchObject({
      kind: "user",
      canFork: true,
      contextMessageIndex: 1
    });
    expect(state.nodes[3]).toMatchObject({
      kind: "agent",
      canFork: true,
      contextMessageIndex: 2
    });
    expect(state.nextContextMessageIndex).toBe(3);
  });

  it("loads session message history into chat nodes", () => {
    const state = reduceCoreEvent(createInitialState(), frame("gui.load_session_history", {
      message_history: [
        {
          run_id: "run-1",
          role: "user",
          content: [{ type: "text", text: "hello" }],
          metadata: { queue: "normal", display_message_type: "normal" },
          context_message_id: "ctxmsg-history-user",
          context_message_index: 0
        },
        {
          run_id: "run-1",
          role: "assistant",
          content: [
            { type: "reasoning", reasoning: "thinking" },
            { type: "text", text: "answer" }
          ],
          metadata: null,
          context_message_id: "ctxmsg-history-assistant",
          context_message_index: 1
        },
        {
          run_id: "run-1-1",
          role: "system",
          content: [{ type: "text", text: "模型重试 1/10: [network] retrying" }],
          metadata: { display_message_type: "model_retry" }
        },
        {
          run_id: "run-1",
          role: "error",
          content: [{ type: "text", text: "Anthropic authentication failed" }],
          metadata: { display_message_type: "model.error" }
        },
        {
          run_id: "run-1",
          role: "event",
          content: [{ type: "text", text: "Compressing context..." }],
          metadata: { display_message_type: "context_compaction", event_type: "agent.compact_start" }
        },
        {
          run_id: "run-1",
          role: "event",
          content: [{ type: "text", text: "Context compacted" }],
          metadata: { display_message_type: "context_compaction", event_type: "agent.compact_stop" }
        }
      ],
      context_usage: {
        used_tokens: 42,
        max_context_tokens: 1000,
        usage_ratio: 0.042,
        source: "provider_usage"
      }
    }));

    expect(state.nodes.map((node) => node.kind)).toEqual(["user", "thinking", "agent", "system", "error", "compact"]);
    expect(state.nodes[0].content).toBe("hello");
    expect(state.nodes[0]).toMatchObject({
      canFork: true,
      contextMessageId: "ctxmsg-history-user",
      contextMessageIndex: 0,
      historyIndex: 0
    });
    expect(state.nodes[1]).toMatchObject({ content: "thinking", complete: true });
    expect(state.nodes[2]).toMatchObject({
      content: "answer",
      complete: true,
      canFork: true,
      contextMessageId: "ctxmsg-history-assistant",
      contextMessageIndex: 1,
      historyIndex: 1
    });
    expect(state.nodes[3].content).toContain("模型重试 1/10");
    expect(state.nodes[4].content).toBe("Anthropic authentication failed");
    expect(state.nodes[5]).toMatchObject({ content: "Context compacted", complete: true });
    expect(state.sessionMessageCount).toBe(6);
    expect(state.contextUsage).toEqual({
      usedTokens: 42,
      maxContextTokens: 1000,
      ratio: 0.042,
      source: "provider_usage"
    });
  });

  it("preserves structured media parts when replaying session history", () => {
    const userImage = {
      type: "image",
      source: {
        kind: "blob",
        blob_id: "c".repeat(64),
        uri: `hawi-blob://${"c".repeat(64)}`,
        mime_type: "image/png",
        filename: "input.png"
      }
    };
    const assistantImage = {
      type: "image",
      source: {
        kind: "blob",
        blob_id: "d".repeat(64),
        uri: `hawi-blob://${"d".repeat(64)}`,
        mime_type: "image/png",
        filename: "output.png"
      }
    };

    const state = reduceCoreEvent(createInitialState(), frame("gui.load_session_history", {
      message_history: [
        {
          run_id: "run-media-history",
          role: "user",
          content: [{ type: "text", text: "look" }, userImage],
          context_message_id: "ctxmsg-media-user",
          context_message_index: 0
        },
        {
          run_id: "run-media-history",
          role: "assistant",
          content: [assistantImage],
          context_message_id: "ctxmsg-media-assistant",
          context_message_index: 1
        }
      ]
    }));

    expect(state.nodes.map((node) => node.kind)).toEqual(["user", "agent"]);
    expect(state.nodes[0]).toMatchObject({
      content: "look",
      contentParts: [{ type: "text", text: "look" }, userImage]
    });
    expect(state.nodes[1]).toMatchObject({
      content: "",
      contentParts: [assistantImage],
      contextMessageId: "ctxmsg-media-assistant"
    });
  });

  it("replays persisted injection and plugin events from session history", () => {
    const state = reduceCoreEvent(createInitialState(), frame("gui.load_session_history", {
      message_history: [
        {
          run_id: "run-replay",
          role: "user",
          content: [{ type: "text", text: "hello" }],
          metadata: { message_id: "msg-replay", queue: "normal" },
          context_message_index: 0
        },
        {
          run_id: "run-replay",
          role: "event",
          content: [{ type: "text", text: "Injected context" }],
          timestamp: 50,
          metadata: {
            display_message_type: "core_event",
            event_type: "agent.context_injected",
            event_payload: {
              run_id: "run-replay",
              role: "user",
              text: "Injected context",
              hook_type: "before_conversation",
              merge_target: "user_message",
              merge_position: "before",
              target_message_id: "msg-replay",
              position: 0,
              plugin_id: "research",
              plugin_name: "ResearchPlugin",
              plugin_role: "plugin",
              injection_name: "inject_notes"
            }
          }
        },
        {
          run_id: "run-replay",
          role: "event",
          content: [{ type: "text", text: "system material" }],
          metadata: {
            display_message_type: "core_event",
            event_type: "agent.system_prompt",
            event_payload: {
              run_id: "run-replay",
              text: "system material",
              origin: "model_input",
              plugin_role: "framework",
              injection_name: "system_prompt"
            }
          }
        },
        {
          run_id: "run-replay",
          role: "event",
          content: [{ type: "text", text: "legacy injected parameters" }],
          metadata: {
            display_message_type: "core_event",
            event_type: "agent.tool_parameter_injected",
            event_payload: {
              run_id: "run-replay",
              tool_name: "read_file",
              tool_call_id: "tc-read",
              parameters: { tool_call_purpose: "Read notes" }
            }
          }
        },
        {
          run_id: "run-replay",
          role: "event",
          content: [{ type: "text", text: "Collected notes" }],
          metadata: {
            display_message_type: "core_event",
            event_type: "plugin.message",
            event_payload: {
              plugin_id: "planner",
              plugin_name: "PlannerPlugin",
              title: "Plan",
              message: "Collected notes",
              data: { count: 3 }
            }
          }
        },
        {
          run_id: "run-replay",
          role: "event",
          content: [{ type: "text", text: "artifact" }],
          metadata: {
            display_message_type: "core_event",
            event_type: "plugin.artifact.upsert",
            event_payload: {
              plugin_id: "planner",
              plugin_name: "PlannerPlugin",
              artifact: {
                id: "plan",
                type: "plan",
                title: "Plan",
                content: "# Plan\n",
                language: "markdown"
              }
            }
          }
        },
        {
          run_id: "run-replay",
          role: "event",
          content: [{ type: "text", text: "status" }],
          metadata: {
            display_message_type: "core_event",
            event_type: "plugin.status",
            event_payload: {
              plugin_id: "planner",
              plugin_name: "PlannerPlugin",
              status: "active",
              message: "Working"
            }
          }
        }
      ]
    }));

    expect(state.nodes.map((node) => node.kind)).toEqual(["user", "framework", "framework"]);
    expect(state.nodes[0].injections?.[0]).toMatchObject({
      kind: "context_injected",
      content: "Injected context",
      pluginName: "ResearchPlugin",
      targetMessageId: "msg-replay",
      mergePosition: "before",
      contextPosition: 0
    });
    expect(state.nodes[1].framework).toMatchObject({
      kind: "system_prompt",
      content: "system material"
    });
    expect(state.nodes[2].framework).toMatchObject({
      kind: "plugin_message",
      pluginName: "PlannerPlugin"
    });
    expect(state.pluginMessages[0]).toMatchObject({
      pluginId: "planner",
      message: "Collected notes"
    });
    expect(state.artifactOrder).toEqual(["planner:plan"]);
    expect(state.artifacts["planner:plan"].content).toBe("# Plan\n");
    expect(state.pluginStatuses.planner).toMatchObject({
      status: "active",
      message: "Working"
    });
  });

  it("replaces older taskflow review messages for the same step", () => {
    let state = createInitialState();

    state = reduceCoreEvent(state, frame("plugin.event", {
      plugin_id: "hawi/taskflow",
      plugin_name: "Taskflow",
      event_name: "taskflow.review.requested",
      message_id: "taskflow-review-r1",
      level: "warning",
      title: "Review required: Draft",
      message: "Step Draft is awaiting human review.",
      data: {
        kind: "human_review_request",
        plugin_id: "hawi/taskflow",
        review_id: "r1",
        step_id: "draft",
        approve_action: "approve_taskflow_review",
        reject_action: "reject_taskflow_review"
      },
      state: {
        steps: [{ id: "draft", status: "reviewing", children: [] }]
      }
    }));

    state = reduceCoreEvent(state, frame("plugin.event", {
      plugin_id: "hawi/taskflow",
      plugin_name: "Taskflow",
      event_name: "taskflow.review.requested",
      message_id: "taskflow-review-r2",
      level: "warning",
      title: "Review required: Draft",
      message: "Step Draft is awaiting human review.",
      data: {
        kind: "human_review_request",
        plugin_id: "hawi/taskflow",
        review_id: "r2",
        step_id: "draft",
        approve_action: "approve_taskflow_review",
        reject_action: "reject_taskflow_review"
      },
      state: {
        steps: [{ id: "draft", status: "reviewing", children: [] }]
      }
    }));

    expect(state.pluginMessages).toHaveLength(1);
    expect(state.pluginMessages[0].id).toBe("taskflow-review-r2");
    expect(state.pluginMessages[0].data).toMatchObject({ review_id: "r2" });
  });

  it("removes taskflow review messages once their step leaves review", () => {
    let state = createInitialState();

    state = reduceCoreEvent(state, frame("plugin.event", {
      plugin_id: "hawi/taskflow",
      plugin_name: "Taskflow",
      event_name: "taskflow.review.requested",
      message_id: "taskflow-review-r1",
      level: "warning",
      title: "Review required: Draft",
      message: "Step Draft is awaiting human review.",
      data: {
        kind: "human_review_request",
        plugin_id: "hawi/taskflow",
        review_id: "r1",
        step_id: "draft",
        approve_action: "approve_taskflow_review",
        reject_action: "reject_taskflow_review"
      },
      state: {
        steps: [{ id: "draft", status: "reviewing", children: [] }]
      }
    }));

    state = reduceCoreEvent(state, frame("plugin.event", {
      plugin_id: "hawi/taskflow",
      plugin_name: "Taskflow",
      event_name: "taskflow.step.updated",
      action: "completed",
      message: "completed: draft",
      state: {
        steps: [{ id: "draft", status: "completed", children: [] }]
      }
    }));

    expect(state.pluginMessages).toHaveLength(1);
    expect(state.pluginMessages[0].message).toBe("completed: draft");
    expect(state.pluginMessages[0].data).toBeUndefined();
  });

  it("clears stale context usage when loading a session without usage", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("model.metadata", {
      input_tokens: 4,
      output_tokens: 2,
      total_tokens: 6,
      context_tokens: 900,
      max_context_tokens: 1000,
      context_ratio: 0.9,
      context_source: "provider_usage"
    }));

    state = reduceCoreEvent(state, frame("gui.load_session_history", { message_history: [] }));

    expect(state.contextUsage).toBeUndefined();
    expect(state.modelUsage).toBeUndefined();
  });

  it("counts one assistant message across thinking and answer deltas", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("run.start", { run_id: "run-count", user_content: "hi", queue: "normal" }));
    state = reduceCoreEvent(state, frame("run.thinking_delta", { run_id: "run-count", delta: "thinking" }));
    state = reduceCoreEvent(state, frame("run.text_delta", { run_id: "run-count", delta: "answer" }));

    expect(state.nodes.map((node) => node.kind)).toEqual(["user", "thinking", "agent"]);
    expect(state.sessionMessageCount).toBe(2);
  });

  it("replaces the pending processing line with the first text delta", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("run.start", { run_id: "run-prefill", user_content: "hi", queue: "normal" }));
    const processingId = state.processing?.id;

    state = reduceCoreEvent(state, frame("run.text_delta", { run_id: "run-prefill", delta: "answer" }));

    expect(state.nodes.map((node) => node.kind)).toEqual(["user", "agent"]);
    expect(state.nodes.some((node) => node.id === processingId)).toBe(false);
    expect(state.nodes[1]).toMatchObject({
      content: "answer",
      complete: false
    });
    expect(state.runs["run-prefill"]).toMatchObject({
      agentNodeId: state.nodes[1].id,
      assistantMessageCounted: true
    });
    expect(state.runs["run-prefill"].processingId).toBeUndefined();
    expect(state.processing).toBeUndefined();
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
    expect(state.processing).toMatchObject({
      content: "处理中...",
      runId: "run-1"
    });
    expect(state.activeRunId).toBe("run-1");
    expect(state.runs["run-1"].processingId).toBe(state.processing?.id);
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
    expect(state.nodes[1].complete).toBe(true);
    expect(state.nodes[3].content).toBe("after");
    expect(state.nodes[3].complete).toBe(false);
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

  it("shows a processing line after the last tool result while waiting for model output", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("run.start", { run_id: "run-wait-tool", user_content: "hi", queue: "normal" }));
    state = reduceCoreEvent(state, frame("tool.call_start", { run_id: "run-wait-tool", tool_call_id: "tc-wait", tool_name: "calc" }));

    expect(state.processing).toBeUndefined();

    state = reduceCoreEvent(state, frame("tool.result", {
      run_id: "run-wait-tool",
      tool_call_id: "tc-wait",
      tool_name: "calc",
      success: true,
      output: "4"
    }));

    expect(state.processing).toMatchObject({
      runId: "run-wait-tool",
      content: "处理中..."
    });
    expect(state.runs["run-wait-tool"].processingId).toBe(state.processing?.id);

    state = reduceCoreEvent(state, frame("run.text_delta", { run_id: "run-wait-tool", delta: "done" }));

    expect(state.processing).toBeUndefined();
    expect(state.runs["run-wait-tool"].processingId).toBeUndefined();
    expect(state.nodes.map((node) => node.kind)).toEqual(["user", "tool", "agent"]);
  });

  it("waits for all known tools to finish before showing model processing", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("run.start", { run_id: "run-two-tools", user_content: "hi", queue: "normal" }));
    state = reduceCoreEvent(state, frame("tool.call_start", { run_id: "run-two-tools", tool_call_id: "tc-a", tool_name: "a" }));
    state = reduceCoreEvent(state, frame("tool.call_start", { run_id: "run-two-tools", tool_call_id: "tc-b", tool_name: "b" }));
    state = reduceCoreEvent(state, frame("tool.result", {
      run_id: "run-two-tools",
      tool_call_id: "tc-a",
      tool_name: "a",
      success: true,
      output: "a"
    }));

    expect(state.processing).toBeUndefined();

    state = reduceCoreEvent(state, frame("tool.result", {
      run_id: "run-two-tools",
      tool_call_id: "tc-b",
      tool_name: "b",
      success: true,
      output: "b"
    }));

    expect(state.processing).toMatchObject({
      runId: "run-two-tools",
      content: "处理中..."
    });
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

  it("records stream durations for thinking, answer, and tool call blocks", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("run.start", { run_id: "run-stream-times", user_content: "hi", queue: "normal" }, 10));
    state = reduceCoreEvent(state, frame("run.thinking_delta", { run_id: "run-stream-times", delta: "thinking" }, 11));
    state = reduceCoreEvent(state, frame("run.text_delta", { run_id: "run-stream-times", delta: "answer" }, 12.5));
    state = reduceCoreEvent(state, frame("tool.call_start", { run_id: "run-stream-times", tool_call_id: "tc-time", tool_name: "calc" }, 13));
    state = reduceCoreEvent(state, frame("tool.call_stop", { tool_call_id: "tc-time", tool_name: "calc", arguments: { expression: "1+1" } }, 13.75));

    expect(state.nodes.map((node) => node.kind)).toEqual(["user", "thinking", "agent", "tool"]);
    expect(state.nodes[1]).toMatchObject({
      complete: true,
      streamStartedAt: 11000,
      streamFinishedAt: 12500,
      streamDurationMs: 1500
    });
    expect(state.nodes[2]).toMatchObject({
      complete: true,
      streamStartedAt: 12500,
      streamFinishedAt: 13000,
      streamDurationMs: 500
    });
    expect(state.nodes[3].tool).toMatchObject({
      streamStartedAt: 13000,
      streamFinishedAt: 13750,
      streamDurationMs: 750
    });
  });

  it("marks thinking complete when run stops", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("run.start", { run_id: "run-stop-thinking", user_content: "hi", queue: "normal" }));
    state = reduceCoreEvent(state, frame("run.thinking_delta", { run_id: "run-stop-thinking", delta: "thinking" }));
    state = reduceCoreEvent(state, frame("run.stop", { run_id: "run-stop-thinking", stop_reason: "end_turn" }));

    expect(state.nodes.map((node) => node.kind)).toEqual(["user", "thinking", "divider"]);
    expect(state.nodes[1].complete).toBe(true);
  });

  it("finishes the active answer block when the run stops", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("run.start", { run_id: "run-stop-agent-time", user_content: "hi", queue: "normal" }, 20));
    state = reduceCoreEvent(state, frame("run.text_delta", { run_id: "run-stop-agent-time", delta: "answer" }, 21));
    state = reduceCoreEvent(state, frame("run.stop", { run_id: "run-stop-agent-time", stop_reason: "end_turn" }, 22.25));

    expect(state.nodes.map((node) => node.kind)).toEqual(["user", "agent", "divider"]);
    expect(state.nodes[1]).toMatchObject({
      complete: true,
      streamStartedAt: 21000,
      streamFinishedAt: 22250,
      streamDurationMs: 1250
    });
  });

  it("removes the pending processing line when a run stops before model output", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("run.start", { run_id: "run-empty", user_content: "hi", queue: "normal" }));
    state = reduceCoreEvent(state, frame("run.stop", { run_id: "run-empty", stop_reason: "error" }));

    expect(state.nodes.map((node) => node.kind)).toEqual(["user", "divider"]);
    expect(state.processing).toBeUndefined();
    expect(state.sessionMessageCount).toBe(1);
  });

  it("removes the pending processing line before any new chat content", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("run.start", { run_id: "run-error", user_content: "hi", queue: "normal" }));
    state = reduceCoreEvent(state, frame("error", { message: "boom" }));

    expect(state.nodes.map((node) => node.kind)).toEqual(["user", "error"]);
    expect(state.runs["run-error"].processingId).toBeUndefined();
    expect(state.processing).toBeUndefined();
  });

  it("keeps the pending processing line when hidden debug content arrives", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("run.start", { run_id: "run-debug", user_content: "hi", queue: "normal" }));
    state = reduceCoreEvent(state, frame("debug.info", { message: "model stream started" }));

    expect(state.nodes.map((node) => node.kind)).toEqual(["user", "debug"]);
    expect(state.processing).toMatchObject({
      content: "处理中...",
      runId: "run-debug"
    });
    expect(state.runs["run-debug"].processingId).toBe(state.processing?.id);
  });

  it("keeps the pending processing line when model retry is reported", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("run.start", { run_id: "run-retry", user_content: "hi", queue: "normal" }));
    state = reduceCoreEvent(state, frame("model.retry", {
      attempt: 1,
      max_retries: 10,
      error_type: "network",
      error_message: "Anthropic connection error: Connection error."
    }));

    expect(state.nodes.map((node) => node.kind)).toEqual(["user", "system"]);
    expect(state.nodes[1].content).toContain("模型重试 1/10");
    expect(state.processing).toMatchObject({
      content: "处理中...",
      runId: "run-retry"
    });
    expect(state.runs["run-retry"].processingId).toBe(state.processing?.id);
  });

  it("marks agent messages complete when run stops", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("run.start", { run_id: "run-stop-agent", user_content: "hi", queue: "normal" }));
    state = reduceCoreEvent(state, frame("run.text_delta", { run_id: "run-stop-agent", delta: "answer" }));
    state = reduceCoreEvent(state, frame("run.stop", { run_id: "run-stop-agent", stop_reason: "end_turn" }));

    expect(state.nodes.map((node) => node.kind)).toEqual(["user", "agent", "divider"]);
    expect(state.nodes[1].complete).toBe(true);
  });

  it("marks agent messages complete when model decoding is interrupted", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("run.start", { run_id: "run-model-interrupt", user_content: "hi", queue: "normal" }));
    state = reduceCoreEvent(state, frame("run.text_delta", { run_id: "run-model-interrupt", delta: "partial" }, 10));

    state = reduceCoreEvent(state, frame("model.interrupted", {
      run_id: "run-model-interrupt",
      request_id: "req-model-interrupt",
      reason: "user",
      stop_reason: "interrupted"
    }, 11));

    expect(state.nodes.map((node) => node.kind)).toEqual(["user", "agent"]);
    expect(state.nodes[1].complete).toBe(true);
    expect(state.nodes[1].streamDurationMs).toBe(1000);
    expect(state.processing).toBeUndefined();
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

  it("marks active tools interrupted without assigning history indexes", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("tool.call_start", {
      run_id: "run-interrupt",
      tool_call_id: "tc-interrupt",
      tool_name: "shell"
    }, 11));
    const beforeIndex = state.nextContextMessageIndex;

    state = reduceCoreEvent(state, frame("tool.interrupted", {
      run_id: "run-interrupt",
      tool_call_id: "tc-interrupt",
      tool_name: "shell",
      reason: "user"
    }, 12));

    const tool = state.nodes[0].tool;
    expect(tool?.status).toBe("fail");
    expect(tool?.argsState).toBe("complete");
    expect(tool?.resultPreview).toBe("Tool call interrupted before completion (reason: user).");
    expect(tool?.resultData).toBe("Tool call interrupted before completion (reason: user).");
    expect(tool?.contextMessageIndex).toBeUndefined();
    expect(tool?.streamDurationMs).toBe(1000);
    expect(state.nextContextMessageIndex).toBe(beforeIndex);

    state = reduceCoreEvent(state, frame("tool.result", {
      run_id: "run-interrupt",
      tool_call_id: "tc-interrupt",
      tool_name: "shell",
      success: false,
      output: "Tool call interrupted before completion (reason: user).",
      interrupted: true,
      context_message_id: "ctx-tool-interrupt"
    }, 13));

    expect(state.nodes[0].tool?.contextMessageId).toBe("ctx-tool-interrupt");
    expect(state.nodes[0].tool?.contextMessageIndex).toBeUndefined();
    expect(state.nextContextMessageIndex).toBe(beforeIndex);
  });

  it("shows workspace switch notices as metadata bubbles", () => {
    const state = reduceCoreEvent(createInitialState(), frame("gui.workspace_changed", {
      message: "已根据 Session 记录切换工作目录：/a -> /b",
      previous_cwd: "/a",
      last_cwd: "/b"
    }));

    expect(state.nodes[0]).toMatchObject({
      kind: "meta",
      content: "已根据 Session 记录切换工作目录：/a -> /b"
    });
  });

  it("preserves full long tool results for scrollable rendering", () => {
    const output = `${"line\n".repeat(300)}final line`;
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("tool.call_start", { run_id: "run-long-result", tool_call_id: "tc-long-result", tool_name: "shell" }));
    state = reduceCoreEvent(state, frame("tool.result", {
      tool_call_id: "tc-long-result",
      tool_name: "shell",
      success: true,
      output
    }));

    expect(state.nodes[0].tool?.resultPreview).toBe(output);
    expect(state.nodes[0].tool?.resultPreview).toContain("final line");
  });

  it("appends streaming tool result parts without completing the tool", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("tool.call_start", {
      run_id: "run-streaming-tool",
      tool_call_id: "tc-streaming-tool",
      tool_name: "run_shell"
    }, 100));
    state = reduceCoreEvent(state, frame("tool.result", {
      run_id: "run-streaming-tool",
      tool_call_id: "tc-streaming-tool",
      part: "first\n",
      is_part: true
    }, 110));

    expect(state.nodes[0].tool?.status).toBe("running");
    expect(state.nodes[0].tool?.resultPreview).toBe("first\n");
    expect(state.nodes[0].tool?.streamFinishedAt).toBeUndefined();

    state = reduceCoreEvent(state, frame("tool.result", {
      run_id: "run-streaming-tool",
      tool_call_id: "tc-streaming-tool",
      tool_name: "run_shell",
      success: true,
      output: "Exit code: 0\n\nStdout:\nfirst\nsecond\n",
      is_part: false
    }, 150));

    expect(state.nodes[0].tool?.status).toBe("success");
    expect(state.nodes[0].tool?.resultPreview).toContain("second");
    expect(state.nodes[0].tool?.streamFinishedAt).toBe(150000);
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

  it("keeps structured filesystem results while using readable previews", () => {
    const readOutput = {
      type: "text",
      file: {
        filePath: "/tmp/example.py",
        content: "   1|print('hi')\n",
        numLines: 1,
        startLine: 0,
        totalLines: 1,
        language: "python"
      }
    };
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("tool.call_start", {
      run_id: "run-fs",
      tool_call_id: "tc-read",
      tool_name: "read_file"
    }));
    state = reduceCoreEvent(state, frame("tool.result", {
      tool_call_id: "tc-read",
      tool_name: "read_file",
      success: true,
      output: readOutput
    }));

    expect(state.nodes[0].tool?.resultData).toEqual(readOutput);
    expect(state.nodes[0].tool?.resultPreview).toBe("   1|print('hi')");

    state = reduceCoreEvent(state, frame("tool.call_start", {
      run_id: "run-fs",
      tool_call_id: "tc-list",
      tool_name: "list_dir"
    }));
    state = reduceCoreEvent(state, frame("tool.result", {
      tool_call_id: "tc-list",
      tool_name: "list_dir",
      success: true,
      output: {
        type: "ls_output",
        text: "total 0\n-rw-r--r--  1 hayden  staff  0 May 14 12:00 a.py\n",
        numEntries: 1,
        isTruncated: false
      }
    }));

    expect(state.nodes[1].tool?.resultPreview).toBe("total 0\n-rw-r--r--  1 hayden  staff  0 May 14 12:00 a.py");

    const directoryOutput = {
      type: "directory",
      path: "/tmp",
      entries: [{ name: "a.py", relativePath: "a.py", type: "file" }],
      text: "total 0\n-rw-r--r--  1 hayden  staff  0 May 14 12:00 a.py\n",
      numEntries: 1,
      isTruncated: false
    };
    state = reduceCoreEvent(state, frame("tool.call_start", {
      run_id: "run-fs",
      tool_call_id: "tc-list-directory",
      tool_name: "list_dir"
    }));
    state = reduceCoreEvent(state, frame("tool.result", {
      tool_call_id: "tc-list-directory",
      tool_name: "list_dir",
      success: true,
      output: directoryOutput
    }));

    expect(state.nodes[2].tool?.resultData).toEqual(directoryOutput);
    expect(state.nodes[2].tool?.resultPreview).toBe("total 0\n-rw-r--r--  1 hayden  staff  0 May 14 12:00 a.py");
  });

  it("loads JSON string tool arguments from session history for specialized renderers", () => {
    const state = reduceCoreEvent(createInitialState(), frame("gui.load_session_history", {
      message_history: [
        {
          run_id: "run-history-tool",
          role: "assistant",
          content: [
            {
              type: "tool_call",
              id: "tc-edit-history",
              name: "edit_file",
              arguments: "{\"file_path\":\"src/app.ts\",\"old_string\":\"old\",\"new_string\":\"new\"}"
            }
          ]
        }
      ]
    }));

    expect(state.nodes[0].tool?.arguments).toEqual({
      file_path: "src/app.ts",
      old_string: "old",
      new_string: "new"
    });
  });

  it("updates status, metadata, retry, and error nodes", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("core.status", {
      runner_state: "RUNNING",
      agent_state: "RUNNING",
      queue_lengths: { urgent: 1, high_prio: 2, normal: 3 },
      queue_messages: {
        urgent: [{ id: "u1", queue: "urgent", content_preview: "stop now", created_at: 100 }],
        high_prio: [{ id: "h1", queue: "high_prio", content_preview: "merge this", created_at: 101 }],
        normal: [{
          id: "n1",
          queue: "normal",
          content_preview: "first",
          content: "first full task",
          content_parts: [{ type: "text", text: "first full task" }],
          created_at: 102
        }]
      },
      auto_compact: {
        enabled: true,
        max_context_tokens: 1000,
        trigger_tokens: 720,
        trigger_ratio: 0.72,
        max_trigger_ratio: 0.95,
        compression_budget: 200,
        token_limit: 720,
        token_limit_ratio: 0.72
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
      ttft_ms: 120,
      prefill_tokens_per_second: 1066.7,
      decode_tokens_per_second: 42.5,
      latency_ms: 44
    }));
    state = reduceCoreEvent(state, frame("model.retry", { attempt: 1, max_retries: 3, error_type: "network", error_message: "retrying" }));
    state = reduceCoreEvent(state, frame("error", { message: "boom" }));

    expect(state.runnerState).toBe("RUNNING");
    expect(state.queueLengths).toEqual({ urgent: 1, high_prio: 2, normal: 3 });
    expect(state.queueMessages.urgent[0].contentPreview).toBe("stop now");
    expect(state.queueMessages.high_prio[0].contentPreview).toBe("merge this");
    expect(state.queueMessages.normal[0].contentPreview).toBe("first");
    expect(state.queueMessages.normal[0].content).toBe("first full task");
    expect(state.queueMessages.normal[0].contentParts).toEqual([
      { type: "text", text: "first full task" }
    ]);
    expect(state.metadataLines[0]).toContain("cache_read=1");
    expect(state.metadataLines[0]).toContain("reasoning=2");
    expect(state.metadataLines[0]).toContain("ctx=128/1024 (13%)");
    expect(state.metadataLines[0]).toContain("ttft=120ms");
    expect(state.metadataLines[0]).toContain("prefill≈1067 tok/s");
    expect(state.metadataLines[0]).toContain("decode≈43 tok/s");
    expect(state.metadataLines[0]).not.toContain("estimated");
    expect(state.modelUsage).toEqual({
      totalTokens: 7,
      inputTokens: 2,
      outputTokens: 3,
      cacheReadTokens: 1,
      cacheWriteTokens: 0
    });
    expect(state.contextUsage).toEqual({ usedTokens: 128, maxContextTokens: 1024, ratio: 0.125, source: "provider_usage" });
    expect(state.contextAutoCompact).toEqual({
      enabled: true,
      maxContextTokens: 1000,
      triggerTokens: 720,
      triggerRatio: 0.72,
      maxTriggerRatio: 0.95,
      compressionBudget: 200,
      tokenLimit: 720,
      tokenLimitRatio: 0.72
    });
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
    expect(state.modelUsage).toBeUndefined();
    expect(state.contextUsage).toEqual({ usedTokens: 64, maxContextTokens: 1000, ratio: 0.064, source: "estimate" });
  });

  it("aggregates model usage and reports input tokens excluding cached tokens", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("model.metadata", {
      input_tokens: 12,
      output_tokens: 3,
      total_tokens: 15,
      cache_read_tokens: 4,
      cache_write_tokens: 1
    }));
    state = reduceCoreEvent(state, frame("model.metadata", {
      input_tokens: 6,
      output_tokens: 2,
      total_tokens: 10,
      cache_read_tokens: 1,
      cache_write_tokens: 1
    }));

    expect(state.modelUsage).toEqual({
      totalTokens: 25,
      inputTokens: 13,
      outputTokens: 5,
      cacheReadTokens: 5,
      cacheWriteTokens: 2
    });
  });

  it("attaches model profile metadata to the current user and assistant bubbles", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("run.start", { run_id: "run-profile", user_content: "hi", queue: "normal" }));
    state = reduceCoreEvent(state, frame("run.text_delta", { run_id: "run-profile", delta: "hello" }));
    state = reduceCoreEvent(state, frame("model.metadata", {
      run_id: "run-profile",
      input_tokens: 20,
      output_tokens: 5,
      cache_read_tokens: 8,
      prefill_tokens: 12,
      prefill_total_tokens: 48,
      prefill_ms: 246,
      prefill_tokens_per_second: 48.8,
      ttft_ms: 698,
      decode_tokens: 5,
      decode_ms: 123,
      decode_tokens_per_second: 40.7,
      peak_decode_tokens_per_second: 52.1
    }));

    const user = state.nodes.find((node) => node.kind === "user");
    const agent = state.nodes.find((node) => node.kind === "agent");

    expect(user?.profile).toEqual({
      cacheTokens: 8,
      prefillTokens: 12,
      prefillTotalTokens: 48,
      prefillMs: 246,
      prefillTokensPerSecond: 48.8
    });
    expect(agent?.profile).toEqual({
      ttftMs: 698,
      decodeTokens: 5,
      decodeMs: 123,
      decodeTokensPerSecond: 40.7,
      peakDecodeTokensPerSecond: 52.1
    });
  });

  it("attaches streaming model profile before the assistant bubble exists", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("run.start", { run_id: "run-live-profile", user_content: "hi", queue: "normal" }));
    state = reduceCoreEvent(state, frame("model.profile", {
      run_id: "run-live-profile",
      cache_tokens: 8,
      prefill_tokens: 12,
      prefill_total_tokens: 48,
      prefill_ms: 246,
      prefill_tokens_per_second: 48.8,
      ttft_ms: 698,
      decode_tokens: 5,
      decode_ms: 123,
      decode_tokens_per_second: 40.7,
      peak_decode_tokens_per_second: 52.1
    }));

    const userBeforeText = state.nodes.find((node) => node.kind === "user");
    expect(userBeforeText?.profile).toEqual({
      cacheTokens: 8,
      prefillTokens: 12,
      prefillTotalTokens: 48,
      prefillMs: 246,
      prefillTokensPerSecond: 48.8
    });
    expect(state.nodes.find((node) => node.kind === "agent")).toBeUndefined();

    state = reduceCoreEvent(state, frame("run.text_delta", { run_id: "run-live-profile", delta: "hello" }));

    const agent = state.nodes.find((node) => node.kind === "agent");
    expect(agent?.profile).toEqual({
      ttftMs: 698,
      decodeTokens: 5,
      decodeMs: 123,
      decodeTokensPerSecond: 40.7,
      peakDecodeTokensPerSecond: 52.1
    });
  });

  it("replays persisted model metadata into model usage state", () => {
    const state = reduceCoreEvent(createInitialState(), frame("gui.load_session_history", {
      message_history: [
        {
          run_id: "req-usage",
          role: "event",
          content: [{ type: "text", text: "event" }],
          metadata: {
            display_message_type: "core_event",
            event_type: "model.metadata",
            event_payload: {
              request_id: "req-usage",
              input_tokens: 9,
              output_tokens: 2,
              total_tokens: 11,
              cache_read_tokens: 3,
              cache_write_tokens: 0,
              context_tokens: 11,
              max_context_tokens: 100,
              context_ratio: 0.11,
              context_source: "provider_usage"
            }
          }
        }
      ]
    }));

    expect(state.nodes).toHaveLength(0);
    expect(state.modelUsage).toEqual({
      totalTokens: 11,
      inputTokens: 6,
      outputTokens: 2,
      cacheReadTokens: 3,
      cacheWriteTokens: 0
    });
    expect(state.contextUsage).toEqual({
      usedTokens: 11,
      maxContextTokens: 100,
      ratio: 0.11,
      source: "provider_usage"
    });
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
      runner_state: "IDLE",
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

  it("keeps provider label while accepting higher context growth estimates", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("model.metadata", {
      context_tokens: 900,
      max_context_tokens: 1000,
      context_ratio: 0.9,
      context_source: "provider_usage"
    }));
    state = reduceCoreEvent(state, frame("core.status", {
      runner_state: "RUNNING",
      agent_state: "RUNNING",
      queue_lengths: { urgent: 0, high_prio: 0, normal: 0 },
      context_usage: {
        used_tokens: 950,
        max_context_tokens: 1000,
        usage_ratio: 0.95,
        source: "estimate"
      }
    }));

    expect(state.contextUsage).toEqual({ usedTokens: 950, maxContextTokens: 1000, ratio: 0.95, source: "provider_usage" });
  });

  it("tracks active context compression for the status strip", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("agent.compact_start", {
      run_id: "run-compact",
      mode: "auto",
      tokens_before: 1000,
      message_count_before: 20
    }, 100));

    expect(state.contextCompression).toMatchObject({
      active: true,
      nodeId: state.nodes[0].id,
      mode: "auto",
      tokensBefore: 1000,
      messageCountBefore: 20,
      startedAt: 100000,
      updatedAt: 100000
    });
    expect(state.nodes).toHaveLength(1);
    expect(state.nodes[0]).toMatchObject({
      kind: "compact",
      content: "Compressing context...",
      complete: false,
      streamStartedAt: 100000
    });
    expect(state.sessionMessageCount).toBe(1);

    state = reduceCoreEvent(state, frame("agent.compact_stop", {
      run_id: "run-compact",
      mode: "auto",
      status: "success",
      tokens_before: 1000,
      tokens_after: 250,
      message_count_before: 20,
      message_count_after: 5
    }, 110));

    expect(state.contextCompression).toMatchObject({
      active: false,
      mode: "auto",
      status: "success",
      tokensBefore: 1000,
      tokensAfter: 250,
      messageCountBefore: 20,
      messageCountAfter: 5,
      startedAt: 100000,
      updatedAt: 110000
    });
    expect(state.nodes).toHaveLength(1);
    expect(state.nodes[0]).toMatchObject({
      kind: "compact",
      content: "Context compacted",
      complete: true,
      streamFinishedAt: 110000,
      streamDurationMs: 10000
    });
    expect(state.sessionMessageCount).toBe(2);
  });

  it("clears active context compression when a run stops", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("run.start", { run_id: "run-compact-stop", user_content: "hi", queue: "normal" }, 20));
    state = reduceCoreEvent(state, frame("agent.compact_start", {
      run_id: "run-compact-stop",
      mode: "auto",
      tokens_before: 1000,
      message_count_before: 20
    }, 21));
    state = reduceCoreEvent(state, frame("run.stop", {
      run_id: "run-compact-stop",
      stop_reason: "interrupted"
    }, 22));

    expect(state.contextCompression).toMatchObject({
      active: false,
      status: "interrupted",
      updatedAt: 22000
    });
    expect(state.nodes.some((node) => (
      node.kind === "compact"
      && node.content === "Context compaction interrupted"
      && node.complete === true
    ))).toBe(true);
  });

  it("adds debug output as low-emphasis chat notes", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("debug.info", { message: "stderr line" }));

    expect(state.debugLines).toEqual(["stderr line"]);
    expect(state.nodes.map((node) => node.kind)).toEqual(["debug"]);
    expect(state.nodes[0].content).toBe("stderr line");
  });

  it("adds system prompt injections as collapsed framework chat nodes", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("agent.system_prompt", {
      run_id: "run-system",
      text: "You are Hawi.",
      origin: "model_input",
      plugin_role: "framework",
      injection_name: "system_prompt",
      metadata: { content_scope: "full_prompt" }
    }, 30));

    expect(state.nodes.map((node) => node.kind)).toEqual(["framework"]);
    expect(state.nodes[0].framework).toMatchObject({
      kind: "system_prompt",
      label: "System prompt",
      content: "You are Hawi.",
      runId: "run-system",
      pluginRole: "framework",
      injectionName: "system_prompt",
      timestamp: 30000
    });
  });

  it("renders empty system prompts as blank content instead of JSON", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("agent.system_prompt", {
      run_id: "run-empty-system",
      content: [
        { type: "text", text: "" },
        { type: "cache_point", cache_point: { type: "ephemeral" } }
      ],
      text: "",
      origin: "model_input",
      plugin_role: "framework",
      injection_name: "system_prompt",
      metadata: { content_scope: "full_prompt" }
    }, 30));

    expect(state.nodes.map((node) => node.kind)).toEqual(["framework"]);
    expect(state.nodes[0].framework).toMatchObject({
      kind: "system_prompt",
      content: ""
    });
  });

  it("groups system prompt injected segments as child framework bubbles", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("agent.system_prompt", {
      run_id: "run-system",
      text: "Plugin system material",
      origin: "before_session",
      plugin_id: "research",
      plugin_name: "ResearchPlugin",
      plugin_role: "plugin",
      injection_name: "inject_prompt",
      metadata: { content_scope: "injected_segment", change_type: "append" }
    }, 20));
    state = reduceCoreEvent(state, frame("agent.system_prompt", {
      run_id: "run-system",
      text: "Base system",
      origin: "session_start",
      plugin_role: "framework",
      injection_name: "system_prompt",
      metadata: { content_scope: "full_prompt" }
    }, 30));

    expect(state.nodes).toHaveLength(1);
    expect(state.nodes[0].framework).toMatchObject({
      kind: "system_prompt",
      label: "System prompt",
      content: "Base system",
      runId: "run-system"
    });
    expect(state.nodes[0].injections).toHaveLength(1);
    expect(state.nodes[0].injections?.[0]).toMatchObject({
      kind: "system_prompt",
      label: "System prompt 注入信息",
      content: "Plugin system material",
      pluginId: "research",
      pluginName: "ResearchPlugin",
      injectionName: "inject_prompt"
    });
  });

  it("dedupes repeated system prompt injected segments in release fallback", () => {
    enableReleaseDedupFallbackForTest();
    let state = createInitialState();
    const childPayload = {
      run_id: "run-system-dupe",
      text: "Plugin system material",
      origin: "before_session",
      plugin_id: "research",
      plugin_name: "ResearchPlugin",
      plugin_role: "plugin",
      injection_name: "inject_prompt",
      metadata: { content_scope: "injected_segment", change_type: "append" }
    };

    state = reduceCoreEvent(state, frame("agent.system_prompt", childPayload, 20));
    state = reduceCoreEvent(state, frame("agent.system_prompt", childPayload, 21));
    state = reduceCoreEvent(state, frame("agent.system_prompt", {
      run_id: "run-system-dupe",
      text: "Base system",
      origin: "session_start",
      plugin_role: "framework",
      injection_name: "system_prompt",
      metadata: { content_scope: "full_prompt" }
    }, 30));

    expect(state.nodes).toHaveLength(1);
    expect(state.nodes[0].injections).toHaveLength(1);
    expect(state.nodes[0].injections?.[0]).toMatchObject({
      kind: "system_prompt",
      content: "Plugin system material",
      pluginId: "research",
      injectionName: "inject_prompt"
    });
  });

  it("attaches user-targeted context injections to the matching user message", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("run.start", {
      run_id: "run-context",
      message_id: "msg-context",
      user_content: "hello",
      queue: "normal"
    }));
    state = reduceCoreEvent(state, frame("agent.context_injected", {
      run_id: "run-context",
      role: "user",
      text: "Injected after",
      hook_type: "before_conversation",
      merge_target: "user_message",
      merge_position: "after",
      target_message_id: "msg-context",
      position: 1,
      plugin_id: "research",
      plugin_name: "ResearchPlugin",
      plugin_role: "plugin",
      injection_name: "inject_notes"
    }));
    state = reduceCoreEvent(state, frame("agent.context_injected", {
      run_id: "run-context",
      role: "user",
      text: "Injected before",
      hook_type: "before_conversation",
      merge_target: "user_message",
      merge_position: "before",
      target_message_id: "msg-context",
      position: 0,
      plugin_id: "research",
      plugin_name: "ResearchPlugin",
      plugin_role: "plugin",
      injection_name: "inject_earlier_notes"
    }));

    expect(state.nodes).toHaveLength(1);
    expect(state.nodes[0]).toMatchObject({ kind: "user", content: "hello" });
    expect(state.nodes[0].injections).toHaveLength(2);
    expect(state.nodes[0].injections?.[0]).toMatchObject({
      kind: "context_injected",
      content: "Injected before",
      pluginId: "research",
      pluginName: "ResearchPlugin",
      mergeTarget: "user_message",
      mergePosition: "before",
      targetMessageId: "msg-context",
      contextPosition: 0
    });
    expect(state.nodes[0].injections?.[1]).toMatchObject({
      content: "Injected after",
      mergePosition: "after",
      contextPosition: 1
    });
    expect(state.processing).toMatchObject({ runId: "run-context" });
  });

  it("shows runtime context injections as framework bubbles", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("agent.tool_runtime_context_injected", {
      run_id: "run-tool",
      tool_name: "inspect",
      tool_call_id: "tc-inspect",
      parameter_name: "context",
      plugin_id: "inspector",
      plugin_name: "InspectorPlugin",
      plugin_role: "tool_owner",
      injection_name: "runtime_context"
    }));

    expect(state.nodes.map((node) => node.kind)).toEqual(["framework"]);
    expect(state.nodes[0].framework).toMatchObject({
      kind: "tool_runtime_context_injected",
      toolName: "inspect",
      parameterName: "context",
      pluginName: "InspectorPlugin"
    });
  });

  it("keeps plugin messages in the plugin log and mirrors them as framework bubbles", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("plugin.message", {
      plugin_id: "planner",
      plugin_name: "PlannerPlugin",
      title: "Plan",
      message: "Collected research notes",
      data: { count: 3 }
    }, 40));

    expect(state.pluginMessages).toHaveLength(1);
    expect(state.pluginMessages[0]).toMatchObject({
      pluginId: "planner",
      pluginName: "PlannerPlugin",
      title: "Plan",
      message: "Collected research notes"
    });
    expect(state.nodes.map((node) => node.kind)).toEqual(["framework"]);
    expect(state.nodes[0].framework).toMatchObject({
      kind: "plugin_message",
      label: "插件消息",
      pluginId: "planner",
      pluginName: "PlannerPlugin"
    });
    expect(state.nodes[0].framework?.content).toContain("Collected research notes");
    expect(state.nodes[0].framework?.content).toContain('"count": 3');
  });

  it("tracks subagent events as a first-class subsystem", () => {
    let state = createInitialState();
    const status = {
      id: "sub_1",
      name: "worker-1",
      role: "worker",
      state: "RUNNING",
      runner_state: "RUNNING",
      executor_state: "RUNNING",
      queue_lengths: { normal: 0 }
    };

    state = reduceCoreEvent(state, frame("subagent.created", {
      subagent_id: "sub_1",
      subagent_name: "worker-1",
      subagent_role: "worker",
      status
    }, 10));
    state = reduceCoreEvent(state, frame("subagent.event", {
      subagent_id: "sub_1",
      subagent_name: "worker-1",
      subagent_role: "worker",
      status,
      child_event: {
        type: "model.content_block_delta",
        run_id: "run-sub",
        delta_type: "text",
        delta: "hel"
      }
    }, 11));

    expect(state.subagentOrder).toEqual(["sub_1"]);
    expect(state.subagents.sub_1.status?.plugins).toEqual([]);
    expect(state.subagents.sub_1.status?.toolNames).toEqual([]);
    expect(state.subagents.sub_1.nodes).toHaveLength(1);
    expect(state.subagents.sub_1.nodes[0]).toMatchObject({
      kind: "agent",
      content: "hel",
      complete: false
    });
    expect(state.pluginMessages).toHaveLength(0);

    state = reduceCoreEvent(state, frame("subagent.event", {
      subagent_id: "sub_1",
      subagent_name: "worker-1",
      subagent_role: "worker",
      status,
      child_event: {
        type: "agent.message_added",
        run_id: "run-sub",
        role: "assistant"
      },
      message_entry: {
        version: 1,
        timestamp: 12,
        run_id: "run-sub",
        role: "assistant",
        content: [{ type: "text", text: "hello" }]
      }
    }, 12));

    expect(state.subagents.sub_1.messageHistory).toHaveLength(1);
    expect(state.subagents.sub_1.nodes).toHaveLength(1);
    expect(state.subagents.sub_1.nodes[0]).toMatchObject({
      kind: "agent",
      content: "hello",
      complete: true
    });

    state = reduceCoreEvent(state, frame("subagent.closed", {
      subagent_id: "sub_1",
      subagent_name: "worker-1",
      subagent_role: "worker",
      reason: "done",
      status: { ...status, state: "CLOSED", runner_state: "IDLE", executor_state: "IDLE" }
    }, 13));

    expect(state.subagents.sub_1.state).toBe("CLOSED");
  });

  it("keeps a subagent's creation timestamp stable across status updates", () => {
    let state = createInitialState();

    state = reduceCoreEvent(state, frame("subagent.created", {
      subagent_id: "sub_1",
      status: {
        id: "sub_1",
        name: "worker-1",
        role: "worker",
        state: "RUNNING",
        runner_state: "RUNNING",
        executor_state: "RUNNING",
        queue_lengths: { normal: 0 },
        created_at: 100,
        updated_at: 100
      }
    }, 100));
    state = reduceCoreEvent(state, frame("subagent.event", {
      subagent_id: "sub_1",
      status: {
        id: "sub_1",
        name: "worker-1",
        role: "worker",
        state: "RUNNING",
        runner_state: "RUNNING",
        executor_state: "RUNNING",
        queue_lengths: { normal: 0 },
        updated_at: 999
      },
      child_event: { type: "subagent.status" }
    }, 999));

    expect(state.subagents.sub_1.createdAt).toBe(100);
    expect(state.subagents.sub_1.status?.updatedAt).toBe(999);
  });

  it("captures subagent plugin and tool configuration from status snapshots", () => {
    let state = createInitialState();

    state = reduceCoreEvent(state, frame("subagent.created", {
      subagent_id: "sub_tools",
      status: {
        id: "sub_tools",
        name: "worker-tools",
        role: "worker",
        state: "RUNNING",
        runner_state: "RUNNING",
        executor_state: "IDLE",
        queue_lengths: { normal: 0 },
        plugins: [
          {
            id: "hawi/filesystem",
            name: "Filesystem",
            display_name: "Filesystem",
            class_name: "FileSystemPlugin"
          }
        ],
        plugin_ids: ["hawi/filesystem"],
        tool_names: ["read_file", "grep"],
        tool_count: 2
      }
    }, 30));

    expect(state.subagents.sub_tools.status?.pluginIds).toEqual(["hawi/filesystem"]);
    expect(state.subagents.sub_tools.status?.plugins[0]).toMatchObject({
      id: "hawi/filesystem",
      name: "Filesystem",
      displayName: "Filesystem",
      className: "FileSystemPlugin"
    });
    expect(state.subagents.sub_tools.status?.toolNames).toEqual(["read_file", "grep"]);
    expect(state.subagents.sub_tools.status?.toolCount).toBe(2);
  });

  it("marks shared-context subagents with a handoff node", () => {
    let state = createInitialState();

    state = reduceCoreEvent(state, frame("subagent.created", {
      subagent_id: "sub_shared",
      subagent_name: "reviewer-1",
      subagent_role: "reviewer",
      status: {
        id: "sub_shared",
        name: "reviewer-1",
        role: "reviewer",
        state: "RUNNING",
        runner_state: "RUNNING",
        executor_state: "IDLE",
        queue_lengths: { normal: 1 },
        mode: "fork",
        shared_context: true
      }
    }, 20));

    expect(state.subagents.sub_shared.sharedContext).toBe(true);
    expect(state.subagents.sub_shared.mode).toBe("fork");
    expect(state.subagents.sub_shared.nodes).toHaveLength(1);
    expect(state.subagents.sub_shared.nodes[0]).toMatchObject({
      kind: "handoff",
      content: expect.stringContaining("前文延续")
    });

    state = reduceCoreEvent(state, frame("subagent.event", {
      subagent_id: "sub_shared",
      subagent_name: "reviewer-1",
      subagent_role: "reviewer",
      status: {
        id: "sub_shared",
        name: "reviewer-1",
        role: "reviewer",
        state: "RUNNING",
        runner_state: "RUNNING",
        executor_state: "RUNNING",
        queue_lengths: { normal: 0 },
        mode: "fork",
        shared_context: true
      },
      child_event: {
        type: "model.content_block_delta",
        run_id: "run-shared",
        delta_type: "text",
        delta: "working"
      }
    }, 21));

    expect(state.subagents.sub_shared.nodes).toHaveLength(2);
    expect(state.subagents.sub_shared.nodes[0].kind).toBe("handoff");
    expect(state.subagents.sub_shared.nodes[1]).toMatchObject({
      kind: "agent",
      content: "working",
      complete: false
    });
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
    expect(state.nodes[1].streamDurationMs).toBe(1234);
  });

  it("records run stop duration from the run start time", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("run.start", { run_id: "run-total-time", user_content: "hi", queue: "normal" }, 10));
    state = reduceCoreEvent(state, frame("run.text_delta", { run_id: "run-total-time", delta: "answer" }, 12));
    state = reduceCoreEvent(state, frame("run.stop", { run_id: "run-total-time", stop_reason: "end_turn" }, 13.5));

    expect(state.nodes.map((node) => node.kind)).toEqual(["user", "agent", "divider"]);
    expect(state.nodes[2]).toMatchObject({
      content: "end_turn",
      streamStartedAt: 10000,
      streamFinishedAt: 13500,
      streamDurationMs: 3500
    });
  });

  it("clears visible chat while preserving status", () => {
    let state = createInitialState();
    state = reduceCoreEvent(state, frame("core.status", { runner_state: "RUNNING", agent_state: "RUNNING", queue_lengths: { urgent: 1, high_prio: 0, normal: 0 } }));
    state = reduceCoreEvent(state, frame("run.start", { run_id: "run-4", user_content: "hi", queue: "normal" }));
    state = reduceCoreEvent(state, frame("debug.info", { message: "debug" }));

    state = reduceCoreEvent(state, frame("gui.clear_chat", {}));

    expect(state.nodes).toEqual([]);
    expect(state.debugLines).toEqual([]);
    expect(state.runnerState).toBe("RUNNING");
    expect(state.queueLengths).toEqual({ urgent: 1, high_prio: 0, normal: 0 });
  });
});
