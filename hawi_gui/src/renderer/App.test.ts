import { describe, expect, it } from "vitest";
import { createElement } from "react";
import { renderToString } from "react-dom/server";
import type { GuiMetadata } from "../shared/protocol";
import { VERSION } from "../shared/protocol";
import App, { artifactTypeLabel, formatSessionTimestamp, formatStreamFinishedLabel, formatToolCopyText, groupArtifactsByType, inputHistoryFromChatNodes, isNearChatBottom, mergeInputHistory, reduceSessionStates, renderMarkdown, renderPriorityStatusText, renderSessionCounterText, resolveEscapeDismissTarget, resolveFollowTailOnScroll, sessionLoadStateLabel, shouldBubbleNestedVerticalScroll, shouldInitializeSessionState, shouldNavigateInputHistoryFromKeyEvent, shouldSubmitInputFromKeyEvent, sortSessionsByCreatedAt, thinkingExcerpt, upsertSessionRuntime } from "./App";
import type { PluginArtifactState } from "./state";

describe("App", () => {
  it("renders the boot screen without crashing", () => {
    expect(renderToString(createElement(App))).toContain("Loading Hawi metadata");
  });
});

function makeMetadata(coreRunning: boolean): GuiMetadata {
  return {
    inspect: {
      version: VERSION,
      models: [],
      plugin_catalog: [],
      default_system_prompt: "你是Hawi，一个通用agent"
    },
    config: {
      version: 1,
      modelName: "",
      systemPrompt: "你是Hawi，一个通用agent",
      selectedPlugins: [],
      pluginConfigs: {},
      showDebug: true
    },
    coreRunning
  };
}

describe("thinkingExcerpt", () => {
  it("does not add an ellipsis when the summary is the full content", () => {
    expect(thinkingExcerpt("exactly twelve", 14)).toBe("exactly twelve");
  });

  it("adds an ellipsis only when content is truncated", () => {
    expect(thinkingExcerpt("too long", 3)).toBe("too...");
  });
});

describe("formatStreamFinishedLabel", () => {
  it("formats stream completion durations in seconds", () => {
    expect(formatStreamFinishedLabel(1250)).toBe("finished in 1.3s");
  });

  it("skips missing stream durations", () => {
    expect(formatStreamFinishedLabel()).toBeNull();
  });
});

describe("renderPriorityStatusText", () => {
  it("describes high priority as a priority slot and normal as queue length", () => {
    expect(renderPriorityStatusText({ urgent: 1, high_prio: 3, normal: 4 })).toBe("Message Insert 1 · Queue 4");
  });

  it("shows empty priority and queue states", () => {
    expect(renderPriorityStatusText({ urgent: 0, high_prio: 0, normal: 0 })).toBe("Message Insert 0 · Queue 0");
  });

  it("counts pending high priority previews as a priority slot", () => {
    expect(renderPriorityStatusText(
      { urgent: 0, high_prio: 0, normal: 2 },
      {
        urgent: [],
        high_prio: [{ id: "steer-1", queue: "high_prio", contentPreview: "steer" }],
        normal: []
      }
    )).toBe("Message Insert 1 · Queue 2");
  });

  it("counts pending normal previews as ordinary queue work", () => {
    expect(renderPriorityStatusText(
      { urgent: 0, high_prio: 0, normal: 0 },
      {
        urgent: [],
        high_prio: [],
        normal: [{ id: "steer-plain", queue: "normal", contentPreview: "plain" }]
      }
    )).toBe("Message Insert 0 · Queue 1");
  });
});

describe("session runtime display helpers", () => {
  it("renders running over loaded counts", () => {
    expect(renderSessionCounterText(2, 5)).toBe("2/5");
  });

  it("labels all session load states", () => {
    expect(sessionLoadStateLabel("unloaded")).toBe("未加载");
    expect(sessionLoadStateLabel("loaded")).toBe("已加载待命");
    expect(sessionLoadStateLabel("running")).toBe("运行中");
  });

  it("keeps separate chat state per session id", () => {
    let states = reduceSessionStates({}, {
      sessionId: "session-a",
      frame: {
        version: VERSION,
        type: "run.start",
        payload: { run_id: "run-a", user_content: "hello a", queue: "normal" }
      }
    });
    states = reduceSessionStates(states, {
      sessionId: "session-b",
      frame: {
        version: VERSION,
        type: "run.start",
        payload: { run_id: "run-b", user_content: "hello b", queue: "normal" }
      }
    });

    expect(states["session-a"].nodes[0].content).toBe("hello a");
    expect(states["session-b"].nodes[0].content).toBe("hello b");
  });

  it("does not create a session-list item from empty runtime status", () => {
    const sessions = upsertSessionRuntime([], "session-empty", {
      load_state: "loaded",
      loaded_at: 1000,
      last_finished_at: undefined
    });

    expect(sessions).toEqual([]);
  });

  it("creates a session-list item once runtime status is visibly materialized", () => {
    const sessions = upsertSessionRuntime([], "session-active", {
      load_state: "running",
      loaded_at: 1000,
      last_finished_at: undefined
    }, { createIfMissing: true });

    expect(sessions.map((session) => session.session_id)).toEqual(["session-active"]);
  });
});

describe("resolveEscapeDismissTarget", () => {
  const closed = {
    contextCompactDialogOpen: false,
    contextCompactBusy: false,
    pluginDialogOpen: false,
    modelDialogOpen: false,
    debugMenuOpen: false,
    queuePopoverOpen: false,
    editingQueueTaskId: null,
    sessionDialogOpen: false
  };

  it("dismisses modal dialogs before popovers", () => {
    expect(resolveEscapeDismissTarget({
      ...closed,
      modelDialogOpen: true,
      queuePopoverOpen: true
    })).toBe("modelDialog");
    expect(resolveEscapeDismissTarget({
      ...closed,
      pluginDialogOpen: true,
      modelDialogOpen: true
    })).toBe("pluginDialog");
  });

  it("does not close the context compact dialog while it is busy", () => {
    expect(resolveEscapeDismissTarget({
      ...closed,
      contextCompactDialogOpen: true,
      contextCompactBusy: true,
      pluginDialogOpen: true
    })).toBeNull();
  });

  it("cancels queue item editing before closing the queue popover", () => {
    expect(resolveEscapeDismissTarget({
      ...closed,
      queuePopoverOpen: true,
      editingQueueTaskId: "task-1"
    })).toBe("queueTaskEdit");
  });
});

describe("artifact sidebar helpers", () => {
  function artifact(key: string, artifactType: string): PluginArtifactState {
    return {
      key,
      id: key,
      pluginId: "plugin",
      pluginName: "Plugin",
      artifactType,
      title: key,
      updatedAt: 1
    };
  }

  it("groups artifacts by type in first-seen order", () => {
    const groups = groupArtifactsByType([
      artifact("a", "plan"),
      artifact("b", "file"),
      artifact("c", "plan")
    ]);

    expect(groups.map((group) => [group.type, group.artifacts.map((item) => item.key)])).toEqual([
      ["plan", ["a", "c"]],
      ["file", ["b"]]
    ]);
  });

  it("formats artifact type labels for sidebar tabs", () => {
    expect(artifactTypeLabel("tool_result")).toBe("Tool Result");
    expect(artifactTypeLabel("")).toBe("Artifact");
  });
});

describe("shouldInitializeSessionState", () => {
  it("waits until the core process is running", () => {
    expect(shouldInitializeSessionState(null)).toBe(false);
    expect(shouldInitializeSessionState(makeMetadata(false))).toBe(false);
    expect(shouldInitializeSessionState(makeMetadata(true))).toBe(true);
  });
});

describe("sortSessionsByCreatedAt", () => {
  it("uses creation time instead of last update time", () => {
    const sessions = sortSessionsByCreatedAt([
      {
        session_id: "older-but-updated",
        name: "older",
        created_at: "2024-01-01T00:00:00",
        updated_at: "2026-01-01T00:00:00",
        last_checkpoint_event: null,
        components_present: []
      },
      {
        session_id: "newer",
        name: "newer",
        created_at: "2025-01-01T00:00:00",
        updated_at: "2025-01-01T00:00:00",
        last_checkpoint_event: null,
        components_present: []
      }
    ]);

    expect(sessions.map((session) => session.session_id)).toEqual([
      "newer",
      "older-but-updated"
    ]);
  });
});

describe("formatSessionTimestamp", () => {
  const now = Date.parse("2026-05-15T12:00:00Z");

  it("uses compact relative labels for recent sessions", () => {
    expect(formatSessionTimestamp("2026-05-15T11:59:40Z", now)).toBe("刚刚");
    expect(formatSessionTimestamp("2026-05-15T11:55:00Z", now)).toBe("5分钟前");
    expect(formatSessionTimestamp("2026-05-14T12:00:00Z", now)).toBe("昨天");
  });

  it("falls back to a compact date for older sessions", () => {
    expect(formatSessionTimestamp("2026-05-01T12:00:00Z", now)).toMatch(/5.*1/);
  });
});

describe("isNearChatBottom", () => {
  it("allows auto scroll when the chat is less than 5px from the bottom", () => {
    expect(isNearChatBottom({ scrollHeight: 1000, scrollTop: 595.5, clientHeight: 400 })).toBe(true);
  });

  it("does not allow auto scroll when the chat is 5px or more from the bottom", () => {
    expect(isNearChatBottom({ scrollHeight: 1000, scrollTop: 595, clientHeight: 400 })).toBe(false);
  });
});

describe("shouldBubbleNestedVerticalScroll", () => {
  it("keeps wheel events inside a nested scroller while it can scroll down", () => {
    expect(shouldBubbleNestedVerticalScroll({ scrollHeight: 1000, scrollTop: 200, clientHeight: 400 }, 20)).toBe(false);
  });

  it("bubbles wheel events once a nested scroller reaches the bottom", () => {
    expect(shouldBubbleNestedVerticalScroll({ scrollHeight: 1000, scrollTop: 600, clientHeight: 400 }, 20)).toBe(true);
  });

  it("bubbles wheel events when the current wheel delta would cross the bottom", () => {
    expect(shouldBubbleNestedVerticalScroll({ scrollHeight: 1000, scrollTop: 580, clientHeight: 400 }, 40)).toBe(true);
  });

  it("bubbles wheel events once a nested scroller reaches the top", () => {
    expect(shouldBubbleNestedVerticalScroll({ scrollHeight: 1000, scrollTop: 0, clientHeight: 400 }, -20)).toBe(true);
  });

  it("bubbles wheel events when the current wheel delta would cross the top", () => {
    expect(shouldBubbleNestedVerticalScroll({ scrollHeight: 1000, scrollTop: 10, clientHeight: 400 }, -20)).toBe(true);
  });

  it("bubbles wheel events when nested content is not scrollable", () => {
    expect(shouldBubbleNestedVerticalScroll({ scrollHeight: 400, scrollTop: 0, clientHeight: 400 }, 20)).toBe(true);
  });
});

describe("resolveFollowTailOnScroll", () => {
  it("keeps following during programmatic scroll lag", () => {
    expect(resolveFollowTailOnScroll(true, false, false, false, false)).toBe(true);
  });

  it("keeps following while auto scrolling is still settling", () => {
    expect(resolveFollowTailOnScroll(false, false, false, false, true)).toBe(true);
  });

  it("stops following when the user scrolls away from the bottom", () => {
    expect(resolveFollowTailOnScroll(true, false, true, false, false)).toBe(false);
  });

  it("resumes following when the chat is back near the bottom", () => {
    expect(resolveFollowTailOnScroll(false, true, true, false, false)).toBe(true);
  });
});

describe("renderMarkdown", () => {
  it("renders fenced code blocks with syntax highlighting classes", () => {
    const html = renderMarkdown("```ts\nconst answer = 42;\n```");

    expect(html).toContain("class=\"code-block\"");
    expect(html).toContain("class=\"hljs language-typescript\"");
    expect(html).toContain("hljs-keyword");
  });

  it("escapes unknown language code blocks", () => {
    const html = renderMarkdown("```unknown\n<script>\n```");

    expect(html).toContain("&lt;script&gt;");
    expect(html).not.toContain("<script>");
  });

  it("renders links for external opening instead of current-window navigation", () => {
    const html = renderMarkdown("https://example.com");

    expect(html).toContain("href=\"https://example.com\"");
    expect(html).toContain("target=\"_blank\"");
    expect(html).toContain("rel=\"noopener noreferrer\"");
  });
});

describe("formatToolCopyText", () => {
  it("formats tool calls with arguments and result for clipboard copy", () => {
    const text = formatToolCopyText({
      runId: "run-1",
      toolCallId: "call-1",
      name: "read_file",
      status: "success",
      argsRaw: "",
      argsState: "complete",
      arguments: { path: "docs/todo.md" },
      resultPreview: "done"
    });

    expect(text).toContain("Tool: read_file");
    expect(text).toContain("Status: success");
    expect(text).toContain("\"path\": \"docs/todo.md\"");
    expect(text).toContain("Result:\ndone");
  });
});

describe("shouldSubmitInputFromKeyEvent", () => {
  it("submits plain Enter", () => {
    expect(shouldSubmitInputFromKeyEvent({
      key: "Enter",
      shiftKey: false,
      nativeEvent: {}
    }, false)).toBe(true);
  });

  it("does not submit Shift+Enter", () => {
    expect(shouldSubmitInputFromKeyEvent({
      key: "Enter",
      shiftKey: true,
      nativeEvent: {}
    }, false)).toBe(false);
  });

  it("does not submit Enter while an IME composition is active", () => {
    expect(shouldSubmitInputFromKeyEvent({
      key: "Enter",
      shiftKey: false,
      nativeEvent: { isComposing: true }
    }, false)).toBe(false);
  });

  it("does not submit Enter for IME key events reported as keyCode 229", () => {
    expect(shouldSubmitInputFromKeyEvent({
      key: "Enter",
      shiftKey: false,
      nativeEvent: { keyCode: 229 }
    }, false)).toBe(false);
  });
});

describe("input history navigation", () => {
  it("collects user chat nodes as editable input history", () => {
    expect(inputHistoryFromChatNodes([
      { id: "u1", kind: "user", content: " first " },
      { id: "a1", kind: "agent", content: "answer" },
      { id: "u2", kind: "user", content: "" },
      { id: "u3", kind: "user", content: "second" }
    ])).toEqual(["first", "second"]);
  });

  it("merges local and replayed input history without duplicate entries", () => {
    expect(mergeInputHistory(["first", "second"], ["second", "third"])).toEqual([
      "first",
      "second",
      "third"
    ]);
  });

  it("uses ArrowUp at the start of the input to select older messages", () => {
    expect(shouldNavigateInputHistoryFromKeyEvent({
      key: "ArrowUp",
      shiftKey: false,
      nativeEvent: {}
    }, "draft", 0, 0, false, false)).toBe("previous");
  });

  it("uses ArrowUp from a single-line draft to start browsing history", () => {
    expect(shouldNavigateInputHistoryFromKeyEvent({
      key: "ArrowUp",
      shiftKey: false,
      nativeEvent: {}
    }, "draft", 3, 3, false, false)).toBe("previous");
  });

  it("keeps ArrowUp as cursor movement below the first line of multi-line text", () => {
    expect(shouldNavigateInputHistoryFromKeyEvent({
      key: "ArrowUp",
      shiftKey: false,
      nativeEvent: {}
    }, "line one\nline two", 10, 10, false, false)).toBeNull();
  });

  it("uses ArrowDown to move forward only while browsing history", () => {
    expect(shouldNavigateInputHistoryFromKeyEvent({
      key: "ArrowDown",
      shiftKey: false,
      nativeEvent: {}
    }, "history item", 12, 12, false, true)).toBe("next");
    expect(shouldNavigateInputHistoryFromKeyEvent({
      key: "ArrowDown",
      shiftKey: false,
      nativeEvent: {}
    }, "history item", 12, 12, false, false)).toBeNull();
  });

  it("does not navigate history while composing with an IME", () => {
    expect(shouldNavigateInputHistoryFromKeyEvent({
      key: "ArrowUp",
      shiftKey: false,
      nativeEvent: { isComposing: true }
    }, "", 0, 0, false, false)).toBeNull();
  });
});
