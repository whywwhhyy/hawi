import { describe, expect, it } from "vitest";
import { createElement } from "react";
import { renderToString } from "react-dom/server";
import type { GuiMetadata } from "../shared/protocol";
import { VERSION } from "../shared/protocol";
import App, { artifactTypeLabel, buildFocusTranscriptItems, canStopRunnerState, formatSessionTimestamp, formatStreamFinishedLabel, formatToolCopyText, groupArtifactsByType, inputHistoryFromChatNodes, isNearChatBottom, MERMAID_RENDER_CONFIG, mergeInputHistory, middleEllipsizePath, modelProviderConfigPreviewLines, projectNameFromPath, reduceSessionStates, renderMarkdown, renderPriorityStatusText, renderSessionCounterText, renderUsageStatusText, resolveEscapeDismissTarget, resolveFollowTailOnScroll, resumePayloadFromInput, sanitizeRenderedMermaidHtml, sessionLoadStateLabel, shouldBubbleNestedVerticalScroll, shouldInitializeSessionState, shouldNavigateInputHistoryFromKeyEvent, shouldSubmitInputFromKeyEvent, sortSessionsByCreatedAt, sortSubAgentsByCreatedAt, thinkingExcerpt, upsertSessionRuntime } from "./App";
import { resolveOverflowVisibleCount } from "./OverflowToolbar";
import type { ChatNode, PluginArtifactState, SubAgentRuntimeState } from "./state";

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
      toolCallPurposeEnabled: true,
      showDebug: true,
      focusModeEnabled: true
    },
    coreRunning
  };
}

function makeSubAgent(overrides: Partial<SubAgentRuntimeState>): SubAgentRuntimeState {
  return {
    id: "subagent",
    name: "SubAgent",
    role: "general",
    state: "CREATED",
    messageHistory: [],
    nodes: [],
    partial: { text: "", reasoning: "" },
    eventCount: 0,
    ...overrides
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

describe("buildFocusTranscriptItems", () => {
  it("folds everything after a user message except the last agent reply", () => {
    const nodes: ChatNode[] = [
      { id: "user-1", kind: "user", content: "question" },
      { id: "thinking-1", kind: "thinking", content: "working", streamDurationMs: 1500 },
      {
        id: "tool-1",
        kind: "tool",
        content: "",
        tool: {
          runId: "run-1",
          toolCallId: "tool-1",
          name: "read_file",
          status: "success",
          argsRaw: "{}",
          argsState: "complete",
          resultPreview: "ok"
        }
      },
      { id: "agent-1", kind: "agent", content: "draft" },
      { id: "divider-1", kind: "divider", content: "end_turn · 4.1s" },
      { id: "agent-2", kind: "agent", content: "final" },
      { id: "user-2", kind: "user", content: "next" },
      { id: "agent-3", kind: "agent", content: "answer" },
    ];

    const items = buildFocusTranscriptItems(nodes);

    expect(items.map((item) => item.type)).toEqual(["node", "focus-fold", "node", "node", "node"]);
    expect(items[1]).toMatchObject({
      type: "focus-fold",
      group: {
        nodes: [
          { id: "thinking-1" },
          { id: "tool-1" },
          { id: "agent-1" },
          { id: "divider-1" },
        ],
        summary: {
          toolCount: 1,
          activity: "reading",
          active: false,
          label: "1 tool · reading"
        }
      }
    });
    expect(items[2]).toMatchObject({ type: "node", node: { id: "agent-2" } });
  });

  it("folds active work before the final reply exists", () => {
    const nodes: ChatNode[] = [
      { id: "user-1", kind: "user", content: "question" },
      { id: "thinking-1", kind: "thinking", content: "working", complete: false },
    ];

    const items = buildFocusTranscriptItems(nodes);

    expect(items.map((item) => item.type)).toEqual(["node", "focus-fold"]);
    expect(items[1]).toMatchObject({
      type: "focus-fold",
      group: {
        nodes: [{ id: "thinking-1" }],
        summary: {
          toolCount: 0,
          activity: "thinking",
          active: true,
          label: "0 tools · thinking"
        }
      }
    });
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

describe("renderUsageStatusText", () => {
  it("renders an empty usage summary", () => {
    expect(renderUsageStatusText()).toBe("Usage Total 0 · Input 0 · Output 0 · Cache Write 0 · Cache Read 0");
  });

  it("renders total, input/output, and cache write/read counters", () => {
    expect(renderUsageStatusText({
      totalTokens: 15400,
      inputTokens: 12000,
      outputTokens: 1800,
      cacheReadTokens: 1400,
      cacheWriteTokens: 200
    })).toBe("Usage Total 15.4K · Input 12K · Output 1.8K · Cache Write 200 · Cache Read 1.4K");
  });
});

describe("modelProviderConfigPreviewLines", () => {
  it("renders loaded provider config lines", () => {
    expect(modelProviderConfigPreviewLines({
      adapter: "OpenAIModel",
      model_count: 2,
      properties: {
        base_url: "http://localhost:1234/v1",
        api_key: "sk-...abcd"
      }
    })).toEqual([
      "adapter: OpenAIModel",
      "models: 2",
      "base_url: http://localhost:1234/v1",
      "api_key: sk-...abcd"
    ]);
  });

  it("renders summarized provider config without block labels", () => {
    expect(modelProviderConfigPreviewLines({
      adapter: "OpenAIModel, AnthropicModel",
      model_count: 4,
      properties: { base_url: "http://localhost:1234/v1 | https://example.test" }
    })).toEqual([
      "adapter: OpenAIModel, AnthropicModel",
      "models: 4",
      "base_url: http://localhost:1234/v1 | https://example.test"
    ]);
  });
});

describe("resolveOverflowVisibleCount", () => {
  it("keeps every item visible when the toolbar has enough width", () => {
    expect(resolveOverflowVisibleCount(196, [50, 60, 70], 40, 8)).toBe(3);
  });

  it("moves trailing items into overflow while preserving prefix order", () => {
    expect(resolveOverflowVisibleCount(170, [50, 60, 70], 40, 8)).toBe(2);
  });

  it("shows only the overflow trigger when the first item cannot fit beside it", () => {
    expect(resolveOverflowVisibleCount(95, [50, 60, 70], 40, 8)).toBe(0);
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

describe("project status helpers", () => {
  it("uses the directory basename as the project name", () => {
    expect(projectNameFromPath("/Users/hayden/Projects/Python/Hawi")).toBe("Hawi");
    expect(projectNameFromPath("C:\\Users\\hayden\\Project")).toBe("Project");
    expect(projectNameFromPath(null)).toBe("-");
  });

  it("shortens long paths from the middle", () => {
    const path = "/Users/hayden/Projects/Python/Hawi/hawi_gui/src/renderer/App.tsx";
    const shortened = middleEllipsizePath(path, 40);
    expect(shortened).toContain("…");
    expect(shortened.startsWith("/Users/hayden")).toBe(true);
    expect(shortened.endsWith("/renderer/App.tsx")).toBe(true);
  });
});

describe("resolveEscapeDismissTarget", () => {
  const closed = {
    mediaPreviewOpen: false,
    contextPopoverOpen: false,
    projectPopoverOpen: false,
    pluginDialogOpen: false,
    modelDialogOpen: false,
    subagentObserverOpen: false,
    settingsMenuOpen: false,
    queuePopoverOpen: false,
    editingQueueTaskId: null,
    sessionDialogOpen: false
  };

  it("dismisses media previews before stopping the agent or other dialogs", () => {
    expect(resolveEscapeDismissTarget({
      ...closed,
      mediaPreviewOpen: true
    })).toBe("mediaPreview");
    expect(resolveEscapeDismissTarget({
      ...closed,
      mediaPreviewOpen: true,
      subagentObserverOpen: true
    })).toBe("mediaPreview");
  });

  it("dismisses modal dialogs before popovers", () => {
    expect(resolveEscapeDismissTarget({
      ...closed,
      subagentObserverOpen: true,
      pluginDialogOpen: true
    })).toBe("subagentObserver");
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

  it("closes the context popover after modal dialogs", () => {
    expect(resolveEscapeDismissTarget({
      ...closed,
      pluginDialogOpen: true
    })).toBe("pluginDialog");
    expect(resolveEscapeDismissTarget({
      ...closed,
      projectPopoverOpen: true,
      contextPopoverOpen: true
    })).toBe("projectPopover");
    expect(resolveEscapeDismissTarget({
      ...closed,
      contextPopoverOpen: true,
      queuePopoverOpen: true
    })).toBe("contextPopover");
  });

  it("cancels queue item editing before closing the queue popover", () => {
    expect(resolveEscapeDismissTarget({
      ...closed,
      queuePopoverOpen: true,
      editingQueueTaskId: "task-1"
    })).toBe("queueTaskEdit");
  });
});

describe("canStopRunnerState", () => {
  it("allows escape-stop only while the runner is active", () => {
    expect(canStopRunnerState("RUNNING")).toBe(true);
    expect(canStopRunnerState("INTERRUPTING")).toBe(true);
    expect(canStopRunnerState("IDLE")).toBe(false);
    expect(canStopRunnerState("READY")).toBe(false);
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

describe("resumePayloadFromInput", () => {
  it("uses non-empty input as the resume prompt", () => {
    expect(resumePayloadFromInput("  从这里继续处理  ")).toEqual({
      message: "从这里继续处理"
    });
  });

  it("falls back to the engine default resume prompt for empty input", () => {
    expect(resumePayloadFromInput("   ")).toEqual({});
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

describe("sortSubAgentsByCreatedAt", () => {
  it("keeps the sidebar order tied to creation time instead of recent updates", () => {
    const sorted = sortSubAgentsByCreatedAt([
      makeSubAgent({
        id: "newer-but-updated",
        createdAt: 200,
        lastEventAt: 1000,
        state: "RUNNING"
      }),
      makeSubAgent({
        id: "older",
        createdAt: 100,
        lastEventAt: 10,
        state: "COMPLETED"
      })
    ]);

    expect(sorted.map((item) => item.id)).toEqual(["older", "newer-but-updated"]);
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

  it("does not resume following while text selection is active", () => {
    expect(resolveFollowTailOnScroll(true, true, false, true, false)).toBe(false);
  });
});

describe("renderMarkdown", () => {
  it("renders fenced code blocks with syntax highlighting classes", () => {
    const html = renderMarkdown("```ts\nconst answer = 42;\n```");

    expect(html).toContain("class=\"code-copy-button\"");
    expect(html).toContain("class=\"code-block\"");
    expect(html).toContain("class=\"hljs language-typescript\"");
    expect(html).toContain("hljs-keyword");
    expect(html).not.toContain("<pre><code");
    expect(html.match(/<pre\b/g) ?? []).toHaveLength(1);
  });

  it("escapes unknown language code blocks", () => {
    const html = renderMarkdown("```unknown\n<script>\n```");

    expect(html).toContain("&lt;script&gt;");
    expect(html).not.toContain("<script>");
  });

  it("renders svg fenced blocks as a sanitized preview plus copyable code", () => {
    const html = renderMarkdown("```svg\n<svg viewBox=\"0 0 10 10\"><circle cx=\"5\" cy=\"5\" r=\"4\" /></svg>\n```");

    expect(html).toContain("class=\"svg-preview-shell\"");
    expect(html).toContain("data:image/svg+xml;charset=utf-8,");
    expect(html).toContain("class=\"code-copy-button\"");
    expect(html).toContain("class=\"hljs language-xml\"");
  });

  it("renders mermaid fenced blocks as an async preview target plus copyable code", () => {
    const html = renderMarkdown("```mermaid\ngraph TD\n  A-->B\n```");

    expect(html).toContain("class=\"mermaid-preview-shell\"");
    expect(html).toContain("data-mermaid-source=");
    expect(html).toContain(encodeURIComponent("graph TD\n  A-->B\n"));
    expect(html).toContain("Rendering diagram...");
    expect(html).toContain("class=\"code-copy-button\"");
    expect(html).toContain("language-mermaid");
  });

  it("configures mermaid flowcharts to avoid foreignObject labels", () => {
    expect(MERMAID_RENDER_CONFIG.flowchart.htmlLabels).toBe(false);
  });

  it("passes mermaid svg output through without local sanitizing", () => {
    const html = sanitizeRenderedMermaidHtml(`
      <svg>
        <style>
          @import url("https://example.test/bad.css");
          .node rect { fill: #4CAF50; color: #fff; }
          .bad { background: url("https://example.test/bad.png"); }
        </style>
        <foreignObject><div>Label</div></foreignObject>
        <rect onclick="alert(1)" style="fill: #4CAF50; background: url(https://example.test/bad.png)" />
      </svg>
    `);

    expect(html).toContain("<style>");
    expect(html).toContain("<foreignObject>");
    expect(html).toContain("onclick");
    expect(html).toContain("@import");
    expect(html).toContain("url(");
  });

  it("sanitizes svg fenced previews without hiding the source code block", () => {
    const html = renderMarkdown("```svg\n<svg viewBox=\"0 0 10 10\"><script>alert(1)</script><rect onclick=\"alert(2)\" width=\"10\" height=\"10\" /></svg>\n```");
    const src = html.match(/src="([^"]+)"/)?.[1] ?? "";
    const decodedSvg = decodeURIComponent(src.replace("data:image/svg+xml;charset=utf-8,", ""));

    expect(decodedSvg).not.toContain("<script");
    expect(decodedSvg).not.toContain("onclick");
    expect(html).toContain("script");
    expect(html).toContain("onclick");
  });

  it("renders sanitized raw HTML blocks", () => {
    const html = renderMarkdown("<details open><summary>More</summary><table><tr><td>A</td></tr></table></details>");

    expect(html).toContain("<details open>");
    expect(html).toContain("<summary>More</summary>");
    expect(html).toContain("<table>");
    expect(html).toContain("<td>A</td>");
  });

  it("sanitizes dangerous raw HTML", () => {
    const html = renderMarkdown("<script>alert(1)</script><img src=\"javascript:alert(2)\" onerror=\"alert(3)\"><a href=\"jav&#x3a;ascript:alert(4)\">bad</a>");

    expect(html).not.toContain("<script");
    expect(html).not.toContain("javascript:");
    expect(html).not.toContain("onerror");
    expect(html).toContain("<img>");
    expect(html).toContain("<a>bad</a>");
  });

  it("renders sanitized inline raw SVG", () => {
    const html = renderMarkdown("<svg viewBox=\"0 0 10 10\" onload=\"alert(1)\"><script>alert(2)</script><circle cx=\"5\" cy=\"5\" r=\"4\" /></svg>");

    expect(html).toContain("<svg viewBox=\"0 0 10 10\">");
    expect(html).toContain("<circle");
    expect(html).not.toContain("<script");
    expect(html).not.toContain("onload");
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
