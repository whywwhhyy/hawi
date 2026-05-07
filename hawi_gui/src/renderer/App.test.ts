import { describe, expect, it } from "vitest";
import { createElement } from "react";
import { renderToString } from "react-dom/server";
import App, { isNearChatBottom, renderMarkdown, renderPriorityStatusText, resolveFollowTailOnScroll, shouldSubmitInputFromKeyEvent, thinkingExcerpt } from "./App";

describe("App", () => {
  it("renders the boot screen without crashing", () => {
    expect(renderToString(createElement(App))).toContain("Loading Hawi metadata");
  });
});

describe("thinkingExcerpt", () => {
  it("does not add an ellipsis when the summary is the full content", () => {
    expect(thinkingExcerpt("exactly twelve", 14)).toBe("exactly twelve");
  });

  it("adds an ellipsis only when content is truncated", () => {
    expect(thinkingExcerpt("too long", 3)).toBe("too...");
  });
});

describe("renderPriorityStatusText", () => {
  it("describes urgent as interruption, high priority as a merged slot, and normal as queue length", () => {
    expect(renderPriorityStatusText({ urgent: 1, high_prio: 3, normal: 4 })).toBe("打断 待打断 · 合并 1 · 队列 4");
  });

  it("shows empty interruption and merge states", () => {
    expect(renderPriorityStatusText({ urgent: 0, high_prio: 0, normal: 0 })).toBe("打断 无 · 合并 0 · 队列 0");
  });

  it("counts pending high priority previews as a merged slot", () => {
    expect(renderPriorityStatusText(
      { urgent: 0, high_prio: 0, normal: 2 },
      {
        urgent: [],
        high_prio: [{ id: "steer-1", queue: "high_prio", contentPreview: "steer" }],
        normal: []
      }
    )).toBe("打断 无 · 合并 1 · 队列 2");
  });

  it("counts pending normal previews as ordinary queue work", () => {
    expect(renderPriorityStatusText(
      { urgent: 0, high_prio: 0, normal: 0 },
      {
        urgent: [],
        high_prio: [],
        normal: [{ id: "steer-plain", queue: "normal", contentPreview: "plain" }]
      }
    )).toBe("打断 无 · 合并 0 · 队列 1");
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
