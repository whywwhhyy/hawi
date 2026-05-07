import { describe, expect, it } from "vitest";
import { createElement } from "react";
import { renderToString } from "react-dom/server";
import App, { isNearChatBottom, renderMarkdown, resolveFollowTailOnScroll, shouldSubmitInputFromKeyEvent, thinkingExcerpt } from "./App";

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
