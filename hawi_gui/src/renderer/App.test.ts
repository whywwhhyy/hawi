import { describe, expect, it } from "vitest";
import { isNearChatBottom, thinkingExcerpt } from "./App";

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
