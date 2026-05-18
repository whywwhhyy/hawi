import { describe, expect, it } from "vitest";
import { MIN_CONTENT_SIZE, minimumWindowSizeForContent, normalizeMinimumContentSize } from "./layout";

describe("layout sizing helpers", () => {
  it("keeps the configured minimum content size as the lower bound", () => {
    expect(normalizeMinimumContentSize({ width: 900, height: 500 })).toEqual(MIN_CONTENT_SIZE);
  });

  it("rounds dynamic content measurements up to whole pixels", () => {
    expect(normalizeMinimumContentSize({ width: 1200.2, height: 700.1 })).toEqual({
      width: 1201,
      height: 701,
    });
  });

  it("falls back to configured minimums for invalid measurements", () => {
    expect(normalizeMinimumContentSize({ width: Number.NaN, height: Number.POSITIVE_INFINITY })).toEqual(MIN_CONTENT_SIZE);
  });

  it("adds native frame insets to content minimums", () => {
    expect(minimumWindowSizeForContent(MIN_CONTENT_SIZE, { width: 16, height: 39 })).toEqual({
      width: 1096,
      height: 699,
    });
  });
});
