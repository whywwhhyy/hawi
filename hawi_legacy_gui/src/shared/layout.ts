export interface LayoutSize {
  width: number;
  height: number;
}

export const MIN_CONTENT_SIZE: LayoutSize = {
  width: 640,
  height: 660,
};

export function normalizeMinimumContentSize(size: Partial<LayoutSize> | null | undefined): LayoutSize {
  return {
    width: normalizeDimension(size?.width, MIN_CONTENT_SIZE.width),
    height: normalizeDimension(size?.height, MIN_CONTENT_SIZE.height),
  };
}

export function minimumWindowSizeForContent(contentSize: LayoutSize, frameSize: Partial<LayoutSize>): LayoutSize {
  return {
    width: Math.ceil(contentSize.width + Math.max(0, frameSize.width ?? 0)),
    height: Math.ceil(contentSize.height + Math.max(0, frameSize.height ?? 0)),
  };
}

function normalizeDimension(value: number | undefined, fallback: number): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return fallback;
  }
  return Math.ceil(Math.max(fallback, value));
}
