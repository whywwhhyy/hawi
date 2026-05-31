import { useCallback, useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { Ellipsis } from "lucide-react";

const useBrowserLayoutEffect = typeof window === "undefined" ? useEffect : useLayoutEffect;

export type OverflowToolbarPlacement = "toolbar" | "overflow";

export interface OverflowToolbarItem {
  id: string;
  render: (placement: OverflowToolbarPlacement, closeOverflow: () => void) => ReactNode;
}

interface OverflowToolbarProps {
  items: OverflowToolbarItem[];
  label: string;
  className?: string;
  overflowOpen: boolean;
  onOverflowOpenChange: (open: boolean) => void;
}

export function resolveOverflowVisibleCount(
  availableWidth: number,
  itemWidths: number[],
  overflowButtonWidth: number,
  gap: number
): number {
  if (itemWidths.length === 0) return 0;
  if (availableWidth <= 0) return 0;
  const normalizedGap = Math.max(0, gap);
  const fullWidth = itemWidths.reduce((total, width) => total + width, 0)
    + normalizedGap * Math.max(0, itemWidths.length - 1);
  if (fullWidth <= availableWidth) return itemWidths.length;

  let visibleWidth = 0;
  let visibleCount = 0;
  for (const itemWidth of itemWidths) {
    const nextVisibleWidth = visibleWidth + itemWidth + (visibleCount > 0 ? normalizedGap : 0);
    const totalWithOverflow = nextVisibleWidth + normalizedGap + overflowButtonWidth;
    if (totalWithOverflow > availableWidth) break;
    visibleWidth = nextVisibleWidth;
    visibleCount += 1;
  }
  return visibleCount;
}

export function OverflowToolbar({
  items,
  label,
  className = "",
  overflowOpen,
  onOverflowOpenChange
}: OverflowToolbarProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const measureRef = useRef<HTMLDivElement | null>(null);
  const [visibleCount, setVisibleCount] = useState(items.length);

  const closeOverflow = useCallback(() => {
    onOverflowOpenChange(false);
  }, [onOverflowOpenChange]);

  const measure = useCallback(() => {
    const container = containerRef.current;
    const measureRoot = measureRef.current;
    if (!container || !measureRoot) return;
    const itemWidths = Array.from(measureRoot.querySelectorAll<HTMLElement>("[data-overflow-measure-item]"))
      .map((element) => element.getBoundingClientRect().width);
    const overflowButtonWidth = measureRoot
      .querySelector<HTMLElement>("[data-overflow-measure-more]")
      ?.getBoundingClientRect().width ?? 0;
    const style = window.getComputedStyle(measureRoot);
    const gap = Number.parseFloat(style.columnGap || style.gap || "0") || 0;
    const nextVisibleCount = resolveOverflowVisibleCount(
      container.getBoundingClientRect().width,
      itemWidths,
      overflowButtonWidth,
      gap
    );
    setVisibleCount((current) => current === nextVisibleCount ? current : nextVisibleCount);
  }, []);

  useBrowserLayoutEffect(() => {
    measure();
  });

  useBrowserLayoutEffect(() => {
    const container = containerRef.current;
    const measureRoot = measureRef.current;
    if (!container || !measureRoot || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(() => {
      measure();
    });
    observer.observe(container);
    observer.observe(measureRoot);
    return () => {
      observer.disconnect();
    };
  }, [items.length, measure]);

  useEffect(() => {
    setVisibleCount((current) => Math.min(current, items.length));
  }, [items.length]);

  const clampedVisibleCount = Math.min(visibleCount, items.length);
  const visibleItems = items.slice(0, clampedVisibleCount);
  const overflowItems = items.slice(clampedVisibleCount);
  const hasOverflow = overflowItems.length > 0;

  useEffect(() => {
    if (!hasOverflow && overflowOpen) {
      onOverflowOpenChange(false);
    }
  }, [hasOverflow, overflowOpen, onOverflowOpenChange]);

  return (
    <div className={`overflow-toolbar ${className}`.trim()} aria-label={label} ref={containerRef}>
      <div className="overflow-toolbar-list">
        {visibleItems.map((item) => (
          <div className="overflow-toolbar-item" key={item.id}>
            {item.render("toolbar", closeOverflow)}
          </div>
        ))}
        {hasOverflow && (
          <div className="overflow-toolbar-menu-anchor">
            <button
              type="button"
              className={`tool-button overflow-toolbar-more ${overflowOpen ? "active" : ""}`}
              title="更多操作"
              aria-label="更多操作"
              aria-haspopup="menu"
              aria-expanded={overflowOpen}
              onClick={() => onOverflowOpenChange(!overflowOpen)}
            >
              <Ellipsis size={18} />
            </button>
            {overflowOpen && (
              <div className="menu-popover overflow-toolbar-menu" role="menu">
                {overflowItems.map((item) => (
                  <div className="overflow-toolbar-menu-item" key={item.id}>
                    {item.render("overflow", closeOverflow)}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
      <div className="overflow-toolbar-measure" aria-hidden="true" ref={measureRef}>
        {items.map((item) => (
          <div data-overflow-measure-item="" key={item.id}>
            {item.render("toolbar", closeOverflow)}
          </div>
        ))}
        <div data-overflow-measure-more="">
          <button type="button" className="tool-button overflow-toolbar-more" tabIndex={-1}>
            <Ellipsis size={18} />
          </button>
        </div>
      </div>
    </div>
  );
}
