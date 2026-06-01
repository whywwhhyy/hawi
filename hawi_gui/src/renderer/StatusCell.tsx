import { forwardRef, type ButtonHTMLAttributes, type HTMLAttributes, type ReactNode } from "react";

export const StatusCell = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement> & {
  active?: boolean;
}>(function StatusCell({
  active = false,
  className,
  children,
  ...props
}, ref) {
  return (
    <div ref={ref} {...props} className={classNames("status-cell", className, active && "active")}>
      {children}
    </div>
  );
});

export const StatusCellTrigger = forwardRef<HTMLButtonElement, ButtonHTMLAttributes<HTMLButtonElement> & {
  label: ReactNode;
  contentClassName: string;
  contentAriaHidden?: boolean;
}>(function StatusCellTrigger({
  label,
  contentClassName,
  contentAriaHidden = true,
  children,
  type = "button",
  className,
  ...props
}, ref) {
  return (
    <button ref={ref} {...props} type={type} className={classNames("status-cell-trigger", className)}>
      <span className="status-cell-label">{label}</span>
      <span className={contentClassName} aria-hidden={contentAriaHidden}>
        {children}
      </span>
    </button>
  );
});

export function StatusCellDisplay({
  label,
  contentClassName,
  contentAriaHidden = true,
  children,
  className,
  ...props
}: HTMLAttributes<HTMLDivElement> & {
  label: ReactNode;
  contentClassName: string;
  contentAriaHidden?: boolean;
}) {
  return (
    <div {...props} className={classNames("status-cell", "status-cell-display", className)}>
      <span className="status-cell-label">{label}</span>
      <span className={contentClassName} aria-hidden={contentAriaHidden}>
        {children}
      </span>
    </div>
  );
}

export function StatusPopoverHeader({
  title,
  value
}: {
  title: ReactNode;
  value?: ReactNode;
}) {
  return (
    <header>
      <span>{title}</span>
      {value !== undefined && <strong>{value}</strong>}
    </header>
  );
}

function classNames(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(" ");
}
