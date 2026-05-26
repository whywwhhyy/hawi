import type { ButtonHTMLAttributes, HTMLAttributes, ReactNode } from "react";

export function StatusCell({
  active = false,
  className,
  children,
  ...props
}: HTMLAttributes<HTMLDivElement> & {
  active?: boolean;
}) {
  return (
    <div {...props} className={classNames(className, active && "active")}>
      {children}
    </div>
  );
}

export function StatusCellTrigger({
  label,
  contentClassName,
  contentAriaHidden = true,
  children,
  type = "button",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  label: ReactNode;
  contentClassName: string;
  contentAriaHidden?: boolean;
}) {
  return (
    <button {...props} type={type}>
      <span className="status-cell-label">{label}</span>
      <span className={contentClassName} aria-hidden={contentAriaHidden}>
        {children}
      </span>
    </button>
  );
}

export function StatusCellDisplay({
  label,
  contentClassName,
  contentAriaHidden = true,
  children,
  ...props
}: HTMLAttributes<HTMLDivElement> & {
  label: ReactNode;
  contentClassName: string;
  contentAriaHidden?: boolean;
}) {
  return (
    <div {...props}>
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
