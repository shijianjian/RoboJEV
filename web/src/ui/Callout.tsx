import type { ReactNode } from "react";
import { IconAlert } from "./Icons";

/**
 * A boxed note: a warning, an error, or a plain aside.
 *
 * A `warn` or `bad` callout is announced when it appears (`role="alert"`), because those are
 * rendered in response to something the reader just did — a submission the server refused, a
 * recording that would not load — and a screen reader would otherwise never mention it. A
 * `neutral` one is prose that happens to be boxed, so it stays silent; `live={false}` opts a
 * bad/warn one out (for a callout rendered with the page rather than in reply to an action).
 */
export function Callout({ tone = "warn", title, children, testId, live }: {
  tone?: "warn" | "bad" | "neutral";
  title?: ReactNode;
  children: ReactNode;
  testId?: string;
  live?: boolean;
}) {
  const announce = live ?? tone !== "neutral";
  return (
    <div
      className={`callout${tone === "neutral" ? "" : ` callout--${tone}`}`}
      role={announce ? "alert" : undefined}
      data-testid={testId}
    >
      <span className="callout__icon" aria-hidden><IconAlert /></span>
      <div>
        {title != null && <div className="callout__title">{title}</div>}
        <div>{children}</div>
      </div>
    </div>
  );
}
