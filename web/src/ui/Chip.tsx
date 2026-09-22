import type { ReactNode } from "react";

export type Tone = "neutral" | "success" | "failure" | "running" | "queued" | "warn" | "accent";

export function Chip({ children, tone = "neutral", mono = false, dot = false, large = false, title, testId, className = "" }: {
  children: ReactNode;
  tone?: Tone;
  mono?: boolean;
  dot?: boolean;
  large?: boolean;
  title?: string;
  testId?: string;
  className?: string;
}) {
  return (
    <span
      className={`chip chip--${tone}${mono ? " chip--mono" : ""}${large ? " chip--lg" : ""}${className ? " " + className : ""}`}
      title={title}
      data-testid={testId}
    >
      {dot && <span className="chip__dot" aria-hidden />}
      {children}
    </span>
  );
}

/** One filter chip: the same pill the rest of the site labels things with, but pressable, so
 *  a row of them reads as the states of one choice rather than as a row of buttons. Pressed is
 *  `aria-pressed`, which is also what the styling hangs off - one source for both. */
export function FilterChip({ pressed, onClick, children, testId }: {
  pressed: boolean;
  onClick: () => void;
  children: ReactNode;
  testId: string;
}) {
  return (
    <button
      type="button"
      className="chip chip-filter"
      aria-pressed={pressed}
      onClick={onClick}
      data-testid={testId}
    >
      {children}
    </button>
  );
}

/** Run lifecycle status -> chip tone. Covers the states plan 2b will add. */
export function statusTone(status: string): Tone {
  switch (status) {
    case "done": return "neutral";
    case "running": return "running";
    case "queued": return "queued";
    case "failed": case "error": return "failure";
    default: return "neutral";
  }
}

/** The outcome chip: the benchmark's own success flag, or the lifecycle status if there is none yet. */
export function OutcomeChip({ success, status, large = false, testId }: { success: boolean | null; status: string; large?: boolean; testId?: string }) {
  if (success === null) return <Chip tone={statusTone(status)} dot large={large} testId={testId}>{status}</Chip>;
  return <Chip tone={success ? "success" : "failure"} dot large={large} testId={testId}>{success ? "success" : "failure"}</Chip>;
}
