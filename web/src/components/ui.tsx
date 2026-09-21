/** The three primitives every card on this site is made of. Kept tiny and local: a UI kit would
 *  be more code than the site. */
import type { ReactNode } from "react";

export function Card({ title, sub, actions, children, bodyClass = "card__body", testId }: {
  title?: ReactNode;
  sub?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  bodyClass?: string;
  testId?: string;
}) {
  return (
    <section className="card" data-testid={testId}>
      {(title !== undefined || sub !== undefined) && (
        <header className="card__head">
          {title !== undefined && <h2 className="card__title">{title}</h2>}
          {sub !== undefined && <span className="card__sub">{sub}</span>}
          {actions !== undefined && <span style={{ marginLeft: "auto" }}>{actions}</span>}
        </header>
      )}
      <div className={bodyClass}>{children}</div>
    </section>
  );
}

export function Chip({ tone = "neutral", mono = false, children, testId }: {
  tone?: "neutral" | "success" | "failure" | "warn" | "accent";
  mono?: boolean;
  children: ReactNode;
  testId?: string;
}) {
  const cls = ["chip", tone === "neutral" ? "" : `chip--${tone}`, mono ? "chip--mono" : ""]
    .filter(Boolean)
    .join(" ");
  return <span className={cls} data-testid={testId}>{children}</span>;
}

/** success / failure, said in words as well as in colour. */
export function OutcomeChip({ success, testId }: { success: boolean; testId?: string }) {
  return (
    <Chip tone={success ? "success" : "failure"} testId={testId}>
      {success ? "success" : "failure"}
    </Chip>
  );
}
