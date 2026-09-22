import type { ReactNode } from "react";

export function Card({ children, className = "", flush = false, id }: { children: ReactNode; className?: string; flush?: boolean; id?: string }) {
  return <section id={id} className={`card${flush ? " card--flush" : ""}${className ? " " + className : ""}`}>{children}</section>;
}

export function CardHead({ title, sub, actions }: { title: ReactNode; sub?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="card__head">
      <span className="card__title">{title}</span>
      {sub != null && <span className="card__sub">{sub}</span>}
      {actions != null && <span className="card__actions">{actions}</span>}
    </div>
  );
}

export function CardBody({ children, pad = "normal", className = "" }: { children: ReactNode; pad?: "normal" | "tight" | "none"; className?: string }) {
  const mod = pad === "tight" ? " card__body--tight" : pad === "none" ? " card__body--none" : "";
  return <div className={`card__body${mod}${className ? " " + className : ""}`}>{children}</div>;
}
