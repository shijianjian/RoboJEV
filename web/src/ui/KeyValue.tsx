import { Fragment, type ReactNode } from "react";

export interface KV { k: string; v: ReactNode; mono?: boolean; title?: string }

export function KeyValue({ items, rule = false }: { items: KV[]; rule?: boolean }) {
  return (
    <dl className={`kv${rule ? " kv--rule" : ""}`}>
      {items.map((it) => (
        <Fragment key={it.k}>
          <dt className="kv__k">{it.k}</dt>
          <dd className={`kv__v${it.mono ? " kv__v--mono" : ""}`} title={it.title}>{it.v}</dd>
        </Fragment>
      ))}
    </dl>
  );
}
