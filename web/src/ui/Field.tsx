import type { CSSProperties, ReactNode, InputHTMLAttributes, SelectHTMLAttributes } from "react";

/** A labelled control for toolbars and forms. The label is a real <label>, so the
    control keeps an accessible name without an aria-label duplicate. */
export function Field({ label, htmlFor, children, style }: { label: string; htmlFor: string; children: ReactNode; style?: CSSProperties }) {
  return (
    <div className="field" style={style}>
      <label className="field__label" htmlFor={htmlFor}>{label}</label>
      {children}
    </div>
  );
}

export function Select({ className = "", ...rest }: SelectHTMLAttributes<HTMLSelectElement>) {
  return <select className={`select${className ? " " + className : ""}`} {...rest} />;
}

export function Input({ className = "", ...rest }: InputHTMLAttributes<HTMLInputElement>) {
  return <input className={`input${className ? " " + className : ""}`} {...rest} />;
}
