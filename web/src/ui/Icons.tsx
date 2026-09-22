/* Inline SVG icons — no icon library. Every icon is decorative: the control that
   carries it supplies the accessible name. */
const base = { width: 16, height: 16, viewBox: "0 0 16 16", fill: "none", "aria-hidden": true as const, focusable: "false" as const };

export function IconPlay() {
  return <svg {...base}><path d="M5 3.5v9l7.5-4.5z" fill="currentColor" /></svg>;
}

export function IconPause() {
  return <svg {...base}><path d="M5 3h2.2v10H5zM8.8 3H11v10H8.8z" fill="currentColor" /></svg>;
}

export function IconPrev() {
  return <svg {...base}><path d="M11 3.5v9L4.5 8z" fill="currentColor" /><rect x="3" y="3.5" width="1.4" height="9" fill="currentColor" /></svg>;
}

export function IconNext() {
  return <svg {...base}><path d="M5 3.5v9L11.5 8z" fill="currentColor" /><rect x="11.6" y="3.5" width="1.4" height="9" fill="currentColor" /></svg>;
}

export function IconCamera() {
  return (
    <svg {...base}>
      <rect x="1.5" y="4.5" width="13" height="8" rx="1.5" stroke="currentColor" strokeWidth="1.2" />
      <circle cx="8" cy="8.5" r="2.2" stroke="currentColor" strokeWidth="1.2" />
      <path d="M5.5 4.5 6.4 3h3.2l.9 1.5" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" />
    </svg>
  );
}

export function IconAlert() {
  return (
    <svg {...base}>
      <path d="M8 2.2 15 13.8H1z" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" />
      <path d="M8 6.4v3.2" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
      <circle cx="8" cy="11.6" r="0.75" fill="currentColor" />
    </svg>
  );
}

export function IconDownload() {
  return (
    <svg {...base}>
      <path d="M8 2.5v7.5m0 0L5.2 7.2M8 10l2.8-2.8" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M2.8 11.5v1.2c0 .4.3.8.8.8h8.8c.5 0 .8-.4.8-.8v-1.2" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
    </svg>
  );
}

export function IconExternal() {
  return (
    <svg {...base} width={12} height={12} viewBox="0 0 12 12">
      <path d="M4.5 2h5.5v5.5M10 2 5 7" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M8 8.5V10H2V4h1.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

/** GitHub's mark, inline: no icon font and no request for it. */
export function IconGitHub() {
  return (
    <svg {...base} viewBox="0 0 16 16">
      <path fill="currentColor" d="M8 0a8 8 0 0 0-2.53 15.59c.4.07.55-.17.55-.38v-1.33c-2.23.48-2.7-1.07-2.7-1.07-.36-.92-.89-1.17-.89-1.17-.73-.5.06-.49.06-.49.8.06 1.23.83 1.23.83.72 1.22 1.88.87 2.34.66.07-.52.28-.87.5-1.07-1.78-.2-3.65-.89-3.65-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.6 7.6 0 0 1 4 0c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.28.82 2.15 0 3.07-1.87 3.75-3.66 3.95.29.25.54.73.54 1.48v2.2c0 .21.15.46.55.38A8 8 0 0 0 8 0Z" />
    </svg>
  );
}
