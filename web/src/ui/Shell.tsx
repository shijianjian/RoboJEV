import type { ReactNode } from "react";
import { IconGitHub } from "./Icons";

/**
 * robopp's shell (`app/layout.tsx`): the header bar with the mark, and the page. There
 * is no nav: the app's pages are the sidebar's tabs. Where robopp's header ends in "your runs",
 * this one ends in a GitHub mark. robopp's footer (one sentence and an About link) is not drawn.
 */
export function Shell({ repository, children }: {
  /** The repository's page (`showcase.json`), for the GitHub mark at the header's right end. */
  repository: string;
  children: ReactNode;
}) {
  return (
    <>
      <header className="app-header">
        <div className="shell app-header__inner">
          <a href="#/" className="app-header__brand">
            <span className="app-header__mark" aria-hidden />
            RoboJEV
          </a>
          <div className="app-header__slot">
            <a className="btn btn--icon btn--ghost app-header__github" href={repository} target="_blank"
               rel="noopener noreferrer" title="GitHub" aria-label="GitHub" data-testid="github-link">
              <IconGitHub />
            </a>
          </div>
        </div>
      </header>
      <main className="app-main">
        <div className="shell">{children}</div>
      </main>
    </>
  );
}
