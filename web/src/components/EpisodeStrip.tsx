/**
 * The episode switcher: one item per bundle, across the top.
 *
 * A poster, a short label and the outcome — everything else about an episode is a click away, in
 * the episode itself. The poster is a still from the bundle's own video, taken a little past the
 * middle (`recorder.py#poster_time`), so the four items are four different pictures of four
 * different scenes rather than four identical untouched tables.
 *
 * Each item is an `<a href="#/<id>">`, so a middle click opens a tab and the browser's own
 * history works; the click handler only stops the default navigation so switching does not
 * reload.
 *
 * The console, when there is one, is the first item and the only one that is not a recording: it
 * has no poster (nothing has been rendered) and no outcome (nothing has happened), so it says
 * "live" where the others say success or failure. On the static deployment there is no such item
 * and this strip is exactly the four it always was.
 */
import type { EpisodeIndexEntry } from "../data/types";
import { LIVE_ID } from "../data/live";
import { shortLabel } from "../data/lookup";
import { hashFor } from "../data/route";
import { Chip, OutcomeChip } from "./ui";

export function EpisodeStrip({ entries, current, posterUrl, onPick }: {
  entries: EpisodeIndexEntry[];
  current: string;
  posterUrl: (entry: EpisodeIndexEntry) => string | null;
  onPick: (id: string) => void;
}) {
  return (
    <nav className="strip" aria-label="episodes" data-testid="episode-strip">
      {entries.map((entry, i) => {
        const poster = posterUrl(entry);
        const active = entry.id === current;
        return (
          <a
            key={entry.id}
            className={`strip__item${active ? " strip__item--on" : ""}`}
            href={hashFor(entry.id)}
            aria-current={active ? "true" : undefined}
            title={`${entry.instruction} — ${entry.success ? "success" : "failure"}, ${entry.decisions} of ${entry.max_decisions} decisions (key ${i + 1})`}
            data-testid={`strip-${entry.id}`}
            onClick={(e) => {
              if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
              e.preventDefault();
              onPick(entry.id);
            }}
          >
            {poster !== null && (
              <img className="strip__poster" src={poster} alt="" width={96} height={96} loading="lazy" />
            )}
            <span className="strip__body">
              <span className="strip__label">{shortLabel(entry.instruction)}</span>
              <span className="strip__meta">
                {entry.id === LIVE_ID ? (
                  <Chip tone="accent" testId="strip-outcome-live">live</Chip>
                ) : (
                  <OutcomeChip success={entry.success} testId={`strip-outcome-${entry.id}`} />
                )}
                {entry.max_decisions > 0 && (
                  <span className="num dim">{entry.decisions}/{entry.max_decisions}</span>
                )}
              </span>
            </span>
          </a>
        );
      })}
    </nav>
  );
}
