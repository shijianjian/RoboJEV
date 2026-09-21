/**
 * The whole site is one page: an episode, playing, and a strip to switch between them.
 *
 * Routing is the hash, because GitHub Pages cannot rewrite `/RoboJEV/drawer` back to
 * `index.html` and a deep link that 404s is worse than a `#`. `#/<id>` is the address of an
 * episode; `#/replay/<id>`, which is what the first version of this site handed out, still
 * resolves; and an empty hash is the drawer — the episode with the most to look at.
 *
 * Switching does not reload: the hash changes, the episode is fetched (and thereafter cached by
 * `ReplaySource`), and the `<video>` is rebuilt by its key.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { EpisodeStrip } from "./components/EpisodeStrip";
import { Replay } from "./components/Replay";
import { Chip } from "./components/ui";
import { episodeFor, hashFor } from "./data/route";
import { makeSource, ReplaySource } from "./data/source";
import type { Episode, EpisodeIndexEntry } from "./data/types";

const source = makeSource();

function Failed({ what, error }: { what: string; error: unknown }) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    <div className="notice notice--bad" data-testid="load-error">
      <b>{what} could not be loaded.</b>
      <pre>{message}</pre>
    </div>
  );
}

export function App() {
  const [hash, setHash] = useState(() => (typeof location === "undefined" ? "" : location.hash));
  const [entries, setEntries] = useState<EpisodeIndexEntry[] | null>(null);
  const [listError, setListError] = useState<unknown>(null);
  const [episode, setEpisode] = useState<Episode | null>(null);
  const [episodeError, setEpisodeError] = useState<unknown>(null);

  useEffect(() => {
    const onHash = () => setHash(location.hash);
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    let alive = true;
    source.list().then(
      (list) => { if (alive) setEntries(list); },
      (err) => { if (alive) setListError(err); },
    );
    return () => { alive = false; };
  }, []);

  const known = useMemo(() => (entries ?? []).map((e) => e.id), [entries]);
  const id = episodeFor(hash, known);

  useEffect(() => {
    let alive = true;
    setEpisode(null);
    setEpisodeError(null);
    source.load(id).then(
      (ep) => { if (alive) setEpisode(ep); },
      (err) => { if (alive) setEpisodeError(err); },
    );
    return () => { alive = false; };
  }, [id]);

  const pick = useCallback((next: string) => { location.hash = hashFor(next); }, []);

  // ↑/↓ walk the strip, 1–9 jump straight to one. The replay owns space and ←/→.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target !== null && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;
      const list = entries ?? [];
      if (list.length === 0) return;
      const at = Math.max(0, list.findIndex((entry) => entry.id === id));
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        const delta = e.key === "ArrowDown" ? 1 : -1;
        pick(list[(at + delta + list.length) % list.length].id);
      } else if (/^[1-9]$/.test(e.key)) {
        const wanted = list[Number(e.key) - 1];
        if (wanted !== undefined) { e.preventDefault(); pick(wanted.id); }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [entries, id, pick]);

  const mediaUrl = useCallback(
    (path: string) => (source instanceof ReplaySource ? source.mediaUrl(id, path) : path),
    [id],
  );
  const posterUrl = useCallback(
    (entry: EpisodeIndexEntry) => (
      source instanceof ReplaySource && entry.poster != null
        ? source.mediaUrl(entry.id, entry.poster)
        : null
    ),
    [],
  );

  return (
    <div className="shell">
      <header className="masthead">
        <span className="masthead__name"><a href="#/">RoboJEV</a></span>
        <span className="masthead__right"><Chip mono>{source.kind}</Chip></span>
      </header>

      {listError !== null && <Failed what="The episode index" error={listError} />}
      {entries !== null && entries.length > 0 && (
        <EpisodeStrip entries={entries} current={id} posterUrl={posterUrl} onPick={pick} />
      )}

      {episodeError !== null ? (
        <Failed what={`Bundle “${id}”`} error={episodeError} />
      ) : episode === null ? (
        <p className="muted" data-testid="loading">Loading…</p>
      ) : (
        <Replay episode={episode} mediaUrl={mediaUrl} />
      )}
    </div>
  );
}
