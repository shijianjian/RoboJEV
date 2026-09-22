/**
 * The address of an episode.
 *
 * The hash, because GitHub Pages cannot rewrite `/RoboJEV/drawer` back to `index.html` and a deep
 * link that 404s is worse than a `#`. `#/<id>` is today's shape; `#/replay/<id>` is the one the
 * first version of this site handed out and still has to resolve, because links do not get a
 * chance to be updated. Nothing at all is the default episode.
 */

/** The episode that opens when the URL names none: the drawer, which has the most to look at. */
export const DEFAULT_EPISODE = "drawer";

/** The id in `hash`, or null when it names none. A `?t=<frame>` after it is the playhead. */
export function routeOf(hash: string): string | null {
  const m = /^#\/?(?:replay\/)?([\w.-]+)\/?(?:\?.*)?$/.exec(hash);
  return m === null ? null : m[1];
}

/** The playhead a hash names (`#/drawer?t=72`), or null. robopp's run page keeps it as `?t=`;
 *  here the query string lives inside the hash, because Pages serves one file. */
export function frameOf(hash: string): number | null {
  const q = hash.indexOf("?");
  if (q === -1) return null;
  const raw = new URLSearchParams(hash.slice(q + 1)).get("t");
  if (raw === null || raw.trim() === "") return null;
  const n = Number(raw);
  return Number.isInteger(n) && n >= 0 ? n : null;
}

/** The hash for an episode id — one place, so a link and a `location.hash` cannot disagree.
 *  Frame 0 is the address with no `?t=` at all, as on robopp's run page. */
export function hashFor(id: string, frame = 0): string {
  return frame > 0 ? `#/${id}?t=${frame}` : `#/${id}`;
}

/**
 * Which episode a hash means, given the episodes that exist.
 *
 * An id the index does not know falls back to the default rather than to an error page: the point
 * of the site is that something is playing. An empty `known` means the index has not arrived yet,
 * and then a named id is taken at its word — otherwise the first paint would always be the
 * default and would then jump.
 *
 * `fallback` is what an address that names nothing opens. It is the drawer on the static site and
 * the console on a page a console is serving, which is the one difference between the two
 * deployments' routing: `robojev console` should open on the thing it is a console for.
 */
export function episodeFor(hash: string, known: readonly string[],
                           fallback: string = DEFAULT_EPISODE): string {
  const named = routeOf(hash);
  if (named === null) return fallback;
  if (known.length === 0 || known.includes(named)) return named;
  return fallback;
}
