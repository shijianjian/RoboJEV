/**
 * Where a decision comes from.
 *
 * Two answers, one interface. Today it is a recorded bundle on disk ({@link ReplaySource}) and
 * the page is a replay; later it is meant to be a `robojev console` on localhost streaming the
 * same objects over a WebSocket as the arm moves ({@link LiveSource}). The panel, the timeline
 * and the state text must not be able to tell the difference, so they are written against
 * `EpisodeSource` and never against `fetch`.
 *
 * The live half is a **stub on purpose**: its message types are written down in `PROTOCOL.md` and
 * `connect()` refuses with the reason, rather than a half-built client nobody has run against a
 * server nobody has written. What decides which one is used is `liveUrl()` -- `VITE_LIVE_URL` at
 * build time, or `?live=ws://…` at run time, so a static build dropped on Pages stays a replay
 * and the same files pointed at a console become a console.
 */
import { validateEpisode } from "./schema";
import type { Decision, Episode, EpisodeIndexEntry } from "./types";

export interface EpisodeSource {
  readonly kind: "replay" | "live";
  /** The bundles this source can show. A live source has one: the episode in progress. */
  list(): Promise<EpisodeIndexEntry[]>;
  /** Everything about one episode. For a live source this resolves once the episode has begun
   *  and then grows -- which is what `subscribe` is for. */
  load(id: string): Promise<Episode>;
  /** New decisions as they arrive. Returns an unsubscribe function. A replay never calls back:
   *  its episode is complete the moment it is loaded. */
  subscribe(id: string, onDecision: (d: Decision) => void): () => void;
}

/** Where the bundles live, relative to the page. Relative on purpose: the site is served from
 *  `/`, from `/RoboJEV/`, and from a `file://`-ish sub-path during verification, and all three
 *  have to resolve the same. */
const REPLAY_ROOT = "replays";

async function getJson(url: string): Promise<unknown> {
  const response = await fetch(url, { cache: "no-cache" });
  if (!response.ok) throw new Error(`${url}: ${response.status} ${response.statusText}`);
  return (await response.json()) as unknown;
}

export class ReplaySource implements EpisodeSource {
  readonly kind = "replay" as const;
  private readonly root: string;
  private readonly cache = new Map<string, Promise<Episode>>();

  constructor(root: string = REPLAY_ROOT) {
    this.root = root.replace(/\/$/, "");
  }

  /** The directory a bundle's media sits in, for the `<video src>`. */
  mediaUrl(id: string, path: string): string {
    return `${this.root}/${id}/${path}`;
  }

  async list(): Promise<EpisodeIndexEntry[]> {
    const raw = await getJson(`${this.root}/index.json`);
    if (!Array.isArray(raw)) throw new Error("replays/index.json: expected an array of bundles");
    return raw as EpisodeIndexEntry[];
  }

  load(id: string): Promise<Episode> {
    const hit = this.cache.get(id);
    if (hit !== undefined) return hit;
    const pending = (async () => {
      const raw = await getJson(`${this.root}/${id}/episode.json`);
      const checked = validateEpisode(raw);
      if (!checked.ok) {
        throw new Error(`${id}/episode.json is not a bundle this page can read:\n- ${checked.problems.join("\n- ")}`);
      }
      return checked.episode;
    })();
    this.cache.set(id, pending);
    return pending;
  }

  subscribe(): () => void {
    return () => {};
  }
}

/**
 * The live console, not built.
 *
 * Everything it needs from a server is in `PROTOCOL.md`: a WebSocket that sends one `hello` with
 * the episode's header, then one `decision` message per decision carrying exactly the object
 * `recorder.py` writes into `episode.json`, then `done`. Frames come from an MJPEG or WebRTC
 * stream named in `hello`, because a browser cannot decode a 20 Hz mp4 that is still being
 * written. Until that server exists this class states the contract and refuses.
 */
export class LiveSource implements EpisodeSource {
  readonly kind = "live" as const;
  readonly url: string;

  constructor(url: string) {
    this.url = url;
  }

  private refuse(): never {
    throw new Error(
      `live source ${this.url}: the console server is not built yet. The page speaks the protocol ` +
      `in PROTOCOL.md; nothing serves it. Drop the ?live= parameter to read the recorded bundles.`,
    );
  }

  list(): Promise<EpisodeIndexEntry[]> {
    return Promise.reject(new Error(`live source ${this.url}: not implemented (see PROTOCOL.md)`));
  }

  load(): Promise<Episode> {
    return Promise.reject(new Error(`live source ${this.url}: not implemented (see PROTOCOL.md)`));
  }

  subscribe(): () => void {
    this.refuse();
  }
}

/** The live console's address, or null for the ordinary static site. `?live=` wins over the
 *  build-time default so one deployed build can be pointed at a console without rebuilding. */
export function liveUrl(search: string = typeof location === "undefined" ? "" : location.search): string | null {
  const fromQuery = new URLSearchParams(search).get("live");
  if (fromQuery !== null && fromQuery !== "") return fromQuery;
  const fromEnv = import.meta.env?.VITE_LIVE_URL;
  return typeof fromEnv === "string" && fromEnv !== "" ? fromEnv : null;
}

/** The source this page run uses. */
export function makeSource(): EpisodeSource {
  const live = liveUrl();
  return live === null ? new ReplaySource() : new LiveSource(live);
}
