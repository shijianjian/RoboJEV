/**
 * Where a decision comes from.
 *
 * Two answers, one interface. Today it is a recorded bundle on disk ({@link ReplaySource}) and
 * the page is a replay; later it is meant to be a `robojev console` on localhost streaming the
 * same objects over a WebSocket as the arm moves ({@link LiveSource}). The panel, the timeline
 * and the state text must not be able to tell the difference, so they are written against
 * `EpisodeSource` and never against `fetch`.
 *
 * The live half is {@link LiveSource} in `./live.ts`, and what decides whether it is used is
 * {@link liveUrl}: the console injects its own address into the page it serves, `?live=ws://…`
 * overrides that, and `VITE_LIVE_URL` is the build-time default. A static build dropped on Pages
 * has none of the three, makes no probe, and is a replay viewer -- which is the point: the
 * deployed site must not make one request it did not mean to.
 */
import { LiveSource } from "./live";
import { validateEpisode } from "./schema";
import type { Decision, Episode, EpisodeIndexEntry } from "./types";
import type { BundleFiles } from "../viewer/files";

export { LiveSource } from "./live";

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

  /**
   * Every file the viewer reads for one bundle, as absolute URLs: the scene is fetched from the
   * MuJoCo worker, and a worker resolves a relative URL against its own script, not the page.
   * The scene is `scenes/<hash>/` at the site's root: robopp's compiled bundle, served from the
   * repository's `data/scenes` (a build copies the ones its runs use).
   */
  files(episode: Episode): BundleFiles {
    const abs = (path: string) => new URL(path, typeof document === "undefined" ? "http://localhost/" : document.baseURI).href;
    const base = `${this.root}/${episode.id}/`;
    const media: Record<string, string> = {};
    for (const [camera, track] of Object.entries(episode.media)) {
      if (track !== undefined) media[camera] = abs(base + track.path);
    }
    return {
      media,
      poster: episode.poster != null ? abs(base + episode.poster) : null,
      qpos: episode.qpos != null ? { url: abs(base + episode.qpos.path), spec: episode.qpos } : null,
      scene: episode.scene != null
        ? { xml: abs(`scenes/${episode.scene.hash}/scene.xml`), assets_base: abs(`scenes/${episode.scene.hash}/assets/`) }
        : null,
      episodeJson: abs(base + "episode.json"),
    };
  }

  /** Forget what has been read, so a bundle saved by the console is read fresh. */
  refresh(): void {
    this.cache.clear();
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

/** What `robojev console` writes into the page it serves, before the bundle runs. */
export interface ConsoleHint { ws?: unknown }

/**
 * The live console's address, or null for the ordinary static site.
 *
 * Three sources, in the order that lets one build be all three deployments: `?live=` on the URL
 * wins, so a page served from anywhere can be pointed at a console without rebuilding; then the
 * global the console injects into the page it serves itself, which is why `robojev console` alone
 * opens a working console; then `VITE_LIVE_URL`, the build-time default. A GitHub Pages visit has
 * none of them and makes no request looking for one.
 */
export function liveUrl(
  search: string = typeof location === "undefined" ? "" : location.search,
  hint: ConsoleHint | null = typeof window === "undefined"
    ? null
    : ((window as unknown as { __ROBOJEV_CONSOLE__?: ConsoleHint }).__ROBOJEV_CONSOLE__ ?? null),
): string | null {
  const fromQuery = new URLSearchParams(search).get("live");
  if (fromQuery !== null && fromQuery !== "") return fromQuery;
  if (hint !== null && typeof hint.ws === "string" && hint.ws !== "") return hint.ws;
  const fromEnv = import.meta.env?.VITE_LIVE_URL;
  return typeof fromEnv === "string" && fromEnv !== "" ? fromEnv : null;
}

/** The recorded bundles, always: the strip is the four episodes on disk whether or not a console
 *  is attached, and a console adds an item in front of them rather than replacing them. */
export function makeSource(): EpisodeSource {
  return new ReplaySource();
}

/** The console this page run talks to, or null. */
export function makeLive(): LiveSource | null {
  const url = liveUrl();
  return url === null ? null : new LiveSource(url);
}
