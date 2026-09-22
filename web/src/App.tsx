/**
 * Two pages behind one sidebar: **Runs**, the replay page (a recorded episode: its 3D scene, its
 * videos and its decisions), and **Dataset**, the inference page (robopp's RoboJEV tab: pick a
 * task and a start state and drive a policy through `robojev console`).
 *
 * Routing is the hash, because GitHub Pages cannot rewrite `/RoboJEV/drawer` back to
 * `index.html`: `#/dataset` is the inference page, `#/<id>` an episode (`?t=<frame>` opens it at a
 * frame), and an empty hash opens the drawer episode. The GitHub Pages build (`VITE_SITE=pages`)
 * has no Dataset tab at all - it needs a console - and sends `#/dataset` to the Runs tab; a local
 * build keeps both, and without a console its Dataset tab says one is needed.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { EMPTY_LIVE, LIVE_ID, type LiveState } from "./data/live";
import { DEFAULT_EPISODE, episodeFor, frameOf, hashFor, routeOf } from "./data/route";
import { shortLabel } from "./data/lookup";
import { makeLive, ReplaySource } from "./data/source";
import type { CatalogueSuite, Episode, EpisodeIndexEntry } from "./data/types";
import { Replay } from "./replay/Replay";
import { Inference, type SceneChoice } from "./robojev/Inference";
import { DEFAULT_INIT_STATES, policyName, suiteLabel } from "./robojev/labels";
import { Sidebar, type RailRun } from "./robojev/Sidebar";
import { Callout } from "./ui/Callout";
import { Shell } from "./ui/Shell";
import showcase from "../showcase.json";

const source = new ReplaySource();
const live = makeLive();

/** The GitHub Pages build: the Runs tab and nothing that needs a console. Decided when the site is
 *  built (`npm run build:pages`), never by looking for a console. */
const PAGES = import.meta.env.VITE_SITE === "pages";

/** The inference page's address. `#/live`, what the console's page used to open on, still works. */
const DATASET = "dataset";
const DATASET_ALIASES = [DATASET, LIVE_ID];

/** Below this the sidebar is a drawer over the page: robopp's `NARROW`. */
const NARROW = "(max-width: 999px)";

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function App() {
  const [hash, setHash] = useState(() => (typeof location === "undefined" ? "" : location.hash));
  const [entries, setEntries] = useState<EpisodeIndexEntry[] | null>(null);
  const [listError, setListError] = useState<unknown>(null);
  const [state, setState] = useState<LiveState>(EMPTY_LIVE);
  const [choice, setChoice] = useState<SceneChoice>({ suite: "libero_spatial", task: 0, init: 0 });

  useEffect(() => {
    const onHash = () => setHash(location.hash);
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  // The task catalogue: what the Dataset tab lists, thumbnails and start states included. Served
  // from the repository's `data/` by the console, and copied into a local build.
  const [catalogue, setCatalogue] = useState<CatalogueSuite[]>([]);
  useEffect(() => {
    if (PAGES) return;
    fetch("catalogue/index.json", { cache: "no-cache" })
      .then((r) => (r.ok ? r.json() : []))
      .then((rows: unknown) => { if (Array.isArray(rows)) setCatalogue(rows as CatalogueSuite[]); })
      .catch(() => {});
  }, []);

  const readIndex = useCallback(() => {
    source.refresh();
    source.list().then(setEntries, setListError);
  }, []);
  useEffect(readIndex, [readIndex]);

  useEffect(() => {
    if (live === null) return;
    const off = live.subscribeState(setState);
    live.connect();
    return () => { off(); live.close(); };
  }, []);

  // The console's own defaults, once; and a new episode puts the sidebar on its own scene.
  const [seeded, setSeeded] = useState(false);
  if (!seeded && state.config !== null) {
    setSeeded(true);
    setChoice({ suite: state.config.default.suite, task: state.config.default.task, init: state.config.default.init });
  }
  const [adopted, setAdopted] = useState<string | null>(null);
  if (state.header !== null && adopted !== state.header.id) {
    setAdopted(state.header.id);
    setChoice({ suite: state.header.suite, task: state.header.task_index, init: state.header.init_state_index });
  }
  // The task sentences, for whichever suite the sidebar is on.
  const [asked, setAsked] = useState("");
  useEffect(() => {
    if (live === null || state.connection !== "open" || asked === choice.suite) return;
    setAsked(choice.suite);
    live.askTasks(choice.suite);
  }, [state.connection, choice.suite, asked]);

  const known = useMemo(() => [...(PAGES ? [] : DATASET_ALIASES), ...(entries ?? []).map((e) => e.id)], [entries]);
  const route = episodeFor(hash, known, DEFAULT_EPISODE);
  // On Pages an address naming the Dataset tab is rewritten to the run it falls back to.
  useEffect(() => {
    const named = routeOf(hash);
    if (!PAGES || named === null || !DATASET_ALIASES.includes(named)) return;
    const to = hashFor(DEFAULT_EPISODE);
    history.replaceState(null, "", `${location.pathname}${location.search}${to}`);
    setHash(to);
  }, [hash]);
  const tab = DATASET_ALIASES.includes(route) ? "dataset" : "runs";
  const [lastRun, setLastRun] = useState(DEFAULT_EPISODE);
  if (tab === "runs" && lastRun !== route) setLastRun(route);

  const [sideOpen, setSideOpen] = useState(true);
  useEffect(() => {
    const mq = window.matchMedia(NARROW);
    const apply = () => setSideOpen(!mq.matches);
    apply();
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, []);

  // ---------------------------------------------------------------- the sidebar's two lists

  const list = useMemo(() => entries ?? [], [entries]);
  const catalogued = catalogue.find((c) => c.suite === choice.suite) ?? null;
  const tasks = useMemo(() => {
    if (catalogued !== null) {
      return catalogued.tasks.map((t) => ({
        taskIndex: t.task_index, instruction: t.instruction,
        thumbnail: `catalogue/${t.suite}/${t.task_index}/${t.thumbnail ?? "thumb.png"}`,
      }));
    }
    const byTask = new Map<number, EpisodeIndexEntry>();
    for (const e of list) if (e.suite === choice.suite && !byTask.has(e.task_index)) byTask.set(e.task_index, e);
    const rows = state.tasks ?? Array.from({ length: 10 }, (_, i) => ({
      index: i, instruction: byTask.get(i)?.instruction ?? null, init_states: DEFAULT_INIT_STATES,
    }));
    return rows.map((t) => {
      const e = byTask.get(t.index);
      return { taskIndex: t.index, instruction: t.instruction, thumbnail: e?.poster != null ? source.mediaUrl(e.id, e.poster) : null };
    });
  }, [catalogued, state.tasks, list, choice.suite]);
  // The datasets: the catalogue's suites with their real task counts; without a catalogue, the
  // console's suites or the suites the runs on disk are from.
  const families = [{
    simulator: "libero",
    heading: "LIBERO",
    suites: catalogue.length > 0
      ? catalogue.map((c) => ({ id: c.suite, displayName: suiteLabel(c.suite), nTasks: c.tasks.length }))
      : (state.config?.suites ?? [...new Set([choice.suite, ...list.map((e) => e.suite)])]).map((id) => ({
          id, displayName: suiteLabel(id), nTasks: id === choice.suite ? tasks.length : 10,
        })),
  }];

  const runs: RailRun[] = useMemo(() => {
    // Recorded runs only: a live episode is the Dataset tab's, and is a run once it is saved.
    const out: RailRun[] = [];
    for (const e of list) {
      out.push({
        id: e.id, status: "done", policyLabel: shortLabel(e.instruction), policy: policyName(e.policy),
        initStateIndex: e.init_state_index,
        success: e.success, steps: e.decisions, family: "decision", href: hashFor(e.id),
        suite: e.suite, suiteLabel: suiteLabel(e.suite), taskIndex: e.task_index,
      });
    }
    return out;
  }, [list]);

  const go = (target: string) => { location.hash = target; };
  const onSelectRun = (id: string) => go(hashFor(id));
  const onGoToScene = (suite: string, task: number, init: number) => {
    setChoice({ suite, task, init });
    go(`#/${DATASET}`);
  };


  return (
    <Shell repository={showcase.repository}>
      {listError !== null && (
        <Callout tone="bad" title="The episode index could not be loaded" testId="load-error">{message(listError)}</Callout>
      )}
      <div className={`run-tab run-tab--app run-tab--${tab === "runs" ? "replay" : "dataset"}`}>
        <details className="run-tab__side" open={sideOpen} onToggle={(e) => setSideOpen(e.currentTarget.open)} data-testid="run-side">
          <summary className="run-tab__side-summary">{tab === "dataset" ? "Dataset" : "Runs"}</summary>
          <Sidebar
            tab={tab}
            tabs={PAGES ? ["runs"] : ["runs", "dataset"]}
            datasetHref={`#/${DATASET}`}
            runsHref={hashFor(lastRun)}
            families={families}
            suite={choice.suite}
            suiteLabel={suiteLabel(choice.suite)}
            tasks={tasks}
            taskIndex={choice.task}
            onSuite={(suite) => setChoice({ suite, task: 0, init: 0 })}
            // Without a console there is no start-state picker, so a task opens on the start state
            // its recording began from - the one start state this page can show.
            onTask={(task) => setChoice({
              ...choice, task,
              init: live === null ? list.find((e) => e.suite === choice.suite && e.task_index === task)?.init_state_index ?? 0 : 0,
            })}
            runs={runs}
            selectedRun={tab === "runs" ? route : null}
            onSelectRun={onSelectRun}
            onGoToScene={onGoToScene}
          />
        </details>
        {tab === "dataset" ? (
          <Inference
            live={live}
            state={state}
            entries={list}
            source={source}
            choice={choice}
            onChoice={setChoice}
            instruction={tasks.find((t) => t.taskIndex === choice.task)?.instruction ?? null}
            task={catalogued?.tasks.find((t) => t.task_index === choice.task) ?? null}
            onSaved={readIndex}
            // Newest first: a run the console has just saved goes to the top.
            taskRuns={list.filter((e) => e.suite === choice.suite && e.task_index === choice.task)
              .sort((x, y) => Number(y.id.startsWith("live-")) - Number(x.id.startsWith("live-")) || (x.id.startsWith("live-") ? y.id.localeCompare(x.id) : 0))
              .map((e) => ({
              id: e.id, label: shortLabel(e.instruction), success: e.success,
              poster: e.poster != null ? source.mediaUrl(e.id, e.poster) : null,
            }))}
          />
        ) : (
          <div className="run-tab__main">
            {/* Keyed by the frame too: a link to another frame of the same episode is a new page. */}
            <RunPage key={`${route}@${frameOf(hash) ?? 0}`} id={route} frame={frameOf(hash) ?? 0} />
          </div>
        )}
      </div>
    </Shell>
  );
}

function RunPage({ id, frame }: { id: string; frame: number }) {
  const [episode, setEpisode] = useState<Episode | null>(null);
  const [error, setError] = useState<unknown>(null);
  useEffect(() => {
    let alive = true;
    source.load(id).then((ep) => { if (alive) setEpisode(ep); }, (err) => { if (alive) setError(err); });
    return () => { alive = false; };
  }, [id]);
  const files = useMemo(() => (episode === null ? null : source.files(episode)), [episode]);
  if (error !== null) return <Callout tone="bad" title={`Could not load ${id}`} testId="load-error">{message(error)}</Callout>;
  if (episode === null || files === null) return <p className="u-dim" data-testid="loading">Loading…</p>;
  return <Replay episode={episode} files={files} initialTime={frame / episode.control_rate} />;
}
