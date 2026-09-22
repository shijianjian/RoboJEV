/**
 * The Dataset tab: the chosen task, driven live, in the Runs page's layout.
 *
 * The MuJoCo scene is the main view, posed from the console's `pose` stream (and, before an
 * episode, at the task's start state wherever a recorded run begins from it); the console's two
 * camera renders are stacked on its right; the decision grid under them updates as the model
 * answers. The runs recorded for the task are listed on the right, newest first.
 *
 * The controls are one bar along the bottom: the weights, and Start - which runs the episode to its
 * end - or Stop. An episode that ends, or is stopped, is saved as a run by itself and appears at the
 * top of that list. A candidate bar is still a control while it runs (robopp's assisted override):
 * clicking one holds that answer for the next decision, drawn only when the bar is hovered.
 * Without a console (the static site) the bar is one label saying a console is needed.
 */
import { lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";
import { commandsEnabled, type LivePose, type LiveSceneRef, type LiveSource, type LiveState, type LiveWeights } from "../data/live";
import type { ReplaySource } from "../data/source";
import type { CatalogueTask, EpisodeIndexEntry } from "../data/types";
import { DecisionGrid } from "../replay/DecisionGrid";
import { Callout } from "../ui/Callout";
import { Card, CardBody } from "../ui/Card";
import { Chip, OutcomeChip, type Tone } from "../ui/Chip";
import { fetchQpos } from "../viewer/record";
import { suiteLabel } from "./labels";
import { LiveFrame } from "./LiveFrame";

const SceneOnly = lazy(() => import("../viewer/SceneOnly").then((m) => ({ default: m.SceneOnly })));

/** The scene the sidebar has chosen. */
export interface SceneChoice { suite: string; task: number; init: number }

/** One recorded run of the chosen task, for the list on the right. */
export interface TaskRun { id: string; label: string; success: boolean; poster: string | null }

/** The one-word status: waiting (nothing running), running, done or failed (the last episode). */
function statusWord(state: LiveState, last: "done" | "failed" | null): { word: string; tone: Tone } {
  if (state.done !== null) return state.done.success ? { word: "done", tone: "success" } : { word: "failed", tone: "failure" };
  if (state.run === "error") return { word: "failed", tone: "failure" };
  if (state.episode || state.run === "starting") return { word: "running", tone: "running" };
  if (last !== null) return last === "done" ? { word: "done", tone: "success" } : { word: "failed", tone: "failure" };
  return { word: "waiting", tone: "neutral" };
}

/** What the weights picker offers: the console's `weights`, or - from a console older than the
 *  picker - one entry per engine it serves. */
function weightsOf(state: LiveState): LiveWeights[] {
  const offered = state.config?.weights;
  if (offered !== undefined) return offered;
  return (state.config?.policies ?? []).map((p) => ({ id: p, label: p, revision: null, policy: p, checkpoint: null }));
}

export function Inference({ live, state, entries, source, choice, onChoice, instruction, onSaved, taskRuns, task: catalogued }: {
  live: LiveSource | null;
  state: LiveState;
  entries: EpisodeIndexEntry[];
  source: ReplaySource;
  choice: SceneChoice;
  onChoice: (next: SceneChoice) => void;
  /** The chosen task's sentence, where it is known. */
  instruction: string | null;
  /** A bundle was written: the index should be read again. */
  onSaved: () => void;
  taskRuns: TaskRun[];
  /** The chosen task's catalogue entry: its compiled scene and its start states' poses. */
  task: CatalogueTask | null;
}) {
  const { suite, task, init } = choice;
  const header = state.header;
  const weights = weightsOf(state);
  const [weightId, setWeightId] = useState<string | null>(null);
  const weight = weights.find((w) => w.id === weightId) ?? weights[0] ?? null;
  const [last, setLast] = useState<"done" | "failed" | null>(null);

  const saved = state.saved?.id ?? null;
  useEffect(() => { if (saved !== null) onSaved(); }, [saved, onSaved]);

  const can = commandsEnabled(state);
  const maxSteps = header?.max_steps ?? 220;
  const decisions = state.decisions;
  const decision = decisions.length > 0 ? decisions[decisions.length - 1] : null;
  const overridable = state.episode && state.connection === "open" && state.done === null
    && (state.run === "paused" || state.run === "waiting" || state.run === "running");

  // Start runs the episode to its end: once the settling steps are done and the console reports
  // the episode paused, it is told to run. And an episode that has ended - by itself, or by Stop -
  // is saved and let go of, which is what puts it in the list of runs and "Start" back on the bar.
  const autorun = useRef<string | null>(null);
  const closing = useRef<string | null>(null);
  useEffect(() => {
    if (live === null || header === null || !state.episode) return;
    if (autorun.current === "pending" && state.run === "paused" && state.done === null) {
      autorun.current = header.id;
      live.run();
    }
    if ((state.done !== null || state.run === "error") && closing.current !== header.id) {
      closing.current = header.id;
      setLast(state.done?.success ? "done" : "failed");
      live.save();
      live.reset();
    }
  }, [live, header, state.episode, state.run, state.done]);

  const onStart = useCallback(() => {
    if (live === null || weight === null) return;
    autorun.current = "pending";
    live.start({ suite, task, init, policy: weight.policy, checkpoint: weight.checkpoint, selection: "argmax", temperature: null });
  }, [live, weight, suite, task, init]);
  // Stop keeps what was run: the episode so far is saved as a run, then let go of.
  const onStop = useCallback(() => {
    if (live === null || header === null) return;
    closing.current = header.id;
    setLast(null);
    live.save();
    live.reset();
  }, [live, header]);
  const onOverride = useCallback((qid: string, candidate: string) => {
    live?.override(qid, state.overrides[qid] === candidate ? null : candidate);
  }, [live, state.overrides]);

  // ------------------------------------------------------------------ the start state

  // Before an episode, the chosen task's scene at the chosen start state: robopp's catalogue
  // preview - the compiled scene posed from `init_qpos.json`, with no simulator. Without a
  // catalogue entry, a recorded run's first frame of the same start state stands in.
  const startEntry = entries.find((e) => e.suite === suite && e.task_index === task && e.init_state_index === init) ?? null;
  const [startScene, setStartScene] = useState<{ scene: LiveSceneRef; pose: LivePose } | null>(null);
  useEffect(() => {
    let alive = true;
    const abs = (path: string) => new URL(path, document.baseURI).href;
    if (catalogued !== null && catalogued.scene_bundle !== null) {
      const hash = catalogued.scene_bundle;
      fetch(`catalogue/${suite}/${task}/init_qpos.json`)
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
        .then((table: { nq: number; qpos: number[][] }) => {
          if (!alive) return;
          setStartScene({
            scene: { hash, xml: abs(`scenes/${hash}/scene.xml`), assets: abs(`scenes/${hash}/assets/`), nq: table.nq },
            pose: { step: 0, qpos: table.qpos[Math.min(init, table.qpos.length - 1)] },
          });
        })
        .catch(() => { if (alive) setStartScene(null); });
      return () => { alive = false; };
    }
    if (startEntry === null) { setStartScene(null); return; }
    source.load(startEntry.id).then(async (ep) => {
      const f = source.files(ep);
      if (f.scene === null || f.qpos === null) { if (alive) setStartScene(null); return; }
      const table = await fetchQpos(f.qpos.url, f.qpos.spec);
      if (!alive) return;
      setStartScene({
        scene: { hash: ep.scene!.hash, xml: f.scene.xml, assets: f.scene.assets_base, nq: f.qpos.spec.nq },
        pose: { step: 0, qpos: Array.from(table.slice(0, f.qpos.spec.nq)) },
      });
    }).catch(() => { if (alive) setStartScene(null); });
    return () => { alive = false; };
  }, [catalogued, suite, task, init, startEntry, source]);

  const elsewhere = header !== null && state.episode && (header.suite !== suite || header.task_index !== task);
  const sentence = header !== null && state.episode && !elsewhere ? header.instruction : instruction;
  const scene = state.episode ? state.scene : startScene?.scene ?? null;
  const pose = state.episode ? state.pose : startScene?.pose ?? null;
  const cameras = state.episode && live !== null ? { agentview: live.frameUrl("agentview"), wrist: live.frameUrl("wrist") } : null;
  const status = statusWord(state, last);
  const open = state.episode || state.run === "starting";

  return (
    <>
      <div className="run-tab__main">
        <div className="rj-page">
          <div className="rj-episode-title">
            <h1 data-testid="task-instruction">{sentence ?? `task ${task}`}</h1>
            <Chip mono>{suite}</Chip>
            <Chip mono>task {task} · init {init}</Chip>
            {state.done !== null && <OutcomeChip success={state.done.success} status="done" testId="live-outcome" />}
            {header !== null && state.episode && <Chip mono testId="live-readout">step {state.step ?? 0}/{maxSteps}</Chip>}
          </div>
          {elsewhere && header !== null && (
            <Callout tone="warn" title="The session is on another scene" testId="session-elsewhere">
              {suiteLabel(header.suite)} task <span className="u-num">{header.task_index}</span>.{" "}
              <button type="button" className="btn btn--sm" data-testid="session-back"
                      onClick={() => onChoice({ suite: header.suite, task: header.task_index, init: header.init_state_index })}>
                Back to it
              </button>
            </Callout>
          )}

          <div className="rj-view">
            <div className="rj-view__main">
              <div className="rj-scene" data-testid="live-scene">
                {scene !== null ? (
                  <Suspense fallback={<div className="skeleton scene" />}>
                    <SceneOnly xml={scene.xml} assetsBase={scene.assets} qpos={pose?.qpos} heavy={false} />
                  </Suspense>
                ) : (
                  <div className="scene-wrap"><div className="scene" data-testid="live-no-scene" /></div>
                )}
              </div>
            </div>
            <div className="rj-view__cams">
              {(["agentview", "wrist"] as const).map((camera) => (
                <figure className="cam" key={camera}>
                  <div className="cam__frame">
                    {cameras !== null
                      ? <LiveFrame url={cameras[camera]} alt={`${camera}, live`} testId={`live-${camera}`} className="rj-liveframe" />
                      : camera === "agentview" && catalogued !== null && init === 0
                        ? <img className="rj-liveframe" src={`catalogue/${suite}/${task}/thumb.png`} alt="agentview at start state 0" data-testid="start-thumb" />
                        : <div className="rj-liveframe skeleton" />}
                  </div>
                  <figcaption><span>{camera}</span><span className="u-dim">camera</span></figcaption>
                </figure>
              ))}
            </div>
          </div>

          <Card className="rj-decision-card">
            <CardBody pad="tight">
              <div data-testid="decision-card">
                {decision === null ? (
                  <p className="u-dim" data-testid="decision-empty">{state.episode ? "—" : "no session"}</p>
                ) : (
                  <DecisionGrid
                    decision={decision}
                    hot={null}
                    onHot={() => {}}
                    armed={overridable ? state.overrides : {}}
                    onOverride={overridable ? onOverride : undefined}
                  />
                )}
              </div>
            </CardBody>
          </Card>
        </div>
      </div>

      <aside className="run-tab__panel rj-taskruns" aria-label="runs of this task">
        <div className="sidenav__heading">Runs</div>
        {taskRuns.length === 0 ? (
          <p className="u-dim rj-taskruns__empty" data-testid="dataset-runs-empty">none recorded</p>
        ) : (
          <ul className="rj-taskruns__list" data-testid="dataset-runs">
            {taskRuns.map((run) => (
              <li key={run.id}>
                <a className="rj-taskrun" href={`#/${run.id}`} data-testid={`dataset-run-${run.id}`} title={run.label}>
                  {run.poster !== null ? <img className="replay-task__thumb" src={run.poster} alt="" /> : <span className="replay-task__thumb replay-task__thumb--none" />}
                  <span className="rj-taskrun__label">{run.label}</span>
                  <OutcomeChip success={run.success} status="done" />
                </a>
              </li>
            ))}
          </ul>
        )}
      </aside>

      <div className="rj-dock card" data-testid="dock">
        {live === null ? (
          <p className="u-dim" data-testid="session-needs-console">needs a local <code>robojev console</code></p>
        ) : (
          <>
            <div className="rj-dock__row" role="group" aria-label="the session">
              <label className="field">
                <span className="field__label">weights</span>
                <select className="select" value={weight?.id ?? ""} disabled={open || weights.length === 0}
                        onChange={(e) => setWeightId(e.target.value)} data-testid="session-weights">
                  {weights.length === 0 && <option value="">none</option>}
                  {weights.map((w) => (
                    <option key={w.id} value={w.id}>{w.label}{w.revision ? ` · ${w.revision}` : ""}</option>
                  ))}
                </select>
              </label>
              {open ? (
                <button type="button" className="btn" disabled={!can.reset || closing.current === header?.id}
                        onClick={onStop} data-testid="session-stop">Stop</button>
              ) : (
                <button type="button" className="btn btn--primary" disabled={!can.start || weight === null}
                        onClick={onStart} data-testid="session-start">Start</button>
              )}
              {state.connection === "open" ? (
                <Chip tone={status.tone} dot testId="session-status">{status.word}</Chip>
              ) : (
                // Said only when it is a problem: the socket is down, and a click tries again now.
                <button type="button" className="chip chip--failure rj-disconnected" onClick={() => live.connect()}
                        title="the console cannot be reached; click to try again" data-testid="disconnected">
                  <span className="chip__dot" aria-hidden />disconnected
                </button>
              )}
            </div>
            {state.error !== null && <p className="footnote u-bad" data-testid="session-error">{state.error}</p>}
          </>
        )}
      </div>
    </>
  );
}
