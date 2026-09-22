/**
 * The Runs page: one recorded episode, with the MuJoCo scene as the main view.
 *
 * The 3D scene (robopp's `SceneView`, posed from the bundle's `qpos.bin`, one row per video frame)
 * takes the middle of the page; the two recorded cameras are stacked on its right; under it one
 * block holds the transport, the decision ticks and the stage bar; the decisions, the state-text
 * toggle and the details row follow.
 *
 * **The video's clock drives everything.** A rAF loop while playing (plus `timeupdate` while
 * paused) sets `time`; `time` picks the decision, the frame the scene is posed at, and the bars.
 * Seeking - a tick, the stage bar, the arrow keys - sets `video.currentTime` and nothing else.
 *
 * **It opens paused on its first frame**, once the scene is in (robopp's loading line is on the
 * stage meanwhile). The transport's **autoplay** toggle - off unless this viewer turned it on, and
 * remembered in this browser - starts it by itself instead, when the scene has compiled, so the arm
 * in 3D and the arm on camera start together; it plays once and stops at the end. A scene that
 * fails to load does not hold the page: the videos are playable, and a chip says the replay is
 * video-only.
 */
import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Episode } from "../data/types";
import { decisionAt, episodeDuration, seekTimeFor, stageSegments } from "../data/lookup";
import { Card, CardBody } from "../ui/Card";
import { Chip, OutcomeChip } from "../ui/Chip";
import { fetchQpos, trajectoryOf, type Trajectory } from "../viewer/record";
import type { BundleFiles } from "../viewer/files";
import { DecisionGrid } from "./DecisionGrid";
import { Details } from "./Details";
import { StageTimeline } from "./StageTimeline";
import { StateText } from "./StateText";
import { Transport } from "./Transport";

const SceneView = lazy(() => import("../viewer/SceneView").then((m) => ({ default: m.SceneView })));

/** Where the autoplay choice is kept: per browser, and only as a convenience. */
const AUTOPLAY_KEY = "robojev.autoplay";

function readAutoplay(): boolean {
  try {
    return window.localStorage.getItem(AUTOPLAY_KEY) === "on";
  } catch {
    return false;
  }
}

function writeAutoplay(on: boolean): void {
  try {
    window.localStorage.setItem(AUTOPLAY_KEY, on ? "on" : "off");
  } catch {
    // No storage (a private window, blocked site data): the toggle still works for this page.
  }
}

/** Where the 3D scene is: `none` for a bundle that carries none (schema 1). */
type SceneGate = "loading" | "ready" | "failed" | "none";

function prefersReducedMotion(): boolean {
  return typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/** The bundle's pose table, as the frame clock the 3D scene reads. */
function useTrajectory(episode: Episode, files: BundleFiles): { traj: Trajectory | null; error: string | null } {
  const [state, setState] = useState<{ traj: Trajectory | null; error: string | null }>({ traj: null, error: null });
  useEffect(() => {
    let alive = true;
    setState({ traj: null, error: null });
    if (files.qpos === null || files.scene === null) return;
    fetchQpos(files.qpos.url, files.qpos.spec).then(
      (table) => { if (alive) setState({ traj: trajectoryOf(episode, table), error: null }); },
      (err) => { if (alive) setState({ traj: null, error: String(err) }); },
    );
    return () => { alive = false; };
  }, [episode, files]);
  return state;
}

export function Replay({ episode, files, initialTime = 0 }: {
  episode: Episode;
  files: BundleFiles;
  /** Where the page opens (`#/<id>?t=<frame>`): a deep link opens there, paused. */
  initialTime?: number;
}) {
  const main = useRef<HTMLVideoElement>(null);
  const wrist = useRef<HTMLVideoElement>(null);
  const [time, setTime] = useState(initialTime);
  const [playing, setPlaying] = useState(false);
  const [rate, setRate] = useState(1);
  const [hot, setHot] = useState<string | null>(null);
  const [showState, setShowState] = useState(false);
  const [metadata, setMetadata] = useState(false);
  const [autoplay, setAutoplay] = useState(readAutoplay);
  const onAutoplay = useCallback(() => {
    setAutoplay((on) => { writeAutoplay(!on); return !on; });
  }, []);
  const { traj, error: poseError } = useTrajectory(episode, files);
  const hasScene = files.scene !== null && files.qpos !== null;
  const [sceneState, setSceneState] = useState<"loading" | "ready" | "failed">("loading");
  const gate: SceneGate = !hasScene ? "none" : poseError !== null ? "failed" : sceneState;

  const duration = episodeDuration(episode);
  const segments = useMemo(() => stageSegments(episode), [episode]);
  const current = decisionAt(episode.decisions, time);
  const decision = current >= 0 ? episode.decisions[current] : null;
  const grounded = useMemo(() => episode.decisions.find((d) => d.grounding !== null) ?? null, [episode]);
  const frame = Math.min(Math.max(Math.floor(time * episode.control_rate + 1e-6), 0), episode.total_frames - 1);

  const seek = useCallback((t: number) => {
    const v = main.current;
    const clamped = Math.max(0, Math.min(t, duration - 1e-3));
    if (v !== null) {
      v.currentTime = clamped;
      if (wrist.current !== null) wrist.current.currentTime = clamped;
    }
    setTime(clamped);
  }, [duration]);

  const stepDecision = useCallback((delta: number) => {
    const next = Math.max(0, Math.min(current + delta, episode.decisions.length - 1));
    seek(seekTimeFor(episode.decisions, next));
  }, [current, episode.decisions, seek]);

  const toggle = useCallback(() => {
    const v = main.current;
    if (v === null) return;
    if (v.paused) void v.play().catch(() => {}); else v.pause();
  }, []);

  const onLoaded = useCallback(() => {
    const v = main.current;
    if (v !== null) v.currentTime = initialTime;
    setMetadata(true);
  }, [initialTime]);

  // The start, with autoplay on: once, when the videos have their metadata *and* the scene is in
  // (or will not be). A deep link to a frame is a picture somebody pointed at, and stays paused.
  const started = useRef(false);
  useEffect(() => {
    if (started.current || !metadata || gate === "loading") return;
    started.current = true;
    if (!autoplay || initialTime > 0 || prefersReducedMotion()) return;
    void main.current?.play().catch(() => {});
    // `autoplay` is read when the gate opens; turning it on later is the reader's own play press.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [metadata, gate, initialTime]);

  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    const tick = () => {
      const v = main.current;
      if (v !== null) setTime(v.currentTime);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing]);

  useEffect(() => {
    const w = wrist.current;
    if (w === null) return;
    w.playbackRate = rate;
    if (playing && w.paused) void w.play().catch(() => {});
    if (!playing && !w.paused) w.pause();
    if (Math.abs(w.currentTime - time) > 1 / episode.control_rate) w.currentTime = time;
  }, [playing, rate, time, episode.control_rate]);

  useEffect(() => {
    const v = main.current;
    if (v !== null) v.playbackRate = rate;
  }, [rate, episode.id]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target !== null && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;
      if (e.key === " ") { e.preventDefault(); toggle(); }
      else if (e.key === "ArrowLeft") { e.preventDefault(); stepDecision(-1); }
      else if (e.key === "ArrowRight") { e.preventDefault(); stepDecision(1); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggle, stepDecision]);

  const agentview = episode.media.agentview;
  const wristTrack = episode.media.wrist;

  return (
    <div className="rj-page" data-scene-gate={gate}>
      <div className="rj-episode-title">
        <h1 data-testid="replay-title">{episode.instruction}</h1>
        <OutcomeChip success={episode.success} status="done" testId="replay-outcome" />
        <Chip mono>{episode.decisions.length}/{episode.max_decisions}</Chip>
        <Chip mono>task {episode.task_index} · init {episode.init_state_index}</Chip>
        {gate === "failed" && (
          <Chip tone="warn" testId="scene-fallback" title={poseError ?? "the 3D scene did not load"}>video-only</Chip>
        )}
      </div>

      <div className="rj-view">
        <div className="rj-view__main">
          <div className="rj-scene" data-testid="replay-scene">
            {traj !== null && files.scene !== null ? (
              <Suspense fallback={<div className="skeleton scene" />}>
                <SceneView
                  files={{ scene: files.scene }}
                  trajectory={traj}
                  frame={frame}
                  onReady={() => setSceneState("ready")}
                  onFailed={() => setSceneState("failed")}
                />
              </Suspense>
            ) : (
              <div className="skeleton scene" />
            )}
          </div>
          <Transport
            duration={duration}
            time={time}
            playing={playing}
            rate={rate}
            decisions={episode.decisions}
            current={current}
            segments={segments}
            onSeek={seek}
            onToggle={toggle}
            onStepDecision={stepDecision}
            onRate={setRate}
            extra={
              <button
                type="button"
                className="btn btn--sm rj-autoplay"
                aria-pressed={autoplay}
                onClick={onAutoplay}
                title="start each replay by itself once its 3D scene has loaded"
                data-testid="autoplay"
              >
                autoplay
              </button>
            }
          >
            <StageTimeline segments={segments} duration={duration} current={current} onSeek={seek} />
          </Transport>
        </div>

        <div className="rj-view__cams">
          <figure className="cam">
            <div className="cam__frame">
              {agentview !== undefined && files.media.agentview !== undefined ? (
                <video
                  key={`${episode.id}-agentview`}
                  ref={main}
                  data-testid="agentview"
                  src={files.media.agentview}
                  poster={files.poster ?? undefined}
                  width={agentview.width}
                  height={agentview.height}
                  preload="auto"
                  playsInline
                  muted
                  onLoadedMetadata={onLoaded}
                  onPlay={() => setPlaying(true)}
                  onPause={() => setPlaying(false)}
                  onEnded={() => setPlaying(false)}
                  onTimeUpdate={(e) => setTime(e.currentTarget.currentTime)}
                  onClick={toggle}
                />
              ) : <div className="skeleton rj-cam-empty" />}
            </div>
            <figcaption><span>agentview</span><span className="u-dim">camera</span></figcaption>
          </figure>
          {wristTrack !== undefined && files.media.wrist !== undefined && (
            <figure className="cam">
              <div className="cam__frame">
                <video
                  key={`${episode.id}-wrist`}
                  ref={wrist}
                  data-testid="wrist"
                  src={files.media.wrist}
                  width={wristTrack.width}
                  height={wristTrack.height}
                  preload="auto"
                  playsInline
                  muted
                />
              </div>
              <figcaption><span>wrist</span><span className="u-dim">camera</span></figcaption>
            </figure>
          )}
        </div>
      </div>

      <div className="rj-under">
        <div className="rj-statetoggle">
          <button
            type="button"
            className="btn btn--sm"
            aria-expanded={showState}
            onClick={() => setShowState((v) => !v)}
            title="the paragraph the model read for this decision — hovering a question highlights the lines its answer is read from"
            data-testid="state-toggle"
          >
            {showState ? "▾" : "▸"} state text
          </button>
          {showState && decision === null && <span className="u-dim" data-testid="state-empty">—</span>}
        </div>
        <Details episode={episode} decision={decision} grounding={grounded} />
      </div>
      {showState && decision !== null && <StateText state={decision.state} hot={hot} onHot={setHot} />}

      <Card className="rj-decision-card">
        <CardBody pad="tight">
          <div data-testid="decision-card">
            {decision === null ? (
              <p className="u-dim" data-testid="decision-empty">—</p>
            ) : (
              <DecisionGrid decision={decision} hot={hot} onHot={setHot} />
            )}
          </div>
        </CardBody>
      </Card>
    </div>
  );
}
