/**
 * The site: one episode, playing, with the decisions beside it.
 *
 * **The video's clock drives everything.** A rAF loop while playing (plus `timeupdate` while
 * paused) sets `time`, `time` picks the decision, and the decision draws the panel, the details
 * row and — when it is open — the state text. Seeking, from a tick, the timeline or the arrow
 * keys, sets `video.currentTime` and nothing else; there is no second cursor to keep in step,
 * which is the bug this arrangement exists to make impossible.
 *
 * **It plays by itself**, because the page is a demo: muted (there is no audio track at all),
 * starting when the metadata is in and again whenever the episode is switched, looping back to
 * the start after a short pause on the last frame. `prefers-reduced-motion: reduce` turns all of
 * that off and leaves the poster up, paused.
 *
 * The wrist camera is a second `<video>` kept in step by assignment rather than by its own clock:
 * two elements playing independently drift, and the small one is a detail of the big one.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Episode } from "../data/types";
import { decisionAt, episodeDuration, seekTimeFor, stageSegments } from "../data/lookup";
import { DecisionPanel } from "./DecisionPanel";
import { Details } from "./Details";
import { StageTimeline } from "./StageTimeline";
import { StateText } from "./StateText";
import { Scrubber } from "./Scrubber";
import { Card, Chip, OutcomeChip } from "./ui";

/** How long the last frame is held before the episode starts over: long enough to see how it
 *  ended, short enough that nobody thinks it has frozen. */
const LOOP_PAUSE_MS = 1400;

function prefersReducedMotion(): boolean {
  return typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
}

export function Replay({ episode, mediaUrl }: {
  episode: Episode;
  mediaUrl: (path: string) => string;
}) {
  const main = useRef<HTMLVideoElement>(null);
  const wrist = useRef<HTMLVideoElement>(null);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [rate, setRate] = useState(1);
  const [hot, setHot] = useState<string | null>(null);
  const [showState, setShowState] = useState(false);

  const duration = episodeDuration(episode);
  const segments = useMemo(() => stageSegments(episode), [episode]);
  const current = decisionAt(episode.decisions, time);
  const decision = current >= 0 ? episode.decisions[current] : null;
  const grounded = useMemo(
    () => episode.decisions.find((d) => d.grounding !== null) ?? null,
    [episode],
  );

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

  // A new episode starts at the top. The `<video>` is keyed by episode id, so the element itself
  // is rebuilt and this is a fresh load rather than a stale frame of the one before.
  useEffect(() => {
    setTime(0);
    setHot(null);
  }, [episode.id]);

  const onLoaded = useCallback(() => {
    const v = main.current;
    if (v === null || prefersReducedMotion()) return;
    v.currentTime = 0;
    void v.play().catch(() => {});
  }, []);

  // The loop: hold the last frame, then start over. The `loop` attribute would restart instantly
  // and nobody would see how the episode ended.
  useEffect(() => {
    const v = main.current;
    if (v === null) return;
    const onEnded = () => {
      if (prefersReducedMotion()) return;
      window.setTimeout(() => {
        const el = main.current;
        if (el === null) return;
        el.currentTime = 0;
        void el.play().catch(() => {});
      }, LOOP_PAUSE_MS);
    };
    v.addEventListener("ended", onEnded);
    return () => v.removeEventListener("ended", onEnded);
  }, [episode.id]);

  // While playing, `timeupdate` alone is too coarse — it fires about four times a second, and so
  // do the decisions — so the clock is read every frame; while paused the loop stops entirely.
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
    <div>
      <div className="episode-title">
        <h1 data-testid="replay-title">{episode.instruction}</h1>
        <OutcomeChip success={episode.success} testId="replay-outcome" />
        <Chip mono>{episode.decisions.length}/{episode.max_decisions}</Chip>
        <Chip mono>task {episode.task_index} · init {episode.init_state_index}</Chip>
      </div>

      <div className="replay">
        <div className="replay__left">
          <div className="stage">
            <div className="videobox">
              {agentview !== undefined ? (
                <video
                  key={`${episode.id}-agentview`}
                  ref={main}
                  data-testid="agentview"
                  src={mediaUrl(agentview.path)}
                  poster={episode.poster != null ? mediaUrl(episode.poster) : undefined}
                  width={agentview.width}
                  height={agentview.height}
                  preload="auto"
                  playsInline
                  muted
                  autoPlay
                  onLoadedMetadata={onLoaded}
                  onPlay={() => setPlaying(true)}
                  onPause={() => setPlaying(false)}
                  onEnded={() => setPlaying(false)}
                  onTimeUpdate={(e) => setTime(e.currentTarget.currentTime)}
                  onClick={toggle}
                />
              ) : <p className="notice">This bundle carries no scene camera.</p>}
            </div>
            {wristTrack !== undefined && (
              <div className="videobox videobox--wrist" title="in-hand camera">
                <video
                  key={`${episode.id}-wrist`}
                  ref={wrist}
                  data-testid="wrist"
                  src={mediaUrl(wristTrack.path)}
                  width={wristTrack.width}
                  height={wristTrack.height}
                  preload="auto"
                  playsInline
                  muted
                />
              </div>
            )}
          </div>

          <Scrubber
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
          >
            <StageTimeline segments={segments} duration={duration} current={current} onSeek={seek} />
          </Scrubber>

          <div className="statetoggle">
            <button
              type="button"
              className="btn"
              aria-expanded={showState}
              onClick={() => setShowState((v) => !v)}
              title="the paragraph the model read for this decision — hovering a question highlights the lines its answer is read from"
              data-testid="state-toggle"
            >
              {showState ? "▾" : "▸"} state text
            </button>
            {showState && decision === null && (
              <span className="dim" data-testid="state-empty">—</span>
            )}
          </div>
          {showState && decision !== null && (
            <StateText state={decision.state} hot={hot} onHot={setHot} />
          )}
          <Details episode={episode} decision={decision} grounding={grounded} />
        </div>

        <div className="replay__right">
          <Card testId="decision-card" bodyClass="card__body card__body--tight">
            {decision === null ? (
              <p className="muted" data-testid="decision-empty">—</p>
            ) : (
              <DecisionPanel decision={decision} hot={hot} onHot={setHot} />
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}
