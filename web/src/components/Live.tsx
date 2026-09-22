/**
 * The console: the same page as a replay, with the episode still happening.
 *
 * Deliberately the same layout, the same components and the same reading order — the episode
 * strip, the two cameras, the transport and its decision ticks, the state-text disclosure, the
 * details row, and the panel of bars on the right. A replay and a live episode are two pictures of
 * one thing, and a console that arranged them differently would be a second site.
 *
 * Three things are the live view's own, and `PROTOCOL.md` names all three:
 *
 * 1. **There is no seekable video**, so the two cameras are `<img>`s the console refreshes one
 *    picture at a time — an mp4 that is still being written is not something a browser can play.
 * 2. **The newest decision is the current one.** The scrubber is a history rather than a control
 *    over what is shown: scrubbing back stops the panel following, and the transport's ⤓ puts it
 *    back on the newest.
 * 3. **A candidate bar is a control.** Clicking one holds that answer for the next decision; the
 *    console executes it and records it with the `overridden` flag the bundle already carries.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { LiveSource, LiveState, StartSpec } from "../data/live";
import { commandsEnabled, liveEpisode } from "../data/live";
import { decisionAt, episodeDuration, stageSegments } from "../data/lookup";
import { DecisionPanel } from "./DecisionPanel";
import { Details } from "./Details";
import { LiveControls } from "./LiveControls";
import { Scrubber } from "./Scrubber";
import { StageTimeline } from "./StageTimeline";
import { StateText } from "./StateText";
import { Card, Chip, OutcomeChip } from "./ui";

/** The shortest gap between two picture requests. The simulator produces twenty frames a second,
 *  so asking faster than that is asking for the same picture twice. */
const FRAME_GAP_MS = 45;

/** How long to wait after a picture request that failed. A console between episodes has none to
 *  give, and hammering it changes nothing except the size of its log. */
const FRAME_RETRY_MS = 600;

/**
 * One camera, refreshed a picture at a time.
 *
 * Self-paced: the next request goes out when the previous picture has painted, so a slow machine
 * simply shows fewer frames instead of queueing requests behind itself. It asks for nothing at all
 * while no episode is open, which is what keeps an idle console idle.
 */
function LiveFrame({ url, alt, testId, className }: {
  url: string | null;
  alt: string;
  testId: string;
  className: string;
}) {
  const img = useRef<HTMLImageElement>(null);

  useEffect(() => {
    const element = img.current;
    if (element === null || url === null) return;
    let alive = true;
    let seq = 0;
    let timer = 0;
    const next = () => {
      if (!alive) return;
      seq += 1;
      element.src = `${url}${url.includes("?") ? "&" : "?"}n=${seq}`;
    };
    const again = (ms: number) => {
      if (!alive) return;
      window.clearTimeout(timer);
      timer = window.setTimeout(next, ms);
    };
    const onLoad = () => again(FRAME_GAP_MS);
    const onError = () => again(FRAME_RETRY_MS);
    element.addEventListener("load", onLoad);
    element.addEventListener("error", onError);
    next();
    return () => {
      alive = false;
      window.clearTimeout(timer);
      element.removeEventListener("load", onLoad);
      element.removeEventListener("error", onError);
    };
  }, [url]);

  return (
    <div className={className} title={alt}>
      <img ref={img} className="liveframe" alt={alt} data-testid={testId} width={256} height={256} />
    </div>
  );
}

export function Live({ source, state }: { source: LiveSource; state: LiveState }) {
  const [cursor, setCursor] = useState(-1);
  const [follow, setFollow] = useState(true);
  const [hot, setHot] = useState<string | null>(null);
  const [showState, setShowState] = useState(false);

  const decisions = state.decisions;
  const newest = decisions.length - 1;
  // A new episode is a new console: the cursor must not be left pointing into the previous one.
  const [watched, setWatched] = useState(state.header?.id ?? null);
  if (watched !== (state.header?.id ?? null)) {
    setWatched(state.header?.id ?? null);
    setCursor(-1);
    setFollow(true);
    setHot(null);
  }

  const current = follow ? newest : Math.max(-1, Math.min(cursor, newest));
  const decision = current >= 0 ? decisions[current] : null;
  const episode = liveEpisode(state);
  const segments = useMemo(() => (episode === null ? [] : stageSegments(episode)), [episode]);
  const grounded = useMemo(
    () => decisions.find((d) => d.grounding !== null) ?? null,
    [decisions],
  );
  const can = commandsEnabled(state);

  const look = useCallback((at: number) => {
    setCursor(at);
    // Landing on the newest decision is the same intent as pressing ⤓, so it re-follows: a console
    // stuck at "history" while pointing at the live decision would be lying about which it is.
    setFollow(at >= newest);
  }, [newest]);

  const onSeek = useCallback((t: number) => {
    if (decisions.length === 0) return;
    look(Math.max(0, decisionAt(decisions, t)));
  }, [decisions, look]);

  const onStepDecision = useCallback((delta: number) => {
    if (decisions.length === 0) return;
    look(Math.max(0, Math.min((current < 0 ? newest : current) + delta, newest)));
  }, [current, decisions.length, newest, look]);

  const onToggle = useCallback(() => {
    if (state.run === "running") source.pause();
    else if (can.run) source.run();
  }, [source, state.run, can.run]);

  const onStart = useCallback((spec: StartSpec) => { source.start(spec); }, [source]);
  const onCommand = useCallback((op: "step" | "run" | "pause" | "reset") => {
    if (op === "step") source.step();
    else if (op === "run") source.run();
    else if (op === "pause") source.pause();
    else source.reset();
  }, [source]);
  const onSave = useCallback(() => { source.save(); }, [source]);

  // Only the newest decision can be overridden: an override is an instruction about the decision
  // that has not been taken yet, and arming one from a decision five back would mean nothing.
  const onOverride = useCallback((qid: string, candidate: string) => {
    source.override(qid, state.overrides[qid] === candidate ? null : candidate);
  }, [source, state.overrides]);

  // Space runs and pauses; ←/→ walk the history. The same keys as the replay, doing the live
  // version of the same thing.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target !== null && /^(INPUT|TEXTAREA|SELECT|BUTTON)$/.test(target.tagName)) return;
      if (e.key === " ") { e.preventDefault(); onToggle(); }
      else if (e.key === "ArrowLeft") { e.preventDefault(); onStepDecision(-1); }
      else if (e.key === "ArrowRight") { e.preventDefault(); onStepDecision(1); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onToggle, onStepDecision]);

  const duration = episode === null ? 1 : Math.max(episodeDuration(episode), 0.001);
  const armed = follow && can.override ? state.overrides : {};

  return (
    <div>
      <div className="episode-title">
        <h1 data-testid="live-title">
          {state.header?.instruction ?? "The console is waiting for an episode."}
        </h1>
        {state.done !== null && <OutcomeChip success={state.done.success} testId="live-outcome" />}
        {state.header !== null && (
          <>
            <Chip mono>{decisions.length}/{state.header.max_decisions}</Chip>
            <Chip mono>task {state.header.task_index} · init {state.header.init_state_index}</Chip>
          </>
        )}
      </div>

      <LiveControls state={state} onStart={onStart} onCommand={onCommand} onSave={onSave} />

      <div className="replay live">
        <div className="replay__left">
          {/* Nothing is requested while no episode is open: an idle console holds a simulator and
              should not also be answering forty requests a second for a picture of nothing. */}
          {state.episode ? (
            <div className="stage">
              <LiveFrame
                className="videobox"
                url={source.frameUrl("agentview")}
                alt="the scene camera, live"
                testId="live-agentview"
              />
              <LiveFrame
                className="videobox videobox--wrist"
                url={source.frameUrl("wrist")}
                alt="the in-hand camera, live"
                testId="live-wrist"
              />
            </div>
          ) : (
            <p className="notice" data-testid="live-no-episode">
              No episode. Start one above and the scene and in-hand cameras appear here, a frame at
              a time.
            </p>
          )}

          <Scrubber
            duration={duration}
            time={decision?.t ?? 0}
            playing={state.run === "running"}
            rate={1}
            decisions={decisions}
            current={current}
            segments={segments}
            onSeek={onSeek}
            onToggle={onToggle}
            onStepDecision={onStepDecision}
            onRate={() => {}}
            speeds={[]}
            disabled={!can.run && state.run !== "running"}
            playLabels={{ play: "▶ run", pause: "❚❚ pause" }}
            extra={
              <button
                type="button"
                className="btn btn--icon"
                aria-pressed={follow}
                disabled={decisions.length === 0}
                onClick={() => { setFollow(true); setCursor(newest); }}
                title="follow the newest decision"
                aria-label="follow the newest decision"
                data-testid="live-follow"
              >
                ⤓
              </button>
            }
            readout={
              <>
                step {state.step ?? 0} / {state.header?.max_steps ?? "—"}
                <span className="dim"> · decision </span>
                {current < 0 ? "—" : current + 1} / {decisions.length}
                {!follow && <span className="dim"> · history</span>}
              </>
            }
            hint={<><kbd>space</kbd> run · <kbd>←</kbd><kbd>→</kbd> decision</>}
          >
            <StageTimeline segments={segments} duration={duration} current={current} onSeek={onSeek} />
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
          {episode !== null && (
            <Details episode={episode} decision={decision} grounding={grounded} />
          )}
        </div>

        <div className="replay__right">
          <Card testId="decision-card" bodyClass="card__body card__body--tight">
            {decision === null ? (
              <p className="muted" data-testid="decision-empty">
                {state.episode
                  ? "No decision yet — Step takes one."
                  : "Start an episode and the model's answers appear here, one row of bars per question."}
              </p>
            ) : (
              <DecisionPanel
                decision={decision}
                hot={hot}
                onHot={setHot}
                armed={armed}
                onOverride={follow && can.override ? onOverride : undefined}
              />
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}
