/**
 * The transport: play, speed, and a track whose ticks are the decision points.
 *
 * The range input is the real control — invisible, stretched over the whole track, so dragging
 * and the keyboard both work exactly as a browser's range does — and the coloured ticks sit on
 * top of it as buttons, because clicking a specific decision is a different intent from dragging
 * to a time and should not depend on hitting a 3-pixel notch by scrubbing.
 *
 * Time, not frames: the video element's clock is the single source of truth, so seeking is
 * expressed in seconds and the decision in force is looked up from it (`data/lookup.ts`).
 */
import type { ReactNode } from "react";
import type { Decision } from "../data/types";
import { stageColor } from "../data/lookup";
import type { StageSegment } from "../data/lookup";

const SPEEDS = [0.25, 0.5, 1, 2];

export function Transport({
  duration, time, playing, rate, decisions, current, segments, onSeek, onToggle, onStepDecision,
  onRate, children, speeds = SPEEDS, playLabels, readout, hint, extra, disabled = false,
}: {
  duration: number;
  time: number;
  playing: boolean;
  rate: number;
  decisions: readonly Decision[];
  current: number;
  segments: readonly StageSegment[];
  onSeek: (t: number) => void;
  onToggle: () => void;
  onStepDecision: (delta: number) => void;
  onRate: (r: number) => void;
  /** The playback speeds on offer. Empty hides the group: a live episode runs at the simulator's
   *  pace, and a 2× button that did nothing would be worse than no button. */
  speeds?: readonly number[];
  /** What the primary button says. Defaults to play/pause. */
  playLabels?: { play: string; pause: string };
  /** The numbers on the right of the transport. Defaults to time and decision. */
  readout?: ReactNode;
  /** The keyboard hint under them. */
  hint?: ReactNode;
  /** One more control in the transport, after the three. */
  extra?: ReactNode;
  /** The primary button is unavailable. */
  disabled?: boolean;
  /** The stage timeline, drawn inside this same card: it is the same axis at a coarser grain, and
   *  a card of its own with a heading of its own would be a second explanation of one thing. */
  children?: ReactNode;
}) {
  const labels = playLabels ?? { play: "▶ play", pause: "❚❚ pause" };
  const pct = (t: number) => `${duration > 0 ? Math.min(100, Math.max(0, (t / duration) * 100)) : 0}%`;
  return (
    <div className="rj-controls" data-testid="controls">
      <div>
        <div className="rj-transport" role="group" aria-label="playback">
          <div className="rj-transport__buttons">
            <button type="button" className="btn btn--icon" onClick={() => onStepDecision(-1)}
                    aria-label="previous decision" title="previous decision (←)">‹</button>
            <button type="button" className="btn btn--primary" onClick={onToggle} disabled={disabled}
                    aria-label={playing ? "pause" : "play"} title={playing ? "pause (space)" : "play (space)"}
                    data-testid="play">
              {playing ? labels.pause : labels.play}
            </button>
            <button type="button" className="btn btn--icon" onClick={() => onStepDecision(1)}
                    aria-label="next decision" title="next decision (→)">›</button>
            {extra}
          </div>
          {speeds.length > 0 && (
            <div className="rj-speed" role="group" aria-label="speed">
              {speeds.map((s) => (
                <button key={s} type="button" className="btn" aria-pressed={rate === s}
                        onClick={() => onRate(s)} data-testid={`speed-${s}`}>
                  {s}×
                </button>
              ))}
            </div>
          )}
          <div className="rj-transport__readout" data-testid="transport-readout">
            {readout ?? (
              <>
                {time.toFixed(2)} / {duration.toFixed(2)} s
                <span className="u-dim"> · decision </span>
                {current < 0 ? "—" : current + 1} / {decisions.length}
              </>
            )}
          </div>
          <div className="rj-transport__hint" aria-hidden>
            {hint ?? <><kbd>space</kbd> play · <kbd>←</kbd><kbd>→</kbd> decision</>}
          </div>
        </div>

        <div className="rj-scrub">
          <input
            className="rj-scrub__input"
            type="range"
            min={0}
            max={Math.max(duration, 0.001)}
            step={0.01}
            value={Math.min(time, duration)}
            onChange={(e) => onSeek(Number(e.target.value))}
            aria-label="video time in seconds"
            data-testid="scrub-input"
          />
          <div className="rj-scrub__track" aria-hidden>
            {segments.map((seg, i) => (
              <span
                key={i}
                className="rj-scrub__stage"
                style={{
                  left: pct(seg.from),
                  width: `calc(${pct(seg.to)} - ${pct(seg.from)})`,
                  background: stageColor(seg.stage),
                }}
              />
            ))}
          </div>
          {decisions.map((d, i) => (
            <button
              key={i}
              type="button"
              className={`rj-scrub__tick${i === current ? " rj-scrub__tick--current" : ""}`}
              style={{ left: pct(d.t), color: stageColor(d.subgoal) }}
              onClick={() => onSeek(d.t + 1e-3)}
              title={`decision ${i + 1} — ${d.subgoal ?? "?"} at ${d.t.toFixed(2)} s`}
              aria-label={`seek to decision ${i + 1}, stage ${d.subgoal ?? "unknown"}, ${d.t.toFixed(2)} seconds`}
              data-testid={`tick-${i}`}
            />
          ))}
          <span className="rj-scrub__cursor" style={{ left: pct(time) }} aria-hidden />
        </div>
        {children}
      </div>
    </div>
  );
}
