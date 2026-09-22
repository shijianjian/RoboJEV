/**
 * The episode in one bar: each stage as a segment the width of the time it held, clickable.
 *
 * It is the answer to "where in the task am I" that a 44-tick scrubber cannot give — on the
 * drawer episode it shows at a glance that reach is a third of the run, because the wrist has to
 * turn before the hand can descend. It sits inside the transport card, directly under the
 * scrubber it shares an axis with, and it names its own stages: a legend under it would be a
 * second row explaining the first.
 */
import { stageColor } from "../data/lookup";
import type { StageSegment } from "../data/lookup";

export function StageTimeline({ segments, duration, current, onSeek }: {
  segments: readonly StageSegment[];
  duration: number;
  /** The decision in force, so the segment containing it can be marked. */
  current: number;
  onSeek: (t: number) => void;
}) {
  // One time axis with the scrubber above it: every segment is placed at its own start and sized
  // to its own span as a fraction of the whole episode, and the settling steps before the first
  // decision - which belong to no stage - are a neutral segment at the front. So the bar starts at
  // t = 0 and ends at the episode's end, exactly where the scrubber's track does.
  const pct = (t: number) => (duration > 0 ? Math.min(100, Math.max(0, (t / duration) * 100)) : 0);
  const settle = segments.length > 0 ? segments[0].from : duration;
  return (
    <div className="rj-timeline" data-testid="stage-timeline">
      {settle > 0 && (
        <span
          className="rj-timeline__seg rj-timeline__seg--settle"
          style={{ left: "0%", width: `${pct(settle)}%` }}
          title={`settling: 0.00–${settle.toFixed(2)} s, no decision`}
          aria-hidden
          data-testid="timeline-settle"
        />
      )}
      {segments.map((seg, i) => {
        const width = pct(seg.to) - pct(seg.from);
        const active = current >= seg.fromDecision && current <= seg.toDecision;
        return (
          <button
            key={i}
            type="button"
            className="rj-timeline__seg"
            style={{ left: `${pct(seg.from)}%`, width: `${width}%`, background: stageColor(seg.stage) }}
            aria-current={active ? "true" : undefined}
            aria-label={`${seg.stage}, ${seg.from.toFixed(2)} to ${seg.to.toFixed(2)} seconds`}
            onClick={() => onSeek(seg.from + 1e-3)}
            title={`${seg.stage}: ${seg.from.toFixed(2)}–${seg.to.toFixed(2)} s, decisions ${seg.fromDecision + 1}–${seg.toDecision + 1}`}
            data-testid={`timeline-${i}`}
            data-from={seg.fromDecision}
          >
            {width > 9 ? seg.stage : ""}
          </button>
        );
      })}
    </div>
  );
}
