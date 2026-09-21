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
  return (
    <div className="timeline" data-testid="stage-timeline">
      {segments.map((seg, i) => {
        const width = duration > 0 ? ((seg.to - seg.from) / duration) * 100 : 0;
        const active = current >= seg.fromDecision && current <= seg.toDecision;
        return (
          <button
            key={i}
            type="button"
            className="timeline__seg"
            style={{ width: `${width}%`, background: stageColor(seg.stage) }}
            aria-current={active ? "true" : undefined}
            aria-label={`${seg.stage}, ${seg.from.toFixed(2)} to ${seg.to.toFixed(2)} seconds`}
            onClick={() => onSeek(seg.from + 1e-3)}
            title={`${seg.stage}: ${seg.from.toFixed(2)}–${seg.to.toFixed(2)} s, decisions ${seg.fromDecision + 1}–${seg.toDecision + 1}`}
            data-testid={`timeline-${i}`}
          >
            {width > 9 ? seg.stage : ""}
          </button>
        );
      })}
    </div>
  );
}
