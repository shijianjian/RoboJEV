import type { ReactNode } from "react";
import { Chip, OutcomeChip } from "./Chip";

/** One run as a cell shows it: a recorded bundle, or the console's live episode. */
export interface RunCellRun {
  id: string;
  /** `done` for a recording; the live episode's run state otherwise. */
  status: string;
  /** The cell's first line. robopp puts the model there; here every run is the same kind of
   *  model and what tells them apart is the task, so it is the task's short label. */
  policyLabel: string;
  /** The engine that answered, on the last line. */
  policy?: string;
  initStateIndex: number;
  success: boolean | null;
  /** Decisions taken, for the `· n` after the outcome. */
  steps: number | null;
  /** `decision` for every run this app shows: they all answer questions. */
  family?: string;
  /** Where the cell's ↗ goes: the run's own page. */
  href: string;
}

/**
 * One run, as a cell you can put on a stage: robopp's `ui/RunCell.tsx`, with the two things a
 * static app has no use for taken out - the compare tick (there is no compare page) and the seed
 * (a bundle does not record one).
 *
 * The whole cell is the replay control, one button stretched under the ↗ that opens the run's own
 * page, exactly as robopp's is.
 */
export function RunCell({ run, where, whereLabel, selected, onSelect }: {
  run: RunCellRun;
  where?: ReactNode;
  whereLabel?: string;
  selected: boolean;
  onSelect: (id: string) => void;
}) {
  const done = run.status === "done";
  const replayLabel = `replay ${run.policyLabel}${whereLabel ? `, ${whereLabel}` : ""}, state ${run.initStateIndex}`;
  return (
    <li className={`run-cell${selected ? " is-selected" : ""}`} data-testid="run-row" data-run={run.id}>
      <button
        type="button"
        className="run-cell__hit"
        onClick={() => onSelect(run.id)}
        aria-pressed={selected}
        aria-label={replayLabel}
        data-testid={`replay-${run.id}`}
      />
      <span className="run-cell__top">
        <span className="runs-here__policy" title={run.policyLabel}>{run.policyLabel}</span>
        <span className="run-cell__tools">
          <a href={run.href} className="run-cell__link" aria-label={`open run ${run.id}`} title="this run's own page">↗</a>
        </span>
      </span>
      {where !== undefined && <span className="run-cell__where u-dim">{where}</span>}
      <span className="run-cell__outcome">
        {done
          ? <OutcomeChip success={run.success} status={run.status} />
          : <Chip tone={run.status === "running" ? "running" : "queued"} dot testId={`live-status-${run.id}`}>{run.status}</Chip>}
        {run.family === "decision" && (
          <Chip title="a decision policy: it answers a fixed set of questions and composes the action from the answers" testId="family-decision">decision</Chip>
        )}
        {run.steps != null && <span className="u-dim"> · <span className="u-num">{run.steps}</span></span>}
      </span>
      <span className="run-cell__meta u-dim">
        {run.policy !== undefined && <>{run.policy} · </>}state <span className="u-num">{run.initStateIndex}</span>
      </span>
    </li>
  );
}
