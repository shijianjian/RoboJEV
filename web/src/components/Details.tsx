/**
 * The slim row under the decision panel: labels and values, no sentences.
 *
 * Four things that used to be four cards with four explanations — where the plan is steering,
 * which grasp option the `rim` answer picked, what the sentence was grounded to, and what ran
 * this episode. Each is a `label value` pair; the `title` attribute carries the sentence for
 * anyone who wants it, and the screen does not.
 *
 * The grasp option appears only while it is live — in `reach` and `grasp`, which are the stages
 * whose waypoint it decides. Once the bowl is in the air every option trivially "fits", and a
 * row that is always there but only sometimes means anything is worse than one that comes and
 * goes.
 */
import type { Decision, Episode } from "../data/types";
import { fmtCm, fmtRoom } from "../data/lookup";

const RIM_STAGES = new Set(["reach", "grasp"]);

function Fact({ label, children, title, testId }: {
  label: string;
  children: React.ReactNode;
  title?: string;
  testId?: string;
}) {
  return (
    <span className="fact" title={title} data-testid={testId}>
      <span className="fact__label">{label}</span>
      <span className="fact__value num">{children}</span>
    </span>
  );
}

export function Details({ episode, decision, grounding }: {
  episode: Episode;
  decision: Decision | null;
  /** The decision that grounded, if any — it is an episode-level fact, shown once. */
  grounding: Decision | null;
}) {
  const g = grounding?.grounding ?? null;
  const rim = decision !== null && RIM_STAGES.has(decision.subgoal ?? "") ? decision.rim_chosen : null;
  return (
    <div className="details" data-testid="details">
      {decision?.waypoint_cm != null && (
        <Fact label="waypoint" title={decision.waypoint ?? undefined} testId="fact-waypoint">
          x {fmtCm(decision.waypoint_cm[0])} y {fmtCm(decision.waypoint_cm[1])} z {fmtCm(decision.waypoint_cm[2])} cm
        </Fact>
      )}
      {rim !== null && (
        <Fact
          label="grasp"
          title={`rim point ${rim.letter}: ${rim.side} side, wrist turn ${rim.turn_deg}°, outer finger room ${fmtRoom(rim.room_cm)} — ${rim.verdict}`}
          testId="fact-rim"
        >
          {rim.letter} {rim.side} · {rim.turn_deg}° · {fmtRoom(rim.room_cm)}
          {!rim.fits && <span className="fact__bad"> blocked</span>}
        </Fact>
      )}
      {g !== null && (
        <Fact
          label="grounded"
          title={`the text rule answered ${g.target} → ${g.destination}; the model's own pick was ${g.model?.target} → ${g.model?.destination}`}
          testId="fact-grounding"
        >
          {g.target} <span className="dim">→</span> {g.destination}
          <span className="dim"> {g.source}</span>
          {/* Only when there *is* a model opinion to compare against. The scripted expert runs no
              grounding forward, so `model_agrees` is null on its episodes — and drawing that as a
              disagreement would be the page inventing a second answer nobody ever gave. */}
          {g.model_agrees !== null && (
            <span className={g.model_agrees ? "fact__ok" : "fact__bad"}>
              {g.model_agrees ? " model agrees" : ` model: ${g.model?.target ?? "?"}`}
            </span>
          )}
        </Fact>
      )}
      <Fact label="policy" title={`checkpoint revision ${episode.checkpoint_revision}`} testId="fact-policy">
        {episode.policy} {episode.checkpoint_revision}
      </Fact>
      <Fact label="steps" title={`${episode.steps} control steps at ${episode.control_rate} Hz, ended ${episode.terminated_by}`}>
        {episode.steps} @ {episode.control_rate} Hz
      </Fact>
    </div>
  );
}
