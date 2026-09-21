/**
 * What the model answered: one labelled row of candidate bars per question.
 *
 * The qid and the bars, and nothing else on screen — what each question asks is in the row's
 * `title`, for whoever wants it. The bar's fill is the probability, the accent edge is the answer
 * that executed, and a `grip` the guard refused is struck through in red beside the answer that
 * replaced it: a refusal is the one place where the model's answer and the robot's action differ,
 * so it must never be silently redrawn as agreement.
 *
 * Hovering (or focusing) a question tells the caller which question is "hot"; the state text, when
 * it is open, lights the lines that answer it. That link is the question set's own
 * (`data/stateText.ts`), not an attention map.
 */
import type { Decision } from "../data/types";
import { PANEL_LAYOUT, QUESTION_BY_ID, orderQuestions } from "../data/questions";
import { fmtProbability, refusedCandidate } from "../data/lookup";
import { Chip } from "./ui";

function Bar({ qid, id, p, chosen, refused }: {
  qid: string; id: string; p: number; chosen: boolean; refused: boolean;
}) {
  const cls = ["bar", chosen ? "bar--chosen" : "", refused ? "bar--refused" : ""].filter(Boolean).join(" ");
  return (
    <span className={cls} aria-current={chosen ? "true" : undefined} data-testid={`bar-${qid}-${id}`}>
      <span className="bar__fill" aria-hidden style={{ ["--p" as string]: Math.max(p, 0) }} />
      <span className="sr">{qid}: </span>
      <span className="bar__id">{id === "-" ? "−" : id}</span>
      {chosen && <span className="sr">, chosen</span>}
      {refused && <span className="sr">, asked for by the model and refused by the grip guard</span>}
      <span className="bar__p">{fmtProbability(p)}</span>
    </span>
  );
}

export function DecisionPanel({ decision, hot, onHot }: {
  decision: Decision;
  /** The question the reader is pointing at, or null. */
  hot: string | null;
  onHot: (qid: string | null) => void;
}) {
  const qids = orderQuestions(Object.keys(decision.questions), PANEL_LAYOUT);
  const latch = decision.grip_latch;
  const refused = refusedCandidate(latch);

  return (
    <div data-testid="decision-panel">
      <p className="decision__meta" data-testid="decision-meta">
        <span>
          decision <b className="num">{decision.index + 1}</b>
          <span className="dim"> · </span>
          <span className="num">{decision.t.toFixed(2)} s</span>
        </span>
        {decision.subgoal !== null && (
          <Chip testId="decision-stage">
            {decision.subgoal}
            {decision.substage !== null && <span className="dim"> {decision.substage}</span>}
          </Chip>
        )}
        {latch.refused && <Chip tone="failure" testId="decision-refused">grip refused</Chip>}
      </p>

      <div className="decision__grid" data-testid="decision-groups">
        {qids.map((qid) => {
          const group = decision.questions[qid];
          const spec = QUESTION_BY_ID[qid];
          const isHot = hot === qid;
          return (
            <section
              key={qid}
              className={`decision__group${isHot ? " decision__group--hot" : ""}`}
              data-testid={`decision-${qid}`}
              title={spec === undefined ? qid : `${spec.asks} ${spec.means}`}
              onMouseEnter={() => onHot(qid)}
              onMouseLeave={() => onHot(null)}
              onFocus={() => onHot(qid)}
              onBlur={() => onHot(null)}
              tabIndex={0}
            >
              <h3 className="decision__head">
                <span className="decision__qid">{qid}</span>
                {qid === "grip" && latch.refused && (
                  <Chip tone="failure">{latch.closed ? "held closed" : "held open"}</Chip>
                )}
              </h3>
              {group.candidates.map((c) => (
                <Bar
                  key={c.id}
                  qid={qid}
                  id={c.id}
                  p={c.p}
                  chosen={c.id === group.choice}
                  refused={qid === "grip" && c.id === refused}
                />
              ))}
            </section>
          );
        })}
      </div>
    </div>
  );
}
