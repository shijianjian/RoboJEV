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
 *
 * **Live, a bar is also a control.** Given `onOverride`, every candidate becomes a button: clicking
 * one arms it for the *next* decision, which the console then executes and records with the
 * `overridden` flag the bundle schema already carries. What is drawn as armed is `armed`, which is
 * the console's own answer and not this page's memory of the click — a bar lit for an override the
 * server refused is the worst thing this panel could draw.
 */
import type { Decision } from "../data/types";
import { PANEL_LAYOUT, QUESTION_BY_ID, orderQuestions } from "../data/questions";
import { fmtProbability, refusedCandidate } from "../data/lookup";
import { Chip } from "./ui";

function Bar({ qid, id, p, chosen, refused, armed = false, onOverride }: {
  qid: string; id: string; p: number; chosen: boolean; refused: boolean;
  armed?: boolean;
  onOverride?: (qid: string, candidate: string) => void;
}) {
  const cls = ["bar", chosen ? "bar--chosen" : "", refused ? "bar--refused" : "",
               armed ? "bar--armed" : "", onOverride !== undefined ? "bar--live" : ""]
    .filter(Boolean).join(" ");
  const inside = (
    <>
      <span className="bar__fill" aria-hidden style={{ ["--p" as string]: Math.max(p, 0) }} />
      <span className="sr">{qid}: </span>
      <span className="bar__id">{id === "-" ? "−" : id}</span>
      {chosen && <span className="sr">, chosen</span>}
      {armed && <span className="sr">, held for the next decision</span>}
      {refused && <span className="sr">, asked for by the model and refused by the grip guard</span>}
      <span className="bar__p">{fmtProbability(p)}</span>
    </>
  );
  if (onOverride === undefined) {
    return (
      <span className={cls} aria-current={chosen ? "true" : undefined} data-testid={`bar-${qid}-${id}`}>
        {inside}
      </span>
    );
  }
  return (
    <button
      type="button"
      className={cls}
      aria-pressed={armed}
      // A second click on the held candidate takes it back off: there is no "no override"
      // candidate to click instead.
      title={armed ? `held: click to let ${qid} answer itself` : `hold ${qid} = ${id} for the next decision`}
      onClick={() => onOverride(qid, id)}
      data-testid={`bar-${qid}-${id}`}
    >
      {inside}
    </button>
  );
}

export function DecisionPanel({ decision, hot, onHot, armed, onOverride }: {
  decision: Decision;
  /** The question the reader is pointing at, or null. */
  hot: string | null;
  onHot: (qid: string | null) => void;
  /** `{qid: candidate}` the console is holding for the next decision. Live only. */
  armed?: Record<string, string>;
  /** Given, every bar is a control; absent — a replay — they are read-only. */
  onOverride?: (qid: string, candidate: string) => void;
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
                {group.overridden && (
                  <Chip tone="accent" testId={`overridden-${qid}`}>overridden</Chip>
                )}
                {armed?.[qid] !== undefined && (
                  <Chip tone="accent" testId={`armed-${qid}`}>next: {armed[qid]}</Chip>
                )}
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
                  armed={armed?.[qid] === c.id}
                  onOverride={onOverride}
                />
              ))}
            </section>
          );
        })}
      </div>
    </div>
  );
}
