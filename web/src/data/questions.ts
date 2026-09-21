/**
 * The ten questions, in the order they are asked, with the one-line version of what each one
 * means. The wording is compressed from the question set itself
 * (`robojev/questions.py`); the model reads the full instructions, the reader of this
 * page reads these.
 *
 * `target` and `destination` are here too, at the end, because they are asked in the same shape
 * -- once per episode instead of once per decision -- and the panel draws them with the same bars.
 */
export interface QuestionSpec {
  qid: string;
  /** The question, in six words. */
  asks: string;
  /** What the answer does. */
  means: string;
  /** The candidate ids, as the state's own vocabulary spells them. */
  candidates: string;
  group: "direction" | "size" | "wrist" | "grasp" | "fingers" | "stage" | "episode";
}

export const QUESTIONS: QuestionSpec[] = [
  { qid: "move_x", group: "direction", candidates: "−, hold, +", asks: "Which way along x?",
    means: "The sign of the waypoint's x offset, against the tolerance printed on the same line." },
  { qid: "move_y", group: "direction", candidates: "−, hold, +", asks: "Which way along y?",
    means: "The same question about y — forward is +, towards the robot is −." },
  { qid: "move_z", group: "direction", candidates: "−, hold, +", asks: "Which way along z?",
    means: "The same question about z — up is +. The three answers compose into one diagonal move." },
  { qid: "size_x", group: "size", candidates: "large, medium, small", asks: "How far along x?",
    means: "5.0 / 1.7 / 0.5 cm, read off x's own remaining offset against the printed bands." },
  { qid: "size_y", group: "size", candidates: "large, medium, small", asks: "How far along y?",
    means: "Per axis, so an approach 6 cm out in y and 1 cm out in z does not crawl in y." },
  { qid: "size_z", group: "size", candidates: "large, medium, small", asks: "How far along z?",
    means: "Each axis takes the largest step its own error can afford to overshoot by." },
  { qid: "yaw", group: "wrist", candidates: "−, hold, +", asks: "Turn the wrist?",
    means: "The sign of the wrist's yaw error. A right angle costs six decisions; without it the drawer is ungraspable." },
  { qid: "rim", group: "grasp", candidates: "A – H", asks: "Which side of the target?",
    means: "One lettered line per grasp option: the side, the turn it needs, the room the outer finger gets, and whether that fits." },
  { qid: "grip", group: "fingers", candidates: "true, false", asks: "Keep the fingers closed?",
    means: "An instruction to the fingers, not a prediction. A table guards it: a change only applies in a stage whose plan makes it." },
  { qid: "subgoal", group: "stage", candidates: "reach, grasp, lift, carry, place, retreat", asks: "Which stage is this?",
    means: "Which stage the state's own facts imply — the diagnostic that says when the model and the plan disagree." },
  { qid: "target", group: "episode", candidates: "the scene's objects", asks: "Which object does the sentence mean?",
    means: "Asked once, before any motion. A text rule answers it for the arm; the model's own pick is shown beside it." },
  { qid: "destination", group: "episode", candidates: "the scene's objects", asks: "Where must it end up?",
    means: "Asked once, in the same forward pass as the target." },
];

export const QUESTION_BY_ID: Record<string, QuestionSpec> =
  Object.fromEntries(QUESTIONS.map((q) => [q.qid, q]));

/** The question set's own order — what `qids` is in the server, and what the panel sorts by. */
export const PANEL_ORDER: string[] = QUESTIONS.map((q) => q.qid);

/**
 * The order the panel *draws* them in, which is not the order they are asked in.
 *
 * The panel is a two-column grid, so the pairing is the layout: each axis gets a row with its
 * direction beside its own step size, which is how the two are read ("which way, and how far")
 * and which makes the three axes three rows instead of six stacked boxes. The wrist and the grasp
 * side share the next row, the fingers and the stage the last.
 */
export const PANEL_LAYOUT: string[] = [
  "move_x", "size_x", "move_y", "size_y", "move_z", "size_z",
  // `rim` has eight candidates against everyone else's two or three, so it goes last, beside the
  // tallest of the rest: put it next to a two-bar question and the row is six bars of white space.
  "yaw", "grip", "subgoal", "rim", "target", "destination",
];

/** Sort a decision's question ids into `order`, keeping anything unknown at the end rather than
 *  dropping it -- a question set that grows must still be drawn. */
export function orderQuestions(qids: string[], order: string[] = PANEL_ORDER): string[] {
  const rank = (q: string) => {
    const i = order.indexOf(q);
    return i === -1 ? order.length : i;
  };
  return [...qids].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
}
