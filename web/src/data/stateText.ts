/**
 * Which line of the state text answers which question.
 *
 * This is the one claim the page makes that is not already in the bundle, so it is worth being
 * exact about what it is: **it is the question set's own wording, read back off the state**. Each
 * `move_*` question's instructions say "the state prints the waypoint's x offset ... and the
 * tolerance it is judged against"; the state prints exactly one such line per axis. Each `size_*`
 * question says "the state prints the x offset and the three bands the sizes cover"; that is the
 * same axis line plus the bands line. `rim` says "the state lists the grasp candidates as
 * lettered lines". So hovering a question and seeing its lines light up is not a guess about what
 * the model attended to -- it is the mapping the questions were written against
 * (`robojev/questions.py`, and the template in `robojev/state.py#serialise_v2`).
 *
 * It is a *lexical* classification of a generated paragraph, which means it can go stale if that
 * template changes. Hence `unmapped`: a line this file does not recognise is still drawn, plainly,
 * rather than dropped, and the tests pin the classification against a real recorded state.
 */

export type LineKind =
  | "header"
  | "task"
  | "grounding"
  | "subgoal"
  | "gripper"
  | "yaw"
  | "rests"
  | "rim-head"
  | "rim-row"
  | "waypoint"
  | "axis"
  | "largest"
  | "bands"
  | "objects"
  | "events"
  | "counters"
  | "history"
  | "none"
  | "other";

export interface StateLine {
  /** 0-based line number in the state text. */
  index: number;
  text: string;
  kind: LineKind;
  /** The questions this line is the evidence for. Empty for a line that is context only. */
  qids: string[];
  /** For an axis line, the axis it belongs to; for a rim row, the candidate's letter. */
  key: string | null;
}

const AXIS_RE = /^\s{2}([xyz]):\s/;
const RIM_RE = /^\s{2}([A-H]):\s/;

const MOVE_SIZE = (axis: string) => [`move_${axis}`, `size_${axis}`];

/** Classify one line. Order matters: the indented axis and rim rows are matched before any of the
 *  prefix rules, because both begin with two spaces and nothing else does. */
function classify(line: string): { kind: LineKind; qids: string[]; key: string | null } {
  const axis = AXIS_RE.exec(line);
  if (axis) return { kind: "axis", qids: MOVE_SIZE(axis[1]), key: axis[1] };

  const rim = RIM_RE.exec(line);
  if (rim) return { kind: "rim-row", qids: ["rim"], key: rim[1] };

  if (line.startsWith("Robot:")) return { kind: "header", qids: [], key: null };
  if (line.startsWith("Task:")) return { kind: "task", qids: ["target", "destination"], key: null };
  if (line.startsWith("Target:") && line.includes("Destination:")) {
    return { kind: "grounding", qids: ["target", "destination"], key: null };
  }
  if (line.startsWith("Subgoal so far:")) return { kind: "subgoal", qids: ["subgoal"], key: null };
  if (line.startsWith("Gripper:")) return { kind: "gripper", qids: ["grip", "subgoal"], key: null };
  if (line.startsWith("Wrist yaw error:")) return { kind: "yaw", qids: ["yaw"], key: null };
  if (line.startsWith("Target rests")) return { kind: "rests", qids: ["rim"], key: null };
  if (line.startsWith("Grasp candidates")) return { kind: "rim-head", qids: ["rim"], key: null };
  if (line.startsWith("Waypoint")) {
    return {
      kind: "waypoint",
      qids: ["move_x", "move_y", "move_z", "size_x", "size_y", "size_z", "grip", "subgoal"],
      key: null,
    };
  }
  if (line.startsWith("Largest remaining offset:")) {
    return { kind: "largest", qids: ["size_x", "size_y", "size_z"], key: null };
  }
  if (line.startsWith("Step bands:")) {
    return { kind: "bands", qids: ["size_x", "size_y", "size_z"], key: null };
  }
  if (line.startsWith("Other objects:")) return { kind: "objects", qids: [], key: null };
  if (line.startsWith("Events:")) return { kind: "events", qids: ["subgoal"], key: null };
  if (line.startsWith("Attempts:")) return { kind: "counters", qids: ["rim", "subgoal"], key: null };
  if (/^Last \d+:/.test(line)) return { kind: "history", qids: ["subgoal", "grip"], key: null };
  if (line.trim() === "") return { kind: "none", qids: [], key: null };
  return { kind: "other", qids: [], key: null };
}

/** The whole paragraph, line by line, with each line's questions attached. */
export function parseStateText(state: string): StateLine[] {
  return state.split("\n").map((text, index) => ({ index, text, ...classify(text) }));
}

/** The line numbers a question's answer is read off. */
export function linesForQuestion(lines: readonly StateLine[], qid: string): number[] {
  return lines.filter((l) => l.qids.includes(qid)).map((l) => l.index);
}

/** Every line the state text has that no question reads -- the context lines. Exported because
 *  a jump in this number is how a template change makes itself known in the tests. */
export function unmapped(lines: readonly StateLine[]): StateLine[] {
  return lines.filter((l) => l.qids.length === 0);
}
