/**
 * The clock. One episode has two of them -- seconds of video and decision points -- and every
 * disagreement between the panel and the picture would be a bug in this file, so it is the only
 * place either is converted into the other.
 *
 * The rule is the recorder's: the video holds **every** control step at the backend's control
 * rate, settle steps included, so `t = control_step / control_rate` exactly, and the decision in
 * force at time `t` is the last one whose own `t` is not in the future. Before the first decision
 * (the ten settle steps) there is no decision in force, and the page says so rather than
 * pretending the first one is already answering.
 */
import type { Decision, Episode, GripLatch } from "./types";

/**
 * The `grip` candidate the model asked for and the guard refused, or null when nothing was
 * refused.
 *
 * The bundle records the **executed** answer in `questions.grip.choice`, because that is what the
 * robot did; the model's own answer survives only in `grip_latch.asked`. When the two differ, the
 * panel has to show both — the guard's answer as the chosen one and the model's struck through —
 * because a decision policy whose refusals are redrawn as agreement is exactly the picture this
 * page exists to not paint.
 */
export function refusedCandidate(latch: GripLatch): string | null {
  if (!latch.refused || latch.asked === null) return null;
  return latch.asked ? "true" : "false";
}

/** The stage vocabulary, in the order the task goes through it. The colours are the scrubber's
 *  tick colours and the timeline's segments -- one hue per stage, quiet against the panel, with
 *  the accent reserved for the cursor and for what the model chose. */
export const STAGES = ["reach", "grasp", "lift", "carry", "place", "retreat"] as const;
export type Stage = (typeof STAGES)[number];

export const STAGE_COLOR: Record<string, string> = {
  reach: "#58a6ff",
  grasp: "#f5b301",
  lift: "#3ecf84",
  carry: "#5ed3c4",
  place: "#b18cff",
  retreat: "#97a3b6",
};

export function stageColor(stage: string | null | undefined): string {
  return (stage && STAGE_COLOR[stage]) || "#7f8b9e";
}

/**
 * The index of the decision in force at video time `t`, or -1 before the first one.
 *
 * Binary search, because it runs on every `timeupdate` and a 44-entry linear scan that is called
 * 60 times a second is a habit that stops being free the day someone records a long episode.
 */
export function decisionAt(decisions: readonly Decision[], t: number): number {
  let lo = 0;
  let hi = decisions.length - 1;
  let found = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (decisions[mid].t <= t + 1e-9) {
      found = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return found;
}

/** The video time to seek to in order to land on decision `i`: its own instant, nudged a hair
 *  forward so a seek that rounds down does not land on the decision before it. */
export function seekTimeFor(decisions: readonly Decision[], i: number): number {
  const d = decisions[Math.max(0, Math.min(i, decisions.length - 1))];
  return d === undefined ? 0 : d.t + 1e-3;
}

/** How long the episode's video is, in seconds, from the bundle alone -- so the scrubber can be
 *  drawn before the video element has any metadata. */
export function episodeDuration(episode: Episode): number {
  return episode.total_frames / episode.control_rate;
}

export interface StageSegment {
  stage: string;
  from: number;
  to: number;
  fromDecision: number;
  toDecision: number;
}

/**
 * The stage timeline: consecutive decisions that share a sub-goal, merged into one bar.
 *
 * The last segment runs to the end of the video rather than to the last decision, because the
 * final stage is still in force while the arm finishes executing its last action -- a timeline
 * that stopped 0.25 s early would leave a gap nobody could seek into.
 */
export function stageSegments(episode: Episode): StageSegment[] {
  const out: StageSegment[] = [];
  const end = episodeDuration(episode);
  episode.decisions.forEach((d, i) => {
    const stage = d.subgoal ?? "unknown";
    const last = out[out.length - 1];
    if (last !== undefined && last.stage === stage) {
      last.to = d.t;
      last.toDecision = i;
    } else {
      out.push({ stage, from: d.t, to: d.t, fromDecision: i, toDecision: i });
    }
  });
  for (let i = 0; i < out.length; i += 1) {
    out[i].to = i + 1 < out.length ? out[i + 1].from : end;
  }
  return out;
}

/** `0.25 s` as the page says it: two decimals, no trailing unit surprises. */
export function fmtTime(t: number): string {
  return `${t.toFixed(2)} s`;
}

/** A probability as a percentage that never lies by rounding: a 99.98 % answer is not "100 %"
 *  and a 0.02 % one is not "0 %", because both of those would claim a certainty the model did
 *  not have. */
export function fmtProbability(p: number): string {
  if (p >= 1) return "100%";
  if (p <= 0) return "0%";
  if (p > 0.999) return ">99.9%";
  if (p < 0.001) return "<0.1%";
  if (p >= 0.1) return `${(p * 100).toFixed(1)}%`;
  return `${(p * 100).toFixed(2)}%`;
}

/** A signed centimetre offset: `+0.9`, `−4.0`, and never `-0.0`. */
export function fmtCm(v: number): string {
  const abs = Math.abs(v).toFixed(1);
  return `${v < 0 && abs !== "0.0" ? "−" : "+"}${abs}`;
}

/**
 * The task sentence, short enough for an episode-strip item: `bowl in the top drawer of the
 * wooden cabinet`.
 *
 * LIBERO's instructions are all "pick up X and place it on Y", and the four episodes here differ
 * only in the X — leading every item with "pick up the black bowl" would make them look identical
 * at the glance the strip exists for. So the strip shows X, with the colour word dropped (all
 * four bowls are black) and the destination left to the title line. A sentence that does not
 * match the pattern comes back unchanged rather than cut somewhere arbitrary; the CSS truncates
 * to one line either way.
 */
export function shortLabel(instruction: string): string {
  const m = /^\s*pick up (.+?) and place it (?:on|in) (.+?)\s*$/i.exec(instruction);
  if (m === null) return instruction.trim();
  return m[1].replace(/^the\s+/i, "").replace(/^black\s+/i, "").trim();
}

/** The sentinel the plan writes for "no wall anywhere near this rim point". A measurement that
 *  was never taken must not be printed as one, so the cards say `open` — while the verbatim state
 *  text, which is the record of what the model read, keeps saying whatever the server said. */
export const NO_WALL_CM = 999;

export function fmtRoom(cm: number): string {
  return cm >= NO_WALL_CM ? "open" : `${cm.toFixed(1)} cm`;
}
