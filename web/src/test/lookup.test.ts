/** The clock and the stage timeline: the two pure things a wrong panel would be blamed on. */
import { describe, expect, it } from "vitest";
import {
  decisionAt, episodeDuration, fmtCm, fmtProbability, fmtRoom, refusedCandidate, seekTimeFor,
  shortLabel, stageSegments, stageColor,
} from "../data/lookup";
import type { Decision, Episode } from "../data/types";

function decision(index: number, t: number, subgoal: string): Decision {
  return {
    index, step: index, control_step: Math.round(t * 20), t, state: "", questions: {},
    grip_latch: { asked: false, closed: false, refused: false }, grounding: null,
    subgoal, substage: null, target: null, destination: null, waypoint: null, waypoint_cm: null,
    rim_candidates: [], rim_chosen: null, action: [0, 0, 0, 0, 0, 0, -1],
    forward_passes: 1, candidate_paths: 36, step_sizes_cm: null,
  };
}

const DECISIONS = [
  decision(0, 0.5, "reach"),
  decision(1, 0.75, "reach"),
  decision(2, 1.0, "grasp"),
  decision(3, 1.25, "lift"),
];

describe("decisionAt", () => {
  it("has no decision in force during the settle steps", () => {
    expect(decisionAt(DECISIONS, 0)).toBe(-1);
    expect(decisionAt(DECISIONS, 0.49)).toBe(-1);
  });

  it("takes the decision whose instant has just passed", () => {
    expect(decisionAt(DECISIONS, 0.5)).toBe(0);
    expect(decisionAt(DECISIONS, 0.74)).toBe(0);
    expect(decisionAt(DECISIONS, 0.75)).toBe(1);
    expect(decisionAt(DECISIONS, 1.24999)).toBe(2);
  });

  it("holds the last decision to the end of the video", () => {
    expect(decisionAt(DECISIONS, 9)).toBe(3);
  });

  it("is exact on a float that should land on the boundary", () => {
    // 0.25 s steps are not representable, so the lookup must tolerate the last bit.
    const t = 3 * 0.25 + 0.5;
    expect(decisionAt(DECISIONS, t)).toBe(3);
  });

  it("agrees with a linear scan on every decision of a long episode", () => {
    const many = Array.from({ length: 44 }, (_, i) => decision(i, 0.5 + i * 0.25, "carry"));
    for (let i = 0; i < many.length; i += 1) {
      const t = seekTimeFor(many, i);
      let expected = -1;
      many.forEach((d, j) => { if (d.t <= t + 1e-9) expected = j; });
      expect(decisionAt(many, t)).toBe(expected);
      expect(decisionAt(many, t)).toBe(i);
    }
  });

  it("returns -1 for an episode with no decisions at all", () => {
    expect(decisionAt([], 5)).toBe(-1);
  });
});

describe("seekTimeFor", () => {
  it("lands inside the decision it names, never on the one before", () => {
    const t = seekTimeFor(DECISIONS, 2);
    expect(t).toBeGreaterThan(DECISIONS[2].t);
    expect(decisionAt(DECISIONS, t)).toBe(2);
  });

  it("clamps out-of-range indices", () => {
    expect(decisionAt(DECISIONS, seekTimeFor(DECISIONS, -5))).toBe(0);
    expect(decisionAt(DECISIONS, seekTimeFor(DECISIONS, 99))).toBe(3);
  });
});

const EPISODE = {
  total_frames: 40, control_rate: 20, decisions: DECISIONS,
} as unknown as Episode;

describe("stageSegments", () => {
  it("merges consecutive decisions that share a sub-goal", () => {
    const segments = stageSegments(EPISODE);
    expect(segments.map((s) => s.stage)).toEqual(["reach", "grasp", "lift"]);
    expect(segments[0].fromDecision).toBe(0);
    expect(segments[0].toDecision).toBe(1);
  });

  it("covers the video with no gaps and runs the last stage to the end", () => {
    const segments = stageSegments(EPISODE);
    expect(segments[0].from).toBe(0.5);
    for (let i = 1; i < segments.length; i += 1) {
      expect(segments[i].from).toBe(segments[i - 1].to);
    }
    expect(segments[segments.length - 1].to).toBe(episodeDuration(EPISODE));
  });

  it("gives every stage its own colour and falls back for an unknown one", () => {
    expect(stageColor("reach")).not.toBe(stageColor("grasp"));
    expect(stageColor(null)).toMatch(/^#/);
  });
});

describe("refusedCandidate", () => {
  it("is null when the guard let the answer through", () => {
    expect(refusedCandidate({ asked: true, closed: true, refused: false })).toBeNull();
    expect(refusedCandidate({ asked: false, closed: false, refused: false })).toBeNull();
  });

  it("names the model's own answer when the guard overrode it", () => {
    // The model asked to close, the guard kept the fingers open: the struck-through bar is
    // `true`, and `questions.grip.choice` (the executed answer) is `false`.
    expect(refusedCandidate({ asked: true, closed: false, refused: true })).toBe("true");
    expect(refusedCandidate({ asked: false, closed: true, refused: true })).toBe("false");
  });

  it("says nothing rather than guessing when the record does not carry the answer", () => {
    expect(refusedCandidate({ asked: null, closed: true, refused: true })).toBeNull();
  });
});

describe("formatting", () => {
  it("never rounds a probability into a certainty it did not have", () => {
    expect(fmtProbability(1)).toBe("100%");
    expect(fmtProbability(0.9998)).toBe(">99.9%");
    expect(fmtProbability(0.0002)).toBe("<0.1%");
    expect(fmtProbability(0)).toBe("0%");
    expect(fmtProbability(0.5)).toBe("50.0%");
  });

  it("shortens the four task sentences into four different-looking labels", () => {
    const labels = [
      "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
      "pick up the black bowl between the plate and the ramekin and place it on the plate",
      "pick up the black bowl on the cookie box and place it on the plate",
      "pick up the black bowl on the wooden cabinet and place it on the plate",
    ].map(shortLabel);
    expect(labels).toEqual([
      "bowl in the top drawer of the wooden cabinet",
      "bowl between the plate and the ramekin",
      "bowl on the cookie box",
      "bowl on the wooden cabinet",
    ]);
    expect(new Set(labels).size).toBe(4);
  });

  it("leaves a sentence it does not recognise alone", () => {
    expect(shortLabel("open the drawer")).toBe("open the drawer");
  });

  it("prints the no-wall sentinel as a word, not as a measurement", () => {
    expect(fmtRoom(9990)).toBe("open");
    expect(fmtRoom(3.14)).toBe("3.1 cm");
  });

  it("signs centimetres and refuses a negative zero", () => {
    expect(fmtCm(14.8)).toBe("+14.8");
    expect(fmtCm(-4)).toBe("−4.0");
    expect(fmtCm(-0.02)).toBe("+0.0");
  });
});
