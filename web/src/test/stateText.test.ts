/**
 * The line ↔ question mapping, pinned against a **real recorded state text** — the first decision
 * of the drawer episode, copied verbatim out of `public/replays/drawer/episode.json`.
 *
 * Verbatim is the point: this mapping is a lexical reading of a template that lives in another
 * repository, so the test that matters is "does it still classify what the server actually
 * prints", not "does it classify what I remember it printing".
 */
import { describe, expect, it } from "vitest";
import { linesForQuestion, parseStateText, unmapped } from "../data/stateText";

const STATE = [
  "Robot: Franka Panda, gripper-relative frame; x right, y forward, z up; distances in cm.",
  "Task: pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
  "Target: bowl_1 (chosen at t=0). Destination: plate_1.",
  "Subgoal so far: reach (approach). Done: nothing. Decision 1 of 44; 43 left.",
  "Gripper: open 7.7 cm; holding nothing.",
  "Wrist yaw error: +90.0 deg [tolerance 8.0]",
  "Target rests inside cabinet_1, whose walls reach 6.5 cm above it; it is lifted clear of them before it travels.",
  "Grasp candidates (outer finger needs 1.0 cm):",
  "  A: +x, turn -90, room 3.1 cm -> fits",
  "  B: -x, turn +90, room 1.2 cm -> fits",
  "  C: +x, turn -135, room 2.8 cm -> fits",
  "  D: +y, turn +0, room 0.0 cm -> blocked by cabinet_1",
  "  E: -y, turn +180, room 0.1 cm -> blocked by cabinet_1",
  "  F: +y, turn -45, room 0.2 cm -> blocked by cabinet_1",
  "  G: -y, turn +135, room 0.0 cm -> blocked by cabinet_1",
  "  H: -x, turn +45, room 0.8 cm -> blocked by cabinet_1",
  "Waypoint (approach: 8 cm above the rim of bowl_1, +x side): x +34.4, y -16.0, z -1.7   [tolerance 0.3; arrived within 1.3]",
  "  x: +34.4 is outside tolerance -> not aligned",
  "  y: -16.0 is outside tolerance -> not aligned",
  "  z: -1.7 is outside tolerance -> not aligned",
  "Largest remaining offset: 34.4 cm (large range: 3.50 and above).",
  "Step bands: small below 1.17 cm, medium 1.17 to 3.50 cm, large 3.50 and above; one step executes 0.5 / 1.7 / 5.0 cm.",
  "Other objects: bowl_2 x +22.1 y -30.0 z -5.8; cookies_1 x +28.7 y -0.1 z -27.5; ramekin_1 x +2.1 y +17.6 z -28.5; plate_1 x +27.2 y +18.0 z -28.2.",
  "Events: t=0 start, waypoint 38.0 cm away",
  "Attempts: grasps 0, held 0. Closest so far 38.0 cm at t=0. Moved 0.0 cm, net 0.0 cm (not looping).",
  "Last 3: none yet.",
].join("\n");

const LINES = parseStateText(STATE);

describe("parseStateText", () => {
  it("keeps every line, in order, verbatim", () => {
    expect(LINES).toHaveLength(26);
    expect(LINES.map((l) => l.text).join("\n")).toBe(STATE);
    LINES.forEach((line, i) => expect(line.index).toBe(i));
  });

  it("classifies the axis offset lines and nothing else as axes", () => {
    const axes = LINES.filter((l) => l.kind === "axis");
    expect(axes.map((l) => l.key)).toEqual(["x", "y", "z"]);
  });

  it("does not mistake a lettered grasp row for an axis row", () => {
    const rim = LINES.filter((l) => l.kind === "rim-row");
    expect(rim.map((l) => l.key)).toEqual(["A", "B", "C", "D", "E", "F", "G", "H"]);
  });
});

describe("linesForQuestion", () => {
  it("ties move_x to the x offset line (and not to y or z)", () => {
    const lines = linesForQuestion(LINES, "move_x").map((i) => LINES[i].text);
    expect(lines).toContain("  x: +34.4 is outside tolerance -> not aligned");
    expect(lines.join("\n")).not.toContain("  y:");
    expect(lines.join("\n")).not.toContain("  z:");
  });

  it("gives size_z the z line and the bands it is judged against", () => {
    const kinds = linesForQuestion(LINES, "size_z").map((i) => LINES[i].kind);
    expect(kinds).toContain("axis");
    expect(kinds).toContain("bands");
    expect(kinds).toContain("largest");
    const axis = linesForQuestion(LINES, "size_z").filter((i) => LINES[i].kind === "axis");
    expect(axis.map((i) => LINES[i].key)).toEqual(["z"]);
  });

  it("ties rim to the whole candidate block, its header and what the target rests in", () => {
    const lines = linesForQuestion(LINES, "rim").map((i) => LINES[i].kind);
    expect(lines.filter((k) => k === "rim-row")).toHaveLength(8);
    expect(lines).toContain("rim-head");
    expect(lines).toContain("rests");
  });

  it("ties yaw to the wrist error line alone", () => {
    const lines = linesForQuestion(LINES, "yaw");
    expect(lines).toHaveLength(1);
    expect(LINES[lines[0]].text).toMatch(/^Wrist yaw error:/);
  });

  it("ties grip to the gripper line and grounding to the task sentence", () => {
    expect(linesForQuestion(LINES, "grip").map((i) => LINES[i].kind)).toContain("gripper");
    const target = linesForQuestion(LINES, "target").map((i) => LINES[i].kind);
    expect(target).toEqual(["task", "grounding"]);
  });

  it("gives every asked question at least one line", () => {
    const asked = ["move_x", "move_y", "move_z", "size_x", "size_y", "size_z",
      "yaw", "rim", "grip", "subgoal", "target", "destination"];
    for (const qid of asked) {
      expect(linesForQuestion(LINES, qid).length, `${qid} has no line`).toBeGreaterThan(0);
    }
  });

  it("leaves only the context lines unmapped", () => {
    expect(unmapped(LINES).map((l) => l.kind)).toEqual(["header", "objects"]);
  });
});
