/** A bundle as the frame clock the 3D scene and the videos share. */
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import type { Decision, Episode } from "../data/types";
import { parseQpos, qposAt, trajectoryOf } from "../viewer/record";

const ROOT = join(__dirname, "..", "..", "..", "showcase", "replays");

function decision(index: number, extra: Partial<Decision> = {}): Decision {
  return {
    index, step: index, control_step: 10 + index * 5, t: (10 + index * 5) / 20, state: "",
    questions: {
      move_x: { candidates: [{ id: "-", p: 0.2 }, { id: "hold", p: 0.2 }, { id: "+", p: 0.6 }], choice: "hold", overridden: true },
      grip: { candidates: [{ id: "true", p: Number.NaN }, { id: "false", p: 0.5 }], choice: "true", overridden: false },
    },
    grip_latch: { asked: true, closed: false, refused: false },
    grounding: null, subgoal: "reach", substage: "approach", target: null, destination: null,
    waypoint: null, waypoint_cm: [1, -2, 0.5], rim_candidates: [], rim_chosen: null,
    action: [0.1, 0, 0, 0, 0, 0, -1], forward_passes: 1, candidate_paths: null, step_sizes_cm: null,
    ...extra,
  };
}

describe("the frame clock", () => {
  const episode = {
    total_frames: 22, control_rate: 20, wait_steps: 10, decisions: [decision(0), decision(1)],
    qpos: { path: "qpos.bin", frames: 22, nq: 2, dtype: "<f4" },
  } as unknown as Episode;

  it("reads qpos.bin as little-endian float32 and refuses one of the wrong size", () => {
    const buf = new ArrayBuffer(8);
    new DataView(buf).setFloat32(0, 1.5, true);
    new DataView(buf).setFloat32(4, -2, true);
    expect(Array.from(parseQpos(buf, { path: "q", frames: 1, nq: 2 }))).toEqual([1.5, -2]);
    expect(() => parseQpos(buf, { path: "q", frames: 2, nq: 2 })).toThrow(/bytes/);
  });

  it("holds each decision's action over its chunk, with gaps where there was none", () => {
    const qpos = Float32Array.from({ length: 44 }, (_, i) => i);
    const t = trajectoryOf(episode, qpos);
    expect(t.numFrames).toBe(22);
    expect(t.decision[9]).toBe(-1);
    expect(t.decision[10]).toBe(0);
    expect(t.decision[16]).toBe(1);
    expect(Number.isNaN(t.action[0])).toBe(true);
    expect(t.action[12 * 7]).toBeCloseTo(0.1);
    expect(Number.isNaN(t.action[21 * 7])).toBe(true);       // the terminal frame
    expect(t.isWait[9]).toBe(1);
    expect(Array.from(qposAt(t, 3))).toEqual([6, 7]);
    expect(trajectoryOf(episode, null).nq).toBe(0);
  });

  it.runIf(existsSync(join(ROOT, "drawer", "qpos.bin")))("reads a shipped bundle's pose table", () => {
    const ep = JSON.parse(readFileSync(join(ROOT, "drawer", "episode.json"), "utf-8")) as Episode;
    const raw = readFileSync(join(ROOT, "drawer", "qpos.bin"));
    const table = parseQpos(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength), ep.qpos!);
    const t = trajectoryOf(ep, table);
    expect(t.nq).toBe(48);
    expect(t.decision[t.numFrames - 1]).toBe(ep.decisions.length - 1);
    expect(qposAt(t, 72)[0]).toBeCloseTo(-0.180553, 5);
  });
});
