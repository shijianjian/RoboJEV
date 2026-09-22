/**
 * An episode bundle as the viewer's frame clock reads it.
 *
 * robopp's `viewer/record.ts` reads a run's parquet into per-frame columns; a bundle already *is*
 * one frame per video frame (`t = control_step / control_rate`), so this is the same shape built
 * from `episode.json` and `qpos.bin`: the pose to put the 3D scene in, the action executed, and
 * which decision was in force, for every frame the videos hold.
 */
import { decisionAt } from "../data/lookup";
import type { Episode, QposTable } from "../data/types";

/** The action's seven components, in LIBERO's order - robopp's `ACTION_LABELS`. */
export const ACTION_LABELS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"];

export interface Trajectory {
  numFrames: number;
  fps: number;
  /** Frame-major, `nq` wide; empty when the bundle has no pose table. */
  qpos: Float32Array;
  nq: number;
  /** Frame-major, 7 wide; NaN on a frame that executed no policy action (the settle steps and
   *  the terminal frame), which the plot draws as a gap. */
  action: Float32Array;
  /** The decision in force at each frame, or -1 before the first. */
  decision: Int32Array;
  isWait: Uint8Array;
}

/** `qpos.bin`, checked against the table the bundle says it is. */
export function parseQpos(buf: ArrayBuffer, spec: QposTable): Float32Array {
  const want = spec.frames * spec.nq * 4;
  if (buf.byteLength !== want) {
    throw new Error(`${spec.path}: ${buf.byteLength} bytes, the bundle says ${spec.frames}x${spec.nq} float32 (${want})`);
  }
  // Little-endian by the format, which is every platform a browser runs on; read through a
  // DataView anyway so a big-endian one would still get the right numbers.
  const view = new DataView(buf);
  const out = new Float32Array(spec.frames * spec.nq);
  for (let i = 0; i < out.length; i += 1) out[i] = view.getFloat32(i * 4, true);
  return out;
}

export async function fetchQpos(url: string, spec: QposTable): Promise<Float32Array> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${spec.path}: ${res.status}`);
  return parseQpos(await res.arrayBuffer(), spec);
}

/** The bundle as frames. `qpos` is the parsed table, or null for a bundle without one. */
export function trajectoryOf(episode: Episode, qpos: Float32Array | null): Trajectory {
  const numFrames = Math.max(episode.total_frames, 1);
  const fps = episode.control_rate;
  const action = new Float32Array(numFrames * 7).fill(Number.NaN);
  const decision = new Int32Array(numFrames).fill(-1);
  const isWait = new Uint8Array(numFrames);
  const last = numFrames - 1;
  for (let f = 0; f < numFrames; f += 1) {
    isWait[f] = f < episode.wait_steps ? 1 : 0;
    const d = decisionAt(episode.decisions, f / fps);
    decision[f] = d;
    // The terminal frame is where the arm ended up; there is no action for it.
    if (d >= 0 && f < last) {
      const a = episode.decisions[d].action;
      for (let k = 0; k < 7; k += 1) action[f * 7 + k] = a[k] ?? Number.NaN;
    }
  }
  const nq = episode.qpos?.nq ?? 0;
  const usable = qpos !== null && nq > 0 && qpos.length >= numFrames * nq;
  return {
    numFrames, fps, action, decision, isWait,
    qpos: usable ? qpos : new Float32Array(0),
    nq: usable ? nq : 0,
  };
}

/** One frame's pose - robopp's `qposAt`. */
export function qposAt(t: Trajectory, frame: number): Float32Array {
  const f = Math.min(Math.max(Math.floor(frame), 0), t.numFrames - 1);
  return t.qpos.slice(f * t.nq, (f + 1) * t.nq);
}
