/**
 * Bundle validation: a JSON blob is an {@link Episode}, or it is a list of reasons why not.
 *
 * Why validate at all, when the file was written half an hour ago by `recorder.py`? Because this
 * page is the only reader of a format two programs write -- the recorder now, a live console
 * later -- and the failure it protects against is not a malformed file but a *silently changed*
 * one: a renamed field shows up as an empty panel and a plausible-looking replay, which is the
 * worst way to be wrong about what a model answered. A bundle that fails here is named, with the
 * field that is missing, on the page.
 *
 * The rules are the ones the UI actually depends on. Everything else is carried through as-is.
 */
import type { Episode } from "./types";

/** The newest bundle this page reads. 2 added the pose table and the 3D scene; a version-1
 *  bundle is still read, and replays on its videos alone. */
export const SUPPORTED_SCHEMA = 2;
export const SUPPORTED_SCHEMAS: readonly number[] = [1, 2];

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function checkDecision(d: unknown, i: number, problems: string[]): void {
  const where = `decisions[${i}]`;
  if (!isRecord(d)) {
    problems.push(`${where}: not an object`);
    return;
  }
  if (typeof d.control_step !== "number") problems.push(`${where}.control_step: expected a number`);
  if (typeof d.t !== "number") problems.push(`${where}.t: expected a number`);
  if (typeof d.state !== "string") problems.push(`${where}.state: expected the state text`);
  if (!isRecord(d.questions)) {
    problems.push(`${where}.questions: expected an object of questions`);
    return;
  }
  for (const [qid, group] of Object.entries(d.questions)) {
    if (!isRecord(group) || !Array.isArray(group.candidates)) {
      problems.push(`${where}.questions.${qid}: expected {candidates, choice}`);
      continue;
    }
    for (const c of group.candidates) {
      if (!isRecord(c) || typeof c.id !== "string" || typeof c.p !== "number") {
        problems.push(`${where}.questions.${qid}: a candidate is not {id, p}`);
        break;
      }
    }
  }
}

/** `{ok: true, episode}` or `{ok: false, problems}` -- never a throw, because the caller draws
 *  the problems rather than a blank page. */
export function validateEpisode(raw: unknown): { ok: true; episode: Episode } | { ok: false; problems: string[] } {
  const problems: string[] = [];
  if (!isRecord(raw)) return { ok: false, problems: ["episode.json: not a JSON object"] };

  if (!SUPPORTED_SCHEMAS.includes(raw.schema_version as number)) {
    problems.push(
      `schema_version: this page reads ${SUPPORTED_SCHEMAS.join(" and ")}, the bundle says ${JSON.stringify(raw.schema_version)}`,
    );
  }
  if (raw.qpos != null) {
    const q = raw.qpos;
    if (!isRecord(q) || typeof q.path !== "string" || typeof q.frames !== "number" || typeof q.nq !== "number") {
      problems.push("qpos: expected {path, frames, nq, dtype}");
    } else if (q.dtype !== undefined && q.dtype !== "<f4") {
      problems.push(`qpos.dtype: this page reads little-endian float32, the bundle says ${JSON.stringify(q.dtype)}`);
    }
  }
  if (raw.scene != null && (!isRecord(raw.scene) || typeof raw.scene.hash !== "string")) {
    problems.push("scene: expected {hash, nq}");
  }
  for (const key of ["id", "instruction", "policy", "suite"] as const) {
    if (typeof raw[key] !== "string") problems.push(`${key}: expected a string`);
  }
  if (typeof raw.success !== "boolean") problems.push("success: expected a boolean");
  if (typeof raw.control_rate !== "number" || raw.control_rate <= 0) {
    problems.push("control_rate: expected a positive number of control steps per second");
  }
  if (!isRecord(raw.media) || !isRecord(raw.media.agentview)) {
    problems.push("media.agentview: every bundle carries the scene camera");
  }
  if (!Array.isArray(raw.decisions)) {
    problems.push("decisions: expected an array");
  } else {
    if (raw.decisions.length === 0) problems.push("decisions: the episode recorded none");
    raw.decisions.forEach((d, i) => checkDecision(d, i, problems));
  }

  if (problems.length > 0) return { ok: false, problems };
  return { ok: true, episode: raw as unknown as Episode };
}
