/**
 * The shape of an episode bundle, and of the one thing the page ever asks of a data source.
 *
 * A bundle is `recorder.py`'s output: `episode.json` beside `agentview.mp4` (+ `wrist.mp4`), in a
 * directory of its own. Nothing here is Vite-specific or fetch-specific on purpose -- a live
 * console streaming the same objects over a socket produces the same `Decision`s, which is the
 * whole reason `EpisodeSource` exists (see `source.ts` and `../../PROTOCOL.md`).
 */

/** One candidate of one question, and how sure the model was of it. */
export interface Candidate {
  id: string;
  p: number;
}

/** One question, as the model answered it. `overridden` is an operator's forced answer -- never
 *  true in a replay, and the reason the field is here is that a live session's is. */
export interface QuestionAnswer {
  candidates: Candidate[];
  choice: string | null;
  overridden: boolean;
}

/** One lettered grasp option, as the state text lists it. */
export interface RimCandidate {
  letter: string;
  side: string;
  turn_deg: number;
  room_cm: number;
  verdict: string;
  fits: boolean;
  /** The plan has already stood on this rim point and come away with nothing. */
  tried?: boolean;
}

/** The grip guard's verdict: what the model asked for, what the fingers were told, and whether
 *  the table refused the change. */
export interface GripLatch {
  asked: boolean | null;
  closed: boolean | null;
  refused: boolean;
}

/** Which object the sentence meant. Carried only by the decision that paid for the grounding
 *  forward -- the first, and again after a failed grasp attempt. */
export interface Grounding {
  mode: string | null;
  source: string | null;
  sources: Record<string, string> | null;
  target: string | null;
  destination: string | null;
  model: Record<string, string | null> | null;
  model_agrees: boolean | null;
  rule: Record<string, string | null> | null;
  probability: Record<string, number> | null;
  min_probability: number | null;
  regrounded: boolean | null;
}

export interface Decision {
  /** Position in the episode's decision list, 0-based. */
  index: number;
  /** The policy's own decision counter (`meta.step`); equals `index` in a replay. */
  step: number | null;
  /** The control step this decision was asked at -- the video's frame number. */
  control_step: number;
  /** The same instant in seconds of video: `control_step / control_rate`. */
  t: number;
  /** The whole paragraph the model read. */
  state: string;
  questions: Record<string, QuestionAnswer>;
  grip_latch: GripLatch;
  grounding: Grounding | null;
  subgoal: string | null;
  substage: string | null;
  target: string | null;
  destination: string | null;
  waypoint: string | null;
  waypoint_cm: number[] | null;
  rim_candidates: RimCandidate[];
  rim_chosen: RimCandidate | null;
  /** The 7-d LIBERO action this decision composed into, held for `execute_steps` control steps. */
  action: number[];
  forward_passes: number | null;
  candidate_paths: number | null;
  step_sizes_cm: Record<string, number> | null;
}

export interface MediaTrack {
  path: string;
  width: number;
  height: number;
  fps: number;
  frames: number;
  codec: string;
}

export interface Episode {
  schema_version: number;
  id: string;
  suite: string;
  task_index: number;
  init_state_index: number;
  instruction: string;
  title: string;
  note: string;
  policy: string;
  checkpoint_revision: string | null;
  checkpoint_repo: string | null;
  questions_version: string | null;
  success: boolean;
  terminated_by: string;
  error: string | null;
  steps: number;
  control_rate: number;
  wait_steps: number;
  execute_steps: number;
  max_steps: number;
  max_decisions: number;
  total_frames: number;
  duration_s: number;
  recorded_at: string;
  wall_seconds: number;
  media: Partial<Record<"agentview" | "wrist", MediaTrack>>;
  /** The still inside the bundle that the strip shows and the `<video>` uses as its poster. */
  poster?: string | null;
  decisions: Decision[];
}

/** One row of `replays/index.json`: enough to draw a strip item without loading the episode. */
export interface EpisodeIndexEntry {
  id: string;
  title: string;
  instruction: string;
  note: string;
  success: boolean;
  decisions: number;
  max_decisions: number;
  task_index: number;
  init_state_index: number;
  suite: string;
  poster?: string | null;
}
