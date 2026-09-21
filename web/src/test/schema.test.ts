/**
 * Bundle validation, and — the part that matters — validation run against the **bundles this
 * site actually ships**, read off disk. A schema test that only checks hand-written fixtures
 * proves the validator is self-consistent; this one proves the recorder and the page agree.
 */
import { readdirSync, readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { validateEpisode } from "../data/schema";
import { decisionAt } from "../data/lookup";
import { orderQuestions } from "../data/questions";
import { DEFAULT_EPISODE } from "../data/route";

const ROOT = join(__dirname, "..", "..", "public", "replays");

function minimal(): Record<string, unknown> {
  return {
    schema_version: 1, id: "x", suite: "libero_spatial", instruction: "pick it up", policy: "robojev",
    success: true, control_rate: 20, media: { agentview: { path: "agentview.mp4" } },
    decisions: [{
      control_step: 10, t: 0.5, state: "Task: pick it up",
      questions: { grip: { candidates: [{ id: "true", p: 1 }], choice: "true", overridden: false } },
    }],
  };
}

describe("validateEpisode", () => {
  it("accepts a minimal well-formed bundle", () => {
    expect(validateEpisode(minimal()).ok).toBe(true);
  });

  it("names the version it cannot read rather than guessing", () => {
    const bad = { ...minimal(), schema_version: 2 };
    const result = validateEpisode(bad);
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.problems[0]).toMatch(/schema_version/);
  });

  it("refuses an episode with no decisions and one with no camera", () => {
    const noDecisions = validateEpisode({ ...minimal(), decisions: [] });
    expect(noDecisions.ok).toBe(false);
    const noCamera = validateEpisode({ ...minimal(), media: {} });
    expect(noCamera.ok).toBe(false);
    if (!noCamera.ok) expect(noCamera.problems.join()).toMatch(/agentview/);
  });

  it("names the decision and the field when a candidate is malformed", () => {
    const raw = minimal();
    (raw.decisions as Record<string, unknown>[])[0].questions = {
      grip: { candidates: [{ id: "true" }], choice: "true" },
    };
    const result = validateEpisode(raw);
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.problems[0]).toMatch(/decisions\[0\]\.questions\.grip/);
  });

  it("collects every problem rather than stopping at the first", () => {
    const result = validateEpisode({ schema_version: 9, decisions: [] });
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.problems.length).toBeGreaterThan(3);
  });

  it("refuses a blob that is not an object at all", () => {
    expect(validateEpisode(null).ok).toBe(false);
    expect(validateEpisode([1, 2]).ok).toBe(false);
  });
});

const bundles = existsSync(ROOT)
  ? readdirSync(ROOT, { withFileTypes: true }).filter((e) => e.isDirectory()).map((e) => e.name)
  : [];

describe.runIf(bundles.length > 0)("the bundles this site ships", () => {
  it("has an index listing every bundle directory, each with a poster on disk", () => {
    const index = JSON.parse(readFileSync(join(ROOT, "index.json"), "utf-8")) as
      { id: string; poster?: string | null }[];
    expect(index.map((e) => e.id).sort()).toEqual([...bundles].sort());
    for (const entry of index) {
      // The strip is four pictures; a row whose picture 404s is the one thing it cannot survive.
      expect(entry.poster, `${entry.id} has no poster`).toBeTruthy();
      expect(existsSync(join(ROOT, entry.id, entry.poster!)), `${entry.id}/${entry.poster}`).toBe(true);
    }
  });

  it("opens on an episode that exists", () => {
    expect(bundles).toContain(DEFAULT_EPISODE);
  });

  it.each(bundles)("%s validates, and its decisions are a monotone clock", (name) => {
    const raw = JSON.parse(readFileSync(join(ROOT, name, "episode.json"), "utf-8")) as unknown;
    const result = validateEpisode(raw);
    if (!result.ok) throw new Error(`${name}: ${result.problems.join("; ")}`);
    const episode = result.episode;

    expect(episode.decisions.length).toBeGreaterThan(0);
    expect(episode.decisions.length).toBeLessThanOrEqual(episode.max_decisions);
    episode.decisions.forEach((d, i) => {
      expect(d.index).toBe(i);
      // The recorder's one invariant: video time is the control step over the control rate.
      expect(d.t).toBeCloseTo(d.control_step / episode.control_rate, 6);
      if (i > 0) expect(d.t).toBeGreaterThan(episode.decisions[i - 1].t);
      // Every decision is inside the video, and after the settle steps.
      expect(d.control_step).toBeGreaterThanOrEqual(episode.wait_steps);
      expect(d.control_step).toBeLessThan(episode.total_frames);
      expect(d.action).toHaveLength(7);
      expect(d.state.length).toBeGreaterThan(200);
      // Every question has a choice, and that choice is one of its own candidates.
      for (const [qid, group] of Object.entries(d.questions)) {
        expect(group.candidates.length, `${qid}`).toBeGreaterThan(1);
        expect(group.candidates.map((c) => c.id), `${qid} choice`).toContain(group.choice);
        const total = group.candidates.reduce((a, c) => a + c.p, 0);
        expect(total, `${qid} probabilities sum`).toBeCloseTo(1, 2);
      }
    });

    // Every decision is reachable from its own instant, and the settle steps have none.
    expect(decisionAt(episode.decisions, 0)).toBe(-1);
    episode.decisions.forEach((d, i) => expect(decisionAt(episode.decisions, d.t)).toBe(i));

    // The ten motor questions are all asked, and grounding happens once, on the first decision.
    const motor = orderQuestions(Object.keys(episode.decisions[1].questions));
    expect(motor).toEqual([
      "move_x", "move_y", "move_z", "size_x", "size_y", "size_z", "yaw", "rim", "grip", "subgoal",
    ]);
    expect(episode.decisions[0].grounding).not.toBeNull();
  });
});
