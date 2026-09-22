/**
 * What a build publishes beside the app: runs, the compiled scenes they use, and the catalogue.
 *
 * Vite copies nothing from outside `src/`; this step does, from the checkout's three data
 * directories - `showcase/` (tracked: the runs GitHub carries, their scenes and their tasks'
 * catalogue entries), `runs/` (console saves) and `data/` (the full catalogue and scenes):
 *
 *     VITE_SITE=pages npm run build   # GitHub Pages: showcase/ only - the runs web/showcase.json
 *                                     # names, their scenes, their tasks' catalogue entries
 *     npm run build                   # local: showcase/ + runs/ + $ROBOJEV_HOME/replays, the
 *                                     # full catalogue, and every scene those runs use
 *
 * Output: `dist/replays/<id>/…` + `dist/replays/index.json`, `dist/scenes/<hash>/…`,
 * `dist/catalogue/<suite>/<task>/…` + `dist/catalogue/index.json`.
 */
import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const REPO = resolve(WEB, "..");
const argOut = process.argv.indexOf("--out");
const DIST = argOut > 0 ? resolve(process.argv[argOut + 1]) : join(WEB, "dist");
const PAGES = process.env.VITE_SITE === "pages";
const HOME = process.env.ROBOJEV_HOME || process.env.ROBOPP_HOME || join(homedir(), ".robojev");
const showcase = JSON.parse(readFileSync(join(WEB, "showcase.json"), "utf-8"));
const BUNDLE_FILES = ["episode.json", "agentview.mp4", "wrist.mp4", "poster.jpg", "qpos.bin"];

const SHOW = join(REPO, "showcase");
const replayRoots = PAGES ? [join(SHOW, "replays")] : [join(SHOW, "replays"), join(REPO, "runs"), join(HOME, "replays")];
const sceneRoots = PAGES ? [join(SHOW, "scenes")] : [join(SHOW, "scenes"), join(REPO, "data", "scenes"), join(HOME, "scenes")];
const catalogueRoots = PAGES ? [join(SHOW, "catalogue")] : [join(SHOW, "catalogue"), join(REPO, "data", "catalogue"), join(HOME, "catalogue")];
const readJson = (path) => JSON.parse(readFileSync(path, "utf-8"));

let bytes = 0;
const copy = (src, dst) => { mkdirSync(dirname(dst), { recursive: true }); copyFileSync(src, dst); bytes += statSync(src).size; };
const copyTree = (src, dst) => {
  for (const entry of readdirSync(src, { withFileTypes: true })) {
    const s = join(src, entry.name), d = join(dst, entry.name);
    if (entry.isDirectory()) copyTree(s, d); else copy(s, d);
  }
};

// ------------------------------------------------------------------------------------ runs
const found = new Map();
for (const root of replayRoots.filter((d) => existsSync(d))) {
  for (const name of readdirSync(root)) {
    if (!found.has(name) && existsSync(join(root, name, "episode.json"))) found.set(name, join(root, name));
  }
}
const ids = PAGES ? showcase.runs
  : [...showcase.runs.filter((id) => found.has(id)), ...[...found.keys()].filter((id) => !showcase.runs.includes(id)).sort()];
const missing = ids.filter((id) => !found.has(id));
if (missing.length > 0) {
  console.error(`publish-replays: web/showcase.json names runs not in showcase/replays: ${missing.join(", ")}`);
  process.exit(1);
}
for (const dir of ["replays", "scenes", "catalogue"]) rmSync(join(DIST, dir), { recursive: true, force: true });
const index = [];
const scenes = new Set();
const tasks = new Set();
for (const id of ids) {
  const dir = found.get(id);
  const bundle = readJson(join(dir, "episode.json"));
  for (const name of BUNDLE_FILES) if (existsSync(join(dir, name))) copy(join(dir, name), join(DIST, "replays", id, name));
  if (bundle.scene?.hash) scenes.add(bundle.scene.hash);
  tasks.add(`${bundle.suite}/${bundle.task_index}`);
  index.push({
    id, title: bundle.title, instruction: bundle.instruction, note: bundle.note, success: bundle.success,
    decisions: bundle.decisions.length, max_decisions: bundle.max_decisions, task_index: bundle.task_index,
    init_state_index: bundle.init_state_index, suite: bundle.suite, policy: bundle.policy, poster: bundle.poster ?? null,
  });
}
writeFileSync(join(DIST, "replays", "index.json"), JSON.stringify(index, null, 1) + "\n");

// ------------------------------------------------------------------------------------ scenes
for (const hash of scenes) {
  const root = sceneRoots.find((r) => existsSync(join(r, hash, "scene.xml")));
  if (root === undefined) { console.error(`publish-replays: no scene ${hash}; its runs replay without the 3D view`); continue; }
  copyTree(join(root, hash), join(DIST, "scenes", hash));
}

// ------------------------------------------------------------------------------------ catalogue
const suites = new Map();
for (const root of catalogueRoots.filter((d) => existsSync(d))) {
  for (const suite of readdirSync(root)) {
    for (const task of existsSync(join(root, suite)) ? readdirSync(join(root, suite)) : []) {
      const dir = join(root, suite, task);
      if (!existsSync(join(dir, "task.json")) || (PAGES && !tasks.has(`${suite}/${task}`))) continue;
      const byTask = suites.get(suite) ?? new Map();
      if (byTask.has(Number(task))) continue;
      byTask.set(Number(task), readJson(join(dir, "task.json")));
      suites.set(suite, byTask);
      for (const name of ["task.json", "thumb.png", "init_qpos.json"]) {
        if (existsSync(join(dir, name))) copy(join(dir, name), join(DIST, "catalogue", suite, task, name));
      }
    }
  }
}
const order = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"];
const catalogue = [...suites.keys()].sort((a, b) => (order.indexOf(a) + 99) % 99 - (order.indexOf(b) + 99) % 99 || a.localeCompare(b))
  .map((suite) => ({ suite, tasks: [...suites.get(suite).keys()].sort((a, b) => a - b).map((t) => suites.get(suite).get(t)) }));
mkdirSync(join(DIST, "catalogue"), { recursive: true });
writeFileSync(join(DIST, "catalogue", "index.json"), JSON.stringify(catalogue) + "\n");

console.log(`publish-replays: ${PAGES ? "GitHub Pages (showcase/)" : "local"} — ${index.length} runs, ${scenes.size} scenes, `
  + `${catalogue.reduce((n, s) => n + s.tasks.length, 0)} catalogued tasks, ${(bytes / 1e6).toFixed(1)} MB into ${relative(REPO, DIST)}`);
