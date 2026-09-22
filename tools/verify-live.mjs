/**
 * Prove the live console works in a real browser, against a real simulator, and photograph it.
 *
 * `tools/verify.mjs` does this for the static half against `tools/serve.mjs`; this one does it
 * against `robojev console`, which serves the same built app *and* runs an episode. It drives the
 * Dataset tab as an operator would - pick a task in the sidebar, pick weights, Start, hold one
 * candidate while it runs, let it run to its end, open the run it was saved as from the list on
 * the right; then Start and Stop one more - and asserts what a screenshot cannot: that decisions
 * arrived, that the 3D scene loaded and moved with the arm, that the episode was saved by itself
 * with its pose table and its scene, and that the forced answer is recorded as overridden.
 *
 *     node tools/verify-live.mjs [baseURL] [screenshot dir]
 *
 * Defaults: http://127.0.0.1:8765/ and ./screenshots. The console must offer weights it can run:
 * on a box whose console interpreter cannot load a checkpoint, start it with `--dev-expert` (the
 * scripted expert, for development), which is what this picks when it is offered. It leaves two
 * saved runs in the console's saves directory; delete them afterwards if they are not wanted.
 *
 * Playwright drives the **system Chrome** (H.264 for the replay half), with SwiftShader allowed so
 * a headless box still has WebGL for the 3D stage.
 */
import { mkdirSync } from "node:fs";
import { chromium } from "playwright";

const BASE = process.argv[2] ?? "http://127.0.0.1:8765/";
const SHOTS = process.argv[3] ?? "screenshots";
mkdirSync(SHOTS, { recursive: true });

let failures = 0;
function check(name, ok, detail = "") {
  console.log(`${ok ? "  ok  " : " FAIL "} ${name}${detail ? ` — ${detail}` : ""}`);
  if (!ok) failures += 1;
}

const browser = await chromium.launch({ channel: "chrome", args: ["--enable-unsafe-swiftshader", "--use-gl=swiftshader"] });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
const consoleErrors = [];
page.on("console", (m) => { if (m.type() === "error") consoleErrors.push(m.text()); });
page.on("pageerror", (e) => consoleErrors.push(String(e)));

const by = (id) => page.locator(`[data-testid="${id}"]`);
const text = async (id) => (await by(id).first().innerText()).trim();
async function waitStatus(want, timeout = 60000) {
  await page.waitForFunction((w) => {
    const el = document.querySelector('[data-testid="session-status"]');
    return el !== null && w.includes(el.innerText.trim());
  }, want, { timeout });
}
async function waitDecisionAtLeast(n, timeout = 60000) {
  await page.waitForFunction((want) => {
    const m = /decision\s+(\d+)/.exec(document.querySelector('[data-testid="decision-meta"]')?.innerText ?? "");
    return m !== null && Number(m[1]) >= want;
  }, n, { timeout });
}
const topRun = () => page.locator('[data-testid="dataset-runs"] [data-testid^="dataset-run-"]').first().getAttribute("data-testid");
const liveScene = '[data-testid="live-scene"]';
async function liveSceneReady() {
  await page.waitForFunction((sel) => document.querySelector(`${sel} [data-testid="scene-status"]`)?.innerText.trim() === "ready",
    liveScene, { timeout: 120000 });
}
const canvasShot = () => page.locator(`${liveScene} canvas.scene`).screenshot();

// ------------------------------------------------------------------------ idle
await page.goto(BASE, { waitUntil: "domcontentloaded" });
await page.waitForSelector('[data-testid="side-tab-runs"][aria-selected="true"]');
check("the console's page opens on the Runs tab, like the site", true);
check("a local build has both tabs, and the GitHub mark in the header",
      (await page.locator('[role="tablist"] [role="tab"]').allInnerTexts()).join(",") === "Runs,Dataset"
      && (await page.locator('.app-header [data-testid="github-link"]').getAttribute("href")) === "https://github.com/shijianjian/RoboJEV"
      && (await page.locator('.app-side [data-testid="github-link"]').count()) === 0);
const rows = await page.locator('[data-testid="run-row"]').evaluateAll((els) => els.map((e) => e.getAttribute("data-run")));
check("the Runs tab lists recorded runs only, with no live item", !rows.includes("live") && rows.includes("drawer"), rows.join(","));
await by("side-tab-dataset").click();
await waitStatus(["waiting", "done", "failed"], 15000);
check("the Dataset tab is the console: the task list, and the controls in one bar along the bottom",
      (await by("scene-sidebar").count()) === 1 && (await by("dock").count()) === 1 && (await by("session-start").count()) === 1);
const noise = ["session-step", "session-run", "session-pause", "session-save", "session-temperature", "state-toggle", "play", "tick-0"];
const present = [];
for (const id of noise) if ((await by(id).count()) > 0) present.push(id);
check("and nothing else: no Step, Run, Save, temperature, state text or scrubber", present.length === 0, present.join(","));
await page.waitForFunction(() => document.querySelector('[data-testid="task-pick-0"]')
  ?.closest("label")?.innerText.includes("pick up"), null, { timeout: 30000 });
check("the sidebar names the tasks once the console has read them", true);
const weights = await page.locator('[data-testid="session-weights"] option').allInnerTexts();
check("the bottom bar offers weights, not engines", weights.length > 0 && !weights.some((w) => /^(expert|model|jev)$/.test(w.trim())),
      weights.join(" | "));
const expert = weights.findIndex((w) => w.includes("scripted expert"));
if (expert >= 0) await by("session-weights").selectOption({ index: expert });
check("no episode, so no picture is being requested", (await by("live-agentview").count()) === 0);

// ------------------------------------------------------------------------ start, and it runs
await by("task-pick-0").check();
await page.waitForFunction(() => document.querySelector('[data-testid="dataset-run-bowl-plate"]') !== null, null, { timeout: 5000 })
  .then(() => check("the task's recorded runs are listed on the right", true), () => check("the task's recorded runs are listed on the right", false));
const before = await canvasShot();
const firstTop = await topRun();
await by("session-start").click();
await waitStatus(["running", "done"], 30000);
await liveSceneReady();
check("the live 3D scene loaded", true);
await page.waitForFunction(() => ["live-agentview", "live-wrist"].every((id) => {
  const img = document.querySelector(`[data-testid="${id}"]`);
  return img !== null && img.complete && img.naturalWidth === 256;
}), null, { timeout: 60000 });
check("both cameras decoded a real picture", true);
const instruction = await text("task-instruction");
check("the page names the task being driven", instruction.startsWith("pick up"), instruction);
await waitDecisionAtLeast(2);
check("Start runs it: decisions arrive with nobody pressing anything", true);

// One override, while it runs: a candidate bar is a control, drawn as one only on hover. The bars
// are redrawn with every decision, so the click is retried until the console reports it held.
let bars = [];
let wanted = null;
for (let attempt = 0; attempt < 10 && wanted === null; attempt += 1) {
  bars = await page.locator('[data-testid="decision-move_x"] .rj-bar').evaluateAll(
    (els) => els.map((e) => ({ tag: e.tagName, id: e.getAttribute("data-testid").replace(/^bar-move_x-/, ""),
                               chosen: e.classList.contains("rj-bar--chosen"), testid: e.getAttribute("data-testid") })));
  const pick = bars.find((b) => !b.chosen && b.id !== "hold") ?? bars.find((b) => !b.chosen);
  if (pick === undefined) break;
  await page.locator(`[data-testid="${pick.testid}"]`).click({ timeout: 1000 }).catch(() => {});
  const held = await page.waitForFunction(() => document.querySelector('[data-testid="armed-move_x"], [data-testid="overridden-move_x"]') !== null,
    null, { timeout: 1500 }).then(() => true, () => false);
  if (held) wanted = pick;
}
check("a candidate bar is a control while the episode runs", bars.length === 3 && bars.every((b) => b.tag === "BUTTON") && wanted !== null,
      bars.map((b) => b.tag).join(","));
await page.waitForTimeout(1500);
const after = await canvasShot();
check("the 3D scene moves with the arm", !before.equals(after));
await page.mouse.move(5, 5);
await page.evaluate(() => window.scrollTo(0, 0));
await page.screenshot({ path: `${SHOTS}/live-1440.png` });
await page.setViewportSize({ width: 1024, height: 820 });
await page.waitForTimeout(500);
await page.screenshot({ path: `${SHOTS}/live-1024.png` });
await page.setViewportSize({ width: 1440, height: 900 });

// ------------------------------------------------------------------------ to its end, saved by itself
await waitStatus(["done", "failed"], 240000);
check("the episode ran to its own end, successfully", (await text("session-status")) === "done");
await page.waitForFunction((was) => {
  const top = document.querySelector('[data-testid="dataset-runs"] [data-testid^="dataset-run-"]');
  return top !== null && top.getAttribute("data-testid") !== was && top.getAttribute("data-testid").startsWith("dataset-run-live-");
}, firstTop, { timeout: 120000 });
const savedId = (await topRun()).replace(/^dataset-run-/, "");
check("it was saved as a run by itself and is at the top of the task's runs", savedId.startsWith("live-"), savedId);
await page.waitForFunction(() => document.querySelector('[data-testid="session-start"]') !== null, null, { timeout: 30000 });
check("and the bar reads Start again", await by("session-start").isEnabled());
await page.screenshot({ path: `${SHOTS}/live-done-1440.png` });
const bundle = await page.evaluate(async (id) => (await fetch(`replays/${id}/episode.json`)).json(), savedId);
check("the saved run succeeded", bundle.success === true && bundle.terminated_by === "success", bundle.terminated_by);
const forced = bundle.decisions.filter((d) => d.questions.move_x?.overridden === true);
check("the answer held while it ran is recorded as overridden", forced.length === 1 && forced[0].questions.move_x.choice === wanted?.id,
      `${forced.length} overridden, move_x = ${forced[0]?.questions.move_x.choice}`);
const qposBytes = await page.evaluate(async (b) => (await (await fetch(`replays/${b.id}/${b.qpos.path}`)).arrayBuffer()).byteLength, bundle);
check("the saved run carries one pose row per video frame", qposBytes === bundle.total_frames * bundle.qpos.nq * 4,
      `${qposBytes} bytes, ${bundle.total_frames} frames`);
const sceneStatus = await page.evaluate(async (h) => (await fetch(`replays/scenes/${h}/scene.xml`)).status, bundle.scene.hash);
check("and its scene is served beside it", sceneStatus === 200, bundle.scene.hash.slice(0, 12));

// ------------------------------------------------------------------------ open it on the Runs tab
await by(`dataset-run-${savedId}`).click();
await page.waitForFunction((id) => {
  const v = document.querySelector('[data-testid="agentview"]');
  const s = document.querySelector('[data-testid="replay-scene"] [data-testid="scene-status"]');
  return location.hash === `#/${id}` && v !== null && v.readyState >= 1 && s !== null && s.innerText.trim() === "ready";
}, savedId, { timeout: 120000 });
check("the saved run opens on the Runs tab, 3D scene and all",
      (await by(`replay-${savedId}`).getAttribute("aria-pressed")) === "true");
await page.screenshot({ path: `${SHOTS}/live-saved-replay-1440.png` });

// ------------------------------------------------------------------------ Stop keeps what was run
await by("side-tab-dataset").click();
await page.waitForSelector('[data-testid="session-start"]');
const topBefore = await topRun();
await by("session-start").click();
await waitStatus(["running"], 30000);
await waitDecisionAtLeast(3);
await by("session-stop").click();
await page.waitForFunction((was) => {
  const top = document.querySelector('[data-testid="dataset-runs"] [data-testid^="dataset-run-"]');
  return top !== null && top.getAttribute("data-testid") !== was;
}, topBefore, { timeout: 120000 });
const stoppedId = (await topRun()).replace(/^dataset-run-/, "");
const stopped = await page.evaluate(async (id) => (await fetch(`replays/${id}/episode.json`)).json(), stoppedId);
check("Stop ends it, and what was run is saved too", stopped.terminated_by === "in_progress" && stopped.decisions.length >= 3,
      `${stoppedId}: ${stopped.decisions.length} decisions, ${stopped.terminated_by}`);
await page.waitForSelector('[data-testid="session-start"]', { timeout: 30000 });
check("and the console is ready for another", await by("session-start").isEnabled());

check("no console errors", consoleErrors.length === 0, consoleErrors.join(" | ").slice(0, 400));
await browser.close();
console.log(failures === 0 ? `\nALL CHECKS PASSED (saved ${savedId}, ${stoppedId})` : `\n${failures} CHECK(S) FAILED (saved ${savedId}, ${stoppedId})`);
process.exit(failures === 0 ? 0 : 1);
