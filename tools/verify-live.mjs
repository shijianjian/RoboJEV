/**
 * Prove the live console works in a real browser, against a real simulator, and photograph it.
 *
 * `tools/verify.mjs` does this for the replay half against `tools/serve.mjs`; this one does it
 * against `robojev console`, which is serving the same built app *and* running an episode. It
 * drives the page as an operator would — pick a scene, Start, Step, hold a candidate, Step again,
 * Run to the end, Save — and asserts the things a screenshot cannot: that two decisions really
 * arrived with pictures beside them, that the answer the operator forced is the answer the bundle
 * records as overridden, that the episode ended, and that the bundle is on disk.
 *
 *     node tools/verify-live.mjs [baseURL] [screenshot dir]
 *
 * Defaults: http://127.0.0.1:8765/ and ./screenshots.
 *
 * Playwright is resolved from wherever node finds it and drives the **system Chrome**, as
 * `verify.mjs` does and for the same reason: the replay half of the same page needs H.264.
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

const browser = await chromium.launch({ channel: "chrome" });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
const consoleErrors = [];
page.on("console", (m) => { if (m.type() === "error") consoleErrors.push(m.text()); });
page.on("pageerror", (e) => consoleErrors.push(String(e)));

const text = (id) => page.locator(`[data-testid="${id}"]`).innerText();
const count = (id) => page.locator(`[data-testid="${id}"]`).count();

/** How many decisions the transport says have arrived. */
async function decisionsShown() {
  const readout = await text("transport-readout");
  return Number(/\/\s*(\d+)\s*$/.exec(readout.replace(/\s+/g, " ").trim())?.[1] ?? "0");
}

async function waitForDecisions(n, timeout = 120000) {
  await page.waitForFunction((want) => {
    const el = document.querySelector('[data-testid="transport-readout"]');
    if (el === null) return false;
    const m = /\/\s*(\d+)\s*$/.exec(el.innerText.replace(/\s+/g, " ").trim());
    return m !== null && Number(m[1]) >= want;
  }, n, { timeout });
}

/**
 * The candidate ids of one question on screen, and which is chosen.
 *
 * Off the test id and not off the label: the panel draws `-` as a typographic minus, and a script
 * that clicked what it read would be looking for a candidate no question declares.
 */
async function bars(qid) {
  return page.evaluate((q) => {
    const group = document.querySelector(`[data-testid="decision-${q}"]`);
    if (group === null) return null;
    return [...group.querySelectorAll(".bar")].map((b) => ({
      id: b.getAttribute("data-testid").slice(`bar-${q}-`.length),
      chosen: b.classList.contains("bar--chosen"),
      armed: b.classList.contains("bar--armed"),
      tag: b.tagName,
    }));
  }, qid);
}

// ------------------------------------------------------------------ the console serves the app
await page.goto(BASE, { waitUntil: "domcontentloaded" });
await page.waitForSelector('[data-testid="live-controls"]', { timeout: 30000 });
check("the console serves the app and the page finds the console",
      (await count("live-controls")) === 1);
await page.waitForFunction(
  () => document.querySelector('[data-testid="live-connection"]')?.innerText.trim() === "idle",
  null, { timeout: 20000 });
check("the socket is open and idle", (await text("live-connection")).trim() === "idle");
check("the console is the first item in the strip and it is current",
      (await page.locator('[data-testid="strip-live"]').getAttribute("aria-current")) === "true");
check("the recorded bundles are still in the strip beside it",
      (await page.locator('[data-testid="strip-drawer"]').count()) === 1);
check("no episode, so no picture is being requested",
      (await count("live-no-episode")) === 1 && (await count("live-agentview")) === 0);

// The task list arrives once the console has read LIBERO's task definitions: seconds, off the
// main thread, so the page is usable before it lands.
await page.waitForFunction(
  () => document.querySelector('[data-testid="live-task"]')?.tagName === "SELECT",
  null, { timeout: 90000 }).catch(() => {});
const picker = await page.locator('[data-testid="live-task"]').evaluate((el) => el.tagName);
check("the task picker names the tasks once the console has read them", picker === "SELECT",
      picker);

await page.screenshot({ path: `${SHOTS}/console-idle.png` });

// --------------------------------------------------------------------------- start an episode
if (picker === "SELECT") await page.locator('[data-testid="live-task"]').selectOption("0");
else await page.locator('[data-testid="live-task"]').fill("0");
await page.locator('[data-testid="live-init"]').fill("0");
await page.locator('[data-testid="live-policy"]').selectOption("expert");
await page.locator('[data-testid="live-start"]').click();

await page.waitForFunction(
  () => document.querySelector('[data-testid="live-connection"]')?.innerText.trim() === "paused",
  null, { timeout: 180000 });
const title = (await text("live-title")).trim();
check("the episode header names the task", title.startsWith("pick up"), title);
check("the scene line says what is being driven",
      (await text("live-scene")).includes("task 0"), (await text("live-scene")).trim());
check("the cameras are being fetched now that there is an episode",
      (await count("live-agentview")) === 1 && (await count("live-wrist")) === 1);

// ------------------------------------------------------------------------------ two decisions
await page.locator('[data-testid="live-step"]').click();
await waitForDecisions(1);
await page.locator('[data-testid="live-step"]').click();
await waitForDecisions(2);
check("two Steps are two decisions", (await decisionsShown()) === 2);

const meta = await text("decision-meta");
check("the panel is showing the newest of them", /decision\s+2\b/.test(meta.replace(/\s+/g, " ")),
      meta.replace(/\s+/g, " ").trim());
// Waited for rather than sampled: the two `<img>`s are being replaced several times a second, so
// any single instant may catch one of them mid-decode.
// The sizes are read *inside* the wait and handed back, not sampled after it: a second evaluate
// would run a frame later, by which time the src has been swapped again.
let pictures = null;
try {
  const handle = await page.waitForFunction(() => {
    const out = ["live-agentview", "live-wrist"].map((id) => {
      const img = document.querySelector(`[data-testid="${id}"]`);
      return img === null ? null : { w: img.naturalWidth, h: img.naturalHeight };
    });
    return out.every((p) => p !== null && p.w > 0) ? out : false;
  }, null, { timeout: 20000 });
  pictures = await handle.jsonValue();
} catch { /* reported by the check below */ }
check("both cameras decoded a real picture",
      pictures !== null && pictures.every((p) => p.w === 256 && p.h === 256),
      JSON.stringify(pictures));
const stateShown = await page.evaluate(() => {
  const el = document.querySelector('[data-testid="state-toggle"]');
  el.click();
  return true;
});
await page.waitForSelector('[data-testid="state-text"]');
const paragraph = await text("state-text");
check("the paragraph the model read is on the page",
      stateShown && paragraph.includes("x:") && paragraph.length > 200, `${paragraph.length} chars`);
await page.locator('[data-testid="state-toggle"]').click();

// ----------------------------------------------------------------------------- the override
const before = await bars("move_x");
check("a live candidate bar is a control", before !== null && before.every((b) => b.tag === "BUTTON"),
      JSON.stringify(before));
const wanted = before.find((b) => !b.chosen);
await page.locator(`[data-testid="bar-move_x-${wanted.id}"]`).click();
await page.waitForSelector('[data-testid="armed-move_x"]', { timeout: 10000 });
check(`holding move_x = ${wanted.id} is shown as held`,
      (await text("armed-move_x")).includes(wanted.id), await text("armed-move_x"));
const armedBars = await bars("move_x");
check("and it is the clicked candidate that is marked",
      armedBars.find((b) => b.armed)?.id === wanted.id, JSON.stringify(armedBars));

await page.mouse.move(5, 5);
await page.screenshot({ path: `${SHOTS}/console-armed.png` });

// The picture really is the episode's, not one still frame: the console stamps each render with a
// sequence number, and five control steps later it has to have moved.
const frameSeq = () => page.evaluate(async () =>
  Number((await fetch("frame/agentview.png", { cache: "no-store" })).headers.get("x-frame-seq")));
const seqBefore = await frameSeq();
await page.locator('[data-testid="live-step"]').click();
await waitForDecisions(3);
const seqAfter = await frameSeq();
check("the camera stream advances with the episode", seqAfter >= seqBefore + 5,
      `frame ${seqBefore} → ${seqAfter}`);
await page.waitForSelector('[data-testid="overridden-move_x"]', { timeout: 20000 });
const forced = await bars("move_x");
check("the forced answer is the one that executed",
      forced.find((b) => b.chosen)?.id === wanted.id, JSON.stringify(forced));
check("and the decision says it was overridden",
      (await text("overridden-move_x")).trim() === "overridden");
check("nothing is held any more: an override is one decision, not a setting",
      (await count("armed-move_x")) === 0);

// ------------------------------------------------------------------------------ run to the end
await page.locator('[data-testid="live-run"]').click();
await page.waitForFunction(() => {
  const chip = document.querySelector('[data-testid="live-connection"]')?.innerText.trim();
  return chip === "done" || chip === "error";
}, null, { timeout: 600000 });
const outcome = (await count("live-outcome")) === 1 ? (await text("live-outcome")).trim() : "—";
const line = (await text("live-line")).trim();
check("the episode ran to its own end", (await text("live-connection")).trim() === "done", line);
check("and it succeeded", outcome === "success", `${outcome} · ${line}`);
const total = await decisionsShown();
check("with more decisions than the three taken by hand", total > 3, `${total} decisions`);

await page.screenshot({ path: `${SHOTS}/console-done.png` });

// ------------------------------------------------------------------------------------- save
await page.locator('[data-testid="live-save"]').click();
await page.waitForSelector('[data-testid="live-saved"]', { timeout: 120000 });
const saved = (await text("live-saved")).replace(/\s+/g, " ").trim();
const savedId = /saved\s+(\S+)/.exec(saved)?.[1] ?? "";
check("the episode was written out as a bundle", savedId.startsWith("live-"), saved);
const bundle = await page.evaluate(
  async (id) => (await fetch(`replays/${id}/episode.json`)).json(), savedId);
check("the bundle is served by the console without a rebuild",
      bundle.id === savedId && bundle.decisions.length === total,
      `${bundle.decisions.length} decisions, success=${bundle.success}`);
check("the overridden decision survived into the file",
      bundle.decisions[2].questions.move_x.overridden === true
      && bundle.decisions[2].questions.move_x.choice === wanted.id,
      JSON.stringify(bundle.decisions[2].questions.move_x.choice));
check("the bundle carries both videos and a poster",
      bundle.media.agentview?.codec === "h264" && bundle.media.wrist !== undefined
      && bundle.poster === "poster.jpg",
      JSON.stringify(Object.keys(bundle.media)));

// --------------------------------------------------------------- back to a replay, and reset
await page.locator('[data-testid="strip-drawer"]').click();
await page.waitForSelector('[data-testid="agentview"]', { timeout: 20000 });
check("a recorded bundle still replays in the same page",
      (await page.locator('[data-testid="replay-title"]').count()) === 1);
await page.locator('[data-testid="strip-live"]').click();
await page.waitForSelector('[data-testid="live-controls"]');
check("and the console is still there when you come back",
      (await text("live-connection")).trim() === "done");

await page.locator('[data-testid="live-reset"]').click();
await page.waitForFunction(
  () => document.querySelector('[data-testid="live-connection"]')?.innerText.trim() === "idle",
  null, { timeout: 60000 });
check("Reset lets go of the simulator and the console is ready for another",
      (await text("live-connection")).trim() === "idle");

check("no console errors", consoleErrors.length === 0, consoleErrors.join(" | ").slice(0, 400));

await browser.close();
console.log(`\nsaved bundle: ${savedId}`);
console.log(failures === 0 ? "ALL LIVE CHECKS PASSED" : `${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
