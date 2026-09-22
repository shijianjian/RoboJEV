/**
 * Prove the built site works as plain files in a real browser, and photograph it.
 *
 * Run against `node serve.mjs <port> /RoboJEV/` on `web/dist`, which is what GitHub Pages will
 * be (`npm run build:pages`). It drives the page as a visitor would - open the root (an episode,
 * paused until autoplay is turned on), seek to two
 * decisions, switch episodes from the sidebar, follow a deep link - and asserts the things
 * a screenshot cannot: that the videos really decoded and advance, that the MuJoCo scene loaded
 * from the bundle's own scene and is posed differently at two decisions, that the panel shows the
 * decision the clock says it does, and that the Pages build has no Dataset tab.
 *
 *     node tools/verify.mjs [baseURL] [screenshot dir] [pages|local]
 *
 * Defaults: http://127.0.0.1:8142/RoboJEV/, ./screenshots and `pages` (`npm run build:pages`: the
 * Runs tab alone). `local` checks a plain `npm run build` instead: both tabs, and a Dataset tab
 * that says a console is needed.
 *
 * Playwright drives the **system Chrome** (the bundled Chromium has no H.264), with SwiftShader
 * allowed so a headless box still has WebGL for the 3D stage. The MuJoCo WASM is fetched from
 * jsDelivr, as on robopp, so the box needs the network.
 */
import { mkdirSync } from "node:fs";
import { chromium } from "playwright";

const BASE = process.argv[2] ?? "http://127.0.0.1:8142/RoboJEV/";
const SHOTS = process.argv[3] ?? "screenshots";
const SITE = process.argv[4] ?? "pages";
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
const requests = [];
page.on("request", (r) => requests.push({ url: r.url(), at: Date.now() }));

const by = (id) => page.locator(`[data-testid="${id}"]`);

/** Wait until the <video> has decoded a frame: readyState >= 2 and a real duration. */
async function videoReady(testid = "agentview") {
  await page.waitForFunction((id) => {
    const v = document.querySelector(`[data-testid="${id}"]`);
    return v !== null && v.readyState >= 2 && Number.isFinite(v.duration) && v.duration > 0;
  }, testid, { timeout: 20000 });
  return page.evaluate((id) => {
    const v = document.querySelector(`[data-testid="${id}"]`);
    return { duration: v.duration, w: v.videoWidth, paused: v.paused, t: v.currentTime, src: v.currentSrc };
  }, testid);
}
async function sceneReady() {
  await page.waitForFunction(() => document.querySelector('[data-testid="scene-status"]')?.innerText.trim() === "ready",
    null, { timeout: 120000 });
  return page.locator(".scene-stage").getAttribute("data-scene");
}
/** The panel's own statement of which decision it is showing. */
async function shownDecision() {
  const text = await by("decision-meta").innerText();
  return Number(/decision\s+(\d+)/.exec(text)[1]) - 1;
}
async function seekToDecision(i) {
  await by(`tick-${i}`).click();
  await page.waitForFunction((want) => {
    const el = document.querySelector('[data-testid="decision-meta"]');
    return el !== null && Number(/decision\s+(\d+)/.exec(el.innerText)[1]) - 1 === want;
  }, i, { timeout: 5000 });
  await page.waitForTimeout(600);           // the pose round trip to the worker
}
const bundleOf = (id) => page.evaluate(async (name) => (await fetch(`replays/${name}/episode.json`)).json(), id);

// ---------------------------------------------------------------- the root IS the demo
const opened = Date.now();
await page.goto(BASE, { waitUntil: "domcontentloaded" });
await page.waitForSelector('[data-testid="agentview"]');
// The clock must not move before the scene has compiled: sample it until the scene says ready.
const beforeReady = [];
while (true) {
  const s = await page.evaluate(() => {
    const v = document.querySelector('[data-testid="agentview"]');
    return { t: v?.currentTime ?? 0, paused: v?.paused ?? true,
             ready: document.querySelector('[data-testid="scene-status"]')?.innerText.trim() === "ready" };
  });
  if (s.ready || Date.now() - opened > 120000) break;
  beforeReady.push(s);
  await page.waitForTimeout(100);
}
const firstPaint = await page.evaluate(() => performance.getEntriesByType("paint").map((e) => `${e.name} ${Math.round(e.startTime)} ms`).join(", "));
check("the videos wait on their first frame until the scene is in",
      beforeReady.length > 3 && beforeReady.every((s) => s.t === 0 && s.paused),
      `${beforeReady.length} samples before the scene was ready; ${firstPaint}`);
check("no landing page: the root opens an episode on the Runs tab",
      (await by("replay-title").count()) === 1
      && (await by("side-tab-runs").getAttribute("aria-selected")) === "true");
check("the sidebar marks the drawer as the run on screen",
      (await by("replay-drawer").getAttribute("aria-pressed")) === "true");
check("the header is the name and the GitHub mark: no navigation, no status chip",
      (await page.locator(".app-header nav, .app-header__nav, .app-header .chip").count()) === 0);

const first = await videoReady();
check("the video decoded", first.w === 256, `${first.w}px ${first.duration.toFixed(2)}s`);
await page.waitForTimeout(1200);
const idle = await page.evaluate(() => { const v = document.querySelector('[data-testid="agentview"]'); return { t: v.currentTime, paused: v.paused }; });
check("autoplay is off by default: the run opens paused on its first frame",
      idle.paused && idle.t === 0 && (await by("autoplay").getAttribute("aria-pressed")) === "false");
await by("autoplay").click();
check("the autoplay toggle is remembered in this browser",
      (await page.evaluate(() => localStorage.getItem("robojev.autoplay"))) === "on");
await page.reload({ waitUntil: "domcontentloaded" });
await page.waitForSelector('[data-testid="agentview"]');
const gated = [];
while (true) {
  const s = await page.evaluate(() => {
    const v = document.querySelector('[data-testid="agentview"]');
    return { t: v?.currentTime ?? 0, paused: v?.paused ?? true,
             ready: document.querySelector('[data-testid="scene-status"]')?.innerText.trim() === "ready" };
  });
  if (s.ready) break;
  gated.push(s);
  await page.waitForTimeout(100);
}
check("with autoplay on, the clock still waits for the scene", gated.every((s) => s.t === 0 && s.paused), `${gated.length} samples`);
check("and then plays by itself", await page.waitForFunction(
  () => !document.querySelector('[data-testid="agentview"]').paused, null, { timeout: 5000 }).then(() => true, () => false));
await page.waitForTimeout(900);
const advanced = await page.evaluate(() => {
  const v = document.querySelector('[data-testid="agentview"]');
  return { t: v.currentTime, paused: v.paused };
});
check("it is playing on its own", !advanced.paused && advanced.t > 0.2, `t=${advanced.t.toFixed(2)}s`);
check("the panel followed the video", (await shownDecision()) >= 0);

const drawer = await bundleOf("drawer");
const drawerScene = await sceneReady();
const sceneAt = Date.now() - opened;
check("the MuJoCo scene loaded: the bundle's own scene", drawerScene?.endsWith(drawer.scene.hash),
      `${drawer.scene.hash.slice(0, 12)} at ${sceneAt} ms`);
const threeChunk = requests.find((r) => /\/assets\/scene-loading-[^/]*\.js$/.test(r.url));
const wasm = requests.find((r) => r.url.includes("@mujoco/mujoco"));
const box = (sel) => page.locator(sel).first().evaluate((el) => { const r = el.getBoundingClientRect(); return { l: r.left, r: r.right, t: r.top, b: r.bottom, w: r.width, h: r.height }; });
const sceneBox = await box(".rj-scene canvas.scene");
const scrubBox = await box(".rj-scrub");
const barBox = await box('[data-testid="stage-timeline"]');
const camBoxes = await page.locator(".rj-view__cams .cam").evaluateAll((els) => els.map((e) => { const r = e.getBoundingClientRect(); return { l: r.left, t: r.top, w: r.width }; }));
check("the scrubber and the stage bar start at the 3D view's left edge and span its width",
      Math.abs(scrubBox.l - sceneBox.l) <= 1 && Math.abs(barBox.l - sceneBox.l) <= 1
      && Math.abs(scrubBox.r - sceneBox.r) <= 1 && Math.abs(barBox.r - sceneBox.r) <= 1,
      `scene ${Math.round(sceneBox.l)}-${Math.round(sceneBox.r)}, scrubber ${Math.round(scrubBox.l)}-${Math.round(scrubBox.r)}, bar ${Math.round(barBox.l)}-${Math.round(barBox.r)}`);
check("the 3D view is the main view: most of the width, most of the height",
      sceneBox.w > 700 && sceneBox.h > 0.55 * 900, `${Math.round(sceneBox.w)}x${Math.round(sceneBox.h)}`);
check("the two cameras are stacked on its right, one width, level with its top",
      camBoxes.length === 2 && camBoxes.every((c) => c.l > sceneBox.r) && camBoxes[0].w === camBoxes[1].w
      && Math.abs(camBoxes[0].t - sceneBox.t) <= 2 && camBoxes[1].t > camBoxes[0].t,
      camBoxes.map((c) => `${Math.round(c.w)}px at ${Math.round(c.l)},${Math.round(c.t)}`).join(" / "));
// One time axis: each stage segment starts where its first decision's tick is, the settle segment
// starts where the track does, and both bars end at the episode's end.
const axis = await page.evaluate(() => {
  const mid = (el) => { const r = el.getBoundingClientRect(); return r.left + r.width / 2; };
  const segs = [...document.querySelectorAll('[data-testid^="timeline-"][data-from]')].map((el) => ({
    left: el.getBoundingClientRect().left, right: el.getBoundingClientRect().right,
    tick: mid(document.querySelector(`[data-testid="tick-${el.getAttribute("data-from")}"]`)),
  }));
  const settle = document.querySelector('[data-testid="timeline-settle"]')?.getBoundingClientRect() ?? null;
  const track = document.querySelector(".rj-scrub").getBoundingClientRect();
  const bar = document.querySelector('[data-testid="stage-timeline"]').getBoundingClientRect();
  return { segs, settle: settle && { left: settle.left, right: settle.right }, track: { l: track.left, r: track.right }, bar: { l: bar.left, r: bar.right } };
});
const worst = Math.max(...axis.segs.map((g) => Math.abs(g.left - g.tick)));
check("every stage segment starts at its first decision's tick (one time axis)", worst <= 1, `worst ${worst.toFixed(2)} px over ${axis.segs.length} stages`);
check("the settling steps are a neutral segment at the front, from the track's own left edge",
      axis.settle !== null && Math.abs(axis.settle.left - axis.track.l) <= 1 && Math.abs(axis.settle.right - axis.segs[0].left) <= 1);
check("the last stage ends where the track ends",
      Math.abs(axis.segs[axis.segs.length - 1].right - axis.track.r) <= 1 && Math.abs(axis.bar.r - axis.track.r) <= 1);
check("the first screen holds the 3D view, both cameras and the transport",
      scrubBox.b < 900 && barBox.b <= 900, `stage bar bottom ${Math.round(barBox.b)}`);
check("the 3D stage is its own chunk, and MuJoCo comes from jsDelivr as on robopp",
      threeChunk !== undefined && wasm !== undefined, `three.js chunk at ${threeChunk ? threeChunk.at - opened : "?"} ms`);

const bodyText = await page.locator("body").innerText();
for (const phrase of ["one forward pass", "click to seek", "Hover a question", "before any motion",
                      "what to look for", "Code plans", "Which way along", "Recorded runs of existing"]) {
  check(`no explanatory prose: "${phrase}"`, !bodyText.includes(phrase));
}
check("the state text is closed by default", (await by("state-text").count()) === 0);
await page.screenshot({ path: `${SHOTS}/root.png` });

// ------------------------------------------------------------------------ what it shows
// Pause, so a seek stays put (it may already have played to the end: a replay does not loop).
if (!(await page.evaluate(() => document.querySelector('[data-testid="agentview"]').paused))) await by("play").click();
await page.waitForFunction(() => document.querySelector('[data-testid="agentview"]').paused, null, { timeout: 5000 });

// Two decisions, one reaching and one carrying: the scene, the video and the bars all move.
const reach = drawer.decisions.find((d) => d.subgoal === "reach" && d.index > 2);
const carry = drawer.decisions.find((d) => d.subgoal === "carry");
const canvas = page.locator(".rj-scene canvas.scene");
await seekToDecision(reach.index);
const reachShot = await canvas.screenshot({ path: `${SHOTS}/scene-reach.png` });
await by("agentview").screenshot({ path: `${SHOTS}/camera-reach.png` });
await seekToDecision(carry.index);
const carryShot = await canvas.screenshot({ path: `${SHOTS}/scene-carry.png` });
await by("agentview").screenshot({ path: `${SHOTS}/camera-carry.png` });
check("the scene is posed differently at a reaching and a carrying decision", !reachShot.equals(carryShot),
      `decisions ${reach.index + 1} and ${carry.index + 1}`);
const videoTime = await page.evaluate(() => document.querySelector('[data-testid="agentview"]').currentTime);
check("the video followed the panel", Math.abs(videoTime - carry.t) < 0.06, `${videoTime.toFixed(3)}s vs ${carry.t}s`);

const yawTurns = drawer.decisions.filter((d) => d.questions.yaw.choice !== "hold").map((d) => d.index);
await seekToDecision(yawTurns[1]);
const yawChosen = await page.locator('[data-testid="decision-yaw"] .rj-bar--chosen .rj-bar__id').innerText();
check("the wrist-turn decision shows its answer", yawChosen.trim() === "+", yawChosen.trim());
const rimDecision = drawer.decisions.find((d) => d.rim_candidates.some((r) => !r.fits));
await seekToDecision(rimDecision.index);
const rimBar = await page.locator('[data-testid="decision-rim"] .rj-bar--chosen .rj-bar__id').innerText();
check("the chosen rim bar is the bundle's", rimBar.trim() === rimDecision.rim_chosen.letter, rimBar);
const graspFact = await by("fact-rim").innerText();
check("the grasp fact names the rim point while reaching", graspFact.includes(rimDecision.rim_chosen.letter),
      graspFact.replace(/\n/g, " "));
const segments = await page.locator('[data-testid^="timeline-"]').count();
check("the stage bar is under the scrubber, one segment per stage", segments >= 4, `${segments} segments`);
await by("timeline-1").click();
await page.waitForTimeout(300);
check("clicking a stage segment seeks into it", (await by("timeline-1").getAttribute("aria-current")) === "true");

// The state text opens, and the hover link still works inside it.
await by("state-toggle").click();
await page.waitForSelector('[data-testid="state-text"]');
await by("decision-rim").hover();
await page.waitForTimeout(150);
const litKinds = await page.locator(".rj-statetext__line--hot").evaluateAll((els) => els.map((e) => e.getAttribute("data-kind")));
check("hovering rim lights the candidate block", litKinds.filter((k) => k === "rim-row").length === 8 && litKinds.includes("rim-head"),
      litKinds.join(","));
await by("state-toggle").click();
await page.mouse.move(5, 5);

// Keyboard: ←/→ decisions, space play.
const before = await shownDecision();
await page.keyboard.press("ArrowRight");
await page.waitForTimeout(150);
check("→ advances one decision", (await shownDecision()) === before + 1);
await page.keyboard.press("ArrowLeft");
await page.waitForTimeout(150);
check("← goes back one decision", (await shownDecision()) === before);
await page.keyboard.press(" ");
await page.waitForTimeout(500);
check("space plays", await page.evaluate(() => !document.querySelector('[data-testid="agentview"]').paused));
await page.keyboard.press(" ");

// ---------------------------------------------------------- switching, from the sidebar
for (const id of ["bowl-plate", "cookie-box", "failure"]) {
  await by(`replay-${id}`).click();
  await page.waitForFunction((want) => location.hash === `#/${want}`
      && (document.querySelector('[data-testid="agentview"]')?.currentSrc ?? "").includes(`/${want}/`),
    id, { timeout: 10000 });
  await videoReady();
  const bundle = await bundleOf(id);
  const scene = await sceneReady();
  await page.waitForFunction(() => !document.querySelector('[data-testid="agentview"]').paused, null, { timeout: 5000 }).catch(() => {});
  const v = await videoReady();
  check(`switching to ${id} loads it, plays it and poses its own scene`,
        Math.abs(v.duration - bundle.total_frames / bundle.control_rate) < 0.15 && !v.paused && scene?.endsWith(bundle.scene.hash),
        `${v.duration.toFixed(2)}s, scene ${bundle.scene.hash.slice(0, 12)}`);
  check(`${id} shows its own task sentence`, (await by("replay-title").innerText()).trim() === bundle.instruction.trim());
}

// --------------------------------------------------------- the Pages build: Runs, and the source
const tabs = await page.locator('[role="tablist"] [role="tab"]').allInnerTexts();
if (SITE === "pages") {
  check("the Pages build's sidebar has the Runs tab and nothing that needs a console", tabs.join(",") === "Runs", tabs.join(","));
} else {
  check("a local build has both tabs", tabs.join(",") === "Runs,Dataset", tabs.join(","));
}
const gh = await by("github-link").evaluate((a) => {
  const r = a.getBoundingClientRect();
  const header = document.querySelector(".app-header").getBoundingClientRect();
  const pad = parseFloat(getComputedStyle(document.querySelector(".app-header__inner")).paddingRight);
  return { href: a.href, target: a.target, rel: a.rel, label: a.getAttribute("aria-label"), title: a.title,
           svg: a.querySelector("svg") !== null, text: a.innerText.trim(), inHeader: a.closest(".app-header") !== null,
           inSide: a.closest(".app-side") !== null, gap: header.right - r.right, pad,
           top: r.top >= header.top && r.bottom <= header.bottom };
});
check("the GitHub mark is in the header, at its right end, not in the sidebar",
      gh.inHeader && !gh.inSide && gh.top && Math.abs(gh.gap - gh.pad) <= 1,
      `${Math.round(gh.gap)} px from the right edge, header padding ${gh.pad} px`);
check("an icon with a GitHub tooltip, opening the repository in a new tab with noopener",
      gh.href === "https://github.com/shijianjian/RoboJEV" && gh.target === "_blank" && gh.rel.includes("noopener")
      && gh.svg && gh.text === "" && gh.label === "GitHub" && gh.title === "GitHub", `${gh.href} ${gh.target} ${gh.rel}`);
if (SITE === "pages") {
  for (const link of ["#/dataset", "#/live"]) {
    await page.goto(`${BASE}${link}`, { waitUntil: "domcontentloaded" });
    await page.waitForSelector('[data-testid="agentview"]');
    check(`${link} goes to the Runs tab on Pages`,
          (await page.evaluate(() => location.hash)) === "#/drawer" && (await by("session-needs-console").count()) === 0
          && (await by("side-tab-runs").getAttribute("aria-selected")) === "true");
  }
} else {
  await by("side-tab-dataset").click();
  await page.waitForFunction(() => location.hash === "#/dataset", null, { timeout: 5000 });
  await page.waitForSelector('[data-testid="task-pick-4"]');
  await by("task-pick-4").check();
  const startScene = await sceneReady();
  check("the local Dataset tab lists the tasks and shows a recorded task's start state in 3D",
        startScene?.endsWith(drawer.scene.hash) && (await by("dataset-run-drawer").count()) === 1);
  check("and says, where its controls would be, that they need a console",
        (await by("session-needs-console").innerText()).trim() === "needs a local robojev console");
  await page.screenshot({ path: `${SHOTS}/dataset-1440.png` });
  await by("dataset-run-drawer").click();
  await page.waitForFunction(() => location.hash === "#/drawer", null, { timeout: 5000 });
  check("a run listed there opens on the Runs tab", (await by("side-tab-runs").getAttribute("aria-selected")) === "true");
  // A console that cannot be reached is said once, in the bar, and nowhere else.
  await page.goto(`${BASE}?live=ws://127.0.0.1:9/ws#/dataset`, { waitUntil: "domcontentloaded" });
  await page.waitForSelector('[data-testid="disconnected"]', { timeout: 10000 });
  check("an unreachable console is one 'disconnected' chip in the bar, and the header says nothing",
        (await by("disconnected").innerText()).trim() === "disconnected" && (await page.locator(".app-header .chip").count()) === 0);
  consoleErrors.splice(0);   // the refused WebSocket is the point of this check, not an error
}

// ------------------------------------------------------------------------- deep links
await page.goto(`${BASE}#/failure?t=190`, { waitUntil: "domcontentloaded" });
await videoReady();
await page.waitForTimeout(500);
const deep = await page.evaluate(() => document.querySelector('[data-testid="agentview"]').currentTime);
check("a deep link opens that episode at its frame",
      (await by("replay-outcome").innerText()).trim() === "failure" && Math.abs(deep - 190 / 20) < 0.06, `t=${deep.toFixed(2)}s`);
check("the sidebar marks it as current", (await by("replay-failure").getAttribute("aria-pressed")) === "true");
await sceneReady();
await page.waitForTimeout(400);
await page.screenshot({ path: `${SHOTS}/failure.png` });

await page.goto(`${BASE}#/replay/cookie-box`, { waitUntil: "domcontentloaded" });
await videoReady();
check("an old #/replay/<id> link still resolves", (await by("replay-cookie-box").getAttribute("aria-pressed")) === "true");

// ----------------------------------------------------------- narrow windows, and errors
await page.setViewportSize({ width: 1024, height: 820 });
await page.goto(`${BASE}#/drawer?t=72`, { waitUntil: "domcontentloaded" });
await videoReady();
await sceneReady();
await page.waitForTimeout(500);
const wide = await page.evaluate(() => ({ scroll: document.documentElement.scrollWidth, client: document.documentElement.clientWidth }));
check("no horizontal overflow at laptop width", wide.scroll <= wide.client + 1, `${wide.scroll} vs ${wide.client}`);
await page.screenshot({ path: `${SHOTS}/laptop-1024.png` });
await page.screenshot({ path: `${SHOTS}/laptop-1024-full.png`, fullPage: true });
if (SITE === "local") {
  await page.goto(`${BASE}#/dataset`, { waitUntil: "domcontentloaded" });
  await page.waitForSelector('[data-testid="session-needs-console"]');
  await page.waitForTimeout(3000);
  await page.screenshot({ path: `${SHOTS}/dataset-1024.png` });
}

await page.setViewportSize({ width: 390, height: 844 });
await page.goto(`${BASE}#/drawer`, { waitUntil: "domcontentloaded" });
await videoReady();
await page.waitForTimeout(500);
const phone = await page.evaluate(() => ({ scroll: document.documentElement.scrollWidth, client: document.documentElement.clientWidth }));
check("no horizontal overflow on a phone", phone.scroll <= phone.client + 1, `${phone.scroll} vs ${phone.client}`);
await page.screenshot({ path: `${SHOTS}/phone.png`, fullPage: true });

check("no console errors", consoleErrors.length === 0, consoleErrors.join(" | ").slice(0, 300));

await browser.close();
console.log(failures === 0 ? "\nALL CHECKS PASSED" : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
