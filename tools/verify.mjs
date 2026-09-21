/**
 * Prove the built site works as plain files in a real browser, and photograph it.
 *
 * Run against `node serve.mjs <port> /RoboJEV/` on `web/dist`, which is what GitHub Pages will
 * be. It drives the page as a visitor would — open the root, watch it play, switch episodes from
 * the strip, follow a deep link — and asserts the things a screenshot cannot: that the video
 * really decoded metadata and is advancing (a bundle whose mp4 the browser cannot play would
 * still *look* fine on its poster), that the URL follows the switch, and that the panel shows the
 * decision the clock says it does.
 *
 *     node tools/verify.mjs [baseURL] [screenshot dir]
 *
 * Defaults: http://127.0.0.1:8142/RoboJEV/ and ./screenshots.
 *
 * Playwright is resolved from wherever node finds it (`npm i -D playwright` in `web/`, or
 * `NODE_PATH=...`) and drives the **system Chrome**: the bundled Chromium has no H.264, and
 * H.264 is exactly what is being verified.
 */
import { mkdirSync } from "node:fs";
import { chromium } from "playwright";

const BASE = process.argv[2] ?? "http://127.0.0.1:8142/RoboJEV/";
const SHOTS = process.argv[3] ?? "screenshots";
mkdirSync(SHOTS, { recursive: true });

let failures = 0;
function check(name, ok, detail = "") {
  console.log(`${ok ? "  ok  " : " FAIL "} ${name}${detail ? ` — ${detail}` : ""}`);
  if (!ok) failures += 1;
}

const browser = await chromium.launch({ channel: "chrome" });
// 1440x900 is the "does the whole thing fit on one screen" case, so it is the default viewport
// and the screenshots taken at it are deliberately NOT fullPage.
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
const consoleErrors = [];
page.on("console", (m) => { if (m.type() === "error") consoleErrors.push(m.text()); });
page.on("pageerror", (e) => consoleErrors.push(String(e)));

/** Wait until the <video> has decoded metadata: readyState >= 1 and a real duration. */
async function videoReady(testid = "agentview") {
  await page.waitForFunction((id) => {
    const v = document.querySelector(`[data-testid="${id}"]`);
    return v !== null && v.readyState >= 1 && Number.isFinite(v.duration) && v.duration > 0;
  }, testid, { timeout: 15000 });
  return page.evaluate((id) => {
    const v = document.querySelector(`[data-testid="${id}"]`);
    return { readyState: v.readyState, duration: v.duration, w: v.videoWidth, h: v.videoHeight,
             paused: v.paused, t: v.currentTime, src: v.currentSrc };
  }, testid);
}

/** The panel's own statement of which decision it is showing. */
async function shownDecision() {
  const text = await page.locator('[data-testid="decision-meta"]').innerText();
  return Number(/decision\s+(\d+)/.exec(text)[1]) - 1;
}

async function seekToDecision(i) {
  await page.locator(`[data-testid="tick-${i}"]`).click();
  await page.waitForFunction((want) => {
    const el = document.querySelector('[data-testid="decision-meta"]');
    return el !== null && Number(/decision\s+(\d+)/.exec(el.innerText)[1]) - 1 === want;
  }, i, { timeout: 5000 });
}

const bundleOf = (id) =>
  page.evaluate(async (name) => (await fetch(`replays/${name}/episode.json`)).json(), id);

// ---------------------------------------------------------------- the root IS the demo
await page.goto(BASE, { waitUntil: "networkidle" });
await page.waitForSelector('[data-testid="agentview"]');
check("no landing page: the root opens an episode",
      (await page.locator('[data-testid="replay-title"]').count()) === 1
      && (await page.locator('[data-testid="episode-strip"]').count()) === 1);
check("the default episode is the drawer",
      (await page.locator('[data-testid="strip-drawer"]').getAttribute("aria-current")) === "true");

const first = await videoReady();
check("the video decoded metadata", first.readyState >= 1 && first.w === 256,
      `readyState=${first.readyState} ${first.w}x${first.h} ${first.duration.toFixed(2)}s`);

// Playing by itself: the clock has to advance without anybody pressing anything.
await page.waitForTimeout(900);
const advanced = await page.evaluate(() => {
  const v = document.querySelector('[data-testid="agentview"]');
  return { t: v.currentTime, paused: v.paused };
});
check("it is playing on its own", !advanced.paused && advanced.t > 0.2, `t=${advanced.t.toFixed(2)}s`);
check("the panel followed the video", (await shownDecision()) >= 0);

const posters = await page.locator(".strip__poster").count();
check("every strip item has a poster", posters === 4, `${posters} posters`);
const posterOk = await page.evaluate(() =>
  [...document.querySelectorAll(".strip__poster")].every((img) => img.complete && img.naturalWidth > 0));
check("the posters actually loaded", posterOk);
const labels = await page.locator(".strip__label").allInnerTexts();
check("labels are short and distinct", new Set(labels).size === 4 && labels.every((l) => l.length < 50),
      labels.join(" | "));

const bodyText = await page.locator("body").innerText();
for (const phrase of ["one forward pass", "click to seek", "Hover a question", "before any motion",
                      "what to look for", "Code plans", "Which way along"]) {
  check(`no explanatory prose: "${phrase}"`, !bodyText.includes(phrase));
}
check("the state text is closed by default",
      (await page.locator('[data-testid="state-text"]').count()) === 0);

// The whole panel on the first screen at 1440x900, nothing cut off below the fold.
const fits = await page.evaluate(() => {
  const box = document.querySelector('[data-testid="decision-card"]').getBoundingClientRect();
  return { bottom: Math.round(box.bottom), viewport: window.innerHeight };
});
check("the decision panel fits the first screen at 1440x900",
      fits.bottom <= fits.viewport, `panel bottom ${fits.bottom} vs viewport ${fits.viewport}`);

await page.screenshot({ path: `${SHOTS}/root.png` });

// ------------------------------------------------------------------------ what it shows
const drawer = await bundleOf("drawer");
const yawTurns = drawer.decisions.filter((d) => d.questions.yaw.choice !== "hold").map((d) => d.index);
await page.locator('[data-testid="play"]').click();           // pause, so a seek stays put
await seekToDecision(yawTurns[1]);
const yawChosen = await page.locator('[data-testid="decision-yaw"] .bar--chosen .bar__id').innerText();
check("the wrist-turn decision shows its answer", yawChosen.trim() === "+", yawChosen.trim());
const videoTime = await page.evaluate(() => document.querySelector('[data-testid="agentview"]').currentTime);
check("the video followed the panel",
      Math.abs(videoTime - drawer.decisions[yawTurns[1]].t) < 0.06,
      `video ${videoTime.toFixed(3)}s vs decision ${drawer.decisions[yawTurns[1]].t}s`);

const rimDecision = drawer.decisions.find((d) => d.rim_candidates.some((r) => !r.fits));
await seekToDecision(rimDecision.index);
const rimBar = await page.locator('[data-testid="decision-rim"] .bar--chosen .bar__id').innerText();
check("the chosen rim bar is the bundle's", rimBar.trim() === rimDecision.rim_chosen.letter, rimBar);
const graspFact = await page.locator('[data-testid="fact-rim"]').innerText();
check("the grasp fact names the rim point while reaching",
      graspFact.includes(rimDecision.rim_chosen.letter), graspFact.replace(/\n/g, " "));

// The state text opens, and the hover link still works inside it.
await page.locator('[data-testid="state-toggle"]').click();
await page.waitForSelector('[data-testid="state-text"]');
await page.locator('[data-testid="decision-rim"]').hover();
await page.waitForTimeout(150);
const litKinds = await page.locator(".statetext__line--hot").evaluateAll(
  (els) => els.map((e) => e.getAttribute("data-kind")));
check("hovering rim still lights the candidate block",
      litKinds.filter((k) => k === "rim-row").length === 8 && litKinds.includes("rim-head"),
      litKinds.join(","));
await page.locator('[data-testid="decision-move_x"]').hover();
await page.waitForTimeout(150);
const xLit = await page.locator(".statetext__line--hot").evaluateAll((els) => els.map((e) => e.innerText.trim()));
check("hovering move_x lights the x offset line and not y or z",
      xLit.some((t) => t.startsWith("x:")) && !xLit.some((t) => t.startsWith("y:")),
      xLit.map((t) => t.slice(0, 18)).join(" | "));
await page.locator('[data-testid="state-toggle"]').click();
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

// ---------------------------------------------------------- switching, and the URL with it
for (const id of ["bowl-plate", "cookie-box", "failure"]) {
  await page.locator(`[data-testid="strip-${id}"]`).click();
  await page.waitForFunction(
    (want) => location.hash === `#/${want}`
      && (document.querySelector('[data-testid="agentview"]')?.currentSrc ?? "").includes(`/${want}/`),
    id, { timeout: 10000 });
  const v = await videoReady();
  const bundle = await bundleOf(id);
  check(`switching to ${id} loads and plays it`,
        v.w === 256 && Math.abs(v.duration - bundle.total_frames / bundle.control_rate) < 0.15 && !v.paused,
        `${v.duration.toFixed(2)}s, hash ${await page.evaluate(() => location.hash)}`);
  const title = await page.locator('[data-testid="replay-title"]').innerText();
  check(`${id} shows its own task sentence`, title.trim() === bundle.instruction.trim());
}

await page.keyboard.press("1");
await page.waitForFunction(() => location.hash === "#/drawer", null, { timeout: 5000 });
check("the 1 key jumps to the first episode", (await page.evaluate(() => location.hash)) === "#/drawer");
await page.keyboard.press("ArrowDown");
await page.waitForFunction(() => location.hash === "#/bowl-plate", null, { timeout: 5000 });
check("↓ walks the strip", (await page.evaluate(() => location.hash)) === "#/bowl-plate");

// ------------------------------------------------------------------------- deep links
await page.goto(`${BASE}#/failure`, { waitUntil: "networkidle" });
await videoReady();
check("a deep link opens that episode",
      (await page.locator('[data-testid="replay-outcome"]').innerText()).trim() === "failure");
check("the strip marks it as current",
      (await page.locator('[data-testid="strip-failure"]').getAttribute("aria-current")) === "true");
await page.waitForTimeout(700);
await seekToDecision(38);
await page.mouse.move(5, 5);
await page.waitForTimeout(250);
await page.screenshot({ path: `${SHOTS}/failure.png` });

await page.goto(`${BASE}#/replay/cookie-box`, { waitUntil: "networkidle" });
await videoReady();
check("an old #/replay/<id> link still resolves",
      (await page.locator('[data-testid="strip-cookie-box"]').getAttribute("aria-current")) === "true");

// ----------------------------------------------------------- narrow windows, and errors
await page.setViewportSize({ width: 1024, height: 820 });
await page.goto(`${BASE}#/drawer`, { waitUntil: "networkidle" });
await videoReady();
await page.waitForTimeout(700);
await seekToDecision(18);
await page.mouse.move(5, 5);
await page.waitForTimeout(250);
const wide = await page.evaluate(() => ({
  scroll: document.documentElement.scrollWidth, client: document.documentElement.clientWidth }));
check("no horizontal overflow at laptop width", wide.scroll <= wide.client + 1, `${wide.scroll} vs ${wide.client}`);
await page.screenshot({ path: `${SHOTS}/laptop-1024.png`, fullPage: true });

await page.setViewportSize({ width: 390, height: 844 });
await page.waitForTimeout(500);
const phone = await page.evaluate(() => ({
  scroll: document.documentElement.scrollWidth, client: document.documentElement.clientWidth }));
check("no horizontal overflow on a phone", phone.scroll <= phone.client + 1, `${phone.scroll} vs ${phone.client}`);
await page.screenshot({ path: `${SHOTS}/phone.png`, fullPage: true });

check("no console errors", consoleErrors.length === 0, consoleErrors.join(" | ").slice(0, 300));

await browser.close();
console.log(failures === 0 ? "\nALL CHECKS PASSED" : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
