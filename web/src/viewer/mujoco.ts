import type { BodyPose, GeomKind, GeomSpec, MeshData, MeshIndices, SceneGeometry, TextureData } from "./sceneProtocol";

export const MUJOCO_WASM_URL = "https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.12.0/mujoco.js";

// robosuite's own convention (not MuJoCo's default) is: geom group 0 holds the *collision*
// geoms (simplified proxy shapes, e.g. the green robot links and yellow object hulls seen in
// this viewer before this fix) and group 1 holds the *visual* geoms actually meant to be seen,
// including the floor and walls. robosuite's own renderer hides group 0 and shows group 1 (and
// group 2, used for some markers/sites). MuJoCo itself defaults to showing groups 0-2, which is
// why rendering "everything below group 3" - the naive reading of that default - looks correct
// in general but wrong specifically for robosuite scenes, which repurpose group 0 for collision
// geometry. So this viewer must special-case robosuite's convention rather than use MuJoCo's.
export const VISIBLE_GEOM_GROUPS = new Set([1, 2]);

/**
 * The MuJoCo WASM module, loaded once per thread that asks for it.
 *
 * This module is imported by `mujoco.worker.ts`, so everything in this file runs with no DOM and
 * no three.js — deliberately: pulling three.js into the worker bundle would double it for code
 * the worker can never use. Model arrays are read here into the plain buffers of
 * `sceneProtocol.ts`, and three.js objects are built from those on the main thread
 * (`threeStage.ts`).
 *
 * The import is left to the browser rather than bundled (both bundlers' ignore comments) because
 * the CDN ships the `.wasm` beside the `.js` and resolves it relative to its own URL.
 */
let mujocoPromise: Promise<any> | undefined;
export function loadMujoco(): Promise<any> {
  if (!mujocoPromise) {
    mujocoPromise = import(/* @vite-ignore */ MUJOCO_WASM_URL).then((m: any) => (m.default ?? m)());
  }
  return mujocoPromise;
}

export function assetFilesFromXml(xml: string): string[] {
  const out = new Set<string>();
  for (const m of xml.matchAll(/\sfile="([^"]+)"/g)) {
    const f = m[1];
    // Mirrors assertKey's traversal check: a bare "." or ".." path segment (e.g. "assets/..")
    // would otherwise slip past the character-class regex below and let a fetch escape assetsBase.
    if (!/^assets\/[A-Za-z0-9._-]+$/.test(f) || f.split("/").some((s) => s === "." || s === "..")) {
      throw new Error(`asset path outside assets/: ${f}`);
    }
    out.add(f);
  }
  return [...out];
}

// The exported bundle's <size njmax="..." nconmax="..."/> (robosuite's legacy defaults) makes
// MuJoCo 3 pre-reserve a huge constraint arena (~1.2 GB) that pushes the compiled model past the
// WASM build's 2 GiB heap ceiling. The viewer only ever calls mj_forward (never steps physics),
// so a much smaller arena is safe here and doesn't change any kinematics; the exported bundle
// itself is left untouched -- this patch is applied only to the copy written into the WASM FS.
//
// 40M, not the 64M this started at: measured across all 47 bundles in live storage, the worst
// case after mj_forward at the model's own default pose is maxuse_arena 8.12 MiB with
// maxuse_stack 2.35 MiB alongside it -- both are carved from this one pool, so ~10.5 MiB is the
// high-water mark. 40M is ~4x that, which is the margin a *posed* frame needs and the sweep could
// not measure: arena use in mj_forward is dominated by contacts, and a frame mid-manipulation
// carries more of them than the default pose the sweep saw. Every MjData reserves this much and
// the compare page holds two, so the cut from 64M is ~48 MB a tab keeps - against a 1815 MB
// plateau, which is why this is not where the memory problem was solved (MAX_CACHED_MODELS is).
//
// A scene that overflows it anyway is not left to fail: the pose path reloads that one bundle with
// {@link doubledArena} once. See `sceneEngine.ts`.
export const VIEWER_ARENA = "40M";

/** The arena to try when a scene turns out to need more than {@link VIEWER_ARENA}. */
export function doubledArena(arena: string): string {
  const m = /^(\d+)([KMG]?)$/.exec(arena);
  return m ? `${Number(m[1]) * 2}${m[2]}` : arena;
}

export function patchSizeForViewer(xml: string, arena: string = VIEWER_ARENA): string {
  const replacement = `<size memory="${arena}"/>`;
  const sizeRe = /<size\b[^>]*\/>|<size\b[^>]*>[\s\S]*?<\/size>/;
  if (sizeRe.test(xml)) return xml.replace(sizeRe, replacement);
  const compilerRe = /<compiler\b[^>]*\/>|<compiler\b[^>]*>[\s\S]*?<\/compiler>/;
  const compilerMatch = xml.match(compilerRe);
  if (compilerMatch) {
    const idx = compilerMatch.index! + compilerMatch[0].length;
    return xml.slice(0, idx) + replacement + xml.slice(idx);
  }
  const mujocoMatch = xml.match(/<mujoco\b[^>]*>/);
  if (mujocoMatch) {
    const idx = mujocoMatch.index! + mujocoMatch[0].length;
    return xml.slice(0, idx) + replacement + xml.slice(idx);
  }
  return replacement + xml;
}

// FNV-1a over the URL string, base36. Only a stable, collision-unlikely name for a bundle
// whose URL carries no hash of its own - never a content hash and never security-relevant.
function hashUrl(url: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < url.length; i++) {
    h ^= url.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return "u" + h.toString(36);
}

/**
 * The WASM FS directory a bundle is written to, one per bundle rather than one shared
 * `/working`: two viewers on the same page (the compare page) can show two *different*
 * bundles - robosuite samples fixture placement per reset, so two runs on the same task
 * usually have different bundle hashes - and a shared directory would have the second load
 * overwrite the first's assets under the first model's file names.
 *
 * The name comes from the bundle's own content hash, the `scenes/<hash>/` segment of its URL;
 * anything else (a URL shaped differently, or a `..` segment - the leading character class
 * rejects one) falls back to a hash of the URL string, so the result is always one safe
 * path segment under /scenes.
 */
export function sceneDirFromXmlUrl(xmlUrl: string): string {
  const path = xmlUrl.split("?")[0].split("#")[0];
  const m = /(?:^|\/)scenes\/([A-Za-z0-9][A-Za-z0-9._-]*)\/scene\.xml$/.exec(path);
  if (m) return `/scenes/${m[1]}`;
  return `/scenes/${hashUrl(xmlUrl)}`;
}

let loggedDisposeFailure = false;

// model/data are embind-wrapped WASM objects: their heap memory is only released by an
// explicit .delete() call, never by JS garbage collection. Without this, switching scenes
// (or remounting the viewer) leaks the previous model's WASM heap until the tab is reloaded.
// `.delete` may be absent on some embind builds, hence the optional calls; any failure is
// logged once (not per call) so a bad teardown doesn't spam the console.
//
// Either argument may be null: a viewer's `MjData` is released when it unmounts, while the model
// it was built from belongs to the cache and outlives it (see {@link SceneHandle}).
export function disposeMujoco(model: any, data: any): void {
  try {
    data?.delete?.();
    model?.delete?.();
  } catch (e) {
    if (!loggedDisposeFailure) {
      loggedDisposeFailure = true;
      console.error("disposeMujoco: failed to release MuJoCo WASM objects", e);
    }
  }
}

/** How many materialized bundle directories the WASM FS keeps. Each is up to 50 MiB of meshes and
 *  textures, and the WASM heap ceiling is 2 GiB, so this cannot be unbounded; three covers
 *  going back and forth between a compare page's pair and one other scene. Past that, the
 *  least recently used dirs are unlinked and their bundles re-fetched if visited again.
 *
 *  Left at three deliberately: browsing six live bundles with a two-model cache plateaus at the
 *  same 1815 MB whether this is two or three (report, fix round 2), so the dirs are not what
 *  fills the heap - the compiled models are - and a smaller number here would only buy re-fetches. */
export const MAX_SCENE_DIRS = 3;

/** The slice of Emscripten's `FS` this module uses. Named so the bookkeeping below can be
 *  unit-tested against a plain object instead of a WASM module. */
export interface SceneFs {
  analyzePath(path: string): { exists: boolean };
  mkdirTree(path: string): void;
  writeFile(path: string, data: Uint8Array | string): void;
  unlink(path: string): void;
  rmdir(path: string): void;
  readdir?(path: string): string[];
}

/** Materialized scene dirs, least recently used first. Module-level, i.e. one per tab: the
 *  WASM module it describes is itself memoised for the tab's lifetime. */
const sceneDirs: string[] = [];

/** dir -> the promise that materializes that bundle into the FS, while it is in flight and
 *  after it resolves. Two viewers mounting together (the compare page, when both runs share a
 *  bundle hash) then fetch and write it once: the second awaits the first. */
const materializing = new Map<string, Promise<void>>();

/** dir -> how many loads are between "started materializing" and "mj_loadXML returned". A
 *  compiled MjModel holds its own copy of the meshes and textures, so a dir is safe to remove
 *  once the compile is done - but not before, so these are never evicted. */
const loading = new Map<string, number>();

/** Move `dir` to the most-recently-used end, adding it if new. Mutates and returns `lru`. */
export function touchSceneDir(lru: string[], dir: string): string[] {
  const at = lru.indexOf(dir);
  if (at >= 0) lru.splice(at, 1);
  lru.push(dir);
  return lru;
}

/**
 * Which dirs to drop to get back to `max`: the least recently used first, skipping any that a
 * load is still holding. Pure, so the arithmetic is testable without an FS.
 *
 * A dir held by a load is skipped rather than counted out, so a page holding `max` dirs open at
 * once keeps them all — one bundle over the cap beats evicting a scene that is still compiling.
 */
export function evictionPlan(lru: readonly string[], held: ReadonlySet<string>, max = MAX_SCENE_DIRS): string[] {
  const out: string[] = [];
  let keep = lru.length;
  for (const dir of lru) {
    if (keep <= max) break;
    if (held.has(dir)) continue;
    out.push(dir);
    keep--;
  }
  return out;
}

/**
 * Unlink everything under one materialized bundle dir and remove the dir itself.
 *
 * `readdir` drives it where the FS offers one (Emscripten's does), with `known` - the relative
 * paths this module wrote - as the fallback, so a build without readdir still frees the bytes
 * that matter. Every call is guarded: a half-removed dir is fine (the next materialize rewrites
 * what is missing), a throw here would take the viewer down with it.
 */
export function removeSceneDir(fs: SceneFs, dir: string, known: readonly string[] = []): void {
  const drop = (path: string) => { try { fs.unlink(path); } catch {} };
  const listed = (path: string): string[] => {
    try { return (fs.readdir?.(path) ?? []).filter((n) => n !== "." && n !== ".."); } catch { return []; }
  };
  const names = new Set<string>([...listed(dir + "/assets"), ...known.filter((f) => f.startsWith("assets/")).map((f) => f.slice("assets/".length))]);
  for (const name of names) drop(`${dir}/assets/${name}`);
  try { fs.rmdir(dir + "/assets"); } catch {}
  const top = new Set<string>([...listed(dir).filter((n) => n !== "assets"), ...known.filter((f) => !f.includes("/")), "scene.xml"]);
  for (const name of top) drop(`${dir}/${name}`);
  try { fs.rmdir(dir); } catch {}
}

/** dir -> the relative paths written into it, for {@link removeSceneDir}'s fallback. */
const sceneDirFiles = new Map<string, string[]>();

/** Forget every materialize promise for a dir, whatever arena it was written for. */
function forgetMaterialized(dir: string): void {
  for (const key of [...materializing.keys()]) {
    if (key === dir || key.startsWith(dir + "#")) materializing.delete(key);
  }
}

/**
 * Fetch a bundle and write it into `dir`, skipping files the FS already holds.
 *
 * Only called through {@link materializing}, so at most one of these runs per dir at a time:
 * without that the compare page's two viewers both compute `missing` before either writes and
 * both download the full ~75 MiB.
 */
async function materializeSceneBundle(
  mujoco: any, xmlUrl: string, assetsBase: string, dir: string, arena: string,
  onProgress?: (loaded: number, total: number) => void,
): Promise<void> {
  const fs = mujoco.FS as SceneFs;
  const exists = (p: string) => Boolean(fs.analyzePath(p).exists);
  const xml = await (await fetch(xmlUrl)).text();
  const files = assetFilesFromXml(xml);
  sceneDirFiles.set(dir, [...(sceneDirFiles.get(dir) ?? []), ...files, sceneXmlName(arena)]);
  fs.mkdirTree(dir + "/assets");
  const missing = files.filter((f) => !exists(dir + "/" + f));
  let done = 0;
  onProgress?.(0, missing.length);
  const blobs = await Promise.all(missing.map(async (f) => {
    const res = await fetch(assetsBase + f.slice("assets/".length));
    if (!res.ok) throw new Error(`asset ${f}: ${res.status}`);
    const bytes = new Uint8Array(await res.arrayBuffer());
    onProgress?.(++done, missing.length);
    return bytes;
  }));
  missing.forEach((f, i) => fs.writeFile(dir + "/" + f, blobs[i]));
  // One xml per arena, assets shared: a scene that needs a bigger `<size memory>` is the same
  // bundle, and re-fetching 50 MB of meshes to change one attribute would be absurd.
  const xmlPath = dir + "/" + sceneXmlName(arena);
  if (!exists(xmlPath)) fs.writeFile(xmlPath, patchSizeForViewer(xml, arena));
}

/** The name the patched scene is written under. Carries the arena, because the same bundle can be
 *  in the filesystem twice with two different `<size memory>` values. */
function sceneXmlName(arena: string): string {
  return `scene-${arena}.xml`;
}

/** Drop the least recently used dirs past {@link MAX_SCENE_DIRS}, freeing their WASM heap. */
function evictSceneDirs(fs: SceneFs): void {
  for (const dir of evictionPlan(sceneDirs, new Set(loading.keys()))) {
    removeSceneDir(fs, dir, sceneDirFiles.get(dir) ?? []);
    sceneDirs.splice(sceneDirs.indexOf(dir), 1);
    sceneDirFiles.delete(dir);
    // The files are gone, so the next load of this bundle must fetch and write them again.
    materializing.delete(dir);
  }
}

/** `performance.measure` names this module records, so the cost of opening a scene can be read
 *  out of a real browser (see `scripts/scene-bench.ts`) rather than guessed at. */
export const PERF_FETCH = "robojev:scene-fetch";
export const PERF_COMPILE = "robojev:scene-compile";
/** Building the three.js stage out of the worker's buffers — the one part of opening a scene
 *  that is still main-thread work, and so the one worth watching now. */
export const PERF_BUILD = "robojev:scene-build";

/** Time `body` under `name`, where this environment has a `performance` to record it with.
 *  Marks made in the worker stay in the worker's own timeline; `PERF_BUILD` is the main
 *  thread's. */
export function measured<T>(name: string, body: () => T): T {
  if (typeof performance === "undefined" || typeof performance.measure !== "function") return body();
  const start = performance.now();
  const finish = () => { try { performance.measure(name, { start, end: performance.now() }); } catch {} };
  let result: T;
  try {
    result = body();
  } catch (e) {
    finish();
    throw e;
  }
  if (result instanceof Promise) return result.finally(finish) as T;
  finish();
  return result;
}

/**
 * Give the browser a frame to paint before the caller blocks the main thread.
 *
 * One `requestAnimationFrame` gets React's pending update into a rendered frame; the
 * `setTimeout(0)` after it lets that frame actually reach the screen before the next
 * synchronous block starts. Without both, the progress line and the newly selected task's own
 * text are queued behind a compile that will not yield for seconds, and the click looks lost.
 *
 * rAF never fires in a hidden tab (or a test environment that has none), so the timeout also
 * stands in for it rather than waiting forever.
 */
function paintOnce(): Promise<void> {
  return new Promise((resolve) => {
    let done = false;
    const finish = () => { if (!done) { done = true; resolve(); } };
    if (typeof requestAnimationFrame === "function") {
      requestAnimationFrame(() => setTimeout(finish, 0));
      setTimeout(finish, 50); // hidden tab: no rAF is ever delivered
    } else {
      setTimeout(finish, 0);
    }
  });
}

/**
 * How many compiled models are kept for re-use.
 *
 * Compiling one LIBERO bundle is the expensive step of opening a scene - seconds of mesh and
 * texture parsing - and a model holds its own copy of both, so this cannot be unbounded. It was
 * four; two is what the measurement says it has to be. Browsing six live bundles in one worker,
 * alternating, `HEAPU8.byteLength` after each load:
 *
 *     cache of 4:  872 -> 1515 -> 1671 -> 1815 -> 1926 -> ... -> 2147 (the ceiling), and the
 *                  twelfth load fails with "Could not allocate memory"
 *     cache of 2:  842 -> 1509 -> 1671 -> 1671 -> 1671 -> ... -> 1815, fifteen loads, no failure
 *
 * The heap never shrinks, so every model kept is heap spent for the life of the tab. Two still
 * covers the compare page's pair and going back and forth between two tasks; the third visit
 * recompiles, which costs a second the reader can see rather than a tab that eventually cannot
 * open any scene at all.
 *
 * **Two is the default, not the only cap.** A *heavy* scene - the registry's own flag, a RoboCasa
 * kitchen today ({@link loadSceneBundle}'s `heavy`) - caps the cache at one: the spike measured
 * 1 146 MB of heap for a single texture-capped kitchen against the same 2 GiB ceiling, so a second
 * cached model of any size is close to the allocation that fails. The pair the compare page mounts
 * is the reason the default stays two, and the reason a pair with a kitchen in it is replayed as
 * video instead (`viewer/stage.ts`).
 *
 * The cap is a property of *what is in the cache*, not of the load that happens to be running: see
 * {@link cacheCap}. A kitchen released and left cached still holds the cache to one, so the next
 * LIBERO scene replaces it rather than joining it.
 */
export const MAX_CACHED_MODELS = 2;

/** What the caller is waiting on, so the badge can say. `total` is the number of assets this
 *  load has to fetch, which is zero for a bundle the WASM FS already holds. */
export type SceneProgress =
  | { phase: "fetching"; loaded: number; total: number }
  | { phase: "compiling" };

/** One compiled model plus how many viewers are currently holding it. A model is only
 *  {@link MjModel.delete}d once nothing is using it, so an eviction can never pull the scene out
 *  from under a canvas that is still rendering it.
 *
 *  `heavy` is the suite's registry flag, carried by the *entry* rather than only by the load that
 *  compiled it: a kitchen costs its gigabyte for as long as it is cached, not only while it is
 *  being compiled, and the cap has to know that on every later release and insert. */
interface CachedModel { model: any; users: number; heavy: boolean }

/** Compiled models, least recently used first (a `Map` iterates in insertion order, and a
 *  re-used entry is re-inserted). Module-level, i.e. one per tab, like the FS dirs above. */
const modelCache = new Map<string, CachedModel>();

/** xmlUrl -> the compile in flight for it, so two viewers mounting on the same bundle compile
 *  once and share the result instead of paying for it twice.
 *
 * `stale` holds one predicate per caller waiting on that compile. The compile checks them in the
 * last moment it still can — after the bundle is in the FS, before `mj_loadXML` takes the thread
 * for seconds — and skips the compile only if *every* waiter has given up on it. One viewer
 * moving on must not cancel a scene another viewer is still waiting for. */
interface PendingCompile { promise: Promise<CachedModel>; stale: Set<() => boolean> }
const compilingModels = new Map<string, PendingCompile>();

/** Thrown by {@link loadSceneBundle} when every caller cancelled before the compile began. Its
 *  own class so a caller can tell "you asked me to stop" from a real failure and stay quiet. */
export class SceneCancelled extends Error {
  constructor(message = "scene load cancelled") {
    super(message);
    this.name = "SceneCancelled";
  }
}

/** A viewer's hold on a cached model, with its own `MjData`.
 *
 * The data is *not* shared: two viewers of one bundle (the compare page) each set `qpos` and run
 * `mj_forward` on their own, and one `MjData` between them would have each frame overwrite the
 * other's pose. It is cheap next to the model, which is the part worth caching.
 */
export interface SceneHandle { model: any; data: any; release: () => void }

/**
 * The WASM module's heap, in bytes.
 *
 * The number that matters for the failure this exists for: WebAssembly memory grows and never
 * shrinks, and this build is compiled with a 2 GiB maximum (32768 pages of 64 KiB from an initial
 * 17 MB), so what a tab has spent it has spent — a scene that compiles happily in a fresh tab can
 * fail with "Could not allocate memory" in one that has been browsing for a while.
 *
 * **Reading it is not as simple as `Module.HEAPU8`.** This build does not export it: Emscripten
 * leaves a guarded accessor in its place that calls `abort()`, and an abort is permanent — the
 * module is dead for the rest of its life, every later call throwing `Aborted`. Measured, the hard
 * way, by a probe that hung a worker doing exactly that. So:
 *
 * - Any typed array the module hands out (`data.qpos`, `model.names`, …) is a *view onto its
 *   memory*, and `view.buffer.byteLength` is the heap size, exactly, for free. That is the
 *   primary source, and the last one seen is remembered for the moments when there is no model to
 *   hand (a load that failed).
 * - Failing that, `HEAPU8`/`wasmMemory` are read **only if they are plain data properties** — a
 *   descriptor check, never a property read, because reading a guarded one is the abort above.
 * - Failing that, 0: "not known", which the viewer shows by leaving the attribute off.
 */
let lastHeapView: ArrayBufferView | null = null;

/** Remember a typed array the module handed us, as a handle on its memory. Growing the memory
 *  detaches the old buffer (`byteLength` 0), which is why this is refreshed on every load. */
export function noteHeapView(view: unknown): void {
  if (ArrayBuffer.isView(view as ArrayBufferView)) lastHeapView = view as ArrayBufferView;
}

/** A property's value only if it is a plain one. Emscripten's placeholder for an unexported
 *  runtime method is a getter that aborts the module, so it must never be invoked. */
function plainProperty(obj: any, name: string): any {
  if (!obj || typeof obj !== "object") return undefined;
  const descriptor = Object.getOwnPropertyDescriptor(obj, name);
  return descriptor && !descriptor.get ? descriptor.value : undefined;
}

export function heapBytes(mujoco: any, sample?: unknown): number {
  if (ArrayBuffer.isView(sample as ArrayBufferView)) lastHeapView = sample as ArrayBufferView;
  const seen = lastHeapView?.buffer?.byteLength ?? 0;
  if (seen > 0) return seen;
  const heap = plainProperty(mujoco, "HEAPU8");
  if (typeof heap?.byteLength === "number") return heap.byteLength;
  const memory = plainProperty(mujoco, "wasmMemory");
  if (typeof memory?.buffer?.byteLength === "number") return memory.buffer.byteLength;
  return 0;
}

/**
 * A typed array the module still owns, over its *current* memory.
 *
 * Growing WebAssembly memory detaches every view onto the old buffer, so the view remembered by
 * the last load reads 0 bytes exactly when a load has just failed for want of memory — the moment
 * the number is worth having. Embind rebuilds these views on each property access, so any model
 * still in the cache answers it. The engine prefers a live session's `data.qpos` and falls back to
 * this.
 */
export function cachedHeapView(): ArrayBufferView | null {
  for (const entry of modelCache.values()) {
    const names = entry.model?.names;
    if (ArrayBuffer.isView(names) && names.buffer.byteLength > 0) return names;
  }
  return null;
}

/**
 * Does this failure look like the WASM heap running out?
 *
 * Emscripten reports the condition in several voices depending on where it hit: MuJoCo's own
 * `mju_error` ("Could not allocate memory"), the allocator's abort path ("Aborted", "Cannot
 * enlarge memory arrays", "out of memory"), or a `RangeError` from the JS engine when a buffer
 * allocation fails. Recognising it is what turns a dead viewer into a retry.
 *
 * Matched on the message even for a `RangeError`: "Maximum call stack size exceeded" is one too,
 * and the rung this decides terminates a worker and kills every session it holds, so a false
 * positive is expensive.
 */
export function isOutOfMemory(e: unknown): boolean {
  const message = e instanceof Error ? e.message : String(e);
  if (e instanceof RangeError) return /array buffer allocation failed|invalid (typed )?array length|memory/i.test(message);
  return /could not allocate|out of memory|\boom\b|cannot enlarge memory|memory allocation failed|\baborted\b/i.test(message);
}

/**
 * Give back everything this module is holding that nothing is using: every compiled model no
 * viewer has open, and every materialised bundle directory no load is still reading.
 *
 * The recovery step for an allocation failure. It cannot free what is in use — deleting a model a
 * viewer is still showing would take that viewer's scene down with it — and freeing does not
 * shrink the heap in any case: this is about making the heap the tab already has usable again.
 * When it is not enough, the client's answer is a fresh worker, which is the only way a WASM heap
 * is ever really reclaimed.
 */
export function purgeSceneMemory(mujoco: any): { models: number; dirs: number } {
  let models = 0;
  for (const [url, entry] of [...modelCache]) {
    // `users` is 0 for a moment on every freshly compiled model, between the insert and the
    // `holdModel` its caller is on its way to; `compilingModels` still holds that load, so this
    // is the same guard `evictModels` takes as `keep`.
    if (entry.users > 0 || compilingModels.has(url)) continue;
    modelCache.delete(url);
    disposeMujoco(entry.model, null);
    models++;
  }
  const held = new Set(loading.keys());
  let dirs = 0;
  for (const dir of [...sceneDirs]) {
    if (held.has(dir)) continue;
    removeSceneDir(mujoco.FS as SceneFs, dir, sceneDirFiles.get(dir) ?? []);
    sceneDirs.splice(sceneDirs.indexOf(dir), 1);
    sceneDirFiles.delete(dir);
    forgetMaterialized(dir);
    dirs++;
  }
  return { models, dirs };
}

/**
 * How many compiled models this cache may hold *right now*: one while it holds a heavy model,
 * {@link MAX_CACHED_MODELS} otherwise.
 *
 * Asked on every release and every insert rather than fixed by the load that is running, because
 * the gigabyte a kitchen occupies is spent for as long as the model is cached. The sequence that
 * made this necessary: open a RoboCasa run (kitchen cached, heap ~1 150 MB), navigate to a LIBERO
 * run. The kitchen is released - cap 2, one entry, nothing dropped - and the LIBERO model then
 * inserts under cap 2 as well, so the tab ends up holding ~1.15 GB + ~0.8 GB against a 2 GiB
 * ceiling, having never opened a second kitchen. With the flag on the entry, the LIBERO insert
 * sees cap 1 and the kitchen goes.
 *
 * The other direction takes care of itself: once the last heavy entry is evicted, this reads 2
 * again and two light scenes can share the cache as they always could.
 */
function cacheCap(): number {
  for (const entry of modelCache.values()) if (entry.heavy) return 1;
  return MAX_CACHED_MODELS;
}

/**
 * How much room to leave *before* compiling a scene with this `heavy` flag: the cap the cache will
 * be under once the new model is in it, minus the slot that model is about to take.
 *
 * `mj_loadXML` allocates the whole model while everything already cached is still resident, and
 * evicting only after the insert means a second kitchen is compiled *beside* the first: ~2.3 GB
 * against a 2 GiB ceiling, i.e. a failed compile on essentially every kitchen-to-kitchen
 * navigation. The OOM ladder recovers it (`purgeSceneMemory`, then a retry, then a fresh worker),
 * so it was never a broken viewer — it was a wasted compile, every time, in the one place this
 * flag exists to protect. Freeing first is what the ladder would have done anyway, without the
 * failure.
 *
 * Zero is a legitimate answer, and the one a heavy load usually gets: nothing released stays.
 */
function capBeforeCompile(heavy: boolean): number {
  return Math.max(0, (heavy ? 1 : cacheCap()) - 1);
}

/**
 * Drop cached models past `cap` ({@link cacheCap} unless the caller says otherwise), oldest first,
 * skipping any a viewer still holds and the one named by `keep`.
 *
 * Called on a release, before a compile ({@link capBeforeCompile}) and after the insert. The
 * insert is the case that used to be missing: a load whose caller gave up *during* the compile
 * leaves a compiled model in the cache that nobody ever holds and therefore nobody ever releases,
 * so a reader clicking through tasks faster than they compile could grow the cache without bound —
 * hundreds of megabytes of WASM heap, against a 2 GiB ceiling.
 *
 * `keep` is the entry just inserted: it has no users yet (its caller is still on its way to
 * `holdModel`) and is the last in iteration order, so without this it would be the one evicted
 * whenever every older entry is held.
 *
 * `cap` is 1 whenever a heavy model is in the cache, which is what makes room for it: everything
 * the tab was keeping for convenience goes, and the kitchen is the one model left. It cannot free
 * a model another viewer is holding, so it is a cap on *cached* models, not a guarantee of one -
 * which is why the compare page changes the stage instead of relying on this.
 */
function evictModels(keep?: string, cap: number = cacheCap()): void {
  for (const [url, entry] of [...modelCache]) {
    if (modelCache.size <= cap) break;
    // `compilingModels` is the same guard `purgeSceneMemory` takes, and it matters more now that a
    // cap of 1 (or 0) can reach past the entry named by `keep`: a model inserted by *another*
    // load that has not reached its `holdModel` yet still has no users, and deleting it would
    // leave that caller building an `MjData` from a freed model.
    if (entry.users > 0 || url === keep || compilingModels.has(url)) continue;
    modelCache.delete(url);
    disposeMujoco(entry.model, null);
  }
}

function holdModel(mujoco: any, url: string, entry: CachedModel): SceneHandle {
  // The MjData first, and the bookkeeping only once it exists: `mj_makeData` allocates the whole
  // `<size memory>` arena and reports failure as "Could not allocate memory", and an increment
  // left behind by a throw would pin this model for the life of the worker - invisible to
  // `evictModels` and to `purgeSceneMemory`, both of which skip anything with users, so the very
  // model whose MjData just failed could never be given back.
  const data = new mujoco.MjData(entry.model);
  entry.users++;
  modelCache.delete(url);
  modelCache.set(url, entry); // most recently used
  let released = false;
  return {
    model: entry.model,
    data,
    release: () => {
      if (released) return;
      released = true;
      disposeMujoco(null, data); // the data is this viewer's; the model belongs to the cache
      entry.users = Math.max(0, entry.users - 1);
      evictModels();
    },
  };
}

/**
 * Compile a scene bundle into a MuJoCo model, materializing it into the WASM FS first.
 *
 * The FS holds one directory per bundle (see {@link sceneDirFromXmlUrl}) so that two viewers on
 * the same page can show two different bundles, and re-mounting a scene already in the FS costs
 * no network. Two things keep that cheap rather than unbounded: writes for one dir happen once
 * even under concurrent callers, and no more than {@link MAX_SCENE_DIRS} dirs are kept.
 *
 * On top of that the *compiled* model is cached by bundle URL, so going back to a task already
 * visited costs neither the network nor the parse. The caller owns the returned handle and must
 * `release()` it; until every holder has, the model stays out of the eviction's reach.
 */
export async function loadSceneBundle(
  mujoco: any, xmlUrl: string, assetsBase: string, onProgress?: (p: SceneProgress) => void,
  /** Asked, at every point this load can still be abandoned cheaply, whether the caller has
   *  given up. A load already inside `mj_loadXML` cannot be stopped — WASM has no yield — so
   *  the last such point is the instant before it starts. */
  stale: () => boolean = () => false,
  /** `<size memory>` for this load. The cache key includes it: the same bundle with a bigger
   *  arena is a different compiled model, not a cache hit that would fail the same way. */
  arena: string = VIEWER_ARENA,
  /** The suite's `scene.heavy`, from the registry by way of the page (`suiteIsHeavy` ->
   *  `SceneClient.load` -> the worker). It is recorded on the cache entry, and caps the
   *  compiled-model cache at one for as long as that model is cached - not merely while it
   *  compiles: see {@link cacheCap} and {@link MAX_CACHED_MODELS}. */
  heavy: boolean = false,
): Promise<SceneHandle> {
  if (stale()) throw new SceneCancelled();
  const key = `${xmlUrl}#${arena}`;
  const cached = modelCache.get(key);
  if (cached) return holdModel(mujoco, key, cached);

  let pending = compilingModels.get(key);
  if (!pending) {
    const stales = new Set<() => boolean>();
    pending = { promise: compileSceneBundle(mujoco, xmlUrl, assetsBase, stales, arena, heavy, onProgress), stale: stales };
    compilingModels.set(key, pending);
  }
  pending.stale.add(stale);
  let entry: CachedModel;
  try {
    entry = await pending.promise;
  } finally {
    pending.stale.delete(stale);
    if (compilingModels.get(key) === pending) compilingModels.delete(key);
  }
  if (stale()) throw new SceneCancelled();
  return holdModel(mujoco, key, entry);
}

async function compileSceneBundle(
  mujoco: any, xmlUrl: string, assetsBase: string, stales: Set<() => boolean>, arena: string,
  heavy: boolean, onProgress?: (p: SceneProgress) => void,
): Promise<CachedModel> {
  const dir = sceneDirFromXmlUrl(xmlUrl);
  loading.set(dir, (loading.get(dir) ?? 0) + 1);
  try {
    // Room first, work second. Everything below - the bundle's bytes into the WASM FS, then the
    // model `mj_loadXML` builds out of them - lands in the same heap as whatever is still cached,
    // and that heap never shrinks. So the models this load is going to make room for are given
    // back *before* it starts rather than after it has inserted its own: see
    // {@link capBeforeCompile}. Nothing a viewer is holding is touched, so this can free less than
    // it asks for, which is the same promise the cap has always made.
    evictModels(undefined, capBeforeCompile(heavy));
    // Keyed by dir *and* arena: the assets are written once (the second call finds them there),
    // the patched xml once per arena.
    const materializeKey = `${dir}#${arena}`;
    let materialized = materializing.get(materializeKey);
    if (!materialized) {
      materialized = measured(PERF_FETCH, () => materializeSceneBundle(mujoco, xmlUrl, assetsBase, dir, arena, (loaded, total) =>
        onProgress?.({ phase: "fetching", loaded, total })));
      materializing.set(materializeKey, materialized);
    }
    try {
      await materialized;
    } catch (e) {
      // A failed fetch must not be remembered as "this bundle is in the FS": the next attempt
      // (a retry, or another viewer) has to be allowed to try again.
      if (materializing.get(materializeKey) === materialized) materializing.delete(materializeKey);
      throw e;
    }
    touchSceneDir(sceneDirs, dir);
    onProgress?.({ phase: "compiling" });
    // Let a frame paint first. In the worker there is nothing to paint and no rAF, so this is
    // just a turn of the event loop - which is what matters there: a `cancel` message posted
    // while the bundle was being fetched is delivered in it, and the check below then skips a
    // compile nobody is waiting for any more. On the main thread (the no-Worker fallback)
    // mj_loadXML blocks for seconds, so the badge that says so has to be on screen before it
    // starts, and "on screen" means a rendered frame rather than a queued React update.
    await paintOnce();
    // The last moment this load can still be abandoned: once mj_loadXML is entered, nothing -
    // not a cancel, not terminating the worker's message queue - stops it short of finishing.
    if (stales.size > 0 && [...stales].every((f) => f())) throw new SceneCancelled();
    const xmlPath = `${dir}/${sceneXmlName(arena)}`;
    const entry: CachedModel = { model: measured(PERF_COMPILE, () => mujoco.MjModel.mj_loadXML(xmlPath)), users: 0, heavy };
    const key = `${xmlUrl}#${arena}`;
    modelCache.set(key, entry);
    // On insert, not only on release: a model whose caller cancelled while it was compiling is
    // never held and so would never be released. The cap comes from what is in the cache - this
    // entry included, hence the insert first - so a heavy scene keeps nothing else company, and
    // neither does anything loaded while one is still cached.
    evictModels(key);
    return entry;
  } finally {
    const held = (loading.get(dir) ?? 1) - 1;
    if (held <= 0) loading.delete(dir); else loading.set(dir, held);
    // After the compile, not before: the model now holds its own meshes and textures, so this
    // dir is as evictable as any other.
    evictSceneDirs(mujoco.FS as SceneFs);
  }
}

/** One NUL-terminated name out of the model's packed `names` blob. */
function nameAt(names: Uint8Array, adr: number): string {
  let e = adr;
  while (e < names.length && names[e] !== 0) e++;
  return new TextDecoder().decode(names.subarray(adr, e));
}

/**
 * Per-vertex normals for a mesh, computed here rather than by three.js on the main thread.
 *
 * Same result as `BufferGeometry.computeVertexNormals`: an indexed mesh gets area-weighted
 * normals averaged over the faces sharing each vertex; a non-indexed one gets the face normal on
 * all three of its corners. It is a few million floats per bundle - exactly the kind of work the
 * worker exists to keep off the thread that has to stay responsive.
 */
export function computeNormals(positions: Float32Array, indices: MeshIndices): Float32Array {
  const normals = new Float32Array(positions.length);
  const tris = indices ? indices.length / 3 : positions.length / 9;
  for (let t = 0; t < tris; t++) {
    const a = (indices ? indices[t * 3] : t * 3) * 3;
    const b = (indices ? indices[t * 3 + 1] : t * 3 + 1) * 3;
    const c = (indices ? indices[t * 3 + 2] : t * 3 + 2) * 3;
    const abx = positions[b] - positions[a], aby = positions[b + 1] - positions[a + 1], abz = positions[b + 2] - positions[a + 2];
    const acx = positions[c] - positions[a], acy = positions[c + 1] - positions[a + 1], acz = positions[c + 2] - positions[a + 2];
    const nx = aby * acz - abz * acy, ny = abz * acx - abx * acz, nz = abx * acy - aby * acx;
    normals[a] += nx; normals[a + 1] += ny; normals[a + 2] += nz;
    normals[b] += nx; normals[b + 1] += ny; normals[b + 2] += nz;
    normals[c] += nx; normals[c + 1] += ny; normals[c + 2] += nz;
  }
  for (let i = 0; i < normals.length; i += 3) {
    const len = Math.hypot(normals[i], normals[i + 1], normals[i + 2]) || 1;
    normals[i] /= len; normals[i + 1] /= len; normals[i + 2] /= len;
  }
  return normals;
}

/** One mesh of the model, in the layout three.js takes. A mesh MuJoCo gave texture coordinates
 *  must be non-indexed: `mesh_face` and `mesh_facetexcoord` index positions and texcoords
 *  independently per corner, so one vertex can carry different UVs on different faces. */
function meshData(model: any, m: number): MeshData {
  const vAdr = model.mesh_vertadr[m], vNum = model.mesh_vertnum[m];
  const fAdr = model.mesh_faceadr[m], fNum = model.mesh_facenum[m];
  const texAdr = model.mesh_texcoordadr ? model.mesh_texcoordadr[m] : -1;
  if (texAdr >= 0) {
    const positions = new Float32Array(fNum * 3 * 3);
    const uvs = new Float32Array(fNum * 3 * 2);
    for (let i = 0; i < fNum; i++) {
      for (let k = 0; k < 3; k++) {
        const corner = (fAdr + i) * 3 + k;
        const vi = vAdr + model.mesh_face[corner];
        const ti = texAdr + model.mesh_facetexcoord[corner];
        const dst = i * 3 + k;
        positions[dst * 3] = model.mesh_vert[vi * 3];
        positions[dst * 3 + 1] = model.mesh_vert[vi * 3 + 1];
        positions[dst * 3 + 2] = model.mesh_vert[vi * 3 + 2];
        uvs[dst * 2] = model.mesh_texcoord[ti * 2];
        uvs[dst * 2 + 1] = 1 - model.mesh_texcoord[ti * 2 + 1]; // MuJoCo's V axis is flipped vs. three's
      }
    }
    return { positions, uvs, indices: null, normals: computeNormals(positions, null) };
  }
  const positions = Float32Array.from(model.mesh_vert.slice(vAdr * 3, (vAdr + vNum) * 3));
  const face = model.mesh_face.slice(fAdr * 3, (fAdr + fNum) * 3);
  // 16-bit indices where the mesh fits in them, which is every LIBERO mesh: `mesh_face` is
  // mesh-local, so `vertnum` is the bound. three would have picked the narrow type itself from a
  // plain array (`setIndex(Array)`); handing it a typed array means choosing here instead, and
  // 32-bit indices for a 900-vertex mesh are twice the GPU memory for nothing.
  const indices = vNum <= 65535 ? Uint16Array.from(face) : Uint32Array.from(face);
  return { positions, uvs: null, indices, normals: computeNormals(positions, indices) };
}

/** A material's RGB texture as RGBA bytes, whatever channel count MuJoCo stored it with. */
function textureData(model: any, texId: number): TextureData {
  const w = model.tex_width[texId], h = model.tex_height[texId], ch = model.tex_nchannel[texId], off = Number(model.tex_adr[texId]);
  const rgba = new Uint8Array(w * h * 4);
  const src = model.tex_data;
  for (let p = 0; p < w * h; p++) {
    rgba[p * 4] = src[off + p * ch];
    rgba[p * 4 + 1] = ch > 1 ? src[off + p * ch + 1] : rgba[p * 4];
    rgba[p * 4 + 2] = ch > 2 ? src[off + p * ch + 2] : rgba[p * 4];
    rgba[p * 4 + 3] = 255;
  }
  return { width: w, height: h, rgba };
}

/**
 * Read one compiled model into the plain buffers a stage is built from.
 *
 * This is the whole of the MuJoCo-shaped knowledge that used to live in `buildThreeScene`: geom
 * types become {@link GeomKind} names, meshes and textures are copied out of the WASM heap (once
 * each, however many geoms share them) and nothing that leaves here points back into the module.
 * `threeStage.ts` turns the result into three.js objects and knows nothing about MuJoCo.
 */
export function extractSceneGeometry(mujoco: any, model: any, scene: string): SceneGeometry {
  const G = mujoco.mjtGeom;
  const kinds = new Map<number, GeomKind>([
    [G.mjGEOM_PLANE.value, "plane"], [G.mjGEOM_SPHERE.value, "sphere"], [G.mjGEOM_CAPSULE.value, "capsule"],
    [G.mjGEOM_CYLINDER.value, "cylinder"], [G.mjGEOM_BOX.value, "box"], [G.mjGEOM_ELLIPSOID.value, "ellipsoid"],
    [G.mjGEOM_MESH.value, "mesh"],
  ]);
  const names = new Uint8Array(model.names);
  const bodyNames: string[] = [];
  for (let b = 0; b < model.nbody; b++) bodyNames.push(nameAt(names, model.name_bodyadr[b]));
  const cameras: { name: string; fovy: number }[] = [];
  for (let c = 0; c < model.ncam; c++) cameras.push({ name: nameAt(names, model.name_camadr[c]), fovy: model.cam_fovy[c] });

  const meshes: MeshData[] = [];
  const meshIndex = new Map<number, number>();
  const textures: TextureData[] = [];
  const textureIndex = new Map<number, number>();
  const geoms: GeomSpec[] = [];
  for (let g = 0; g < model.ngeom; g++) {
    if (!VISIBLE_GEOM_GROUPS.has(model.geom_group[g])) continue;
    const kind = kinds.get(model.geom_type[g]);
    if (!kind) continue; // heightfields, sdfs, decorations: the viewer draws none of them
    let mesh = -1;
    if (kind === "mesh") {
      const m = model.geom_dataid[g];
      mesh = meshIndex.get(m) ?? -1;
      if (mesh < 0) { mesh = meshes.push(meshData(model, m)) - 1; meshIndex.set(m, mesh); }
    }
    let rgba: [number, number, number, number] = [model.geom_rgba[g * 4], model.geom_rgba[g * 4 + 1], model.geom_rgba[g * 4 + 2], model.geom_rgba[g * 4 + 3]];
    let texture = -1;
    let texRepeat: [number, number] = [1, 1];
    const matId = model.geom_matid[g];
    if (matId >= 0) {
      rgba = [model.mat_rgba[matId * 4], model.mat_rgba[matId * 4 + 1], model.mat_rgba[matId * 4 + 2], model.mat_rgba[matId * 4 + 3]];
      const texId = model.mat_texid[matId * 10 + 1]; // mjNTEXROLE = 10, RGB role = 1
      if (texId >= 0) {
        texture = textureIndex.get(texId) ?? -1;
        if (texture < 0) { texture = textures.push(textureData(model, texId)) - 1; textureIndex.set(texId, texture); }
        texRepeat = [model.mat_texrepeat[matId * 2], model.mat_texrepeat[matId * 2 + 1]];
      }
    }
    geoms.push({
      kind,
      body: model.geom_bodyid[g],
      size: [model.geom_size[g * 3], model.geom_size[g * 3 + 1], model.geom_size[g * 3 + 2]],
      pos: [model.geom_pos[g * 3], model.geom_pos[g * 3 + 1], model.geom_pos[g * 3 + 2]],
      quat: [model.geom_quat[g * 4], model.geom_quat[g * 4 + 1], model.geom_quat[g * 4 + 2], model.geom_quat[g * 4 + 3]],
      rgba, mesh, texture, texRepeat,
    });
  }
  return { scene, nbody: model.nbody, bodyNames, geoms, meshes, textures, cameras, nq: model.nq, qpos0: Float32Array.from(model.qpos0) };
}

/**
 * Where every body and camera ends up for one `qpos`.
 *
 * Forward kinematics only — the viewer never steps physics — and the arrays are copies, not
 * views onto the WASM heap, so they can be transferred to the main thread and read there while
 * the next frame is already being computed here.
 */
export function bodyPose(mujoco: any, model: any, data: any, qpos: Float32Array): BodyPose {
  if (qpos.length !== model.nq) throw new Error(`qpos length ${qpos.length} != model.nq ${model.nq}`);
  noteHeapView(data.qpos); // a view onto the WASM memory, which is how its size is read (heapBytes)
  data.qpos.set(qpos);
  mujoco.mj_forward(model, data);
  return {
    xpos: copyFloats(data.xpos, model.nbody * 3),
    xquat: copyFloats(data.xquat, model.nbody * 4),
    camPos: copyFloats(data.cam_xpos, model.ncam * 3),
    camMat: copyFloats(data.cam_xmat, model.ncam * 9),
  };
}

/** The first `n` values of a MuJoCo array as a `Float32Array` of its own — a copy, because
 *  everything the worker sends is transferred and must not be a view onto the WASM heap. */
function copyFloats(src: ArrayLike<number>, n: number): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = src[i];
  return out;
}
