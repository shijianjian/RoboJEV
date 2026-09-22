import { PERF_COMPILE, PERF_FETCH, SceneCancelled, type SceneProgress } from "./mujoco";
import type { BodyPose, SceneChannel, SceneEvent, SceneGeometry, SceneRequest, SceneTiming } from "./sceneProtocol";

/**
 * The main thread's half of the MuJoCo worker: generations, cancellation, and one worker shared
 * by every viewer on the page.
 *
 * What the components see is {@link SceneClient.load}: a promise for a scene, and a `cancel` that
 * means "I have moved on" — which is the whole point of this architecture. Clicking another task
 * while one is compiling used to mean waiting for a scene nobody wanted any more, because the
 * compile held the only thread there was. Now the click is handled at once, the load in flight is
 * abandoned where it can be, and the scene already on the canvas keeps rendering (and orbiting)
 * throughout.
 *
 * One worker, not one per viewer: the compare page mounts two viewers, and a second WASM module
 * would double the ~2 GiB-ceilinged heap and re-fetch every bundle. Serialising the two loads
 * costs nothing the compile was not going to cost anyway.
 *
 * **Why a load behind a stale compile waits rather than starting a fresh worker.**
 * A compile that has started cannot be stopped — WASM does not yield — so a load requested while
 * one is running waits out its remainder. The alternative considered (and prototyped, and
 * measured) was to terminate the worker and spawn another, so the abandoned compile cannot delay
 * the scene the reader actually asked for.
 *
 * Measured in Chrome against a production build, 2026-09-13 (detail in the report). Clicking
 * away from a task mid-compile and back again, from the second click to the scene on screen,
 * median of three:
 *
 * | policy | to ready |
 * | --- | --- |
 * | wait out the stale compile (this) | **4.9 s** |
 * | terminate and spawn a fresh worker | 6.2 s |
 *
 * Terminating buys the remainder of the stale compile, and a fresh module worker with MuJoCo
 * initialised in it is cheap — 171 ms warm, 959 ms cold. But it throws away everything the worker
 * was holding, and that is worth more: the stale compile's own model goes into the cache, so
 * waiting it out leaves the next load to serve from memory (no fetch, no compile), while the
 * fresh worker has to re-materialise the bundle (0.7 s, from the HTTP cache) and compile it again
 * (2.9 s).
 *
 * So waiting is what this client does, always. The restart machinery below ({@link reset}) stays
 * for the case it is actually right for: a worker that has *died*, where there is nothing left to
 * preserve.
 */

/** A loaded scene, held by one viewer. Releasing it frees that viewer's `MjData` in the worker
 *  and lets the compiled model be evicted once nothing else holds it. */
export interface SceneSession {
  readonly geometry: SceneGeometry;
  /**
   * Pose every body for one `qpos`, in the worker.
   *
   * Resolves to `null` when the answer no longer matters — another pose was requested before this
   * one was sent (scrubbing outruns the round trip), or the viewer has released the scene — which
   * callers treat as "keep the frame you have". *Rejects* if the scene's worker is gone: that
   * frame is not merely late, it can never be computed, and the viewer has to say so rather than
   * leave a stale pose under a chip reading `ready`.
   */
  pose(qpos: Float32Array): Promise<BodyPose | null>;
  release(): void;
}

/** A load in flight. `cancel` retires it: the promise rejects with {@link SceneCancelled}, and
 *  the worker drops whatever it was doing for it as soon as it can. */
export interface SceneLoad {
  promise: Promise<SceneSession>;
  cancel: () => void;
}

export interface SceneClient {
  /** `heavy` is the suite's registry flag (`suiteIsHeavy`), carried down to the compile so a
   *  RoboCasa kitchen caps the worker's compiled-model cache at one. */
  load(xmlUrl: string, assetsBase: string, onProgress?: (p: SceneProgress) => void, heavy?: boolean): SceneLoad;
  /** The WASM heap as of the last load, in bytes, or 0 before there has been one. The viewers put
   *  it on the stage as `data-heap-mb`: it only ever grows, and this build dies at 2 GiB. */
  heapBytes(): number;
  /** Where scenes are being compiled right now. The viewers put it on the stage as `data-engine`,
   *  because "the page froze" and "the fallback engaged" are the same symptom and only this tells
   *  them apart. `"none"` until the first load, which is when a channel is made. */
  engine(): SceneEngineKind;
}

/** `"worker"` is this commit's whole point; `"in-process"` is the fallback, which behaves exactly
 *  as the viewer did before it — compile on the main thread, freeze and all. */
export type SceneEngineKind = "worker" | "in-process" | "none";

/** How many times a worker may die before the tab gives up on workers altogether. A worker that
 *  fails to start usually fails the same way every time (a blocked CDN, a bad chunk), but a
 *  one-off — an out-of-memory kill on a huge bundle, say — must not cost every later scene the
 *  main thread. */
export const MAX_WORKER_ATTEMPTS = 3;

/** What the client remembers about a load it has asked for and not yet resolved. */
interface PendingLoad {
  xmlUrl: string;
  assetsBase: string;
  /** Replayed with the load on a respawn: a fresh worker has to be told the same thing about
   *  this scene the dead one was, or the kitchen it recompiles gets the light cache cap. */
  heavy: boolean;
  onProgress?: (p: SceneProgress) => void;
  resolve: (s: SceneSession) => void;
  reject: (e: unknown) => void;
  /** Set once this load has already been given a fresh worker after an allocation failure. The
   *  ladder is: the worker drops its caches and retries, then this, then the error is the
   *  reader's. Each rung is tried once. */
  triedFreshWorker?: boolean;
}

/** Whoever is waiting on one pose. Resolved with `null` when the answer stopped mattering,
 *  rejected when it can never come (see {@link SceneSession.pose}). */
interface PoseWaiter {
  resolve: (p: BodyPose | null) => void;
  reject: (e: unknown) => void;
}

/** What the client remembers about a scene a viewer is holding. */
interface SessionState {
  geometry: SceneGeometry;
  /** Cleared when the worker this session lived in is gone: its poses can no longer be computed. */
  alive: boolean;
  seq: number;
  /** The pose request the worker is working on, and who is waiting for it. */
  inFlight: ({ seq: number } & PoseWaiter) | null;
  /** At most one waiting request: a newer `qpos` makes an older waiting one pointless. */
  queued: ({ qpos: Float32Array } & PoseWaiter) | null;
}

/** How a channel is made: given the callback its events are delivered to, and one to call if the
 *  channel dies under it. Returns null when this browser cannot run a module worker, which puts
 *  the engine on this thread instead. */
export type SceneChannelFactory = (onEvent: (e: SceneEvent) => void, onFail: (why: string) => void) => SceneChannel | null;

export function createSceneClient(spawn: SceneChannelFactory = spawnSceneWorker): SceneClient {
  let channel: SceneChannel | null = null;
  /** How many workers have died under this client. Past {@link MAX_WORKER_ATTEMPTS} it stops
   *  spawning them. */
  let workerFailures = 0;
  let engine: SceneEngineKind = "none";
  let announced: SceneEngineKind = "none";
  let heap = 0;
  let generation = 0;
  const pending = new Map<number, PendingLoad>();
  const sessions = new Map<number, SessionState>();

  function ensure(): SceneChannel {
    if (channel) return channel;
    const spawned = workerFailures >= MAX_WORKER_ATTEMPTS ? null : spawn(onEvent, onChannelFailure);
    engine = spawned ? "worker" : "in-process";
    channel = spawned ?? inProcessChannel(onEvent);
    // Said once per engine, not per load: a page that silently compiles on the main thread looks
    // exactly like a page whose worker is slow, and this is the line that tells them apart in a
    // bug report.
    if (engine !== announced) {
      announced = engine;
      const why = engine === "worker" ? "" : workerFailures > 0 ? " (worker died)" : " (no module worker in this browser)";
      console.info(`robojev viewer: engine=${engine}${why}`);
    }
    return channel;
  }

  /** The worker died (it failed to start, or threw where nothing could catch it). Everything it
   *  was holding is gone; the loads still wanted are re-issued, on a fresh worker if this tab has
   *  attempts left and on this thread otherwise. */
  function onChannelFailure(why: string): void {
    workerFailures++;
    console.error(`robojev viewer: mujoco worker failed (${why}), attempt ${workerFailures} of ${MAX_WORKER_ATTEMPTS}`);
    reset();
  }

  function post(msg: SceneRequest, transfer?: Transferable[]): void {
    ensure().post(msg, transfer);
  }

  /**
   * Throw away the channel and start again: every session it held is dead, every load it was
   * running is re-posted to the next one.
   *
   * A dead session cannot be posed again — its model and its `MjData` went with the worker — so
   * everyone waiting on one is told so with an error rather than a silent `null`. The viewer turns
   * that into its failure state; a scrubber that moves the frame counter over a scene that can no
   * longer move is worse than an honest "reload me".
   */
  function reset(): void {
    channel?.terminate();
    channel = null;
    // Unknown again until the next channel is made, which the re-posted loads below do straight
    // away: `engine()` must never name a channel that no longer exists, and `heapBytes()` must
    // not keep reporting a dead module's last figure.
    engine = "none";
    heap = 0;
    const gone = new Error("the scene's worker stopped; reload the page to get it back");
    for (const session of sessions.values()) {
      session.alive = false;
      session.inFlight?.reject(gone);
      session.inFlight = null;
      session.queued?.reject(gone);
      session.queued = null;
    }
    sessions.clear();
    // The loads still wanted are wanted on the new channel too, from the beginning.
    for (const [gen, load] of pending) post({ op: "load", gen, xmlUrl: load.xmlUrl, assetsBase: load.assetsBase, heavy: load.heavy });
  }

  function onEvent(e: SceneEvent): void {
    switch (e.op) {
      case "progress": {
        const load = pending.get(e.gen);
        if (!load) return;
        load.onProgress?.(e.phase === "fetching" ? { phase: "fetching", loaded: e.fetched, total: e.total } : { phase: "compiling" });
        return;
      }
      case "loaded": {
        heap = e.heapBytes;
        const load = pending.get(e.gen);
        // Cancelled while it was compiling: the worker has already released it (a `cancel` for a
        // loaded generation releases it), so there is nothing to do but forget it.
        if (!load) return;
        pending.delete(e.gen);
        recordTiming(e.timing);
        load.resolve(makeSession(e.gen, e.geometry));
        return;
      }
      case "failed": {
        heap = e.heapBytes;
        const load = pending.get(e.gen);
        if (!load) return;
        // The worker has already dropped every cached model and materialised bundle it could and
        // tried again. A WASM heap never shrinks, so the only thing left that can free one is a
        // new worker: give this load that, once. A respawn for this reason is not a worker
        // *failure* - the worker did nothing wrong - so it does not count against the death bound.
        // Only on a real worker. On the in-process engine `reset()` would kill every live session
        // without releasing the models they hold - and the new channel shares the very module that
        // ran out, so it reclaims nothing. There, the ladder ends at the worker's own purge.
        if (e.outOfMemory && !load.triedFreshWorker && engine === "worker") {
          load.triedFreshWorker = true;
          console.warn(`robojev viewer: scene ran the worker out of memory at ${Math.round(e.heapBytes / 1e6)} MB; starting a fresh one and trying once more`);
          reset(); // terminates, respawns, and re-posts every load still wanted, this one included
          return;
        }
        pending.delete(e.gen);
        load.reject(new Error(e.message));
        return;
      }
      case "pose": {
        const session = sessions.get(e.gen);
        if (!session?.inFlight || session.inFlight.seq !== e.seq) return;
        const { resolve } = session.inFlight;
        session.inFlight = null;
        resolve(e.pose);
        drain(e.gen, session);
        return;
      }
      case "poseFailed": {
        const session = sessions.get(e.gen);
        if (!session?.inFlight || session.inFlight.seq !== e.seq) return;
        const { resolve, reject } = session.inFlight;
        session.inFlight = null;
        // A released scene is not worth reporting - the viewer that asked is on its way out with
        // it - but a frame the worker could not compute for want of memory is. The worker has
        // already reloaded this scene with a bigger arena and failed again by the time this
        // arrives, so the alternative to an error is a stale frame under a chip reading `ready`.
        if (e.outOfMemory) reject(new Error(e.message));
        else resolve(null);
        drain(e.gen, session);
        return;
      }
    }
  }

  /** Send the request that piled up behind the one just answered, if any. */
  function drain(gen: number, session: SessionState): void {
    const next = session.queued;
    if (!next) return;
    session.queued = null;
    sendPose(gen, session, next.qpos, next);
  }

  function sendPose(gen: number, session: SessionState, qpos: Float32Array, waiter: PoseWaiter): void {
    const seq = ++session.seq;
    session.inFlight = { seq, ...waiter };
    // A copy of exactly this frame, which is then transferred. `qposAt` hands out a *view* into
    // the trajectory's own buffer (record.ts), and structured-cloning a view serialises the whole
    // buffer behind it - the entire trajectory, on every frame of playback. `slice` is the
    // smallest message there is, and owning it means it can be transferred rather than copied.
    const own = qpos.slice();
    post({ op: "pose", gen, seq, qpos: own }, [own.buffer]);
  }

  function makeSession(gen: number, geometry: SceneGeometry): SceneSession {
    const state: SessionState = { geometry, alive: true, seq: 0, inFlight: null, queued: null };
    sessions.set(gen, state);
    let released = false;
    return {
      geometry,
      pose(qpos: Float32Array): Promise<BodyPose | null> {
        if (released) return Promise.resolve(null);
        if (!state.alive) return Promise.reject(new Error("the scene's worker stopped; reload the page to get it back"));
        return new Promise<BodyPose | null>((resolve, reject) => {
          if (state.inFlight) {
            // Scrubbing faster than the round trip: only the newest frame is worth computing, so
            // the one waiting is dropped (its caller is told "keep what you have").
            state.queued?.resolve(null);
            state.queued = { qpos, resolve, reject };
            return;
          }
          sendPose(gen, state, qpos, { resolve, reject });
        });
      },
      release(): void {
        if (released) return;
        released = true;
        state.queued?.resolve(null);
        state.queued = null;
        state.inFlight?.resolve(null);
        state.inFlight = null;
        if (!state.alive) return;
        sessions.delete(gen);
        post({ op: "release", gen });
      },
    };
  }

  function cancel(gen: number): void {
    const load = pending.get(gen);
    if (!load) return;
    pending.delete(gen);
    channel?.post({ op: "cancel", gen });
    load.reject(new SceneCancelled());
  }

  return {
    engine: () => engine,
    heapBytes: () => heap,
    load(xmlUrl: string, assetsBase: string, onProgress?: (p: SceneProgress) => void, heavy = false): SceneLoad {
      const gen = ++generation;
      let resolve!: (s: SceneSession) => void;
      let reject!: (e: unknown) => void;
      const promise = new Promise<SceneSession>((res, rej) => { resolve = res; reject = rej; });
      pending.set(gen, { xmlUrl, assetsBase, heavy, onProgress, resolve, reject });
      post({ op: "load", gen, xmlUrl, assetsBase, heavy });
      return { promise, cancel: () => cancel(gen) };
    },
  };
}

/**
 * Re-record the worker's phase timings in the page's own timeline.
 *
 * `performance.measure` entries made inside a worker are invisible to the page and start from a
 * different time origin, so what comes back is durations, laid end to end against the moment the
 * geometry arrived. `scripts/scene-bench.ts` and anyone reading `robopp:scene-fetch` /
 * `robopp:scene-compile` out of a real browser sees the same two names it always did — which is
 * the point: the numbers moved threads, not meanings.
 */
function recordTiming(timing: SceneTiming): void {
  if (typeof performance === "undefined" || typeof performance.measure !== "function") return;
  if (timing.fetchMs <= 0 && timing.compileMs <= 0) return; // served from the model cache
  const end = performance.now();
  const compileStart = end - timing.compileMs;
  try {
    performance.measure(PERF_FETCH, { start: Math.max(0, compileStart - timing.fetchMs), end: compileStart });
    performance.measure(PERF_COMPILE, { start: compileStart, end });
  } catch {
    // A browser that refuses the measure (a clamped clock, a locked-down timeline) costs a
    // measurement, never a scene.
  }
}

/**
 * The one client every viewer on the page shares.
 *
 * Module-level, so the compare page's two viewers, and every scene opened over the life of the
 * tab, go through a single worker — and therefore a single WASM heap, a single materialised
 * filesystem and a single compiled-model cache.
 */
let shared: SceneClient | undefined;
export function sceneClient(): SceneClient {
  if (!shared) shared = createSceneClient();
  return shared;
}

/**
 * A real module worker, or null when this browser cannot run one.
 *
 * The feature test is not `typeof Worker`: a browser without module-worker support accepts the
 * constructor and then fails to load the script asynchronously, so what is tested is whether the
 * `type` option is read at all. A worker that dies later is handled too — `onerror` puts the
 * client on the in-process engine rather than leaving the viewer with a dead scene.
 */
function spawnSceneWorker(onEvent: (e: SceneEvent) => void, onFail: (why: string) => void): SceneChannel | null {
  if (typeof Worker !== "function" || typeof URL === "undefined") return null;
  if (!moduleWorkersSupported()) return null;
  try {
    const worker = new Worker(new URL("./mujoco.worker.ts", import.meta.url), { type: "module" });
    worker.onmessage = (e: MessageEvent<SceneEvent>) => onEvent(e.data);
    worker.onerror = (e) => onFail(e.message || "worker error");
    return {
      post: (msg, transfer) => worker.postMessage(msg, transfer ?? []),
      terminate: () => {
        // Detached, not just terminated: an event already queued on this thread is still
        // delivered after `terminate()`, and a `loaded` from the dead worker would then be
        // applied to a generation that has since been re-posted to the new channel.
        worker.onmessage = null;
        worker.onerror = null;
        worker.terminate();
      },
    };
  } catch {
    return null;
  }
}

/**
 * Does this browser understand `{ type: "module" }`?
 *
 * The getter fires during the constructor's dictionary conversion, so it says whether the option
 * was *read* — which is the only thing a browser without module workers gets wrong (it ignores the
 * option and loads the script as a classic worker, where the imports fail asynchronously).
 *
 * A throw from the probe says nothing about module support: a CSP with `worker-src 'self'`, a
 * sandboxed frame or a policy blocking `blob:` all fail this probe while a same-origin module
 * worker would have worked fine. So a throw is inconclusive and the real spawn gets its chance —
 * it has its own `try` and its own `onerror`, and falling back from there costs one load. What is
 * *not* acceptable is what this used to do: reset an already-successful probe to false in the
 * catch, and pin the tab to the main-thread engine for its whole life.
 */
let moduleWorkerSupport: boolean | undefined;
function moduleWorkersSupported(): boolean {
  if (moduleWorkerSupport !== undefined) return moduleWorkerSupport;
  let read = false;
  let threw = false;
  try {
    const url = URL.createObjectURL(new Blob([""], { type: "text/javascript" }));
    try {
      new Worker(url, { get type() { read = true; return "module"; } } as WorkerOptions).terminate();
    } finally {
      URL.revokeObjectURL(url);
    }
  } catch {
    threw = true;
  }
  moduleWorkerSupport = read || threw;
  return moduleWorkerSupport;
}

/**
 * The fallback: the same engine, on this thread.
 *
 * Everything still works — this is how the viewer behaved before the worker existed — except
 * that a compile blocks the page while it runs. The engine module is imported dynamically so a
 * browser that *does* have workers never loads it, and requests made before the import lands are
 * queued rather than dropped.
 *
 * Events are delivered in a microtask, never synchronously inside `post`: the client would
 * otherwise re-enter its own state (a pose reply arriving inside the call that sent it) in a way
 * it never can with a real worker.
 */
function inProcessChannel(onEvent: (e: SceneEvent) => void): SceneChannel {
  const queue: SceneRequest[] = [];
  let handle: ((msg: SceneRequest) => void) | null = null;
  let live = true;
  void import("./sceneEngine").then(({ createSceneEngine }) => {
    if (!live) return;
    const engine = createSceneEngine((msg) => { if (live) queueMicrotask(() => onEvent(msg)); });
    handle = (msg) => engine.handle(msg);
    for (const msg of queue.splice(0)) handle(msg);
  });
  return {
    post: (msg) => { if (handle) handle(msg); else queue.push(msg); },
    terminate: () => { live = false; handle = null; queue.length = 0; },
  };
}
