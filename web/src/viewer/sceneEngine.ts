import {
  bodyPose, cachedHeapView, doubledArena, extractSceneGeometry, heapBytes, isOutOfMemory, loadMujoco,
  loadSceneBundle, noteHeapView, purgeSceneMemory, sceneDirFromXmlUrl, SceneCancelled, VIEWER_ARENA,
  type SceneHandle, type SceneProgress,
} from "./mujoco";
import { geometryTransfer, poseTransfer, type SceneEvent, type SceneRequest } from "./sceneProtocol";

/**
 * The MuJoCo side of the viewer: requests in, events out, and not one reference to `self`,
 * `Worker` or the DOM.
 *
 * `mujoco.worker.ts` is three lines of wiring around this, which is the point — the same engine
 * runs on the main thread when the browser has no module workers (see `sceneClient.ts`), and a
 * unit test can drive it with a fake MuJoCo module and assert what it replies. Nothing here
 * decides *policy* (which generation is current, when to give up on one): that is the client's,
 * because only the client knows what is on screen.
 *
 * Cancel semantics, which are the whole reason generations exist:
 *
 * - A `cancel` for a load that has not started its compile abandons it — `loadSceneBundle` is
 *   given a predicate it checks at every point a load can still be dropped cheaply, the last
 *   being the instant before `mj_loadXML`.
 * - A `cancel` for a load already inside the compile cannot stop it (WASM does not yield). The
 *   compile finishes and says nothing; its model goes into the cache, where the *next* visit to
 *   that bundle finds it, so the seconds it cost are not wasted even though its viewer has gone.
 * - A `cancel` for a scene that is already loaded releases it, exactly as `release` would: the
 *   `loaded` event and the `cancel` may cross in flight, and a session nobody claims would
 *   otherwise pin its model in the cache forever.
 */
export interface SceneEnginePort {
  (msg: SceneEvent, transfer?: Transferable[]): void;
}

export interface SceneEngine {
  handle(msg: SceneRequest): void;
}

/** One loaded scene: the model (shared, cached) and this generation's own `MjData`, plus what it
 *  would take to load it again with a bigger arena (see the pose path below). */
interface Session {
  handle: SceneHandle;
  mujoco: any;
  xmlUrl: string;
  assetsBase: string;
  arena: string;
  /** The suite's `scene.heavy`, kept so the reload below (a bigger arena) caps the model cache
   *  the same way the original load did. */
  heavy: boolean;
  /** Set once this scene has been given a bigger arena: it is tried once, not every frame. */
  grewArena?: boolean;
}

/** A clock that exists in a worker, a browser and node alike. */
const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

/** Give the queue a turn. `postMessage` delivers as a task, and a promise continuation is a
 *  microtask, so a message sent while this thread was blocked is only seen after a macrotask
 *  boundary — this one. */
const yieldToMessages = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

export function createSceneEngine(post: SceneEnginePort, mujocoLoader: () => Promise<any> = loadMujoco): SceneEngine {
  const sessions = new Map<number, Session>();
  /** Generations whose load is running right now. */
  const inFlight = new Set<number>();
  /** Generations the client has given up on, while their load is still running. */
  const cancelled = new Set<number>();

  const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

  /**
   * The heap size, read through something the module still owns *now*.
   *
   * Growing the memory detaches every view onto the old buffer, and a load that failed for want of
   * memory has grown it several times on the way — so the view the last successful load left
   * behind reads 0 bytes at exactly the moment this number is worth reporting. A live session's
   * `data.qpos`, or any model still in the cache, is a view onto the current buffer.
   */
  const heapNow = (mujoco: any): number => {
    for (const session of sessions.values()) {
      const qpos = session.handle?.data?.qpos;
      if (ArrayBuffer.isView(qpos) && qpos.buffer.byteLength > 0) return heapBytes(mujoco, qpos);
    }
    return heapBytes(mujoco, cachedHeapView());
  };

  async function load(gen: number, xmlUrl: string, assetsBase: string, heavy: boolean): Promise<void> {
    inFlight.add(gen);
    // The phase split, for the main thread to re-record in its own timeline (see SceneTiming).
    // Taken from the progress the load reports rather than from inside it: "compiling" is posted
    // in the last moment before mj_loadXML takes the thread, which is exactly the boundary.
    const started = now();
    let compileStarted = 0;
    let mujoco: any;
    try {
      mujoco = await mujocoLoader();
      if (cancelled.has(gen)) return;
      const onProgress = (p: SceneProgress) => {
        if (p.phase === "compiling") compileStarted = now();
        if (cancelled.has(gen)) return;
        post(p.phase === "fetching"
          ? { op: "progress", gen, phase: "fetching", fetched: p.loaded, total: p.total }
          : { op: "progress", gen, phase: "compiling" });
      };
      const stale = () => cancelled.has(gen);
      let handle: SceneHandle;
      try {
        handle = await loadSceneBundle(mujoco, xmlUrl, assetsBase, onProgress, stale, undefined, heavy);
      } catch (e) {
        // The heap ran out. It is 2 GiB, it never shrinks, and what is in it is mostly scenes
        // nobody is looking at any more: give those back and try this load once more before
        // telling anyone. If it fails again the client's answer is a fresh worker, which is the
        // only way a WASM heap is ever really reclaimed.
        if (!isOutOfMemory(e) || e instanceof SceneCancelled || cancelled.has(gen)) throw e;
        // Read before the purge: it is about to delete the models this is measured through.
        const before = heapNow(mujoco);
        const freed = purgeSceneMemory(mujoco);
        console.warn(`robojev viewer: out of memory at ${Math.round(before / 1e6)} MB; dropped ${freed.models} cached model(s) and ${freed.dirs} bundle(s), retrying`);
        compileStarted = 0;
        handle = await loadSceneBundle(mujoco, xmlUrl, assetsBase, onProgress, stale, undefined, heavy);
      }
      const done = now();
      const timing = compileStarted
        // A cache hit reports neither phase, and says so with two zeroes rather than crediting
        // the compile it did not do.
        ? { fetchMs: compileStarted - started, compileMs: done - compileStarted }
        : { fetchMs: 0, compileMs: 0 };
      // Before paying for the geometry, let whatever the client sent *while* the compile held
      // this thread be delivered. A cancel posted during those seconds is only a queued task
      // until the compile returns, and the promise continuation above is a microtask, so without
      // this yield a scene the reader has already clicked away from would be copied out of the
      // WASM heap in full — a second or more of work, and a second the load they do want spends
      // waiting behind it. Measured at ~2 s per abandoned LIBERO scene.
      await yieldToMessages();
      // The handle is real whether or not anyone still wants it, and has to be given back either
      // way, or its model is pinned in the cache for the life of the worker.
      if (cancelled.has(gen)) { handle.release(); return; }
      // `names` is a view onto the module's memory: the handle heapBytes reads its size through.
      noteHeapView(handle.model.names);
      const geometry = extractSceneGeometry(mujoco, handle.model, sceneDirFromXmlUrl(xmlUrl));
      // Registered only once the client has actually been told about it. If the post throws (a
      // transfer list the browser refuses, a message over its size limit) the client rejects and
      // never sends `release`, so the handle is given back here instead — otherwise it would pin
      // its model in the cache for the life of the worker.
      try {
        post({ op: "loaded", gen, geometry, timing, heapBytes: heapBytes(mujoco, handle.data.qpos) }, geometryTransfer(geometry));
      } catch (e) {
        handle.release();
        throw e;
      }
      sessions.set(gen, { handle, mujoco, xmlUrl, assetsBase, arena: VIEWER_ARENA, heavy });
    } catch (e) {
      // A load nobody is waiting for any more ends in silence: the client retired this
      // generation when it cancelled it, and has nothing left to do with the news.
      if (!(e instanceof SceneCancelled) && !cancelled.has(gen)) {
        post({ op: "failed", gen, message: message(e), heapBytes: heapNow(mujoco), outOfMemory: isOutOfMemory(e) });
      }
    } finally {
      inFlight.delete(gen);
      cancelled.delete(gen);
    }
  }

  /**
   * Pose one frame, and deal with the one way `mj_forward` can fail that is not the caller's
   * fault: an arena too small for this frame's contacts.
   *
   * `<size memory>` is chosen at compile time from a measurement taken at each model's *default*
   * pose (viewer/mujoco.ts). A frame mid-manipulation carries more contacts, and MuJoCo reports
   * the overflow as "Could not allocate memory" — the same words as a full heap, from a completely
   * different cause, for which purging and respawning do nothing. What does work is compiling this
   * one bundle again with twice the arena, which is done once per scene. If that still fails, the
   * reader is told: a stale frame under a chip reading `ready` is the failure this viewer goes out
   * of its way to avoid.
   */
  async function pose(gen: number, seq: number, qpos: Float32Array): Promise<void> {
    const session = sessions.get(gen);
    // A pose for a scene that has been released is not an error worth reporting to the user: the
    // viewer that asked has already moved on.
    if (!session) { post({ op: "poseFailed", gen, seq, message: "scene released", outOfMemory: false }); return; }
    try {
      const computed = bodyPose(session.mujoco, session.handle.model, session.handle.data, qpos);
      post({ op: "pose", gen, seq, pose: computed }, poseTransfer(computed));
      return;
    } catch (e) {
      if (!isOutOfMemory(e) || session.grewArena) {
        post({ op: "poseFailed", gen, seq, message: message(e), outOfMemory: isOutOfMemory(e) });
        return;
      }
    }
    session.grewArena = true;
    const arena = doubledArena(session.arena);
    try {
      console.warn(`robojev viewer: this frame needs more than a ${session.arena} arena; reloading the scene with ${arena}`);
      const bigger = await loadSceneBundle(session.mujoco, session.xmlUrl, session.assetsBase, undefined, () => false, arena, session.heavy);
      // Still the same model, so the stage the viewer built is still right: only the pool the
      // frame is computed in has changed.
      if (sessions.get(gen) !== session) { bigger.release(); return; }
      session.handle.release();
      session.handle = bigger;
      session.arena = arena;
      const computed = bodyPose(session.mujoco, bigger.model, bigger.data, qpos);
      post({ op: "pose", gen, seq, pose: computed }, poseTransfer(computed));
    } catch (e) {
      post({ op: "poseFailed", gen, seq, message: message(e), outOfMemory: isOutOfMemory(e) });
    }
  }

  function release(gen: number): void {
    const session = sessions.get(gen);
    if (!session) return;
    sessions.delete(gen);
    session.handle.release();
  }

  return {
    handle(msg: SceneRequest): void {
      switch (msg.op) {
        case "load":
          void load(msg.gen, msg.xmlUrl, msg.assetsBase, msg.heavy);
          return;
        case "cancel":
          if (sessions.has(msg.gen)) release(msg.gen);
          else if (inFlight.has(msg.gen)) cancelled.add(msg.gen);
          return;
        case "release":
          release(msg.gen);
          return;
        case "pose":
          void pose(msg.gen, msg.seq, msg.qpos);
          return;
      }
    },
  };
}
