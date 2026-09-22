/**
 * The MuJoCo worker: a module worker that owns the WASM module, its filesystem and the compiled
 * models, so that compiling a LIBERO scene — seconds of uninterruptible WASM — happens on a
 * thread nobody is looking at.
 *
 * Everything it does lives in `sceneEngine.ts`; this file is only the wiring, so the same engine
 * can run in-process when a browser has no module workers (`sceneClient.ts` falls back to that)
 * and can be unit-tested without a worker at all.
 *
 * It is loaded by `new Worker(new URL("./mujoco.worker.ts", import.meta.url), { type: "module" })`
 * — the form both Turbopack and webpack recognise and compile into their own chunk.
 */
import { createSceneEngine } from "./sceneEngine";
import type { SceneRequest } from "./sceneProtocol";

// `self` in a worker is a DedicatedWorkerGlobalScope, which the DOM lib this project compiles
// against does not describe: postMessage there takes a transfer list where window's takes a
// target origin. The cast is to the two members this file uses, rather than to `any`.
const scope = self as unknown as {
  postMessage(msg: unknown, transfer?: Transferable[]): void;
  onmessage: ((e: MessageEvent<SceneRequest>) => void) | null;
};

const engine = createSceneEngine((msg, transfer) => scope.postMessage(msg, transfer ?? []));

scope.onmessage = (e) => engine.handle(e.data);
