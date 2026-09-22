/**
 * The messages the viewer and its MuJoCo worker exchange, and the shapes they carry.
 *
 * MuJoCo compiles a LIBERO bundle for seconds with no chance to yield, so it runs in a Web
 * Worker (`mujoco.worker.ts`) and the main thread never touches the WASM module at all. That
 * makes this file the whole seam between them: everything three.js needs to build a stage, and
 * everything a frame needs to be posed, crosses it as plain buffers.
 *
 * Two rules shape it:
 *
 * - **Nothing here is a MuJoCo object.** No embind handles, no enum values, no views onto the
 *   WASM heap — geom types arrive as their own names, meshes as `Float32Array`s of their own.
 *   The main thread can therefore build a scene without a MuJoCo module in the page at all.
 * - **Every message carries its generation.** A viewer that has moved on (another task clicked,
 *   another run opened) is not interested in the load it asked for a moment ago, and the reply to
 *   that load may arrive at any time — including after the reply to the load that replaced it,
 *   because a compile already under way cannot be interrupted. The generation is what lets the
 *   client drop it. See `sceneClient.ts` for the cancel semantics and `sceneEngine.ts` for what
 *   the worker does with a cancelled generation.
 */

/** The geom shapes the viewer draws, named rather than numbered: `mjtGeom` values live only in
 *  the worker, and anything else (an sdf, a height field) is dropped at extraction time. */
export type GeomKind = "plane" | "sphere" | "capsule" | "cylinder" | "box" | "ellipsoid" | "mesh";

/** One drawable geom: which body carries it, where it sits on that body, and how it looks.
 *  Sizes, positions and quaternions are MuJoCo's own (z-up, quaternion in w-x-y-z order); the
 *  z-up-to-y-up rotation is the stage's business, exactly as it was when this was read straight
 *  off the model. */
export interface GeomSpec {
  kind: GeomKind;
  body: number;
  size: [number, number, number];
  pos: [number, number, number];
  /** MuJoCo order: w, x, y, z. */
  quat: [number, number, number, number];
  rgba: [number, number, number, number];
  /** Index into {@link SceneGeometry.meshes}, or -1 for a primitive. */
  mesh: number;
  /** Index into {@link SceneGeometry.textures}, or -1. */
  texture: number;
  texRepeat: [number, number];
}

/** A mesh as three.js wants it: positions and normals per vertex, `uvs` when the mesh is
 *  textured, `indices` when its corners can be shared. A textured mesh is non-indexed, because
 *  MuJoCo indexes positions and texture coordinates independently per corner. Normals are
 *  computed in the worker so the main thread does no per-vertex arithmetic at all. */
export interface MeshData {
  positions: Float32Array;
  normals: Float32Array;
  uvs: Float32Array | null;
  indices: MeshIndices;
}

/** Mesh indices, as narrow as the mesh allows: 16-bit is half the GPU memory and covers every
 *  LIBERO mesh, and a mesh past 65 535 vertices needs the wide type. */
export type MeshIndices = Uint16Array | Uint32Array | null;

/** An RGBA image, already expanded from whatever channel count MuJoCo stored it with. */
export interface TextureData { width: number; height: number; rgba: Uint8Array }

/** Everything three.js needs to build one scene, once. */
export interface SceneGeometry {
  /** The bundle's identity — its `scenes/<hash>` directory. Shown as `data-scene` on the stage
   *  once the scene is actually on the canvas, so a test can tell which bundle is being looked
   *  at (and see that a superseded one never appeared). */
  scene: string;
  nbody: number;
  bodyNames: string[];
  geoms: GeomSpec[];
  meshes: MeshData[];
  textures: TextureData[];
  /** Named cameras, in model order: `cameraPoseFrom` matches by name and reads `fovy` here. */
  cameras: { name: string; fovy: number }[];
  /** The model's `nq`, to check a trajectory's `qpos` against, and its default pose. */
  nq: number;
  qpos0: Float32Array;
}

/** Where the bodies and cameras ended up for one `qpos`. Body poses are quaternions (what
 *  three.js takes); camera orientations stay 3x3 matrices, which is what MuJoCo has and what
 *  the one caller (the agentview button) converts. */
export interface BodyPose {
  xpos: Float32Array;
  xquat: Float32Array;
  camPos: Float32Array;
  camMat: Float32Array;
}

/**
 * How the load's wall time split, so the main thread can record it.
 *
 * The worker makes its own `performance.measure` entries, but they live in the worker's timeline
 * — a different time origin, and invisible to anything reading `performance` in the page
 * (`scripts/scene-bench.ts`, the browser's own profiler). So the durations come back with the
 * geometry and are re-measured on the main thread, where the numbers were always read.
 */
export interface SceneTiming {
  /** Fetching the bundle and writing it into the WASM filesystem. Zero for a bundle already
   *  there, and for a model served straight from the compiled-model cache. */
  fetchMs: number;
  /** `mj_loadXML`. Zero when the model came from the cache. */
  compileMs: number;
}

/** Main thread -> worker. */
export type SceneRequest =
  /** `heavy` is the suite's `scene.heavy` (registry, via `suiteIsHeavy`), and travels with the
   *  load because only the page knows it and only the worker can act on it: a heavy scene caps
   *  the compiled-model cache at one for the duration of that load. */
  | { op: "load"; gen: number; xmlUrl: string; assetsBase: string; heavy: boolean }
  | { op: "pose"; gen: number; seq: number; qpos: Float32Array }
  /** Retire a generation: its result is dropped, and a load that has not reached the compile
   *  yet is abandoned before it starts. */
  | { op: "cancel"; gen: number }
  /** The viewer is done with a loaded scene: free its `MjData` and let the model be evicted. */
  | { op: "release"; gen: number };

/**
 * Worker -> main thread.
 *
 * Every terminal reply carries `heapBytes`, the WASM module's `HEAPU8.byteLength`. That heap only
 * ever grows — WebAssembly memory cannot shrink — and this build's ceiling is 2 GiB, so a tab that
 * has opened many scenes can fail an allocation that the same scene passes in a fresh tab. The
 * viewer puts the number on the stage as `data-heap-mb`, which is the only way to see it coming.
 */
export type SceneEvent =
  | { op: "progress"; gen: number; phase: "fetching"; fetched: number; total: number }
  | { op: "progress"; gen: number; phase: "compiling" }
  | { op: "loaded"; gen: number; geometry: SceneGeometry; timing: SceneTiming; heapBytes: number }
  | { op: "failed"; gen: number; message: string; heapBytes: number; outOfMemory: boolean }
  | { op: "pose"; gen: number; seq: number; pose: BodyPose }
  | { op: "poseFailed"; gen: number; seq: number; message: string; outOfMemory: boolean };

/** The half of a worker the client uses, so a fake one (a test) or none at all (the
 *  no-Worker fallback, which runs the engine in-process) can stand in for it. */
export interface SceneChannel {
  post(msg: SceneRequest, transfer?: Transferable[]): void;
  terminate(): void;
}

/** Every `ArrayBuffer` in a geometry, for `postMessage`'s transfer list: the meshes and textures
 *  of a LIBERO bundle are tens of megabytes, and structured-cloning them would copy the lot. */
export function geometryTransfer(g: SceneGeometry): ArrayBuffer[] {
  const out: ArrayBuffer[] = [g.qpos0.buffer as ArrayBuffer];
  for (const m of g.meshes) {
    out.push(m.positions.buffer as ArrayBuffer, m.normals.buffer as ArrayBuffer);
    if (m.uvs) out.push(m.uvs.buffer as ArrayBuffer);
    if (m.indices) out.push(m.indices.buffer as ArrayBuffer);
  }
  for (const t of g.textures) out.push(t.rgba.buffer as ArrayBuffer);
  return out;
}

/** The same, for a pose reply. */
export function poseTransfer(p: BodyPose): ArrayBuffer[] {
  return [p.xpos.buffer as ArrayBuffer, p.xquat.buffer as ArrayBuffer, p.camPos.buffer as ArrayBuffer, p.camMat.buffer as ArrayBuffer];
}
