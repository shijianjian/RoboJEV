import { useEffect, useRef, useState } from "react";
import type * as THREE from "three";
import { SceneCancelled } from "./mujoco";
import { sceneClient, type SceneEngineKind, type SceneSession } from "./sceneClient";
import type { BodyPose, SceneGeometry } from "./sceneProtocol";
import { SceneProgressLine, useSceneStatus } from "./SceneStatus";
import { applyBodyPose, attachTexturesOverFrames, buildSceneRoot, frameOnCamera, createThreeStage, disposeSceneRoot, type ThreeStage } from "./threeStage";
import "./scene-loading.css";

/** The pose to show: the caller's `qpos` when it fits this model, the model's own default
 *  otherwise. A mismatched length is a data problem (an `init_qpos.json` recorded against a
 *  different bundle), not a render error, so it warns and falls back rather than throwing the
 *  viewer into its error state. */
function poseFor(geometry: SceneGeometry, qpos: ArrayLike<number> | undefined): Float32Array {
  if (qpos === undefined) return geometry.qpos0;
  if (qpos.length !== geometry.nq) {
    console.warn(`SceneOnly: qpos length ${qpos.length} != model.nq ${geometry.nq}; showing the default pose`);
    return geometry.qpos0;
  }
  return Float32Array.from(qpos);
}

interface Loaded {
  session: SceneSession;
  bodies: THREE.Group[];
  root: THREE.Group;
  /** Stops the per-frame texture upload, for a scene swapped out before it finished arriving. */
  stopTextures: () => void;
}

/**
 * `heavy` defaults to **true**, unlike `SceneView`'s.
 *
 * This component is handed a bundle hash and nothing else - `/scenes/[hash]` knows no suite, so
 * there is no `suiteIsHeavy` to read. Defaulting to false made its cache entry light
 * (`cacheCap()` = 2), so opening one kitchen and navigating to another compiled the second beside
 * the first: ~2.9 GB against the 2 GiB build ceiling, which is the "Could not allocate memory"
 * the heavy-scene rules exist to prevent. This view shows exactly one scene at a time, in every
 * caller, so capping its cache at one costs a recompile when the reader goes back to a scene they
 * already opened and nothing else - and the cap is correct for the one case that cannot survive
 * being wrong.
 */
export function SceneOnly({ xml, assetsBase, qpos, heavy = true }: { xml: string; assetsBase: string; qpos?: ArrayLike<number>; heavy?: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [status, statusControls] = useSceneStatus(heavy);
  // Which bundle is on the canvas *now*, which is not the same question as which one the page is
  // about: the previous scene stays up while the next compiles. Only set when a scene is swapped
  // in, so `data-scene` never names a bundle you cannot see.
  const [scene, setScene] = useState<string | null>(null);
  // Which engine compiled it. Read where the fallback can already have engaged (the channel is
  // made by the first load) so a page that quietly went back to compiling on the main thread says
  // so, in the DOM and in one console line - see sceneClient.ts.
  const [engine, setEngine] = useState<SceneEngineKind>("none");
  // The WASM heap after the last load, in MB. It only ever grows and the build dies at 2 GiB, so
  // this is the one number that says a tab is heading for "Could not allocate memory".
  const [heapMb, setHeapMb] = useState<number | null>(null);
  // What the load effect needs to re-pose the scene later, kept out of the effect's own scope so
  // the last effect below can apply a new `qpos` without reloading the bundle.
  const stateRef = useRef<Loaded | null>(null);
  const stageRef = useRef<ThreeStage | null>(null);
  // Which load is the current one. A click made while another task is still loading must not end
  // up showing the scene that load was going to produce, nor let its progress reports overwrite
  // the new one's. Everything a load does is gated on still being the latest generation - and the
  // worker is told to drop it, so a load that has not reached its compile never runs one.
  const generation = useRef(0);
  // The load effect must not restart when only `qpos` changes, so it reads the current value
  // through a ref instead of taking it as a dependency.
  const qposRef = useRef(qpos);
  qposRef.current = qpos;

  // The canvas, its renderer and the render loop outlive any one bundle: that is what keeps the
  // previous scene up (and orbitable) while the next one is being compiled in the worker.
  useEffect(() => {
    const stage = createThreeStage(canvasRef.current!);
    stageRef.current = stage;
    return () => {
      stageRef.current = null;
      stage.dispose();
      const loaded = stateRef.current;
      if (loaded) {
        stateRef.current = null;
        loaded.stopTextures();
        disposeSceneRoot(loaded.root);
        loaded.session.release();
      }
    };
  }, []);

  useEffect(() => {
    const mine = ++generation.current;
    const current = () => generation.current === mine;
    statusControls.startLoad();
    const load = sceneClient().load(xml, assetsBase, (p) => { if (current()) statusControls.progress(p); }, heavy);
    setEngine(sceneClient().engine());
    (async () => {
      const session = await load.promise;
      // Nothing below may run for a bundle this component has moved on from - but the session is
      // real either way and has to be given back, or its model is pinned in the worker's cache.
      if (!current() || !stageRef.current) { session.release(); return; }
      const { root, bodies, textures } = buildSceneRoot(session.geometry);
      // Posed before it is swapped in, and nothing above is touched until it succeeds: a failure
      // here leaves the scene already on the canvas exactly as it was, rather than stranding it
      // un-disposed behind a half-built replacement.
      let pose: BodyPose | null;
      try {
        pose = await session.pose(poseFor(session.geometry, qposRef.current));
      } catch (e) {
        disposeSceneRoot(root);
        session.release();
        throw e;
      }
      if (!current() || !stageRef.current) { disposeSceneRoot(root); session.release(); return; }
      if (pose) applyBodyPose(bodies, pose);
      const previous = stateRef.current;
      // Uploading this bundle's textures is the last main-thread cost of opening a scene, and the
      // only one three.js can do nowhere but here; spread over frames it is never a block.
      const stopTextures = attachTexturesOverFrames(textures);
      stateRef.current = { session, bodies, root, stopTextures };
      // Swapped in one go: the old scene comes off the canvas only now that the new one is built
      // and posed, so there is never a frame with an empty stage.
      stageRef.current.scene.add(root);
      // The first scene on this canvas opens on the agentview's view of the table; a later one
      // keeps whatever view the reader has orbited to.
      if (!previous && pose) frameOnCamera(stageRef.current, session.geometry, pose, root);
      if (previous) {
        previous.stopTextures();
        stageRef.current.scene.remove(previous.root);
        disposeSceneRoot(previous.root);
        previous.session.release();
      }
      setScene(session.geometry.scene);
      setEngine(sceneClient().engine());
      setHeapMb(Math.round(sceneClient().heapBytes() / 1e6) || null);
      statusControls.ready();
    })().catch((e) => {
      setHeapMb(Math.round(sceneClient().heapBytes() / 1e6) || null);
      if (current() && !(e instanceof SceneCancelled)) statusControls.failed(e);
    });
    // Bumping the counter retires this load for this component; cancelling it tells the worker,
    // which abandons it outright unless its compile has already begun.
    return () => { generation.current++; load.cancel(); };
  }, [xml, assetsBase, heavy, statusControls]);

  useEffect(() => {
    const s = stateRef.current;
    if (!s) return; // still loading; the load effect applies the current qpos itself
    let live = true;
    s.session.pose(poseFor(s.session.geometry, qpos))
      // A null pose means the request was superseded by a newer one, or the scene is gone: either
      // way what is on the canvas is what should stay there.
      .then((pose) => { if (live && pose && stateRef.current === s) applyBodyPose(s.bodies, pose); })
      .catch((e) => { if (live) statusControls.failed(e); });
    return () => { live = false; };
  }, [qpos, statusControls]);

  return (
    <div className="scene-wrap">
      <div className="scene-stage" data-scene={scene ?? undefined} data-engine={engine === "none" ? undefined : engine} data-heap-mb={heapMb ?? undefined}>
        <canvas ref={canvasRef} className="scene" />
        <SceneProgressLine state={status} />
      </div>
      <div className="scene-bar" style={{ borderRadius: "var(--r-3)", marginTop: "var(--s-2)" }}>
        <span className={`chip chip--${status.tone} chip--mono`}>
          <span className="chip__dot" aria-hidden />
          <span data-testid="scene-status">{status.text}</span>
        </span>
      </div>
    </div>
  );
}
