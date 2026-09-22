import { useEffect, useRef, useState } from "react";
import type * as THREE from "three";
import { IconCamera } from "../ui/Icons";
import { SceneCancelled } from "./mujoco";
import type { Trajectory } from "./record";
import { qposAt } from "./record";
import { sceneClient, type SceneEngineKind, type SceneSession } from "./sceneClient";
import type { BodyPose, SceneGeometry } from "./sceneProtocol";
import { SceneProgressLine, useSceneStatus } from "./SceneStatus";
import { applyBodyPose, attachTexturesOverFrames, buildSceneRoot, frameOnCamera, cameraPoseFrom, createThreeStage, disposeSceneRoot, type ThreeStage } from "./threeStage";
import "./scene-loading.css";

/** One frame of the trajectory, checked against the model it is about to be applied to. A length
 *  mismatch means this record was made against a different bundle - a data problem the viewer
 *  says out loud rather than posing something meaningless. */
function qposForFrame(geometry: SceneGeometry, trajectory: Trajectory, frame: number): Float32Array {
  const qpos = qposAt(trajectory, frame);
  if (qpos.length !== geometry.nq) throw new Error(`qpos length ${qpos.length} != model.nq ${geometry.nq}`);
  return qpos;
}

interface Loaded {
  session: SceneSession;
  bodies: THREE.Group[];
  root: THREE.Group;
  /** The last pose applied, which is what the agentview button reads the camera out of. */
  pose: BodyPose | null;
  /** Stops the per-frame texture upload, for a scene swapped out before it finished arriving. */
  stopTextures: () => void;
}

/** A bundle's scene: where its `scene.xml` is and where the shared asset pool is. Both absolute,
 *  because the worker fetches them and a worker resolves a relative URL against its own script. */
export interface SceneFiles { xml: string; assets_base: string }

/** robopp's `viewer/SceneView.tsx`: the recorded episode's MuJoCo scene, posed at the viewer's
 *  frame, orbitable. `files` is the bundle's scene rather than a run's storage listing; the rest is
 *  robopp's. `heavy` stays false: no LIBERO scene is. */
export function SceneView({ files, trajectory, frame, heavy = false, onReady, onFailed }: {
  files: { scene: SceneFiles | null }; trajectory: Trajectory; frame: number; heavy?: boolean;
  /** The scene is compiled and on the canvas - what the replay waits for before it plays. */
  onReady?: () => void;
  /** It could not be loaded; the replay then plays its videos without it. */
  onFailed?: (e: unknown) => void;
}) {
  const callbacks = useRef({ onReady, onFailed });
  callbacks.current = { onReady, onFailed };
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [status, statusControls] = useSceneStatus(heavy);
  // Which bundle is on the canvas now - set only when a scene is actually swapped in, so it
  // never names the one still compiling. See SceneOnly.tsx.
  const [scene, setScene] = useState<string | null>(null);
  // Which engine compiled it. Read where the fallback can already have engaged (the channel is
  // made by the first load) so a page that quietly went back to compiling on the main thread says
  // so, in the DOM and in one console line - see sceneClient.ts.
  const [engine, setEngine] = useState<SceneEngineKind>("none");
  // The WASM heap after the last load, in MB. It only ever grows and the build dies at 2 GiB, so
  // this is the one number that says a tab is heading for "Could not allocate memory".
  const [heapMb, setHeapMb] = useState<number | null>(null);
  const stateRef = useRef<Loaded | null>(null);
  const stageRef = useRef<ThreeStage | null>(null);
  // Which load is the current one: a run opened while another is still loading must not end up
  // showing the scene that load was going to produce. See SceneOnly.tsx.
  const generation = useRef(0);
  const frameRef = useRef({ trajectory, frame });
  frameRef.current = { trajectory, frame };

  // The stage outlives the bundle on it: switching runs keeps the previous scene (and the
  // camera you had orbited to) on screen while the next one is compiled in the worker.
  useEffect(() => {
    const stage = createThreeStage(canvasRef.current!, { sunlight: true, damping: true, maxPixelRatio: 2 });
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
    // A run with no scene bundle has nothing for this view to load, and is never given one:
    // `stageModeFor` (viewer/stage.ts) picks the video stage for it and this component is not
    // mounted at all. The guard stays as the type narrowing `files.scene` needs, and says which
    // decision was skipped if one ever is.
    const sceneFiles = files.scene;
    if (sceneFiles === null) {
      statusControls.failed(new Error("this run has no scene bundle to replay"));
      return;
    }
    statusControls.startLoad();
    const load = sceneClient().load(sceneFiles.xml, sceneFiles.assets_base, (p) => { if (current()) statusControls.progress(p); }, heavy);
    setEngine(sceneClient().engine());
    (async () => {
      const session = await load.promise;
      // The session is real whether or not this component still wants it, and has to be given
      // back either way, or its model is pinned in the worker's cache forever.
      if (!current() || !stageRef.current) { session.release(); return; }
      const { root, bodies, textures } = buildSceneRoot(session.geometry);
      const { trajectory: traj, frame: at } = frameRef.current;
      // Posed before it is swapped in, and nothing above is touched until it succeeds: a
      // trajectory that does not fit this model (or a worker that went away mid-load) leaves the
      // scene already on the canvas exactly as it was, rather than stranding it un-disposed
      // behind a half-built replacement.
      let pose: BodyPose | null;
      try {
        pose = await session.pose(qposForFrame(session.geometry, traj, at));
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
      stateRef.current = { session, bodies, root, pose, stopTextures };
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
      callbacks.current.onReady?.();
    })().catch((e) => {
      setHeapMb(Math.round(sceneClient().heapBytes() / 1e6) || null);
      if (current() && !(e instanceof SceneCancelled)) {
        statusControls.failed(e);
        callbacks.current.onFailed?.(e);
      }
    });
    // Bumping the counter retires this load for this component; cancelling it tells the worker,
    // which abandons it outright unless its compile has already begun.
    return () => { generation.current++; load.cancel(); };
    // The bundle URL is the identity of what this effect loads; `files` itself is a fresh object
    // on every render of the page around it.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [files.scene?.xml, files.scene?.assets_base, heavy, statusControls]);

  useEffect(() => {
    const s = stateRef.current;
    if (!s) return;
    let live = true;
    (async () => {
      const pose = await s.session.pose(qposForFrame(s.session.geometry, trajectory, frame));
      // A null pose means a newer frame was asked for before this one came back (scrubbing
      // outruns the round trip, and only the newest frame is worth drawing) or the scene is gone.
      if (!live || !pose || stateRef.current !== s) return;
      s.pose = pose;
      applyBodyPose(s.bodies, pose);
    })().catch((e) => { if (live) statusControls.failed(e); });
    return () => { live = false; };
  }, [frame, trajectory, statusControls]);

  const jumpToAgentview = () => {
    const s = stateRef.current, stage = stageRef.current;
    if (!s?.pose || !stage) return;
    const pose = cameraPoseFrom(s.session.geometry, s.pose, "agentview");
    if (!pose) return;
    const worldPos = pose.position.clone().applyMatrix4(s.root.matrixWorld);
    stage.camera.position.copy(worldPos);
    stage.camera.quaternion.copy(s.root.quaternion.clone().multiply(pose.quaternion));
    stage.camera.fov = pose.fovy;
    stage.camera.updateProjectionMatrix();
  };

  return (
    <div className="scene-wrap">
      <div className="scene-stage" data-scene={scene ?? undefined} data-engine={engine === "none" ? undefined : engine} data-heap-mb={heapMb ?? undefined}>
        <canvas ref={canvasRef} className="scene" />
        <SceneProgressLine state={status} />
      </div>
      <div className="scene-bar">
        <span className={`chip chip--${status.tone} chip--mono`}>
          <span className="chip__dot" aria-hidden />
          <span data-testid="scene-status">{status.text}</span>
        </span>
        <span className="u-row">
          <span className="u-dim">drag to orbit · scroll to zoom</span>
          <button type="button" className="btn btn--sm" onClick={jumpToAgentview} title="move the free camera to the recorded agentview pose">
            <IconCamera /> agentview camera
          </button>
        </span>
      </div>
    </div>
  );
}
