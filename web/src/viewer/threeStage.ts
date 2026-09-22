import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { measured, PERF_BUILD } from "./mujoco";
import type { BodyPose, MeshData, SceneGeometry } from "./sceneProtocol";

/**
 * The parts of a scene view that outlive the scene being viewed.
 *
 * Renderer, camera, orbit controls and the render loop belong to the canvas, not to the bundle
 * on it: keeping them alive across a bundle change is what lets the previous scene stay on
 * screen (and stay orbitable) for the seconds the next one takes to compile. Only the model's
 * body groups are swapped — see {@link disposeSceneRoot}.
 *
 * Both viewers build the same stage; the options are the two differences between a start-state
 * preview and a replay.
 */
export interface ThreeStage {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  /** Draw one frame now, outside the loop (a re-pose that must land before the next rAF). */
  render: () => void;
  /** Put the free camera at `position`, orbiting `target`. */
  frame: (position: THREE.Vector3, target: THREE.Vector3) => void;
  dispose: () => void;
}

export interface ThreeStageOptions {
  /** A shadow-casting key light, and the shadow map to receive it. */
  sunlight?: boolean;
  /** Inertia on the orbit controls: nice while scrubbing, noise in a still preview. */
  damping?: boolean;
  /** Cap on `devicePixelRatio`; left alone (renderer default) when absent. */
  maxPixelRatio?: number;
}

export function createThreeStage(canvas: HTMLCanvasElement, opts: ThreeStageOptions = {}): ThreeStage {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  if (opts.maxPixelRatio) renderer.setPixelRatio(Math.min(window.devicePixelRatio, opts.maxPixelRatio));
  renderer.shadowMap.enabled = Boolean(opts.sunlight);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x060a12);
  scene.add(new THREE.HemisphereLight(0xdfe7f5, 0x3a3a3a, 1.6));
  if (opts.sunlight) {
    const sun = new THREE.DirectionalLight(0xffffff, 2.2);
    sun.position.set(2, 4, 1.5);
    sun.castShadow = true;
    scene.add(sun);
  }

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 50);
  camera.position.set(1.6, 1.2, 1.6);
  const orbit = new OrbitControls(camera, canvas);
  orbit.target.set(0, 0.8, 0);
  orbit.enableDamping = Boolean(opts.damping);

  const resize = () => {
    const w = canvas.clientWidth || 800, h = canvas.clientHeight || 500;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  window.addEventListener("resize", resize);
  resize();

  const render = () => { orbit.update(); renderer.render(scene, camera); };
  let raf = 0;
  let running = true;
  const loop = () => {
    render();
    if (running) raf = requestAnimationFrame(loop);
  };
  loop();

  return {
    scene,
    camera,
    render,
    frame: (position, target) => {
      camera.position.copy(position);
      orbit.target.copy(target);
      orbit.update();
    },
    dispose: () => {
      running = false;
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
      orbit.dispose();
      renderer.dispose();
    },
  };
}

/**
 * Build one scene's body groups from the buffers the worker sent.
 *
 * This is the other half of `extractSceneGeometry` (viewer/mujoco.ts): everything that needed to
 * read a compiled MuJoCo model happened in the worker, and what arrives here is plain arrays. So
 * this function does no per-vertex arithmetic at all — it wraps buffers in `BufferAttribute`s and
 * builds materials — which is what keeps a scene appearing without a long task on the thread
 * that has to stay responsive.
 *
 * The returned `bodies` are indexed by MuJoCo body id, so {@link applyBodyPose} can walk them
 * against a pose reply directly; body 0 is the world and never moves.
 */
export function buildSceneRoot(geometry: SceneGeometry): SceneRoot {
  return measured(PERF_BUILD, () => buildRoot(geometry));
}

/**
 * A built scene, plus the textures it is still waiting for.
 *
 * The scene is usable the moment this returns — every material has its colour — and each entry in
 * `textures` is one GPU upload that {@link attachTexturesOverFrames} hands to the renderer a frame
 * at a time. See that function for why the uploads are not simply done here.
 */
export interface SceneRoot {
  root: THREE.Group;
  bodies: THREE.Group[];
  textures: PendingTexture[];
}

/** One MuJoCo texture, and the materials waiting to be given it. `bytes` is what its upload will
 *  cost, which is how the per-frame budget is spent. */
export interface PendingTexture {
  bytes: number;
  attach: () => void;
}

function buildRoot(geometry: SceneGeometry): SceneRoot {
  const root = new THREE.Group();
  root.rotation.x = -Math.PI / 2; // z-up to y-up
  const bodies: THREE.Group[] = [];
  for (let b = 0; b < geometry.nbody; b++) {
    const group = new THREE.Group();
    group.name = geometry.bodyNames[b] ?? `body${b}`;
    bodies.push(group);
    root.add(group);
  }
  // One BufferGeometry per mesh, however many geoms draw it: a LIBERO bundle re-uses the robot's
  // links across several geoms, and a second copy of those vertices would be paid for on the GPU.
  const meshes = geometry.meshes.map(bufferGeometry);
  // And one Texture per *texture*, not per geom. three allocates a GL texture per `Texture.source`
  // (and the cache key does not include `repeat`), so a fresh `DataTexture` per geom means the
  // same pixels uploaded and held several times over - on the fixture bundle, 28 uploads for 16
  // textures, ~38 MB of GPU memory for nothing. A `clone()` shares the source by reference, which
  // is what lets every geom keep its own `repeat` without paying for its own copy.
  const bases = geometry.textures.map((tex) => {
    const base = new THREE.DataTexture(tex.rgba, tex.width, tex.height, THREE.RGBAFormat, THREE.UnsignedByteType);
    base.wrapS = base.wrapT = THREE.RepeatWrapping;
    base.colorSpace = THREE.SRGBColorSpace;
    base.needsUpdate = true;
    return base;
  });
  const waiting: { material: THREE.MeshStandardMaterial; map: THREE.Texture }[][] = geometry.textures.map(() => []);
  for (const spec of geometry.geoms) {
    const [sx, sy, sz] = spec.size;
    let buffer: THREE.BufferGeometry;
    switch (spec.kind) {
      case "plane": buffer = new THREE.PlaneGeometry(sx * 2 || 40, sy * 2 || 40); break;
      case "sphere": buffer = new THREE.SphereGeometry(sx, 32, 24); break;
      case "capsule": buffer = new THREE.CapsuleGeometry(sx, sy * 2, 8, 20).rotateX(Math.PI / 2); break;
      case "cylinder": buffer = new THREE.CylinderGeometry(sx, sx, sy * 2, 32).rotateX(Math.PI / 2); break;
      case "box": buffer = new THREE.BoxGeometry(sx * 2, sy * 2, sz * 2); break;
      case "ellipsoid": buffer = new THREE.SphereGeometry(1, 24, 16).scale(sx, sy, sz); break;
      case "mesh": {
        const shared = meshes[spec.mesh];
        if (!shared) continue;
        buffer = shared;
        break;
      }
    }
    const [r, g, b, a] = spec.rgba;
    // Built without its map: the material's own colour is on screen immediately, and the texture
    // is attached later, one upload per frame. See attachTexturesOverFrames.
    const material = new THREE.MeshStandardMaterial({
      color: new THREE.Color(r, g, b), roughness: 0.85, metalness: 0.05, transparent: a < 1, opacity: a,
    });
    if (spec.texture >= 0 && bases[spec.texture]) {
      const map = bases[spec.texture].clone(); // shares `source`, and therefore the GL texture
      map.repeat.set(spec.texRepeat[0], spec.texRepeat[1]);
      waiting[spec.texture].push({ material, map });
    }
    const mesh = new THREE.Mesh(buffer, material);
    mesh.castShadow = spec.kind !== "plane";
    mesh.receiveShadow = true;
    mesh.position.set(spec.pos[0], spec.pos[1], spec.pos[2]);
    mesh.quaternion.set(spec.quat[1], spec.quat[2], spec.quat[3], spec.quat[0]); // MuJoCo stores w first
    (bodies[spec.body] ?? root).add(mesh);
  }
  const textures: PendingTexture[] = geometry.textures.map((tex, i) => ({
    bytes: tex.rgba.byteLength,
    attach: () => {
      for (const { material, map } of waiting[i]) {
        material.map = map;
        material.needsUpdate = true; // the map changes the shader program, not just a uniform
      }
    },
  })).filter((_, i) => waiting[i].length > 0);
  return { root, bodies, textures };
}

/**
 * Hand the scene's textures to the renderer a frame at a time.
 *
 * Uploading a texture is `texImage2D` on the main thread, and there is no worker in the world that
 * can do it for you. On the fixture bundle that is 742 ms in one task if every material is handed
 * its map at once — the single biggest block left after the compile moved off this thread, and on
 * a larger scene it was measured at ~3 s. Attaching one texture (or one budget's worth of small
 * ones) per animation frame turns that one task into a dozen short ones: the stage is on screen
 * from the first frame with its materials' flat colours, and the textures arrive over the next few
 * frames while the page stays interactive.
 *
 * Returns a cancel: a scene swapped out mid-upload must stop attaching maps to materials that are
 * on their way to `disposeSceneRoot`.
 *
 * `budgetBytes` is a *soft* cap — one texture is always attached per frame, however large it is,
 * because a single upload cannot be split — so a bundle carrying 4096² textures still has ~60 ms
 * frames. Capping textures at build time (`worker`'s `--texture-max`) is the fix for that, not
 * anything this can do.
 */
export function attachTexturesOverFrames(textures: readonly PendingTexture[], budgetBytes = 4 * 1024 * 1024): () => void {
  if (textures.length === 0) return () => {};
  if (typeof requestAnimationFrame !== "function") {
    // No frames to spread over (a test, a hidden tab that never paints): do it all now rather
    // than leave the scene untextured.
    for (const texture of textures) texture.attach();
    return () => {};
  }
  let next = 0;
  let raf = requestAnimationFrame(step);
  let live = true;
  function step(): void {
    if (!live) return;
    let spent = 0;
    do {
      const texture = textures[next++];
      texture.attach();
      spent += texture.bytes;
    } while (next < textures.length && spent + textures[next].bytes <= budgetBytes);
    if (next < textures.length) raf = requestAnimationFrame(step);
  }
  return () => { live = false; cancelAnimationFrame(raf); };
}

function bufferGeometry(mesh: MeshData): THREE.BufferGeometry {
  const buffer = new THREE.BufferGeometry();
  buffer.setAttribute("position", new THREE.BufferAttribute(mesh.positions, 3));
  buffer.setAttribute("normal", new THREE.BufferAttribute(mesh.normals, 3));
  if (mesh.uvs) buffer.setAttribute("uv", new THREE.BufferAttribute(mesh.uvs, 2));
  if (mesh.indices) buffer.setIndex(new THREE.BufferAttribute(mesh.indices, 1));
  return buffer;
}

/** Move the body groups to where the worker's forward kinematics put them. Body 0 is the world
 *  frame, which the root's own rotation already accounts for. */
export function applyBodyPose(bodies: THREE.Group[], pose: BodyPose): void {
  for (let b = 1; b < bodies.length; b++) {
    bodies[b].position.set(pose.xpos[b * 3], pose.xpos[b * 3 + 1], pose.xpos[b * 3 + 2]);
    bodies[b].quaternion.set(pose.xquat[b * 4 + 1], pose.xquat[b * 4 + 2], pose.xquat[b * 4 + 3], pose.xquat[b * 4]);
  }
}

/** Where a named MuJoCo camera is, in the scene root's own frame, for the pose just applied.
 *  MuJoCo cameras look down -z with y up, same as three's, so the rotation matrix is used as it
 *  comes. Null when the model has no camera by that name. */
export function cameraPoseFrom(
  geometry: SceneGeometry, pose: BodyPose, name: string,
): { position: THREE.Vector3; quaternion: THREE.Quaternion; fovy: number } | null {
  const c = geometry.cameras.findIndex((cam) => cam.name === name);
  if (c < 0) return null;
  const m = pose.camMat;
  const rot = new THREE.Matrix4().set(
    m[c * 9 + 0], m[c * 9 + 1], m[c * 9 + 2], 0,
    m[c * 9 + 3], m[c * 9 + 4], m[c * 9 + 5], 0,
    m[c * 9 + 6], m[c * 9 + 7], m[c * 9 + 8], 0,
    0, 0, 0, 1,
  );
  return {
    position: new THREE.Vector3(pose.camPos[c * 3], pose.camPos[c * 3 + 1], pose.camPos[c * 3 + 2]),
    quaternion: new THREE.Quaternion().setFromRotationMatrix(rot),
    fovy: geometry.cameras[c].fovy,
  };
}

/** Free the GPU-side buffers of one model's body groups. three.js does not reference-count
 *  geometries, materials or textures, so a scene swapped out without this leaks the whole
 *  bundle's meshes for the life of the tab. */
export function disposeSceneRoot(root: THREE.Object3D): void {
  root.traverse((obj) => {
    if (!(obj instanceof THREE.Mesh)) return;
    obj.geometry.dispose();
    const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
    for (const mat of materials) {
      const m = mat as THREE.MeshStandardMaterial;
      if (m.map) m.map.dispose();
      m.dispose();
    }
  });
}

/**
 * The free camera's first view of a scene: from the model's own `agentview` camera, drawn back a
 * little, orbiting the point that camera looks at on the table. robopp opens on a fixed view of the
 * whole room, which on a LIBERO tabletop leaves the task's objects a few pixels wide; this is the
 * view the policy's own camera has of them. False when the model has no such camera.
 */
export function frameOnCamera(stage: ThreeStage, geometry: SceneGeometry, pose: BodyPose, root: THREE.Object3D,
                              name = "agentview", back = 0.35, reach = 1.0): boolean {
  const cam = cameraPoseFrom(geometry, pose, name);
  if (cam === null) return false;
  root.updateMatrixWorld(true);
  const position = cam.position.clone().applyMatrix4(root.matrixWorld);
  const quaternion = root.quaternion.clone().multiply(cam.quaternion);
  const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(quaternion);
  stage.frame(position.clone().addScaledVector(forward, -back), position.clone().addScaledVector(forward, reach));
  return true;
}
