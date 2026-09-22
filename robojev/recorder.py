"""One episode as a replay bundle the web front end can serve.

A bundle is a directory `<name>/` holding

    episode.json      the schema below (`SCHEMA_VERSION`)
    agentview.mp4     H.264 / yuv420p / +faststart, one frame per control step
    wrist.mp4         the same, from the in-hand camera
    poster.jpg        one still
    qpos.bin          the simulator's joint positions, one float32 row per video frame

and the hash of the compiled scene those rows pose (`robojev.scene_bundle`, robopp's export): the
repository's `data/scenes/<hash>/` for a task it ships, `$ROBOJEV_HOME/scenes/<hash>/` otherwise.
 The videos are written at the environment's own control rate (20 Hz on LIBERO)
with **every** frame of the episode in them -- the settling steps at the start included -- so
video time and control step are the same clock:

    t_seconds = control_step / control_rate

which is what lets the page drive the decision panel off `video.currentTime` without a lookup
table of its own. `web/PROTOCOL.md` is the schema as the front end reads it.

The decision objects come straight off the frames the episode loop carries
(`Frame.decisions`), so nothing here reaches into a policy's internals.
"""
from __future__ import annotations

import json
import pathlib
import re
import shutil
import subprocess
import time

import numpy as np

#: 2 added `qpos` and `scene`. A version-1 bundle is still read: it simply has no 3D scene.
SCHEMA_VERSION = 2

#: The pose file's name and its element type: little-endian float32, frame-major.
QPOS_NAME = "qpos.bin"
QPOS_DTYPE = "<f4"

#: The lettered grasp-candidate lines of the state text:
#: `  C: -y side, turn +90, room 3.1 cm -> fits`.
RIM_LINE = re.compile(
    r"^\s{2}([A-H]):\s*(.+?),\s*turn\s*([+-]?\d+),\s*room\s*(-?[\d.]+)\s*cm\s*->\s*(.+?)\s*$")

#: The bundle's poster: the still the episode strip shows for it. Small on purpose -- it is a
#: 96 px thumbnail and there is one per bundle on the first paint.
POSTER_NAME = "poster.jpg"
POSTER_WIDTH = 224


def _clean(value):
    """JSON without numpy in it: every number a float or an int, every array a list."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), 6)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    return value


def _groups(decisions: dict) -> dict:
    """`{qid: {candidates: [{id, p}], choice, overridden}}` for every question in one decision.

    The candidate order is the question set's declared order, which is the order the bars are
    drawn in -- `hold` in the middle of an axis question is a property of the vocabulary and must
    survive the trip.
    """
    out = {}
    for qid, group in decisions.items():
        if qid == "meta" or not isinstance(group, dict) or "probabilities" not in group:
            continue
        probabilities = group["probabilities"]
        out[qid] = {
            "candidates": [{"id": cid, "p": round(float(p), 6)} for cid, p in probabilities.items()],
            "choice": group.get("choice"),
            "overridden": bool(group.get("overridden", False)),
        }
    return out


def rim_rows(state: str) -> list[dict]:
    """The grasp-option block of the state text, parsed back out of it.

    The page highlights these lines when the reader hovers `rim`; having them as data as well
    means the panel can name the chosen one without re-parsing the paragraph in the browser.
    """
    rows = []
    for line in state.splitlines():
        m = RIM_LINE.match(line)
        if m:
            letter, side, turn, room, verdict = m.groups()
            text = verdict.strip()
            rows.append({
                "letter": letter, "side": side.strip(), "turn_deg": int(turn),
                "room_cm": float(room), "verdict": text,
                # The verdict is one of `fits`, `blocked by <fixture>`, and either of those
                # prefixed `tried, ` once the plan has stood there and come away with nothing. So
                # "does it fit" is the absence of `blocked`, not the presence of `fits`.
                "fits": "blocked" not in text.lower(),
                "tried": text.lower().startswith("tried"),
            })
    return rows


def _decision(frame, index: int, fps: float) -> dict:
    meta = dict(frame.decisions.get("meta") or {})
    state = meta.get("state") or ""
    rim = rim_rows(state)
    choice = (frame.decisions.get("rim") or {}).get("choice")
    chosen = next((r for r in rim if r["letter"] == choice), None)
    latch = dict(meta.get("grip_latch") or {})
    grounding = meta.get("grounding")
    return _clean({
        "index": index,
        "step": meta.get("step"),
        "control_step": int(frame.frame_index),
        "t": round(int(frame.frame_index) / float(fps), 4),
        "state": state,
        "questions": _groups(frame.decisions),
        "grip_latch": {
            "asked": latch.get("asked"),
            "closed": latch.get("closed"),
            "refused": bool(latch.get("refused", False)),
        },
        "grounding": None if not grounding else {
            "mode": grounding.get("mode"),
            "source": grounding.get("source"),
            "sources": grounding.get("sources"),
            "target": grounding.get("target"),
            "destination": grounding.get("destination"),
            "model": grounding.get("model"),
            "model_agrees": grounding.get("model_agrees"),
            "rule": grounding.get("rule"),
            "probability": grounding.get("probability"),
            "min_probability": grounding.get("min_probability"),
            "regrounded": grounding.get("regrounded"),
        },
        "subgoal": meta.get("subgoal"),
        "substage": meta.get("substage"),
        "target": meta.get("target"),
        "destination": meta.get("destination"),
        "waypoint": meta.get("waypoint"),
        "waypoint_cm": meta.get("waypoint_cm"),
        "rim_candidates": rim,
        "rim_chosen": chosen,
        "action": [float(v) for v in np.asarray(frame.action).reshape(-1)],
        "forward_passes": meta.get("forward_passes"),
        "candidate_paths": meta.get("candidate_paths"),
        "step_sizes_cm": meta.get("step_sizes_cm"),
    })


def decision_entry(frame, index: int, fps: float) -> dict:
    """One decision, exactly as `episode.json` holds it.

    Public because a **live** console sends this same object down its socket, decision by decision
    (`robojev.console`, `web/PROTOCOL.md`): the page has one reader for a recorded decision and a
    streamed one, and it only does if the two are the same bytes. Calling this rather than
    re-deriving it is what makes that true by construction instead of by agreement.
    """
    return _decision(frame, index, fps)


# --------------------------------------------------------------------------------------- video

def ffmpeg() -> str:
    """Whatever `ffmpeg` this box has, checked before an episode is run rather than after it."""
    found = shutil.which("ffmpeg")
    if not found:
        raise SystemExit("robojev record: no ffmpeg on this box; the bundle's videos cannot be "
                         "encoded. Install ffmpeg and try again.")
    return found


def write_video(path: pathlib.Path, frames: list[np.ndarray], fps: float, crf: int = 26) -> None:
    """One H.264 / yuv420p / `+faststart` mp4, fed raw RGB over a pipe.

    H.264 and not something newer because the point of a bundle is a file a browser can open with
    no server and no transcode step, and that includes a visitor's Safari.
    """
    h, w = frames[0].shape[:2]
    cmd = [
        ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.0",
        # Every control step is a decision boundary someone may seek to, so keyframes are cheap
        # insurance against a seek landing on the wrong side of a GOP.
        "-g", "20", "-movflags", "+faststart",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for img in frames:
            proc.stdin.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes())
    finally:
        proc.stdin.close()
        if proc.wait() != 0:
            raise SystemExit(f"robojev record: ffmpeg failed writing {path}")


def write_poster(bundle: pathlib.Path, seconds: float, video: str = "agentview.mp4") -> str | None:
    """One frame of `video` at `seconds`, as `poster.jpg` inside `bundle`. Returns its name.

    Taken from the bundle's own video rather than from the episode's frames, so a bundle recorded
    before posters existed gets exactly the same picture from `--reparse` as it would have got at
    record time: there is one source for it, and it is the file that shipped.
    """
    src = bundle / video
    if not src.exists():
        return None
    out = bundle / POSTER_NAME
    cmd = [
        ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{max(seconds, 0):.3f}", "-i", str(src), "-frames:v", "1",
        "-vf", f"scale={POSTER_WIDTH}:-2:flags=lanczos", "-q:v", "6", str(out),
    ]
    if subprocess.run(cmd).returncode != 0 or not out.exists():
        raise SystemExit(f"robojev record: ffmpeg failed writing {out}")
    return POSTER_NAME


def poster_time(bundle: dict) -> float:
    """A bit past the middle of the episode, which on these tasks is the hand already on or over
    the object rather than the untouched table every episode starts with."""
    decisions = bundle.get("decisions") or []
    if not decisions:
        return 0.0
    return float(decisions[int(0.55 * (len(decisions) - 1))]["t"])


def _frames(episode, camera: str) -> list[np.ndarray]:
    out = []
    for f in episode.frames:
        img = f.images.get(camera)
        if img is None:
            return []
        out.append(np.asarray(img, dtype=np.uint8))
    return out


# -------------------------------------------------------------------------------------- bundles

def reparse(out: pathlib.Path) -> int:
    """Re-derive every parsed-from-the-state-text field of an existing bundle, in place.

    The state text is the record; everything parsed out of it is a view, so a fix to the parser
    must not cost a simulator run -- and cannot, since re-running would be a different episode.
    """
    path = out / "episode.json"
    bundle = json.loads(path.read_text(encoding="utf-8"))
    for d in bundle["decisions"]:
        rim = rim_rows(d.get("state") or "")
        choice = (d.get("questions", {}).get("rim") or {}).get("choice")
        d["rim_candidates"] = rim
        d["rim_chosen"] = next((r for r in rim if r["letter"] == choice), None)
    bundle["poster"] = write_poster(out, poster_time(bundle))
    path.write_text(json.dumps(_clean(bundle), indent=1) + "\n", encoding="utf-8")
    print(f"REPARSED {path} ({len(bundle['decisions'])} decisions)", flush=True)
    return 0


def write_qpos(out: pathlib.Path, rows: list[np.ndarray]) -> dict | None:
    """`qpos.bin` and its description, or None when the frames carry no pose."""
    if not rows or any(r is None for r in rows):
        return None
    table = np.stack([np.asarray(r, dtype=np.float32).reshape(-1) for r in rows])
    (out / QPOS_NAME).write_bytes(table.astype(QPOS_DTYPE).tobytes())
    return {"path": QPOS_NAME, "frames": int(table.shape[0]), "nq": int(table.shape[1]),
            "dtype": QPOS_DTYPE}


def read_qpos(out: pathlib.Path, bundle: dict) -> np.ndarray | None:
    """The pose table a bundle carries, as `[frames, nq]`, or None."""
    spec = bundle.get("qpos")
    if not spec:
        return None
    raw = np.frombuffer((out / spec["path"]).read_bytes(), dtype=spec.get("dtype", QPOS_DTYPE))
    return raw.reshape(int(spec["frames"]), int(spec["nq"]))


def export_scene(env, scenes_dir: pathlib.Path) -> str | None:
    """The scene this environment is showing, exported into `scenes_dir`; its hash, or None for
    an environment with no MJCF to export. A failed export is said, not fatal: the bundle is
    still a replay without it."""
    if not hasattr(env, "scene_xml"):
        return None
    from robojev import scene_bundle

    try:
        return scene_bundle.ensure_scene(env.scene_xml()) if scenes_dir is None else \
            scene_bundle.export_scene_bundle(env.scene_xml(), scenes_dir)
    except Exception as exc:                                     # noqa: BLE001 - reported
        print(f"robojev record: scene export failed ({type(exc).__name__}: {exc}); "
              f"the bundle has no 3D scene", flush=True)
        return None


def record(env, policy, episode, out: pathlib.Path, *, suite: str, task_index: int,
           init_state_index: int, engine: str, protocol, started: float,
           name: str | None = None, note: str = "", title: str | None = None,
           crf: int = 26, scene: str | None = None,
           scenes_dir: pathlib.Path | None = None) -> dict:
    """Write `episode` into `out` as a bundle, and return the `episode.json` that was written.

    `scene` is a hash already exported (the console exports once per episode); without one the
    environment's scene is exported here - into `scenes_dir`, or `$ROBOJEV_HOME/scenes` unless the
    repository's `data/scenes` already holds it.
    """
    out.mkdir(parents=True, exist_ok=True)
    info = policy.describe()
    fps = float(getattr(env, "control_freq", 20.0))

    decisions, questions_version = [], None
    for frame in episode.frames:
        if frame.policy_query and frame.decisions:
            questions_version = (frame.decisions.get("meta") or {}).get(
                "questions_version", questions_version)
            decisions.append(_decision(frame, len(decisions), fps))

    media = {}
    for camera, filename in (("agentview", "agentview.mp4"), ("wrist", "wrist.mp4")):
        frames = _frames(episode, camera)
        if not frames:
            continue
        write_video(out / filename, frames, fps, crf)
        h, w = frames[0].shape[:2]
        media[camera] = {"path": filename, "width": int(w), "height": int(h),
                         "fps": fps, "frames": len(frames), "codec": "h264"}

    qpos = write_qpos(out, [f.qpos for f in episode.frames])
    if qpos is not None and scene is None:
        scene = export_scene(env, scenes_dir)

    checkpoint = info.get("checkpoint") or {}
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "id": name or out.name,
        "suite": suite,
        "task_index": task_index,
        "init_state_index": init_state_index,
        "instruction": env.instruction,
        "title": title or env.instruction,
        "note": note,
        "policy": engine,
        "checkpoint_revision": checkpoint.get("revision"),
        "checkpoint_repo": checkpoint.get("repo"),
        "questions_version": questions_version,
        "success": bool(episode.success),
        "terminated_by": episode.terminated_by,
        "error": episode.error,
        "steps": int(episode.steps),
        "control_rate": fps,
        "wait_steps": int(protocol.wait_steps),
        "execute_steps": int(protocol.execute_steps),
        "max_steps": int(protocol.max_steps),
        "max_decisions": int(protocol.max_steps // protocol.execute_steps),
        "total_frames": len(episode.frames),
        "duration_s": round(len(episode.frames) / fps, 3),
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": round(time.time() - started, 1),
        "media": media,
        "qpos": qpos,
        "scene": None if qpos is None or scene is None else {"hash": scene, "nq": qpos["nq"]},
        "decisions": decisions,
    }
    bundle["poster"] = write_poster(out, poster_time(bundle))
    (out / "episode.json").write_text(json.dumps(_clean(bundle), indent=1) + "\n",
                                      encoding="utf-8")
    return bundle


# ------------------------------------------------------------------------------- adding a scene

def decode_video(path: pathlib.Path, width: int, height: int) -> np.ndarray:
    """Every frame of an mp4, as `[n, h, w, 3]` uint8."""
    cmd = [ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", str(path),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3)


def replay_actions(env, bundle: dict, init_state_index: int, seed: int = 7):
    """Step `env` through a bundle's own executed actions and return what it went through:
    `(qpos rows, agentview images, success, steps)`, one row and one image per video frame.

    The actions are the recorded ones -- the settling steps' dummy action, then each decision's
    `action` held for `execute_steps` -- so no policy is involved and the decisions stay exactly
    as they were recorded. `bundle_with_scene` then checks the pictures against the bundle's own
    video, which is what says the replay went through the same states.
    """
    from robojev.episode import env_qpos

    try:
        from robojev.envs.libero import set_seed_everywhere
        set_seed_everywhere(seed)
    except ImportError:                                          # pragma: no cover
        pass
    obs = env.reset(init_state_index)
    actions = [np.asarray(env.dummy_action(), dtype=np.float32)] * int(bundle["wait_steps"])
    for d in bundle["decisions"]:
        actions += [np.asarray(d["action"], dtype=np.float32)] * int(bundle["execute_steps"])
    rows, images, success, steps = [], [], False, 0
    for t, action in enumerate(actions):
        rows.append(env_qpos(env))
        images.append(env.images(obs).get("agentview"))
        result = env.step(action.tolist())
        obs = result.obs
        if t >= int(bundle["wait_steps"]):
            steps += 1
        if result.done:
            success = True
            break
    rows.append(env_qpos(env))
    images.append(env.images(obs).get("agentview"))
    return rows, images, success, steps


def bundle_with_scene(out: pathlib.Path, env, *, init_state_index: int, seed: int = 7,
                      scenes_dir: pathlib.Path | None = None, tolerance: float = 6.0) -> dict:
    """Give an existing bundle its `qpos.bin` and its scene by replaying its recorded actions.

    Refuses -- writes nothing -- unless the replay ends the way the recording did (same success,
    same number of steps, same number of frames) and every replayed agentview picture matches the
    bundle's own video frame within `tolerance` (mean absolute difference, 0-255; H.264 at crf 26
    alone accounts for a couple of units).
    """
    out = pathlib.Path(out)
    path = out / "episode.json"
    bundle = json.loads(path.read_text(encoding="utf-8"))
    rows, images, success, steps = replay_actions(env, bundle, init_state_index, seed)
    problems = []
    if bool(success) != bool(bundle["success"]):
        problems.append(f"success {success} != recorded {bundle['success']}")
    if steps != int(bundle["steps"]):
        problems.append(f"steps {steps} != recorded {bundle['steps']}")
    if len(rows) != int(bundle["total_frames"]):
        problems.append(f"frames {len(rows)} != recorded {bundle['total_frames']}")
    track = (bundle.get("media") or {}).get("agentview")
    worst = 0.0
    if track is not None and not problems:
        video = decode_video(out / track["path"], int(track["width"]), int(track["height"]))
        for i, img in enumerate(images[: len(video)]):
            diff = float(np.abs(video[i].astype(np.int16) - np.asarray(img, np.int16)).mean())
            worst = max(worst, diff)
        if worst > tolerance:
            problems.append(f"replayed pictures differ from the video (worst frame {worst:.2f})")
    if problems:
        raise SystemExit(f"robojev scene {out.name}: the replay is not the recording: "
                         + "; ".join(problems))
    qpos = write_qpos(out, rows)
    scene = export_scene(env, scenes_dir)
    bundle["schema_version"] = SCHEMA_VERSION
    bundle["qpos"] = qpos
    bundle["scene"] = None if scene is None else {"hash": scene, "nq": qpos["nq"]}
    path.write_text(json.dumps(_clean(bundle), indent=1) + "\n", encoding="utf-8")
    print(f"SCENE {out.name}: {len(rows)} frames, nq {qpos['nq']}, worst frame {worst:.2f}, "
          f"scene {scene}", flush=True)
    return bundle


__all__ = ["POSTER_NAME", "QPOS_NAME", "RIM_LINE", "SCHEMA_VERSION", "bundle_with_scene",
           "decision_entry", "export_scene", "ffmpeg", "poster_time", "read_qpos", "record",
           "replay_actions", "reparse", "rim_rows", "write_poster", "write_qpos", "write_video"]
