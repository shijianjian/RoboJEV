"""One episode as a replay bundle the web front end can serve.

A bundle is a directory `<name>/` holding

    episode.json      the schema below (`SCHEMA_VERSION`)
    agentview.mp4     H.264 / yuv420p / +faststart, one frame per control step
    wrist.mp4         the same, from the in-hand camera
    poster.jpg        one still, for the episode strip

and nothing else. The videos are written at the environment's own control rate (20 Hz on LIBERO)
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

SCHEMA_VERSION = 1

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


def record(env, policy, episode, out: pathlib.Path, *, suite: str, task_index: int,
           init_state_index: int, engine: str, protocol, started: float,
           name: str | None = None, note: str = "", title: str | None = None,
           crf: int = 26) -> dict:
    """Write `episode` into `out` as a bundle, and return the `episode.json` that was written."""
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
        "decisions": decisions,
    }
    bundle["poster"] = write_poster(out, poster_time(bundle))
    (out / "episode.json").write_text(json.dumps(_clean(bundle), indent=1) + "\n",
                                      encoding="utf-8")
    return bundle


__all__ = ["POSTER_NAME", "RIM_LINE", "SCHEMA_VERSION", "ffmpeg", "poster_time", "record",
           "reparse", "rim_rows", "write_poster", "write_video"]
