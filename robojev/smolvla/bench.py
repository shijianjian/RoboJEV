"""Closed-loop benchmark on frozen layouts: a SmolVLA checkpoint, or the scripted expert, driving the
simulated XLeRobot to put the bowl on the plate.

    python -m robojev.smolvla.bench --policy expert --out DIR
    python -m robojev.smolvla.bench --policy CHECKPOINT_DIR --out DIR [--n-action-steps 10] [--videos 6]

Evaluation layouts are seeds EVAL_SEED_BASE .. EVAL_SEED_BASE + n - 1: never drawn for demonstrations.
An episode succeeds when the bowl rests on the plate, released, within the time limit (20 s). The
policy sees what the real robot would: three camera images, the 16 joint fields, the instruction.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from .demos import EVAL_SEED_BASE
from .env import CAMERAS, CONTROL_HZ, TableEnv, task
from .scene import sample_layout

MAX_SECONDS = 20.0
# the pretrained SmolVLA knows its cameras as camera1..3
RENAME = {f"observation.images.{k}": f"observation.images.camera{i + 1}" for i, k in enumerate(CAMERAS)}


class SmolVLA:
    def __init__(self, path: str, device: str, n_action_steps: int | None):
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.torch = torch
        self.policy = SmolVLAPolicy.from_pretrained(path).to(device).eval()
        if n_action_steps:
            self.policy.config.n_action_steps = n_action_steps
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=self.policy.config, pretrained_path=path,
            preprocessor_overrides={"device_processor": {"device": device},
                                    "rename_observations_processor": {"rename_map": RENAME}})
        self.device = device

    def reset(self) -> None:
        self.policy.reset()

    def __call__(self, obs: dict, instruction: str) -> np.ndarray:
        t = self.torch
        batch = {"observation.state": t.from_numpy(obs["observation.state"]).float()[None], "task": [instruction]}
        for k in CAMERAS:
            img = t.from_numpy(obs[f"observation.images.{k}"]).permute(2, 0, 1).float() / 255.0
            batch[f"observation.images.{k}"] = img[None]
        batch = self.pre(batch)
        with t.inference_mode():
            a = self.policy.select_action(batch)
        a = self.post(a)
        return a.squeeze(0).float().cpu().numpy()


def run_episode(env: TableEnv, act, video: list | None = None) -> dict:
    obs = env.reset()
    steps, t0, success = int(MAX_SECONDS * CONTROL_HZ), time.time(), False
    lifted = False
    instruction = task(env.layout.arm)
    for k in range(steps):
        a = act(obs, instruction)
        obs = env.step(a)
        lifted |= float(env.data.xpos[env.bowl][2]) > env.rest_z + 0.03
        if video is not None and k % 2 == 0:
            video.append(np.concatenate([obs["observation.images.head"], env.render("front", (256, 342))], 1))
        if env.success():
            success = True
            break
    _, off, _ = env.bowl_on_plate()
    return {"success": success, "seconds": round((k + 1) / CONTROL_HZ, 2), "lifted": lifted,
            "bowl_to_plate_cm": round(off * 100, 1), "wall_s": round(time.time() - t0, 1)}


def _episode(i: int, policy, out: Path, videos: int) -> dict:
    seed = EVAL_SEED_BASE + i
    lay = sample_layout(seed)
    env = TableEnv(lay)
    video = [] if i < videos else None
    if policy is None:
        from .expert import Expert

        env.reset()
        r = _run_expert(env, Expert(env).actions(), video)
    else:
        policy.reset()
        r = run_episode(env, policy, video)
    env.close()
    r.update({"episode": i, "seed": seed, "arm": lay.arm, "distractor": lay.ramekin is not None})
    if video:
        import imageio.v2 as iio

        iio.mimwrite(out / f"ep{i:02d}_{lay.arm}_{'ok' if r['success'] else 'fail'}.mp4", video, fps=CONTROL_HZ // 2,
                     macro_block_size=1)
    print(json.dumps(r), flush=True)
    return r


def _worker(args: tuple) -> list[dict]:
    """One process: its own policy copy and its own share of the episodes (i % n == k)."""
    k, n, a = args
    os.environ.setdefault("MUJOCO_GL", "egl")
    policy = None if a.policy == "expert" else SmolVLA(a.policy, a.device, a.n_action_steps)
    return [_episode(i, policy, a.out, a.videos) for i in range(a.episodes) if i % n == k]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy", required=True, help="'expert' or a SmolVLA checkpoint directory (pretrained_model)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--n-action-steps", type=int, default=None, help="actions executed per chunk (SmolVLA predicts 50)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--videos", type=int, default=6, help="save this many episodes as video")
    ap.add_argument("--workers", type=int, default=1, help="episodes run in this many processes")
    a = ap.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    a.out.mkdir(parents=True, exist_ok=True)
    if a.workers > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(a.workers) as pool:
            eps = [r for rs in pool.map(_worker, [(k, a.workers, a) for k in range(a.workers)]) for r in rs]
    else:
        eps = _worker((0, 1, a))
    eps.sort(key=lambda e: e["episode"])
    summary = {"policy": a.policy, "episodes": len(eps), "success_rate": round(float(np.mean([e["success"] for e in eps])), 3),
               "by_arm": {arm: round(float(np.mean([e["success"] for e in eps if e["arm"] == arm])), 3)
                          for arm in ("left", "right") if any(e["arm"] == arm for e in eps)},
               "mean_seconds_success": round(float(np.mean([e["seconds"] for e in eps if e["success"]])), 2)
               if any(e["success"] for e in eps) else None,
               "n_action_steps": a.n_action_steps, "max_seconds": MAX_SECONDS, "episodes_detail": eps}
    (a.out / "results.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes_detail"}))


def _run_expert(env: TableEnv, gen, video) -> dict:
    """The scripted expert on the same terms: same limit, same success test."""
    steps, t0, success, lifted = int(MAX_SECONDS * CONTROL_HZ), time.time(), False, False
    k = 0
    for k in range(steps):
        a = next(gen, None)
        if a is None:
            a = env.to_driver(env.joints())  # the script is over: hold still
        env.step(a)
        lifted |= float(env.data.xpos[env.bowl][2]) > env.rest_z + 0.03
        if video is not None and k % 2 == 0:
            obs = env.observe()
            video.append(np.concatenate([obs["observation.images.head"], env.render("front", (256, 342))], 1))
        if env.success():
            success = True
            break
    _, off, _ = env.bowl_on_plate()
    return {"success": success, "seconds": round((k + 1) / CONTROL_HZ, 2), "lifted": lifted,
            "bowl_to_plate_cm": round(off * 100, 1), "wall_s": round(time.time() - t0, 1)}


if __name__ == "__main__":
    main()
