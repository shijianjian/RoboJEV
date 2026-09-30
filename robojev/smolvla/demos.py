"""Scripted XLeRobot demonstrations as a LeRobot dataset, for fine-tuning SmolVLA.

    python -m robojev.smolvla.demos --out DIR --episodes 400 [--workers 16] [--first-seed 0]

Each worker runs layouts (one seed each) through the scripted expert and writes its own dataset
shard; only successful demonstrations are kept. The shards are then merged into DIR/merged. Seeds
at and above EVAL_SEED_BASE are the benchmark's and are refused here.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

from .env import CAMERAS, CONTROL_HZ, FIELDS, TASK, task

EVAL_SEED_BASE = 1_000_000


def features(image: int) -> dict:
    f = {"observation.state": {"dtype": "float32", "shape": (len(FIELDS),), "names": FIELDS},
         "action": {"dtype": "float32", "shape": (len(FIELDS),), "names": FIELDS}}
    for key in CAMERAS:
        f[f"observation.images.{key}"] = {"dtype": "video", "shape": (image, image, 3),
                                          "names": ["height", "width", "channels"]}
    return f


def _worker(args: tuple) -> dict:
    wid, seeds, out, image = args
    os.environ.setdefault("MUJOCO_GL", "egl")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from .env import TableEnv
    from .expert import Expert
    from .scene import sample_layout

    root = Path(out) / "shards" / f"w{wid:02d}"
    ds = LeRobotDataset.create(repo_id=f"local/xlerobot_bowl_w{wid:02d}", fps=CONTROL_HZ, features=features(image),
                               root=root, robot_type="xlerobot_2wheels", use_videos=True)
    kept, failed, log = 0, 0, []
    for seed in seeds:
        lay = sample_layout(seed)
        env = TableEnv(lay, image)
        obs = env.reset()
        frames = []
        for a in Expert(env).actions():
            frames.append({**obs, "action": a})
            obs = env.step(a)
        ok = env.success()
        env.close()
        log.append({"seed": seed, "arm": lay.arm, "success": bool(ok), "steps": len(frames)})
        if not ok:
            failed += 1
            continue
        for fr in frames:
            ds.add_frame({**fr, "task": task(lay.arm)})
        ds.save_episode(parallel_encoding=False)  # this worker is already one of many processes
        kept += 1
    ds.finalize()
    return {"worker": wid, "root": str(root), "kept": kept, "failed": failed, "log": log}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=400, help="layouts to try (failures are dropped)")
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--image", type=int, default=256)
    a = ap.parse_args()
    seeds = list(range(a.first_seed, a.first_seed + a.episodes))
    if seeds[-1] >= EVAL_SEED_BASE:
        raise SystemExit(f"seeds >= {EVAL_SEED_BASE} are reserved for the benchmark")
    chunks = [seeds[i::a.workers] for i in range(a.workers)]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        results = pool.map(_worker, [(i, c, str(a.out), a.image) for i, c in enumerate(chunks) if c])
    from lerobot.datasets.aggregate import aggregate_datasets

    shards = [r for r in results if r["kept"]]
    aggregate_datasets(repo_ids=[f"local/xlerobot_bowl_w{r['worker']:02d}" for r in shards],
                       aggr_repo_id="local/xlerobot_bowl", roots=[Path(r["root"]) for r in shards],
                       aggr_root=a.out / "merged")
    log = sorted((e for r in results for e in r["log"]), key=lambda e: e["seed"])
    (a.out / "demos.json").write_text(json.dumps(
        {"task": TASK, "fps": CONTROL_HZ, "fields": FIELDS, "tried": len(log),
         "kept": sum(e["success"] for e in log), "episodes": log}, indent=1))
    print(f"kept {sum(e['success'] for e in log)} of {len(log)} demonstrations -> {a.out / 'merged'}")


if __name__ == "__main__":
    main()
