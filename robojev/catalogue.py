"""The task catalogue: what the Dataset tab lists, and what it previews with no simulator running.

robopp's `robopp/catalogue.py`, ported for LIBERO: one directory per task,

    <root>/catalogue/<suite>/<task>/task.json       name, instruction, objects, goal, scene hash, ...
    <root>/catalogue/<suite>/<task>/thumb.png       the agentview at start state 0
    <root>/catalogue/<suite>/<task>/init_qpos.json  {n, nq, qpos}: every start state's pose

and the task's compiled scene under `<root>/scenes/<hash>/` (`robojev.scene_bundle`, robopp's
export). The page poses that scene from `init_qpos.json`, so choosing a task or a start state
shows it at once and costs no simulator.

The repository tracks the entries its showcased runs use in `showcase/`; the full catalogue of the
five LIBERO suites (robopp's own export) sits in the untracked `data/`. `robojev catalogue`
regenerates entries, into `data/` for a suite it holds and into `$ROBOJEV_HOME` otherwise.
"""
from __future__ import annotations

import json
import os
import pathlib

from robojev.scene_bundle import DEFAULT_TEXTURE_MAX, REPO_DATA, SHOWCASE, export_scene_bundle

#: The version of `task.json` (robopp's `task.schema.json` `x-schema-version`).
SCHEMA_VERSION = 1


def roots() -> list[pathlib.Path]:
    """Where catalogue entries are read from, in order: `showcase/`, `data/`, `$ROBOJEV_HOME`."""
    from robojev.home import home

    return [SHOWCASE, REPO_DATA, home()]


def default_root(suite: str) -> pathlib.Path:
    """Where `robojev catalogue` writes: `data/` for a suite the repository ships, else home."""
    from robojev.home import home

    return REPO_DATA if (REPO_DATA / "catalogue" / suite).is_dir() else home()


def parse_bddl(suite: str, task_index: int) -> dict:
    """robopp's `parse_bddl`: name, instruction, objects, fixtures, goal, objects of interest."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.bddl_utils import robosuite_parse_problem

    task = benchmark.get_benchmark_dict()[suite]().get_task(task_index)
    parsed = robosuite_parse_problem(os.path.join(get_libero_path("bddl_files"), task.problem_folder,
                                                  task.bddl_file))
    instruction = parsed["language_instruction"]
    if isinstance(instruction, list):
        instruction = " ".join(instruction)
    return {
        "name": task.bddl_file[:-len(".bddl")],
        "instruction": instruction,
        "objects": parsed["objects"],
        "fixtures": parsed["fixtures"],
        "goal": parsed["goal_state"],
        "objects_of_interest": parsed["obj_of_interest"],
    }


def export_task(suite: str, task_index: int, root: pathlib.Path, env=None,
                texture_max: int = DEFAULT_TEXTURE_MAX) -> dict:
    """Write one task's entry under `root` and return its `task.json`: robopp's `export_task` for
    a LIBERO suite. `env` is built (and closed) here when not given."""
    from robojev.console.images import encode_png

    root = pathlib.Path(root)
    task_dir = root / "catalogue" / suite / str(task_index)
    task_dir.mkdir(parents=True, exist_ok=True)
    owns = env is None
    if owns:
        from robojev import envs

        env = envs.make(suite, task_index, render_size=256, env_seed=0)
    try:
        obs0 = env.reset(0)
        (task_dir / "thumb.png").write_bytes(encode_png(env.images(obs0)["agentview"]))
        parsed = parse_bddl(suite, task_index)
        scene = export_scene_bundle(env.scene_xml(), root / "scenes", texture_max=texture_max)
        rows = []
        for i in range(env.num_init_states):
            env.reset(i)
            rows.append(env.ground_truth()[0].tolist())
        (task_dir / "init_qpos.json").write_text(json.dumps({"n": len(rows), "nq": len(rows[0]), "qpos": rows}))
        doc = {
            "schema_version": SCHEMA_VERSION,
            "suite": suite,
            "task_index": task_index,
            "simulator": "libero",
            "name": parsed["name"],
            "instruction": env.instruction,
            "objects": parsed["objects"],
            "fixtures": parsed["fixtures"],
            "goal": parsed["goal"],
            "objects_of_interest": parsed["objects_of_interest"],
            "scene_bundle": scene,
            "max_steps": env.max_steps_default(),
            "n_init_states": len(rows),
            "thumbnail": "thumb.png",
            "init_qpos": "init_qpos.json",
        }
        # task.json last: a directory without one is an export that did not finish.
        (task_dir / "task.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    finally:
        if owns:
            env.close()
    return doc


def run_catalogue(suite: str, tasks: list[int] | None = None, root: pathlib.Path | None = None,
                  log=print) -> list[dict]:
    """Export every task of `suite` (or `tasks`), one simulator at a time."""
    from libero.libero import benchmark

    count = benchmark.get_benchmark_dict()[suite]().n_tasks
    root = default_root(suite) if root is None else pathlib.Path(root)
    docs = []
    for task_index in (tasks if tasks is not None else range(count)):
        doc = export_task(suite, task_index, root)
        log(f"{suite}/{task_index}: {doc['scene_bundle'][:12]} {doc['name']}")
        docs.append(doc)
    return docs


def index() -> list[dict]:
    """Every catalogued suite and its tasks, from both roots (the repository's first):
    `[{suite, tasks: [task.json + {root}]}]`, suites in LIBERO's order, tasks by index."""
    order = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]
    suites: dict[str, dict[int, dict]] = {}
    for root in roots():
        base = root / "catalogue"
        if not base.is_dir():
            continue
        for suite_dir in base.iterdir():
            if not suite_dir.is_dir():
                continue
            for task_dir in suite_dir.iterdir():
                path = task_dir / "task.json"
                if not path.is_file() or not task_dir.name.isdigit():
                    continue
                tasks = suites.setdefault(suite_dir.name, {})
                if int(task_dir.name) in tasks:
                    continue
                try:
                    doc = json.loads(path.read_text(encoding="utf-8"))
                except ValueError:
                    continue
                tasks[int(task_dir.name)] = doc
    names = sorted(suites, key=lambda s: (order.index(s) if s in order else len(order), s))
    return [{"suite": s, "tasks": [suites[s][i] for i in sorted(suites[s])]} for s in names]


__all__ = ["SCHEMA_VERSION", "default_root", "export_task", "index", "parse_bddl", "roots",
           "run_catalogue"]
