"""`robojev` -- run one episode, record one for the web replay, and make a checkpoint.

Six verbs, in the order somebody meets them:

    robojev run      one closed-loop episode; prints whether it succeeded
    robojev record   the same, written out as a replay bundle the web front end serves
    robojev console  a local server: the web app, plus one episode driven from the browser
    robojev harvest  the scripted expert's rollouts, as training rows
    robojev train    NanoJev's own trainer over those rows
    robojev dagger   one round of relabelling the trained policy's own visited states

`run`, `record` and `console` need a simulator (`pip install 'robojev[libero]'`); `train` needs
none on this box, because the fine-tune happens in its own pinned environment. Nothing heavy is
imported until the verb that needs it runs, so `robojev --help` costs the standard library and
numpy.

Progress goes to stderr and the answer goes to stdout, so `$(robojev harvest …)` is a path and
`robojev train --json | jq` works.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time


#: Printed before a local fine-tune: it wants the whole card for minutes, and anything else using
#: the same GPU will either slow it down or be pushed out of memory by it.
TRAIN_WARNING = ("this takes the whole GPU for minutes; nothing else should be using it")


def log(message: str) -> None:
    print(f"robojev: {message}", file=sys.stderr, flush=True)


def indices(text: str) -> list[int]:
    """`"0-7,9"` -> `[0,…,7,9]`. Ranges as well as a list, because a DAgger round names eight
    tasks and ten init states and spelling both out is noise."""
    found: list[int] = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            lo, hi = part.split("-", 1)
            found.extend(range(int(lo), int(hi) + 1))
        else:
            found.append(int(part))
    return found


def build_env(suite: str, task: int, render_size: int = 256, env_seed: int = 0):
    """One environment from the configured adapter (`robojev.envs`)."""
    from robojev import envs

    return envs.make(suite, task, render_size=render_size, env_seed=env_seed)


# ------------------------------------------------------------------------------- run / record

def _one_episode(args, *, images: bool):
    """Build the policy, build the environment, run one episode, and give all three back."""
    from robojev import episode as episode_mod
    from robojev import policy as policy_mod

    policy = policy_mod.build(
        args.policy, args.suite, checkpoint=args.checkpoint, seed=args.seed,
        task_index=args.task, selection=args.selection, ground=args.ground,
        revision=args.revision or None,
    )
    protocol = policy.protocol(args.max_steps)
    env = build_env(args.suite, args.task, render_size=args.render_size,
                    env_seed=protocol.env_seed)
    try:
        result = episode_mod.run_episode(env, policy, protocol, init_state_index=args.init,
                                         seed=args.seed, images=images)
    except BaseException:
        env.close()
        policy.close()
        raise
    return policy, env, protocol, result


def run(args: argparse.Namespace) -> int:
    """One closed-loop episode. Prints one line, and exits non-zero when it failed."""
    from robojev import episode as episode_mod

    started = time.time()
    policy, env, protocol, result = _one_episode(args, images=False)
    try:
        decisions = episode_mod.decision_count(result)
        instruction = env.instruction
    finally:
        env.close()
        policy.close()
    if result.error:
        log(result.error)
    print(f"success={result.success} decisions={decisions} steps={result.steps} "
          f"terminated_by={result.terminated_by} suite={args.suite} task={args.task} "
          f"init={args.init} policy={args.policy} seconds={time.time() - started:.0f} "
          f"instruction={instruction!r}")
    return 0 if result.success else 1


def record(args: argparse.Namespace) -> int:
    """One episode, written into `--out` as an episode bundle (`web/PROTOCOL.md`)."""
    from robojev import recorder

    out = pathlib.Path(args.out)
    if args.reparse:
        return recorder.reparse(out)
    recorder.ffmpeg()                     # fail before an episode is run, not after it
    started = time.time()
    policy, env, protocol, result = _one_episode(args, images=True)
    try:
        bundle = recorder.record(env, policy, result, out, suite=args.suite, task_index=args.task,
                                 init_state_index=args.init, engine=args.policy,
                                 protocol=protocol, started=started, name=args.name,
                                 note=args.note, title=args.title, crf=args.crf)
    finally:
        env.close()
        policy.close()
    size = sum(f.stat().st_size for f in out.iterdir())
    print(f"success={bundle['success']} decisions={len(bundle['decisions'])} "
          f"steps={bundle['steps']} bytes={size} out={out}")
    return 0


# ------------------------------------------------------------------------------------ console

def console(args: argparse.Namespace) -> int:
    """The local console: the built web app on one port, and one episode driven from it.

    The heavy import is the simulator and it happens when an episode is started, not here -- so a
    console comes up in milliseconds and says what it is serving before anything is loaded. What
    *is* checked here is the thing that would otherwise fail in the middle of a browser session:
    a `--policy` this interpreter cannot serve.
    """
    from robojev.console import server as console_mod
    from robojev.console import wire

    policies = console_mod.available_policies()
    if args.policy not in policies:
        why = {
            "model": "a local checkpoint needs torch: pip install 'robojev[model]'. LIBERO pins "
                     "Python 3.10 and NanoJev's predictor pins 3.14, so the two share an "
                     "interpreter only in an environment where both install (README, Limits).",
            "jev": "the hosted model needs $JEV_API_KEY, and every decision it answers is a paid "
                   "request.",
        }.get(args.policy, "this interpreter cannot serve it")
        print(f"console: --policy {args.policy} is not available here: {why}", file=sys.stderr)
        print(f"console: this interpreter serves {', '.join(policies)}", file=sys.stderr)
        return 2

    dist = pathlib.Path(args.dist) if args.dist else console_mod.default_dist()
    if dist is not None and not (dist / "index.html").is_file():
        print(f"console: {dist} has no index.html; build the app with "
              f"`cd web && npm ci && npm run build`", file=sys.stderr)
        return 2
    replays = pathlib.Path(args.replays_dir) if args.replays_dir else console_mod.default_replays()
    default = wire.StartSpec(suite=args.suite, task=args.task, init=args.init,
                             policy=args.policy, selection=args.selection,
                             checkpoint=args.checkpoint, seed=args.seed,
                             max_steps=args.max_steps)
    return console_mod.serve(console_mod.Console(
        host=args.host, port=args.port, dist=dist, replays=replays, policies=policies,
        default=default, render_size=args.render_size, idle_timeout=args.idle_timeout, log=log))


# ------------------------------------------------------------------------------------ harvest

def harvest(args: argparse.Namespace) -> int:
    """The scripted expert's rollouts under ε-noise, as training rows.

    Imported inside the function: this reaches for a simulator, and `robojev --help` must not.
    """
    import contextlib

    from robojev import rollout

    started = time.time()
    with contextlib.redirect_stdout(sys.stderr):
        tasks = tuple(indices(args.tasks)) if args.tasks else tuple(range(args.n_tasks))
        if not tasks:
            print("harvest: --tasks selected no task", file=sys.stderr)
            return 2
        cfg = rollout.RolloutConfig(
            suite=args.suite, tasks=tasks, suite_tasks=tuple(range(args.n_tasks)),
            episodes_per_task=args.episodes_per_task, epsilon=args.epsilon,
            annotate=not args.no_annotate, grounding_rows=not args.no_grounding_rows,
            grounding_repeats=args.grounding_repeats, seed=args.seed)
        out_dir = pathlib.Path(args.out) if args.out else rollout.out_root(args.suite)
        manifest = rollout.harvest_expert(cfg, out_dir, jobs=args.jobs,
                                          grounding_only=args.grounding_only)
    rows = manifest["rows"]
    print(f"robojev harvest: {rows['total']} rows ({rows['grounding']} grounding) in "
          f"{time.time() - started:.0f}s, "
          f"{manifest['success']['successes']}/{manifest['success']['episodes']} episodes "
          f"succeeded", file=sys.stderr)
    for qid, record_ in sorted(manifest["parse_agreement"].items()):
        print(f"  parse {qid}: {record_['agree']}/{record_['rows']}", file=sys.stderr)
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(out_dir / rollout.MANIFEST)
    return 0


# ------------------------------------------------------------------------------------- dagger

def dagger(args: argparse.Namespace) -> int:
    """One DAgger round: roll the trained policy out and relabel the states **it** visited.

    The round's whole claim is that its states are the states a graded run of the same checkpoint
    visits, so the episode loop and the protocol are this CLI's, not the round's own invention.
    """
    import contextlib

    from robojev import dagger as dagger_mod
    from robojev import episode as episode_mod
    from robojev import policy as policy_mod

    started = time.time()
    tasks, inits = indices(args.tasks), indices(args.inits)
    if not tasks or not inits:
        print("dagger: --tasks/--inits selected nothing", file=sys.stderr)
        return 2
    data = pathlib.Path(args.data) if args.data else _harvest_root(args.suite)
    out = pathlib.Path(args.out) if args.out else data.parent / f"{data.name}-dagger{args.round}"
    try:
        manifest = dagger_mod.read_manifest(data)
    except ValueError as exc:
        print(f"dagger: {exc}", file=sys.stderr)
        return 2
    cfg = dagger_mod.config_from_manifest(manifest, suite=args.suite, seed=args.seed,
                                          round=args.round)
    result = out
    if not args.merge_only:
        with contextlib.redirect_stdout(sys.stderr):
            policy = policy_mod.build(args.policy, args.suite, checkpoint=args.checkpoint,
                                      seed=args.seed, selection=args.selection,
                                      ground=args.ground)
            protocol = policy.protocol(args.max_steps)
            try:
                round_manifest = dagger_mod.collect(
                    policy, args.suite, tasks, inits, out, seed=args.seed, cfg=cfg,
                    round=args.round,
                    env_factory=lambda suite, task: build_env(
                        suite, task, render_size=protocol.render_size,
                        env_seed=protocol.env_seed),
                    protocol=protocol, execute_steps=protocol.execute_steps,
                    run=episode_mod.run_episode,
                )
            finally:
                policy.close()
        print(f"robojev dagger: {round_manifest['rows']['total']} rows from "
              f"{len(round_manifest['episodes'])} episodes in {time.time() - started:.0f}s "
              f"(success {round_manifest['success_rate']:.2f}, "
              f"{round_manifest['state_source']})", file=sys.stderr)
    if args.merge:
        merged = pathlib.Path(args.merge)
        extra = [pathlib.Path(p) for p in (args.also or [])]
        try:
            summary = dagger_mod.merge(data, [out, *extra], merged)
        except ValueError as exc:
            print(f"dagger: {exc}", file=sys.stderr)
            return 2
        print(f"robojev dagger: merged {summary['rows']['by_source']} -> {merged}",
              file=sys.stderr)
        result = merged
    if args.json:
        print(json.dumps(dagger_mod.read_manifest(result), indent=2, sort_keys=True))
    else:
        print(result)
    return 0


def _harvest_root(suite: str) -> pathlib.Path:
    from robojev import home

    return home.data_dir("robojev", suite)


# -------------------------------------------------------------------------------------- train

def train(args: argparse.Namespace) -> int:
    """NanoJev's own trainer over the harvested rows, in its own pinned environment.

    `--validate-only` and `--finish` are the two modes that do not touch a GPU: the first is
    upstream's stdlib row audit, the second only reads files a finished run left behind.
    """
    from robojev import runtime
    from robojev import train as trainer

    if args.build_env:
        r = runtime.trainer()
        runtime.nanojev_dir(r, log=log)
        venv = runtime.build_env(r, rebuild=args.rebuild, log=log)
        print(venv)
        return 0

    cfg = trainer.TrainConfig(
        steps=args.steps, seed=args.seed, from_scratch=args.from_scratch,
        eval_every=args.eval_every,
        **({} if args.max_microbatch_tokens is None
           else {"max_microbatch_tokens": args.max_microbatch_tokens}),
        **({} if args.backbone_lr is None else {"backbone_lr": args.backbone_lr}),
        **({} if args.head_lr is None else {"head_lr": args.head_lr}),
        gradient_checkpointing=not args.no_gradient_checkpointing,
        **({} if args.base_model is None else {"base_model": args.base_model}),
        base_revision=args.base_revision,
        data=pathlib.Path(args.data) if args.data else None,
        out=pathlib.Path(args.out) if args.out else None,
    )
    data_dir = cfg.data or trainer.data_root(args.policy, args.suite)
    pinned = set()
    if args.max_microbatch_tokens is not None:
        pinned.add("max_microbatch_tokens")
    if args.gradient_checkpointing or args.no_gradient_checkpointing:
        pinned.add("gradient_checkpointing")
    if args.gradient_checkpointing and args.no_gradient_checkpointing:
        print("train: --gradient-checkpointing and --no-gradient-checkpointing are opposites; "
              "pass one", file=sys.stderr)
        return 2

    if args.validate_only:
        import subprocess
        import tempfile

        r = runtime.trainer()
        with tempfile.TemporaryDirectory() as tmp:
            rows = pathlib.Path(tmp) / trainer.ROWS
            counts = trainer.merge_rows(data_dir, rows)
            print(f"robojev train: {counts['total']} rows ({counts['by_split']})",
                  file=sys.stderr)
            argv, cwd = trainer.validate_command(r, rows)
            print("robojev train: " + " ".join(argv), file=sys.stderr, flush=True)
            return subprocess.run(argv, cwd=cwd).returncode

    try:
        if args.slurm:
            from robojev import slurm as cluster

            place = cluster.SlurmConfig(
                host=args.slurm_host or "", root=args.slurm_root or "",
                partition=args.slurm_partition or "", qos=args.slurm_qos or "",
                user=args.slurm_user or None, resume=not args.no_resume,
                base_revision=args.base_revision,
                **({} if args.slurm_cpus is None else {"cpus": args.slurm_cpus}),
                **({} if args.slurm_mem is None else {"mem": args.slurm_mem}),
                **({} if args.slurm_time is None else {"time_limit": args.slurm_time}))
            report = cluster.train_on_slurm(args.policy, args.suite, cfg, place,
                                            dry_run=args.dry_run, pinned=pinned, log=log)
        elif args.finish:
            report = trainer.finish(pathlib.Path(args.finish), log=log)
        else:
            log(TRAIN_WARNING)
            report = trainer.train(args.policy, args.suite, cfg, log=log)
    except trainer.TrainError as exc:
        print(f"train: {exc}", file=sys.stderr)
        return exc.exit_code
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(json.dumps(report.get("eval") or {}, indent=2, sort_keys=True))
    return 0


# --------------------------------------------------------------------------------- the parser

def _episode_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--init", dest="init", type=int, default=0,
                   help="which of the task's start states")
    p.add_argument("--policy", default="expert", choices=("expert", "model", "jev"),
                   help="expert: the scripted plan. model: a local checkpoint. jev: the hosted "
                        "model (needs $JEV_API_KEY)")
    p.add_argument("--checkpoint", default=None,
                   help="the checkpoint directory (--policy model) or model name (--policy jev)")
    p.add_argument("--revision", default=None)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--selection", default="argmax",
                   help="argmax (default), sample, or sample@<T>")
    p.add_argument("--ground", default="rule", choices=("rule", "forward"),
                   help="rule (default): the grounding forward still runs and is reported, but "
                        "the pair committed is the text rule's. forward: commit the model's own "
                        "answer when it is at least 0.5 sure")
    p.add_argument("--max-steps", type=int, default=None,
                   help="override the suite's own episode cap; a run that does is not comparable")
    p.add_argument("--render-size", type=int, default=256)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="robojev", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="one closed-loop episode; prints whether it succeeded")
    _episode_flags(r)
    r.set_defaults(func=run)

    rec = sub.add_parser("record", help="one episode, written out as a replay bundle")
    _episode_flags(rec)
    rec.add_argument("--out", required=True, help="the bundle directory to write")
    rec.add_argument("--name", default=None, help="the bundle's id; defaults to the directory name")
    rec.add_argument("--note", default="", help="one line: what to look for in this episode")
    rec.add_argument("--title", default=None, help="overrides the task sentence as the card title")
    rec.add_argument("--crf", type=int, default=26)
    rec.add_argument("--reparse", action="store_true",
                     help="run nothing: re-derive the fields parsed out of the state text in an "
                          "existing bundle at --out and rewrite its episode.json")
    rec.set_defaults(func=record)

    c = sub.add_parser("console", help="a local server: the web app, plus one live episode")
    _episode_flags(c)
    c.add_argument("--host", default="127.0.0.1",
                   help="the interface to bind. Loopback by default and on purpose: the console "
                        "is unauthenticated and drives a simulator")
    c.add_argument("--port", type=int, default=8765)
    c.add_argument("--dist", default=None,
                   help="the built front end to serve; defaults to web/dist in this checkout")
    c.add_argument("--replays-dir", default=None,
                   help="where `save` writes a bundle and where the episode strip reads them; "
                        "defaults to web/public/replays in this checkout")
    c.add_argument("--idle-timeout", type=float, default=900.0,
                   help="seconds without a command before the episode is closed and the simulator "
                        "let go of")
    c.set_defaults(func=console)

    h = sub.add_parser("harvest", help="training rows from the scripted expert's rollouts")
    h.add_argument("--suite", default="libero_spatial")
    h.add_argument("--tasks", default=None, help="e.g. 0-7,9. Default: every task of the suite")
    h.add_argument("--n-tasks", type=int, default=10, help="how many tasks the suite has")
    h.add_argument("--episodes-per-task", type=int, default=20)
    h.add_argument("--epsilon", type=float, default=0.10,
                   help="how often a rollout takes a corrupted answer, so the rows cover states "
                        "a perfect controller never visits")
    h.add_argument("--grounding-repeats", type=int, default=1)
    h.add_argument("--grounding-only", action="store_true")
    h.add_argument("--no-grounding-rows", action="store_true")
    h.add_argument("--no-annotate", action="store_true")
    h.add_argument("--seed", type=int, default=17)
    h.add_argument("--jobs", type=int, default=1,
                   help="harvest this many tasks at once, one simulator per process")
    h.add_argument("--out", default=None)
    h.add_argument("--json", action="store_true")
    h.set_defaults(func=harvest)

    d = sub.add_parser("dagger", help="one DAgger round over the policy's own visited states")
    _episode_flags(d)
    d.set_defaults(policy="model")
    d.add_argument("--tasks", default="0-7")
    d.add_argument("--inits", default="0-9")
    d.add_argument("--round", type=int, default=1)
    d.add_argument("--data", default=None, help="the harvest these rows will join")
    d.add_argument("--out", default=None)
    d.add_argument("--merge", default=None, help="write a merged directory here")
    d.add_argument("--also", nargs="*", default=None, help="other rounds to merge in")
    d.add_argument("--merge-only", action="store_true")
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=dagger)

    t = sub.add_parser("train", help="NanoJev's own trainer over the harvested rows")
    t.add_argument("--policy", default="robojev")
    t.add_argument("--suite", default="libero_spatial")
    t.add_argument("--steps", type=int, default=300)
    t.add_argument("--seed", type=int, default=17)
    t.add_argument("--eval-every", type=int, default=50)
    t.add_argument("--from-scratch", action="store_true",
                   help="build fresh heads on a bare backbone instead of warm-starting")
    t.add_argument("--base-model", default=None)
    t.add_argument("--base-revision", default=None)
    t.add_argument("--max-microbatch-tokens", type=int, default=None)
    t.add_argument("--backbone-lr", type=float, default=None)
    t.add_argument("--head-lr", type=float, default=None)
    t.add_argument("--gradient-checkpointing", action="store_true")
    t.add_argument("--no-gradient-checkpointing", action="store_true")
    t.add_argument("--data", default=None)
    t.add_argument("--out", default=None)
    t.add_argument("--validate-only", action="store_true",
                   help="upstream's stdlib row audit: no tokenizer, no weights, no GPU")
    t.add_argument("--finish", default=None,
                   help="skip the trainer and finish a run that already produced weights")
    t.add_argument("--build-env", action="store_true",
                   help="clone the pinned NanoJev and build the trainer's uv environment")
    t.add_argument("--rebuild", action="store_true")
    t.add_argument("--slurm", action="store_true", help="run the fine-tune on a SLURM cluster")
    t.add_argument("--slurm-host", default=None, help="or $ROBOJEV_SLURM_HOST")
    t.add_argument("--slurm-root", default=None,
                   help="the filesystem root every byte goes under, or $ROBOJEV_SLURM_ROOT")
    t.add_argument("--slurm-partition", default=None, help="or $ROBOJEV_SLURM_PARTITION")
    t.add_argument("--slurm-qos", default=None, help="or $ROBOJEV_SLURM_QOS")
    t.add_argument("--slurm-user", default=None)
    t.add_argument("--slurm-cpus", type=int, default=None)
    t.add_argument("--slurm-mem", default=None)
    t.add_argument("--slurm-time", default=None, help="the job's time limit, e.g. 06:00:00")
    t.add_argument("--dry-run", action="store_true",
                   help="--slurm only: print the script and the commands and touch nothing")
    t.add_argument("--no-resume", action="store_true",
                   help="a requeued attempt starts the run again instead of continuing")
    t.add_argument("--json", action="store_true")
    t.set_defaults(func=train)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":                                       # pragma: no cover
    raise SystemExit(main())
