"""`robojev train --slurm` -- the same training run, on a SLURM cluster's big card instead.

Nothing here trains anything either, and nothing here is a second trainer. The command this
renders is the command `robojev.train.trainer_command` builds -- literally, by asking it
and keeping the flags (`trainer_flags`) -- with two numbers changed, and the eval table at the end
is produced by `train.finish` on this box over files the job wrote. The cluster is a bigger card,
not a different pipeline.

**Nothing about one particular cluster is built in.** The host, the filesystem root and the
partition have no defaults: they are given on the command line or in `$ROBOJEV_SLURM_HOST`,
`$ROBOJEV_SLURM_ROOT`, `$ROBOJEV_SLURM_PARTITION` and `$ROBOJEV_SLURM_QOS`, and a run that names
none of them fails before it touches the network.

What is actually different from a local run, and why:

1. **The wheel.** Many GPU clusters are a driver generation behind, and the trainer's own
   `uv.lock` resolves torch from PyPI, which is the newest CUDA build; it fails at
   `torch.cuda.init()` on an older driver. The cluster therefore builds from
   `robojev/trainer/cluster/`, whose lock is the same packages with torch from an older CUDA
   wheel index. See that project's `pyproject.toml`.
2. **The memory budget.** `TrainConfig`'s `max_microbatch_tokens=6000` and
   `gradient_checkpointing=True` are facts about a 24 GB card, argued at length in
   `train.trainer_command`. A 141 GB card does not need either, so `cluster_config` raises the cap
   and turns the recompute off -- *only* when the caller left the local defaults alone, so
   `--max-microbatch-tokens` still means what it says. Neither changes what the trainer optimizes:
   the cap is how many backward passes one optimizer step is split into, and `--batch-questions`
   is unchanged.
3. **Where things live.** Cluster home directories are usually small and quota'd, so every byte --
   the environment, the uv cache, the Python download, the Hugging Face cache, the rows, the
   checkpoint -- goes under the given `--slurm-root`. Nothing is written to `$HOME`, and nothing
   on this box is written outside `$ROBOJEV_HOME`.
4. **Preemption.** A preemptible partition can take the job away at any moment, so it asks for
   `--signal=TERM@60 --requeue`. NanoJev's trainer has no exact resume: `--init-checkpoint` is
   documented as "optimizer is new, not an exact training resume" and there is no optimizer state
   on disk. A requeued attempt therefore **warm-starts from the interrupted attempt's own
   weights** and asks for the steps the run has left (`RESUME_FROM_BEST`), rather than throwing
   away the hours it already spent. What that costs is AdamW's two moments; what it does not cost
   is the learning rate, because there is no schedule. `resume=False` restores the
   exactly-reproducible start-over, which is the better answer only for a run short enough to fit
   between two preemptions.

The local half never runs on the login node beyond `sbatch`, `squeue`, `tail` and `rsync`: login
processes are commonly killed at maintenance, so the driver loop lives here.
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import time
from typing import ClassVar

from robojev import runtime, train as _train

#: The subdirectory holding the older-CUDA project: `pyproject.toml` + `uv.lock`, staged onto
#: the cluster as-is. Both names are fixed by uv.
CLUSTER_PROJECT = "cluster"
PROJECT_FILES = ("pyproject.toml", "uv.lock")

#: Written into a built environment once `uv sync` has returned 0. Its absence -- not the presence
#: of a `.venv` -- is what makes the job build: an interrupted sync leaves a directory that looks
#: finished and is not.
ENV_STAMP = ".robojev-cluster-built"

#: The job's own receipt, written into the checkpoint directory and rsynced back with it: the
#: resolved warm-start revision (which only the node knows, because only the node downloads it),
#: the node, the restart count and the trainer's wall clock. `train.finish` never reads it;
#: `TrainOnSlurm` does, to fill in the provenance `train.train` would have recorded locally.
RUN_RECEIPT = "robojev-slurm-run.json"

#: The microbatch cap for a large card. 32768 is upstream's own default (16384) doubled: at
#: 3,738 padded tokens for a 7-candidate LIBERO-Spatial question it admits eight questions per
#: backward, so a 12-question optimizer step is two backward passes instead of six. Applied by
#: `cluster_config` only over `TrainConfig`'s own defaults.
CLUSTER_MICROBATCH_TOKENS = 32768

#: `--signal=TERM@60` is a one-minute grace window, and `--requeue` is what makes the job come
#: back. Both are the usual pattern for a preemptible partition.
PREEMPTION_SIGNAL = "TERM@60"

#: One poll is one ssh. 20 s keeps the log readable without hammering the login node; a queue wait
#: is usually measured in tens of seconds and an epoch in tens of minutes.
POLL_SECONDS = 20.0

#: Bytes of new log a single poll will carry back. The trainer prints one short JSON object every
#: twelve steps, so this is never reached in practice; it bounds a runaway stack trace.
LOG_CHUNK = 1 << 16

#: Separates the `squeue` answer from the log chunk in one poll's output. Chosen so it cannot
#: occur in either.
POLL_MARK = b"\n--8<--robojev--\n"

#: The `--init-checkpoint` value in the rendered script is resolved *on the node* (the warm start
#: is downloaded there, 2.4 GB at 123 MB/s, never rsynced). `trainer_flags` puts this token in the
#: argument list and `render_sbatch` swaps it for the shell variable after quoting.
INIT_PLACEHOLDER = "@ROBOJEV_INIT_CHECKPOINT@"

#: `--steps` is the other value the script decides for itself, and for the same reason: a resumed
#: attempt asks for the steps the run has left rather than for the whole epoch again. Same
#: treatment -- `trainer_flags` puts the token in the argument list and `_quote` swaps it for the
#: shell variable after quoting, so the flag list stays `train.trainer_command`'s.
STEPS_PLACEHOLDER = "@ROBOJEV_STEPS@"


#: What a requeued attempt does when `SlurmConfig.resume` is false: throw the interrupted output
#: away and run the whole thing again from the warm start. Reproducible, and -- on a partition
#: that preempts about hourly -- a job that can never finish anything longer than an hour.
RESTART_FROM_SCRATCH = '  # NanoJev writes no optimizer state and refuses an output directory that already holds\n  # config.json, so a preempted run restarts from the warm start rather than resuming.\n  echo "robojev: restart $RESTART -- clearing $OUT and training again from the warm start"\n  rm -rf "$OUT"'

#: And what it does by default: keep the interrupted attempt beside the run, warm-start the next
#: one from its weights, and ask for the steps the run has left.
#:
#: This is **not** an exact resume, and the script says so out loud. AdamW's two moments are lost,
#: because upstream writes none. What is *not* lost is the learning rate: there is no schedule at
#: all -- a constant-LR AdamW, diagnosis the measurements -- so the next attempt optimizes the same
#: objective at the same rate from a better initialisation, which is exactly what upstream
#: documents `--init-checkpoint` to be ("optimizer is new, not an exact training resume").
#:
#: The alternative was measured rather than reasoned about. One run took three attempts and
#: 3 h 18 of wall clock to land 1 h 05 of work, and one of those three *finished the trainer* and
#: was preempted before it could exit. Starting over each time means a run longer than the gap
#: between two preemptions never completes at all, which is
#: a worse answer than a warm-started continuation.
#:
#: The steps already taken are subtracted, so the run is one epoch *across* the attempts rather
#: than one epoch per attempt. Every attempt is kept at `$RUN/attempt-<n>/` and nothing is deleted.
#:
#: **They are counted from the job's own stdout**, and that is the fix for the failure the run-4
#: the run-4 diagnosis records. The count used to come out of every previous attempt's
#: `train_log.json`, and upstream writes that file **once, after the final evaluation**
#: (`train_pipeline_decisions.py:562`) -- so a *preempted* attempt never has one, the count is
#: zero from the only source it reads, and the restart asks for the whole epoch again. One run
#: was preempted at step 2329 of 2906 and its restart printed `0 of 2906 steps already taken`:
#: the right weights, for 2,906 more steps instead of the 577 it had left. On a partition that
#: preempts about hourly, a run that resets its own progress never finishes.
#:
#: Stdout has the data. `--open-mode=append` keeps one file across restarts and upstream logs one
#: JSON record per logged step whose `step` counts from zero within its own attempt, so the total
#: is the sum of each attempt's high-water mark and an attempt boundary is a step that did not
#: increase. `train_log.json` stays as a floor: where both exist they agree, and taking the larger
#: means a source going missing can only ever *under*-resume, never skip work.
RESUME_FROM_BEST = '  PRIOR="$RUN/attempt-$RESTART"\n  if [ -f "$OUT/best.safetensors" ] && [ -f "$OUT/config.json" ]; then\n    rm -rf "$PRIOR"; mv "$OUT" "$PRIOR"\n    DONE=$("$VENV/bin/python" - "$RUN" "$STDOUT_LOG" <<\'ROBOJEV_COUNT\'\nimport glob, json, sys\n\nrun, log = sys.argv[1], sys.argv[2]\n\n\ndef from_stdout(path):\n    # Steps taken across every attempt, from the job\'s own stdout. `--open-mode=append` keeps one\n    # file across restarts, and upstream logs one JSON record per logged step whose `step` counts\n    # from zero within its own attempt -- so the total is the sum of each attempt\'s high-water\n    # mark, and an attempt boundary is a step that did not increase.\n    total = attempt = 0\n    try:\n        handle = open(path, errors="replace")\n    except OSError:\n        return 0\n    with handle:\n        for line in handle:\n            line = line.strip()\n            if line[:1] != chr(123) or \'"phase"\' not in line:\n                continue\n            try:\n                record = json.loads(line)\n            except ValueError:\n                continue\n            if record.get("phase") != "full":\n                continue\n            step = record.get("step")\n            if not isinstance(step, int):\n                continue\n            if step <= attempt:          # the counter restarted: a new attempt began\n                total += attempt\n                attempt = 0\n            attempt = max(attempt, step)\n    return total + attempt\n\n\ndef from_train_logs(run):\n    # The old source, kept as a floor. Upstream writes `train_log.json` once, after the final\n    # evaluation, so a preempted attempt has none -- but a completed one does, and where both\n    # exist they agree.\n    done = 0\n    for path in sorted(glob.glob(run + "/attempt-*/train_log.json")):\n        try:\n            done += sum(1 for r in json.load(open(path)) if r.get("phase") == "full")\n        except Exception:\n            pass\n    return done\n\n\nprint(max(from_stdout(log), from_train_logs(run)))\nROBOJEV_COUNT\n)\n    STEPS=$(( {total} - DONE ))\n    if [ "$STEPS" -lt 1 ]; then STEPS=1; fi\n    INIT_ARGS=(--init-checkpoint "$PRIOR")\n    echo "robojev: restart $RESTART -- $DONE of {total} steps already taken; continuing from $PRIOR for $STEPS more (not an exact resume: AdamW moments are new, and there is no LR schedule to lose)"\n  else\n    echo "robojev: restart $RESTART -- the interrupted attempt left no weights; starting over"\n    rm -rf "$OUT"\n  fi'


#: How often the job samples the card's own memory use while the trainer runs (seconds). Thirty
#: is far finer than the thing being measured -- a peak that lasts less than that is not a peak a
#: run can be sized against -- and 120 samples an hour is nothing beside a training step.
VRAM_SAMPLE_SECONDS: int = 30

#: How many dev evaluations a cluster run gets, when the caller did not set the cadence itself.
#: Ten is enough to read the curve's shape and select a checkpoint from it -- one run's eight told
#: the whole story (1.117 at step 0, 0.92 by 303, noise around 0.90 for 1,500 steps) -- and few
#: enough that they cost minutes rather than hours. See `cluster_config`.
CLUSTER_EVALS: int = 10


class SlurmError(_train.TrainError):
    """A cluster-side failure, carrying `train.TrainError`'s exit code contract."""


# ------------------------------------------------------------------------------ the configuration


@dataclasses.dataclass(frozen=True)
class SlurmConfig:
    """Where the job goes and what it asks for.

    **`host`, `root` and `partition` have no defaults.** They are a property of somebody's cluster
    and nothing in this package can guess them, so they are given on the command line or in
    `$ROBOJEV_SLURM_HOST` / `$ROBOJEV_SLURM_ROOT` / `$ROBOJEV_SLURM_PARTITION`, and a config built
    without them raises. `qos` is optional (`$ROBOJEV_SLURM_QOS`) because many partitions need
    none; prefer a preemptible partition if the site has one, and expect the `--signal`/`--requeue`
    handling above to earn its keep there.

    The resource defaults are sized for the job rather than for a site: `cpus=16` because a large
    card finishes a 32k-token microbatch faster than one core tokenizes the next one; `mem=64G`
    covers fp32 weights, the row file and the tokenizer's arenas with room to spare;
    `time=06:00:00` is a cap, not a booking.
    """

    #: `ROBOJEV_SLURM_<FIELD>` supplies any of the four required-ish fields below.
    ENV_PREFIX: ClassVar[str] = "ROBOJEV_SLURM_"

    host: str = ""                                # required: --slurm-host or $ROBOJEV_SLURM_HOST
    user: str | None = None                       # None: whatever ssh resolves remotely
    root: str = ""                                # required: --slurm-root or $ROBOJEV_SLURM_ROOT
    partition: str = ""                           # required: --slurm-partition or $..._PARTITION
    qos: str = ""                                 # optional: --slurm-qos or $ROBOJEV_SLURM_QOS
    gres: str = "gpu:1"
    cpus: int = 16
    mem: str = "64G"
    time_limit: str = "06:00:00"
    uv: str = "$HOME/.local/bin/uv"               # uv is rarely on a non-interactive PATH
    poll_seconds: float = POLL_SECONDS
    #: The warm start's commit, or `None` for whatever `C-Tianyu/NanoJev` is at now -- which is
    #: exactly what a local `robojev train` gets (`train.base_checkpoint` pins no revision either),
    #: so the default is the *same* behaviour on both boxes and not a cluster quirk.
    #:
    #: It has to be settable because those weights are unversioned and they **move**: the
    #: repository's `main` carried `best.safetensors` sha256 `fff62d14…` on the morning of
    #: 2026-09-20 and `f68c47d6…` that afternoon, under two different commits. A cluster run meant
    #: to be compared against a local run from before the change has to be given the commit that
    #: run recorded in its own `robojev.json` (`base_checkpoint`), or the two runs are fine-tunes
    #: of two different models and every number is incomparable.
    base_revision: str | None = None
    #: Whether a requeued attempt continues from the interrupted one's weights (`RESUME_FROM_BEST`,
    #: the default) or throws them away and starts the run again (`RESTART_FROM_SCRATCH`). False is
    #: the exactly-reproducible answer and the right one for a run short enough to fit between two
    #: preemptions; the default is the one that finishes.
    resume: bool = True

    #: The fields that must come from somewhere, and the one that may be empty.
    REQUIRED: ClassVar[tuple[str, ...]] = ("host", "root", "partition")
    FROM_ENV: ClassVar[tuple[str, ...]] = ("host", "root", "partition", "qos", "user")

    def __post_init__(self) -> None:
        """Fill the site-specific fields from `$ROBOJEV_SLURM_*`, then insist on the three.

        Reading the environment here rather than at the call site means the same rule holds for
        the CLI, for a script that builds a config directly and for a test, and there is exactly
        one error message when a cluster was never named.
        """
        for name in self.FROM_ENV:
            if not getattr(self, name):
                value = os.environ.get(f"{self.ENV_PREFIX}{name.upper()}")
                if value:
                    object.__setattr__(self, name, value.strip())
        object.__setattr__(self, "root", (self.root or "").rstrip("/"))
        missing = [name for name in self.REQUIRED if not getattr(self, name)]
        if missing:
            flags = ", ".join(f"--slurm-{name}" for name in missing)
            env = ", ".join(f"${self.ENV_PREFIX}{name.upper()}" for name in missing)
            raise SlurmError(
                f"this run names no cluster: {', '.join(missing)} must be given. They are "
                f"properties of somebody's SLURM site and nothing here can guess them -- pass "
                f"{flags}, or set {env}.",
                exit_code=2,
            )


def cluster_config(cfg: _train.TrainConfig, pinned=frozenset()) -> _train.TrainConfig:
    """`cfg` with a large card's memory budget, and only where the caller kept the small one's.

    The two fields this touches are the two `train.TrainConfig` documents as facts about the card
    rather than tuning choices. An operator who passed `--max-microbatch-tokens 6000` or asked for
    gradient checkpointing explicitly meant it -- comparing the cluster against the local card step
    for step is exactly the reason to -- so their value survives.

    `pinned` is the set of field names the operator named on the command line, which is the only
    way to say "yes, gradient checkpointing, **on the cluster too**": its default is `True`, so
    asking for it is indistinguishable from leaving it alone by value. That is not hypothetical --
    a 4B backbone needs the recompute on a 141 GB card exactly as a 0.6B one needs it on a 24 GB
    card, and without this the cluster would silently turn it off and OOM.
    """
    defaults = _train.TrainConfig()
    changes = {}
    # Evaluations, which are the run's other big cost and the one nobody was counting. The
    # default cadence is 50 steps, which over an epoch of 2906 is **58 dev evaluations**; one run
    # measured a 4B dev evaluation at 299 s, of which ~60 s is writing a 16 GB checkpoint. That is
    # 4.8 hours of evaluation attached to 2.5 hours of training, on a partition that preempts about
    # hourly and only writes the held-out table *after* the last one. `CLUSTER_EVALS` says
    # how many are worth having instead: enough to see the curve's shape and pick a checkpoint,
    # and few enough that they are a rounding error beside the steps.
    if cfg.eval_every == defaults.eval_every and cfg.steps > 0:
        spaced = max(defaults.eval_every, -(-cfg.steps // CLUSTER_EVALS))
        if spaced != cfg.eval_every:
            changes["eval_every"] = spaced
    if ("max_microbatch_tokens" not in pinned
            and cfg.max_microbatch_tokens == defaults.max_microbatch_tokens):
        changes["max_microbatch_tokens"] = CLUSTER_MICROBATCH_TOKENS
    if ("gradient_checkpointing" not in pinned
            and cfg.gradient_checkpointing == defaults.gradient_checkpointing):
        changes["gradient_checkpointing"] = False
    return dataclasses.replace(cfg, **changes) if changes else cfg


# ------------------------------------------------------------------------------- the remote paths


@dataclasses.dataclass(frozen=True)
class Layout:
    """Every path the cluster side uses, derived from the root and two content hashes.

    The two caches are keyed by content, not by run: `envs/robojev@<sha256(uv.lock)[:12]>` is the
    same name `runtime.TrainerEnv.env_name` gives the local environment (a different hash, because
    it is a different lockfile) and `src/nanojev@<commit[:12]>` is the same name
    `runtime.nanojev_clone` gives the local clone. Change the lock or the pin and the next run builds beside the old one
    rather than into it; leave them alone and every run after the first is a no-op.
    """

    root: str
    env_name: str
    nanojev_name: str
    nanojev_url: str
    nanojev_commit: str
    nanojev_subdir: str
    run_id: str

    @property
    def envs(self) -> str:
        return f"{self.root}/envs"

    @property
    def env(self) -> str:
        return f"{self.envs}/{self.env_name}"

    @property
    def venv(self) -> str:
        return f"{self.env}/.venv"

    @property
    def src(self) -> str:
        return f"{self.root}/src/{self.nanojev_name}"

    @property
    def scripts(self) -> str:
        return f"{self.src}/{self.nanojev_subdir}"

    @property
    def hf(self) -> str:
        return f"{self.root}/hf"

    @property
    def uv_cache(self) -> str:
        return f"{self.root}/uv/cache"

    @property
    def uv_python(self) -> str:
        return f"{self.root}/uv/python"

    @property
    def run(self) -> str:
        return f"{self.root}/runs/{self.run_id}"

    @property
    def data(self) -> str:
        return f"{self.run}/data"

    @property
    def rows(self) -> str:
        return f"{self.data}/{_train.ROWS}"

    @property
    def checkpoint(self) -> str:
        return f"{self.run}/checkpoint"

    @property
    def logs(self) -> str:
        return f"{self.run}/logs"

    @property
    def script(self) -> str:
        return f"{self.run}/job.sbatch"

    def log(self, job_id: str) -> str:
        """The job's stdout+stderr. One file across requeues (`--open-mode=append`), so the
        poller's byte offset stays meaningful when a preempted job comes back."""
        return f"{self.logs}/slurm-{job_id}.out"


def cluster_project(r: runtime.TrainerEnv) -> pathlib.Path:
    """`robojev/trainer/cluster/`, checked to be a uv project. Never the trainer's own directory."""
    directory = r.dir / CLUSTER_PROJECT
    missing = [name for name in PROJECT_FILES if not (directory / name).is_file()]
    if missing:
        raise SlurmError(
            f"no cluster project: {directory} is missing {', '.join(missing)}. It is the "
            f"older-CUDA twin of the trainer's own lock, and the only thing that installs a torch "
            f"a cluster on an older driver can initialise.",
            exit_code=2,
        )
    return directory


def lock_hash(directory: pathlib.Path) -> str:
    """sha256 of a project's `uv.lock`, the way `runtime.lockfile_hash` computes it."""
    import hashlib

    return hashlib.sha256((directory / "uv.lock").read_bytes()).hexdigest()


def layout(r: runtime.TrainerEnv, cfg: SlurmConfig, run_id: str) -> Layout:
    pin = runtime.nanojev_pin(r)
    if pin is None:  # pragma: no cover -- only if the sidecar is removed from the recipe
        raise SlurmError(f"{r.dir} pins no NanoJev checkout ({runtime.NANOJEV_PIN})")
    url, commit, subdir = pin
    return Layout(
        root=cfg.root.rstrip("/"),
        env_name=f"{r.server}@{lock_hash(cluster_project(r))[:12]}",
        nanojev_name=f"nanojev@{commit[:12]}",
        nanojev_url=url, nanojev_commit=commit, nanojev_subdir=subdir,
        run_id=run_id,
    )


# ------------------------------------------------------------------------------ the trainer's own


def trainer_flags(r: runtime.TrainerEnv, cfg: _train.TrainConfig, place: Layout) -> list[str]:
    """The flags `train.trainer_command` would pass, with the cluster's paths. Asked, not copied.

    `trainer_command` returns a whole argv -- an interpreter, the script, then the flags -- and
    which interpreter depends on what is built on *this* box. The flags do not, so they are taken
    from after the script's own path. Building them here instead would be a second list to keep in
    step with the first, and the entire claim of this module is that the cluster runs the same
    command; the test that asserts it (`test_slurm.py`) compares against `trainer_command` too.
    """
    argv, _ = _train.trainer_command(
        r, place.rows, place.checkpoint, dataclasses.replace(cfg, steps=STEPS_PLACEHOLDER),
        init_checkpoint=INIT_PLACEHOLDER if _train.warm_starts_from_nanojev(cfg) else None,
    )
    for index, token in enumerate(argv):
        if token.endswith(_train.TRAINER):
            return argv[index + 1:]
    raise SlurmError(  # pragma: no cover -- trainer_command always names the script
        f"could not find {_train.TRAINER} in the trainer command {argv!r}")


# --------------------------------------------------------------------------------- the job script


def _quote(flags) -> str:
    """Shell-quote a flag list, leaving the two placeholders as live shell expansions.

    `--init-checkpoint` is dropped from the line entirely and replaced by `"${INIT_ARGS[@]}"`
    appended after it: the flag is present on some attempts of one job and absent on others (a
    bare-backbone run starts without one and resumes with one), and a bash array is the only way
    to say "these two words, or nothing" without quoting an empty argument into the argv.
    """
    out, skip = [], False
    for flag in flags:
        if skip:
            skip = False
            continue
        if flag == "--init-checkpoint":
            skip = True
            continue
        out.append(shlex.quote(str(flag)))
    return " ".join(out).replace(shlex.quote(STEPS_PLACEHOLDER), '"$STEPS"')


def render_sbatch(r: runtime.TrainerEnv, cfg: _train.TrainConfig, slurm: SlurmConfig,
                  place: Layout) -> str:
    """The whole job, as one `bash` script. Deterministic: same inputs, same bytes.

    It does four things in order, and the first three are cached across runs by content:

    * **the environment** -- `uv sync --frozen` from the staged cu126 project into
      `envs/<name>/.venv`, under an `flock` so two runs submitted together build once, and behind
      a stamp file so the second run is a no-op. `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR` and
      `UV_PROJECT_ENVIRONMENT` all point under the root: uv's defaults are all in `$HOME`, which
      is the quota that is nearly full.
    * **the clone** -- NanoJev at the pinned commit, into `src/nanojev@<commit>`, cloned to a
      `.partial` name and renamed, so an interrupted clone is never mistaken for a finished one.
    * **the warm start** -- `snapshot_download` of the four entries upstream's loader needs, into
      the shared `HF_HOME`. Downloaded on the node (123 MB/s from HF) rather than rsynced (2.4 GB
      up a home link), and the revision it resolves is written into the receipt because it is
      provenance this box cannot otherwise know.
    * **the trainer** -- upstream's script, with `trainer_flags`, from the clone's `scripts/`.

    Everything before the trainer is idempotent; the trainer is not, which is what the restart
    branch is about. On a requeue (`SLURM_RESTART_COUNT` > 0) the output directory is cleared and
    the run starts over: NanoJev writes no optimizer state and documents `--init-checkpoint` as
    not a resume, so there is nothing to continue from and a "resume" would silently be a second
    fine-tune of a half-trained model at a restarted learning rate.
    """
    # What a requeued attempt does with the output of the one before it. See `SlurmConfig.resume`.
    restart = (RESUME_FROM_BEST.format(total=cfg.steps) if slurm.resume
               else RESTART_FROM_SCRATCH)
    resume = "true" if slurm.resume else "false"
    flags = trainer_flags(r, cfg, place)
    # A run from a bare backbone (`--base-model Qwen/Qwen3-4B`) downloads nothing here: the
    # trainer's own `AutoModel.from_pretrained` pulls it into the same shared `HF_HOME` on the
    # node, and there is no local `DecisionModel` directory for `--init-checkpoint` to name.
    # Two flags can pin the warm start's commit and they mean the same thing: `--slurm-base-revision`
    # (this module's, because only the node downloads it) and `--base-revision` (the trainer's, which
    # is what a *local* run would use). The cluster-specific one wins when both are given.
    base_revision = slurm.base_revision if slurm.base_revision is not None else cfg.base_revision
    warm_start = "" if not _train.warm_starts_from_nanojev(cfg) else f"""
# --- the warm start: {_train.BASE_CHECKPOINT_REPO}, on the node, into the shared HF_HOME --------
"$VENV/bin/python" - <<'PY' > "$RUN/base.json"
import json, pathlib
from huggingface_hub import snapshot_download
directory = snapshot_download(
    repo_id={_train.BASE_CHECKPOINT_REPO!r},
    revision={base_revision!r},
    allow_patterns=["config.json", "best.safetensors", "backbone_config/*", "tokenizer/*"],
)
print(json.dumps({{"path": directory, "revision": pathlib.Path(directory).name}}))
PY
INIT_CHECKPOINT="$("$VENV/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["path"])' "$RUN/base.json")"
BASE_REVISION="$("$VENV/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["revision"])' "$RUN/base.json")"
INIT_ARGS=(--init-checkpoint "$INIT_CHECKPOINT")
echo "robojev: warm start $INIT_CHECKPOINT"
"""
    return f"""#!/bin/bash
#SBATCH --job-name=robojev-{place.run_id}
#SBATCH --partition={slurm.partition}
#SBATCH --qos={slurm.qos}
#SBATCH --gres={slurm.gres}
#SBATCH --cpus-per-task={slurm.cpus}
#SBATCH --mem={slurm.mem}
#SBATCH --time={slurm.time_limit}
#SBATCH --signal={PREEMPTION_SIGNAL}
#SBATCH --requeue
#SBATCH --chdir={place.run}
#SBATCH --output={place.logs}/slurm-%j.out
#SBATCH --open-mode=append
# Written by robojev.slurm. Every path is under {place.root}; nothing touches $HOME.
set -euo pipefail

ROOT={shlex.quote(place.root)}
ENV_DIR={shlex.quote(place.env)}
VENV={shlex.quote(place.venv)}
SRC={shlex.quote(place.src)}
SCRIPTS={shlex.quote(place.scripts)}
RUN={shlex.quote(place.run)}
OUT={shlex.quote(place.checkpoint)}
UV={slurm.uv}

export UV_CACHE_DIR={shlex.quote(place.uv_cache)}
export UV_PYTHON_INSTALL_DIR={shlex.quote(place.uv_python)}
export UV_PROJECT_ENVIRONMENT="$VENV"
export HF_HOME={shlex.quote(place.hf)}
# Long padded microbatches of varying width fragment the caching allocator; torch's own answer,
# and the same default `train.run_trainer` sets locally.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
unset VIRTUAL_ENV CONDA_PREFIX PYTHONHOME PYTHONPATH || true

mkdir -p "$ROOT/envs" "$ROOT/src" "$RUN" {shlex.quote(place.logs)} "$HF_HOME" "$UV_CACHE_DIR"
RESTART="${{SLURM_RESTART_COUNT:-0}}"
# This job's own stdout, which `--open-mode=append` keeps across restarts: one JSON record per
# logged step, and the only place a *preempted* attempt's progress survives (see
# `RESUME_FROM_BEST`, and the run-4 diagnosis note §R4.5).
STDOUT_LOG={shlex.quote(place.logs)}/slurm-$SLURM_JOB_ID.out
# Either empty, or the two words `--init-checkpoint <dir>`. See `_quote`.
INIT_ARGS=()
echo "robojev: job $SLURM_JOB_ID on $(hostname), restart $RESTART, $(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

# --- the environment: cu126, cached by lock hash, built once under a lock ----------------------
(
  flock 9
  if [ ! -f "$ENV_DIR/{ENV_STAMP}" ]; then
    echo "robojev: uv sync --frozen $ENV_DIR"
    "$UV" sync --frozen --project "$ENV_DIR"
    date -Is > "$ENV_DIR/{ENV_STAMP}"
  else
    echo "robojev: environment already built at $ENV_DIR"
  fi
) 9>"$ROOT/envs/.{place.env_name}.lock"

# --- NanoJev at the pinned commit --------------------------------------------------------------
(
  flock 9
  if [ ! -d "$SRC/.git" ]; then
    echo "robojev: cloning {place.nanojev_url} @ {place.nanojev_commit}"
    rm -rf "$SRC.partial"
    git clone --quiet {shlex.quote(place.nanojev_url)} "$SRC.partial"
    git -C "$SRC.partial" checkout --quiet --detach {place.nanojev_commit}
    mv "$SRC.partial" "$SRC"
  else
    echo "robojev: NanoJev already at $SRC"
  fi
) 9>"$ROOT/src/.{place.nanojev_name}.lock"
test -d "$SCRIPTS"
{warm_start}
# --- the trainer -------------------------------------------------------------------------------
STEPS={cfg.steps}
if [ "$RESTART" != "0" ]; then
{restart}
fi
mkdir -p "$OUT"
STARTED=$(date +%s)
# Peak VRAM, sampled while the run is alive. `summary.json`'s `max_gpu_allocated_gb` is torch's
# own high-water mark and it is only written in the final block, *after* the evaluations -- so a
# preempted attempt reports none at all, which is exactly the attempt whose memory one wants to
# know about. This is the card's number rather than the allocator's (it includes the CUDA context
# and any fragmentation), sampled every {VRAM_SAMPLE_SECONDS}s, and the maximum goes into the
# receipt below. It costs one `nvidia-smi` a sample and dies with the job.
VRAM_LOG="$RUN/vram-$SLURM_JOB_ID.csv"
( while true; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$VRAM_LOG" 2>/dev/null || true
    sleep {VRAM_SAMPLE_SECONDS}
  done ) &
VRAM_SAMPLER=$!
trap 'kill $VRAM_SAMPLER 2>/dev/null || true' EXIT
cd "$SCRIPTS"
"$VENV/bin/python" {shlex.quote(place.scripts + "/" + _train.TRAINER)} {_quote(flags)} "${{INIT_ARGS[@]}}"
ELAPSED=$(( $(date +%s) - STARTED ))
kill $VRAM_SAMPLER 2>/dev/null || true
PEAK_VRAM_MIB=$(sort -n "$VRAM_LOG" 2>/dev/null | tail -1)
PEAK_VRAM_MIB=${{PEAK_VRAM_MIB:-0}}

# --- the receipt: what only the node knows -----------------------------------------------------
cat > "$OUT/{RUN_RECEIPT}" <<EOF
{{"job_id": "$SLURM_JOB_ID", "node": "$(hostname)", "restart_count": $RESTART,
 "base_checkpoint_revision": "${{BASE_REVISION:-}}", "trainer_seconds": $ELAPSED,
 "peak_vram_mib": $PEAK_VRAM_MIB,
 "final_attempt_steps": $STEPS, "resume": {resume},
 "partition": "{slurm.partition}", "finished_at": "$(date -Is)"}}
EOF
echo "robojev: trainer finished in ${{ELAPSED}}s -> $OUT"
"""


# ------------------------------------------------------------------------------------ the commands


def ssh_command(slurm: SlurmConfig, remote: str) -> list[str]:
    """`ssh <host> "bash -lc '<remote>'"` -- one argument, quoted twice, on purpose.

    ssh does not pass an argv: it joins everything after the host with spaces and hands the
    result to the far side's login shell, which then re-splits it. A command passed as separate
    arguments therefore loses its quoting on the way -- `bash -lc 'a && b'` arrives as
    `bash -lc a && b`, which runs `a` and then `b` in the *outer* shell. So the whole remote
    command is quoted into a single word here, and `bash -lc` is the only thing ssh's own shell
    sees. The login shell is needed because `uv` lives in `~/.local/bin`, which the cluster's
    non-interactive PATH does not carry (probe the measurements).
    """
    return ["ssh", "-o", "BatchMode=yes", _host(slurm), f"bash -lc {shlex.quote(remote)}"]


def _host(slurm: SlurmConfig) -> str:
    return slurm.host if slurm.user is None else f"{slurm.user}@{slurm.host}"


def rsync_up_command(slurm: SlurmConfig, local: pathlib.Path, remote_dir: str) -> list[str]:
    """Push a local directory's *contents* into `remote_dir`, creating the path.

    `--mkpath` is not assumed (it is uv-era rsync and the login node's may predate it): the
    directory is made by `--rsync-path`, which runs on the far end before the transfer. Trailing
    slash on the source, so `local/x` lands at `remote_dir/x` rather than `remote_dir/local/x`.
    """
    return [
        "rsync", "-a", "--stats", "--rsync-path", f"mkdir -p {shlex.quote(remote_dir)} && rsync",
        f"{str(local).rstrip('/')}/", f"{_host(slurm)}:{remote_dir}/",
    ]


def rsync_down_command(slurm: SlurmConfig, remote_dir: str, local: pathlib.Path) -> list[str]:
    """Pull a remote directory's contents into `local`. Pulled, never pushed: the compute nodes
    cannot reach this box."""
    return ["rsync", "-a", "--stats", f"{_host(slurm)}:{remote_dir.rstrip('/')}/",
            f"{str(local).rstrip('/')}/"]


def sbatch_command(slurm: SlurmConfig, place: Layout) -> list[str]:
    """`sbatch --parsable`, from the run directory, so stdout is the job id and nothing else."""
    return ssh_command(slurm, f"cd {shlex.quote(place.run)} && sbatch --parsable "
                              f"{shlex.quote(place.script)}")


def poll_command(slurm: SlurmConfig, place: Layout, job_id: str, offset: int) -> list[str]:
    """One ssh that answers both questions a poll asks: what is the job doing, and what has it
    said since byte `offset`. `squeue` is authoritative while the job exists and silent once it
    does not, which is exactly the signal to switch to `sacct`."""
    log = shlex.quote(place.log(job_id))
    mark = POLL_MARK.decode().strip("\n")
    return ssh_command(
        slurm,
        f"squeue -h -j {job_id} -o '%T|%R' 2>/dev/null || true; "
        f"printf '\\n{mark}\\n'; "
        f"tail -c +{offset + 1} {log} 2>/dev/null | head -c {LOG_CHUNK} || true",
    )


def sacct_command(slurm: SlurmConfig, job_id: str) -> list[str]:
    """The finished job's accounting row, `|`-separated and header-less. `-X` keeps it to the job
    rather than its steps, which is what `Elapsed` and `State` are wanted for."""
    return ssh_command(
        slurm,
        f"sacct -j {job_id} -X -n -P -o "
        f"JobID,State,Submit,Start,End,Elapsed,ExitCode,NodeList,MaxRSS",
    )


# -------------------------------------------------------------------------------------- the parses


#: `squeue -h -o '%T|%R'` for one job: a state and a reason, or nothing at all once the job has
#: left the queue. `PENDING|(Resources)`, `RUNNING|node-17`, `PREEMPTED|(null)`.
def parse_squeue(text: str) -> tuple[str, str] | None:
    """`(state, reason)` for the one job, or `None` when the queue no longer knows it.

    An array or a requeue can momentarily show two lines; the first is taken, because the poller
    only ever asks about a single job id and a second line is the same job in another state.
    """
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        state, _, reason = line.partition("|")
        return state.strip(), reason.strip()
    return None


_SACCT_FIELDS = ("job_id", "state", "submit", "start", "end", "elapsed", "exit_code",
                 "nodes", "max_rss")


def parse_sacct(text: str) -> dict | None:
    """The first `sacct -X -n -P` row as a dict, or `None` if the row has not appeared yet.

    `sacct` lags `squeue` by a few seconds at job exit, so "no row" is a retry rather than a
    failure. `state` keeps SLURM's own spelling, including the `CANCELLED by <uid>` form.
    """
    for line in text.splitlines():
        parts = line.strip().split("|")
        if len(parts) < len(_SACCT_FIELDS) or not parts[0]:
            continue
        return dict(zip(_SACCT_FIELDS, parts))
    return None


_RSYNC_SENT = re.compile(r"^[Tt]otal bytes sent:\s*([\d,]+)", re.MULTILINE)
_RSYNC_RECEIVED = re.compile(r"^[Tt]otal bytes received:\s*([\d,]+)", re.MULTILINE)


def parse_rsync_stats(text: str) -> dict:
    """`{"sent": int, "received": int}` out of `rsync --stats`. Zeroes when it did not say."""
    def number(pattern) -> int:
        match = pattern.search(text)
        return int(match.group(1).replace(",", "")) if match else 0

    return {"sent": number(_RSYNC_SENT), "received": number(_RSYNC_RECEIVED)}


#: States `squeue`/`sacct` report for a job that is over. `PREEMPTED` and `REQUEUED` are *not*
#: here: with `--requeue` they are a job on its way back to `PENDING`, not an end.
TERMINAL_STATES = ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL",
                   "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE")


def is_terminal(state: str) -> bool:
    return state.split()[0].upper() in TERMINAL_STATES if state else False


# ------------------------------------------------------------------------------------- the driving


def _run(argv: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    """`subprocess.run`, with a failure that says what the far side said.

    `CalledProcessError` prints the argv and swallows stderr, which for an ssh or an rsync is the
    entire diagnosis ("Permission denied", "No space left on device", "Invalid partition name").
    """
    result = subprocess.run(argv, capture_output=True, text=True)
    if check and result.returncode:
        raise SlurmError(
            f"`{' '.join(shlex.quote(token) for token in argv)}` exited {result.returncode}: "
            f"{(result.stderr or result.stdout).strip() or '(no output)'}"
        )
    return result


def submit(slurm: SlurmConfig, place: Layout, log=_train.log_to_stderr) -> str:
    result = _run(sbatch_command(slurm, place))
    job_id = result.stdout.strip().split(";")[0]        # `--parsable` is `<id>[;<cluster>]`
    if not job_id.isdigit():
        raise SlurmError(f"sbatch did not return a job id: {result.stdout!r} {result.stderr!r}")
    log(f"submitted job {job_id} to {slurm.partition}")
    return job_id


def wait(slurm: SlurmConfig, place: Layout, job_id: str, log=_train.log_to_stderr,
         sleep=time.sleep) -> dict:
    """Poll until the job leaves the queue, relaying the log. Returns the `sacct` row plus timings.

    Three clocks, because a preempted job makes one clock a lie. `queue_seconds` is the total time
    the job was *not* running -- the first wait plus every wait after a requeue, which is the
    number that says whether the partition is worth using. `run_seconds` is the total time it *was*
    running, including the attempts that were thrown away. `last_run_seconds` is the final attempt
    alone, and it is the only one that describes the checkpoint that came back, because a
    requeued job starts the epoch again from the warm start.

    A requeue does not end the wait: the job id does not change, `squeue` shows it PENDING again,
    and `restarts` counts it.
    """
    mark = POLL_MARK.decode().strip("\n")
    offset, started, running_at, restarts, state = 0, time.time(), None, 0, "PENDING"
    last_state, unreachable = None, 0
    pending_since, queued, ran = started, 0.0, 0.0

    def read() -> tuple[str | None, int]:
        """One poll: `(squeue text or None when ssh failed, bytes of log relayed)`."""
        nonlocal offset
        result = _run(poll_command(slurm, place, job_id, offset), check=False)
        if mark not in result.stdout:
            # A dropped ssh, not a finished job: the marker is printed unconditionally. Saying
            # "the job is gone" here would end the wait on a flaky login node.
            return None, 0
        head, _, tail = result.stdout.partition(mark)
        chunk = tail.lstrip("\n")
        if chunk:
            offset += len(chunk.encode("utf-8", "replace"))
            print(chunk, end="" if chunk.endswith("\n") else "\n", file=sys.stderr, flush=True)
        return head, len(chunk)

    while True:
        head, _relayed = read()
        if head is None:
            unreachable += 1
            if unreachable > 10:
                raise SlurmError(
                    f"{slurm.host} has been unreachable for {unreachable} polls while job "
                    f"{job_id} was running. The job has NOT been cancelled and neither has "
                    f"anything under {place.run}; read {place.log(job_id)} on the cluster and "
                    f"pull {place.checkpoint} back by hand when it finishes."
                )
            sleep(slurm.poll_seconds)
            continue
        unreachable = 0
        current = parse_squeue(head)
        if current is None:
            read()                                   # whatever the job said after the last poll
            break
        state, reason = current
        if state == "RUNNING" and running_at is None:
            running_at = time.time()
            queued += running_at - (pending_since or running_at)
            pending_since = None
            log(f"job {job_id} {'re' if restarts else ''}started on {reason} after "
                f"{queued:.0f}s in the queue" + (f" (attempt {restarts + 1})" if restarts else ""))
        if state == "PENDING" and running_at is not None:
            # Preempted and on its way back: `--requeue` returns it to PENDING under the same id.
            # Only PENDING counts -- a job on its way *out* passes through COMPLETING, and
            # treating that as a requeue would report every clean run as having restarted once.
            restarts += 1
            ran += time.time() - running_at
            running_at, pending_since = None, time.time()
            log(f"job {job_id} was requeued ({reason}) after {ran:.0f}s of running; NanoJev "
                f"cannot resume, so it starts the run again from the warm start")
        if state != last_state:
            log(f"job {job_id}: {state} {reason}".rstrip())
            last_state = state
        sleep(slurm.poll_seconds)

    row = None
    for _ in range(10):                      # sacct lags squeue by a few seconds at job exit
        row = parse_sacct(_run(sacct_command(slurm, job_id), check=False).stdout)
        if row is not None:
            break
        sleep(slurm.poll_seconds)
    finished = time.time()
    last_run = finished - running_at if running_at is not None else 0.0
    return {
        "job_id": job_id,
        "sacct": row,
        "state": (row or {}).get("state", state),
        "queue_seconds": round(queued + (finished - pending_since if pending_since else 0.0), 1),
        "run_seconds": round(ran + last_run, 1),
        "last_run_seconds": round(last_run, 1),
        "restarts": restarts,
        "log": place.log(job_id),
    }


# ------------------------------------------------------------------------------------ the command


def run_id_for(suite: str, stamp: str) -> str:
    return f"{suite}-{stamp}"


def train_on_slurm(policy: str, suite: str, cfg: _train.TrainConfig, slurm: SlurmConfig,
                   *, dry_run: bool = False, pinned=frozenset(),
                   log=_train.log_to_stderr, sleep=time.sleep) -> dict:
    """The whole cluster run: stage, submit, wait, pull back, and `train.finish` on this box.

    The split between the two boxes is drawn where the dependencies are. The cluster gets the
    trainer and nothing else -- a bash script, uv, git and the recipe's own environment -- because
    installing this package there would mean a second copy to keep in step for the sake
    of one function. This box gets everything either side of it: `merge_rows` (stdlib, seconds,
    and it is where the cross-split check belongs -- before a queue, not after one) and
    `train.finish`, which is a pure function of the files the job wrote and therefore produces
    *byte-identically* the eval table a local run produces.

    The `.tmp-<stamp>` directory is this box's, made before the job is submitted and filled by
    the rsync at the end, so the run lands exactly where `robojev train --finish` expects it and
    the failure story is `train.train`'s: a run that dies leaves a directory nothing reads.
    """
    r = runtime.trainer()
    cfg = cluster_config(cfg, pinned)
    data_dir = pathlib.Path(cfg.data) if cfg.data else _train.data_root(policy, suite)
    final = pathlib.Path(cfg.out) if cfg.out else _train.checkpoint_root(policy, suite)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    place = layout(r, slurm, run_id_for(suite, stamp))
    # Before anything is created, as `train.train` does -- and the same resolution of the path
    # budget from the rows' own question set, so the job script the cluster runs carries the
    # `--max-length` a local run would have passed.
    cfg = _train.for_rows(cfg, _train.read_manifest(data_dir))

    work = final.parent / f"{final.name}.tmp-{stamp}"
    scratch = final.parent / f"{final.name}.tmp-{stamp}.rows"
    project = cluster_project(r)
    script = render_sbatch(r, cfg, slurm, place)

    if dry_run:
        # No ssh, no rsync, nothing created: the script and the four commands, on stdout, so a
        # reviewer can read what would run and a test can assert it.
        print(script)
        for argv in (rsync_up_command(slurm, project, place.env),
                     rsync_up_command(slurm, scratch, place.data),
                     sbatch_command(slurm, place),
                     poll_command(slurm, place, "<job>", 0),
                     rsync_down_command(slurm, place.checkpoint, work)):
            print("$ " + " ".join(shlex.quote(token) for token in argv))
        return {"dry_run": True, "script": script, "layout": place, "config": cfg,
                "work": str(work), "run_id": place.run_id}

    scratch.mkdir(parents=True, exist_ok=False)
    moved = {"up": 0, "down": 0}
    try:
        rows = scratch / _train.ROWS
        counts = _train.merge_rows(data_dir, rows)
        log(f"{counts['total']} rows from {len(counts['files'])} task file(s) -> {rows} "
            f"({counts['by_split']})")
        # The manifest travels with the rows only so the run directory on the cluster describes
        # itself; `finish` reads the local copy, which is the one that was measured.
        (scratch / _train.HARVEST_MANIFEST).write_bytes(
            (data_dir / _train.HARVEST_MANIFEST).read_bytes())
        (scratch / "job.sbatch").write_text(script, encoding="utf-8")

        log(f"staging the cu126 project -> {slurm.host}:{place.env}")
        moved["up"] += parse_rsync_stats(
            _run(rsync_up_command(slurm, project, place.env)).stdout)["sent"]
        log(f"staging the rows -> {slurm.host}:{place.data}")
        moved["up"] += parse_rsync_stats(
            _run(rsync_up_command(slurm, scratch, place.data)).stdout)["sent"]
        # The script is submitted from the run directory, so it has to be there and not in data/.
        _run(ssh_command(slurm, f"mkdir -p {shlex.quote(place.logs)} && "
                                f"cp {shlex.quote(place.data)}/job.sbatch "
                                f"{shlex.quote(place.script)}"))

        job_id = submit(slurm, place, log=log)
        outcome = wait(slurm, place, job_id, log=log, sleep=sleep)
        if not outcome["state"].startswith("COMPLETED"):
            raise SlurmError(
                f"job {job_id} ended {outcome['state']!r}; the log is "
                f"{slurm.host}:{outcome['log']} and the run directory {place.run} was left alone."
            )

        work.mkdir(parents=True, exist_ok=False)
        log(f"pulling the checkpoint back -> {work}")
        moved["down"] += parse_rsync_stats(
            _run(rsync_down_command(slurm, place.checkpoint, work)).stdout)["received"]
        receipt = _train._read_json(work / RUN_RECEIPT, {}) or {}
        missing = runtime.missing_checkpoint_files(work)
        if missing:
            raise SlurmError(
                f"job {job_id} completed but {work} is missing {', '.join(missing)} after the "
                f"rsync; the cluster's copy is still at {place.checkpoint}."
            )
    except BaseException:
        # Same rule as `train.train`: up to the point where the four files are here there is no
        # checkpoint, so a half-pulled directory is removed rather than left to be verified. The
        # cluster's copy is never deleted -- it is hours of a GPU and the only other copy.
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(scratch, ignore_errors=True)
        raise
    shutil.rmtree(scratch, ignore_errors=True)
    log(f"job {job_id}: {outcome['queue_seconds']:.0f}s queued, {outcome['run_seconds']:.0f}s "
        f"running, {outcome['restarts']} restart(s), {moved['up']} B up / {moved['down']} B down")

    # Everything `train.train` would have written before launching the trainer, written now that
    # the trainer's output is here, so `finish` is the same function over the same inputs.
    flags = trainer_flags(r, cfg, place)
    base = receipt.get("base_checkpoint_revision") or None
    # The receipt is the one file of this that outlives `finish` -- the context is unlinked at the
    # rename, and `robojev.json`'s shape is `train.finish`'s, not this module's. So the local half
    # of the run (what the queue cost, what moved, which cluster) is added to the node's half and
    # travels with the weights.
    receipt = {**receipt, "host": slurm.host, "root": place.root, "run": place.run,
               "queue_seconds": outcome["queue_seconds"], "wait_seconds": outcome["run_seconds"],
               "restarts": outcome["restarts"], "sacct": outcome["sacct"],
               "bytes_sent": moved["up"], "bytes_received": moved["down"],
               "base_revision_asked": slurm.base_revision}
    (work / RUN_RECEIPT).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                                    encoding="utf-8")
    (work / _train.TRAIN_CONTEXT).write_text(json.dumps({
        "policy": policy, "suite": suite, "data": str(data_dir), "out": str(final),
        "stamp": stamp, "started_at": time.time() - float(outcome["run_seconds"]),
        "trainer": [f"{place.venv}/bin/python", f"{place.scripts}/{_train.TRAINER}", *flags],
        "trainer_cwd": place.scripts, "rows": counts, "max_path_tokens": cfg.max_length,
        "base_checkpoint": None if not _train.warm_starts_from_nanojev(cfg) or not base
        else f"{_train.BASE_CHECKPOINT_REPO}@{base}",
        "base_model": "{}@{}".format(*_train.backbone_for(cfg)),
        "slurm": {**outcome, "receipt": receipt, "root": place.root, "run": place.run,
                  "partition": slurm.partition, "host": slurm.host,
                  "bytes_sent": moved["up"], "bytes_received": moved["down"]},
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = _train.finish(work, log=log)
    summary["slurm"] = {**outcome, "receipt": receipt, "bytes": moved}
    return summary
