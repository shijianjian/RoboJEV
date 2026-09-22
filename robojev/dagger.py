"""One DAgger round: the states the trained policy actually visits, labelled by the label function.

Note §5.3. The policy is trained on states whose tracker blocks were filled by the harvest's own
label rollouts -- a loop that follows its labels and therefore rarely wanders. At test
time the same blocks are filled from the model's own answers, and a successful rollout never
contains a failed attempt, so the history a rollout writes for itself is drawn from a distribution
no training row was ever sampled from. That is the classic covariate shift behind imitation
learning, and one round of DAgger (Ross et al.) is the cheapest honest fix:

1. roll the trained policy out on the training tasks;
2. record the state it was actually given at every decision point -- **its own prompt, verbatim**,
   out of `decisions.meta.state`, never a reconstruction;
3. relabel each visited state with `TrackerV2.gold_answers` on a tracker replayed over the
   episode -- the **same** label function the harvest's own rows come from;
4. add the rows to the training set and train again.

**What is not relabelled.** The `state` text. It is the policy's own prompt, byte for byte,
including the tracker block it wrote itself; rebuilding it here would defeat the entire point of
the round. Where a policy reports none at all (a fake, a non-decision baseline) the relabeller
renders its own and the row says so in `metadata.state_source`.

**What is replayed rather than re-decided.** The grounding commit (which bowl the model named) and
the serving latch's refusals. If the model named the wrong bowl, the correction for that is a
corrected `target` answer; a motor label quietly computed about a different bowl would be a label
of a state nobody read.

The round is **pure on-policy**: every executed chunk is the policy's own. A label source that
also drove the arm would be measuring a different policy than the one being corrected.

Nothing here writes into a harvest directory. `collect` writes its own, in the harvest's layout
(`task_<i>.jsonl` plus a manifest), and `merge` builds a third directory holding both -- the
harvest rows copied byte for byte, the DAgger rows appended after them under the same task file so
that `train.merge_rows`'s `task_*.jsonl` glob and its cross-split check keep working untouched.
"""
from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import pathlib
from typing import Iterator

import numpy as np

from robojev import dataset as dataset_mod
from robojev import envs as env_mod
from robojev import roles as roles_mod
from robojev import train as train_mod
from robojev.dataset import log_to_stderr
from robojev.compose import CHUNK_STEPS

#: The manifest a `collect` writes, and the one `merge` writes. The harvest's own name, because a
#: DAgger directory is a training directory: `robojev train` reads it the same way.
MANIFEST = dataset_mod.MANIFEST

#: `metadata.source` on every row this module writes. The one key that tells a merged file's rows
#: apart, and what the per-source counts in a merged manifest are keyed by.
SOURCE = "dagger"

#: `metadata.source` the harvest's rows carry once they are merged. They do not carry it on disk
#: -- a harvest row is never rewritten -- so `merge` counts them by the absence of the key.
HARVEST_SOURCE = "harvest"

#: What a row's `state` text was: the policy's own prompt, or one the relabeller rendered because
#: the policy reported none. A served checkpoint always reports one and the relabeller asserts
#: its own rebuild against it byte for byte; `rebuilt` is a fake or a baseline, and the manifest
#: counts the two so a round that measured nothing is visible.
STATE_SOURCES = ("policy", "rebuilt")

PROVENANCE = dataset_mod.PROVENANCE

#: `rows.by_source` key for the v2 harvest's BDDL grounding rows, carried into a merged
#: directory as their own file (see `merge`).
GROUNDING_SOURCE = "grounding"

#: Who says what should have been answered at a state the model visited: `TrackerV2.gold_answers`
#: on a tracker rebuilt from the recorded observations, which is the **same** label function the
#: rows were harvested with. A DAgger round labelled by a different rule than the rows it will be
#: trained beside is a second dataset, not a correction of the first, which is why there is one
#: name here and not a choice.
LABELLER = "tracker"


# --------------------------------------------------------------------------------------------
# configuration


@dataclasses.dataclass
class DaggerConfig:
    """Everything that decides what a DAgger row says, in one object the manifest echoes.

    Almost all of it is read off the harvest manifest by `config_from_manifest`, and that is the
    point: a DAgger row has to be labelled with the numbers the rows it will be trained beside
    were labelled with -- the same δ pair, the same floors, the same memory settings, the same
    per-task split -- or the merged directory holds two datasets. `merge` re-checks the ones the
    harvest calls `ROW_SHAPE_KEYS`.
    """

    suite: str = "libero_spatial"
    #: Which DAgger round this is. It is the second field of every row id, so a second round
    #: against a better policy can be collected into the same directory without a collision.
    round: int = 1
    seed: int = 17

    #: The composer's step sizes, from the harvest manifest. A row is only meaningful beside the
    #: δ it was labelled with.
    delta_t: float = 0.0
    delta_r: float = 0.0
    chunk_steps: int = CHUNK_STEPS
    fine_label_threshold: float = 0.5
    move_floor: float = 0.005
    yaw_floor: float = 0.05

    #: The clock the state's `Decision k of N` line counts against, and how many decisions and
    #: events its blocks carry. Only the settings that change a rendered byte -- the ones the
    #: harvest manifest's `memory` block carries.
    horizon: int = 220
    every: int = CHUNK_STEPS
    history_k: int = 3
    max_events: int = 8
    target_rule: str = "obj_of_interest"

    #: The split each task's rows inherit, from the harvest manifest. A DAgger row of task 3 is a
    #: row of task 3: it lands in whatever split task 3 is already in, or the trainer's
    #: cross-split check fails on the merged file (`train.merge_rows`).
    splits: dict[int, str] = dataclasses.field(default_factory=dict)

    def shape(self) -> dict:
        """The row-shape keys this module can speak for, for the manifest.

        A DAgger directory has no grip rollout and no demonstration, so the keys that describe
        those are absent rather than guessed at; `merge` compares only the keys both manifests
        carry. The round's states are `v2.state.TrackerV2`'s and its labels are
        `TrackerV2.gold_answers`', which is exactly what the harvest manifest calls
        `memory_rule: "v2-tracker-1"` and `label_source: "tracker"` -- so those are the two
        strings reported here, and `merge` accepts the round against the harvest it was collected
        from.
        """
        from robojev import rollout as v2rollout               # noqa: PLC0415
        from robojev.state import MEMORY_RULE_V2               # noqa: PLC0415

        return {
            "suite": self.suite, "every": self.every, "chunk_steps": self.chunk_steps,
            "fine_label_threshold": self.fine_label_threshold,
            "move_floor": self.move_floor, "yaw_floor": self.yaw_floor,
            "seed": self.seed,
            "label_source": v2rollout.LABEL_SOURCE,
            "labeller": LABELLER,
            "questions_version": "v2",
            "memory_rule": MEMORY_RULE_V2,
            "horizon": self.horizon,
            "history_k": self.history_k, "max_events": self.max_events,
            "target_rule": self.target_rule,
            "row_source": v2rollout.DAGGER_SOURCE,
        }


def config_from_manifest(manifest: dict, **overrides) -> DaggerConfig:
    """A `DaggerConfig` that agrees with the harvest whose rows these will be trained beside.

    Everything is read, nothing is defaulted silently: a manifest without δ_t/δ_r is a
    `ValueError`, because the step sizes every row is labelled against are functions of it, and a
    DAgger round labelled against the wrong δ describes a different policy than the one being
    corrected.
    """
    missing = [k for k in ("delta_t", "delta_r") if manifest.get(k) is None]
    if missing:
        raise ValueError(
            f"the harvest manifest carries no {', '.join(missing)}: it is what the composer scales "
            "a chosen axis by, so a DAgger round cannot be labelled or driven without it"
        )
    memory = manifest.get("memory") or {}
    splits: dict[int, str] = {}
    for split, tasks in (manifest.get("splits") or {}).items():
        for task in tasks:
            splits[int(task)] = split
    cfg = DaggerConfig(
        suite=str(manifest.get("suite", "libero_spatial")),
        delta_t=float(manifest["delta_t"]),
        delta_r=float(manifest["delta_r"]),
        chunk_steps=int(memory.get("chunk_steps", manifest.get("chunk_steps", CHUNK_STEPS))),
        fine_label_threshold=float(manifest.get("fine_label_threshold", 0.5)),
        move_floor=float(manifest.get("move_floor", 0.005)),
        yaw_floor=float(manifest.get("yaw_floor", 0.05)),
        horizon=int(memory.get("horizon") or manifest.get("horizon") or 220),
        every=int(memory.get("every", manifest.get("every", CHUNK_STEPS))),
        history_k=int(memory.get("history_k", 3)),
        max_events=int(memory.get("max_events", 8)),
        target_rule=str(memory.get("target_rule", manifest.get("target_rule", "obj_of_interest"))),
        splits=splits,
    )
    return dataclasses.replace(cfg, **overrides) if overrides else cfg


def read_manifest(directory) -> dict:
    """The manifest in `directory`, or a `ValueError` naming what to do about it.

    Unlike `harvest.read_manifest`, which returns None for a directory it is about to write, this
    one is a read of somebody else's finished work: a DAgger round without it would be labelled
    against guessed offsets and guessed splits, which is worse than not running.
    """
    path = pathlib.Path(directory) / MANIFEST
    if not path.is_file():
        raise ValueError(
            f"no {MANIFEST} under {directory}. A DAgger round reads the δ pair, the label "
            "thresholds, the memory settings, the per-task splits and the measured grasp offsets "
            "out of the harvest manifest; harvest first (`robojev harvest robojev`)."
        )
    found = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(found, dict):
        raise ValueError(f"{path} is not a JSON object")
    return found


# --------------------------------------------------------------------------------------------
# rows


def episode_rng(seed: int, task_index: int, init_state: int) -> np.random.Generator:
    """A generator that depends on the episode, not on the run.

    `harvest._row_rng`'s rule one level up: a blake2b digest of the three numbers, so collecting
    task 3 alone draws the same coin flips it would have drawn beside tasks 0-7. `hash()` is
    salted per process for strings and cannot be used.
    """
    key = f"{seed}:{task_index}:{init_state}".encode()
    return np.random.default_rng(int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big"))


def row_id(round_index: int, task_index: int, init_state: int, step: int) -> str:
    """`dagger:<round>:<task>:<init>:<t>` -- the id scheme, in one place.

    Deliberately unlike the harvest's `<suite>:<task>:<demo>:<step>`: the two kinds of row share a
    file after `merge`, and the trainer refuses a `state_id` that appears twice
    (`train_pipeline_decisions.py:168-173`), so they must not be able to collide even for a task
    and step they share.
    """
    return f"{SOURCE}:{round_index}:{task_index}:{init_state}:{step}"


def source_group_id(round_index: int, task_index: int, init_state: int) -> str:
    """One episode is one source group: every row of it lands in one split, the way one
    demonstration does in a harvest."""
    return f"{SOURCE}:{round_index}:{task_index}:{init_state}"


def target_for(env, objects: dict, proprio, instruction: str, cfg: DaggerConfig) -> dict:
    """Which object to pick up and which to place it on, by the harvest's configured rule.

    `obj_of_interest` reads the task definition through the simulator (`roles.grip_target`), which
    a rollout *does* have -- the env is right there -- and `scene_roles` is the rule a server can
    apply from the instruction and the poses alone. Which one names the memory's `Target:` line is
    the harvest's `target_rule`, because these rows are labelled to sit beside its rows; the other
    is computed anyway and the agreement recorded, exactly as the harvest records it.
    """
    roles = roles_mod.scene_roles(objects, instruction, np.asarray(proprio)[0:3])
    try:
        bddl_target, target_source = roles_mod.grip_target(env, objects, np.asarray(proprio)[0:3])
    except (AttributeError, ValueError):
        # No BDDL to read (a fake env, a backend that is not LIBERO): the rule a server can apply
        # is the only one there is, and the row says so rather than claiming a source it has not
        # got.
        bddl_target, target_source = roles["target"], "scene_roles"
    if cfg.target_rule == "obj_of_interest":
        target = bddl_target
    else:
        target = roles["target"]
    destination = _destination(env, objects, target, roles)
    return {"target": target, "target_source": target_source, "destination": destination,
            "agrees": bool(roles["target"] == bddl_target), "rule": roles["rule"]}


def _destination(env, objects: dict, target: str | None, roles: dict) -> str | None:
    """`roles.destination_for`, guarded for a backend with no task definition on it."""
    try:
        return roles_mod.destination_for(env, objects, target, roles)
    except (AttributeError, ValueError):
        found = roles.get("destination")
        return found if found != target else None


# --------------------------------------------------------------------------------------------
# the rollout


class _Recorder:
    """The policy, with a tap on it.

    Wrapped rather than reimplemented: `robojev.episode.run_episode` drives this exactly as
    A graded run drives the same policy object, so the states these rows describe are the states a
    graded evaluation of the same checkpoint would visit -- same wait steps, same chunking, same
    stop-on-success. All this adds is, at each `act`:

    * read the state text the policy was actually given, out of `decisions.meta.state`;
    * hand the policy's own answers to a memory of our own, so `Subgoal` (and therefore the
      waypoint and the latch) follows the rollout rather than a demonstration.

    `wants_privileged` is True unconditionally: the labels need object poses, and a DAgger round
    of a policy that did not want them would otherwise have nothing to label against.
    """

    wants_privileged = True

    def __init__(self, inner, env, cfg: DaggerConfig, task_index: int, init_state: int,
                 split: str, on_error=None):
        self.inner = inner
        self.env = env
        self.cfg = cfg
        self.task_index = task_index
        self.init_state = init_state
        self.split = split
        self.instruction = ""
        self.rows: list[dict] = []
        self.skipped: list[tuple[int, str]] = []
        self._step = 0
        self._last_decisions: dict | None = None
        self._on_error = on_error
        # No row is built here: the visited states are collected and relabelled in one pass at
        # the end of the episode, because the label source is a *tracker* that has to be driven
        # forward decision by decision and the cleanest place to drive it is `rollout.relabel` --
        # the same function the harvest's own rows come from.
        self.trace: list[dict] = []
        self.privileged: list[dict] = []
        self.roles: dict | None = None

    # -- the Policy surface ---------------------------------------------------------------
    def reset(self, instruction: str) -> None:
        self.instruction = instruction
        self._step = 0
        self.inner.reset(instruction)

    def describe(self):
        return self.inner.describe()

    @property
    def last_decisions(self) -> dict | None:
        return self._last_decisions

    def close(self) -> None:  # pragma: no cover -- the collector closes the inner policy itself
        close = getattr(self.inner, "close", None)
        if close is not None:
            close()

    def act(self, obs: dict, *args, **kwargs):
        chunk, decisions = self.inner.act(obs, *args, **kwargs)
        chunk = np.asarray(chunk, dtype=np.float32)
        self._last_decisions = decisions
        step = self._step
        self._step += 1
        try:
            self._record(obs, decisions, step)
        except Exception as exc:  # a decision point that cannot be labelled is dropped, not fatal
            self.skipped.append((step, repr(exc)))
            if self._on_error is not None:
                self._on_error(step, exc)
        return chunk, decisions

    # -- the tap --------------------------------------------------------------------------
    def _record(self, obs: dict, decisions: dict | None, step: int):
        privileged = obs.get("privileged")
        if not privileged:
            raise ValueError(
                "the backend supplied no privileged scene state, so there are no object poses to "
                "measure a waypoint against: a DAgger round needs a simulator that has one"
            )
        proprio = np.asarray(self.env.state_vector(obs), dtype=np.float32)
        names = target_for(self.env, privileged, proprio, self.instruction, self.cfg)
        # The whole decision point, kept as observed and **as served**. Nothing is labelled yet:
        # the label at a state is a function of the tracker's plan at that state, and the
        # tracker is replayed once, over the episode, in `relabel_states`. What is kept
        # here is everything that replay needs.
        #
        # The clock is the harvest's, not the simulator's: decision index times the harvest's own
        # `every`, so the state's `Decision k of N` line counts the same thing a training row
        # counts. `run_episode`'s own `t` is offset by `protocol.wait_steps`, which no training
        # row has ever seen.
        chosen = self._chosen(decisions)
        meta = (decisions or {}).get("meta") or {}
        grounding = meta.get("grounding")
        commit = None
        if grounding:
            # The server ran its grounding forward on this decision and committed a pair
            # (`server._ground`). Replayed, never re-decided: if the model named the wrong
            # bowl, the correction for that is a corrected `target` answer, and a motor label
            # quietly computed about a different bowl would be a label of a state nobody read.
            commit = {"target": grounding.get("target"),
                      "destination": grounding.get("destination"),
                      "source": grounding.get("source") or "model"}
        elif not self.trace:
            # A policy that never grounds (a fake, a baseline) still has to name a pair, and
            # the rule is the only thing here that can.
            commit = {"target": names["target"], "destination": names["destination"],
                      "source": names["target_source"]}
        if commit is not None:
            self.roles = dict(commit)
        latch = meta.get("grip_latch") or None
        self.trace.append({
            "step": step * self.cfg.every,
            "decision": step,
            "proprio": proprio,
            # The prompt the policy was served, byte for byte. `relabel` asserts its own
            # rebuild against it before it labels anything.
            "state": meta.get("state") or None,
            # The **executed** answers -- after the serving latch guard -- which is the dict
            # the server hands its own tracker and therefore what its `Last 3` block prints.
            "answers": chosen or {},
            "commit": commit,
            # A refusal by the guard is written into the server's tracker as an event, so it
            # has to be written into the replayed one too or every state after it differs.
            "latch": (None if not latch
                      else {"asked": bool(latch.get("asked")),
                            "refused": bool(latch.get("refused"))}),
        })
        self.privileged.append(privileged)

    @staticmethod
    def _chosen(decisions: dict | None) -> dict | None:
        """`{qid: candidate}` out of the the design notes decision block, or None."""
        if not decisions:
            return None
        found = {qid: entry["choice"] for qid, entry in decisions.items()
                 if qid != "meta" and isinstance(entry, dict) and "choice" in entry}
        return found or None


@dataclasses.dataclass
class EpisodeRows:
    """One rolled-out episode: its rows, its outcome, and whatever could not be labelled."""

    task_index: int
    init_state: int
    rows: list[dict]
    success: bool
    steps: int
    terminated_by: str
    error: str | None
    first_success_step: int | None
    skipped: list[tuple[int, str]]

    def outcome(self) -> dict:
        return {"success": bool(self.success), "steps": int(self.steps),
                "terminated_by": self.terminated_by, "error": self.error,
                "first_success_step": self.first_success_step}


def run_one(policy, env, protocol, cfg: DaggerConfig, task_index: int, init_state: int,
            split: str, *, execute_steps: int, run) -> EpisodeRows:
    """One episode, through the caller's episode loop, with every decision point recorded.

    `run(env, policy, protocol, init_state_index=…, seed=…, execute_steps=…)` is **the host's own
    evaluation loop**, and it is required rather than reimplemented here. That is the whole claim
    of the round: the states these rows describe are the states a graded evaluation of the same
    checkpoint would visit -- same wait steps, same chunking, same stop-on-success -- and a loop
    written here to avoid an argument would be a second definition of "an episode".

    The outcome is stamped onto every row of the episode *after* the loop ends, because it is not
    known while the rows are being made: a row says whether the episode it came from succeeded,
    which is how a later round can weight or filter them.
    """
    recorder = _Recorder(policy, env, cfg, task_index, init_state, split)
    result = run(env, recorder, protocol, init_state_index=init_state, seed=cfg.seed,
                 execute_steps=execute_steps)
    outcome = {"success": bool(result.success), "steps": int(result.steps),
               "terminated_by": result.terminated_by, "error": result.error,
               "first_success_step": result.first_success_step}
    recorder.rows = relabel_states(
        recorder.trace, recorder.instruction, recorder.privileged, cfg=cfg,
        task_index=task_index, init_state=init_state, split=split, roles=recorder.roles)
    for row in recorder.rows:
        row["metadata"]["outcome"] = dict(outcome)
    return EpisodeRows(task_index=task_index, init_state=init_state, rows=recorder.rows,
                       skipped=recorder.skipped, **outcome)


def relabel_states(trace, instruction: str, privileged_seq, *, cfg: DaggerConfig,
                   task_index: int, init_state: int, split: str,
                   roles: dict | None = None) -> list[dict]:
    """The DAgger labeller: `TrackerV2.gold_answers` on the model's **own** visited states.

    `robojev.rollout.relabel` does the work, and that is the point rather than an
    implementation detail: the rows this returns come out of exactly the function the harvest's
    own rows come out of, so a merged directory holds one label definition. What differs from a
    harvested row is the *state* -- it is the prompt the policy was actually given, and the
    history block in it carries the policy's own answers (`HISTORY_SOURCES = "policy"`, the measurements:
    the only source under which the block is the distribution that actually shifted).

    The round is **pure on-policy** (DAgger's β = 0, the measurements): every executed chunk is the
    policy's own. There is no mixing parameter.
    """
    from robojev import rollout                                 # noqa: PLC0415

    if not trace:
        return []
    rcfg = rollout.RolloutConfig(suite=cfg.suite, delta_t=cfg.delta_t, delta_r=cfg.delta_r,
                                 max_steps=cfg.horizon * cfg.every)
    return rollout.relabel(trace, instruction, privileged_seq, cfg=rcfg, roles=roles,
                           suite=cfg.suite, task_index=task_index, init_state=init_state,
                           episode=cfg.round, split=split)


# --------------------------------------------------------------------------------------------
# the round


def collect(policy, suite: str, tasks, init_states, out_dir, *,
            seed: int = 17, harvest_dir=None, manifest: dict | None = None,
            cfg: DaggerConfig | None = None, round: int = 1, env_factory=None,
            protocol=None, execute_steps: int | None = None, run=None,
            log=log_to_stderr) -> dict:
    """Roll `policy` out over `tasks` x `init_states` and write the visited states as rows.

    `policy` is whatever a graded evaluation drives -- a remote policy behind the policy-server
    protocol, or a fake in a test. It is driven, not owned: the caller opened it and the caller
    closes it.

    Three things come from the host and none of them has a default invented here: `env_factory`
    (`(suite, task_index) -> DecisionEnv`, or the configured adapter -- see `decision.env`),
    `protocol` (the wait steps, the horizon and the chunk the rollout runs under) and `run` (the
    host's own episode loop). The round's claim is that its states are the states a graded run of
    the same checkpoint visits, so all three are the host's or the claim is not true.

    The configuration comes from the harvest these rows will be trained beside: pass
    `harvest_dir`, or a `manifest` already read, or a `cfg` built by `config_from_manifest`. There
    is no fourth option, because every number a row is labelled with -- δ, the thresholds, the
    tracker settings, the per-task split -- belongs to that harvest, and a DAgger round labelled
    against defaults is a second dataset rather than an addition to the first.

    One env at a time, closed before the next task's is built: a ten-task round is a couple of
    hours of CPU simulation and the box it runs on is the box the live worker is using.

    Returns (and writes) the round's manifest.
    """
    if cfg is None:
        if manifest is None:
            if harvest_dir is None:
                raise ValueError(
                    "collect needs the harvest these rows will join: pass harvest_dir, manifest "
                    "or cfg (see config_from_manifest)"
                )
            manifest = read_manifest(harvest_dir)
        cfg = config_from_manifest(manifest, suite=suite, seed=int(seed), round=int(round))
    tasks = [int(t) for t in tasks]
    init_states = [int(i) for i in init_states]
    if not tasks or not init_states:
        raise ValueError("a DAgger round needs at least one task and one init state")
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_factory = env_mod.factory_or_default(env_factory)
    if protocol is None:
        raise ValueError(
            "collect needs the evaluation protocol the rollout runs under (wait steps, the "
            "horizon, the chunk): it is what makes these states the states a graded run of the "
            "same checkpoint visits, and a default invented here would be a second definition "
            "of an episode"
        )
    if run is None:
        raise ValueError(
            "collect needs the episode loop the rollout runs under: the states these rows "
            "describe have to be the states a graded run of the same checkpoint visits, and a "
            "loop written here would be a second definition of an episode"
        )
    if execute_steps is None:
        execute_steps = int(getattr(policy, "execute_steps", cfg.every))

    per_task: dict[str, dict] = {}
    episodes: list[dict] = []
    for task_index in tasks:
        split = cfg.splits.get(task_index, "train")
        record = {"rows": 0, "split": split, "episodes": 0, "successes": 0, "skipped": 0,
                  "translate": {}, "rotate": {}, "magnitude": {}, "grip": {},
                  "subgoal": {}, "waypoint": {}, "state_source": {},
                  "terminated_by": {},
                  # How often each question was asked, and how often the model's **executed**
                  # answer differed from the label of that same state. Their ratio
                  # is the round's own yield: the per-question error rate of the policy being
                  # corrected, on the states it actually visits.
                  "asked": {}, "disagrees": {},
                  "episode_records": []}
        env = env_factory(cfg.suite, task_index)
        try:
            path = out_dir / f"task_{task_index}.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                for init_state in init_states:
                    episode = run_one(policy, env, protocol, cfg, task_index, init_state, split,
                                      execute_steps=execute_steps, run=run)
                    for row in episode.rows:
                        fh.write(json.dumps(row, sort_keys=True) + "\n")
                        _count(record, row)
                    record["rows"] += len(episode.rows)
                    record["episodes"] += 1
                    record["successes"] += int(episode.success)
                    record["skipped"] += len(episode.skipped)
                    _bump(record["terminated_by"], episode.terminated_by)
                    outcome = {"task": task_index, "init_state": init_state,
                               "split": split, "rows": len(episode.rows), **episode.outcome()}
                    episodes.append(outcome)
                    record["episode_records"].append(outcome)
                    log(f"task {task_index} init {init_state}: {len(episode.rows)} rows, "
                        f"{'success' if episode.success else episode.terminated_by}")
            per_task[str(task_index)] = record
        finally:
            close = getattr(env, "close", None)
            if close is not None:
                close()

    manifest_out = {
        "policy": "robojev",
        "source": SOURCE,
        **cfg.shape(),
        "round": cfg.round,
        "delta_t": cfg.delta_t,
        "delta_r": cfg.delta_r,
        "tasks": sorted(int(t) for t in per_task),
        "init_states": init_states,
        "splits": _splits_of(per_task),
        "rows": {"total": sum(r["rows"] for r in per_task.values()),
                 "by_split": _totals(per_task, lambda r: {r["split"]: r["rows"]}),
                 "by_task": {t: r["rows"] for t, r in sorted(per_task.items(),
                                                             key=lambda kv: int(kv[0]))}},
        "episodes": episodes,
        "success_rate": (sum(e["success"] for e in episodes) / len(episodes)) if episodes else 0.0,
        # The closed-loop number the round is actually run for, in the shape the harvest's
        # manifest reports it: per episode above, and per task here. A DAgger round whose
        # successes are not written down is a round whose only record is its stderr.
        "success": {
            "episodes": len(episodes),
            "successes": sum(int(e["success"]) for e in episodes),
            "rate": (sum(int(e["success"]) for e in episodes) / len(episodes)) if episodes else None,
            "by_task": {t: {"successes": r["successes"], "episodes": r["episodes"],
                            "split": r["split"]}
                        for t, r in sorted(per_task.items(), key=lambda kv: int(kv[0]))},
        },
        # Per question: how often it was asked and how often the policy's executed answer was not
        # the label. This is what the round measured.
        "disagreement": _disagreement(per_task),
        "translate": _totals(per_task, lambda r: r["translate"]),
        "rotate": _totals(per_task, lambda r: r["rotate"]),
        "magnitude": _totals(per_task, lambda r: r["magnitude"]),
        "grip": _totals(per_task, lambda r: r["grip"]),
        "subgoal": _totals(per_task, lambda r: r["subgoal"]),
        "waypoint": _totals(per_task, lambda r: r["waypoint"]),
        # Every row whose text is `rebuilt` rather than `policy` is a row that does **not** carry
        # the model's own memory, which is the one thing this round is for. A nonzero count here
        # means the policy reported no `decisions.meta.state` and the round measured nothing.
        "state_source": _totals(per_task, lambda r: r["state_source"]),
        "skipped": sum(r["skipped"] for r in per_task.values()),
        "terminated_by": _totals(per_task, lambda r: r["terminated_by"]),
        "per_task": per_task,
        "provenance": PROVENANCE,
        "created_at": _now(),
    }
    (out_dir / MANIFEST).write_text(json.dumps(manifest_out, indent=2, sort_keys=True) + "\n")
    return manifest_out


def _count(record: dict, row: dict) -> None:
    """Every answer's marginal, keyed off the row's own `gold` rather than off a fixed list of
    question ids: which questions a round answers is the question set's business, and a counter
    that indexed a name the set does not have would be a `KeyError` in the middle of an
    hour-long rollout."""
    for qid, value in row["gold"].items():
        _bump(record.setdefault(qid, {}),
              str(bool(value)).lower() if isinstance(value, bool) else str(value))
    meta = row["metadata"]
    for qid in row["questions"]:
        _bump(record["asked"], qid)
    for qid in meta.get("disagrees", ()):
        _bump(record["disagrees"], qid)
    _bump(record["subgoal"], meta.get("memory_subgoal", "unknown"))
    if meta.get("waypoint") is not None:
        _bump(record["waypoint"], meta["waypoint"])
    _bump(record["state_source"], meta.get("state_source", "rebuilt"))


def _disagreement(per_task: dict) -> dict[str, dict]:
    """`{qid: {"asked", "differs", "rate"}}` over every row of the round."""
    asked = _totals(per_task, lambda r: r.get("asked", {}))
    differs = _totals(per_task, lambda r: r.get("disagrees", {}))
    return {qid: {"asked": n, "differs": differs.get(qid, 0),
                  "rate": (differs.get(qid, 0) / n) if n else None}
            for qid, n in sorted(asked.items())}


def _splits_of(per_task: dict) -> dict[str, list[int]]:
    splits: dict[str, list[int]] = {}
    for task, record in sorted(per_task.items(), key=lambda kv: int(kv[0])):
        splits.setdefault(record["split"], []).append(int(task))
    return splits


# --------------------------------------------------------------------------------------------
# merging


#: The `harvest.ROW_SHAPE_KEYS` a DAgger manifest can speak for. A merged directory has to agree
#: on every one of them or it holds two datasets under one manifest -- the harvest's own argument,
#: applied across the seam this module adds. The keys a DAgger round has no opinion about (the
#: grip rollout's, the demonstration's) are not compared, because it did not run one.
#:
MERGE_SHAPE_KEYS = ("suite", "every", "chunk_steps", "fine_label_threshold", "move_floor",
                    "yaw_floor", "label_source", "memory_rule", "horizon", "history_k",
                    "max_events", "target_rule",
                    # A round merged into a harvest of another question set answers different
                    # questions about a differently-rendered state; the one string that says so.
                    "questions_version")


def effective_shape(dagger_manifest: dict) -> dict:
    """A round's shape, with the labeller taken as the ground truth about what its rows are.

    A round whose manifest recorded a stale `memory_rule` or `label_source` for rows that are this
    labeller's -- the states came from `v2.state.TrackerV2` and the labels from
    `TrackerV2.gold_answers` whatever the manifest says -- is read by its `labeller`, which is
    decisive, so a round already on disk does not have to be re-rolled for an hour to be merged.
    """
    from robojev import rollout as v2rollout                   # noqa: PLC0415
    from robojev.state import MEMORY_RULE_V2                   # noqa: PLC0415

    if dagger_manifest.get("labeller") != LABELLER:
        return dict(dagger_manifest)
    corrected = {k: v for k, v in dagger_manifest.items() if k != "grip_label"}
    corrected.update(label_source=v2rollout.LABEL_SOURCE, memory_rule=MEMORY_RULE_V2,
                     questions_version="v2", row_source=v2rollout.DAGGER_SOURCE)
    return corrected


def merge(harvest_dir, dagger_dir, out_dir) -> dict:
    """A training directory holding both sources, with the harvest's rows byte for byte.

    The layout is the harvest's, unchanged: one `task_<i>.jsonl` per task plus a manifest, which
    is what `train.merge_rows` globs and what its cross-split check reads. A task's DAgger rows
    are **appended after** its harvested ones in the same file, so every harvested line is at the
    same offset with the same bytes it had -- the file is a prefix-preserving extension of itself,
    and `test_the_harvest_rows_survive_the_merge_byte_for_byte` pins exactly that.

    The DAgger rows inherit their task's split from the harvest manifest at collection time, so a
    merge is only ever confirming it; a disagreement is refused rather than silently resolved,
    because a `source_group_id` that crosses splits is a hard error in the trainer and a much
    worse error to debug there.

    Returns (and writes) the merged manifest: the harvest's, plus `sources` -- the row counts of
    each -- and the DAgger round's own manifest under `dagger`.
    """
    harvest_dir = pathlib.Path(harvest_dir)
    # One round or several: a second DAgger round is more of the same rows, and a training
    # directory that could hold only one of them would make "train on every round so far" a
    # manual concatenation nobody's manifest describes.
    dagger_dirs = ([pathlib.Path(dagger_dir)] if isinstance(dagger_dir, (str, pathlib.Path))
                   else [pathlib.Path(d) for d in dagger_dir])
    if not dagger_dirs:
        raise ValueError("merge needs at least one DAgger round directory")
    out_dir = pathlib.Path(out_dir)
    if out_dir.resolve() in {harvest_dir.resolve()} | {d.resolve() for d in dagger_dirs}:
        raise ValueError(
            f"merge writes a third directory: {out_dir} is one of its inputs, and merging in "
            "place would rewrite the harvest whose rows are supposed to survive untouched"
        )
    harvest_manifest = read_manifest(harvest_dir)
    dagger_manifests = [read_manifest(d) for d in dagger_dirs]
    for manifest in dagger_manifests:
        _check_shape(harvest_manifest, effective_shape(manifest))
    rounds = [m.get("round") for m in dagger_manifests]
    if len(set(rounds)) != len(rounds):
        raise ValueError(
            f"two of {[str(d) for d in dagger_dirs]} report the same --round ({rounds}): their "
            "row ids collide, and a merged directory with two rows claiming one id is a "
            "directory the trainer reads twice and counts once"
        )
    dagger_manifest = dagger_manifests[0]

    harvest_splits = {int(t): s for s, tasks in (harvest_manifest.get("splits") or {}).items()
                      for t in tasks}
    out_dir.mkdir(parents=True, exist_ok=True)

    harvest_files = {_task_of(p): p for p in sorted(harvest_dir.glob("task_*.jsonl"))
                     if _task_of(p) is not None}
    dagger_files: dict[int, list[pathlib.Path]] = {}
    for directory in dagger_dirs:
        for path in sorted(directory.glob("task_*.jsonl")):
            if _task_of(path) is not None:
                dagger_files.setdefault(_task_of(path), []).append(path)
    if not harvest_files and not dagger_files:
        raise ValueError(
            f"no task_*.jsonl under {harvest_dir} or "
            f"{', '.join(str(d) for d in dagger_dirs)}: nothing to merge")

    counts = {HARVEST_SOURCE: 0, SOURCE: 0}
    by_task: dict[str, dict] = {}
    for task_index in sorted(set(harvest_files) | set(dagger_files)):
        target = out_dir / f"task_{task_index}.jsonl"
        harvested = (harvest_files[task_index].read_bytes() if task_index in harvest_files else b"")
        if harvested and not harvested.endswith(b"\n"):  # pragma: no cover -- a truncated harvest
            raise ValueError(f"{harvest_files[task_index]} does not end in a newline: refusing to "
                             "append DAgger rows to a half-written line")
        added = b""
        n_harvest = _count_lines(harvested)
        n_dagger = 0
        split = harvest_splits.get(task_index)
        if task_index in dagger_files:
            lines = []
            for path in dagger_files[task_index]:
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if split is not None and row["split"] != split:
                        raise ValueError(
                            f"DAgger row {row['id']} is split {row['split']!r} but task "
                            f"{task_index} is {split!r} in the harvest manifest: a source group "
                            "that crosses splits is refused by the trainer "
                            "(train_pipeline_decisions.py:168-173). Re-collect the round against "
                            "this harvest's manifest."
                        )
                    lines.append(line)
                    n_dagger += 1
            added = "".join(line + "\n" for line in lines).encode("utf-8")
            split = split if split is not None else _split_of(dagger_manifest, task_index)
        target.write_bytes(harvested + added)
        counts[HARVEST_SOURCE] += n_harvest
        counts[SOURCE] += n_dagger
        by_task[str(task_index)] = {"rows": n_harvest + n_dagger, "split": split or "train",
                                    "by_source": {HARVEST_SOURCE: n_harvest, SOURCE: n_dagger}}

    # The frames sidecar belongs to the harvest -- a DAgger round renders none -- and a training
    # directory without it is one the frame-reading half of a later vision variant cannot use.
    # The JPEG tree itself is **not** copied: it is gigabytes, and duplicating it per merge would
    # fill the box. The merged manifest's `frames.dir` therefore points back at the harvest's own
    # tree rather than at a directory beside the sidecar, so nothing is left indexing files that
    # are not where it says they are.
    sidecar = harvest_dir / dataset_mod.FRAMES
    if sidecar.is_file():
        (out_dir / dataset_mod.FRAMES).write_bytes(sidecar.read_bytes())

    # v2's BDDL grounding rows ride beside the task files rather than inside them -- their own
    # file, their own instruction-level split -- and `train._task_files` globs `task_*.jsonl`
    # **plus** `grounding.jsonl`. A merged directory without it would train the motor questions
    # on two sources and drop the grounding question entirely, which is the one the DAgger round
    # cannot correct: a rollout visits the states its own grounding chose.
    grounding = harvest_dir / train_mod.GROUNDING_ROWS
    n_grounding = 0
    if grounding.is_file():
        blob = grounding.read_bytes()
        (out_dir / train_mod.GROUNDING_ROWS).write_bytes(blob)
        n_grounding = _count_lines(blob)

    splits: dict[str, list[int]] = {}
    for task, record in sorted(by_task.items(), key=lambda kv: int(kv[0])):
        splits.setdefault(record["split"], []).append(int(task))
    merged = {
        **harvest_manifest,
        "tasks": sorted(int(t) for t in by_task),
        "splits": splits,
        "rows": {
            "total": counts[HARVEST_SOURCE] + counts[SOURCE] + n_grounding,
            "by_source": {**counts, GROUNDING_SOURCE: n_grounding},
            "by_split": _totals(by_task, lambda r: {r["split"]: r["rows"]}),
            "by_task": {t: r["rows"] for t, r in sorted(by_task.items(), key=lambda kv: int(kv[0]))},
        },
        "sources": {
            HARVEST_SOURCE: {"rows": counts[HARVEST_SOURCE], "dir": str(harvest_dir),
                             "manifest": harvest_manifest.get("created_at")},
            SOURCE: {"rows": counts[SOURCE],
                     "dirs": [str(d) for d in dagger_dirs],
                     "dir": str(dagger_dirs[0]),
                     "rounds": [m.get("round") for m in dagger_manifests],
                     "round": dagger_manifest.get("round"),
                     "labeller": dagger_manifest.get("labeller"),
                     "success_rate": dagger_manifest.get("success_rate"),
                     "success": [m.get("success") for m in dagger_manifests]},
            GROUNDING_SOURCE: {"rows": n_grounding, "file": train_mod.GROUNDING_ROWS,
                               "dir": str(harvest_dir),
                               "repeats": harvest_manifest.get("grounding_repeats")},
        },
        "per_task_by_source": by_task,
        "frames": {**(harvest_manifest.get("frames") or {}),
                   "sidecar": dataset_mod.FRAMES,
                   "dir": str(harvest_dir / dataset_mod.FRAMES_DIR)},
        "dagger": dagger_manifest,
        "dagger_rounds": dagger_manifests,
        "merged_at": _now(),
    }
    (out_dir / MANIFEST).write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    return merged


def _check_shape(harvest_manifest: dict, dagger_manifest: dict) -> None:
    """Refuse to merge rows built two different ways -- `harvest._check_row_shape`'s argument,
    across the seam between a harvest and a round collected against a different one."""
    clashes = {k: (harvest_manifest[k], dagger_manifest[k]) for k in MERGE_SHAPE_KEYS
               if k in harvest_manifest and k in dagger_manifest
               and harvest_manifest[k] != dagger_manifest[k]}
    if clashes:
        detail = "; ".join(f"{k}: harvest {a!r} vs dagger {b!r}" for k, (a, b) in sorted(clashes.items()))
        raise ValueError(
            f"these rows were built differently and cannot share a directory ({detail}). "
            "Collect the DAgger round against this harvest's manifest."
        )


def _task_of(path: pathlib.Path) -> int | None:
    try:
        return int(path.stem.split("_", 1)[1])
    except (IndexError, ValueError):  # pragma: no cover -- a file called task_x.jsonl
        return None


def _split_of(manifest: dict, task_index: int) -> str | None:
    for split, tasks in (manifest.get("splits") or {}).items():
        if task_index in [int(t) for t in tasks]:
            return split
    return None


def _count_lines(blob: bytes) -> int:
    return sum(1 for line in blob.decode("utf-8").splitlines() if line.strip())


def _bump(counter: dict, key: str, by: int = 1) -> None:
    counter[key] = counter.get(key, 0) + by


def _totals(records: dict, pick) -> dict:
    out: dict[str, int] = {}
    for record in records.values():
        for key, n in pick(record).items():
            _bump(out, key, n)
    return out


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def rows_of(directory) -> Iterator[dict]:
    """Every row under `directory`, in task order -- the reader a report or a relabelling wants.

    `train.merge_rows`'s ordering rule, restated: `task_10` sorts between `task_1` and `task_2` as
    a string, and a file concatenated in an order its manifest does not describe is one nobody can
    reproduce.
    """
    directory = pathlib.Path(directory)
    files = [(t, p) for p in directory.glob("task_*.jsonl") if (t := _task_of(p)) is not None]
    for _, path in sorted(files):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)
