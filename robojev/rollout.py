"""RoboJEV v2's training rows, harvested from the scripted expert's **own rollouts**.

Why this module exists, in one sentence: v1's labels were what a teleoperator did next, which is
not a function of the state the model reads (failure F2), and the latest NanoJev note measured the
whole of its Basic gap closing when the label source became *one authoritative controller action
per decision*. `robojev.expert` is that controller; this is the harvester that runs it and
writes down what it said.

Three things about the way it runs are deliberate, and each is a measured decision rather than a
style choice.

**The rows are what the expert would do; the arm does something slightly else.** Every motion
answer is corrupted with probability ε (`RolloutConfig.epsilon`) *after* the gold is recorded, so
the state distribution the model trains on contains the states a ~90 %-accurate policy actually
wanders into -- upstream's Basic keeps every visited state, exploration included. `grip` is exempt
(the design notes, the measurements): the latch is the one answer the closed loop cannot absorb an error in
(19/20 -> 7/20 at p = 0.10 with it corrupted, 11-13/20 with it exempt), and an exploring
controller plus an irreversible action is most of upstream's own Predict-Position gap.

**No outcome filter.** A failed episode is kept in full. The one filter is upstream's ammo
analogue: after the release there is no grasp to decide about, so a `retreat` row carries every
question *except* `grip`.

**The gold is read off the tracker, never off the controller.** `expert.gold_for_state` returns
`TrackerV2.gold_answers()` -- the label function applied to the numbers the state string prints --
and raises when the tracker's waypoint and the expert's own differ by more than a tolerance that
could change an answer. So gate G3's invariant ("the state text alone reproduces the label") is
asserted at the moment a row is written, on every row, rather than discovered weeks later as a
97 % parser. `parse_agreement` re-reads what was written and reports it per question.

The grounding rows of `robojev.grounding` join the same directory as their own rows,
with `metadata.source = "grounding"` and their own **instruction-level** split: a held-out task
and a held-out instruction are two different held-out things, and concatenating a camera-frame
grounding state with a motion state would make one row that is neither.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
from importlib import import_module
from typing import Any, Iterator

import numpy as np

from robojev import dataset as dataset_mod
from robojev import envs as env_mod
from robojev import expert as expert_mod
from robojev import home
from robojev import roles as roles_mod
from robojev.dataset import log_to_stderr
from robojev import grounding as grounding_mod
from robojev import questions as questions_mod
from robojev import state as state_mod

#: `import_module`, not `from robojev import parse`: the package's `__init__`
#: re-exports a *function* called `parse`, so the plain form hands back the function and every
#: attribute lookup on it raises. `compose` has the same trap.
parse_mod = import_module("robojev.parse")
from robojev.compose import MEASURED_CM_PER_UNIT, MEASURED_STEPS

#: `metadata.source` of a row written from an expert rollout, and the value a directory's manifest
#: carries. A directory holds **one** source: `--source demos` rows and `--source expert` rows
#: answer different questions about different states and merging them into one file would train
#: one model on two definitions of a label (the same rule `_check_row_shape` below enforces for
#: the row shape).
SOURCE: str = "expert_rollout"
#: `metadata.source` of a BDDL grounding row riding in the same directory (Task 4's rows).
GROUNDING_SOURCE: str = "grounding"
#: And of a DAgger row relabelled by this expert on the model's own visited states.
DAGGER_SOURCE: str = "dagger_expert"

#: What `--source` may be on `robojev harvest robojev`.
SOURCES: tuple[str, ...] = ("demos", "expert")

#: Every gold here is a controller's reference answer, not a sampled outcome: the kind the
#: trainer's `paired_brier_pg` must *not* be fed. `grip` included -- it is a latch, a decision the
#: expert makes, which is exactly what the design notes's principle 3 asks for.
LABEL_KIND: str = "reference_argmax_compatibility"

#: The mass the **soft** target puts on the chosen candidate, for the comparison arm only
#: (`RolloutConfig.soft_targets`). NanoJev's own declared-actuator-noise level, unchanged from the
#: number every earlier run softened its labels with.
SOFT_RHO: float = 0.85

#: **Hard** one-hot targets, not `expert.probabilities`' declared-noise softening. Note §3 and
#: §6.1: the hard arm won action agreement 81.6 % against the soft arm's 77.1 % on upstream's own
#: comparison. The soft arm stays reachable through `RolloutConfig.soft_targets` so the two can be
#: measured on identical states, and it is not the default.
PROBS_KIND: str = dataset_mod.PROBS_KIND

PROVENANCE: str = dataset_mod.PROVENANCE

MANIFEST: str = dataset_mod.MANIFEST

#: One JSONL per task, exactly as `robojev.harvest` lays a directory out, so a re-harvest
#: of one task rewrites one file and `--jobs N` is N processes that never touch the same path.
def task_file(task_index: int) -> str:
    return f"task_{task_index}.jsonl"


#: Where a task's own record of the run lands while the harvest is still going. The manifest is
#: assembled from these at the end, which is what makes the parallel harvest a merge rather than
#: N processes fighting over one JSON file.
PARTS_DIR: str = "parts"

#: The grounding rows' file, kept separate from the task files for the same reason their split is
#: separate: they are a different dataset sharing a directory.
GROUNDING_FILE: str = "grounding.jsonl"


#: The policy directory v2's rows live under. **Not** `robojev`: that directory holds the
#: demonstrations' v1 rows, which the deployed checkpoints were trained on and which nothing here
#: may overwrite. v2 answers different questions about differently-rendered states, so it is a
#: different dataset and gets a different root.
POLICY_DIR: str = "robojev-v2"


def out_root(suite: str, policy: str = POLICY_DIR) -> pathlib.Path:
    """`$ROBOJEV_HOME/data/robojev-v2/<suite>` -- where an expert harvest lands by default."""
    return home.home() / "data" / policy / suite


class RolloutError(RuntimeError):
    """A row could not be written the way this module promises it writes rows."""


# --------------------------------------------------------------------------------------------
# configuration


@dataclasses.dataclass
class RolloutConfig:
    """Everything that changes a byte or a number of a harvested row, in one object.

    The manifest echoes all of it, because Task 7's server reads `questions_version`,
    `cm_per_unit` and the tracker settings back out of the checkpoint's `robojev.json` and refuses
    to launch against a mismatch. A knob that is not here is a way for a checkpoint and a server
    to disagree silently.
    """

    suite: str = "libero_spatial"
    #: The tasks this run harvests.
    tasks: tuple[int, ...] = tuple(range(10))
    #: The task list the **split** is resolved over, which is not the same thing: `--tasks 0,9`
    #: must still put task 9 in `test`, and a split derived from the subset would put task 0 in
    #: `dev`. Echoed in the manifest so a later partial run resolves the same way.
    suite_tasks: tuple[int, ...] = tuple(range(10))
    #: LIBERO init-state indices, cycled over `episodes_per_task`.
    inits: tuple[int, ...] = tuple(range(10))
    episodes_per_task: int = 20
    #: Per-question corruption of the **executed** motion answers. Never of a recorded gold.
    epsilon: float = 0.10
    #: Spec §5: no noise on the irreversible answer -- **and none on the wrist**, which is a
    #: different reason for the same rule. A wrong `move_*` mis-steps the hand and the next
    #: decision re-reads the waypoint and corrects it; a wrong `yaw` turns the wrist, and the rim
    #: direction *is* the wrist's closing axis rotated by the candidate's turn, so the waypoint
    #: itself moves and the hand chases it round the object. Measured, and it is not a small
    #: effect: at ε = 0.10 with `yaw` corrupted a 2-episode-per-task proof harvest of tasks 0, 4
    #: and 9 scored **0/6**, against the 18/20 on task 0 that the deployed checkpoint's own
    #: harvest recorded at the same ε (docs/DESIGN.md §4C).
    #:
    #: **And none on `rim`, for a third reason: it is not a step, it is a commitment.** A wrong
    #: `move_*` is undone by the next decision and a wrong `yaw` by the one after; a wrong `rim`
    #: sends the plan to a different rim point and the plan *obeys* it -- which is the point of
    #: the question -- so the cost is the whole approach and descent that rim point takes, and
    #: the drawer task has no such decisions to spare. Decided by the numbers rather than by the
    #: argument (`scratch/v2-drawer/round3/proof_sweep.py`, tasks 0, 4 and 9, 8 episodes each,
    #: ε = 0.10, seed 17, everything else identical):
    #:
    #: | `rim` | task 0 | task 4 | task 9 | all |
    #: |---|---|---|---|---|
    #: | corrupted | 8/8 | 2/8 | 4/8 | **14/24** |
    #: | exempt | 8/8 | 5/8 | 6/8 | **19/24** |
    #:
    #: Gate G3 is 1.000 on all ten questions either way, so this is about the state distribution
    #: the rows are drawn from and not about whether they can be read back.
    epsilon_exempt: tuple[str, ...] = ("grip", "yaw", "rim")
    annotate: bool = True
    questions_version: str = questions_mod.QUESTION_SET_VERSION
    #: `v2.compose.MEASURED_CM_PER_UNIT` -- what one unit of normalised δ_t actually executes.
    cm_per_unit: float = MEASURED_CM_PER_UNIT
    #: §7 a design ruling: a v2 checkpoint is served at δ_t = 1.0, so its labels are harvested there.
    delta_t: float = expert_mod.EXPERT_DELTA_T
    delta_r: float = expert_mod.EXPERT_DELTA_R
    grounding_rows: bool = True
    #: How many copies of every grounding scene to write (see `grounding_rows`). 1 -- the
    #: default -- is the un-augmented file every existing directory holds, byte for byte.
    grounding_repeats: int = 1
    seed: int = 17
    max_steps: int = 220
    #: Which questions are asked. Follows `v2.questions.ACTIVE_QIDS`, which is the set the
    #: heuristic server serves; `yaw` is defined but off on LIBERO-Spatial.
    qids: tuple[str, ...] = questions_mod.ACTIVE_QIDS
    #: The held-out pair, over `suite_tasks`. `None` means `dataset.resolve_splits`' rule: the
    #: last task is `test` and the one before it is `dev`.
    dev_task: int | None = None
    test_task: int | None = None
    #: Hard one-hot gold (the default, the measurements) or `expert.probabilities`' softening.
    soft_targets: bool = False
    #: Raise the moment `expert.gold_for_state` says the tracker's waypoint and the expert's
    #: disagree. On by default: a silent drop would turn a planner bug into a quiet 3 % of missing
    #: episodes.
    strict_gold: bool = True
    #: Raise the moment a written row's own state text does not re-parse to its own gold. Gate
    #: G3's invariant, asserted per row at write time.
    strict_parse: bool = True
    #: Where the BDDL grounding rows come from. The file Task 4 wrote, or -- if it is missing --
    #: rebuilt from the scene cache.
    grounding_path: str = str(grounding_mod.SCENE_CACHE.parent / "rows.jsonl")
    #: The scene cache the augmented rows are built from. **Settled** by default: a scene
    #: captured straight after `set_init_state` is not the scene a server grounds against, and on
    #: LIBERO-Spatial task 3 the ten no-op steps flip a relation word (`grounding.SETTLED_STEPS`).
    scenes_path: str = str(grounding_mod.SETTLED_SCENE_CACHE)

    def tracker_kwargs(self, instruction: str | None = None) -> dict:
        """The `TrackerV2` constructor arguments this configuration implies.

        `instruction` is passed through because the tracker steers by the expert's own sub-stage
        machine when it has one, and by `waypoint_for`'s plainer geometry when it does not. A
        harvest always has one: the labels of record are the plan's, and a tracker planning a
        different route from the expert would fail `gold_for_state`'s waypoint check on row one.
        """
        return {"horizon": self.horizon(), "qids": tuple(self.qids), "annotate": self.annotate,
                "instruction": instruction}

    def horizon(self) -> int:
        """Decisions in an episode, which is what the state's `Decision k of N` line prints."""
        return int(self.max_steps // expert_mod.CHUNK_STEPS)


def split_config(cfg: RolloutConfig) -> dataset_mod.Splits:
    """The held-out pair this configuration implies, for `dataset.split_for`.

    Built over `suite_tasks` and never over the subset being harvested, so a run of `--tasks 0,9`
    gives task 9 the same `test` it would have had in a full run.
    """
    return dataset_mod.resolve_splits(dataset_mod.Splits(
        tasks=tuple(cfg.suite_tasks), dev_task=cfg.dev_task, test_task=cfg.test_task))


def noise_exempt(cfg: RolloutConfig) -> tuple[str, ...]:
    """Which of the controller's answers ε leaves alone.

    `epsilon_exempt` (the design notes's `grip`, and `yaw` -- see the field) **plus every answer the
    question set does not ask**. ε is a model of the model: the state distribution a row is drawn
    from should be the one a policy with 1 - ε accuracy on *the questions it answers* produces,
    and corrupting a question nobody asks would move the arm in states no deployed policy can
    reach, so every row after that would describe a scene the model will never see.
    """
    asked = set(cfg.qids)
    return tuple(sorted(set(cfg.epsilon_exempt)
                        | {qid for qid in expert_mod.AXISWISE_CANDIDATES if qid not in asked}))


def episode_seed(cfg: RolloutConfig, task_index: int, episode: int) -> int:
    """The corruption stream of one episode, as a function of the episode and not of the run.

    `dataset.row_rng`'s rule: a blake2b digest of the three numbers, so harvesting task 3 alone
    draws the coins it would have drawn beside tasks 0-9 and `--jobs 8` is byte-identical to
    `--jobs 1`.
    """
    rng = dataset_mod.row_rng(cfg.seed, task_index, episode, 0)
    return int(rng.integers(0, 2 ** 62))


def row_id(suite: str, task_index: int, init_state: int, episode: int, step: int) -> str:
    return f"expert:{suite}:{task_index}:{init_state}:{episode}:{step}"


def dagger_row_id(source: str, suite: str, task_index: int, init_state: int, round_index: int,
                  step: int) -> str:
    """A relabelled row's id, which must not be able to collide with a harvested one.

    `expert:libero_spatial:0:0:1:0` is a *harvest* row -- task 0, init 0, episode 1, decision 0 --
    and a DAgger round numbering its rounds from 1 would have produced exactly that string. The
    source leads the id instead, so a merged directory cannot have two rows claiming one id.
    """
    return f"{source}:{suite}:{task_index}:{init_state}:r{round_index}:{step}"


def source_group_id(suite: str, task_index: int, init_state: int, episode: int) -> str:
    """Every row of one episode shares a group, so the trainer's no-crossing rule holds over
    episodes and not merely over tasks."""
    return f"{suite}:{task_index}:{init_state}:{episode}"


# --------------------------------------------------------------------------------------------
# roles


def roles_for(env, objects: dict, eef, instruction: str) -> dict:
    """`{"target", "destination", "source"}`, pinned to the **task definition's** object.

    G1 was measured with the target pinned this way (85/100); the rule that guesses it from the
    sentence is 0/10 on two of the ten tasks, which is failure F3 measured and exactly what the
    `target` grounding question exists to fix. A label source may read the BDDL -- it knows which
    task it is labelling -- where a server may not.
    """
    target, source = roles_mod.grip_target(env, objects, eef)
    destination = None
    for name in roles_mod.obj_of_interest(env):
        if name in objects and name != target:
            destination = name
            break
    if destination is None:
        found = expert_mod.resolve_roles(objects, instruction, eef).get("destination")
        destination = found if found != target else None
    return {"target": target, "destination": destination, "source": source}


# --------------------------------------------------------------------------------------------
# one row


def motion_row(*, cfg: RolloutConfig, suite: str, task_index: int, init_state: int, episode: int,
               step: int, split: str, state_text: str, gold: dict, tracker, meta: dict,
               executed: dict, roles: dict, decision_index: int,
               controller: dict | None = None, identifier: str | None = None) -> dict:
    """One NanoJev training row for one decision point of one expert rollout.

    Three answers to the same question are in play here and the row keeps all three apart,
    because conflating any two of them is how a dataset acquires a bug nobody can see:

    * **`gold`** -- the label of record, `TrackerV2.gold_answers()`: the label function applied to
      the numbers the state string prints. This is what the row is labelled with, and what the
      parser has to reproduce.
    * **`controller`** -- what `expert.answers_v2` itself answered, before any noise. It is the
      same function of the same geometry, computed in metres rather than in the state's rounded
      centimetres, so it agrees with `gold` on all but the band edges and the one decision either
      side of a stage change. `metadata.gold_differs` counts where it does not, per row, which is
      the honest measurement of that seam rather than an assumption about it.
    * **`executed`** -- what the arm was actually commanded: `controller` with ε applied.
      `metadata.noised` is exactly the set ε moved, which is what makes "ε never touches `grip`"
      a checkable claim rather than a promise.
    """
    qids = [qid for qid in cfg.qids if qid in gold]
    # Upstream's ammo filter, in this vocabulary: after the release there is no grasp to decide
    # about, so `grip` is not asked. Every other question still is -- the retreat is a real
    # decision and dropping the whole row would teach the policy that episodes end at the release.
    if tracker.subgoal == "retreat" and "grip" in qids:
        qids.remove("grip")
    questions = questions_mod.questions_block(qids)
    answers = {qid: gold[qid] for qid in qids}
    probs = {qid: (_soft(qid, value, questions[qid]) if cfg.soft_targets
                   else _one_hot(qid, value, questions[qid]))
             for qid, value in answers.items()}

    identifier = identifier or row_id(suite, task_index, init_state, episode, step)
    row = {
        "id": identifier,
        "state_id": identifier,
        "family_id": suite,
        "split": split,
        "state": state_text,
        "questions": questions,
        "gold": answers,
        "gold_probs": probs,
        "gold_probs_kind": {qid: PROBS_KIND for qid in qids},
        "gold_label_kind": {qid: LABEL_KIND for qid in qids},
        "metadata": {
            "source": SOURCE,
            "source_group_id": source_group_id(suite, task_index, init_state, episode),
            "suite": suite,
            "task_index": task_index,
            "init_state": init_state,
            "episode": episode,
            "decision": decision_index,
            "step": step,
            "questions_version": cfg.questions_version,
            "memory_rule": state_mod.MEMORY_RULE_V2,
            "annotate": cfg.annotate,
            "cm_per_unit": cfg.cm_per_unit,
            "delta_t": cfg.delta_t,
            "delta_r": cfg.delta_r,
            "epsilon": cfg.epsilon,
            "epsilon_exempt": list(cfg.epsilon_exempt),
            "target_object": roles["target"],
            "target_source": roles["source"],
            "destination_object": roles["destination"],
            "memory_subgoal": tracker.subgoal,
            "memory_target": roles["target"],
            "memory_destination": roles["destination"],
            "history_source": "executed",
            "label_source": "expert_tracker",
            "gold_source": meta.get("gold_source", "tracker"),
            "expert_phase": meta.get("phase"),
            "tracker_subgoal": meta.get("tracker_subgoal"),
            "tracker_offset_cm": meta.get("tracker_offset_cm"),
            "waypoint_cm": meta.get("waypoint_cm"),
            "attempts": meta.get("attempts"),
            "held": meta.get("held"),
            # What the arm was actually commanded, what the controller meant to command, and the
            # two disagreements: ε's, and the label function's with the controller it labels.
            "executed": {qid: _as_label(executed.get(qid)) for qid in executed},
            "controller": ({qid: _as_label(v) for qid, v in controller.items()}
                           if controller else None),
            "noised": sorted(qid for qid in (controller or {})
                             if qid in executed
                             and _as_label(executed[qid]) != _as_label(controller[qid])),
            # What was *executed* against what should have been. In a harvest this is ε plus the
            # seam below; in a DAgger round it is the model's own error, which is the round's
            # whole yield and the number a per-qid accuracy is read off.
            "disagrees": sorted(qid for qid in answers
                                if qid in executed
                                and _as_label(executed[qid]) != _as_label(answers[qid])),
            "gold_differs": sorted(qid for qid in answers
                                   if qid in (controller or {})
                                   and _as_label(controller[qid]) != _as_label(answers[qid])),
            "provenance": PROVENANCE,
        },
    }
    dataset_mod.check_probs(row)
    return row


def _as_label(value):
    """A boolean answer spelled the way its candidate set spells it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _candidates(question: dict) -> list[str]:
    """A question's candidate ids, whichever of the two shapes it is in."""
    return ["false", "true"] if question["type"] == "boolean" else list(question["criteria"])


def _soft(qid: str, value, question: dict) -> dict[str, float]:
    """NanoJev's declared-noise target: `SOFT_RHO` on the answer, the rest split evenly.

    The **comparison arm**, not the default -- `soft_targets`, which the measurements measured losing to
    the hard one-hot gold by 4.5 points of action agreement.
    """
    criteria = _candidates(question)
    chosen = _as_label(value) if question["type"] == "boolean" else value
    if chosen not in criteria:
        raise RolloutError(f"{qid}: the expert answered {value!r}, not one of {criteria}")
    spread = (1.0 - SOFT_RHO) / max(len(criteria) - 1, 1)
    return {candidate: (SOFT_RHO if candidate == chosen else spread) for candidate in criteria}


def _one_hot(qid: str, value, question: dict) -> dict[str, float]:
    """A hard target over exactly this question's candidates (the measurements's winning arm)."""
    if question["type"] == "boolean":
        chosen = _as_label(value)
        return {"false": 1.0 if chosen == "false" else 0.0,
                "true": 1.0 if chosen == "true" else 0.0}
    criteria = _candidates(question)
    if value not in criteria:
        raise RolloutError(f"{qid}: the expert answered {value!r}, which is not one of {criteria}")
    return {candidate: (1.0 if candidate == value else 0.0) for candidate in criteria}


# --------------------------------------------------------------------------------------------
# one episode


class _Episode:
    """The `on_decision` callback `expert.run_episode` writes a row through.

    The expert's phase carry is mirrored here rather than read out of the callback, because the
    callback is handed the decision *after* it was made and `gold_for_state` has to make the same
    one. `decide` is pure and total, so a phase advanced from the same start over the same states
    is the same phase, and `gold_for_state`'s own waypoint check is what says so out loud if it
    ever stops being true.
    """

    def __init__(self, cfg: RolloutConfig, env, task_index: int, init_state: int, episode: int,
                 split: str, roles: dict):
        self.cfg = cfg
        self.env = env
        self.task_index = task_index
        self.init_state = init_state
        self.episode = episode
        self.split = split
        self.roles = roles
        self.instruction = env.instruction
        self.tracker = state_mod.TrackerV2(
            target=roles["target"], destination=roles["destination"],
            **cfg.tracker_kwargs(self.instruction))
        self.tracker.commit(roles["target"], roles["destination"], step=0, source=roles["source"])
        self.phase = expert_mod.new_phase(roles)
        self.rows: list[dict] = []
        self.error: str | None = None

    def on_decision(self, index: int, choices: dict, meta: dict, state8, privileged: dict) -> None:
        step = index * expert_mod.CHUNK_STEPS
        self.tracker.observe(step=step, proprio=state8, objects=privileged)
        state_text = state_mod.serialise_v2(state8, privileged, self.instruction, self.tracker,
                                            annotate=self.cfg.annotate)
        # The controller's own answer, before the noise and before the label function. Computed
        # from the phase carry *as it stands*, which is the same carry `gold_for_state` is about
        # to read, so it is the answer `run_episode` itself made at this decision.
        controller, _nxt, _meta = expert_mod.answers_v2(
            state8, privileged, self.instruction, self.phase, delta_t=self.cfg.delta_t,
            roles=self.roles)
        gold, self.phase, gold_meta = expert_mod.gold_for_state(
            self.tracker, state8, privileged, self.instruction, self.phase,
            delta_t=self.cfg.delta_t, roles=self.roles)
        executed = {**choices, "grip": _as_label(choices.get("grip"))}
        row = motion_row(
            cfg=self.cfg, suite=self.cfg.suite, task_index=self.task_index,
            init_state=self.init_state, episode=self.episode, step=step, split=self.split,
            state_text=state_text, gold=gold, tracker=self.tracker, meta=gold_meta,
            executed=executed, roles=self.roles, decision_index=index, controller=controller)
        check_row(row, strict=self.cfg.strict_parse)
        self.rows.append(row)
        # The history block says what was **executed**, never what was merely answered: the effect
        # column beside it is measured from the next observed state, and an action column
        # describing a move that never happened is the one fiction the note rules out. Under ε
        # that is the corrupted answer, which is the whole point of harvesting this way.
        self.tracker.answer({**executed, "subgoal": gold.get("subgoal", self.tracker.subgoal)})
        # ...and the mirrored carry takes the executed `rim` letter exactly as the tracker's and
        # `run_episode`'s own do, so the three plans that have to agree about which rim point the
        # episode is standing on stay one plan.
        self.phase = expert_mod.with_executed(self.phase, executed)


def run_one(cfg: RolloutConfig, env, task_index: int, init_state: int, episode: int,
            split: str) -> dict:
    """One expert episode: its rows, whether it succeeded, and what it cost.

    The success flag is the closed-loop sanity number for the label policy **under ε-noise**: it
    is the expert scored at exactly the accuracy the model is being asked to reach, so a harvest
    whose episodes stop succeeding is a harvest whose ε is above what the vocabulary tolerates
    (G4's table), not merely a noisier file.
    """
    obs = env.reset(init_state)
    privileged = env.privileged(obs)
    state8 = np.asarray(env.state_vector(obs), dtype=np.float64).reshape(-1)
    roles = roles_for(env, privileged, state8[0:3], env.instruction)
    recorder = _Episode(cfg, env, task_index, init_state, episode, split, roles)
    outcome: dict[str, Any]
    try:
        outcome = expert_mod.run_episode(
            env, init_state, max_steps=cfg.max_steps, delta_t=cfg.delta_t, delta_r=cfg.delta_r,
            roles=roles, corruption=cfg.epsilon,
            corruption_exempt=noise_exempt(cfg),
            seed=episode_seed(cfg, task_index, episode), on_decision=recorder.on_decision)
    except expert_mod.ExpertError as exc:
        if cfg.strict_gold:
            raise
        return {"task_index": task_index, "init_state": init_state, "episode": episode,
                "split": split, "success": False, "steps": None, "decisions": len(recorder.rows),
                "rows": [], "error": repr(exc)}
    return {
        "task_index": task_index, "init_state": init_state, "episode": episode, "split": split,
        "success": bool(outcome["success"]), "steps": int(outcome["steps"]),
        "decisions": int(outcome["decisions"]), "attempts": int(outcome["attempts"]),
        "out_of_horizon": bool(outcome["out_of_horizon"]), "target": roles["target"],
        "rows": recorder.rows, "error": None,
    }


# --------------------------------------------------------------------------------------------
# gate G3's invariant, at write time


def check_row(row: dict, *, strict: bool = True) -> dict[str, bool]:
    """Re-parse a row's own state text and compare it to the row's own gold, per question.

    This is gate G3's contract asserted where the row is *made*: if a rule over the string cannot
    recover the label, neither can the model, and the honest moment to find out is now rather than
    after a cluster epoch. It is also the check that says whether a change to the state text is
    right -- a tracker whose waypoint stops matching the text it prints fails here, on row one.
    """
    agree = {}
    try:
        parsed = parse_mod.parse(row["state"], qids=tuple(row["gold"]))
    except parse_mod.UnparseableState as exc:
        if strict:
            raise RolloutError(
                f"{row['id']}: the state text this row carries cannot be parsed ({exc}). Gate G3 "
                f"says every label is a function of the text; a row whose text is unreadable is a "
                f"row the model cannot learn."
            ) from exc
        return {qid: False for qid in row["gold"]}
    for qid, gold in row["gold"].items():
        agree[qid] = _as_label(parsed.get(qid)) == _as_label(gold)
    wrong = sorted(qid for qid, ok in agree.items() if not ok)
    if wrong and strict:
        raise RolloutError(
            f"{row['id']}: the state text re-parses to a different answer for {wrong} "
            f"(parsed {[_as_label(parsed.get(q)) for q in wrong]}, gold "
            f"{[_as_label(row['gold'][q]) for q in wrong]}). Gate G3's invariant is that a row's "
            f"own text reproduces its own label; it does not."
        )
    return agree


def parse_agreement(rows) -> dict[str, dict]:
    """Per question: how many rows were re-parsed and how many re-parsed to their own gold.

    Reported at the end of a harvest and written into the manifest, because "the parser is at
    100 %" is a claim about *the rows that were written*, not about a fixture. Grounding rows have
    no waypoint and no motion label and are skipped rather than counted as failures.
    """
    counts: dict[str, dict] = {}
    for row in rows:
        if row.get("metadata", {}).get("source") == GROUNDING_SOURCE:
            continue
        agree = check_row(row, strict=False)
        for qid, ok in agree.items():
            record = counts.setdefault(qid, {"rows": 0, "agree": 0})
            record["rows"] += 1
            record["agree"] += int(ok)
    for record in counts.values():
        record["rate"] = (record["agree"] / record["rows"]) if record["rows"] else None
    return counts


def _is_v2_state(text: str) -> bool:
    """Can `v2.parse` read this string at all? The cheapest possible check that a prompt is v2's
    and not v1's, run before a v2 label is attached to it."""
    try:
        parse_mod.read(text)
    except parse_mod.UnparseableState:
        return False
    return True


def rows_of(path) -> Iterator[dict]:
    """Every row of a file or of a harvest directory, in the order they were written."""
    path = pathlib.Path(path)
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    for one in files:
        with one.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)


# --------------------------------------------------------------------------------------------
# the grounding rows


def grounding_rows(cfg: RolloutConfig) -> list[dict]:
    """The BDDL grounding rows, stamped with their own `metadata.source`.

    At `grounding_repeats == 1` (the default) this is exactly what it always was: the file Task 4
    wrote if it is there, rebuilt from the cached scenes if it is not, so an existing directory
    re-harvests byte for byte.

    **Above 1 it augments**, and the augmentation is two measured things at once.

    *Repetition with a different alphabet.* Run 6 saw 540 grounding rows once among 46 000
    questions and its held-out `target` accuracy is 0.78; on LIBERO-Spatial task 0 it names the
    wrong bowl at p = 0.957. `N` copies of every scene, each built with its own seed, means each
    copy numbers and orders the candidates differently (`assign_ids` shuffles inside a noun
    group), so the repetition teaches the sentence rather than the digit.

    *Both phrasings.* The scenes carry the BDDL's own `(:language …)` -- "Pick the akita black
    bowl next to the cookies box" -- and a server asks with the simulator's sentence -- "pick up
    the black bowl next to the cookie box". Different verb, article, noun and case for the same
    scene, and a head trained on one and asked the other is answering a question it never saw.
    The copies alternate between the two, so the total stays `N` per scene and half of them are
    the wording that is actually served.

    The split is the same for every copy of a scene and for **both** its phrasings
    (`grounding.scene_keys`): a sentence in train under one wording and in test under another is
    leakage wearing a paraphrase.
    """
    repeats = max(1, int(cfg.grounding_repeats))
    path = pathlib.Path(cfg.grounding_path).expanduser()
    if repeats == 1:
        if path.is_file():
            rows = list(rows_of(path))
        else:
            rows, _report = grounding_mod.build_rows(_scenes(cfg, path))
    else:
        scenes = _scenes(cfg, path)
        # One split map for the whole augmentation, drawn over groups of co-occurring keys so
        # both phrasings of a task are one held-out thing.
        splits = grounding_mod.split_map(scenes, key_of=grounding_mod.scene_keys)
        rows = []
        for index in range(repeats):
            phrasing = grounding_mod.PHRASINGS[index % len(grounding_mod.PHRASINGS)]
            built, _report = grounding_mod.build_rows(
                scenes, seed=grounding_mod.SPLIT_SEED + index, splits=splits,
                phrasing=phrasing, id_suffix=f":g{index}")
            rows.extend(built)
    out = []
    for row in rows:
        metadata = {k: v for k, v in row["metadata"].items() if not k.startswith("_")}
        metadata["source"] = GROUNDING_SOURCE
        metadata["grounding_repeats"] = repeats
        metadata.setdefault("provenance", PROVENANCE)
        out.append({**row, "metadata": metadata})
    return out


def _scenes(cfg: RolloutConfig, path: pathlib.Path) -> list[dict]:
    cache = pathlib.Path(cfg.scenes_path).expanduser()
    scenes = grounding_mod.load_scenes(cache)
    if not scenes:
        raise RolloutError(
            f"no grounding rows at {path} and no scene cache at {cache}: run "
            f"`python -m robojev.grounding scenes --cache {cache}`, or harvest with "
            f"--no-grounding-rows"
        )
    unsettled = sorted({int(s.get("settled_steps", 0)) for s in scenes})
    if unsettled != [grounding_mod.SETTLED_STEPS]:
        raise RolloutError(
            f"{cache} holds scenes captured after {unsettled} no-op steps and a server grounds "
            f"after {grounding_mod.SETTLED_STEPS} (`robojev.episode.run_episode`'s wait steps). "
            f"Rebuild it: `python -m robojev.grounding scenes --cache {cache} "
            f"--jobs 10`."
        )
    return scenes


# --------------------------------------------------------------------------------------------
# the harvest


def _record() -> dict:
    return {"rows": 0, "episodes": 0, "successes": 0, "split": None, "labels": {},
            "questions": {}, "subgoal": {}, "noised": {}, "gold_differs": {}, "decisions": 0,
            "errors": {}, "episode_records": []}


def harvest_task(cfg: RolloutConfig, env, task_index: int, out_dir, log=log_to_stderr) -> dict:
    """Every episode of one task, written to `<out_dir>/task_<i>.jsonl`. Returns the record."""
    out_dir = pathlib.Path(out_dir)
    splits = split_config(cfg)
    split = dataset_mod.split_for(task_index, splits)
    record = _record()
    record["split"] = split
    path = out_dir / task_file(task_index)
    with path.open("w", encoding="utf-8") as handle:
        for episode in range(cfg.episodes_per_task):
            init_state = cfg.inits[episode % len(cfg.inits)]
            result = run_one(cfg, env, task_index, init_state, episode, split)
            record["episodes"] += 1
            record["successes"] += int(result["success"])
            record["decisions"] += int(result["decisions"])
            if result["error"]:
                dataset_mod.bump(record["errors"], result["error"][:200])
            for row in result["rows"]:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                _count(record, row)
            record["episode_records"].append(
                {k: v for k, v in result.items() if k != "rows"})
            log(f"task {task_index} episode {episode} init {init_state}: "
                f"{len(result['rows'])} rows, "
                f"{'success' if result['success'] else 'fail'} in {result['decisions']} decisions")
    return record


def _count(record: dict, row: dict) -> None:
    record["rows"] += 1
    for qid, value in row["gold"].items():
        dataset_mod.bump(record["labels"].setdefault(qid, {}), _as_label(value))
    for qid in row["questions"]:
        dataset_mod.bump(record["questions"], qid)
    dataset_mod.bump(record["subgoal"], row["metadata"]["memory_subgoal"])
    for qid in row["metadata"].get("noised", ()):
        dataset_mod.bump(record["noised"], qid)
    for qid in row["metadata"].get("gold_differs", ()):
        dataset_mod.bump(record["gold_differs"], qid)


def _run_tasks(cfg: RolloutConfig, out_dir, tasks, log=log_to_stderr, env_factory=None) -> None:
    """Harvest `tasks` into `out_dir`, leaving one part file per task behind.

    The unit of work of `--jobs N`: a process that owns its own simulator and its own output
    files and shares nothing with its siblings, which is what makes the parallel harvest a plain
    merge at the end rather than a lock.

    `env_factory(suite, task_index) -> DecisionEnv` is how the simulator gets in (`decision.env`);
    `None` builds the configured adapter, which is what a `--jobs N` worker subprocess -- handed
    nothing but JSON -- has to fall back on.
    """
    out_dir = pathlib.Path(out_dir)
    (out_dir / PARTS_DIR).mkdir(parents=True, exist_ok=True)
    factory = env_mod.factory_or_default(env_factory)
    for task_index in tasks:
        env = factory(cfg.suite, task_index)
        try:
            record = harvest_task(cfg, env, task_index, out_dir, log=log)
        finally:
            close = getattr(env, "close", None)
            if close is not None:
                close()
        (out_dir / PARTS_DIR / f"task_{task_index}.json").write_text(
            json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
        log(f"task {task_index}: {record['rows']} rows, "
            f"{record['successes']}/{record['episodes']} episodes succeeded")


def _worker(payload: str) -> None:  # pragma: no cover -- exercised by the real parallel harvest
    """The `--jobs N` entry point: a config and a task list, as JSON, in a fresh process."""
    blob = json.loads(payload)
    cfg = RolloutConfig(**{k: (tuple(v) if isinstance(v, list) else v)
                           for k, v in blob["cfg"].items()})
    _run_tasks(cfg, blob["out_dir"], blob["tasks"])


def harvest_expert(cfg: RolloutConfig, out_dir, log=log_to_stderr, *, jobs: int = 1,
                   env_factory=None, grounding_only: bool = False) -> dict:
    """Harvest every task in `cfg.tasks` into `out_dir`, and return (and write) the manifest.

    With `jobs > 1` the tasks are dealt round-robin to that many processes, each owning its own
    simulator: ten tasks of twenty episodes is about an hour on one core and about six minutes on
    ten, and nothing is shared between them because a task's rows live in a task's file and a
    task's counters in a task's part file. `episode_seed` is a function of (seed, task, episode)
    alone, so `--jobs 8` and `--jobs 1` write the same bytes.

    The grounding rows and the manifest are written by the parent, once, after every worker has
    finished.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    previous = dataset_mod.read_manifest(out_dir) or {}
    if previous and previous.get("source") not in (None, SOURCE):
        raise RolloutError(
            f"{out_dir} already holds {previous.get('source')!r} rows and this run writes "
            f"{SOURCE!r}. A directory holds one source: harvest into a different --out."
        )
    _check_shape(previous, shape_of(cfg), cfg.tasks)
    (out_dir / PARTS_DIR).mkdir(parents=True, exist_ok=True)

    tasks = list(cfg.tasks)
    if grounding_only:
        # Rewrite `grounding.jsonl` and the manifest over a directory whose task files are
        # already there. No simulator, no re-labelling, no touching a motion row: the grounding
        # rows are a separate dataset sharing a directory (their own file, their own split), and
        # re-augmenting them is a file rewrite rather than a harvest.
        if not any((out_dir / task_file(t)).is_file() for t in tasks) and not tasks:
            raise RolloutError(f"{out_dir} holds no task files to regenerate grounding beside")
        log(f"regenerating grounding rows only ({cfg.grounding_repeats} per scene)")
    elif jobs > 1 and len(tasks) > 1:
        _run_parallel(cfg, out_dir, tasks, jobs, log)
    else:
        _run_tasks(cfg, out_dir, tasks, log=log, env_factory=env_factory)

    grounding = grounding_rows(cfg) if cfg.grounding_rows else []
    if cfg.grounding_rows:
        with (out_dir / GROUNDING_FILE).open("w", encoding="utf-8") as handle:
            for row in grounding:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    elif (out_dir / GROUNDING_FILE).is_file():
        (out_dir / GROUNDING_FILE).unlink()

    manifest = write_manifest(cfg, out_dir, previous=previous, log=log)
    return manifest


def _run_parallel(cfg: RolloutConfig, out_dir, tasks, jobs: int, log) -> None:  # pragma: no cover
    """`jobs` processes, each with its own simulator, dealt the tasks round-robin.

    Round-robin rather than contiguous blocks: the tasks do not cost the same (task 4 runs to the
    horizon every time, and a task that never succeeds is the longest one), so dealing them out
    keeps the slowest worker from being handed the four slowest tasks.
    """
    import concurrent.futures
    import multiprocessing

    chunks: list[list[int]] = [[] for _ in range(min(jobs, len(tasks)))]
    for i, task_index in enumerate(tasks):
        chunks[i % len(chunks)].append(task_index)
    payloads = [json.dumps({"cfg": dataclasses.asdict(cfg), "out_dir": str(out_dir),
                            "tasks": chunk}) for chunk in chunks if chunk]
    log(f"harvesting {len(tasks)} task(s) in {len(payloads)} process(es)")
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(payloads),
                                                mp_context=context) as pool:
        for _ in pool.map(_worker, payloads):
            pass


# --------------------------------------------------------------------------------------------
# the manifest


def shape_of(cfg: RolloutConfig) -> dict:
    """The knobs two runs into one directory must agree on, or the merged manifest would describe
    a file it does not describe -- the row-shape rule, in v2's vocabulary."""
    return {
        "source": SOURCE,
        "suite": cfg.suite,
        "questions_version": cfg.questions_version,
        "qids": list(cfg.qids),
        "annotate": cfg.annotate,
        "epsilon": cfg.epsilon,
        "epsilon_exempt": list(cfg.epsilon_exempt),
        # The exemption as it was actually applied: the declared one plus every answer the
        # question set does not ask (see `noise_exempt`).
        "epsilon_exempt_effective": list(noise_exempt(cfg)),
        "cm_per_unit": cfg.cm_per_unit,
        "delta_t": cfg.delta_t,
        "delta_r": cfg.delta_r,
        "seed": cfg.seed,
        "max_steps": cfg.max_steps,
        "horizon": cfg.horizon(),
        "memory_rule": state_mod.MEMORY_RULE_V2,
        "soft_targets": cfg.soft_targets,
        "label_source": "expert_tracker",
        "suite_tasks": list(cfg.suite_tasks),
        "inits": list(cfg.inits),
        "episodes_per_task": cfg.episodes_per_task,
    }


#: The subset of `shape_of` a re-harvest of one task must match: everything that changes a byte
#: of a rendered state or the meaning of a label. `episodes_per_task` is deliberately *not* here
#: -- more episodes of one task beside fewer of another is a bigger file, not a second dataset.
SHAPE_KEYS: tuple[str, ...] = (
    "source", "suite", "questions_version", "qids", "annotate", "epsilon", "epsilon_exempt",
    "epsilon_exempt_effective",
    "cm_per_unit", "delta_t", "delta_r", "seed", "max_steps", "horizon", "memory_rule",
    "soft_targets", "label_source", "suite_tasks",
)


def _check_shape(previous: dict, shape: dict, tasks) -> None:
    clashes = {k: (previous[k], shape[k]) for k in SHAPE_KEYS
               if k in previous and previous[k] != shape[k]}
    if clashes:
        detail = "; ".join(f"{k}: {was!r} -> {now!r}" for k, (was, now) in sorted(clashes.items()))
        raise RolloutError(
            f"this harvest would put task(s) {list(tasks)} beside rows built differently "
            f"({detail}). Harvest every task again, or use a different --out."
        )


def write_manifest(cfg: RolloutConfig, out_dir, previous: dict | None = None,
                   log=log_to_stderr) -> dict:
    """Assemble the manifest from every part file on disk and write it.

    Every knob that changes a byte or a number is echoed, because Task 7's server reads
    `questions_version`, `cm_per_unit` and the tracker settings out of the checkpoint's
    `robojev.json` and refuses to launch against a mismatch -- the same guard `rho` already is for
    v1. The two measured numbers that are *not* knobs are here too: the per-question parse
    agreement over the rows actually written (gate G3, re-run on the real file), and the per-task
    closed-loop success count of the ε-corrupted expert, which is the sanity number for the label
    policy itself.
    """
    out_dir = pathlib.Path(out_dir)
    previous = previous if previous is not None else (dataset_mod.read_manifest(out_dir) or {})
    records: dict[str, dict] = {}
    for part in sorted((out_dir / PARTS_DIR).glob("task_*.json")):
        index = part.stem.split("_", 1)[1]
        if (out_dir / task_file(int(index))).is_file():
            records[index] = json.loads(part.read_text())

    tasks = sorted(int(t) for t in records)
    splits: dict[str, list[int]] = {}
    for task_index in tasks:
        splits.setdefault(records[str(task_index)]["split"], []).append(task_index)

    agreement = parse_agreement(rows_of(out_dir))
    grounding_count = 0
    grounding_splits: dict[str, int] = {}
    grounding_file = out_dir / GROUNDING_FILE
    if grounding_file.is_file():
        for row in rows_of(grounding_file):
            grounding_count += 1
            dataset_mod.bump(grounding_splits, row["split"])

    tracker = state_mod.TrackerV2(**cfg.tracker_kwargs())
    successes = {t: records[str(t)]["successes"] for t in tasks}
    episodes = {t: records[str(t)]["episodes"] for t in tasks}
    manifest = {
        "policy": "robojev",
        **shape_of(cfg),
        "tasks": tasks,
        "harvested": list(cfg.tasks),
        "dev_task": split_config(cfg).dev_task,
        "test_task": split_config(cfg).test_task,
        "splits": splits,
        # Everything a server or a trainer has to agree with, in one block (Task 7 reads it).
        "tracker": tracker.settings(),
        "steps": {"cm": dict(MEASURED_STEPS.cm), "cm_per_unit": cfg.cm_per_unit},
        "bands": {
            "tolerance_cm": tracker.tolerance_cm,
            "arrival_cm": tracker.arrival_cm,
            "yaw_tolerance_deg": tracker.yaw_tolerance_deg,
        },
        "rows": {
            "total": sum(r["rows"] for r in records.values()) + grounding_count,
            "expert_rollout": sum(r["rows"] for r in records.values()),
            "grounding": grounding_count,
            "by_split": _merge_counts(
                [{r["split"]: r["rows"]} for r in records.values()] + [grounding_splits]),
            "by_task": {str(t): records[str(t)]["rows"] for t in tasks},
        },
        "questions": _merge_counts([r["questions"] for r in records.values()]),
        # Every label's marginal, per question. A label's base rate is the number a trained model
        # has to beat to have learned anything at all, and counting it here means it never has to
        # be recovered from tens of thousands of rows afterwards.
        "labels": _merge_nested([r["labels"] for r in records.values()]),
        "subgoal": _merge_counts([r["subgoal"] for r in records.values()]),
        # What ε actually moved, per question -- the check that the exemption holds.
        "noised": _merge_counts([r["noised"] for r in records.values()]),
        # And where the label function and the controller it labels disagree: the band-edge and
        # stage-change seam `answers_v2` documents, measured rather than assumed. A number that
        # climbs here is a label definition drifting away from the controller that produced the
        # states, which is the one thing a relabelling cannot fix afterwards.
        "gold_differs": _merge_counts([r["gold_differs"] for r in records.values()]),
        # Gate G3, re-measured on the rows that were actually written.
        "parse_agreement": agreement,
        # The closed-loop sanity number for the label policy under ε-noise, per task.
        "success": {
            "episodes": sum(episodes.values()),
            "successes": sum(successes.values()),
            "rate": (sum(successes.values()) / sum(episodes.values())) if episodes else None,
            "by_task": {str(t): {"successes": successes[t], "episodes": episodes[t]}
                        for t in tasks},
        },
        "label_errors": _merge_counts([r["errors"] for r in records.values()]),
        "episodes": [e for t in tasks for e in records[str(t)]["episode_records"]],
        "grounding_rows": bool(cfg.grounding_rows),
        "grounding_repeats": max(1, int(cfg.grounding_repeats)),
        "grounding_path": cfg.grounding_path,
        "grounding_phrasings": list(grounding_mod.PHRASINGS),
        "grounding_scenes": cfg.scenes_path,
        "grounding_settled_steps": grounding_mod.SETTLED_STEPS,
        "grounding_splits": grounding_splits,
        "provenance": PROVENANCE,
        "versions": dataset_mod.versions(),
        "created_at": previous.get("created_at") or dataset_mod.now(),
        "updated_at": dataset_mod.now(),
    }
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    log(f"{manifest['rows']['total']} rows ({manifest['rows']['grounding']} grounding), "
        f"{manifest['success']['successes']}/{manifest['success']['episodes']} episodes succeeded")
    for qid, record in sorted(agreement.items()):
        log(f"  parse {qid}: {record['agree']}/{record['rows']}"
            + (f" ({record['rate']:.4f})" if record["rate"] is not None else ""))
    return manifest


def _merge_counts(blocks) -> dict:
    out: dict[str, int] = {}
    for block in blocks:
        for key, value in (block or {}).items():
            out[key] = out.get(key, 0) + int(value)
    return dict(sorted(out.items()))


def _merge_nested(blocks) -> dict:
    out: dict[str, dict] = {}
    for block in blocks:
        for qid, counts in (block or {}).items():
            merged = out.setdefault(qid, {})
            for key, value in counts.items():
                merged[key] = merged.get(key, 0) + int(value)
    return {qid: dict(sorted(counts.items())) for qid, counts in sorted(out.items())}


# --------------------------------------------------------------------------------------------
# the DAgger labeller


def rebuild_tracker(cfg: RolloutConfig, instruction: str):
    """A `TrackerV2` built exactly the way `robojev/policy.py` builds one.

    The constructor arguments are the contract: the server passes `horizon =
    UPSTREAM_MAX_STEPS[suite] // CHUNK_SIZE`, `every = CHUNK_SIZE` and the active qids, and takes
    `annotate` and the bands from their defaults. Every one of those changes a byte of the state
    text -- `Decision 3 of 44; 41 left` is not `Decision 3 of 220; 217 left` -- so a relabeller
    that guessed any of them would rebuild a state the model never read and label it anyway.
    """
    return state_mod.TrackerV2(**cfg.tracker_kwargs(instruction))


def _diff_line(rebuilt: str, recorded: str) -> str:
    """The first line the two texts disagree on, both sides, for an error message.

    A 500-token prompt printed twice is unreadable; the one line that moved is the whole
    diagnosis -- a different `Target:` means the grounding was not replayed, a different
    `Decision k of N` means the horizon was, a different `Events:` means the latch guard was.
    """
    ours, theirs = rebuilt.splitlines(), recorded.splitlines()
    for index in range(max(len(ours), len(theirs))):
        mine = ours[index] if index < len(ours) else "<missing>"
        yours = theirs[index] if index < len(theirs) else "<missing>"
        if mine != yours:
            return f"line {index + 1}: rebuilt {mine!r}, served {yours!r}"
    return "the two differ only in trailing whitespace"


def relabel(trace, instruction: str, privileged_seq, *, cfg: RolloutConfig | None = None,
            roles: dict | None = None, suite: str = "libero_spatial", task_index: int = 0,
            init_state: int = 0, episode: int = 0, split: str = "train",
            source: str = DAGGER_SOURCE) -> list[dict]:
    """Relabel the states a **model** visited, with this expert, on the model's own history.

    `trace` is one entry per decision the policy made and `privileged_seq` the matching object
    poses. An entry carries everything needed to **replay the server's own tracker**, because the
    label has to be a function of the text the model actually read and nothing else:

    * `proprio` and `step` -- the observation and the control step the server counted it at;
    * `state` -- the prompt the policy was served, out of `decisions.meta.state`;
    * `commit` -- `{"target", "destination", "source"}` on the decisions where the server ran its
      grounding forward and committed a pair (`meta.grounding`), and `None` on every other one;
    * `answers` -- the **executed** answers, `{qid: choice}` after the latch guard, which is what
      the server hands to `tracker.answer` and therefore what its `Last 3` block prints;
    * `latch` -- `{"asked", "refused"}` from `meta.grip_latch`, so a refusal that the server wrote
      into the tracker's `Events:` line is written into this one too.

    **Why all of it rather than a fresh tracker with the rule's target.** The round's first real
    run died on `size_x`: the model had grounded the *other* black bowl, so the server's waypoint
    was a different rim point and the relabeller -- which committed the BDDL's own target -- was
    labelling a text it had not rendered. A DAgger row is only a correction if the label is what
    should have been answered *at the state the model read*; if the grounding was wrong, the
    correction for that is a corrected `target` answer, not a motor label quietly computed about
    a different bowl. So the grounding is replayed, not re-decided, and the rebuild is asserted
    byte-identical to the served text before anything is labelled.

    The other half of the round is unchanged: the states are the policy's own, which is the
    distribution that actually shifted, and the history block in them carries the policy's own
    answers (`HISTORY_SOURCES = "policy"`, the measurements).
    """
    cfg = cfg or RolloutConfig(suite=suite)
    tracker = rebuild_tracker(cfg, instruction)
    roles = dict(roles) if roles else None
    phase = None
    rows: list[dict] = []
    for index, (entry, privileged) in enumerate(zip(trace, privileged_seq)):
        step = int(entry.get("step", index * expert_mod.CHUNK_STEPS))
        proprio = np.asarray(entry["proprio"], dtype=np.float64).reshape(-1)
        commit = entry.get("commit")
        if commit is None and index == 0 and roles is not None:
            # A policy that does not ground (a fake, a fixture) still has to name a pair, and the
            # caller's roles are the only thing that can. A served v2 checkpoint always grounds on
            # its first decision, so this is the test path and not the deployed one.
            commit = dict(roles)
        if commit is not None:
            # `commit` **before** `observe`, exactly as the server's `_ground` does, and at the
            # same step: it is what fills `Target: … (chosen at t=…)` and it rebuilds the
            # planner's carry, so a re-grounding mid-episode re-plans here as it re-planned there.
            tracker.commit(commit.get("target"), commit.get("destination"), step=step,
                           source=str(commit.get("source") or "model"))
            roles = {"target": commit.get("target"), "destination": commit.get("destination")}
            phase = expert_mod.new_phase(roles)
        if roles is None:
            raise RolloutError(
                f"the trace names no target at step {step}: a served v2 checkpoint commits one "
                f"on its first decision (`meta.grounding`) and a caller that has no such record "
                f"must pass `roles=`. A motor label about an unnamed object is not a label."
            )
        tracker.observe(step=step, proprio=proprio, objects=privileged)
        rebuilt = state_mod.serialise_v2(proprio, privileged, instruction, tracker,
                                         annotate=tracker.annotate)
        recorded = entry.get("state") or None
        # The coarsest mismatch first, because it has the most useful message: a prompt with no
        # `Waypoint (...)` line at all is a v1 checkpoint being rolled out under v2's labeller,
        # not a tracker replayed slightly wrong.
        if recorded is not None and not _is_v2_state(recorded):
            raise RolloutError(
                f"the policy's own prompt at step {step} is not a v2 state (no `Waypoint (...)` "
                f"line): this round is rolling out a checkpoint served on v1's text, and a v2 "
                f"label read off it would be a label its own state cannot carry. Roll out a v2 "
                f"checkpoint, or relabel without the policy's prompt."
            )
        if recorded is not None and recorded != rebuilt:
            raise RolloutError(
                f"the state rebuilt for step {step} is not the state the policy was served, so a "
                f"label read off this tracker would be a label of a text nobody read "
                f"({_diff_line(rebuilt, recorded)}). The tracker must be constructed and driven "
                f"exactly as `robojev/policy.py` drives its own -- same horizon, same "
                f"qids, same committed grounding, same executed answers, same latch events."
            )
        text = recorded or rebuilt
        gold, phase, meta = expert_mod.gold_for_state(tracker, proprio, privileged, instruction,
                                                      phase, delta_t=cfg.delta_t, roles=roles)
        answers = {qid: _as_label(v) for qid, v in (entry.get("answers") or {}).items()}
        row = motion_row(
            cfg=cfg, suite=suite, task_index=task_index, init_state=init_state, episode=episode,
            step=step, split=split, state_text=text, gold=gold, tracker=tracker, meta=meta,
            executed=answers, roles={**roles, "source": str((commit or {}).get("source")
                                                            or tracker.source or "unknown")},
            decision_index=index,
            identifier=dagger_row_id(source, suite, task_index, init_state, episode, step))
        row["metadata"]["source"] = source
        row["metadata"]["history_source"] = "policy"
        row["metadata"]["state_source"] = "policy" if recorded is not None else "rebuilt"
        row["metadata"]["state_matches_rebuild"] = True
        row["metadata"]["grounded_here"] = commit is not None
        row["metadata"]["policy_choice"] = answers or None
        check_row(row, strict=cfg.strict_parse)
        rows.append(row)
        _replay_latch(tracker, entry.get("latch"))
        # The policy's own executed answers drive the history, verbatim -- the same dict the
        # server hands its own tracker. A policy that reported nothing leaves the block blank
        # rather than having the expert's answers attributed to it.
        tracker.answer(answers or None)
        # ...and the **labeller's own carry takes the executed `rim` letter too**, because the
        # tracker's just did and the two are the same plan. Skip it and the server's plan and
        # this one stand on different candidates from the first selection that reads a letter --
        # the episode's first decision, a blocked stage, a grasp that held nothing -- and the
        # first retry puts them a rim axis and a retry offset apart. Round 1's second failure
        # was exactly that: `in subgoal 'reach' the tracker steers to ... and the expert to ...
        # they differ by 0.0150 m`, the model having asked again for the letter it had just
        # tried while this carry moved on to the plan's next direction.
        phase = expert_mod.with_executed(phase, answers or None, qids=tracker.qids)
    return rows


def _replay_latch(tracker, latch: dict | None) -> None:
    """Re-run the serving latch guard so its refusals land in this tracker's `Events:` too.

    The server calls `GripLatch.update(..., log=tracker.log_latch)` after it has rendered a state
    and before the next one, so a refusal is invisible in the state that caused it and printed in
    the one after. Replaying it is not optional bookkeeping: skip it and every state from the
    first refusal onwards differs from the one the model read.
    """
    if not latch:
        return
    row = tracker.last_row
    offset = row.waypoint_cm
    arrived = bool(offset is not None
                   and all(abs(float(v)) < tracker.arrival_cm for v in offset))
    tracker.latch.update(bool(latch.get("asked")), subgoal=tracker.subgoal, arrived=arrived,
                         log=tracker.log_latch)


def main(argv=None) -> int:  # pragma: no cover -- the module entry point, for a worker shell
    """`python -m robojev.rollout --tasks 0,9 --out DIR` -- the harvest, standalone.

    The CLI (`robojev harvest robojev --source expert`) is the supported surface; this exists so a
    cluster job script can run one task subset without going through it.
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--no-grounding-rows", action="store_true")
    parser.add_argument("--no-annotate", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    tasks = (tuple(int(t) for t in args.tasks.split(",") if t.strip())
             if args.tasks else tuple(range(10)))
    cfg = RolloutConfig(suite=args.suite, tasks=tasks, episodes_per_task=args.episodes_per_task,
                        epsilon=args.epsilon, seed=args.seed,
                        grounding_rows=not args.no_grounding_rows, annotate=not args.no_annotate)
    manifest = harvest_expert(cfg, pathlib.Path(args.out), jobs=args.jobs)
    print(json.dumps(manifest["rows"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
