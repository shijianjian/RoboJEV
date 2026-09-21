"""Rows from the scripted expert's own rollouts, driven against a stub arm.

The whole file is CPU: `robojev.rollout` reaches the simulator through exactly four
seams -- `env.reset`, `env.state_vector`, `env.privileged` and `env.step` -- and `StubArm` below
implements all four in thirty lines of arithmetic, so the row shape, the ε rule, the split rule,
the grounding join, the manifest and gate G3's write-time invariant are all testable with no
MuJoCo, no EGL and no LIBERO. The `sim`-marked proof of the real thing is the harvest itself.

What is worth knowing about the stub: it is a hand that moves 1 cm per unit of commanded action
per control step (`EXECUTED_COARSE_PER_DELTA_T / CHUNK_STEPS`, which is what the real arm was
*measured* to do), a gripper that shuts over two steps, and a bowl that follows the fingers once
they are closed around it. That is enough for the expert's phase machine to reach, descend,
close, lift, carry and release, which is every sub-stage a row can be written in.
"""
from __future__ import annotations

import collections
import dataclasses
import json
import pathlib
from importlib import import_module

import numpy as np
import pytest

from robojev import expert as expert_mod
from robojev import questions as questions_mod
from robojev import rollout as R
from test_dataset import validate_training_row   # tests/ is on sys.path: no __init__.py, by design

#: `import_module`, not `from robojev import parse`: the package re-exports a
#: *function* called `parse`, so the plain form hands back the function.
parse_mod = import_module("robojev.parse")


class StubArm:
    """A LIBERO env reduced to what an expert rollout reads."""

    instruction = "pick up the black bowl and place it on the plate"
    obj_of_interest = ("akita_black_bowl_1", "plate_1")

    #: The measured executed displacement, per control step, per unit of commanded action.
    PER_STEP_M = expert_mod.EXECUTED_COARSE_PER_DELTA_T / expert_mod.CHUNK_STEPS

    def __init__(self, bowl=(0.02, 0.02, 0.90), plate=(0.22, 0.18, 0.90),
                 start=(-0.05, -0.10, 1.12), decoy=None):
        self._bowl0 = np.asarray(bowl, np.float64)
        self._plate = np.asarray(plate, np.float64)
        self._start = np.asarray(start, np.float64)
        # A second, identical bowl. Off by default; the grounding tests need it, because the
        # whole failure this file reproduces is a model naming the *other* one.
        self._decoy = None if decoy is None else np.asarray(decoy, np.float64)
        if self._decoy is not None:
            self.obj_of_interest = ("akita_black_bowl_1", "plate_1")
        self.reset(0)

    # -- the four seams --------------------------------------------------------------------
    def reset(self, init_state_index: int):
        # Each init state is a slightly different start pose, the way LIBERO's are.
        self.eef = self._start + np.array([0.01, 0.01, 0.0]) * float(init_state_index)
        self.bowl = self._bowl0.copy()
        self.width = 0.08
        self.holding = False
        self.steps = 0
        return self._obs()

    def privileged(self, obs):
        poses = {"akita_black_bowl_1": {"pos": self.bowl.copy(),
                                        "quat": np.array([1.0, 0.0, 0.0, 0.0])},
                 "plate_1": {"pos": self._plate.copy(),
                             "quat": np.array([1.0, 0.0, 0.0, 0.0])}}
        if self._decoy is not None:
            poses["akita_black_bowl_2"] = {"pos": self._decoy.copy(),
                                           "quat": np.array([1.0, 0.0, 0.0, 0.0])}
        return poses

    @staticmethod
    def state_vector(obs):
        return np.asarray(obs["state"], np.float64)

    def step(self, action):
        from robojev.envs import StepResult

        action = np.asarray(action, np.float64)
        self.steps += 1
        self.eef = self.eef + self.PER_STEP_M * action[0:3]
        if action[6] > 0:
            self.width = max(self.width - 0.02, 0.0)
            if self.width <= 0.04 and not self.holding:
                self.holding = bool(np.linalg.norm(self.eef - self.bowl) <= 0.06)
        else:
            self.width = min(self.width + 0.02, 0.08)
            self.holding = False
        if self.holding:
            self.bowl = self.eef.copy()
        done = bool(not self.holding
                    and float(np.hypot(*(self.bowl[0:2] - self._plate[0:2]))) <= 0.05
                    and abs(float(self.bowl[2] - self._plate[2])) <= 0.06
                    and self.steps > 5)
        return StepResult(obs=self._obs(), reward=0.0, done=done, info={})

    def close(self) -> None:
        pass

    def _obs(self):
        state = np.concatenate([self.eef, [0.0, 0.0, 0.0],
                                [self.width / 2.0, -self.width / 2.0]])
        return {"robot0_eef_pos": self.eef.copy(), "state": state.astype(np.float64)}


def stub_factory(cfg, task_index: int):
    """One arm per task, at a different bowl, so a two-task harvest is two different scenes."""
    shift = 0.01 * float(task_index)
    return StubArm(bowl=(0.02 + shift, 0.02 - shift, 0.90))


def config(**overrides) -> R.RolloutConfig:
    base = dict(tasks=(0,), suite_tasks=tuple(range(10)), inits=(0,), episodes_per_task=1,
                epsilon=0.0, grounding_rows=False, max_steps=15)
    base.update(overrides)
    return R.RolloutConfig(**base)


def harvest(tmp_path, cfg) -> dict:
    return R.harvest_expert(cfg, tmp_path, log=lambda _m: None, env_factory=stub_factory)


def written(tmp_path, task_index: int = 0) -> list[dict]:
    return list(R.rows_of(pathlib.Path(tmp_path) / R.task_file(task_index)))


# ----------------------------------------------------------------------------- the row shape


def test_a_three_decision_episode_writes_one_row_per_decision(tmp_path):
    """Fifteen control steps is three chunks, so it is three decisions and three rows -- and the
    `step` each row carries is the control step of its own decision point, not its index."""
    manifest = harvest(tmp_path, config())
    rows = written(tmp_path)
    assert len(rows) == 3, [row["id"] for row in rows]
    assert [row["metadata"]["step"] for row in rows] == [0, 5, 10]
    assert [row["metadata"]["decision"] for row in rows] == [0, 1, 2]
    assert manifest["rows"]["total"] == 3
    assert manifest["rows"]["expert_rollout"] == 3


def test_the_gold_is_hard_and_kinded_as_a_reference_answer(tmp_path):
    """Note §6.1: the **hard** arm won action agreement 81.6 % against the soft arm's 77.1 %, so a
    one-hot target is what a row carries unless `soft_targets` is asked for by name."""
    harvest(tmp_path, config())
    for row in written(tmp_path):
        for qid, probs in row["gold_probs"].items():
            assert sorted(probs.values())[-1] == 1.0, (row["id"], qid, probs)
            assert sum(probs.values()) == pytest.approx(1.0)
            assert set(probs.values()) <= {0.0, 1.0}
            assert row["gold_probs_kind"][qid] == R.PROBS_KIND
            assert row["gold_label_kind"][qid] == R.LABEL_KIND


def test_the_soft_arm_is_reachable_and_is_not_the_default(tmp_path):
    """Kept as the comparison NanoJev's declared-noise recipe is, and not switched on."""
    assert R.RolloutConfig().soft_targets is False
    harvest(tmp_path, config(soft_targets=True))
    row = written(tmp_path)[0]
    assert set(row["gold_probs"]["move_x"].values()) != {0.0, 1.0}


def test_every_active_question_is_asked_and_only_the_active_ones(tmp_path):
    """The shared `step` is defined and not asked; a row that asked it would be tokens for zero
    bits, and a row that missed `size_z` would be unbuildable. `yaw` **is** asked since the
    drawer task (note `2026-09-21-stage9-drawer-task.md`)."""
    harvest(tmp_path, config())
    for row in written(tmp_path):
        assert set(row["questions"]) <= set(questions_mod.ACTIVE_QIDS)
        assert "step" not in row["questions"] and "yaw" in row["questions"]
        assert set(row["gold"]) == set(row["questions"])
        assert {"move_x", "move_y", "move_z", "size_x", "size_y", "size_z", "subgoal"} <= set(
            row["questions"])


# --------------------------------------------------------------------------- gate G3's invariant


def test_every_written_row_reparses_to_its_own_gold(tmp_path):
    """Spec §2's principle 1, asserted on the rows that were actually written: a rule over the
    string alone reproduces every label. `harvest_expert` raises rather than writing a row that
    fails this, so the agreement is 1.0 by construction -- and this is what says so."""
    manifest = harvest(tmp_path, config())
    agreement = manifest["parse_agreement"]
    assert agreement, manifest
    for qid, record in agreement.items():
        assert record["rate"] == 1.0, (qid, record)
        assert record["rows"] == 3
    # And independently of the manifest, from the file.
    for row in written(tmp_path):
        assert parse_mod.parse(row["state"], qids=tuple(row["gold"])) == row["gold"]


def test_a_row_whose_text_disagrees_with_its_gold_is_refused():
    """The check is the point of the check: a state text that has been edited away from its label
    must raise where the row is made, not surface as a 97 % gate weeks later."""
    tracker = type("T", (), {"subgoal": "reach"})()
    row = {"id": "x", "state": "Waypoint (r): x +1.0, y +0.0, z +0.0   [tolerance 0.3; arrived "
                               "within 1.3]\nGripper: open 8.0 cm; holding nothing.\n"
                               "Subgoal so far: reach.\n",
           "gold": {"move_x": "-"}}
    with pytest.raises(R.RolloutError, match="re-parses to a different answer"):
        R.check_row(row, strict=True)
    assert R.check_row(row, strict=False) == {"move_x": False}
    assert tracker.subgoal == "reach"


def test_an_unparseable_state_is_refused_rather_than_scored_zero():
    with pytest.raises(R.RolloutError, match="cannot be parsed"):
        R.check_row({"id": "x", "state": "nothing here", "gold": {"move_x": "-"}}, strict=True)


# ------------------------------------------------------------------------------------ the noise


def test_the_noise_moves_the_arm_and_never_the_label(tmp_path):
    """ε is applied to the **executed** answer: the row says what the expert would have done, the
    arm does something slightly else, and that is what puts the model's own error distribution
    into the training states. A corrupted gold would be a corrupted dataset."""
    harvest(tmp_path, config(epsilon=1.0, max_steps=30))
    rows = written(tmp_path)
    assert any(row["metadata"]["noised"] for row in rows), "p=1 corrupted nothing"
    for row in rows:
        # The label is still the label the text implies, whatever the arm was told to do.
        assert parse_mod.parse(row["state"], qids=tuple(row["gold"])) == row["gold"]


def test_the_latch_the_wrist_and_the_rim_are_exempt_from_the_noise_at_any_epsilon(tmp_path):
    """Spec §5 and note §6.9: success falls 19/20 -> 7/20 at p = 0.10 with `grip` corrupted and
    holds at 11-13/20 with it exempt, because an exploring controller plus an irreversible action
    is most of upstream's Predict-Position gap.

    `yaw` is exempt for a different reason and a measured one: the rim direction **is** the
    wrist's closing axis rotated by the candidate's turn, so a corrupted wrist moves the
    waypoint itself rather than mis-stepping towards it. At ε = 0.10 with it corrupted, a proof
    harvest of tasks 0, 4 and 9 scored 0/6.

    `rim` is exempt for a third reason, and one the numbers decided rather than the argument: it
    is not a step but a commitment, and the plan *obeys* it, so a wrong letter costs the whole
    approach and descent that rim point takes. Tasks 0, 4 and 9, 8 episodes each at ε = 0.10:
    14/24 with it corrupted against 19/24 with it exempt, the whole difference on the two fixture
    tasks (`scratch/v2-drawer/round3/proof_sweep.py`).
    """
    assert R.RolloutConfig().epsilon_exempt == ("grip", "yaw", "rim")
    harvest(tmp_path, config(epsilon=1.0, max_steps=40))
    for row in written(tmp_path):
        assert "grip" not in row["metadata"]["noised"]
        assert "yaw" not in row["metadata"]["noised"]
        assert "rim" not in row["metadata"]["noised"]
        for qid in ("grip", "yaw", "rim"):
            if qid in row["gold"]:
                assert row["metadata"]["executed"][qid] == row["metadata"]["controller"][qid]


def test_the_same_seed_writes_the_same_bytes(tmp_path):
    """The reproducibility `episode_seed` promises, and the thing `--jobs N` relies on: the
    corruption stream is a function of (seed, task, episode) and of nothing else."""
    first, second = tmp_path / "a", tmp_path / "b"
    harvest(first, config(epsilon=0.5, max_steps=40))
    harvest(second, config(epsilon=0.5, max_steps=40))
    assert (first / R.task_file(0)).read_bytes() == (second / R.task_file(0)).read_bytes()


def test_a_task_harvested_alone_is_the_task_harvested_beside_others(tmp_path):
    """`--tasks 3` must reproduce what `--tasks 0-9` produced for task 3, or a re-harvest of one
    task silently makes a second dataset."""
    both, alone = tmp_path / "both", tmp_path / "alone"
    harvest(both, config(tasks=(0, 1), epsilon=0.5, max_steps=30))
    harvest(alone, config(tasks=(1,), epsilon=0.5, max_steps=30))
    assert (both / R.task_file(1)).read_bytes() == (alone / R.task_file(1)).read_bytes()


# ----------------------------------------------------------------------------------- the filter


def test_a_post_release_row_does_not_ask_grip():
    """Upstream's ammo filter, in this vocabulary: after the release there is no grasp to decide
    about. The rest of the row stays -- the retreat is a real decision, and dropping it would
    teach the policy that episodes end where the bowl lands."""
    cfg = config()
    gold = {qid: "hold" if qid.startswith("move") else "small"
            for qid in ("move_x", "move_y", "move_z", "size_x", "size_y", "size_z")}
    gold.update(grip=False, subgoal="retreat")
    row = R.motion_row(
        cfg=cfg, suite="libero_spatial", task_index=0, init_state=0, episode=0, step=0,
        split="train", state_text="", gold=gold,
        tracker=type("T", (), {"subgoal": "retreat"})(), meta={}, executed={},
        roles={"target": "b", "destination": "p", "source": "obj_of_interest"},
        decision_index=0)
    assert "grip" not in row["questions"]
    assert "grip" not in row["gold"]
    assert "grip" not in row["gold_probs"]
    assert "subgoal" in row["gold"]


def test_a_failed_episode_is_kept_in_full(tmp_path):
    """No outcome filter (`SONIC_PREDICT_POSITION.md:48-51` via the note): a fifteen-step horizon
    cannot finish the task, and every one of its decision points is still a row."""
    manifest = harvest(tmp_path, config())
    assert manifest["success"]["successes"] == 0
    assert manifest["rows"]["expert_rollout"] == 3


# ------------------------------------------------------------------------------------ the split


def test_the_split_follows_the_task_and_not_the_subset(tmp_path):
    """Tasks 0-7 train, 8 dev, 9 test -- resolved over the suite and never over the tasks this run
    happens to touch, or `--tasks 0,9` would call task 0 `dev`."""
    manifest = harvest(tmp_path, config(tasks=(0, 9)))
    assert manifest["splits"] == {"train": [0], "test": [9]}
    assert {row["split"] for row in written(tmp_path, 0)} == {"train"}
    assert {row["split"] for row in written(tmp_path, 9)} == {"test"}


def test_the_grounding_rows_land_with_their_own_source_and_their_own_split(tmp_path):
    """A held-out task and a held-out instruction are two different held-out things, so Task 4's
    rows keep their instruction-level split and are never concatenated with a motion state."""
    grounding = tmp_path / "grounding-source.jsonl"
    rows = [{"id": "g1", "state_id": "g1", "family_id": "libero_object", "split": "test",
             "state": "Camera frame...", "questions": {}, "gold": {}, "gold_probs": {},
             "gold_probs_kind": {}, "gold_label_kind": {},
             "metadata": {"suite": "libero_object", "_private": 1}}]
    grounding.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    out = tmp_path / "out"
    manifest = harvest(out, config(grounding_rows=True, grounding_path=str(grounding)))
    landed = list(R.rows_of(out / R.GROUNDING_FILE))
    assert [row["metadata"]["source"] for row in landed] == [R.GROUNDING_SOURCE]
    assert "_private" not in landed[0]["metadata"]
    assert landed[0]["split"] == "test"
    assert manifest["rows"]["grounding"] == 1
    assert manifest["rows"]["total"] == 4
    assert manifest["grounding_splits"] == {"test": 1}
    # And the motion rows still say what they are.
    assert {row["metadata"]["source"] for row in written(out)} == {R.SOURCE}


# --------------------------------------------------------------------------------- the manifest


def test_the_manifest_echoes_every_knob_that_changes_a_byte(tmp_path):
    """Task 7's server reads `questions_version`, `cm_per_unit` and the tracker settings out of
    the checkpoint's `robojev.json` and refuses a mismatch at launch, so a knob that is not here is
    a way for a checkpoint and a server to disagree in silence."""
    cfg = config(tasks=(0,), epsilon=0.07, annotate=False)
    manifest = harvest(tmp_path, cfg)
    assert manifest["source"] == R.SOURCE
    assert manifest["questions_version"] == "v2"
    assert manifest["epsilon"] == 0.07
    assert manifest["epsilon_exempt"] == ["grip", "yaw", "rim"]
    assert manifest["annotate"] is False
    assert manifest["episodes_per_task"] == 1
    assert manifest["inits"] == [0]
    assert manifest["seed"] == cfg.seed
    assert manifest["cm_per_unit"] == 5.0
    assert manifest["delta_t"] == 1.0
    assert manifest["delta_r"] == pytest.approx(expert_mod.EXPERT_DELTA_R)
    assert manifest["memory_rule"] == "v2-tracker-1"
    assert manifest["qids"] == list(questions_mod.ACTIVE_QIDS)
    assert manifest["tracker"]["memory_rule"] == "v2-tracker-1"
    assert manifest["tracker"]["annotate"] is False
    assert manifest["steps"]["cm"]["large"] == 5.0
    assert manifest["bands"]["tolerance_cm"] == 0.3
    assert manifest["bands"]["arrival_cm"] == pytest.approx(1.3)
    assert manifest["provenance"] == R.PROVENANCE
    # The two measured blocks: gate G3 re-run on the real rows, and the ε-corrupted expert's own
    # closed-loop count, which is the sanity number for the label policy itself.
    assert set(manifest["parse_agreement"]) == set(questions_mod.ACTIVE_QIDS)
    assert manifest["success"]["by_task"] == {"0": {"successes": 0, "episodes": 1}}
    assert manifest["rows"]["by_task"] == {"0": 3}


def test_the_manifest_counts_every_labels_marginal(tmp_path):
    """A label's base rate is the number a trained model has to beat to have learned anything at
    all, so the harvest counts it rather than leaving it to be recovered from 20,000 rows."""
    manifest = harvest(tmp_path, config())
    labels = manifest["labels"]
    assert set(labels) == set(questions_mod.ACTIVE_QIDS)
    for qid, counts in labels.items():
        assert sum(counts.values()) == 3, (qid, counts)
        assert set(counts) <= set(_candidates(qid))


def _candidates(qid: str):
    question = questions_mod.QUESTIONS[qid]
    return ("false", "true") if question["type"] == "boolean" else tuple(question["criteria"])


def test_a_directory_holds_one_source(tmp_path):
    """`--source demos` rows and `--source expert` rows answer different questions about
    different states; a directory holding both would train one model on two label definitions."""
    harvest(tmp_path, config())
    manifest = json.loads((tmp_path / R.MANIFEST).read_text())
    manifest["source"] = "demos"
    (tmp_path / R.MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(R.RolloutError, match="already holds"):
        harvest(tmp_path, config())


def test_a_reharvest_under_a_different_shape_is_refused(tmp_path):
    """The same rule `harvest._check_row_shape` applies to the demonstrations' rows: a directory
    holding rows rendered with the annotations on and rows rendered without them holds two
    datasets under one manifest."""
    harvest(tmp_path, config(tasks=(0,)))
    with pytest.raises(R.RolloutError, match="built differently"):
        harvest(tmp_path, config(tasks=(1,), annotate=False))


# ------------------------------------------------------------------------------ the relabeller


def test_relabel_labels_the_policys_own_states_with_the_experts_answer(tmp_path):
    """The DAgger round: the states are the model's, the history is the model's, and the label is
    what this expert would have answered there -- checked, as every row is, against the text."""
    env = StubArm()
    obs = env.reset(0)
    trace, privileged_seq = [], []
    for index in range(4):
        proprio = env.state_vector(obs)
        privileged_seq.append(env.privileged(obs))
        trace.append({"step": index * 5, "proprio": proprio,
                      "answers": {"move_x": "+", "move_y": "hold", "move_z": "-",
                                  "size_x": "large", "size_y": "small", "size_z": "medium",
                                  "grip": "false"}})
        for _ in range(5):
            obs = env.step(np.array([0.2, 0.2, -0.2, 0, 0, 0, -1.0])).obs

    roles = {"target": "akita_black_bowl_1", "destination": "plate_1", "source": "obj_of_interest"}
    rows = R.relabel(trace, env.instruction, privileged_seq, roles=roles)
    assert len(rows) == 4
    for row in rows:
        assert row["metadata"]["source"] == R.DAGGER_SOURCE
        assert row["metadata"]["history_source"] == "policy"
        assert row["metadata"]["policy_choice"]["move_x"] == "+"
        assert parse_mod.parse(row["state"], qids=tuple(row["gold"])) == row["gold"]


def test_relabel_prefers_the_prompt_the_policy_was_actually_given():
    """`dagger._Recorder._state_text`'s rule: after the tracker the prompt is a function of the
    whole episode, and the point of a DAgger row is that the episode is the policy's."""
    env = StubArm()
    obs = env.reset(0)
    proprio = env.state_vector(obs)
    privileged = env.privileged(obs)
    roles = {"target": "akita_black_bowl_1", "destination": "plate_1", "source": "obj_of_interest"}
    rebuilt = R.relabel([{"step": 0, "proprio": proprio, "answers": {}}], env.instruction,
                        [privileged], roles=roles)[0]
    given = R.relabel([{"step": 0, "proprio": proprio, "answers": {},
                        "state": rebuilt["state"]}], env.instruction, [privileged], roles=roles)[0]
    assert rebuilt["metadata"]["state_source"] == "rebuilt"
    assert given["metadata"]["state_source"] == "policy"
    assert given["metadata"]["state_matches_rebuild"] is True
    assert given["gold"] == rebuilt["gold"]


def test_a_question_that_is_not_asked_is_not_noised_either(tmp_path):
    """ε is a model of the model: a policy that is never asked a question never gets it wrong, so
    corrupting it would move the arm in states no deployed policy can reach and every row after
    that would describe a scene the model will never see.

    Asked with the question set a pre-wrist checkpoint was harvested under, which is the case
    this rule exists for -- and with the wrist asked, `yaw` is noised like any other answer.
    """
    cfg = config(epsilon=1.0, max_steps=40, qids=questions_mod.active_qids(yaw=False),
                 epsilon_exempt=("grip",))
    assert "yaw" in R.noise_exempt(cfg) and "grip" in R.noise_exempt(cfg)
    assert "move_x" not in R.noise_exempt(cfg)
    manifest = harvest(tmp_path, cfg)
    assert "yaw" not in manifest["noised"]
    assert "grip" not in manifest["noised"]
    assert manifest["noised"], "p=1 corrupted nothing"


def test_every_row_validates_against_the_trainers_own_rule(tmp_path):
    """NanoJev's `validate_training_row`, vendored in `test_harvest`: a harvest that finishes has
    produced a file the trainer accepts, and finding out otherwise on the cluster is the most
    expensive possible moment."""
    harvest(tmp_path, config(tasks=(0, 9), epsilon=0.1, max_steps=40))
    rows = written(tmp_path, 0) + written(tmp_path, 9)
    assert rows
    for row in rows:
        validate_training_row(row)


# ------------------------------------------------ replaying the server's own tracker (round 1)
#
# The first real DAgger round died on its first episode:
#
#   RolloutError: expert:libero_spatial:0:0:1:0: the state text re-parses to a different answer
#   for ['size_x'] (parsed ['medium'], gold ['large'])
#
# The model had grounded the *other* black bowl, so the served state named a different rim point
# than the relabeller -- which committed the BDDL's own target -- rebuilt. These tests are that
# failure, reduced: a served episode whose tracker is committed to one bowl, relabelled against
# the other, and the same trace relabelled correctly.

SERVED_ANSWERS = {"move_x": "+", "move_y": "hold", "move_z": "-", "size_x": "large",
                  "size_y": "small", "size_z": "medium", "yaw": "hold", "rim": "A",
                  "grip": "false", "subgoal": "reach"}


def serve(env, *, target, destination="plate_1", decisions=4, asked=lambda i: False,
          commit_at=(0,)):
    """Drive `env` the way `recipes/robojev/server.py` drives a rollout, and record the trace.

    Deliberately a copy of the server's *order* -- commit, observe, serialise, answer the model,
    update the latch guard against the state just rendered, hand the executed answers to the
    tracker -- because that order is what `rollout.relabel` has to replay. A helper that took a
    shortcut here would prove nothing about the thing that broke.
    """
    from robojev.state import TrackerV2, serialise_v2

    tracker = TrackerV2(horizon=44, every=5, qids=questions_mod.ACTIVE_QIDS,
                        instruction=env.instruction)
    obs = env.reset(0)
    trace, privileged_seq = [], []
    for index in range(decisions):
        proprio = env.state_vector(obs)
        privileged = env.privileged(obs)
        commit = None
        if index in commit_at:
            commit = {"target": target, "destination": destination, "source": "model"}
            tracker.commit(target, destination, step=index * 5, source="model")
        tracker.observe(step=index * 5, proprio=proprio, objects=privileged)
        text = serialise_v2(proprio, privileged, env.instruction, tracker,
                            annotate=tracker.annotate)
        trace.append({"step": index * 5, "decision": index, "proprio": proprio, "state": text,
                      "answers": dict(SERVED_ANSWERS), "commit": commit,
                      "latch": {"asked": bool(asked(index)), "refused": False}})
        privileged_seq.append(privileged)
        row = tracker.last_row
        offset = row.waypoint_cm
        arrived = bool(offset is not None
                       and all(abs(float(v)) < tracker.arrival_cm for v in offset))
        tracker.latch.update(bool(asked(index)), subgoal=tracker.subgoal, arrived=arrived,
                             log=tracker.log_latch)
        tracker.answer(dict(SERVED_ANSWERS))
        for _ in range(5):
            obs = env.step(np.array([0.3, 0.3, -0.3, 0, 0, 0, -1.0])).obs
    return trace, privileged_seq


def two_bowls():
    return StubArm(bowl=(0.02, 0.02, 0.90), decoy=(-0.16, 0.14, 0.90))


def test_the_round_relabels_about_the_bowl_the_model_actually_grounded(tmp_path):
    """Round 1's bug. The model committed `akita_black_bowl_2`; every state it read named that
    bowl's rim. A relabeller that re-decided the target from the BDDL would label a waypoint the
    model never saw -- and quietly, since both labels are well-formed. Replaying the commit is
    what makes the row a correction of the state the model read."""
    env = two_bowls()
    trace, privileged = serve(env, target="akita_black_bowl_2")
    rows = R.relabel(trace, env.instruction, privileged)
    assert len(rows) == len(trace)
    for row, entry in zip(rows, trace):
        assert row["state"] == entry["state"]
        assert row["metadata"]["state_source"] == "policy"
        assert row["metadata"]["target_object"] == "akita_black_bowl_2"
        assert parse_mod.parse(row["state"], qids=tuple(row["gold"])) == row["gold"]
    assert rows[0]["metadata"]["grounded_here"] is True
    assert rows[1]["metadata"]["grounded_here"] is False


def test_a_rebuild_that_is_not_the_served_state_is_refused_and_names_the_line(tmp_path):
    """The same trace, relabelled as if the BDDL's bowl had been grounded: the rebuild is a
    different text, and the round stops with the line that moved rather than writing a row whose
    label is about a different object."""
    env = two_bowls()
    trace, privileged = serve(env, target="akita_black_bowl_2")
    wrong = [{**entry, "commit": (None if entry["commit"] is None
                                  else {**entry["commit"], "target": "akita_black_bowl_1"})}
             for entry in trace]
    with pytest.raises(R.RolloutError) as caught:
        R.relabel(wrong, env.instruction, privileged)
    message = str(caught.value)
    assert "not the state the policy was served" in message
    assert "Target:" in message or "Waypoint" in message


def test_a_latch_refusal_is_replayed_into_the_rebuilt_trackers_events(tmp_path):
    """The serving guard writes its refusals into the tracker as events, and an event is printed
    in the state *after* the decision that caused it. Skip the replay and every state from the
    first refusal onwards is a state the model never read."""
    env = two_bowls()
    trace, privileged = serve(env, target="akita_black_bowl_2", decisions=4,
                              asked=lambda i: i == 0)
    assert "Events: t=" in trace[1]["state"], "the fixture did not produce a refusal to replay"
    rows = R.relabel(trace, env.instruction, privileged)
    assert [row["state"] for row in rows] == [entry["state"] for entry in trace]
    blind = [{**entry, "latch": None} for entry in trace]
    with pytest.raises(R.RolloutError, match="not the state the policy was served"):
        R.relabel(blind, env.instruction, privileged)


def test_a_rebuilt_tracker_at_the_wrong_horizon_is_refused(tmp_path):
    """`Decision 3 of 44; 41 left` is not `Decision 3 of 220; 217 left`. The horizon is one of the
    constructor arguments that changes a byte of every state, so getting it wrong has to stop the
    round rather than relabel 44 states that were rendered at another one."""
    env = two_bowls()
    trace, privileged = serve(env, target="akita_black_bowl_2")
    with pytest.raises(R.RolloutError, match="not the state the policy was served"):
        R.relabel(trace, env.instruction, privileged,
                  cfg=R.RolloutConfig(max_steps=1100))


def test_the_relabelled_row_says_where_the_policy_and_the_expert_disagreed(tmp_path):
    """The round's whole yield, per row: which questions the model got wrong at a state it chose
    to visit. The fixture answers the same thing at every decision, so it is wrong somewhere."""
    env = two_bowls()
    trace, privileged = serve(env, target="akita_black_bowl_2")
    rows = R.relabel(trace, env.instruction, privileged)
    for row in rows:
        expected = sorted(qid for qid, value in row["gold"].items()
                          if R._as_label(SERVED_ANSWERS[qid]) != R._as_label(value))
        assert row["metadata"]["disagrees"] == expected
        assert row["metadata"]["policy_choice"]["move_x"] == "+"
    assert any(row["metadata"]["disagrees"] for row in rows)


def test_a_trace_with_no_committed_target_and_no_roles_is_refused(tmp_path):
    """A motor label about an unnamed object is not a label."""
    env = two_bowls()
    trace, privileged = serve(env, target="akita_black_bowl_2")
    bare = [{**entry, "commit": None, "state": None} for entry in trace]
    with pytest.raises(R.RolloutError, match="names no target"):
        R.relabel(bare, env.instruction, privileged)


# ------------------------------------ the model's own `rim` letter, replayed (round 1, again)
#
# The round died a second time, on the seventh episode of the first checkpoint served with `yaw`
# and `rim`:
#
#   ExpertError: in subgoal 'reach' the tracker steers to [-0.1651, 0.2632, 1.0184] and the
#   expert to [-0.1654, 0.3632, 1.0034] (grasp point [-0.1654, 0.3632, 0.9234] ...): they differ
#   by 0.0150 m
#
# 10 cm apart in y is two rim radii -- the two ends of one rim axis -- and 1.5 cm in z is one
# retry offset: the served tracker and the labelling expert were standing on different grasp
# candidates. `TrackerV2.answer` puts the executed `rim` letter into the carry that renders the
# state; nothing put it into the carry that labels it, so from the first selection that reads a
# letter the two were different plans. These tests are that failure, reduced.


def serve_until_retry(env, letter: str, *, decisions: int = 18, stall_from: int = 5,
                      target="akita_black_bowl_1", destination="plate_1"):
    """A served episode that **re-chooses a candidate mid-way**, with the model's letter obeyed.

    `serve` above never leaves the approach, and a candidate that is never re-selected hides
    exactly the bug this reproduces: the deferred first choice and the re-choice after a failed
    grasp are the two moments the plan reads `rim` out of its carry. So this one drives the hand
    onto the tracker's own waypoint until `stall_from` and then holds it still with the fingers
    open -- the close fails its outcome check, the candidate is marked spent, and the next
    selection is the one that has to obey the letter.

    The order is `serve`'s, which is `recipes/robojev/server.py`'s: observe, serialise, answer,
    hand the executed answers to the tracker.
    """
    from robojev.state import TrackerV2, serialise_v2

    answers = {**SERVED_ANSWERS, "rim": letter}
    tracker = TrackerV2(horizon=44, every=5, qids=questions_mod.ACTIVE_QIDS,
                        instruction=env.instruction)
    obs = env.reset(0)
    trace, privileged_seq = [], []
    for index in range(decisions):
        proprio = np.asarray(env.state_vector(obs), np.float64)
        privileged = env.privileged(obs)
        commit = None
        if index == 0:
            commit = {"target": target, "destination": destination, "source": "model"}
            tracker.commit(target, destination, step=0, source="model")
        tracker.observe(step=index * 5, proprio=proprio, objects=privileged)
        text = serialise_v2(proprio, privileged, env.instruction, tracker,
                            annotate=tracker.annotate)
        trace.append({"step": index * 5, "decision": index, "proprio": proprio, "state": text,
                      "answers": dict(answers), "commit": commit,
                      "latch": {"asked": False, "refused": False}})
        privileged_seq.append(privileged)
        tracker.answer(dict(answers))
        waypoint = tracker.waypoint
        step_m = env.PER_STEP_M * 5
        move = (np.zeros(3) if (index >= stall_from or waypoint is None)
                else np.clip((np.asarray(waypoint.point, np.float64) - proprio[0:3]) / step_m,
                             -1.0, 1.0))
        for _ in range(5):
            obs = env.step(np.array([move[0], move[1], move[2], 0.0, 0.0, 0.0, -1.0])).obs
    return trace, privileged_seq


def rim_sides(trace) -> list[str]:
    """The side of the target each served state names, out of its own `Waypoint (...)` line."""
    sides = []
    for entry in trace:
        line = next(l for l in entry["state"].splitlines() if l.startswith("Waypoint"))
        sides.append(line.split(" side)")[0].rsplit(",", 1)[-1].strip())
    return sides


@pytest.mark.parametrize("letter", ["A", "B"])
def test_the_round_replays_the_rim_letter_the_model_actually_executed(letter):
    """Round 1's second bug, both of the ways it arises.

    `A` is the model asking again for the direction it has just tried, so the plan retries the
    same axis one retry offset higher; `B` is the model standing on the far end of the axis from
    the start and the plan moving on when that fails. Either way the served states name the
    candidate the *letter* chose, and a labeller whose own carry never saw the letter labels a
    waypoint two rim radii away -- which `gold_for_state` refuses, as it should.
    """
    env = StubArm()
    trace, privileged = serve_until_retry(env, letter)
    sides = rim_sides(trace)
    assert len(set(sides)) > 1, f"the fixture never re-chose a candidate: {sides}"

    rows = R.relabel(trace, env.instruction, privileged)

    assert len(rows) == len(trace)
    for row, entry in zip(rows, trace):
        assert row["state"] == entry["state"]
        assert row["metadata"]["state_source"] == "policy"
        assert parse_mod.parse(row["state"], qids=tuple(row["gold"])) == row["gold"]
        # The two plans are one plan, decision for decision: the point the state names and the
        # point the labelling expert steers to are the same point, in the state's own
        # centimetres. This is the assertion the drift check makes loudly and this makes always.
        assert row["metadata"]["waypoint_cm"] == row["metadata"]["tracker_offset_cm"]


def test_a_labeller_that_drops_the_executed_rim_letter_is_caught_by_the_waypoint_check(
        monkeypatch):
    """The guard that found this, pinned. `_raise_on_drift` is the only thing that noticed a
    labeller standing on the wrong candidate -- both labels are well-formed, both re-parse -- so
    a relabeller that stops replaying the letter must still fail loudly rather than write a
    round's worth of rows about rim points nobody visited."""
    env = StubArm()
    trace, privileged = serve_until_retry(env, "A")
    monkeypatch.setattr(R.expert_mod, "with_executed",
                        lambda phase, answers, qids=None: phase)
    with pytest.raises(expert_mod.ExpertError, match="the tracker steers to"):
        R.relabel(trace, env.instruction, privileged)


def test_the_executed_rim_letter_lands_in_every_copy_of_the_plans_carry():
    """One rule, four carries (`run_episode`, `_Episode`, `TrackerV2` and the relabeller), and
    the rule is `TrackerV2.answer`'s: a run that is asked `rim` takes whatever letter came back,
    including none, and a run that is not asked it keeps its own order."""
    phase = expert_mod.new_phase({"target": "bowl", "destination": "plate"})
    assert phase["chose"] is None
    assert expert_mod.with_executed(phase, {"rim": "C"})["chose"] == "C"
    # No `rim` in the answers and none in the question set: the carry is untouched.
    assert expert_mod.with_executed(phase, {"move_x": "+"}) is phase
    assert expert_mod.with_executed(phase, {"move_x": "+"}, qids=("move_x",)) is phase
    # Asked, and unanswered: blanked, exactly as the tracker blanks its own, so the two carries
    # do not part company on the next selection.
    chosen = expert_mod.with_executed(phase, {"rim": "C"})
    assert expert_mod.with_executed(chosen, {"move_x": "+"}, qids=("move_x", "rim"))["chose"] is None
    # Nothing to mirror: a policy that reported no answers leaves the carry alone.
    assert expert_mod.with_executed(phase, None) is phase
    assert expert_mod.with_executed(None, {"rim": "C"}) is None


# ------------------------------------------------------------- the grounding augmentation (A)


def _fake_scenes(monkeypatch, n_tasks=3, n_init=5):
    """A scene cache the row builder can run over with no LIBERO and no simulator."""
    grounding = import_module("robojev.grounding")
    bddl = grounding.parse_bddl(
        "(define (problem x) (:objects akita_black_bowl_1 akita_black_bowl_2 - akita_black_bowl "
        "plate_1 - plate) (:obj_of_interest akita_black_bowl_1) (:goal (and (on "
        "akita_black_bowl_1 plate_1))))")
    scenes = []
    for task in range(n_tasks):
        for init in range(n_init):
            scenes.append({
                "suite": "libero_spatial", "task_index": task, "init_index": init,
                # The scenes a harvest builds from are the settled ones; an unsettled cache is
                # refused, so the fixture has to be what the real cache is.
                "settled_steps": grounding.SETTLED_STEPS,
                "task_name": f"pick_up_the_black_bowl_number_{task}_and_place_it_on_the_plate",
                "instruction": f"Pick the akita black bowl number {task} and place it on the plate",
                "entities": [
                    {"name": "akita_black_bowl_1", "category": "akita_black_bowl",
                     "kind": "object", "pos": [0.0 + init, 1.0, 0.0]},
                    {"name": "akita_black_bowl_2", "category": "akita_black_bowl",
                     "kind": "object", "pos": [10.0, 2.0 + init, 0.0]},
                    {"name": "plate_1", "category": "plate", "kind": "object",
                     "pos": [20.0, 3.0, 0.0]},
                ],
            })
    monkeypatch.setattr(grounding, "load_scenes", lambda *a, **k: scenes)
    monkeypatch.setattr(grounding, "bddl_task_from_record", lambda record: bddl)
    return scenes, grounding


def test_one_repeat_is_the_file_that_is_already_on_disk(tmp_path, monkeypatch):
    """The default must not move a byte: an existing directory re-harvests to the file it holds,
    which is what lets the augmentation be opt-in rather than a migration."""
    _scenes, grounding = _fake_scenes(monkeypatch)
    missing = tmp_path / "nothing.jsonl"
    cfg = config(grounding_rows=True, grounding_path=str(missing))
    rows = R.grounding_rows(cfg)
    assert R.RolloutConfig().grounding_repeats == 1
    assert all(row["id"].endswith(":grounding") for row in rows)
    assert {row["metadata"]["phrasing"] for row in rows} == {"bddl"}


def test_the_repeats_multiply_the_scenes_and_alternate_the_two_phrasings(tmp_path, monkeypatch):
    """N copies of every scene, half under the sentence a server asks with -- the measured gap
    that made run 6's grounding head wrong at p = 0.96 on the served wording."""
    scenes, _grounding = _fake_scenes(monkeypatch)
    cfg = config(grounding_rows=True, grounding_repeats=8,
                 grounding_path=str(tmp_path / "nothing.jsonl"))
    rows = R.grounding_rows(cfg)
    one = R.grounding_rows(config(grounding_rows=True,
                                  grounding_path=str(tmp_path / "nothing.jsonl")))
    assert len(rows) == 8 * len(one)
    phrasings = collections.Counter(row["metadata"]["phrasing"] for row in rows)
    assert phrasings == {"bddl": 4 * len(one), "served": 4 * len(one)}
    # Distinct ids, and every scene appears exactly N times.
    assert len({row["id"] for row in rows}) == len(rows)
    per_scene = collections.Counter(row["id"].rsplit(":g", 1)[0] for row in rows)
    assert set(per_scene.values()) == {8}
    assert {row["metadata"]["grounding_repeats"] for row in rows} == {8}
    # A served copy carries the sentence the simulator serves, not the BDDL's.
    served = next(r for r in rows if r["metadata"]["phrasing"] == "served")
    assert f"Task: {served['metadata']['served_instruction']}" in served["state"]
    assert served["metadata"]["served_instruction"] != served["metadata"]["bddl_instruction"]


def test_every_copy_of_a_sentence_lands_in_one_split(tmp_path, monkeypatch):
    """No leakage, under either wording: a grounding row of a sentence the model met in training
    proves nothing about grounding, and a paraphrase of it proves nothing either."""
    _scenes, grounding = _fake_scenes(monkeypatch)
    rows = R.grounding_rows(config(grounding_rows=True, grounding_repeats=6,
                                   grounding_path=str(tmp_path / "nothing.jsonl")))
    by_task = collections.defaultdict(set)
    for row in rows:
        by_task[row["metadata"]["task_index"]].add(row["split"])
    for task, splits in by_task.items():
        assert len(splits) == 1, (task, splits)
    for row in rows:
        for key in grounding.scene_keys({"instruction": row["metadata"]["bddl_instruction"],
                                         "task_name": row["metadata"]["task_name"]}):
            assert grounding.instruction_key(row["metadata"]["instruction"]) in (
                key, grounding.instruction_key(row["metadata"]["instruction"]))


def test_the_manifest_echoes_the_repeat_count(tmp_path, monkeypatch):
    _scenes, _grounding = _fake_scenes(monkeypatch)
    manifest = harvest(tmp_path, config(grounding_rows=True, grounding_repeats=4,
                                        grounding_path=str(tmp_path / "nothing.jsonl")))
    assert manifest["grounding_repeats"] == 4
    assert manifest["grounding_phrasings"] == ["bddl", "served"]
    assert manifest["rows"]["grounding"] == 4 * 15


def test_grounding_only_rewrites_the_file_and_leaves_the_task_rows_alone(tmp_path, monkeypatch):
    """The augmentation of an existing directory is a file rewrite: no simulator, no relabelling,
    and every harvested motion row still byte for byte where it was."""
    _scenes, _grounding = _fake_scenes(monkeypatch)
    cfg = config(grounding_rows=True, grounding_path=str(tmp_path / "nothing.jsonl"))
    harvest(tmp_path, cfg)
    before = (tmp_path / R.task_file(0)).read_bytes()
    manifest = R.harvest_expert(
        dataclasses.replace(cfg, grounding_repeats=8), tmp_path, log=lambda _m: None,
        env_factory=None, grounding_only=True)
    assert (tmp_path / R.task_file(0)).read_bytes() == before
    assert manifest["grounding_repeats"] == 8
    assert manifest["rows"]["grounding"] == 8 * 15
    assert manifest["rows"]["expert_rollout"] == 3


def test_an_unsettled_scene_cache_is_refused(tmp_path, monkeypatch):
    """A cache captured straight after `set_init_state` describes a scene no server grounds
    against -- on LIBERO-Spatial task 3 the ten no-op steps flip the bowl's relation from "on top
    of cookies_1" to "inside cookies_1". Building rows from it would bake that into the file, so
    it raises with the command that rebuilds it."""
    scenes, grounding = _fake_scenes(monkeypatch)
    stale = [{**scene, "settled_steps": 0} for scene in scenes]
    monkeypatch.setattr(grounding, "load_scenes", lambda *a, **k: stale)
    with pytest.raises(R.RolloutError, match="no-op steps"):
        R.grounding_rows(config(grounding_rows=True, grounding_repeats=2,
                                grounding_path=str(tmp_path / "nothing.jsonl")))


def test_the_manifest_records_which_scene_cache_the_rows_came_from(tmp_path, monkeypatch):
    _scenes, grounding = _fake_scenes(monkeypatch)
    manifest = harvest(tmp_path, config(grounding_rows=True, grounding_repeats=2,
                                        grounding_path=str(tmp_path / "nothing.jsonl")))
    assert manifest["grounding_scenes"] == str(grounding.SETTLED_SCENE_CACHE)
    assert manifest["grounding_settled_steps"] == grounding.SETTLED_STEPS
