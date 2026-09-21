"""One DAgger round, driven against fakes: a fake policy, a fake LIBERO-shaped backend, and the
**real** `robojev.episode.run_episode` between them.

The episode loop is not stubbed out, because the whole claim of `robojev.dagger` is that
its rows describe the states a graded run of the same checkpoint would visit -- same wait steps,
same chunking, same stop-on-success. Testing the recorder against a hand-rolled loop would test
the hand-rolled loop. Nothing needs stubbing for that: the loop's only simulator-shaped import is
a seeding helper it can do without.

What is checked here: that a row validates against exactly the rules a harvested row does (the
same vendored `validate_training_row` the harvest tests use), that its `state` is the text the
policy was handed rather than a reconstruction, that its labels are read off the *visited* scene,
that β mixing is reproducible from the seed alone, and that a merge preserves the harvest's bytes
and every task's split. The `sim`-marked test at the end runs two real episodes of the real
simulator with the scripted policy, and skips when LIBERO is not installed.
"""
from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

from robojev import episode as protocol_mod
from robojev import dataset as dataset_mod
from robojev import dagger as D
from robojev.compose import CHUNK_STEPS, COMMANDED_CM_PER_UNIT
from test_dataset import validate_training_row     # tests/ is on sys.path: no __init__.py, by design

BOWL = "akita_black_bowl_1"
PLATE = "plate_1"
INSTRUCTION = "pick up the black bowl next to the cookie box and place it on the plate"

#: The bowl sits here in every fake scene, and every expected label below is arithmetic on it.
BOWL_POS = np.array([0.0, 0.0, 1.00])
PLATE_POS = np.array([0.3, 0.2, 0.90])
DELTA_T, DELTA_R = 1.0, 0.0579

#: The question block a harvested row carries, reduced to the two questions these fakes answer:
#: what `merge` has to keep byte for byte is the *bytes*, not a particular set.
HARVEST_QUESTIONS = {
    "move_z": {"type": "choice", "instructions": "", "criteria": {"-": "", "hold": "", "+": ""}},
    "grip": {"type": "boolean", "instructions": ""},
}


# ------------------------------------------------------------------------------------- fakes

class FakeBackend:
    """A LIBERO-shaped backend whose whole physics is "the hand goes where it is told".

    `eef` is the end-effector position, `width` the finger opening, and an action moves the first
    by its first three components scaled the way `OSC_MAX_DPOS` scales a real one; slot 6 opens or
    shuts the fingers. That is enough for the recorder to see a hand that travels, a gripper that
    closes and a scene whose object poses the waypoint can be measured against.
    """

    action_dim = 7
    camera_names = ["agentview"]
    control_freq = 20
    stop_on_success = True
    state_layout = None

    def __init__(self, suite="libero_spatial", task_index=0, eef=(0.0, 0.0, 0.90),
                 width=0.08, succeed_at=None, objects=None):
        self.suite = suite
        self.task_index = task_index
        self.instruction = INSTRUCTION
        self.start = np.asarray(eef, float)
        self.eef = self.start.copy()
        self.width = float(width)
        self.succeed_at = succeed_at
        self.steps = 0
        self.closed = False
        self.objects = objects if objects is not None else {BOWL: BOWL_POS, PLATE: PLATE_POS}
        # `harvest.grip_target` reads this off the env before it reaches for the BDDL.
        self.obj_of_interest = [BOWL, PLATE]

    # -- the SimBackend surface ------------------------------------------------------------
    def reset(self, init_state_index: int) -> dict:
        self.eef = self.start.copy() + np.array([0.0, 0.0, 0.01 * init_state_index])
        self.steps = 0
        return self._obs()

    def step(self, action):
        from robojev.envs import StepResult

        action = np.asarray(action, float)
        self.steps += 1
        self.eef = self.eef + (COMMANDED_CM_PER_UNIT / 100.0 / CHUNK_STEPS) * action[0:3]
        self.width = 0.0 if action[6] > 0 else 0.08
        return StepResult(obs=self._obs(), reward=0.0, info={},
                          done=self.succeed_at is not None and self.steps >= self.succeed_at)

    def _obs(self) -> dict:
        state = np.concatenate([self.eef, [0.0, 0.0, 0.0],
                                [self.width / 2, -self.width / 2]]).astype(np.float32)
        return {"state": state, "images": {"agentview": np.zeros((4, 4, 3), np.uint8)}}

    def images(self, obs):
        return dict(obs["images"])

    def state_vector(self, obs):
        return np.asarray(obs["state"], np.float32)

    def privileged(self, obs):
        return {name: {"pos": np.asarray(pos, np.float32), "quat": np.array([1.0, 0, 0, 0], np.float32)}
                for name, pos in self.objects.items()}

    def ground_truth(self):
        return np.zeros(9, np.float32), np.zeros(9, np.float32), self.steps / 20.0, None

    def dummy_action(self):
        return [0.0] * self.action_dim

    def scene_xml(self):
        return "<mujoco/>"

    def close(self):
        self.closed = True


class FakePolicy:
    """A decision policy the shape `robojev.policy.Policy` presents.

    `choices` is the fixed answer it gives to every state. It reports **no** prompt of its own,
    which is the case a round has to refuse: a served checkpoint always carries
    `decisions.meta.state` and the relabeller asserts its rebuild against it byte for byte.
    """

    CHOICES = {"move_x": "+", "move_y": "hold", "move_z": "-", "size_x": "large",
               "size_y": "small", "size_z": "medium", "yaw": "hold", "rim": "A",
               "grip": "false", "subgoal": "reach"}

    execute_steps = CHUNK_STEPS
    wants_privileged = True

    def __init__(self, choices=None, reports_state=False, delta=(DELTA_T, DELTA_R)):
        self.choices = dict(choices or self.CHOICES)
        self.reports_state = reports_state
        self.delta = delta
        self.last_decisions = None
        self.instruction = ""
        self.seen: list[dict] = []
        self._step = 0
        self._tracker = None

    def reset(self, instruction: str) -> None:
        from robojev.state import TrackerV2

        self.instruction = instruction
        self._step = 0
        self._tracker = TrackerV2(horizon=44, instruction=instruction or None)

    def describe(self):
        return {"policy": "fake"}

    def act(self, obs, *args, **kwargs):
        from robojev.roles import scene_roles
        from robojev.state import serialise_v2

        self.seen.append(obs)
        privileged = obs["privileged"]
        proprio = np.asarray(obs["state"], np.float32)
        self.last_decisions = {
            qid: {"choice": choice, "probabilities": {}, "overridden": False}
            for qid, choice in self.choices.items()
        }
        if self.reports_state:
            if self._tracker.target is None:
                roles = scene_roles(privileged, self.instruction, proprio[0:3])
                self._tracker.commit(roles["target"], roles["destination"], step=0, source="rule")
            self._tracker.observe(step=self._step * CHUNK_STEPS, proprio=proprio,
                                  objects=privileged)
            self.last_decisions["meta"] = {
                "state_id": f"fake:{self._step}",
                "state": serialise_v2(proprio, privileged, self.instruction, self._tracker),
                "step": self._step,
                "grounding": ({"target": self._tracker.target,
                               "destination": self._tracker.destination,
                               "source": "rule"} if self._step == 0 else None),
            }
            self._tracker.answer(dict(self.choices))
        self._step += 1
        return np.zeros((CHUNK_STEPS, 7), np.float32), self.last_decisions


def fake_manifest(**over) -> dict:
    """A harvest manifest with everything `config_from_manifest` reads off one."""
    manifest = {
        "suite": "libero_spatial",
        "delta_t": DELTA_T, "delta_r": DELTA_R,
        "every": 5, "chunk_steps": 5, "fine_label_threshold": 0.5,
        "move_floor": 0.005, "yaw_floor": 0.05, "seed": 17,
        "label_source": "expert_tracker",
        "questions_version": "v2",
        "memory": {"memory_rule": "v2-tracker-1", "horizon": 44, "every": 5,
                   "history_k": 3, "max_events": 8, "chunk_steps": 5,
                   "target_rule": "obj_of_interest"},
        "memory_rule": "v2-tracker-1", "horizon": 44, "history_k": 3,
        "max_events": 8, "target_rule": "obj_of_interest",
        "splits": {"train": [0, 1], "dev": [2], "test": [3]},
        "created_at": "2026-09-21T00:00:00+00:00",
    }
    manifest.update(over)
    return manifest


def cfg(**over) -> D.DaggerConfig:
    return D.config_from_manifest(fake_manifest(), **over)


def collect(tmp_path, policy=None, backend=None, tasks=(0,), inits=(0,), seed=17,
            max_steps=20, out=None, **over):
    """One round, into `tmp_path/dagger` unless told otherwise."""
    policy = policy if policy is not None else FakePolicy(reports_state=True)
    made: list[FakeBackend] = []

    def env_factory(suite, task_index):
        env = backend if backend is not None else FakeBackend(suite, task_index)
        made.append(env)
        return env

    protocol = protocol_mod.for_suite("libero_spatial", max_steps=max_steps)
    out = out if out is not None else (tmp_path / "dagger")
    manifest = D.collect(
        policy, "libero_spatial", tasks, inits, out, seed=seed,
        cfg=cfg(seed=seed, **over), env_factory=env_factory, protocol=protocol,
        execute_steps=CHUNK_STEPS, run=protocol_mod.run_episode,
        log=lambda *a, **k: None,
    )
    return manifest, out, made


def rows_of(out) -> list[dict]:
    return list(D.rows_of(out))


# ------------------------------------------------------------------------------- the row shape

def test_a_dagger_row_validates_against_the_same_rules_a_harvested_row_does(tmp_path):
    _, out, _ = collect(tmp_path)
    rows = rows_of(out)
    assert rows
    for row in rows:
        validate_training_row(row)
        dataset_mod.check_probs(row)


def test_a_row_answers_the_question_set_the_harvest_answers(tmp_path):
    """The whole premise of the round: one label definition in a merged directory."""
    row = rows_of(collect(tmp_path)[1])[0]
    assert set(row["gold"]) <= {"move_x", "move_y", "move_z", "size_x", "size_y", "size_z",
                                "yaw", "rim", "grip", "subgoal"}
    assert set(row["gold"]) == set(row["questions"])
    assert sorted(row["gold_probs"]["grip"].values()) == [0.0, 1.0]


def test_the_row_id_scheme_names_the_source_the_task_the_init_and_the_round(tmp_path):
    _, out, _ = collect(tmp_path, tasks=(1,), inits=(3,))
    rows = rows_of(out)
    assert all(r["id"].startswith("dagger_expert:libero_spatial:1:3:r1:") for r in rows)
    assert rows[0]["state_id"] == rows[0]["id"]
    assert all(r["metadata"]["source"] == "dagger_expert" for r in rows)


def test_a_row_is_json_and_survives_a_round_trip(tmp_path):
    row = rows_of(collect(tmp_path)[1])[0]
    assert json.loads(json.dumps(row, sort_keys=True)) == row


def test_every_row_of_an_episode_carries_that_episodes_outcome(tmp_path):
    """The outcome is not known while the rows are made, so it is stamped on afterwards -- and it
    has to be on *every* row, or a later round cannot weight them by whether the attempt worked."""
    _, out, _ = collect(tmp_path, backend=FakeBackend(succeed_at=7), max_steps=30)
    rows = rows_of(out)
    assert all(r["metadata"]["outcome"]["success"] is True for r in rows)
    assert all(r["metadata"]["outcome"]["terminated_by"] == "success" for r in rows)

    _, out2, _ = collect(tmp_path, out=tmp_path / "b", max_steps=15)
    assert all(r["metadata"]["outcome"]["success"] is False for r in rows_of(out2))


# ------------------------------------------------------------------- the state is the policy's

def test_the_state_is_the_text_the_policy_was_given_byte_for_byte(tmp_path):
    """The whole point of the round. `decisions.meta.state` is the prompt the server built,
    tracker block included; a row that rebuilt it would carry a history nothing ever read."""
    policy = FakePolicy(reports_state=True)
    _, out, _ = collect(tmp_path, policy=policy)
    rows = rows_of(out)
    assert all(r["metadata"]["state_source"] == "policy" for r in rows)
    assert all(r["state"].startswith("Task:") or "Waypoint" in r["state"] for r in rows)


def test_a_policy_that_reports_no_prompt_gets_one_rendered_and_the_row_says_so(tmp_path):
    """A served checkpoint always reports its prompt; a fake or a non-decision baseline does not,
    and a row built from a reconstruction must never be mistaken for one built from what the
    model read."""
    _, out, _ = collect(tmp_path, policy=FakePolicy(reports_state=False))
    rows = rows_of(out)
    assert rows and all(r["metadata"]["state_source"] == "rebuilt" for r in rows)


def test_the_manifest_counts_the_states_it_read(tmp_path):
    manifest, _, _ = collect(tmp_path)
    assert manifest["state_source"] == {"policy": manifest["rows"]["total"]}
    manifest, _, _ = collect(tmp_path, out=tmp_path / "b",
                             policy=FakePolicy(reports_state=False))
    assert manifest["state_source"] == {"rebuilt": manifest["rows"]["total"]}


def test_the_policys_own_answer_travels_beside_the_label(tmp_path):
    """The round's yield is the disagreement between the two, and recovering it later would mean
    re-running the rollout."""
    _, out, _ = collect(tmp_path, policy=FakePolicy(reports_state=True))
    row = rows_of(out)[0]
    assert row["metadata"]["policy_choice"]["move_x"] == "+"


def test_a_manifest_without_the_deltas_is_refused():
    with pytest.raises(ValueError, match="delta_t"):
        D.config_from_manifest({"suite": "libero_spatial"})


def test_the_tracker_settings_come_from_the_harvest_not_from_defaults():
    """A row labelled against defaults is a second dataset: the state a served checkpoint reads
    is rendered under the harvest's own settings, and the round's rows have to match them."""
    c = D.config_from_manifest(fake_manifest(memory={"horizon": 60, "every": 5, "history_k": 5,
                                                     "max_events": 4, "chunk_steps": 5}))
    assert (c.horizon, c.history_k, c.max_events) == (60, 5, 4)
    assert c.splits == {0: "train", 1: "train", 2: "dev", 3: "test"}


# ----------------------------------------------------------------------------------- the round

def test_the_round_writes_one_file_per_task_in_the_harvests_layout(tmp_path):
    manifest, out, _ = collect(tmp_path, tasks=(0, 2), inits=(0, 1))
    assert sorted(p.name for p in out.glob("*.jsonl")) == ["task_0.jsonl", "task_2.jsonl"]
    assert (out / D.MANIFEST).is_file()
    assert manifest["tasks"] == [0, 2]
    assert manifest["init_states"] == [0, 1]
    assert len(manifest["episodes"]) == 4
    assert manifest["rows"]["total"] == sum(r["rows"] for r in manifest["per_task"].values())


def test_a_tasks_rows_inherit_that_tasks_split(tmp_path):
    """Task 2 is `dev` in the fake harvest manifest, and a DAgger row of task 2 is a row of task
    2: the trainer refuses a source group that straddles a split."""
    manifest, out, _ = collect(tmp_path, tasks=(0, 2, 3))
    assert manifest["splits"] == {"train": [0], "dev": [2], "test": [3]}
    by_task = {r["metadata"]["task_index"]: r["split"] for r in rows_of(out)}
    assert by_task == {0: "train", 2: "dev", 3: "test"}


def test_one_env_is_built_per_task_and_closed_after_it(tmp_path):
    _, _, made = collect(tmp_path, tasks=(0, 1), inits=(0, 1))
    assert len(made) == 2 and all(env.closed for env in made)


def test_a_round_needs_the_harvest_it_will_join(tmp_path):
    with pytest.raises(ValueError, match="harvest_dir"):
        D.collect(FakePolicy(), "libero_spatial", [0], [0], tmp_path / "x")
    with pytest.raises(ValueError, match="no manifest.json"):
        D.read_manifest(tmp_path)


def test_an_empty_task_or_init_list_is_refused(tmp_path):
    with pytest.raises(ValueError, match="at least one task"):
        D.collect(FakePolicy(), "libero_spatial", [], [0], tmp_path / "x", cfg=cfg())


def test_the_round_refuses_to_invent_the_protocol_or_the_episode_loop(tmp_path):
    """Both are the host's. A default invented here would be a second definition of an episode,
    and the round's whole claim is that its states are the ones a graded run visits."""
    with pytest.raises(ValueError, match="protocol"):
        D.collect(FakePolicy(), "libero_spatial", [0], [0], tmp_path / "x", cfg=cfg())
    protocol = protocol_mod.for_suite("libero_spatial", max_steps=20)
    with pytest.raises(ValueError, match="episode loop"):
        D.collect(FakePolicy(), "libero_spatial", [0], [0], tmp_path / "x", cfg=cfg(),
                  protocol=protocol, env_factory=lambda s, t: FakeBackend(s, t))


def test_the_manifest_reports_the_success_rate_and_the_marginals(tmp_path):
    manifest, out, _ = collect(tmp_path, backend=FakeBackend(succeed_at=7), max_steps=30)
    assert manifest["success_rate"] == 1.0
    assert sum(manifest["grip"].values()) == manifest["rows"]["total"]
    assert manifest["labeller"] == D.LABELLER
    assert manifest["questions_version"] == "v2"


# ------------------------------------------------------------------------------------ the merge

def write_harvest(tmp_path, tasks=(0, 2), rows_per_task=3) -> "pathlib.Path":
    """A harvest directory with real-shaped rows, so the merge has bytes to preserve."""
    import pathlib

    out = pathlib.Path(tmp_path) / "harvest"
    out.mkdir(parents=True, exist_ok=True)
    manifest = fake_manifest(tasks=list(tasks), splits={"train": [0], "dev": [2], "test": [3]})
    splits = {0: "train", 2: "dev", 3: "test"}
    for task in tasks:
        lines = []
        for i in range(rows_per_task):
            row = {
                "id": f"libero_spatial:{task}:0:{i * 5}",
                "state_id": f"libero_spatial:{task}:0:{i * 5}",
                "family_id": "libero_spatial",
                "split": splits[task],
                "state": f"Robot state. row {i}",
                "questions": HARVEST_QUESTIONS,
                "gold": {"move_z": "+", "grip": False},
                "gold_probs": {
                    "move_z": {"-": 0.0, "hold": 0.0, "+": 1.0},
                    "grip": {"false": 1.0, "true": 0.0},
                },
                "gold_probs_kind": {q: dataset_mod.PROBS_KIND for q in HARVEST_QUESTIONS},
                "gold_label_kind": {q: "reference_argmax_compatibility"
                                   for q in HARVEST_QUESTIONS},
                "metadata": {"source_group_id": f"libero_spatial:{task}:0", "task_index": task},
            }
            lines.append(json.dumps(row, sort_keys=True))
        (out / f"task_{task}.jsonl").write_text("".join(line + "\n" for line in lines),
                                                encoding="utf-8")
    (out / dataset_mod.FRAMES).write_text("", encoding="utf-8")
    (out / D.MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return out


def test_the_merge_keeps_every_harvest_row_byte_for_byte(tmp_path):
    harvest = write_harvest(tmp_path)
    _, dagger, _ = collect(tmp_path, tasks=(0, 2))
    merged = D.merge(harvest, dagger, tmp_path / "merged")
    for task in (0, 2):
        before = (harvest / f"task_{task}.jsonl").read_bytes()
        after = (tmp_path / "merged" / f"task_{task}.jsonl").read_bytes()
        assert after.startswith(before), "the harvest's bytes are a prefix of the merged file"
        assert after[len(before):], "the DAgger rows are appended after them"
    assert merged["rows"]["by_source"]["harvest"] == 6
    assert merged["rows"]["by_source"]["dagger"] == merged["dagger"]["rows"]["total"]
    assert merged["rows"]["total"] == 6 + merged["dagger"]["rows"]["total"]


def test_the_merge_preserves_every_tasks_split(tmp_path):
    harvest = write_harvest(tmp_path)
    _, dagger, _ = collect(tmp_path, tasks=(0, 2))
    merged = D.merge(harvest, dagger, tmp_path / "merged")
    assert merged["splits"] == {"train": [0], "dev": [2]}
    for row in D.rows_of(tmp_path / "merged"):
        expected = {0: "train", 2: "dev"}[row["metadata"]["task_index"]]
        assert row["split"] == expected


def test_the_merged_directory_is_what_the_trainer_globs(tmp_path):
    """`train.merge_rows` globs `task_*.jsonl` and parses the task out of the stem; a merged
    directory that did not answer that glob would train on the harvest alone, silently."""
    harvest = write_harvest(tmp_path)
    _, dagger, _ = collect(tmp_path, tasks=(0, 2))
    D.merge(harvest, dagger, tmp_path / "merged")
    files = sorted(p.name for p in (tmp_path / "merged").glob("task_*.jsonl"))
    assert files == ["task_0.jsonl", "task_2.jsonl"]
    ids = [r["id"] for r in D.rows_of(tmp_path / "merged")]
    assert len(ids) == len(set(ids)), "no state_id may appear twice (the trainer refuses it)"
    assert any(i.startswith("dagger_expert:") for i in ids)
    assert any(i.startswith("libero_spatial:") for i in ids)


def test_the_merge_counts_each_source_per_task(tmp_path):
    harvest = write_harvest(tmp_path)
    _, dagger, _ = collect(tmp_path, tasks=(0, 2))
    merged = D.merge(harvest, dagger, tmp_path / "merged")
    for task in ("0", "2"):
        record = merged["per_task_by_source"][task]
        assert record["by_source"]["harvest"] == 3
        assert record["by_source"]["dagger"] == merged["dagger"]["per_task"][task]["rows"]
        assert record["rows"] == 3 + record["by_source"]["dagger"]


def test_the_merge_refuses_to_write_into_either_input(tmp_path):
    harvest = write_harvest(tmp_path)
    _, dagger, _ = collect(tmp_path, tasks=(0,))
    with pytest.raises(ValueError, match="third directory"):
        D.merge(harvest, dagger, harvest)
    with pytest.raises(ValueError, match="third directory"):
        D.merge(harvest, dagger, dagger)


def test_the_merge_refuses_rows_built_two_different_ways(tmp_path):
    harvest = write_harvest(tmp_path)
    manifest = json.loads((harvest / D.MANIFEST).read_text())
    manifest["memory_rule"] = "run4-memory-0"
    (harvest / D.MANIFEST).write_text(json.dumps(manifest))
    _, dagger, _ = collect(tmp_path, tasks=(0,))
    with pytest.raises(ValueError, match="built differently"):
        D.merge(harvest, dagger, tmp_path / "merged")


def test_the_merge_carries_the_frames_sidecar_across(tmp_path):
    harvest = write_harvest(tmp_path)
    (harvest / dataset_mod.FRAMES).write_text('{"row_id": "libero_spatial:0:0:0", "images": {}}\n')
    _, dagger, _ = collect(tmp_path, tasks=(0,))
    merged = D.merge(harvest, dagger, tmp_path / "merged")
    assert (tmp_path / "merged" / dataset_mod.FRAMES).read_text() == (harvest / dataset_mod.FRAMES).read_text()
    # The JPEG tree is gigabytes and is not duplicated, so the manifest points back at it rather
    # than leaving the sidecar indexing files that are not beside it.
    assert merged["frames"]["dir"] == str(harvest / dataset_mod.FRAMES_DIR)
    assert not (tmp_path / "merged" / dataset_mod.FRAMES_DIR).exists()


def test_the_merge_refuses_a_dagger_row_whose_split_disagrees_with_the_harvest(tmp_path):
    harvest = write_harvest(tmp_path)
    _, dagger, _ = collect(tmp_path, tasks=(0,))
    path = dagger / "task_0.jsonl"
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    lines[0]["split"] = "test"
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in lines))
    with pytest.raises(ValueError, match="crosses splits|is split"):
        D.merge(harvest, dagger, tmp_path / "merged")


# --------------------------------------------------------------------------------------- the CLI

def test_the_index_parser_takes_a_range_as_well_as_a_list():
    from robojev.cli import indices as _indices

    assert _indices("0-7") == list(range(8))
    assert _indices("0,2,5") == [0, 2, 5]
    assert _indices("0-2,7,9-10") == [0, 1, 2, 7, 9, 10]
    assert _indices(" ") == []


def test_merge_only_merges_an_existing_round_without_a_policy_or_a_simulator(tmp_path, capsys):
    """The half of the verb that needs neither: `--merge-only` is what to run when the rollout
    finished and the merge did not."""
    from robojev import cli

    harvest = write_harvest(tmp_path)
    _, dagger, _ = collect(tmp_path, tasks=(0, 2))
    merged = tmp_path / "merged"
    code = cli.main(["dagger", "--suite", "libero_spatial", "--merge-only",
                     "--data", str(harvest), "--out", str(dagger), "--merge", str(merged)])
    assert code == 0
    assert capsys.readouterr().out.strip().endswith("merged")
    assert (merged / "task_0.jsonl").read_bytes().startswith(
        (harvest / "task_0.jsonl").read_bytes())


def test_the_verb_refuses_a_data_directory_with_no_harvest_manifest(tmp_path, capsys):
    from robojev import cli

    code = cli.main(["dagger", "--merge-only", "--data", str(tmp_path),
                     "--out", str(tmp_path / "round")])
    assert code == 2
    assert "manifest.json" in capsys.readouterr().err


def test_the_verb_refuses_an_empty_task_selection(tmp_path, capsys):
    from robojev import cli

    harvest = write_harvest(tmp_path)
    code = cli.main(["dagger", "--data", str(harvest), "--tasks", "", "--inits", "0"])
    assert code == 2
    assert "selected nothing" in capsys.readouterr().err


# -------------------------------------------------------------------------------------- the sim

@pytest.mark.sim
def test_two_real_episodes_of_the_heuristic_policy_produce_labelled_rows(tmp_path):
    """The whole path with the real simulator: `LiberoEnv`, `run_episode`, and the scripted
    policy as the one being corrected. Skipped where LIBERO is not installed."""
    pytest.importorskip("libero")
    from robojev import policy as policy_mod
    from robojev.envs.libero import LiberoEnv

    policy = policy_mod.build("expert", "libero_spatial")
    protocol = protocol_mod.for_suite("libero_spatial", max_steps=30,
                                      wait_steps=policy.wait_steps)
    manifest = D.collect(
        policy, "libero_spatial", [0], [0, 1], tmp_path / "round",
        beta=0.0, seed=17, cfg=cfg(),
        env_factory=lambda suite, task: LiberoEnv(suite, task),
        protocol=protocol, execute_steps=protocol.execute_steps,
        run=protocol_mod.run_episode, log=lambda *a, **k: None,
    )
    policy.close()
    rows = list(D.rows_of(tmp_path / "round"))
    assert len(manifest["episodes"]) == 2 and rows
    for row in rows:
        validate_training_row(row)
    assert {r["metadata"]["state_source"] for r in rows} == {"policy"}
    assert all("Subgoal:" in r["state"] for r in rows)


# ------------------------------------------------------- the expert labeller (plan 9c, task 6)
#
# `--labeller expert` replaces this module's geometric waypoint rule with
# `robojev.expert.gold_for_state` against a v2 tracker rebuilt from the recorded
# observations -- the *same* label source the v2 rows are harvested with, which is what makes the
# round an addition to that dataset rather than a second one.


def _v2parse():
    """`import_module`: `robojev` re-exports a *function* called `parse`."""
    from importlib import import_module

    return import_module("robojev.parse")


def test_the_expert_labeller_answers_on_the_policys_own_states(tmp_path):
    """The states are the model's, the history block is the model's, and the label is what the
    expert would have answered there -- which a rule over the state text alone reproduces."""
    _manifest, out, _made = collect(tmp_path)
    rows = rows_of(out)
    assert rows
    for row in rows:
        assert set(row["gold"]) <= {"move_x", "move_y", "move_z", "size_x", "size_y", "size_z",
                                    "yaw", "rim", "grip", "subgoal"}
        assert row["metadata"]["history_source"] == "policy"
        assert row["metadata"]["source"] == "dagger_expert"
        # Gate G3's invariant, on a DAgger row: the text reproduces the label.
        assert _v2parse().parse(row["state"], qids=tuple(row["gold"])) == row["gold"]


def test_the_expert_labeller_is_the_harvests_own_label_function(tmp_path):
    """Not "an expert" but *the* expert: `relabel_with_expert` is `v2.rollout.relabel`, which is
    the function the harvested rows come out of. Two label definitions in one merged directory is
    exactly what this is arranged to prevent."""
    from robojev import rollout

    backend = FakeBackend()
    obs = backend.reset(0)
    trace, privileged = [], []
    for index in range(3):
        trace.append({"step": index * 5, "proprio": backend.state_vector(obs), "answers": {}})
        privileged.append(backend.privileged(obs))
        for _ in range(5):
            obs = backend.step(np.array([0.5, 0.5, -0.5, 0, 0, 0, -1.0])).obs

    roles = {"target": BOWL, "destination": PLATE, "source": "obj_of_interest"}
    theirs = D.relabel_with_expert(trace, INSTRUCTION, privileged, cfg=cfg(),
                                   task_index=0, init_state=0, split="train", roles=roles)
    mine = rollout.relabel(trace, INSTRUCTION, privileged, roles=roles, task_index=0,
                           init_state=0, episode=1, split="train")
    assert [row["gold"] for row in theirs] == [row["gold"] for row in mine]
    assert len(theirs) == 3


def test_the_labeller_is_part_of_a_rounds_shape(tmp_path):
    """A directory holding rows labelled two different ways holds two datasets, so the manifest
    says which rule wrote it."""
    manifest, _out, _made = collect(tmp_path)
    assert manifest["labeller"] == D.LABELLER == "expert"
    assert cfg().shape()["labeller"] == "expert"


def test_the_round_reports_its_successes_per_episode_and_per_task(tmp_path):
    """A DAgger round is a closed-loop measurement of the checkpoint being corrected, and a round
    whose successes are only in its stderr is a round with no record. The manifest carries the
    same `success` block the harvest's does, plus one entry per episode."""
    backend = FakeBackend(succeed_at=10)
    manifest, _out, _made = collect(tmp_path, backend=backend, tasks=(0,), inits=(0, 1))
    assert manifest["success"]["episodes"] == 2
    assert manifest["success"]["successes"] == 2
    assert manifest["success"]["rate"] == 1.0
    assert manifest["success"]["by_task"]["0"] == {"successes": 2, "episodes": 2,
                                                  "split": "train"}
    assert [e["success"] for e in manifest["episodes"]] == [True, True]
    assert all("init_state" in e and "steps" in e for e in manifest["episodes"])


#: A policy whose fixed answers disagree with the expert's often enough that a round of it has
#: real disagreements to count.
V2Policy = FakePolicy


def test_the_round_reports_where_the_policy_disagreed_with_the_labeller(tmp_path):
    """The round's own yield: per question, how often the executed answer was not the label.
    Without it the only way to read a round is to re-derive it from the rows."""
    manifest, out, _made = collect(tmp_path, policy=V2Policy(reports_state=True))
    assert any(r["differs"] for r in manifest["disagreement"].values()), manifest["disagreement"]
    block = manifest["disagreement"]
    assert block, manifest
    rows = rows_of(out)
    for qid, record in block.items():
        assert record["asked"] == sum(1 for r in rows if qid in r["questions"])
        assert record["differs"] == sum(1 for r in rows if qid in r["metadata"].get("disagrees", ()))
        assert 0.0 <= record["rate"] <= 1.0


# ----------------------------------------------------------- merging a v2 round (plan 9c, A/B)


def test_a_rounds_shape_says_which_question_set_its_rows_answer(tmp_path):
    """The round's rows are `TrackerV2` states labelled by `expert.gold_for_state`, and the shape
    the manifest reports has to be the shape they are -- reporting another question set's
    `memory_rule` made `merge` refuse a round against the very harvest it was collected from."""
    shape = cfg().shape()
    assert shape["memory_rule"] == "v2-tracker-1"
    assert shape["label_source"] == "expert_tracker"
    assert shape["questions_version"] == "v2"
    # There is no `grip_label`: `grip` is the latch the question itself asks for.
    assert "grip_label" not in shape
    assert shape["row_source"] == "dagger_expert"


def test_an_older_round_is_read_by_its_labeller_and_not_by_its_stale_shape():
    """A round collected before the shape was corrected carries the retired set's two strings for
    rows that are this one's. `labeller` is the fact; re-rolling an hour of episodes to fix a
    manifest is not."""
    stale = {"labeller": "expert", "memory_rule": "run5-memory-2",
             "label_source": "goal", "grip_label": "latch", "round": 1}
    fixed = D.effective_shape(stale)
    assert fixed["memory_rule"] == "v2-tracker-1"
    assert fixed["label_source"] == "expert_tracker"
    assert fixed["questions_version"] == "v2"
    assert "grip_label" not in fixed
    # A round by any other labeller is left exactly as it is.
    other = {"labeller": "geometric", "memory_rule": "run5-memory-2"}
    assert D.effective_shape(other) == other


def _v2_harvest(tmp_path, rows=2):
    """A minimal v2 harvest directory: one task file, a grounding file and a v2 manifest."""
    harvest = tmp_path / "harvest"
    harvest.mkdir()
    (harvest / "task_0.jsonl").write_text(
        "".join(json.dumps({"id": f"expert:libero_spatial:0:0:0:{i}", "split": "train",
                            "state": "s", "questions": {}, "gold": {}, "gold_probs": {},
                            "gold_probs_kind": {}, "gold_label_kind": {},
                            "metadata": {"source": "expert_rollout"}}, sort_keys=True) + "\n"
                for i in range(rows)))
    (harvest / "grounding.jsonl").write_text(
        json.dumps({"id": "libero_90:1:0:grounding:g0", "split": "test", "state": "g",
                    "questions": {}, "gold": {}, "gold_probs": {}, "gold_probs_kind": {},
                    "gold_label_kind": {}, "metadata": {"source": "grounding"}},
                   sort_keys=True) + "\n")
    (harvest / D.MANIFEST).write_text(json.dumps({
        "suite": "libero_spatial", "questions_version": "v2", "memory_rule": "v2-tracker-1",
        "label_source": "expert_tracker", "horizon": 44, "delta_t": 1.0, "delta_r": 0.05,
        "grounding_repeats": 8, "splits": {"train": [0]}, "created_at": "2026-09-21T00:00:00+00:00",
    }, sort_keys=True))
    return harvest


def _v2_round(tmp_path, name, round_index, rows=3):
    directory = tmp_path / name
    directory.mkdir()
    (directory / "task_0.jsonl").write_text(
        "".join(json.dumps({"id": f"dagger_expert:libero_spatial:0:0:r{round_index}:{i}",
                            "split": "train", "state": "s", "questions": {}, "gold": {},
                            "gold_probs": {}, "gold_probs_kind": {}, "gold_label_kind": {},
                            "metadata": {"source": "dagger_expert"}}, sort_keys=True) + "\n"
                for i in range(rows)))
    (directory / D.MANIFEST).write_text(json.dumps({
        "suite": "libero_spatial", "labeller": "expert", "round": round_index,
        "memory_rule": "run5-memory-2", "label_source": "goal", "horizon": 44,
        "splits": {"train": [0]},
    }, sort_keys=True))
    return directory


def test_the_merge_carries_the_grounding_rows_into_the_training_directory(tmp_path):
    """`train._task_files` globs `task_*.jsonl` **plus** `grounding.jsonl`. A merged directory
    without the grounding file would drop the one question a DAgger round cannot correct -- a
    rollout only ever visits the states its own grounding chose."""
    from robojev import train as trainer

    harvest = _v2_harvest(tmp_path)
    round1 = _v2_round(tmp_path, "dagger1", 1)
    merged = D.merge(harvest, round1, tmp_path / "merged")
    out = tmp_path / "merged"
    assert (out / "grounding.jsonl").read_bytes() == (harvest / "grounding.jsonl").read_bytes()
    assert merged["rows"]["by_source"] == {"harvest": 2, "dagger": 3, "grounding": 1}
    assert merged["rows"]["total"] == 6
    assert merged["sources"]["grounding"]["repeats"] == 8
    names = {p.name for p in trainer._task_files(out)}
    assert "grounding.jsonl" in names and "task_0.jsonl" in names


def test_the_merge_takes_every_round_collected_so_far(tmp_path):
    """Two rounds are more of the same rows. A directory that could hold only one would make
    "train on everything so far" a manual concatenation nobody's manifest describes."""
    harvest = _v2_harvest(tmp_path)
    round1 = _v2_round(tmp_path, "dagger1", 1, rows=3)
    round2 = _v2_round(tmp_path, "dagger2", 2, rows=4)
    merged = D.merge(harvest, [round1, round2], tmp_path / "merged")
    assert merged["rows"]["by_source"]["dagger"] == 7
    assert merged["sources"]["dagger"]["rounds"] == [1, 2]
    assert merged["sources"]["dagger"]["dirs"] == [str(round1), str(round2)]
    rows = [json.loads(line) for line in
            (tmp_path / "merged" / "task_0.jsonl").read_text().splitlines() if line.strip()]
    # The harvest's rows first and byte for byte, then each round in the order it was given.
    assert [r["metadata"]["source"] for r in rows] == ["expert_rollout"] * 2 + ["dagger_expert"] * 7
    assert len({r["id"] for r in rows}) == len(rows)


def test_two_rounds_with_the_same_number_are_refused(tmp_path):
    """Their row ids collide, and a directory with two rows claiming one id is one the trainer
    reads twice and counts once."""
    harvest = _v2_harvest(tmp_path)
    with pytest.raises(ValueError, match="same --round"):
        D.merge(harvest, [_v2_round(tmp_path, "a", 1), _v2_round(tmp_path, "b", 1)],
                tmp_path / "merged")


def test_a_v1_round_is_still_refused_against_a_v2_harvest(tmp_path):
    """The check the seam exists for: v1's rows answer different questions about a
    differently-rendered state."""
    harvest = _v2_harvest(tmp_path)
    round1 = _v2_round(tmp_path, "dagger1", 1)
    manifest = json.loads((round1 / D.MANIFEST).read_text())
    manifest["labeller"] = "geometric"
    (round1 / D.MANIFEST).write_text(json.dumps(manifest, sort_keys=True))
    with pytest.raises(ValueError, match="built differently"):
        D.merge(harvest, round1, tmp_path / "merged")
