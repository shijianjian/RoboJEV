"""The learned policy's contract, on a CPU against a fake `DecisionPredictor`.

The real forward pass needs CUDA and 2.4 GB of weights, so it is a `gpu`-marked proof run by
hand. The fake is injected into `sys.modules` under upstream's own module name, which is what
`_ensure_loaded()` imports after putting the pinned clone's `scripts/` on `sys.path`;
`$ROBOJEV_HOME` points at a `tmp_path`, so the clone it looks for and the checkpoint it reads are
both fabricated and nothing touches a real runtime root.

What is worth testing without a GPU is everything that decides *what is asked*: the question set
comes from the checkpoint and never from an argument, the step scale comes with it, the grounding
forward happens twice at most and is its own request in its own frame, the latch guard is the
only serving rule and it is reported, and the `decisions` block has one group per active qid --
by name, never by count, because `yaw` and the shared `step` are defined and off.
"""
from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

from robojev import expert, registry, runtime
from robojev import grounding as v2grounding
from robojev import policy as policy_mod
from robojev.compose import MEASURED_CM_PER_UNIT, MEASURED_STEPS
from robojev.parse import parse as v2_parse
from robojev.questions import QUESTIONS as V2_QUESTIONS, candidates as v2_candidates
from robojev.roles import scene_roles
from robojev.state import TrackerV2

#: Read from the package's own sidecar, so this file cannot drift from the pin.
PIN = json.loads((runtime.TRAINER_DIR / "nanojev.json").read_text())
CLONE = f"nanojev@{PIN['commit'][:12]}"

WEIGHTS = b"not really 2.4 GB of safetensors"

QIDS = registry.qids("v2")

#: The proprio vector and a two-bowl LIBERO-Spatial scene: the hand above the table, the two
#: bowls 20 cm apart, the plate to the right and the ramekin to the left -- the scene the hardest
#: instruction in the suite is written about.
STATE = np.array([0.0, -0.10, 1.05, 3.14, 0.0, 0.0, 0.04, -0.04], dtype=np.float32)
PRIVILEGED = {
    "akita_black_bowl_1": {"pos": np.array([-0.06, -0.15, 0.97], np.float32),
                           "quat": np.array([1, 0, 0, 0], np.float32)},
    "akita_black_bowl_2": {"pos": np.array([-0.19, 0.20, 0.97], np.float32),
                           "quat": np.array([1, 0, 0, 0], np.float32)},
    "plate_1": {"pos": np.array([0.05, 0.19, 0.97], np.float32),
                "quat": np.array([1, 0, 0, 0], np.float32)},
    "glazed_rim_porcelain_ramekin_1": {"pos": np.array([-0.20, -0.30, 0.97], np.float32),
                                       "quat": np.array([1, 0, 0, 0], np.float32)},
}
INSTRUCTION = "pick up the black bowl between the plate and the ramekin and place it on the plate"


def v2_meta(**kw) -> dict:
    """The provenance file `robojev train` writes out of a harvest manifest."""
    tracker = TrackerV2(horizon=44, every=5, qids=QIDS)
    return {
        "questions_version": "v2",
        "qids": list(QIDS),
        "delta_t": MEASURED_STEPS.units("large"),
        "delta_r": 0.05785714285714285,
        "cm_per_unit": MEASURED_CM_PER_UNIT,
        "steps": {"cm": dict(MEASURED_STEPS.cm),
                  "cm_per_unit": MEASURED_CM_PER_UNIT},
        "tracker": tracker.settings(),
        "memory_rule": tracker.settings()["memory_rule"],
        **kw,
    }


class FakeV2Predictor:
    """`DecisionPredictor`'s surface, answering whatever the request asks.

    Built from the request's own criteria rather than from a table, so it answers the grounding
    questions (whose candidates are this scene's short ids) and the motion questions with one
    body. `answers` can be pinned per qid by a test; anything else gets a sharp distribution over
    the first candidate.
    """

    instances: list["FakeV2Predictor"] = []
    #: `{qid: {candidate: p}}`, consulted before the default.
    pinned: dict[str, dict[str, float]] = {}

    def __init__(self, checkpoint_dir, max_length=None, precision="bf16", **kw):
        self.checkpoint_dir = checkpoint_dir
        self.max_length = max_length
        self.precision = precision
        self.calls: list[tuple[dict, dict]] = []
        FakeV2Predictor.instances.append(self)

    @staticmethod
    def _distribution(qid: str, candidates: list[str]) -> dict[str, float]:
        pinned = FakeV2Predictor.pinned.get(qid)
        if pinned is not None:
            return {c: float(pinned.get(c, 0.0)) for c in candidates}
        head = 0.9 if len(candidates) > 1 else 1.0
        rest = (1.0 - head) / max(len(candidates) - 1, 1)
        return {c: (head if i == 0 else rest) for i, c in enumerate(candidates)}

    def predict(self, payload, batch_questions=0, temperature=1.0):
        self.calls.append((payload, {"batch_questions": batch_questions,
                                     "temperature": temperature}))
        state = payload["states"][0]
        return {
            "states": [{
                "id": state["id"],
                "answers": {qid: {"type": q["type"],
                                  "probabilities": self._distribution(qid, list(q["criteria"]))}
                            for qid, q in state["questions"].items()},
            }],
            "execution": {"forward_passes": 1, "candidate_paths": 29,
                          "questions": len(state["questions"]), "device": "cuda:0"},
        }


@pytest.fixture(scope="module")
def server():
    return policy_mod


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOJEV_HOME", str(tmp_path))
    (tmp_path / "src" / CLONE / PIN["subdir"]).mkdir(parents=True)
    checkpoint = tmp_path / "checkpoints" / "robojev" / "libero_spatial"
    (checkpoint / "backbone_config").mkdir(parents=True)
    (checkpoint / "tokenizer").mkdir()
    (checkpoint / "best.safetensors").write_bytes(WEIGHTS)
    (checkpoint / "config.json").write_text(json.dumps({"set_head": "attention"}))
    (checkpoint / policy_mod.CHECKPOINT_META).write_text(json.dumps(v2_meta()))
    return checkpoint


@pytest.fixture
def fake_upstream(monkeypatch):
    FakeV2Predictor.instances.clear()
    FakeV2Predictor.pinned = {}
    module = types.ModuleType("predict_toy_decisions")
    module.DecisionPredictor = FakeV2Predictor
    monkeypatch.setitem(sys.modules, "predict_toy_decisions", module)
    return module


def make(server, checkpoint, **kw):
    s = server.ModelPolicy("libero_spatial", str(checkpoint), **kw)
    s.reset(INSTRUCTION)
    return s


def obs(**kw):
    return {"state": STATE, "privileged": PRIVILEGED, **kw}


def meta_write(checkpoint, meta: dict) -> None:
    (checkpoint / policy_mod.CHECKPOINT_META).write_text(json.dumps(meta))


# -------------------------------------------------------- the version comes from the checkpoint

def test_the_question_set_is_read_from_the_checkpoint_and_not_from_a_flag(server, home):
    s = make(server, home)
    assert s.questions_version == "v2"
    assert tuple(s._qids) == QIDS == ("move_x", "move_y", "move_z", "size_x", "size_y", "size_z",
                                      "yaw", "rim", "grip", "subgoal")
    # Defined and off by default: the count is never the thing that is asserted.
    assert "step" not in s._qids
    assert "yaw" in V2_QUESTIONS and "step" in V2_QUESTIONS
    assert s.max_length == registry.max_path_tokens("v2") == 1024


def test_a_checkpoint_trained_before_the_wrist_is_served_with_its_own_question_set(server, home):
    """**The deployed weights keep serving.** `robojev-v2r1` was harvested before `yaw` was asked
    and its manifest says so; the active set moved underneath it when the drawer task turned
    the wrist on. The qids come from the checkpoint, exactly as `questions_version`,
    `cm_per_unit` and the tracker's settings do -- reading today's default instead would refuse
    every checkpoint trained before the default last moved.

    Three things have to follow from that, and all three are what makes it safe:
    the served **request** asks that set, the **state text** carries no line for a question this
    run does not ask, and the plan offers **no grasp candidate** that would need one (a candidate
    is an instruction to the arm, and an instruction this vocabulary cannot express is a decision
    that silently does something else).
    """
    old = [q for q in QIDS if q not in ("yaw", "rim")]
    tracker = TrackerV2(horizon=44, every=5, qids=old)
    meta_write(home, v2_meta(qids=old, tracker=tracker.settings()))
    s = make(server, home)
    assert tuple(s._qids) == tuple(old) and "yaw" not in s._qids and "rim" not in s._qids
    assert tuple(s._questions_block()) == tuple(old)
    assert set(s._tracker.qids) == set(old)
    assert {g.turn for g in expert.candidates(s._qids)} == {0.0, 180.0}

    # **And its state text is the text it was trained on.** The two lines and the block this
    # round added are printed only when their own question is asked, and the room's furniture --
    # which the privileged state now carries -- is not in `Other objects:` either.
    from robojev.state import serialise_v2

    objects = {**PRIVILEGED, "wooden_cabinet_1": {
        "pos": np.array([0.04, -0.26, 0.9], np.float32),
        "quat": np.array([0, 0, 0, 1], np.float32), "fixture": True,
        "boxes": [[0.13, -0.05, 1.09, 0.003, 0.034, 0.109, 1, 0, 0, 0, 1, 0, 0, 0, 1]]}}
    s._tracker.commit("akita_black_bowl_1", "plate_1", step=0, source="bddl")
    s._tracker.observe(step=0, proprio=STATE, objects=objects)
    text = serialise_v2(STATE, objects, INSTRUCTION, s._tracker)
    assert "Wrist yaw" not in text and "Grasp candidates" not in text
    assert "Target rests" not in text and "cabinet" not in text
    assert v2_parse(text, s._tracker.qids) == s._tracker.gold_answers()


def test_a_checkpoint_naming_a_question_this_build_cannot_ask_is_refused(server, home):
    meta_write(home, v2_meta(qids=["move_x", "move_y", "grip", "telepathy"]))
    with pytest.raises(policy_mod.PolicyError, match="telepathy"):
        server.ModelPolicy("libero_spatial", str(home))


def test_a_checkpoint_that_says_nothing_declares_the_retired_set_and_is_refused_by_name(
        server, home):
    """A manifest with no `questions_version` predates the key, which means the first
    question set -- retired. Serving it under today's default would ask weights trained on one
    vocabulary about another, silently, for a whole run; the refusal names the set instead."""
    meta_write(home, {"delta_t": 0.3536, "delta_r": 0.0579})
    with pytest.raises(policy_mod.PolicyError) as caught:
        server.ModelPolicy("libero_spatial", str(home))
    assert "'v1'" in str(caught.value) and "retired" in str(caught.value)


def test_a_checkpoint_declaring_the_retired_set_outright_is_refused_the_same_way(server, home):
    meta_write(home, v2_meta(questions_version="v1"))
    with pytest.raises(policy_mod.PolicyError) as caught:
        server.ModelPolicy("libero_spatial", str(home))
    assert "retired" in str(caught.value) and "0/40" in str(caught.value)


def test_the_flag_may_assert_the_version_and_may_never_change_it(server, home):
    assert make(server, home, questions_version="v2").questions_version == "v2"
    with pytest.raises(policy_mod.PolicyError, match="does not match"):
        server.ModelPolicy("libero_spatial", str(home), questions_version="v0")


def test_a_manifest_naming_a_version_this_build_does_not_know_fails_at_construction(server, home):
    meta_write(home, v2_meta(questions_version="v3"))
    with pytest.raises(policy_mod.PolicyError) as caught:
        server.ModelPolicy("libero_spatial", str(home))
    assert "v3" in str(caught.value) and "v2" in str(caught.value)


# ------------------------------------------------------------------------------- the step scale

def test_a_v2_checkpoint_without_its_step_scale_does_not_load(server, home):
    meta = v2_meta()
    del meta["cm_per_unit"]
    meta_write(home, meta)
    with pytest.raises(policy_mod.PolicyError, match="cm_per_unit"):
        server.ModelPolicy("libero_spatial", str(home))


def test_a_checkpoint_served_at_another_scale_is_refused_rather_than_run_five_times_too_fast(
        server, home):
    """The single most expensive silent failure available here: 5 cm steps executed at 25 cm."""
    meta_write(home, v2_meta(cm_per_unit=25.0))
    with pytest.raises(policy_mod.PolicyError, match="delta_t"):
        server.ModelPolicy("libero_spatial", str(home))


def test_the_bands_printed_in_the_state_have_to_be_the_ones_it_was_trained_on(server, home):
    meta_write(home, v2_meta(steps={"cm": {"large": 9.0, "medium": 1.7, "small": 0.5},
                                    "cm_per_unit": 5.0}))
    with pytest.raises(policy_mod.PolicyError, match="step sizes"):
        server.ModelPolicy("libero_spatial", str(home))


def test_a_checkpoint_trained_on_another_tracker_is_refused_at_launch(server, home):
    meta = v2_meta()
    meta["tracker"] = {**meta["tracker"], "history_k": 9}
    meta_write(home, meta)
    with pytest.raises(policy_mod.PolicyError) as caught:
        server.ModelPolicy("libero_spatial", str(home))
    assert "history_k" in str(caught.value) and "state text" in str(caught.value)


# ------------------------------------------------------------------------------------ describe

def test_describe_reports_the_version_the_scale_the_tracker_and_the_grounding(server, home):
    d = make(server, home).describe()
    block = d["protocol"]
    assert d["questions_version"] == block["questions_version"] == "v2"
    assert set(d["questions"]) == set(QIDS)
    assert d["questions"]["move_x"]["candidates"] == list(v2_candidates("move_x"))
    assert block["cm_per_unit"] == 5.0
    assert block["step_sizes_cm"] == dict(MEASURED_STEPS.cm)
    # Nothing saturates at the measured scale -- the sizes were read off what the arm executes.
    assert block["step_saturates"] == {"large": False, "medium": False, "small": False}
    assert block["delta_t"] == 1.0
    assert block["memory"]["memory_rule"] == "v2-tracker-1"
    assert block["memory"]["annotate"] is True
    assert block["memory"]["qids"] == list(QIDS)
    assert block["grip_latch"] == {"grip_guard": True}
    assert block["ground"]["mode"] == "rule"
    assert block["ground"]["order"] == ["rule", "model", "scene_roles"]
    assert "resolve_request" in block["ground"]["resolver"]
    forward = make(server, home, ground="forward").describe()["protocol"]["ground"]
    assert forward["mode"] == "forward" and forward["order"] == ["model", "scene_roles"]
    assert block["ground"]["questions"] == ["target", "destination"]
    assert block["ground"]["min_probability"] == 0.5
    assert block["ground"]["frame"] == "camera"
    assert block["max_path_tokens"] == 1024
    # ρ is v1's declared label noise and v2's state says nothing about it.
    assert d["rho"] is None and block["rho"] is None


def test_the_policy_states_how_it_wants_to_be_run(server, home):
    """`protocol()` is what the episode loop obeys. The learned policy asks for the settling
    steps the harvest that trained it rendered its first state after; the scripted one asks for
    none. The number is the policy's, not the loop's."""
    policy = make(server, home)
    protocol = policy.protocol()
    assert protocol.family == "decision" and protocol.privileged_state is True
    assert protocol.chunk_size == protocol.execute_steps == 5
    assert protocol.wait_steps == 10 == policy.describe()["protocol"]["wait_steps"]
    assert protocol.max_steps == 220
    # An override is possible and is a different experiment, which is why it has to be asked for.
    assert policy.protocol(30).max_steps == 30


def test_describe_still_loads_nothing(server, home, fake_upstream):
    make(server, home).describe()
    assert FakeV2Predictor.instances == []


# ---------------------------------------------------------------------------------------- act

def test_a_motion_frame_has_one_group_per_active_question_and_meta(server, home, fake_upstream):
    s = make(server, home)
    s.act(obs())                                   # decision 0 also grounds
    chunk, decisions = s.act(obs())
    assert set(decisions) == set(QIDS) | {"meta"}
    assert chunk.shape == (5, 7) and chunk.dtype == np.float32
    assert np.array_equal(np.unique(chunk, axis=0), chunk[:1])
    assert all(decisions[qid]["overridden"] is False for qid in QIDS)
    assert list(decisions["move_x"]["probabilities"]) == list(v2_candidates("move_x"))
    assert decisions["meta"]["questions_version"] == "v2"
    assert decisions["meta"]["forward_passes"] == 1
    assert decisions["meta"]["cm_per_unit"] == 5.0


def test_one_forward_carries_every_question_of_the_motion_request(server, home, fake_upstream):
    s = make(server, home)
    s.act(obs())
    payload, kwargs = FakeV2Predictor.instances[0].calls[-1]
    state = payload["states"][0]
    assert list(payload) == ["states"] and set(state) == {"id", "state", "questions"}
    assert set(state["questions"]) == set(QIDS)
    assert kwargs == {"batch_questions": 0, "temperature": 1.0}
    assert state["id"] == "libero_spatial:0:0"
    assert s.max_length == FakeV2Predictor.instances[0].max_length == 1024


def test_the_state_is_v2s_tracker_text_and_carries_the_committed_target(server, home,
                                                                       fake_upstream):
    _chunk, decisions = make(server, home).act(obs())
    text = decisions["meta"]["state"]
    assert text == FakeV2Predictor.instances[0].calls[-1][0]["states"][0]["state"]
    assert "gripper-relative" in text
    assert "Target: " in text and "Decision 1 of 44" in text
    # v1's serialiser is nowhere near this: no absolute eef, no declared-noise sentence.
    assert "eef_pos=" not in text and "probability 0.85" not in text


def test_the_composed_action_is_the_checkpoints_own_step_scale(server, home, fake_upstream):
    """`move_x: -` at the large size is one full normalised unit at the measured 5 cm, and a
    fifth of one if this were served at v1's δ_t -- which is the whole of failure G2."""
    FakeV2Predictor.pinned = {
        "move_x": {"-": 1.0}, "move_y": {"hold": 1.0}, "move_z": {"hold": 1.0},
        "size_x": {"large": 1.0}, "size_y": {"small": 1.0}, "size_z": {"small": 1.0},
        "grip": {"false": 1.0}, "subgoal": {"reach": 1.0},
    }
    chunk, decisions = make(server, home).act(obs())
    assert decisions["move_x"]["choice"] == "-" and decisions["size_x"]["choice"] == "large"
    assert chunk[0][0] == pytest.approx(-1.0)
    assert chunk[0][6] == pytest.approx(-1.0)              # fingers open


# ------------------------------------------------------------------------------ the grounding

def _ground_call(instance):
    return [payload for payload, _ in instance.calls
            if "target" in payload["states"][0]["questions"]]


def test_the_grounding_forward_happens_on_decision_zero_and_nowhere_else(server, home,
                                                                        fake_upstream):
    s = make(server, home)
    for _ in range(4):
        s.act(obs())
    instance = FakeV2Predictor.instances[0]
    assert len(_ground_call(instance)) == 1
    assert len(instance.calls) == 5                         # one grounding + four motion
    assert instance.calls[0][0]["states"][0]["id"].endswith(":ground")


def test_the_grounding_request_is_its_own_request_in_the_cameras_frame(server, home,
                                                                      fake_upstream):
    """Never concatenated with the motor state: that one is gripper-relative and mirrored left
    for right, so "the bowl on the left" read against it resolves the wrong bowl."""
    make(server, home).act(obs())
    ground, motion = FakeV2Predictor.instances[0].calls[0][0], \
        FakeV2Predictor.instances[0].calls[1][0]
    text = ground["states"][0]["state"]
    assert list(ground["states"][0]["questions"]) == ["target", "destination"]
    assert "Frame: the work surface" in text and "x is front" in text
    assert f"Task: {INSTRUCTION}" in text
    assert "gripper-relative" not in text
    assert text not in motion["states"][0]["state"]
    assert "Grounding." not in motion["states"][0]["state"]


def test_a_confident_grounding_is_committed_as_the_models_own_answer(server, home, fake_upstream):
    """`--ground forward`: the other arm, and what `rule` is measured against."""
    names = v2grounding.grounding_names(PRIVILEGED)
    bowl = next(sid for sid, name in names.items() if name == "akita_black_bowl_2")
    plate = next(sid for sid, name in names.items() if name == "plate_1")
    FakeV2Predictor.pinned = {"target": {bowl: 0.92}, "destination": {plate: 0.88}}
    s = make(server, home, ground="forward")
    _chunk, decisions = s.act(obs())
    grounding = decisions["meta"]["grounding"]
    assert grounding["mode"] == "forward" and grounding["source"] == "model"
    assert grounding["target"] == "akita_black_bowl_2"
    assert grounding["destination"] == "plate_1"
    assert grounding["regrounded"] is False
    assert s._tracker.committed() == {"target": "akita_black_bowl_2", "destination": "plate_1",
                                      "source": "model", "step": 0}
    # And the two groups render like any other question, so the tab needs no schema change.
    assert set(decisions) == set(QIDS) | {"target", "destination", "meta"}
    assert decisions["target"]["choice"] == bowl
    assert decisions["target"]["probabilities"][bowl] == pytest.approx(0.92)
    assert decisions["target"]["overridden"] is False
    assert decisions["meta"]["target"] == "akita_black_bowl_2"


def test_by_default_the_text_rule_commits_and_a_confidently_wrong_model_does_not(server, home,
                                                                                 fake_upstream):
    """Run 6 in a unit test. Its forward picked `akita_black_bowl_2` at p 0.957 on exactly this
    scene, whose `bowl_1` line says "between plate_1 and ramekin_1" -- the bowl the sentence
    names. A probability threshold cannot catch a confident error, so `--ground rule` commits
    what the text says and reports what the model wanted."""
    names = v2grounding.grounding_names(PRIVILEGED)
    wrong = next(sid for sid, name in names.items() if name == "akita_black_bowl_2")
    plate = next(sid for sid, name in names.items() if name == "plate_1")
    FakeV2Predictor.pinned = {"target": {wrong: 0.957}, "destination": {plate: 0.88}}
    s = make(server, home)
    _chunk, decisions = s.act(obs())
    grounding = decisions["meta"]["grounding"]
    assert grounding["mode"] == "rule" and grounding["source"] == "rule"
    assert grounding["sources"] == {"target": "rule", "destination": "rule"}
    assert grounding["target"] == "akita_black_bowl_1"
    assert grounding["destination"] == "plate_1"
    assert s._tracker.committed()["target"] == "akita_black_bowl_1"
    assert "Target: bowl_1" in decisions["meta"]["state"]
    # The model's opinion is still shown, with its probabilities, and the record says it differed.
    assert grounding["model"] == {"target": "akita_black_bowl_2", "destination": "plate_1"}
    assert grounding["model_agrees"] is False
    assert grounding["probability"]["target"] == pytest.approx(0.957)
    assert decisions["target"]["choice"] == wrong
    assert decisions["target"]["probabilities"][wrong] == pytest.approx(0.957)


def test_the_model_answers_when_the_text_rule_cannot_and_is_sure_enough(server, home,
                                                                       fake_upstream):
    """The chain: rule, then the model at or above 0.5, then `scene_roles`. A sentence the rule
    cannot cut leaves the question to whichever of the two can answer it."""
    names = v2grounding.grounding_names(PRIVILEGED)
    bowl = next(sid for sid, name in names.items() if name == "akita_black_bowl_2")
    s = server.ModelPolicy("libero_spatial", str(home))
    s.reset("do the thing")                      # no noun the rule can match
    FakeV2Predictor.pinned = {"target": {bowl: 0.91},
                              "destination": {sid: 0.25 for sid in names}}
    _chunk, decisions = s.act(obs())
    grounding = decisions["meta"]["grounding"]
    assert grounding["rule"] == {"target": None, "destination": None}
    # One flat question drops *both* to the geometric rule: the pair is committed together.
    assert set(grounding["sources"].values()) == {"scene_roles"}
    assert grounding["target"] in PRIVILEGED and grounding["destination"] in PRIVILEGED
    assert grounding["source"] == "scene_roles"


def test_a_coin_flip_between_two_identical_bowls_falls_back_to_the_rule(server, home,
                                                                       fake_upstream):
    """A flat distribution is not a decision. Below `GROUNDING_MIN_P` the rule that named every
    harvested row's target names this one, and the record says which of the two did it."""
    names = v2grounding.grounding_names(PRIVILEGED)
    bowls = sorted(sid for sid, name in names.items() if "bowl" in name)
    FakeV2Predictor.pinned = {"target": {bowls[0]: 0.34, bowls[1]: 0.33},
                              "destination": {sid: 0.25 for sid in names}}
    _chunk, decisions = make(server, home).act(obs())
    grounding = decisions["meta"]["grounding"]
    assert grounding["source"] == "rule"
    assert grounding["target"] in PRIVILEGED and grounding["destination"] in PRIVILEGED
    # Under `forward` a flat distribution falls back to the geometric rule; under the default the
    # text rule answered before the probability was ever looked at.
    flat = make(server, home, ground="forward")
    _c, forward = flat.act(obs())
    roles = scene_roles(PRIVILEGED, INSTRUCTION, np.asarray(STATE[0:3], np.float64))
    assert forward["meta"]["grounding"]["source"] == "scene_roles"
    assert forward["meta"]["grounding"]["target"] == roles["target"]
    assert decisions["meta"]["target_rule"] == "rule"
    # The model's own distribution is still reported: the operator sees what it thought.
    assert decisions["target"]["probabilities"][bowls[0]] == pytest.approx(0.34)


def test_a_failed_attempt_grounds_again_and_says_so(server, home, fake_upstream):
    """§3: the two answers are re-asked "only when the tracker records a failed attempt", which
    on a scene with two identical bowls is the evidence that the wrong one was named."""
    s = make(server, home)
    s.act(obs())
    assert len(_ground_call(FakeV2Predictor.instances[0])) == 1
    # The tracker's own fact, forced here rather than played out over forty decisions of a
    # simulator this test does not have.
    s._tracker._grasp_attempts = s._tracker._committed_attempts + 1
    s._tracker._rose_at = None
    _chunk, decisions = s.act(obs())
    assert len(_ground_call(FakeV2Predictor.instances[0])) == 2
    assert decisions["meta"]["grounding"]["regrounded"] is True
    assert decisions["meta"]["grounding"]["decision"] == 1
    # And not a third time while nothing new has failed.
    s.act(obs())
    assert len(_ground_call(FakeV2Predictor.instances[0])) == 2


def test_reset_clears_the_tracker_and_the_committed_roles(server, home, fake_upstream):
    s = make(server, home)
    for _ in range(3):
        s.act(obs())
    assert s._tracker.decisions == 3 and s._tracker.target is not None
    s.reset("pick up the plate and place it on the black bowl")
    assert s._tracker.decisions == 0 and s._step == 0
    assert s._tracker.target is None and s._tracker.committed() is None
    assert s._grounding is None
    s.act(obs())
    # A fresh episode grounds again, from scratch.
    assert len(_ground_call(FakeV2Predictor.instances[0])) == 2


# ---------------------------------------------------------------------- overrides and the latch

def test_an_override_naming_a_v1_question_is_refused_under_v2(server, home, fake_upstream):
    s = make(server, home)
    with pytest.raises(policy_mod.PolicyError, match=r"\['translate'\]"):
        s.act(obs(), overrides={"translate": "+x"})
    with pytest.raises(policy_mod.PolicyError, match="not one of"):
        s.act(obs(), overrides={"move_x": "+q"})


def test_an_override_forces_one_axis_and_leaves_the_distribution_alone(server, home,
                                                                      fake_upstream):
    s = make(server, home)
    _chunk, decisions = s.act(obs(), overrides={"move_z": "-"})
    assert decisions["move_z"]["choice"] == "-" and decisions["move_z"]["overridden"] is True
    assert decisions["move_z"]["probabilities"] == pytest.approx(
        {c: (0.9 if i == 0 else 0.05) for i, c in enumerate(v2_candidates("move_z"))})
    assert decisions["meta"]["overridden"] == ["move_z"]


def test_the_latch_guard_refuses_a_close_the_geometry_does_not_permit(server, home,
                                                                     fake_upstream):
    """Spec §7 ruling (c), and note §5's cliff: the expert falls from 19/20 to 7/20 when an
    unguarded latch is corrupted at 10 %, while the motion answers shrug it off."""
    FakeV2Predictor.pinned = {"grip": {"true": 0.99, "false": 0.01}}
    s = make(server, home)
    chunk, decisions = s.act(obs())
    assert decisions["grip"]["choice"] == "false", "the hand is nowhere near the rim"
    assert chunk[0][6] == pytest.approx(-1.0)
    assert decisions["meta"]["grip_latch"]["asked"] is True
    assert decisions["meta"]["grip_latch"]["refused"] is True
    assert decisions["meta"]["grip_latch"]["closed"] is False
    assert decisions["meta"]["grip_latch"]["grip_guard"] is True
    # The refusal is an event, so the next state the model reads says its request was not applied.
    _chunk, nxt = s.act(obs())
    assert "refused" in nxt["meta"]["state"]


def test_an_operator_override_outranks_the_latch_guard(server, home, fake_upstream):
    """"Force close" from the console must close the gripper, not be argued with by a rule."""
    s = make(server, home)
    chunk, decisions = s.act(obs(), overrides={"grip": True})
    assert decisions["grip"]["choice"] == "true" and decisions["grip"]["overridden"] is True
    assert chunk[0][6] == pytest.approx(1.0)
    assert decisions["meta"]["grip_latch"]["refused"] is False
    assert s._latch.closed is True


def test_the_executed_answers_are_what_the_next_history_block_explains(server, home,
                                                                      fake_upstream):
    """An override or a refused latch is what the arm did, so it is what the tracker records --
    the model's own preference is already in `decisions[qid]["probabilities"]`."""
    s = make(server, home)
    s.act(obs(), overrides={"move_x": "+"})
    assert s._tracker._rows[-1].answers["move_x"] == "+"
