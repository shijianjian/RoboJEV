"""RoboJEV v2's core: the question set, the composer, the single label function, the state
string and gate G3's parser.

Pure CPU, no LIBERO and no torch. The gate this file actually runs is **G3** -- "a rule-based
parser of the state string alone reproduces every label", spec §5 -- and the property test below
is the measurement: thousands of synthetic states, with the per-axis annotations and without
them, parsed back to the answers the expert gave. If it ever fails, v2's principle 1 is broken
and the labels are not a function of the text the model reads.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

import robojev as v2

# `robojev.compose` is a *function* on the package, so the submodules are reached
# through `importlib` rather than `from … import compose`.
c2 = importlib.import_module("robojev.compose")
q2 = importlib.import_module("robojev.questions")
s2 = importlib.import_module("robojev.state")
p2 = importlib.import_module("robojev.parse")
expert = importlib.import_module("robojev.expert")
v2questions = q2

OBJECTS = {
    "akita_black_bowl_1": {"pos": np.array([0.02, 0.10, 0.92])},
    "akita_black_bowl_2": {"pos": np.array([-0.12, 0.22, 0.92])},
    "glazed_rim_porcelain_ramekin_1": {"pos": np.array([-0.12, 0.095, 0.92])},
    "plate_1": {"pos": np.array([0.17, 0.085, 0.925])},
}
INSTRUCTION = ("pick up the black bowl between the plate and the ramekin and place it "
               "on the plate")


def _tracker(**kwargs):
    kwargs.setdefault("horizon", 44)
    kwargs.setdefault("target", "akita_black_bowl_1")
    kwargs.setdefault("destination", "plate_1")
    return v2.TrackerV2(**kwargs)


# --------------------------------------------------------------------------- the question set


def _validate_request(req: dict) -> None:
    """Upstream's `validate_request`, vendored as a check exactly as `test_decision.py` vendors
    it for v1 -- NanoJev is not a package and its scripts import torch at module scope, so the
    rules it enforces are asserted here and the real one runs in the `sim`/`gpu` proof."""
    assert set(req) == {"states"}
    assert isinstance(req["states"], list) and req["states"]
    for state in req["states"]:
        assert set(state) == {"id", "state", "questions"}
        assert isinstance(state["id"], str) and state["id"]
        assert isinstance(state["state"], str) and state["state"]
        assert isinstance(state["questions"], dict) and state["questions"]
        for qid, q in state["questions"].items():
            assert set(q) <= {"type", "instructions", "criteria"}
            assert q["type"] in ("boolean", "choice", "score")
            assert isinstance(q["instructions"], str) and q["instructions"]
            if q["type"] == "choice":
                assert 2 <= len(q["criteria"]) <= 255
                assert all(isinstance(k, str) and isinstance(vv, str)
                           for k, vv in q["criteria"].items())
            else:
                assert set(q["criteria"]) <= {"false", "true"}


def test_the_motion_request_is_the_shape_nanojev_accepts():
    req = v2.request("libero_spatial:0:0", state_id="libero_spatial:0:0")
    _validate_request(req)
    questions = req["states"][0]["questions"]
    assert tuple(questions) == v2.ACTIVE_QIDS
    assert tuple(questions) == ("move_x", "move_y", "move_z", "size_x", "size_y", "size_z",
                                "yaw", "rim", "grip", "subgoal")
    for axis in ("move_x", "move_y", "move_z"):
        assert tuple(questions[axis]["criteria"]) == ("-", "hold", "+")
    for size in ("size_x", "size_y", "size_z"):
        assert tuple(questions[size]["criteria"]) == ("large", "medium", "small")
    assert tuple(questions["yaw"]["criteria"]) == ("-", "hold", "+")
    assert tuple(questions["rim"]["criteria"]) == v2.RIM_CANDIDATES == tuple("ABCDEFGH")
    # The shared `step` is defined and not asked -- see `test_the_active_set_...`.
    assert "step" not in questions
    assert questions["grip"]["type"] == "boolean"
    assert set(questions["grip"]["criteria"]) == {"true", "false"}
    assert tuple(questions["subgoal"]["criteria"]) == (
        "reach", "grasp", "lift", "carry", "place", "retreat")


def test_the_episode_request_grounds_against_the_scene_in_words():
    """Failure F3: "the bowl between the plate and the ramekin" had to be grounded implicitly
    inside every motor decision, off ten instructions. Now it is one question whose candidates
    spell out where each object is and what it is between."""
    cm = v2.objects_cm(np.zeros(8), OBJECTS)
    req = v2.request("state", qids=v2.EPISODE_QIDS, objects_cm=cm)
    _validate_request(req)
    questions = req["states"][0]["questions"]
    assert tuple(questions) == ("target", "destination")
    assert set(questions["target"]["criteria"]) == {"bowl_1", "bowl_2", "ramekin_1", "plate_1"}
    bowl_1 = questions["target"]["criteria"]["bowl_1"]
    assert bowl_1.startswith("bowl_1, at x ")
    assert "cm" in bowl_1
    assert "between ramekin_1 and plate_1" in bowl_1 or "between plate_1 and ramekin_1" in bowl_1
    assert "right of ramekin_1" in bowl_1
    assert "nearest to" in bowl_1


def test_an_episode_question_without_a_scene_is_an_error_not_an_empty_candidate_set():
    with pytest.raises(ValueError):
        v2.questions_block(("target",))


def test_the_active_set_asks_the_wrist_and_drops_the_shared_step():
    """Plan 9c's ruling on the motor vocabulary, from the expert's own 100 episodes
    (`notes/2026-09-21-stage9-scripted-expert.md`): the per-axis sizes scored **85/100 at a
    median of 27 decisions** against the shared-step vocabulary's 83/100 at 33 (§4), so `step`
    stays defined and unasked.

    `yaw` moved the other way (note `2026-09-21-stage9-drawer-task.md`): it was `hold` in 100 %
    of the expert's decisions only because the served yaw size was scaled by what the controller
    is *commanded* rather than by what it executes, so one answer turned the wrist 3.6 degrees
    and a right angle cost 25 of the 44 decisions. At the measured 28 degrees per unit a right
    angle costs six, and the drawer task is 0/10 without it. It stays switchable, because every
    checkpoint trained before it is served from its own `robojev.json`."""
    assert "yaw" in v2.QUESTIONS and "step" in v2.QUESTIONS
    assert "yaw" in v2.ACTIVE_QIDS and "rim" in v2.ACTIVE_QIDS
    assert "step" not in v2.ACTIVE_QIDS
    assert v2.active_qids() == v2.ACTIVE_QIDS
    assert v2.active_qids(yaw=False, rim=False) == ("move_x", "move_y", "move_z", "size_x",
                                                    "size_y", "size_z", "grip", "subgoal")
    assert v2.active_qids(shared_step=True) == ("move_x", "move_y", "move_z", "step", "yaw",
                                                "rim", "grip", "subgoal")
    assert set(v2.ALL_MOTION_QIDS) > set(v2.ACTIVE_QIDS)


def test_every_motion_candidate_states_its_commit_horizon():
    """Latest-NanoJev note §6.2: their Doom candidates say "…for 4 Doom ticks". A model choosing
    an action has to be told what the action costs. `subgoal` is exempt -- it names a stage, it
    does not command the arm for a quarter of a second."""
    assert v2.HELD_FOR == "Held for the next 5 control steps (0.25 s)."
    for qid in [q for q in v2.ALL_MOTION_QIDS if q != "subgoal"]:
        for text in v2.QUESTIONS[qid]["criteria"].values():
            assert text.endswith(v2.HELD_FOR)
    for text in v2.QUESTIONS["subgoal"]["criteria"].values():
        assert v2.HELD_FOR not in text
    assert v2.COMMIT_STEPS == v2.CHUNK_STEPS


def test_the_question_block_is_a_copy():
    a = v2.questions_block()
    a["move_x"]["criteria"]["+"] = "mutated"
    assert v2.questions_block()["move_x"]["criteria"]["+"] != "mutated"


def test_scene_relations_name_the_phrases_libero_instructions_use():
    cm = {"a": (0.0, 0.0, 0.0), "b": (20.0, 0.0, 0.0), "c": (10.0, 1.0, 0.0),
          "lid": (0.0, 0.0, 6.0)}
    texts = v2.scene_candidates(cm)
    assert "between a and b" in texts["c"]
    assert "on top of a" in texts["lid"]
    assert "left of b" in texts["a"] and "right of a" in texts["b"]
    forward = v2.scene_candidates({"n": (0.0, 0.0, 0.0), "f": (0.0, 20.0, 0.0)})
    assert "behind n" in forward["f"] and "in front of f" in forward["n"]


# ------------------------------------------------------------------------------- the composer


def test_the_step_sizes_are_the_experts_measured_ones():
    """`notes/2026-09-21-stage9-scripted-expert.md` §4: 5.0 / 1.7 / 0.5 cm **executed** per axis
    per decision, which is `expert.AXISWISE_SCALE` times the measured travel. Gate G2's pass
    condition is that `small` is under the positioning tolerance the rim grasp needs -- 0.5 cm
    against the ±1.5 cm the note's tolerance table holds at 100 % on every axis."""
    assert v2.STEP_LARGE_CM == pytest.approx(5.0)
    assert v2.STEP_MEDIUM_CM == pytest.approx(5.0 / 3.0)
    assert v2.STEP_SMALL_CM == pytest.approx(0.5)
    assert v2.STEP_SMALL_CM < v2.STEP_MEDIUM_CM < v2.STEP_LARGE_CM
    # The sizes are the expert's own scale, not a second copy of it.
    for name, cm in (("large", v2.STEP_LARGE_CM), ("medium", v2.STEP_MEDIUM_CM),
                     ("small", v2.STEP_SMALL_CM)):
        assert cm == pytest.approx(expert.AXISWISE_SCALE[name] * v2.MEASURED_CM_PER_UNIT)


def test_the_measured_scale_is_the_expert_notes_and_nothing_saturates():
    """Note §2: the arm **executes** 4.49 / 5.41 / 5.49 cm on x / y / z at δ_t = 1.0 against the
    0.25 m the composer *commands* -- a fifth. v1 believed the commanded number, so every one of
    its thresholds was five times too wide and a `hold` at 4.4 cm of real error was a controller
    stopping 4.4 cm short of the bowl."""
    assert 4.4 <= c2.MEASURED_CM_PER_UNIT <= 5.6
    assert c2.MEASURED_CM_PER_UNIT < c2.COMMANDED_CM_PER_UNIT / 4
    assert c2.MEASURED_STEPS.cm_per_unit == c2.MEASURED_CM_PER_UNIT
    assert c2.DEFAULT_STEPS is c2.MEASURED_STEPS      # v2 is served at delta_t = 1.0
    # Nothing saturates: the sizes were chosen from what the arm executes, so `large` is exactly
    # the OSC range limit. The guard stays for the day gate G2 moves STEP_LARGE_CM.
    assert not c2.saturates("large") and not c2.saturates("small")
    assert c2.saturates("large", c2.calibrate(2.0))
    assert "executes" in c2.step_range_words("large", c2.calibrate(2.0))


def test_a_large_step_commands_no_more_than_the_controller_takes():
    action = v2.compose({"move_x": "+", "move_y": "hold", "move_z": "hold", "size_x": "large",
                         "size_y": "small", "size_z": "small", "grip": False})
    assert abs(float(action[0])) <= 1.0
    assert float(action[0]) == pytest.approx(1.0)      # delta_t = 1.0, the OSC range limit


def test_calibrate_replaces_the_scale_without_changing_a_size_in_centimetres():
    measured = v2.calibrate(10.0)
    assert measured.cm == v2.DEFAULT_STEPS.cm             # same centimetres…
    assert measured.units("large") == pytest.approx(0.5)  # …a smaller command to reach them
    assert v2.calibrate(2.0).units("large") == 1.0        # clipped at the range limit
    with pytest.raises(ValueError):
        v2.calibrate(0.0)


def test_the_registry_names_one_question_set_and_refuses_the_retired_one():
    from robojev import registry
    assert registry.VERSIONS == ("v2",) and registry.DEFAULT_VERSION == "v2"
    assert registry.qids("v2") == q2.ACTIVE_QIDS
    assert registry.max_path_tokens("v2") <= 1024
    assert registry.composer("v2").compose is v2.compose
    with pytest.raises(ValueError, match="v2"):
        registry.check("v3")
    # The retired set is refused by name rather than as an unknown string, so a checkpoint that
    # declares it (or declares nothing, which means it) says why it cannot be served.
    with pytest.raises(ValueError, match="retired"):
        registry.check(registry.UNVERSIONED)


def test_three_axes_move_at_once():
    """§2 principle 5, and failure F5a's countermeasure: one axis per quarter-second made the
    scripted expert's paths about three times longer than the demonstrations' diagonals."""
    action = v2.compose({"move_x": "+", "move_y": "-", "move_z": "hold", "size_x": "large",
                         "size_y": "medium", "size_z": "small", "grip": False,
                         "subgoal": "reach"})
    assert action.shape == (7,)
    # Each axis takes **its own** size: that is the ruling, and it is worth six decisions an
    # episode (note §4).
    assert action[0] == pytest.approx(1.0)
    assert action[1] == pytest.approx(-1.0 / 3.0)
    assert action[2] == 0.0
    assert action[5] == 0.0
    assert action[6] == -1.0


def test_yaw_and_the_gripper_latch_compose_too():
    action = v2.compose({"move_x": "hold", "move_y": "hold", "move_z": "hold",
                         "size_x": "medium", "size_y": "small", "size_z": "small",
                         "yaw": "+", "grip": True, "subgoal": "grasp"})
    # **The wrist has one speed and it is not the hand's.** It used to take the largest size any
    # axis asked for, which gave 1.5 degrees a decision in exactly the case the wrist is for --
    # the hand arrived over the rim, every axis answering `hold`/`small`, 60 degrees still to
    # turn (note `2026-09-21-stage9-drawer-task.md` §3).
    assert action[5] == pytest.approx(v2.DEFAULT_STEPS.yaw_units("large"))
    assert action[6] == 1.0
    still = {"move_x": "hold", "move_y": "hold", "move_z": "hold", "size_x": "small",
             "size_y": "small", "size_z": "small"}
    # The string form a probability dict's keys take must mean the same thing as the bool.
    assert v2.compose({**still, "grip": "true"})[6] == 1.0
    assert v2.compose({**still, "grip": "false"})[6] == -1.0
    # The shared-step ablation composes too: one size for every axis.
    shared = v2.compose({"move_x": "+", "move_y": "+", "move_z": "+", "step": "medium",
                         "grip": False})
    assert shared[0] == shared[1] == shared[2] == pytest.approx(1.0 / 3.0)


def test_a_decision_is_five_identical_control_steps():
    answers = {"move_x": "+", "move_y": "hold", "move_z": "-", "size_x": "small",
               "size_y": "small", "size_z": "small", "yaw": "hold", "grip": True}
    block = v2.chunk(answers)
    assert block.shape == (5, 7)
    assert np.all(block == block[0])
    assert np.allclose(block[0], v2.compose(answers))


def test_the_composer_refuses_an_answer_the_question_set_cannot_produce():
    for bad in ({"move_x": "+x"}, {"size_x": "coarse"}, {"yaw": "+yaw"}):
        answers = {"move_x": "hold", "move_y": "hold", "move_z": "hold", "size_x": "small",
                   "size_y": "small", "size_z": "small", "yaw": "hold", "grip": False}
        answers.update(bad)
        with pytest.raises(KeyError):
            v2.compose(answers)


# -------------------------------------------------------------------- the one label function


@pytest.mark.parametrize("axis,index", [("move_x", 0), ("move_y", 1), ("move_z", 2)])
def test_each_axis_is_labelled_independently_by_its_own_sign_and_tolerance(axis, index):
    """§2 principle 2: one relation per question. Nothing on the other two axes changes this
    axis's answer -- which is what v1's seven-way `translate` could not say (failure F4)."""
    for value, expected in ((+5.0, "+"), (-5.0, "-"), (0.0, "hold"),
                            (+0.3, "hold"), (-0.3, "hold"),   # exactly the printed band
                            (+0.4, "+"), (-0.4, "-")):
        offset = [0.0, 0.0, 0.0]
        offset[index] = value
        labels = v2.labels_from_waypoint(offset, 0.0, False, "reach")
        assert labels[axis] == expected
        # The other two axes are untouched by this one.
        for other in ("move_x", "move_y", "move_z"):
            if other != axis:
                assert labels[other] == "hold"
    # And an axis is not perturbed by a large offset on another axis -- neither its direction
    # nor, now, its size.
    noisy = [0.0, 0.0, 0.0]
    noisy[index] = 0.1
    noisy[(index + 1) % 3] = 30.0
    labels = v2.labels_from_waypoint(noisy, 0.0, False, "reach")
    assert labels[axis] == "hold" and labels[f"size_{axis[-1]}"] == "small"


def test_each_size_band_is_read_against_the_step_above_it():
    """The one implementation warning of note §4, which cost the expert 34 of its 85 points:
    reading each band against its **own** step answers `large` for a 1.2 cm error, the hand ends
    4.6 cm above the rim when the fingers shut, and the same vocabulary scores 51/100."""
    assert v2.step_size_for(1.2) == "medium"        # not `large`: 1.2 < 0.7 x 5.0
    assert v2.step_size_for(20.0) == "large"
    assert v2.step_size_for(3.6) == "large"
    assert v2.step_size_for(3.5) == "large"         # 0.7 x the large step, and the band is open
    assert v2.step_size_for(3.4) == "medium"
    assert v2.step_size_for(1.2) == "medium"        # just above 0.7 x the medium step (1.17)
    assert v2.step_size_for(1.1) == "small"
    assert v2.step_size_for(0.0) == "small"
    # Both bands are above a half, because the arm does not stop when the command does:
    # sustained travel runs ~21 % above from-rest (note §2).
    assert c2.HOLD_BAND == 0.6 and c2.FINE_BAND == 0.7


def test_each_axis_gets_its_own_size_and_the_shared_step_stays_as_the_ablation():
    labels = v2.labels_from_waypoint([6.0, 1.5, 0.1], 0.0, False, "reach",
                                     qids=v2.ALL_MOTION_QIDS)
    assert labels["size_x"] == "large"      # 6.0 cm: still travelling
    assert labels["size_y"] == "medium"     # 1.5 cm: close
    assert labels["size_z"] == "small"      # inside the hold band, and not moving
    assert labels["move_z"] == "hold"
    # The ablation's single `step` is the largest offset *still to be travelled*.
    assert labels["step"] == "large"
    assert v2.largest_remaining([0.1, 1.5, 6.0]) == pytest.approx(6.0)
    assert v2.largest_remaining([0.1, 0.1, 0.1]) == 0.0


def test_yaw_is_a_sign_against_its_own_tolerance():
    for value, expected in ((+12.0, "+"), (-12.0, "-"), (0.0, "hold"), (4.9, "hold")):
        labels = v2.labels_from_waypoint([0.0, 0.0, 0.0], value, False, "reach",
                                         qids=v2.ALL_MOTION_QIDS)
        assert labels["yaw"] == expected


def test_grip_is_a_latch_decided_by_the_stage_and_not_a_predicted_outcome():
    """Failure F1: v1 asked for a calibrated probability that closing *would work*, and a
    well-calibrated answer to a rare outcome never crosses 0.5, so the arm never closed. v2 asks
    what the fingers should do, and the answer is the **stage** -- which is exactly how
    `expert.decide` decides it, sub-stage by sub-stage, and the stage is a fact the state
    prints."""
    far, at = [0.0, 0.0, 6.0], [0.1, -0.2, 0.3]   # `at` is inside the 1.3 cm arrival band
    # reach (the expert's `approach` and `descend`): open, always -- whatever the fingers do.
    assert v2.labels_from_waypoint(far, 0.0, False, "reach")["grip"] is False
    assert v2.labels_from_waypoint(at, 0.0, True, "reach")["grip"] is False
    # grasp (its `close`) and lift: shut. The stage is only ever entered at the rim.
    assert v2.labels_from_waypoint(far, 0.0, False, "grasp")["grip"] is True
    assert v2.labels_from_waypoint(far, 0.0, True, "lift")["grip"] is True
    # carry: shut while the fingers still have it; a dropped bowl leaves them and the state
    # says `holding nothing`.
    assert v2.labels_from_waypoint(far, 0.0, True, "carry")["grip"] is True
    assert v2.labels_from_waypoint(far, 0.0, False, "carry")["grip"] is False
    # place (its `lower` then `release`): hold until the hand has arrived, then open.
    assert v2.labels_from_waypoint(far, 0.0, True, "place")["grip"] is True
    assert v2.labels_from_waypoint(at, 0.0, True, "place")["grip"] is False
    # retreat: open, always.
    assert v2.labels_from_waypoint(at, 0.0, True, "retreat")["grip"] is False


def test_the_subgoal_label_maps_v1s_phase_names_onto_v2s_candidates():
    assert v2.normalise_phase("locate") == "reach"      # grounding is a separate question now
    for name in v2questions.SUBGOAL_CANDIDATES:
        assert v2.normalise_phase(name) == name
    with pytest.raises(KeyError):
        v2.normalise_phase("hovering")
    assert v2.labels_from_waypoint([0, 0, 0], 0.0, True, "carry")["subgoal"] == "carry"
    # `expert.py`'s own eight-state phase machine maps through here, so an expert-labelled row
    # and a tracker-labelled one name the same stage.
    assert v2.PHASE_FROM_EXPERT["close"] == "grasp"
    assert v2.PHASE_FROM_EXPERT["lower"] == "place"
    assert set(v2.PHASE_FROM_EXPERT.values()) <= set(v2questions.SUBGOAL_CANDIDATES)


# --------------------------------------------------------------------------- the state string


def test_the_rim_rule_has_two_copies_in_this_tree_and_they_agree():
    """`v2.state.rim_point` and `expert.rim_point` are two implementations of one measured rule
    (`2026-09-21-stage9-grasp-feasibility.md`: `target_xy + r * normalize(eef_xy - target_xy)`,
    `z = target_z + dz`). Grasping the *centre* holds 0.16-0.29 against 0.73-0.93 at the
    demonstrators' rim pose -- failure F5b."""
    rng = np.random.default_rng(3)
    cases = [((0.30, 0.10), np.array([0.10, 0.10, 0.90])),
             ((-0.10, 0.10), np.array([0.10, 0.10, 0.90])),
             ((0.10, 0.10), np.array([0.10, 0.10, 0.90]))]   # the degenerate hand-above case
    cases += [(rng.uniform(-0.3, 0.3, 2), rng.uniform([-0.3, -0.3, 0.9], [0.3, 0.3, 1.0]))
              for _ in range(50)]
    for eef_xy, pos in cases:
        mine = v2.rim_point(eef_xy, pos, 0.05, 0.025)
        theirs = expert.rim_point(pos[:2], eef_xy, 0.05)
        assert mine[:2] == pytest.approx(theirs, abs=1e-9)
        assert mine[2] == pytest.approx(float(pos[2]) + 0.025, abs=1e-9)
    # `dz` is the expert's **measurement**, not the note's midpoint: at +0.045 the fingers shut
    # to 2 mm on air, at +0.025 the same close catches the rim (expert note §1.2).
    assert v2.DEFAULT_RIM_DZ_M == expert.GRASP_DZ == 0.025
    assert v2.DEFAULT_RIM_RADIUS_M == expert.RIM_RADIUS == 0.05


def test_the_rim_direction_is_the_wrists_closing_axis_not_the_approach_side():
    """Expert note §1.1, and it is worth six episodes on task 2 alone (3/10 -> 9/10, same
    radius). The fingers close along the gripper's local y and span 8 cm about a rim ~6 cm
    across: radially, one finger goes inside the bowl and one outside; tangentially, both land on
    the ring and the descent pushes the bowl across the table."""
    pos = np.array([0.10, 0.10, 0.90])
    # A wrist rotated so its closing axis is world +x, with the hand approaching from +y.
    proprio = np.array([0.10, 0.30, 1.00, 0.0, 0.0, -np.pi / 2, 0.04, -0.04])
    wp = v2.waypoint_for(proprio, {"b_1": {"pos": pos}, "p_1": {"pos": pos}}, "b_1", "p_1",
                         "grasp")
    direction = expert.grasp_direction(proprio, pos[:2])
    assert wp.point[:2] == pytest.approx(pos[:2] + 0.05 * direction)
    # …which is *not* where the approach-side rule would have put it.
    assert wp.point[:2] != pytest.approx(v2.rim_point(proprio[:2], pos, 0.05, 0.025)[:2])
    assert "closing axis" in wp.label


def test_the_place_waypoint_aims_the_bowl_not_the_hand():
    """Expert note §1.5: a rim grasp holds the bowl a rim-radius to one side, so aiming the hand
    at the plate's centre lands the bowl beside it and LIBERO's `on(bowl, plate)` never fires --
    observed as a clean pick, carry and release that scored nothing."""
    objects = {"akita_black_bowl_1": {"pos": np.array([0.05, 0.10, 1.00])},
               "plate_1": {"pos": np.array([0.20, 0.00, 0.92])}}
    proprio = np.array([0.00, 0.10, 1.05, 0, 0, 0, 0.01, -0.01])
    wp = v2.waypoint_for(proprio, objects, "akita_black_bowl_1", "plate_1", "carry")
    carried = objects["akita_black_bowl_1"]["pos"][:2] - proprio[:2]
    assert wp.point[:2] == pytest.approx(objects["plate_1"]["pos"][:2] - carried)
    # The bowl, not the hand, is what ends up over the plate.
    assert (wp.point[:2] + carried) == pytest.approx(objects["plate_1"]["pos"][:2])
    assert wp.label == "bowl_1 above plate_1"


def test_the_waypoint_follows_the_stage_and_walks_the_retry_heights():
    proprio = np.array([0.0, 0.0, 1.05, 0, 0, 0, 0.04, -0.04])
    for stage, label in (("reach", "rim of bowl_1, closing axis"),
                         ("grasp", "rim of bowl_1, closing axis"),
                         ("lift", "clear above bowl_1"),
                         ("carry", "bowl_1 above plate_1"),
                         ("place", "bowl_1 above plate_1"),
                         ("retreat", "retreat above plate_1")):
        wp = v2.waypoint_for(proprio, OBJECTS, "akita_black_bowl_1", "plate_1", stage)
        assert wp is not None and wp.label == label
    assert v2.waypoint_for(proprio, OBJECTS, None, None, "reach") is None
    # A failed grasp comes back at the next candidate's height (`expert.GRASP_CANDIDATES`):
    # upwards first, because too high shuts on air and too low pushes the bowl away before it
    # shuts. This plainer geometry has no notion of the candidate's *side* -- it re-reads the
    # closing axis every decision -- so it walks the heights alone.
    heights = [v2.waypoint_for(proprio, OBJECTS, "akita_black_bowl_1", "plate_1", "grasp",
                               attempts=n).point[2] for n in range(4)]
    assert heights == pytest.approx([OBJECTS["akita_black_bowl_1"]["pos"][2] + 0.025 + dz
                                     for dz in expert.RETRY_DZ])


def test_the_serialised_state_is_section_4s_template():
    """The golden string. Every line after `Task:` is computed by code from observations (§2
    principle 6); distances are gripper-relative signed centimetres, the waypoint is named in the
    text (principle 1) and each axis gets the line its own question asks about (principle 2)."""
    tracker = _tracker()
    proprio = np.array([-0.02, -0.02, 1.10, 3.14, 0.0, 0.0, 0.04, -0.04])
    for t in range(0, 40, 5):
        p = proprio.copy()
        p[0] += 0.004 * t
        p[1] += 0.0035 * t
        p[2] -= 0.0042 * t
        tracker.observe(step=t, proprio=p, objects=OBJECTS, yaw_err_deg=1.0)
        text = v2.serialise_v2(p, OBJECTS, INSTRUCTION, tracker)
        tracker.answer(tracker.gold_answers())
    assert text == "\n".join([
        "Robot: Franka Panda, gripper-relative frame; x right, y forward, z up; "
        "distances in cm.",
        "Task: pick up the black bowl between the plate and the ramekin and place it on "
        "the plate",
        "Target: bowl_1 (chosen at t=0). Destination: plate_1.",
        "Subgoal so far: reach (approach). Done: nothing. Decision 8 of 44; 36 left.",
        "Gripper: open 8.0 cm; holding nothing.",
        "Wrist yaw error: +1.0 deg [tolerance 8.0]",
        "Waypoint (rim of bowl_1, closing axis): x -10.0, y +4.8, z -0.8   "
        "[tolerance 0.3; arrived within 1.3]",
        "  x: -10.0 is outside tolerance -> not aligned",
        "  y: +4.8 is outside tolerance -> not aligned",
        "  z: -0.8 is outside tolerance -> not aligned",
        "Largest remaining offset: 10.0 cm (large range: 3.50 and above).",
        "Step bands: small below 1.17 cm, medium 1.17 to 3.50 cm, large 3.50 and above; "
        "one step executes 0.5 / 1.7 / 5.0 cm.",
        "Other objects: bowl_2 x -24.0 y +11.8 z -3.3; ramekin_1 x -24.0 y -0.7 z -3.3; "
        "plate_1 x +5.0 y -1.7 z -2.8.",
        "Events: t=0 start, waypoint 17.5 cm away | t=5 within 20 cm | t=15 within 10 cm",
        "Attempts: grasps 0, held 0. Closest so far 8.0 cm at t=25. Moved 23.7 cm, "
        "net 23.7 cm (not looping).",
        "Last 3: t=20 x-l z-l, open -> 0.1 cm closer | "
        "t=25 x-l y-m z-l, open -> 1.2 cm farther | "
        "t=30 x-l y-l z-m, open -> 1.9 cm farther",
    ])
    # And a tracker asked for the question set a pre-wrist checkpoint was trained on prints the
    # same state **without** that line -- which is how those weights keep being served.
    without = _tracker(qids=v2.active_qids(yaw=False))
    without.observe(step=0, proprio=proprio, objects=OBJECTS, yaw_err_deg=1.0)
    assert "Wrist yaw" not in v2.serialise_v2(proprio, OBJECTS, INSTRUCTION, without)


def test_the_annotation_is_the_ablation_and_only_the_annotation():
    """§4: whether `-> not aligned` stays is an ablation -- with it the task is reading, without
    it one comparison. Turning it off changes those three lines and nothing else."""
    tracker = _tracker()
    proprio = np.array([-0.02, -0.02, 1.10, 3.14, 0.0, 0.0, 0.04, -0.04])
    tracker.observe(step=0, proprio=proprio, objects=OBJECTS)
    on = v2.serialise_v2(proprio, OBJECTS, INSTRUCTION, tracker, annotate=True).splitlines()
    off = v2.serialise_v2(proprio, OBJECTS, INSTRUCTION, tracker, annotate=False).splitlines()
    assert len(on) == len(off)
    differing = [i for i, (a, b) in enumerate(zip(on, off)) if a != b]
    assert [on[i].split(":")[0].strip() for i in differing] == ["x", "y", "z"]
    for i in differing:
        assert "aligned" in on[i] and "aligned" not in off[i]
        assert off[i].split(":")[1].strip() in on[i]


def test_the_state_carries_no_absolute_coordinates_and_no_quaternions():
    """§4: "quaternions and absolute coordinates gone (they carried no decision)". Every number
    the model reads is gripper-relative centimetres."""
    tracker = _tracker()
    proprio = np.array([-0.02, -0.02, 1.10, 3.14, 0.0, 0.0, 0.04, -0.04])
    tracker.observe(step=0, proprio=proprio, objects=OBJECTS)
    text = v2.serialise_v2(proprio, OBJECTS, INSTRUCTION, tracker)
    assert "quat" not in text
    assert "eef_pos" not in text
    assert "akita_black_bowl_1" not in text          # short ids only
    assert "1.10" not in text and "0.92" not in text  # no world-frame metres


def test_serialising_before_observing_is_an_error():
    with pytest.raises(ValueError):
        v2.serialise_v2(np.zeros(8), OBJECTS, INSTRUCTION, _tracker())


def test_the_tracker_records_the_retreat_stage_v1s_phase_rules_have_no_name_for():
    tracker = _tracker()
    proprio = np.array([0.17, 0.085, 1.10, 0, 0, 0, 0.04, -0.04])
    tracker.observe(step=0, proprio=proprio, objects=OBJECTS, released=True)
    assert tracker.subgoal == "retreat"
    text = v2.serialise_v2(proprio, OBJECTS, INSTRUCTION, tracker)
    assert "Subgoal so far: retreat (approach)." in text
    assert "Waypoint (retreat above plate_1)" in text
    assert v2.parse(text)["grip"] is False


def test_the_tracker_counts_and_the_counters_say_when_it_is_looping():
    """Failure F6: run 3's model hovered beside the bowl for two hundred steps and the state
    could not say so."""
    tracker = _tracker()
    base = np.array([-0.02, -0.02, 1.10, 3.14, 0.0, 0.0, 0.04, -0.04])
    for t in range(0, 40, 5):
        p = base.copy()
        p[0] += 0.02 * (1 if (t // 5) % 2 else -1)
        tracker.observe(step=t, proprio=p, objects=OBJECTS)
        tracker.answer(tracker.gold_answers())
    assert "looping" in tracker.counters_line()
    assert "not looping" not in tracker.counters_line()
    assert tracker.decisions == 8


# ------------------------------------------------------------------- gate G3: the text parser


def test_the_parser_reads_only_the_state_string():
    tracker = _tracker()
    proprio = np.array([-0.02, -0.02, 1.10, 3.14, 0.0, 0.0, 0.04, -0.04])
    tracker.observe(step=0, proprio=proprio, objects=OBJECTS, yaw_err_deg=-9.0)
    text = v2.serialise_v2(proprio, OBJECTS, INSTRUCTION, tracker)
    facts = v2.read(text)
    assert facts["tolerance"] == 0.3
    assert facts["arrival"] == 1.3
    assert facts["holding"] is False
    assert facts["phase"] == "reach"
    assert v2.parse(text) == tracker.gold_answers()
    assert tuple(v2.parse(text)) == v2.PARSED_QIDS == v2.ACTIVE_QIDS


def test_a_state_missing_a_number_a_label_depends_on_is_refused_not_guessed():
    with pytest.raises(v2.UnparseableState):
        v2.parse("Robot: …\nTask: pick up the bowl\n")


def _random_episode(rng, *, snap: bool, qids=None):
    """One synthetic decision point. `snap` places the hand on the waypoint's own axis so the
    offsets land on chosen grid values -- including the exact tolerance and the exact step
    thresholds, which is where a parser that rounded differently would break."""
    radius = 0.05
    dz = 0.03
    objects = {
        "akita_black_bowl_1": {"pos": rng.uniform([-0.25, -0.25, 0.88], [0.25, 0.25, 0.98])},
        "akita_black_bowl_2": {"pos": rng.uniform([-0.25, -0.25, 0.88], [0.25, 0.25, 0.98])},
        "plate_1": {"pos": rng.uniform([-0.25, -0.25, 0.88], [0.25, 0.25, 0.98])},
    }
    stage = rng.choice(["reach", "grasp", "lift", "carry", "place", "retreat"])
    kwargs = {} if qids is None else {"qids": qids}
    tracker = _tracker(target="akita_black_bowl_1", destination="plate_1",
                       rim_radius_m=radius, rim_dz_m=dz, **kwargs)
    if snap:
        # A hand due +x of the bowl with an identity wrist, whose closing axis is world y --
        # signed toward the hand, `grasp_direction` returns +x, so the rim point is exactly
        # bowl + (r, 0, dz) and the offsets are whatever grid values we choose.
        # The exact hold band (0.3), the size thresholds either side (1.1/1.2 around 1.17,
        # 3.5/3.6 around 3.50) and the arrival band (1.3) are all on the grid: that is where a
        # parser that rounded differently would break.
        grid = [0.0, 0.2, 0.3, 0.4, 1.1, 1.2, 1.3, 1.4, 3.4, 3.5, 3.6, 12.3,
                -0.3, -1.2, -1.3, -3.5, -20.1]
        bowl = objects["akita_black_bowl_1"]["pos"]
        wanted = np.array([rng.choice(grid) for _ in range(3)]) / 100.0
        eef = np.array([bowl[0] + radius, bowl[1], bowl[2] + dz]) - wanted
        eef[0] = max(eef[0], bowl[0] + 0.001)      # stay on the +x side of the bowl
    else:
        eef = rng.uniform([-0.35, -0.35, 0.90], [0.35, 0.35, 1.25])
    width = float(rng.choice([0.08, 0.075, 0.01, 0.0]))
    wrist = np.zeros(3) if snap else rng.uniform(-3.2, 3.2, 3)
    proprio = np.concatenate([eef, wrist, [width / 2, -width / 2]])
    tracker.observe(step=int(rng.integers(0, 200)), proprio=proprio, objects=objects,
                    yaw_err_deg=float(rng.choice([0.0, 4.9, 5.0, 5.1, -5.0, -12.3, 30.0])),
                    released=(stage == "retreat"))
    return tracker, proprio, objects


@pytest.mark.parametrize("annotate", [True, False])
def test_gate_g3_the_parser_reproduces_every_label_from_the_text_alone(annotate):
    """**Gate G3's parser half**, spec §5: "a rule-based parser of the state string alone
    reproduces every label -- parser 100 %".

    Two thousand synthetic states per annotation setting, half of them snapped onto the exact
    tolerance and the exact step thresholds. The comparison is against `TrackerV2.gold_answers()`
    -- the answers the expert gives from the full-precision geometry, rounded through
    `rounded_cm` exactly as the string is -- so this is not the parser checking its own working:
    it is the claim that nothing the expert knows is missing from the text (§2 principle 1).

    Run with the annotations and without them, because §4 makes that an ablation and a label that
    survived only the annotated form would be a label the harder condition cannot teach.
    """
    rng = np.random.default_rng(20260921)
    checked = 0
    for i in range(2000):
        tracker, proprio, objects = _random_episode(rng, snap=bool(i % 2))
        text = v2.serialise_v2(proprio, objects, INSTRUCTION, tracker, annotate=annotate)
        assert v2.parse(text, tracker.qids) == tracker.gold_answers(), text
        checked += 1
    assert checked == 2000


def test_the_parser_is_indifferent_to_the_annotation():
    rng = np.random.default_rng(7)
    for i in range(200):
        tracker, proprio, objects = _random_episode(rng, snap=bool(i % 2))
        on = v2.serialise_v2(proprio, objects, INSTRUCTION, tracker, annotate=True)
        off = v2.serialise_v2(proprio, objects, INSTRUCTION, tracker, annotate=False)
        assert v2.parse(on, tracker.qids) == v2.parse(off, tracker.qids)


# ------------------------------------------------------------------------- gate G6: the budget


def test_the_parser_reproduces_the_expert_label_with_yaw_and_the_shared_step_switched_on():
    """The ablations are ablations, not a second label rule: turning `yaw` and the shared `step`
    back on changes which keys are asked and nothing about how they are answered."""
    rng = np.random.default_rng(11)
    for i in range(300):
        tracker, proprio, objects = _random_episode(rng, snap=bool(i % 2),
                                                    qids=v2.ALL_MOTION_QIDS)
        text = v2.serialise_v2(proprio, objects, INSTRUCTION, tracker)
        assert "Wrist yaw error:" in text
        assert v2.parse(text, tracker.qids) == tracker.gold_answers(), text


def test_the_measured_path_fits_the_budget_with_margin():
    """Gate G6: "token count of the state + longest question < 1024 with margin".

    `MEASURED_PATH_TOKENS_V2` is measured, not estimated -- the Qwen3-0.6B tokenizer at NanoJev's
    pinned revision, in the robojev recipe environment, over a worst-case five-object
    LIBERO-Spatial scene with the suite's longest instruction, the annotations on, a full events
    line and a full history, across every (question, candidate) path including the once-per-
    episode `target` candidates built from that scene. The tokenizer is not importable from this
    test environment, so the number lives in the module and this is its guard; re-measure it
    whenever the state text or the question set changes, because upstream raises rather than
    truncating an oversized path.
    """
    assert v2.MEASURED_PATH_TOKENS_V2 == 682
    assert v2.MEASURED_PATH_TOKENS_V2 < v2.MAX_PATH_TOKENS_V2 == 1024
    assert 1024 - v2.MEASURED_PATH_TOKENS_V2 > 300      # the margin G6 asks for


def test_the_module_is_light():
    """A policy server and a bare install both import this; neither has torch."""
    import subprocess
    import sys
    out = subprocess.run([sys.executable, "-c",
                          "import robojev, sys; "
                          "print(sorted(m for m in sys.modules if m.split('.')[0] in "
                          "('torch','transformers','libero','robosuite','h5py','mujoco')))"],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


# ------------------------------------------- the label function against the expert that measured


def test_the_label_function_is_the_experts_axiswise_rule():
    """**The text-derived label and the expert cannot drift.**

    `robojev.expert.axiswise_answers` is the rule the 85/100 run was measured with
    (`notes/2026-09-21-stage9-scripted-expert.md` §4). `labels_from_waypoint` is what a state
    string is labelled by. If they ever disagree, every training row is labelled by something no
    closed-loop number was ever measured for -- which is failure F2 all over again.

    A thousand random error vectors plus every band boundary, in the expert's metres and in the
    state's centimetres, compared answer for answer on the six motion questions.
    """
    rng = np.random.default_rng(1234)
    edges = [0.0, 0.29, 0.3, 0.31, 1.16, 7.0 / 6.0, 1.17, 3.49, 3.5, 3.51, 9.9, -0.3, -3.5]
    vectors = [np.array([a, b, c]) for a in edges[:6] for b in edges[6:] for c in (0.0, -4.2)]
    vectors += [rng.uniform(-20.0, 20.0, 3) for _ in range(1000)]
    for offset_cm in vectors:
        mine = v2.labels_from_waypoint(offset_cm, 0.0, False, "reach")
        theirs = expert.axiswise_answers(np.asarray(offset_cm) / 100.0)
        for axis in "xyz":
            assert mine[f"move_{axis}"] == theirs[f"move_{axis}"], (offset_cm, axis)
            assert mine[f"size_{axis}"] == theirs[f"size_{axis}"], (offset_cm, axis)


def test_the_label_functions_hold_band_is_the_experts_own():
    """0.6 x the small step, and the tolerance the state prints is that same number -- so the
    band the model is told about is the band the label was decided by."""
    assert v2.DEFAULT_TOLERANCE_CM == 0.3 == pytest.approx(c2.HOLD_BAND * v2.STEP_SMALL_CM)
    assert v2.DEFAULT_ARRIVAL_CM == pytest.approx(expert.XY_TOL * 100.0) == 1.3
    # Arrival is deliberately looser than `hold`: one is "this axis needs no step", the other is
    # "the phase may advance, the fingers may close".
    assert v2.DEFAULT_TOLERANCE_CM < v2.DEFAULT_ARRIVAL_CM


def test_v2s_active_set_is_the_experts_own_v2_vocabulary():
    """The label source and the question set are one vocabulary, not two that happen to match:
    every question the set asks, the expert answers."""
    assert set(v2.ACTIVE_QIDS) == set(expert.V2_QIDS)
    assert v2.MOVE_QIDS == expert.V2_MOVE_QIDS
    assert v2.SIZE_QIDS == expert.V2_SIZE_QIDS
    assert v2.PHASE_FROM_EXPERT is expert.V2_PHASE_SUBGOAL


@pytest.mark.sim
def test_the_label_function_agrees_with_the_expert_over_a_real_episode():
    """The same equality, over the states an expert rollout actually visits rather than states
    this test invented -- the only ones the model will ever be asked about.

    Every decision of one LIBERO-Spatial episode is driven by the expert in the per-axis
    vocabulary, a `TrackerV2` observes the same states, and three things are asserted at each:
    the motion answers agree with `expert.answers_v2`, the tracker's own gold agrees with them
    on the axes, and **`parse(serialise_v2(...))` reproduces the tracker's gold** -- gate G3 on
    real states. ~40 s on a CPU, most of it the env build.
    """
    from robojev.envs.libero import LiberoEnv

    env = LiberoEnv("libero_spatial", 0, render_size=128)
    seen = []
    try:
        tracker = _tracker(horizon=44)

        def on_decision(index, choices, meta, state8, privileged):
            if index == 0:
                # The same rule the expert names its own target with, so the tracker and the
                # controller are acting on one bowl.
                roles = expert.resolve_roles(privileged, env.instruction,
                                             np.asarray(state8, dtype=float)[0:3])
                tracker.commit(roles["target"], roles["destination"], step=0, source="rule")
            tracker.observe(step=index * 5, proprio=state8, objects=privileged)
            text = v2.serialise_v2(state8, privileged, env.instruction, tracker)
            gold = tracker.gold_answers()
            assert v2.parse(text, tracker.qids) == gold, text
            theirs = expert.axiswise_answers(np.asarray(meta["error"], dtype=float))
            mine = v2.labels_from_waypoint(np.asarray(meta["error"], dtype=float) * 100.0,
                                           0.0, False, "reach")
            for axis in "xyz":
                assert mine[f"move_{axis}"] == theirs[f"move_{axis}"]
                assert mine[f"size_{axis}"] == theirs[f"size_{axis}"]
            tracker.answer(gold)
            seen.append(index)

        result = expert.run_episode(env, 0, max_steps=220,
                                    on_decision=on_decision)
    finally:
        env.close()
    assert result["success"], result
    assert len(seen) >= 10


# ------------------------------------------------------------- the grip latch's structural guard


def test_the_guard_is_a_table_over_the_two_facts_the_state_prints():
    """Note §5's cliff, and its own recommendation. At p = 0.10 the expert scores **1/20** with
    `grip` corrupted and 13/20 with it exempt: a wrong axis is undone by the next decision
    because the waypoint is re-read from the state, but closing the fingers mid-approach wastes a
    retry and opening them mid-carry drops the bowl. "No motion accuracy rescues a `grip` head
    below ~0.98", so the latch is protected structurally instead -- by the table, and by nothing
    else. Every cell of it, both ways round."""
    permitted = {(subgoal, arrived, proposed):
                 v2.latch_permitted(proposed, not proposed, subgoal=subgoal, arrived=arrived)
                 for subgoal in v2.LATCH_TABLE
                 for arrived in (False, True)
                 for proposed in (False, True)}
    closing = {(subgoal, arrived): permitted[(subgoal, arrived, True)]
               for subgoal, arrived in [(s, a) for s in v2.LATCH_TABLE for a in (False, True)]}
    opening = {(subgoal, arrived): permitted[(subgoal, arrived, False)]
               for subgoal, arrived in [(s, a) for s in v2.LATCH_TABLE for a in (False, True)]}

    # Shutting the fingers is the plan's own act, and `grasp` is the stage that is it.
    assert closing == {("grasp", False): True, ("grasp", True): True,
                       ("reach", False): False, ("reach", True): False,
                       ("lift", False): False, ("lift", True): False,
                       ("carry", False): False, ("carry", True): False,
                       ("place", False): False, ("place", True): False,
                       ("retreat", False): False, ("retreat", True): False}
    # Opening them is never wrong while reaching or retreating -- which is what makes a retry
    # possible -- right over the destination, and nowhere else.
    assert opening == {("reach", False): True, ("reach", True): True,
                       ("retreat", False): True, ("retreat", True): True,
                       ("place", True): True, ("place", False): False,
                       ("grasp", False): False, ("grasp", True): False,
                       ("lift", False): False, ("lift", True): False,
                       ("carry", False): False, ("carry", True): False}
    # A latch that is already where it is asked to be is not a change at all.
    for subgoal in v2.LATCH_TABLE:
        for state in (False, True):
            assert v2.latch_permitted(state, state, subgoal=subgoal, arrived=False)


def test_the_table_is_total_over_the_question_set_s_own_sub_goals():
    """A sub-goal the table has no row for is an episode that dies in the serving path."""
    from robojev import questions

    assert set(v2.LATCH_TABLE) == set(questions.SUBGOAL_CANDIDATES)


def test_the_guard_applies_a_permitted_change_at_once_and_holds_a_refused_one():
    """No confirmation count: inside a window the change is the plan's own next action, so
    confirming it only costs a decision; outside one it is refused however often it is asked."""
    latch = v2.GripLatch()
    assert latch.closed is False
    for _ in range(3):
        assert latch.update(True, subgoal="reach", arrived=False) is False
    latch.reset()
    assert latch.update(True, subgoal="grasp", arrived=True) is True
    # And once closed it stays closed through a lift and a carry, however the model answers.
    assert latch.update(False, subgoal="lift", arrived=True) is True
    assert latch.update(False, subgoal="carry", arrived=False) is True


def test_the_latch_opens_over_the_destination_and_whenever_the_plan_is_reaching_again():
    latch = v2.GripLatch(closed=True)
    assert latch.update(False, subgoal="carry", arrived=False) is True
    assert latch.update(False, subgoal="place", arrived=False) is True    # not there yet
    assert latch.update(False, subgoal="place", arrived=True) is False

    # **The retry.** The plan reopens and re-approaches when the grasp caught nothing, and the
    # guard used to refuse it: it read `retrying` off `TrackerV2.needs_regrounding`, which the
    # re-grounding cleared one decision later, so 0 of 9 retries in run 7 ever succeeded. A hand
    # that is reaching has nothing to drop, so opening is simply permitted there.
    latch = v2.GripLatch(closed=True)
    assert latch.update(False, subgoal="reach", arrived=False) is False


def test_the_guard_logs_every_refusal_into_the_memory_block():
    tracker = _tracker()
    proprio = np.array([-0.02, -0.02, 1.10, 3.14, 0.0, 0.0, 0.04, -0.04])
    tracker.observe(step=0, proprio=proprio, objects=OBJECTS)
    tracker.latch.update(True, subgoal="reach", arrived=False, log=tracker.log_latch)
    text = v2.serialise_v2(proprio, OBJECTS, INSTRUCTION, tracker)
    assert "grip close refused (geometry)" in text
    # The geometry is the only reason there is, now that the confirmation count is gone: a
    # permitted change is applied on the decision it is asked on and logs nothing.
    refusals = text.count("refused")
    tracker.latch.reset()
    tracker.latch.update(True, subgoal="grasp", arrived=True, log=tracker.log_latch)
    assert v2.serialise_v2(proprio, OBJECTS, INSTRUCTION, tracker).count("refused") == refusals


def test_the_guard_is_an_ablation_and_its_config_is_reportable():
    unguarded = v2.GripLatch(v2.LatchGuard(enabled=False))
    assert unguarded.update(True, subgoal="reach", arrived=False) is True
    assert unguarded.config() == {"grip_guard": False}
    assert v2.DEFAULT_LATCH_GUARD.config() == {"grip_guard": True}
    assert v2.latch_permitted(True, False, subgoal="grasp", arrived=True)
    assert not v2.latch_permitted(True, False, subgoal="reach", arrived=True)


def test_the_label_function_still_reports_the_experts_raw_latch():
    """The guard protects a *model's* latch at run time. It never touches the gold answer, or a
    training row would be labelled with the guard's opinion instead of the expert's."""
    labels = v2.labels_from_waypoint([0.1, 0.1, 0.1], 0.0, False, "grasp")
    assert labels["grip"] is True                     # the stage is the decision, not a streak


# -------------------------------------------------------------- the serving hooks (plan 9c T2)


def _drive(tracker, objects=None, n=12, released_at=None):
    objects = OBJECTS if objects is None else objects
    base = np.array([-0.02, -0.02, 1.10, 3.14, 0.0, 0.0, 0.04, -0.04])
    text = None
    for i in range(n):
        p = base.copy()
        p[0] += 0.004 * i * 5
        p[1] += 0.0035 * i * 5
        p[2] -= 0.0042 * i * 5
        tracker.observe(step=i * 5, proprio=p, objects=objects,
                        released=(released_at is not None and i >= released_at))
        text = v2.serialise_v2(p, objects, INSTRUCTION, tracker)
        tracker.answer(tracker.gold_answers())
    return text


def test_two_trackers_fed_the_same_observations_render_the_same_bytes():
    """The whole reason `robojev` is shared code: a training row and an inference request
    must be the same function of the same numbers, or the fine-tune generalises to nothing."""
    a, b = _tracker(), _tracker()
    assert _drive(a) == _drive(b)
    assert a.settings() == b.settings()
    assert a.settings()["memory_rule"] == s2.MEMORY_RULE_V2
    assert a.settings()["annotate"] is True
    assert _tracker(annotate=False).settings()["annotate"] is False


def test_reset_empties_everything():
    """What stopped v1 leaking one episode's memory into the next."""
    tracker = _tracker()
    first = _drive(tracker)
    assert "t=25" in first
    tracker.reset()
    tracker.commit("akita_black_bowl_1", "plate_1", step=0, source="bddl")
    second = _drive(tracker, n=1)
    assert "t=5" not in second and "t=25" not in second
    assert "Last 3: none yet." in second
    assert "Moved 0.0 cm" in second
    assert "Decision 1 of 44" in second


def test_regrounding_is_triggered_by_a_failed_attempt_and_nothing_else():
    """Spec §3: the two answers are re-asked "only when the tracker records a failed attempt".
    On a scene with two identical bowls, a grasp that shut on nothing is the evidence that the
    wrong one was named (failure F3). Observation-only: no model in the loop."""
    tracker = _tracker()
    tracker.commit("akita_black_bowl_1", "plate_1", step=0, source="rule")
    assert tracker.committed() == {"target": "akita_black_bowl_1", "destination": "plate_1",
                                   "source": "rule", "step": 0}
    _drive(tracker, n=4)
    assert tracker.needs_regrounding() is False       # a clean approach asks nothing
    objects = {k: {"pos": np.array(v["pos"], dtype=float)} for k, v in OBJECTS.items()}
    close = np.array([0.02, 0.10, 0.95, 3.14, 0.0, 0.0, 0.005, -0.005])
    tracker.observe(step=100, proprio=close, objects=objects)
    assert tracker.needs_regrounding() is True        # the fingers shut and nothing came up
    objects["akita_black_bowl_1"]["pos"] = objects["akita_black_bowl_1"]["pos"] + [0, 0, 0.06]
    tracker.observe(step=105, proprio=close, objects=objects)
    assert tracker.needs_regrounding() is False       # something rose: the target was right


def test_an_offset_that_rounds_to_zero_is_not_written_as_a_direction():
    """`-0.0` reads as "just below" and it is not: at a tenth of a centimetre it is level."""
    objects = {"akita_black_bowl_1": {"pos": np.array([0.02, 0.10, 0.92])},
               "plate_1": {"pos": np.array([0.02, 0.10, 0.92 - 1e-7])}}
    tracker = _tracker()
    proprio = np.array([0.02, 0.10, 0.92, 3.14, 0.0, 0.0, 0.04, -0.04])
    tracker.observe(step=0, proprio=proprio, objects=objects)
    text = v2.serialise_v2(proprio, objects, INSTRUCTION, tracker)
    assert "-0.0" not in text
    assert "plate_1 x +0.0 y +0.0 z +0.0" in text


# ------------------------------------------ the plan: the expert's sub-stages, not a straight line


def test_the_plan_is_the_experts_sub_stages_and_not_a_line_to_the_rim():
    """**The waypoint the state names must be the one the expert steers to, in every sub-stage.**

    The expert's approach flies to `HOVER` (8 cm) above the rim point and descends only once x
    and y are aligned. A tracker whose `reach` waypoint *is* the rim point would be labelling
    "descend diagonally straight at the rim" -- a policy nobody has measured closed-loop, while
    the 85/100 of `notes/2026-09-21-stage9-scripted-expert.md` is the expert's, hover and all.
    """
    tracker = _tracker(instruction=INSTRUCTION)
    high = np.array([0.02, 0.10, 1.15, 0.0, 0.0, 0.0, 0.04, -0.04])   # right over the bowl
    tracker.observe(step=0, proprio=high, objects=OBJECTS)
    assert tracker.substage == "approach"
    bowl_z = float(OBJECTS["akita_black_bowl_1"]["pos"][2])
    # The approach's waypoint sits a hover above the rim, not on it.
    assert tracker.waypoint.point[2] == pytest.approx(
        bowl_z + expert.GRASP_DZ + expert.HOVER, abs=1e-6)
    text = v2.serialise_v2(high, OBJECTS, INSTRUCTION, tracker)
    assert "Subgoal so far: reach (approach)." in text
    assert "Waypoint (approach: 8 cm above the rim of bowl_1," in text
    assert v2.parse(text, tracker.qids) == tracker.gold_answers()
    # Once x and y are aligned at hover height the plan descends, and the waypoint is the rim.
    hover = tracker.waypoint.point
    at_hover = np.array([hover[0], hover[1], hover[2], 0.0, 0.0, 0.0, 0.04, -0.04])
    # The decision that *arrives* is still an `approach` -- the machine advances on its way out,
    # exactly as `expert.decide` does -- and the one after it descends.
    tracker.observe(step=5, proprio=at_hover, objects=OBJECTS)
    assert tracker.substage == "approach"
    assert v2.parse(v2.serialise_v2(at_hover, OBJECTS, INSTRUCTION, tracker),
                    tracker.qids)["move_z"] == "hold"
    tracker.observe(step=10, proprio=at_hover, objects=OBJECTS)
    assert tracker.substage == "descend"
    assert tracker.waypoint.point[2] == pytest.approx(bowl_z + expert.GRASP_DZ, abs=1e-6)
    text = v2.serialise_v2(at_hover, OBJECTS, INSTRUCTION, tracker)
    assert "Waypoint (descend: rim of bowl_1," in text
    assert v2.parse(text, tracker.qids) == tracker.gold_answers()


def test_the_state_names_the_sub_stage_and_the_side_of_the_bowl():
    """The waypoint jumps 8 cm in z between `approach` and `descend`, and it sits on whichever
    end of the wrist's closing axis the grasp candidate in force names -- an axis the expert
    latches at the first decision. Both are printed, so the text explains a waypoint that would
    otherwise look like it had moved for no reason the model can see."""
    tracker = _tracker(instruction=INSTRUCTION)
    proprio = np.array([0.02, 0.10, 1.15, 0.0, 0.0, 0.0, 0.04, -0.04])
    tracker.observe(step=0, proprio=proprio, objects=OBJECTS)
    text = v2.serialise_v2(proprio, OBJECTS, INSTRUCTION, tracker)
    side = next(w for w in ("+x side", "-x side", "+y side", "-y side") if w in text)
    assert side
    assert set(s2.SUBSTAGE_LABEL) == set(expert.PHASES)


def test_the_planner_is_the_experts_own_machine_and_the_fallback_is_the_geometry():
    """With an instruction the tracker steers by `expert.decide`'s phase machine -- imported, not
    re-implemented, so the plan and the 85/100 cannot drift apart. Without one it falls back to
    `waypoint_for`'s stage geometry, which is what a caller with no sentence (a fixture, a token
    measurement) wants."""
    proprio = np.array([0.02, 0.10, 1.15, 0.0, 0.0, 0.0, 0.04, -0.04])
    planned = _tracker(instruction=INSTRUCTION)
    planned.observe(step=0, proprio=proprio, objects=OBJECTS)
    bare = _tracker()
    bare.observe(step=0, proprio=proprio, objects=OBJECTS)
    assert planned.waypoint.point[2] > bare.waypoint.point[2]      # the hover
    assert bare.substage == "approach" and planned.substage == "approach"
    # Both are still fully parseable: the label is a function of the text either way.
    for tracker in (planned, bare):
        text = v2.serialise_v2(proprio, OBJECTS, INSTRUCTION, tracker)
        assert v2.parse(text, tracker.qids) == tracker.gold_answers()


@pytest.mark.sim
def test_the_closed_loop_runs_on_the_parsed_answers_alone():
    """**Gate G1 for the labels**, and the only number that means anything about them.

    The loop is: observe -> `serialise_v2` -> `parse` the string -> `chunk(..., MEASURED_STEPS)`
    -> step. Nothing in it asks the expert for an *answer*; the only thing the expert supplies is
    the plan (`v2.state.plan`, which is code planning, the NanoJev division of labour) and the
    once-per-episode grounding, which at serving time is the model's own `target` answer.

    Measured over 5 init states of each task: **task 0 5/5** (median 24 decisions) and **task 9
    5/5** (median 39), against the scripted expert's own 5/5 and 4/5 through `act`. This test
    runs one init state of each to keep it to a couple of minutes; the ten-episode sweep is
    `scratchpad/g1_loop.py`.
    """
    from robojev.envs.libero import LiberoEnv

    for task in (0, 9):
        env = LiberoEnv("libero_spatial", task, render_size=128)
        try:
            obs = env.reset(0)
            state8 = env.state_vector(obs)
            privileged = env.privileged(obs)
            roles = expert.resolve_roles(privileged, env.instruction,
                                         np.asarray(state8, dtype=float)[0:3])
            tracker = v2.TrackerV2(horizon=44, instruction=env.instruction)
            tracker.commit(roles["target"], roles["destination"], step=0, source="rule")
            success = False
            for decision in range(44):
                state8 = env.state_vector(obs)
                privileged = env.privileged(obs)
                tracker.observe(step=decision * 5, proprio=state8, objects=privileged)
                text = v2.serialise_v2(state8, privileged, env.instruction, tracker)
                answers = v2.parse(text, tracker.qids)
                tracker.answer(answers)
                for row in v2.chunk(answers, c2.MEASURED_STEPS):
                    result = env.step(row)
                    obs = result.obs
                    if result.done:
                        success = True
                        break
                if success:
                    break
        finally:
            env.close()
        assert success, f"task {task} did not finish on the parsed answers alone"
