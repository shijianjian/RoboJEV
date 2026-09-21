"""`robojev.expert`: the scripted expert's geometry, phase machine and retry.

Everything but the last test is synthetic -- hand-built poses and a hand-built phase carry, no
simulator -- which is the point of `decide` being a pure function: the controller's reasoning is
testable at a millisecond per case, and the one `sim`-marked test at the bottom is there to prove
the synthetic states are the same states the simulator produces.
"""
from __future__ import annotations

from importlib import import_module

import numpy as np
import pytest

from robojev import expert, skill

BOWL = "akita_black_bowl_1"
OTHER = "akita_black_bowl_2"
PLATE = "plate_1"
RAMEKIN = "glazed_rim_porcelain_ramekin_1"
INSTRUCTION = "pick up the black bowl between the plate and the ramekin and place it on the plate"


def scene(bowl=(-0.06, 0.20, 0.90), other=(-0.19, 0.32, 0.90), plate=(0.05, 0.21, 0.90),
          ramekin=(-0.20, 0.19, 0.90)) -> dict:
    """A LIBERO-Spatial-shaped privileged state: two identical bowls, a plate, a ramekin."""
    return {name: {"pos": np.asarray(pos, np.float32), "quat": np.asarray([0, 0, 0, 1], np.float32)}
            for name, pos in ((BOWL, bowl), (OTHER, other), (PLATE, plate), (RAMEKIN, ramekin))}


def proprio(pos, width: float = 0.08, axisangle=(np.pi, 0.0, 0.0)) -> np.ndarray:
    """The 8-d vector `LiberoEnv.state_vector` builds: position, axis-angle, two finger joints.

    The default orientation is LIBERO's own starting wrist, a half turn about x -- top-down, with
    the fingers closing along the world's y. `grasp_direction` reads it, so it decides which side
    of the bowl these tests' hand stands on.
    """
    return np.asarray([*pos, *axisangle, width / 2, -width / 2], np.float64)


def phase_at(name: str, **fields) -> dict:
    p = expert.new_phase()
    p["name"] = name
    p.update(fields)
    return p


# ------------------------------------------------------------------------------ the constants


def test_the_chunk_is_the_composer_s():
    """One number, read from the other side of the fine-tune.

    `import_module` rather than `import robojev.compose as compose`: the package's
    `__init__` re-exports a *function* called `compose`, and `import a.b as c` binds
    `getattr(a, "b")` when that succeeds -- so the plain form hands back the function and every
    attribute below raises.
    """
    compose = import_module("robojev.compose")

    assert expert.CHUNK_STEPS == compose.CHUNK_STEPS


def test_a_full_step_is_calibrated_against_what_the_arm_executes():
    """`EXECUTED_COARSE_PER_DELTA_T` is a fifth of what the composer *commands*, which is the
    whole reason the thresholds are not the commanded ones. If someone ever "fixes" it to the
    commanded value, every threshold in the controller silently grows five-fold."""
    compose = import_module("robojev.compose")

    assert expert.EXECUTED_COARSE_PER_DELTA_T < 0.3 * compose.COMMANDED_CM_PER_UNIT / 100.0
    assert expert.EXECUTED_COARSE_PER_DELTA_T == pytest.approx(0.050, abs=0.01)


# -------------------------------------------------------------------------- the rim geometry


def test_the_rim_point_sits_a_radius_toward_the_hand():
    point = expert.rim_point((0.0, 0.0), (0.20, 0.0), radius=0.05)
    assert point == pytest.approx([0.05, 0.0])


def test_the_rim_point_follows_whichever_side_the_hand_is_on():
    """The feasibility note's rule never needs to know which side a demonstrator used."""
    for eef, expected in (((0.0, 0.3), [0.0, 0.05]), ((-0.3, 0.0), [-0.05, 0.0]),
                          ((-0.3, -0.3), [-0.05 / np.sqrt(2), -0.05 / np.sqrt(2)])):
        assert expert.rim_point((0.0, 0.0), eef, radius=0.05) == pytest.approx(expected)


def test_the_rim_point_is_total_over_the_bowl_s_own_axis():
    """Degenerate input is the one case where any direction is as good as any other; it still has
    to be *a* direction, and the same one every time."""
    point = expert.rim_point((1.0, 2.0), (1.0, 2.0), radius=0.05)
    assert point == pytest.approx([1.05, 2.0])


def test_the_grasp_direction_is_the_wrist_s_closing_axis():
    """The fingers close along the gripper's local y; the rim point has to be on that axis or the
    two of them land on the rim ring instead of straddling it."""
    top_down = (np.pi, 0.0, 0.0)              # LIBERO's starting wrist: closing axis is world y
    for eef, expected in (((0.0, 0.5), [0.0, 1.0]),       # the hand is on the +y side
                          ((0.0, -0.5), [0.0, -1.0]),     # ...and on the -y side
                          ((0.5, 0.02), [0.0, 1.0]),      # approaching along x: still the y axis
                          ((0.5, -0.02), [0.0, -1.0])):
        direction = expert.grasp_direction(np.asarray([*eef, 1.0, *top_down, 0.04, -0.04]),
                                           (0.0, 0.0))
        assert direction == pytest.approx(expected, abs=1e-6)


def _matrix_to_axisangle(matrix) -> np.ndarray:
    """Rodrigues, backwards: the `(3,)` rotation vector of a `(3, 3)` matrix.

    The inverse of `expert._rotation`, written here because it is only ever needed to *build* a
    test input -- the expert reads an orientation and never writes one.
    """
    angle = float(np.arccos(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0)))
    if angle < 1e-12:
        return np.zeros(3)
    sine = float(np.sin(angle))
    if abs(sine) < 1e-8:
        # A half turn: the skew part vanishes, so the axis is read off `(R + I) / 2 = a aᵀ`.
        symmetric = matrix + np.eye(3)
        column = symmetric[:, int(np.argmax(np.diag(symmetric)))]
        return column / float(np.linalg.norm(column)) * angle
    axis = np.array([matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0],
                     matrix[1, 0] - matrix[0, 1]]) / (2.0 * sine)
    return axis * angle


def test_the_grasp_direction_turns_with_the_wrist():
    """A wrist yawed by 90 degrees closes along world x, and the rim point follows it."""
    yaw = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # Rz(90 degrees)
    axisangle = _matrix_to_axisangle(yaw @ expert._rotation((np.pi, 0.0, 0.0)))
    direction = expert.grasp_direction(np.asarray([0.5, 0.0, 1.0, *axisangle, 0.04, -0.04]),
                                       (0.0, 0.0))
    assert abs(direction[0]) == pytest.approx(1.0, abs=1e-6)
    assert direction[1] == pytest.approx(0.0, abs=1e-6)


def test_the_rotation_helper_round_trips_its_own_inverse():
    for v in ((np.pi, 0, 0), (0.3, -0.7, 1.1), (0, 0, 0)):
        assert _matrix_to_axisangle(expert._rotation(v)) == pytest.approx(np.asarray(v, float))


def test_the_rim_radius_is_the_measured_one():
    """5 cm: the feasibility note's per-task medians were 0.048-0.056 m."""
    assert 0.045 <= expert.RIM_RADIUS <= 0.06
    assert 0.015 <= expert.GRASP_DZ <= 0.055      # the note's measured dz band


# -------------------------------------------------------------------- the wrist and its turns


def test_an_axis_is_a_line_so_the_wrist_never_turns_more_than_a_right_angle():
    """`wrap_yaw`. Fingers along `+d` and along `-d` are the same grasp, so a rule that turned
    through 135 degrees to reach a pose 45 degrees away would spend six decisions buying
    nothing."""
    for given, expected in ((0, 0), (45, 45), (90, 90), (91, -89), (135, -45), (180, 0),
                            (225, 45), (270, 90), (-45, -45), (-135, 45), (359, -1)):
        assert expert.wrap_yaw(given) == pytest.approx(expected)
    assert all(-90 < expert.wrap_yaw(a) <= 90 for a in range(-720, 720, 7))


def test_the_yaw_error_is_zero_for_the_candidates_that_stand_on_the_wrist_s_own_axis():
    """Which is what keeps the eight tasks that never leave them answering `hold` for ever: the
    rim direction of a `turn` of 0 or 180 degrees *is* the closing axis."""
    hand = proprio((-0.30, -0.02, 1.18))
    axis = expert.grasp_direction(hand, (0.0, 0.0))
    for turn in (0.0, 180.0):
        assert expert.wrist_yaw_error(hand, expert.rotate_xy(axis, turn)) == pytest.approx(0.0)
    for turn in (45.0, 90.0, -45.0, -90.0, 135.0, -135.0):
        assert expert.wrist_yaw_error(hand, expert.rotate_xy(axis, turn)) == pytest.approx(
            expert.wrap_yaw(turn), abs=1e-6)


def test_a_candidate_that_needs_the_wrist_says_so_and_is_not_offered_without_it():
    """The generic half of serving a checkpoint that was trained before the wrist existed: a
    candidate is an instruction to the arm, and an instruction the served question set cannot
    express is not an alternative -- it is a decision that silently does something else."""
    assert expert.requires(expert.Grasp(0.0, 0.0)) == ()
    assert expert.requires(expert.Grasp(180.0, 0.0)) == ()
    assert expert.requires(expert.Grasp(-90.0, 0.0)) == ("yaw",)
    without = expert.candidates(("move_x", "move_y", "move_z", "grip", "subgoal"))
    assert {g.turn for g in without} == {0.0, 180.0}
    assert len(without) == len(expert.RETRY_DZ) * 2
    # ...and it is the list this module had before the wrist, in the order it had it.
    assert [(g.turn, g.dz) for g in without] == [
        (turn, dz) for dz in expert.RETRY_DZ for turn in (0.0, 180.0)]
    assert len(expert.candidates(("yaw",))) == len(expert.GRASP_CANDIDATES) == (
        len(expert.RETRY_DZ) * len(expert.GRASP_TURNS))


def test_the_candidate_order_is_grouped_by_the_wrist_and_then_read_off_the_scene():
    """`candidate_turns` is a **partition, not a sort**: what does not fit goes to the back and
    the measured order stands among the rest."""
    objects = scene()
    axis = np.array([0.0, 1.0])
    turns = expert.candidate_turns((0.0, 0.0, 0.90), objects, axis, expert.RIM_RADIUS)
    assert set(turns) == set(expert.GRASP_TURNS)
    # Nothing is in the way in an empty-ish scene, so the declared order stands, whole.
    assert turns[0] == 0.0
    assert set(turns[1:4]) == {180.0, 90.0, -90.0}
    assert set(turns[4:]) == {45.0, -45.0, 135.0, -135.0}
    # A **wall** hard against the `-90` rim point demotes that candidate and leaves the order of
    # the others exactly as it was -- which is the whole of how the drawer task stops spending
    # its horizon on rim points two drawer walls are standing in. A *movable* object never
    # demotes anything: its pose has no extent and a box drawn round it is fiction.
    wall = [expert.RIM_RADIUS + expert.FINGER_HALF_SPAN, 0.0, 1.0,
            0.002, 0.2, 0.1, 1, 0, 0, 0, 1, 0, 0, 0, 1]
    blocked = {**scene(), "wooden_cabinet_1": {
        "pos": np.asarray([0.1, 0.0, 0.90], np.float32),
        "quat": np.asarray([0, 0, 0, 1], np.float32), "fixture": True, "boxes": [wall]}}
    turns = expert.candidate_turns((0.0, 0.0, 0.90), blocked, axis, expert.RIM_RADIUS)
    assert turns[-1] == -90.0
    assert turns[:3] == (0.0, 180.0, 90.0)


def test_a_candidate_fits_when_the_outer_finger_has_room_for_itself():
    """The room is measured against **objects and fixtures alike**, and against the fixture's own
    collision boxes rather than its bounding box -- a drawer's hull contains the bowl inside it,
    so a rule measured against it would call every grasp blocked."""
    from robojev import scene as scene_mod

    axis = np.array([0.0, 1.0])
    wall = np.array([0.0, expert.RIM_RADIUS + expert.FINGER_HALF_SPAN + 0.002, 0.95])
    objects = {**scene(bowl=(0.0, 0.0, 0.90)), "wooden_cabinet_1": {
        "pos": np.asarray([0.0, 0.2, 0.90], np.float32),
        "quat": np.asarray([0, 0, 0, 1], np.float32), "fixture": True,
        "boxes": [[*wall, 0.2, 0.002, 0.1, 1, 0, 0, 0, 1, 0, 0, 0, 1]]}}
    room, who = expert.candidate_room((0.0, 0.0, 0.90), objects, axis, 0.0, expert.RIM_RADIUS,
                                      expert.GRASP_DZ)
    assert who == "wooden_cabinet_1" and room < expert.FINGER_HALF_WIDTH
    behind, _ = expert.candidate_room((0.0, 0.0, 0.90), objects, axis, 180.0, expert.RIM_RADIUS,
                                      expert.GRASP_DZ)
    assert behind > expert.FINGER_HALF_WIDTH
    # ...and the fixture is furniture: nothing that renders the scene sees it.
    assert "wooden_cabinet_1" not in scene_mod.movable(objects)


def test_the_yaw_answer_is_a_sign_against_the_tolerance_the_state_prints():
    for error, axiswise in ((0.0, "hold"), (8.0, "hold"), (-8.0, "hold"),
                            (8.1, "+"), (-8.1, "-"), (90.0, "+"), (-45.0, "-")):
        assert expert._yaw_answer(error) == axiswise


def test_the_wrist_tolerance_is_at_least_half_a_step_or_it_would_oscillate():
    """The wrist has one speed, so a tolerance below half a step is a wrist that steps across the
    band and back for ever."""
    assert expert.YAW_TOL_DEG >= 0.5 * expert.YAW_SCALE["large"]


def test_the_wrist_constants_are_the_composer_s():
    compose = import_module("robojev.compose")

    assert expert.YAW_TOL_DEG == compose.DEFAULT_YAW_TOLERANCE_DEG
    assert expert.EXECUTED_DEG_PER_UNIT == compose.MEASURED_DEG_PER_UNIT
    assert expert.YAW_SCALE["large"] == compose.YAW_LARGE_DEG
    # And a rotation is scaled by what the arm **executes**, not by what it is commanded --
    # the same mistake `EXECUTED_COARSE_PER_DELTA_T` exists to stop for a translation.
    assert expert.EXECUTED_DEG_PER_UNIT < 0.3 * compose.COMMANDED_DEG_PER_UNIT


def test_the_two_composers_turn_the_wrist_by_the_same_amount():
    """`compose_axiswise` is what the 100-episode table executes and `v2.compose.compose` is what
    the served policy executes; a wrist that turned at two speeds would make the table a
    measurement of something nobody runs."""
    compose = import_module("robojev.compose")

    answers = {"move_x": "hold", "move_y": "hold", "move_z": "hold", "size_x": "small",
               "size_y": "small", "size_z": "small", "grip": False}
    for yaw in ("-", "hold", "+"):
        mine = expert.compose_axiswise({**answers, "yaw": yaw})
        theirs = compose.compose({**answers, "yaw": yaw, "subgoal": "reach"})
        assert mine[5] == pytest.approx(theirs[5])


def test_the_rim_axis_does_not_chase_a_wrist_the_plan_is_turning():
    """The bug that cost the drawer task its first working candidate: the rim direction is the
    latched axis rotated by the candidate's turn, so an axis re-read from a wrist that is
    *obeying* that turn rotates with it and the rim point runs away from the hand.

    A candidate that needs no rotation keeps the old rule -- re-read while the hand is farther
    than `LATCH_RADIUS` -- which is what leaves the other tasks' behaviour alone.
    """
    objects = scene()
    far = proprio((-0.40, 0.20, 1.10))                       # well outside LATCH_RADIUS
    turned = proprio((-0.40, 0.20, 1.10), axisangle=(2.36, 2.08, 0.0))   # a yawed wrist
    common = {"roles": {"target": BOWL, "destination": PLATE}, "rim_axis": [0.0, 1.0]}
    _, straight, _ = expert.decide(turned, objects, INSTRUCTION,
                                   phase_at("approach", candidate=0, **common))
    assert straight["rim_axis"] != [0.0, 1.0]                # turn 0: the old rule, re-read
    _, held, _ = expert.decide(turned, objects, INSTRUCTION,
                               phase_at("approach", candidate=2, **common))
    assert held["rim_axis"] == [0.0, 1.0]                    # a turning candidate: latched


# ------------------------------------------------------------------------- the step-size tiling


def test_the_tolerances_are_reachable_by_the_smallest_step():
    """A phase tolerance tighter than the smallest step's own hold band is a phase that can never
    be satisfied: the controller answers `hold`, the hand stops, and the transition never fires."""
    smallest = expert.HOLD_BAND * expert.AXISWISE_SCALE["small"] * expert.EXECUTED_COARSE_PER_DELTA_T
    assert smallest < expert.XY_TOL
    assert smallest < expert.Z_TOL


# ------------------------------------------------------------------------- reading the sentence


def test_the_target_is_the_shared_rule_s_target():
    """The expert must never name a different bowl from the harvester and the servers."""
    from robojev.roles import scene_roles

    objects, eef = scene(), (-0.21, -0.02, 1.18)
    _, _, meta = expert.decide(proprio(eef), objects, INSTRUCTION, expert.new_phase())
    shared = scene_roles(objects, INSTRUCTION, eef)
    assert (meta["target"], meta["destination"]) == (shared["target"], shared["destination"])
    assert meta["target"] == BOWL          # the bowl between the plate and the ramekin


def test_an_empty_scene_is_an_expert_error():
    with pytest.raises(expert.ExpertError):
        expert.decide(proprio((0, 0, 1.1)), {}, INSTRUCTION, expert.new_phase())


# ------------------------------------------------------------------------------- the phases


def test_the_approach_flies_to_the_hover_point_above_the_rim():
    objects = scene()
    choices, nxt, meta = expert.decide(proprio((-0.30, -0.02, 1.18)), objects, INSTRUCTION,
                                       expert.new_phase())
    assert meta["phase"] == "approach" and nxt["name"] == "approach"
    assert choices["grip"] == "false"
    # The hover point is a rim radius from the bowl and a hover above the grasp height.
    assert meta["goal"][2] == pytest.approx(0.90 + expert.GRASP_DZ + expert.HOVER, abs=1e-6)
    assert np.hypot(meta["goal"][0] + 0.06, meta["goal"][1] - 0.20) == pytest.approx(
        expert.RIM_RADIUS, abs=1e-3)      # `meta` rounds to 4 decimals
    # ...and it is on the wrist's closing axis (world -y for this orientation), not on the line
    # from the hand: the hand is at -0.30 in x and the waypoint does not move in x at all.
    assert meta["goal"][0] == pytest.approx(-0.06, abs=1e-3)


def test_the_approach_hands_over_to_the_descent_above_the_rim():
    objects = scene()
    rim = (-0.06, 0.20 - expert.RIM_RADIUS)      # the closing axis, on the side the hand is on
    hover = 0.90 + expert.GRASP_DZ + expert.HOVER
    _, nxt, _ = expert.decide(proprio((rim[0], rim[1], hover)), objects, INSTRUCTION,
                              phase_at("approach", rim_axis=[0.0, -1.0]))
    assert nxt["name"] == "descend"


def test_the_descent_closes_only_at_the_grasp_height():
    objects = scene()
    rim = (-0.06, 0.20 - expert.RIM_RADIUS)
    high = proprio((rim[0], rim[1], 0.90 + expert.GRASP_DZ + 0.05))
    at = proprio((rim[0], rim[1], 0.90 + expert.GRASP_DZ))

    choices, still, _ = expert.decide(high, objects, INSTRUCTION,
                                      phase_at("descend", rim_axis=[0.0, -1.0]))
    assert still["name"] == "descend"          # still on its way down
    assert choices["move_z"] == "-" and choices["grip"] == "false"

    _, nxt, _ = expert.decide(at, objects, INSTRUCTION, phase_at("descend", rim_axis=[0.0, -1.0]))
    assert nxt["name"] == "close"


def test_the_close_holds_still_and_waits_for_the_fingers():
    """`grip` true is a command, not a grasp: the expert spends `SETTLE_DECISIONS` decisions
    shut and motionless before it trusts the fingers with the bowl's weight."""
    objects = scene()
    at = proprio((-0.01, 0.20, 0.90 + expert.GRASP_DZ), width=0.08)
    for tick in range(expert.SETTLE_DECISIONS):
        choices, nxt, meta = expert.decide(at, objects, INSTRUCTION,
                                           phase_at("close", ticks=tick, rim_axis=[1.0, 0.0]))
        assert choices["grip"] == "true"
        assert choices["move_z"] == "hold"
        expected = "lift" if tick + 1 >= expert.SETTLE_DECISIONS else "close"
        assert nxt["name"] == expected, f"tick {tick}"


def test_the_close_records_what_the_grasp_is_measured_from():
    """The close hands over the two numbers every later question about the grasp is answered
    from: where the hand was when the fingers settled (what the lift climbs from) and what the
    hand-to-bowl offset was at that moment (what `held` compares against). Recorded as the stage
    *leaves*, i.e. once the fingers have settled on whatever they caught -- not on the way in,
    when they are still shutting and the bowl may still move."""
    objects = scene()
    hand = proprio((-0.01, 0.20, 0.925))
    _, waiting, _ = expert.decide(hand, objects, INSTRUCTION,
                                  phase_at("close", ticks=0, rim_axis=[1.0, 0.0]))
    assert waiting["name"] == "close"
    assert waiting["close_eef_z"] is None and waiting["grasp_offset"] is None

    _, nxt, _ = expert.decide(hand, objects, INSTRUCTION,
                              phase_at("close", ticks=expert.SETTLE_DECISIONS - 1,
                                       rim_axis=[1.0, 0.0]))
    assert nxt["name"] == "lift"
    assert nxt["close_eef_z"] == pytest.approx(0.925)
    assert nxt["grasp_offset"] == pytest.approx([-0.05, 0.0, -0.025], abs=1e-6)


def test_the_lift_goal_does_not_chase_the_rising_bowl():
    """The regression that cost 26 of 44 decisions and put the bowl at the top of the workspace:
    a lift height measured from the *target's* pose recedes as fast as the hand climbs, because a
    grasped bowl rises with it.

    It is measured from two fixed things instead -- where the hand was when it closed, and the
    height the carry needs -- so it is the **same number** at every rise, which is what this
    asserts. (The carry height wins here: the lift is also the extraction, and a hand that
    translates before it has cleared what the target was standing in drags it through the wall.)
    """
    goals = []
    for rise in (0.0, 0.05, 0.10, 0.20):
        objects = scene(bowl=(-0.01, 0.20, 0.90 + rise))
        _, _, meta = expert.decide(proprio((-0.01, 0.20, 0.925 + rise)), objects, INSTRUCTION,
                                   phase_at("lift", close_eef_z=0.925, rim_axis=[1.0, 0.0],
                                            roles={"target": BOWL, "destination": PLATE}))
        goals.append(meta["goal"][2])
    carry_height = 0.90 + expert.CARRY_CLEARANCE + 0.025      # the plate, plus the grasp offset
    assert goals == pytest.approx([max(0.925 + expert.LIFT, carry_height)] * 4)


def test_the_lift_hands_over_once_the_bowl_has_come_with_it():
    """The bowl rose with the hand, so the offset between the two is what it was at the close:
    `skill.held`, the same test the carry and the tracker's printed `holding` use."""
    rise = expert.CARRY_CLEARANCE
    objects = scene(bowl=(-0.01, 0.20, 0.90 + rise))
    choices, nxt, meta = expert.decide(
        proprio((-0.01, 0.20, 0.925 + rise), width=0.02), objects, INSTRUCTION,
        phase_at("lift", grasp_offset=[0.0, 0.0, -0.025], close_eef_z=0.925,
                 rim_axis=[1.0, 0.0], target_z0=0.90))
    assert meta["held"] is True
    assert nxt["name"] == "carry" and choices["grip"] == "true"


def test_the_carry_clears_the_higher_of_where_the_target_came_from_and_where_it_goes():
    """**The extraction rule.** A target that was standing inside something has to come out of it
    before it can go anywhere, and how far up that is, is a fact about where it *came from*.

    Measured against the demonstrations (note `2026-09-21-stage9-drawer-task.md` §1): on the
    drawer task they raise the bowl to 1.211 m from a resting pose of 1.063, and on the cabinet
    task to 1.251 from 1.128 -- while the destination-only rule asked for 1.110 m on both, which
    is 9 cm below the drawer's own front panel. On a flat scene the destination is the higher of
    the two and this changes nothing.
    """
    heights = {}
    for name, (bowl_z, plate_z) in {"flat": (0.90, 0.90), "drawer": (1.063, 0.90),
                                    "cabinet": (1.128, 0.90)}.items():
        # The destination is a whole `CARRY_RAMP_M` away, so this is the carry's own height and
        # not a point on its way down (see the test below).
        objects = scene(bowl=(-0.01, 0.20, bowl_z), plate=(0.05, 0.21 + expert.CARRY_RAMP_M,
                                                           plate_z))
        rise = {"flat": 0.0, "drawer": 0.065, "cabinet": 0.002}[name]
        _, _, meta = expert.decide(
            proprio((-0.01, 0.20, bowl_z + 0.025), width=0.02), objects, INSTRUCTION,
            phase_at("carry", grasp_offset=[0.0, 0.0, -0.025], close_eef_z=bowl_z + 0.025,
                     rim_axis=[1.0, 0.0], target_z0=bowl_z, support_rise=rise,
                     roles={"target": BOWL, "destination": PLATE}))
        heights[name] = meta["goal"][2] - 0.025          # the *bowl's* carried height
    # Flat: the destination's own clearance, as before. In a drawer: over the panel that stands
    # 6.5 cm above the bowl. On a cabinet top: barely anything stands above it, so the lift is
    # the 8 cm margin and not the 14 cm a blanket clearance used to ask for.
    assert heights["flat"] == pytest.approx(0.90 + expert.CARRY_CLEARANCE)
    assert heights["drawer"] == pytest.approx(1.063 + 0.065 + expert.EXTRACT_MARGIN)
    assert heights["cabinet"] == pytest.approx(1.128 + 0.002 + expert.EXTRACT_MARGIN)


def test_the_carry_comes_back_down_to_the_destinations_own_clearance():
    """The extraction height is a fact about where the target came from, and it stops mattering
    once the hand has left there. Holding it all the way across buys nothing and costs the
    decisions the descent onto the plate then needs -- and the demonstrations do not do it either
    (the drawer task's bowl peaks at 1.199 m and is at 1.149 half way to the plate)."""
    heights = []
    for gap in (0.30, 0.20, 0.10, 0.05, 0.0):
        objects = scene(bowl=(-0.01, 0.20, 1.063), plate=(-0.01, 0.20 + gap, 0.90))
        _, _, meta = expert.decide(
            proprio((-0.01, 0.20, 1.088), width=0.02), objects, INSTRUCTION,
            phase_at("carry", grasp_offset=[0.0, 0.0, -0.025], close_eef_z=1.088,
                     rim_axis=[1.0, 0.0], target_z0=1.063, support_rise=0.065,
                     roles={"target": BOWL, "destination": PLATE}))
        heights.append(meta["goal"][2] - 0.025)
    assert heights[0] == pytest.approx(1.063 + 0.065 + expert.EXTRACT_MARGIN)  # over the panel
    assert heights[-1] == pytest.approx(0.90 + expert.CARRY_CLEARANCE)     # over the plate
    assert heights == sorted(heights, reverse=True)                        # and monotone between


def test_a_carry_with_no_record_of_where_the_target_stood_clears_the_destination_alone():
    """The fallback, and it is the old rule: a carry built straight into a later stage (every
    synthetic case here, and any caller that resumes one) has no `target_z0`, and a clearance
    read off the target's *current* pose would chase a bowl that rises with the hand."""
    objects = scene(bowl=(-0.01, 0.20, 1.20))
    _, _, meta = expert.decide(proprio((-0.01, 0.20, 1.225), width=0.02), objects, INSTRUCTION,
                               phase_at("carry", grasp_offset=[0.0, 0.0, -0.025],
                                        close_eef_z=1.225, rim_axis=[1.0, 0.0],
                                        roles={"target": BOWL, "destination": PLATE}))
    assert meta["goal"][2] == pytest.approx(0.90 + expert.CARRY_CLEARANCE + 0.025)


def test_a_grasp_that_lifted_nothing_reopens_and_retries():
    """The hand has climbed 12 cm and the bowl has not: whatever is between the fingers, it is
    not the target. The outcome is checked at **every** decision of the lift, so this is noticed
    on the first one rather than after a patience nobody could justify per stage."""
    objects = scene(bowl=(-0.01, 0.20, 0.90))            # the bowl never moved
    phase = phase_at("lift", ticks=0, grasp_offset=[0.0, 0.0, -0.025], close_eef_z=0.925,
                     rim_axis=[1.0, 0.0])
    _, nxt, meta = expert.decide(proprio((-0.01, 0.20, 1.05), width=0.02), objects, INSTRUCTION,
                                 phase)
    assert meta["held"] is False and meta["failed"] is True
    assert nxt["name"] == "approach"
    assert nxt["attempts"] == 1                          # a grasp was tried and found empty
    assert nxt["candidate"] == 1                         # ...so the next candidate is in force
    assert nxt["grasp_offset"] is None and nxt["close_eef_z"] is None
    # The fingers open on the decision that is actually an approach, which is the next one.
    choices, _, _ = expert.decide(proprio((-0.01, 0.20, 1.05), width=0.02), objects, INSTRUCTION,
                                  nxt)
    assert choices["grip"] == "false"


def test_each_candidate_stands_somewhere_the_last_one_did_not():
    """One ordered list of alternatives -- a way round the rim and a height above the bowl --
    where there used to be two (`rim_flips` for the side, `RETRY_DZ` for the height), each with
    its own counter and neither able to change the other."""
    objects = scene()
    poses = []
    order = None
    for candidate in range(len(expert.GRASP_CANDIDATES) + 1):
        _, nxt, meta = expert.decide(proprio((-0.30, -0.02, 1.18)), objects, INSTRUCTION,
                                     phase_at("approach", candidate=candidate, turns=order))
        order = nxt["turns"]
        poses.append((meta["grasp_turn"], meta["grasp_dz"], tuple(meta["grasp_point"][0:2])))
    assert [(turn, dz) for turn, dz, _ in poses[:-1]] == pytest.approx(
        [(grasp.turn, round(expert.GRASP_DZ + grasp.dz, 4))
         for grasp in expert.candidates(turns=tuple(order))])
    assert poses[-1] == poses[-2]              # the last candidate is held, not indexed past
    assert len({pose for pose in poses}) == len(expert.GRASP_CANDIDATES)
    # Every one of them is **on the rim**, a radius from the bowl: what a candidate changes is
    # which way round it stands (and, with it, how far the wrist must turn to close radially
    # there), never how far out.
    bowl = np.asarray(scene()[BOWL]["pos"][0:2], float)
    for _, _, point in poses:
        assert np.hypot(*(np.asarray(point) - bowl)) == pytest.approx(expert.RIM_RADIUS, abs=1e-3)
    # ...and the opposite end of the axis the hand holds is among them, at the index
    # `TURN_GROUPS` puts it: the stalled-descent note's own recovery is still in the list.
    opposite = tuple(2 * bowl - np.asarray(poses[0][2]))
    at = [i for i, (_, _, point) in enumerate(poses)
          if np.allclose(point, opposite, atol=1e-3)]
    assert at and order[at[0]] == 180.0


def test_a_bowl_dropped_mid_carry_is_picked_up_again():
    """Dropped means *stopped moving with the hand*, not "lower than it was": the fixture tasks
    carry downhill and a height test calls their whole carry a drop."""
    objects = scene(bowl=(0.0, 0.20, 0.90))              # back on the table, 15 cm below the hand
    _, nxt, meta = expert.decide(proprio((0.0, 0.20, 1.05), width=0.02), objects, INSTRUCTION,
                                 phase_at("carry", close_eef_z=0.925,
                                          grasp_offset=[0.0, 0.0, -0.03], rim_axis=[1.0, 0.0]))
    assert meta["held"] is False
    assert nxt["name"] == "approach" and nxt["attempts"] == 1 and nxt["candidate"] == 1


def test_the_place_waypoint_aims_the_bowl_and_not_the_hand():
    """A rim grasp holds the bowl a rim radius to one side; aiming the hand at the plate's centre
    puts the bowl over its edge and LIBERO's `on(bowl, plate)` never fires."""
    objects = scene(bowl=(-0.01, 0.25, 1.00), other=(-0.60, 0.60, 0.90), plate=(0.05, 0.21, 0.90))
    eef = (-0.06, 0.25, 1.03)                            # holding it 5 cm to the -x side
    _, _, meta = expert.decide(proprio(eef, width=0.02), objects, INSTRUCTION,
                               phase_at("carry", close_eef_z=0.925,
                                        grasp_offset=[0.05, 0.0, -0.03], rim_axis=[1.0, 0.0]))
    assert meta["place_point"] == pytest.approx([0.05 - 0.05, 0.21 - 0.0], abs=1e-6)
    assert meta["goal"][0:2] == pytest.approx(meta["place_point"], abs=1e-6)


def test_the_descent_over_the_plate_is_measured_by_the_bowl_s_own_height():
    objects = scene(bowl=(0.05, 0.21, 1.05), other=(-0.60, 0.60, 0.90), plate=(0.05, 0.21, 0.90))
    _, _, meta = expert.decide(proprio((0.05, 0.21, 1.08), width=0.02), objects, INSTRUCTION,
                               phase_at("lower", close_eef_z=0.925,
                                        grasp_offset=[0.0, 0.0, -0.03], rim_axis=[1.0, 0.0]))
    # The bowl is 1.05 and has to reach 0.90 + RELEASE_DZ, so the hand drops by the difference.
    assert meta["goal"][2] == pytest.approx(1.08 - (1.05 - 0.90 - expert.RELEASE_DZ))


def test_the_release_opens_and_waits_before_retreating():
    objects = scene(bowl=(0.05, 0.21, 0.935), plate=(0.05, 0.21, 0.90))
    for tick in range(expert.RELEASE_DECISIONS):
        choices, nxt, _ = expert.decide(proprio((0.05, 0.21, 0.96), width=0.02), objects,
                                        INSTRUCTION, phase_at("release", ticks=tick, close_eef_z=0.925))
        assert choices["grip"] == "false" and choices["move_z"] == "hold"
        assert nxt["name"] == ("retreat" if tick + 1 >= expert.RELEASE_DECISIONS else "release")


def test_the_retreat_rises_and_then_stops():
    objects = scene()
    choices, nxt, _ = expert.decide(proprio((0.05, 0.21, 0.96)), objects, INSTRUCTION,
                                    phase_at("retreat", ticks=expert.RETREAT_DECISIONS - 1))
    assert choices["move_z"] == "+" and choices["grip"] == "false"
    assert nxt["name"] == "done"
    choices, nxt, _ = expert.decide(proprio((0.05, 0.21, 1.10)), objects, INSTRUCTION,
                                    phase_at("done"))
    assert all(choices[f"move_{axis}"] == "hold" for axis in "xyz")
    assert choices["yaw"] == "hold" and choices["grip"] == "false"
    assert nxt["name"] == "done"


def test_a_stage_that_stops_getting_closer_takes_the_next_candidate():
    """The one progress rule, on the stage it was measured on. A hand that cannot reach this rim
    point answers `move_z -` at it for as long as it is allowed to; what is observable is that
    the distance to the waypoint has stopped falling, and the answer is to stand somewhere else.
    There is no clock: a descent that is still descending is never interrupted."""
    objects = scene()
    rim = (-0.06, 0.20 - expert.RIM_RADIUS)
    stuck = proprio((rim[0], rim[1], 0.90 + expert.GRASP_DZ + 0.06))
    phase = phase_at("descend", rim_axis=[0.0, -1.0])
    names = []
    for _ in range(skill.PROGRESS_PATIENCE + 1):
        _, phase, meta = expert.decide(stuck, objects, INSTRUCTION, phase)
        names.append(phase["name"])
    assert names == ["descend"] * skill.PROGRESS_PATIENCE + ["approach"]
    assert meta["blocked"] and phase["candidate"] == 1


def test_a_descent_that_is_descending_is_never_interrupted():
    objects = scene()
    rim = (-0.06, 0.20 - expert.RIM_RADIUS)
    phase = phase_at("descend", rim_axis=[0.0, -1.0])
    for height in (0.09, 0.07, 0.05, 0.04, 0.03, 0.02):
        _, phase, meta = expert.decide(proprio((rim[0], rim[1], 0.90 + expert.GRASP_DZ + height)),
                                       objects, INSTRUCTION, phase)
        assert not meta["blocked"] and phase["candidate"] == 0


def test_the_best_pose_reached_is_accepted_only_once_the_candidates_are_spent():
    """The one "proceed anyway" left, and it is reached after every alternative was tried rather
    than after twelve decisions had passed -- which is what used to close the fingers 2 cm above
    the rim and lose the bowl on the lift."""
    objects = scene()
    rim = (-0.06, 0.20 - expert.RIM_RADIUS)
    stuck = proprio((rim[0], rim[1], 0.90 + expert.GRASP_DZ + 0.06))
    phase = phase_at("descend", rim_axis=[0.0, -1.0],
                     candidate=len(expert.GRASP_CANDIDATES) - 1,
                     tried=list(range(len(expert.GRASP_CANDIDATES) - 1)),
                     stuck=skill.PROGRESS_PATIENCE - 1, best=0.0)
    _, nxt, meta = expert.decide(stuck, objects, INSTRUCTION, phase)
    assert meta["blocked"] and nxt["name"] == "close"


def test_every_phase_answers_the_whole_question_set():
    """A label source that omits a question is a training row that cannot be built."""
    objects = scene()
    for name in expert.PHASES:
        choices, _, _ = expert.decide(proprio((-0.01, 0.20, 1.00)), objects, INSTRUCTION,
                                      phase_at(name, close_eef_z=0.925))
        assert set(choices) == set(expert.AXISWISE_CANDIDATES)
        for qid, candidates in expert.AXISWISE_CANDIDATES.items():
            value = choices[qid]
            assert value in candidates, qid


def test_the_gripper_is_shut_exactly_while_it_is_carrying():
    objects = scene(bowl=(-0.01, 0.20, 0.95))
    shut = {"close", "lift", "carry", "lower"}
    for name in expert.PHASES:
        choices, _, _ = expert.decide(proprio((-0.01, 0.20, 0.98)), objects, INSTRUCTION,
                                      phase_at(name, close_eef_z=0.925,
                                               grasp_offset=[0.0, 0.0, -0.03]))
        assert choices["grip"] == ("true" if name in shut else "false"), name


# ------------------------------------------------------------------------------ determinism


def test_the_same_state_gives_the_same_answer():
    objects, state = scene(), proprio((-0.15, 0.10, 1.05))
    first = expert.decide(state, objects, INSTRUCTION, phase_at("approach"))
    for _ in range(5):
        again = expert.decide(state, objects, INSTRUCTION, phase_at("approach"))
        assert again[0] == first[0] and again[1] == first[1] and again[2] == first[2]


def test_decide_does_not_mutate_what_it_is_given():
    objects, state = scene(), proprio((-0.15, 0.10, 1.05))
    phase = phase_at("approach")
    before_phase, before_objects = dict(phase), {k: v["pos"].copy() for k, v in objects.items()}
    expert.decide(state, objects, INSTRUCTION, phase)
    assert phase == before_phase
    for name, pos in before_objects.items():
        assert objects[name]["pos"] == pytest.approx(pos)


def test_a_missing_phase_starts_a_fresh_episode():
    objects = scene()
    a = expert.decide(proprio((-0.15, 0.10, 1.05)), objects, INSTRUCTION, None)
    b = expert.decide(proprio((-0.15, 0.10, 1.05)), objects, INSTRUCTION, expert.new_phase())
    assert a[0] == b[0] and a[1] == b[1]


# ----------------------------------------------------------------------------- the real thing


@pytest.mark.sim
def test_one_real_episode_is_picked_and_placed():
    """The whole contract, once, against the simulator: task 0's first init state inside the
    suite's own horizon, with LIBERO's own success predicate ending the episode.

    The synthetic tests above are only worth anything if the states they build are the states
    `LiberoEnv` produces, and this is what says so. ~40 s on a CPU, most of it the env build.
    """
    from robojev.envs.libero import LiberoEnv

    env = LiberoEnv("libero_spatial", 0, render_size=128)
    try:
        result = expert.run_episode(env, 0, max_steps=220)
    finally:
        env.close()
    assert result["success"], result
    assert result["decisions"] < 44
    assert result["steps"] < 220


# ------------------------------------------------------------------------- the v2 vocabulary
#
# Plan 9c task 3: the expert is the authoritative labeller in RoboJEV v2's question set. These
# tests are about the *shape* of the answer and its agreement with the two other things that
# produce or consume it -- `v2.compose.labels_from_waypoint` (the single label function) and
# `v2.parse` (gate G3's text-only parser). The controller's own behaviour is tested above and is
# not re-tested here, because `answers_v2` does not have any: it is `decide`'s.


def v2questions():
    return import_module("robojev.questions")


def v2compose():
    """`import_module` rather than `from robojev import compose`: the package's
    `__init__` re-exports a *function* called `compose`, so the plain form hands back the
    function and every attribute lookup on it raises."""
    return import_module("robojev.compose")


def test_the_subgoal_map_is_total_over_the_phase_machine():
    """A label source that has no `subgoal` for a phase it can be in is a training row that
    cannot be built -- and the phase it would be missing is whichever one nobody thought of."""
    questions = v2questions()

    assert set(expert.V2_PHASE_SUBGOAL) == set(expert.PHASES)
    assert set(expert.V2_PHASE_SUBGOAL.values()) <= set(questions.SUBGOAL_CANDIDATES)


def test_the_subgoal_map_is_the_composer_s():
    """It is a literal here because `v2.compose` imports *this* module and importing it back
    would be a cycle. Two copies of one mapping is two chances for `descend` to become `grasp`
    on one side only, so they are pinned to each other."""
    assert expert.V2_PHASE_SUBGOAL == v2compose().PHASE_FROM_EXPERT


def test_descend_is_reach_and_close_is_grasp():
    """The one line of the map worth asserting on its own. `grasp` is the candidate whose text
    says "this is where they close"; the expert's `close` is the decision that shuts the fingers
    and `descend` is still on the way down. A `descend` labelled `grasp` teaches the model to
    close a decision early, which is `GRASP_DZ`'s own measured 4.5 cm miss."""
    assert expert.V2_PHASE_SUBGOAL["descend"] == "reach"
    assert expert.V2_PHASE_SUBGOAL["close"] == "grasp"


def _random_privileged(rng):
    """A scene and a hand somewhere plausible over LIBERO-Spatial's table."""
    return (scene(bowl=rng.uniform([-0.25, 0.0, 0.90], [0.25, 0.40, 1.00]),
                  other=rng.uniform([-0.25, 0.0, 0.90], [0.25, 0.40, 1.00]),
                  plate=rng.uniform([-0.25, 0.0, 0.90], [0.25, 0.40, 1.00]),
                  ramekin=rng.uniform([-0.25, 0.0, 0.90], [0.25, 0.40, 1.00])),
            proprio(rng.uniform([-0.35, -0.15, 0.92], [0.35, 0.50, 1.30]),
                    width=float(rng.choice([0.08, 0.02]))))


def test_answers_v2_is_total_over_two_hundred_random_states():
    """Pure and total: no exception, every qid present, every value a declared candidate.

    Two hundred draws over every phase of the machine, because an expert that can raise in the
    middle of a relabelling is not a label source -- the run dies halfway and the rows already
    written are a dataset nobody can reproduce.
    """
    questions = v2questions()
    rng = np.random.default_rng(20260921)

    for index in range(200):
        objects, state = _random_privileged(rng)
        name = expert.PHASES[int(rng.integers(len(expert.PHASES)))]
        answers, nxt, meta = expert.answers_v2(
            state, objects, INSTRUCTION,
            phase_at(name, close_eef_z=0.925, grasp_offset=[0.0, 0.0, -0.03],
                     candidate=int(rng.integers(len(expert.GRASP_CANDIDATES)))))

        assert set(answers) == set(expert.V2_QIDS), index
        for qid in expert.V2_MOVE_QIDS:
            assert answers[qid] in questions.AXIS_CANDIDATES, (index, qid, answers[qid])
        for qid in expert.V2_SIZE_QIDS:
            assert answers[qid] in questions.SIZE_CANDIDATES, (index, qid, answers[qid])
        assert answers["yaw"] in questions.YAW_CANDIDATES
        assert isinstance(answers["grip"], bool)
        assert answers["subgoal"] in questions.SUBGOAL_CANDIDATES
        assert nxt["name"] in expert.PHASES
        assert meta["questions_version"] == "v2"


def test_the_v2_answers_cover_the_active_question_set():
    """`yaw` is answered here and not asked on the wire (`ACTIVE_QIDS`), which is the right way
    round: a suite that switches the wrist on has labels for it without a re-harvest. What must
    never happen is the wire asking something the labeller has no answer for."""
    assert set(v2questions().ACTIVE_QIDS) <= set(expert.V2_QIDS)


def test_the_v2_answers_are_the_v2_label_function():
    """The expert and gate G3's parser are the same function of the same numbers.

    `answers_v2` reads its bands in metres and `labels_from_waypoint` reads them in centimetres
    -- deliberately, because `v2.compose` imports this module and calling it from here would be
    a cycle -- so this is what says the two tilings are one tiling. Offsets that land on a band
    edge to the last bit are skipped: there the two differ by one float ulp, which is one size
    on one axis for one decision, and the labels of record come from `gold_for_state` (the label
    function on the rounded numbers the state prints) rather than from here.
    """
    compose = v2compose()
    edges = (compose.DEFAULT_TOLERANCE_CM,
             compose.FINE_BAND * compose.STEP_MEDIUM_CM,
             compose.FINE_BAND * compose.STEP_LARGE_CM)
    rng = np.random.default_rng(7)
    compared = 0

    for _ in range(1000):
        offset_cm = np.round(rng.uniform(-14.0, 14.0, 3), 1)
        if any(abs(abs(v) - edge) < 1e-9 for v in offset_cm for edge in edges):
            continue
        mine = expert.axiswise_answers(offset_cm / 100.0)
        theirs = compose.labels_from_waypoint(
            offset_cm, 0.0, False, "reach",
            qids=(*expert.V2_MOVE_QIDS, *expert.V2_SIZE_QIDS))
        for qid, value in theirs.items():
            assert mine[qid] == value, (offset_cm, qid, mine[qid], value)
        compared += 1
    assert compared > 900, compared


def test_the_v2_answers_are_the_axiswise_controller_s():
    """Ruling (a): v2's motor vocabulary *is* the expert's measured axiswise set, so `answers_v2`
    must be `decide(..., )` and not a second controller to measure. The only
    differences allowed are `grip`'s spelling and the addition of `subgoal`."""
    objects, state = scene(), proprio((-0.15, 0.10, 1.05))
    phase = phase_at("descend", rim_axis=[0.0, -1.0])

    choices, _, meta = expert.decide(state, objects, INSTRUCTION, phase, )
    answers, _, _ = expert.answers_v2(state, objects, INSTRUCTION, phase)

    for qid in (*expert.V2_MOVE_QIDS, *expert.V2_SIZE_QIDS, "yaw"):
        assert answers[qid] == choices[qid], qid
    assert answers["grip"] is (choices["grip"] == "true")
    assert answers["subgoal"] == expert.V2_PHASE_SUBGOAL[meta["phase"]]


def test_the_v2_grip_is_the_phase_machine_s_latch_and_not_a_geometric_one():
    """`grip` is the answer the closed loop cannot absorb an error in (note §5: 1/20 at p = 0.10
    corrupted against 13/20 exempt), and the expert's is its phase machine's -- shut from the
    close to the release, whatever the geometry says about arrival."""
    objects = scene(bowl=(-0.01, 0.20, 0.95))
    shut = {"close", "lift", "carry", "lower"}
    for name in expert.PHASES:
        answers, _, _ = expert.answers_v2(
            proprio((-0.01, 0.20, 0.98)), objects, INSTRUCTION,
            phase_at(name, close_eef_z=0.925, grasp_offset=[0.0, 0.0, -0.03]))
        assert answers["grip"] is (name in shut), name


def test_answers_v2_does_not_mutate_what_it_is_given():
    objects, state = scene(), proprio((-0.15, 0.10, 1.05))
    phase = phase_at("approach")
    before = dict(phase)
    expert.answers_v2(state, objects, INSTRUCTION, phase)
    assert phase == before


def test_the_v2_meta_carries_what_a_console_needs_to_show_the_decision():
    """The offsets in the unit the state prints them in and the per-axis sizes beside them --
    plan 9c's requirement on the served `decisions.meta`. A console converting metres to
    centimetres itself is a second place for the scale error of note §2 to live."""
    answers, _, meta = expert.answers_v2(proprio((-0.15, 0.10, 1.05)), scene(), INSTRUCTION)

    assert meta["questions_version"] == "v2"
    assert meta["subgoal"] == answers["subgoal"]
    assert meta["offset_cm"] == [round(v * 100.0, 1) for v in meta["error"]]
    assert meta["sizes"] == {axis: answers[f"size_{axis}"] for axis in "xyz"}
    assert meta["delta_t"] == expert.EXPERT_DELTA_T == 1.0
    assert meta["cm_per_unit"] == pytest.approx(5.0)


# ------------------------------------------------------------ the label of record (gate G3)


def _tracker(objects=None, target=BOWL, destination=PLATE, **kwargs):
    state = import_module("robojev.state")
    return state.TrackerV2(horizon=44, target=target, destination=destination, **kwargs)


def test_gold_for_state_is_what_the_parser_reads_out_of_the_state_string():
    """Gate G3's contract, over 500 fixture states: the expert's label and a rule that has seen
    nothing but the text are the same dict.

    This is spec §2's principle 1 -- *if a rule cannot recover the label from the state string,
    the model cannot either* -- and it is the one v1 failed (F2: the harvest labelled from an
    executed move, the expert decided from geometry, and the two agreed 82.7 % of the time).
    `gold_for_state` cannot disagree with the parser by construction, because it reads the
    waypoint off the tracker that rendered the text; what this test proves is that the
    construction is actually wired that way.
    """
    state_module = import_module("robojev.state")
    parse = import_module("robojev.parse")
    rng = np.random.default_rng(90210)
    checked = 0

    for _ in range(500):
        objects = scene()
        name = expert.PHASES[int(rng.integers(len(expert.PHASES)))]
        hand = proprio(rng.uniform([-0.30, -0.10, 0.93], [0.30, 0.45, 1.25]),
                       width=float(rng.choice([0.08, 0.02])))
        tracker = _tracker()
        tracker.observe(step=0, proprio=hand, objects=objects)
        text = state_module.serialise_v2(hand, objects, INSTRUCTION, tracker)

        gold, _, meta = expert.gold_for_state(
            tracker, hand, objects, INSTRUCTION,
            phase_at(name, close_eef_z=0.925, grasp_offset=[0.0, 0.0, -0.03]))

        assert gold == parse.parse(text), (name, text)
        assert meta["gold_source"] == "tracker"
        checked += 1
    assert checked == 500


def test_gold_for_state_reads_the_tracker_s_waypoint_and_not_its_own():
    """The whole difference from `answers_v2`, and the reason the gold is a function of the text:
    move the tracker's waypoint (a wider rim) and the gold moves with it."""
    objects, hand = scene(), proprio((-0.01, 0.35, 1.00))
    near = _tracker()
    near.observe(step=0, proprio=hand, objects=objects)
    gold_near, _, _ = expert.gold_for_state(near, hand, objects, INSTRUCTION, expert.new_phase())

    assert gold_near == near.gold_answers()


def test_gold_for_state_raises_when_the_tracker_names_a_different_bowl():
    """Naming a different object is the sharpest disagreement there is: the state would say one
    bowl and the label would be about another, and a harvest would write thousands of rows
    before a gate noticed. It fails here instead."""
    objects, hand = scene(), proprio((-0.01, 0.30, 1.05))
    tracker = _tracker(target=OTHER)
    tracker.observe(step=0, proprio=hand, objects=objects)

    with pytest.raises(expert.ExpertError, match="target"):
        expert.gold_for_state(tracker, hand, objects, INSTRUCTION, expert.new_phase())


def test_gold_for_state_raises_when_the_two_rim_geometries_disagree():
    """A tracker grasping at a different radius from the expert's is the quiet version of the
    same bug: same object, different waypoint, and a parser that scores 97 % for a reason nobody
    can find. `GOLD_WAYPOINT_TOL` is the expert's own arrival tolerance -- a drift smaller than
    it cannot change an answer, and a drift larger than it can."""
    objects, hand = scene(), proprio((-0.01, 0.30, 1.05))
    tracker = _tracker(rim_radius_m=expert.RIM_RADIUS + 0.05)
    tracker.observe(step=0, proprio=hand, objects=objects)

    with pytest.raises(expert.ExpertError, match="differ by"):
        expert.gold_for_state(tracker, hand, objects, INSTRUCTION, expert.new_phase())


def test_gold_for_state_needs_a_tracker_that_has_observed_something():
    objects, hand = scene(), proprio((-0.01, 0.30, 1.05))

    with pytest.raises(expert.ExpertError, match="no waypoint"):
        expert.gold_for_state(_tracker(), hand, objects, INSTRUCTION, expert.new_phase())


# ------------------------------------------------- the episode's first candidate is the model's


def _first_selection(letter):
    """`(candidate, turn, rim_followed, chose_applied)` for an episode whose first answer was
    `letter` -- decision 0 rendered with the block, decision 1 taken with the answer."""
    objects, hand = scene(), proprio((-0.30, -0.02, 1.18))
    qids = expert.V2_QIDS
    _choices, carry, meta = expert.decide(hand, objects, INSTRUCTION,
                                          expert.new_phase({"target": BOWL, "destination": PLATE}),
                                          qids=qids)
    assert meta["candidate"] == 0            # decision 0 has no answer yet: the plan's own order
    _choices, nxt, meta = expert.decide(hand, objects, INSTRUCTION, {**carry, "chose": letter},
                                        qids=qids)
    return meta["candidate"], meta["grasp_turn"], meta["rim_followed"], nxt["chose_applied"]


def test_the_first_candidate_of_an_episode_is_the_one_the_model_asked_for():
    """**The point of asking.** The executor chooses a candidate when something goes wrong, and at
    the episode's first decision nothing has -- so the plan used to take the first alternative in
    its own order and the model's first `rim` answer selected nothing at all. The selection is
    deferred one decision instead: decision 0 renders the whole block, the letter that comes back
    selects the candidate for decision 1.

    A **wrong** answer is obeyed, which is what makes the question worth its tokens.
    """
    candidate, turn, followed, applied = _first_selection("C")
    assert (candidate, turn) == (2, -90.0) and followed and applied


def test_the_gold_first_answer_changes_nothing():
    """`rim_gold` is the first listed candidate that fits and `candidate_turns` has already put a
    fitting direction first, so on a scene where the plan is right the deferred selection is a
    no-op -- which is what keeps the expert's own harvest byte-identical to the one before it."""
    objects, hand = scene(), proprio((-0.30, -0.02, 1.18))
    choices, _, _ = expert.decide(hand, objects, INSTRUCTION,
                                  expert.new_phase({"target": BOWL, "destination": PLATE}),
                                  qids=expert.V2_QIDS)
    assert _first_selection(choices["rim"])[0:2] == (0, 0.0)


def test_a_letter_this_run_was_not_offered_falls_back_to_the_plan_s_order():
    candidate, turn, followed, applied = _first_selection("Z")
    assert (candidate, turn) == (0, 0.0) and not followed and applied


def test_the_first_selection_happens_once_and_the_hand_does_not_walk_round_the_bowl():
    """A model that changes its mind at decision 4 is not a reason to abandon a rim point the
    hand is already standing on: `chose_applied` latches, and every later selection goes through
    the executor's own -- a blocked stage or a grasp that held nothing."""
    objects, hand = scene(), proprio((-0.30, -0.02, 1.18))
    qids = expert.V2_QIDS
    carry = expert.new_phase({"target": BOWL, "destination": PLATE})
    for letter in (None, "C", "D", "B"):
        _choices, carry, meta = expert.decide(hand, objects, INSTRUCTION,
                                              {**carry, "chose": letter} if letter else carry,
                                              qids=qids)
    assert (meta["candidate"], meta["grasp_turn"]) == (2, -90.0)     # still the first answer's
