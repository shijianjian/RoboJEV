"""`robojev.roles` -- which object the task means, and what stage the episode is in.

Two rules that every side of the fine-tune reads: the expert steers by them, both recipe servers
name their target with them, and the harvest labels against them. A disagreement between two
readers of the same sentence is a row labelled about one bowl and executed against another, so
they are pinned here on hand-built scenes rather than left to an episode to discover.
"""
from __future__ import annotations

import numpy as np
import pytest

from robojev import roles as R

TARGET = "akita_black_bowl_1"
DESTINATION = "plate_1"

#: A five-object LIBERO-Spatial scene at the poses the plan 9a task-2 report measured.
SCENE = {
    "akita_black_bowl_1": {"pos": np.array([-0.051, 0.206, 0.898]), "quat": np.zeros(4)},
    "akita_black_bowl_2": {"pos": np.array([-0.184, 0.333, 0.898]), "quat": np.zeros(4)},
    "cookies_1": {"pos": np.array([-0.055, 0.016, 0.909]), "quat": np.zeros(4)},
    "glazed_rim_porcelain_ramekin_1": {"pos": np.array([-0.187, 0.206, 0.899]), "quat": np.zeros(4)},
    "plate_1": {"pos": np.array([-0.084, 0.197, 0.902]), "quat": np.zeros(4)},
}

BETWEEN = "pick up the black bowl between the plate and the ramekin and place it on the plate"


def scene(bowl_z: float = 0.898) -> dict:
    out = {name: {"pos": pose["pos"].copy(), "quat": pose["quat"]} for name, pose in SCENE.items()}
    out[TARGET]["pos"][2] = bowl_z
    return out


def proprio(x: float, y: float, z: float, width: float = 0.08) -> np.ndarray:
    """The 8-d vector `LiberoEnv.state_vector` produces, with the fingers parted by `width`."""
    return np.array([x, y, z, 3.134, -0.011, -0.089, width / 2, -width / 2], np.float64)


# ------------------------------------------------------------------------------ the stage rule

def test_the_first_decision_is_always_locate_even_with_the_hand_already_at_the_object():
    """`locate` is decision 0 only, and it is checked *first* rather than read off the bottom of
    the table. At decision 0 the hand can already be inside `reach`'s window (or even `grasp`'s),
    and an episode that never passed through `locate` could never say it had finished naming what
    it is acting on."""
    at_the_bowl = proprio(-0.051, 0.206, 0.918)
    assert R.phase(at_the_bowl, scene(), TARGET, DESTINATION, 0.898, decision=0) == ("locate", 1)
    # The same observation one decision later is what the geometry says it is.
    assert R.phase(at_the_bowl, scene(), TARGET, DESTINATION, 0.898, decision=1) == ("grasp", 3)


def test_no_target_named_is_locate_whenever_it_happens():
    assert R.phase(proprio(0, 0, 1.1), scene(), None, DESTINATION, None, decision=7) == ("locate", 1)
    # A target the scene does not contain is the same thing: nothing to measure to.
    assert R.phase(proprio(0, 0, 1.1), scene(), "ghost_1", DESTINATION, None, decision=7) \
        == ("locate", 1)


@pytest.mark.parametrize("name,index,x,y,z,width,bowl_z", [
    # far away, fingers open -> reach
    ("reach", 2, -0.213, -0.002, 1.176, 0.080, 0.898),
    # within 5 cm horizontally and 3 cm of grasp height, fingers open -> grasp
    ("grasp", 3, -0.053, 0.206, 0.918, 0.080, 0.898),
    # the same place with the fingers shut and the bowl still down -> lift
    ("lift", 4, -0.053, 0.206, 0.918, 0.020, 0.898),
    # fingers shut, bowl risen, far from the plate -> carry
    ("carry", 5, -0.400, 0.500, 1.050, 0.020, 0.948),
    # fingers shut, bowl risen, within 10 cm of the plate -> place
    ("place", 6, -0.084, 0.197, 1.050, 0.020, 0.948),
])
def test_every_row_of_the_subgoal_table(name, index, x, y, z, width, bowl_z):
    """One case per row of `phase`'s table. The thresholds are the ones a state text already
    prints with: `0.04` is `GRIPPER_FULL_OPEN / 2`, the width a gripper reads `open`/`closed`
    at, and `0.02` is `RISE`."""
    got = R.phase(proprio(x, y, z, width), scene(bowl_z), TARGET, DESTINATION, 0.898, decision=4)
    assert got == (name, index)
    assert R.SUBGOALS[index - 1] == name


def test_the_finger_threshold_is_the_one_a_state_string_prints_with():
    """A stage saying `grasp` (fingers open) beside a `gripper=closed` on the line above it would
    be two statements about one number. One constant, so they cannot disagree."""
    assert R.CLOSED_WIDTH == R.GRIPPER_FULL_OPEN / 2 == 0.04
    just_closed = proprio(-0.053, 0.206, 0.918, R.CLOSED_WIDTH)
    assert R.phase(just_closed, scene(), TARGET, DESTINATION, 0.898, decision=4)[0] == "lift"


def test_a_grasp_that_never_rose_is_lift_and_not_carry():
    """The whole of "is it actually held?": the simulator reports the bowl's pose every step, so
    a grasp that closed on air is a target that never rose."""
    closed = proprio(-0.051, 0.206, 0.900, 0.020)
    assert R.phase(closed, scene(0.898), TARGET, DESTINATION, 0.898, decision=4)[0] == "lift"
    # The same pose with the bowl risen is a real grasp, and (here, over the plate) a `place`.
    risen = scene(0.898 + R.RISE + 0.001)
    assert R.phase(closed, risen, TARGET, DESTINATION, 0.898, decision=4)[0] == "place"
    assert R.RISE == 0.02


def test_the_offsets_are_the_two_numbers_an_object_line_prints():
    horiz, dz = R.offsets(proprio(0.0, 0.206, 1.000), scene(), TARGET)
    assert horiz == pytest.approx(0.051, abs=1e-3)
    assert dz == pytest.approx(-0.102, abs=1e-3)
    assert R.offsets(proprio(0, 0, 1.1), scene(), "ghost_1") is None


# ------------------------------------------------------------------------------- the role rule

def test_the_spatial_phrase_stops_before_the_destination_clause():
    """Every LIBERO-Spatial sentence ends "and place it on the plate", so reading anchors out of
    the whole sentence would find the plate in all four of them."""
    assert R.spatial_phrase(BETWEEN).strip() == "between the plate and the ramekin"
    assert R.spatial_phrase(
        "pick up the black bowl next to the cookie box and place it on the plate"
    ).strip() == "next to the cookie box"


def test_the_two_anchor_phrase_uses_the_midpoint_and_not_the_nearer_anchor():
    """Measured on task 0's own poses: "nearest the ramekin" alone picks the *wrong* bowl by 3 mm,
    while the midpoint of the two anchors sits 1 cm from the right bowl and 17 cm from the wrong
    one. This is the rule both the harvest and the RoboJEV server name the target with."""
    roles = R.scene_roles(SCENE, BETWEEN, [0.0, 0.0, 1.1])
    assert roles["target"] == "akita_black_bowl_1"
    assert roles["destination"] == "plate_1"
    assert roles["rule"] == "the bowl nearest the midpoint of the plate and the ramekin"
    assert set(R.ROLE_KEYS) <= set(roles)


def test_an_anchor_with_no_observable_pose_falls_back_to_the_bowl_farther_from_the_plate():
    """The stove and the wooden cabinet are robosuite *fixtures*: body ids, no `<name>_pos`
    observable, so there is nothing to measure a distance to."""
    roles = R.scene_roles(SCENE, "pick up the black bowl on the stove and place it on the plate",
                          [0.0, 0.0, 1.1])
    assert roles["target"] == "akita_black_bowl_2"
    assert "no observable pose" in roles["rule"]


def test_an_empty_scene_is_a_value_error_in_the_library():
    """The recipe servers wrap it in their own exception type; the library itself knows nothing
    about policy servers."""
    with pytest.raises(ValueError, match="empty"):
        R.scene_roles({}, BETWEEN, [0.0, 0.0, 1.0])


def test_the_furniture_is_never_a_target():
    """A rule that matched "cabinet" against an object name would start naming the cabinet itself
    the moment the privileged state could see one."""
    with_fixture = {**SCENE, "wooden_cabinet_1": {"pos": np.array([-0.05, 0.30, 0.90]),
                                                  "fixture": True, "boxes": []}}
    roles = R.scene_roles(with_fixture, BETWEEN, [0.0, 0.0, 1.1])
    assert roles["target"] in ("akita_black_bowl_1", "akita_black_bowl_2")
    assert roles["destination"] == "plate_1"


# ------------------------------------------------------- the task definition's own answer

class _Env:
    def __init__(self, names):
        self.obj_of_interest = list(names)


def test_the_task_definitions_first_object_is_the_target():
    target, source = R.grip_target(_Env([TARGET, DESTINATION]), SCENE, [0.0, 0.0, 1.1])
    assert (target, source) == (TARGET, "obj_of_interest")
    assert R.destination_for(_Env([TARGET, DESTINATION]), SCENE, target, {}) == DESTINATION


def test_an_object_with_no_observable_pose_falls_back_to_the_nearest_and_records_it():
    """LIBERO-Spatial task 4 names a bowl inside a cabinet, and a cabinet is a fixture with a
    body id and no pose. A row labelled against a guess must never look like one labelled against
    the task definition."""
    target, source = R.grip_target(_Env(["wooden_cabinet_1"]), SCENE, [-0.05, 0.20, 0.90])
    assert source == "nearest" and target in SCENE
