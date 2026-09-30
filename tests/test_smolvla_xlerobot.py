"""The SmolVLA table task on the simulated XLeRobot: units, layouts, and one scripted demonstration."""
import importlib.util
import os

import numpy as np
import pytest

from robojev.smolvla.env import FIELDS, TableEnv
from robojev.smolvla.scene import FRONT, sample_layout

HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None


def test_fields_are_the_real_drivers():
    assert len(FIELDS) == 16 and FIELDS[:6] == [
        "left_arm_shoulder_pan.pos", "left_arm_shoulder_lift.pos", "left_arm_elbow_flex.pos",
        "left_arm_wrist_flex.pos", "left_arm_wrist_roll.pos", "left_arm_gripper.pos"]
    assert FIELDS[-4:] == ["head_motor_1.pos", "head_motor_2.pos", "x.vel", "theta.vel"]


def test_driver_units_round_trip():
    q = {"right": np.array([0.1, 1.2, -0.3, 0.4, 2.0, 1.7]), "left": np.array([-0.5, 0.0, 3.0, -1.0, 0.2, 0.0])}
    a = TableEnv.to_driver(q)
    assert np.isclose(a[11], 100.0) and np.isclose(a[5], 0.0)  # grippers 0-100
    assert np.isclose(a[6], np.degrees(0.1))  # right arm starts at field 6
    back = TableEnv.from_driver(a)
    for arm in q:
        np.testing.assert_allclose(back[arm], q[arm], atol=1e-5)


def test_layouts_are_seeded_and_in_the_acting_arms_reach():
    assert sample_layout(3) == sample_layout(3)
    arms = set()
    for seed in range(200):
        lay = sample_layout(seed)
        arms.add(lay.arm)
        side = -1 if lay.arm == "right" else 1
        assert 0.06 <= lay.bowl[0] - FRONT <= 0.10 and 0.12 <= side * lay.bowl[1] <= 0.22
        assert side * lay.plate[1] >= -0.04 - 1e-9  # at most 4 cm across the midline
        assert abs(lay.bowl[1] - lay.plate[1]) >= 0.15 - 1e-9  # bowl and plate never overlap
    assert arms == {"left", "right"}


def test_benchmark_seeds_are_not_demonstration_seeds():
    import subprocess
    import sys

    r = subprocess.run([sys.executable, "-m", "robojev.smolvla.demos", "--out", "/tmp/unused", "--episodes", "2",
                        "--first-seed", "999999"], capture_output=True, text=True)
    assert r.returncode != 0 and "reserved for the benchmark" in (r.stderr + r.stdout)


@pytest.mark.skipif(not HAS_MUJOCO or not os.environ.get("ROBOJEV_LIBERO_OBJECTS"),
                    reason="needs MuJoCo and ROBOJEV_LIBERO_OBJECTS")
@pytest.mark.parametrize("seed", [1, 2])  # one left-arm and one right-arm layout the script is known to place
def test_the_scripted_demonstration_places_the_bowl(seed):
    from robojev.smolvla.expert import Expert

    env = TableEnv(sample_layout(seed), render=False)
    assert env.layout.arm == ("left" if seed == 1 else "right")
    env.reset()
    n = 0
    for a in Expert(env).actions():
        assert a.shape == (16,) and not a[12:].any()  # head and base fields stay 0: parked, head fixed
        env.step(a)
        n += 1
    assert env.success() and 100 < n < 400
    env.close()
