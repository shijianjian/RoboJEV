"""What the package promises about itself: it imports light, the loop is the loop, and the CLI
answers without a simulator.

Everything here runs on a laptop with numpy and nothing else. That is the claim: a contributor
who wants to read the state text, change a question or fix the parser should not have to install
a simulator, a GPU stack or a checkpoint first.
"""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from robojev import cli, episode as episode_mod, policy as policy_mod, recorder


# ------------------------------------------------------------------------------ importing it

def test_importing_the_package_costs_numpy_and_nothing_heavier():
    """A subprocess, because this one's whole point is what is *not* in `sys.modules`, and the
    test suite has already imported half the package by the time this runs."""
    code = (
        "import sys; import robojev; from robojev import skill, expert, policy, cli;"
        "heavy = [m for m in ('torch', 'transformers', 'libero', 'robosuite', 'mujoco')"
        " if m in sys.modules];"
        "print(heavy)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_the_simulator_adapter_imports_without_a_simulator():
    """`robojev.envs.libero` is resolved by name at call time, but the module itself has to be
    importable everywhere: it is what `DEFAULT_ADAPTER` points at, and a package whose default
    binding cannot even be named is a package with a broken error message."""
    code = ("import sys, robojev.envs.libero as m;"
            "print(m.UPSTREAM_MAX_STEPS['libero_spatial'], 'libero' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["220", "False"]


def test_the_cli_answers_for_help_without_importing_anything_heavy():
    out = subprocess.run([sys.executable, "-m", "robojev.cli", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0
    for verb in ("run", "record", "harvest", "dagger", "train"):
        assert verb in out.stdout


def test_the_index_parser_takes_a_range_as_well_as_a_list():
    assert cli.indices("0-7") == list(range(8))
    assert cli.indices("0,2,5") == [0, 2, 5]
    assert cli.indices("0-2,7,9-10") == [0, 1, 2, 7, 9, 10]
    assert cli.indices(" ") == []


# ---------------------------------------------------------------------------- the episode loop

BOWL = np.array([0.0, 0.0, 1.00], dtype=np.float32)


class _Step:
    def __init__(self, obs, done):
        self.obs, self.done, self.reward, self.info = obs, done, 0.0, {}


class FakeEnv:
    """A `DecisionEnv` whose physics is "the hand goes where it is told", succeeding at `succeed_at`."""

    instruction = "pick up the black bowl and place it on the plate"
    control_freq = 20.0
    stop_on_success = True

    def __init__(self, succeed_at: int | None = None):
        self.succeed_at = succeed_at
        self.steps = 0
        self.closed = False
        self.eef = np.array([0.0, 0.0, 1.2], dtype=np.float32)

    @property
    def num_init_states(self) -> int:
        return 2

    def reset(self, init_state_index: int) -> dict:
        self.steps = 0
        self.eef = np.array([0.0, 0.0, 1.2], dtype=np.float32)
        return {"i": init_state_index}

    def step(self, action):
        self.steps += 1
        self.eef = self.eef + np.asarray(action[:3], dtype=np.float32) * 0.05
        done = self.succeed_at is not None and self.steps >= self.succeed_at
        return _Step({"i": self.steps}, done)

    def state_vector(self, obs) -> np.ndarray:
        return np.array([*self.eef, np.pi, 0.0, 0.0, 0.04, -0.04], dtype=np.float32)

    def privileged(self, obs) -> dict:
        return {"akita_black_bowl_1": {"pos": BOWL, "quat": np.array([0, 0, 0, 1], np.float32)},
                "plate_1": {"pos": np.array([0.3, 0.2, 0.9], np.float32),
                            "quat": np.array([0, 0, 0, 1], np.float32)}}

    def images(self, obs) -> dict:
        return {"agentview": np.zeros((8, 8, 3), np.uint8), "wrist": np.zeros((8, 8, 3), np.uint8)}

    def dummy_action(self):
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]

    def close(self) -> None:
        self.closed = True


class FakePolicy:
    """Answers the same thing every time and returns `(chunk, decisions)`, like the real ones."""

    def __init__(self, chunk_steps: int = 5):
        self.chunk_steps = chunk_steps
        self.calls = 0
        self.instruction = ""

    def reset(self, instruction: str) -> None:
        self.instruction = instruction
        self.calls = 0

    def act(self, obs, *a, **kw):
        self.calls += 1
        chunk = np.zeros((self.chunk_steps, 7), np.float32)
        chunk[:, 2] = -1.0
        return chunk, {"move_z": {"probabilities": {"-": 1.0, "hold": 0.0, "+": 0.0},
                                  "choice": "-", "overridden": False},
                       "meta": {"step": self.calls - 1, "state": "Robot: fake\n",
                                "subgoal": "reach", "grip_latch": {"asked": False,
                                                                   "closed": False,
                                                                   "refused": False}}}


def test_the_loop_waits_then_queries_once_per_chunk():
    env, policy = FakeEnv(), FakePolicy()
    protocol = episode_mod.Protocol(max_steps=20, wait_steps=10, execute_steps=5)
    result = episode_mod.run_episode(env, policy, protocol)

    assert policy.instruction == env.instruction
    # 20 policy steps at 5 a chunk is four queries, and the ten wait steps ask nothing.
    assert policy.calls == 4 and result.steps == 20
    assert episode_mod.decision_count(result) == 4
    assert result.terminated_by == "max_steps" and result.success is False


def test_a_wait_step_executes_the_no_op_and_carries_no_decision():
    env, policy = FakeEnv(), FakePolicy()
    result = episode_mod.run_episode(env, policy, episode_mod.Protocol(max_steps=5, wait_steps=3))
    waits = [f for f in result.frames if f.is_wait_step]
    assert len(waits) == 3
    assert all(f.decisions is None and f.policy_query is False for f in waits)
    assert all(np.allclose(f.action, env.dummy_action()) for f in waits)


def test_every_frame_of_a_chunk_carries_the_decision_that_produced_it():
    """Not only the frame that asked: a replay bundle needs something to show at every step, and
    `policy_query` is already the flag that says which one did the asking."""
    env, policy = FakeEnv(), FakePolicy()
    result = episode_mod.run_episode(env, policy, episode_mod.Protocol(max_steps=10, wait_steps=0))
    executed = [f for f in result.frames if not f.is_wait_step and not f.is_terminal]
    assert len(executed) == 10
    assert all(f.decisions is not None for f in executed)
    assert [f.policy_query for f in executed[:5]] == [True, False, False, False, False]


def test_the_first_success_ends_the_episode():
    env, policy = FakeEnv(succeed_at=7), FakePolicy()
    result = episode_mod.run_episode(env, policy, episode_mod.Protocol(max_steps=50, wait_steps=0))
    assert result.success is True and result.terminated_by == "success"
    assert result.first_success_step == 6 and result.steps == 7


def test_a_terminal_frame_records_where_the_arm_ended_up():
    env, policy = FakeEnv(), FakePolicy()
    result = episode_mod.run_episode(env, policy, episode_mod.Protocol(max_steps=5, wait_steps=0))
    last = result.frames[-1]
    assert last.is_terminal and last.decisions is None
    assert np.isnan(last.action).all()
    # It is not a step: `steps` counts only real policy steps.
    assert result.steps == 5


def test_a_policy_that_raises_ends_the_episode_with_the_reason_in_the_record():
    class Broken(FakePolicy):
        def act(self, obs, *a, **kw):
            raise ValueError("no privileged state")

    result = episode_mod.run_episode(FakeEnv(), Broken(),
                                     episode_mod.Protocol(max_steps=5, wait_steps=0))
    assert result.success is False and result.terminated_by == "error"
    assert "no privileged state" in result.error


def test_the_privileged_scene_reaches_the_policy_without_mutating_the_observation():
    seen = {}

    class Nosy(FakePolicy):
        def act(self, obs, *a, **kw):
            seen["keys"] = set(obs)
            return super().act(obs, *a, **kw)

    episode_mod.run_episode(FakeEnv(), Nosy(), episode_mod.Protocol(max_steps=5, wait_steps=0))
    assert "privileged" in seen["keys"]


def test_a_vision_shaped_policy_is_never_handed_the_privileged_scene():
    seen = {}

    class Blind(FakePolicy):
        def act(self, obs, *a, **kw):
            seen["keys"] = set(obs)
            return super().act(obs, *a, **kw)

    episode_mod.run_episode(FakeEnv(), Blind(),
                            episode_mod.Protocol(max_steps=5, wait_steps=0,
                                                 privileged_state=False))
    assert "privileged" not in seen["keys"]


# -------------------------------------------------------------------------------- the recorder

def test_the_grasp_options_are_parsed_back_out_of_the_state_text():
    """The state text is the record; the rim table beside it in a bundle is a view of it, which
    is why `record --reparse` can rebuild it without re-running a simulator."""
    state = ("Grasp options:\n"
             "  A: +x side, turn +0, room 4.2 cm -> fits\n"
             "  B: -y side, turn -90, room 0.4 cm -> blocked by wooden_cabinet_1\n"
             "  C: -x side, turn +90, room 3.1 cm -> tried, fits\n")
    rows = recorder.rim_rows(state)
    assert [r["letter"] for r in rows] == ["A", "B", "C"]
    assert rows[0] == {"letter": "A", "side": "+x side", "turn_deg": 0, "room_cm": 4.2,
                       "verdict": "fits", "fits": True, "tried": False}
    # "does it fit" is the absence of `blocked`, not the presence of `fits`.
    assert rows[1]["fits"] is False and rows[1]["tried"] is False
    assert rows[2]["fits"] is True and rows[2]["tried"] is True


def test_the_poster_is_taken_a_little_past_the_middle():
    bundle = {"decisions": [{"t": float(i)} for i in range(21)]}
    assert recorder.poster_time(bundle) == 11.0
    assert recorder.poster_time({"decisions": []}) == 0.0


def test_a_bundle_is_json_with_no_numpy_left_in_it(tmp_path, monkeypatch):
    """Every number a float or an int, every array a list -- a bundle that cannot be serialised
    is a recording that was made and then lost."""
    env, policy = FakeEnv(succeed_at=6), FakePolicy()
    protocol = episode_mod.Protocol(max_steps=20, wait_steps=2)
    result = episode_mod.run_episode(env, policy, protocol, images=True)
    # No ffmpeg in the unit suite: the videos are the one part that needs a binary.
    monkeypatch.setattr(recorder, "write_video", lambda *a, **k: None)
    monkeypatch.setattr(recorder, "write_poster", lambda *a, **k: None)

    class Describes(FakePolicy):
        def describe(self):
            return {"checkpoint": {"repo": None, "revision": "fake-1"}}

    policy.describe = Describes().describe
    bundle = recorder.record(env, policy, result, tmp_path / "b", suite="libero_spatial",
                             task_index=0, init_state_index=0, engine="expert",
                             protocol=protocol, started=0.0, note="a note")
    written = json.loads((tmp_path / "b" / "episode.json").read_text())
    assert written == bundle
    assert written["schema_version"] == recorder.SCHEMA_VERSION
    assert written["success"] is True and written["note"] == "a note"
    assert written["control_rate"] == 20.0 and written["wait_steps"] == 2
    assert len(written["decisions"]) == episode_mod.decision_count(result)
    # Video time and control step are the same clock, which is what lets the page drive the
    # panel off `video.currentTime`.
    first = written["decisions"][0]
    assert first["t"] == pytest.approx(first["control_step"] / written["control_rate"])


# ----------------------------------------------------------------------------- the three engines

def test_the_three_engines_are_reachable_by_name():
    assert policy_mod.ENGINES == ("expert", "model", "jev")
    assert isinstance(policy_mod.build("expert"), policy_mod.ExpertPolicy)
    with pytest.raises(policy_mod.PolicyError, match="unknown engine"):
        policy_mod.build("nonesuch")


def test_a_model_engine_with_no_checkpoint_on_the_box_says_where_it_looked(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOJEV_HOME", str(tmp_path))
    with pytest.raises(policy_mod.PolicyError):
        policy_mod.build("model", "libero_spatial")
    assert policy_mod.default_checkpoint("libero_spatial") == \
        tmp_path / "checkpoints" / "robojev" / "libero_spatial"
