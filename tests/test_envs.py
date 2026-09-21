"""`robojev.envs` and `robojev.home` -- the package's two seams to its host.

The decision package harvests, trains and serves a model; it does not own a simulator and it does
not own a directory. Both arrive from outside: an environment (or a factory for one) that matches
`DecisionEnv`, and `$ROBOJEV_HOME`. What is pinned here is that neither seam reaches for a
particular host at import time, that a thirty-line fake satisfies the protocol, and that the
failure when an adapter cannot be loaded says what to do about it.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from robojev import envs as env_mod
from robojev import home as home_mod


# ------------------------------------------------------------------------------ the protocol

class _Step:
    def __init__(self, obs, done=False):
        self.obs, self.done = obs, done


class FakeEnv:
    """A `DecisionEnv` in thirty lines and no imports -- which is the point of the protocol."""

    instruction = "pick up the black bowl and place it on the plate"
    obj_of_interest = ["bowl_1", "plate_1"]

    def __init__(self, suite="libero_spatial", task_index=0):
        self.suite, self.task_index = suite, task_index
        self.closed = False
        self._eef = np.array([0.0, 0.0, 1.1])

    @property
    def num_init_states(self) -> int:
        return 3

    def reset(self, init_state_index: int) -> dict:
        self._eef = np.array([0.0, 0.0, 1.1 + 0.01 * init_state_index])
        return {"state": self.state_vector(None)}

    def step(self, action) -> _Step:
        self._eef = self._eef + np.asarray(action, float)[0:3] * 0.01
        return _Step({"state": self.state_vector(None)})

    def state_vector(self, obs) -> np.ndarray:
        return np.array([*self._eef, np.pi, 0.0, 0.0, 0.04, -0.04], np.float32)

    def privileged(self, obs) -> dict:
        return {"bowl_1": {"pos": np.zeros(3), "quat": np.zeros(4)}}

    def dummy_action(self):
        return [0.0] * 6 + [-1.0]

    def close(self) -> None:
        self.closed = True


def test_a_fake_satisfies_the_protocol_without_inheriting_it():
    """`runtime_checkable`, so an adapter is one by *shape*: an existing wrapper is already a
    `DecisionEnv` and nothing has to be edited to say so."""
    assert isinstance(FakeEnv(), env_mod.DecisionEnv)


def test_a_thing_that_is_not_an_environment_is_not_one():
    class Half:
        def reset(self, i):
            return {}

    assert not isinstance(Half(), env_mod.DecisionEnv)


# --------------------------------------------------------------------------- the default binding

def test_the_default_adapter_is_a_name_and_not_an_import(monkeypatch):
    """A string, resolved at call time: the package stays importable on a box with no simulator,
    a `robojev --help` pays nothing for MuJoCo, and a subprocess worker handed only JSON can still
    build one."""
    assert ":" in env_mod.DEFAULT_ADAPTER
    source = pathlib.Path(env_mod.__file__).read_text(encoding="utf-8")
    assert "import robojev.envs.libero" not in source
    assert "from robojev.envs.libero" not in source
    monkeypatch.setenv(env_mod.ADAPTER_ENV, "tests.fake:Env")
    assert env_mod.adapter_name() == "tests.fake:Env"


def test_an_adapter_that_cannot_be_loaded_says_what_to_do_about_it(monkeypatch):
    monkeypatch.setenv(env_mod.ADAPTER_ENV, "no_such_module_at_all:Env")
    with pytest.raises(ImportError) as caught:
        env_mod.adapter()
    assert "env_factory" in str(caught.value) and env_mod.ADAPTER_ENV in str(caught.value)


def test_a_name_without_an_attribute_is_refused_rather_than_guessed(monkeypatch):
    monkeypatch.setenv(env_mod.ADAPTER_ENV, "robojev.envs.libero")
    with pytest.raises(ValueError, match="module:attribute"):
        env_mod.adapter()


def test_make_builds_the_configured_adapter(monkeypatch):
    monkeypatch.setenv(env_mod.ADAPTER_ENV, f"{__name__}:FakeEnv")
    env = env_mod.make("libero_spatial", 3)
    assert isinstance(env, FakeEnv) and env.task_index == 3


def test_a_caller_with_a_factory_never_reaches_the_default():
    made = []

    def factory(suite, task_index):
        made.append((suite, task_index))
        return FakeEnv(suite, task_index)

    resolved = env_mod.factory_or_default(factory)
    assert resolved is factory
    resolved("libero_spatial", 7)
    assert made == [("libero_spatial", 7)]


def test_no_factory_falls_back_to_the_configured_adapter():
    assert env_mod.factory_or_default(None) is env_mod.make


# ------------------------------------------------------------------------------------ the home

def test_the_home_is_robojev_home_and_is_read_per_call(monkeypatch, tmp_path):
    """`$ROBOJEV_HOME` stays the root: an operator's harvests and checkpoints are already there,
    and a second variable naming a second directory would be a migration nobody asked for."""
    monkeypatch.setenv(home_mod.HOME_ENV, str(tmp_path / "elsewhere"))
    assert home_mod.home() == tmp_path / "elsewhere"
    monkeypatch.delenv(home_mod.HOME_ENV)
    assert home_mod.home() == pathlib.Path(home_mod.DEFAULT_HOME).expanduser()


def test_reading_the_home_creates_nothing(monkeypatch, tmp_path):
    monkeypatch.setenv(home_mod.HOME_ENV, str(tmp_path / "unmade"))
    home_mod.home()
    home_mod.data_dir("robojev-v2", "libero_spatial")
    home_mod.checkpoints_dir()
    assert not (tmp_path / "unmade").exists()


def test_the_two_roots_are_where_the_pipeline_looks_for_them(monkeypatch, tmp_path):
    monkeypatch.setenv(home_mod.HOME_ENV, str(tmp_path))
    assert home_mod.data_dir("robojev-v2", "libero_spatial") == (
        tmp_path / "data" / "robojev-v2" / "libero_spatial")
    assert home_mod.checkpoints_dir() == tmp_path / "checkpoints"


# ---------------------------------------------------------------------------- the version block

def test_the_package_reports_only_what_it_can_see_of_itself():
    from robojev import dataset

    dataset.use_version_collector(None)
    own = dataset.versions()
    assert set(own) == {"python", "numpy"}


def test_a_host_installs_its_own_collector_and_a_failing_one_degrades(monkeypatch):
    from robojev import dataset

    try:
        dataset.use_version_collector(lambda: {"host": "deadbeef", "torch": "2.14"})
        assert dataset.versions()["host"] == "deadbeef"

        def boom():
            raise RuntimeError("no torch on this box")

        dataset.use_version_collector(boom)
        # A finished harvest must not be lost to its provenance block.
        assert set(dataset.versions()) == {"python", "numpy"}
    finally:
        dataset.use_version_collector(None)
