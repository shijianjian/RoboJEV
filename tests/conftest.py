"""What every test in this directory gets: no simulator unless there is one, and no `$HOME`.

Three markers, and each means "this test needs something the CPU suite does not have":

* `sim` -- a simulator. LIBERO pulls robosuite and MuJoCo and downloads its own assets, so a
  contributor running `pytest` on a laptop has none of it; these are skipped rather than failed,
  because "you have not installed the simulator" is not a defect in this package.
* `gpu` -- a CUDA device and a 2.4 GB checkpoint.
* `network` -- a hosted API and a key.

A skipped test is not a passing test, so the skips are counted and named rather than hidden: the
CI job that has a simulator runs the same file and does not skip them.
"""
from __future__ import annotations

import importlib.util
import os

import pytest


def _importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


HAVE_LIBERO = _importable("libero")
HAVE_TORCH = _importable("torch")


def pytest_collection_modifyitems(config, items):
    no_sim = pytest.mark.skip(reason="no simulator: pip install 'robojev[libero]'")
    no_gpu = pytest.mark.skip(reason="no torch/CUDA: pip install 'robojev[model]'")
    no_net = pytest.mark.skip(reason="network tests are opt-in: set ROBOJEV_TEST_NETWORK=1")
    want_network = os.environ.get("ROBOJEV_TEST_NETWORK") == "1"
    for item in items:
        if "sim" in item.keywords and not HAVE_LIBERO:
            item.add_marker(no_sim)
        if "gpu" in item.keywords and not HAVE_TORCH:
            item.add_marker(no_gpu)
        if "network" in item.keywords and not want_network:
            item.add_marker(no_net)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """No test reads or writes the developer's real runtime root.

    Set for every test, not only the ones that look like they need it: a harvest, a checkpoint
    lookup and a key read all resolve their path at call time, and the one that forgets to point
    it somewhere safe is the one that writes into somebody's `~/.robojev`.
    """
    monkeypatch.setenv("ROBOJEV_HOME", str(tmp_path / "robojev-home"))
    monkeypatch.delenv("ROBOPP_HOME", raising=False)
    monkeypatch.delenv("JEV_API_KEY", raising=False)
