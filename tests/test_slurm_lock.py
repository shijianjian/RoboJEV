"""The two locks are one environment resolved twice: cu130 for the 3090, cu126 for the cluster.

A checkpoint trained under one and served under the other has to be the same arithmetic in the
same library versions, or the eval table the cluster produces is not comparable with the 3090's
and the weights the worker serves were made by something else. The only difference either lock is
allowed is which CUDA runtime torch carries.
"""
from __future__ import annotations

import pathlib
import re

from robojev import runtime
from robojev import slurm


#: A cluster for the tests to point at. `host`, `root` and `partition` have no defaults -- naming
#: one particular site in the package is exactly what this repository must not do -- so every test
#: that builds a config goes through here and the three values are visibly made up.
HOST = "login.cluster.example"
ROOT = "/scratch/robojev"
PARTITION = "gpu"


def config(**kw):
    kw.setdefault("host", HOST)
    kw.setdefault("root", ROOT)
    kw.setdefault("partition", PARTITION)
    kw.setdefault("qos", "normal")
    return slurm.SlurmConfig(**kw)


#: The CUDA runtime is the whole point of the second lock, so these are the packages allowed to
#: differ. Everything else -- transformers, tokenizers, numpy, safetensors, huggingface-hub,
#: msgpack and the rest of the 56 -- must match version for version.
CUDA_PACKAGES = re.compile(r"^(nvidia-|cuda-)")

_PACKAGE = re.compile(r'^\[\[package\]\]\nname = "([^"]+)"\nversion = "([^"]+)"', re.MULTILINE)


def _packages(lock: pathlib.Path) -> dict[str, str]:
    return dict(_PACKAGE.findall(lock.read_text(encoding="utf-8")))


def _locks() -> tuple[pathlib.Path, pathlib.Path]:
    r = runtime.trainer()
    return r.dir / "uv.lock", slurm.cluster_project(r) / "uv.lock"


def test_both_locks_pin_the_same_torch_version():
    here, cluster = (_packages(path) for path in _locks())
    assert here["torch"] == "2.14.0"
    # The cluster's is the same version from PyTorch's cu126 index, which spells the build into
    # the version. Anything but a `+cu126` local segment on 2.14.0 is a different torch.
    assert cluster["torch"] == "2.14.0+cu126"
    assert cluster["torch"].split("+")[0] == here["torch"]


def test_the_cluster_lock_takes_torch_from_the_cu126_index_and_nothing_else_from_it():
    _, cluster = _locks()
    text = cluster.read_text(encoding="utf-8")
    assert 'source = { registry = "https://download.pytorch.org/whl/cu126" }' in text
    # One package's worth of override: every other wheel still comes from PyPI.
    assert text.count("download.pytorch.org/whl/cu126") == len(
        [1 for line in text.splitlines() if "download.pytorch.org/whl/cu126" in line])
    non_torch = re.split(r'^name = "torch"$', text, flags=re.MULTILINE)[0]
    assert "download.pytorch.org" not in non_torch.replace(
        'index = "https://download.pytorch.org/whl/cu126"', "")


def test_everything_that_is_not_a_cuda_runtime_is_the_same_version_in_both():
    here, cluster = (_packages(path) for path in _locks())
    ours = {"robojev-trainer", "robojev-trainer-cluster"}   # the two project names
    shared = {name for name in set(here) | set(cluster)
              if not CUDA_PACKAGES.match(name) and name not in ours and name != "torch"}
    assert len(shared) > 30, "the locks resolved almost nothing in common; something is wrong"
    differ = {name: (here.get(name), cluster.get(name))
              for name in shared if here.get(name) != cluster.get(name)}
    assert differ == {}


def test_both_projects_ask_for_the_same_five_pins():
    r = runtime.trainer()
    pins = re.compile(r'^\s*"((?:torch|transformers|safetensors|numpy|huggingface-hub|msgpack)'
                      r'[^"]*)"', re.MULTILINE)
    here = set(pins.findall((r.dir / "pyproject.toml").read_text(encoding="utf-8")))
    cluster = set(pins.findall(
        (slurm.cluster_project(r) / "pyproject.toml").read_text(encoding="utf-8")))
    assert here == cluster
    assert "torch==2.14.0" in here


def test_the_cluster_runs_the_nanojev_commit_the_recipe_pins():
    """NanoJev is not a lock entry -- it is not a package (`nanojev.json`) -- so
    the one place its commit can disagree is the remote clone's name. It is derived from the same
    pin, and this is what says so."""
    r = runtime.trainer()
    url, commit, subdir = runtime.nanojev_pin(r)
    place = slurm.layout(r, config(), "libero_spatial-x")
    assert place.nanojev_commit == commit
    assert place.nanojev_url == url
    assert place.nanojev_name == f"nanojev@{commit[:12]}"
    # ...and it is the name `runtime.nanojev_clone` would give the local checkout, so the two boxes
    # run the same scripts directory relative to their own roots.
    assert place.nanojev_name == f"nanojev@{commit[:12]}"
    assert place.scripts.endswith(f"/{subdir}")


def test_the_two_environments_never_share_a_directory():
    """Same recipe, different lock, so `env_name` must differ: building the cu126 environment on
    top of the cu130 one would give the 3090 a torch its driver is newer than for no reason, and
    a box that does both keeps both."""
    r = runtime.trainer()
    cluster_hash = slurm.lock_hash(slurm.cluster_project(r))
    assert cluster_hash != r.lockfile_hash
    assert slurm.layout(r, config(), "x").env_name != r.env_name


def test_the_cluster_project_is_not_the_trainer_project():
    """A nested project is a *second* environment, built only on the cluster. Nothing here may
    treat it as the one a local `robojev train` runs in."""
    r = runtime.trainer()
    assert slurm.cluster_project(r) != r.dir
    assert slurm.cluster_project(r).parent == r.dir
