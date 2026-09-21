"""The trainer's environment, the pinned upstream, and where a trained checkpoint lands.

The fine-tune is run by NanoJev's own `train_pipeline_decisions.py`, which pins Python 3.14 and
torch 2.14 -- an interpreter no simulator in this project can share. So the trainer gets an
environment of its own, built by `uv` from the project in `robojev/trainer/`, and this module is
the three questions `robojev.train` and `robojev.slurm` ask about it:

* **where is NanoJev?** `trainer/nanojev.json` pins a URL, a commit and a subdirectory. NanoJev is
  not a Python package -- `scripts/` is a flat directory of sibling modules importing each other
  by bare name -- so it cannot be a lockfile entry and is cloned into `$ROBOJEV_HOME/src/` and put
  on `sys.path` by whoever needs it.
* **how do I run something in that environment?** `run_command`, which returns the argv and the
  working directory and starts nothing, so a test can assert the command.
* **where do the weights go?** `$ROBOJEV_HOME/checkpoints/<policy>/<suite>`, identified by
  `sha256(best.safetensors)[:12]` -- the only identity a checkpoint nobody published has.

Nothing here downloads or builds on import.
"""
from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import pathlib
import subprocess
import sys

from robojev import home

#: The sidecar that pins an upstream no lockfile can name.
NANOJEV_PIN = "nanojev.json"

#: The trainer project inside the package: `pyproject.toml`, `uv.lock`, the pin, and `cluster/`
#: (the same pins resolved against a CUDA 12 wheel index, for a cluster node with an older driver).
TRAINER_DIR = pathlib.Path(__file__).resolve().parent / "trainer"

#: The four files NanoJev's own loader requires of a checkpoint directory. A run that produced
#: fewer of them produced nothing servable.
LOCAL_CHECKPOINT_FILES = ("best.safetensors", "config.json", "backbone_config", "tokenizer")


class RuntimeError_(RuntimeError):
    """The trainer environment or the pinned upstream is not in a usable state."""


PullError = RuntimeError_


@dataclasses.dataclass(frozen=True)
class TrainerEnv:
    """The uv project the trainer runs in.

    `lockfile_hash` is the sha256 of `uv.lock`: it identifies the *environment* the project
    builds, so two versions of this package whose lockfiles differ get two environments and never
    collide. `env_name` is that pairing, `<server>@<sha256[:12]>`.
    """

    server: str
    dir: pathlib.Path
    lockfile_hash: str
    entry: str = "train_pipeline_decisions.py"

    @property
    def env_name(self) -> str:
        return f"{self.server}@{self.lockfile_hash[:12]}"

    @property
    def env_dir(self) -> pathlib.Path:
        """`$ROBOJEV_HOME/envs/<env_name>` -- where `robojev train --build-env` builds it."""
        return home.env_dir(self.env_name)

    @property
    def venv(self) -> pathlib.Path:
        """The environment beside the project, as `uv sync --project <dir>` would create it."""
        return self.dir / ".venv"

    def built_venv(self) -> pathlib.Path | None:
        """The environment to run in, or `None` when nothing is built yet.

        A built environment under `$ROBOJEV_HOME` wins over one beside the project: same lockfile,
        same hash, so they are the same environment twice.
        """
        for candidate in (self.env_dir / ".venv", self.venv):
            if (candidate / "bin" / "python").exists():
                return candidate
        return None


def lockfile_hash(directory: pathlib.Path) -> str:
    lock = pathlib.Path(directory) / "uv.lock"
    if not lock.is_file():
        raise RuntimeError_(f"{directory} has no uv.lock; it is not a trainer project")
    return hashlib.sha256(lock.read_bytes()).hexdigest()


@functools.lru_cache(maxsize=None)
def trainer(directory: str | pathlib.Path | None = None) -> TrainerEnv:
    """The trainer project shipped with this package, or another one by path."""
    d = pathlib.Path(directory) if directory is not None else TRAINER_DIR
    return TrainerEnv(server="robojev", dir=d, lockfile_hash=lockfile_hash(d))


def nanojev_pin(r: TrainerEnv) -> tuple[str, str, str] | None:
    """`(url, commit, subdir)` for the pinned upstream, or None when the sidecar is absent."""
    path = r.dir / NANOJEV_PIN
    if not path.is_file():
        return None
    pin = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in ("url", "commit", "subdir") if not pin.get(key)]
    if missing:
        raise RuntimeError_(f"{path}: {NANOJEV_PIN} is missing {', '.join(missing)}")
    return str(pin["url"]), str(pin["commit"]), str(pin["subdir"])


def nanojev_clone(r: TrainerEnv) -> pathlib.Path | None:
    """Where `nanojev_dir` would put it, *if it is already there*. Never clones.

    The read-only half: a launch or a dry run must be able to ask "is the checkout on this box?"
    without a network round trip.
    """
    pin = nanojev_pin(r)
    if pin is None:
        return None
    directory = home.src_dir() / f"nanojev@{pin[1][:12]}"
    return directory if (directory / pin[2]).is_dir() else None


def nanojev_dir(r: TrainerEnv, log=None) -> pathlib.Path:
    """The pinned checkout, cloning it if it is not there yet."""
    pin = nanojev_pin(r)
    if pin is None:
        raise RuntimeError_(f"{r.dir} pins no upstream checkout ({NANOJEV_PIN})")
    url, commit, subdir = pin
    root = home.src_dir() / f"nanojev@{commit[:12]}"
    if not (root / subdir).is_dir():
        root.parent.mkdir(parents=True, exist_ok=True)
        if log is not None:
            log(f"cloning {url} @ {commit[:12]} -> {root}")
        if not (root / ".git").is_dir():
            subprocess.run(["git", "clone", "--filter=blob:none", url, str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "checkout", "--detach", commit], check=True)
    if not (root / subdir).is_dir():
        raise RuntimeError_(
            f"{root / subdir} is not a directory; {NANOJEV_PIN} names subdir {subdir!r}, which "
            f"{url} @ {commit[:12]} does not have")
    return root


def build_env(r: TrainerEnv, *, rebuild: bool = False, log=None) -> pathlib.Path:
    """`uv sync --frozen` the trainer project into `$ROBOJEV_HOME/envs/<env_name>`."""
    target = r.env_dir
    target.mkdir(parents=True, exist_ok=True)
    for name in ("pyproject.toml", "uv.lock"):
        (target / name).write_bytes((r.dir / name).read_bytes())
    if rebuild:
        venv = target / ".venv"
        if venv.exists():
            import shutil
            shutil.rmtree(venv)
    if log is not None:
        log(f"uv sync --frozen {target}")
    subprocess.run(["uv", "sync", "--frozen", "--project", str(target)], check=True)
    return target / ".venv"


def run_command(r: TrainerEnv, args: list[str], entry: str | pathlib.Path) -> tuple[list[str], str]:
    """The argv and cwd that run `entry` inside `r`'s environment, and nothing else.

    Three shapes, decided only by which environment `built_venv()` found: a built environment's
    interpreter directly; `uv run --project <dir>` when the environment sits beside the project;
    and this interpreter when nothing is built, which is how a test runs it.
    """
    venv = r.built_venv()
    entry = pathlib.Path(entry)
    if venv is None:
        return [sys.executable, str(entry), *args], str(r.dir)
    if venv.parent == r.dir:
        return ["uv", "run", "--project", str(r.dir), "python", str(entry), *args], str(r.dir)
    return [str(venv / "bin" / "python"), str(entry), *args], str(r.dir)


# ------------------------------------------------------------------------- the trained weights

def local_checkpoint_dir(policy: str, suite: str) -> pathlib.Path:
    """`$ROBOJEV_HOME/checkpoints/<policy>/<suite>`."""
    return home.checkpoints_dir() / policy / suite


def local_checkpoint_revision(directory) -> str:
    """`sha256(best.safetensors)[:12]` -- the identity of a checkpoint nothing published.

    Only `best.safetensors` is hashed: it is the only file training changes, the other three are
    the backbone's own config and tokenizer copied in unchanged, and a directory digest would
    depend on mtimes and on however the trainer happened to order its writes.
    """
    weights = pathlib.Path(directory) / "best.safetensors"
    h = hashlib.sha256()
    with weights.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:12]


def missing_checkpoint_files(directory) -> list[str]:
    directory = pathlib.Path(directory)
    return [name for name in LOCAL_CHECKPOINT_FILES if not (directory / name).exists()]


__all__ = ["LOCAL_CHECKPOINT_FILES", "NANOJEV_PIN", "TRAINER_DIR", "TrainerEnv", "build_env",
           "lockfile_hash", "local_checkpoint_dir", "local_checkpoint_revision",
           "missing_checkpoint_files", "nanojev_clone", "nanojev_dir", "nanojev_pin",
           "run_command", "trainer"]
