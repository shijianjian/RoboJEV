"""Where this package writes at runtime.

The package harvests rows, trains checkpoints and caches scenes; all three are runtime output,
and all three go under one root rather than into the checkout. `$ROBOJEV_HOME` is that root, and
`~/.robojev` is where it points when nobody says otherwise.

`$ROBOPP_HOME` is read as a fallback and nothing more: it is the name this code used while it
lived inside an evaluation platform, and an operator whose harvests and checkpoints are already
under that root should not have to move them to keep using them.
"""
from __future__ import annotations

import os
import pathlib

#: The environment variable that names the root, and its default.
HOME_ENV = "ROBOJEV_HOME"
#: The older spelling, honoured when the current one is unset.
LEGACY_HOME_ENV = "ROBOPP_HOME"
DEFAULT_HOME = "~/.robojev"


def home() -> pathlib.Path:
    """The runtime root: `$ROBOJEV_HOME`, else `$ROBOPP_HOME`, else `~/.robojev`.

    Expanded per call, so a test's `monkeypatch.setenv` always wins, and never created as a side
    effect of reading it -- whatever writes under it makes its own directory.
    """
    named = os.environ.get(HOME_ENV) or os.environ.get(LEGACY_HOME_ENV) or DEFAULT_HOME
    return pathlib.Path(named).expanduser()


def data_dir(policy: str, suite: str) -> pathlib.Path:
    """`$ROBOJEV_HOME/data/<policy>/<suite>` -- where a harvest's rows land."""
    return home() / "data" / policy / suite


def checkpoints_dir() -> pathlib.Path:
    """`$ROBOJEV_HOME/checkpoints` -- weights this package *made*, one `<policy>/<suite>` each."""
    return home() / "checkpoints"


def src_dir() -> pathlib.Path:
    """`$ROBOJEV_HOME/src` -- pinned upstream checkouts, e.g. NanoJev's `scripts/`."""
    return home() / "src"


#: The runtime root this code used while it lived inside an evaluation platform, looked in for
#: pinned upstream checkouts when no root has been named at all.
LEGACY_DEFAULT_HOME = "~/.robopp"


def src_dirs() -> list[pathlib.Path]:
    """Every directory a pinned upstream checkout may be found in, in the order to look.

    `src_dir()` first. When **neither** root variable is set, the legacy default root's `src/` is
    looked in too: a checkout cloned there before the rename is the same pinned commit, and
    cloning 2 GB of history again to find it is not worth anything. A root that *was* named is
    the whole truth, which is also what keeps a test's `$ROBOJEV_HOME` from finding a real clone.
    """
    found = [src_dir()]
    if not (os.environ.get(HOME_ENV) or os.environ.get(LEGACY_HOME_ENV)):
        legacy = pathlib.Path(LEGACY_DEFAULT_HOME).expanduser() / "src"
        if legacy not in found:
            found.append(legacy)
    return found


def env_dir(name: str) -> pathlib.Path:
    """`$ROBOJEV_HOME/envs/<name>` -- a built Python environment for an upstream that needs one."""
    return home() / "envs" / name


__all__ = ["DEFAULT_HOME", "HOME_ENV", "LEGACY_DEFAULT_HOME", "LEGACY_HOME_ENV",
           "checkpoints_dir", "data_dir", "env_dir", "home", "src_dir", "src_dirs"]
