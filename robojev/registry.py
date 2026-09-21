"""Which question set a run speaks, and nothing else.

A checkpoint carries its version in its manifest, and the server, the harvester and the trainer
all have to agree on what that string means. This module is the one place it is decoded.

`check()` raises on anything else, naming the whole of `VERSIONS`: a server handed `"v3"` from a
manifest must fail at launch, not at the tenth step of the first episode. A **retired** set gets
its own sentence, because "unknown version" would be a lie about a string this repository used to
write -- see `RETIRED`.

Stdlib only at module scope, and the question modules are imported **inside** the functions, so
`import robojev` stays light (no torch, no simulator, no numpy pulled in by this file).
"""
from __future__ import annotations

from types import SimpleNamespace

VERSIONS: tuple[str, ...] = ("v2",)
DEFAULT_VERSION: str = "v2"

#: What a checkpoint whose `robojev.json` names no version is: the first question set, which
#: predates the key. Reading it as the retired name rather than as today's default is what makes
#: such a checkpoint refuse at launch instead of being served under a vocabulary it never saw.
UNVERSIONED: str = "v1"

#: Question sets this repository used to write into a manifest and can no longer serve, and why.
#: A checkpoint naming one is refused by `check` in one line, rather than failing later with a
#: `KeyError` about a question nothing defines.
RETIRED: dict[str, str] = {
    "v1": "the single-axis translate/rotate/magnitude/grip set, retired after it scored 0/40 "
          "closed loop; its weights, harvest and serving path were deleted",
}


def check(version: str) -> str:
    """`version` if it names a question set this package serves, else `ValueError`."""
    if version in VERSIONS:
        return version
    if version in RETIRED:
        raise ValueError(
            f"this checkpoint declares question set {version!r}, which is retired: "
            f"{RETIRED[version]}. Re-harvest and retrain at "
            f"{', '.join(VERSIONS)} rather than serving it."
        )
    raise ValueError(
        f"unknown question-set version {version!r}; known versions are {', '.join(VERSIONS)}")


def qids(version: str) -> tuple[str, ...]:
    """The question ids `version` asks per decision, in the order they are answered."""
    check(version)
    from robojev.questions import ACTIVE_QIDS                # noqa: PLC0415

    return ACTIVE_QIDS


def defined_qids(version: str) -> tuple[str, ...]:
    """Every question `version` **defines**, asked by default or not.

    Wider than `qids` on purpose: the set defines `yaw` and the shared `step` and asks whichever
    of them a run was harvested with. A checkpoint declares its own set in `robojev.json` and is
    served with it (`robojev/policy.py`), so what a server has to validate is "is every
    question this checkpoint names one I know how to ask", not "is it today's default".
    """
    check(version)
    from robojev.questions import ALL_MOTION_QIDS            # noqa: PLC0415

    return ALL_MOTION_QIDS


def questions_block(version: str, **kw) -> dict:
    """The `{qid: {type, instructions, criteria}}` block for one request."""
    check(version)
    from robojev.questions import questions_block as block   # noqa: PLC0415

    return block(**kw)


def max_path_tokens(version: str) -> int:
    """The `max_length` both halves of the model must be run at for this version.

    Upstream raises rather than truncating an oversized path, so this is a real constraint and
    not a style note -- a checkpoint trained at one budget and served at another is being asked a
    different question about the same state.
    """
    check(version)
    from robojev.questions import MAX_PATH_TOKENS_V2         # noqa: PLC0415

    return MAX_PATH_TOKENS_V2


def composer(version: str):
    """`SimpleNamespace(compose=…, chunk=…)`: answers -> the LIBERO action, for this version."""
    check(version)
    from robojev.compose import chunk, compose               # noqa: PLC0415

    return SimpleNamespace(compose=compose, chunk=chunk)
