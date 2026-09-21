"""Gate G3's parser: the state string alone, in; every gold answer, out.

§2's principle 1 is that *every label is a deterministic function of the text the model reads* --
"if a rule cannot recover the label from the state string, the model cannot either". This module
is that rule. It takes **only** the string (no observation, no tracker, no privileged pose) and
reproduces the motion answers, `grip` and `subgoal` by reading four things out of the text and
handing them to the one label function, `robojev.compose.labels_from_waypoint`:

* the waypoint's three signed centimetre offsets and the tolerance they are judged against,
* the wrist's yaw error and its tolerance,
* whether the gripper is holding the target,
* which stage the tracker says it is in.

It deliberately does **not** read the per-axis `-> aligned / not aligned` lines, the
`Largest remaining offset` line or the `step` range words, even though all three are present:
those are the state *restating* the relation for the model's benefit, and a parser that leaned on
them would prove only that the state contains its own answer key. Reading the raw offsets instead
is what makes the property test in `tests/test_decision_v2.py` mean something -- it passes with
the annotations on and off, unchanged.

The failure mode this guards against is failure F2: in v1 the labels were *not* a function of the
state (they were what a human did next), which is irreducible noise no model size fixes.
"""
from __future__ import annotations

import re

from robojev.compose import (
    DEFAULT_ARRIVAL_CM,
    DEFAULT_TOLERANCE_CM,
    DEFAULT_YAW_TOLERANCE_DEG,
    labels_from_waypoint,
)
from robojev.questions import ACTIVE_QIDS

_WAYPOINT = re.compile(
    r"^Waypoint \(.*?\): x (?P<x>[-+][\d.]+), y (?P<y>[-+][\d.]+), z (?P<z>[-+][\d.]+)"
    r"\s+\[tolerance (?P<tol>[\d.]+); arrived within (?P<arrival>[\d.]+)\]\s*$", re.M)
_YAW = re.compile(
    r"^Wrist yaw error: (?P<yaw>[-+][\d.]+) deg \[tolerance (?P<tol>[\d.]+)\]\s*$", re.M)
_GRIPPER = re.compile(r"^Gripper: (?:open|closed) [\d.]+ cm; holding (?P<holds>.+?)\.\s*$", re.M)
#: `Subgoal so far: reach (approach).` -- the v2 subgoal is what is labelled; the expert
#: sub-stage in brackets is what explains the waypoint, and the parser does not need it.
_SUBGOAL = re.compile(r"^Subgoal so far: (?P<subgoal>\w+)\b", re.M)
#: `  A: -y side, wrist turn -90 deg, room 2.9 cm -> fits` -- one grasp candidate, one verdict.
#: The letter and the verdict are all a label needs; the numbers before the arrow are the state
#: showing its work, exactly as the per-axis lines do.
_RIM = re.compile(r"^  (?P<letter>[A-H]): .*? -> (?P<verdict>.+?)\s*$", re.M)

class UnparseableState(ValueError):
    """The state string does not carry a number a label depends on.

    Raised rather than guessed: a parser that silently defaulted a missing waypoint to zero would
    report a 100 % gate pass over states whose labels it had invented.
    """


def read(state_text: str) -> dict:
    """The four facts the label function needs, read out of the text and nothing else."""
    waypoint = _WAYPOINT.search(state_text)
    if waypoint is None:
        raise UnparseableState("no `Waypoint (...)` line with three signed offsets")
    gripper = _GRIPPER.search(state_text)
    if gripper is None:
        raise UnparseableState("no `Gripper: ... holding ...` line")
    subgoal = _SUBGOAL.search(state_text)
    if subgoal is None:
        raise UnparseableState("no `Subgoal so far:` line")
    yaw = _YAW.search(state_text)
    rim = tuple({"letter": m["letter"],
                 "fits": m["verdict"].endswith("fits"),
                 "tried": m["verdict"].startswith("tried")}
                for m in _RIM.finditer(state_text))
    return {
        "rim": rim,
        "offset_cm": (float(waypoint["x"]), float(waypoint["y"]), float(waypoint["z"])),
        "tolerance": float(waypoint["tol"]),
        "arrival": float(waypoint["arrival"]),
        "yaw_err": float(yaw["yaw"]) if yaw else 0.0,
        "yaw_tolerance": (float(yaw["tol"]) if yaw else DEFAULT_YAW_TOLERANCE_DEG),
        "holding": gripper["holds"] != "nothing",
        "phase": subgoal["subgoal"],
    }


def parse(state_text: str, qids=ACTIVE_QIDS) -> dict:
    """`state_text` -> `{move_x, move_y, move_z, step, yaw, grip, subgoal}`, the gold answers.

    Equal, byte for byte of its inputs, to what the expert's own
    `TrackerV2.gold_answers()` returns for the state it rendered -- with the annotations and
    without them. That equality *is* gate G3's parser half.
    """
    facts = read(state_text)
    return labels_from_waypoint(
        facts["offset_cm"], facts["yaw_err"], facts["holding"], facts["phase"],
        facts["tolerance"], yaw_tolerance=facts["yaw_tolerance"], arrival=facts["arrival"],
        rim=facts["rim"], qids=qids)


#: What `parse` returns a key for. `target` and `destination` are not here: they are asked once,
#: from the scene, and no rule over the motion state can recover them.
PARSED_QIDS: tuple[str, ...] = ACTIVE_QIDS

DEFAULTS = {"tolerance": DEFAULT_TOLERANCE_CM, "yaw_tolerance": DEFAULT_YAW_TOLERANCE_DEG,
            "arrival": DEFAULT_ARRIVAL_CM}
