"""Which object the task means, and what stage the episode is in.

Two rules, both pure functions of what a *server* can see -- the instruction, the object poses and
the 8-d proprio vector -- and both used on every side of the fine-tune, which is why they live
here rather than half in a harvester and half in a recipe:

* **the roles** (`scene_roles`): which object to pick up and which to put it on, read out of the
  instruction's spatial phrase and the poses it points at. A label source that knows which task it
  is labelling may instead read the task definition's own `(:obj_of_interest …)` (`grip_target`),
  and the harvest measures how often the two agree; a server has only the sentence.
* **the stage** (`phase`): which of `SUBGOALS` one observation is in, decided from the finger
  width, the target's offset from the fingers and how far the target has risen since the episode
  began.

Both were `decision/memory.py`'s until the v1 question set was deleted; nothing about either is
v1, and v2's tracker (`v2.state.TrackerV2`) and both recipe servers read them unchanged.

Pure numpy, no simulator and no upstream import, except for `obj_of_interest`/`grip_target`, which
ask the environment adapter a question about the task definition.
"""
from __future__ import annotations

import numpy as np

from robojev import scene as scene_mod

#: The pick-and-place the suite is, as one enum. `locate` is decision 0 only (nothing has been
#: named yet); the other five are decided by `phase()` from geometry.
SUBGOALS: tuple[str, ...] = ("locate", "reach", "grasp", "lift", "carry", "place")

#: The Panda gripper's two finger joints move symmetrically: `finger_joint1` ranges over
#: `[0, 0.04]` and `finger_joint2` mirrors it over `[-0.04, 0]` (`panda_gripper.xml`), so the
#: opening width `g0 - g1` spans 0 (fully closed) to `0.08` (fully open).
GRIPPER_FULL_OPEN: float = 0.08

#: Half of `GRIPPER_FULL_OPEN`: a grasp closed *on an object* stops well short of the joint limit
#: -- at the object's width -- so a state where the fingers are closer together than this reads as
#: closed even while the numeric width is nonzero.
CLOSED_WIDTH: float = GRIPPER_FULL_OPEN / 2

#: Metres the target must gain over its height at the episode's first observation to count as
#: lifted.
RISE: float = 0.02

#: `grasp`/`lift`: how close, horizontally, the fingers have to be to the target (metres).
NEAR_XY: float = 0.05
#: `grasp`: and how near its grasp height, vertically.
GRASP_DZ: float = 0.03
#: `lift`: the looser vertical window, because the fingers are already shut on it.
LIFT_DZ: float = 0.05
#: `place`: how close, horizontally, to the destination.
PLACE_XY: float = 0.10


# ------------------------------------------------------------------ reading the instruction

#: LIBERO-Spatial's four instructions are one sentence with one spatial phrase:
#:
#:     pick up the black bowl between the plate and the ramekin and place it on the plate
#:     pick up the black bowl next to the cookie box and place it on the plate
#:     pick up the black bowl on the stove and place it on the plate
#:     pick up the black bowl on the wooden cabinet and place it on the plate
#:
#: The destination is the plate in every one of them, and the target is whichever of the two
#: identical black bowls the phrase points at. This table maps a word the phrase may use to the
#: object name that names it, or to `None` when the scene has no observable for it: the stove and
#: the cabinet are robosuite *fixtures*, which get body ids but no `<name>_pos`/`<name>_quat`
#: observables, so there is no pose to measure a distance to.
ANCHORS: tuple[tuple[str, str | None], ...] = (
    ("plate", "plate"),
    ("ramekin", "ramekin"),
    ("cookie", "cookie"),
    ("stove", None),
    ("cabinet", None),
)

BOWL = "bowl"
PLATE = "plate"
#: Where the spatial phrase ends. Everything after it is the destination clause, which names the
#: plate in every LIBERO-Spatial task and must not be read as part of the phrase -- otherwise
#: "next to the cookie box **and place it on the plate**" would name two anchors instead of one.
PHRASE_END = "and place"

#: What `scene_roles` decides.
ROLE_KEYS: tuple[str, ...] = ("target", "destination", "anchors", "rule")


def _pos(objects: dict, name: str) -> np.ndarray:
    return np.asarray(objects[name]["pos"], dtype=np.float64)[0:3]


def _match(objects: dict, word: str) -> str | None:
    """The first object whose name contains `word`, or None.

    Substring rather than equality: LIBERO's object names are the BDDL's own
    (`akita_black_bowl_1`, `glazed_rim_porcelain_ramekin_1`, `cookies_1`), and every word this
    rule looks for appears inside one of them.
    """
    for name in objects:
        if word in name:
            return name
    return None


def spatial_phrase(instruction: str) -> str:
    """The part of the instruction that says *which* bowl: between "bowl" and "and place".

    "pick up the black bowl between the plate and the ramekin and place it on the plate" ->
    " between the plate and the ramekin ". Scoping matters: every LIBERO-Spatial instruction ends
    "and place it on the plate", so reading anchors out of the whole sentence would find the
    plate in all four of them and turn every phrase into a two-anchor one. A sentence with
    neither marker falls back to itself, which is the right answer for an instruction this
    rule was not written for.
    """
    text = (instruction or "").lower()
    after = text.partition(BOWL)[2] or text
    return after.partition(PHRASE_END)[0] or after


def _anchors_in(phrase: str) -> list[tuple[str, str | None]]:
    """The anchor words the phrase uses, in the order it uses them."""
    return sorted(
        ((word, obj) for word, obj in ANCHORS if word in phrase),
        key=lambda pair: phrase.index(pair[0]),
    )


def scene_roles(objects: dict, instruction: str, eef) -> dict:
    """Which object to pick up, which to put it on, and by what rule.

    The suite decides the destination: LIBERO-Spatial's four tasks all end "place it on the
    plate", so the plate is it. The target is the interesting half, because the two black bowls
    are identical objects at different places and the instruction distinguishes them only by a
    spatial phrase. The rule is **the bowl nearest the point the phrase names** -- the anchor
    object's own position, or the midpoint of the two when the phrase names two ("between the
    plate and the ramekin").

    The midpoint half is not decoration. Measured on LIBERO-Spatial task 0, "nearest the ramekin"
    alone picks the *wrong* bowl by 3 mm -- 0.131 m to the far bowl against 0.134 m to the one
    actually between the plate and the ramekin -- while the midpoint of the two anchors sits 1 cm
    from the right bowl and 17 cm from the wrong one. Reading both halves of a two-anchor phrase
    is both more faithful to the sentence and the only one of the two that is not a coin flip.

    For the two phrases whose anchor is a fixture with no observable pose (the stove, the wooden
    cabinet) there is nothing to measure to, so the fallback is **the bowl farther from the
    plate**: both of those tasks stand the target away on furniture, and the other bowl is the
    one the plate's own scene was built around.

    It reads nothing the state string does not already print -- the instruction and the object
    poses -- which is what makes it a rule a *server* can honestly apply. The task definition's
    `(:obj_of_interest …)` would be the easy answer and it is not available at run time; a
    harvest measures how often the two agree and records it in the manifest.

    The rule that fired is returned and travels in `decisions.meta`, so a console reading a
    wrong answer can see whether the decider misread the sentence or mis-executed the motion.
    """
    # The **movable** scene: the privileged state carries the room's furniture now
    # (`robojev.scene`), and a rule that matched "cabinet" against an object name would
    # start naming the cabinet itself as a target the moment it could see one.
    objects = scene_mod.movable(objects)
    names = list(objects)
    if not names:
        raise ValueError("the privileged scene state is empty: there is nothing to act on")
    eef = np.asarray(eef, dtype=np.float64)[0:3]

    phrase = spatial_phrase(instruction)
    anchors = _anchors_in(phrase)
    anchor_words = [word for word, _ in anchors]
    # Only the anchors that are actually in the scene: the stove and the cabinet are fixtures
    # with no observable pose, and an anchor word the scene has no object for measures nothing.
    resolved = [(word, _match(objects, obj)) for word, obj in anchors if obj is not None]
    resolved = [(word, name) for word, name in resolved if name is not None]

    bowls = [n for n in names if BOWL in n]
    plate = _match(objects, PLATE)
    if plate is None or not bowls:
        # Nothing recognisable in the scene: the generic fallback, recorded as such rather than
        # guessed at silently.
        by_distance = sorted(names, key=lambda n: float(np.linalg.norm(_pos(objects, n) - eef)))
        return {
            "target": by_distance[0],
            "destination": by_distance[-1],
            "anchors": anchor_words,
            "rule": "fallback: nearest object to the end effector as target, farthest as destination",
        }

    roles = {"destination": plate, "anchors": anchor_words}
    if len(bowls) == 1:
        return {**roles, "target": bowls[0], "rule": "the only bowl in the scene"}
    if resolved:
        point = np.mean([_pos(objects, name) for _, name in resolved], axis=0)
        named = " and the ".join(word for word, _ in resolved)
        rule = (f"the bowl nearest the {named}" if len(resolved) == 1
                else f"the bowl nearest the midpoint of the {named}")
        return {**roles, "target": min(bowls, key=lambda n: float(np.linalg.norm(_pos(objects, n) - point))),
                "rule": rule}
    if anchors:
        target = max(bowls, key=lambda n: float(np.linalg.norm(_pos(objects, n) - _pos(objects, plate))))
        return {**roles, "target": target,
                "rule": f"the {anchor_words[0]} has no observable pose: the bowl farther from the plate"}
    target = min(bowls, key=lambda n: float(np.linalg.norm(_pos(objects, n) - eef)))
    return {**roles, "target": target,
            "rule": "no spatial phrase recognised: the bowl nearest the end effector"}


# ------------------------------------------------------------------- the task definition's own


def obj_of_interest(env) -> list[str]:
    """The task definition's `(:obj_of_interest …)`, in its declared order.

    LIBERO's `BDDLBaseDomain.__init__` parses it onto the robosuite env
    (`bddl_base_domain.py:127`), so it is already there and no BDDL file has to be re-parsed. An
    adapter (or a fake) that declares the attribute itself is used exactly the same way.
    """
    own = getattr(env, "obj_of_interest", None)
    if own is not None:
        return list(own() if callable(own) else own)
    return list(env.env.env.obj_of_interest)


def grip_target(env, objects: dict, eef_pos) -> tuple[str, str]:
    """`(name, source)`: which object a label source acts on, and how it was chosen.

    The task's `(:obj_of_interest …)[0]`, with the nearest object as a **recorded** fallback. The
    fallback is not hypothetical -- LIBERO-Spatial task 4 is "the black bowl in the top drawer of
    the wooden cabinet", and a cabinet is a BDDL *fixture*, which gets a body id but no
    `<name>_pos` observable, so it is absent from the movable scene and there is no pose to
    measure a lift of. When the named object is not an observable movable object, the nearest
    movable object to the end effector stands in and the source says `"nearest"`, so a row
    labelled against a guess is never mistaken for one labelled against the task definition.
    """
    for name in obj_of_interest(env):
        if name in objects:
            return name, "obj_of_interest"
    if not objects:
        raise ValueError("the task has no movable objects: nothing to measure a grasp of")
    eef = np.asarray(eef_pos, np.float64)
    nearest = min(objects, key=lambda n: float(np.linalg.norm(np.asarray(objects[n]["pos"], np.float64) - eef)))
    return nearest, "nearest"


def destination_for(env, objects: dict, target: str | None, roles: dict) -> str | None:
    """Which object the target is going **onto**.

    The task definition's `(:obj_of_interest …)` names both in order, and every LIBERO-Spatial
    task's second name is the plate, so the first observable name that is not the target is the
    destination. `scene_roles`'s answer stands in when it names nothing usable -- it reads "place
    it on the plate" out of the instruction, which is the only thing a server has.
    """
    for name in obj_of_interest(env):
        if name in objects and name != target:
            return name
    found = roles.get("destination")
    return found if found != target else None


# ---------------------------------------------------------------------------- the stage rule


def offsets(proprio, objects: dict, name: str | None) -> tuple[float, float] | None:
    """`(horiz, dz)` from the fingers to `name`, the two numbers the object line already prints."""
    if name is None or name not in objects:
        return None
    eef = np.asarray(proprio, dtype=np.float64)[0:3]
    delta = _pos(objects, name) - eef
    return float(np.hypot(delta[0], delta[1])), float(delta[2])


def phase(proprio, objects: dict, target: str | None, destination: str | None,
          start_z: float | None, *, decision: int = 1) -> tuple[str, int]:
    """`(name, index)`: which subgoal this observation is in.

    `w = g0 - g1` is the finger width, `risen = target_z - start_z` is how far the target has come
    up since the episode's first observation, and the offsets are the ones the object lines print.
    First match wins, from `place` downwards:

    | # | subgoal | rule |
    |---|---|---|
    | 6 | `place` | `w <= 0.04` and `risen > 0.02` and destination `horiz <= 0.10` |
    | 5 | `carry` | `w <= 0.04` and `risen > 0.02` |
    | 4 | `lift`  | `w <= 0.04`, target `horiz <= 0.05`, `|dz| <= 0.05`, `risen <= 0.02` |
    | 3 | `grasp` | `w > 0.04`, target `horiz <= 0.05`, `|dz| <= 0.03` ("at grasp height") |
    | 2 | `reach` | a target is named and none of the above |
    | 1 | `locate`| no target named yet (decision 0 only) |

    `locate` is checked **first**, not last, and that is the one departure from reading the table
    bottom-up: at decision 0 the hand can already be inside `reach`'s window, and a run that never
    passes through `locate` would never be able to say it had finished naming what it is acting
    on. The condition is the table's own -- "decision 0, or no target" -- so the subgoal appears
    exactly once per episode.

    Pure: everything it reads is in the arguments, so a label source's replayed state and a
    server's live observation get the same answer from the same numbers.
    """
    if decision <= 0 or target is None or target not in objects:
        return SUBGOALS[0], 1

    proprio = np.asarray(proprio, dtype=np.float64)
    width = float(proprio[6] - proprio[7])
    closed = width <= CLOSED_WIDTH
    horiz, dz = offsets(proprio, objects, target)
    risen = (float(_pos(objects, target)[2]) - float(start_z)) if start_z is not None else 0.0
    to_destination = offsets(proprio, objects, destination)

    if closed and risen > RISE:
        if to_destination is not None and to_destination[0] <= PLACE_XY:
            return "place", 6
        return "carry", 5
    if closed and horiz <= NEAR_XY and abs(dz) <= LIFT_DZ and risen <= RISE:
        return "lift", 4
    if not closed and horiz <= NEAR_XY and abs(dz) <= GRASP_DZ:
        return "grasp", 3
    return "reach", 2


__all__ = ["ANCHORS", "CLOSED_WIDTH", "GRASP_DZ", "GRIPPER_FULL_OPEN", "LIFT_DZ", "NEAR_XY",
           "PLACE_XY", "RISE", "ROLE_KEYS", "SUBGOALS", "destination_for", "grip_target",
           "obj_of_interest", "offsets", "phase", "scene_roles", "spatial_phrase"]
