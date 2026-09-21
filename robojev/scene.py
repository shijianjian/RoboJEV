"""The scene as the planner needs it: which entries are furniture, and how much room a point has.

The privileged scene used to carry the BDDL's `(:objects ...)` and nothing else -- every
*movable* thing, each as a pose. The drawer task made the omission expensive: the thing that
stops the fingers is a drawer wall, the plan could not see it, and it spent its horizon
discovering by collision what the model had in front of it all along (note
docs/DESIGN.md §1). So the privileged state now carries the `(:fixtures ...)`
too -- the cabinet, the stove -- as a pose **and a box list**, and this module is the seam:

* `movable()` is what everything that was written before fixtures existed reads, so the state
  text, the grounding candidates and the role rules see exactly the scene they saw before;
* `obstacles()` and `clearance()` are what the planner reads, and they are the only things in the
  tree that look at a fixture's geometry.

**Why boxes and not one bounding box.** A drawer's bounding box contains the bowl inside it, so a
clearance measured against it says every grasp is blocked. What blocks a finger is a *wall*, and
the walls are exactly what the simulator's collision geoms are: thin boxes with a hole in the
middle of them, which is the drawer's interior. Each box is stored **oriented** -- `[cx, cy, cz, hx, hy, hz, r00 … r22]`, its centre, its
half-extents and its world rotation -- because the axis-aligned hull of a thin oblique plate is
not a conservative approximation of it but a different piece of furniture: this cabinet is yawed
155 degrees, and the hull of its drawer floor is a cube that fills the interior the fingers have
to enter. Measured, with hulls: all eight candidates blocked, 0.0 cm of room, on a task where
three of them grasp.

Pure numpy, no simulator, no upstream import: a server reads these same dicts off the wire.
"""
from __future__ import annotations

import numpy as np

#: The key that marks a privileged entry as furniture rather than an object of the task.
FIXTURE_KEY: str = "fixture"
#: The key its oriented collision boxes ride under: `[[cx, cy, cz, hx, hy, hz, r00 … r22], ...]`.
BOXES_KEY: str = "boxes"
#: How wide a box row is: a centre, three half-extents and a flattened 3x3 world rotation.
BOX_WIDTH: int = 15


def is_fixture(pose: dict) -> bool:
    return bool(pose.get(FIXTURE_KEY)) if isinstance(pose, dict) else False


def short_id(name: str, taken: set[str] | None = None) -> str:
    """The last two `_`-separated segments of `name`, joined by `_`.

    `akita_black_bowl_1` -> `bowl_1`, `glazed_rim_porcelain_ramekin_1` -> `ramekin_1`,
    `plate_1` -> `plate_1` (already two segments). Falls back to the full name when the short
    form collides with one already in `taken`, or when `name` has fewer than two segments.

    Object names dominate the token cost of a state string: a short id saves ~10 tokens per
    object, and every block that names an object -- the state's object lines, the waypoint, the
    events, the grounding candidates -- names it this way, so a reader never has to hold two
    spellings of one bowl.
    """
    taken = taken if taken is not None else set()
    parts = name.split("_")
    if len(parts) < 2:
        return name
    short = "_".join(parts[-2:])
    if short in taken:
        return name
    return short


def movable(objects: dict | None) -> dict:
    """The scene without its furniture -- the dict every caller written before fixtures existed
    means when it says `objects`, byte for byte."""
    return {n: p for n, p in (objects or {}).items() if not is_fixture(p)}


def fixtures(objects: dict | None) -> dict:
    return {n: p for n, p in (objects or {}).items() if is_fixture(p)}


def obstacles(objects: dict | None, exclude: str | None = None,
              fixtures_only: bool = True) -> list[tuple[str, np.ndarray]]:
    """`[(name, boxes)]` for everything a finger could hit: each fixture's own collision boxes,
    and -- only if asked for -- every other movable object as one small box about its pose.

    **Fixtures only, by default, and that is a statement about what the privileged state knows.**
    A fixture carries its own collision geometry, so a wall is where the wall is. A movable
    object carries a *pose* and no extent, and its origin is wherever its mesh was authored: this
    suite's plate reports z 0.970 and the bowl beside it 0.898, seven centimetres apart on one
    flat table. A box drawn round such a pose is fiction, and it costs episodes -- measured, a
    rim point several centimetres clear of a plate read as blocked by it, on a task that was
    10/10 and became 9/10.

    `fixtures_only=False` adds each movable object as one `MOVABLE_HALF` box about its pose, for
    a caller that wants a rough answer and knows that is what it is getting.
    """
    out: list[tuple[str, np.ndarray]] = []
    for name, pose in (objects or {}).items():
        if name == exclude:
            continue
        if is_fixture(pose):
            boxes = np.asarray(pose.get(BOXES_KEY) or [], dtype=np.float64).reshape(-1, BOX_WIDTH)
            if boxes.size:
                out.append((name, boxes))
            continue
        if fixtures_only:
            continue
        centre = np.asarray(pose["pos"], dtype=np.float64).reshape(3)
        out.append((name, np.array([[*centre, MOVABLE_HALF, MOVABLE_HALF, MOVABLE_HALF,
                                     1, 0, 0, 0, 1, 0, 0, 0, 1]])))
    return out


#: Half the width a movable object is treated as having when a finger is measured against it.
#: The suite's bowls are ~6 cm across the rim and its ramekins less.
MOVABLE_HALF: float = 0.03


def _local(point: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """`point` in each box's own frame: `(N, 3)`, the rotation applied transposed."""
    rotations = boxes[:, 6:15].reshape(-1, 3, 3)
    delta = point[None, :] - boxes[:, 0:3]
    return np.einsum("nji,nj->ni", rotations, delta)


def _distance_to_boxes(point: np.ndarray, boxes: np.ndarray) -> float:
    """The distance from a point to the nearest of a set of oriented boxes; 0 inside one.

    **Boxes whose top is below the point are not obstacles**, they are the floor. A fingertip at
    the grasp height stands 2.5 cm above whatever the object is resting on, by construction, and
    a rule that counted that surface would report the same 2.3 cm of room for every direction and
    rank nothing (measured, on the task whose bowl stands on the cabinet's top). What stops a
    finger is something that reaches *past* it.
    """
    standing = boxes[_tops(boxes) >= float(point[2]) - FLOOR_TOL_M]
    if not standing.size:
        return float("inf")
    gap = np.abs(_local(point, standing)) - standing[:, 3:6]
    return float(np.min(np.linalg.norm(np.clip(gap, 0.0, None), axis=1)))


def clearance(point, objects: dict | None, exclude: str | None = None,
              fixtures_only: bool = True) -> tuple[float, str | None]:
    """`(metres to the nearest obstacle, what it is)` for a point in world metres.

    The one measurement the candidate ranking and the printed block are both made of, so what the
    plan prefers and what the state says cannot disagree.
    """
    point = np.asarray(point, dtype=np.float64).reshape(3)
    best, who = float("inf"), None
    for name, boxes in obstacles(objects, exclude, fixtures_only):
        distance = _distance_to_boxes(point, boxes)
        if distance < best:
            best, who = distance, name
    return (best, who) if who is not None else (float("inf"), None)


def _tops(boxes: np.ndarray) -> np.ndarray:
    """The world z of each box's highest point: its centre plus its own extent along world z."""
    rotations = boxes[:, 6:15].reshape(-1, 3, 3)
    return boxes[:, 2] + np.einsum("nj,nj->n", np.abs(rotations[:, 2, :]), boxes[:, 3:6])


def _footprints(boxes: np.ndarray) -> np.ndarray:
    """Each box's world-axis-aligned horizontal half-extent: `(N, 2)`.

    The hull rather than the oriented rectangle, and here that is the right approximation: this
    answers "what is the target standing in", where including a little too much means naming the
    drawer the bowl is in rather than nothing at all.
    """
    rotations = boxes[:, 6:15].reshape(-1, 3, 3)
    return np.einsum("nij,nj->ni", np.abs(rotations[:, 0:2, :]), boxes[:, 3:6])


#: How far below a fingertip a surface may be and still be floor rather than an obstacle.
FLOOR_TOL_M: float = 0.005

#: How far from the target a fixture's box still counts as a wall the extraction has to clear.
#: A drawer's own front panel is 10 cm from the bowl; the far side of the cabinet is not a wall
#: this lift has anything to do with.
NEAR_M: float = 0.25


def support(target: str, objects: dict | None) -> tuple[str | None, float]:
    """`(what the target is resting in or on, how far that thing reaches above it, in metres)`.

    The obstacle with a box **under** the target's own pose and horizontally around it, and then
    the highest that obstacle reaches above the target within `NEAR_M` -- which is the height an
    extraction has to clear, and the number the state prints. A bowl on the table gets
    `(None, 0.0)`; a bowl in an open drawer gets the cabinet and the height of the drawer's front
    panel above the bowl's own base.
    """
    poses = objects or {}
    if target not in poses:
        return None, 0.0
    centre = np.asarray(poses[target]["pos"], dtype=np.float64).reshape(3)
    best_name, best_rise, best_top = None, 0.0, -float("inf")
    for name, boxes in obstacles(poses, exclude=target):
        tops = _tops(boxes)
        around = np.all(np.abs(centre[None, 0:2] - boxes[:, 0:2])
                        <= _footprints(boxes) + MOVABLE_HALF, axis=1)
        under = around & (tops <= centre[2] + 1e-6)
        if not under.any() or float(tops[under].max()) <= best_top:
            continue
        near = np.linalg.norm(centre[None, 0:2] - boxes[:, 0:2], axis=1) <= NEAR_M
        best_top = float(tops[under].max())
        best_name = name
        best_rise = float(max(tops[near].max() - centre[2], 0.0)) if near.any() else 0.0
    return best_name, best_rise


__all__ = ["BOXES_KEY", "BOX_WIDTH", "FIXTURE_KEY", "FLOOR_TOL_M", "MOVABLE_HALF", "NEAR_M",
           "clearance", "fixtures", "is_fixture", "movable", "obstacles", "short_id", "support"]
