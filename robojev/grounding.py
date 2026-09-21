"""RoboJEV v2 grounding data: which object the sentence means, decided once per episode.

Spec `docs/DESIGN.md` §1 F3, §3 and gate G5.

Failure F3 was measured, not guessed: on the held-out task the arm never came within 0.25 m of
the bowl, because every single motor decision had to implicitly ground *"the black bowl between
the plate and the ramekin"* to one of two identical bowls -- and ten instructions are far too few
to learn grounding from. §2's principle 4 separates the two jobs: grounding is asked **once**, at
reset, before any motion, and its answer is written into the tracker so every later question can
say "the target".

That makes the training data cheap and large. Every LIBERO suite's BDDL already names the object
of interest and the goal predicate, so the ~130 instructions across `libero_spatial`,
`libero_object`, `libero_goal`, `libero_10` and `libero_90` give grounding rows **without a single
rollout** -- one env construction per task to read the initial positions out of the simulator, and
nothing else. This module is that pipeline, plus gate G5's two CPU probes.

Three parts, in order:

* **the BDDL** (`parse_bddl`, `task_kind`, `target_of`, `destination_of`) -- a self-contained
  s-expression reader. Self-contained on purpose: the tests parse a fixture file in the CPU suite,
  and `libero.libero.envs.bddl_utils` cannot be imported without robosuite.
* **the scene** (`scene_records`, `build_scene_cache`) -- one environment per task, several init
  states each, movable poses from the environment's `privileged` and fixture poses read off the
  MuJoCo bodies. Cached to JSONL so every re-run of the row builder and the probes is instant.
* **the rows** (`relations`, `grounding_state`, `build_rows`) -- the NanoJev training row, whose
  `state` is a *grounding state*: the instruction, then one line per candidate with its short id,
  its category word, its position in centimetres and its spatial relations **in words computed
  from the positions**. The relation words are the whole point: they are what lets a reader
  resolve "between the plate and the ramekin" or "next to the cookie box" or "on the stove"
  without having to do coordinate geometry in its head.

**The frame is the viewer's, not the robot's.** LIBERO's instructions say "the black bowl on the
left", "the bowl at the back", "the book on the right", and those words are written from in front
of the table -- the `agentview` camera's side, looking at the robot. Measured against the BDDL's
own region names (`akita_black_bowl_left_init_region` at world y = -0.15,
`..._right_init_region` at y = +0.05; `..._front_init_region` at world x = +0.10,
`..._back_init_region` at x = -0.15): **+x is front (toward the viewer, away from the robot), +y
is right, +z is up**, measured from the centre of the surface the objects rest on and from
its height (`_scene_frame`). In the *robot's*
own frame that left/right is mirrored, so a grounding state written in gripper-relative
coordinates would have to flip the words; that is why this module carries its own renderer rather
than reusing the motor state's.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import math
import os
import pathlib
import random
import re
import sys

from robojev import envs as env_mod

#: Every LIBERO suite whose BDDL files carry a `(:language …)` and a `(:goal …)`. `libero_100` is
#: `libero_90` + `libero_10` under one name in `benchmark.get_benchmark_dict()`, so listing it
#: would double every row it contains.
SUITES: tuple[str, ...] = ("libero_spatial", "libero_object", "libero_goal", "libero_10",
                           "libero_90")

#: Where `build_scene_cache` writes, and `load_scenes` reads. One env construction per task costs
#: a few seconds and there are ~130 of them; the cache is what makes re-running the row builder
#: and the two G5 probes free.
SCENE_CACHE: pathlib.Path = pathlib.Path(
    os.environ.get("ROBOPP_V2_SCENES", "~/scratch/v2-grounding/scenes.jsonl")
).expanduser()

#: Where a cache of **settled** scenes lives, and the default this module now builds.
SETTLED_SCENE_CACHE: pathlib.Path = pathlib.Path(
    os.environ.get("ROBOPP_V2_SCENES_SETTLED",
                   "~/scratch/v2-grounding/scenes-settled.jsonl")
).expanduser()

#: No-op control steps to run after `reset` before a scene's poses are read.
#:
#: **The second train/serve gap, measured.** The platform's episode loop runs
#: `protocol.wait_steps` dummy actions before the policy's first `act` (upstream's
#: `GenerateConfig.num_steps_wait`), so a server grounds against a scene that has *settled*: the
#: objects have dropped the millimetre or two that `set_init_state` leaves them above their
#: resting place, and a stack has closed. A cache captured immediately after `set_init_state`
#: describes a different scene. On LIBERO-Spatial task 3 those ten steps flip the bowl's relation
#: from "on top of cookies_1" to "inside cookies_1" -- one relation word, in the candidate line
#: the model scores.
SETTLED_STEPS: int = 10

#: How many init states per task. Positions differ between them, so the same sentence appears in
#: several different scenes and a model cannot ground it by memorising one coordinate triple.
INIT_STATES_PER_TASK: int = 5

# ------------------------------------------------------------------ relation thresholds, in cm

#: Two entities closer than this along an axis are level, not "left of" / "in front of".
LEVEL_CM: float = 3.0
#: How far off the segment between two entities a third may sit and still be "between" them, and
#: how far apart those two must be for the phrase to carry information at all.
BETWEEN_OFF_CM: float = 6.0
BETWEEN_SPAN_CM: float = 12.0
#: Horizontal radius and height difference that make one entity "on top of" another.
ON_TOP_XY_CM: float = 9.0
ON_TOP_DZ_CM: float = 3.0
#: Horizontal radius and height difference that make one entity "inside" another (co-located: a
#: bowl in a drawer, a box in the basket).
INSIDE_XY_CM: float = 6.0
INSIDE_DZ_CM: float = 2.0
#: How close an object must be to one of a fixture's **region sites** to be called in or on that
#: part of it, and how far above the site it may still be. A fixture's body sits at its base and
#: can be 15 cm from the part an object rests on -- the stove's burner is at the far end of the
#: stove, the cabinet's top drawer slides out of it -- so the body alone gets "the bowl on the
#: stove" wrong, and the sites (LIBERO's own `(:regions …)`, compiled into MuJoCo sites) get it
#: right.
ANCHOR_XY_CM: float = 9.0
ANCHOR_DZ_CM: float = 25.0
#: Two of an entity's regions this close together horizontally are the same place as far as the
#: words go, and then height decides which one an object is resting on (`_containment`).
ANCHOR_TIE_CM: float = 5.0
#: Two entities within this distance are "next to" each other.
NEXT_TO_CM: float = 16.0
#: How many `between A and B` phrases one candidate carries, closest to the line first. Every
#: pair of neighbours is a candidate pair, so an eight-object scene offers 21 of them and most
#: say nothing -- "between the bowl and the table" is true of half the table.
MAX_BETWEEN_RELATIONS: int = 2
#: How many pairwise left/right/front/behind phrases one candidate line carries. The nearest
#: neighbours first -- a phrase about an object 60 cm away resolves nothing and costs tokens.
MAX_NEIGHBOUR_RELATIONS: int = 3

#: The held-out split. Every LIBERO-Spatial task 8 and 9 instruction is test (those two are the
#: run-3/run-4 evaluation tasks, so G5 measures exactly the generalisation the closed-loop count
#: needs), plus a seeded random fraction of every other suite's instructions.
HELD_OUT_SPATIAL_TASKS: tuple[int, ...] = (8, 9)
TEST_FRACTION: float = 0.15
SPLIT_SEED: int = 20260921

#: `gold_probs` here is a hard one-hot -- the BDDL *states* which object the sentence means, so
#: there is nothing to soften. Both kinds are in the trainer's accepted sets
#: (`tests/test_harvest.py::PROBS_KINDS`, `LABEL_KINDS`).
PROBS_KIND: str = "deterministic_truth"
LABEL_KIND: str = "deterministic_truth"

EPISODE_INSTRUCTIONS: dict[str, str] = {
    "target": (
        "Which object in the scene does the task sentence tell the robot to pick up or act on? "
        "The scene block above gives every candidate's position and where it lies relative to "
        "the others. Choose the one the sentence describes. This is asked once, before any "
        "motion; every later question refers to your answer as 'the target'."
    ),
    "destination": (
        "Where does the task sentence say the target must end up? The scene block above gives "
        "every candidate's position and where it lies relative to the others. Choose the object "
        "or fixture the sentence names as the destination. This is asked once, before any motion."
    ),
}


# ==================================================================== part 1: the BDDL

@dataclasses.dataclass(frozen=True)
class Predicate:
    """One goal or init literal, name lowercased: `(On akita_black_bowl_1 plate_1)`."""

    name: str
    args: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class BddlTask:
    suite: str
    task_index: int
    name: str
    instruction: str
    #: `{entity name: category}`, from `(:objects …)` -- everything the simulator gives an
    #: observable, i.e. everything that can be picked up.
    objects: dict[str, str]
    #: `{entity name: category}`, from `(:fixtures …)` -- tables, cabinets, stoves, microwaves.
    fixtures: dict[str, str]
    #: `{full region name: owning entity}`. A goal says `flat_stove_1_cook_region`; the region
    #: `cook_region` declares `(:target flat_stove_1)`, so the destination entity is the stove.
    regions: dict[str, str]
    obj_of_interest: tuple[str, ...]
    init: tuple[Predicate, ...]
    goal: tuple[Predicate, ...]

    @property
    def entities(self) -> dict[str, str]:
        return {**self.objects, **self.fixtures}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\(|\)|[^\s()]+", text)


def _read_sexp(tokens: list[str], i: int):
    """The classic two-line reader: `(a b (c))` -> `["a", "b", ["c"]]`."""
    if tokens[i] != "(":
        return tokens[i], i + 1
    out: list = []
    i += 1
    while tokens[i] != ")":
        node, i = _read_sexp(tokens, i)
        out.append(node)
    return out, i + 1


def _typed_names(body: list) -> dict[str, str]:
    """`akita_black_bowl_1 akita_black_bowl_2 - akita_black_bowl` -> `{name: category}`.

    PDDL's typed-list syntax: names, a bare `-`, then the type they all share. A name with no
    `- type` after it (LIBERO never writes one, but the grammar allows it) keeps itself as its
    own category rather than being dropped.
    """
    out: dict[str, str] = {}
    pending: list[str] = []
    i = 0
    while i < len(body):
        token = body[i]
        if token == "-":
            category = body[i + 1]
            for name in pending:
                out[name] = category
            pending = []
            i += 2
            continue
        pending.append(token)
        i += 1
    for name in pending:
        out[name] = name
    return out


def _predicates(body: list) -> tuple[Predicate, ...]:
    """Every literal under a `(:goal …)` or `(:init …)`, with `And` flattened away.

    A goal is `(And (On a b) (Turnon c))` or, in a few files, a bare `(On a b)`; both shapes come
    back as the same flat tuple, which is what `task_kind` and `destination_of` read.
    """
    out: list[Predicate] = []
    for node in body:
        if not isinstance(node, list) or not node:
            continue
        head = node[0]
        if head.lower() == "and":
            out.extend(_predicates(node[1:]))
        else:
            out.append(Predicate(head.lower(), tuple(node[1:])))
    return tuple(out)


def parse_bddl(text: str, *, suite: str = "", task_index: int = -1, name: str = "") -> BddlTask:
    """Read one `.bddl` file's text into a `BddlTask`.

    Self-contained rather than `libero.libero.envs.bddl_utils.robosuite_parse_problem`: that
    module cannot be imported without robosuite and MuJoCo, and the whole point of this pipeline
    is that its parsing half runs in the plain CPU test suite in milliseconds.
    """
    tokens = _tokenize(text)
    tree, _ = _read_sexp(tokens, 0)
    sections: dict[str, list] = {}
    for node in tree:
        if isinstance(node, list) and node and isinstance(node[0], str) and node[0].startswith(":"):
            sections[node[0][1:].lower()] = node[1:]

    regions: dict[str, str] = {}
    for node in sections.get("regions", []):
        if not isinstance(node, list) or not node:
            continue
        region_name = node[0]
        for child in node[1:]:
            if isinstance(child, list) and child and child[0] == ":target":
                regions[f"{child[1]}_{region_name}"] = child[1]

    return BddlTask(
        suite=suite,
        task_index=task_index,
        name=name,
        instruction=" ".join(sections.get("language", [])).strip(),
        objects=_typed_names(sections.get("objects", [])),
        fixtures=_typed_names(sections.get("fixtures", [])),
        regions=regions,
        obj_of_interest=tuple(sections.get("obj_of_interest", [])),
        init=_predicates(sections.get("init", [])),
        goal=_predicates(sections.get("goal", [])),
    )


#: `On`/`In` mean "the first argument ends up carried onto/into the second"; the rest are states
#: of a fixture the arm changes without carrying anything.
PLACE_PREDICATES: frozenset[str] = frozenset({"on", "in"})
ARTICULATE_PREDICATES: frozenset[str] = frozenset({"open", "close"})
TOGGLE_PREDICATES: frozenset[str] = frozenset({"turnon", "turnoff"})


def owner_of(task: BddlTask, argument: str) -> str | None:
    """The entity a goal argument names: itself, or the entity a region belongs to.

    `plate_1` -> `plate_1`; `flat_stove_1_cook_region` -> `flat_stove_1`;
    `kitchen_table_plate_init_region` -> `kitchen_table`.
    """
    if argument in task.entities:
        return argument
    return task.regions.get(argument)


def place_goals(task: BddlTask) -> tuple[tuple[str, str], ...]:
    """`((carried object, destination entity), …)` for every `On`/`In` goal that moves an object."""
    out: list[tuple[str, str]] = []
    for pred in task.goal:
        if pred.name not in PLACE_PREDICATES or len(pred.args) != 2:
            continue
        carried = pred.args[0]
        if carried not in task.objects:  # `(On wooden_cabinet_1 …)` in an init, never a goal
            continue
        destination = owner_of(task, pred.args[1])
        if destination is None:
            continue
        out.append((carried, destination))
    return tuple(out)


def task_kind(task: BddlTask) -> str:
    """What the goal asks for, as a `+`-joined sorted label.

    `place` (an object is carried somewhere), `articulate` (a drawer or door opens or closes),
    `toggle` (a stove or a switch changes state). Only a kind containing `place` has a
    `destination`; "open the top drawer of the cabinet" carries nothing, so asking where it goes
    would be asking for an answer that does not exist.
    """
    kinds: set[str] = set()
    if place_goals(task):
        kinds.add("place")
    for pred in task.goal:
        if pred.name in ARTICULATE_PREDICATES:
            kinds.add("articulate")
        elif pred.name in TOGGLE_PREDICATES:
            kinds.add("toggle")
        elif pred.name in PLACE_PREDICATES and pred.args and pred.args[0] not in task.objects:
            kinds.add("articulate")
    return "+".join(sorted(kinds)) if kinds else "unknown"


def _is_table(task: BddlTask, entity: str) -> bool:
    return task.fixtures.get(entity, "").endswith("table") or entity == "floor"


def target_of(task: BddlTask) -> tuple[str | None, str]:
    """The object the instruction tells the arm to act on, and why it is or is not unique.

    A pick-and-place task's target is the carried object of its single place goal. A task with no
    place goal ("open the top drawer of the cabinet", "turn on the stove") still has a target --
    the fixture the goal names -- so those rows are kept, which is why `target`'s candidate set is
    objects *and* the non-table fixtures rather than objects alone.

    Returns `(None, reason)` when the sentence has more than one answer ("put **both** the
    alphabet soup and the cream cheese box in the basket"): a single-answer choice question
    cannot be labelled from a two-answer sentence, and inventing one of the two as the gold would
    teach the model to guess.
    """
    carried = {c for c, _ in place_goals(task)}
    if len(carried) > 1:
        return None, "several objects are carried"
    if carried:
        return next(iter(carried)), "carried object of the place goal"
    acted: set[str] = set()
    for pred in task.goal:
        if pred.name in ARTICULATE_PREDICATES | TOGGLE_PREDICATES | PLACE_PREDICATES:
            for arg in pred.args:
                owner = owner_of(task, arg)
                if owner is not None and not _is_table(task, owner):
                    acted.add(owner)
    if len(acted) == 1:
        return next(iter(acted)), "fixture the goal changes"
    if not acted:
        return None, "no goal entity"
    interest = [e for e in task.obj_of_interest if e in acted]
    if len(interest) == 1:
        return interest[0], "object of interest among the goal entities"
    return None, "several goal entities"


def destination_of(task: BddlTask) -> tuple[str | None, str]:
    """Where the carried object must end up, and why it is or is not askable.

    A goal that places onto a *table* region ("put the chocolate pudding to the left of the
    plate" is `(On chocolate_pudding_1 kitchen_table_left_of_plate_region)`) has no object-level
    destination: the honest answer to "which object does it go on" is "none of them", and
    labelling it `table_1` would train the model to answer "the table" to a sentence that names
    the plate. Those rows keep their `target` and drop their `destination`.
    """
    places = place_goals(task)
    if not places:
        return None, "no place goal"
    destinations = {d for _, d in places}
    if len(destinations) > 1:
        return None, "several destinations"
    destination = next(iter(destinations))
    if _is_table(task, destination):
        return None, "destination is a table region"
    return destination, "destination of the place goal"


# ==================================================================== part 2: the scene

@dataclasses.dataclass(frozen=True)
class Entity:
    """One candidate: where it is, what it is, and whether it can be picked up.

    `pos` is `(x, y, z)` in **centimetres in the viewer's table frame** (module docstring), or
    `None` for a fixture whose body the simulator does not expose -- such a fixture is still
    listed, by name, as "position unknown", because leaving it out of the candidate set would
    silently change the question.
    """

    name: str
    category: str
    kind: str  # "object" | "fixture"
    pos: tuple[float, float, float] | None
    #: `((word, "region" | "side", (x, y, z)), …)` -- the entity's own BDDL regions, as MuJoCo
    #: sites: the cabinet's `top`, `middle` and `bottom` drawers and its `top side`, the stove's
    #: `cook` surface, a tray's `contain`. These are what "in the top drawer of the cabinet" and
    #: "on the stove" actually name.
    anchors: tuple[tuple[str, str, tuple[float, float, float]], ...] = ()


def _scene_frame(sim, object_positions) -> tuple[float, float, float]:
    """The origin every position in a row is measured from: metres, world axes.

    `x, y` come from whatever body holds the scene up -- `table` in the LIBERO-Spatial and kitchen
    scenes, `<fixture>` in the living-room and study ones, `floor` in LIBERO-Object -- and are
    the world origin when none of them exists. In practice all of them sit at `(0, 0)`, so this
    is a robustness measure rather than a shift.

    `z` is the **median resting height of the scene's movable objects**, not a table top. The
    tables are not comparable across suites (LIBERO-Object has no table body at all, the living
    room's is a mesh whose geom sizes describe its bounding volume rather than its surface), and
    what the relation words need is the surface the objects are *on*. The median is that surface
    by construction and survives an outlier -- a bowl already inside a drawer, a box standing on
    the cabinet -- which is exactly the case the text has to describe.
    """
    model, data = sim.model, sim.data
    x = y = 0.0
    for name in ("table",) + tuple(n for n in (model.body_id2name(i) or "" for i in range(model.nbody))
                                   if n.endswith("table")) + ("floor",):
        try:
            body = model.body_name2id(name)
        except Exception:  # noqa: BLE001 -- mujoco raises a bare ValueError for an unknown name
            continue
        x, y = float(data.body_xpos[body][0]), float(data.body_xpos[body][1])
        break
    heights = sorted(float(pos[2]) for pos in object_positions)
    surface = heights[len(heights) // 2] if heights else 0.0
    return x, y, surface


def _fixture_pos(sim, name: str, category: str, frame) -> tuple[float, float, float] | None:
    """A fixture's position, read off whichever MuJoCo body carries it.

    LIBERO gives an articulated fixture a `<name>_main` body and no observable
    (`bddl_base_domain.py:470-476` builds observables for `self.objects` only), so this is the
    only way to put a cabinet or a stove in the scene at all.
    """
    candidates = ["table", name] if category.endswith("table") else [f"{name}_main", name,
                                                                     f"{name}_base"]
    for body_name in candidates:
        try:
            body = sim.model.body_name2id(body_name)
        except Exception:  # noqa: BLE001 -- mujoco raises a bare ValueError for an unknown name
            continue
        return _to_frame(sim.data.body_xpos[body], frame)
    return None


def _anchors(sim, name: str, frame) -> list[list]:
    """An entity's BDDL regions, read off the MuJoCo sites LIBERO compiles them into.

    `wooden_cabinet_1_top_region` -> `["top", "region", [x, y, z]]`,
    `wooden_cabinet_1_top_side` -> `["top", "side", …]`, `flat_stove_1_cook_region` -> `["cook",
    "region", …]`. `<name>_default_site` is the body's own origin and carries no region, so it is
    skipped.
    """
    model, data = sim.model, sim.data
    out: list[list] = []
    for site in range(model.nsite):
        site_name = model.site_id2name(site) or ""
        if not site_name.startswith(f"{name}_"):
            continue
        suffix = site_name[len(name) + 1:]
        for kind in ("region", "side"):
            if suffix.endswith(f"_{kind}"):
                word = suffix[: -len(kind) - 1].replace("_", " ")
                out.append([word, kind, list(_to_frame(data.site_xpos[site], frame))])
                break
    return out


def _to_frame(pos, frame) -> tuple[float, float, float]:
    ox, oy, oz = frame
    return (round((float(pos[0]) - ox) * 100.0, 1),
            round((float(pos[1]) - oy) * 100.0, 1),
            round((float(pos[2]) - oz) * 100.0, 1))


def bddl_task(suite: str, task_index: int) -> BddlTask:
    """The suite's task `task_index`, parsed. Goes through the same benchmark API
    the environment adapter uses, so the task order is the one the env builds."""
    from libero.libero import benchmark, get_libero_path

    suites = benchmark.get_benchmark_dict()
    task = suites[suite]().get_task(task_index)
    path = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    return parse_bddl(path.read_text(encoding="utf-8"), suite=suite, task_index=task_index,
                      name=task.name)


def suite_size(suite: str) -> int:
    from libero.libero import benchmark

    return benchmark.get_benchmark_dict()[suite]().n_tasks


def settle(env, obs, steps: int = SETTLED_STEPS):
    """Run `steps` no-op control steps and return the observation they end at.

    The **same** action and the same count the platform's episode loop runs before a policy's
    first decision: `env.dummy_action()` (LIBERO's `[0, 0, 0, 0, 0, 0, -1]`, fingers open) for
    the protocol's wait steps. Separated out so the settling can be tested without a simulator: the
    only thing worth checking is that it steps the right action exactly the right number of times
    before anybody reads a pose.
    """
    # A plain list, not a numpy array: this module is stdlib-only at module scope (`_xyz`) and
    # every backend's `step` accepts a sequence.
    action = list(env.dummy_action())
    for _ in range(max(int(steps), 0)):
        obs = env.step(action).obs
    return obs


def scene_records(suite: str, task_index: int, *, n_init: int = INIT_STATES_PER_TASK,
                  render_size: int = 128, settled_steps: int = SETTLED_STEPS,
                  env_factory=None) -> list[dict]:
    """One record per init state: the task's entities with their positions in the table frame.

    Costs one environment construction (a few seconds) and `n_init` resets (milliseconds each),
    which is why the caller caches the result rather than rebuilding it per run. `env_factory()`
    is how the simulator gets in (`decision.env`); `None` builds the configured adapter.

    **The poses are read after `settled_steps` no-op steps**, which is the scene a server
    actually grounds against (`settle`). `settled_steps=0` is the unsettled capture the first
    cache was built with, kept so the two can be compared rather than argued about.
    """
    task = bddl_task(suite, task_index)
    factory = env_factory or (lambda: env_mod.make(suite, task_index, render_size=render_size))
    env = factory()
    try:
        total = env.num_init_states
        indices = sorted({int(round(i * (total - 1) / max(n_init - 1, 1))) for i in range(n_init)})
        out = []
        for init_index in indices:
            obs = settle(env, env.reset(init_index), settled_steps)
            sim = _mujoco_sim(env)
            poses = env.privileged(obs)
            frame = _scene_frame(sim, [pose["pos"] for pose in poses.values()])
            entities = []
            for name in env.object_names:
                entities.append({"name": name, "category": task.objects.get(name, name),
                                 "kind": "object",
                                 "pos": list(_to_frame(poses[name]["pos"], frame)),
                                 "anchors": _anchors(sim, name, frame)})
            for name, category in task.fixtures.items():
                pos = _fixture_pos(sim, name, category, frame)
                entities.append({"name": name, "category": category, "kind": "fixture",
                                 "pos": list(pos) if pos is not None else None,
                                 "anchors": _anchors(sim, name, frame)})
            out.append({"suite": suite, "task_index": task_index, "task_name": task.name,
                        "instruction": task.instruction, "init_index": init_index,
                        # What the scene was captured after, so a row can say which of the two
                        # scenes it describes rather than leaving it to a file's mtime.
                        "settled_steps": int(settled_steps),
                        "entities": entities})
        return out
    finally:
        env.close()


def _mujoco_sim(env):
    """The MuJoCo handle behind `env`, for the two anchor measurements the poses do not carry.

    The **one** place in this package that looks past `decision.env.DecisionEnv` at a simulator's
    own internals, and it is offline: the scene cache is built by hand, read from a file at run
    time, and nothing in the serving or harvesting path calls it. An adapter says how to get
    there with `mujoco_sim()`; the fallback is robosuite's own nesting, which is what every
    LIBERO wrapper in sight presents.
    """
    own = getattr(env, "mujoco_sim", None)
    if own is not None:
        return own()
    return env.env.env.sim


def _scene_worker(payload: str) -> str:  # pragma: no cover -- the parallel cache builder
    """One `(suite, task)`'s records, as JSON, in a fresh process with its own simulator."""
    blob = json.loads(payload)
    records = scene_records(blob["suite"], blob["task_index"], n_init=blob["n_init"],
                            settled_steps=blob["settled_steps"])
    return json.dumps(records)


def build_scene_cache(path: pathlib.Path = SETTLED_SCENE_CACHE, suites=SUITES, *,
                      n_init: int = INIT_STATES_PER_TASK, settled_steps: int = SETTLED_STEPS,
                      jobs: int = 1, log=print) -> pathlib.Path:
    """Fill the JSONL cache, skipping every `(suite, task)` already in it.

    Resumable by construction: a run that dies half way through `libero_90` is restarted by
    calling this again.

    **Settled by default** (`SETTLED_STEPS`), and into `SETTLED_SCENE_CACHE` by default, because
    a cache captured before the episode's no-op steps describes a scene no server ever grounds
    against. `settled_steps=0` reproduces the first cache.

    `jobs > 1` deals the tasks to that many processes, each owning its own simulator: 130 env
    constructions at a few seconds each is twenty minutes sequentially and about two on ten
    cores. The records are written by the parent in task order whatever the workers finish in,
    so the file does not depend on the scheduling.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["suite"], r["task_index"]) for r in load_scenes(path)} if path.is_file() else set()
    todo = [(suite, task_index) for suite in suites for task_index in range(suite_size(suite))
            if (suite, task_index) not in done]
    if not todo:
        return path

    def write(handle, suite, task_index, records) -> None:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        log(f"{suite}:{task_index} {records[0]['instruction']!r} "
            f"({len(records)} init states, settled {settled_steps})")

    with path.open("a", encoding="utf-8") as handle:
        if jobs <= 1:
            for suite, task_index in todo:
                write(handle, suite, task_index,
                      scene_records(suite, task_index, n_init=n_init,
                                    settled_steps=settled_steps))
            return path
        import concurrent.futures                                          # noqa: PLC0415
        import multiprocessing                                             # noqa: PLC0415

        payloads = [json.dumps({"suite": s, "task_index": t, "n_init": n_init,
                                "settled_steps": settled_steps}) for s, t in todo]
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(jobs, len(payloads)),
                                                    mp_context=context) as pool:
            for (suite, task_index), blob in zip(todo, pool.map(_scene_worker, payloads)):
                write(handle, suite, task_index, json.loads(blob))
    return path


def load_scenes(path: pathlib.Path = SCENE_CACHE) -> list[dict]:
    if not pathlib.Path(path).is_file():
        return []
    return [json.loads(line) for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


# ==================================================================== part 3: the rows

def base_word(name: str) -> str:
    """The noun a short id is built on: `akita_black_bowl_1` -> `bowl`, `wooden_cabinet_1` ->
    `cabinet`, `main_table` -> `table`.

    This is `robojev.state.short_id`'s stem -- that function keeps the last two
    `_`-separated segments (`bowl_1`), and the numbering is re-assigned per row by
    `assign_ids`.
    """
    parts = name.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return parts[-2]
    return parts[-1]


def assign_ids(entities, rng: random.Random) -> dict[str, str]:
    """`{entity name: short id}`, with the numbering **shuffled inside each noun group**.

    Two black bowls are `bowl_1` and `bowl_2`, and which of the two the BDDL calls
    `akita_black_bowl_1` is decided by `rng`. Without this the gold answer is `_1` in almost
    every LIBERO-Spatial row -- the BDDL names the object of interest first -- and a model can
    score 90 % by reading the digit instead of the sentence.
    """
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for entity in entities:
        groups[base_word(entity.name)].append(entity.name)
    out: dict[str, str] = {}
    for word, names in sorted(groups.items()):
        numbers = list(range(1, len(names) + 1))
        rng.shuffle(numbers)
        for name, number in zip(sorted(names), numbers):
            out[name] = f"{word}_{number}"
    return out


def _fmt(value: float) -> str:
    rounded = round(float(value), 1)
    return format(rounded if rounded != 0.0 else 0.0, "+.1f")


def _horizontal(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _containment(here, there, anchors) -> str | None:
    """`here` in or on `there`, in words, or `None`.

    An entity's own regions win when the object is sitting in one of them, because they are what
    the sentence names: "the bowl **in the top drawer** of the cabinet" and "the bowl **on** the
    stove" are both 13-15 cm from the fixture's body and would otherwise be related to nothing.
    A `_side` region is an outside surface ("on top of"); a `contain` region is an interior
    ("inside"); every other region is a named part ("in the top part of").

    Two anchors can both be in reach, and then the nearer one wins **unless they are within
    `ANCHOR_TIE_CM` of each other horizontally**, in which case the higher one does. That is the
    difference between the two LIBERO-Spatial cabinet tasks: drawer is *open*, so its
    site is 15 cm out from the cabinet's lid and horizontal distance settles it; drawer
    is *closed*, so the drawer site and the lid site sit on top of each other and only the height
    says whether the bowl is in the drawer or standing on the lid.
    """
    reachable = []
    for word, kind, pos in anchors:
        horizontal = _horizontal(here, pos)
        if horizontal > ANCHOR_XY_CM:
            continue
        rise = here[2] - pos[2]
        if not -INSIDE_DZ_CM <= rise <= ANCHOR_DZ_CM:
            continue
        reachable.append((horizontal, pos[2], word, kind))
    if reachable:
        nearest = min(horizontal for horizontal, _, _, _ in reachable)
        word, kind = max((height, -horizontal, word, kind)
                         for horizontal, height, word, kind in reachable
                         if horizontal - nearest <= ANCHOR_TIE_CM)[2:]
        if kind == "side" or word == "cook":
            return "on top of"
        if word.startswith("contain"):
            return "inside"
        return f"in the {word} part of"
    horizontal = _horizontal(here, there)
    if horizontal <= INSIDE_XY_CM and abs(here[2] - there[2]) <= INSIDE_DZ_CM:
        return "inside"
    if horizontal <= ON_TOP_XY_CM and here[2] - there[2] > ON_TOP_DZ_CM:
        return "on top of"
    return None


def relations(positions: dict[str, tuple[float, float, float]],
              anchors: dict[str, list] | None = None) -> dict[str, list[str]]:
    """`{short id: position}` -> `{short id: relation phrases}`, computed from the numbers alone.

    The order is the order a reader needs: containment first (`inside`, `on top of`), then
    `between A and B`, then the pairwise left/right/front/behind phrases nearest neighbour first,
    then `next to` and `nearest to`. `between` is the phrase LIBERO-Spatial's hardest instruction
    actually uses, and `on top of` / `next to` cover "the bowl on the cookie box" and "the bowl
    next to the ramekin"; every one of them is a comparison of two coordinates, so a rule can
    recover them and gate G3's principle 1 survives.
    """
    anchors = anchors or {}
    out: dict[str, list[str]] = {}
    names = sorted(positions)
    for sid in names:
        here = positions[sid]
        others = {n: positions[n] for n in names if n != sid}
        phrases: list[str] = []
        for other, pos in others.items():
            phrase = _containment(here, pos, anchors.get(other, ()))
            if phrase:
                phrases.append(f"{phrase} {other}")

        neighbours = list(others)
        betweens: list[tuple[float, str]] = []
        for i, a in enumerate(neighbours):
            for b in neighbours[i + 1:]:
                pa, pb = others[a], others[b]
                span = (pb[0] - pa[0], pb[1] - pa[1])
                length = math.hypot(*span)
                if length < BETWEEN_SPAN_CM:
                    continue
                unit = (span[0] / length, span[1] / length)
                rel = (here[0] - pa[0], here[1] - pa[1])
                along = rel[0] * unit[0] + rel[1] * unit[1]
                across = abs(rel[0] * unit[1] - rel[1] * unit[0])
                if 0.15 * length < along < 0.85 * length and across <= BETWEEN_OFF_CM:
                    betweens.append((across, f"between {a} and {b}"))
        phrases.extend(phrase for _, phrase in sorted(betweens)[:MAX_BETWEEN_RELATIONS])

        order = sorted(others, key=lambda n: _horizontal(here, others[n]))
        pairwise = 0
        for other in order:
            if pairwise >= MAX_NEIGHBOUR_RELATIONS:
                break
            pos = others[other]
            words = []
            if here[1] - pos[1] >= LEVEL_CM:
                words.append("right of")
            elif here[1] - pos[1] <= -LEVEL_CM:
                words.append("left of")
            if here[0] - pos[0] >= LEVEL_CM:
                words.append("in front of")
            elif here[0] - pos[0] <= -LEVEL_CM:
                words.append("behind")
            for word in words:
                phrases.append(f"{word} {other}")
                pairwise += 1
        if order:
            nearest = order[0]
            if _horizontal(here, others[nearest]) <= NEXT_TO_CM:
                phrases.append(f"next to {nearest}")
            phrases.append(f"nearest to {nearest}")
        out[sid] = phrases
    return out


def extremes(positions: dict[str, tuple[float, float, float]], groups: dict[str, list[str]]
             ) -> dict[str, list[str]]:
    """The absolute phrases: `the leftmost of the 3 bowls`, `the middle of the 3 bowls`.

    LIBERO says "the black bowl on the left", "the book on the right", "the black bowl in the
    middle" and "the black bowl at the back" -- those are not relations to a named neighbour,
    they are a rank among the identical objects of the same noun, and nothing in the pairwise
    phrases states a rank. Computed only inside a noun group of two or more, because "the
    leftmost of the 1 plate" says nothing.
    """
    out: dict[str, list[str]] = {sid: [] for sid in positions}
    for word, sids in groups.items():
        if len(sids) < 2:
            continue
        by_right = sorted(sids, key=lambda s: positions[s][1])
        by_front = sorted(sids, key=lambda s: positions[s][0])
        n = len(sids)
        out[by_right[0]].append(f"the leftmost of the {n} {word}s")
        out[by_right[-1]].append(f"the rightmost of the {n} {word}s")
        out[by_front[-1]].append(f"the frontmost of the {n} {word}s")
        out[by_front[0]].append(f"the backmost of the {n} {word}s")
        if n == 3:
            out[by_front[1]].append(f"the middle of the {n} {word}s front to back")
            out[by_right[1]].append(f"the middle of the {n} {word}s left to right")
    return out


FRAME_HEADER = (
    "Frame: the work surface, seen from the front of the workspace. x is front (+) and back (-), "
    "y is right (+) and left (-), z is height above the surface the objects rest on. Distances "
    "in centimetres from the centre of that surface."
)


#: What each BDDL category is *called* in the instructions. LIBERO's asset names and its language
#: disagree in a handful of places, and the disagreement is not cosmetic: the candidate line is
#: the only place the sentence's words can be matched to an object, so `porcelain_mug` written out
#: as "porcelain mug" makes "the **white** mug" match `white_yellow_mug` instead -- which is the
#: wrong bowl-equivalent, measured on three LIBERO-90 tasks. Only names that differ are listed;
#: everything else is its category with the underscores opened up.
CATEGORY_WORDS: dict[str, str] = {
    "akita_black_bowl": "black bowl",
    "porcelain_mug": "white mug",
    "white_yellow_mug": "yellow and white mug",
    "red_coffee_mug": "red mug",
    "chefmate_8_frypan": "frying pan",
    "new_salad_dressing": "salad dressing",
    "glazed_rim_porcelain_ramekin": "ramekin",
    "cookies": "cookie box",
    "wooden_two_layer_shelf": "cabinet shelf",
    "desk_caddy": "caddy",
    "wooden_tray": "tray",
}


def category_word(category: str) -> str:
    return CATEGORY_WORDS.get(category, category.replace("_", " "))


def candidate_line(sid: str, entity: Entity, phrases: list[str]) -> str:
    """One scene line: short id, category word, position, relations. The category word is the
    BDDL's own type with the underscores opened up -- it is what the instruction gives the reader
    ("the white bowl", "the red mug"), and nothing beyond it."""
    category = category_word(entity.category)
    if entity.pos is None:
        return f"  {sid} ({category}): position unknown."
    x, y, z = entity.pos
    where = f"x {_fmt(x)}, y {_fmt(y)}, z {_fmt(z)}"
    tail = f" -- {'; '.join(phrases)}." if phrases else "."
    return f"  {sid} ({category}) at {where}{tail}"


def grounding_state(instruction: str, entities: list[Entity], ids: dict[str, str]) -> str:
    """The grounding state: the sentence, then the scene, in the order a reader needs them."""
    positions = {ids[e.name]: tuple(e.pos) for e in entities if e.pos is not None}
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for entity in entities:
        if entity.pos is not None:
            groups[base_word(entity.name)].append(ids[entity.name])
    phrases = relations(positions, {ids[e.name]: e.anchors for e in entities})
    ranks = extremes(positions, groups)
    lines = [
        "Grounding. One decision before the arm moves: which object the instruction means.",
        FRAME_HEADER,
        f"Task: {instruction}",
        "Objects that can be picked up:",
    ]
    for entity in sorted((e for e in entities if e.kind == "object"), key=lambda e: ids[e.name]):
        sid = ids[entity.name]
        lines.append(candidate_line(sid, entity, ranks.get(sid, []) + phrases.get(sid, [])))
    fixtures = sorted((e for e in entities if e.kind == "fixture"), key=lambda e: ids[e.name])
    if fixtures:
        lines.append("Fixtures that cannot be picked up:")
        for entity in fixtures:
            sid = ids[entity.name]
            lines.append(candidate_line(sid, entity, ranks.get(sid, []) + phrases.get(sid, [])))
    return "\n".join(lines)


def candidate_sets(task: BddlTask, entities: list[Entity], ids: dict[str, str]
                   ) -> dict[str, list[str]]:
    """Which entities each question chooses between.

    `target` is the movable objects **plus the non-table fixtures**: "open the top drawer of the
    cabinet" and "turn on the stove" have a fixture as their target, and a candidate set that
    changed shape depending on the answer would leak the answer. `destination` is everything
    except the table, which is never an object-level destination (`destination_of`).
    """
    targets, destinations = [], []
    for entity in entities:
        sid = ids[entity.name]
        if _is_table(task, entity.name):
            continue
        targets.append(sid)
        destinations.append(sid)
    return {"target": sorted(targets), "destination": sorted(destinations)}


def question_block(qid: str, sids: list[str], entities: list[Entity], ids: dict[str, str],
                   phrases: dict[str, list[str]]) -> dict:
    """NanoJev's `{type, instructions, criteria}` for one scene-dependent question.

    The criteria repeat each candidate's own line, because a candidate path carries the state and
    **one** candidate: the model is scoring that line, and a path that named only `bowl_2` would
    make the two bowls' paths differ by a digit.
    """
    by_name = {ids[e.name]: e for e in entities}
    criteria = {}
    for sid in sids:
        entity = by_name[sid]
        criteria[sid] = candidate_line(sid, entity, phrases.get(sid, [])).strip()
    return {"type": "choice", "instructions": EPISODE_INSTRUCTIONS[qid], "criteria": criteria}


#: The scene label LIBERO puts on a task *name* and never on its `language`:
#: `KITCHEN_SCENE10_…`, `LIVING_ROOM_SCENE2_…`, `STUDY_SCENE1_…`. Anchored, upper-case and
#: `SCENE<digits>`-terminated, so a sentence that merely begins with a capitalised word is left
#: alone.
SCENE_LABEL = re.compile(r"^[A-Z0-9]+(?: [A-Z0-9]+)* SCENE\d+ ")


def served_instruction(task_name: str) -> str:
    """The sentence the **simulator** serves for a task, from the task's own name.

    LIBERO's `task.language` -- which `libero_utils.get_libero_env` returns and which becomes
    the environment's `instruction`, and therefore the sentence a served request carries -- is
    the BDDL file's stem with its underscores turned back into spaces, **minus the scene label
    the stem carries outside LIBERO-Spatial**. Derived here rather than imported so the row
    builder stays stdlib-only and needs no simulator.

    **It is not the BDDL's own `(:language …)`**, and that gap is why this function exists. Task 6
    declares `"Pick the akita black bowl next to the cookies box and place it on the plate"` and
    is served `"pick up the black bowl next to the cookie box and place it on the plate"`:
    different verb, different article, different noun for the same box, different case. A
    grounding head trained on one and asked the other is being asked a question it has not seen,
    and run 6's was systematically wrong on the served phrasing at p ~ 0.96.

    **And it is not the whole stem either**, which is what `SCENE_LABEL` removes. Measured
    against `benchmark.get_benchmark_dict()[suite]().get_task(i)`:

        libero_10  name  LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce…
                   language  'put both the alphabet soup and the tomato sauce in the basket'
        libero_90  name  KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet
                   language  'close the top drawer of the cabinet'
        libero_spatial  name == language, which is why the spatial tasks never showed it.

    Keeping `"KITCHEN SCENE3 "` on the front is a sentence the simulator never serves, and it is
    not harmless: it hides the lead verb that `split_clauses` reads, so every "turn on the stove
    and put the moka pot **on it**" row grounds to the wrong entity. Measured over the cached
    scenes, the label costs the served phrasing 25 of 610 `target` and 25 of 520 `destination`;
    without it the rule is 610/610 and 520/520.
    """
    return SCENE_LABEL.sub("", str(task_name).replace("_", " ").strip()).strip()


#: `metadata.phrasing` of a row: whose sentence its `Task:` line carries.
PHRASINGS: tuple[str, ...] = ("bddl", "served")


def phrasing_of(record: dict, phrasing: str) -> str:
    """The sentence a row renders under `phrasing`, for one scene record."""
    if phrasing == "served":
        return served_instruction(record.get("task_name") or "") or record["instruction"]
    return record["instruction"]


def scene_keys(record: dict) -> tuple[str, ...]:
    """Every instruction key one scene can be written under -- both phrasings, de-duplicated.

    The unit the split is drawn over. Two phrasings of one task are one held-out thing: a model
    that saw "pick up the black bowl next to the cookie box" in train has seen the sentence, and
    scoring it on "Pick the akita black bowl next to the cookies box" in test would be measuring
    paraphrase robustness while calling it grounding.
    """
    keys = [instruction_key(phrasing_of(record, p)) for p in PHRASINGS]
    return tuple(dict.fromkeys(keys))


def instruction_key(instruction: str) -> str:
    """The split is by *instruction*, so the key is the sentence with its case and spacing
    normalised -- `libero_spatial` and `libero_90` share sentences ("put the black bowl on the
    plate"), and a row of one in train with a row of the other in test is leakage."""
    return re.sub(r"\s+", " ", instruction.strip().lower())


def split_map(scenes: list[dict], *, seed: int = SPLIT_SEED,
              fraction: float = TEST_FRACTION, key_of=None) -> dict[str, str]:
    """`{instruction key: "train" | "test"}`.

    Every LIBERO-Spatial task 8 and 9 sentence is test by construction; the rest of the test set
    is a seeded `fraction` of the remaining sentences, drawn over the sentences and not over the
    rows, so the five init states of one task never straddle the split.

    `key_of(record) -> keys` makes one scene contribute **several** keys that must land in the
    same split -- the two phrasings of one task (`scene_keys`). Keys that co-occur in any scene
    are merged into one group and the draw is over the groups, so a sentence cannot be in train
    under one wording and in test under another. Left `None` the behaviour is exactly the
    single-key one above, which is what the harvested `grounding.jsonl` on disk was written with
    and what a re-run must reproduce byte for byte.
    """
    if key_of is None:
        forced = {instruction_key(s["instruction"]) for s in scenes
                  if s["suite"] == "libero_spatial" and s["task_index"] in HELD_OUT_SPATIAL_TASKS}
        others = sorted({instruction_key(s["instruction"]) for s in scenes} - forced)
        rng = random.Random(seed)
        rng.shuffle(others)
        held = set(others[:int(round(fraction * len(others)))])
        return {key: ("test" if key in forced | held else "train")
                for key in {instruction_key(s["instruction"]) for s in scenes}}

    parent: dict[str, str] = {}

    def find(key: str) -> str:
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    forced_groups: set[str] = set()
    per_scene: list[tuple[dict, tuple[str, ...]]] = [(s, tuple(key_of(s))) for s in scenes]
    for _scene, keys in per_scene:
        for key in keys:
            find(key)
        for other in keys[1:]:
            union(keys[0], other)
    for scene, keys in per_scene:
        if scene["suite"] == "libero_spatial" and scene["task_index"] in HELD_OUT_SPATIAL_TASKS:
            forced_groups.update(find(key) for key in keys)
    groups = sorted({find(key) for key in parent})
    others = sorted(set(groups) - forced_groups)
    rng = random.Random(seed)
    rng.shuffle(others)
    held = forced_groups | set(others[:int(round(fraction * len(others)))])
    return {key: ("test" if find(key) in held else "train") for key in parent}


def is_frame(entity: Entity) -> bool:
    """The table (and, in the study scenes, the floor) is the frame rather than a candidate.

    It is dropped from the scene block as well as from both candidate sets. Keeping it would cost
    a line, and worse: a table's body sits at the centre of everything, so it turns up as "between
    A and B" and "nearest to" for half the scene and dilutes the phrases that do resolve
    something.
    """
    return entity.category.endswith("table") or entity.category == "floor"


def entities_of(record: dict, *, keep_frame: bool = False) -> list[Entity]:
    entities = [Entity(e["name"], e["category"], e["kind"],
                       tuple(e["pos"]) if e["pos"] is not None else None,
                       tuple((word, kind, tuple(pos))
                             for word, kind, pos in e.get("anchors", ())))
                for e in record["entities"]]
    return entities if keep_frame else [e for e in entities if not is_frame(e)]


def build_rows(scenes: list[dict], *, seed: int = SPLIT_SEED, splits: dict[str, str] | None = None,
               task_lookup=None, phrasing: str = "bddl", id_suffix: str = ""
               ) -> tuple[list[dict], dict]:
    """The training rows, and a report of what was kept and dropped.

    One row per `(task, init state)`, carrying whichever of `target` and `destination` the BDDL
    can label unambiguously. A row with neither is not written at all.

    `task_lookup` maps a scene record to its `BddlTask`; the default re-reads the real `.bddl`
    file, and a test passes its own so the row builder can run without LIBERO installed.

    `phrasing` picks which sentence the `Task:` line carries (`PHRASINGS`): the BDDL's own
    `(:language …)`, or the one the simulator serves (`served_instruction`). **The split is
    looked up under the BDDL key either way**, so both phrasings of a scene land in the same
    split whatever `splits` was built from -- a served sentence in test whose BDDL twin is in
    train would be the leakage this is arranged to prevent.

    `id_suffix` distinguishes repeated copies of one scene (`…:g3`). The copies differ in their
    candidate numbering and ordering, because `seed` reaches `assign_ids`, and a model that has
    seen the same scene under four numberings cannot answer it by memorising a digit.
    """
    if phrasing not in PHRASINGS:
        raise ValueError(f"phrasing must be one of {PHRASINGS}, not {phrasing!r}")
    splits = splits if splits is not None else split_map(scenes, seed=seed)
    task_lookup = task_lookup if task_lookup is not None else bddl_task_from_record
    rows: list[dict] = []
    report = {"kinds": collections.Counter(), "dropped_target": collections.Counter(),
              "dropped_destination": collections.Counter(),
              "rows_per_suite": collections.Counter(), "tasks_per_suite": collections.Counter(),
              "questions": collections.Counter(), "split_rows": collections.Counter(),
              "split_instructions": collections.Counter()}
    seen_tasks: set[tuple[str, int]] = set()
    for record in scenes:
        suite, task_index = record["suite"], record["task_index"]
        task = task_lookup(record)
        target, target_why = target_of(task)
        destination, destination_why = destination_of(task)
        if (suite, task_index) not in seen_tasks:
            seen_tasks.add((suite, task_index))
            report["kinds"][task_kind(task)] += 1
            report["tasks_per_suite"][suite] += 1
            if target is None:
                report["dropped_target"][target_why] += 1
            if destination is None and "place" in task_kind(task):
                report["dropped_destination"][destination_why] += 1
        if target is None and destination is None:
            continue

        entities = entities_of(record)
        rng = random.Random(f"{seed}:{suite}:{task_index}:{record['init_index']}")
        ids = assign_ids(entities, rng)
        positions = {ids[e.name]: tuple(e.pos) for e in entities if e.pos is not None}
        groups: dict[str, list[str]] = collections.defaultdict(list)
        for entity in entities:
            if entity.pos is not None:
                groups[base_word(entity.name)].append(ids[entity.name])
        ranks = extremes(positions, groups)
        phrases = {sid: ranks.get(sid, []) + rel
                   for sid, rel in relations(
                       positions, {ids[e.name]: e.anchors for e in entities}).items()}
        sets = candidate_sets(task, entities, ids)

        questions, gold, gold_probs = {}, {}, {}
        for qid, answer in (("target", target), ("destination", destination)):
            if answer is None or ids.get(answer) not in sets[qid]:
                continue
            questions[qid] = question_block(qid, sets[qid], entities, ids, phrases)
            gold[qid] = ids[answer]
            gold_probs[qid] = {sid: (1.0 if sid == ids[answer] else 0.0) for sid in sets[qid]}
            report["questions"][qid] += 1
        if not questions:
            continue

        key = instruction_key(record["instruction"])
        split = splits[key]
        sentence = phrasing_of(record, phrasing)
        row_id = f"{suite}:{task_index}:{record['init_index']}:grounding{id_suffix}"
        rows.append({
            "id": row_id,
            "state_id": row_id,
            "family_id": suite,
            "split": split,
            "state": grounding_state(sentence, entities, ids),
            "questions": questions,
            "gold": gold,
            "gold_probs": gold_probs,
            "gold_probs_kind": {qid: PROBS_KIND for qid in questions},
            "gold_label_kind": {qid: LABEL_KIND for qid in questions},
            "metadata": {
                "source_group_id": key,
                "suite": suite,
                "task_index": task_index,
                "task_name": record["task_name"],
                "init_index": record["init_index"],
                # How many no-op steps the scene had settled for when its poses were read. A
                # row of an unsettled scene describes a scene no server grounds against.
                "settled_steps": record.get("settled_steps"),
                "instruction": sentence,
                # Which sentence this copy carries, and the other one, so a file says outright
                # whether a row was written under the wording a server asks with.
                "phrasing": phrasing,
                "bddl_instruction": record["instruction"],
                "served_instruction": phrasing_of(record, "served"),
                "task_kind": task_kind(task),
                "ids": ids,
                "gold_names": {qid: name for qid, name in
                               (("target", target), ("destination", destination))
                               if name is not None and qid in questions},
            },
        })
        report["rows_per_suite"][suite] += 1
        report["split_rows"][split] += 1
    report["split_instructions"] = collections.Counter(splits.values())
    return rows, report


#: The BDDL is re-parsed from the file rather than cached beside the scene, so a fix to the parser
#: takes effect without a 20-minute simulator re-run.
_BDDL_CACHE: dict[tuple[str, int], BddlTask] = {}


def bddl_task_from_record(record: dict) -> BddlTask:
    key = (record["suite"], record["task_index"])
    if key not in _BDDL_CACHE:
        _BDDL_CACHE[key] = bddl_task(*key)
    return _BDDL_CACHE[key]


# ==================================================================== gate G5: the two probes

#: Words an instruction uses for a noun the BDDL spells differently. The left side is what a
#: sentence says, the right side is `base_word` of the entity it means.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "drawer": ("cabinet", "shelf"), "drawers": ("cabinet", "shelf"),
    "cabinet": ("cabinet", "shelf"),  # KITCHEN_SCENE9's "cabinet" is a wooden_two_layer_shelf
    "shelf": ("shelf", "cabinet"), "cookie": ("cookies",), "cookies": ("cookies",),
    "pan": ("frypan",), "frying": ("frypan",), "moka": ("pot",), "pot": ("pot",),
    "rack": ("rack",), "caddy": ("caddy",), "stove": ("stove",), "microwave": ("microwave",),
    "basket": ("basket",), "tray": ("tray",), "bowl": ("bowl",), "bowls": ("bowl",),
    "plate": ("plate",), "plates": ("plate",), "mug": ("mug",), "mugs": ("mug",),
    "book": ("book",), "books": ("book",), "ramekin": ("ramekin",), "butter": ("butter",),
    "milk": ("milk",), "ketchup": ("ketchup",), "juice": ("juice",), "soup": ("soup",),
    "cheese": ("cheese",), "pudding": ("pudding",), "bottle": ("bottle",),
    "dressing": ("dressing",), "sauce": ("sauce",), "table": ("table",),
}


def nouns_for(word: str) -> tuple[str, ...]:
    """The entity nouns a sentence's word can mean. One word can mean two: LIBERO's KITCHEN_SCENE9
    says "cabinet" for a `wooden_two_layer_shelf`, and a scene holding both leaves the ambiguity
    for the spatial phrases to settle -- which is the honest behaviour, not a bug to hide."""
    return SYNONYMS.get(word, (word,))


#: The words LIBERO uses for a named part of a fixture: "the **top** drawer of the cabinet", "the
#: **middle** layer of the drawer". They are the BDDL's own region words, which is why they are
#: the words `_containment` writes into the scene text.
PART_WORDS: frozenset[str] = frozenset({"top", "middle", "bottom", "front", "back", "left",
                                        "right", "cook"})

#: Where a sentence may be cut into "what to move" and "where it goes". Every one of them is
#: tried (`_phrase_splits`), because neither the first nor the last is right in general.
_SPLIT_WORDS = ("and put", "and place", " on ", " in ", " into ", " onto ", " to ", " under ",
                " inside ", " at ")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def _strip_verb(text: str) -> str:
    """The sentence without its leading action verb, which names no object."""
    text = text.lower().strip()
    for verb in ("pick up ", "pick ", "put ", "place ", "push ", "stack ", "move "):
        if text.startswith(verb):
            return text[len(verb):]
    return text


#: The join between "what to pick up" and "where to put it", when the sentence has one: LIBERO
#: writes "… **and place it** on the plate" / "… **and put it** on the plate" / "… **then place**
#: …". Everything before it grounds the target and everything after it grounds the destination,
#: and neither may borrow a noun from the other.
_PLACING_JOIN = re.compile(r"\b(?:and|then)\s+(?:place|put|set|drop|stack)\b")

#: What a placing clause opens with before it names the destination: the pronoun for the thing
#: just picked up, then the preposition.
_PLACING_LEAD = re.compile(
    r"^(?:it|them)\b|^(?:on|in|into|onto|to|under|inside|at|the top of|top of)\b")


def placing_clauses(sentence: str) -> tuple[str, str] | None:
    """`(the picking clause, the placing clause)`, or `None` when the sentence has no join.

    **This is what keeps "on the cookie box" out of the destination.** LIBERO-Spatial task 3 is
    "pick up the black bowl **on the cookie box** and place it **on the plate**": two `on`
    phrases, one describing *which bowl* and one describing *where it goes*, and a resolver that
    offers every cut of the whole sentence will happily answer `destination` with the cookie box
    -- which it did, closed-loop, whenever the scene text called the bowl "inside cookies_1"
    rather than "on top of cookies_1" (the two differ by where the objects settle, so the same
    task grounds differently before and after LIBERO's ten settling steps).

    A sentence without a join ("put the black bowl on the plate", "open the top drawer of the
    cabinet") is left to `_phrase_splits`, which is the whole reason that function tries every
    cut: there, the cut *is* the question.
    """
    text = _strip_verb(sentence)
    match = _PLACING_JOIN.search(text)
    if match is None:
        return None
    pick, place = text[:match.start()].strip(), text[match.end():].strip()
    while True:
        stripped = _PLACING_LEAD.sub("", place).strip()
        if stripped == place:
            break
        place = stripped
    return (pick, place) if pick and place else None


def _phrase_splits(instruction: str):
    """Every way of cutting a sentence into a target phrase and a destination phrase.

    Non-greedy "first preposition" is wrong on LIBERO ("put the black bowl **at the back** on the
    plate" would make the target "the black bowl" and the destination "the back on the plate"),
    and greedy is wrong on "put the bowl in the top drawer of the cabinet". So every split is
    offered and the resolver keeps the first one that resolves both halves.
    """
    text = instruction.lower().strip()
    for verb in ("pick up ", "pick ", "put ", "place ", "push ", "stack ", "move "):
        if text.startswith(verb):
            text = text[len(verb):]
            break
    points = sorted({m.start() for word in _SPLIT_WORDS for m in re.finditer(word, text)})
    for point in points:
        head, tail = text[:point], text[point:]
        tail = re.sub(r"^(and put|and place|on|in|into|onto|to|under|inside|at)\b", "", tail).strip()
        tail = re.sub(r"^(it|them)\b", "", tail).strip()
        tail = re.sub(r"^(in|on|into|onto|to|under|inside|the top of|top of)\b", "", tail).strip()
        if head.strip() and tail:
            yield head.strip(), tail
    yield text, ""


def _matching(phrase: str, sids: list[str], ids_to_entity: dict[str, Entity]) -> list[str]:
    """The candidates whose noun the phrase names, narrowed by their category words.

    Two stages, because both are things the sentence genuinely gives: the noun (`bowl`, `mug`,
    `cabinet`) picks the group, and the adjectives (`white`, `red`, `black`) pick inside it. What
    it deliberately does not use is the short id's digit.
    """
    words = set(_tokens(phrase))
    nouns = {n for w in words for n in nouns_for(w)}
    hits = [sid for sid in sids if base_word(ids_to_entity[sid].name) in nouns]
    if not hits:
        return []
    scored = []
    for sid in hits:
        category = set(_tokens(category_word(ids_to_entity[sid].category)))
        # Words of the category the phrase says, minus words it does not: "the white mug" fits
        # `white mug` exactly and `yellow and white mug` with two words left over, and without the
        # second term the two score the same.
        scored.append(((len(category & words), -len(category - words)), sid))
    best = max(score for score, _ in scored)
    return [sid for score, sid in scored if score == best]


def _relation_cut(text: str, keys) -> tuple[str, str] | None:
    """`(what is before the relation word, what is after it)`, at its **first** occurrence.

    The sentence puts the candidate on one side and its neighbour on the other, and both halves
    are needed: the left one says what is being grounded ("the black bowl"), the right one says
    what it is beside ("the cookie box"). Returns `None` when the phrase does not use this
    relation at all.
    """
    padded = text + " "
    found = [(padded.index(key), key) for key in keys if key in padded]
    if not found:
        return None
    at, key = min(found)
    return padded[:at].strip(), padded[at + len(key):].strip()


def resolve(phrase: str, sids: list[str], ids_to_entity: dict[str, Entity],
            positions: dict[str, tuple[float, float, float]],
            phrases: dict[str, list[str]], *, strict: bool = False) -> str | None:
    """The rule-based resolver: a phrase and a scene in, one candidate out.

    This is gate G5(a). It measures whether the **text** is sufficient -- if a rule that reads
    only the sentence and the relation words cannot pick the right object, no model reading the
    same string can either (§2 principle 1), and the state is what has to change.

    `strict` returns `None` instead of guessing when the phrase leaves several candidates. That
    is what `resolve_row` uses to *choose the split*: "pick the bowl **on** the cookies box and
    place it **on** the plate" has three places to cut, and the right one is the cut whose two
    halves both resolve -- a guess on the first cut would silently make "the cookies box" the
    destination.
    """
    hits = _matching(phrase, sids, ids_to_entity)
    if len(hits) == 1:
        return hits[0]
    if not hits:
        return None
    words = _tokens(phrase)
    text = " ".join(words)

    # "between the plate and the ramekin" -- the only phrase that needs two named neighbours.
    match = re.search(r"between the (\w+[\w ]*?) and the (\w+)", text)
    if match:
        groups = (nouns_for(match.group(1).split()[-1]), nouns_for(match.group(2)))
        for sid in hits:
            for phrase_text in phrases.get(sid, []):
                if not phrase_text.startswith("between "):
                    continue
                named = {base_word(p) for p in phrase_text[len("between "):].split(" and ")}
                if all(set(group) & named for group in groups):
                    return sid

    # "on the cookie box", "next to the ramekin", "in the top drawer of the cabinet": one named
    # neighbour and the relation word that goes with it. `in` matches both "inside X" and the
    # named-part phrase "in the top part of X", and when the sentence names the part ("the **top**
    # drawer", "the **middle** layer") the part has to match too -- that is the only thing telling
    # a bowl in the cabinet's top drawer from a bowl standing on the cabinet's lid.
    # Each entry's readings are tried in order, and `on` has two: a bowl resting on a **flat**
    # box has its centre within `INSIDE_XY_CM` and `INSIDE_DZ_CM` of the box's own, so
    # `_containment` writes "inside cookies_1" for the arrangement the sentence calls "on the
    # cookie box" -- and which of the two it writes depends on where the scene settles, so the
    # same task says one before LIBERO's ten settling steps and the other after. The sentence's
    # "on" means either; the *named neighbour* is what disambiguates, and the stricter reading is
    # still tried first so "on the cabinet" never matches a bowl inside its drawer.
    for readings, keys in ((((("on top of",), ("on top of", "inside"))), ("on ", "on top of ")),
                           ((("inside", "in the"),), ("in ", "inside ")),
                           ((("next to",),), ("next to ", "beside "))):
        cut = _relation_cut(text, keys)
        if cut is None:
            continue
        before, after = cut
        # **Which side of the relation each noun is on.** "the black **bowl** next to the
        # **cookie box**" names the candidate on the left and its neighbour on the right, and
        # reading the scene line "next to bowl_1" from the wrong side answers with the box. That
        # is not hypothetical: it is LIBERO-Spatial task 6 under the phrasing the *environment*
        # gives ("the cookie box" matches the `cookie box` category word exactly, so `_matching`
        # keeps the box as a candidate and the box is "next to bowl_1" too).
        near = _matching(before, sids, ids_to_entity) or hits
        parts = [w for w in words if w in PART_WORDS]
        for relations_wanted in readings:
            for word in _tokens(after):
                if word not in SYNONYMS:
                    continue
                for wanted_parts in ([parts] if parts else []) + [[]]:
                    for sid in near:
                        for phrase_text in phrases.get(sid, []):
                            if not phrase_text.startswith(relations_wanted):
                                continue
                            if base_word(phrase_text.split()[-1]) not in nouns_for(word):
                                continue
                            if wanted_parts and not any(f" {part} part of" in phrase_text
                                                        for part in wanted_parts):
                                continue
                            return sid

    # "on the left", "at the back", "in the middle": a rank among the identical objects.
    ranked = [sid for sid in hits if sid in positions]
    if ranked:
        if "left" in words:
            return min(ranked, key=lambda s: positions[s][1])
        if "right" in words:
            return max(ranked, key=lambda s: positions[s][1])
        if "front" in words:
            return max(ranked, key=lambda s: positions[s][0])
        if "back" in words or "behind" in words:
            return min(ranked, key=lambda s: positions[s][0])
        if "middle" in words or "center" in words or "centre" in words:
            if "table" in words or "center" in words or "centre" in words:
                return min(ranked, key=lambda s: math.hypot(*positions[s][:2]))
            # Along whichever axis they are spread over: LIBERO stands three bowls front to
            # back in one scene and three books left to right in another, and "in the middle"
            # means the middle of that row either way.
            spread_x = max(positions[s][0] for s in ranked) - min(positions[s][0] for s in ranked)
            spread_y = max(positions[s][1] for s in ranked) - min(positions[s][1] for s in ranked)
            axis = 0 if spread_x >= spread_y else 1
            in_order = sorted(ranked, key=lambda s: positions[s][axis])
            return in_order[len(in_order) // 2]
    return None if strict else hits[0]


#: A leading clause that changes a fixture rather than carrying anything: "**turn on the stove**
#: and put the moka pot on it", "**close the top drawer of the cabinet** and put the black bowl on
#: top of it". The object being grounded is in the *second* clause, and the first clause's fixture
#: is what the second clause's "it" points at.
_LEAD_VERBS = ("turn on", "turn off", "open", "close")
_CARRY_VERBS = ("put ", "place ", "pick ", "stack ", "push ")


def split_clauses(instruction: str) -> tuple[str, str]:
    """`(the clause that grounds the target, the leading clause it follows)`.

    Eight of the 130 instructions are two actions joined by "and", and in every one of them the
    first action is on a fixture ("turn on the stove and put the moka pot on it"). Grounding the
    first clause would answer `target` with the stove, which is why this is a parser rule and not
    a scoring heuristic -- it decides *which sentence* is being grounded, not which object.
    """
    text = instruction.lower().strip()
    for match in re.finditer(r"\band\b", text):
        head, tail = text[:match.start()].strip(), text[match.end():].strip()
        if (head.startswith(_LEAD_VERBS) and tail.startswith(_CARRY_VERBS)):
            return tail, head
    return text, ""


def resolve_scene(instruction: str, qid: str, sids: list[str],
                  entities: dict[str, Entity], positions: dict[str, tuple[float, float, float]],
                  phrases: dict[str, list[str]]) -> str | None:
    """The whole rule, over a scene however it was obtained: a training row or a live request.

    Split out of `resolve_row` so that the serving path and the row path are **the same three
    passes over the same numbers** rather than two implementations that agree today
    (`test_grounding_v2.py` asserts they resolve identically on the cached scenes). This is the
    rule gate G5(a) measured: 610/610 `target` and 515/520 `destination` over every LIBERO suite.
    """
    sentence, lead = split_clauses(instruction)
    # "…and put the moka pot **on it**", "…and put the bowl **inside**": the destination is the
    # fixture the dropped clause named.
    if lead and qid == "destination" and re.search(r"\b(on|in|inside|into|of) it\b|\binside$", sentence):
        anaphor = resolve(lead, sids, entities, positions, phrases, strict=True)
        if anaphor is not None:
            return anaphor
    # **When the sentence says where the join is, it is not the resolver's to guess.** A
    # "… and place it …" sentence has one clause per question, and each question reads only its
    # own: the picking clause may describe the target by what it is standing on ("the black bowl
    # on the cookie box") without that ever becoming the destination.
    clauses = placing_clauses(sentence)
    if clauses is not None:
        phrase = clauses[0] if qid == "target" else clauses[1]
        hit = resolve(phrase, sids, entities, positions, phrases, strict=True)
        # Then, and only inside that clause, the guess `resolve` makes without `strict` -- gate
        # G5 is a measurement, so an unresolvable phrase is counted wrong rather than skipped.
        return hit if hit is not None else resolve(phrase, sids, entities, positions, phrases)
    splits = list(_phrase_splits(sentence))
    # Pass one: the cut whose *both* halves resolve without a guess (see `resolve`'s `strict`).
    for head, tail in splits:
        if not tail:
            continue
        target = resolve(head, sids, entities, positions, phrases, strict=True)
        destination = resolve(tail, sids, entities, positions, phrases, strict=True)
        if target is not None and destination is not None:
            return target if qid == "target" else destination
    # Pass two: no cut resolves cleanly, so take the first that resolves the half being asked for.
    for head, tail in splits:
        phrase = head if qid == "target" else tail
        if phrase:
            hit = resolve(phrase, sids, entities, positions, phrases, strict=True)
            if hit is not None:
                return hit
    # Pass three: guess, and be counted wrong if the guess is wrong -- G5 is a measurement.
    for head, tail in splits:
        phrase = head if qid == "target" else tail
        if phrase:
            hit = resolve(phrase, sids, entities, positions, phrases)
            if hit is not None:
                return hit
    return None


def resolve_row(row: dict, qid: str) -> str | None:
    """`resolve_scene` against one built row, using only what the row's own text carries."""
    entities = row["metadata"]["_entities"]
    positions = {sid: e.pos for sid, e in entities.items() if e.pos is not None}
    return resolve_scene(row["metadata"]["instruction"], qid,
                         sorted(row["questions"][qid]["criteria"]), entities, positions,
                         row["metadata"]["_phrases"])


#: One candidate line of a grounding state, read back: `  bowl_1 (black bowl) at x +1.2, y -15.0,
#: z +0.0 -- the leftmost of the 2 bowls; between plate_1 and ramekin_1.`
_CANDIDATE_RE = re.compile(
    r"^\s+(?P<sid>\S+) \((?P<category>[^)]*)\)"
    r"(?:: position unknown\.| at x (?P<x>[-+][\d.]+), y (?P<y>[-+][\d.]+), z (?P<z>[-+][\d.]+)"
    r"(?: -- (?P<phrases>.*?))?\.)$"
)


def scene_from_request(request: dict) -> tuple[str, dict[str, Entity], dict[str, list[str]]]:
    """`(instruction, {sid: Entity}, {sid: phrases})` read back out of a grounding request.

    **Out of the text, deliberately.** The rule's whole claim is that the state string is
    sufficient (§2 principle 1, gate G3's principle applied to grounding), so the resolver a
    server runs reads exactly what the model reads and nothing beside it -- no privileged dict, no
    BDDL, no second source of positions that could disagree with the sentence the model was given.
    Every line it parses is one `candidate_line` wrote.

    The reconstructed `Entity` carries the **short id** as its name and the already-worded
    category, which is what `_matching` needs and all it uses: `base_word("bowl_1")` is `bowl`,
    and `category_word("black bowl")` is itself.
    """
    state = request["states"][0]["state"]
    instruction, entities, phrases = "", {}, {}
    kind = "object"
    for line in state.splitlines():
        if line.startswith("Task: "):
            instruction = line[len("Task: "):].strip()
            continue
        if line.startswith("Fixtures"):
            kind = "fixture"
            continue
        if line.startswith("Objects"):
            kind = "object"
            continue
        match = _CANDIDATE_RE.match(line)
        if not match:
            continue
        sid = match.group("sid")
        position = (None if match.group("x") is None else
                    (float(match.group("x")), float(match.group("y")), float(match.group("z"))))
        entities[sid] = Entity(sid, match.group("category"), kind, position)
        phrases[sid] = [p.strip() for p in (match.group("phrases") or "").split(";") if p.strip()]
    return instruction, entities, phrases


def resolve_request(request: dict, qids=None) -> dict[str, str | None]:
    """`{qid: short id or None}` for a grounding request built from a live observation.

    The serving-time entry to the rule `resolve_row` is for training rows, and the reason a server
    has one: a checkpoint's grounding **forward** can be confidently wrong (run 6 committed
    `bowl_2` at p 0.957 on a scene whose `bowl_1` line says "between plate_1 and ramekin_1"), so a
    probability threshold is not a guard. The rule reads the relation words the sentence names and
    scored 610/610 / 515/520 in gate G5(a), which is what `robojev/policy.py --ground
    rule` commits instead.

    `None` for a question the rule cannot settle -- never a guess dressed as an answer; the server
    decides what to do with that.
    """
    instruction, entities, phrases = scene_from_request(request)
    positions = {sid: e.pos for sid, e in entities.items() if e.pos is not None}
    questions = request["states"][0]["questions"]
    return {qid: resolve_scene(instruction, qid, sorted(questions[qid]["criteria"]),
                               entities, positions, phrases)
            for qid in (qids if qids is not None else questions)}


def attach_scene(rows: list[dict], scenes: list[dict]) -> None:
    """Put the parsed entities and relation phrases back on each row, under private keys.

    The probes need the numbers the state was written from; the rows on disk carry only the text,
    which is exactly what the model gets. These keys start with `_` and are stripped by
    `write_rows`, so nothing that reaches the trainer ever contains them.
    """
    by_id = {(s["suite"], s["task_index"], s["init_index"]): s for s in scenes}
    for row in rows:
        meta = row["metadata"]
        record = by_id[(meta["suite"], meta["task_index"], meta["init_index"])]
        entities = {meta["ids"][e.name]: e for e in entities_of(record)}
        positions = {sid: e.pos for sid, e in entities.items() if e.pos is not None}
        groups: dict[str, list[str]] = collections.defaultdict(list)
        for sid, entity in entities.items():
            if entity.pos is not None:
                groups[base_word(entity.name)].append(sid)
        ranks = extremes(positions, groups)
        meta["_entities"] = entities
        meta["_phrases"] = {sid: ranks.get(sid, []) + rel
                            for sid, rel in relations(
                                positions,
                                {sid: e.anchors for sid, e in entities.items()}).items()}


# ---------------------------------------------------------------- G5(b): the linear probe

#: The per-candidate features the bag of words is crossed with. Each one is a number a rule could
#: read off the scene lines; the cross with the sentence's tokens is what lets a *linear* model
#: learn that "left" goes with `leftmost` and "between" with `between_two_named`.
CANDIDATE_FEATURES: tuple[str, ...] = (
    "noun_match", "first_noun", "last_noun", "noun_position", "category_overlap",
    "leftmost", "rightmost", "frontmost", "backmost",
    "middle", "between_two_named", "on_top_of_named", "inside_named", "named_part_of_named",
    "next_to_named",
    "nearest_named", "is_fixture", "peers",
)


def candidate_vector(row: dict, qid: str, sid: str) -> dict[str, float]:
    entities: dict[str, Entity] = row["metadata"]["_entities"]
    phrases: dict[str, list[str]] = row["metadata"]["_phrases"]
    entity = entities[sid]
    words = set(_tokens(row["metadata"]["instruction"]))
    nouns = {n for w in words for n in nouns_for(w)}
    named = {s for s, e in entities.items() if base_word(e.name) in nouns}
    category = set(_tokens(entity.category))
    mine = phrases.get(sid, [])

    def rel(prefix) -> float:
        return float(any(p.startswith(prefix) and p.split()[-1] in named for p in mine))

    part = 0.0
    for phrase in mine:
        if phrase.startswith("in the") and phrase.split()[-1] in named:
            said = {w for w in words if w in PART_WORDS}
            part = max(part, 1.0 + float(any(f" {w} part of" in phrase for w in said)))

    between = 0.0
    for phrase in mine:
        if phrase.startswith("between "):
            pair = set(phrase[len("between "):].split(" and "))
            between = max(between, float(len(pair & named) == 2))
    peers = sum(1 for e in entities.values() if base_word(e.name) == base_word(entity.name))
    # Where in the sentence this candidate's noun is first said. "Pick the **cream cheese** and
    # place it in the **basket**" names both, so `noun_match` alone cannot tell the target from
    # the destination; which noun comes first can, and it is a property of the (sentence,
    # candidate) pair rather than a parse of the sentence.
    order = _tokens(row["metadata"]["instruction"])
    where = {s: next((i for i, w in enumerate(order) if base_word(e.name) in nouns_for(w)),
                     None)
             for s, e in entities.items()}
    said = [i for i in where.values() if i is not None]
    mine_at = where.get(sid)
    return {
        "noun_match": float(base_word(entity.name) in nouns),
        "first_noun": float(mine_at is not None and said and mine_at == min(said)),
        "last_noun": float(mine_at is not None and said and mine_at == max(said)),
        "noun_position": (mine_at / len(order)) if mine_at is not None else 0.0,
        "category_overlap": len(category & words) / max(len(category), 1),
        "leftmost": float(any(p.startswith("the leftmost") for p in mine)),
        "rightmost": float(any(p.startswith("the rightmost") for p in mine)),
        "frontmost": float(any(p.startswith("the frontmost") for p in mine)),
        "backmost": float(any(p.startswith("the backmost") for p in mine)),
        "middle": float(any(p.startswith("the middle") for p in mine)),
        "between_two_named": between,
        "on_top_of_named": rel("on top of"),
        "inside_named": rel(("inside", "in the")),
        "named_part_of_named": part,
        "next_to_named": rel("next to"),
        "nearest_named": rel("nearest to"),
        "is_fixture": float(entity.kind == "fixture"),
        "peers": float(peers > 1),
    }


def _features(row: dict, qid: str, sid: str, vocabulary: set[str]) -> dict[str, float]:
    base = candidate_vector(row, qid, sid)
    out = dict(base)
    out["bias"] = 1.0
    for token in set(_tokens(row["metadata"]["instruction"])) & vocabulary:
        for name, value in base.items():
            if value:
                out[f"{token}*{name}"] = value
    return out


def linear_probe(rows: list[dict], qid: str, *, min_count: int = 3, steps: int = 400,
                 lr: float = 0.5, l2: float = 1e-3) -> dict:
    """A bag-of-words × per-candidate-features conditional logit, trained on the train split.

    Gate G5(b). One weight vector scores every candidate of a row and the loss is the softmax
    cross-entropy over that row's candidates -- the same shape NanoJev's own head has, which is
    why a linear model failing here would mean the *text* is at fault and not the reader.

    NumPy only: the pixi environment has no scikit-learn, and 400 full-batch gradient steps over
    a few thousand candidate rows takes under a second.
    """
    import numpy as np

    train = [r for r in rows if r["split"] == "train" and qid in r["questions"]]
    counts: collections.Counter = collections.Counter()
    for row in train:
        counts.update(set(_tokens(row["metadata"]["instruction"])))
    vocabulary = {token for token, n in counts.items() if n >= min_count}

    def encode(subset):
        groups = []
        for row in subset:
            sids = sorted(row["questions"][qid]["criteria"])
            groups.append((row, sids, [_features(row, qid, sid, vocabulary) for sid in sids]))
        return groups

    train_groups = encode(train)
    index: dict[str, int] = {}
    for _, _, feats in train_groups:
        for feature in feats:
            for key in feature:
                index.setdefault(key, len(index))

    def matrix(groups):
        out = []
        for row, sids, feats in groups:
            block = np.zeros((len(sids), len(index)))
            for i, feature in enumerate(feats):
                for key, value in feature.items():
                    if key in index:
                        block[i, index[key]] = value
            out.append((row, sids, block))
        return out

    train_blocks = matrix(train_groups)
    weights = np.zeros(len(index))
    for _ in range(steps):
        gradient = np.zeros(len(index))
        for row, sids, block in train_blocks:
            scores = block @ weights
            probabilities = np.exp(scores - scores.max())
            probabilities /= probabilities.sum()
            truth = np.zeros(len(sids))
            truth[sids.index(row["gold"][qid])] = 1.0
            gradient += block.T @ (probabilities - truth)
        gradient = gradient / max(len(train_blocks), 1) + l2 * weights
        weights -= lr * gradient

    def accuracy(subset):
        blocks = matrix(encode(subset))
        hits = 0
        misses = []
        for row, sids, block in blocks:
            pick = sids[int(np.argmax(block @ weights))]
            hits += int(pick == row["gold"][qid])
            if pick != row["gold"][qid]:
                misses.append((row["metadata"]["instruction"], pick, row["gold"][qid]))
        return (hits / len(blocks) if blocks else float("nan")), misses

    return {"weights": weights, "index": index, "vocabulary": vocabulary,
            "accuracy": accuracy, "n_train": len(train)}


def resolver_report(rows: list[dict], qid: str) -> dict:
    """Gate G5(a)'s numbers: accuracy overall, per split, per suite, and every miss."""
    subset = [r for r in rows if qid in r["questions"]]
    hits, misses = 0, []
    per_suite: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    per_split: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for row in subset:
        pick = resolve_row(row, qid)
        ok = pick == row["gold"][qid]
        hits += int(ok)
        per_suite[row["family_id"]][0] += int(ok)
        per_suite[row["family_id"]][1] += 1
        per_split[row["split"]][0] += int(ok)
        per_split[row["split"]][1] += 1
        if not ok:
            misses.append({"instruction": row["metadata"]["instruction"], "suite": row["family_id"],
                           "task": row["metadata"]["task_index"], "picked": pick,
                           "gold": row["gold"][qid], "split": row["split"]})
    return {"n": len(subset), "accuracy": hits / len(subset) if subset else float("nan"),
            "per_suite": {k: v for k, v in sorted(per_suite.items())},
            "per_split": {k: v for k, v in sorted(per_split.items())},
            "misses": misses}


# ==================================================================== part 4: serving

# The rows above are built from a BDDL file and a simulator. A *server* has neither: at reset it
# has the instruction and `obs["privileged"]`, which is every movable object's world pose and
# nothing else (`decision.env.DecisionEnv.privileged`). This part builds the same request out
# of that, so the sentence the checkpoint was trained to ground is the sentence it is asked at
# run time -- byte for byte, for the same scene (`test_grounding_v2.py` pins it).
#
# **What a live scene cannot say, and is therefore not said.** Three things in a training row
# come from the simulator's model rather than from an observation, and every one of them is
# absent here rather than approximated:
#
# 1. **Fixtures.** LIBERO builds observables for its movable objects only
#    (`bddl_base_domain.py:470-476`), so a cabinet, a stove or a tray is not in `privileged` and
#    cannot be a candidate. On LIBERO-Spatial that costs nothing -- both grounding answers of all
#    ten tasks are movable objects -- but on `libero_goal`'s "open the top drawer of the cabinet"
#    the answer is not in the candidate set at all, and a server there must fall back to the rule
#    (which is what `meta.grounding.source` records).
# 2. **Region sites.** `<fixture>_top_region`, `<stove>_cook_region` and the rest are MuJoCo
#    sites, not observables, so `Entity.anchors` is empty and the containment phrases the sites
#    produce ("in the top part of cabinet_1") cannot appear. The position-only containment rules
#    (`inside`, `on top of`) still do, because they are arithmetic on the candidates' own
#    coordinates.
# 3. **The table's body.** `_scene_frame` reads the surface's `x, y` off whichever body holds the
#    scene up; here it is taken as the world origin, which is where every LIBERO scene's table
#    actually is (see `_scene_frame`: "in practice all of them sit at (0, 0)"). The frame's *z*
#    needs no model at all -- it is the median resting height of the movable objects, which
#    `privileged` gives in full.

#: The seed the serving-time numbering is drawn with. Training shuffles the digits per row so a
#: model cannot score by reading `_1` (`assign_ids`); a served episode has no such hazard and
#: wants the opposite property -- the same scene must get the same ids on a re-ground ten
#: decisions later, or the answer committed at t=0 and the answer committed after a failed grasp
#: would be written in two different alphabets.
SERVING_SEED: int = 0

#: The two questions a serving-time grounding forward asks, in the order they are declared.
SERVING_QIDS: tuple[str, ...] = tuple(EPISODE_INSTRUCTIONS)


def serving_frame(privileged: dict) -> tuple[float, float, float]:
    """`_scene_frame` without a simulator: the world origin, and the objects' median height."""
    heights = sorted(_xyz(body["pos"])[2] for body in privileged.values())
    return 0.0, 0.0, (heights[len(heights) // 2] if heights else 0.0)


def serving_entities(privileged: dict) -> list[Entity]:
    """One `Entity` per movable object in a live observation, in the row builder's own units.

    The category is the object's name with its trailing index removed, which **is** the BDDL's
    own type for every LIBERO asset (`akita_black_bowl_1` is an `akita_black_bowl`), so the
    candidate lines read exactly as the training rows' do -- including the `CATEGORY_WORDS`
    corrections, without which "the white mug" matches the wrong mug.
    """
    frame = serving_frame(privileged)
    return [Entity(name, _category_of(name), "object", _to_frame(_xyz(privileged[name]["pos"]), frame))
            for name in sorted(privileged)]


def _xyz(pos) -> tuple[float, float, float]:
    """`(x, y, z)` out of whatever a backend puts in `privileged[name]["pos"]` -- a numpy array,
    a list or a tuple. Spelt out rather than reached for with numpy, because this module is
    stdlib-only at module scope and a grounding row is text."""
    x, y, z = (float(v) for v in tuple(pos)[:3])
    return x, y, z


def _category_of(name: str) -> str:
    parts = name.split("_")
    return "_".join(parts[:-1]) if len(parts) >= 2 and parts[-1].isdigit() else name


def serving_ids(privileged: dict, *, seed: int = SERVING_SEED) -> dict[str, str]:
    """`{scene object name: short id}` for a live scene -- `assign_ids`' numbering, seeded."""
    return assign_ids(serving_entities(privileged), random.Random(seed))


def grounding_names(privileged: dict, *, seed: int = SERVING_SEED) -> dict[str, str]:
    """The inverse: `{short id: scene object name}`, which is how a chosen candidate becomes the
    name `TrackerV2.commit` (and every later waypoint) is given."""
    return {sid: name for name, sid in serving_ids(privileged, seed=seed).items()}


def grounding_request(instruction: str, privileged: dict, *, state_id: str = "grounding",
                      qids=SERVING_QIDS, seed: int = SERVING_SEED,
                      ids: dict[str, str] | None = None) -> dict:
    """NanoJev's request for the once-per-episode grounding forward, from a live observation.

    `{"states": [{"id", "state", "questions"}]}` and no other key, exactly as
    `v2.questions.request` builds a motion request -- but rendered by **this** module, in the
    camera frame the instructions are written in. It is never concatenated with the motor state:
    that one is gripper-relative and mirrored left for right, so a sentence saying "the bowl on
    the left" read against it would resolve the wrong bowl (module docstring, "the frame is the
    viewer's").

    `ids` is an escape hatch for the test that pins this against `build_rows`; a server leaves it
    alone and gets `serving_ids`.
    """
    entities = serving_entities(privileged)
    ids = ids if ids is not None else assign_ids(entities, random.Random(seed))
    positions = {ids[e.name]: tuple(e.pos) for e in entities if e.pos is not None}
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for entity in entities:
        if entity.pos is not None:
            groups[base_word(entity.name)].append(ids[entity.name])
    ranks = extremes(positions, groups)
    phrases = {sid: ranks.get(sid, []) + rel
               for sid, rel in relations(
                   positions, {ids[e.name]: e.anchors for e in entities}).items()}
    sids = sorted(ids.values())
    return {
        "states": [
            {
                "id": state_id,
                "state": grounding_state(instruction, entities, ids),
                "questions": {qid: question_block(qid, sids, entities, ids, phrases)
                              for qid in qids},
            }
        ]
    }


def write_rows(rows: list[dict], path: pathlib.Path) -> pathlib.Path:
    """The rows as JSONL, with the probes' private `_`-prefixed metadata stripped."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            clean = dict(row)
            clean["metadata"] = {k: v for k, v in row["metadata"].items() if not k.startswith("_")}
            handle.write(json.dumps(clean, sort_keys=True) + "\n")
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("scenes", "rows", "probe"))
    parser.add_argument("--cache", type=pathlib.Path, default=SETTLED_SCENE_CACHE)
    parser.add_argument("--jobs", type=int, default=1,
                        help="scenes: build this many tasks at once, one simulator per process")
    parser.add_argument("--settled-steps", type=int, default=SETTLED_STEPS,
                        help="scenes: no-op control steps after reset before the poses are read "
                             "(0 reproduces the first, unsettled cache)")
    parser.add_argument("--out", type=pathlib.Path, default=SCENE_CACHE.parent / "rows.jsonl")
    parser.add_argument("--n-init", type=int, default=INIT_STATES_PER_TASK)
    args = parser.parse_args(argv)

    if args.command == "scenes":
        build_scene_cache(args.cache, n_init=args.n_init, jobs=args.jobs,
                          settled_steps=args.settled_steps)
        return 0

    scenes = load_scenes(args.cache)
    if not scenes:
        print(f"no scenes in {args.cache}; run `scenes` first", file=sys.stderr)
        return 1
    rows, report = build_rows(scenes)
    attach_scene(rows, scenes)
    if args.command == "rows":
        write_rows(rows, args.out)
        print(json.dumps({k: dict(v) if isinstance(v, collections.Counter) else v
                          for k, v in report.items()}, indent=2, sort_keys=True))
        return 0

    for qid in ("target", "destination"):
        rule = resolver_report(rows, qid)
        probe = linear_probe(rows, qid)
        train_accuracy, _ = probe["accuracy"]([r for r in rows if r["split"] == "train"
                                               and qid in r["questions"]])
        test_accuracy, misses = probe["accuracy"]([r for r in rows if r["split"] == "test"
                                                   and qid in r["questions"]])
        print(f"== {qid}: {rule['n']} rows")
        print(f"   resolver {rule['accuracy']:.3f} overall, per split {rule['per_split']}")
        print(f"   probe train {train_accuracy:.3f} test {test_accuracy:.3f} "
              f"({len(probe['index'])} features)")
        for miss in rule["misses"][:20]:
            print(f"   rule miss [{miss['suite']}:{miss['task']} {miss['split']}] "
                  f"{miss['instruction']!r} -> {miss['picked']} (gold {miss['gold']})")
        for instruction, pick, gold in misses[:10]:
            print(f"   probe miss {instruction!r} -> {pick} (gold {gold})")
    return 0


if __name__ == "__main__":  # pragma: no cover -- the CLI entry point
    raise SystemExit(main())
