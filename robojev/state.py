"""v2's state string: §4's template, and the tracker that fills it.

Everything after the `Task:` line is computed by code from observations -- §2's principle 6,
*memory is a structured tracker kept by code; the model reads it*. Nothing here is a prediction
and nothing is prose: the differences from v1's block are that distances are signed centimetres
rather than metres, that each axis gets its own line stating the relation its own question asks
(principle 2), that the waypoint the whole decision is steering to is **named in the state**
(principle 1 -- otherwise the label is not a function of the text), and that the summary is
counters.

The waypoint is produced **by code**, not by the model: the rim point of the committed target
while reaching and grasping, a point above the destination while carrying and placing, a point up
and away after the release. That is the NanoJev division of labour -- the model judges, code
plans. The rim rule is the feasibility note's
(`docs/DESIGN.md`):

    target_xy = object_xy + r * normalize(eef_xy - object_xy)
    target_z  = object_z + dz

with `r` and `dz` **per task**, from the harvest manifest's `grasp_rim_radius.by_task` and
`grasp_rim_dz.by_task`; the defaults below are that note's own medians. Grasping the object's
*centre* holds 0.16-0.29 of the time against 0.73-0.93 at the demonstrators' rim pose, which is
failure F5b.

`robojev.roles.phase` does the stage judgement -- the same rule, from the same finger
width and the same offsets, that every other reader of a stage uses -- and `TrackerV2` keeps the
per-decision bookkeeping v2 prints and a stage name has no line for: the waypoint, its distance,
the release, the retreat stage.
"""
from __future__ import annotations

import dataclasses

import numpy as np

# The planner's own geometry, imported rather than re-derived: `grasp_direction` and the tuned
# constants are what the closed-loop tables were measured with, and a second copy of them here
# would be a second thing to retune (docs/DESIGN.md §1).
from robojev.planner.pick_and_place import (
    CARRY_CLEARANCE,
    FINGER_HALF_WIDTH,
    DROP_TOL,
    GRASP_DZ,
    HOVER,
    LIFT,
    RETRY_DZ,
    RIM_RADIUS,
    SETTLE_DECISIONS,
    SUBGOALS,
    grasp_direction,
    new_phase as _new_carry,
    plan as _plan,
    rim_point as _rim_point,
    with_executed as _with_executed,
)
from robojev.planner.executor import held as skill_held
from robojev.roles import CLOSED_WIDTH, phase
from robojev.scene import movable, short_id, support as scene_support
from robojev.compose import (
    AXIS_SLOT,
    CHUNK_STEPS,
    DEFAULT_ARRIVAL_CM,
    DEFAULT_TOLERANCE_CM,
    DEFAULT_YAW_TOLERANCE_DEG,
    GripLatch,
    band_words,
    labels_from_waypoint,
    largest_remaining,
    normalise_phase,
    step_range_words,
    step_size_for,
)
from robojev.questions import ACTIVE_QIDS

#: The Panda's finger span, fully open, in metres (v1's `GRIPPER_FULL_OPEN`).
GRIPPER_FULL_OPEN: float = 0.08

#: **The expert's tuned constants**, not the feasibility note's estimates, and imported so they
#: cannot drift. `r = 0.05` is the note's median; `dz = 0.025` is the expert's **measurement**
#: against the note's 0.02-0.05 spread -- at +0.045 the fingers shut to 2 mm on air and the bowl
#: does not move, at +0.025 the same close catches the rim (docs/DESIGN.md). Per-task values from the
#: harvest manifest's `grasp_rim_radius.by_task` / `grasp_rim_dz.by_task` override them.
DEFAULT_RIM_RADIUS_M: float = RIM_RADIUS
DEFAULT_RIM_DZ_M: float = GRASP_DZ
#: How far above the grasp point the approach flies before it descends (`expert.HOVER`), how far
#: the hand lifts (`expert.LIFT`), and how high the bowl is carried over the destination
#: (`expert.CARRY_CLEARANCE`), in metres.
DEFAULT_HOVER_M: float = HOVER
DEFAULT_LIFT_HEIGHT_M: float = LIFT
DEFAULT_PLACE_HEIGHT_M: float = CARRY_CLEARANCE
#: Where the hand goes once the fingers have opened: straight up, clear of what it put down.
DEFAULT_RETREAT_HEIGHT_M: float = 0.15
#: The grasp heights the plan retries at, in the order it walks them (`expert.RETRY_DZ`): the
#: tuned height first, then upwards, because a close that is too low pushes the bowl away before
#: it shuts. (A candidate also carries which way round the rim to stand and how far the wrist
#: must turn to close there, which this plainer stage geometry has no notion of -- it re-reads
#: the closing axis every decision and never rotates.)
DEFAULT_RETRY_DZ_M: tuple[float, ...] = RETRY_DZ
#: How many decisions the fingers hold still and closed before the lift (`expert.SETTLE_DECISIONS`).
DEFAULT_SETTLE_DECISIONS: int = SETTLE_DECISIONS

#: Bumped whenever a **rendered byte** changes. The guard a checkpoint is checked against at
#: launch: a model fine-tuned on one rendering and served another has been asked a different
#: question about the same state, silently, for a whole episode.
MEMORY_RULE_V2: str = "v2-tracker-1"

#: The waypoint distances the `Events` line records the first crossing of, in centimetres.
EVENT_MILESTONES_CM: tuple[float, ...] = (20.0, 10.0, 5.0)
#: How many events and how many history rows the state carries.
DEFAULT_MAX_EVENTS: int = 4
DEFAULT_HISTORY_K: int = 3
#: Net displacement below this fraction of the path walked, over at least `LOOP_MIN_DECISIONS`
#: decisions, is what the state calls "looping" -- run 3's failure was an arm that hovered beside
#: the bowl for two hundred steps and could not see that it had.
LOOP_RATIO: float = 0.5
LOOP_MIN_DECISIONS: int = 6


def _cm(value) -> str:
    """One decimal, explicit sign, never `-0.0` -- at this resolution a value that rounds to zero
    is level, and a minus sign there reads as a direction it is not."""
    rounded = round(float(value), 1)
    return format(rounded if rounded != 0.0 else 0.0, "+.1f")


def _mag(value) -> str:
    """One decimal, no sign: a distance."""
    return format(round(abs(float(value)), 1), ".1f")


def rounded_cm(values) -> np.ndarray:
    """The offsets **as the state prints them**.

    The label function must be applied to the numbers the model reads, not to the full-precision
    ones behind them, or a value a hair outside the tolerance could print as a value inside it
    and gate G3 would fail on rounding alone. Every caller that wants the gold answer for a state
    rounds first, through here.
    """
    return np.round(np.asarray(values, dtype=np.float64).reshape(3) + 0.0, 1)


# ------------------------------------------------------------------------------ the waypoint


@dataclasses.dataclass(frozen=True)
class Waypoint:
    """Where the hand is steering, and what the state calls it."""

    label: str
    point: np.ndarray          # world metres
    offset_m: np.ndarray       # point - eef, world metres

    @property
    def offset_cm(self) -> np.ndarray:
        return self.offset_m * 100.0

    @property
    def distance_cm(self) -> float:
        return float(np.linalg.norm(self.offset_m)) * 100.0


def rim_point(eef_xy, object_pos, radius_m: float = DEFAULT_RIM_RADIUS_M,
              dz_m: float = DEFAULT_RIM_DZ_M) -> np.ndarray:
    """The feasibility note's rule, delegated to `expert.rim_point` so the three copies of it in
    this tree (here, `expert.py`, `dagger.py`) cannot disagree: `object_xy + r *
    normalize(eef_xy - object_xy)`, `object_z + dz`, with `+x` for the degenerate hand-directly-
    above case so the state text never depends on floating-point noise.

    This is the note's **approach-side** rule and it is kept because the radius is what the note
    measured. `waypoint_for` does *not* use this direction -- see `grasp_direction`.
    """
    obj = np.asarray(object_pos, dtype=np.float64).reshape(3)
    xy = _rim_point(obj[:2], eef_xy, radius_m)
    return np.array([xy[0], xy[1], obj[2] + dz_m])


def waypoint_for(proprio, objects: dict, target: str | None, destination: str | None,
                 subgoal: str, *, rim_radius_m: float = DEFAULT_RIM_RADIUS_M,
                 rim_dz_m: float = DEFAULT_RIM_DZ_M,
                 lift_height_m: float = DEFAULT_LIFT_HEIGHT_M,
                 place_height_m: float = DEFAULT_PLACE_HEIGHT_M,
                 retreat_height_m: float = DEFAULT_RETREAT_HEIGHT_M,
                 attempts: int = 0) -> "Waypoint | None":
    """The one point this decision is steering to, chosen by the stage. Pure.

    Three corrections to the feasibility note's rule, each measured and each worth episodes
    (`notes/2026-09-21-stage9-scripted-expert.md` §1):

    1. **The rim direction is the wrist's closing axis, not the approach side.** The fingers close
       along the gripper's local y and span 8 cm about a rim ~6 cm across: radially, one finger
       goes inside the bowl and one outside; tangentially, both land on the ring and the descent
       *pushes the bowl across the table*. Task 2 went **3/10 to 9/10** on this alone, same
       radius. `expert.grasp_direction` is that axis, signed toward the hand.
    2. **The place waypoint aims the bowl, not the hand.** A rim grasp holds the bowl a rim-radius
       to one side, so aiming the hand at the plate's centre lands the bowl beside it and LIBERO's
       `on(bowl, plate)` never fires -- a clean pick, carry and release that scored nothing. The
       waypoint is therefore `destination_xy - (target_xy - eef_xy)`.
    3. **Retries walk the grasp height.** `attempts` indexes `DEFAULT_RETRY_DZ_M`, upwards first:
       too high shuts on air, too low pushes the bowl away before the fingers close.
    """
    eef = np.asarray(proprio, dtype=np.float64).reshape(-1)[0:3]
    stage = normalise_phase(subgoal)
    have_target = target is not None and target in objects
    have_destination = destination is not None and destination in objects
    attempt_dz = rim_dz_m + DEFAULT_RETRY_DZ_M[min(int(attempts), len(DEFAULT_RETRY_DZ_M) - 1)]
    if stage in ("reach", "grasp") and have_target:
        pos = np.asarray(objects[target]["pos"], dtype=np.float64).reshape(3)
        direction = grasp_direction(proprio, pos[:2])
        point = np.array([pos[0] + rim_radius_m * direction[0],
                          pos[1] + rim_radius_m * direction[1],
                          pos[2] + attempt_dz])
        label = f"rim of {short_id(target)}, closing axis"
    elif stage == "lift" and have_target:
        pos = np.asarray(objects[target]["pos"], dtype=np.float64).reshape(3)
        direction = grasp_direction(proprio, pos[:2])
        point = np.array([pos[0] + rim_radius_m * direction[0],
                          pos[1] + rim_radius_m * direction[1],
                          pos[2] + attempt_dz + lift_height_m])
        label = f"clear above {short_id(target)}"
    elif stage in ("carry", "place") and have_destination and have_target:
        pos = np.asarray(objects[destination]["pos"], dtype=np.float64).reshape(3)
        held = np.asarray(objects[target]["pos"], dtype=np.float64).reshape(3)
        # The bowl is what has to land on the plate, and it is not in the hand's axis.
        carried = held[:2] - eef[:2]
        point = np.array([pos[0] - carried[0], pos[1] - carried[1],
                          pos[2] + place_height_m + (float(eef[2]) - float(held[2]))])
        label = f"{short_id(target)} above {short_id(destination)}"
    elif stage == "retreat" and have_destination:
        pos = np.asarray(objects[destination]["pos"], dtype=np.float64).reshape(3)
        point = np.array([float(eef[0]), float(eef[1]), pos[2] + retreat_height_m])
        label = f"retreat above {short_id(destination)}"
    else:
        return None
    return Waypoint(label=label, point=point, offset_m=point - eef)


#: What the state calls each of the expert's sub-stages. The *plan* changes between them even
#: where the v2 subgoal does not -- `approach` flies to a point `HOVER` above the rim and
#: `descend` goes down to the rim itself -- so the sub-stage is printed beside the subgoal and
#: named in the waypoint's own label. Without it the model would read one `reach` stage whose
#: waypoint jumps 8 cm in z for no reason the text gives.
SUBSTAGE_LABEL: dict[str, str] = {
    "approach": "approach: {hover} cm above the rim of {target}, {side} side",
    "descend": "descend: rim of {target}, {side} side",
    "close": "close: hold still and shut the fingers",
    "lift": "lift: clear above {target}",
    "carry": "carry: {target} above {destination}",
    "lower": "lower: {target} onto {destination}",
    "release": "release: hold still and open the fingers",
    "retreat": "retreat: up and away from {destination}",
    "done": "done: hold still",
}


def plan(proprio, objects: dict, instruction: str, carry: dict | None, *,
         qids=None) -> tuple["Waypoint", str, dict, dict]:
    """The waypoint, the sub-stage, the next phase carry and the planner's facts.

    **This is the expert's own phase machine, imported and not re-implemented.** The plan is
    where the 85/100 lives (`notes/2026-09-21-stage9-scripted-expert.md`): the approach flies to
    `HOVER` above the rim and only descends once x and y are aligned; the lift is measured from
    where the hand was when it closed, not from the target, which recedes as fast as the hand
    climbs; the carry aims the **bowl** over the destination, not the hand; the release height is
    read off the bowl; a grasp that caught nothing or a stage that stopped getting closer takes
    the next of `expert.GRASP_CANDIDATES`; and the rim **axis** is latched inside `LATCH_RADIUS`
    so the hand does not walk around the bowl.

    A tracker that named the rim point as its `reach` waypoint would be teaching "descend
    diagonally straight at the rim" -- a policy nobody has measured closed-loop. Copying the
    machine here would be a second thing to retune; calling it is one.

    This is **planning by code**, the NanoJev division of labour, and it is not the policy: the
    answers still come from the state text through `labels_from_waypoint`, and
    `test_the_closed_loop_runs_on_the_parsed_answers_alone` drives the simulator with nothing but
    `parse(serialise_v2(...))`.
    """
    carry = _new_carry() if carry is None else carry
    _goal, nxt, meta = _plan(proprio, objects, instruction, carry, qids=qids)
    eef = np.asarray(proprio, dtype=np.float64).reshape(-1)[0:3]
    offset = np.asarray(meta["error"], dtype=np.float64).reshape(3)
    substage = meta["phase"]
    grasp_point = np.asarray(meta["grasp_point"], dtype=np.float64).reshape(3)
    target_pos = (np.asarray(objects[meta["target"]]["pos"], dtype=np.float64).reshape(3)
                  if meta.get("target") in objects else grasp_point)
    side = _side_words(grasp_point[:2] - target_pos[:2])
    label = SUBSTAGE_LABEL[substage].format(
        hover=f"{HOVER * 100:.0f}", side=side,
        target=short_id(meta.get("target") or "target"),
        destination=short_id(meta.get("destination") or "destination"))
    return Waypoint(label=label, point=eef + offset, offset_m=offset), substage, nxt, meta


def _verdict(row: dict) -> str:
    """What one candidate line ends with: the relation its own question asks, already made.

    `fits` / `blocked by <what>` / `tried, ...` -- one comparison per line rather than an
    arg-max over a column of centimetres, which is what a small model is poor at (§2
    principle 2), and the three words `v2.parse` reads the `rim` label back out of.
    """
    verdict = "fits" if row["fits"] else f"blocked by {short_id(row['blocker'] or 'something')}"
    return f"tried, {verdict}" if row["tried"] else verdict


def _rests_words(target, objects: dict, meta: dict) -> str:
    """The one line that says what the target is standing in, and how high the carry has to be.

    The thing a coordinate list cannot say: a bowl at z 1.06 is a bowl in a drawer whose front
    panel stands 6.5 cm above it, and that is what makes the extraction 14 cm rather than 10.
    """
    what, rise = scene_support(target, objects)
    carry_cm = rise * 100.0
    if not what or rise <= 0.005:
        return f"Target rests on a flat surface; it is carried {CARRY_CLEARANCE * 100:.0f} cm "\
               f"above the destination."
    return (f"Target rests inside {short_id(what)}, whose walls reach {carry_cm:.1f} cm above it; "
            f"it is lifted clear of them before it travels.")


def _side_words(direction) -> str:
    """Which side of the target the rim point sits on, in words -- the end of the wrist's closing
    axis the grasp candidate in force stands on. Printed so the text explains a waypoint that
    would otherwise look like it had moved for no reason."""
    dx, dy = float(direction[0]), float(direction[1])
    if abs(dx) >= abs(dy):
        return "+x" if dx >= 0 else "-x"
    return "+y" if dy >= 0 else "-y"


# ------------------------------------------------------------------------------ the tracker


@dataclasses.dataclass
class _Row:
    """One decision, as observed. `answers` and `effect_cm` are filled in afterwards."""

    step: int
    pos: np.ndarray
    width: float
    subgoal: str
    substage: str
    holding: bool
    waypoint_cm: np.ndarray | None
    distance_cm: float | None
    yaw_err_deg: float
    rim: tuple = ()
    rests: str = ""
    answers: dict | None = None
    effect_cm: float | None = None


class TrackerV2:
    """The structured tracker behind `serialise_v2`: one episode, as v2 prints it.

    Three calls per decision:

        tracker.observe(step=t, proprio=…, objects=…)
        text = serialise_v2(proprio, objects, instruction, tracker, annotate=True)
        tracker.answer(answers)

    `observe` closes out the previous decision: the `Last 3` block's effect column and the path
    counters are differences between two *observed* states, never predictions of what a command
    would do. The target and the destination are committed once, at construction or through
    `commit`, which is where the once-per-episode `target`/`destination` answers land.
    """

    def __init__(self, *, horizon: int, instruction: str | None = None,
                 target: str | None = None,
                 destination: str | None = None, every: int = CHUNK_STEPS,
                 history_k: int = DEFAULT_HISTORY_K, max_events: int = DEFAULT_MAX_EVENTS,
                 qids=ACTIVE_QIDS, annotate: bool = True,
                 tolerance_cm: float = DEFAULT_TOLERANCE_CM,
                 arrival_cm: float = DEFAULT_ARRIVAL_CM,
                 yaw_tolerance_deg: float = DEFAULT_YAW_TOLERANCE_DEG,
                 rim_radius_m: float = DEFAULT_RIM_RADIUS_M,
                 rim_dz_m: float = DEFAULT_RIM_DZ_M,
                 lift_height_m: float = DEFAULT_LIFT_HEIGHT_M,
                 place_height_m: float = DEFAULT_PLACE_HEIGHT_M,
                 retreat_height_m: float = DEFAULT_RETREAT_HEIGHT_M):
        self.horizon = int(horizon)
        self.every = int(every)
        self.history_k = int(history_k)
        self.max_events = int(max_events)
        self.qids = tuple(qids)
        self.annotate = bool(annotate)
        self.tolerance_cm = float(tolerance_cm)
        self.arrival_cm = float(arrival_cm)
        self.yaw_tolerance_deg = float(yaw_tolerance_deg)
        self.rim_radius_m = float(rim_radius_m)
        self.rim_dz_m = float(rim_dz_m)
        self.lift_height_m = float(lift_height_m)
        self.place_height_m = float(place_height_m)
        self.retreat_height_m = float(retreat_height_m)
        #: The instruction the planner reads. With it the tracker steers by the expert's own
        #: sub-stage machine (`plan`), which is what the 85/100 was measured with; without it it
        #: falls back to `waypoint_for`'s stage geometry, which is what a caller with no
        #: instruction (a token measurement, a fixture) wants.
        self.instruction = instruction
        self.target = target
        self.destination = destination
        self.chosen_at: int | None = None
        self.source: str | None = None
        #: The runtime guard on a *model's* `grip` answer (the measurements's cliff). Not used by
        #: `gold_answers`, which reports the expert's raw latch.
        self.latch = GripLatch()
        self.reset()

    # -------------------------------------------------------------------------------- state

    def reset(self) -> None:
        self._rows: list[_Row] = []
        self._events: list[str] = []
        self._milestones: set[float] = set()
        self._done: list[str] = []
        self._closest: tuple[float, int] | None = None
        self._path_cm = 0.0
        self._grasp_attempts = 0
        self._committed_attempts = 0
        self._held = 0
        self._rose_at: int | None = None
        self._released = False
        self._start_z: float | None = None
        self._carry: dict | None = None
        self._closed_offset: list[float] | None = None
        self._substage = "approach"
        #: The planner's own facts about the decision last observed (`planner.plan`'s `meta`),
        #: or None where there is no plan. A harvest records a few of them beside each row.
        self.plan_meta: dict | None = None
        self.latch.reset()

    def commit(self, target: str | None, destination: str | None, step: int = 0,
               source: str = "model") -> None:
        """The once-per-episode `target`/`destination` answers, written into memory (§3). Every
        motor question afterwards refers to "the target".

        `source` is how the pair was named -- `"model"` (the grounding forward), `"rule"`
        (`roles.scene_roles`) or `"bddl"` (the suite's own `obj_of_interest`) -- and a record
        carries it, because the expert the measurements measured the rule wrong on 2 of 10 tasks and a run
        has to be able to say which rule filled its `Target:` line.
        """
        self.target = target
        self.destination = destination
        self.chosen_at = int(step)
        self.source = str(source)
        self._committed_attempts = self._grasp_attempts
        # The plan follows the *committed* pair, not the planner's own reading of the sentence:
        # the whole point of asking `target` once is that everything afterwards refers to it.
        self._carry = _new_carry({"target": target, "destination": destination})

    def committed(self) -> dict | None:
        """What was committed, and how."""
        if self.target is None and self.destination is None:
            return None
        return {"target": self.target, "destination": self.destination,
                "source": self.source, "step": self.chosen_at}

    def needs_regrounding(self) -> bool:
        """Should `target`/`destination` be asked again?

        True when a grasp has been attempted since the commit and **nothing has ever risen** --
        the tracker's own facts, no model in the loop. §3: the two answers are re-asked "only when
        the tracker records a failed attempt", and a failed attempt on a scene with two identical
        bowls is the evidence that the wrong one was named (failure F3).
        """
        return self._grasp_attempts > self._committed_attempts and self._rose_at is None

    def settings(self) -> dict:
        """The knobs that change a rendered byte -- the guard a checkpoint is checked against."""
        return {
            "memory_rule": MEMORY_RULE_V2,
            "horizon": self.horizon,
            "every": self.every,
            "history_k": self.history_k,
            "max_events": self.max_events,
            "annotate": self.annotate,
            "qids": list(self.qids),
            "tolerance_cm": self.tolerance_cm,
            "arrival_cm": self.arrival_cm,
            "rim_radius_m": self.rim_radius_m,
            "rim_dz_m": self.rim_dz_m,
        }

    def log_latch(self, text: str) -> None:
        """What `GripLatch` calls when it refuses a change: the refusal becomes an event, so the
        next state the model reads says its own latch request was not applied."""
        step = self._rows[-1].step if self._rows else 0
        self._events.append(f"t={step} {text}")

    @property
    def decisions(self) -> int:
        return len(self._rows)

    @property
    def subgoal(self) -> str:
        return self._rows[-1].subgoal if self._rows else "reach"

    @property
    def substage(self) -> str:
        """The expert sub-stage the plan is in: `approach`, `descend`, `close`, `lift`, `carry`,
        `lower`, `release`, `retreat` or `done`."""
        return self._rows[-1].substage if self._rows else "approach"

    @property
    def waypoint(self) -> Waypoint | None:
        return self._waypoint

    @property
    def last_row(self) -> "_Row":
        """The decision point most recently observed."""
        if not self._rows:
            raise ValueError("nothing observed yet")
        return self._rows[-1]

    # ------------------------------------------------------------------------------ updating

    def observe(self, *, step: int, proprio, objects: dict, yaw_err_deg: float = 0.0,
                released: bool | None = None) -> None:
        """Record one decision point, and close out the one before it."""
        proprio = np.asarray(proprio, dtype=np.float64).reshape(-1)
        if proprio.shape[0] < 8:
            raise ValueError(f"expected the 8-d proprio vector, got shape {proprio.shape}")
        if self.chosen_at is None and self.target is not None:
            self.chosen_at = int(step)
        if released is not None:
            self._released = self._released or bool(released)

        index = len(self._rows)
        if index == 0 and self.target in objects:
            self._start_z = float(np.asarray(objects[self.target]["pos"], np.float64)[2])
        width = float(proprio[6] - proprio[7])
        closed = width <= CLOSED_WIDTH
        # **The one `held`** (`skill.held`): the fingers are shut *and* the target has not moved
        # relative to them since they shut on it. The planner's own carry is the reference where
        # there is a plan; where there is not (no instruction, the plainer stage geometry below)
        # the tracker records the offset itself, the first time the fingers close.
        offset, reference = self._grasp_offset(proprio, objects, closed)
        holding = skill_held(closed, offset, reference, DROP_TOL)
        if (self._start_z is not None and self.target in objects
                and float(np.asarray(objects[self.target]["pos"], np.float64)[2])
                - self._start_z > 0.02 and self._rose_at is None):
            self._rose_at = int(step)

        if self.instruction is not None and self.target in objects:
            if self._carry is None:
                self._carry = _new_carry({"target": self.target,
                                                 "destination": self.destination})
            wp, substage, self._carry, meta = plan(proprio, objects, self.instruction,
                                                   self._carry, qids=self.qids)
            self.plan_meta = meta
            stage = SUBGOALS[substage]
            self._grasp_attempts = int(meta["attempts"])
            holding = bool(meta["held"])
            # **The wrist's error is the plan's, not the caller's.** It is how far the wrist must
            # still turn for the fingers to close radially on the rim point the candidate in
            # force stands on (`expert.wrist_yaw_error`), and zero for a candidate on the axis
            # the wrist already holds -- which is every candidate the eight non-drawer tasks
            # reach. The argument stays for the plainer stage geometry below, which has no plan
            # to ask.
            yaw_err_deg = float(meta["yaw_error"])
            rim = tuple(meta.get("rim") or ())
            rests = _rests_words(self.target, objects, meta)
        else:
            self.plan_meta = None
            rim, rests = (), ""
            substage = "approach"
            stage = self._stage(proprio, objects, index)
            wp = waypoint_for(proprio, objects, self.target, self.destination, stage,
                              rim_radius_m=self.rim_radius_m, rim_dz_m=self.rim_dz_m,
                              lift_height_m=self.lift_height_m,
                              place_height_m=self.place_height_m,
                              retreat_height_m=self.retreat_height_m,
                              attempts=self._grasp_attempts)
        self._waypoint = wp
        row = _Row(step=int(step), pos=np.asarray(proprio[0:3], np.float64).copy(), width=width,
                   subgoal=stage, substage=substage, holding=holding,
                   waypoint_cm=None if wp is None else rounded_cm(wp.offset_cm),
                   distance_cm=None if wp is None else wp.distance_cm,
                   yaw_err_deg=float(yaw_err_deg), rim=rim, rests=rests)

        previous = self._rows[-1] if self._rows else None
        self._rows.append(row)
        if stage not in self._done:
            self._done.append(stage)
        if row.distance_cm is not None and (self._closest is None
                                            or row.distance_cm < self._closest[0]):
            self._closest = (row.distance_cm, row.step)
        if previous is not None:
            self._path_cm += float(np.linalg.norm(row.pos - previous.pos)) * 100.0
            if previous.distance_cm is not None and row.distance_cm is not None:
                previous.effect_cm = previous.distance_cm - row.distance_cm
            if not previous.holding and holding:
                self._held += 1
            if self.instruction is None and previous.width > CLOSED_WIDTH >= width:
                self._grasp_attempts += 1
        self._record_events(previous, row)

    def answer(self, answers: dict | None) -> None:
        """What was actually decided from the state this tracker last rendered.

        The `rim` answer goes **into the plan's carry**, where the next selection reads it: the
        model says which way round the target to stand and the plan obeys it at the moments it
        chooses a candidate (the start, a blocked stage, a grasp that held nothing). An answer
        naming a letter this run was not offered, or a direction already spent, is ignored and
        the plan keeps its own order -- `expert._choose_candidate`, and `meta["rim_followed"]`
        records which of the two happened.
        """
        if self._rows and answers is not None:
            self._rows[-1].answers = dict(answers)
            self._carry = _with_executed(self._carry, answers, qids=self.qids)

    # ------------------------------------------------------------------------------ internals

    def _grasp_offset(self, proprio, objects: dict, closed: bool):
        """`(offset, reference)` for `skill.held`, from the tracker's own observations.

        The reference is latched the first decision the fingers are shut and dropped when they
        open again, which is the same moment `expert`'s plan records it -- so the two agree
        without the tracker having to know whether there is a plan.
        """
        if self.target is None or self.target not in objects:
            return None, None
        pos = np.asarray(objects[self.target]["pos"], dtype=np.float64).reshape(3)
        offset = pos - np.asarray(proprio, dtype=np.float64).reshape(-1)[0:3]
        if not closed:
            self._closed_offset = None
        elif self._closed_offset is None:
            self._closed_offset = [float(v) for v in offset]
        return offset, self._closed_offset

    def _stage(self, proprio, objects: dict, index: int) -> str:
        """v1's `phase` rules, with v2's `retreat` in front of them.

        `phase` has no `retreat`: v1's episode ends at `place` and the arm stays where it let go.
        A released object with the fingers open is a stage of its own -- it has its own waypoint
        (up and away) and its own `grip` answer (false, always) -- so it is decided here and
        everything else is delegated.
        """
        if self._released:
            return "retreat"
        name, _ = phase(proprio, objects, self.target, self.destination, self._start_z,
                        decision=max(index, 1))
        return normalise_phase(name)

    def _record_events(self, previous: _Row | None, row: _Row) -> None:
        if previous is None:
            if row.distance_cm is None:
                self._events.append(f"t={row.step} start")
            else:
                self._events.append(
                    f"t={row.step} start, waypoint {_mag(row.distance_cm)} cm away")
            return
        if row.distance_cm is not None:
            for milestone in EVENT_MILESTONES_CM:
                if milestone not in self._milestones and row.distance_cm <= milestone:
                    self._milestones.add(milestone)
                    self._events.append(f"t={row.step} within {milestone:g} cm")
        if not previous.holding and row.holding:
            self._events.append(f"t={row.step} fingers closed on {self._short(self.target)}")
        if previous.subgoal != row.subgoal:
            self._events.append(f"t={row.step} {row.subgoal}")

    def _short(self, name: str | None) -> str:
        return "the target" if name is None else short_id(name)

    # -------------------------------------------------------------------------------- facts

    def gold_answers(self) -> dict:
        """The expert's answer to every question at the decision this tracker last observed,
        from exactly the numbers the state prints -- rounded first, through `rounded_cm`."""
        row = self._rows[-1]
        if row.waypoint_cm is None:
            raise ValueError("no waypoint: the target is not in the scene")
        return labels_from_waypoint(row.waypoint_cm, round(row.yaw_err_deg, 1), row.holding,
                                    row.subgoal, self.tolerance_cm,
                                    yaw_tolerance=self.yaw_tolerance_deg,
                                    arrival=self.arrival_cm, rim=self._rim_rows(row),
                                    qids=self.qids)

    def _rim_rows(self, row: "_Row") -> tuple:
        """The candidate block **as the state prints it** -- the verdicts and nothing else, which
        is exactly what `v2.parse` reads back out of the text."""
        return tuple({"letter": r["letter"], "fits": bool(r["fits"]), "tried": bool(r["tried"])}
                     for r in row.rim)

    def _history_rows(self) -> list[str]:
        out = []
        for row in self._rows[-(self.history_k + 1):-1][-self.history_k:]:
            if row.answers is None:
                continue
            moves = [f"{axis[-1]}{row.answers[axis]}{row.answers.get('size_' + axis[-1], '')[:1]}"
                     for axis in AXIS_SLOT if row.answers.get(axis) in ("+", "-")]
            move = " ".join(moves) if moves else "hold"
            # The shared-step ablation names one size for the whole move; the per-axis set has
            # already put each axis's initial on its own token above.
            size = row.answers.get("step")
            move = f"{move} {size}" if size else move
            fingers = "closed" if row.answers.get("grip") else "open"
            if row.effect_cm is None:
                effect = "?"
            elif row.effect_cm >= 0:
                effect = f"{_mag(row.effect_cm)} cm closer"
            else:
                effect = f"{_mag(row.effect_cm)} cm farther"
            out.append(f"t={row.step} {move}, {fingers} -> {effect}")
        return out

    def counters_line(self) -> str:
        if self._closest is None:
            closest = "Closest so far n/a."
        else:
            closest = f"Closest so far {_mag(self._closest[0])} cm at t={self._closest[1]}."
        net_cm = 0.0
        if len(self._rows) > 1:
            net_cm = float(np.linalg.norm(self._rows[-1].pos - self._rows[0].pos)) * 100.0
        looping = (len(self._rows) >= LOOP_MIN_DECISIONS and self._path_cm > 0.0
                   and net_cm < LOOP_RATIO * self._path_cm)
        return (f"Attempts: grasps {self._grasp_attempts}, held {self._held}. {closest} "
                f"Moved {_mag(self._path_cm)} cm, net {_mag(net_cm)} cm "
                f"({'looping' if looping else 'not looping'}).")

    def lines(self, objects: dict, proprio) -> list[str]:
        """The tracker's own lines of §4's template, in order."""
        row = self._rows[-1]
        target = self._short(self.target) if self.target else "not chosen"
        destination = self._short(self.destination) if self.destination else "not chosen"
        chosen = f" (chosen at t={self.chosen_at})" if self.chosen_at is not None else ""
        done = ", ".join(self._done[:-1]) if len(self._done) > 1 else "nothing"
        left = max(self.horizon - self.decisions, 0)
        width_cm = row.width * 100.0
        openness = "open" if row.width > CLOSED_WIDTH else "closed"
        holds = self._short(self.target) if row.holding else "nothing"

        lines = [
            f"Target: {target}{chosen}. Destination: {destination}.",
            f"Subgoal so far: {row.subgoal} ({row.substage}). Done: {done}. "
            f"Decision {self.decisions} of {self.horizon}; {left} left.",
            f"Gripper: {openness} {_mag(width_cm)} cm; holding {holds}.",
        ]
        # A line whose question is not asked is tokens for zero bits, so each of these is
        # printed exactly when its own question is in the set this run serves -- which is how a
        # checkpoint trained before either of them keeps reading the text it was trained on.
        if "yaw" in self.qids:
            lines.append(f"Wrist yaw error: {_cm(row.yaw_err_deg)} deg "
                         f"[tolerance {_mag(self.yaw_tolerance_deg)}]")
        if "rim" in self.qids and row.rim:
            if row.rests:
                lines.append(row.rests)
            # **Short lines, and it is the token budget that made them short.** The block adds
            # one line per offered candidate to every motion path, and a path is the state plus
            # one question's instructions plus one criterion: measured with the recipe
            # environment's own Qwen3 tokenizer, the long form ("+x side, wrist turn -90 deg,
            # room 2.9 cm") put the worst motion path at 922 tokens of the 1024 budget. The
            # words that went are the ones the reader supplies anyway -- the units and the nouns
            # -- and none of them is read by `v2.parse`, which takes the letter and the verdict
            # after the arrow. 886 tokens.
            lines.append(f"Grasp candidates (outer finger needs "
                         f"{FINGER_HALF_WIDTH * 100:.1f} cm):")
            for r in row.rim:
                lines.append(f"  {r['letter']}: {r['side']}, turn "
                             f"{format(int(round(r['turn'])), '+d')}, "
                             f"room {_mag(r['room_cm'])} cm -> {_verdict(r)}")
        return lines

    def other_objects_line(self, objects: dict, proprio) -> str:
        # **The movable scene only.** The privileged state carries the room's furniture now
        # (`robojev.scene`), and it is for the planner: putting a cabinet in this line
        # would change every state string of every task, including the ones a deployed
        # checkpoint was trained on.
        objects = movable(objects)
        eef = np.asarray(proprio, dtype=np.float64).reshape(-1)[0:3]
        taken: set[str] = set()
        parts = []
        for name, pose in objects.items():
            sid = short_id(name, taken)
            taken.add(sid)
            if name == self.target:
                continue
            delta = (np.asarray(pose["pos"], dtype=np.float64).reshape(3) - eef) * 100.0
            parts.append(f"{sid} x {_cm(delta[0])} y {_cm(delta[1])} z {_cm(delta[2])}")
        return "Other objects: " + ("; ".join(parts) if parts else "none") + "."

    def events_line(self) -> str:
        return "Events: " + " | ".join(self._events[-self.max_events:])

    def history_line(self) -> str:
        rows = self._history_rows()
        if not rows:
            return f"Last {self.history_k}: none yet."
        return f"Last {self.history_k}: " + " | ".join(rows)


# ----------------------------------------------------------------------------- the serialiser


HEADER: str = ("Robot: Franka Panda, gripper-relative frame; x right, y forward, z up; "
               "distances in cm.")


def serialise_v2(obs_state8, privileged: dict, instruction: str, memory: TrackerV2, *,
                 annotate: bool | None = None) -> str:
    """§4's template, exactly, for the decision `memory` last observed.

    `obs_state8` is LIBERO's own 8-d proprio vector and `privileged` the task-object poses in
    world metres -- the same two inputs v1's `serialise` takes, so a harvested row and a live
    request are still the same function of the same numbers. Every distance is converted here and
    printed as **gripper-relative signed centimetres**: the model reads "the rim is 5.5 cm
    forward", never two coordinate triples it has to subtract (failure F4).

    `annotate` is the §4 ablation. With it, each axis line states the relation its own question
    asks and its verdict (`-> aligned` / `-> not aligned`) and the task is reading; without it,
    the line is the number alone and the task is one comparison. Gate G3 is run both ways and the
    parser in `robojev.parse` reads either, which is the proof that the annotation is
    an aid and not the label's source.
    """
    if memory.decisions == 0:
        raise ValueError("observe() a decision point before serialising it")
    annotate = memory.annotate if annotate is None else bool(annotate)
    row = memory.last_row
    wp = memory.waypoint
    lines = [HEADER, f"Task: {instruction}"]
    lines.extend(memory.lines(privileged, obs_state8))

    if wp is None or row.waypoint_cm is None:
        lines.append("Waypoint: none -- the target is not in the scene.")
    else:
        offset = row.waypoint_cm
        lines.append(
            f"Waypoint ({wp.label}): x {_cm(offset[0])}, y {_cm(offset[1])}, z {_cm(offset[2])}"
            f"   [tolerance {_mag(memory.tolerance_cm)}; arrived within "
            f"{_mag(memory.arrival_cm)}]")
        for axis, index in AXIS_SLOT.items():
            name = axis[-1]
            value = offset[index]
            if annotate:
                inside = abs(float(value)) <= memory.tolerance_cm
                verdict = ("is inside tolerance -> aligned" if inside
                           else "is outside tolerance -> not aligned")
                lines.append(f"  {name}: {_cm(value)} {verdict}")
            else:
                lines.append(f"  {name}: {_cm(value)}")
        remaining = largest_remaining(offset, memory.tolerance_cm)
        lines.append(f"Largest remaining offset: {_mag(remaining)} cm "
                     f"({step_range_words(step_size_for(remaining))}).")
        # The bands every `size_*` question is answered against, printed once: without them the
        # size label would not be a function of the text (principle 1).
        lines.append(band_words())

    lines.append(memory.other_objects_line(privileged, obs_state8))
    lines.append(memory.events_line())
    lines.append(memory.counters_line())
    lines.append(memory.history_line())
    return "\n".join(lines)


def objects_cm(obs_state8, privileged: dict) -> dict[str, tuple[float, float, float]]:
    """`{short_id: (x, y, z)}` in gripper-relative centimetres -- what
    `robojev.questions.scene_candidates` builds the `target`/`destination` candidates
    from, in the same units and from the same numbers as every other line of the state."""
    eef = np.asarray(obs_state8, dtype=np.float64).reshape(-1)[0:3]
    taken: set[str] = set()
    out: dict[str, tuple[float, float, float]] = {}
    # The grounding question asks which *object* the sentence means; the furniture is not a
    # candidate and printing it would change every scene block (`robojev.scene`).
    for name, pose in movable(privileged).items():
        sid = short_id(name, taken)
        taken.add(sid)
        delta = (np.asarray(pose["pos"], dtype=np.float64).reshape(3) - eef) * 100.0
        out[sid] = (float(delta[0]), float(delta[1]), float(delta[2]))
    return out
