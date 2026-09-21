"""One small plan executor: a skill's stages as data, and the four rules that drive them.

Why this module exists, in one sentence: `expert.decide` used to be a hand-written if/elif phase
machine in which *three* ideas were each implemented several times, differently -- **am I making
progress**, **is the object held**, **what do I try next** -- because every fix so far had been a
new constant in one branch (`PHASE_PATIENCE`, `DESCEND_PATIENCE`, `LIFT_PATIENCE`, `MAX_RIM_FLIPS`,
`RETRY_DZ` indexed by attempts, `risen`, `carrying`, `_within_fingers`). Each of those was right
about the failure it was written for and blind to the same failure one stage over: the measured
stalled *descent* was caught in six decisions and the measured stalled *approach*, ten episodes of
the same table, was not noticed at all.

So the stages become data and the rules become one each
(`docs/DESIGN.md` §2):

1. **Progress.** Every moving stage's distance to its own goal must improve by `PROGRESS_EPS`
   within `PROGRESS_PATIENCE` decisions, measured against the closest it has already come (a stage
   that oscillates is not progressing). It does not: the stage is **blocked**. There is no other
   timeout anywhere -- no stage has a clock.
2. **Alternatives.** One ordered list of candidates belongs to the whole skill, and its length is
   all this module knows about it (`Plan.candidates`); what a candidate *is* -- a side of a rim, a
   grasp height -- is the task's business. A blocked stage or a failed check marks the current
   candidate as spent, backs off through `on_blocked`/`on_failed` and takes the next. When the
   list is exhausted the executor **accepts the best pose it reached** and goes on through
   `on_done`: the one "proceed anyway" left, and it is reached only after every alternative was
   tried rather than after a clock ran out. A stage that has no alternative worth taking says so
   in its own row -- `ACCEPT` to hand over at once, `None` to keep steering -- and `ACCEPT` is
   what the old `stalled and within 2 x XY_TOL` was approximating.
3. **Outcomes.** `Stage.check` is a fact that must hold at *every* decision of the stage, not one
   tested on the way out -- a bowl that leaves the fingers half way through a carry is a failed
   carry from that decision, not from the one that arrives over the plate.
4. **Dwell.** A stage with no `arrived` leaves after `dwell` decisions. That is the whole of
   "hold still and shut the fingers" and "hold still and open them", and (with a goal) of the
   retreat.

The executor knows nothing about bowls, rims, grippers or LIBERO. It reads exactly one attribute
of the context it is handed -- `ctx.eef`, the point it measures distances from -- and everything
else about the task arrives through the stages' own callables. `robojev.expert` is the one
task written against it today (`PICK_AND_PLACE`).

**The carry is a plain JSON-able dict** and stays one: DAgger persists it between rounds and
`TrackerV2` keeps it beside a rendered state, so a numpy array or a dataclass in here would be a
row that cannot be written. `step` is pure -- it mutates neither the carry nor the context -- for
the reason `decide` is: a label source that cannot be replayed is not a label source.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable

import numpy as np

#: **How much closer to its goal a stage has to get before it counts as moving**, in metres,
#: measured against the closest it has already come rather than against the decision before.
#:
#: Measured (docs/DESIGN.md §3: all 106 descents of the 100-episode
#: table traced, 11 of them stalls): a stalled descent's distance to its waypoint moves 0.2 cm a
#: decision, up and down, for as long as it is allowed to; a descent that is working takes
#: 1.7-5.0 cm out of that distance per decision. At 0.3 cm this catches 11 of 11 stalls at a
#: median of 6 decisions rather than 12; 0.5 cm fires on six healthy descents, 0.1 cm misses a
#: stall. The band is 0.2-0.3 and 0.3 is what the end-to-end tables were measured at.
PROGRESS_EPS: float = 0.003
#: How many consecutive decisions a stage may make no progress before it is called blocked. Three:
#: at two the same trace fires on eight healthy descents instead of five and saves one decision, at
#: five a stall is eight decisions old before it is noticed.
PROGRESS_PATIENCE: int = 3


#: **What being blocked means for a stage that has nothing else to stand on.** `on_blocked` and
#: `on_failed` name a stage to back off to before the next candidate is tried; `ACCEPT` says to
#: hand over through `on_done` from the best pose reached instead; `None` says there is nothing to
#: do about it and the stage keeps steering.
#:
#: All three are measured, not tastes (docs/DESIGN.md §2):
#:
#: * the **descent** takes the next candidate -- a rim point can be one the arm cannot stand on
#:   while the opposite end of the same axis is reachable to 0.4 cm;
#: * the **approach** accepts -- it stops 2-3 cm from a hover point whose descent steers at the
#:   same x and y, so descending from there costs nothing while going round the bowl costs six
#:   decisions and loses episodes;
#: * the **lower** and the **carry** keep steering -- releasing 2.4 cm short of the place point
#:   puts the bowl beside the plate and scores nothing, while carrying on sometimes arrives. The
#:   episode's horizon is the only clock, and it is the episode's rather than the stage's.
ACCEPT: str = "<accept the best pose reached>"


class PlanError(ValueError):
    """A plan that cannot be executed: a transition naming a stage the plan does not have.

    Raised when the plan is built, not when a decision reaches the dangling name -- a controller
    that stops halfway through an episode because nobody spelled `retreat` the same way twice is
    a bug that should never reach a simulator.
    """


@dataclasses.dataclass(frozen=True)
class Stage:
    """One sub-stage of a skill, as data.

    `name` is what the state text prints (`approach`, `descend`, `close`, ...) and `subgoal` the
    coarser vocabulary the model answers in. `grip` is the plan's own finger command while in the
    stage -- a fact about the stage, not about the geometry, which is why the expert's `grip`
    answer is a latch and not a prediction.

    `goal(ctx)` is the point this stage steers to, or `None` for a stage that holds still.
    `residual(ctx)` is **how far the stage still is from done beyond the hand's position**, in the
    same metres, and it is what lets the progress rule watch something that is not a translation:
    a hand standing still over the rim while the wrist turns onto it is making progress, and a
    rule that only measured the distance to a point would call it blocked three decisions in and
    descend with the fingers pointing the wrong way. Everything else about the stage is unchanged
    by it -- the goal is still a point and the arm is still steered at it.

    `arrived(ctx)` is what ends it, or `None` for a stage that ends after `dwell` decisions.
    `check(ctx)` is an outcome that must hold throughout it. `on_exit(ctx)` is the carry fields the
    stage writes as it leaves -- the reference a later check is measured from, typically.

    `on_done` is where the stage goes when it arrives. `on_blocked` and `on_failed` say what
    being blocked, or losing the outcome, means here: the name of a stage to back off to before
    the next candidate, `ACCEPT` to hand over through `on_done` from the best pose reached, or
    `None` to keep steering.
    """

    name: str
    subgoal: str
    grip: bool
    goal: Callable[[Any], Any] | None = None
    residual: Callable[[Any], float] | None = None
    arrived: Callable[[Any], bool] | None = None
    dwell: int = 0
    check: Callable[[Any], bool] | None = None
    on_done: str = ""
    on_blocked: str | None = None
    on_failed: str | None = None
    on_exit: Callable[[Any], dict] | None = None


@dataclasses.dataclass(frozen=True)
class Plan:
    """A whole skill: its stages, where it starts, how many alternatives it may try.

    `candidates` is a **count** and deliberately nothing more. The executor's business is that
    there are alternatives and that they run out; which side of which rim at which height the
    n-th one is belongs to the task (`expert.GRASP_CANDIDATES`), and a plan executor that knew
    would be a phase machine again.

    `per_attempt` names the carry fields a new candidate invalidates -- what was recorded about
    the attempt that has just been abandoned. They are set to `None`, so the carry stays a dict
    with a fixed set of keys and a reader never has to know how far an episode got.
    """

    stages: tuple[Stage, ...]
    start: str
    candidates: int = 1
    per_attempt: tuple[str, ...] = ()
    #: `choose(carry, ctx) -> int | None`: which alternative to take next, when the task has an
    #: opinion that is not "the next untried one in order" -- a model's answer, typically. `None`
    #: from it, or an index that is not untried, falls back to the order. The executor's own
    #: business is still only that there are alternatives and that they run out.
    choose: Callable[[dict, Any], int | None] | None = None

    def __post_init__(self) -> None:
        names = {stage.name for stage in self.stages}
        if self.start not in names:
            raise PlanError(f"the plan starts in {self.start!r}, which is not one of {sorted(names)}")
        for stage in self.stages:
            for field in ("on_done", "on_blocked", "on_failed"):
                target = getattr(stage, field)
                if target not in (None, "", ACCEPT) and target not in names:
                    raise PlanError(
                        f"stage {stage.name!r} leaves through {field}={target!r}, which is not "
                        f"one of {sorted(names)}"
                    )
            if stage.arrived is None and stage.check is not None:
                raise PlanError(
                    f"stage {stage.name!r} holds still for {stage.dwell} decisions and has a "
                    f"check: a stage that cannot fail to arrive cannot fail its outcome either"
                )
        if self.candidates < 1:
            raise PlanError(f"a plan needs at least one candidate, got {self.candidates}")

    def stage(self, name: str) -> Stage:
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise PlanError(f"no stage named {name!r} in this plan")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(stage.name for stage in self.stages)

    @property
    def subgoals(self) -> dict[str, str]:
        """`{stage name: subgoal}` -- the map the state text and the label function share."""
        return {stage.name: stage.subgoal for stage in self.stages}


@dataclasses.dataclass(frozen=True)
class Step:
    """What the executor decided for one decision point.

    `goal` is where to steer (the hand's own position for a stage that holds still, so the caller
    needs no special case for it), `grip` the finger command, `carry` the next decision's carry.
    `blocked` and `failed` are reported rather than inferred, because they are what a measurement
    counts: an episode that spends its candidates is a different episode from one that arrives.
    """

    stage: Stage
    goal: np.ndarray
    grip: bool
    carry: dict
    blocked: bool = False
    failed: bool = False
    exhausted: bool = False
    distance: float = 0.0


def new_carry(plan: Plan, **fields: Any) -> dict:
    """The carry an episode starts with: the first stage, no progress, the first candidate.

    `fields` are the task's own (the roles it latched, the geometry it recorded); they sit in the
    same flat dict, which is what keeps the carry one JSON object rather than two.
    """
    carry = {
        "name": plan.start,
        "ticks": 0,          # decisions spent in `name`
        "decisions": 0,      # decisions in the episode
        "candidate": 0,      # which alternative is in force
        "tried": [],         # the alternatives already spent, in the order they were spent
        "attempts": 0,       # outcomes that did not hold: grasps tried and found empty
        "best": None,        # the closest this stage has come to its goal, in metres
        "stuck": 0,          # consecutive decisions it has not improved on that
    }
    carry.update({key: None for key in plan.per_attempt})
    carry.update(fields)
    return carry


def held(closed: bool, offset, reference, tolerance: float) -> bool:
    """**The** definition of "the object is in the hand", used by every stage that needs it.

    The fingers are shut *and* the object is **no farther from the hand than the grasp left it**.
    Dropping is a thing that puts distance between the hand and the object -- the hand climbs and
    the object stays where it was -- so the quantity to watch is the *length* of the offset, not
    the offset itself. The two rules it replaces each assumed which way the skill then goes: "the
    object rose since the close" calls a correct downhill carry a drop (the fixture tasks, 0/10),
    and "the object is within 8 cm of the hand horizontally" says a hand that shut on air is
    holding the bowl it is standing next to.

    **And a rim grasp re-seats.** The third rule -- "the offset has not *changed* by more than the
    tolerance" -- was measured calling a good grasp a drop: on the drawer task the bowl closed at
    4.6 cm from the grip site and slid to 0.9 cm as the lift took its weight, a 3.7 cm change with
    the bowl 7.5 cm off the drawer floor and plainly in the fingers (trace
    `scratch/v2-drawer/round3/trace_t4_i3.json`, decision 15). The plan then opened the fingers
    over the cabinet and spent the rest of the episode chasing what it had dropped by saying so.
    An object settling *into* the fingers is the grasp improving; only an object leaving them is a
    drop, and this asks about that one direction.

    `reference` is `None` until the grasp has been made, and nothing is held before then.
    """
    if not closed or reference is None:
        return False
    offset = np.asarray(offset, dtype=np.float64).reshape(3)
    reference = np.asarray(reference, dtype=np.float64).reshape(3)
    return bool(float(np.linalg.norm(offset))
                <= float(np.linalg.norm(reference)) + float(tolerance))


def step(plan: Plan, carry: dict, ctx: Any) -> Step:
    """One decision: where to steer, what the fingers do, and the carry for the next one.

    Pure. The order of the rules is the design and is worth reading as such:

        an outcome that stopped holding   -> failed   (`on_failed`)
        arrived                           -> done     (`on_done`)
        no progress for PROGRESS_PATIENCE -> blocked   (`on_blocked`)
        otherwise                         -> keep going

    The check comes first because a carry whose bowl left the fingers is a failed carry from that
    decision, not from the one that reaches the plate; arrival comes before blocked because a
    stage that has arrived cannot be stuck. What `on_failed`/`on_blocked` then mean is the
    stage's own to say -- see `ACCEPT`.
    """
    stage = plan.stage(carry["name"])
    eef = np.asarray(ctx.eef, dtype=np.float64).reshape(3)
    goal = eef.copy() if stage.goal is None else np.asarray(stage.goal(ctx), np.float64).reshape(3)
    # The one quantity the progress rule watches: how far this stage still is from done. The
    # hand's distance to the goal, plus whatever else the stage says it is still waiting for.
    distance = float(np.linalg.norm(goal - eef))
    if stage.residual is not None:
        distance += float(stage.residual(ctx))

    best = carry.get("best")
    stuck = int(carry.get("stuck") or 0)
    blocked = failed = False
    nxt = stage.name

    if stage.arrived is None:
        # A dwell stage: it cannot fail to arrive, it waits. `dwell = 0` leaves at once, which is
        # what a terminal stage (`done`) wants -- it names itself in `on_done` and stays.
        if carry["ticks"] + 1 >= stage.dwell:
            nxt = stage.on_done
    else:
        improved = best is None or distance < float(best) - PROGRESS_EPS
        stuck = 0 if improved else stuck + 1
        best = distance if best is None else min(distance, float(best))
        if stage.check is not None and not stage.check(ctx):
            failed = True
        elif stage.arrived(ctx):
            nxt = stage.on_done
        elif stuck >= PROGRESS_PATIENCE:
            blocked = True

    updates: dict[str, Any] = {}
    exhausted = False
    if blocked or failed:
        back = stage.on_failed if failed else stage.on_blocked
        # **Spent, not incremented.** Which alternative comes next is the task's to say (a model
        # answers it here), so what the executor keeps is the set that has been used up; the
        # order is only the fallback.
        spent = list(carry.get("tried") or [])
        if int(carry["candidate"]) not in spent:
            spent.append(int(carry["candidate"]))
        untried = [i for i in range(plan.candidates) if i not in spent]
        exhausted = not untried
        if back is None:
            # Nothing to do about it: the stage keeps steering (see `ACCEPT`).
            pass
        elif back == ACCEPT or exhausted:
            # **Accept the best pose reached** -- because this stage never had an alternative
            # worth taking, or because every one of them has been tried. The one "proceed anyway"
            # the design keeps, and it is reached by exhausting the alternatives rather than by a
            # clock running out.
            nxt = stage.on_done
        else:
            nxt = back
            picked = plan.choose(carry, ctx) if plan.choose is not None else None
            updates["candidate"] = picked if picked in untried else untried[0]
            updates["tried"] = spent
            updates.update({key: None for key in plan.per_attempt})
        if failed:
            updates["attempts"] = int(carry["attempts"]) + 1
        if (blocked or failed) and back is not None and back != ACCEPT and exhausted:
            updates["tried"] = spent

    if nxt != stage.name and stage.on_exit is not None:
        updates.update(stage.on_exit(ctx))
    if nxt != stage.name or blocked or failed:
        # A new stage steers at a new goal and a new candidate stands somewhere else: either way
        # the distance already walked says nothing about the one about to be.
        best, stuck = None, 0

    nextcarry = dict(carry)
    nextcarry.update(updates)
    nextcarry["best"] = best
    nextcarry["stuck"] = stuck
    nextcarry["name"] = nxt
    nextcarry["ticks"] = (carry["ticks"] + 1) if nxt == stage.name else 0
    nextcarry["decisions"] = int(carry.get("decisions") or 0) + 1
    return Step(stage=stage, goal=goal, grip=stage.grip, carry=nextcarry, blocked=blocked,
                failed=failed, exhausted=exhausted, distance=distance)


__all__ = ["ACCEPT", "PROGRESS_EPS", "PROGRESS_PATIENCE", "Plan", "PlanError", "Stage", "Step",
           "held", "new_carry", "step"]
