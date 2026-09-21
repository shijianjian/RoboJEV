"""`robojev.skill`: the plan executor, on a toy skill that knows nothing about robots.

The point of the module is that the four rules are written once and are the same rules for every
stage, so these tests use a two-stage plan over a one-dimensional "arm" rather than the expert's
pick-and-place: if `Progress`, the candidate list, the outcome check and the dwell need the bowl
to be testable, they are not the rules the design claims they are. `test_expert.py` is where they
meet the real task.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from robojev import skill


class Ctx:
    """Everything a stage callable here asks about: where the hand is, and two facts."""

    def __init__(self, x: float, *, closed: bool = False, offset=(0.0, 0.0, 0.0),
                 reference=None, goal=10.0):
        self.eef = np.array([float(x), 0.0, 0.0])
        self.closed = closed
        self.offset = np.asarray(offset, dtype=float)
        self.reference = reference
        self.target = np.array([float(goal), 0.0, 0.0])


def toy(*, candidates: int = 1, check=None, on_blocked: str | None = "travel",
        on_failed: str | None = "travel") -> skill.Plan:
    """`travel` (moving, with an optional outcome) -> `wait` (a dwell) -> `done`."""
    return skill.Plan(
        stages=(
            skill.Stage("travel", "reach", False, goal=lambda c: c.target,
                        arrived=lambda c: abs(float(c.target[0] - c.eef[0])) < 0.5,
                        check=check, on_done="wait", on_blocked=on_blocked,
                        on_failed=on_failed,
                        on_exit=lambda c: {"arrived_at": round(float(c.eef[0]), 3)}),
            skill.Stage("wait", "grasp", True, dwell=3, on_done="done"),
            skill.Stage("done", "retreat", False, on_done="done"),
        ),
        start="travel", candidates=candidates, per_attempt=("arrived_at",),
    )


def run(plan, carry, positions, **kwargs):
    """Drive the plan through a list of hand positions; returns one `Step` per decision."""
    steps = []
    for x in positions:
        out = skill.step(plan, carry, Ctx(x, **kwargs))
        carry = out.carry
        steps.append(out)
    return steps


# ------------------------------------------------------------------------------ the carry


def test_the_carry_is_json_able_and_stays_so():
    """DAgger persists it between rounds and the tracker keeps it beside a rendered state: a
    numpy scalar or a dataclass in here is a row that cannot be written."""
    plan = toy(candidates=3)
    carry = skill.new_carry(plan, roles={"target": "bowl_1"})
    for out in run(plan, carry, [0.0, 0.1, 0.1, 0.1, 0.1, 9.9, 9.9, 9.9, 9.9]):
        assert json.loads(json.dumps(out.carry)) == out.carry


def test_the_executor_does_not_mutate_the_carry_it_is_given():
    plan = toy()
    carry = skill.new_carry(plan)
    before = dict(carry)
    skill.step(plan, carry, Ctx(0.0))
    assert carry == before


def test_a_fresh_carry_starts_at_the_plan_s_own_start():
    plan = toy(candidates=4)
    carry = skill.new_carry(plan, roles=None)
    assert carry["name"] == "travel"
    assert (carry["candidate"], carry["attempts"], carry["stuck"]) == (0, 0, 0)
    assert carry["best"] is None and carry["arrived_at"] is None
    assert carry["roles"] is None


# --------------------------------------------------------------------------- one progress rule


def test_a_stage_that_is_closing_on_its_goal_is_never_blocked():
    """Well beyond `PROGRESS_PATIENCE` decisions: progress is the only thing measured, so a long
    stage is not a suspicious one. There is no clock anywhere in the executor."""
    plan = toy()
    steps = run(plan, skill.new_carry(plan), [float(x) for x in range(0, 9)])
    assert not any(s.blocked for s in steps)
    assert all(s.carry["name"] == "travel" for s in steps)


def test_a_stage_that_stops_closing_is_blocked_after_the_measured_patience():
    """0.3 cm in 3 decisions, from the trace of 106 descents. A stall oscillates rather than
    stands still, which is why the rule measures against the *closest so far* and not against the
    decision before: 0.29, 0.28, 0.29 is not progress."""
    plan = toy(on_blocked=None)
    carry = skill.new_carry(plan)
    steps = run(plan, carry, [0.0, 0.0029, 0.0, 0.0029])
    assert [s.blocked for s in steps] == [False, False, False, True]
    assert steps[2].carry["stuck"] == skill.PROGRESS_PATIENCE - 1


def test_progress_is_measured_against_the_closest_the_stage_has_come():
    """A decision that gets closer than anything before resets the count; one that merely
    recovers ground it had already covered does not."""
    plan = toy(on_blocked=None)
    steps = run(plan, skill.new_carry(plan), [0.0, 1.0, 0.0, 1.0, 0.0])
    # The last decision is the third without progress: it blocks, and blocking starts the next
    # stage's count from nothing.
    assert [s.carry["stuck"] for s in steps] == [0, 0, 1, 2, 0]
    assert [s.blocked for s in steps] == [False] * 4 + [True]


# ------------------------------------------------------------------ one notion of alternatives


def test_a_blocked_stage_backs_off_and_takes_the_next_candidate():
    plan = toy(candidates=3)
    carry = skill.new_carry(plan)
    steps = run(plan, carry, [0.0, 0.0, 0.0, 0.0])
    assert steps[-1].blocked and not steps[-1].exhausted
    assert steps[-1].carry["candidate"] == 1
    assert steps[-1].carry["name"] == "travel"          # the stage it backs off to
    # A new candidate stands somewhere else, so the progress it had made says nothing about the
    # progress it is about to make.
    assert steps[-1].carry["best"] is None and steps[-1].carry["stuck"] == 0
    # ...and what the abandoned attempt recorded is dropped with it.
    assert steps[-1].carry["arrived_at"] is None


def test_the_candidates_run_out_and_then_the_best_pose_reached_is_accepted():
    """The one "proceed anyway" the design keeps, and it is reached only after every alternative
    was tried -- not after a clock ran out, which is what closed the fingers 2 cm above the rim."""
    plan = toy(candidates=2)
    carry = skill.new_carry(plan)
    steps = run(plan, carry, [0.0] * 8)
    switches = [i for i, s in enumerate(steps) if s.blocked]
    assert len(switches) == 2
    assert steps[switches[0]].carry["candidate"] == 1 and not steps[switches[0]].exhausted
    assert steps[switches[1]].exhausted
    assert steps[switches[1]].carry["candidate"] == 1          # no third candidate to take
    assert steps[switches[1]].carry["name"] == "wait"          # it goes on regardless


def test_a_stage_with_no_alternative_keeps_steering():
    """Being blocked is only actionable when there is something else to try. Where there is not
    -- the carry, the lower -- the executor's two options are to carry on and to hand over early,
    and handing over early is the worse of them: a `lower` that releases 2.4 cm short puts the
    bowl beside the plate and scores nothing, while one that keeps trying sometimes arrives. The
    episode's horizon is the only clock, and it is the episode's rather than the stage's."""
    plan = toy(candidates=4, on_blocked=None)
    steps = run(plan, skill.new_carry(plan), [0.0] * 8)
    assert all(s.carry["name"] == "travel" for s in steps)
    assert steps[-1].carry["candidate"] == 0                   # untouched: nothing was tried
    assert [s.blocked for s in steps].count(True) == 2         # reported every patience, not once
    # ...and it still arrives if it ever gets there.
    assert skill.step(plan, steps[-1].carry, Ctx(9.9)).carry["name"] == "wait"


def test_a_failed_check_takes_the_next_candidate_through_the_back_off_stage():
    """The grasp did not hold. The plan does not stay where it is and it does not carry on: it
    goes back to the hover and comes down on the next candidate."""
    plan = toy(candidates=3, check=lambda c: c.closed)
    carry = skill.new_carry(plan)
    carry["name"] = "travel"
    out = skill.step(plan, carry, Ctx(5.0, closed=False))
    assert out.failed and not out.blocked
    assert out.carry["name"] == "travel" and out.carry["candidate"] == 1
    assert out.carry["attempts"] == 1


def test_only_a_failed_outcome_counts_as_an_attempt():
    """`attempts` is a fact about grasps -- the state text prints it and the re-grounding rule
    reads it -- and taking the other side of a rim is not a grasp. The candidate pointer moves
    for both; the attempt counter only for an outcome that did not hold."""
    plan = toy(candidates=4)
    steps = run(plan, skill.new_carry(plan), [0.0] * 4)
    assert steps[-1].blocked and steps[-1].carry["attempts"] == 0


def test_an_outcome_is_checked_at_every_decision_and_not_only_on_the_way_out():
    """A bowl that leaves the fingers half way through a carry is a failed carry from that
    decision, not from the one that arrives over the plate."""
    plan = toy(candidates=3, check=lambda c: c.closed)
    carry = skill.new_carry(plan)
    ok = skill.step(plan, carry, Ctx(0.0, closed=True))
    assert not ok.failed and ok.carry["name"] == "travel"
    lost = skill.step(plan, ok.carry, Ctx(1.0, closed=False))
    assert lost.failed                       # mid-stage, nowhere near arriving


def test_a_check_outranks_an_arrival():
    plan = toy(candidates=3, check=lambda c: c.closed)
    out = skill.step(plan, skill.new_carry(plan), Ctx(10.0, closed=False))
    assert out.failed and out.carry["name"] == "travel"


# ------------------------------------------------------------------------------ dwell stages


def test_a_dwell_stage_holds_still_and_leaves_after_its_own_decisions():
    plan = toy()
    carry = skill.new_carry(plan)
    carry["name"] = "wait"
    for tick in range(3):
        out = skill.step(plan, carry, Ctx(4.0))
        assert out.goal == pytest.approx([4.0, 0.0, 0.0])      # the hand's own position
        assert out.grip is True                                 # the stage's own finger command
        assert out.carry["name"] == ("done" if tick == 2 else "wait")
        carry = out.carry
    assert not any_blocked(plan)


def any_blocked(plan) -> bool:
    """A dwell stage can neither block nor fail: it has no goal to fall short of."""
    carry = skill.new_carry(plan)
    carry["name"] = "wait"
    return any(skill.step(plan, carry, Ctx(4.0)).blocked for _ in range(5))


def test_a_terminal_stage_names_itself_and_stays():
    plan = toy()
    carry = skill.new_carry(plan)
    carry["name"] = "done"
    for _ in range(3):
        out = skill.step(plan, carry, Ctx(1.0))
        assert out.carry["name"] == "done"
        carry = out.carry


def test_ticks_count_decisions_in_one_stage_and_reset_when_it_changes():
    plan = toy()
    steps = run(plan, skill.new_carry(plan), [0.0, 1.0, 2.0, 9.8])
    # The carry a decision *returns* has counted that decision, so the first one leaves `ticks`
    # at 1 and the decision that changes stage leaves it at 0.
    assert [s.carry["ticks"] for s in steps] == [1, 2, 3, 0]
    assert [s.carry["decisions"] for s in steps] == [1, 2, 3, 4]


def test_a_stage_records_what_the_next_one_is_measured_from_as_it_leaves():
    """`on_exit` is where the reference an outcome is judged against is latched -- the offset from
    the hand to the object once the fingers have settled on it."""
    plan = toy()
    steps = run(plan, skill.new_carry(plan), [0.0, 5.0, 9.8])
    assert steps[0].carry["arrived_at"] is None            # still travelling
    assert steps[-1].carry["name"] == "wait"
    assert steps[-1].carry["arrived_at"] == 9.8


# -------------------------------------------------------------------------------- one `held`


def test_held_is_the_fingers_and_the_object_not_moving_relative_to_them():
    assert skill.held(True, (0.0, 0.0, -0.03), (0.0, 0.0, -0.03), 0.04)
    assert skill.held(True, (0.0, 0.01, -0.03), (0.0, 0.0, -0.03), 0.04)
    assert not skill.held(True, (0.0, 0.0, -0.13), (0.0, 0.0, -0.03), 0.04)
    assert not skill.held(False, (0.0, 0.0, -0.03), (0.0, 0.0, -0.03), 0.04)


def test_nothing_is_held_before_the_grasp_has_been_made():
    """No reference is the honest answer to "is it in the hand" before the fingers have shut on
    anything -- and it is what the old rule got wrong, printing "holding bowl_1" for a hand that
    was merely standing near one."""
    assert not skill.held(True, (0.0, 0.0, -0.03), None, 0.04)


def test_held_does_not_care_which_way_the_skill_then_goes():
    """The fixture tasks carry *downhill*: a bowl that starts on a stove and ends on a plate is
    below the height it was grasped at for the whole carry, and a rule testing that it rose calls
    the correct carry a drop (measured: those three tasks at 0/10)."""
    assert skill.held(True, (0.01, 0.0, -0.03), (0.0, 0.0, -0.03), 0.04)


# ------------------------------------------------------------------------ the plan is checked


def test_a_transition_naming_a_stage_that_does_not_exist_is_refused_when_the_plan_is_built():
    with pytest.raises(skill.PlanError, match="retreet"):
        skill.Plan(stages=(skill.Stage("a", "reach", False, dwell=1, on_done="retreet"),),
                   start="a")


def test_a_plan_that_starts_nowhere_is_refused():
    with pytest.raises(skill.PlanError, match="starts in"):
        skill.Plan(stages=(skill.Stage("a", "reach", False, dwell=1, on_done="a"),), start="b")


def test_a_dwell_stage_may_not_carry_an_outcome():
    """It cannot fail to arrive, so it cannot fail its outcome either, so the check would be a
    rule that never fires -- which is how a phase machine grows a branch nobody can measure."""
    with pytest.raises(skill.PlanError, match="check"):
        skill.Plan(stages=(skill.Stage("a", "reach", False, dwell=1, on_done="a",
                                       check=lambda c: True),), start="a")
