"""A scripted expert that *finishes* LIBERO-Spatial pick-and-place in the decision vocabulary.

Why this exists. The latest NanoJev took ViZDoom Basic from 56/128 to 128/128 with the state text
byte-identical, purely by changing the **label source** to one authoritative expert action per
decision (docs/DESIGN.md). Our labels come from human
demonstrations replayed at 20 Hz, and a teleoperator's derivative is not a function of the scene:
two decision points 0.25 s apart in the same reach can carry different `translate` golds. This
module is the other kind of source -- a controller that is *right by construction* at every state,
including the states a half-trained model wanders into, which is what DAgger needs and what a
demonstration cannot give.

It is a **pure function** of the privileged state plus its own small phase carry:

    choices, phase, meta = decide(state8, privileged, instruction, phase)

`choices` is one answer per question of the per-axis set (`AXISWISE_CANDIDATES`) -- a direction
and a step size per world axis, `yaw`, `rim` and a boolean `grip` -- which `compose_axiswise`
turns into the `(7,)` action the answer is held for. No randomness, no I/O, no simulator: the same
arguments always give the same answer, which is what makes a relabelling reproducible.

How it differs from the `heuristic` recipe (`robojev/policy.py`), which reaches
the bowl in 17/20 episodes and lifts it in 0/20 (note `2026-09-20-stage9-robojev-run3-closed-
loop.md`):

1. **It grasps the rim, not the centre.** Note docs/DESIGN.md measured
   every demonstration grasping 5-6 cm off centre horizontally and 2-5 cm above the reported pose;
   a centre waypoint holds 0.16-0.74 against the demo pose's 0.73-0.93. `rim_point` is that note's
   own recommended waypoint, `bowl_xy + r * normalize(eef_xy - bowl_xy)`.
2. **Its step sizes are calibrated against what the arm executes, not what it commands.** The two
   halves of this are `EXECUTED_COARSE_PER_DELTA_T` (the arm executes about a fifth of the
   translation it is commanded, so every threshold written in the composer's units is five times
   too wide) and `axis_answer` (the step size is chosen from the error on the axis being *moved*,
   not from the norm of the 3-d error, which would take a whole large step along a small
   component and oscillate). Between them they are most of the difference between the heuristic's
   0/20 and the table in the note.
3. **It waits for the fingers.** `grip` going true is a command, not a grasp: the Panda's fingers
   take about `SETTLE_DECISIONS` chunks to close on a rim. Lifting on the next decision tears the
   bowl out from between them.
4. **It checks and retries.** Whether the bowl came with the hand is observable (`skill.held`:
   the fingers are shut and the bowl has not moved relative to them since the grasp), so a grasp
   that caught nothing is noticed rather than carried to the plate: the expert reopens, backs off
   to the hover and comes down on the next of `GRASP_CANDIDATES` -- the other end of the rim axis,
   then the next grasp height.

The target/destination rule is *not* re-invented here: it is `robojev.roles.scene_roles`,
the same function the harvester, the `heuristic` recipe and the RoboJEV server all read an
instruction with, so nothing can name a different bowl from the same sentence.
"""
from __future__ import annotations

import collections
import dataclasses
from typing import Any

import numpy as np

from robojev import roles as roles_mod, scene, skill

# ------------------------------------------------------------------------ the loop's own numbers

#: An answer is held for this many control steps. `v2.compose.CHUNK_STEPS` is the same number
#: read from the other side of the fine-tune, and `test_expert.py` pins the two together.
CHUNK_STEPS: int = 5

#: **How far a coarse chunk actually moves the hand, in metres per unit of δ_t.** The one
#: measurement this whole controller is built on, and it is not the number the composer's units
#: suggest. `OSC_POSE` maps a full-scale component onto 5 cm *commanded* per control step, so five
#: of them command `0.25 × δ_t` metres -- but robosuite re-sets the OSC goal to `current + delta`
#: every step and the arm's impedance only closes a fifth of that gap in one 20 Hz step, so what
#: it *executes* is a fifth of what it asks for.
#:
#: Measured (`scratch/stage9-expert/calibrate.py`, LIBERO-Spatial task 0 from its settled reset
#: pose, five identical steps per cell, metres travelled):
#:
#: | δ_t | ±x | ±y | ±z | (coarse) | ±x | ±y | ±z | (fine) |
#: |---|---|---|---|---|---|---|---|---|
#: | 0.3536 | 0.0156 | 0.0190 | 0.0193 | | 0.0051 | 0.0062 | 0.0063 | |
#: | 0.50 | 0.0222 | 0.0270 | 0.0274 | | 0.0073 | 0.0088 | 0.0090 | |
#: | 0.75 | 0.0335 | 0.0405 | 0.0411 | | 0.0110 | 0.0134 | 0.0136 | |
#: | 1.00 | 0.0449 | 0.0541 | 0.0549 | | 0.0147 | 0.0179 | 0.0182 | |
#:
#: Exactly linear in δ_t, and `fine` is exactly a third of `coarse` at every δ_t -- the composer's
#: `FINE_FRACTION` survives the controller. x is ~17 % slower than y and z (the Panda's reach
#: direction), which one scalar cannot express; 0.050 is the mean and the error it makes is
#: absorbed by the next decision, because every threshold below is re-read from the current state.
EXECUTED_COARSE_PER_DELTA_T: float = 0.050

#: **δ_t = 1.0, and the demonstrations' 0.3536 is what kept the `heuristic` recipe at 0/20.**
#: The manifest's δ_t is the *demonstrations'* median moving step, which is the right number for
#: labelling what a human did and the wrong one for a controller that gets 44 decisions: at
#: 0.3536 a coarse chunk executes 1.8 cm, and one LIBERO-Spatial episode is about 0.40 m of
#: axis-wise reach, 0.10 m of lift and 0.25 m of carry -- 40+ decisions before the gripper has
#: closed. Measured on task 0 init 0: the hand needed 24 of the 44 decisions to arrive over the
#: rim and the episode ended in `approach`, which is exactly the note's "closed on the bowl at
#: step 175, too late to lift" failure.
#:
#: At 1.0 a coarse chunk executes ~5 cm and a fine one ~1.7 cm, so the three answers tile the line
#: at 0.8 cm / 2.5 cm (`_thresholds`) -- finer than the 1.9-8.1 cm spread the demonstrations' own
#: grasps sit in (feasibility note), so the resolution lost is below the resolution the grasp
#: needs. **Consequence for the training run:** these labels are answers about a 5 cm step. A
#: server composing them with the manifest's 0.3536 would execute a fifth of every move the expert
#: meant, so a checkpoint trained on expert labels has to be served with `delta_t = 1.0` in its
#: `robojev.json`. That is the whole of what changing the label source costs downstream.
EXPERT_DELTA_T: float = 1.0
#: δ_r as the manifest carries it, for the **grouped** (v1) composer, which is the only thing that
#: still scales a rotation by it. The per-axis composer below uses the measured
#: `EXECUTED_DEG_PER_UNIT` instead, for the reason `EXECUTED_COARSE_PER_DELTA_T` exists: a
#: rotation command executes a fraction of what it asks for, and a threshold written in commanded
#: units is a controller that stops short.
EXPERT_DELTA_R: float = 0.05785714285714285

#: **Degrees of wrist yaw the arm actually executes per normalised unit, per decision.** The
#: rotation half of `EXECUTED_COARSE_PER_DELTA_T`, and measured the same way
#: (`scratch/v2-drawer/calib_yaw.py`: five identical control steps from the settled reset pose,
#: the change in the gripper's own closing-axis angle, repeated eight times):
#:
#: | command | executed deg / decision | eef drift over 8 decisions |
#: |---|---|---|
#: | 0.105 | 3.6 | 1 mm |
#: | 0.25 | 8.0 | 3 mm |
#: | **0.50** | **14.1** | 8 mm |
#: | 1.00 | 18.9 | **18 cm** -- the OSC drags the hand across the table; unusable |
#:
#: So the wrist turns ~28 degrees per unit per decision, not the 143 the commanded scale
#: (`5 steps x 0.5 rad`) suggests. **This is why the wrist was unaffordable**: the note's
#: "δ_r buys 8.3 degrees per decision" was a *commanded* figure, the served `yaw` size executed
#: 3.6, and a right angle cost 25 decisions of a 44-decision horizon. At the measured scale a
#: right angle costs 6, which is what makes `yaw` a question worth asking (note
#: docs/DESIGN.md).
EXECUTED_DEG_PER_UNIT: float = 28.0

#: What each step size turns the wrist by, in **degrees executed per decision** -- the rotation
#: counterpart of `AXISWISE_SCALE`, and the numbers `v2.compose.YAW_*_DEG` are. One decision of
#: `large` is a sixth of a right angle; the wrist is held to the same size as the hand's largest
#: translation, because it is one hand.
YAW_SCALE: dict[str, float] = {"large": 15.0, "medium": 5.0, "small": 1.5}

#: How far the wrist may be from the angle the grasp asks for before the plan turns it, in
#: degrees. `robojev.compose.DEFAULT_YAW_TOLERANCE_DEG`, copied here for the reason the
#: composer's other constants are (this module must import while the package is being edited) and
#: pinned against it in `test_expert.py`.
#:
#: **Eight degrees, and it is `YAW_SCALE["large"]` that sets it**: the wrist has one speed, so a
#: tolerance below half a step is a wrist that steps across the band and back for ever (15
#: degrees of step against a 5-degree band: +7 -> -8 -> +7). At 8 an error inside one step lands
#: inside the tolerance and an error outside it shrinks by 15 degrees a decision. It is also well
#: inside what the grasp can absorb: 8 degrees moves a fingertip 0.6 cm along the rim, against
#: the +-1.5 cm the rim grasp holds at 100 % (the measurements's tolerance table), and the demonstrations'
#: own grasps sit 15-30 degrees off radial.
YAW_TOL_DEG: float = 8.0

# ----------------------------------------------------------------------------- the controller
# Every length is metres in LIBERO's world frame: the frame `obs["state"][0:3]` (the grip site,
# i.e. the point between the fingertips) and `privileged[name]["pos"]` (a body origin) share.

#: The rim radius of docs/DESIGN.md: the horizontal offset from the
#: bowl's reported pose to the point the demonstrations actually close at (their per-task medians
#: were 0.048-0.056 m).
RIM_RADIUS: float = 0.05
#: How far above the bowl's reported pose the fingers close. The pose is the body's origin, at the
#: bowl's base, and the demonstrations' own medians were +0.022 to +0.051 m above it -- a 3 cm
#: spread, which turns out to be the difference between a grasp and a miss. **Measured** (task 0,
#: init 0, otherwise identical runs): at +0.045, where the `heuristic` recipe's `GRASP_HEIGHT` put
#: it, the fingers close to 2 mm on nothing and the bowl does not move; at +0.025 the same close
#: catches the rim and the bowl comes up with the hand. The fingers are 8 cm across and the rim is
#: a thin ring: a centimetre too high and they shut above it.
GRASP_DZ: float = 0.025
#: How far above the grasp point the approach flies before it descends. Clears the ramekin, the
#: cookie box and the plate, which are all shorter than this.
HOVER: float = 0.08
#: Where the three step sizes hand over to each other, as fractions of what they execute. The
#: textbook answer is 0.5 and 0.5 -- hand over exactly half way, so the residual after a step is
#: never more than half the step. Both are **above** a half here because the arm does not stop when
#: the command does: an OSC goal set five times in a row leaves the hand moving, and a chunk
#: commanded `-z` is measurably still drifting `+y` from the chunk before it. Measured on task 0:
#: at 0.5/0.5 the approach overshot the rim in y and spent four decisions oscillating across it.
#: Preferring the smaller step over a wider band costs one extra decision per reach and removes
#: the oscillation, which cost four.
FINE_BAND: float = 0.7
HOLD_BAND: float = 0.6

#: Horizontal tolerance on the rim point, in the approach and at the release, and the vertical one
#: on the grasp height. Both are a little **above** `_thresholds`' hold threshold (0.8 cm at
#: δ_t = 1.0) and that ordering is load-bearing: a tolerance tighter than the smallest step the
#: vocabulary can take is a phase that answers `hold` for ever without ever being satisfied.
XY_TOL: float = 0.013
Z_TOL: float = 0.013
#: How many decisions the gripper holds still and closed before the lift. Measured: the fingers
#: reach the bowl's rim about 2 chunks after the command (note §"failure modes").
SETTLE_DECISIONS: int = 3
#: How far the hand lifts above the grasp point before carrying.
LIFT: float = 0.10
#: How far the target may drift from where it sat in the fingers before the plan calls it dropped
#: -- the tolerance of the one `held` test (`skill.held`), used by the lift, the carry and the
#: tracker's printed `holding` alike. Generous on purpose: a carried bowl swings a centimetre or
#: two in the fingers, and the cost of a false "dropped" (reopening over the middle of the table)
#: is much higher than of a late one.
DROP_TOL: float = 0.04
#: Below this finger width (metres) the gripper counts as shut -- `memory.CLOSED_WIDTH`, half the
#: Panda's 8 cm span, copied here for the reason the composer's constants are (this module has to
#: import while the package around it is being edited) and pinned against it in `test_expert.py`.
#: It is a *shut* test and not a *holding* one: measured over the deployed run's own states, a
#: hand carrying a bowl reports 0.2-2.0 cm of gap and a hand that shut on air reports 0.1 cm, so
#: the width says whether the fingers closed and only `skill.held` says on what.
CLOSED_WIDTH: float = 0.04
#: How high above the destination's pose the bowl is carried.
CARRY_CLEARANCE: float = 0.14
#: How high above the destination's pose the bowl is released.
RELEASE_DZ: float = 0.035
#: Decisions the gripper stays open, holding still, before the retreat -- the mirror of
#: `SETTLE_DECISIONS`. Retreating on the same decision as the release drags the bowl off the plate.
RELEASE_DECISIONS: int = 2
#: Decisions spent rising away from the placed bowl. LIBERO's success predicate is checked every
#: step and the episode ends on it, so this only ever runs when the placement was not counted.
RETREAT_DECISIONS: int = 2
#: How far the hand rises while retreating from the bowl it has put down.
RETREAT_RISE: float = 0.10
#: Below this horizontal distance to the target the rim **axis** is latched. Re-reading the
#: wrist's closing axis every decision is right while the hand is far away -- it names whichever
#: end of the axis the hand is coming from, and the hand is still choosing where to come from --
#: and self-defeating once the plan is standing on the rim: a fresh read taken there names the
#: side the plan has just abandoned straight back. So the axis is latched here and the grasp
#: candidate in force says which *end* of it to stand on.
LATCH_RADIUS: float = 0.12

#: **One grasp, as the plan may try it**: which way round the target's rim to stand (`turn`, in
#: degrees from the wrist's own closing axis, measured anticlockwise seen from above) and how far
#: above the target's pose the fingers close (`dz`, added to the task's `grasp_dz`).
#:
#: `turn` is **one number for two things**, which is the whole of why the drawer task becomes
#: expressible: it says where round the rim to stand *and* how far the wrist has to turn to get
#: there, because the fingers must close **radially** at whatever rim point is chosen (a
#: tangential pair of fingers lands both on the ring and pushes the object away -- task 2, 3/10
#: against 9/10). A turn of 0 or 180 degrees is the wrist's own axis and needs no rotation, which
#: is exactly the two candidates this list used to hold.
Grasp = collections.namedtuple("Grasp", ("turn", "dz"))

#: Which questions a candidate needs answered before it can be executed. A quarter- or
#: eighth-turn of the rim needs the wrist, and the wrist is the `yaw` question; the two ends of
#: the axis the hand already holds need nothing. `candidates()` filters on this, so **a
#: checkpoint served without `yaw` is never offered a grasp it cannot perform** -- it sees the
#: eight candidates this module had before the wrist existed, in the same order.
def requires(grasp: "Grasp") -> tuple[str, ...]:
    return () if round(float(grasp.turn)) % 180 == 0 else ("yaw",)


#: The heights the retry walks, in the order it walks them: the tuned height first, then upwards,
#: because the measured failure at +0.045 shuts on air while a close that is too *low* pushes the
#: object away before it shuts and moves the pose the next attempt aims at.
RETRY_DZ: tuple[float, ...] = (0.0, 0.015, -0.01, 0.03)

#: The rim directions the plan may try, as turns from the wrist's own closing axis, **grouped by
#: what the wrist has to do to reach them**:
#:
#: * `0, 180` -- the two ends of the axis the hand already holds. No rotation, and they are this
#:   module's whole vocabulary before this change: the first is where the hand is standing and
#:   the second is the measured recovery from a rim point the arm cannot hold (note
#:   docs/DESIGN.md §2: reachable to 0.4 cm where the first end is 2.2 cm
#:   short). Their order is measured and is deliberately *not* re-ordered by the rule below.
#: * `±90` -- the other principal axis. Together with the first pair these are the four rim
#:   points the stalled-descent note probed, and on the drawer task one of them is the grasp that
#:   works: 9/10 inits at `-90` and 6/10 at `+90`, against 0/10 at either end of the axis the
#:   hand starts with (docs/DESIGN.md §2).
#: * `±45, ±135` -- the diagonals, last: a quarter turn costs six decisions and a diagonal costs
#:   three, but the diagonals measured 5/9 on the same inits where `-90` measured 9/10, so the
#:   cheaper rotation is not the better grasp.
GRASP_TURNS: tuple[float, ...] = (0.0, 180.0, -90.0, 90.0, -45.0, 135.0, 45.0, -135.0)

#: **The one ordered list of alternatives** (the design notes.3), replacing the two the machine used to
#: have -- `rim_flips`, a side to take after a stalled descent, and `RETRY_DZ`, a height indexed
#: by failed grasps. They were one idea written twice, so a stalled *descent* could take the other
#: side and a failed *grasp* could not, and neither could change the other's variable.
#:
#: The **direction** alternates fastest because it is the cheaper and the better-evidenced move;
#: the **height** then walks the bracket. Which of the two ends of a *new* axis comes first is
#: not fixed here -- `candidate_turns` reads it off the scene.
GRASP_CANDIDATES: tuple[Grasp, ...] = tuple(
    Grasp(turn, dz) for dz in RETRY_DZ for turn in GRASP_TURNS
)

#: The turns, **grouped in the order the groups are tried**, and this tuple is the whole of the
#: candidate order that is not read off the scene:
#:
#: 1. `0` -- where the hand is already standing. Unchanged and first, because it is what the
#:    100-episode table was measured with, and never re-ordered by anything.
#: 2. `180, ±90` -- **the other three rim points of the two principal axes**, the set the
#:    stalled-descent note probed, ordered by the free space `candidate_turns` reads off the
#:    scene. One group and not two, and that is the whole ruling: the opposite end of the axis
#:    the hand holds is the *measured* recovery from a rim point the **arm** cannot stand on, and
#:    a quarter turn is the only recovery from one the **scene** blocks, and which of those two
#:    an episode is in is a question about the scene rather than about the vocabulary. Ranking
#:    them by how much room the outer finger has answers it: on the drawer task `180` ranks
#:    **last** of the three (8.9 cm against 19.9 for `-90`) because the second bowl stands behind
#:    it -- and `180` is measurably hopeless there, blocked by the same drawer wall as `0` (§1) --
#:    while on tasks 1 and 9, the two that actually spend candidates, `180` ranks first (21.2 and
#:    36.1 cm) and keeps its place.
#: 3. the diagonals, last: they cost less wrist than a quarter turn and measured worse (5/10
#:    against 9/10 on the drawer task's inits).
TURN_GROUPS: tuple[tuple[float, ...], ...] = (
    (0.0,), (180.0, -90.0, 90.0), (-45.0, 45.0, 135.0, -135.0),
)

#: Half the Panda's finger span. The fingers straddle the rim point, so **this far outside it is
#: where the outer finger stands** -- and that is the point a candidate's free space is measured
#: at, because it is the point that lands on a wall. (Measured, drawer task: the outer finger at
#: `RIM_RADIUS + FINGER_HALF_SPAN` = 9 cm from the bowl's centre against 7.1 and 9.5 cm of
#: interior depth, and it rests on `wooden_cabinet_1_g9`/`g11` for every control step of the
#: descent.)
FINGER_HALF_SPAN: float = 0.04

#: How much room the outer finger needs beside it before a candidate counts as one that **fits**:
#: its own half-thickness, about a centimetre of Panda fingertip.
#:
#: The threshold is measured rather than chosen. Over the drawer task's eight rim directions the
#: room the scene reports and the grasp that results agree at exactly this line: `-90` 2.9 cm and
#: 9/10 inits grasped, `-135` 2.6 cm and 5/10, `+90` 1.4 cm and 6/10, `+45` 1.1 cm and 5/10,
#: against `+135` 0.7 cm, `180` 0.5 cm and `0`/`-45` 0.0 cm, which grasp **nothing**.
FINGER_HALF_WIDTH: float = 0.01

#: How far above what it was standing in the target is carried, in metres, on top of whatever
#: that thing reaches above it (`scene.support`). **Measured, and 8 cm is not enough**: at 8 the
#: drawer task's bowl travels 6 mm over the front panel it has to cross, catches it, and the
#: carry stalls for 27 of the episode's 44 decisions (three of ten inits). At 10 it crosses 2.6
#: cm clear and they finish. The demonstrations peak 1.4 cm over that panel, which is the same
#: story told by a human who can see the panel.
#:
#: It is still cheaper than the blanket clearance it replaces wherever the walls are low: the
#: cabinet-top task's bowl stands on a flat surface (0.2 cm of lip) and is carried 12.8 cm above
#: its origin rather than 14, and the flat tasks are unchanged because their destination's own
#: clearance is the higher of the two.
EXTRACT_MARGIN: float = 0.10


def candidates(qids=None, turns: tuple[float, ...] = GRASP_TURNS) -> tuple[Grasp, ...]:
    """The alternatives a run may try: `turns` x `RETRY_DZ`, keeping only the ones whose wrist
    rotation the **served question set** can actually execute.

    A candidate is an instruction to the arm, and an instruction the vocabulary cannot express is
    not an alternative -- it is a decision that silently does something else. A checkpoint trained
    before `yaw` was asked declares its own question set in `robojev.json` and is served with it
    (`robojev/policy.py`), so this hands that run the eight candidates it was trained
    against, in their old order, and hands a run that asks `yaw` all thirty-two.
    """
    have = None if qids is None else set(qids)
    order = tuple(t for t in turns
                  if have is None or not set(requires(Grasp(t, 0.0))) - have)
    return tuple(Grasp(t, dz) for dz in RETRY_DZ for t in order)


def rotate_xy(direction, degrees: float) -> np.ndarray:
    """A horizontal unit vector, turned `degrees` anticlockwise seen from above."""
    v = np.asarray(direction, dtype=np.float64).reshape(2)
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def wrap_yaw(degrees: float) -> float:
    """An angle folded into `(-90, +90]`.

    **The gripper's closing axis is a line, not an arrow**: fingers pointing along `+d` and along
    `-d` are the same grasp. So the wrist never has to turn more than a right angle to align with
    a direction, and a rule that turned through 135 degrees to reach a pose 45 degrees away would
    spend six decisions buying nothing.
    """
    a = (float(degrees) + 180.0) % 360.0 - 180.0
    if a > 90.0:
        return a - 180.0
    if a <= -90.0:
        return a + 180.0
    return a


def wrist_yaw_error(proprio, want_xy) -> float:
    """How far the wrist must still turn, in degrees, for the fingers to close along `want_xy`.

    Signed, and wrapped into `(-90, +90]` by `wrap_yaw`. Positive is anticlockwise seen from
    above, which is what the `yaw` question's `+` candidate says. This is the number the state
    prints on its `Wrist yaw error:` line and the only thing the `yaw` label is a function of.
    """
    proprio = np.asarray(proprio, dtype=np.float64).reshape(-1)
    want = np.asarray(want_xy, dtype=np.float64).reshape(2)
    if proprio.shape[0] < 6:
        return 0.0
    closing = (_rotation(proprio[3:6]) @ np.array([0.0, 1.0, 0.0]))[0:2]
    if float(np.linalg.norm(closing)) < 1e-3 or float(np.linalg.norm(want)) < 1e-9:
        return 0.0
    return wrap_yaw(np.degrees(np.arctan2(want[1], want[0]) - np.arctan2(closing[1], closing[0])))


def candidate_room(target, objects: dict, axis, turn: float, radius: float,
                   grasp_dz: float, name: str | None = None) -> tuple[float, str | None]:
    """`(metres of room beside the outer finger, what would stop it)` for one rim direction.

    The one measurement the ranking, the state's printed block and the `candidate` question's
    gold are all made of, so what the plan prefers, what the text says and what the model is
    scored against cannot disagree.
    """
    target = np.asarray(target, dtype=np.float64).reshape(3)
    direction = rotate_xy(np.asarray(axis, np.float64).reshape(2), turn)
    probe = np.array([*(target[0:2] + (radius + FINGER_HALF_SPAN) * direction),
                      float(target[2]) + grasp_dz])
    # The target itself is not an obstacle: the fingers are *meant* to close on it, and its own
    # rim is the thing they straddle.
    return scene.clearance(probe, objects, exclude=name)


def candidate_turns(target, objects: dict, axis, radius: float,
                    turns: tuple[float, ...] = GRASP_TURNS,
                    grasp_dz: float = GRASP_DZ, name: str | None = None) -> tuple[float, ...]:
    """The rim directions to try, in the order **this scene** prefers them.

    **One rule, and it is a partition rather than a sort**: a rim direction whose outer finger
    does not fit goes to the back; among the rest, and among the blocked ones, the declared order
    of `TURN_GROUPS` stands.

    Both halves are deliberate. Demoting what does not fit is the whole of what the drawer task
    needed -- `0` and `180` are stopped by the same two drawer walls, and the plan used to spend
    eleven decisions discovering that by collision. *Not* sorting the rest by room is what keeps
    every task that works today reaching for what it reaches for today: on the flat scenes the
    hand's own side has the most room anyway (10.7 cm on task 0) and a sort would still be free
    to reorder it on the strength of a centimetre.

    "Fits" is `FINGER_HALF_WIDTH` of room measured against **objects and fixtures alike**
    (`scene.clearance`), which is the measurement that used to be missing: the cabinet and the
    stove are in the privileged state now.
    """
    target = np.asarray(target, dtype=np.float64).reshape(3)
    axis = np.asarray(axis, dtype=np.float64).reshape(2)
    declared = [t for group in TURN_GROUPS for t in turns if t in group]
    declared += [t for t in turns if t not in declared]

    def fits(turn: float) -> bool:
        room, _ = candidate_room(target, objects, axis, turn, radius, grasp_dz, name)
        return room >= FINGER_HALF_WIDTH

    return tuple(sorted(declared, key=lambda t: (not fits(t), declared.index(t))))


class ExpertError(RuntimeError):
    """The expert was asked for a decision it cannot make: no privileged state, an empty scene."""


# ------------------------------------------------------------------------------ per-task tuning

#: Per-task constants, keyed by the task's instruction (lower-cased and whitespace-collapsed --
#: `_task_key`). Only the keys that differ from the module defaults above need to be present.
#:
#: **Measured, not guessed**: every entry here was set by running the 10 init states of that task
#: in `scratch/stage9-expert/measure.py` and keeping the value that scored best; the note carries
#: the table. An empty dict for a task means the defaults already scored 10/10 on it, which is
#: worth recording explicitly -- it says the default rim rule generalises, rather than that nobody
#: looked.
TASK_TUNING: dict[str, dict[str, float]] = {}


def _task_key(instruction: str) -> str:
    return " ".join((instruction or "").lower().split())


def tuning(instruction: str) -> dict[str, float]:
    """The constants this task runs with: the module defaults, overridden by `TASK_TUNING`."""
    base = {
        "rim_radius": RIM_RADIUS,
        "grasp_dz": GRASP_DZ,
        "hover": HOVER,
        "lift": LIFT,
        "carry_clearance": CARRY_CLEARANCE,
        "release_dz": RELEASE_DZ,
    }
    base.update(TASK_TUNING.get(_task_key(instruction), {}))
    return base


def _rotation(axisangle) -> np.ndarray:
    """Rodrigues' formula, a `(3,)` rotation vector to a `(3, 3)` matrix.

    Kept here rather than imported from the composer: the expert reads a wrist orientation to
    decide from, which is a different job from turning an answer into an action.
    """
    v = np.asarray(axisangle, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return np.eye(3)
    k = v / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def grasp_direction(proprio, target_xy) -> np.ndarray:
    """Which side of the bowl to put the fingers on: **the wrist's own closing axis**, signed
    toward the hand.

    The feasibility note's rule is `normalize(eef_xy - bowl_xy)` -- the side the hand is coming
    from -- and that rule is right about the *radius* and silent about the direction, because
    every demonstration it was measured from also had a human turning the wrist. This axis is
    what the hand can stand on for **free**, and it is what the first two candidates use. A
    candidate with a non-zero `turn` pays for a rim point off it with the rotation
    `wrist_yaw_error` measures -- six decisions for a right angle at the measured
    `EXECUTED_DEG_PER_UNIT`.

    It matters as much as the radius. The fingers close along the gripper's local y and span 8 cm
    about the grip site; the bowl's rim is a ring of radius ~6 cm. Put the hand on the rim with
    that axis pointing **radially** and one finger goes inside the bowl and one outside, with the
    rim wall between them. Put it there with the axis **tangential** and both fingers land on the
    ring itself -- and the descent pushes the bowl across the table instead of grasping it, which
    is exactly what task 2 did for 8 decisions at a time while the bowl slid away from it
    (observed: the target's own pose moving 2 cm during the descent, three failed attempts, 3/10).

    The sign is whichever end of the axis the hand is already nearer, so choosing the direction
    never costs a trip around the bowl. The `axis` fallback keeps the note's own rule for a wrist
    whose closing axis is vertical, where there is no side to choose.
    """
    proprio = np.asarray(proprio, dtype=np.float64).reshape(-1)
    target_xy = np.asarray(target_xy, dtype=np.float64).reshape(2)
    approach = proprio[0:2] - target_xy
    if proprio.shape[0] >= 6:
        closing = (_rotation(proprio[3:6]) @ np.array([0.0, 1.0, 0.0]))[0:2]
        norm = float(np.linalg.norm(closing))
        if norm > 1e-3:
            closing = closing / norm
            return closing if float(np.dot(closing, approach)) >= 0 else -closing
    norm = float(np.linalg.norm(approach))
    return approach / norm if norm > 1e-6 else np.array([1.0, 0.0])


def rim_point(target_xy, eef_xy, radius: float = RIM_RADIUS) -> np.ndarray:
    """The grasp waypoint of docs/DESIGN.md:
    `bowl_xy + radius * normalize(eef_xy - bowl_xy)`.

    The direction is whichever side the hand is already approaching from, so the rule never needs
    to know which side a demonstrator used. Degenerate only when the hand is exactly over the
    bowl's axis, and then any direction is as good as any other: `+x` is chosen so the function
    stays a total, deterministic function of its arguments.
    """
    target_xy = np.asarray(target_xy, dtype=np.float64).reshape(2)
    direction = np.asarray(eef_xy, dtype=np.float64).reshape(2) - target_xy
    norm = float(np.linalg.norm(direction))
    unit = direction / norm if norm > 1e-6 else np.array([1.0, 0.0])
    return target_xy + radius * unit


# ---------------------------------------------------------------------- reading the instruction


def resolve_roles(objects: dict, instruction: str, eef) -> dict:
    """`robojev.roles.scene_roles`, with this module's own error type on an empty scene.

    The rule itself is documented there: it is the one a **server** can apply, from the
    instruction and the object poses alone, and the expert uses it so that a scripted episode and
    a served one are acting on the same bowl.
    """
    try:
        return roles_mod.scene_roles(objects, instruction, eef)
    except ValueError as exc:        # an empty scene: the rule's own error, in this module's type
        raise ExpertError(str(exc)) from exc


def _pos(objects: dict, name: str) -> np.ndarray:
    return np.asarray(objects[name]["pos"], dtype=np.float64)[0:3]


# ------------------------------------------------------------------ the task, as data
#
# Pick-and-place is a `robojev.skill.Plan`: nine stages, each a `Stage` carrying its own
# goal, its own arrival test and where it goes when it arrives, is blocked or loses its outcome.
# **No branch of `decide` names a stage.** What used to be five per-stage rules -- a patience for
# every phase, a second patience for the descent, a third for the lift, a rim-flip counter and a
# retry table indexed by attempts -- is now one progress rule, one candidate list and one `held`,
# all of them in `skill.step` and none of them here.


@dataclasses.dataclass(frozen=True)
class Ctx:
    """Everything one decision knows, computed once and read by the stages' own callables.

    Frozen and plain: the stage table has to be data, so nothing in it may reach back into
    `decide`'s locals, and a context that could be mutated by a goal function would make the
    executor's purity a matter of manners.
    """

    proprio: np.ndarray
    eef: np.ndarray
    target: np.ndarray
    destination: np.ndarray
    roles: dict
    tune: dict
    rim_axis: np.ndarray        # the wrist's closing axis, latched inside LATCH_RADIUS
    rim_view: tuple             # one row per offered rim direction: what the state prints
    alternatives: tuple         # the (turn, dz) list this run may try, in its own order
    choice: str | None          # the letter a model asked for, if the question is asked at all
    grasp: Grasp                # the candidate in force
    grasp_xy: np.ndarray
    grasp_z: float
    grasp_dir: np.ndarray       # the rim direction the candidate stands on, from the target
    yaw_err: float              # degrees the wrist must still turn to close along `grasp_dir`
    clear_z: float              # the height the target is carried at, clear of what it left
    hover_z: float
    lift_from: float            # the hand's own z when the fingers settled, else the hand's now
    place_xy: np.ndarray
    closed: bool
    offset: np.ndarray          # target - eef, the constant of a real grasp
    reference: Any              # what `offset` was when the fingers settled, or None
    horizontal_to_rim: float
    horizontal_to_target: float
    horizontal_to_destination: float
    drop: float                 # how far the *bowl* is above its release height


#: The letters the offered rim directions are listed under, and the order they are listed in.
#: `v2.questions.RIM_CANDIDATES`, copied for the reason the composer's constants are and pinned
#: against it in `test_expert.py`.
RIM_LETTERS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G", "H")


def rim_view(target, objects: dict, axis, turns, radius: float, grasp_dz: float,
             tried: tuple[float, ...] = (), name: str | None = None) -> tuple[dict, ...]:
    """One row per offered rim direction: the letter, where it stands, what the wrist must do,
    how much room the outer finger would have there and whether it fits.

    **The state prints this and the `rim` question is scored against it**, so the plan's own
    preference, the text the model reads and the gold are one computation rather than three.
    """
    axis = np.asarray(axis, dtype=np.float64).reshape(2)
    rows = []
    for letter, turn in zip(RIM_LETTERS, turns):
        room, blocker = candidate_room(target, objects, axis, turn, radius, grasp_dz, name)
        rows.append({
            "letter": letter,
            "turn": float(turn),
            "side": _side_word(rotate_xy(axis, turn)),
            "room_cm": round(min(room, 99.9) * 100.0, 1),
            "fits": bool(room >= FINGER_HALF_WIDTH),
            "blocker": blocker,
            "tried": bool(turn in tried),
        })
    return tuple(rows)


def _side_word(direction) -> str:
    """Which side of the target a rim direction stands on, in the state's own four words."""
    dx, dy = float(direction[0]), float(direction[1])
    if abs(dx) >= abs(dy):
        return "+x" if dx >= 0 else "-x"
    return "+y" if dy >= 0 else "-y"


def rim_gold(rows) -> str:
    """The `rim` answer the state's own block implies: **the first listed candidate that fits and
    has not been tried**, and the first untried one when none of them fits.

    One relation per line and no arg-max over a column (§2 principle 2): every row carries its
    own verdict, and this reads them in order.
    """
    rows = list(rows)
    if not rows:
        return RIM_LETTERS[0]
    for row in rows:
        if row["fits"] and not row["tried"]:
            return row["letter"]
    for row in rows:
        if not row["tried"]:
            return row["letter"]
    return rows[0]["letter"]


def _hover(ctx: Ctx) -> np.ndarray:
    return np.array([ctx.grasp_xy[0], ctx.grasp_xy[1], ctx.hover_z])


def _rim(ctx: Ctx) -> np.ndarray:
    return np.array([ctx.grasp_xy[0], ctx.grasp_xy[1], ctx.grasp_z])


#: Over what horizontal distance the carry comes back **down** to the destination's own
#: clearance, in metres. The extraction height is a fact about where the target *came from*, and
#: it stops being relevant once the hand has left there: holding it all the way across the table
#: buys nothing and costs the decisions the descent onto the plate then needs (the drawer task's
#: release is 25 cm below the extraction height and 10 cm below the destination's -- three
#: decisions of the horizon it has none to spare in).
#:
#: The demonstrations do exactly this rather than carrying flat: on the drawer task the bowl
#: peaks at 1.199 m twenty control steps after the close and is at 1.149 by the time it is half
#: way to the plate (docs/DESIGN.md §1). 20 cm is a little over twice
#: what the drawer's own walls stand out from the bowl, so the ramp starts only once the target
#: is clear of them.
CARRY_RAMP_M: float = 0.20


def _carry_height(ctx: Ctx) -> float:
    """**How high the hand holds the target while it travels**, in world metres.

    From where the hand was when the fingers settled, the lift clears `lift`; but a target that
    was standing *inside* something has to come out of it before it can go anywhere, and the
    height that takes is a fact about **where it came from**, not about where it is going. So the
    clearance is measured from the higher of the two (`ctx.clear_z`), and the hand carries the
    target that far up -- `eef - target` is the constant of the grasp, so adding it turns a height
    for the object into a height for the hand.

    Measured against the demonstrations (docs/DESIGN.md §1): they raise
    the bowl to 1.211 m on the drawer task (this rule: 1.203) and to 1.251 m on the cabinet task
    (this rule: 1.268), where the old destination-only rule asked for 1.110 m on both -- 9 cm
    *below* the drawer's front panel and 2 cm below the cabinet top the bowl was standing on.
    On the eight flat tasks the destination is the higher of the two and nothing changes.
    """
    low = float(ctx.destination[2]) + ctx.tune["carry_clearance"]
    ramp = min(max(ctx.horizontal_to_destination / CARRY_RAMP_M, 0.0), 1.0)
    return low + (ctx.clear_z - low) * ramp + (float(ctx.eef[2]) - float(ctx.target[2]))


def _clear_above(ctx: Ctx) -> np.ndarray:
    """From where the hand was when the fingers settled, **not** from the grasp height: the grasp
    height is read off the target's current pose, and once the grasp is real the target rises with
    the hand, so a lift goal defined against it recedes exactly as fast as the hand climbs (26 of
    44 decisions spent in `lift`, the bowl 0.73 m above the table).

    The lift is also the **extraction**: it goes straight up to the carry height when that is
    higher than `lift`, because moving sideways before then drags the target through whatever it
    was standing in (the drawer's own front panel, 13 cm above the bowl's resting pose).
    """
    return np.array([ctx.eef[0], ctx.eef[1],
                     max(ctx.lift_from + ctx.tune["lift"], _carry_height(ctx))])


def _above_destination(ctx: Ctx) -> np.ndarray:
    return np.array([ctx.place_xy[0], ctx.place_xy[1], _carry_height(ctx)])


def _onto_destination(ctx: Ctx) -> np.ndarray:
    """The *bowl's* height is what has to end up above the plate, and it is in the state: the hand
    descends by however far the bowl is above its release height, not to a fixed z."""
    return np.array([ctx.place_xy[0], ctx.place_xy[1], float(ctx.eef[2]) - ctx.drop])


def _up_and_away(ctx: Ctx) -> np.ndarray:
    return np.array([ctx.eef[0], ctx.eef[1], float(ctx.eef[2]) + RETREAT_RISE])


#: What one degree of wrist is worth in the executor's one unit, metres of remaining travel.
#: **A millimetre**, so a right angle of wrist counts as 9 cm of approach -- which is about what
#: walking round the rim to the opposite side costs, and those two are exactly the alternatives
#: the plan is choosing between. It has to be well above `skill.PROGRESS_EPS` per decision or a
#: turning wrist would read as a stalled one: at 15 degrees a decision the wrist closes 1.5 cm of
#: it, five times the 0.3 cm the progress rule asks for.
YAW_RESIDUAL_M_PER_DEG: float = 0.001


def _wrist_residual(ctx: Ctx) -> float:
    """How far the approach still is from done because of the **wrist**, in metres.

    Zero for every candidate that stands on the axis the wrist already holds, so the eight tasks
    that never leave those two see the approach they always saw.
    """
    return abs(float(ctx.yaw_err)) * YAW_RESIDUAL_M_PER_DEG


def _over_the_rim(ctx: Ctx) -> bool:
    # **And the wrist has turned.** The fingers have to close radially at the rim point the
    # candidate chose, and the place to turn them is up here, where nothing is in the way: a
    # descent begun with the wrist still 40 degrees out puts a finger where the rim is not.
    # For the two candidates that need no rotation `yaw_err` is 0 by construction, so this
    # clause is invisible to every task that never leaves them.
    return (ctx.horizontal_to_rim < XY_TOL
            and abs(float(ctx.eef[2]) - ctx.hover_z) < ctx.tune["hover"] * 0.5
            and abs(ctx.yaw_err) <= YAW_TOL_DEG)


def _at_the_rim(ctx: Ctx) -> bool:
    # Horizontal first: closing 3 cm to the side of the rim is the one mistake this controller
    # cannot recover from without a whole retry.
    return ctx.horizontal_to_rim < XY_TOL and float(ctx.eef[2]) - ctx.grasp_z < Z_TOL


def _clear_of_the_table(ctx: Ctx) -> bool:
    return float(ctx.eef[2]) > float(_clear_above(ctx)[2]) - Z_TOL


def _over_the_destination(ctx: Ctx) -> bool:
    return ctx.horizontal_to_destination < XY_TOL


def _at_release_height(ctx: Ctx) -> bool:
    # **Both** halves, and the horizontal one is not decoration: releasing at the right height
    # over the wrong point puts the bowl on the table beside the plate, which looks like a
    # finished episode and scores nothing (observed on task 7: released 3 cm off in y).
    return ctx.drop < Z_TOL and ctx.horizontal_to_destination < XY_TOL


def _held(ctx: Ctx) -> bool:
    """The one `held`, for the lift's outcome, the carry's and the tracker's printed `holding`."""
    return skill.held(ctx.closed, ctx.offset, ctx.reference, DROP_TOL)


def _the_grasp_that_was_made(ctx: Ctx) -> dict:
    """What the fingers settled on, recorded as the close hands over to the lift: the offset every
    later `held` is measured against, and the height the lift is measured from."""
    return {"grasp_offset": [float(v) for v in ctx.offset], "close_eef_z": float(ctx.eef[2])}


def _first_untried(turn: float | None, alternatives, spent) -> int | None:
    """The first alternative standing at `turn` that has not been spent, or `None`.

    **Which grasp height to try a direction at stays the plan's own ladder**: the `rim` answer
    names where round the rim to stand and nothing else, so this walks `alternatives` in their own
    order and takes the first one that stands there.
    """
    if turn is None:
        return None
    spent = set(spent)
    for index, grasp in enumerate(alternatives):
        if grasp.turn == turn and index not in spent:
            return index
    return None


def _wanted_turn(choice: str | None, rows) -> float | None:
    """The rim direction a `rim` letter names, read off the block the state printed.

    `None` for no answer and for a letter this run was not offered -- the two cases in which the
    plan keeps its own order. One lookup, used by the first selection and by every later one, so
    the two cannot read the letters differently.
    """
    if not choice:
        return None
    return next((float(row["turn"]) for row in rows if row["letter"] == choice), None)


def _choose_candidate(carry: dict, ctx: Ctx) -> int | None:
    """Which alternative to take next: **the one the model asked for**, when it was asked.

    `ctx.choice` is the `rim` answer the state this decision was read off carried. It names a rim
    *direction*; which grasp height to try it at stays the plan's own ladder, so this returns the
    first alternative with that direction that has not been spent. An answer naming a letter this
    run was not offered, or one whose direction is used up, returns `None` and the executor falls
    back to its order -- and `meta["rim_followed"]` says which of the two happened.
    """
    return _first_untried(_wanted_turn(ctx.choice, ctx.rim_view), ctx.alternatives,
                          set(carry.get("tried") or []) | {int(carry["candidate"])})


#: Pick-and-place, whole. Read down the `on_done` column for a successful episode and across the
#: `on_blocked`/`on_failed` ones for every way it recovers.
PICK_AND_PLACE: skill.Plan = skill.Plan(
    stages=(
        # **The approach accepts the best pose it reached: blocked, it descends from where it
        # is.** Measured (docs/DESIGN.md §2): an approach that
        # stops improving has almost always stopped 2-3 cm from the hover point, where the arm's
        # own 1.7 cm step is oscillating across the tolerance -- not on a pose it cannot hold.
        # Sending it round the bowl to the other candidate costs six decisions and loses episodes
        # (9:7, 1:8); descending from 2 cm out costs nothing, because the descent steers at the
        # same x and y. Where a rim point really *is* unreachable, the descent is where that
        # shows (the stalled-descent note's own evidence), and the descent is what takes the next
        # candidate.
        skill.Stage("approach", "reach", False, goal=_hover, residual=_wrist_residual,
                    arrived=_over_the_rim, on_done="descend", on_blocked=skill.ACCEPT),
        skill.Stage("descend", "reach", False, goal=_rim, arrived=_at_the_rim,
                    on_done="close", on_blocked="approach"),
        # Still, shut, and waiting for the fingers: `grip` going true is a command, not a grasp.
        skill.Stage("close", "grasp", True, dwell=SETTLE_DECISIONS, on_done="lift",
                    on_exit=_the_grasp_that_was_made),
        skill.Stage("lift", "lift", True, goal=_clear_above, arrived=_clear_of_the_table,
                    check=_held, on_done="carry", on_blocked="approach", on_failed="approach"),
        skill.Stage("carry", "carry", True, goal=_above_destination,
                    arrived=_over_the_destination, check=_held, on_done="lower",
                    on_failed="approach"),
        skill.Stage("lower", "place", True, goal=_onto_destination, arrived=_at_release_height,
                    on_done="release"),
        skill.Stage("release", "place", False, dwell=RELEASE_DECISIONS, on_done="retreat"),
        skill.Stage("retreat", "retreat", False, goal=_up_and_away, dwell=RETREAT_DECISIONS,
                    on_done="done"),
        # The placement is made and LIBERO ends the episode on its own predicate.
        skill.Stage("done", "retreat", False, on_done="done"),
    ),
    start="approach",
    candidates=len(GRASP_CANDIDATES),
    choose=_choose_candidate,
    # A new candidate stands somewhere else and has grasped nothing yet. The rim *axis* is not
    # here: it is the episode's, latched once, and the candidate's `side` is which end of it to
    # stand on -- so "the other side" stays a statement about the bowl rather than about wherever
    # the hand happened to stall.
    per_attempt=("grasp_offset", "close_eef_z"),
)

#: The sub-stage names, in the order one successful episode passes through them. Derived from the
#: plan, so a stage cannot exist without a name or be named without existing.
PHASES: tuple[str, ...] = PICK_AND_PLACE.names


def new_phase(roles: dict | None = None) -> dict:
    """The carry an episode starts with: nothing latched, nothing attempted.

    `skill.new_carry`'s own fields (the stage, the progress, the candidate, the attempts) plus
    this task's three: the roles, the rim axis, and what the close recorded.

    `roles` pins `{"target", "destination"}` instead of reading them from the instruction. A
    *server* has no business using it -- naming the bowl from the sentence is half the task -- but
    a **label source** does: a relabelling run knows which task it is relabelling, and the BDDL's
    `obj_of_interest` is the answer the success predicate is actually scored against. Measuring the
    expert both ways is what separates "the controller cannot do it" from "the controller was
    pointed at the wrong bowl" (see the note's table).
    """
    return skill.new_carry(
        PICK_AND_PLACE,
        # Pinned, or latched at the first decision: the scene stops being static the moment the
        # grasp moves the target.
        roles=dict(roles) if roles else None,
        rim_axis=None,        # the wrist's closing axis toward the hand, read once
        turns=None,           # the rim directions in this scene's preferred order, read once
        target_z0=None,       # where the target was standing, before the grasp moved it
        support_rise=None,    # how far what it was standing in reaches above it
        rim_world=None,       # the chosen rim direction, in the world frame, latched per candidate
        rim_candidate=-1,     # which candidate `rim_world` was latched for
        rim_turn=None,        # ...and which direction that candidate meant when it was
        chose=None,           # which candidate the model asked for, if it is asked at all
        chose_applied=False,  # ...and whether the episode's first selection has consumed it
    )


def with_executed(phase: dict | None, answers: dict | None, qids=None) -> dict | None:
    """`phase`, carrying the **executed** `rim` letter -- the one answer the plan itself obeys.

    Every other answer is a step the arm takes and the next decision re-reads; `rim` names which
    way round the target to stand, and the plan reads it out of its own carry at the moments it
    selects a candidate (`ctx.choice`, `_choose_candidate`). So a carry that did not receive the
    executed letter is a carry standing on a different candidate from the moment the first
    selection happens -- which is a *different rim point*, two rim-radii away, not a rounding.

    **There is one such carry per copy of the plan, and they all have to be the same plan.** Four
    exist: `run_episode`'s own, `rollout._Episode`'s mirror of it, `TrackerV2._carry` (which
    renders the state the model reads) and `rollout.relabel`'s (which labels it). This is the one
    rule all four apply, because round 1 of the first checkpoint served with `rim` died of the
    fourth of them quietly not applying it: the tracker replayed the model's letter, the
    labeller's carry kept the plan's own order, and at the first selection after a failed grasp
    the two were standing on different candidates -- surfacing as `gold_for_state`'s waypoint
    check, 10 cm (two rim radii) and one retry offset apart.

    `qids` is which questions the run is asked, when that is known: `TrackerV2` blanks the letter
    when a run that *is* asked `rim` reports no answer, and a replay of it has to blank its own
    the same way or the two carries part company on the next selection. A caller with no question
    set reads the answer dict itself, which is what a harvest's own loop has always done.
    """
    if phase is None or answers is None:
        return phase
    if "rim" not in (answers if qids is None else qids):
        return phase
    return {**phase, "chose": answers.get("rim")}


def _yaw_answer(yaw_error: float) -> str:
    """The `yaw` answer for a wrist alignment error: a sign against `YAW_TOL_DEG`, nothing else.

    The same relation `v2.compose.labels_from_waypoint` applies to the number the state prints,
    which is what makes the label recoverable from the text. The two spellings are the two
    set: `+`, `-` or `hold`.
    """
    plus, minus = "+", "-"
    if abs(float(yaw_error)) <= YAW_TOL_DEG:
        return "hold"
    return plus if yaw_error > 0 else minus


def decide(
    state8,
    privileged: dict,
    instruction: str,
    phase: dict | None = None,
    *,
    delta_t: float = EXPERT_DELTA_T,
    qids=None,
) -> tuple[dict[str, Any], dict, dict]:
    """One decision point: `(choices, next_phase, meta)`.

    `state8` is the environment's 8-d proprio vector (grip-site position, the end-effector's
    axis-angle, the two finger joints); `privileged` is its
    `{name: {"pos", "quat"}}`; `phase` is `new_phase()` at the start of an episode and the second
    element of this function's own return after that.

    The answers are the per-axis set (`AXISWISE_CANDIDATES`, composed by `compose_axiswise`):
    one direction and one step size per world axis, plus `yaw`, `rim` and a boolean `grip`.

    **The decision itself is three lines**: read the geometry, hand it to `skill.step` with
    `PICK_AND_PLACE`, turn the offset to the goal it chose into an answer. The stage's own
    `grip`, `goal`, `arrived`, `check` and where it goes next are in the table above; nothing
    here branches on which stage it is.

    Pure and total over its own carry: it reads nothing but its arguments, mutates none of them,
    and every stage of the plan returns an answer -- an expert that can raise in the middle of a
    relabelling is not a label source. It *does* raise on an input it cannot decide from at all
    (no scene, a carry naming a stage this plan has never had), which is a different thing and
    is what `ExpertError` is for.
    """
    proprio = np.asarray(state8, dtype=np.float64).reshape(-1)
    if proprio.shape[0] < 3:
        raise ExpertError(f"expected the 8-d proprio vector, got shape {proprio.shape}")
    if not privileged:
        raise ExpertError("no privileged scene state: the expert decides from object poses")
    eef = proprio[0:3]
    phase = dict(new_phase() if phase is None else phase)
    if phase["name"] not in PHASES:
        raise ExpertError(f"the phase carry names {phase['name']!r}, which is not one of {PHASES}")
    tune = tuning(instruction)

    # --- the roles. Latched at the first decision, because every rule `scene_roles` applies is a
    # statement about where things are and carrying the target is the act of making it untrue
    # (see `heuristic.latched_roles`). A latch naming an object the scene no longer has is
    # dropped rather than trusted.
    roles = phase.get("roles")
    if not (roles and all(roles.get(k) in privileged for k in ("target", "destination"))):
        roles = resolve_roles(privileged, instruction, eef)
    target = _pos(privileged, roles["target"])
    destination = _pos(privileged, roles["destination"])

    # --- the context: every number one decision reads, computed once and handed to the stages.
    ctx, alternatives, latched = _context(proprio, roles, target, destination, tune, phase,
                                          privileged, qids)

    # --- the decision. One call, no branch on the stage's name: `skill.step` applies the progress
    # rule, the candidate list, the outcome check and the dwell, in that order, to whichever stage
    # the carry names. The plan is the module's own, with the number of alternatives **this run**
    # has: a served question set without `yaw` cannot turn the wrist, so it is not offered the
    # candidates that need it (`candidates`).
    plan = (PICK_AND_PLACE if len(alternatives) == PICK_AND_PLACE.candidates
            else dataclasses.replace(PICK_AND_PLACE, candidates=len(alternatives)))
    # The deferred first selection (`_context`) is applied to the carry the executor is handed,
    # not to the one it returns: what it returns may be a *later* selection -- a stage that just
    # blocked has already taken the next candidate -- and that one must stand.
    phase = dict(phase, candidate=int(latched.pop("candidate")))
    out = skill.step(plan, phase, ctx)
    nxt = dict(out.carry)
    nxt["roles"] = dict(roles)
    nxt.update(latched)

    error = out.goal - eef
    choices = axiswise_answers(error, yaw=_yaw_answer(ctx.yaw_err), grip=out.grip,
                               delta_t=delta_t)
    choices["rim"] = rim_gold(ctx.rim_view)
    meta = {
        **roles,
        "phase": out.stage.name,
        "next_phase": nxt["name"],
        "ticks": int(phase["ticks"]),
        "attempts": int(phase["attempts"]),
        "candidate": int(phase["candidate"]),
        "decisions": int(phase.get("decisions", 0)),
        "goal": [round(float(v), 4) for v in out.goal],
        "error": [round(float(v), 4) for v in error],
        "grasp_point": [round(float(ctx.grasp_xy[0]), 4), round(float(ctx.grasp_xy[1]), 4),
                        round(ctx.grasp_z, 4)],
        "grasp_dz": round(ctx.grasp_z - float(target[2]), 4),
        "grasp_turn": float(ctx.grasp.turn),
        "candidates": len(alternatives),
        "rim": [dict(row) for row in ctx.rim_view],
        "rim_gold": rim_gold(ctx.rim_view),
        # Whether the plan took the letter it was handed, or fell back to its own order because
        # the answer named a candidate this run does not have or has already spent.
        "rim_followed": bool(ctx.choice and any(row["letter"] == ctx.choice
                                                for row in ctx.rim_view)),
        "rim_asked": ctx.choice,
        "horizontal_to_rim": round(ctx.horizontal_to_rim, 4),
        "horizontal_to_target": round(ctx.horizontal_to_target, 4),
        "horizontal_to_destination": round(ctx.horizontal_to_destination, 4),
        "place_point": [round(float(ctx.place_xy[0]), 4), round(float(ctx.place_xy[1]), 4)],
        "held": bool(_held(ctx)),
        "blocked": bool(out.blocked),
        "failed": bool(out.failed),
        "distance": round(float(out.distance), 4),
        "finger_width": round(float(proprio[6] - proprio[7]), 4) if proprio.shape[0] >= 8 else None,
        # Rounded to the tenth of a degree the state prints, so the label the tracker reads off
        # the text and the answer this module gives are a function of the same number.
        "yaw_error": round(float(ctx.yaw_err), 1),
    }
    return choices, nxt, meta


def _context(proprio, roles: dict, target, destination, tune: dict, phase: dict,
             objects: dict | None = None, qids=None) -> tuple[Ctx, tuple[Grasp, ...], dict]:
    """The geometry of one decision, as the feasibility note's rule and the candidate in force.

    Returns the context, the **alternatives this run has** (the candidate list, filtered by what
    the served question set can execute) and the carry fields latched here.

    **The rim axis is read while the hand is far and latched inside `LATCH_RADIUS`.**
    `grasp_direction` signs the wrist's closing axis toward the hand, which is the right rule
    while the hand is still choosing where to come from and a self-defeating one once it is
    standing on the rim: a fresh read taken there names the side the plan has just abandoned
    straight back. So the axis is latched, and the candidate's `turn` says which way round it to
    stand -- which makes "take another side" a statement about the bowl rather than about
    wherever the hand happened to stall.

    **The candidate's `turn` is also the wrist's job.** The fingers have to close radially at
    whatever rim point is chosen, so the angle between the wrist's closing axis and the rim
    direction *is* the yaw error, and it is zero by construction for the two candidates that
    stand on the axis the wrist already holds.
    """
    eef = np.asarray(proprio, dtype=np.float64).reshape(-1)[0:3]
    horizontal_to_target = float(np.hypot(*(target[0:2] - eef[0:2])))
    stored = phase.get("rim_axis")
    fresh = grasp_direction(proprio, target[0:2])
    axis = fresh if stored is None else np.asarray(stored, dtype=np.float64).reshape(2)
    # Latched at the first decision, both of them, and for the same reason the roles are: the
    # scene stops being static the moment the grasp moves the target, and an alternative that
    # meant one direction at decision 3 and another at decision 30 is not an alternative.
    # **The order stays live until a candidate has actually been spent.** It is read from where
    # the target *is*, and at the first decision of an episode the target is often not there yet:
    # LIBERO spawns the drawer task's bowl 8.8 cm above the drawer floor, level with the tops of
    # the walls, where every rim direction is clear -- so an order latched then says the blocked
    # ones are fine and the plan walks into them anyway. While nothing has been tried the hand is
    # still choosing where to come from and re-reading costs nothing; from the first switch on it
    # is latched, because an alternative that meant one direction at decision 3 and another at
    # decision 30 is not an alternative.
    turns = phase.get("turns")
    if turns is None or not (phase.get("tried") or []):
        turns = candidate_turns(target, objects or {}, axis, tune["rim_radius"],
                                grasp_dz=tune["grasp_dz"], name=roles["target"])
    turns = tuple(float(t) for t in turns)
    # `target_z0` is **where the target was standing**, and it is only readable before the grasp
    # moves it: latched while the plan is still reaching for it, never afterwards. A carry that
    # has none (one built straight into a later stage, as the tests do) leaves the clearance to
    # the destination alone, which is what this plan did before the drawer -- a height that
    # chased the target's current pose would be the lift bug all over again, 26 decisions spent
    # climbing after a bowl that rises with the hand.
    # The **lowest** the target has been seen at while the plan was still reaching for it, not
    # the first reading: LIBERO spawns an object a few centimetres above what it will rest on and
    # lets it fall, and the drawer task's bowl drops 8.8 cm in the first decision of every
    # episode. A clearance read off the spawn height would carry it 9 cm higher than it needs and
    # tear it out of the fingers on the way (measured: the lift asked for 23 cm, the bowl slipped
    # at the first decision of it).
    target_z0 = phase.get("target_z0")
    if PICK_AND_PLACE.subgoals.get(phase["name"]) in ("reach", "grasp"):
        target_z0 = (float(target[2]) if target_z0 is None
                     else min(float(target_z0), float(target[2])))
    target_z0 = None if target_z0 is None else float(target_z0)
    # How far what the target is standing in reaches **above** it -- a drawer's own front panel,
    # a cabinet top's lip -- read from the fixtures' boxes and latched with the height it is
    # measured from, because once the grasp lifts the target it is standing in nothing.
    support_rise = phase.get("support_rise")
    if support_rise is None and PICK_AND_PLACE.subgoals.get(phase["name"]) in ("reach", "grasp"):
        _what, support_rise = scene.support(roles["target"], objects or {})
    support_rise = None if support_rise is None else float(support_rise)
    alternatives = candidates(qids, turns)
    index = min(int(phase["candidate"]), len(alternatives) - 1)
    # **The episode's first selection is the model's too.** Every later one is
    # (`_choose_candidate`, through `Plan.choose`), but the executor only chooses when something
    # goes wrong, and at the episode's first decision nothing has: the plan took the first
    # alternative in its own order before the model had said anything, because the first answer
    # only exists once the first state has been rendered. So the choice is *deferred* instead:
    # decision 0 is rendered with the whole block and carries the plan's own order, and the
    # letter that comes back selects the candidate for decision 1 -- the first decision at which
    # there is an answer to obey. `chose_applied` latches it, so a letter that stands for a whole
    # episode does not re-select every decision, and a model that changes its mind mid-approach
    # does not walk the hand round the bowl.
    #
    # **For the gold answer nothing changes**, which is what keeps the expert's own harvest
    # byte-identical: `rim_gold` is the first listed candidate that fits, `candidate_turns` has
    # already put a fitting direction first, so the gold letter selects the candidate the plan
    # was already standing on.
    first_choice = (not phase.get("chose_applied") and not (phase.get("tried") or [])
                    and index == 0)
    if first_choice:
        letters = rim_view(target, objects or {}, axis,
                           tuple(dict.fromkeys(g.turn for g in alternatives)),
                           tune["rim_radius"], tune["grasp_dz"], name=roles["target"])
        picked = _first_untried(_wanted_turn(phase.get("chose") or None, letters),
                                alternatives, ())
        if picked is not None:
            index = picked
    grasp = alternatives[index]
    # **The axis is re-read from the wrist while the hand is far -- unless the plan is the one
    # turning the wrist.** Re-reading is right while the hand is still choosing where to come
    # from, and it is self-defeating twice over once a candidate has a `turn`: the rim direction
    # is the axis rotated by that turn, so an axis read off a wrist that is *obeying* the turn
    # rotates with it and the rim point runs away from the hand. For the two candidates that need
    # no rotation this is exactly the old rule, decision for decision.
    if stored is None or (horizontal_to_target > LATCH_RADIUS and not requires(grasp)):
        axis = fresh

    # **The rim direction is latched in the WORLD frame, per candidate.** It is fixed the moment
    # a candidate is selected -- the wrist's closing axis then, turned by the candidate's turn --
    # and re-latched only when the next candidate is selected. That is what makes a wrong `yaw`
    # answer a recoverable mis-rotation instead of a moved waypoint: the wrist target is a world
    # angle the model's answers cannot edit, so the worst a wrong one does is turn the hand away
    # from it and cost the decision it takes to turn back.
    world = phase.get("rim_world")
    # Re-latched when the *candidate* changes, and also when the same index comes to mean a
    # different direction, which is what a live order can do while nothing has been tried yet.
    same = (world is not None and int(phase.get("rim_candidate", -1)) == index
            and float(phase.get("rim_turn", 1e9)) == float(grasp.turn))
    direction = (np.asarray(world, dtype=np.float64).reshape(2) if same
                 else rotate_xy(axis, grasp.turn))
    latched = {"rim_axis": [float(axis[0]), float(axis[1])], "turns": list(turns),
               "target_z0": target_z0, "support_rise": support_rise,
               "rim_world": [float(direction[0]), float(direction[1])],
               "rim_candidate": index, "rim_turn": float(grasp.turn),
               # Latched the first decision at which an answer existed to obey -- or, if the run
               # is not asked `rim` at all, the first decision full stop, so a deferred choice
               # never outlives the approach it belongs to.
               "chose_applied": bool(phase.get("chose_applied") or not first_choice
                                     or phase.get("chose")),
               "candidate": index}
    grasp_xy = target[0:2] + tune["rim_radius"] * direction
    grasp_z = float(target[2]) + tune["grasp_dz"] + grasp.dz
    lift_from = phase.get("close_eef_z")
    width = float(proprio[6] - proprio[7]) if np.asarray(proprio).reshape(-1).shape[0] >= 8 else 0.0
    # **The bowl is what has to land on the plate, and it is not in the hand's axis.** A rim grasp
    # holds it `rim_radius` to one side, so aiming the *hand* at the plate's centre puts the bowl's
    # centre a rim-radius off the plate's edge and LIBERO's `on(bowl, plate)` predicate never
    # fires. (Observed: a clean pick, carry, descent and release on task 0 that scored nothing --
    # the bowl came to rest 1.5 cm above the table, beside the plate.) While the target is held,
    # the offset from the hand to the bowl is a constant of the grasp, so the place waypoint is
    # the destination minus it, and "have we arrived" is asked about the bowl.
    # **The wrist is only asked to turn while the hand is still reaching for the rim.** Once the
    # fingers are on it, turning them twists the object out of the grasp, so the error the plan
    # reports -- and therefore the `yaw` label the state carries -- is zero from the close
    # onwards. It is a property of the stage the carry names, which is a fact the state prints.
    reaching = PICK_AND_PLACE.subgoals.get(phase["name"]) == "reach"
    yaw_err = wrist_yaw_error(proprio, direction) if reaching else 0.0
    spent = set(phase.get("tried") or [])
    view = rim_view(target, objects or {}, axis,
                    tuple(dict.fromkeys(g.turn for g in alternatives)),
                    tune["rim_radius"], tune["grasp_dz"],
                    tried=tuple({alternatives[i].turn for i in spent if i < len(alternatives)}),
                    name=roles["target"])
    return Ctx(
        proprio=np.asarray(proprio, dtype=np.float64).reshape(-1),
        eef=eef, target=target, destination=destination, roles=roles, tune=tune,
        rim_axis=axis, rim_view=view, alternatives=alternatives,
        choice=(phase.get("chose") or None), grasp=grasp, grasp_xy=grasp_xy, grasp_z=grasp_z,
        grasp_dir=direction, yaw_err=float(yaw_err),
        # **The two heights a carry has to clear, and each measured where it is**: the
        # destination's own clearance, and whatever the target was standing in, plus the margin
        # that gets it over that thing's walls. The second used to be the first's constant
        # applied to the target's origin, which asked for 14 cm of lift off a *flat* cabinet top
        # (task 9, whose walls reach 0.2 cm above the bowl) and only 11 cm over a drawer panel
        # that stands 6.5 cm above it.
        clear_z=max(float(destination[2]) + tune["carry_clearance"],
                    (target_z0 + (support_rise or 0.0) + EXTRACT_MARGIN)
                    if target_z0 is not None else -np.inf),
        hover_z=grasp_z + tune["hover"],
        lift_from=float(lift_from) if lift_from is not None else float(eef[2]),
        place_xy=destination[0:2] - (target[0:2] - eef[0:2]),
        # A vector too short to carry the fingers is one of the synthetic states the tests build;
        # the plan only asks about the fingers in stages where it has commanded them shut.
        closed=bool(proprio.shape[0] < 8 or width <= CLOSED_WIDTH),
        offset=target - eef,
        reference=phase.get("grasp_offset"),
        horizontal_to_rim=float(np.hypot(*(grasp_xy - eef[0:2]))),
        horizontal_to_target=horizontal_to_target,
        horizontal_to_destination=float(np.hypot(*(target[0:2] - destination[0:2]))),
        drop=float(target[2]) - (float(destination[2]) + tune["release_dz"]),
    ), alternatives, latched


# ------------------------------------------------------------------------ the answer's shape
#
# Three independent ternary questions (one per world axis), each with its own step size, answered
# in parallel and composed into **one diagonal move**. It was measured against the alternative --
# one axis per decision, at one of two sizes -- and won: that shape costs the controller a factor
# of about sqrt(3) in path length, since a reach a diagonal move covers in 0.25 m of travel takes
# 0.40 m of axis-wise travel, which at 5 cm a decision is three extra decisions per reach.

#: The candidate sets of the per-axis vocabulary.
AXISWISE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "move_x": ("-", "hold", "+"), "move_y": ("-", "hold", "+"), "move_z": ("-", "hold", "+"),
    "size_x": ("large", "medium", "small"), "size_y": ("large", "medium", "small"),
    "size_z": ("large", "medium", "small"),
    "yaw": ("-", "hold", "+"),
    "rim": RIM_LETTERS,
    "grip": ("false", "true"),
}

#: What each size executes, as a fraction of a full-scale component. Measured, not proposed: a
#: full-scale component is the most the OSC controller will take (δ_t = 1.0 is its range limit) and
#: it executes ~5 cm per decision per axis, so a diagonal `large` move covers up to 8.7 cm. The
#: three sizes are then 5.0 / 1.7 / 0.5 cm per axis, which is the coordinator's proposed ~8 / 2.5 /
#: 0.7 cm read as diagonals. `small` exists because `medium` (1.7 cm) is larger than the grasp's
#: own positioning tolerance and a two-size vocabulary therefore cannot place the hand on the rim.
AXISWISE_SCALE: dict[str, float] = {"large": 1.0, "medium": 1.0 / 3.0, "small": 0.1}


def axiswise_answers(error, yaw: str = "hold", grip: bool = False,
                     delta_t: float = EXPERT_DELTA_T) -> dict[str, str]:
    """The per-axis vocabulary's answers for one 3-d position error.

    Each axis is answered on its own -- direction and size -- against the same thresholds
    `axis_answer` uses for the single chosen axis, one band lower because there are three sizes:

        |e| < half a small step  -> `hold`
        |e| < half a medium step -> `small`
        |e| < half a large step  -> `medium`
        otherwise                -> `large`
    """
    error = np.asarray(error, dtype=np.float64).reshape(3)
    large = delta_t * EXECUTED_COARSE_PER_DELTA_T
    steps = {name: AXISWISE_SCALE[name] * large for name in ("small", "medium", "large")}
    # A size is chosen by the step **above** it, exactly as the two-size rule is: take the largest
    # step whose own overshoot the error can afford. Reading each band against its own step instead
    # is an off-by-one that answers `large` for a 1.2 cm error -- measured, on task 0: a 5 cm step
    # for a 1.2 cm error, the hand 4.6 cm above the rim when the fingers shut, 2/10.
    out: dict[str, str] = {"yaw": yaw, "grip": "true" if grip else "false"}
    for index, name in enumerate("xyz"):
        e = float(error[index])
        if abs(e) < HOLD_BAND * steps["small"]:
            out[f"move_{name}"], out[f"size_{name}"] = "hold", "small"
            continue
        if abs(e) < FINE_BAND * steps["medium"]:
            size = "small"
        elif abs(e) < FINE_BAND * steps["large"]:
            size = "medium"
        else:
            size = "large"
        out[f"move_{name}"] = "+" if e > 0 else "-"
        out[f"size_{name}"] = size
    return out


def compose_axiswise(choices: dict, delta_t: float = EXPERT_DELTA_T,
                     delta_r: float = EXPERT_DELTA_R) -> np.ndarray:
    """The per-axis answers as one `(7,)` LIBERO action: a diagonal move, not an axis-aligned one.

    The counterpart of `robojev.compose.compose` for `AXISWISE_CANDIDATES`, and
    **the same action for the same answers** -- the two are pinned against each other in
    `test_expert.py`, because this one is what the 100-episode table is measured with and that
    one is what the served policy executes.

    `delta_r` is not what the wrist is scaled by and is kept only for the signature's sake: a
    rotation is scaled by `YAW_SCALE["large"]` over the **measured**
    `EXECUTED_DEG_PER_UNIT`, exactly as a translation is scaled by what it executes rather than
    by what it commands. The wrist has **one** speed and it is not the hand's -- see
    `v2.compose.compose`, which this is pinned against.
    """
    action = np.zeros(7, dtype=np.float32)
    for index, name in enumerate("xyz"):
        move = choices[f"move_{name}"]
        if move != "hold":
            sign = 1.0 if move == "+" else -1.0
            action[index] = sign * delta_t * AXISWISE_SCALE[choices[f"size_{name}"]]
    if choices.get("yaw", "hold") != "hold":
        sign = 1.0 if choices["yaw"] in ("+", "+yaw") else -1.0
        action[5] = sign * min(YAW_SCALE["large"] / EXECUTED_DEG_PER_UNIT, 1.0)
    action[6] = 1.0 if choices["grip"] in (True, "true") else -1.0
    return action


# ------------------------------------------------------------------------- the v2 vocabulary
#
# RoboJEV v2 (`docs/DESIGN.md` §7, a design ruling) adopts
# the `axiswise` set above **as it is measured** -- per-axis direction *and* per-axis size, the
# shape that scored 85/100 at a median of 27 decisions against the shared-step vocabulary's
# 83/100 at 33. So `answers_v2` is not a new controller and not even a new answer: it is
# `decide`'s own answers, whose question ids are already v2's, plus
# the one question v2 adds (`subgoal`) and `grip` normalised to the bool a `boolean` question
# takes. Anything else would be a second controller to measure.

#: `PHASES` -> `robojev.questions.SUBGOAL_CANDIDATES`, and **total** over `PHASES`
#: (`test_the_subgoal_map_is_total_over_the_phase_machine`).
#:
#: `descend` is `reach` and not `grasp`, which is the one line worth reading twice: the expert's
#: `close` is the decision that shuts the fingers, and `grasp` is the candidate whose text says
#: "this is where they close". A `descend` labelled `grasp` would teach the model to close a
#: decision early, on the way down, which is the 4.5 cm miss of `GRASP_DZ`'s own measurement.
#: `done` is `retreat` because v2 has no candidate for "the episode is over" -- LIBERO ends the
#: episode on its own predicate and nothing is asked after it.
#:
#: **Read off the stage table** rather than written out again: each `Stage` carries the subgoal
#: it belongs to, so a sub-stage cannot exist without one or be mapped to a second one somewhere
#: else. `v2.compose.PHASE_FROM_EXPERT` aliases this dict (it imports *this* module, so importing
#: it back would be a cycle) and `test_the_subgoal_map_is_the_composer_s` pins the two together.
V2_PHASE_SUBGOAL: dict[str, str] = PICK_AND_PLACE.subgoals

#: The v2 motor vocabulary's question ids, in the order §3 declares them. `yaw` is answered here
#: like any other: it is `hold` while the candidate in force stands on the wrist's own axis
#: (which is every candidate the eight non-drawer tasks reach) and a sign while the plan is
#: turning the wrist onto a rim point off it.
V2_QIDS: tuple[str, ...] = (
    "move_x", "move_y", "move_z", "size_x", "size_y", "size_z", "yaw", "rim", "grip", "subgoal",
)

#: The three per-axis size questions, keyed by axis, and the three direction ones.
V2_SIZE_QIDS: tuple[str, ...] = ("size_x", "size_y", "size_z")
V2_MOVE_QIDS: tuple[str, ...] = ("move_x", "move_y", "move_z")

#: How far the tracker's waypoint may sit from the expert's own before `gold_for_state` calls
#: them two different plans, in metres. `XY_TOL` is the expert's own arrival tolerance: two
#: waypoints that agree to inside it produce the same answer to every question, so a drift
#: smaller than this cannot change a label, and a drift larger than it can.
GOLD_WAYPOINT_TOL: float = XY_TOL


def answers_v2(
    state8,
    privileged: dict,
    instruction: str,
    phase: dict | None = None,
    *,
    delta_t: float = EXPERT_DELTA_T,
    roles: dict | None = None,
    qids=None,
) -> tuple[dict[str, Any], dict, dict]:
    """One decision point in the **v2 vocabulary**: `(answers, next_phase, meta)`.

    `answers` has one entry per `V2_QIDS`: the three signs, the three per-axis sizes, `yaw`,
    `grip` (a bool, which is what a `boolean` question's `value` takes) and `subgoal`.

    The same phase machine, the same waypoints and the same thresholds as `decide` -- only the
    answer's *shape* changes, and it changes to the shape `decide` already produces. Serve it at `delta_t = 1.0` (§7 a design ruling): these are answers about a 5 cm
    step, and a server composing them with the demonstrations' 0.3536 would execute a fifth of
    every move the expert meant.

    Pure and total, exactly as `decide` is: `test_answers_v2_is_total_over_random_states` draws
    200 privileged states and asserts every qid is present and every value is a declared
    candidate, because a label source that omits a question is a training row that cannot be
    built.

    **The motion answers are the same function of the same numbers as the parser's.** At
    `delta_t = EXPERT_DELTA_T` every `move_*`/`size_*` here equals
    `robojev.compose.labels_from_waypoint`'s for the same waypoint offset -- the two
    read the same bands against the same steps (`AXISWISE_SCALE`, `HOLD_BAND`, `FINE_BAND`) --
    and `test_the_v2_answers_are_the_v2_label_function` pins it over a thousand offsets. It is
    pinned rather than delegated because `v2.compose` imports this module: calling it from here
    would be a cycle, and a label source that cannot be imported is worse than one that is
    checked.

    The one seam that pinning leaves open, stated rather than hidden: this compares metres
    against a band computed in metres and the label function compares centimetres against the
    same band computed in centimetres, so an offset that lands *exactly* on a band edge (0.30,
    1.17 or 3.50 cm to the last bit) can fall either side of it in the two. It moves one axis by
    one size for one decision and the next decision re-reads the waypoint, which is why the
    labels of record come from `gold_for_state` -- the label function applied to the rounded
    numbers the state prints -- and not from here.
    """
    if phase is None:
        phase = new_phase(roles)
    choices, nxt, meta = decide(state8, privileged, instruction, phase,
                                delta_t=delta_t, qids=qids)
    answers: dict[str, Any] = {qid: choices[qid] for qid in V2_QIDS[:8]}
    # A `boolean` question's value is a bool on the wire and in `compose`; `axiswise_answers`
    # spells it as a candidate id because its own candidate set is strings. Normalised here, once,
    # so a caller never has to know which of the two it is holding.
    answers["grip"] = choices["grip"] in (True, "true")
    answers["subgoal"] = V2_PHASE_SUBGOAL[meta["phase"]]

    offset_cm = [round(float(v) * 100.0, 1) for v in meta["error"]]
    meta = {
        **meta,
        "questions_version": "v2",
        "subgoal": answers["subgoal"],
        # The waypoint, in the unit v2's state prints it in: signed gripper-relative centimetres.
        # A console showing a decision beside the state it was made from should not have to
        # convert metres.
        "waypoint_cm": offset_cm,
        "offset_cm": offset_cm,
        "sizes": {axis: answers[f"size_{axis}"] for axis in "xyz"},
        "delta_t": float(delta_t),
        "cm_per_unit": EXECUTED_COARSE_PER_DELTA_T * 100.0,
    }
    return answers, nxt, meta


def _tracker_waypoint(tracker) -> np.ndarray | None:
    """The point `tracker` is steering to, in world metres, or `None` if it has none."""
    waypoint = getattr(tracker, "waypoint", None)
    if waypoint is None:
        return None
    return np.asarray(waypoint.point, dtype=np.float64).reshape(3)


def gold_for_state(tracker, state8, privileged: dict, instruction: str,
                   phase: dict | None = None, *, delta_t: float = EXPERT_DELTA_T,
                   roles: dict | None = None, qids=None) -> tuple[dict[str, Any], dict, dict]:
    """The label of record for one state: `(answers, next_phase, meta)`.

    The difference from `answers_v2` is the whole point of the function. `answers_v2` computes
    its own waypoint and answers about it; this reads the waypoint **off the tracker** -- the
    same object that rendered the state string -- and returns
    `robojev.state.TrackerV2.gold_answers()`, which is `labels_from_waypoint` applied
    to the offsets *as the state prints them* (rounded, through `rounded_cm`). So the gold is by
    construction a function of the numbers in the text, which is the design notes's principle 1 and gate
    G3's contract: `robojev.parse.parse(state_text)` has to reproduce it from the
    string alone, and it can only fail to if the two read the same numbers differently.

    The expert is still run, as the **check**. Where the tracker's plan and the expert's own
    differ by more than `GOLD_WAYPOINT_TOL` this raises `ExpertError` rather than returning a
    label: a disagreement between the waypoint the state names and the waypoint the controller
    is steering to is a bug that G3 would otherwise surface as a 97 % parser, weeks later and
    attributed to the parser. It should fail loudly during a harvest instead.

    What is compared is what the two claim to compute the same way:

    * **the roles** -- naming a different bowl is the sharpest possible disagreement, and the
      cheapest to detect;
    * **the rim geometry** while reaching, grasping or lifting -- the horizontal radius from the
      target's own pose and the height above it, i.e. `rim_radius`, `grasp_dz` and which retry
      offset is in force. Not the rim *direction*: the expert latches the rim axis at the first
      decision and signs it with the candidate in force, while a stage-geometry tracker re-reads
      it from the wrist every decision, so the two legitimately stand on opposite ends of the
      same axis, and a check that fired on that would fire on every retry;
    * **the place point** while carrying or placing -- `destination_xy - (target_xy - eef_xy)`,
      the offset that aims the bowl rather than the hand.

    `retreat` is not compared: "up and away" is a point the two define from different origins
    (the expert from the hand, the tracker from the destination) and nothing is labelled from it
    that a centimetre of height changes.
    """
    answers, nxt, meta = answers_v2(state8, privileged, instruction, phase,
                                    delta_t=delta_t, roles=roles,
                                    qids=getattr(tracker, "qids", None) if qids is None else qids)
    point = _tracker_waypoint(tracker)
    if point is None:
        raise ExpertError(
            "the tracker has no waypoint for this state: it has observed nothing, or its target "
            "is not in the scene. A gold label read off a waypoint that does not exist would be "
            "a label the state string cannot carry."
        )

    for role in ("target", "destination"):
        theirs = getattr(tracker, role, None)
        if theirs is not None and theirs != meta[role]:
            raise ExpertError(
                f"the tracker's {role} is {theirs!r} and the expert's is {meta[role]!r}: the "
                f"state names one object and the label was decided about another"
            )

    # **The tracker's own stage**, not the expert's. The tracker chose its waypoint for the stage
    # its facts imply, and the expert's phase machine is finer than v2's six subgoals (it has a
    # hover, a settle and a retry that `reach` and `grasp` cover between them), so the two
    # legitimately name different stages for the same state. What is *not* legitimate is the two
    # steering to different points within one stage, and that is what is compared.
    subgoal = getattr(tracker, "subgoal", None) or answers["subgoal"]
    eef = np.asarray(state8, dtype=np.float64).reshape(-1)[0:3]
    target = _pos(privileged, meta["target"])
    goal = np.asarray(meta["goal"], dtype=np.float64).reshape(3)
    # **The cheapest agreement first: the two steer to the same point.** A tracker that runs this
    # module's own plan (`v2.state.plan`, which is `decide` called from the other side) produces
    # the expert's waypoint exactly, and no geometric reconstruction of it can be more accurate
    # than the number itself. The looser comparisons below exist for a tracker that reconstructs
    # the plan from the *stage* rather than running it, and are taken only when this one fails.
    exact = float(np.linalg.norm(point - goal))
    if subgoal in ("reach", "grasp", "lift"):
        # The rim *direction* is not compared: the expert latches the axis once and the
        # candidate in force says which end of it to stand on, while a stage-geometry tracker
        # re-reads it from the wrist every decision. The radius from
        # the target is what both agree on -- except during `lift`, whose goal is measured from
        # where the hand *was* when it closed and has no rim radius at all, which is why the
        # exact comparison above is the one that passes there.
        radius = float(np.hypot(*(point[0:2] - target[0:2])))
        drift_xy = abs(radius - float(np.hypot(*(np.asarray(meta["grasp_point"][0:2]) - target[0:2]))))
        # **Two reconstructed heights are legitimate, and the check accepts either.** With an
        # instruction the tracker flies to `HOVER` above the rim and only descends once x and y
        # are aligned; without one it falls back to `waypoint_for`'s plainer stage geometry,
        # whose `reach` waypoint is the rim point itself. Both are the same plan at different
        # resolutions. A third height is not, and that is what this still fires on.
        stage_z = float(meta["grasp_point"][2]) + (LIFT if subgoal == "lift" else 0.0)
        drift_z = min(abs(float(point[2]) - float(goal[2])), abs(float(point[2]) - stage_z))
        _raise_on_drift(min(exact, max(drift_xy, drift_z)), subgoal, point, meta)
    elif subgoal in ("carry", "place"):
        place = np.asarray(meta["place_point"], dtype=np.float64).reshape(2)
        _raise_on_drift(min(exact, float(np.hypot(*(point[0:2] - place)))), subgoal, point, meta)

    gold = tracker.gold_answers()
    meta = {
        **meta,
        "gold_source": "tracker",
        "tracker_subgoal": getattr(tracker, "subgoal", None),
        "tracker_waypoint": [round(float(v), 4) for v in point],
        "tracker_offset_cm": [round(float(v - e) * 100.0, 1) for v, e in zip(point, eef)],
    }
    return gold, nxt, meta


def _raise_on_drift(drift: float, subgoal: str, point, meta: dict) -> None:
    if drift <= GOLD_WAYPOINT_TOL:
        return
    raise ExpertError(
        f"in subgoal {subgoal!r} the tracker steers to {[round(float(v), 4) for v in point]} and "
        f"the expert to {meta['goal']} (grasp point {meta['grasp_point']}, place point "
        f"{meta['place_point']}): they differ by {drift:.4f} m, more than the {GOLD_WAYPOINT_TOL} "
        f"m that cannot change an answer. The state names one waypoint and the label was decided "
        f"about another."
    )


def corrupt(choices: dict, p: float, rng, *, exempt: tuple[str, ...] = ()) -> dict:
    """Each answer independently replaced, with probability `p`, by a uniformly chosen *other*
    candidate of its own question.

    The model's error model, as an experiment: a learned policy is not a controller that fails
    gracefully, it is one whose per-question accuracy is 1 - p. Running the expert through this
    says what accuracy the closed loop actually needs, and `exempt` says whether the irreversible
    answer (`grip`) needs protecting more than the rest.
    """
    sets = dict(AXISWISE_CANDIDATES)
    out = dict(choices)
    for qid, value in choices.items():
        if qid in exempt or qid not in sets or rng.random() >= p:
            continue
        as_text = ("true" if value else "false") if isinstance(value, bool) else value
        others = [c for c in sets[qid] if c != as_text]
        picked = others[int(rng.integers(len(others)))]
        out[qid] = (picked == "true") if isinstance(value, bool) else picked
    return out


def run_episode(env, init_state_index: int, *, max_steps: int = 220,
                delta_t: float = EXPERT_DELTA_T, delta_r: float = EXPERT_DELTA_R,
                roles: dict | None = None,
                corruption: float = 0.0, corruption_exempt: tuple[str, ...] = (),
                seed: int = 0, on_decision=None, qids=None) -> dict:
    """Drive `env` (a `robojev.envs.DecisionEnv`) with this expert for one episode.

    The closed loop the whole module is measured by, and the one an expert relabelling would run:
    read the privileged state, `decide`, `compose` the answer, step it `CHUNK_STEPS` times, repeat
    until LIBERO's own success predicate ends the episode or the horizon does. `on_decision(i,
    choices, meta, state8, privileged)` is called before each chunk, which is where a harvester
    would write its row.

    It lives in the module rather than in the measurement script because it *is* the contract:
    the expert's answers are only labels if something composes and executes them exactly this way.
    """
    rng = np.random.default_rng(seed)
    obs = env.reset(init_state_index)
    phase = new_phase(roles)
    steps = 0
    success = False
    decisions = 0
    trace: list[dict] = []
    while steps < max_steps and not success:
        state8 = env.state_vector(obs)
        privileged = env.privileged(obs)
        choices, phase, meta = decide(state8, privileged, env.instruction, phase,
                                      delta_t=delta_t, qids=qids)
        if corruption > 0:
            choices = corrupt(choices, corruption, rng, exempt=corruption_exempt)
        if on_decision is not None:
            on_decision(decisions, choices, meta, state8, privileged)
        trace.append({"decision": decisions, "step": steps, **{k: v for k, v in choices.items()},
                      "phase": meta["phase"], "held": meta["held"],
                      "attempts": meta["attempts"], "candidate": meta["candidate"],
                      "grasp_turn": meta["grasp_turn"], "yaw_error": meta["yaw_error"]})
        # **The plan obeys the answer that was executed, including its own corrupted one.** `rim`
        # is the only answer that is not a move: it names which way round the rim to stand, and
        # the plan reads it out of the carry at the moments it selects a candidate (`ctx.choice`).
        # A loop that reported one letter and planned against another would be a harvest whose
        # rows describe an episode nobody ran -- and `gold_for_state`'s waypoint check says so,
        # because `TrackerV2.answer` puts the executed letter into *its* carry either way.
        phase = with_executed(phase, choices)
        action = compose_axiswise(choices, delta_t, delta_r)
        for _ in range(CHUNK_STEPS):
            if steps >= max_steps:
                break
            result = env.step(action)
            obs, steps = result.obs, steps + 1
            if result.done:
                success = True
                break
        decisions += 1
    return {"success": bool(success), "steps": steps, "decisions": decisions,
            "phase": phase["name"], "attempts": int(phase["attempts"]),
            "out_of_horizon": bool(not success and steps >= max_steps), "trace": trace}
