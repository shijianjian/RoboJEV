"""v2's composer, its step-size tiling, the single label function, and the grip latch's guard.

Three things face each other here, and between them they are §2's principle 1 -- *every label is
a deterministic function of the text the model reads*:

* `chunk(answers)` turns the answers into the `[5, 7]` action LIBERO executes,
* `labels_from_waypoint(...)` turns the numbers the state prints into the answers the expert
  would give, and
* `GripLatch` decides which of those a *model's* answers is allowed to change.

`labels_from_waypoint` is the **single label function**: the harvester, the scripted expert, the
DAgger relabeller and gate G3's text-only parser (`robojev.parse`) all call it, so
there is nowhere a label could be decided differently. In v1 the harvest labelled from an executed
move while the expert decided from geometry, and the two agreed 82.7 % of the time (failure F2).

**Every number below is measured, and measured by `robojev.expert`**
(`docs/DESIGN.md`), which is imported rather than
copied so a retune there cannot leave a stale constant here. The two facts that matter most:

1. **The arm executes about a fifth of what the composer commands** -- 0.050 m per decision per
   axis at δ_t = 1.0, the OSC range limit, exactly linear in δ_t (docs/DESIGN.md). v1's composer assumed
   0.25 m, so every threshold written in its units was five times too wide and a `hold` answer at
   4.4 cm of real error was a controller stopping 4.4 cm short of the bowl.
2. **A size band is read against the step above it**, never against its own. That is the
   off-by-one that cost the expert 34 of its 85 points: reading each band against its own step
   answers `large` for a 1.2 cm error, the hand ends 4.6 cm above the rim when the fingers shut,
   and the same vocabulary scores 51/100 instead of 85/100 (docs/DESIGN.md).
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

# The expert is the measurement, and it is imported rather than re-derived: one source of truth
# for the step sizes, the bands and the grasp tolerance (the measurements, §4). It is pure numpy --
# `test_the_module_is_light` still holds.
from robojev.expert import (
    AXISWISE_SCALE,
    rim_gold as expert_rim_gold,
    EXECUTED_COARSE_PER_DELTA_T,
    EXECUTED_DEG_PER_UNIT,
    FINE_BAND,
    HOLD_BAND,
    V2_PHASE_SUBGOAL,
    XY_TOL,
    YAW_SCALE,
    YAW_TOL_DEG,
)
from robojev.questions import (
    ACTIVE_QIDS,
    AXIS_CANDIDATES,
    COMMIT_STEPS,
    MOVE_QIDS,
    SIZE_QIDS,
    STEP_CANDIDATES,
    SUBGOAL_CANDIDATES,
)

#: robosuite's `OSC_POSE` maps the normalised action range [-1, 1] onto `output_max`
#: `[0.05, 0.05, 0.05, 0.5, 0.5, 0.5]` **per control step** -- what the composer *commands*.
OSC_MAX_DPOS: float = 0.05
OSC_MAX_DROT: float = 0.5

CHUNK_STEPS: int = COMMIT_STEPS

#: **Centimetres the end effector actually travels per normalised action unit, per decision.**
#: Measured (the measurements, `calibrate.py`, five identical control steps per decision, task 0):
#: 0.0449/0.0541/0.0549 m on x/y/z at δ_t = 1.0, exactly linear in δ_t, and the expert rounds
#: that to 0.050 m. `DEFAULT_CM_PER_UNIT` is therefore **5.0, not 25.0**: the commanded 0.25 m is
#: what v1 believed and it is a fifth of what happens. A checkpoint trained on these labels must
#: be served with `delta_t = 1.0` (the measurements's "consequence"), which is why `units("large")` is
#: exactly 1.0 below.
#: What the composer **commands**: 5 control steps x 0.05 m of OSC `output_max`. v1 believed the
#: arm went this far and it goes a fifth of it, which is why every v1 threshold was five times too
#: wide (docs/DESIGN.md).
COMMANDED_CM_PER_UNIT: float = CHUNK_STEPS * OSC_MAX_DPOS * 100.0
#: What it **executes**: the mean of the note's measured 4.49 / 5.41 / 5.49 cm on x / y / z at
#: δ_t = 1.0 (x is ~17 % slower -- the Panda's reach), rounded to the 5.0 the expert runs at.
MEASURED_CM_PER_UNIT: float = EXECUTED_COARSE_PER_DELTA_T * 100.0
#: What the **wrist** executes per unit per decision, measured the same way and by the same
#: module (`expert.EXECUTED_DEG_PER_UNIT`): **28 degrees, not the 143 it is commanded**. The
#: commanded figure is `math.degrees(CHUNK_STEPS * OSC_MAX_DROT)` = `COMMANDED_DEG_PER_UNIT`
#: below, and serving the yaw sizes against it is what made `yaw` unaffordable -- a `large`
#: answer turned the wrist 3.6 degrees and a right angle cost 25 of a 44-decision horizon.
MEASURED_DEG_PER_UNIT: float = EXECUTED_DEG_PER_UNIT
#: What the wrist is **commanded**: 5 control steps x 0.5 rad of OSC `output_max`.
COMMANDED_DEG_PER_UNIT: float = math.degrees(CHUNK_STEPS * OSC_MAX_DROT)
#: **The measured scale is the default**, per the plan's ruling that v2 checkpoints are served at
#: δ_t = 1.0. A server composing these labels with the old manifest's 0.3536 would execute a
#: fifth of every move the expert meant.
DEFAULT_CM_PER_UNIT: float = MEASURED_CM_PER_UNIT
#: The measured scale is the default for the wrist too, for the reason it is for the hand: an
#: answer scaled by what the controller is *asked* for rather than by what it *does* is a
#: threshold five (here, four) times too wide.
DEFAULT_DEG_PER_UNIT: float = MEASURED_DEG_PER_UNIT

#: The three step sizes, **in centimetres executed per decision per axis** -- the unit the state
#: prints its offsets in. They are `expert.AXISWISE_SCALE` times the measured travel, so they are
#: the sizes the 85/100 run was measured with and cannot drift from it: **5.0 / 1.67 / 0.5 cm**
#: (a `large` diagonal covers 8.7 cm). `small` exists because `medium` at 1.67 cm is already
#: larger than the grasp's positioning tolerance, so a two-size vocabulary cannot place the hand
#: on the rim -- and at 0.5 cm it is well inside the ±1.5 cm the rim grasp tolerates on every
#: axis at 100 % (the measurements's tolerance table). That is gate G2's pass condition.
#:
#: Derived **through metres and then scaled**, in that order, rather than from the centimetre
#: constant: `expert.axiswise_answers` compares an error in metres against
#: `band * scale * EXECUTED_COARSE_PER_DELTA_T`, and a band computed in the other order lands one
#: unit-in-the-last-place away. That is invisible everywhere except on a value sitting exactly on
#: a band edge -- which the state, printing a tenth of a centimetre, reaches -- and there the two
#: rules would answer differently. Pinned by
#: `test_the_label_function_is_the_experts_axiswise_rule`.
STEP_LARGE_CM: float = AXISWISE_SCALE["large"] * EXECUTED_COARSE_PER_DELTA_T * 100.0
STEP_MEDIUM_CM: float = AXISWISE_SCALE["medium"] * EXECUTED_COARSE_PER_DELTA_T * 100.0
STEP_SMALL_CM: float = AXISWISE_SCALE["small"] * EXECUTED_COARSE_PER_DELTA_T * 100.0

#: The wrist's three sizes, in **degrees executed per decision** -- `expert.YAW_SCALE`, imported
#: rather than copied for the reason the translation sizes are. At `MEASURED_DEG_PER_UNIT` a
#: `large` yaw commands 0.54 units and turns the wrist 15 degrees, so a right angle costs six
#: decisions: what makes the drawer task's grasp affordable inside a 44-decision horizon.
YAW_LARGE_DEG: float = YAW_SCALE["large"]
YAW_MEDIUM_DEG: float = YAW_SCALE["medium"]
YAW_SMALL_DEG: float = YAW_SCALE["small"]

#: How close an axis must be to its waypoint to answer `hold`, in centimetres:
#: `HOLD_BAND * STEP_SMALL_CM` = 0.6 x 0.5 = **0.3 cm**, the expert's own hold band. Both bands
#: are above a half, and deliberately: the arm does not stop when the command does. Sustained
#: travel runs ~21 % above from-rest (repeated +y chunks: 0.0541 then 0.0657, 0.0653, 0.0647),
#: so a decision that reverses direction overshoots, and at 0.5/0.5 the approach oscillated
#: across the rim for four decisions (docs/DESIGN.md).
#:
#: Written as the literal the state **prints**, because the label has to be a function of the
#: text: `HOLD_BAND * STEP_SMALL_CM` is 0.3 plus one unit in the last place, and a parser reading
#: `[tolerance 0.3]` back out of the string would get the other side of it. `_sign_answer`
#: compares `<=` against this and `expert.axiswise_answers` compares `<` against the un-rounded
#: band; on every value either of them can be handed they are the same function, which
#: `test_the_label_function_is_the_experts_axiswise_rule` checks over a thousand of them.
DEFAULT_TOLERANCE_CM: float = 0.3
#: How close the *hand* must be to the waypoint for the geometry to call it **arrived** -- the
#: expert's `XY_TOL`/`Z_TOL`, 1.3 cm. This is a different question from `hold`, and deliberately
#: a looser one: `hold` is "this axis needs no step", arrival is "the phase may advance, the
#: fingers may close". A tolerance tighter than the smallest step the vocabulary can take would
#: be a phase that answers `hold` for ever without ever being satisfied (expert's `XY_TOL` note).
DEFAULT_ARRIVAL_CM: float = XY_TOL * 100.0
#: The same for the wrist, in degrees -- `expert.YAW_TOL_DEG`, the angle the plan stops turning
#: at, printed on the state's own `Wrist yaw error:` line so the `yaw` label is a function of the
#: text. Five degrees is a third of one `large` answer, so it is a band the vocabulary can land
#: inside rather than one it steps across.
DEFAULT_YAW_TOLERANCE_DEG: float = YAW_TOL_DEG

#: Which slot of LIBERO's 7-d action each axis answer drives.
AXIS_SLOT: dict[str, int] = {"move_x": 0, "move_y": 1, "move_z": 2}
SIGN: dict[str, float] = {"-": -1.0, "hold": 0.0, "+": +1.0}
YAW_SLOT: int = 5


@dataclasses.dataclass(frozen=True)
class StepSizes:
    """The three step sizes in both the unit the state speaks (cm, degrees) and the unit the
    controller speaks (normalised action units), plus the conversion between them.

    Frozen: a composer whose scale can be mutated after a checkpoint was trained against it is
    the same drift the shared serialiser exists to prevent. `calibrate` returns a new one.
    """

    cm: dict[str, float]
    deg: dict[str, float]
    cm_per_unit: float
    deg_per_unit: float

    def units(self, size: str) -> float:
        """One `size` as a normalised translation command, clipped to 1.0 -- the OSC range limit,
        above which a command is a lie about what will happen."""
        return min(self.cm[size] / self.cm_per_unit, 1.0)

    def yaw_units(self, size: str) -> float:
        return min(self.deg[size] / self.deg_per_unit, 1.0)


def calibrate(cm_per_unit: float, deg_per_unit: float | None = None) -> StepSizes:
    """The hook for a re-measured displacement per decision.

    `cm_per_unit` is how many centimetres the end effector *actually moves* when a unit command
    is held for one decision -- 5.0 as measured (docs/DESIGN.md). Pass a new measurement and every step
    size keeps its meaning in centimetres while the command that produces it changes, which is
    the only way a `step` answer and the state's "largest remaining offset" stay in one unit.
    """
    if not cm_per_unit > 0.0:
        raise ValueError(f"cm_per_unit must be positive, got {cm_per_unit!r}")
    return StepSizes(
        cm={"large": STEP_LARGE_CM, "medium": STEP_MEDIUM_CM, "small": STEP_SMALL_CM},
        deg={"large": YAW_LARGE_DEG, "medium": YAW_MEDIUM_DEG, "small": YAW_SMALL_DEG},
        cm_per_unit=float(cm_per_unit),
        deg_per_unit=float(deg_per_unit if deg_per_unit is not None else DEFAULT_DEG_PER_UNIT),
    )


#: The measured scale. `units("large") == 1.0` is δ_t = 1.0, which is what the expert's 85/100
#: was run at and what the served `robojev.json` must carry.
MEASURED_STEPS: StepSizes = calibrate(MEASURED_CM_PER_UNIT, MEASURED_DEG_PER_UNIT)
DEFAULT_STEPS: StepSizes = MEASURED_STEPS


def saturates(size: str, steps: StepSizes = MEASURED_STEPS) -> bool:
    """Does this size ask for more than the controller will take?

    True when `cm[size] / cm_per_unit > 1.0`, i.e. the normalised command clips at the OSC range
    limit and the state's band words would be promising travel the arm cannot deliver. At the
    measured scale `large` is **exactly** 1.0 and nothing saturates -- the sizes were chosen from
    what the arm executes, which is the whole point of `MEASURED_CM_PER_UNIT`. It stays as a
    guard: gate G2 is what decides whether `STEP_LARGE_CM` ever moves, and if it moves up this is
    what says so out loud rather than silently clipping.
    """
    return steps.cm[size] / steps.cm_per_unit > 1.0


def _as_bool(value) -> bool:
    """`answers["grip"]` may be a bool or the strings `"true"`/`"false"`."""
    if isinstance(value, str):
        return value == "true"
    return bool(value)


def select(probabilities: dict[str, float], mode: str = "argmax", rng=None) -> str:
    """Pick one candidate from a question's probability distribution.

    `mode="argmax"` breaks ties by candidate order (dict insertion order), so it is deterministic
    regardless of floating-point noise. `mode="sample"` -- the console's mode -- draws with
    `rng.choice` over the probabilities exactly as given; any temperature has already been applied
    by `DecisionPredictor.predict(temperature=...)`, so nothing here re-softmaxes the distribution
    a second time.
    """
    if mode == "argmax":
        return max(probabilities, key=probabilities.get)
    if mode == "sample":
        rng = rng if rng is not None else np.random.default_rng()
        keys = list(probabilities)
        p = np.asarray([probabilities[k] for k in keys], dtype=np.float64)
        p = p / p.sum()
        return keys[int(rng.choice(len(keys), p=p))]
    raise ValueError(f"unknown selection mode {mode!r}")


def compose(answers: dict, steps: StepSizes = DEFAULT_STEPS) -> np.ndarray:
    """The answers -> one `(7,)` float32 LIBERO action.

    Each axis contributes `SIGN[answer] * steps.units(step)` to its own slot, so three axes move
    **at once**: §2's principle 5, and the countermeasure to failure F5a. Measured, the diagonal
    is worth about six decisions an episode -- 85/100 at a median of 27 decisions against the
    axis-at-a-time vocabulary's 83/100 at 33 (docs/DESIGN.md). `subgoal` is auxiliary and drives nothing.
    """
    shared = answers.get("step")
    action = np.zeros(7, dtype=np.float32)
    for qid, slot in AXIS_SLOT.items():
        choice = answers[qid]
        if choice not in SIGN:
            raise KeyError(f"unknown {qid} answer {choice!r}")
        size = answers.get(f"size_{qid[-1]}") or shared or "medium"
        if size not in STEP_CANDIDATES:
            raise KeyError(f"unknown step size {size!r}")
        action[slot] = SIGN[choice] * steps.units(size)
    yaw = answers.get("yaw") or "hold"
    if yaw not in SIGN:
        raise KeyError(f"unknown yaw answer {yaw!r}")
    # **The wrist has one speed**, and it is not the hand's. It used to take the largest size any
    # axis asked for -- "it is one hand, and an operator slowing the arm down slows the wrist with
    # it" -- and that is exactly wrong for the case the wrist exists for: the hand arrives over
    # the rim first, every axis then answers `hold`/`small`, and the wrist that still has 60
    # degrees to turn crawls at 1.5 degrees a decision (measured: 11 degrees in the first
    # decision and 2 a decision after it, and the approach gives up before the fingers are
    # aligned). There is no `size_yaw` question to ask instead and there does not need to be:
    # `yaw` is a **sign against the tolerance the state prints**, which is what its own criteria
    # already say, and one size that converges is the whole of what a sign needs. `large`
    # (15 degrees) against `DEFAULT_YAW_TOLERANCE_DEG` (8) converges without oscillating -- an
    # error inside 15 degrees lands inside the tolerance in one step, and one outside it shrinks
    # by 15 -- and a right angle costs six decisions.
    action[YAW_SLOT] = SIGN[yaw] * steps.yaw_units("large")
    action[6] = 1.0 if _as_bool(answers["grip"]) else -1.0
    return action


def chunk(answers: dict, steps: StepSizes = DEFAULT_STEPS, k: int = CHUNK_STEPS) -> np.ndarray:
    """`compose(...)` held for `k` identical control steps: the `[5, 7]` array one decision is."""
    return np.tile(compose(answers, steps), (k, 1))


# ------------------------------------------------------------------------ the step-size tiling


def step_size_for(offset_cm: float) -> str:
    """The `step` answer for a largest-remaining-offset of `offset_cm`.

    **Each band is read against the step above it** -- take the largest step whose own overshoot
    the error can afford -- which is `expert.axiswise_answers`' rule and the one that measured
    85/100. Reading each band against its own step instead answers `large` for a 1.2 cm error and
    scores 51/100 (the measurements's implementation warning). `FINE_BAND` is 0.7 rather than 0.5 because
    sustained travel runs 21 % above from-rest and a reversing decision overshoots (docs/DESIGN.md).

        |e| <  0.7 x medium (1.17 cm) -> small
        |e| <  0.7 x large  (3.50 cm) -> medium
        otherwise                     -> large
    """
    # Compared **in metres**, which is the unit `expert.axiswise_answers` compares in. The same
    # comparison written in centimetres lands one unit in the last place away and disagrees with
    # the expert on a value sitting exactly on a band edge -- 3.5 cm, which the state, printing a
    # tenth of a centimetre, reaches. `test_the_label_function_is_the_experts_axiswise_rule`
    # pins the two against each other over a thousand values including every edge.
    magnitude = abs(float(offset_cm)) / 100.0
    large = AXISWISE_SCALE["large"] * EXECUTED_COARSE_PER_DELTA_T
    medium = AXISWISE_SCALE["medium"] * EXECUTED_COARSE_PER_DELTA_T
    if magnitude < FINE_BAND * medium:
        return "small"
    if magnitude < FINE_BAND * large:
        return "medium"
    return "large"


def step_range_words(size: str, steps: StepSizes = MEASURED_STEPS) -> str:
    """How the state names the band a size covers, in centimetres. The same two numbers
    `step_size_for` compares against, so the text and the rule cannot drift.

    A **saturating** size says what it executes as well as what band it covers: a candidate that
    promised an 8 cm step the controller clips to 5 would be the state lying to the model about
    the cost of the answer it is scoring.
    """
    small_max = FINE_BAND * STEP_MEDIUM_CM
    medium_max = FINE_BAND * STEP_LARGE_CM
    # Two decimals, not one: the small/medium threshold is 1.1667 cm and a text that rounded it
    # to "1.2" would tell the model `small` for an offset the rule calls `medium`. The offsets
    # themselves are printed to a tenth, so two decimals here is unambiguous.
    words = {
        "large": f"large range: {medium_max:.2f} and above",
        "medium": f"medium range: {small_max:.2f} to {medium_max:.2f}",
        "small": f"small range: below {small_max:.2f}",
    }[size]
    if saturates(size, steps):
        words += f"; executes {steps.units(size) * steps.cm_per_unit:.1f} cm"
    return words


def band_words(steps: StepSizes = MEASURED_STEPS) -> str:
    """The one line the state prints so every `size_*` question is answerable from the text: the
    two thresholds, and what each size actually executes."""
    small_max = FINE_BAND * STEP_MEDIUM_CM
    medium_max = FINE_BAND * STEP_LARGE_CM
    return (f"Step bands: small below {small_max:.2f} cm, medium {small_max:.2f} to "
            f"{medium_max:.2f} cm, large {medium_max:.2f} and above; one step executes "
            f"{STEP_SMALL_CM:.1f} / {STEP_MEDIUM_CM:.1f} / {STEP_LARGE_CM:.1f} cm.")


def _sign_answer(value: float, tolerance: float) -> str:
    """Inside the band is `hold`. `<=` against the printed 0.3 rather than `<` against the
    un-rounded 0.30000000000000004: the same function, on a tolerance a parser can read back."""
    if abs(float(value)) <= float(tolerance):
        return "hold"
    return "+" if value > 0.0 else "-"


def normalise_phase(phase: str) -> str:
    """A stage name -> a `subgoal` candidate.

    v1's `robojev.memory.phase` names a `locate` stage, which v2 has no candidate for:
    grounding is the once-per-episode `target` question now, so by the time the motion questions
    are asked the target is committed and the stage is `reach`. `expert.py`'s own eight-state
    phase machine maps through `PHASE_FROM_EXPERT` below.
    """
    if phase in ("locate", None, ""):
        return "reach"
    if phase == "done":
        return "retreat"
    if phase not in SUBGOAL_CANDIDATES:
        raise KeyError(f"unknown phase {phase!r}")
    return phase


#: `robojev.expert`'s eight-state phase machine, in v2's six subgoals -- **its own
#: table**, aliased rather than copied, so an expert-labelled row and a tracker-labelled one
#: cannot name different stages for the same decision. (`descend` is `reach` and not `grasp`:
#: the expert's `close` is the decision that shuts the fingers, and `grasp` is the candidate
#: whose text says "this is where they close".)
PHASE_FROM_EXPERT: dict[str, str] = V2_PHASE_SUBGOAL


def labels_from_waypoint(offset_cm, yaw_err: float, holding: bool, phase: str,
                         tolerance: float = DEFAULT_TOLERANCE_CM, *,
                         yaw_tolerance: float = DEFAULT_YAW_TOLERANCE_DEG,
                         arrival: float = DEFAULT_ARRIVAL_CM, rim=(),
                         qids=ACTIVE_QIDS) -> dict:
    """**The** label function: the numbers the state prints -> the gold answer to every question.

    `offset_cm` is the signed `(x, y, z)` the gripper must still travel to reach the waypoint, in
    centimetres; `yaw_err` the signed wrist error in degrees; `holding` whether the fingers are
    closed on the target; `phase` the stage the tracker's facts imply.

    Each motion answer is **one relation** (§2 principle 2): the sign of one number against one
    tolerance. `step` looks at the maximum of the remaining offsets against two thresholds, which
    is a comparison the state performs for the model by printing "Largest remaining offset".

    `grip` is a **latch** and a decision, not a prediction (principle 3, failure F1) -- and it is
    the one answer the closed loop cannot absorb an error in. Measured (docs/DESIGN.md): at p = 0.10 the
    expert scores 1/20 with `grip` corrupted and 13/20 with it exempt; motion answers tolerate
    5-10 % error because the waypoint is re-read from the state every decision, while opening the
    fingers mid-carry drops the bowl.

    The rule is **the stage, and the stage is a fact the state prints** -- which is exactly how
    `expert.decide` decides it, sub-stage by sub-stage:

    * `reach` (the expert's `approach` and `descend`): open, always.
    * `grasp` (its `close`): shut. The stage is only entered at the rim, so the geometry is
      already agreed; `GripLatch` is what checks that at run time for a *model's* answer.
    * `lift`: shut -- the whole point of the stage.
    * `carry`: shut while the fingers still have the target; a bowl that has been dropped is no
      longer within them and the state says `holding nothing`.
    * `place` (its `lower` then `release`): shut until the hand has **arrived** over the
      destination (`arrival`, the expert's 1.3 cm, not the 0.3 cm `hold` band), then open.
    * `retreat`: open -- the object has been let go and the hand is leaving.

    This is the **raw expert latch**: `GripLatch` below is what protects a *model's* latch, and
    it never changes what is reported here.
    """
    offset = np.asarray(offset_cm, dtype=np.float64).reshape(3)
    subgoal = normalise_phase(phase)
    answers: dict = {}
    for index, (move_qid, size_qid) in enumerate(zip(MOVE_QIDS, SIZE_QIDS)):
        value = float(offset[index])
        answers[move_qid] = _sign_answer(value, tolerance)
        # A `hold` axis takes the smallest size, exactly as `expert.axiswise_answers` does: it is
        # not moving, and a size is still asked for it.
        answers[size_qid] = ("small" if answers[move_qid] == "hold" else step_size_for(value))
    # The shared-step ablation: the largest offset *still to be travelled*, because an axis
    # already inside its hold band is not remaining.
    remaining = [abs(float(offset[AXIS_SLOT[qid]])) for qid in AXIS_SLOT
                 if answers[qid] != "hold"]
    answers["step"] = step_size_for(max(remaining) if remaining else 0.0)
    answers["yaw"] = _sign_answer(yaw_err, yaw_tolerance)
    arrived = bool(np.all(np.abs(offset) < float(arrival)))
    if subgoal in ("grasp", "lift"):
        grip = True
    elif subgoal == "carry":
        grip = bool(holding)
    elif subgoal == "place":
        grip = bool(holding) and not arrived
    else:                                   # reach, retreat
        grip = False
    answers["grip"] = grip
    answers["subgoal"] = subgoal
    # `rim` is read off the block the state prints, row by row, and never recomputed from
    # geometry here: the label has to be a function of the text (§2 principle 1), and the text
    # carries a verdict per row precisely so that the rule is "the first that fits and is
    # untried" rather than an arg-max over a column of centimetres.
    answers["rim"] = expert_rim_gold(rim)
    return {qid: answers[qid] for qid in qids}


def largest_remaining(offset_cm, tolerance: float = DEFAULT_TOLERANCE_CM) -> float:
    """The number the state's `Largest remaining offset` line prints, by the same rule
    `labels_from_waypoint` picks the `step` size with."""
    offset = np.asarray(offset_cm, dtype=np.float64).reshape(3)
    remaining = [abs(float(v)) for v in offset if abs(float(v)) >= float(tolerance)]
    return max(remaining) if remaining else 0.0


# -------------------------------------------------------------------- the grip latch's guard


@dataclasses.dataclass(frozen=True)
class LatchGuard:
    """How much a *model's* `grip` answer is allowed to move the latch.

    **Why this exists, measured.** Note §5's sweep replaces each answer with a random other
    candidate with probability p. Motion answers are recoverable -- the waypoint is re-read from
    the state every decision, so a wrong axis is undone by the next one -- but the latch is not:

        p      grip corrupted   grip exempt
        0.05        9/20            16/20
        0.10        1/20            13/20
        0.20        3/20             9/20

    "No motion accuracy rescues a `grip` head below ~0.98" (docs/DESIGN.md), and the note's own
    conclusion is that the latch "needs to be protected structurally". `LATCH_TABLE` is that
    protection, and it is the whole of it: a change is applied when the plan's own stage is one
    that makes it, and refused otherwise. Re-measured against the guard it replaces, with the
    `grip` answer negated at p (docs/DESIGN.md §5): **16/20 against
    15/20 at 0.05, 16/20 against 13/20 at 0.10, 15/20 against 11/20 at 0.20**, with half the
    refusals -- the old one spent most of them on correct changes it had not seen twice yet.

    `enabled=False` is the ablation -- the unguarded latch, for measuring what the guard buys.
    """

    enabled: bool = True

    def config(self) -> dict:
        """The knobs that change behaviour, for the manifest, `robojev.json` and `describe`."""
        return {"grip_guard": self.enabled}


DEFAULT_LATCH_GUARD: LatchGuard = LatchGuard()

#: **Which grip changes make sense in which stage** -- the guard, whole, as a table over the two
#: facts the state text prints: the sub-goal the plan is in and whether the hand has arrived at
#: the waypoint. `"always"` is permitted, `"never"` refused, `"arrived"` permitted only within the
#: arrival tolerance.
#:
#: Read it as the plan's own latch, which is exactly what it is: the fingers may change only
#: where the plan itself changes them.
#:
#: | sub-goal | close | open |
#: | --- | --- | --- |
#: | `reach`, `retreat` | never | **always** -- open fingers while reaching are never wrong, and
#:   this is what lets a failed grasp be retried at all (before, reopening after one was refused
#:   for the rest of the episode and 0 of 9 retries ever succeeded) |
#: | `grasp` | always -- the stage *is* "hold still and shut them" | never |
#: | `lift`, `carry` | never | never -- the one irreversible mistake |
#: | `place` | never | once arrived over the destination |
LATCH_TABLE: dict[str, dict[str, str]] = {
    "reach":   {"close": "never",  "open": "always"},
    "grasp":   {"close": "always", "open": "never"},
    "lift":    {"close": "never",  "open": "never"},
    "carry":   {"close": "never",  "open": "never"},
    "place":   {"close": "never",  "open": "arrived"},
    "retreat": {"close": "never",  "open": "always"},
}


def latch_permitted(proposed: bool, current: bool, *, subgoal: str, arrived: bool) -> bool:
    """Do the tracker's facts permit *this* change of the latch? `LATCH_TABLE`, applied.

    Pure, total over `SUBGOAL_CANDIDATES`, and a function of nothing but the two facts the state
    prints. What it no longer reads is `retrying` -- `TrackerV2.needs_regrounding`, a *grounding*
    flag the guard borrowed to mean "a retry is in progress". Re-grounding cleared it one decision
    later, so reopening after a failed grasp was refused for the rest of the episode: 0 of 9
    retries in run 7 ever succeeded. "Open while reaching" says the same thing without the
    coupling, and says it for every retry rather than for one decision of it.
    """
    if proposed == current:
        return True
    rule = LATCH_TABLE[normalise_phase(subgoal)]["close" if proposed else "open"]
    return rule == "always" or (rule == "arrived" and bool(arrived))


class GripLatch:
    """The latch itself: what the fingers are doing, and what it takes to change it.

    Not used by `labels_from_waypoint`, which reports the expert's raw latch. This is the runtime
    guard a *server* wraps a model's `grip` answer in, and it logs every refusal so the memory
    block can carry it (`log` is `TrackerV2.log_latch`).

    There is no confirmation count. A two-decision confirmation was the third of the note's
    suggested protections and it is the one the table makes redundant: inside a window the change
    is the plan's own next action, so confirming it only costs a decision (about two an episode,
    in a 44-decision horizon where every remaining failure is a time-out); outside a window it is
    refused whether it is asked once or twice.
    """

    def __init__(self, guard: LatchGuard = DEFAULT_LATCH_GUARD, closed: bool = False):
        self.guard = guard
        self.reset(closed)

    def reset(self, closed: bool = False) -> None:
        self.closed = bool(closed)

    def update(self, proposed, *, subgoal: str, arrived: bool, log=None) -> bool:
        """The latch state after a model answered `proposed`. Returns what the fingers do."""
        proposed = _as_bool(proposed)
        if proposed == self.closed:
            return self.closed
        if not self.guard.enabled or latch_permitted(proposed, self.closed, subgoal=subgoal,
                                                     arrived=arrived):
            self.closed = proposed
            return self.closed
        if log is not None:
            log(f"grip {'close' if proposed else 'open'} refused (geometry)")
        return self.closed

    def config(self) -> dict:
        return self.guard.config()


__all__ = [
    "AXIS_CANDIDATES", "AXIS_SLOT", "CHUNK_STEPS", "DEFAULT_ARRIVAL_CM", "DEFAULT_CM_PER_UNIT",
    "DEFAULT_DEG_PER_UNIT", "DEFAULT_LATCH_GUARD", "DEFAULT_STEPS", "DEFAULT_TOLERANCE_CM",
    "DEFAULT_YAW_TOLERANCE_DEG", "FINE_BAND", "GripLatch", "HOLD_BAND", "LatchGuard",
    "OSC_MAX_DPOS", "OSC_MAX_DROT", "PHASE_FROM_EXPERT", "SIGN", "STEP_LARGE_CM",
    "LATCH_TABLE", "STEP_MEDIUM_CM", "STEP_SMALL_CM", "StepSizes", "YAW_LARGE_DEG",
    "YAW_MEDIUM_DEG",
    "YAW_SLOT", "YAW_SMALL_DEG", "calibrate", "chunk", "compose", "labels_from_waypoint",
    "largest_remaining", "latch_permitted", "normalise_phase", "step_range_words",
    "step_size_for", "band_words", "saturates", "MEASURED_CM_PER_UNIT", "MEASURED_DEG_PER_UNIT",
    "MEASURED_STEPS", "COMMANDED_CM_PER_UNIT", "COMMANDED_DEG_PER_UNIT",
]
