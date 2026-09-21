"""RoboJEV v2's question set: seven per-decision questions and two once-per-episode ones.

Spec `docs/DESIGN.md` §3. Every question here obeys
§2's principle 2 -- *each question is one simple relation*. The v1 set asked `translate` to be an
`argmax(|dx|, |dy|, |dz|)` plus a sign plus a hold threshold in one seven-way choice (failure F4);
here each axis is its own three-way sign question and they are composed, which is also §2's
principle 5 (parallel questions compose into one diagonal move).

`grip` is a **latch**, not a prediction (principle 3): its text asks what the fingers should *be*
doing for the next 5 control steps, and its target is the expert's own hard 0/1, never a measured
outcome frequency. That is failure F1's countermeasure -- a calibrated probability of a rare
outcome never crosses 0.5 and so the v1 policy never closed.

`target` and `destination` are asked **once**, at reset, before any motion (principle 4): which
object the sentence means is decided once and written into memory, and every motor question
afterwards refers to "the target". Their candidates are built from the scene by
`scene_candidates`, which spells out each object's position and its spatial relations to the
others in words, because failure F3 was the model being asked to ground "the bowl between the
plate and the ramekin" implicitly inside every motor decision.

v1's `robojev.questions` is untouched: both sets exist side by side and a request built
here carries `QUESTION_SET_VERSION` in nothing but this module -- the wire shape is upstream's.
"""
from __future__ import annotations

import copy

import numpy as np

#: Bumped whenever a candidate id, a question id or a candidate's *meaning* changes. A checkpoint
#: trained under one version and served under another is reading a different question about the
#: same state, so harvest manifests and `robojev.json` record this string.
QUESTION_SET_VERSION: str = "v2"

#: The commit horizon, in the words every motion candidate ends with (latest-NanoJev the measurements:
#: their Doom candidates say "…for 4 Doom ticks"). One decision is held for `COMMIT_STEPS`
#: control steps of the 20 Hz OSC loop. A model choosing an action has to be told what the action
#: costs before it can weigh it, and it is the difference between "move down" meaning a nudge and
#: meaning an overshoot past the grasp height.
COMMIT_STEPS: int = 5
CONTROL_HZ: float = 20.0
COMMIT_SECONDS: float = COMMIT_STEPS / CONTROL_HZ
HELD_FOR: str = f"Held for the next {COMMIT_STEPS} control steps ({COMMIT_SECONDS:.2f} s)."

#: Three candidates per axis, `hold` in the middle so no ordering trick makes it the default.
AXIS_CANDIDATES: tuple[str, ...] = ("-", "hold", "+")
STEP_CANDIDATES: tuple[str, ...] = ("large", "medium", "small")
SIZE_CANDIDATES: tuple[str, ...] = STEP_CANDIDATES
YAW_CANDIDATES: tuple[str, ...] = ("-", "hold", "+")
#: `retreat` is v2's addition to `robojev.memory.SUBGOALS`: after the release there is a
#: stage with its own waypoint (up and away), and without a name for it the tracker's facts imply
#: `place` forever. `locate` is *not* here -- grounding is the once-per-episode question now, so
#: the stage that used to mean "no target yet" cannot occur while these questions are asked.
SUBGOAL_CANDIDATES: tuple[str, ...] = (
    "reach", "grasp", "lift", "carry", "place", "retreat",
)

#: Every per-decision question this module defines. Which of them are actually **asked** is
#: `ACTIVE_QIDS`, and it is deliberately smaller -- see `active_qids`.
ALL_MOTION_QIDS: tuple[str, ...] = (
    "move_x", "move_y", "move_z", "size_x", "size_y", "size_z", "step", "yaw", "rim", "grip",
    "subgoal",
)

#: The letters a grasp candidate can be listed under. Eight, because that is how many rim
#: directions `expert.GRASP_TURNS` has; the state prints only the ones a run is actually offered.
RIM_CANDIDATES: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G", "H")

#: **The active set**, and the plan's ruling on the motor vocabulary
#: (`docs/DESIGN.md`, §7 of the v2 proposal).
#:
#: *Per-axis sizes, not one shared `step`*: the scripted expert measured both shapes over the same
#: 100 episodes (`notes/2026-09-21-stage9-scripted-expert.md` §4) and the per-axis one scored
#: **85/100 at a median of 27 decisions** against the shared-step vocabulary's 83/100 at 33. Each
#: axis gets the largest step its own remaining error can afford, so an approach that is 6 cm out
#: in y and 1 cm out in z does not crawl in y or overshoot in z.
#:
#: *`yaw` is on*, and the measurement that turned it on is note
#: docs/DESIGN.md. It used to be off because the expert answered `hold` in
#: 100 % of its decisions -- a question with one answer is compute and tokens for zero bits --
#: and the reason it *could* only answer `hold` was that the served yaw sizes were scaled by what
#: the controller is commanded (143 deg/unit) rather than by what it executes (28), so one
#: `large` answer turned the wrist 3.6 degrees and a right angle cost 25 of the 44 decisions. At
#: the measured scale a right angle costs six, and with it the bowl in the top drawer of the
#: wooden cabinet is graspable at all: the fingers have to straddle the rim on the drawer's roomy
#: side, which is a quarter turn off the axis the wrist starts with. 0/10 to 9/10 on that task's
#: inits, with the eight other tasks' first two candidates -- the ones that need no rotation --
#: unchanged.
#:
#: *`step` is the ablation*: the single shared step size, kept so the two shapes can be compared
#: on identical labels, and not asked by default.
#: *`rim` is on*: which way round the target's rim to stand. The plan used to try its own
#: ordered list and discover a blocked rim point by driving a finger into it -- eleven decisions
#: of a 44-decision horizon, twice, on the drawer task. The state now prints each offered
#: candidate with the room its outer finger would have and whether that fits, so the question is
#: one comparison per line and an answer the model can actually make.
MOTION_QIDS: tuple[str, ...] = (
    "move_x", "move_y", "move_z", "size_x", "size_y", "size_z", "yaw", "rim", "grip", "subgoal",
)
ACTIVE_QIDS: tuple[str, ...] = MOTION_QIDS

#: The three per-axis size questions, in axis order.
SIZE_QIDS: tuple[str, ...] = ("size_x", "size_y", "size_z")
#: The three per-axis direction questions, in axis order.
MOVE_QIDS: tuple[str, ...] = ("move_x", "move_y", "move_z")


def active_qids(*, yaw: bool = True, rim: bool = True,
                shared_step: bool = False) -> tuple[str, ...]:
    """The questions a run asks. Configurable, because two of them are ablations.

    `yaw=False` takes the wrist back out, which is the set every checkpoint trained before the
    drawer task was harvested under -- and those checkpoints are still served, from their own
    `robojev.json` (`robojev/policy.py` reads `qids` from the manifest, and
    `expert.candidates` then never offers a grasp that would need a rotation the run cannot
    make). `shared_step=True` swaps the three per-axis sizes for one `step`, which is the
    vocabulary comparison of the measurements -- 83/100 against 85/100 -- run on identical labels.
    """
    qids = ["move_x", "move_y", "move_z"]
    qids += ["step"] if shared_step else list(SIZE_QIDS)
    if yaw:
        qids.append("yaw")
    if rim:
        qids.append("rim")
    qids += ["grip", "subgoal"]
    return tuple(qids)
#: The once-per-episode questions, asked at reset and again only after a failed attempt (§3).
EPISODE_QIDS: tuple[str, ...] = ("target", "destination")


def _axis_question(axis: str, plus: str, minus: str) -> dict:
    """One axis's three-way sign question. The *only* relation it tests is the sign of that one
    axis's waypoint offset against the tolerance the state prints on the same line."""
    return {
        "type": "choice",
        "instructions": (
            f"The state prints the waypoint's {axis} offset -- how far the gripper must still "
            f"move along {axis} to reach the waypoint -- and the tolerance it is judged against. "
            f"Answer '+' when that offset is positive and outside the tolerance, '-' when it is "
            f"negative and outside the tolerance, and 'hold' when it is inside the tolerance. "
            f"Decide {axis} alone; the other axes are asked separately and move at the same time."
        ),
        "criteria": {
            "-": f"Move the gripper along {minus}. {HELD_FOR}",
            "hold": f"Do not move the gripper along {axis}: it is already within tolerance. "
                    f"{HELD_FOR}",
            "+": f"Move the gripper along {plus}. {HELD_FOR}",
        },
    }


def _size_question(axis: str) -> dict:
    """One axis's step size. The relation is that axis's own remaining offset against two
    thresholds the state prints -- never the other axes', which is what a shared `step` made it.

    The bands are read against the step **above** them (take the largest step whose overshoot the
    error can afford). That is `robojev.expert.axiswise_answers`' own rule, and it is
    load-bearing: reading each band against its own step answers `large` for a 1.2 cm error, the
    hand ends 4.6 cm above the rim when the fingers shut, and the same vocabulary scores 51/100
    instead of 85/100 (docs/DESIGN.md).
    """
    return {
        "type": "choice",
        "instructions": (
            f"How far should the gripper move along {axis} this decision? The state prints the "
            f"{axis} offset to the waypoint and the three bands the sizes cover. Answer with the "
            f"size whose band that offset falls in. Answer for {axis} alone; the other axes have "
            f"their own size and move at the same time."
        ),
        "criteria": {
            "large": f"Take a large step along {axis}, about 5 cm: it is still far from the "
                     f"waypoint. {HELD_FOR}",
            "medium": f"Take a medium step along {axis}, about 1.7 cm: close, but not yet within "
                      f"the positioning tolerance. {HELD_FOR}",
            "small": f"Take a small step along {axis}, about 0.5 cm, shorter than the grasp "
                     f"tolerance: settling onto the waypoint. {HELD_FOR}",
        },
    }


QUESTIONS: dict[str, dict] = {
    "move_x": _axis_question("x", "+x (to the right)", "-x (to the left)"),
    "move_y": _axis_question("y", "+y (forward, away from the robot)",
                             "-y (backward, towards the robot)"),
    "move_z": _axis_question("z", "+z (up)", "-z (down)"),
    "size_x": _size_question("x"),
    "size_y": _size_question("y"),
    "size_z": _size_question("z"),
    "step": {
        "type": "choice",
        # One size for the whole move, not one per axis: an operator slowing down slows the hand
        # down, and three independent scales would be three more things to be wrong about. The
        # relation is the largest remaining offset against two thresholds, and the state prints
        # both the number and the range it falls in.
        "instructions": (
            "The state prints the largest remaining offset in centimetres and the two thresholds "
            "that divide the three step sizes. Answer with the size whose range that number "
            "falls in: the hand is travelling when the offset is large, settling when it is "
            "small."
        ),
        "criteria": {
            "large": f"Take a large step, the travelling step: the waypoint is still far away. "
                     f"{HELD_FOR}",
            "medium": f"Take a medium step: the waypoint is close but not yet within the "
                      f"positioning tolerance. {HELD_FOR}",
            "small": f"Take a small step, shorter than the grasp tolerance: the hand is settling "
                     f"onto the waypoint. {HELD_FOR}",
        },
    },
    "yaw": {
        "type": "choice",
        # Yaw only. Roll and pitch were operator dither about a top-down pose a LIBERO-Spatial
        # pick-and-place never changes (v1 ruling 11), and they are not in v2 either.
        "instructions": (
            "The state prints the wrist's yaw error in degrees -- how far the wrist must still "
            "turn about the vertical -- and the tolerance it is judged against. Answer '+' for a "
            "positive error outside tolerance, '-' for a negative one, and 'hold' when the error "
            "is inside the tolerance."
        ),
        "criteria": {
            "-": f"Turn the wrist clockwise about the world's vertical axis, seen from above. "
                 f"{HELD_FOR}",
            "hold": f"Keep the wrist's orientation where it is. {HELD_FOR}",
            "+": f"Turn the wrist anticlockwise about the world's vertical axis, seen from "
                 f"above. {HELD_FOR}",
        },
    },
    "rim": {
        "type": "choice",
        # One comparison per line, never an arg-max over a column of numbers: the state lists
        # each candidate with the room its outer finger would have, the room it needs and the
        # verdict, and says which ones have already been tried. §2 principle 2.
        "instructions": (
            "The state lists the grasp candidates as lettered lines: which side of the target to "
            "stand on, how far the wrist must turn to close there, how much room the outer "
            "finger would have and whether that fits. Answer with the letter of the first "
            "candidate that fits and has not been tried yet. If none of them fits, answer with "
            "the first that has not been tried."
        ),
        "criteria": {letter: f"Stand on the rim point listed as {letter}. {HELD_FOR}"
                     for letter in RIM_CANDIDATES},
    },
    "grip": {
        "type": "boolean",
        # A latch, and a decision rather than a prediction (principle 3, failure F1). The boolean
        # path is one token path with logits [0, z]; `validate_request` accepts only the keys
        # {"false", "true"} for a boolean question's criteria.
        "instructions": (
            "Keep the fingers closed for the next 5 control steps? Answer true while the fingers "
            "must hold what they are holding, and true on the step that closes them on the "
            "target when the state says the gripper is at the waypoint in the grasp stage. "
            "Answer false while the hand is still travelling, and false once the object has been "
            "released where the task wants it. This is an instruction to the fingers, not a "
            "prediction that the grasp will succeed."
        ),
        "criteria": {
            "true": f"Keep the fingers closed: close them now on the target, or keep holding what "
                    f"is already in them. {HELD_FOR}",
            "false": f"Keep the fingers open: there is nothing to hold yet, or what was carried "
                     f"has been released. {HELD_FOR}",
        },
    },
    "subgoal": {
        "type": "choice",
        # Auxiliary, and the DAgger diagnostic: when the model's own rollout disagrees with the
        # tracker about which stage it is in, that is the row worth relabelling.
        "instructions": (
            "Which stage of the task the state's facts imply. The state prints the stage the "
            "tracker believes it is in, the gripper's width and what it is holding, and the "
            "waypoint it is steering to."
        ),
        "criteria": {
            "reach": "Travelling towards the target, fingers open, not yet at the grasp pose.",
            "grasp": "At the target's rim with the fingers open: this is where they close.",
            "lift": "Holding the target, raising it clear of the table.",
            "carry": "Holding the target clear of the table, moving it over the destination.",
            "place": "Over the destination, lowering the target onto it and opening the fingers.",
            "retreat": "The target has been released; move the hand up and away from it.",
        },
    },
}

#: The once-per-episode questions' fixed half. Their `criteria` are built per scene by
#: `scene_candidates` and spliced in by `episode_question`, because the candidate *set* is the
#: scene: which objects exist is not knowable at import time.
EPISODE_INSTRUCTIONS: dict[str, str] = {
    "target": (
        "Which object in the scene does the task sentence tell the robot to pick up? Each "
        "candidate gives that object's position in centimetres from the gripper and where it "
        "lies relative to the other objects. Choose the one the sentence describes. This is "
        "asked once, before any motion; every later question refers to your answer as 'the "
        "target'."
    ),
    "destination": (
        "Where does the task sentence say the target must end up? Each candidate gives that "
        "object's position in centimetres from the gripper and where it lies relative to the "
        "other objects. Choose the one the sentence names as the destination. This is asked "
        "once, before any motion."
    ),
}

QIDS: tuple[str, ...] = tuple(QUESTIONS)

#: The `max_length` both halves of the model are run at for v2, and the budget gate G6 measures
#: `MEASURED_PATH_TOKENS_V2` against. **1024, not v1's 1536**: v2's state is 40 % of v1's -- the
#: quaternions, the absolute world coordinates and the prose summary are gone -- and the measured
#: worst case sits far enough under 1024 that the padded-token cost of a path halves. Upstream
#: raises rather than truncating an oversized path (`predict_toy_decisions.py:110-112`), so this
#: is a real constraint: raise it rather than shorten the state if a suite ever does not fit.
MAX_PATH_TOKENS_V2: int = 1024

#: **Measured**, not estimated: the longest (state, question, candidate) token path this question
#: set can produce, under the Qwen3-0.6B tokenizer at NanoJev's pinned revision, counted through
#: upstream's own prompt template (`State:` / `Question type:` / `Question:` / `Candidate:` /
#: `Decision:`, plus the EOS).
#:
#: The worst case: LIBERO-Spatial's five objects at their widest spelling, the suite's longest
#: instruction, the per-axis annotations on, a full `Events` line and a full `Last 3` block, and
#: every (question, candidate) path -- including the once-per-episode `target` candidates, whose
#: spatial-relation text is the longest single candidate here, and with **every** question
#: switched on (`ALL_MOTION_QIDS`), none of which the default active set asks together. The
#: **state alone is 521 tokens** in that worst case; the longest motion path is 662
#: (`move_x/hold`) and the longest path of all is 682 (`target/plate_1`).
#:
#: Gate G6 wants **< 1024 with margin** and this is 682: 342 tokens of margin. The default active
#: set is shorter again, since it asks neither `yaw` nor the `step` ablation and its state carries
#: no `Wrist yaw error` line. Upstream raises rather than truncating an oversized path, so
#: re-measure whenever the state text or the question set changes.
MEASURED_PATH_TOKENS_V2: int = 682

# A typo in the tables above (a wrong criteria count, a boolean key outside {"false", "true"})
# should fail at import time, not at the server's first request.
for _qid, _q in QUESTIONS.items():
    if _q["type"] == "choice":
        assert 2 <= len(_q["criteria"]) <= 255, (
            f"question {_qid!r}: choice needs 2-255 criteria, got {len(_q['criteria'])}"
        )
    elif _q["type"] == "boolean":
        assert set(_q["criteria"]) <= {"false", "true"}, (
            f"question {_qid!r}: boolean criteria keys must be a subset of "
            f"{{'false', 'true'}}, got {set(_q['criteria'])}"
        )
# Every motion candidate states what one answer commits to (latest-NanoJev the measurements). `subgoal`
# is exempt: it names a stage, it does not command the arm for a quarter of a second.
for _qid in [q for q in ALL_MOTION_QIDS if q != "subgoal"]:
    for _cid, _text in QUESTIONS[_qid]["criteria"].items():
        assert _text.endswith(HELD_FOR), f"{_qid}/{_cid} does not state the commit horizon"
del _qid, _q, _cid, _text


# ------------------------------------------------------------------ the scene, in words

#: Two objects closer than this along an axis are not "left of" or "in front of" each other;
#: they are level. Centimetres.
LEVEL_CM: float = 2.0
#: How far off the line between two objects a third may be and still be "between" them, and how
#: far apart those two must be for the phrase to mean anything. Centimetres.
BETWEEN_OFF_CM: float = 6.0
BETWEEN_SPAN_CM: float = 8.0
#: Horizontal radius within which one object sits "on top of" another, and the height above it
#: that makes it "on top of" rather than "under". Centimetres.
ON_TOP_XY_CM: float = 8.0
ON_TOP_DZ_CM: float = 2.0
#: How many pairwise relation phrases one candidate carries before it is truncated. The whole
#: scene block is paid for in tokens once per episode, but the budget is still a budget.
MAX_RELATIONS: int = 3


def _fmt1(value: float) -> str:
    """One decimal with an explicit sign, and never `-0.0`: at this resolution a value that
    rounds to zero is level, and a minus sign there reads as a direction it is not."""
    rounded = round(float(value), 1)
    return format(rounded if rounded != 0.0 else 0.0, "+.1f")


def _relations(sid: str, here: np.ndarray, others: dict[str, np.ndarray]) -> list[str]:
    """`sid`'s spatial relations to every other object, in words, best first.

    `on top of` and `between A and B` come first because they are the phrases a LIBERO-Spatial
    instruction actually uses ("the bowl between the plate and the ramekin", "the bowl on the
    cookie box"); the pairwise left/right/front/behind phrases follow, nearest neighbour first,
    and `nearest to` closes it so a candidate always says something even in a two-object scene.
    """
    phrases: list[str] = []
    for name, pos in others.items():
        if (float(np.hypot(here[0] - pos[0], here[1] - pos[1])) <= ON_TOP_XY_CM
                and here[2] - pos[2] > ON_TOP_DZ_CM):
            phrases.append(f"on top of {name}")

    names = list(others)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pa, pb = others[a], others[b]
            span = pb[:2] - pa[:2]
            length = float(np.linalg.norm(span))
            if length < BETWEEN_SPAN_CM:
                continue
            unit = span / length
            rel = here[:2] - pa[:2]
            along = float(rel @ unit)
            across = float(abs(rel[0] * unit[1] - rel[1] * unit[0]))
            if 0.0 < along < length and across <= BETWEEN_OFF_CM:
                phrases.append(f"between {a} and {b}")

    order = sorted(others, key=lambda n: float(np.linalg.norm(others[n][:2] - here[:2])))
    for name in order:
        pos = others[name]
        words = []
        if here[0] - pos[0] <= -LEVEL_CM:
            words.append("left of")
        elif here[0] - pos[0] >= LEVEL_CM:
            words.append("right of")
        if here[1] - pos[1] >= LEVEL_CM:
            words.append("behind")
        elif here[1] - pos[1] <= -LEVEL_CM:
            words.append("in front of")
        for word in words:
            phrases.append(f"{word} {name}")

    phrases = phrases[:MAX_RELATIONS]
    if order:
        phrases.append(f"nearest to {order[0]}")
    return phrases


def scene_candidates(objects_cm: dict[str, tuple[float, float, float]]) -> dict[str, str]:
    """`{short_id: (x, y, z) in gripper-relative centimetres}` -> `{short_id: candidate text}`.

    Each candidate is the object's position and where it lies relative to the others, in words --
    the whole of what `target`/`destination` have to ground against (failure F3). The positions
    are the same gripper-relative centimetres every other line of the v2 state is written in, so
    the model never has to subtract two coordinate triples to use them.
    """
    arrays = {sid: np.asarray(pos, dtype=np.float64).reshape(3)
              for sid, pos in objects_cm.items()}
    out: dict[str, str] = {}
    for sid, here in arrays.items():
        others = {n: p for n, p in arrays.items() if n != sid}
        where = f"at x {_fmt1(here[0])}, y {_fmt1(here[1])}, z {_fmt1(here[2])} cm"
        rel = _relations(sid, here, others)
        out[sid] = f"{sid}, {where}" + (f"; {', '.join(rel)}." if rel else ".")
    return out


def episode_question(qid: str, objects_cm: dict[str, tuple[float, float, float]]) -> dict:
    """The `target` or `destination` question for one scene: upstream's `{type, instructions,
    criteria}`, with `scene_candidates`' text as the criteria."""
    if qid not in EPISODE_INSTRUCTIONS:
        raise KeyError(f"not an episode question: {qid!r}")
    return {
        "type": "choice",
        "instructions": EPISODE_INSTRUCTIONS[qid],
        "criteria": scene_candidates(objects_cm),
    }


def questions_block(qids=ACTIVE_QIDS, *, objects_cm=None) -> dict:
    """A deep copy of the named questions, safe to drop into a request payload.

    `objects_cm` is required exactly when `qids` names `target` or `destination`; the motion
    questions do not depend on the scene at all.
    """
    block: dict[str, dict] = {}
    for qid in qids:
        if qid in EPISODE_INSTRUCTIONS:
            if objects_cm is None:
                raise ValueError(f"question {qid!r} needs objects_cm to build its candidates")
            block[qid] = episode_question(qid, objects_cm)
        else:
            block[qid] = copy.deepcopy(QUESTIONS[qid])
    return block


def request(state_text: str, *, state_id: str = "state", qids=ACTIVE_QIDS,
            objects_cm=None) -> dict:
    """NanoJev's exact request shape for one state: `{"states": [{"id", "state", "questions"}]}`.

    Those three keys and no others -- upstream's `validate_request` rejects a fourth top-level key
    with a 400, which is why the tracker's facts ride *inside* `state` rather than beside it.
    """
    return {
        "states": [
            {
                "id": state_id,
                "state": state_text,
                "questions": questions_block(qids, objects_cm=objects_cm),
            }
        ]
    }


def candidates(qid: str) -> tuple[str, ...]:
    """The candidate ids of question `qid`, in the order they were declared."""
    return tuple(QUESTIONS[qid]["criteria"])
