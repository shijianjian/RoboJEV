"""`robojev`: a NanoJev-style parallel decision policy, and everything that makes one.

A decision policy answers a fixed set of independent questions about the current state -- one
direction and one step size per world axis, a wrist turn, which way round the rim to stand, and
whether to close the fingers -- in a single forward pass, and composes the answers into one action
rather than picking one option that must speak for the whole step. Each question's candidates mix
only within that question's own set head, which is what lets "move forward" and "close the
gripper" be chosen together without one out-voting the other.

The pieces, and which side of the fine-tune each is on:

* **the question set** -- the questions (`questions`), the state text and the tracker that fills
  it (`state`), the composer (`compose`), the parser that reads a state back (`parse`), the scene
  grounding (`grounding`) and the expert harvest (`rollout`). Both sides read all of it, which is
  the whole point: a training row and an inference request are built from the same numbers by the
  same code, so nothing can drift.
* **the plan** -- `expert`, `skill`, `scene`, `roles`: a scripted controller that is right by
  construction at every state, the small step machine it is written in, and the two rules that
  read a scene (which object the task means, and what stage the episode is in). Code plans; the
  model judges.
* **the pipeline** -- `dagger`, `train`, `slurm`, `dataset`, `runtime`: one DAgger round, the
  fine-tune, the cluster job, the shape of the directory they all read and write, and the
  environment the trainer runs in.
* **serving** -- `policy` (three engines behind one interface), `episode` (the closed loop),
  `recorder` (one episode as a replay bundle), `cli`.
* `envs` -- the environment protocol and the LIBERO adapter. The only simulator import.
* `registry` -- which question set a checkpoint speaks, decoded in one place.
* `jev_api` -- a hosted model, wearing the local predictor's interface.
* `home` -- where this package writes at runtime.

Importing this module costs the standard library and numpy and nothing else: no torch, no
simulator, no network. See `tests/test_package.py`.
"""
from __future__ import annotations

from robojev import home, registry
from robojev.compose import (
    CHUNK_STEPS,
    COMMANDED_CM_PER_UNIT,
    DEFAULT_ARRIVAL_CM,
    DEFAULT_CM_PER_UNIT,
    DEFAULT_LATCH_GUARD,
    DEFAULT_STEPS,
    DEFAULT_TOLERANCE_CM,
    DEFAULT_YAW_TOLERANCE_DEG,
    LATCH_TABLE,
    MEASURED_CM_PER_UNIT,
    MEASURED_DEG_PER_UNIT,
    MEASURED_STEPS,
    PHASE_FROM_EXPERT,
    STEP_LARGE_CM,
    STEP_MEDIUM_CM,
    STEP_SMALL_CM,
    GripLatch,
    LatchGuard,
    StepSizes,
    band_words,
    calibrate,
    chunk,
    compose,
    labels_from_waypoint,
    largest_remaining,
    latch_permitted,
    normalise_phase,
    saturates,
    select,
    step_range_words,
    step_size_for,
)
from robojev.parse import PARSED_QIDS, UnparseableState, parse, read
from robojev.questions import (
    ACTIVE_QIDS,
    ALL_MOTION_QIDS,
    AXIS_CANDIDATES,
    COMMIT_STEPS,
    EPISODE_QIDS,
    HELD_FOR,
    MAX_PATH_TOKENS_V2,
    MEASURED_PATH_TOKENS_V2,
    MOTION_QIDS,
    MOVE_QIDS,
    QUESTION_SET_VERSION,
    QUESTIONS,
    RIM_CANDIDATES,
    SIZE_QIDS,
    STEP_CANDIDATES,
    SUBGOAL_CANDIDATES,
    active_qids,
    candidates,
    episode_question,
    questions_block,
    request,
    scene_candidates,
)
from robojev.registry import DEFAULT_VERSION, VERSIONS
from robojev.state import (
    DEFAULT_RIM_DZ_M,
    DEFAULT_RIM_RADIUS_M,
    MEMORY_RULE_V2,
    SUBSTAGE_LABEL,
    TrackerV2,
    Waypoint,
    objects_cm,
    plan,
    rim_point,
    rounded_cm,
    serialise_v2,
    waypoint_for,
)

__version__ = "0.1.0"

__all__ = [
    "ACTIVE_QIDS", "ALL_MOTION_QIDS", "AXIS_CANDIDATES", "CHUNK_STEPS", "COMMANDED_CM_PER_UNIT",
    "COMMIT_STEPS", "DEFAULT_ARRIVAL_CM", "DEFAULT_CM_PER_UNIT", "DEFAULT_LATCH_GUARD",
    "DEFAULT_RIM_DZ_M", "DEFAULT_RIM_RADIUS_M", "DEFAULT_STEPS", "DEFAULT_TOLERANCE_CM",
    "DEFAULT_VERSION", "DEFAULT_YAW_TOLERANCE_DEG", "EPISODE_QIDS", "GripLatch", "HELD_FOR",
    "LATCH_TABLE", "LatchGuard", "MAX_PATH_TOKENS_V2", "MEASURED_CM_PER_UNIT",
    "MEASURED_DEG_PER_UNIT", "MEASURED_PATH_TOKENS_V2", "MEASURED_STEPS", "MEMORY_RULE_V2",
    "MOTION_QIDS", "MOVE_QIDS", "PARSED_QIDS", "PHASE_FROM_EXPERT", "QUESTIONS",
    "QUESTION_SET_VERSION", "RIM_CANDIDATES", "SIZE_QIDS", "STEP_CANDIDATES", "STEP_LARGE_CM",
    "STEP_MEDIUM_CM", "STEP_SMALL_CM", "SUBGOAL_CANDIDATES", "SUBSTAGE_LABEL", "StepSizes",
    "TrackerV2", "UnparseableState", "VERSIONS", "Waypoint", "__version__", "active_qids",
    "band_words", "calibrate", "candidates", "chunk", "compose", "episode_question", "home",
    "labels_from_waypoint", "largest_remaining", "latch_permitted", "normalise_phase",
    "objects_cm", "parse", "plan", "questions_block", "read", "registry", "request", "rim_point",
    "rounded_cm", "saturates", "scene_candidates", "select", "serialise_v2", "step_range_words",
    "step_size_for", "waypoint_for",
]
