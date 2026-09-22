"""The planner: code that proposes where the hand should go next, and nothing else.

Two modules. `executor` drives a skill's stages as data -- progress, alternatives, outcomes,
dwell -- and knows nothing about bowls. `pick_and_place` is the one skill written against it: the
stage table, the grasp candidates round a target's rim and the scene geometry they read.

The planner's output is a waypoint, a sub-stage, the grasp candidates and whether the target is
held. `robojev.state` prints that as text; the model answers questions about the text; the labels
are read back off the printed numbers (`robojev.compose`, `robojev.parse`). The planner never
answers a question and never drives the arm.
"""
from robojev.planner import executor, pick_and_place
from robojev.planner.pick_and_place import PHASES, PICK_AND_PLACE, SUBGOALS, PlannerError, new_phase, plan

__all__ = ["PHASES", "PICK_AND_PLACE", "PlannerError", "SUBGOALS", "executor", "new_phase",
           "pick_and_place", "plan"]
