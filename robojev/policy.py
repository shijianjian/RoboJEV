"""The policy: three engines, one interface.

Everything that drives an arm here goes through two methods -- `reset(instruction)` and
`act(obs) -> (chunk, decisions)` -- and the three things that can be behind them are:

* **`ExpertPolicy`** -- the scripted expert (`robojev.expert`) answering the same ten questions a
  learned policy answers. No weights, a few milliseconds a decision on a CPU, and it **finishes
  the task** (90/100 LIBERO-Spatial episodes), which is the point: a baseline that never picks the
  bowl up cannot tell a half-trained checkpoint from a broken one.
* **`ModelPolicy`** -- a local NanoJev fine-tune. The checkpoint declares which question set it
  speaks, what step size its answers mean and what tracker its rows were rendered with, and a
  mismatch on any of those is refused at construction rather than served: a checkpoint served
  under a vocabulary it was not trained on answers a different question about the same state,
  silently, for a whole episode.
* **`ModelPolicy(api="typesafe")`** -- the *hosted* model (`robojev.jev_api`), answering the same
  question block about the same state text over HTTPS. That is the whole of the difference: the
  questions, the serialiser, the plan, the tracker, the latch guard, the grounding rule and the
  composer are shared, so "their model and ours" on one task is a comparison of two models rather
  than of two harnesses.

`decisions` is the same block from all three: `{qid: {probabilities, choice, overridden}}` per
question, plus `meta` carrying the state text the answer was read off, the plan's sub-stage, the
waypoint, the grounding and what the grip guard did. A replay bundle is that block, frame by
frame.

Three rules this file keeps.

1. **Nothing heavy at module level.** torch, transformers and NanoJev are all behind
   `_ensure_loaded`, which only `act` calls, so importing this module costs numpy.
2. **NanoJev's `scripts/` goes on `sys.path` before `import predict_toy_decisions`.** It is not a
   package -- no `pyproject.toml`, no `__init__.py`, sibling modules imported by bare name -- so
   it is pinned in `robojev/trainer/nanojev.json` and cloned into `$ROBOJEV_HOME/src/`.
3. **CUDA is hard-required by upstream and there is no CPU path.** `DecisionPredictor.__init__`
   raises on a non-CUDA device and again on a device without bf16, and the fp32 parameters are
   ~2.4 GB resident. `describe()` says so, and `describe()` itself deliberately loads nothing.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np

from robojev import episode as episode_mod
from robojev import grounding as grounding_mod
from robojev import jev_api, registry, runtime
from robojev import expert as expert_mod
from robojev.compose import MEASURED_STEPS, calibrate, chunk as chunk_v2, saturates, select
from robojev.questions import (
    ACTIVE_QIDS,
    QUESTION_SET_VERSION,
    candidates as v2_candidates,
    questions_block as v2_questions_block,
    request as v2_request,
)
from robojev.roles import scene_roles as shared_scene_roles
from robojev.state import TrackerV2, serialise_v2

#: k = 5 control steps per decision, 0.25 s at 20 Hz. One decision, one chunk, all of it executed
#: -- there is no sliding window.
CHUNK_STEPS = 5
ACTION_DIM = 7

#: What the provenance file beside a checkpoint is called, and the name it used to have. Reading
#: the older one keeps a checkpoint trained before the rename loadable; only the new one is
#: written (`robojev.train.CHECKPOINT_META`).
CHECKPOINT_META = "robojev.json"
LEGACY_CHECKPOINT_META = "robopp.json"

#: Upstream's own default and the only precision the released checkpoints were run at.
PRECISION = "bf16"
#: `0` means "all questions of all states in one batch". That single forward is the property the
#: whole design rests on: a translation and a grasp are chosen together rather than in sequence.
BATCH_QUESTIONS = 0

DEFAULT_SELECTION = "argmax"

#: The scripted policy has no weights, so its "revision" is this file's behaviour.
EXPERT_REVISION = "scripted-v2"

#: Bytes of `best.safetensors` above which the parameters are stored in bf16 rather than fp32.
#:
#: Upstream loads the weights as fp32 whatever `precision` says, and `bf16` is only an autocast
#: around the forward -- which does not halve the resident copy, it *adds* a cached bf16 copy of
#: every weight the region touches. That is fine for a 0.6B checkpoint (2.3 GB on disk, 3.1 GB
#: resident) and impossible for a 4B one at 16 GB on a 24 GB card. 8 GiB is comfortably above the
#: first and comfortably below the second.
BF16_WEIGHTS_BYTES: int = 8 * 1024 ** 3

#: How sure the grounding forward has to be before its own answer is the one committed.
#:
#: Below it the rule resolver is used instead. The two failure modes are not symmetrical: a rule
#: that names the wrong bowl loses the episode, and a *model* that has no opinion at all -- a flat
#: distribution over two identical bowls -- has not grounded anything, so taking its argmax would
#: be taking a coin flip and recording it as a decision. 0.5 is "more likely than everything else
#: together", which for a two-bowl scene is exactly where the answer stops being a tie.
GROUNDING_MIN_P: float = 0.5

#: `"rule"` commits the **text rule**'s pair and still runs and reports the forward; `"forward"`
#: commits the model's own answer when it is sure enough. The default is `rule` because of a
#: measurement, not a preference: the first v2 checkpoint grounds *confidently* wrong (p 0.957 on
#: the bowl the sentence does not name), so `GROUNDING_MIN_P` does not protect the episode, while
#: the text rule scored 610/610 and 515/520. There is no mode that skips grounding altogether:
#: every motor question says "the target", and a policy that never grounded would be asking about
#: an object nothing named.
GROUND_MODES: tuple[str, ...] = ("rule", "forward")
DEFAULT_GROUND: str = "rule"

#: The three engines, as `build()` spells them.
ENGINES: tuple[str, ...] = ("expert", "model", "jev")


class PolicyError(RuntimeError):
    """The policy was asked for something it cannot answer: no privileged state, no upstream
    clone, an override naming a candidate that does not exist."""


# ------------------------------------------------------------------------------ selection mode

def parse_selection(selection: str | None, temperature: float | None = None) -> tuple[str, float]:
    """`("argmax", 1.0)` or `("sample", T)`, from the record's own spelling.

    A bare `"sample"` takes `temperature`, and `"argmax"` is always T = 1: tempering cannot change
    an argmax, and reporting the model's own calibrated distribution is what a console watching
    the bars actually wants to see.
    """
    text = (selection or DEFAULT_SELECTION).strip()
    if text == "argmax":
        return "argmax", 1.0
    mode, _, tail = text.partition("@")
    if mode != "sample":
        raise PolicyError(f"unknown selection mode {selection!r}; expected 'argmax' or 'sample@<T>'")
    if not tail:
        t = 1.0 if temperature is None else float(temperature)
    else:
        try:
            t = float(tail)
        except ValueError:
            raise PolicyError(f"selection {selection!r}: {tail!r} is not a temperature") from None
    if not (t > 0) or t != t or t in (float("inf"), float("-inf")):
        raise PolicyError(f"selection {selection!r}: the temperature must be a finite positive number")
    return "sample", t


def selection_text(mode: str, temperature: float) -> str:
    """The canonical spelling of `(mode, temperature)`, as a record stores it."""
    return "argmax" if mode == "argmax" else f"sample@{temperature:g}"


def _override(qid: str, value, want) -> str:
    """One override, normalised to the candidate id the question actually declares.

    `grip` is a boolean whose candidate ids are the strings `"true"`/`"false"`, and a caller (or a
    JSON round trip) may send either those or a real bool; everything else must name one of its
    question's candidates exactly, because a typo silently ignored would look like the model
    disagreeing with the operator.
    """
    if qid == "grip" and isinstance(value, bool):
        value = "true" if value else "false"
    value = str(value)
    if value not in want:
        raise PolicyError(f"override for {qid!r}: {value!r} is not one of {list(want)}")
    return value


def _ordered(qid: str, probabilities: dict, want) -> dict[str, float]:
    """One question's distribution, in the question block's own candidate order.

    Two reasons it is reordered rather than passed through. `select`'s argmax breaks ties by dict
    order, and upstream builds the answer's dict from its own candidate list, so a tie would
    otherwise be broken by whichever order the model's answer happened to carry. And anything
    rendering bars should get them in the order the question declared its candidates.

    A distribution whose keys are not this question's candidates is a checkpoint trained against a
    different question block -- the one failure this must not paper over, because the composed
    action would silently mean something else.
    """
    want = tuple(want)
    got = set(probabilities)
    if got != set(want):
        raise PolicyError(
            f"question {qid!r}: the checkpoint answered candidates {sorted(got)}, but this "
            f"question declares {list(want)}. The checkpoint was trained against a different "
            f"question set than this one."
        )
    return {c: float(probabilities[c]) for c in want}


def _questions_for_describe(block: dict) -> dict:
    """`{qid: {"type", "candidates": [...], "criteria", "instructions"}}`.

    A question block keys the candidate descriptions under `criteria`, which is NanoJev's request
    schema and not a shape a caller should have to know.
    """
    return {
        qid: {
            "type": q["type"],
            "candidates": list(q["criteria"]),
            "criteria": dict(q["criteria"]),
            "instructions": q["instructions"],
        }
        for qid, q in block.items()
    }


def _dist_version(name: str) -> str:
    """A distribution's version without importing it."""
    from importlib import metadata

    try:
        return metadata.version(name)
    except Exception:
        return "unknown"


def _choice(qid: str, value) -> str:
    """One answer as the candidate id a record and a front end speak."""
    if qid == "grip":
        return "true" if value in (True, "true") else "false"
    return str(value)


# ----------------------------------------------------------------------------------- the base

class Policy:
    """What the episode loop needs, and what all three engines have.

    Subclasses fill `describe`, `reset` and `act`. `protocol()` is derived from `describe()`, so
    a policy states how it wants to be run in one place and the loop obeys it.
    """

    #: This policy reads the simulator's own object poses rather than pixels.
    wants_privileged: bool = True
    family: str = "decision"
    #: Settling steps before the first decision. A policy reading exact poses needs none.
    wait_steps: int = 0

    suite: str

    def describe(self) -> dict:                                   # pragma: no cover - abstract
        raise NotImplementedError

    def reset(self, instruction: str) -> None:                    # pragma: no cover - abstract
        raise NotImplementedError

    def act(self, obs: dict, overrides: dict | None = None, selection: str | None = None):
        raise NotImplementedError                                 # pragma: no cover - abstract

    def close(self) -> None:
        """Release whatever the engine holds. Nothing, for the scripted one."""

    def protocol(self, max_steps: int | None = None) -> episode_mod.Protocol:
        cap = (max_steps if max_steps is not None
               else episode_mod.UPSTREAM_MAX_STEPS.get(self.suite,
                                                       max(episode_mod.UPSTREAM_MAX_STEPS.values())))
        return episode_mod.Protocol(max_steps=int(cap), wait_steps=self.wait_steps,
                                    execute_steps=CHUNK_STEPS, chunk_size=CHUNK_STEPS,
                                    privileged_state=self.wants_privileged, family=self.family)

    # -- shared helpers -------------------------------------------------------------------

    def _new_tracker(self, qids=ACTIVE_QIDS) -> TrackerV2:
        horizon = episode_mod.UPSTREAM_MAX_STEPS.get(
            self.suite, max(episode_mod.UPSTREAM_MAX_STEPS.values())) // CHUNK_STEPS
        return TrackerV2(horizon=horizon, every=CHUNK_STEPS, qids=tuple(qids))

    @staticmethod
    def _privileged(obs: dict) -> dict:
        privileged = obs.get("privileged")
        if not privileged:
            raise PolicyError(
                "act: obs carries no privileged scene state. This policy declares "
                "privileged_state, so the episode loop supplies it on a simulator that has one; "
                "this one answered None -- which is a configuration error, not something to paper "
                "over with a proprioception-only state string the model was never trained on."
            )
        return privileged

    @staticmethod
    def _reset_tracker(tracker: TrackerV2, instruction: str) -> None:
        """Everything a tracker holds is a statement about *this* episode.

        Carried into a fresh scene it would be a confident, detailed and entirely false account of
        a scene that no longer exists, a `Target` naming an object the new scene may not contain,
        and a closed latch holding something it never grasped.
        """
        tracker.instruction = instruction or None
        tracker.reset()
        tracker.target = tracker.destination = None
        tracker.chosen_at = None
        tracker.source = None


# ------------------------------------------------------------------------- the scripted expert

class ExpertPolicy(Policy):
    """The scripted expert answering the ten questions, with hard 0/1 distributions.

    It does not *have* a belief about `move_z`, it has an answer, and a fabricated 0.85 would be a
    number a reader could mistake for confidence. The softened target exists for the training arm
    (`rollout.RolloutConfig.soft_targets`) and not here.

    The tracker is here even though the expert does not read it back: the baseline and the model
    have to be comparable, and anything reading `meta.state` off one run and off the other should
    be reading the same kind of text about the same scene.
    """

    wait_steps = 0

    def __init__(self, suite: str = "libero_spatial", seed: int = 7,
                 selection: str = DEFAULT_SELECTION, revision: str | None = None):
        self.suite = suite
        self.seed = int(seed)
        # Validated at construction: a typo in the mode should fail the run, not the tenth step.
        parse_selection(selection)
        self.selection = selection
        self.revision = revision or EXPERT_REVISION
        self.instruction = ""
        self.tracker = self._new_tracker()
        self.phase = expert_mod.new_phase()
        self.released = False
        self.rng = np.random.default_rng(self.seed)

    # -- the protocol ----------------------------------------------------------------------

    def describe(self) -> dict:
        mode, _ = parse_selection(self.selection)
        return {
            "policy": "expert",
            "family": self.family,
            "checkpoint": {"repo": None, "revision": self.revision},
            "questions": _questions_for_describe(v2_questions_block(ACTIVE_QIDS)),
            "questions_version": QUESTION_SET_VERSION,
            "protocol": {
                "wait_steps": self.wait_steps,
                "max_steps": dict(episode_mod.UPSTREAM_MAX_STEPS),
                "seed": self.seed,
                "chunk_size": CHUNK_STEPS, "execute_steps": CHUNK_STEPS,
                "action_dim": ACTION_DIM,
                "family": self.family,
                "privileged_state": True,
                "selection": self.selection,
                "questions_version": QUESTION_SET_VERSION,
                "questions": v2_questions_block(ACTIVE_QIDS),
                "delta_t": expert_mod.EXPERT_DELTA_T,
                "delta_r": expert_mod.EXPERT_DELTA_R,
                "cm_per_unit": MEASURED_STEPS.cm_per_unit,
                "step_sizes_cm": dict(MEASURED_STEPS.cm),
                "memory": self.tracker.settings(),
                "ground": {"mode": "rule", "order": ["rule", "scene_roles"]},
            },
            "versions": {"policy_repo": "none", "numpy": np.__version__},
            # Deterministic under argmax: same state, same answer, no model and no sampling.
            "nondeterminism": [] if mode == "argmax" else [
                f"selection {self.selection}: each group is drawn from its distribution with a "
                f"generator seeded from the run seed ({self.seed})"
            ],
        }

    def reset(self, instruction: str) -> None:
        self.instruction = instruction
        self.rng = np.random.default_rng(self.seed)
        self._reset_tracker(self.tracker, instruction)
        self.phase = expert_mod.new_phase()
        self.released = False

    # -- the decision ----------------------------------------------------------------------

    def act(self, obs: dict, overrides: dict | None = None, selection: str | None = None):
        """`([5, 7], decisions)`.

        The three-call order the shared renderer requires -- `observe`, render, `answer` -- and
        for the same reason: `observe` closes out the previous decision, so the `Last 3` block's
        effect column is a difference between two *observed* states rather than a restatement of
        what was commanded.

        The action is composed with the step sizes the expert was measured with
        (`MEASURED_STEPS`, δ_t = 1.0). Composing these answers at the demonstrations' 0.3536 would
        execute a fifth of every move they mean, which is the measured reason an episode at the
        demonstrations' δ ran out of horizon.
        """
        privileged = self._privileged(obs)
        state8 = np.asarray(obs["state"], dtype=np.float64).reshape(-1)

        grounding = None
        if (not (self.tracker.target in privileged and self.tracker.destination in privileged)
                or self.tracker.needs_regrounding()):
            grounding = self._ground(privileged, state8)

        self.tracker.observe(step=self.tracker.decisions * CHUNK_STEPS, proprio=state8,
                             objects=privileged, released=self.released)
        answers, self.phase, meta = expert_mod.answers_v2(
            state8, privileged, self.instruction, self.phase,
            delta_t=expert_mod.EXPERT_DELTA_T,
        )
        # `roles.phase`, which the tracker delegates its stage to, has no `retreat`: a released
        # object with the fingers open is a stage of its own and only the decider that opened them
        # knows it happened. Recorded after the decision, so it colours the *next* state.
        self.released = self.released or meta["phase"] in ("release", "retreat", "done")

        overridden: list[str] = []
        for qid, forced in (overrides or {}).items():
            if qid not in ACTIVE_QIDS:
                raise PolicyError(
                    f"override for {qid!r}: not one of this question set's {list(ACTIVE_QIDS)}")
            value = _override(qid, forced, v2_candidates(qid))
            answers[qid] = (value == "true") if qid == "grip" else value
            overridden.append(qid)

        meta["state"] = serialise_v2(state8, privileged, self.instruction, self.tracker,
                                     annotate=self.tracker.annotate)
        meta["step"] = self.tracker.decisions
        meta["overridden"] = overridden
        meta["selection"] = "argmax"
        meta["questions_version"] = QUESTION_SET_VERSION
        meta["latched"] = bool(self.phase.get("rim_axis") is not None)
        meta["released"] = bool(self.released)
        waypoint = self.tracker.waypoint
        meta["waypoint"] = None if waypoint is None else waypoint.label
        meta["subgoal"] = self.tracker.subgoal
        meta["substage"] = self.tracker.substage
        meta["grounding"] = grounding
        meta["destination"] = self.tracker.destination
        meta["target_rule"] = self.tracker.source
        meta["step_sizes_cm"] = dict(MEASURED_STEPS.cm)
        # `offset_cm` is the offset to the point the **expert** is steering to and `waypoint_cm`
        # the one the **state** names, and they are reported separately because they are not
        # always the same point: the approach flies to a hover 8 cm above the rim before it
        # descends and the tracker's six subgoals have no name for that sub-stage.
        row_offsets = self.tracker.last_row.waypoint_cm
        meta["waypoint_cm"] = None if row_offsets is None else [float(v) for v in row_offsets]
        self.tracker.answer(answers)

        decisions = {
            qid: {
                "probabilities": {c: (1.0 if c == _choice(qid, answers[qid]) else 0.0)
                                  for c in v2_candidates(qid)},
                "choice": _choice(qid, answers[qid]),
                "overridden": qid in overridden,
            }
            for qid in ACTIVE_QIDS
        }
        decisions["meta"] = meta
        return chunk_v2(answers, MEASURED_STEPS, CHUNK_STEPS), decisions

    def _ground(self, privileged: dict, state8) -> dict:
        """Commit `target`/`destination` by the **text rule** the learned policy commits with.

        The same function on the same request: `grounding.grounding_request` renders the scene in
        the camera frame the sentences are written in and `grounding.resolve_request` reads the
        relation words back out of it (610/610 `target`, 515/520 `destination`). There is no
        forward to run here -- this policy has no weights -- so what a baseline run shows is the
        rule's answer and nothing else, which is exactly what `ModelPolicy(ground="rule")` commits.

        `scene_roles` remains the fallback for a sentence the text rule cannot settle: it is the
        geometric rule the expert steers by, and it always answers.
        """
        request = grounding_mod.grounding_request(self.instruction, privileged)
        names = grounding_mod.grounding_names(privileged)
        picked = grounding_mod.resolve_request(request)
        roles, sources, out = None, {}, {}
        for qid in ("target", "destination"):
            sid = picked.get(qid)
            if sid is not None and names.get(sid) in privileged:
                out[qid], sources[qid] = names[sid], "rule"
                continue
            if roles is None:
                roles = _scene_roles(privileged, self.instruction, state8[0:3])
            out[qid], sources[qid] = roles[qid], "scene_roles"
        chosen = set(sources.values())
        source = chosen.pop() if len(chosen) == 1 else "mixed"
        step = self.tracker.decisions * CHUNK_STEPS
        regrounded = self.tracker.target is not None
        self.tracker.commit(out["target"], out["destination"], step=step, source=source)
        return {
            "mode": "rule",
            "source": source,
            "sources": sources,
            "target": out["target"],
            "destination": out["destination"],
            "model": None,
            "model_agrees": None,
            "regrounded": bool(regrounded),
            "decision": self.tracker.decisions,
        }


def _scene_roles(privileged: dict, instruction: str, eef) -> dict:
    """`roles.scene_roles` with this module's error type on the one failure it raises."""
    try:
        return shared_scene_roles(privileged, instruction, eef)
    except ValueError as exc:
        raise PolicyError(str(exc)) from exc


# -------------------------------------------------------------------- the learned / hosted one

def _checkpoint_bytes(checkpoint: pathlib.Path) -> int:
    weights = checkpoint / "best.safetensors"
    return weights.stat().st_size if weights.is_file() else 0


def _cast_to_bf16(engine) -> None:
    """Store the loaded parameters in bf16, in place.

    Cast **after** the load rather than instead of it, deliberately: upstream builds the body from
    its config and supplies every parameter with `load_state_dict(strict=True)`, and loading
    straight into bf16 would mean reaching inside that. The forward already runs under
    `autocast(bfloat16)`, so every matmul was being done in bf16 before this; what changes is the
    *storage*, and with it the rounding of each weight once, at load.
    """
    import torch

    model = getattr(engine, "model", None)
    if model is None:                    # a predictor shape this was not written for: say so
        raise PolicyError(
            "bf16 weight storage was asked for, but this DecisionPredictor has no `.model` to "
            "cast. Upstream's loader changed; re-check predict_toy_decisions.py before serving.")
    model.to(dtype=torch.bfloat16)


class ModelPolicy(Policy):
    """A NanoJev fine-tune, or the hosted model, behind the same two methods.

    Holds the checkpoint directory, the question set's constants read from that checkpoint, the
    instruction `reset` was given, the decision counter that names each state, and -- once `act`
    has been called once -- the loaded predictor. Everything but the predictor is cleared on every
    `reset`; the predictor is not, because reloading 2.4 GB of weights between episodes would be
    the single most expensive thing this does and it is stateless across calls.
    """

    #: The scene is still dropping into place for the first half second, and the harvest that
    #: trained this checkpoint rendered its first state after the same wait.
    wait_steps = 10

    def __init__(self, suite: str, checkpoint: str, revision: str | None = None, seed: int = 7,
                 task_index: int = 0, selection: str = DEFAULT_SELECTION, temperature: float = 1.0,
                 bf16_weights: bool | None = None, questions_version: str | None = None,
                 ground: str = DEFAULT_GROUND, api: str | None = None):
        self.suite = suite
        # **Which engine answers.** Empty is the local fine-tune and every line below is
        # unchanged. `api="typesafe"` is the hosted model, and then `checkpoint` names a *model*
        # rather than a directory.
        self.api = (api or "").strip() or None
        if self.api is not None and self.api != jev_api.PROVIDER:
            raise PolicyError(
                f"unknown decision API {self.api!r}; this speaks {jev_api.PROVIDER!r} "
                f"(robojev.jev_api)")
        self.model = str(checkpoint) if self.api else None
        self.checkpoint = pathlib.Path(checkpoint).expanduser()
        self.checkpoint_repo = (jev_api.repo_word(self.api, self.model) if self.api
                                else str(self.checkpoint))
        self.given_revision = revision or None
        self.seed = int(seed)
        self.task_index = int(task_index)
        mode, t = parse_selection(selection, temperature)
        self.mode, self.temperature = mode, t
        self.selection = selection_text(mode, t)
        if ground not in GROUND_MODES:
            raise PolicyError(f"unknown ground mode {ground!r}; expected one of {list(GROUND_MODES)}")
        self.ground = ground

        meta = self._checkpoint_meta()
        # **The question set comes from the checkpoint, and from nowhere else.** `robojev train`
        # copies it out of the harvest manifest. `questions_version` may only *assert* this, never
        # override it.
        try:
            self.questions_version = registry.check(
                meta.get("questions_version") or registry.UNVERSIONED)
        except ValueError as exc:
            raise PolicyError(f"{self.checkpoint / CHECKPOINT_META}: {exc}") from exc
        if questions_version is not None and questions_version != self.questions_version:
            raise PolicyError(
                f"questions_version {questions_version!r} does not match the checkpoint, which "
                f"says {self.questions_version!r}. The argument asserts the checkpoint's own "
                f"version; it cannot change it, because the weights were trained against one "
                f"question set and only that one can be served.")
        # **The questions asked per decision come from the checkpoint**, by name and never by
        # count. This set defines more questions than it asks -- `yaw` and the shared `step` are
        # ablations -- and which of them a run asks is a property of the *harvest*.
        declared = meta.get("qids")
        known = registry.defined_qids(self.questions_version)
        if declared is None:
            self._qids = registry.qids(self.questions_version)
        else:
            unknown = [qid for qid in declared if qid not in known]
            if unknown:
                raise PolicyError(
                    f"{self.checkpoint / CHECKPOINT_META} was trained on the questions "
                    f"{list(declared)}, and {unknown} are not questions "
                    f"{self.questions_version!r} defines ({list(known)}). Re-harvest and retrain:"
                    f"\n    robojev harvest --suite {self.suite}"
                    f"\n    robojev train --suite {self.suite}")
            self._qids = tuple(declared)
        # The per-candidate-path token budget, passed explicitly because upstream's own default is
        # the checkpoint's `config.json` `max_length` (512), far under what a five-object path
        # measures -- and upstream raises rather than truncating.
        self.max_length = registry.max_path_tokens(self.questions_version)

        if meta.get("delta_t") is None or meta.get("delta_r") is None:
            raise PolicyError(
                f"{self.checkpoint / CHECKPOINT_META} does not give delta_t and delta_r. They are "
                f"the measured step sizes every answer is scaled by -- serving without them would "
                f"compose every action at a size nobody measured, silently and for the whole "
                f"episode.")
        self.delta_t = float(meta["delta_t"])
        self.delta_r = float(meta["delta_r"])

        # The step scale, the tracker and the latch guard. Each is read from the checkpoint and
        # each is refused on a mismatch, for one reason: they are what an answer *means* and what
        # the state text *says*.
        cm_per_unit = meta.get("cm_per_unit")
        if cm_per_unit is None:
            raise PolicyError(
                f"{self.checkpoint / CHECKPOINT_META} does not give cm_per_unit, which is how "
                f"many centimetres one unit of a move answer actually travels. A checkpoint "
                f"trained on 5 cm steps and served at 25 cm executes five times every move it "
                f"meant, for the whole episode and without a word anywhere.")
        self.cm_per_unit = float(cm_per_unit)
        self._steps = calibrate(self.cm_per_unit)
        trained_cm = (meta.get("steps") or {}).get("cm")
        if trained_cm is not None and {k: float(v) for k, v in trained_cm.items()} != {
                k: float(v) for k, v in self._steps.cm.items()}:
            raise PolicyError(
                f"{self.checkpoint / CHECKPOINT_META} was trained with the step sizes "
                f"{trained_cm} and this renders {dict(self._steps.cm)}. Those centimetres are "
                f"printed in every state and are what each answer means.")
        # δ_t is not a second knob: it is `large` expressed in normalised units. A manifest whose
        # two numbers disagree cannot be served at either of them.
        served_delta_t = self._steps.units("large")
        if abs(served_delta_t - self.delta_t) > 1e-9:
            raise PolicyError(
                f"{self.checkpoint / CHECKPOINT_META} says delta_t {self.delta_t!r} and "
                f"cm_per_unit {self.cm_per_unit!r}, which is a large step of {served_delta_t!r} "
                f"units. The two are one number; re-harvest so the manifest agrees with itself.")
        self._tracker = self._new_tracker(self._qids)
        self.memory_settings = self._tracker.settings()
        trained_tracker = meta.get("tracker")
        if trained_tracker is not None:
            differs = {k: (v, self.memory_settings.get(k)) for k, v in trained_tracker.items()
                       if k in self.memory_settings and v != self.memory_settings[k]}
            if differs:
                detail = "; ".join(f"{k}: trained {was!r}, serving {now!r}"
                                   for k, (was, now) in sorted(differs.items()))
                raise PolicyError(
                    f"{self.checkpoint / CHECKPOINT_META} was trained with a different tracker "
                    f"than this renders ({detail}). Every one of those settings changes a byte of "
                    f"the state text the model reads.")
        # The latch guard, always on. It is not a hand-tuned threshold on a probability: it
        # applies a change of the fingers only in a sub-goal whose own plan makes that change. An
        # unguarded latch takes the expert from 19/20 to 7/20 at 10 % answer corruption.
        self._latch = self._tracker.latch
        self.rho = None

        self.checkpoint_bytes = _checkpoint_bytes(self.checkpoint)
        self.bf16_weights = (self.checkpoint_bytes > BF16_WEIGHTS_BYTES
                             if bf16_weights is None else bool(bf16_weights))

        self.instruction = ""
        self._step = 0
        self._rng = np.random.default_rng(self.seed)
        # The hosted engine is built **at launch**: all of it is a key and an endpoint, and a run
        # without `$JEV_API_KEY` has to fail here rather than thirty seconds into an episode. The
        # local one stays lazy for the opposite reason: 2.4 GB of weights and a GPU.
        self._engine = jev_api.engine(self.model) if self.api else None
        self._revision: str | None = None
        self._grounding: dict | None = None

    # -- the checkpoint's own facts -------------------------------------------------------

    def _checkpoint_meta(self) -> dict:
        """The provenance file beside the weights, or `{}` when the checkpoint has none.

        Missing is not an error: any NanoJev-format checkpoint directory can be served, it just
        cannot say what step size its rows were labelled at (and then the guards above refuse it,
        which is the point).
        """
        if self.api:
            # A hosted model has no directory and therefore no manifest. What it must be served at
            # is not its own fact but the harness's, and `jev_api.checkpoint_meta` states them in
            # exactly this shape, so every guard runs unchanged for both engines.
            return jev_api.checkpoint_meta()
        for name in (CHECKPOINT_META, LEGACY_CHECKPOINT_META):
            path = self.checkpoint / name
            if path.is_file():
                meta = json.loads(path.read_text(encoding="utf-8"))
                return meta if isinstance(meta, dict) else {}
        return {}

    def revision(self) -> str:
        """`sha256(best.safetensors)[:12]`, or the revision the caller was told to expect.

        A published checkpoint is identified by its repo revision; a locally trained one has no
        such thing, so the weights hash their own name.
        """
        if self.given_revision:
            return self.given_revision
        if self.api:
            # The hosted model's identity is the **version that answers**, not the alias that was
            # asked for: an alias moves, and two runs a month apart under one name would otherwise
            # claim to be the same model. Only a response carries it, so `describe()` buys one --
            # a five-word state and one yes/no question -- which is also the proof the key works.
            if self._revision is None:
                self._revision = self._ensure_loaded().probe()
            return self._revision
        if self._revision is None:
            weights = self.checkpoint / "best.safetensors"
            if not weights.is_file():
                raise PolicyError(
                    f"{self.checkpoint} has no best.safetensors, so there is no checkpoint to "
                    f"identify. Train one with:\n"
                    f"    robojev harvest --suite {self.suite}\n"
                    f"    robojev train --suite {self.suite}")
            h = hashlib.sha256()
            with weights.open("rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    h.update(block)
            self._revision = h.hexdigest()[:12]
        return self._revision

    # -- the model -------------------------------------------------------------------------

    def _ensure_loaded(self):
        """Load NanoJev and the weights, once. The only place anything heavy happens."""
        if self._engine is not None:
            return self._engine

        scripts = upstream_dir()
        if not scripts.is_dir():
            pin = runtime.nanojev_pin(runtime.trainer())
            raise PolicyError(
                f"NanoJev's checkout is not at {scripts}. It is pinned by "
                f"{runtime.NANOJEV_PIN} in {runtime.TRAINER_DIR} and cloned by "
                f"`robojev train --build-env`; or clone {pin[0]} @ {pin[1][:12]} into "
                f"{scripts.parent} by hand.")
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))

        from predict_toy_decisions import DecisionPredictor

        self._engine = DecisionPredictor(
            str(self.checkpoint), max_length=self.max_length, precision=PRECISION)
        if self.bf16_weights:
            _cast_to_bf16(self._engine)
        return self._engine

    def close(self) -> None:
        self._engine = None

    # -- the question set's own shape ------------------------------------------------------

    def candidates(self, qid: str):
        """One question's candidate ids, in the order its block declares them."""
        return v2_candidates(qid)

    def _questions_block(self) -> dict:
        """The block this policy asks, for *its* qids -- never today's default."""
        return v2_questions_block(self._qids)

    # -- the protocol ----------------------------------------------------------------------

    def describe(self) -> dict:
        """What this policy is, what it asks, and under what protocol -- loading nothing.

        Deliberately light: requiring 2.4 GB of weights on a shared GPU to answer "what questions
        do you ask?" would make a preflight fail for a reason that has nothing to do with the
        environment. Under `api` there is one exception and it is the same bargain rather than a
        different one: `revision()` buys the version that answers.
        """
        questions = _questions_for_describe(self._questions_block())
        notes = [
            f"every answer comes from {jev_api.API_URL} over HTTPS: the serialised state leaves "
            f"this box, the run is reproducible only as far as a hosted model is, and the "
            f"recorded revision is the version that actually answered rather than the alias "
            f"{self.model!r} that was asked for",
            "a request that is rate-limited, times out or fails with a 5xx is retried with "
            f"backoff up to {jev_api.DEFAULT_ATTEMPTS} attempts; a decision that still fails "
            "fails the episode rather than being composed from a partial answer",
        ] if self.api else [
            "CUDA is hard-required: DecisionPredictor raises on a non-CUDA device and again on a "
            "device without bf16 support, so there is no CPU path and no fallback",
            f"bf16 forward autocast over fp32 parameters ({PRECISION}): bit-identical only on the "
            "same GPU model",
            "TF32 is disabled by upstream at load",
        ]
        if self.mode == "sample":
            notes.append(
                f"selection {self.selection}: each group is drawn from its (tempered) "
                f"distribution with a generator seeded from the run seed ({self.seed})")
        return {
            "policy": "jev" if self.api else "robojev",
            "family": self.family,
            "checkpoint": {"repo": self.checkpoint_repo, "revision": self.revision()},
            "questions": questions,
            "questions_version": self.questions_version,
            "max_path_tokens": self.max_length,
            "deltas": {"translate": self.delta_t, "rotate": self.delta_r},
            "rho": self.rho,
            "memory": self.memory_settings,
            "bf16_weights": self.bf16_weights,
            "checkpoint_bytes": self.checkpoint_bytes,
            "protocol": {
                "wait_steps": self.wait_steps,
                "max_steps": {self.suite: episode_mod.UPSTREAM_MAX_STEPS[self.suite]},
                "seed": self.seed, "env_seed": 0,
                "chunk_size": CHUNK_STEPS, "execute_steps": CHUNK_STEPS,
                "action_dim": ACTION_DIM,
                "family": self.family,
                "privileged_state": True,
                "selection": self.selection,
                "questions": questions,
                "task_index": self.task_index,
                "questions_version": self.questions_version,
                "max_path_tokens": self.max_length,
                "bf16_weights": self.bf16_weights,
                "cm_per_unit": self.cm_per_unit,
                "step_sizes_cm": dict(self._steps.cm),
                # True for a size that asks for more travel than the controller will take. At the
                # measured scale nothing saturates, and this is what says so if that changes.
                "step_saturates": {size: saturates(size, self._steps) for size in self._steps.cm},
                "delta_t": self.delta_t,
                "delta_r": self.delta_r,
                "rho": None,
                "memory": self.memory_settings,
                "grip_latch": self._latch.config(),
                "ground": {
                    "mode": self.ground,
                    "questions": list(grounding_mod.SERVING_QIDS),
                    "min_probability": GROUNDING_MIN_P,
                    "resolver": "grounding.resolve_request (610/610, 515/520)",
                    "order": (["rule", "model", "scene_roles"] if self.ground == "rule"
                              else ["model", "scene_roles"]),
                    "regrounds_after_a_failed_attempt": True,
                    "frame": "camera",
                },
            },
            "versions": ({
                "policy_repo": self.checkpoint_repo,
                "api_endpoint": jev_api.API_URL,
                "numpy": np.__version__,
            } if self.api else {
                "policy_repo": _pin()["commit"],
                "torch": _dist_version("torch"),
                "transformers": _dist_version("transformers"),
                "safetensors": _dist_version("safetensors"),
                "numpy": np.__version__,
            }),
            "nondeterminism": notes,
        }

    def reset(self, instruction: str) -> None:
        self.instruction = instruction
        self._step = 0
        # Re-seeded per episode, so a `sample@T` run replays from its seed alone.
        self._rng = np.random.default_rng(self.seed)
        self._reset_tracker(self._tracker, instruction)
        self._grounding = None

    # -- the decision ----------------------------------------------------------------------

    def act(self, obs: dict, overrides: dict | None = None, selection: str | None = None):
        """One decision: at most two forwards, one held action.

        The order is the whole of the method and none of it is negotiable:

        1. **Ground**, on the first decision and again after a failed attempt (and nowhere else:
           `TrackerV2.needs_regrounding` is the tracker's own fact, no model in the loop). It is
           its **own request**, rendered in the camera frame the instructions are written in --
           never concatenated with the motor state, which is gripper-relative and mirrored left
           for right, so "the bowl on the left" read against it would resolve the wrong bowl.
        2. **`observe`**, which closes out the previous decision.
        3. **One motion forward** over `self._qids` -- every candidate path of every question in
           one padded tensor, which is the property the whole design rests on.
        4. **The latch guard** over the model's `grip` answer, then `chunk`, then `answer` with
           the **executed** answers, so an override or a refused latch is what the next history
           block has to explain.
        """
        privileged = self._privileged(obs)
        mode, temperature = (
            (self.mode, self.temperature) if selection is None else parse_selection(selection))
        # An override naming a question this policy does not ask is refused as loudly as one
        # naming a candidate it does not have: dropped silently, the operator would watch the
        # model's own choice execute with nothing anywhere saying the override went nowhere.
        unknown = sorted(set(overrides or {}) - set(self._qids))
        if unknown:
            raise PolicyError(
                f"override for {unknown}: not one of this policy's questions {list(self._qids)}")

        engine = self._ensure_loaded()
        state8 = np.asarray(obs["state"], dtype=np.float64).reshape(-1)
        state_id = f"{self.suite}:{self.task_index}:{self._step}"
        tracker = self._tracker

        decisions: dict[str, dict] = {}
        grounding = None
        if self._step == 0 or tracker.needs_regrounding():
            grounding, groups = self._ground(engine, privileged, state8[0:3], state_id,
                                             mode, temperature)
            decisions.update(groups)

        tracker.observe(step=self._step * CHUNK_STEPS, proprio=state8, objects=privileged)
        payload = v2_request(
            serialise_v2(state8, privileged, self.instruction, tracker, annotate=tracker.annotate),
            state_id=state_id, qids=self._qids,
        )
        out = engine.predict(payload, batch_questions=BATCH_QUESTIONS, temperature=temperature)
        answers = out["states"][0]["answers"]

        probabilities = {qid: _ordered(qid, answers[qid]["probabilities"], self.candidates(qid))
                         for qid in self._qids}
        choices: dict[str, str] = {}
        overridden: list[str] = []
        for qid in self._qids:
            forced = (overrides or {}).get(qid)
            if forced is not None:
                choices[qid] = _override(qid, forced, self.candidates(qid))
                overridden.append(qid)
                continue
            choices[qid] = select(probabilities[qid], mode, self._rng)

        # The guard: a change of the fingers applies only in a sub-goal whose own plan makes it --
        # shut in `grasp`, open while reaching or over the destination, never during a lift or a
        # carry. An operator's override outranks it.
        asked = choices.get("grip") == "true"
        row = tracker.last_row
        offset = row.waypoint_cm
        arrived = bool(offset is not None
                       and all(abs(float(v)) < tracker.arrival_cm for v in offset))
        latched = self._latch.update(asked, subgoal=tracker.subgoal, arrived=arrived,
                                     log=tracker.log_latch)
        refused = bool("grip" not in overridden and latched != asked)
        if "grip" in overridden:
            # The latch still follows the fingers, or the next decision's guard would compare the
            # model's answer against a state of the hand that is not the one the arm is in.
            self._latch.closed = asked
        else:
            choices["grip"] = "true" if latched else "false"

        decisions.update({
            qid: {
                "probabilities": probabilities[qid],
                "choice": choices[qid],
                "overridden": qid in overridden,
            }
            for qid in self._qids
        })
        execution = out.get("execution", {})
        waypoint = tracker.waypoint
        decisions["meta"] = {
            "state_id": state_id,
            "state": payload["states"][0]["state"],
            "step": self._step,
            "questions_version": self.questions_version,
            "selection": selection_text(mode, temperature),
            "overridden": overridden,
            "delta_t": self.delta_t,
            "delta_r": self.delta_r,
            "cm_per_unit": self.cm_per_unit,
            "step_sizes_cm": dict(self._steps.cm),
            "forward_passes": execution.get("forward_passes"),
            "candidate_paths": execution.get("candidate_paths"),
            "questions": execution.get("questions"),
            "subgoal": tracker.subgoal,
            "substage": tracker.substage,
            "target": tracker.target,
            "target_rule": (self._grounding or {}).get("source"),
            "destination": tracker.destination,
            "waypoint": None if waypoint is None else waypoint.label,
            "waypoint_cm": None if offset is None else [float(v) for v in offset],
            "grounding": grounding,
            "grip_latch": {
                "closed": bool(latched),
                "asked": bool(asked),
                # True only when the guard changed the executed answer.
                "refused": refused,
                **self._latch.config(),
            },
            # What the hosted engine's call cost, and which version answered it. `None` under the
            # local engine. The key is not in it -- it is not in anything `jev_api` returns.
            "api": out.get("api"),
        }
        tracker.answer(choices)
        self._step += 1
        return chunk_v2(choices, self._steps, CHUNK_STEPS), decisions

    def _ground(self, engine, privileged: dict, eef, state_id: str, mode: str, temperature: float):
        """The once-per-episode grounding forward, and what it commits.

        Returns `(meta.grounding, {"target": group, "destination": group})`. The groups are the
        ordinary `decisions` shape, so **the model's own opinion is always shown** whichever mode
        decides what is committed.

        The chain per question, in order: the text rule; then the model when it is at least
        `GROUNDING_MIN_P` sure; then `roles.scene_roles`, the geometric rule the harvest and the
        baseline name a target with. `meta.grounding.sources` says which one answered.
        """
        request = grounding_mod.grounding_request(
            self.instruction, privileged, state_id=f"{state_id}:ground")
        out = engine.predict(request, batch_questions=BATCH_QUESTIONS, temperature=temperature)
        answers = out["states"][0]["answers"]
        names = grounding_mod.grounding_names(privileged)
        asked = request["states"][0]["questions"]

        groups, model, confidence = {}, {}, {}
        for qid in asked:
            want = list(asked[qid]["criteria"])
            probabilities = _ordered(qid, answers[qid]["probabilities"], want)
            sid = select(probabilities, mode, self._rng)
            groups[qid] = {"probabilities": probabilities, "choice": sid, "overridden": False}
            model[qid] = sid
            confidence[qid] = float(probabilities[sid])

        # The text rule, read off the very request the model was given -- no privileged dict, no
        # task definition, nothing the model could not see. `None` for a question it cannot settle.
        rule = (grounding_mod.resolve_request(request) if self.ground == "rule"
                else dict.fromkeys(asked))
        roles = None
        picked: dict[str, str | None] = {}
        sources: dict[str, str] = {}
        sure = min(confidence.values(), default=0.0) >= GROUNDING_MIN_P
        for qid in asked:
            if rule.get(qid) is not None:
                picked[qid], sources[qid] = names.get(rule[qid]), "rule"
            elif sure:
                picked[qid], sources[qid] = names.get(model[qid]), "model"
            else:
                if roles is None:
                    roles = _scene_roles(privileged, self.instruction,
                                         np.asarray(eef, dtype=np.float64))
                picked[qid], sources[qid] = roles.get(qid), "scene_roles"

        chosen = set(sources.values())
        source = chosen.pop() if len(chosen) == 1 else "mixed"
        tracker = self._tracker
        regrounded = self._step > 0
        tracker.commit(picked.get("target"), picked.get("destination"),
                       step=self._step * CHUNK_STEPS, source=source)
        self._grounding = {
            "mode": self.ground,
            "source": source,
            "sources": sources,
            "state_id": request["states"][0]["id"],
            "target": picked.get("target"),
            "destination": picked.get("destination"),
            # What the *model* said, always, beside what was committed.
            "model": {qid: names.get(model[qid]) for qid in model},
            "model_agrees": all(names.get(model[qid]) == picked.get(qid) for qid in model),
            "candidates": dict(model),
            "rule": {qid: names.get(sid) if sid else None for qid, sid in rule.items()},
            "probability": confidence,
            "min_probability": GROUNDING_MIN_P,
            # True when this is the *second* grounding of the episode: the tracker recorded a
            # grasp that never lifted anything, which on a scene with two identical bowls is the
            # evidence that the wrong one was named.
            "regrounded": regrounded,
            "decision": self._step,
            "forward_passes": (out.get("execution") or {}).get("forward_passes"),
        }
        return dict(self._grounding), groups


# ------------------------------------------------------------------------- the pinned upstream

def _pin() -> dict:
    url, commit, subdir = runtime.nanojev_pin(runtime.trainer())
    return {"url": url, "commit": commit, "subdir": subdir}


def clone_root() -> pathlib.Path:
    """`$ROBOJEV_HOME/src/nanojev@<commit12>` -- where the pinned clone lives.

    Resolved per call rather than at import, because the root is an environment variable and a
    module-level constant would freeze whatever it happened to be at first import.
    """
    from robojev import home

    return home.src_dir() / f"nanojev@{_pin()['commit'][:12]}"


def upstream_dir() -> pathlib.Path:
    """The directory to put on `sys.path`: the clone's `scripts/`."""
    return clone_root() / _pin()["subdir"]


# -------------------------------------------------------------------------------- the factory

def default_checkpoint(suite: str, policy: str = "robojev") -> pathlib.Path:
    """`$ROBOJEV_HOME/checkpoints/<policy>/<suite>` -- what `robojev train` produced."""
    return runtime.local_checkpoint_dir(policy, suite)


def build(engine: str = "expert", suite: str = "libero_spatial", *, checkpoint: str | None = None,
          seed: int = 7, task_index: int = 0, selection: str = DEFAULT_SELECTION,
          temperature: float = 1.0, ground: str = DEFAULT_GROUND,
          revision: str | None = None, bf16_weights: bool | None = None) -> Policy:
    """One policy by name: `expert`, `model` (a local checkpoint) or `jev` (the hosted model)."""
    if engine == "expert":
        return ExpertPolicy(suite, seed=seed, selection=selection, revision=revision)
    if engine == "model":
        directory = pathlib.Path(checkpoint) if checkpoint else default_checkpoint(suite)
        return ModelPolicy(suite, str(directory), revision=revision, seed=seed,
                           task_index=task_index, selection=selection, temperature=temperature,
                           bf16_weights=bf16_weights, ground=ground)
    if engine == "jev":
        return ModelPolicy(suite, checkpoint or jev_api.DEFAULT_MODEL, revision=revision,
                           seed=seed, task_index=task_index, selection=selection,
                           temperature=temperature, ground=ground, api=jev_api.PROVIDER)
    raise PolicyError(f"unknown engine {engine!r}; expected one of {list(ENGINES)}")


__all__ = ["ACTION_DIM", "BF16_WEIGHTS_BYTES", "CHECKPOINT_META", "CHUNK_STEPS", "DEFAULT_GROUND",
           "DEFAULT_SELECTION", "ENGINES", "EXPERT_REVISION", "GROUNDING_MIN_P", "GROUND_MODES",
           "ExpertPolicy", "ModelPolicy", "Policy", "PolicyError", "build", "clone_root",
           "default_checkpoint", "parse_selection", "selection_text", "upstream_dir"]
