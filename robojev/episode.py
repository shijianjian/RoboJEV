"""One closed-loop episode: the loop, the protocol it runs under, and what it records.

The loop is the published LIBERO evaluation's, kept deliberately literal so a number measured
here is comparable with a number measured there: reset, set the init state, `wait_steps` dummy
steps while the scene settles, then policy steps until the task's own success predicate fires or
the suite's cap is reached. Success is `done` from `env.step`, and the first `done` ends the
episode.

A decision policy is queried once every `execute_steps` control steps and the chunk it returns is
executed whole -- there is no sliding window. Every frame of an executed chunk carries the same
`decisions` object, not only the frame that asked, so a replay bundle has something to show at
every step and `policy_query` is the flag that says which frame did the asking.

Nothing here is simulator-specific: it drives a `robojev.envs.DecisionEnv`.
"""
from __future__ import annotations

import dataclasses
from collections import deque
from typing import Any, Callable

import numpy as np

#: The control rate of the LIBERO tasks this package is measured on, and the clock a replay
#: bundle's video is written at: video time and control step are the same clock.
CONTROL_FPS: float = 20.0

#: The episode cap every published result on these suites is measured under, per suite. A run
#: that used a different one is not comparable with any of them.
UPSTREAM_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclasses.dataclass(frozen=True)
class Protocol:
    """How one episode is run. A policy declares it; the loop obeys it.

    `wait_steps` is the settling period: a scene is still dropping into place for the first half
    second, and a vision policy shown that would be shown a scene that is about to change. A
    policy reading the simulator's own object poses does not need it, which is why the scripted
    expert declares zero and the learned one declares ten -- the number is the policy's, and a
    record that did not carry it could not be read back.
    """

    max_steps: int
    wait_steps: int = 10
    execute_steps: int = 5
    chunk_size: int = 5
    render_size: int = 256
    env_seed: int = 0
    privileged_state: bool = True
    family: str = "decision"


def for_suite(suite: str, max_steps: int | None = None, **kw) -> Protocol:
    """The protocol a suite is measured under, with an optional cap override.

    An override is allowed because a test wants thirty steps, not two hundred and twenty -- but a
    run that uses one is not comparable with a published number, and the cap is in the record so
    that is visible rather than assumed.
    """
    cap = UPSTREAM_MAX_STEPS.get(suite, max(UPSTREAM_MAX_STEPS.values()))
    return Protocol(max_steps=int(cap if max_steps is None else max_steps), **kw)


@dataclasses.dataclass
class Frame:
    """One control step, as the record holds it."""

    timestamp: float
    frame_index: int
    state: np.ndarray
    action: np.ndarray
    next_success: bool
    is_wait_step: bool
    is_terminal: bool
    policy_query: bool
    images: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    #: The decision behind this frame's action, straight from the `act` reply. `None` on a wait
    #: step and on the terminal frame, neither of which executes a policy action.
    decisions: dict | None = None
    #: The simulator's `qpos` at this frame, when the frame was recorded with images: what the
    #: web viewer poses its 3D scene with, one row per video frame.
    qpos: np.ndarray | None = None


@dataclasses.dataclass
class EpisodeResult:
    frames: list[Frame]
    success: bool
    steps: int
    terminated_by: str
    error: str | None = None
    #: The frame index of the first step whose post-step `done` was true, or None.
    first_success_step: int | None = None


def _frame(env, obs: dict, t: int, action: np.ndarray, *, images: bool, **kw) -> Frame:
    return Frame(
        timestamp=t / float(getattr(env, "control_freq", CONTROL_FPS)),
        frame_index=t,
        state=np.asarray(env.state_vector(obs)),
        action=action,
        images=env.images(obs) if images and hasattr(env, "images") else {},
        qpos=env_qpos(env) if images else None,
        **kw,
    )


def env_qpos(env) -> np.ndarray | None:
    """The simulator's joint positions right now, or None for an environment that has none."""
    if not hasattr(env, "ground_truth"):
        return None
    return np.asarray(env.ground_truth()[0], dtype=np.float32).copy()


def _policy_obs(env, obs: dict, privileged: bool) -> dict:
    """What one `act` call sees: the simulator's observation plus the two keys a decision policy
    actually reads.

    `state` is the 8-d proprio vector, put here rather than derived inside the policy so that
    every engine -- and a hand-built test state -- reads the same key, and so that the adapter
    stays the only thing that knows how a particular simulator spells a gripper joint.
    `privileged` is the scene, and only a policy that asked for it gets it.

    A *new* dict, always: the observation written to the record and the payload handed to the
    policy cannot diverge if neither is mutated.
    """
    out = {**obs, "state": env.state_vector(obs)}
    if not privileged:
        return out
    extra = env.privileged(obs)
    return out if extra is None else {**out, "privileged": extra}


def run_episode(env, policy, protocol: Protocol, init_state_index: int = 0, seed: int = 7,
                execute_steps: int | None = None, *, images: bool = False,
                on_step: Callable[[int], None] | None = None) -> EpisodeResult:
    """Drive one episode of `env` with `policy`, and return every frame of it.

    `policy` is anything with `reset(instruction)` and `act(obs) -> (chunk, decisions)`;
    `robojev.policy.Policy` is the one this package ships, and a test's fake is a dozen lines.
    """
    try:
        from robojev.envs.libero import set_seed_everywhere
        set_seed_everywhere(seed)
    except ImportError:                                          # pragma: no cover
        pass

    obs = env.reset(init_state_index)
    policy.reset(env.instruction)

    execute = int(protocol.execute_steps if execute_steps is None else execute_steps)
    privileged = bool(protocol.privileged_state)
    frames: list[Frame] = []
    success = False
    first_success_step: int | None = None
    terminated_by = "max_steps"
    error: str | None = None
    errored = False
    steps = 0
    t = 0
    queue: deque = deque()
    decisions: dict | None = None
    try:
        while t < protocol.max_steps + protocol.wait_steps:
            is_wait = t < protocol.wait_steps
            if is_wait:
                action = np.asarray(env.dummy_action(), dtype=np.float32)
                policy_query = False
                decisions = None
            else:
                if not queue:
                    chunk, decisions = policy.act(_policy_obs(env, obs, privileged))
                    chunk = np.asarray(chunk, dtype=np.float32)
                    if chunk.shape[0] < execute:
                        raise ValueError(
                            f"policy returned {chunk.shape[0]} actions, fewer than "
                            f"execute_steps={execute}")
                    queue.extend(row.copy() for row in chunk[:execute])
                    policy_query = True
                else:
                    policy_query = False
                action = queue.popleft()
            frame = _frame(env, obs, t, action, images=images, is_wait_step=is_wait,
                           next_success=False, is_terminal=False, policy_query=policy_query,
                           decisions=decisions)
            result = env.step(action.tolist())
            frame.next_success = bool(result.done)
            frames.append(frame)
            if on_step is not None:
                on_step(t)
            obs = result.obs
            if result.done and first_success_step is None:
                first_success_step = t
            if not is_wait:
                steps += 1
            if result.done and getattr(env, "stop_on_success", True):
                success = True
                terminated_by = "success"
                break
            t += 1
    except Exception as exc:                                     # the record says what stopped it
        errored = True
        error = f"{type(exc).__name__}: {exc}"
        terminated_by = "error"
    if not errored and frames:
        # One terminal frame for the final state, so the record has a row for where the arm ended
        # up. Nothing new is stepped; its action is all-NaN because there is no action for it.
        frames.append(_frame(env, obs, t + 1, np.full(len(frames[-1].action), np.nan,
                                                      dtype=np.float32),
                             images=images, is_wait_step=False,
                             next_success=frames[-1].next_success, is_terminal=True,
                             policy_query=False, decisions=None))
    return EpisodeResult(frames=frames, success=success, steps=steps,
                         terminated_by=terminated_by, error=error,
                         first_success_step=first_success_step)


def decision_count(episode: EpisodeResult) -> int:
    """How many times the policy was asked."""
    return sum(1 for f in episode.frames if f.policy_query)


__all__ = ["CONTROL_FPS", "UPSTREAM_MAX_STEPS", "EpisodeResult", "Frame", "Protocol",
           "decision_count", "env_qpos", "for_suite", "run_episode"]
