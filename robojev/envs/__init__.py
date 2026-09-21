"""The environment this package acts in, as a protocol -- and nothing about any one simulator.

Everything that rolls the expert or a policy out -- the harvest (`v2.rollout`), the DAgger round
(`dagger`), the scene-cache builder (`v2.grounding`) and `expert.run_episode` -- reads a scene and
steps an arm. None of them needs to know *whose* simulator it is, and until this module existed
they all reached for one particular wrapper by name, which is the dependency that made this
package unshippable on its own: a model that cannot be installed without an evaluation platform
is not a model, it is half of a platform.

So the rule is: **a caller hands in the environment, or a factory for one.** `DecisionEnv` below
is what it has to look like, written as a `Protocol` so an adapter satisfies it by shape rather
than by inheritance -- an existing wrapper is already one, and a test's fake is one in thirty
lines with no import at all.

The members are exactly what this package calls and no more:

* `instruction` -- the sentence the task is, which the state text prints and the role rule reads;
* `num_init_states` -- how many starting scenes the task has;
* `reset(i)` / `step(action)` -- the loop, with `step` returning something carrying `obs` and
  `done` (`StepResult`); `done` is the simulator's **own** success predicate, never a rule here;
* `state_vector(obs)` -- the 8-d proprio vector (grip-site position, the end effector's
  axis-angle, the two finger joints), taken as-is and never re-derived;
* `privileged(obs)` -- `{name: {"pos", "quat"}}` for the movable objects **plus** the fixtures,
  each of those carrying `fixture: True` and its oriented collision boxes (`decision.scene`);
* `dummy_action()` -- the no-op the settling steps are run with (fingers open, no motion);
* `obj_of_interest` -- the task definition's own objects, which a *label source* may read and a
  server may not;
* `close()` -- one simulator alive at a time is what keeps a ten-task harvest inside one box.

`object_names` and `mujoco_sim()` are the two optional extras, wanted only by the offline
scene-cache builder, which measures anchor geometry the poses do not carry. Nothing in the serving
or harvesting path touches them.

**The default binding.** A caller that supplies no factory gets `DEFAULT_ADAPTER`, resolved by
name at call time and overridable with `$ROBOJEV_ENV`. It is a string and not an import,
so this module stays free of the platform at module scope, a subprocess worker that was handed
only JSON can still build an environment, and moving to another host is one constant.
"""
from __future__ import annotations

import dataclasses
import importlib
import os
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StepOutcome(Protocol):
    """What `step` returns: the observation, and whether the task's own predicate has fired."""

    obs: dict
    done: bool


@dataclasses.dataclass
class StepResult:
    """The plain implementation of `StepOutcome`, for an adapter that has nothing else to carry."""

    obs: dict
    reward: float = 0.0
    done: bool = False
    info: dict = dataclasses.field(default_factory=dict)


@runtime_checkable
class DecisionEnv(Protocol):
    """One task of one suite, ready to be stepped. See the module docstring for each member."""

    #: The sentence this task is.
    instruction: str

    @property
    def num_init_states(self) -> int:
        """How many starting scenes the task has."""

    def reset(self, init_state_index: int) -> dict:
        """Put the scene at `init_state_index` and return the first observation."""

    def step(self, action) -> StepOutcome:
        """One control step."""

    def state_vector(self, obs: dict) -> Any:
        """The 8-d proprio vector: `eef_pos(3) + axis-angle(3) + gripper_qpos(2)`."""

    def privileged(self, obs: dict) -> dict:
        """`{name: {"pos", "quat", …}}` -- the movable scene and its fixtures."""

    def dummy_action(self) -> Any:
        """The no-op action the settling steps run: no motion, fingers open."""

    def close(self) -> None:
        """Release the simulator."""


#: What `make()` builds when nobody says otherwise, as `module:attribute`. It names the wrapper
#: this model is measured against; `$ROBOJEV_ENV` replaces it, and every entry point in
#: this package takes a factory argument that replaces it more directly still.
DEFAULT_ADAPTER: str = "robojev.envs.libero:LiberoEnv"

#: The variable that overrides it, read per call so a test's `monkeypatch.setenv` wins.
ADAPTER_ENV: str = "ROBOJEV_ENV"


def adapter_name() -> str:
    return os.environ.get(ADAPTER_ENV) or DEFAULT_ADAPTER


def adapter():
    """The configured adapter class, imported now rather than at module import.

    Now and not earlier for two reasons: importing a simulator wrapper costs seconds and pulls in
    MuJoCo, which a `robojev --help` (or the trainer, or a unit test) must not pay for; and this
    package stays importable on a box that has no simulator at all, which is what makes its unit
    tests run in the CPU suite.
    """
    name = adapter_name()
    module, _, attribute = name.partition(":")
    if not attribute:
        raise ValueError(
            f"{ADAPTER_ENV}={name!r} is not a `module:attribute` adapter name (for example "
            f"{DEFAULT_ADAPTER!r})")
    try:
        return getattr(importlib.import_module(module), attribute)
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            f"the decision environment adapter {name!r} could not be loaded ({exc}). Pass an "
            f"`env_factory` to the call that needs one, or set ${ADAPTER_ENV} to a "
            f"`module:attribute` naming a DecisionEnv implementation."
        ) from exc


def make(suite: str, task_index: int, **kwargs) -> DecisionEnv:
    """One environment from the configured adapter. The last resort, not the usual path: every
    entry point in this package takes a factory, and a caller that has one should pass it."""
    return adapter()(suite, task_index, **kwargs)


def factory_or_default(env_factory):
    """`env_factory`, or one that builds the configured adapter with the same signature.

    The signature is `(suite, task_index, **kwargs) -> DecisionEnv`, which is what every caller in
    this package passes around, so a host installs its wrapper in one place and the rest of the
    package never mentions it.
    """
    return env_factory if env_factory is not None else make


__all__ = ["ADAPTER_ENV", "DEFAULT_ADAPTER", "DecisionEnv", "StepOutcome", "StepResult",
           "adapter", "adapter_name", "factory_or_default", "make"]
