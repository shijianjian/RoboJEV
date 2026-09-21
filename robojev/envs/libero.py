"""One LIBERO task, wearing `robojev.envs.DecisionEnv`.

Task definitions, success checks and environment construction all come from LIBERO itself. This
file adds only what a decision needs on top: the 8-d proprio vector, the privileged scene (every
movable object's pose **and** every fixture's oriented collision boxes), and the two renders a
replay bundle is made of.

It is the only module in the package that imports a simulator, and it is imported by name rather
than at package import (`robojev.envs.DEFAULT_ADAPTER`), so `import robojev` costs nothing on a
box with no MuJoCo on it.

Install the extra to use it::

    pip install 'robojev[libero]'

and then LIBERO's own assets, which its installer downloads on first use.
"""
from __future__ import annotations

import math

import numpy as np

from robojev.envs import StepResult

#: LIBERO's `ControlEnv` default control rate, and the clock a replay bundle's video is written at.
CONTROL_FPS: float = 20.0

#: The episode cap the published LIBERO evaluations use, per suite. It is not LIBERO's own number
#: -- LIBERO has no cap -- but the one every reported result on these suites is measured under, so
#: a run that used a different one is not comparable with any of them.
UPSTREAM_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

#: The no-op the settling steps run: no pose delta, fingers open. LIBERO's OSC_POSE controller
#: takes six pose deltas and one gripper command, and -1 is "open".
DUMMY_ACTION: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)


def quat2axisangle(quat) -> np.ndarray:
    """An xyzw quaternion as axis-angle exponential coordinates.

    robosuite's own `transform_utils.quat2axisangle` where robosuite is installed -- the same
    function LIBERO's observables are built with, so the proprio vector this package reads is the
    one every other consumer of the same simulator reads.

    The four lines below are only the fallback for a box that has LIBERO without robosuite, which
    is not a supported configuration but is a cheap thing to survive. **They are the same four
    lines as upstream's**, because there is one way to write this conversion:
    https://github.com/ARISE-Initiative/robosuite/blob/eafb81f/robosuite/utils/transform_utils.py
    (robosuite, MIT).
    """
    try:
        from robosuite.utils.transform_utils import quat2axisangle as _upstream
    except ImportError:
        pass
    else:
        return np.asarray(_upstream(np.asarray(quat, dtype=np.float64)), dtype=np.float64)
    q = np.asarray(quat, dtype=np.float64).copy()
    q[3] = min(max(float(q[3]), -1.0), 1.0)
    den = math.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def set_seed_everywhere(seed: int) -> None:
    """Seed the generators an episode can reach. torch only if it is already imported."""
    import random
    import sys

    random.seed(seed)
    np.random.seed(seed)
    torch = sys.modules.get("torch")
    if torch is not None:                                       # pragma: no cover - GPU only
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


class LiberoEnv:
    """One task of one LIBERO suite, ready to be stepped."""

    suite: str
    task_index: int
    instruction: str
    action_dim: int = 7          # 6 pose deltas + gripper, per LIBERO's OSC_POSE controller
    # The class attribute is the declaration; `__init__` gives each instance its own copy, so this
    # shared list can never be mutated through one env and seen by another.
    camera_names: list[str] = ["agentview", "wrist"]
    # The task's movable objects, in the BDDL's `(:objects …)` order. Per-instance like
    # `camera_names` -- the class attribute is only the declaration.
    object_names: list[str] = []
    control_freq: float = CONTROL_FPS
    # The published loops break out of the step loop on `done` and count the episode a success.
    stop_on_success: bool = True

    def __init__(self, suite: str, task_index: int, render_size: int = 256, env_seed: int = 0):
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        self.camera_names = list(LiberoEnv.camera_names)
        suites = benchmark.get_benchmark_dict()
        if suite not in suites:
            raise KeyError(f"no LIBERO suite {suite!r}; have {sorted(suites)}")
        self.suite = suite
        self.task_index = int(task_index)
        self.task_suite = suites[suite]()
        self.task = self.task_suite.get_task(self.task_index)
        self.init_states = self.task_suite.get_task_init_states(self.task_index)
        bddl = (f"{get_libero_path('bddl_files')}/{self.task.problem_folder}/"
                f"{self.task.bddl_file}")
        self.env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=render_size,
                                      camera_widths=render_size)
        self.instruction = self.task.language
        # The seed affects object positions even under a fixed initial state, so it is set at
        # construction and again here rather than left to whatever ran before.
        self.env.seed(env_seed)
        self.env_seed = int(env_seed)
        # `bddl_base_domain.py`'s `_setup_observables` creates a `<name>_pos`/`<name>_quat`
        # observable for exactly `self.objects`, the BDDL's `(:objects …)` -- the `(:fixtures …)`
        # get body ids but no observables, so a fixture is deliberately absent from this list.
        # `ControlEnv` has no `__getattr__`, which is why the robosuite env is reached as
        # `self.env.env` rather than through the wrapper.
        self.object_names = [o.name for o in self.env.env.objects]

    @property
    def num_init_states(self) -> int:
        return len(self.init_states)

    @property
    def obj_of_interest(self) -> list[str]:
        """The task definition's own objects. A *label source* may read this; a policy may not."""
        return list(getattr(self.env.env, "obj_of_interest", []) or [])

    def reset(self, init_state_index: int) -> dict:
        self.env.reset()
        return self.env.set_init_state(self.init_states[init_state_index])

    def step(self, action) -> StepResult:
        obs, reward, done, info = self.env.step(list(action))
        return StepResult(obs=obs, reward=float(reward), done=bool(done), info=info)

    def images(self, obs: dict) -> dict[str, np.ndarray]:
        return {
            "agentview": self.upright(obs["agentview_image"]),
            "wrist": self.upright(obs["robot0_eye_in_hand_image"]),
        }

    def privileged(self, obs: dict) -> dict:
        """Every movable object's world pose, and every fixture's pose **and collision boxes**.

        `<name>_pos` is the body's `body_xpos` and `<name>_quat` the same body's `body_xquat`
        converted to **xyzw**, which is what LIBERO's own observables carry. The order is
        `self.object_names`, i.e. the BDDL's `(:objects …)` order, so the short id
        `robojev.scene` assigns is stable across every step of every episode of a task.

        **The fixtures are the scene's furniture** -- the BDDL's `(:fixtures …)`, which have body
        ids and no observables, which is why they are absent from the movable list. A planner that
        cannot see them plans into them: on the drawer task the fingers stop against the cabinet's
        own drawer wall on every control step of every descent, and nothing in the state said the
        wall was there.

        Each fixture carries `fixture: True`, the root body's pose, and `boxes` -- its collision
        geoms as **oriented** boxes, `[cx, cy, cz, hx, hy, hz, r00 … r22]`, read live from
        `geom_xpos`/`geom_xmat` so an articulated part is where it actually is (an open top
        drawer's walls travel 15.6 cm out of the carcass, and its front panel with them).

        Two things about that shape are load-bearing. **Not one box per fixture**: a drawer's
        bounding box contains the bowl inside it, and it is the *walls* that stop a finger. And
        **not axis-aligned**: this cabinet is yawed 155 degrees, so the axis-aligned hull of its
        drawer floor -- a thin plate 21 cm across -- is a 21 cm cube that fills the interior the
        fingers have to enter, and every grasp reads as blocked (measured: all eight candidates at
        0.0 cm of room). The rotation travels with the box and `robojev.scene` measures in the
        box's own frame.
        """
        out = {name: {"pos": np.asarray(obs[f"{name}_pos"], np.float32),
                      "quat": np.asarray(obs[f"{name}_quat"], np.float32)}
               for name in self.object_names}
        out.update(self._fixtures())
        return out

    #: Geom groups that are collision geometry rather than decoration. robosuite puts visual
    #: meshes in group 1 and the collision primitives in group 0.
    COLLISION_GROUP: int = 0

    #: mjtGeom's box.
    BOX_GEOM: int = 6

    def _fixtures(self) -> dict:
        """`{fixture name: {pos, quat, fixture, boxes}}`, read from the live simulator."""
        env = self.env.env
        names = list(getattr(env, "fixtures_dict", {}) or {})
        if not names:
            return {}
        sim = env.sim
        model, data = sim.model, sim.data
        subtree = self._fixture_bodies(model, names)
        out: dict[str, dict] = {}
        for name in names:
            bodies = subtree.get(name, ())
            if not bodies:
                continue
            root = bodies[0]
            boxes = []
            for geom in range(model.ngeom):
                if model.geom_bodyid[geom] not in bodies:
                    continue
                if int(model.geom_group[geom]) != self.COLLISION_GROUP:
                    continue
                centre = np.asarray(data.geom_xpos[geom], np.float64)
                rotation = np.asarray(data.geom_xmat[geom], np.float64).reshape(3, 3)
                if int(model.geom_type[geom]) == self.BOX_GEOM:
                    half = np.asarray(model.geom_size[geom], np.float64)
                else:
                    # A geom whose `size` is not three half-extents (a sphere, a capsule, a mesh):
                    # the model's own bounding radius is the honest conservative answer, and a
                    # sphere has no orientation to get wrong.
                    half = np.full(3, float(model.geom_rbound[geom]))
                    rotation = np.eye(3)
                boxes.append([*np.round(centre, 5), *np.round(half, 5),
                              *np.round(rotation.reshape(9), 5)])
            out[name] = {
                "pos": np.asarray(data.body_xpos[root], np.float32),
                # `body_xquat` is wxyz and every other pose here is xyzw, as the observables are.
                "quat": np.asarray(np.roll(data.body_xquat[root], -1), np.float32),
                "fixture": True,
                "boxes": boxes,
            }
        return out

    @staticmethod
    def _fixture_bodies(model, names) -> dict[str, tuple[int, ...]]:
        """`{fixture name: every body id of its subtree}`, root first.

        By the body-name prefix LIBERO builds its fixtures with (`<name>_main` and the parts under
        it, e.g. `wooden_cabinet_1_cabinet_top`), because a drawer's own moving body is a child of
        the carcass and its geoms are what the plan has to see.
        """
        out: dict[str, list[int]] = {name: [] for name in names}
        for body in range(model.nbody):
            body_name = model.body_id2name(body) or ""
            for name in names:
                if body_name == f"{name}_main":
                    out[name].insert(0, body)
                elif body_name.startswith(f"{name}_"):
                    out[name].append(body)
        return {name: tuple(ids) for name, ids in out.items() if ids}

    def mujoco_sim(self):
        """The live MuJoCo handle. Wanted by the offline scene-cache builder and nothing else."""
        return self.env.sim

    def ground_truth(self):
        d = self.env.sim.data
        return np.array(d.qpos, dtype=np.float32), np.array(d.qvel, dtype=np.float32), float(d.time)

    def scene_xml(self) -> str:
        return self.env.sim.model.get_xml()

    def max_steps_default(self) -> int:
        return UPSTREAM_MAX_STEPS[self.suite]

    def close(self) -> None:
        self.env.close()

    @staticmethod
    def dummy_action() -> list:
        return list(DUMMY_ACTION)

    @staticmethod
    def state_vector(obs: dict) -> np.ndarray:
        """`eef_pos(3) + axis-angle(3) + gripper_qpos(2)`, taken as-is and never re-derived."""
        return np.concatenate((obs["robot0_eef_pos"],
                               quat2axisangle(obs["robot0_eef_quat"]),
                               obs["robot0_gripper_qpos"])).astype(np.float32)

    @staticmethod
    def upright(img: np.ndarray) -> np.ndarray:
        """LIBERO renders both cameras upside down and mirrored; this is the picture a human
        recognises, and it is what a replay bundle's video holds."""
        return np.ascontiguousarray(img[::-1, ::-1])


__all__ = ["CONTROL_FPS", "DUMMY_ACTION", "UPSTREAM_MAX_STEPS", "LiberoEnv", "quat2axisangle",
           "set_seed_everywhere"]
