"""The table task at 20 Hz in the real XLeRobot driver's units.

Observations and actions use the LeRobot `xlerobot_2wheels` driver's 16 fields: arm joints in
degrees, grippers 0-100, head motors in degrees, base x.vel (m/s) and theta.vel (deg/s). The base is
parked and the head fixed, so those four are always 0 here, but they are carried so the same policy
output can go to the real driver. Joint zeros are the URDF's: before a hardware run each joint's
offset between the URDF and the calibrated motors must be measured once.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .scene import ARM_JOINTS, IMAGE, SUFFIX, Layout, build

CONTROL_HZ = 20
SUBSTEPS = 25  # x 2 ms = 50 ms
JAW_MAX = 1.7  # rad, URDF limit; the driver's gripper 100
DRIVER_JOINT = {"Rotation": "shoulder_pan", "Pitch": "shoulder_lift", "Elbow": "elbow_flex",
                "Wrist_Pitch": "wrist_flex", "Wrist_Roll": "wrist_roll"}
FIELDS = ([f"{arm}_arm_{f}.pos" for arm in ("left", "right") for f in [*DRIVER_JOINT.values(), "gripper"]]
          + ["head_motor_1.pos", "head_motor_2.pos", "x.vel", "theta.vel"])
CAMERAS = {"head": "head", "left_wrist": "left_wrist", "right_wrist": "right_wrist"}
TASK = "put the black bowl on the plate with the {arm} arm"
# Both arms start folded onto the cart. The URDF's zero pose holds them out over the table's edge,
# where the right arm hid a bowl on the right from the head camera and the policy started blind.
START = np.array([0.0, 3.1, 3.0, 1.0, 0.0, 0.0])  # Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw


def task(arm: str) -> str:
    """The instruction names the arm: which arm acts is decided above the policy (the planner or
    the Jev judge), the way the arm's own side decides it in the scripted demonstrations."""
    return TASK.format(arm=arm)


def _arm_slice(arm: str) -> slice:
    return slice(0, 6) if arm == "left" else slice(6, 12)


class TableEnv:
    def __init__(self, layout: Layout, image: int = IMAGE, render: bool = True):
        self.layout = layout
        self.model = mujoco.MjModel.from_xml_string(build(layout))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.renderer = mujoco.Renderer(m, image, image) if render else None
        # per arm: joint qpos addresses and actuators in driver order (5 joints + jaw)
        self.q_adr, self.act = {}, {}
        for arm, s in SUFFIX.items():
            names = [f"{n}{s}" for n in ARM_JOINTS] + [f"Jaw{s}"]
            self.q_adr[arm] = [m.jnt_qposadr[m.joint(n).id] for n in names]
            self.act[arm] = [m.actuator(f"a_{n}").id for n in names]
        self.bowl, self.plate = m.body("bowl").id, m.body("plate").id
        self.t = 0.0

    # ---- units ---------------------------------------------------------------------------------
    @staticmethod
    def to_driver(arm_q: dict[str, np.ndarray]) -> np.ndarray:
        """Joint radians (5 + jaw per arm) -> the driver's 16 fields."""
        out = np.zeros(len(FIELDS), np.float32)
        for arm, q in arm_q.items():
            v = np.rad2deg(np.asarray(q[:5], float))
            out[_arm_slice(arm)] = [*v, float(q[5]) / JAW_MAX * 100.0]
        return out

    @staticmethod
    def from_driver(a: np.ndarray) -> dict[str, np.ndarray]:
        a = np.asarray(a, float)
        return {arm: np.array([*np.deg2rad(a[_arm_slice(arm)][:5]), a[_arm_slice(arm)][5] / 100.0 * JAW_MAX])
                for arm in SUFFIX}

    # ---- loop ----------------------------------------------------------------------------------
    def reset(self) -> dict:
        mujoco.mj_resetData(self.model, self.data)
        self.hold = {arm: START.copy() for arm in SUFFIX}
        for arm, adr in self.q_adr.items():
            for i, v in zip(adr, START):
                self.data.qpos[i] = v
        self._apply(self.to_driver(self.hold))
        for _ in range(300):  # let the objects settle on the table (0.6 s, not part of the episode)
            mujoco.mj_step(self.model, self.data)
        self.rest_z = float(self.data.xpos[self.bowl][2])
        self.t = 0.0
        return self.observe()

    def _apply(self, action: np.ndarray) -> None:
        for arm, q in self.from_driver(action).items():
            for aid, v in zip(self.act[arm], q):
                self.data.ctrl[aid] = v

    def step(self, action: np.ndarray) -> dict:
        self._apply(action)
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
        self.t += 1.0 / CONTROL_HZ
        return self.observe()

    def joints(self) -> dict[str, np.ndarray]:
        return {arm: np.array([self.data.qpos[i] for i in adr]) for arm, adr in self.q_adr.items()}

    def observe(self) -> dict:
        obs = {"observation.state": self.to_driver(self.joints())}
        if self.renderer is not None:
            for key, cam in CAMERAS.items():
                self.renderer.update_scene(self.data, camera=cam)
                obs[f"observation.images.{key}"] = self.renderer.render().copy()
        return obs

    def render(self, camera: str = "front", size=(360, 480)) -> np.ndarray:
        """A view for people (videos), from a renderer kept per size."""
        cache = self.__dict__.setdefault("_views", {})
        if size not in cache:
            cache[size] = mujoco.Renderer(self.model, *size)
        cache[size].update_scene(self.data, camera=camera)
        return cache[size].render().copy()

    # ---- outcome -------------------------------------------------------------------------------
    def bowl_on_plate(self) -> tuple[bool, float, float]:
        """(on the plate, horizontal offset m, height above the plate m): within 4 cm of the plate's
        centre and resting at plate height, i.e. placed, not carried above it."""
        b, p = self.data.xpos[self.bowl], self.data.xpos[self.plate]
        horiz, dz = float(np.hypot(*(b[:2] - p[:2]))), float(b[2] - p[2])
        return horiz < 0.04 and -0.01 < dz < 0.05, horiz, dz

    def bowl_touched(self) -> bool:
        m, d = self.model, self.data
        for c in d.contact[: d.ncon]:
            names = (m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name)
            if "bowl" in names and any("Jaw" in n for n in names):
                return True
        return False

    def success(self) -> bool:
        """Placed and let go: on the plate, no jaw touching it, nearly still."""
        on, _, _ = self.bowl_on_plate()
        v = np.linalg.norm(self.data.cvel[self.bowl][3:])
        return on and not self.bowl_touched() and v < 0.02

    def close(self) -> None:
        for r in [self.renderer, *self.__dict__.get("_views", {}).values()]:
            if r is not None:
                r.close()
