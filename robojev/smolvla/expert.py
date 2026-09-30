"""The scripted demonstrator: pick the bowl by its +y wall and place it on the plate, with the arm on
the bowl's side. It emits one 16-field driver action per 20 Hz step, as a teleoperator would.

What had to be right (each cost a failed round in the feasibility test): the grasp point is fixed
to the Fixed_Jaw body (the tips' midpoint swings as the single jaw rotates); the gripper axis runs
from the wrist to the fixed jaw's tip; the IK stays on the branch the arm is on; both arms grip the
bowl's +y wall, because the two jaws are not mirror images in the URDF; and a phase ends on arrival
or when the hand stops making progress, never by waiting out a step limit (the XLeRobot's 5-DoF
arm cannot always deliver the last few millimetres, and waiting for them froze the hand for half
the episode).
"""
from __future__ import annotations

from collections.abc import Iterator

import mujoco
import numpy as np

from .env import TableEnv
from .scene import ARM_JOINTS, SUFFIX

OPEN = 0.55  # jaw rad for the straddle (~4.8 cm gap)
TCP_JAW = 0.05  # the jaw opening at which the grasp point is defined
PITCH = 45.0  # approach pitch below horizontal, degrees
RIM_R, RIM_TOP = 0.047, 0.052  # LIBERO's black bowl
STEP = 0.01  # at most 1 cm of hand motion per 50 ms decision
SEEDS = ([0, 1.0, 1.2, 0.3, 0], [0, 0.5, 0.8, 1.0, 0], [0, 1.6, 1.8, 0.0, 0], [0, 2.0, 2.2, -0.5, 0], [0, 1.2, 2.0, 0.6, 0])


class Expert:
    def __init__(self, env: TableEnv, arm: str | None = None):
        self.env, self.arm = env, arm or env.layout.arm
        m = env.model
        self.dk = mujoco.MjData(m)
        s = SUFFIX[self.arm]
        self.jid = [m.joint(f"{n}{s}").id for n in ARM_JOINTS]
        self.qadr = [m.jnt_qposadr[j] for j in self.jid]
        self.lo = np.array([m.jnt_range[j][0] for j in self.jid])
        self.hi = np.array([m.jnt_range[j][1] for j in self.jid])
        self.fj = m.body(f"Fixed_Jaw{s}").id
        self.shoulder = m.body(f"Rotation_Pitch{s}").id
        self.t_fixed, self.t_moving = m.body(f"Fixed_Jaw_tip{s}").id, m.body(f"Moving_Jaw_tip{s}").id
        self.wrist = m.body(f"Wrist_Pitch_Roll{s}").id
        self.jaw_adr = m.jnt_qposadr[m.joint(f"Jaw{s}").id]
        self.tcp_l, self.across_l, self.axis_l = self._grasp_frame(TCP_JAW)
        self.w_across, self.w_pitch = 0.3, 0.08
        self.trace: list[str] = []

    def _grasp_frame(self, jaw: float):
        m, dd = self.env.model, mujoco.MjData(self.env.model)
        dd.qpos[self.jaw_adr] = jaw
        mujoco.mj_kinematics(m, dd)
        mid = 0.5 * (dd.xpos[self.t_fixed] + dd.xpos[self.t_moving])
        R = dd.xmat[self.fj].reshape(3, 3)
        ax = dd.xpos[self.t_fixed] - dd.xpos[self.wrist]
        ax /= np.linalg.norm(ax)
        across = dd.xpos[self.t_moving] - dd.xpos[self.t_fixed]
        across -= (across @ ax) * ax
        across /= np.linalg.norm(across)
        return R.T @ (mid - dd.xpos[self.fj]), R.T @ across, R.T @ ax

    def _frame(self, dd):
        R = dd.xmat[self.fj].reshape(3, 3)
        return dd.xpos[self.fj] + R @ self.tcp_l, R @ self.axis_l, R @ self.across_l

    def tcp(self) -> np.ndarray:
        mujoco.mj_kinematics(self.env.model, self.env.data)
        return self._frame(self.env.data)[0].copy()

    # ---- IK ------------------------------------------------------------------------------------
    def _ik_from(self, target, q0):
        m, dk = self.env.model, self.dk
        dk.qpos[:] = self.env.data.qpos
        mujoco.mj_kinematics(m, dk)
        h = target[:2] - dk.xpos[self.shoulder][:2]
        h /= np.linalg.norm(h) + 1e-12
        ph = np.radians(PITCH)
        want = np.array([h[0] * np.cos(ph), h[1] * np.cos(ph), -np.sin(ph)])
        q = np.array(q0, float)

        def setq(v):
            for k, i in enumerate(self.qadr):
                dk.qpos[i] = v[k]
            mujoco.mj_kinematics(m, dk)

        def res():
            tcp, axis, across = self._frame(dk)
            # jaws close along y (the +y wall grasp): the across direction wants |y| = 1
            return np.array([*(tcp - target), *(self.w_pitch * (axis - want)), self.w_across * (1 - abs(across[1]))])

        for _ in range(200):
            setq(q)
            r = res()
            if np.linalg.norm(r[:3]) < 5e-4 and np.linalg.norm(r[3:]) < 0.01:
                break
            J = np.zeros((7, 5))
            for k in range(5):
                dq = q.copy()
                dq[k] += 1e-4
                setq(dq)
                J[:, k] = (res() - r) / 1e-4
            q = np.clip(q + np.clip(np.linalg.solve(J.T @ J + 1e-3 * np.eye(5), -J.T @ r), -0.15, 0.15), self.lo, self.hi)
        setq(q)
        return q, float(np.linalg.norm(res()[:3]))

    def ik(self, target, q0):
        q, e = self._ik_from(target, q0)  # stay on the branch the arm is on
        if e < 0.02:
            return q
        best = (q, e)
        for seed in SEEDS:
            q2, e2 = self._ik_from(target, seed)
            if e2 < best[1]:
                best = (q2, e2)
        return best[0]

    # ---- the script ----------------------------------------------------------------------------
    def _action(self, q_arm: np.ndarray, jaw: float) -> np.ndarray:
        other = "left" if self.arm == "right" else "right"
        return self.env.to_driver({self.arm: np.array([*q_arm, jaw]), other: self.env.hold[other]})

    def _move(self, target, jaw, max_steps=80, tol=0.004) -> Iterator[np.ndarray]:
        moved = []
        for step in range(max_steps):
            cur = self.tcp()
            delta = target - cur
            dist = float(np.linalg.norm(delta))
            way = cur + (delta if dist < STEP else delta / dist * STEP)
            self.q = self.ik(way, self.q)
            yield self._action(self.q, jaw)
            moved.append(float(np.linalg.norm(self.tcp() - cur)))
            if dist < tol and step > 1:
                return
            if step >= 4 and sum(moved[-3:]) < 0.0015:  # no progress: go on rather than freeze
                return

    def _jaw(self, jaw, max_steps=20) -> Iterator[np.ndarray]:
        hist = []
        for step in range(max_steps):
            yield self._action(self.q, jaw)
            hist.append(float(self.env.data.qpos[self.jaw_adr]))
            if step >= 3 and abs(hist[-1] - hist[-3]) < 0.004:  # the jaw has stopped moving
                return

    def actions(self) -> Iterator[np.ndarray]:
        """The whole demonstration, one action per control step; read env state between steps."""
        env = self.env
        self.q = np.array([env.data.qpos[i] for i in self.qadr])
        b0, p0 = env.data.xpos[env.bowl].copy(), env.data.xpos[env.plate].copy()
        g = np.array([b0[0], b0[1] + RIM_R, b0[2] + RIM_TOP - 0.013])  # straddle the +y wall
        # from the folded start, come in from above: straight at the bowl the unfolding hand sweeps
        # across the table and knocks it away
        phases = [
            ("unfold", lambda: self._move(self.tcp() + [0, 0, 0.06], 0.0, max_steps=30)),
            ("over", lambda: self._move(g + [0, 0, 0.14], OPEN, max_steps=100)),
            ("pre-grasp", lambda: self._move(g + [0, 0, 0.06], OPEN)),
            ("straddle", lambda: self._move(g, OPEN)),
            ("close", lambda: self._jaw(0.0)),
        ]
        for name, gen in phases:
            self.trace.append(name)
            yield from gen()
        self.w_across, self.w_pitch = 0.03, 0.02  # a held bowl does not care about the approach angle
        off = g - b0
        for name, gen in [
            ("lift", lambda: self._move(g + [0, 0, 0.08], 0.0)),
            ("carry", lambda: self._move(np.array([p0[0] + off[0], p0[1] + off[1], g[2] + 0.08]), 0.0, max_steps=120)),
            ("lower", lambda: self._move(np.array([p0[0] + off[0], p0[1] + off[1], g[2] + 0.012]), 0.0)),
            ("release", lambda: self._jaw(OPEN, max_steps=15)),
            ("retreat", lambda: self._move(self.tcp() + [0, 0, 0.07], OPEN)),
        ]:
            self.trace.append(name)
            yield from gen()
        for _ in range(10):  # hold still a moment so the episode ends at rest
            yield self._action(self.q, OPEN)
