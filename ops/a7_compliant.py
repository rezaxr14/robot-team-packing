#!/usr/bin/env python3
"""Bridge: drive an A7 skill's arm through :class:`NewtonCompliantArm` (A5 controller).

The A7 rung scripts were written with the MJCF position servos.  The runbook's A7
section requires the graded skill trials to run **through NewtonCompliantArm** with
the phase stiffness profiles, so this module wires the existing scripted-IK joint
paths to the A5 compliant controller:

* the arm's position-servo gains are zeroed (the fingers keep their position servo);
* every control tick, the Cartesian pose of the current IK path point becomes the
  supervisor's plan target (arm-base frame), the controller returns a 7-joint torque,
  and it is published through ``control.joint_f``;
* the profile is switched by phase (approach / grasp / transport / place / retreat /
  handover), exactly the allow-list in ``rtp.control.compliance.PROFILES``.

Evidence class: SIM (the torque is integrated by Newton), the same controller as A5.
This is a measurement bridge, not a new controller: no gain, clamp or threshold here
is invented for the skill trials.
"""
from __future__ import annotations

import numpy as np


class CompliantArmRunner:
    """One 7-DoF arm of a Newton model driven by :class:`NewtonCompliantArm`."""

    def __init__(self, model, state, *, dof_offset: int = 0, articulation: int = 0, q_home,
                 profile: str = "approach",
                 dt: float = 0.001, residual_gain: float = 4.0) -> None:
        import warp as wp

        from rtp.control import fr3_kinematics as fk
        from rtp.control.compliance import ComplianceConfig
        from rtp.control.newton_arm import (NewtonArmConfig, NewtonCompliantArm,
                                            matrix_from_rotvec, rotvec_from_matrix)

        self.np = np
        self.wp = wp
        import newton
        self.model = model
        self.state = state
        self.newton = newton
        self.na = int(model.joint_dof_count)
        self.offset = int(dof_offset)
        self.articulation = int(articulation)
        self.fk = fk
        self.NCA = NewtonCompliantArm
        self.rotvec = rotvec_from_matrix
        self.mat = matrix_from_rotvec

        # Zero only this arm's position-servo gains.  The finger joints keep theirs so the
        # hand still closes under position control (the runbook's hand rule).
        ke = np.asarray(model.joint_target_ke.numpy(), dtype=float).copy()
        kd = np.asarray(model.joint_target_kd.numpy(), dtype=float).copy()
        sl = slice(self.offset, self.offset + 7)
        ke[sl] = 0.0
        kd[sl] = 0.0
        model.joint_target_ke = wp.array(ke.astype(np.float32), dtype=model.joint_target_ke.dtype,
                                         device=model.device)
        model.joint_target_kd = wp.array(kd.astype(np.float32), dtype=model.joint_target_kd.dtype,
                                         device=model.device)

        self.arm = NewtonCompliantArm(NewtonArmConfig(
            dt=dt, profile=profile, q_home=tuple(float(v) for v in q_home),
            friction_comp=False, residual_comp_gain=float(residual_gain), residual_beta=0.1,
            speed_limiter_gain=8000.0, compliance=ComplianceConfig(zeta=1.2)))
        p0, R0 = fk.fk_numpy(np.asarray(q_home, float))
        self.home_p = np.asarray(p0, float).copy()
        self.home_R = np.asarray(R0, float).copy()
        self.ref_rotvec = rotvec_from_matrix(self.home_R)
        self.target_p = self.home_p.copy()
        self.target_R = self.home_R.copy()
        self.last_info: dict = {}

    # -- targets ------------------------------------------------------------ #
    def set_target(self, p, R) -> None:
        """Set the Cartesian target (arm-base frame) for the next ticks."""
        self.target_p = np.asarray(p, float).reshape(3).copy()
        self.target_R = np.asarray(R, float).reshape(3, 3).copy()

    def set_joint_target(self, q7) -> None:
        """Set the target from a 7-vector of joint angles (the scripted IK path point)."""
        p, R = self.fk.fk_numpy(np.asarray(q7, float))
        self.set_target(p, R)

    def set_profile(self, profile: str) -> None:
        self.arm.sup.set_profile(profile)

    def _bias_and_mass(self):
        """Gravity/bias torque and joint-space inertia for the whole model.

        ``NewtonCompliantArm.gravity_comp_native``/``mass_matrix_native`` assume one articulation
        with no free joint; this scene has a free box (a second articulation), so the mass matrix
        is ``(articulation_count, max_dofs, max_dofs)`` and the gravity vector is
        ``(joint_dof_count,)``.  Call Newton's passive inverse dynamics with those shapes and
        slice this arm's articulation/dofs.  The A5 model and its helpers are not modified.
        """
        wp = self.wp
        st, model = self.state, self.model
        self.newton.eval_fk(model, st.joint_q, st.joint_qd, st)
        g = wp.zeros((self.na,), dtype=wp.float32, device=model.device)
        self.newton.eval_inverse_dynamics_passive(model, st, gravity_force=g)
        n = int(model.max_dofs_per_articulation)
        M = wp.zeros((int(model.articulation_count), n, n), dtype=wp.float32, device=model.device)
        self.newton.eval_inverse_dynamics_passive(model, st, mass_matrix=M)
        Ma = np.asarray(M.numpy(), dtype=float)[self.articulation]
        return (np.asarray(g.numpy(), dtype=float).reshape(-1), Ma)

    # -- one control tick --------------------------------------------------- #
    def tick(self, control, f_ext=None, plan=None) -> np.ndarray:
        """Compute and publish the arm torque for one solver step.

        ``plan`` overrides the default static target with a time-parameterized task-space plan
        ``plan(s) -> [px,py,pz,rx,ry,rz]`` (arm-base frame, rotvec relative to the home
        orientation).  This is the supervisor's intended usage: the plan clock pauses in
        YIELD and resumes in TRACK, so the arm must not be driven by an externally advancing
        target (a step in the target is seen as a push and triggers YIELD).
        """
        st = self.state
        q_all = np.asarray(st.joint_q.numpy(), dtype=float).reshape(-1)
        dq_all = np.asarray(st.joint_qd.numpy(), dtype=float).reshape(-1)
        sl = slice(self.offset, self.offset + 7)
        q, dq = q_all[sl], dq_all[sl]
        if plan is None:
            rot = self.rotvec(self.target_R @ self.home_R.T)
            plan = (lambda _s: np.concatenate([self.target_p, rot]))
        g_all, M_all = self._bias_and_mass()
        g = g_all[sl]
        M = M_all[sl, sl]
        f = np.zeros(6) if f_ext is None else np.asarray(f_ext, float)
        tau, info = self.arm.step(q, dq, f, plan, gravity_comp=g, mass_matrix=M,
                                  ref_rotvec=self.ref_rotvec)
        jf = np.zeros(q_all.shape[0], dtype=np.float32)
        jf[sl] = tau.astype(np.float32)
        self.NCA.write_effort_native(control, jf)
        self.last_info = info
        self.last_tau = tau
        return tau


class SkillArm:
    """Thin adapter between a skill script's joint-path ramps and the compliant runner.

    The rung scripts move along DLS-IK joint paths.  For the compliant grading, each path
    becomes a time-parameterized Cartesian plan through the FK poses of that path; the
    supervisor's own plan clock (which pauses in YIELD and resumes in TRACK) drives the
    motion, so no externally advancing target is used (a target step is seen as a push).
    """

    def __init__(self, runner: CompliantArmRunner, fl: int, fr: int, control) -> None:
        self.r = runner
        self.fl, self.fr, self.control = int(fl), int(fr), control
        self._finger = 0.0
        self.steps = 0
        self._q_spec = None

    # -- finger servo ------------------------------------------------------- #
    def set_fingers(self, ctrl: float) -> None:
        import warp as wp
        self._finger = float(ctrl)
        c = np.asarray(self.control.joint_target_q.numpy()).astype(float).copy()
        c[self.fl] = c[self.fr] = float(ctrl)
        self.control.joint_target_q = wp.array(c.astype(np.float32),
                                               dtype=self.control.joint_target_q.dtype,
                                               device=self.control.joint_target_q.device)

    # -- plans -------------------------------------------------------------- #
    def path_plan(self, path_q, duration: float | None = None, vmax: float = 0.10,
                  from_current: bool = True):
        """Plan a straight Cartesian move from the current pose to the path's goal pose.

        ``path_q`` is a ``solve_ik`` iterate path, NOT a trajectory: consecutive iterates
        move 35 % of the remaining error, so near convergence they overshoot and oscillate
        (measured on the S2 descent: flange z 0.2204 -> 0.1504 -> 0.1568 -> 0.1212, with
        ``rotvec`` dipping to ``[-0.16, +0.13, 2.36]``).  Following the FK poses of those
        iterates makes the plan itself oscillate -- a 0.19 rad orientation swing the 4 Nm
        moment cap cannot track, so the wrist lagged 0.30 rad and the pads dragged the box.
        Only the converged end pose is used here; the goal pose and the IK tolerances are
        unchanged.  Position is interpolated linearly and orientation along the geodesic
        (``R0 @ exp(u * log(R0^T R1))``), both with the same S-curve as the plan clock.

        The null-space posture reference is the straight joint-space line from the measured
        posture to ``path_q[-1]``, so it advances with the same clock and cannot fight the
        task spring.
        """
        q_now = np.asarray(self.r.state.joint_q.numpy(), dtype=float).reshape(-1)[self.r.offset:
                                                                               self.r.offset + 7]
        q_end = np.asarray(path_q[-1], float)
        if from_current:
            p0, R0 = self.r.fk.fk_numpy(q_now)
            q_null = q_now.copy()
        else:
            p0, R0 = self.r.fk.fk_numpy(np.asarray(path_q[0], float))
            q_null = np.asarray(path_q[0], float)
        p1, R1 = self.r.fk.fk_numpy(q_end)
        p0, R0 = np.asarray(p0, float), np.asarray(R0, float)
        p1 = np.asarray(p1, float)
        # geodesic increment in the start frame: R(u) = R0 @ exp(u * log(R0^T R1))
        rot_step = self.r.rotvec(R0.T @ np.asarray(R1, float))
        length = float(np.linalg.norm(p1 - p0))
        s0 = float(self.r.arm.sup.s)
        if duration is None:
            duration = max(0.45, length / max(float(vmax), 1e-3))
        dur = max(float(duration), 1e-3)
        home_R = self.r.home_R

        def plan(s):
            u = float(np.clip((s - s0) / dur, 0.0, 1.0))
            # S-curve: the linear ramp starts and stops with a velocity step, and the
            # supervisor sees that transient as a push (measured 30 mm excursion -> YIELD).
            u = u * u * (3.0 - 2.0 * u)
            p = p0 + u * (p1 - p0)
            R = R0 @ self.r.mat(rot_step * u)
            return np.concatenate([p, self.r.rotvec(R @ home_R.T)])

        # Leave the runner's static target at the plan's final pose so the hold that follows
        # does not command a different orientation.
        p_end, R_end = self.r.fk.fk_numpy(q_end)
        self.r.set_target(p_end, R_end)
        self._q_spec = ([q_null, q_end], s0, dur)
        return plan, s0 + dur

    def static_plan(self, q7=None):
        """Plan that holds the FK pose of ``q7`` (default: the last path/target pose)."""
        if q7 is not None:
            p, R = self.r.fk.fk_numpy(np.asarray(q7, float))
            self.r.set_target(p, R)
        p, R = self.r.target_p, self.r.target_R
        rot = self.r.rotvec(R @ self.r.home_R.T)
        return lambda _s: np.concatenate([p, rot])

    # -- execution ---------------------------------------------------------- #
    def run_plan(self, plan, t_end: float, step_fn, *, finger: float | None = None,
                 guard: float = 4.0) -> int:
        """Run until the supervisor's plan clock reaches ``t_end``; returns ticks used.

        ``step_fn(plan)`` must execute exactly one physics tick with the given plan.  The
        guard caps the ticks at ``guard``x the planned duration in case the supervisor is
        stuck in FAULT; it never truncates a normal run.
        """
        f = self._finger if finger is None else float(finger)
        self.set_fingers(f)
        max_ticks = int(max(1.0, (t_end - float(self.r.arm.sup.s))) / self.r.arm.cfg.dt * guard) + 10
        k = 0
        while float(self.r.arm.sup.s) < t_end - 1e-9 and k < max_ticks:
            self._advance_null_target()
            step_fn(plan)
            k += 1
        self.steps += k
        return k

    def _advance_null_target(self) -> None:
        """Posture (null-space) reference at the current plan time."""
        if self._q_spec is None:
            return
        qs, s0, dur = self._q_spec
        if len(qs) == 1:
            self.r.arm.q_home = np.asarray(qs[0], float).copy()
            return
        u = float(np.clip((float(self.r.arm.sup.s) - s0) / max(dur, 1e-6), 0.0, 1.0))
        # Same S-curve as the Cartesian plan in `path_plan`.  Without it the posture
        # reference runs ahead of the task target by the smoothstep gap (up to ~7 % of the
        # path, i.e. ~9 deg of wrist rotation at a 135 deg grasp yaw) and the null-space PD
        # drags the wrist off the planned orientation (measured as a 17 deg hand tilt and a
        # 40 mm box drag during the compliant descent).
        u = u * u * (3.0 - 2.0 * u)
        x = u * (len(qs) - 1)
        i = min(int(x), len(qs) - 2)
        w = x - i
        self.r.arm.q_home = (1.0 - w) * qs[i] + w * qs[i + 1]

    def hold(self, step_fn, n: int, *, finger: float | None = None, q7=None) -> None:
        if q7 is not None:
            self._q_spec = ([np.asarray(q7, float).copy()], float(self.r.arm.sup.s), 1.0)
        elif self._q_spec is not None:
            self._q_spec = ([self._q_spec[0][-1]], float(self.r.arm.sup.s), 1.0)
        plan = self.static_plan(q7)
        f = self._finger if finger is None else float(finger)
        self.set_fingers(f)
        for _ in range(int(n)):
            self._advance_null_target()
            step_fn(plan)
        self.steps += int(n)
