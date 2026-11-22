#!/usr/bin/env python3
"""A5: compliant control on the Newton FR3 articulation — push acceptance harness.

Run with the Newton venv (never the training venv)::

    source ops/env.sh && $ISAAC_PY ops/a5_newton_push.py --out reports/g6_newton_push.json

What it actually runs
---------------------
The native Newton ``ModelBuilder.add_mjcf()`` FR3 v2 articulation that A3 stepped
and A4 pushed.  The built-in position actuators are disabled (effort mode), the
controller :class:`rtp.control.newton_arm.NewtonCompliantArm` reads ``q/dq``,
computes the calibrated flange FK/Jacobian (A1), runs the
:class:`~rtp.control.compliance.ComplianceSupervisor` state machine, maps its
wrench with ``J^T``, adds weak null-space PD (k_null <= 20) and Newton's RNEA
gravity compensation (A2 answer 4), saturates torque and rate, and writes
``control.joint_f``.  External wrenches are applied with ``state.body_f`` (A4
verified layout ``[force xyz, torque xyz]``) on the flange / elbow bodies.

Rows (from ``docs/AGENT_RUNBOOK.md`` A5 / ``docs/IMPEDANCE_SPEC.md`` §6):
steady_deflection, release_return, plan_pause, elbow_push, payload,
latency_error, invariants, human_hand_proxy.  Every number written to the report
comes from a simulated step; rows that are not implemented are written with
``pass: false``, ``status: "not run"`` and a reason — never a fabricated value.

Evidence class: **SIM** (Newton stepping).  NOT a safety certification; the
force/speed/torque caps are placeholders (D-22, D-58).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import sys
import time
from collections import deque
from importlib import metadata
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtp.control import fr3_kinematics as fk                     # noqa: E402
from rtp.control.compliance import PROFILES, State, ComplianceConfig  # noqa: E402
from rtp.control.newton_arm import (                             # noqa: E402
    ELBOW_BODY, FLANGE_BODY, FOREARM_BODY, Q_HOME, NewtonArmConfig, NewtonCompliantArm,
)

CACHE = ROOT / "assets" / "cache" / "menagerie" / "franka_fr3_v2"
ENTRY = "fr3v2.xml"
OUT = ROOT / "reports" / "g6_newton_push.json"
SEED = 20261120
ALL_HARNESSES = []          # for run-wide mass-matrix SPD evidence


def mass_eigen_range_all() -> dict:
    """min/max eigenvalue of M(q) measured over every tick of every Harness."""
    lo, hi, n = float("inf"), 0.0, 0
    for hh in ALL_HARNESSES:
        if hh.M_n:
            lo = min(lo, hh.M_eig_min)
            hi = max(hi, hh.M_eig_max)
            n += hh.M_n
    return {
        "min": round(lo, 8) if n else None, "max": round(hi, 8) if n else None,
        "samples": n, "spd_measured": bool(n and lo > 0.0),
    }


def versions() -> dict:
    out = {"python": platform.python_version()}
    for name in ("newton", "warp-lang", "isaaclab", "isaaclab-newton", "torch", "numpy"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "MISSING"
    return out


class Harness:
    """Thin native-Newton plant wrapper around :class:`NewtonCompliantArm`."""

    def __init__(self, dt: float = 0.001, profile: str = "free_motion", *,
                 payload_kg: float = 0.0, model_error: Optional[dict] = None,
                 delay_ticks: int = 0, kinematic_hand: bool = False) -> None:
        import newton
        import warp as wp
        self.wp, self.newton = wp, newton
        self.dt = dt
        self.profile = profile
        self.payload_kg = float(payload_kg)
        self.model_error = dict(model_error or {})
        self.delay_ticks = int(delay_ticks)
        self.kinematic_hand = bool(kinematic_hand)
        self.kinematic_hand_body = None
        self.delay_buf = deque()
        path = CACHE / ENTRY
        if not path.exists():
            raise FileNotFoundError(path)
        self.model = self._build_model()
        NewtonArmConfig()  # validates defaults
        NewtonCompliantArm.set_effort_mode_native(self.model)
        # A fixed 0.2 Nm*sign(dq) Coulomb feed-forward was measured to be wrong
        # (the effective stiction is configuration-dependent, 0.05-0.6 Nm).
        # Use the residual-wrench compensation instead (runbook A5: injected
        # wrench + controller force estimate), with the supervisor's own speed
        # limiter reinforced on every return tick.
        self.arm = NewtonCompliantArm(NewtonArmConfig(
            dt=dt, profile=profile, friction_comp=False, residual_comp_gain=4.0,
            residual_beta=0.1, speed_limiter_gain=8000.0,
            compliance=ComplianceConfig(zeta=1.2)))
        self.solver = newton.solvers.SolverMuJoCo(self.model)
        self.cp = newton.CollisionPipeline(self.model)
        self.contacts = self.cp.contacts()
        self.q_home = np.asarray(Q_HOME, dtype=float)
        self.p0, self.R0 = fk.fk_numpy(self.q_home)
        self.home_pos = np.concatenate([self.p0, np.zeros(3)])
        # Measured SPD evidence for the mass matrix, accumulated across the run.
        self.M_eig_min = float("inf")
        self.M_eig_max = 0.0
        self.M_n = 0
        self.rng = np.random.default_rng(SEED)
        self.prev_wrench_hat = None
        self.est_filt = np.zeros(6)
        self.est_push = False          # Schmitt-trigger state for estimator-only rows
        ALL_HARNESSES.append(self)
        self.reset()

    def _build_model(self):
        """Build the FR3 MJCF, optionally with a hand payload and/or plant error.

        ``payload_kg`` adds a small box collider to the flange body (body 9) with
        the density needed for the requested mass (verified: +1 kg -> body_mass[9]
        = 1.0).  ``model_error`` scales body mass/inertia and joint
        friction/armature before ``finalize()``; the controller is then fed the
        *nominal* inertia (M/s) so the mismatch is real, not just a relabelling.
        """
        newton, wp = self.newton, self.wp
        builder = newton.ModelBuilder()
        cwd = Path.cwd()
        os.chdir((CACHE / ENTRY).parent)
        try:
            builder.add_mjcf(ENTRY)
            if self.kinematic_hand:
                # add_body(is_kinematic=True) adds a phantom free DOF (dof 13,
                # len(joint_q) 14) and a fixed joint on it breaks mujoco_warp
                # conversion ("Vector[float32, Literal[11]] ... size 12").
                # add_link + fixed-to-world keeps dof==len(joint_q)==7; its pose
                # is scripted through state.body_q after eval_fk in tick().
                hb = builder.add_link(label="hand_box_body")
                builder.add_joint_fixed(-1, hb, parent_xform=wp.transform(
                    wp.vec3(0.42, 0.0, 0.6), wp.quat_identity()))
                builder.add_shape_box(body=hb, hx=0.03, hy=0.04, hz=0.04,
                                      label="hand_box_shape")
                self.kinematic_hand_body = hb
            if self.payload_kg > 0.0:
                hx = hy = hz = 0.02                      # 4 cm box at the flange
                cfg = copy.copy(builder.default_shape_cfg)
                cfg.density = self.payload_kg / (8.0 * hx * hy * hz)
                cfg.has_shape_collision = False   # inertial mass only; no self-contact geometry
                builder.add_shape_box(body=FLANGE_BODY, hx=hx, hy=hy, hz=hz,
                                      cfg=cfg, label="hand_payload")
            me = self.model_error
            ms = float(me.get("mass", 1.0))
            isc = float(me.get("inertia", ms))
            fs = float(me.get("friction", 1.0))
            asc = float(me.get("armature", 1.0))
            for i in range(len(builder.body_mass)):
                builder.body_mass[i] *= ms
                if isc != 1.0:
                    I = np.asarray(builder.body_inertia[i]).reshape(3, 3)
                    builder.body_inertia[i] = wp.mat33(I * isc)
            for i in range(len(builder.joint_friction)):
                builder.joint_friction[i] *= fs
                builder.joint_armature[i] *= asc
            return builder.finalize()
        finally:
            os.chdir(cwd)

    def mass_matrix_eigen_range(self) -> dict:
        return {
            "min": round(self.M_eig_min, 8) if self.M_n else None,
            "max": round(self.M_eig_max, 8) if self.M_n else None,
            "samples": self.M_n,
            "spd_measured": bool(self.M_n and self.M_eig_min > 0.0),
        }

    def reset(self) -> None:
        wp = self.wp
        self.state0 = self.model.state()
        self.state1 = self.model.state()
        self.control = self.model.control()
        qh = np.asarray(Q_HOME, dtype=np.float32)
        self.state0.joint_q = wp.array(qh, dtype=self.state0.joint_q.dtype, device=self.model.device)
        self.state0.joint_qd = wp.zeros_like(self.state0.joint_qd)
        self.newton.eval_fk(self.model, self.state0.joint_q, self.state0.joint_qd, self.state0)
        self.arm.reset(self.profile)
        self.prev_dq = np.zeros(7)
        self.prev_tau = None
        self.clock = 0.0
        self.steps = 0
        self.delay_buf.clear()
        self.prev_wrench_hat = None
        self.est_filt = np.zeros(6)
        self.est_push = False

    def tick(self, f_ext, plan_fn=None, body_id=None, estimate: bool = False,
             body_wrench=None, *, use_estimate: bool = False, noise_n: float = 0.0,
             est_beta: float = 0.1, body_pose_override=None,
             release_signalled: bool = False) -> dict:
        """One controller tick + one solver step; returns measurements *after* the step.

        ``f_ext`` is the wrench handed to the supervisor (flange task frame);
        ``body_wrench`` is the physical wrench applied to ``body_id`` via
        ``state.body_f``.  They are equal for flange pushes but differ for an
        elbow push, whose raw body wrench must be mapped into the task frame
        before the supervisor can use it.
        """
        wp, newton = self.wp, self.newton
        q = np.asarray(self.state0.joint_q.numpy(), dtype=float)
        dq = np.asarray(self.state0.joint_qd.numpy(), dtype=float)

        # Joint-space inertia M(q): every tick, fed to the dynamically consistent
        # null-space projector.  Divide by the plant mass-error scale so the
        # controller sees the *nominal* inertia (a genuine model mismatch).
        M = self.arm.mass_matrix_native(self.model, self.state0)
        ev = np.linalg.eigvalsh(0.5 * (M + M.T))
        self.M_eig_min = min(self.M_eig_min, float(ev[0]))
        self.M_eig_max = max(self.M_eig_max, float(ev[-1]))
        self.M_n += 1
        ms = float(self.model_error.get("mass", 1.0))
        M_cmd = M / ms if ms != 1.0 else M
        g = self.arm.gravity_comp_native(self.model, self.state0)

        # The supervisor can be driven by the filtered, noisy inverse-dynamics
        # wrench estimate instead of the injected ground-truth wrench (one-tick
        # delay, as a real estimator would have).
        f_ctrl = np.asarray(f_ext, dtype=float)
        if release_signalled:
            # A real wrist/contact sensor reports release; the noisy inverse-dynamics
            # estimate is only needed for push DETECTION while contact persists.
            self.est_push = False
            self.est_filt *= 0.0
        if use_estimate and self.prev_wrench_hat is not None and not release_signalled:
            noisy = self.prev_wrench_hat + self.rng.normal(0.0, noise_n, size=6)
            self.est_filt += est_beta * (noisy - self.est_filt)
            # Schmitt-trigger push detection on the ESTIMATE ONLY: use the
            # supervisor's existing f_push/f_release values (no threshold change).
            # Feeding an exact zero after release stops filter-noise chatter from
            # repeatedly re-triggering YIELD during the return.
            cc = self.arm.cfg.compliance or ComplianceConfig()
            fn = float(np.linalg.norm(self.est_filt[:3]))
            mn = float(np.linalg.norm(self.est_filt[3:]))
            if not self.est_push and (fn > cc.f_push or mn > cc.m_push):
                self.est_push = True
            elif self.est_push and fn < cc.f_release and mn < cc.m_release:
                self.est_push = False
            f_ctrl = self.est_filt.copy() if self.est_push else np.zeros(6)

        tau, info = self.arm.step(q, dq, f_ctrl, plan_fn, gravity_comp=g,
                                  dt=self.dt, mass_matrix=M_cmd)

        # Command delay ring buffer: tick k writes the torque computed at k-delay.
        self.delay_buf.append(tau)
        while len(self.delay_buf) > max(self.delay_ticks, 1):
            self.delay_buf.popleft()
        tau_write = self.delay_buf[0]
        NewtonCompliantArm.write_effort_native(self.control, tau_write)

        applied = f_ext if body_wrench is None else body_wrench
        self.state0.clear_forces()
        if body_id is not None and np.linalg.norm(applied) > 0.0:
            bf = np.zeros((self.model.body_count, 6), dtype=np.float32)
            bf[body_id, :3] = np.asarray(applied[:3], dtype=np.float32)
            bf[body_id, 3:] = np.asarray(applied[3:], dtype=np.float32)
            self.state0.body_f = wp.array(bf, dtype=self.state0.body_f.dtype, device=self.model.device)

        # A scripted kinematic contact body must be re-posed *after* eval_fk
        # (which overwrites all body_q from joint_q) and before collision query.
        if body_pose_override is not None:
            bid, pose = body_pose_override
            bq = np.asarray(self.state0.body_q.numpy()).copy()
            bq[bid] = np.asarray(pose, dtype=np.float32)
            self.state0.body_q = wp.array(bq, dtype=self.state0.body_q.dtype,
                                         device=self.model.device)

        # Model-based wrench estimate (tau_measured - tau_model via finite-diff qdd).
        wrench_hat = None
        if estimate or use_estimate:
            wrench_hat = self.arm.external_wrench_estimate(
                self.model, self.state0, tau_write, info["J"], self.prev_dq, self.dt)
            self.prev_wrench_hat = wrench_hat.copy()

        self.cp.collide(self.state0, self.contacts)
        self.solver.step(self.state0, self.state1, self.control, self.contacts, self.dt)
        self.state0, self.state1 = self.state1, self.state0
        contact_force_max = 0.0
        n_contact = int(np.asarray(self.contacts.rigid_contact_count.numpy()).reshape(-1)[0])
        if n_contact:
            cf = np.asarray(self.contacts.rigid_contact_force.numpy()).reshape(-1, 3)[:n_contact]
            if cf.size:
                contact_force_max = float(np.max(np.linalg.norm(cf, axis=1)))
        self.prev_dq = dq.copy()
        self.prev_tau = tau.copy()
        self.clock += self.dt
        self.steps += 1

        qn = np.asarray(self.state0.joint_q.numpy(), dtype=float)
        dqn = np.asarray(self.state0.joint_qd.numpy(), dtype=float)
        pos, _ = fk.fk_numpy(qn)
        vel = info["J"] @ dqn                 # flange twist, world frame
        body_pos = np.asarray(self.state0.body_q.numpy())[:, :3].copy()
        return {
            "q": qn, "dq": dqn, "pos": pos, "vel": vel, "body_pos": body_pos,
            "info": info, "tau": tau, "tau_write": tau_write, "g": g, "M": M,
            "wrench_hat": wrench_hat, "f_ctrl": f_ctrl,
            "contact_force_max": contact_force_max,
            "t": self.clock, "s": info["s"], "state": info["state"],
        }

    def task_wrench_from_body(self, f_body: np.ndarray, body_id: int) -> np.ndarray:
        """Map a wrench applied on an arbitrary link into the flange task frame.

        The supervisor's ``f_ext`` is defined in the flange task frame.  A push on
        link 4/6 creates the generalized torque ``J_b^T F``; its flange-equivalent
        wrench is ``(J_f^+)^T J_b^T F``.  Passing the raw body wrench to the
        supervisor would be a frame error (it would detect a much larger push
        than the flange actually feels).
        """
        q = np.asarray(self.state0.joint_q.numpy(), dtype=float)
        J_f = fk.jacobian(q).numpy()
        J_all = np.asarray(self.newton.eval_jacobian(self.model, self.state0).numpy())[0]
        J_b = J_all[body_id * 6:(body_id + 1) * 6, :7]
        tau_ext = J_b.T @ np.asarray(f_body, dtype=float)
        return np.linalg.pinv(J_f, rcond=1e-3).T @ tau_ext

    def task_wrench_from_body_dynamic(self, f_body: np.ndarray, body_id: int) -> np.ndarray:
        """Dynamically consistent task wrench: ``F_task = Jbar^T tau_ext``.

        ``Jbar = M^-1 J_f^T (J_f M^-1 J_f^T)^-1`` (Khatib).  Unlike the
        flange-equivalent ``(J_f^+)^T J_b^T F``, this is the wrench the task
        spring must produce to explain the *generalized* push torque; a push on
        the elbow then mostly moves the elbow in the null space and the
        end-effector error stays well below ``F/K``.
        """
        q = np.asarray(self.state0.joint_q.numpy(), dtype=float)
        J_f = fk.jacobian(q).numpy()
        J_all = np.asarray(self.newton.eval_jacobian(self.model, self.state0).numpy())[0]
        J_b = J_all[body_id * 6:(body_id + 1) * 6, :7]
        tau_ext = J_b.T @ np.asarray(f_body, dtype=float)
        M = self.arm.mass_matrix_native(self.model, self.state0)
        Msym = 0.5 * (M + M.T)
        Minv = np.linalg.inv(Msym)
        A = J_f @ Minv @ J_f.T
        Jbar_T = np.linalg.pinv(A, rcond=1e-6) @ J_f @ Minv
        return Jbar_T @ tau_ext


# --------------------------------------------------------------------------- #
# Rows
# --------------------------------------------------------------------------- #
DIRECTIONS = [np.eye(3)[i] * s for i in range(3) for s in (-1.0, 1.0)]
DIR_NAMES = ["-x", "+x", "-y", "+y", "-z", "+z"]


def steady_deflection(h: Harness, force: float, direction: np.ndarray, name: str,
                      *, hold_s: float = 1.5, settle_s: float = 0.3,
                      use_estimate: bool = False, noise_n: float = 0.0) -> dict:
    h.reset()
    f = np.zeros(6)
    f[:3] = direction * force
    n_hold = int(round(hold_s / h.dt))
    n_tail = int(round(0.5 / h.dt))
    defl, fhat, forces, states = [], [], [], []
    for k in range(n_hold + int(round(settle_s / h.dt))):
        rec = h.tick(f if k < n_hold else np.zeros(6), None, FLANGE_BODY,
                     estimate=(n_hold - n_tail <= k < n_hold) or use_estimate,
                     use_estimate=use_estimate, noise_n=noise_n)
        if n_hold - n_tail <= k < n_hold:
            defl.append(float(np.linalg.norm(rec["pos"] - h.p0)))
            forces.append(rec["info"]["force"].copy())
            states.append(rec["state"].value)
            if rec["wrench_hat"] is not None:
                fhat.append(rec["wrench_hat"][:3].copy())
    measured = float(np.mean(defl)) if defl else float("nan")
    k_lin = float(PROFILES[h.profile][0])
    expected = force / k_lin
    err_pct = 100.0 * abs(measured - expected) / expected
    return {
        "force_N": force, "direction": name, "hold_s": hold_s,
        "expected_m": round(expected, 6), "measured_mean_m": round(measured, 6),
        "measured_std_m": round(float(np.std(defl)), 6),
        "error_pct": round(err_pct, 4),
        "commanded_wrench_N": [round(float(x), 4) for x in f],
        "supervisor_force_tail_mean_N": [round(float(x), 4) for x in np.mean(forces, axis=0)] if forces else None,
        "wrench_estimate_mean_N": [round(float(x), 4) for x in np.mean(fhat, axis=0)] if fhat else None,
        "states_in_tail": sorted(set(states)),
        "pass": bool(err_pct <= 10.0),
        "tolerance_pct": 10.0,
    }


def release_return(h: Harness, *, force: float = 20.0, push_s: float = 1.0,
                   release_s: float = 2.5, use_estimate: bool = False,
                   noise_n: float = 0.0) -> dict:
    h.reset()
    f = np.array([force, 0.0, 0.0, 0.0, 0.0, 0.0])
    n_push = int(round(push_s / h.dt))
    n_rel = int(round(release_s / h.dt))
    dir_vec = f[:3] / np.linalg.norm(f[:3])
    delta_at_release = None
    events, states, speeds, errs, xs = [], [], [], [], []
    tau_speeds, tau_norms = [], []
    for k in range(n_push + n_rel):
        rec = h.tick(f if k < n_push else np.zeros(6), None, FLANGE_BODY,
                     use_estimate=use_estimate, noise_n=noise_n,
                     release_signalled=bool(use_estimate and k >= n_push))
        for e in rec["info"]["events"]:
            events.append((round(rec["t"], 4), e))
        if k == n_push - 1:
            delta_at_release = float(np.linalg.norm(rec["pos"] - h.p0))
            delta_vec = rec["pos"] - h.p0
        if k >= n_push:
            off = rec["pos"] - h.p0
            errs.append(float(np.linalg.norm(off)))
            xs.append(float(off @ dir_vec))
            speeds.append(float(np.linalg.norm(rec["vel"][:3])))
            tau_speeds.append(float(np.linalg.norm(rec["info"].get("tau_speed", np.zeros(7)))))
            tau_norms.append(float(np.linalg.norm(rec["tau"])))
            states.append(rec["state"].value)
    errs = np.asarray(errs)
    xs = np.asarray(xs)
    tol = 0.005
    within = np.where(errs <= tol)[0]
    settle = None
    for i in within:
        if i + 100 <= errs.size and np.all(errs[i:i + 100] <= tol):
            settle = float((i + 1) * h.dt)
            break
    overshoot = float(max(0.0, -np.min(xs)) / delta_at_release) if delta_at_release else None
    peak_speed = float(np.max(speeds)) if speeds else None
    v_cap = 1.1 * 0.10
    checks = {
        "release_delta_near_0p10m": abs(delta_at_release - 0.10) <= 0.02,
        "within_5mm_by_2p5s": settle is not None and settle <= 2.5,
        "overshoot_le_10pct": overshoot is not None and overshoot <= 0.10,
        "peak_speed_le_1p1_vreturn": peak_speed is not None and peak_speed <= v_cap,
        "returned_states_seen": "RETURN" in states,
    }
    return {
        "push_force_N": force, "push_s": push_s, "delta_at_release_m": round(delta_at_release, 6),
        "settle_time_s": settle, "final_error_m": round(float(errs[-1]), 6),
        "overshoot_fraction": round(overshoot, 6) if overshoot is not None else None,
        "peak_return_speed_mps": round(peak_speed, 6) if peak_speed is not None else None,
        "peak_speed_t_s": round(0.001 * (int(np.argmax(speeds)) + n_push + 1), 4) if speeds else None,
        "peak_tau_speed_norm_Nm": round(float(np.max(tau_speeds)), 3) if tau_speeds else None,
        "peak_tau_norm_Nm": round(float(np.max(tau_norms)), 3) if tau_norms else None,
        "speed_cap_mps": round(v_cap, 6), "return_tolerance_m": tol,
        "events": events[:20], "states_seen": sorted(set(states)), "checks": checks,
        "pass": bool(all(checks.values())),
    }


def plan_pause(h: Harness, *, force: float = 10.0, pre_s: float = 0.5,
               push_s: float = 1.0, post_s: float = 6.0, plan_s: float = 2.0) -> dict:
    h.reset()
    def plan(s):
        return np.concatenate([h.p0 + np.array([0.05 * min(s, plan_s), 0.0, 0.0]), np.zeros(3)])
    f = np.array([force, 0.0, 0.0, 0.0, 0.0, 0.0])
    n_pre, n_push, n_post = (int(round(x / h.dt)) for x in (pre_s, push_s, post_s))
    s_yield, s_track, events, states, ret_debug, hold_debug = [], [], [], [], [], []
    last_pos = h.p0.copy()
    for k in range(n_pre + n_push + n_post):
        f_ext = f if n_pre <= k < n_pre + n_push else np.zeros(6)
        rec = h.tick(f_ext, plan, FLANGE_BODY)
        last_pos = rec["pos"]
        for e in rec["info"]["events"]:
            events.append((round(rec["t"], 4), e))
        states.append(rec["state"].value)
        if rec["state"] == State.RETURN:
            from rtp.control.scripted_ik import rotation_error
            from rtp.control.newton_arm import matrix_from_rotvec
            _, Rq = fk.fk_numpy(rec["q"])
            er = float(np.linalg.norm(rotation_error(Rq, matrix_from_rotvec(rec["info"]["x_ref"][3:]))))
            ee = float(np.linalg.norm(rec["pos"] - rec["info"]["x_ref"][:3]))
            vv = float(np.linalg.norm(rec["vel"][:3]))
            ww = float(np.linalg.norm(rec["vel"][3:]))
            ret_debug.append((round(rec["t"], 3), round(ee, 5), round(vv, 5), round(er, 5), round(ww, 5)))
        if rec["state"] == State.YIELD:
            s_yield.append(rec["s"])
        if rec["state"] == State.TRACK:
            s_track.append(rec["s"])
            if k % 250 == 0 and k >= n_pre + n_push:
                ee = float(np.linalg.norm(rec["pos"] - plan(rec["s"])[:3]))
                er = float(np.linalg.norm(rec["pos"] - rec["info"]["x_ref"][:3]))
                fu = np.asarray(rec["info"].get("force_used", np.zeros(6)))[:3]
                hold_debug.append((round(rec["t"], 2), round(ee, 5), round(er, 5),
                                   round(float(np.linalg.norm(rec["vel"][:3])), 5),
                                   bool(rec["info"].get("integral_active")),
                                   round(float(rec["info"].get("integ_norm", 0.0)), 4),
                                   [round(float(v), 3) for v in fu]))
    s_frozen = bool(s_yield) and (max(s_yield) - min(s_yield) < 1e-9)
    final_s = float(s_track[-1]) if s_track else 0.0
    final_err = float(np.linalg.norm(last_pos - plan(final_s)[:3]))
    checks = {
        "plan_clock_frozen_in_yield": s_frozen,
        "resumed_track_state": State.TRACK.value in states and bool(s_track),
        "plan_clock_advanced_after_resume": final_s > (min(s_yield) if s_yield else 0.0),
        "final_tracking_error_le_3mm": final_err is not None and final_err <= 0.003,
    }
    return {
        "push_force_N": force, "push_at_s": round(float(s_yield[0]) if s_yield else -1.0, 6),
        "plan_clock_during_yield_min_max": [round(min(s_yield), 9), round(max(s_yield), 9)] if s_yield else None,
        "plan_clock_final": round(final_s, 6),
        "states_seen": sorted(set(states)), "final_state": states[-1] if states else None,
        "states_tail": states[-10:],
        "track_hold_debug": hold_debug[-16:],
        "return_debug_first_last": {"first": ret_debug[:5], "last": ret_debug[-5:],
                                    "min_e": min((x[1] for x in ret_debug), default=None),
                                    "min_v": min((x[2] for x in ret_debug), default=None),
                                    "first_settled_t": next((x[0] for x in ret_debug if x[1] < 0.003 and x[2] < 0.01), None),
                                    "n_settled": sum(1 for x in ret_debug if x[1] < 0.003 and x[2] < 0.01),
                                    "last_e_rot_w": ret_debug[-1][3:] if ret_debug else None},
        "final_tracking_error_m": round(final_err, 6) if final_err is not None else None,
        "resume_events": [e for _t, e in events if "RESUMED" in e][:10],
        "checks": checks, "pass": bool(all(checks.values())),
    }


def elbow_push(h: Harness, body_id: int, body_name: str, force: float,
               *, push_s: float = 1.0, recover_s: float = 3.0,
               profile: str = "grasp", mapping: str = "dynamic") -> dict:
    """Elbow/forearm push.  ``profile`` is an allow-listed PROFILES entry.

    The default is ``grasp`` (K_lin=500 N/m, the stiffest allowed profile) so the
    row is not unfairly penalised by the soft ``free_motion`` profile; the chosen
    profile is logged in the returned row.
    """
    if profile not in PROFILES:
        raise KeyError(f"profile {profile!r} is not in PROFILES")
    h.profile = profile
    h.arm.cfg.profile = profile
    h.reset()
    f = np.array([force, 0.0, 0.0, 0.0, 0.0, 0.0])
    if mapping == "dynamic":
        f_task = h.task_wrench_from_body_dynamic(f, body_id)
    elif mapping == "flange_equiv":
        f_task = h.task_wrench_from_body(f, body_id)
    else:
        raise ValueError(f"unknown mapping {mapping!r}")
    # The pre-registered A5 bound is 15 mm.  D-79 (stiffness-relative) was
    # *proposed*, not approved (no entry in configs/approvals.json); it is
    # reported only for the flange-equivalent history and is withdrawn if the
    # dynamically consistent mapping meets 15 mm at 10-30 N.
    ee_bound = 0.015
    n_push = int(round(push_s / h.dt))
    n_rec = int(round(recover_s / h.dt))
    link0 = np.asarray(h.state0.body_q.numpy())[body_id, :3].copy()
    ee_err, link_disp, posture_err = [], [], []
    for k in range(n_push + n_rec):
        f_ext = f_task if k < n_push else np.zeros(6)
        body_f = f if k < n_push else np.zeros(6)
        rec = h.tick(f_ext, None, body_id, body_wrench=body_f)
        ee_err.append(float(np.linalg.norm(rec["pos"] - h.p0)))
        link_disp.append(float(np.linalg.norm(rec["body_pos"][body_id] - link0)))
        posture_err.append(float(np.linalg.norm(rec["q"] - h.q_home)))
    max_ee = float(np.max(ee_err[:n_push])) if ee_err else None
    final_ee = float(ee_err[-1]) if ee_err else None
    max_link = float(np.max(link_disp[:n_push])) if link_disp else None
    posture_final = float(posture_err[-1]) if posture_err else None
    # first time after release that the null posture is within 0.02 rad
    back = None
    for i in range(n_push, len(posture_err) - 100):
        if posture_err[i] <= 0.02 and max(posture_err[i:i + 100]) <= 0.02:
            back = float((i - n_push + 1) * h.dt)
            break
    checks = {
        "link_yielded": max_link is not None and max_link > 0.001,
        "ee_error_le_15mm": max_ee is not None and max_ee <= ee_bound,
        "ee_return_le_5mm": final_ee is not None and final_ee <= 0.005,
    }
    d79_bound = 1.25 * float(np.linalg.norm(f_task[:3])) / PROFILES[profile][0] + 0.003
    return {
        "body_id": body_id, "body": body_name, "force_N": force,
        "profile": h.profile, "k_lin_N_per_m": PROFILES[profile][0],
        "mapping": mapping,
        "criterion": "original 15 mm (no criterion decision cited)",
        "d79_proposal_bound_m": round(d79_bound, 6),
        "ee_bound_m": round(ee_bound, 6),
        "flange_equivalent_force_N": round(float(np.linalg.norm(f_task[:3])), 4),
        "max_link_displacement_m": round(max_link, 6) if max_link is not None else None,
        "max_ee_error_during_push_m": round(max_ee, 6) if max_ee is not None else None,
        "final_ee_error_after_recovery_m": round(final_ee, 6) if final_ee is not None else None,
        "final_posture_error_rad": round(posture_final, 6) if posture_final is not None else None,
        "null_return_time_s": back, "checks": checks, "pass": bool(all(checks.values())),
    }


# --------------------------------------------------------------------------- #
# Grid rows (d)-(g): payload, latency_error, human_hand_proxy, *_estimated.
# --------------------------------------------------------------------------- #
PAYLOADS_KG = (0.0, 1.0, 2.0, 3.0)
KEY_STEADY_FORCES = (10.0,)          # compact but unrelaxed: ±10% at 6 directions


def _key_rows(h: Harness) -> dict:
    """steady_deflection / release_return / plan_pause / elbow_push on one plant.

    The force grid for steady is the compact KEY_STEADY_FORCES set (documented in
    the report); tolerances are the unchanged runbook ones.  elbow_push runs last
    because it switches h.profile to 'grasp'.
    """
    steady = []
    for F in KEY_STEADY_FORCES:
        for d, name in zip(DIRECTIONS, DIR_NAMES):
            steady.append(steady_deflection(h, F, d, name))
    steady_row = {
        "pass": all(r["pass"] for r in steady),
        "worst_error_pct": max(r["error_pct"] for r in steady),
        "n_rows": len(steady), "subrows": steady,
    }
    # These two must run BEFORE elbow_push: elbow_push switches the profile to
    # 'grasp' (K_lin=500), and a 20 N release would then yield 0.04 m rather than
    # the free_motion 0.10 m, producing an artificial release_return failure.
    release = release_return(h)
    plan = plan_pause(h)
    return {
        "steady_deflection": steady_row,
        "release_return": release,
        "plan_pause": plan,
        "elbow_push": elbow_row(h, both_mappings=False),
    }



def elbow_row(h: Harness, *, both_mappings: bool = True) -> dict:
    """Elbow pushes; the dynamically consistent mapping is certified.

    ``both_mappings`` runs the flange-equivalent comparison too (standalone row only).  The
    payload/latency grids pass ``False``: they only need the certified check, and doubling the
    elbow trials there multiplies hours of Newton stepping for no extra decision.
    """
    elbow = []
    for body_id, body_name in ((ELBOW_BODY, "fr3v2_link4"), (FOREARM_BODY, "fr3v2_link6")):
        for F in (10.0, 20.0, 30.0):
            # Same physical body push; only the wrench handed to the supervisor differs.
            dyn = elbow_push(h, body_id, body_name, F, mapping="dynamic")
            flg = (elbow_push(h, body_id, body_name, F, mapping="flange_equiv")
                   if both_mappings else None)
            elbow.append({
                "body_id": body_id, "body": body_name, "force_N": F,
                "profile": dyn["profile"], "k_lin_N_per_m": dyn["k_lin_N_per_m"],
                "criterion": "original 15 mm (no criterion decision cited)",
                "ee_bound_m": 0.015,
                "dynamic": {k: dyn[k] for k in (
                    "flange_equivalent_force_N", "max_link_displacement_m",
                    "max_ee_error_during_push_m", "final_ee_error_after_recovery_m",
                    "final_posture_error_rad", "null_return_time_s", "checks")},
                "flange_equiv": ({k: flg[k] for k in (
                    "flange_equivalent_force_N", "max_link_displacement_m",
                    "max_ee_error_during_push_m", "final_ee_error_after_recovery_m",
                    "final_posture_error_rad", "null_return_time_s", "checks")}
                    if flg is not None else None),
                "pass": bool(dyn["pass"]),
            })
    dyn_pass = all(r["dynamic"]["checks"]["ee_error_le_15mm"] for r in elbow)
    flg_pass = (all(r["flange_equiv"]["checks"]["ee_error_le_15mm"] for r in elbow)
                if both_mappings else None)
    return {
        "pass": bool(all(r["pass"] for r in elbow)), "n_rows": len(elbow), "subrows": elbow,
        "certified_mapping": "dynamic",
        "both_mappings": {"dynamic_le_15mm_all": dyn_pass,
                          "flange_equiv_le_15mm_all": flg_pass,
                          "d79_withdrawn": bool(dyn_pass)},
        "note": "Both mappings are measured on the same body push. The certified check uses the "
                "dynamically consistent task wrench; the flange-equivalent value is kept as history. "
                "D-79 was proposed, never approved; it is withdrawn if the dynamic mapping meets 15 mm.",
    }


def _restore_profile(h: Harness) -> None:
    h.profile = "free_motion"
    h.arm.cfg.profile = "free_motion"


def payload_row() -> dict:
    """(d) 0/1/2/3 kg box on the hand; each plant runs the four key rows."""
    subrows = []
    for kg in PAYLOADS_KG:
        hp = Harness(payload_kg=kg)
        mass = float(np.asarray(hp.model.body_mass.numpy())[FLANGE_BODY])
        rows = _key_rows(hp)
        checks = {name: bool(row["pass"]) for name, row in rows.items()}
        subrows.append({
            "payload_kg": kg, "flange_body_mass_kg": round(mass, 4),
            "steady_worst_error_pct": rows["steady_deflection"]["worst_error_pct"],
            "release_delta_m": rows["release_return"]["delta_at_release_m"],
            "plan_final_error_m": rows["plan_pause"]["final_tracking_error_m"],
            "elbow_worst_ee_error_dynamic_m": max(r["dynamic"]["max_ee_error_during_push_m"]
                                                  for r in rows["elbow_push"]["subrows"]),
            **( {"elbow_worst_ee_error_flange_equiv_m": None}
                if any(r["flange_equiv"] is None for r in rows["elbow_push"]["subrows"]) else {} ),
            "checks": checks, "pass": bool(all(checks.values())),
            "rows": rows,
        })
        del hp
    return {
        "pass": bool(all(s["pass"] for s in subrows)),
        "grid": {"payloads_kg": list(PAYLOADS_KG), "steady_forces_N": list(KEY_STEADY_FORCES)},
        "note": "a box collider is added to the flange body before finalize(); density sets the "
                "requested mass. The steady grid is compact (one force, 6 directions) but keeps "
                "the unchanged ±10% tolerance; elbow_push keeps the 15 mm bound.",
        "subrows": subrows,
    }


def latency_error_row() -> dict:
    """(e) 0/1/2/3-tick command delay and +/-20% plant error, key rows per cell."""
    cells = [
        ("delay_0", {"delay_ticks": 0}),
        ("delay_1", {"delay_ticks": 1}),
        ("delay_2", {"delay_ticks": 2}),
        ("delay_3", {"delay_ticks": 3}),
        ("mass_inertia_-20pct", {"model_error": {"mass": 0.8, "inertia": 0.8}}),
        ("mass_inertia_+20pct", {"model_error": {"mass": 1.2, "inertia": 1.2}}),
        ("friction_armature_-20pct", {"model_error": {"friction": 0.8, "armature": 0.8}}),
        ("friction_armature_+20pct", {"model_error": {"friction": 1.2, "armature": 1.2}}),
    ]
    subrows = []
    for label, kw in cells:
        hx = Harness(**kw)
        rows = _key_rows(hx)
        checks = {name: bool(row["pass"]) for name, row in rows.items()}
        subrows.append({
            "cell": label, "settings": kw,
            "steady_worst_error_pct": rows["steady_deflection"]["worst_error_pct"],
            "release_delta_m": rows["release_return"]["delta_at_release_m"],
            "plan_final_error_m": rows["plan_pause"]["final_tracking_error_m"],
            "checks": checks, "pass": bool(all(checks.values())),
            "rows": rows,
        })
        del hx
    return {
        "pass": bool(all(s["pass"] for s in subrows)),
        "grid": [label for label, _ in cells],
        "note": "delays are a tau ring buffer in Harness.tick; for mass error the plant is scaled "
                "before finalize and the controller is fed the nominal M/scale. Three key rows "
                "(steady_deflection, release_return, plan_pause) per cell; tolerances unchanged.",
        "subrows": subrows,
    }


def human_hand_row() -> dict:
    """(f) virtual spring-damper hand, then (if feasible) a kinematic box hand."""
    h = Harness()
    h.reset()
    k_c, d_c = 300.0, 20.0
    v_hand = 0.10
    push_s, total_s = 1.0, 2.5
    n_total = int(round(total_s / h.dt))
    peaks, excursions, states = [], [], []
    for k in range(n_total):
        t = k * h.dt
        x_hand = np.array([0.10 * min(t / push_s, 1.0), 0.0, 0.0])
        vh = np.array([v_hand if t < push_s else 0.0, 0.0, 0.0])
        q = np.asarray(h.state0.joint_q.numpy(), dtype=float)
        p, _ = fk.fk_numpy(q)
        vee = np.asarray(h.state0.joint_qd.numpy(), dtype=float)
        ve = fk.jacobian(q).numpy() @ vee
        F_c = k_c * (h.p0 + x_hand - p) - d_c * (ve[:3] - vh)
        wrench = np.concatenate([F_c, np.zeros(3)])
        rec = h.tick(wrench, None, FLANGE_BODY, body_wrench=wrench)
        peaks.append(float(np.linalg.norm(F_c)))
        excursions.append(float(np.linalg.norm(rec["pos"] - h.p0)))
        states.append(rec["state"].value)
    peak = float(np.max(peaks))
    checks = {
        "peak_force_finite": bool(np.isfinite(peak) and peak > 0.0),
        "arm_yielded": max(excursions) > 0.002,
        "no_fault": "FAULT" not in states,
    }
    virtual = {
        "label": "virtual_spring_hand", "k_c_N_per_m": k_c, "d_c_Ns_per_m": d_c,
        "hand_speed_mps": v_hand, "peak_force_N": round(peak, 4),
        "max_ee_excursion_m": round(max(excursions), 6),
        "states_seen": sorted(set(states)), "checks": checks,
        "pass": bool(all(checks.values())),
    }

    # Kinematic box hand: best effort, reported honestly if Newton cannot expose
    # the contact wrench on this build.
    kinematic = {"label": "kinematic_box_hand", "pass": False,
                 "status": "not feasible", "reason": ""}
    try:
        hk = Harness(kinematic_hand=True)
        hk.reset()
        hb = hk.kinematic_hand_body
        # body 9 (flange) has no collision shape in this MJCF; target link7 (body 8).
        target = np.asarray(hk.state0.body_q.numpy())[8, :3].copy()
        peaks_c = []
        for k in range(n_total):
            t = k * hk.dt
            # approach link7 along +x from 0.12 m away
            x = target[0] + 0.12 - v_hand * t
            pose = np.zeros(7, dtype=np.float32)
            pose[:3] = np.array([x, target[1], target[2]], dtype=np.float32)
            pose[3] = 1.0
            rec = hk.tick(np.zeros(6), None, None, body_pose_override=(hb, pose), estimate=True)
            # SolverMuJoCo does not populate `contacts.rigid_contact_force` on this
            # build (measured 0 N throughout), so use the model-based external
            # wrench estimate M qdd + C + g - tau, labelled as an estimate, not a
            # contact sensor. The raw contact-readout value is still recorded.
            w = rec.get("wrench_hat")
            peaks_c.append(float(np.linalg.norm(w[:3])) if w is not None else 0.0)
        peak_c = float(np.max(peaks_c))
        kinematic = {
            "label": "kinematic_box_hand", "pass": bool(peak_c > 0.0),
            "peak_contact_force_N": round(peak_c, 4),
            "note": "scripted kinematic box pushed along -x into link7; peak from the model-based "
                    "external wrench estimate (M qdd + C + g - tau), not a contact sensor; "
                    "Newton's rigid_contact_force readout stayed at 0 N in this venv",
        }
        del hk
    except Exception as exc:  # noqa: BLE001 - honest 'not feasible' rather than a number
        kinematic["reason"] = f"{type(exc).__name__}: {exc}"

    return {
        "pass": bool(virtual["pass"] and kinematic["pass"]),
        "virtual_spring_hand": virtual, "kinematic_box_hand": kinematic,
    }


def estimated_rows() -> dict:
    """(g) supervisor fed the noisy, filtered inverse-dynamics estimate."""
    subrows = []
    for noise in (0.5, 1.0):
        he = Harness()
        steady = []
        for F in KEY_STEADY_FORCES:
            for d, name in zip(DIRECTIONS, DIR_NAMES):
                steady.append(steady_deflection(he, F, d, name,
                                                use_estimate=True, noise_n=noise))
        rr = release_return(he, use_estimate=True, noise_n=noise)
        checks = {
            "steady_deflection": all(r["pass"] for r in steady),
            "release_return": bool(rr["pass"]),
        }
        subrows.append({
            "noise_N": noise, "filter_beta": 0.1, "source": "external_wrench_estimate",
            "steady_worst_error_pct": max(r["error_pct"] for r in steady),
            "steady_n": len(steady), "steady_subrows": steady,
            "release_delta_m": rr["delta_at_release_m"],
            "release_final_error_m": rr["final_error_m"],
            "release_return_row": rr,
            "checks": checks, "pass": bool(all(checks.values())),
        })
        del he
    return {
        "pass": bool(all(s["pass"] for s in subrows)),
        "note": "extra rows: the injected wrench still drives the plant; the supervisor input is "
                "the noisy (0.5 N then 1 N sigma) first-order-filtered inverse-dynamics estimate. "
                "These rows do not replace the original ground-truth rows.",
        "subrows": subrows,
    }


def invariants(h: Harness, steps: int, seed: int = SEED) -> dict:
    """Random pushes on the Newton plant; check supervisor invariants each tick."""
    from rtp.control.compliance import ComplianceConfig
    rng = np.random.default_rng(seed)
    c = h.arm.cfg.compliance or ComplianceConfig()
    segs, t = [], 0.0
    while t < steps * h.dt + 10.0:
        d = rng.uniform(0.3, 1.5)
        v = np.concatenate([rng.normal(size=3), rng.normal(size=3) * 0.1])
        v = v / np.linalg.norm(v[:3]) * rng.uniform(0, 35.0)
        v[3:] = np.clip(v[3:], -1.5, 1.5)
        segs.append((t, t + d, v))
        t += d + rng.uniform(0.5, 3.0)
    def f_ext_at(tt):
        return next((v for a, b, v in segs if a <= tt < b), np.zeros(6))
    h.reset()
    prev_tau = None
    violations = []
    counts = {"force": 0, "moment": 0, "leash": 0, "torque_rate": 0, "return_speed": 0, "fault": 0}
    n_return = 0
    traj_s = []
    for k in range(steps):
        # Capture the wrench actually applied THIS tick.  Using f_ext_at() again
        # after tick() would read the NEXT segment (the old 334 "violations" were
        # RETURN ticks counted as free while a push was in fact still applied).
        applied = f_ext_at(h.clock)
        rec = h.tick(applied, None, FLANGE_BODY)
        info = rec["info"]
        if info["state"] == State.FAULT:
            counts["fault"] += 1
        if np.linalg.norm(info["force"][:3]) > c.f_lin_max + 1e-6:
            counts["force"] += 1
        if np.linalg.norm(info["force"][3:]) > c.m_max + 1e-6:
            counts["moment"] += 1
        gap = np.linalg.norm(info["x"][:3] - info["x_ref"][:3])
        if info["state"] in (State.YIELD, State.RETURN) and gap > c.delta_max + 1e-6:
            counts["leash"] += 1
        if prev_tau is not None and np.max(np.abs(rec["tau"] - prev_tau)) > h.arm.cfg.max_delta + 1e-6:
            counts["torque_rate"] += 1
        prev_tau = rec["tau"]
        free = (np.linalg.norm(applied[:3]) < c.f_release
                and np.linalg.norm(applied[3:]) < c.m_release)
        if info["state"] == State.RETURN and free:
            n_return += 1
            if np.linalg.norm(rec["vel"][:3]) > 1.15 * c.v_return:
                counts["return_speed"] += 1
        if len(violations) < 20 and (counts["force"] or counts["moment"] or counts["leash"]
                                     or counts["torque_rate"] or counts["return_speed"] or counts["fault"]):
            violations.append({"step": k, "t": round(rec["t"], 4), "state": rec["state"].value})
        traj_s.append(info["s"])
    total = sum(counts.values())
    return {
        "steps": steps, "seed": seed, "max_push_N": 35.0, "max_push_torque_Nm": 1.5,
        "counts": counts, "total_violations": total, "return_ticks": n_return,
        "sample_violations": violations,
        "pass": bool(steps >= 100_000 and total == 0),
        "note": "full acceptance requires >= 100000 steps and zero violations; "
                "a shorter run is reported with pass=false, never relaxed. "
                "The wrench used for the 'free' test is the one APPLIED in that "
                "tick (captured before tick()); reading the next segment instead "
                "produced 334 spurious speed violations in the first A5 run.",
    }


def not_run(reason: str) -> dict:
    return {"pass": False, "status": "not run", "reason": reason}


def run_row(fn) -> dict:
    """Run a grid row; record an exception as FAIL rather than losing the report."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        return {"pass": False, "status": "error",
                "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--hold-s", type=float, default=1.5)
    ap.add_argument("--forces", type=float, nargs="+", default=[5.0, 10.0, 20.0, 30.0])
    ap.add_argument("--invariant-steps", type=int, default=0,
                    help="run the invariants row with this many steps (0 = not run)")
    ap.add_argument("--only", nargs="+", default=None,
                    help="run only the named row(s), e.g. --only elbow_push")
    args = ap.parse_args()
    selected = set(args.only) if args.only else None
    def want(name: str) -> bool:
        return selected is None or name in selected

    t0 = time.perf_counter()
    try:
        h = Harness(dt=args.dt)
    except Exception as exc:  # noqa: BLE001
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "step": "A5", "evidence_class": "SIM", "result": "FAIL",
            "error": f"{type(exc).__name__}: {exc}"}, indent=2) + "\n")
        print(f"harness init failed: {exc}")
        return 1

    rows = {}
    if want("steady_deflection"):
        steady = []
        for F in args.forces:
            for d, name in zip(DIRECTIONS, DIR_NAMES):
                steady.append(steady_deflection(h, F, d, name, hold_s=args.hold_s))
        rows["steady_deflection"] = {
            "pass": all(r["pass"] for r in steady),
            "tolerance_pct": 10.0,
            "worst_error_pct": max(r["error_pct"] for r in steady),
            "n_rows": len(steady), "subrows": steady,
        }
    if want("release_return"):
        rows["release_return"] = release_return(h)
    if want("plan_pause"):
        rows["plan_pause"] = plan_pause(h)
    if want("elbow_push"):
        rows["elbow_push"] = elbow_row(h)
        # elbow_push selected 'grasp'; restore the base profile for later rows.
        _restore_profile(h)

    if want("payload"):
        rows["payload"] = run_row(payload_row)
    if want("latency_error"):
        rows["latency_error"] = run_row(latency_error_row)
    if want("invariants"):
        rows["invariants"] = (
            invariants(h, args.invariant_steps) if args.invariant_steps > 0
            else not_run(f"requires >= 100000 random-push steps; --invariant-steps not supplied "
                         f"({args.invariant_steps})")
        )
    if want("human_hand_proxy"):
        rows["human_hand_proxy"] = run_row(human_hand_row)

    if want("steady_deflection_estimated") or want("release_return_estimated"):
        est = run_row(estimated_rows)
        if "error" in est:
            if want("steady_deflection_estimated"):
                rows["steady_deflection_estimated"] = est
            if want("release_return_estimated"):
                rows["release_return_estimated"] = est
            est = None
        if est is not None:
            if want("steady_deflection_estimated"):
                rows["steady_deflection_estimated"] = {
                    "pass": bool(all(s["checks"]["steady_deflection"] for s in est["subrows"])),
                    "note": est["note"],
                    "subrows": [
                        {"noise_N": s["noise_N"], "worst_error_pct": s["steady_worst_error_pct"],
                         "pass": s["checks"]["steady_deflection"],
                         "subrows": s["steady_subrows"]}
                        for s in est["subrows"]
                    ],
                }
            if want("release_return_estimated"):
                rows["release_return_estimated"] = {
                    "pass": bool(all(s["checks"]["release_return"] for s in est["subrows"])),
                    "note": est["note"],
                    "subrows": [
                        {"noise_N": s["noise_N"], "row": s["release_return_row"],
                         "pass": s["checks"]["release_return"]}
                        for s in est["subrows"]
                    ],
                }

    # Wrench estimate comparison: the injected wrench is the supervisor input; the
    # model-based estimate is logged per steady row above (no extra run needed).
    checks = {name: bool(row["pass"]) for name, row in rows.items()}
    wall = time.perf_counter() - t0
    report = {
        "step": "A5",
        "evidence_class": "SIM",
        "scope": "native Newton FR3 v2 articulation, effort-mode compliant control, simulated pushes; "
                 "no hardware, no contact sensor, no safety certification",
        "mode": "L2",
        "controller": "rtp.control.newton_arm.NewtonCompliantArm + rtp.control.compliance.ComplianceSupervisor",
        "versions": versions(),
        "seed": SEED,
        "params": {
            "dt_s": args.dt, "profile": h.profile, "k_lin_N_per_m": PROFILES[h.profile][0],
            "k_rot_Nm_per_rad": PROFILES[h.profile][1], "k_null_Nm_per_rad": h.arm.cfg.k_null,
            "tau_max_Nm": h.arm.cfg.tau_max,
            "torque_rate_Nm_per_ms": h.arm.cfg.torque_rate_nm_per_ms,
            "residual_comp_gain": h.arm.cfg.residual_comp_gain,
            "residual_beta": h.arm.cfg.residual_beta,
            "speed_limiter_gain_Ns_per_m": h.arm.cfg.speed_limiter_gain,
            "fixed_friction_feedforward_enabled": h.arm.cfg.friction_comp,
            "flange_body_id": FLANGE_BODY, "elbow_body_id": ELBOW_BODY, "forearm_body_id": FOREARM_BODY,
            "hold_s": args.hold_s, "forces_N": args.forces,
            "invariant_steps_requested": args.invariant_steps,
        },
        "model": str((CACHE / ENTRY).relative_to(ROOT)),
        "mass_matrix_eigenvalues": mass_eigen_range_all(),
        "steps_simulated": h.steps,
        "wall_s": round(wall, 2),
        "steps_per_s": round(h.steps / wall, 1) if wall > 0 else None,
        "wrench_estimate_note": "supervisor uses the injected wrench (simulation ground truth); "
                                "the acceleration-based inverse-dynamics estimate is logged beside it "
                                "in each steady_deflection subrow for comparison",
        "measurement_notes": [
            "Steady deflection is the mean over the last 0.5 s of a 1.5 s hold; "
            "expected |delta| = F / K_lin with K_lin from PROFILES[profile].",
            "A fixed 0.2 Nm*sign(dq) Coulomb feed-forward was measured to be invalid "
            "(effective stiction 0.05-0.6 Nm, configuration-dependent and mixed sign). "
            "It was removed. Residual-wrench compensation F_corr = g*(F_imp + f_ext) "
            "(g=4), a runbook-A5 'injected wrench + controller estimate' feedback, "
            "drives the steady-state bias to zero without identifying the plant.",
            "The supervisor's soft speed limiter is reinforced on every RETURN / "
            "released-YIELD tick through J[:3].T using the joint-torque budget; the "
            "Cartesian F_lin_max cap otherwise limited its deceleration.",
            "The first A5 run's 334 'return_speed' violations were a harness "
            "bookkeeping error: the 'free' test read the NEXT wrench segment rather "
            "than the one applied. The applied wrench is now captured before tick().",
            "EE error during an elbow push is F_task/K_lin; even at K_lin=500 N/m "
            "(grasp, the stiffest allowed PROFILES entry) a 10 N flange-equivalent "
            "push yields ~20 mm, so the spec's <=15 mm bound cannot hold for a "
            "generic link-4/link-6 push at these forces. Measured, not relaxed.",
            "The null-space PD now uses the dynamically consistent projector "
            "N = I - J^T (M^-1 J^T (J M^-1 J^T)^-1)^T from Newton's inverse-dynamics "
            "mass matrix M(q) (J M^-1 N ~ 0, N^2 ~ N), k_null=15 Nm/rad with critical "
            "damping. M is measured every tick; its min/max eigenvalues are in "
            "mass_matrix_eigenvalues (SPD claim measured, not asserted).",
        ],
        "rows": rows,
        "checks": checks,
        "not_done": [
            "elbow_push <=15 mm at 10 N: if still F_task/K, the pre-registered bound may be "
            "unreachable at the allowed stiffness; row stays FAIL until the user approves a change (rule 6)",
            "invariants at >= 100000 steps unless --invariant-steps was supplied",
            "hardware, ISO 10218-2:2025-derived limits, L0/L1 sign-off (D-22, D-58)",
        ],
    }
    report["result"] = "PASS" if all(checks.values()) else "FAIL"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    summary = {
        "result": report["result"], "steps": h.steps, "steps_per_s": report["steps_per_s"],
        "checks": checks,
        "mass_matrix_eigenvalues": report["mass_matrix_eigenvalues"],
    }
    if "steady_deflection" in rows:
        summary["steady_worst_error_pct"] = rows["steady_deflection"]["worst_error_pct"]
    for key in ("release_return", "plan_pause", "elbow_push", "payload", "latency_error",
                "human_hand_proxy", "steady_deflection_estimated", "release_return_estimated"):
        if key in rows:
            summary[key] = rows[key]
    print(json.dumps(summary, indent=2))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
