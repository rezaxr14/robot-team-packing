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
import json
import platform
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtp.control import fr3_kinematics as fk                     # noqa: E402
from rtp.control.compliance import PROFILES, State               # noqa: E402
from rtp.control.newton_arm import (                             # noqa: E402
    ELBOW_BODY, FLANGE_BODY, FOREARM_BODY, Q_HOME, NewtonArmConfig, NewtonCompliantArm,
)

CACHE = ROOT / "assets" / "cache" / "menagerie" / "franka_fr3_v2"
ENTRY = "fr3v2.xml"
OUT = ROOT / "reports" / "g6_newton_push.json"
SEED = 20261120


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

    def __init__(self, dt: float = 0.001, profile: str = "free_motion") -> None:
        import newton
        import warp as wp
        self.wp, self.newton = wp, newton
        self.dt = dt
        self.profile = profile
        path = CACHE / ENTRY
        if not path.exists():
            raise FileNotFoundError(path)
        builder = newton.ModelBuilder()
        cwd = Path.cwd()
        import os
        os.chdir(path.parent)
        try:
            builder.add_mjcf(ENTRY)
            self.model = builder.finalize()
        finally:
            os.chdir(cwd)
        NewtonArmConfig()  # validates defaults
        NewtonCompliantArm.set_effort_mode_native(self.model)
        # A fixed 0.2 Nm*sign(dq) Coulomb feed-forward was measured to be wrong
        # (the effective stiction is configuration-dependent, 0.05-0.6 Nm).
        # Use the residual-wrench compensation instead (runbook A5: injected
        # wrench + controller force estimate), with the supervisor's own speed
        # limiter reinforced on every return tick.
        self.arm = NewtonCompliantArm(NewtonArmConfig(
            dt=dt, profile=profile, friction_comp=False, residual_comp_gain=4.0,
            residual_beta=0.1, speed_limiter_gain=4000.0))
        self.solver = newton.solvers.SolverMuJoCo(self.model)
        self.cp = newton.CollisionPipeline(self.model)
        self.contacts = self.cp.contacts()
        self.q_home = np.asarray(Q_HOME, dtype=float)
        self.p0, self.R0 = fk.fk_numpy(self.q_home)
        self.home_pos = np.concatenate([self.p0, np.zeros(3)])
        self.reset()

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

    def tick(self, f_ext, plan_fn=None, body_id=None, estimate: bool = False,
             body_wrench=None) -> dict:
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
        g = self.arm.gravity_comp_native(self.model, self.state0)
        tau, info = self.arm.step(q, dq, f_ext, plan_fn, gravity_comp=g, dt=self.dt)
        NewtonCompliantArm.write_effort_native(self.control, tau)

        applied = f_ext if body_wrench is None else body_wrench
        self.state0.clear_forces()
        if body_id is not None and np.linalg.norm(applied) > 0.0:
            bf = np.zeros((self.model.body_count, 6), dtype=np.float32)
            bf[body_id, :3] = np.asarray(applied[:3], dtype=np.float32)
            bf[body_id, 3:] = np.asarray(applied[3:], dtype=np.float32)
            self.state0.body_f = wp.array(bf, dtype=self.state0.body_f.dtype, device=self.model.device)

        # Model-based wrench estimate (tau_measured - tau_model via finite-diff qdd).
        wrench_hat = None
        if estimate:
            wrench_hat = self.arm.external_wrench_estimate(
                self.model, self.state0, tau, info["J"], self.prev_dq, self.dt)

        self.cp.collide(self.state0, self.contacts)
        self.solver.step(self.state0, self.state1, self.control, self.contacts, self.dt)
        self.state0, self.state1 = self.state1, self.state0
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
            "info": info, "tau": tau, "g": g, "wrench_hat": wrench_hat,
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


# --------------------------------------------------------------------------- #
# Rows
# --------------------------------------------------------------------------- #
DIRECTIONS = [np.eye(3)[i] * s for i in range(3) for s in (-1.0, 1.0)]
DIR_NAMES = ["-x", "+x", "-y", "+y", "-z", "+z"]


def steady_deflection(h: Harness, force: float, direction: np.ndarray, name: str,
                      *, hold_s: float = 1.5, settle_s: float = 0.3) -> dict:
    h.reset()
    f = np.zeros(6)
    f[:3] = direction * force
    n_hold = int(round(hold_s / h.dt))
    n_tail = int(round(0.5 / h.dt))
    defl, fhat, forces, states = [], [], [], []
    for k in range(n_hold + int(round(settle_s / h.dt))):
        rec = h.tick(f if k < n_hold else np.zeros(6), None, FLANGE_BODY,
                     estimate=(n_hold - n_tail <= k < n_hold))
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
                   release_s: float = 2.5) -> dict:
    h.reset()
    f = np.array([force, 0.0, 0.0, 0.0, 0.0, 0.0])
    n_push = int(round(push_s / h.dt))
    n_rel = int(round(release_s / h.dt))
    dir_vec = f[:3] / np.linalg.norm(f[:3])
    delta_at_release = None
    events, states, speeds, errs, xs = [], [], [], [], []
    for k in range(n_push + n_rel):
        rec = h.tick(f if k < n_push else np.zeros(6), None, FLANGE_BODY)
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
    s_yield, s_track, events, states = [], [], [], []
    last_pos = h.p0.copy()
    for k in range(n_pre + n_push + n_post):
        f_ext = f if n_pre <= k < n_pre + n_push else np.zeros(6)
        rec = h.tick(f_ext, plan, FLANGE_BODY)
        last_pos = rec["pos"]
        for e in rec["info"]["events"]:
            events.append((round(rec["t"], 4), e))
        states.append(rec["state"].value)
        if rec["state"] == State.YIELD:
            s_yield.append(rec["s"])
        if rec["state"] == State.TRACK:
            s_track.append(rec["s"])
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
        "final_tracking_error_m": round(final_err, 6) if final_err is not None else None,
        "resume_events": [e for _t, e in events if "RESUMED" in e][:10],
        "checks": checks, "pass": bool(all(checks.values())),
    }


def elbow_push(h: Harness, body_id: int, body_name: str, force: float,
               *, push_s: float = 1.0, recover_s: float = 3.0,
               profile: str = "grasp") -> dict:
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
    f_task = h.task_wrench_from_body(f, body_id)
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
        "ee_error_le_15mm": max_ee is not None and max_ee <= 0.015,
        "null_posture_return_le_3s": back is not None and back <= 3.0,
    }
    return {
        "body_id": body_id, "body": body_name, "force_N": force,
        "profile": h.profile, "k_lin_N_per_m": PROFILES[h.profile][0],
        "max_link_displacement_m": round(max_link, 6) if max_link is not None else None,
        "max_ee_error_during_push_m": round(max_ee, 6) if max_ee is not None else None,
        "final_ee_error_after_recovery_m": round(final_ee, 6) if final_ee is not None else None,
        "final_posture_error_rad": round(posture_final, 6) if posture_final is not None else None,
        "null_return_time_s": back, "checks": checks, "pass": bool(all(checks.values())),
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--hold-s", type=float, default=1.5)
    ap.add_argument("--forces", type=float, nargs="+", default=[5.0, 10.0, 20.0, 30.0])
    ap.add_argument("--invariant-steps", type=int, default=0,
                    help="run the invariants row with this many steps (0 = not run)")
    args = ap.parse_args()

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
    rows["release_return"] = release_return(h)
    rows["plan_pause"] = plan_pause(h)
    elbow = []
    for body_id, body_name in ((ELBOW_BODY, "fr3v2_link4"), (FOREARM_BODY, "fr3v2_link6")):
        for F in (10.0, 20.0, 30.0):
            elbow.append(elbow_push(h, body_id, body_name, F))
    rows["elbow_push"] = {
        "pass": all(r["pass"] for r in elbow), "n_rows": len(elbow), "subrows": elbow,
    }
    # elbow_push selected the stiffest allowed PROFILES entry; restore the base
    # profile so the invariants row uses the same controller as the other rows.
    h.profile = "free_motion"
    h.arm.cfg.profile = "free_motion"
    h.reset()
    rows["payload"] = not_run("adding inertial mass to the hand requires rebuilding the Newton model "
                              "(body mass/inertia), not implemented in this pass")
    rows["latency_error"] = not_run("command delay and +/-20% inertia/friction error grid not implemented "
                                    "in this pass; no numbers claimed")
    rows["invariants"] = (
        invariants(h, args.invariant_steps) if args.invariant_steps > 0
        else not_run(f"requires >= 100000 random-push steps; --invariant-steps not supplied "
                     f"({args.invariant_steps})")
    )
    rows["human_hand_proxy"] = not_run("kinematic spring-damper hand contact body not implemented "
                                       "in this pass; no peak force claimed")

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
        ],
        "rows": rows,
        "checks": checks,
        "not_done": [
            "payload / latency_error / human_hand_proxy rows (marked not run, no numbers invented)",
            "invariants at >= 100000 steps unless --invariant-steps was supplied",
            "hardware, ISO 10218-2:2025-derived limits, L0/L1 sign-off (D-22, D-58)",
        ],
    }
    report["result"] = "PASS" if all(checks.values()) else "FAIL"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "result": report["result"], "steps": h.steps, "steps_per_s": report["steps_per_s"],
        "checks": checks,
        "steady_worst_error_pct": rows["steady_deflection"]["worst_error_pct"],
        "release_return": rows["release_return"],
        "plan_pause": rows["plan_pause"],
    }, indent=2))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
