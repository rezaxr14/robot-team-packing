#!/usr/bin/env python3
"""A7 rung S5: weigh a held box from the gravity-compensation joint-torque difference.

FIXTURE script (code prepared, NOT run by the preparing agent; the Newton/GPU environment is
reserved for other jobs).  Invoked later with the merged venv it runs *one* seeded Newton trial
for one mass and writes ``reports/a7_s5.json``::

    source ops/env.sh && $ISAAC_PY ops/a7_s5_weigh.py --seed 0 --mass 0.3

Method: hold the arm still in effort mode with a payload rigidly attached to the flange and with
``NewtonCompliantArm.gravity_comp_native`` cancelling gravity; do the same for an otherwise
identical bare arm.  The difference ``dtau = tau_with - tau_bare`` is the payload's gravity
torque ``J^T [0,0,-m g,0,0,0]``.  Inverting with the flange Jacobian pseudoinverse recovers the
vertical force, hence ``m = -F_z / g``.  ``evidence_class: SIM`` is written only after a real
solver step returns; build failures write ``result: FAIL`` + ``reason``.

Pass bound (docs/AGENT_RUNBOOK.md A7 S5, verbatim): mass estimate within 15 % over 0.1-3 kg.
One mass per invocation; a shell driver sweeps the range.  No bound is loosened here.
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL = ROOT / "assets" / "derived" / "fr3v2_hand.xml"
OUT = ROOT / "reports" / "a7_s5.json"

BASE_Z = 0.90
DT = 0.001
G = 9.81                       # m/s^2, standard gravity used by the estimator
HOLD_STEPS = 500               # effort-mode ticks used to hold the arm still
PAYLOAD_HALF = 0.02            # 4 cm cube at the flange; density set to the requested mass

# Pass bound from the runbook (do not redefine).
MAX_REL_ERROR = 0.15           # |m_est - m_true| / m_true <= 0.15


def recover_flange_wrench(J, tau_with, tau_bare):
    """Recover the payload wrench at the flange: ``F = pinv(J^T) @ (tau_with - tau_bare)``.

    ``J`` is the (6, 7) world-frame flange Jacobian; the torque vectors are the arm-joint (7,)
    gravity-compensation differences.  Pure linear algebra: no newton, no torch, no GPU.
    """
    J = np.asarray(J, dtype=float).reshape(6, 7)
    dtau = np.asarray(tau_with, dtype=float).reshape(7) - np.asarray(tau_bare, dtype=float).reshape(7)
    F = np.linalg.pinv(J.T, rcond=1e-3) @ dtau
    return F


def estimate_mass_from_gravity_torque(J, tau_with, tau_bare, g: float = G) -> float:
    """Mass estimate (kg) from the gravity-comp joint-torque difference and flange Jacobian.

    A downward point load gives ``F = [0, 0, -m g, 0, 0, 0]`` in the world frame, so
    ``m_est = -F_z / g``.  Returns a float; never invents a measurement.
    """
    F = recover_flange_wrench(J, tau_with, tau_bare)
    return float(-F[2] / g)


def relative_error(estimate: float, true: float) -> float:
    return float(abs(estimate - true) / true)


def build(seed: int, mass: float, with_payload: bool):
    """Build one FR3-with-hand model, optionally with a payload welded to the flange."""
    import os
    import newton                       # lazy: CPU tests import this module without newton
    import warp as wp

    b = newton.ModelBuilder()
    cwd = Path.cwd()
    os.chdir(MODEL.parent)
    try:
        b.add_mjcf(MODEL.name, xform=wp.transform(wp.vec3(0.0, 0.0, BASE_Z), wp.quat_identity()))
        flange_matches = [k for k, lbl in enumerate(b.body_label)
                          if lbl.split("/")[-1] == "fr3v2_link8"]
        # The derived MJCF places the flange at body 9 (base=0, link0=1, ..., link8=9);
        # fall back to that verified index if this Newton build does not expose labels.
        flange_body = int(flange_matches[0]) if flange_matches else 9
        if with_payload:
            h = PAYLOAD_HALF
            cfg = copy.copy(b.default_shape_cfg)
            cfg.density = mass / (8.0 * h * h * h)
            cfg.has_shape_collision = False        # inertial mass only; no self-contact
            b.add_shape_box(body=flange_body, hx=h, hy=h, hz=h,
                            cfg=cfg, label="s5_payload")
        model = b.finalize()
    finally:
        os.chdir(cwd)
    return model, flange_body


def main() -> int:
    import newton
    import warp as wp
    from rtp.control import fr3_kinematics as fk
    from rtp.control.newton_arm import NewtonCompliantArm

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mass", type=float, default=0.3)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()

    report = {
        "step": "A7 S5 (weigh a held box from gravity-comp torque difference)",
        "evidence_class": "NOT_RUN",   # flipped to SIM only after a real solver.step returns
        "scope": "one seeded Newton effort-mode hold with the derived simplified parallel-jaw hand; "
                 "NOT a certification and NOT a safety result",
        "seed": a.seed, "mass_kg_true": a.mass,
        "hand_model": str(MODEL.relative_to(ROOT)),
        "method": "dtau = gravity_comp_with - gravity_comp_bare; F = pinv(J^T) dtau; m = -F_z/g",
        "pass_bounds": {"max_relative_error": MAX_REL_ERROR,
                        "source": "docs/AGENT_RUNBOOK.md A7 S5 (unchanged)"},
        "versions": {"python": platform.python_version(), "newton": metadata.version("newton"),
                     "warp": wp.__version__},
        "result": "FAIL", "reason": "not run",
    }
    stepped = [False]

    try:
        model_with, flange_body = build(a.seed, a.mass, with_payload=True)
        model_bare, _ = build(a.seed, a.mass, with_payload=False)
        NewtonCompliantArm.set_effort_mode_native(model_with)
        NewtonCompliantArm.set_effort_mode_native(model_bare)

        def init(model):
            state, control = model.state(), model.control()
            q = np.asarray(state.joint_q.numpy()).astype(float).copy()
            q[:7] = np.asarray(fk.Q_HOME, float)
            for i in range(7, len(q)):
                q[i] = 0.0
            state.joint_q = wp.array(q.astype(np.float32), dtype=state.joint_q.dtype, device=model.device)
            state.joint_qd = wp.zeros_like(state.joint_qd)
            control.joint_f = wp.zeros_like(control.joint_f)
            solver = newton.solvers.SolverMuJoCo(model, njmax=256, nconmax=256)
            cp = newton.CollisionPipeline(model)
            return state, control, solver, cp.contacts()

        state_w, ctrl_w, solver_w, contacts_w = init(model_with)
        state_b, ctrl_b, solver_b, contacts_b = init(model_bare)

        def hold(model, state, control, solver, contacts):
            """Apply gravity compensation in effort mode for HOLD_STEPS real Newton steps."""
            for _ in range(HOLD_STEPS):
                tau = NewtonCompliantArm.gravity_comp_native(model, state)
                NewtonCompliantArm.write_effort_native(control, tau)
                state.clear_forces()
                solver.step(state, state, control, contacts, DT)
                if not stepped[0]:
                    stepped[0] = True
                    report["evidence_class"] = "SIM"

        t0 = time.perf_counter()
        hold(model_with, state_w, ctrl_w, solver_w, contacts_w)
        hold(model_bare, state_b, ctrl_b, solver_b, contacts_b)
        wall = time.perf_counter() - t0

        tau_with = NewtonCompliantArm.gravity_comp_native(model_with, state_w)
        tau_bare = NewtonCompliantArm.gravity_comp_native(model_bare, state_b)
        q_arm = np.asarray(state_w.joint_q.numpy()).astype(float)[:7]
        J = fk.jacobian_numpy(q_arm)
        F_hat = recover_flange_wrench(J, tau_with[:7], tau_bare[:7])
        mass_est = estimate_mass_from_gravity_torque(J, tau_with[:7], tau_bare[:7])
        err = relative_error(mass_est, a.mass)
        flange_start, _ = fk.fk_numpy(np.asarray(fk.Q_HOME, float))
        flange_end, _ = fk.fk_numpy(q_arm)
        drift = float(np.linalg.norm(flange_end - flange_start))
        total_steps = 2 * HOLD_STEPS
        report.update({
            "flange_body_id": flange_body,
            "tau_with_arm_Nm": [round(float(v), 5) for v in tau_with[:7]],
            "tau_bare_arm_Nm": [round(float(v), 5) for v in tau_bare[:7]],
            "dtau_arm_Nm": [round(float(v), 5) for v in (tau_with[:7] - tau_bare[:7])],
            "recovered_flange_wrench": [round(float(v), 5) for v in F_hat],
            "mass_kg_estimated": round(mass_est, 5),
            "mass_kg_true": round(float(a.mass), 5),
            "relative_error": round(err, 5),
            "flange_drift_m": round(drift, 6),      # diagnostic, not a pass bound
            "wall_s": round(wall, 2), "steps": total_steps,
            "steps_per_s": round(total_steps / wall, 1),
            "checks": {
                "finite": bool(np.isfinite(tau_with).all() and np.isfinite(tau_bare).all()
                               and np.isfinite(F_hat).all()),
                "mass_estimate_within_15pct": err <= MAX_REL_ERROR,
            },
        })
        report["result"] = "PASS" if all(report["checks"].values()) else "FAIL"
        if report["result"] == "FAIL":
            report["reason"] = "mass estimate outside 15% or non-finite (see checks); one seed only"
        else:
            report["reason"] = "one seeded mass passed; >= 90% / >= 200 trials still required"
    except Exception as exc:  # noqa: BLE001
        report["result"] = "FAIL"
        report["reason"] = f"{type(exc).__name__}: {str(exc)[:500]}"

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report.get(k) for k in ("result", "evidence_class", "mass_kg_estimated",
                                                 "mass_kg_true", "relative_error", "checks", "reason")
                      if k in report}, indent=2))
    return 0 if report.get("result") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
