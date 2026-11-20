#!/usr/bin/env python3
"""A3: one FR3 articulation steps in Newton and holds a pose.

Run with the Newton venv::

    source ops/env.sh && $ISAAC_PY ops/a3_arm_smoke.py --seconds 5 --dt 0.001

Scope / honesty
---------------
A2 found that runtime MJCF->USD conversion is unavailable in the Newton-only
venv (``isaacsim`` is absent), so we use the route A2 *did* verify: the native
Newton ``ModelBuilder.add_mjcf()`` articulation of the cached Menagerie FR3 v2
MJCF (also used by G-5 and A1).  This is still a Newton articulation with the
real joint/target gains from the MJCF; it is NOT an Isaac Lab ``Articulation``
wrapper and NOT a manipulation, contact or safety result.

Ladder: L1 = the MJCF's own position actuators hold ``Q_HOME`` (measured here);
L2 = effort mode with gravity compensation is recorded as not implemented if
A2 question 4 cannot be verified.  Report: ``reports/a3_arm_smoke.json``.
Evidence class: SIM (Newton stepping), static pose hold only.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "assets" / "cache" / "menagerie" / "franka_fr3_v2"
ENTRY = "fr3v2.xml"
OUT = ROOT / "reports" / "a3_arm_smoke.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--ke-scale", type=float, default=1.0,
                    help="scale the MJCF position-actuator stiffness (kd scales by sqrt)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    import newton
    import warp as wp
    from rtp.control import fr3_kinematics as fk

    path = CACHE / ENTRY
    if not path.exists():
        args.out.write_text(json.dumps({"step": "A3", "result": "FAIL",
                                        "error": f"missing {path}"}, indent=2) + "\n")
        print(f"missing {path}")
        return 1

    builder = newton.ModelBuilder()
    cwd = Path.cwd()
    os.chdir(path.parent)
    try:
        builder.add_mjcf(ENTRY)
        model = builder.finalize()
    finally:
        os.chdir(cwd)

    if args.ke_scale != 1.0:
        # Increase the MJCF position-actuator stiffness while keeping critical damping.
        ke = np.asarray(model.joint_target_ke.numpy(), dtype=float) * args.ke_scale
        kd = np.asarray(model.joint_target_kd.numpy(), dtype=float) * np.sqrt(args.ke_scale)
        model.joint_target_ke = wp.array(ke.astype(np.float32), dtype=model.joint_target_ke.dtype,
                                          device=model.device)
        model.joint_target_kd = wp.array(kd.astype(np.float32), dtype=model.joint_target_kd.dtype,
                                          device=model.device)

    q_home = np.asarray(fk.Q_HOME, dtype=np.float32)
    state0 = model.state()
    state1 = model.state()
    control = model.control()
    # L1: use the position targets/gains the MJCF already supplies (4500/450 etc.).
    state0.joint_q = wp.array(q_home, dtype=state0.joint_q.dtype, device=model.device)
    state0.joint_qd = wp.zeros_like(state0.joint_qd)
    control.joint_target_q = wp.array(q_home, dtype=control.joint_target_q.dtype, device=model.device)

    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    solver = newton.solvers.SolverMuJoCo(model)

    nsteps = int(round(args.seconds / args.dt))
    pos_err_mm: list[float] = []
    joint_err_rad: list[float] = []
    peak_torque = 0.0
    t0 = time.perf_counter()
    for k in range(nsteps):
        state0.clear_forces()
        collision_pipeline.collide(state0, contacts)
        solver.step(state0, state1, control, contacts, args.dt)
        state0, state1 = state1, state0
        q = np.asarray(state0.joint_q.numpy(), dtype=float)
        if k % 50 == 0 or k == nsteps - 1:
            pos, _ = fk.fk_numpy(q)
            target_pos, _ = fk.fk_numpy(q_home)
            pos_err_mm.append(float(np.linalg.norm(pos - target_pos) * 1000.0))
            joint_err_rad.append(float(np.max(np.abs(q - q_home))))
        tau = np.asarray(control.joint_f.numpy(), dtype=float)
        peak_torque = max(peak_torque, float(np.max(np.abs(tau))))

    wall = time.perf_counter() - t0
    final_q = np.asarray(state0.joint_q.numpy(), dtype=float)
    final_pos, _ = fk.fk_numpy(final_q)
    target_pos, _ = fk.fk_numpy(q_home)
    final_err_mm = float(np.linalg.norm(final_pos - target_pos) * 1000.0)
    max_err_mm = max(pos_err_mm) if pos_err_mm else float("inf")

    report = {
        "step": "A3",
        "evidence_class": "SIM",
        "scope": "Newton FR3 v2 articulation, static pose hold (position-actuator ladder L1); no contact/manipulation",
        "mode": "L1",
        "model": str(path.relative_to(ROOT)),
        "versions": {"python": platform.python_version(), "newton": metadata.version("newton"),
                     "warp": wp.__version__, "numpy": np.__version__},
        "params": {"seconds": args.seconds, "dt": args.dt, "substeps_per_frame": 1,
                   "ke_scale": args.ke_scale,
                   "target": "fr3v2_joint1..7 = Q_HOME",
                   "q_home": [round(float(v), 6) for v in q_home]},
        "steps": nsteps,
        "wall_s": round(wall, 3),
        "steps_per_s": round(nsteps / wall, 1) if wall > 0 else None,
        "final_flange_pos_m": [round(float(v), 6) for v in final_pos],
        "target_flange_pos_m": [round(float(v), 6) for v in target_pos],
        "final_pos_err_mm": round(final_err_mm, 6),
        "max_pos_err_mm": round(max_err_mm, 6),
        "max_joint_err_rad": round(max(joint_err_rad), 9),
        "peak_abs_joint_force": round(peak_torque, 4),
        "joint_force_observable_in_L1": False,
        "joint_force_note": "The MJCF position actuators drive through joint_target_q/joint_target_ke; control.joint_f stays zero in L1, so the 0.0 above is 'not observable', NOT zero actuator effort. Effort-mode torque is measured in A5.",
        "checks": {
            "drift_lt_1mm": final_err_mm < 1.0,
            "max_drift_lt_5mm": max_err_mm < 5.0,
            "finite": bool(np.isfinite(final_q).all()),
        },
        "not_done": [
            "L2 effort mode with gravity compensation (A2 question 4 unresolved).",
            "Newton articulation inside an Isaac Lab Articulation wrapper (MJCF conversion needs isaacsim).",
            "Any manipulation, external push or hardware result.",
        ],
    }
    report["result"] = "PASS" if all(report["checks"].values()) else "FAIL"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("result", "mode", "steps_per_s", "final_pos_err_mm", "max_pos_err_mm",
                       "peak_abs_joint_force", "checks")}, indent=2))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
