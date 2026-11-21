#!/usr/bin/env python3
"""A7 physical `release` / `retreat` trial (one seed per invocation).

The arm first executes the graded S2 grasp+lift recipe, then either opens the fingers over the
table (`release`) or moves the empty hand away from the object (`retreat`). Outcomes are derived
from Newton state: the box must land in the table band and the hand must end clear of it.

    python ops/a7_s7_release_retreat.py --skill release --seed 0 --mass 0.5

Evidence class SIM only after real solver steps; no threshold is defined here beyond the measured
clearance/table-band checks (the A7 gate uses the pre-registered per-skill rate).
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

MODEL = ROOT / "assets" / "derived" / "fr3v2_hand.xml"
BASE_Z = 0.90
BOX = 0.05
CLOSED_CTRL = 0.035
LIFT_DELTA = 0.14
DT = 0.001
CLEARANCE = 0.08


def main() -> int:
    import newton
    import os
    import warp as wp
    from rtp.control import fr3_kinematics as fk
    from rtp.control.scripted_ik import solve_ik

    ap = argparse.ArgumentParser()
    ap.add_argument("--skill", choices=("release", "retreat"), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mass", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or ROOT / "reports" / "a7_physical" / a.skill / f"s{a.seed}.json"

    report = {"step": f"A7 {a.skill} physical trial", "evidence_class": "SIM",
              "scope": "grasp+lift then release or retreat; Newton stepping, no safety claim",
              "skill": a.skill, "seed": a.seed, "mass_kg": a.mass,
              "versions": {"python": platform.python_version(), "newton": metadata.version("newton"),
                           "warp": wp.__version__}}
    try:
        rng = np.random.default_rng(a.seed)
        ph, Rh = fk.fk_numpy(np.asarray(fk.Q_HOME, float))
        x = float(ph[0]) + rng.uniform(-0.005, 0.005)
        y = float(ph[1]) + rng.uniform(-0.005, 0.005)
        grasp_z, zc = 0.12, BASE_Z + 0.12 - 0.095
        Rt = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        cwd = Path.cwd()
        os.chdir(MODEL.parent)
        try:
            b = newton.ModelBuilder()
            b.add_mjcf(MODEL.name, xform=wp.transform(wp.vec3(0.0, 0.0, BASE_Z), wp.quat_identity()))
            tbl = b.add_link(label="s7_table")
            b.add_joint_fixed(-1, tbl)
            b.add_shape_box(body=tbl, xform=wp.transform(wp.vec3(0, 0, BASE_Z - 0.02), wp.quat_identity()),
                            hx=0.9, hy=0.6, hz=0.02, color=(0.24, 0.26, 0.30))
            item = b.add_body(xform=wp.transform(wp.vec3(x, y, zc), wp.quat_identity()),
                              mass=a.mass, label="s7_item")
            b.add_shape_box(body=item, hx=BOX / 2, hy=BOX / 2, hz=BOX / 2, label="s7_item_box")
            model = b.finalize()
        finally:
            os.chdir(cwd)
        labels = [l.split("/")[-1] for l in model.joint_label]
        offs = [int(model.joint_q_start.numpy()[i]) for i, l in enumerate(labels)
                if l in ("hand_finger_left_joint", "hand_finger_right_joint")]
        fl, fr = offs
        state, control = model.state(), model.control()
        q = np.asarray(state.joint_q.numpy()).astype(float).copy()
        q[:7] = np.asarray(fk.Q_HOME, float)
        q[fl] = q[fr] = 0.0
        state.joint_q = wp.array(q.astype(np.float32), dtype=state.joint_q.dtype, device=model.device)
        control.joint_target_q = wp.array(q.astype(np.float32), dtype=control.joint_target_q.dtype,
                                          device=model.device)
        solver = newton.solvers.SolverMuJoCo(model, njmax=512, nconmax=512)
        cp, contacts = newton.CollisionPipeline(model), None
        contacts = cp.contacts()
        item_id = int(np.flatnonzero(["s7_item" in l for l in model.body_label])[0])

        def ik(z, q0):
            r = solve_ik(q0, (x, y, z), Rt, max_iters=250, pos_tol=1e-3, rot_tol=5e-3)
            if not r.converged:
                raise RuntimeError(f"IK failed at z={z}: {r.pos_error_m:.4f}/{r.rot_error_rad:.4f}")
            return r.path

        pre = ik(grasp_z + 0.10, fk.Q_HOME)
        grasp = ik(grasp_z, pre[-1])
        lift = ik(grasp_z + LIFT_DELTA, grasp[-1])
        retreat = ik(grasp_z + LIFT_DELTA, lift[-1]) if a.skill == "retreat" else None

        def set_cmd(arm_q, fc):
            c = np.asarray(control.joint_target_q.numpy()).astype(float).copy()
            c[:7] = np.asarray(arm_q, float)
            c[fl] = c[fr] = float(fc)
            control.joint_target_q = wp.array(c.astype(np.float32), dtype=control.joint_target_q.dtype,
                                              device=model.device)

        def step(n):
            for _ in range(n):
                state.clear_forces(); cp.collide(state, contacts)
                solver.step(state, state, control, contacts, DT)

        def ramp(path, fc, n):
            for k in range(n):
                u = min(1.0, (k + 1) / n)
                idx = u * (len(path) - 1)
                i = min(int(idx), len(path) - 2)
                set_cmd(path[i] + (idx - i) * (path[i + 1] - path[i]), fc)
                step(1)

        t0 = time.perf_counter()
        step(200)
        ramp(pre, 0.0, 1000); ramp(grasp, 0.0, 600)
        ramp(grasp, CLOSED_CTRL, 500); step(500)
        ramp(lift, CLOSED_CTRL, 800)
        z_held = float(np.asarray(state.body_q.numpy())[item_id, 2])
        if a.skill == "release":
            ramp(lift, 0.0, 400)      # open the fingers over the table
            step(800)                 # let the box settle
        else:
            ramp(retreat, CLOSED_CTRL, 800)
            ramp(grasp, 0.0, 400)
            step(400)
        bq = np.asarray(state.body_q.numpy())
        box = bq[item_id, :3]
        flange, _ = fk.fk_numpy(np.asarray(state.joint_q.numpy())[:7])
        dist = float(np.linalg.norm(flange - box))
        checks = {"finite": bool(np.isfinite(bq).all()),
                  "box_in_table_band": bool(BASE_Z - 0.02 <= box[2] <= BASE_Z + 0.10),
                  "lift_observed": bool(z_held >= zc + 0.08),
                  "hand_clear": bool(dist >= CLEARANCE)}
        report.update({"box_world": [round(float(v), 5) for v in box],
                       "flange_world": [round(float(v), 5) for v in flange],
                       "hand_box_distance_m": round(dist, 5),
                       "z_while_held_m": round(z_held, 5),
                       "steps": 4600, "wall_s": round(time.perf_counter() - t0, 2),
                       "checks": checks})
        report["result"] = "PASS" if all(checks.values()) else "FAIL"
    except Exception as exc:  # noqa: BLE001
        report["result"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"result": report["result"], "checks": report.get("checks"),
                      "error": report.get("error")}, indent=2))
    return 0 if report.get("result") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
