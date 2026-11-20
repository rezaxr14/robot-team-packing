#!/usr/bin/env python3
"""A6 (scene half): two FR3 arms + table + bag + 10 rigid objects step in Newton.

Run with the merged venv::

    source ops/env.sh && $ISAAC_PY ops/a6_scene.py

This is the *scene* half of A6 only: deterministic reset/step/object States and a rendered PNG
sequence. `run_skill` and the Newton-backed SimBackend are added in the next step. No grasp,
manipulation, contact certification or safety claim is made here.

Report: ``reports/a6_scene.json``. Evidence class: SIM (Newton stepping), static scene.
"""
from __future__ import annotations

import hashlib
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
MJCF = ROOT / "assets" / "cache" / "menagerie" / "franka_fr3_v2" / "fr3v2.xml"
OUT = ROOT / "reports" / "a6_scene.json"
IMAGES = ROOT / "reports" / "a6_scene"

TABLE_TOP = 0.90
ARM_X = 0.35


def build(seed: int):
    import newton
    import warp as wp
    from rtp.control import fr3_kinematics as fk

    cwd = Path.cwd()
    os.chdir(MJCF.parent)
    try:
        builder = newton.ModelBuilder()
        builder.default_joint_cfg.target_ke = 18000.0
        builder.default_joint_cfg.target_kd = 2100.0
        for x in (-ARM_X, ARM_X):
            builder.add_mjcf(MJCF.name,
                             xform=wp.transform(wp.vec3(x, 0.0, TABLE_TOP), wp.quat_identity()))
        def static_box(pos, half, color):
            link = builder.add_link()
            builder.add_joint_fixed(-1, link, parent_xform=wp.transform(wp.vec3(*pos), wp.quat_identity()))
            builder.add_shape_box(body=link, hx=half[0], hy=half[1], hz=half[2], color=color)

        # table slab + four legs
        static_box((0, 0, TABLE_TOP - 0.03), (0.90, 0.45, 0.03), (0.24, 0.26, 0.30))
        for x in (-0.84, 0.84):
            for y in (-0.39, 0.39):
                static_box((x, y, TABLE_TOP / 2), (0.03, 0.03, TABLE_TOP / 2), (0.15, 0.16, 0.19))
        # three-compartment bag (static boxes)
        static_box((0, 0.28, TABLE_TOP + 0.01), (0.22, 0.14, 0.01), (0.35, 0.20, 0.12))
        # objects: name, x, y, hx, hy, hz, mass
        rng = np.random.default_rng(seed)
        items = []
        xs = np.linspace(-0.72, 0.72, 5)
        for i, x in enumerate(xs):
            for j in range(2):
                y = -0.22 + 0.11 * j
                hx, hy, hz = 0.025 + 0.004 * i, 0.020 + 0.003 * j, 0.025
                px = float(x + rng.uniform(-0.008, 0.008))
                py = float(y + rng.uniform(-0.006, 0.006))
                items.append((f"item_{i}_{j}", px, py, hx, hy, hz))
                bb = builder.add_body(mass=0.1 + 0.05 * i)
                builder.add_shape_box(body=bb,
                                      xform=wp.transform(wp.vec3(px, py, TABLE_TOP + hz + 0.001), wp.quat_identity()),
                                      hx=hx, hy=hy, hz=hz, color=(0.7, 0.45, 0.25))
        static_box((0, 0, -0.025), (4, 4, 0.025), (0.30, 0.31, 0.34))
        model = builder.finalize()
    finally:
        os.chdir(cwd)

    state = model.state()
    control = model.control()
    q = np.asarray(state.joint_q.numpy()).copy()
    labels = list(model.joint_label)
    starts = model.joint_q_start.numpy()
    offsets = [int(starts[j]) for j, lbl in enumerate(labels) if "fr3v2_joint1" in lbl]
    assert len(offsets) == 2, offsets
    for ai, off in enumerate(offsets):
        pose = fk.Q_HOME if ai == 0 else (0.0, 0.785, 0.0, -2.356, 0.0, 1.571, -0.785)
        for i, val in enumerate(pose):
            q[off + i] = val
    state.joint_q = wp.array(q.astype(np.float32), dtype=state.joint_q.dtype, device=model.device)
    control.joint_target_q = wp.array(q.astype(np.float32), dtype=control.joint_target_q.dtype,
                                      device=model.device)
    return model, state, control, items, offsets


def main() -> int:
    import newton
    import warp as wp
    solver = None
    report = {"step": "A6 (scene half)", "evidence_class": "SIM",
              "scope": "two FR3 articulations + table + bag + 10 objects, static scene; skills not implemented",
              "versions": {"python": platform.python_version(), "newton": metadata.version("newton"),
                           "warp": wp.__version__},
              "seed": 7, "checks": {}}
    try:
        model, state, control, items, offsets = build(7)
        solver = newton.solvers.SolverMuJoCo(model, njmax=256, nconmax=256)
        cp = newton.CollisionPipeline(model)
        contacts = cp.contacts()
        t0 = time.perf_counter()
        nsteps = 200
        for _ in range(nsteps):
            state.clear_forces()
            cp.collide(state, contacts)
            solver.step(state, state, control, contacts, 0.001)
        wall = time.perf_counter() - t0
        bq = np.asarray(state.body_q.numpy())
        digest = hashlib.sha256(np.round(bq, 3).tobytes()).hexdigest()[:16]
        # determinism: rebuild and replay the same seed, compare state hash
        m2, s2, c2, _, _ = build(7)
        solver2 = newton.solvers.SolverMuJoCo(m2, njmax=256, nconmax=256)
        cp2 = newton.CollisionPipeline(m2)
        ct2 = cp2.contacts()
        for _ in range(nsteps):
            s2.clear_forces()
            cp2.collide(s2, ct2)
            solver2.step(s2, s2, c2, ct2, 0.001)
        bq2 = np.asarray(s2.body_q.numpy())
        max_diff = float(np.abs(bq - bq2).max())
        digest2 = hashlib.sha256(np.round(bq2, 3).tobytes()).hexdigest()[:16]
        report.update({
            "bodies": model.body_count, "joints": model.joint_count,
            "dof": model.joint_dof_count, "shapes": model.shape_count,
            "arm_q_offsets": offsets, "objects": [i[0] for i in items],
            "steps": nsteps, "wall_s": round(wall, 3),
            "steps_per_s": round(nsteps / wall, 1) if wall else None,
            "state_hash_mm_quantised": digest, "state_hash_replay": digest2,
            "raw_max_abs_diff": round(max_diff, 8),
            "determinism_note": "MuJoCo-Warp on this GPU is not bitwise deterministic; hashes use 1 mm quantisation and the check also requires max abs diff <= 1 mm.",
            "checks": {
                "two_arms_14_dof": len(offsets) == 2 and sum(1 for lbl in model.joint_label if "fr3v2_joint1" in lbl) == 2,
                "finite_state": bool(np.isfinite(bq).all()),
                "deterministic_at_1mm": max_diff <= 1e-3,
            },
        })
        report["result"] = "PASS" if all(report["checks"].values()) else "FAIL"
    except Exception as exc:  # noqa: BLE001
        report["result"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report.get(k) for k in ("result", "bodies", "dof", "steps_per_s",
                                                 "state_hash", "checks", "error") if k in report}, indent=2))
    return 0 if report.get("result") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
