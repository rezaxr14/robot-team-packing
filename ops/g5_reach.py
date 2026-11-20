#!/usr/bin/env python3
"""A1 / G-5: verify ``rtp.control.fr3_kinematics`` against Newton's own FK.

Run with the Newton venv (``$ISAAC_PY``), never the training venv::

    source ops/env.sh && $ISAAC_PY ops/g5_reach.py

What it does
------------
1. Loads ``assets/cache/menagerie/franka_fr3_v2/fr3v2.xml`` into Newton exactly as
   ``ops/g5_fr3_load.py`` does (fixed base, 7 DoF).
2. Samples 100 joint configurations inside the MJCF joint limits.
3. For each: Newton ``eval_fk`` -> world pose of ``fr3v2_link8`` (the flange); our
   ``fr3_kinematics.fk``; records position error (mm) and rotation error (deg).
4. Also records the zero pose reach, the joint limits from the model and the total mass
   (datasheet comparison still pending — not claimed here).

Report: ``reports/g5_reach.json`` with ``result`` = PASS iff
``max_pos_err_mm <= 1.0`` and ``max_rot_err_deg <= 0.05``.

Evidence class: SIM-POSE (Newton forward kinematics only; no stepping, no contact).
"""
from __future__ import annotations

import json
import os
import platform
import sys
from importlib import metadata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "assets" / "cache" / "menagerie" / "franka_fr3_v2"
ENTRY = "fr3v2.xml"
FLANGE_LABEL = "fr3v2_link8"
OUT = ROOT / "reports" / "g5_reach.json"

POS_TOL_MM = 1.0
ROT_TOL_DEG = 0.05


def _rot_err_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Angle of the relative rotation R_a^T R_b, in degrees."""
    R = R_a.T @ R_b
    cos = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def main() -> int:
    import newton
    import warp as wp
    from rtp.control import fr3_kinematics as fkmod

    path = CACHE / ENTRY
    if not path.exists():
        OUT.write_text(json.dumps({"step": "A1", "result": "FAIL",
                                   "error": f"missing model {path}"}, indent=2) + "\n")
        print(f"missing model {path}")
        return 1

    builder = newton.ModelBuilder()
    cwd = Path.cwd()
    os.chdir(path.parent)
    try:
        builder.add_mjcf(ENTRY)
        model = builder.finalize()
    finally:
        os.chdir(cwd)

    labels = list(model.body_label)
    flange_idx = next(i for i, lbl in enumerate(labels) if lbl.endswith("/" + FLANGE_LABEL))

    lo = model.joint_limit_lower.numpy().astype(float)
    hi = model.joint_limit_upper.numpy().astype(float)
    q_start = model.joint_q_start.numpy()
    # The 7 arm joints are the ones with a non-zero q dimension, in chain order.
    arm_joint_idx = [j for j in range(model.joint_count) if q_start[j + 1] - q_start[j] == 1]
    if len(arm_joint_idx) != 7:
        OUT.write_text(json.dumps({"step": "A1", "result": "FAIL",
                                   "error": f"expected 7 driven joints, found {len(arm_joint_idx)}"},
                                  indent=2) + "\n")
        print("unexpected joint layout")
        return 1

    # Cross-check our embedded limits against the model's.
    limits_match = bool(
        np.allclose(lo, np.asarray(fkmod.JOINT_LIMITS_LOWER), atol=1e-6)
        and np.allclose(hi, np.asarray(fkmod.JOINT_LIMITS_UPPER), atol=1e-6)
    )

    state = model.state()
    rng = np.random.default_rng(20261119)
    # 100 random configurations inside the limits, plus the zero pose and Q_HOME.
    samples = [np.zeros(7), np.asarray(fkmod.Q_HOME, dtype=float)]
    samples += [rng.uniform(lo, hi) for _ in range(100)]

    rows = []
    max_pos = 0.0
    max_rot = 0.0
    for k, q in enumerate(samples):
        state.joint_q = wp.array(q.astype(np.float32), dtype=state.joint_q.dtype, device=model.device)
        state.joint_qd = wp.zeros_like(state.joint_qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        bq = state.body_q.numpy()[flange_idx]
        wp_pos = np.asarray(bq[:3], dtype=float)
        wp_quat = np.asarray(bq[3:], dtype=float)  # (x, y, z, w) in Warp
        wp_rot = np.asarray(wp.quat_to_matrix(wp.quat(*wp_quat.tolist())), dtype=float).reshape(3, 3)

        our_pos, our_rot = fkmod.fk_numpy(q)
        our_rot = our_rot.numpy() if hasattr(our_rot, "numpy") else np.asarray(our_rot)
        pos_err_mm = float(np.linalg.norm(our_pos - wp_pos) * 1000.0)
        rot_err = _rot_err_deg(our_rot, wp_rot)
        max_pos = max(max_pos, pos_err_mm)
        max_rot = max(max_rot, rot_err)
        if k < 2:
            rows.append({"case": "zero" if k == 0 else "home",
                         "q": [round(float(v), 4) for v in q],
                         "newton_pos_m": [round(float(v), 6) for v in wp_pos],
                         "ours_pos_m": [round(float(v), 6) for v in our_pos],
                         "pos_err_mm": round(pos_err_mm, 6),
                         "rot_err_deg": round(rot_err, 6)})

    zero_pos = np.asarray(rows[0]["newton_pos_m"])
    report = {
        "step": "A1",
        "gate": "G-5 kinematics half",
        "evidence_class": "SIM-POSE",
        "model": str(path.relative_to(ROOT)),
        "flange_body": labels[flange_idx],
        "versions": {
            "python": platform.python_version(),
            "newton": metadata.version("newton"),
            "warp": wp.__version__,
            "numpy": np.__version__,
        },
        "joint_limits": {"lower": lo.tolist(), "upper": hi.tolist(),
                         "match_embedded": limits_match},
        "total_mass_kg": round(float(np.sum(model.body_mass.numpy())), 4),
        "samples": len(samples),
        "max_pos_err_mm": round(max_pos, 6),
        "max_rot_err_deg": round(max_rot, 6),
        "zero_pose_flange_m": [round(float(v), 6) for v in zero_pos],
        "zero_pose_reach_m": round(float(np.linalg.norm(zero_pos)), 6),
        "example_rows": rows,
        "checks": {
            "pos_within_1mm": max_pos <= POS_TOL_MM,
            "rot_within_0p05deg": max_rot <= ROT_TOL_DEG,
            "limits_match_model": limits_match,
        },
        "pending": [
            "Mass/inertia vs the FR3 datasheet within 2% (needs the datasheet as a recorded source).",
            "Franka official URDF/USD vs Menagerie kinematics within 1 mm (G-5 full).",
        ],
    }
    report["result"] = "PASS" if all(report["checks"].values()) else "FAIL"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("result", "max_pos_err_mm", "max_rot_err_deg", "zero_pose_reach_m",
                       "total_mass_kg", "checks")}, indent=2))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
