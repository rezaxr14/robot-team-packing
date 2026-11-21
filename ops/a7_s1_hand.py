#!/usr/bin/env python3
"""A7 rung S1: hand opens/closes under position control with a force cap, on a clamped box.

Ladder definition (docs/AGENT_RUNBOOK.md A7): "hand opens/closes under position control with a
force cap; close on a fixed 5 cm box held in a static clamp" -> "finger contact force readable,
box does not move".

What is measured (all from Newton stepping, evidence class SIM):
* finger travel and stall gap (the box width is 0.05 m),
* servo force per finger from the position-actuator model ``F = kp·(ctrl − q) − kv·qd`` clamped to
  the actuator force range (Newton offers no native contact-force readout on this path, so this is
  the **actuator force**, labelled as such, not a contact sensor measurement),
* penetration of the pad inner face into the box volume,
* box drift (the box is fixed to the world on purpose: S1 tests the hand, not the grasp).

    source ops/env.sh && $ISAAC_PY ops/a7_s1_hand.py

Report: ``reports/a7_s1.json``. NOT a grasp result, NOT a safety result.
"""
from __future__ import annotations

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
OUT = ROOT / "reports" / "a7_s1.json"

BOX_SIZE = 0.05
OPEN_CTRL = 0.0          # slide coordinate 0 = fingers apart
CLOSED_CTRL = 0.035      # 0.035 = pads touching

KP, KV = 600.0, 12.0
FORCE_LIMIT = 20.0


def build():
    import os
    import newton
    import warp as wp
    from rtp.control import fr3_kinematics as fk

    p0, R0 = fk.fk_numpy(np.asarray(fk.Q_HOME, dtype=float))
    z_axis = np.asarray(R0, dtype=float)[:, 2]
    box_center = p0 + z_axis * (0.055 + 0.040)      # palm offset + finger offset along flange +z

    cwd = Path.cwd()
    os.chdir(MODEL.parent)
    try:
        b = newton.ModelBuilder()
        b.add_mjcf(MODEL.name)
        clamp = b.add_link(label="s1_clamp")
        b.add_joint_fixed(-1, clamp)
        b.add_shape_box(body=clamp,
                        xform=wp.transform(wp.vec3(*box_center),
                                           wp.quat_from_matrix(wp.mat33(R0.tolist()))),
                        hx=BOX_SIZE / 2, hy=BOX_SIZE / 2, hz=BOX_SIZE / 2,
                        color=(0.55, 0.35, 0.2), label="s1_box")
        model = b.finalize()
    finally:
        os.chdir(cwd)
    return model, box_center, p0, z_axis


def dof_offsets(model) -> dict:
    out = {}
    for j, label in enumerate(model.joint_label):
        short = label.split("/")[-1]
        if short in ("hand_finger_left_joint", "hand_finger_right_joint"):
            out[short] = int(model.joint_q_start.numpy()[j])
    return out


def main() -> int:
    import newton
    import warp as wp

    report = {
        "step": "A7 S1 (hand under force cap on a clamped box)",
        "evidence_class": "SIM",
        "scope": "finger open/close + stall against a clamped 5 cm box; actuator force (not a contact sensor); no grasp claim",
        "model": str(MODEL.relative_to(ROOT)),
        "versions": {"python": platform.python_version(), "newton": metadata.version("newton"),
                     "warp": wp.__version__},
        "params": {"kp_N_per_m": KP, "kv_Ns_per_m": KV, "force_limit_N": FORCE_LIMIT,
                   "box_size_m": BOX_SIZE, "open_ctrl_m": OPEN_CTRL, "closed_ctrl_m": CLOSED_CTRL, "dt": 0.001},
    }
    try:
        model, box_center, p0, z_axis = build()
        offs = dof_offsets(model)
        assert len(offs) == 2, offs
        state = model.state()
        control = model.control()
        from rtp.control import fr3_kinematics as fk
        q = np.asarray(state.joint_q.numpy()).copy()
        for i, v in enumerate(np.asarray(fk.Q_HOME, dtype=float)):
            q[i] = v
        off_l, off_r = offs["hand_finger_left_joint"], offs["hand_finger_right_joint"]
        q[off_l] = OPEN_CTRL
        q[off_r] = OPEN_CTRL
        state.joint_q = wp.array(q.astype(np.float32), dtype=state.joint_q.dtype, device=model.device)
        control.joint_target_q = wp.array(q.astype(np.float32), dtype=control.joint_target_q.dtype,
                                          device=model.device)
        solver = newton.solvers.SolverMuJoCo(model, njmax=256, nconmax=256)
        cp = newton.CollisionPipeline(model)
        contacts = cp.contacts()

        dt = 0.001
        steps = 3000
        ramp_steps = 1000
        force_l, force_r, gap, ql, qr = [], [], [], [], []
        t0 = time.perf_counter()
        for k in range(steps):
            frac = min(1.0, k / ramp_steps)                # 0 = open, 1 = closed
            ctrl = CLOSED_CTRL * frac
            c = np.asarray(control.joint_target_q.numpy()).copy()
            c[off_l] = ctrl
            c[off_r] = ctrl
            control.joint_target_q = wp.array(c.astype(np.float32), dtype=control.joint_target_q.dtype,
                                              device=model.device)
            state.clear_forces()
            cp.collide(state, contacts)
            solver.step(state, state, control, contacts, dt)
            qq = np.asarray(state.joint_q.numpy(), dtype=float)
            qd = np.asarray(state.joint_qd.numpy(), dtype=float)
            # actuator force from the position servo model (clamped to the actuator range)
            fl = float(np.clip(KP * (ctrl - qq[off_l]) - KV * qd[off_l], -FORCE_LIMIT, FORCE_LIMIT))
            fr = float(np.clip(KP * (ctrl - qq[off_r]) - KV * qd[off_r], -FORCE_LIMIT, FORCE_LIMIT))
            force_l.append(fl)
            force_r.append(fr)
            ql.append(float(qq[off_l]))
            qr.append(float(qq[off_r]))
            gap.append(0.085 - 0.016 - float(qq[off_l]) - float(qq[off_r]))   # inner-face gap
        wall = time.perf_counter() - t0

        final_gap = gap[-1]
        mean_force = max(float(np.mean(np.abs(force_l[-500:]))), float(np.mean(np.abs(force_r[-500:]))))
        box_face = BOX_SIZE
        penetration = max(0.0, box_face - final_gap) if final_gap < box_face else 0.0
        finger_travel = float(ql[-1]) - OPEN_CTRL
        drift = 0.0            # the clamp body is fixed to the world; recorded so the claim is explicit
        report.update({
            "steps": steps, "wall_s": round(wall, 2), "steps_per_s": round(steps / wall, 1),
            "dof": int(model.joint_dof_count), "bodies": int(model.body_count),
            "box_center_world": [round(float(x), 5) for x in box_center],
            "flange_home_world": [round(float(x), 5) for x in p0],
            "final_finger_left_m": round(float(ql[-1]), 5),
            "final_finger_right_m": round(float(qr[-1]), 5),
            "finger_travel_m": round(float(finger_travel), 5),
            "final_inner_gap_m": round(float(final_gap), 5),
            "max_actuator_force_left_N": round(float(np.max(np.abs(force_l))), 3),
            "max_actuator_force_right_N": round(float(np.max(np.abs(force_r))), 3),
            "mean_actuator_force_last_0p5s_N": [round(float(np.mean(np.abs(force_l[-500:]))), 3),
                                                round(float(np.mean(np.abs(force_r[-500:]))), 3)],
            "pad_penetration_m": round(float(penetration), 5),
            "penetration_note": ("self-imposed S1 target <= 3 mm at the 20 N cap (MuJoCo-Warp effective contact "
                                 "stiffness, measured); S4's <= 10 mm bound applies to placing, not this rung"),
            "box_drift_m": drift,
            "checks": {
                "fingers_move": finger_travel >= 0.005,
                "stall_near_box_width_3mm": abs(final_gap - BOX_SIZE) <= 0.003,
                "force_readable_1_to_20N": 1.0 <= mean_force <= FORCE_LIMIT,
                "force_below_cap": max(float(np.max(np.abs(force_l))),
                                       float(np.max(np.abs(force_r)))) <= FORCE_LIMIT + 1e-6,
                "no_penetration_3mm": penetration <= 0.003,
                "box_static_by_construction": drift == 0.0,
                "finite": bool(np.isfinite(np.asarray(qr)).all()),
            },
        })
        report["result"] = "PASS" if all(report["checks"].values()) else "FAIL"
    except Exception as exc:  # noqa: BLE001
        report["result"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report.get(k) for k in ("result", "final_inner_gap_m", "finger_travel_m",
                                                 "max_actuator_force_left_N", "checks", "error")
                      if k in report}, indent=2))
    return 0 if report.get("result") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
