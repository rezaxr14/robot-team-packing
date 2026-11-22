#!/usr/bin/env python3
"""A7 rung S3 trial: S2 with a friction x grasp-force grid (one trial per invocation).

Pass (runbook A7): box height gain >= 8 cm at the end of the hold in >= 95 % of 50 seeds with
masses 0.1-1 kg. This script runs one seed per invocation (a shell loop or a follow-up driver
aggregates the seeds); every number written to the report comes from Newton stepping.

Evidence class: SIM. This is a physical grasp attempt, NOT the SIM-POSE stand-in, and NOT a
certification until the full seeded suite passes. The hand is the derived simplified parallel-jaw
gripper (not the Franka Hand).

    source ops/env.sh && $ISAAC_PY ops/a7_s2_lift.py --seed 0 --mass 0.3
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
OUT = ROOT / "reports" / "a7_s2.json"

BASE_Z = 0.90                # arm base sits on the table plane
BOX = 0.05                    # cube edge (m)
GRASP_XY = None               # set from the home flange pose (reachable with flange z down)
CLOSED_CTRL = 0.035
DT = 0.001


def build(seed: int, mass: float, friction: float, force_factor: float, lift_delta: float = 0.12):
    import os
    import newton
    import warp as wp
    from rtp.control import fr3_kinematics as fk
    from rtp.control.scripted_ik import solve_ik

    rng = np.random.default_rng(seed)
    phome, Rhome = fk.fk_numpy(np.asarray(fk.Q_HOME, float))
    # Rotate the wrist about its approach axis so the finger closing direction is world x;
    # at Q_HOME it is the 45-degree diagonal, which makes a 50 mm cube 70 mm wide to the pads.
    Rtarget = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    # The home flange already points its local +z down. Keep that orientation so the hand hangs
    # toward the table; target a point near the home flange (verified reachable by DLS IK).
    gx, gy = (phome[0], phome[1]) if GRASP_XY is None else GRASP_XY
    x, y = gx + rng.uniform(-0.005, 0.005), gy + rng.uniform(-0.005, 0.005)
    grasp_z = 0.12                             # flange height in the arm-base frame (IK is local)
    zc = BASE_Z + grasp_z - 0.095              # box centre sits at the grasp point (world)
    table_top = BASE_Z

    cwd = Path.cwd()
    os.chdir(MODEL.parent)
    try:
        b = newton.ModelBuilder()
        b.add_mjcf(MODEL.name, xform=wp.transform(wp.vec3(0.0, 0.0, BASE_Z), wp.quat_identity()))
        # table slab (static)
        tbl = b.add_link(label="s2_table")
        b.add_joint_fixed(-1, tbl)
        b.add_shape_box(body=tbl, xform=wp.transform(wp.vec3(0.0, 0.0, table_top - 0.02), wp.quat_identity()),
                        hx=0.9, hy=0.6, hz=0.02, color=(0.24, 0.26, 0.30), label="s2_table_top")
        # free box to lift
        item = b.add_body(xform=wp.transform(wp.vec3(x, y, zc), wp.quat_identity()),
                          mass=mass, label="s2_item")
        b.add_shape_box(body=item, hx=BOX / 2, hy=BOX / 2, hz=BOX / 2,
                        cfg=newton.ModelBuilder.ShapeConfig(mu=friction),
                        color=(0.70, 0.45, 0.25), label="s3_item_box")
        model = b.finalize()
        # The derived hand's pad material is 1.2 and MuJoCo combines geom friction with a
        # max-like rule, so the object's mu was previously ignored (measured: a 0.1 kg box at
        # mu=0.1 stayed on 2*0.1*1.5=0.3 N of friction).  Set the pad/palm mu to the tested
        # value so both surfaces of the contact pair carry the object's friction; this makes
        # the contact model test the property the Brain reasons about.  Recorded as a
        # one-setting contact-model fix in reports/a7_grasp_diagnosis.json.
        mu_obj = float(friction)
        shape_labels = list(model.shape_label)
        pads = [i for i, lbl in enumerate(shape_labels)
                if "finger" in lbl and "pad" in lbl or lbl.endswith("hand_palm_collision")]
        if pads:
            mu_arr = np.asarray(model.shape_material_mu.numpy(), dtype=float).copy()
            for i in pads:
                mu_arr[i] = mu_obj
            model.shape_material_mu = wp.array(mu_arr.astype(np.float32),
                                               dtype=model.shape_material_mu.dtype,
                                               device=model.device)
        # Grasp force is the finger servo stiffness times its position error. Scaling the two
        # finger target gains by `force_factor` scales the commanded force by the same factor
        # (still clamped by the MJCF actuator force range).
        ke = np.asarray(model.joint_target_ke.numpy(), dtype=float).copy()
        for off in (7, 8):
            ke[off] *= force_factor
        model.joint_target_ke = wp.array(ke.astype(np.float32), dtype=model.joint_target_ke.dtype,
                                          device=model.device)
    finally:
        os.chdir(cwd)

    def ik(z_flange: float, q_start):
        r = solve_ik(q_start, (float(x), float(y), z_flange), Rtarget, max_iters=250, pos_tol=1e-3, rot_tol=5e-3)
        if not r.converged:
            raise RuntimeError(f"IK failed at z={z_flange}: pos {r.pos_error_m:.4f}, rot {r.rot_error_rad:.4f}")
        return r.path

    q_pre = ik(grasp_z + 0.10, fk.Q_HOME)          # 10 cm above the grasp pose
    q_grasp = ik(grasp_z, q_pre[-1])               # pads straddle the box
    q_lift = ik(grasp_z + lift_delta, q_grasp[-1])  # slip-compensated lift target (D-80)
    return model, (x, y, zc), q_pre, q_grasp, q_lift, Rhome


def main() -> int:
    import newton
    import warp as wp
    from rtp.control import fr3_kinematics as fk

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mass", type=float, default=0.3)
    ap.add_argument("--friction", type=float, default=0.6)
    ap.add_argument("--force-factor", type=float, default=1.0)
    ap.add_argument("--lift-delta", type=float, default=0.12)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "a7_s3.json")
    a = ap.parse_args()

    report = {
        "step": "A7 S2 (free-box grasp and lift)",
        "evidence_class": "SIM",
        "scope": "one seeded Newton grasp attempt with the derived simplified parallel-jaw hand; "
                 "not a certification and not a safety result",
        "seed": a.seed, "mass_kg": a.mass, "friction": a.friction,
        "force_factor": a.force_factor, "lift_delta_m": a.lift_delta,
        "contact_model": "box, finger pads and palm all carry the tested mu (derived hand had 1.2 pad mu, "
                         "which dominated the contact and hid the object friction)",
        "hand_model": str(MODEL.relative_to(ROOT)),
        "versions": {"python": platform.python_version(), "newton": metadata.version("newton"),
                     "warp": wp.__version__},
    }
    try:
        model, box0, q_pre, q_grasp, q_lift, Rhome = build(a.seed, a.mass, a.friction, a.force_factor, a.lift_delta)
        state, control = model.state(), model.control()
        labels = [lbl.split("/")[-1] for lbl in model.joint_label]
        offsets = [int(model.joint_q_start.numpy()[i]) for i, lbl in enumerate(labels)
                   if lbl in ("hand_finger_left_joint", "hand_finger_right_joint")]
        assert len(offsets) == 2, offsets
        fl, fr = offsets
        q = np.asarray(state.joint_q.numpy()).astype(float).copy()
        q[:7] = np.asarray(fk.Q_HOME, float)
        q[fl] = q[fr] = 0.0                       # fingers open
        state.joint_q = wp.array(q.astype(np.float32), dtype=state.joint_q.dtype, device=model.device)
        control.joint_target_q = wp.array(q.astype(np.float32), dtype=control.joint_target_q.dtype,
                                          device=model.device)
        finger_ids = [int(np.flatnonzero([lbl.split("/")[-1] == name for lbl in model.body_label])[0])
                      for name in ("hand_finger_left", "hand_finger_right")]
        solver = newton.solvers.SolverMuJoCo(model, njmax=512, nconmax=512)
        cp = newton.CollisionPipeline(model)
        contacts = cp.contacts()
        item_body = int(np.flatnonzero(["s2_item" in lbl for lbl in model.body_label])[0])

        def set_cmd(arm_q, finger_ctrl):
            c = np.asarray(control.joint_target_q.numpy()).astype(float).copy()
            c[:7] = np.asarray(arm_q, float)
            c[fl] = c[fr] = float(finger_ctrl)
            control.joint_target_q = wp.array(c.astype(np.float32), dtype=control.joint_target_q.dtype,
                                              device=model.device)

        def step(n):
            for _ in range(n):
                state.clear_forces()
                cp.collide(state, contacts)
                solver.step(state, state, control, contacts, DT)

        def ramp(path, finger_ctrl, n):
            for k in range(n):
                u = min(1.0, (k + 1) / n)
                idx = u * (len(path) - 1)
                i = min(int(idx), len(path) - 2)
                arm = path[i] + (idx - i) * (path[i + 1] - path[i])
                set_cmd(arm, finger_ctrl)
                step(1)

        t0 = time.perf_counter()
        step(200)                                    # settle on the table
        z_start = float(np.asarray(state.body_q.numpy())[item_body, 2])
        ramp(q_pre, 0.0, 1000)                       # approach 10 cm above
        z_after_pre = float(np.asarray(state.body_q.numpy())[item_body, 2])
        ramp(q_grasp, 0.0, 600)                      # descend to the box
        z_after_descend = float(np.asarray(state.body_q.numpy())[item_body, 2])
        bqd = np.asarray(state.body_q.numpy())
        qact = np.asarray(state.joint_q.numpy())[:7]
        fl_actual = fk.fk_numpy(qact)[0]
        report["debug_at_grasp"] = {
            "q_actual": [round(float(v), 5) for v in np.asarray(state.joint_q.numpy())],
            "q_cmd": [round(float(v), 5) for v in q_grasp[-1]],
            "flange_actual": [round(float(v), 5) for v in fl_actual],
            "flange_cmd": [round(float(v), 5) for v in fk.fk_numpy(q_grasp[-1])[0]],
            "fingers_world": [[round(float(v), 5) for v in bqd[i, :3]] for i in finger_ids],
            "box_world": [round(float(v), 5) for v in bqd[item_body, :3]],
            "finger_q": [float(np.asarray(state.joint_q.numpy())[i]) for i in offsets],
        }
        # Finger squeeze force estimate from the position servo: kp_eff * (ctrl - q).
        # The solver's rigid_contact_force is not populated on this build, so this is a
        # model-based estimate of the pad normal force, not a contact-sensor reading.
        kp_eff = [float(np.asarray(model.joint_target_ke.numpy())[i]) for i in (fl, fr)]
        n_peak, n_trace = 0.0, []

        def pad_force_est() -> float:
            qq = np.asarray(state.joint_q.numpy())
            return float(np.mean([kp_eff[j] * (CLOSED_CTRL - float(qq[off]))
                                  for j, off in enumerate((fl, fr))]))

        ramp(q_grasp, CLOSED_CTRL, 500)              # close the fingers
        for _ in range(500):                         # establish the grasp, track the peak
            step(1)
            n_peak = max(n_peak, abs(pad_force_est()))
            if len(n_trace) < 500 and _ % 50 == 0:
                n_trace.append(round(pad_force_est(), 3))
        z_grasped = float(np.asarray(state.body_q.numpy())[item_body, 2])
        ramp(q_lift, CLOSED_CTRL, 800)               # lift 10 cm
        step(1000)                                   # hold 1 s
        z_hold = float(np.asarray(state.body_q.numpy())[item_body, 2])
        wall = time.perf_counter() - t0
        bq = np.asarray(state.body_q.numpy())
        gain = z_hold - z_start
        report.update({
            "box_center_start": [round(float(v), 5) for v in box0],
            "box_z_start": round(z_start, 5),
            "box_z_after_pre": round(z_after_pre, 5),
            "box_z_after_descend": round(z_after_descend, 5),
            "box_z_after_close": round(z_grasped, 5),
            "box_z_after_hold": round(z_hold, 5),
            "height_gain_m": round(gain, 5),
            "box_xy_drift_m": round(float(np.linalg.norm(bq[item_body, :2] - np.asarray(box0[:2]))), 5),
            "pad_force_est_N": round(pad_force_est(), 4),
            "pad_force_peak_est_N": round(n_peak, 4),
            "pad_force_trace_N": n_trace,
            "pad_force_model": "kp_eff*(ctrl-q) averaged over the two finger servos; "
                               "model estimate, not a contact sensor",
            "wall_s": round(wall, 2), "steps": 5600, "steps_per_s": round(5600 / wall, 1),
            "checks": {
                "finite": bool(np.isfinite(bq).all()),
                "height_gain_ge_8cm": gain >= 0.08,
                "box_xy_drift_le_2cm": float(np.linalg.norm(bq[item_body, :2] - np.asarray(box0[:2]))) <= 0.02,
                "not_fallen": z_hold > BASE_Z - 0.02,
            },
        })
        report["result"] = "PASS" if all(report["checks"].values()) else "FAIL"
    except Exception as exc:  # noqa: BLE001
        report["result"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report.get(k) for k in ("result", "height_gain_m", "box_z_after_hold",
                                                 "box_xy_drift_m", "steps_per_s", "checks", "error")
                      if k in report}, indent=2))
    return 0 if report.get("result") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
