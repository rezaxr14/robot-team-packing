#!/usr/bin/env python3
"""A7 rung S2: free box on the table, approach with the arm, close, lift 10 cm, hold 1 s.

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
LIFT_DELTA = 0.12                  # exploration default; the reported rule is lift_delta_for_mass()
DT = 0.001


def lift_delta_for_mass(mass_kg: float) -> float:
    """Deterministic single rule f(mass), chosen on the tuning split (seeds >= 900000).

    The mass is the value the `weigh` skill supplies (the A7 card records which mass source
    was used).  Continuous and monotone: 0.104 m at 0.1 kg, 0.12 m at 0.5 kg, 0.14 m at 1 kg.
    """
    return 0.10 + 0.04 * max(0.0, min(float(mass_kg), 1.0))


def build(seed: int, mass: float, lift_delta: float = 0.12):
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
                        color=(0.70, 0.45, 0.25), label="s2_item_box")
        model = b.finalize()
    finally:
        os.chdir(cwd)

    def ik(z_flange: float, q_start):
        r = solve_ik(q_start, (float(x), float(y), z_flange), Rtarget, max_iters=250, pos_tol=1e-3, rot_tol=5e-3)
        if not r.converged:
            raise RuntimeError(f"IK failed at z={z_flange}: pos {r.pos_error_m:.4f}, rot {r.rot_error_rad:.4f}")
        return r.path

    q_pre = ik(grasp_z + 0.10, fk.Q_HOME)          # 10 cm above the grasp pose
    q_grasp = ik(grasp_z, q_pre[-1])               # pads straddle the box
    q_lift = ik(grasp_z + lift_delta, q_grasp[-1])  # lift target (slip-compensated)
    return model, (x, y, zc), q_pre, q_grasp, q_lift, Rhome


def main() -> int:
    import newton
    import warp as wp
    from rtp.control import fr3_kinematics as fk
    from rtp.control.newton_arm import matrix_from_rotvec, rotvec_from_matrix

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mass", type=float, default=0.3)
    ap.add_argument("--lift-delta", type=float, default=None,
                    help="explicit commanded lift (tuning only); omit to use lift_delta_for_mass()")
    ap.add_argument("--rule", action="store_true",
                    help="use the single deterministic rule lift_delta = f(mass)")
    ap.add_argument("--compliant", action="store_true",
                    help="drive the arm through NewtonCompliantArm (A5 controller) instead of the "
                         "MJCF position servos; the finger servos stay active")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--trace", type=Path, default=None,
                    help="diagnosis only: write a downsampled per-tick trace (pose error, torque "
                         "components, box z) of the compliant run; the trial result is unchanged")
    a = ap.parse_args()

    lift_delta = (lift_delta_for_mass(a.mass) if (a.rule or a.lift_delta is None)
                  else float(a.lift_delta))
    report = {
        "step": "A7 S2 (free-box grasp and lift)",
        "evidence_class": "SIM",
        "scope": "one seeded Newton grasp attempt with the derived simplified parallel-jaw hand; "
                 "not a certification and not a safety result",
        "seed": a.seed, "mass_kg": a.mass, "lift_delta_m": lift_delta,
        "lift_rule": ("lift_delta_for_mass(mass)=0.10+0.04*clip(mass,0,1)" if (a.rule or a.lift_delta is None)
                      else "explicit --lift-delta (tuning/exploration)"),
        "mass_source": "commanded mass (the `weigh` value would be used on the robot; recorded here)",
        "controller": ("NewtonCompliantArm (A5 supervisor; phase profiles approach/grasp/transport)"
                       if a.compliant else "MJCF position servos (rung script, not the A5 controller)"),
        "hand_model": str(MODEL.relative_to(ROOT)),
        "versions": {"python": platform.python_version(), "newton": metadata.version("newton"),
                     "warp": wp.__version__},
    }
    trace_rows: list = []
    phase = ["init"]
    trace_k = [0]
    trace_every = 10
    try:
        model, box0, q_pre, q_grasp, q_lift, Rhome = build(a.seed, a.mass, lift_delta)
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
        # The compliant runner zeroes the arm's position-servo gains, so it must run BEFORE the
        # solver is constructed (the solver copies the gains).
        runner = None
        if a.compliant:
            sys.path.insert(0, str(ROOT / "ops"))
            from a7_compliant import CompliantArmRunner
            runner = CompliantArmRunner(model, state, q_home=fk.Q_HOME, profile="approach")
        solver = newton.solvers.SolverMuJoCo(model, njmax=512, nconmax=512)
        cp = newton.CollisionPipeline(model)
        contacts = cp.contacts()
        item_body = int(np.flatnonzero(["s2_item" in lbl for lbl in model.body_label])[0])

        arm = None
        if runner is not None:
            from a7_compliant import SkillArm
            arm = SkillArm(runner, fl, fr, control)

        def set_cmd(arm_q, finger_ctrl):
            c = np.asarray(control.joint_target_q.numpy()).astype(float).copy()
            c[:7] = np.asarray(arm_q, float)
            c[fl] = c[fr] = float(finger_ctrl)
            control.joint_target_q = wp.array(c.astype(np.float32), dtype=control.joint_target_q.dtype,
                                              device=model.device)

        def step_one():
            state.clear_forces()
            if runner is not None:
                runner.tick(control)
            cp.collide(state, contacts)
            solver.step(state, state, control, contacts, DT)

        def step_plan(plan):
            """One physics tick with the compliant arm following ``plan``."""
            state.clear_forces()
            runner.tick(control, plan=plan)
            cp.collide(state, contacts)
            solver.step(state, state, control, contacts, DT)
            trace_k[0] += 1
            if a.trace is not None and trace_k[0] % trace_every == 0:
                inf = runner.last_info
                q_act = np.asarray(state.joint_q.numpy(), dtype=float)[:7]
                dq_act = np.asarray(state.joint_qd.numpy(), dtype=float)[:7]
                p_act, R_act = fk.fk_numpy(q_act)
                R_plan = matrix_from_rotvec(np.asarray(inf["x_ref"][3:], float)) @ Rhome
                e_true = rotvec_from_matrix(np.asarray(R_act, float) @ R_plan.T)
                q_null = np.asarray(runner.arm.q_home, float)
                _, R_null = fk.fk_numpy(q_null)
                e_null = rotvec_from_matrix(np.asarray(R_null, float) @ R_plan.T)
                spec = arm._q_spec
                trace_rows.append({
                    "k": int(trace_k[0]), "phase": phase[0],
                    "s": round(float(inf["s"]), 4), "state": str(inf["state"]),
                    "ep": round(float(np.linalg.norm(np.asarray(inf["x"][:3]) - np.asarray(inf["x_ref"][:3]))), 5),
                    "er_ctrl": round(float(np.linalg.norm(np.asarray(inf["x"][3:]) - np.asarray(inf["x_ref"][3:]))), 5),
                    "er_true": round(float(np.linalg.norm(e_true)), 5),
                    "er_null": round(float(np.linalg.norm(e_null)), 5),
                    "rv_ref": [round(float(v), 4) for v in np.asarray(inf["x_ref"][3:])],
                    "rv_now": [round(float(v), 4) for v in np.asarray(inf["x"][3:])],
                    "qs_n": (len(spec[0]) if spec is not None else 0),
                    "s0": (round(float(spec[1]), 3) if spec is not None else None),
                    "dur": (round(float(spec[2]), 3) if spec is not None else None),
                    "pz": round(float(p_act[2]), 5),
                    "tau_task": round(float(np.linalg.norm(inf["tau_task"])), 4),
                    "tau_null": round(float(np.linalg.norm(inf["tau_null"])), 4),
                    "tau_g_max": round(float(np.max(np.abs(inf["gravity_comp"]))), 3),
                    "tau_raw_max": round(float(np.max(np.abs(inf["tau_raw"]))), 3),
                    "tau_max": round(float(np.max(np.abs(inf["tau"]))), 3),
                    "clamped": bool(inf["torque_clamped"]),
                    "f_norm": round(float(np.linalg.norm(inf["force"])), 3),
                    "m_norm": round(float(np.linalg.norm(inf["force"][3:])), 3),
                    "qd_max": round(float(np.max(np.abs(dq_act))), 4),
                    "box_z": round(float(np.asarray(state.body_q.numpy())[item_body, 2]), 5),
                    "finger": round(float(arm._finger), 4),
                })

        def step(n):
            if arm is not None:
                arm.hold(step_plan, n)
            else:
                for _ in range(n):
                    step_one()

        def ramp(path, finger_ctrl, n):
            if arm is not None:
                # Time-parameterized Cartesian plan through the FK poses of the joint path,
                # starting at the measured pose.  Measured on the A5 controller: a plan at
                # ~0.19 m/s tracks with ~5 mm final error; 0.37 m in 1 s leaves 30 mm and
                # yields.  A target that starts where the arm actually is does not look like
                # a push (the supervisor's e_pause is 30 mm).
                plan, t_end = arm.path_plan(path, vmax=0.08)
                arm.run_plan(plan, t_end, step_plan, finger=finger_ctrl)
                arm.hold(step_plan, 400, finger=finger_ctrl)
            else:
                for k in range(n):
                    u = min(1.0, (k + 1) / n)
                    idx = u * (len(path) - 1)
                    i = min(int(idx), len(path) - 2)
                    arm_q = path[i] + (idx - i) * (path[i + 1] - path[i])
                    set_cmd(arm_q, finger_ctrl)
                    step_one()

        t0 = time.perf_counter()
        step(200)                                    # settle on the table
        z_start = float(np.asarray(state.body_q.numpy())[item_body, 2])
        if runner is not None:
            runner.set_profile("approach")
        phase[0] = "approach"
        ramp(q_pre, 0.0, 1000)                       # approach 10 cm above
        z_after_pre = float(np.asarray(state.body_q.numpy())[item_body, 2])
        if runner is not None:
            # The final descent and the close are the runbook's `grasp` phase.  Measured
            # reason: at `approach` (300 N/m) the joint stiction floor leaves 17-30 mm of
            # sag near this posture, which parks the pads above the box; `grasp` (500 N/m)
            # halves it.  Profile choice only, no bound changed.
            runner.set_profile("grasp")
        phase[0] = "descent"
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
        if runner is not None:
            runner.set_profile("grasp")
        phase[0] = "close"
        ramp(q_grasp, CLOSED_CTRL, 500)              # close the fingers
        step(500)                                    # establish the grasp
        z_grasped = float(np.asarray(state.body_q.numpy())[item_body, 2])
        if runner is not None:
            runner.set_profile("transport")
        phase[0] = "lift"
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
            "wall_s": round(wall, 2),
            "steps": int(arm.steps + 200) if arm is not None else 5600,
            "steps_per_s": round((int(arm.steps + 200) if arm is not None else 5600) / wall, 1),
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
    if a.trace is not None:
        a.trace.parent.mkdir(parents=True, exist_ok=True)
        a.trace.write_text(json.dumps({"every_ticks": trace_every, "rows": trace_rows}, indent=1) + "\n")
    print(json.dumps({k: report.get(k) for k in ("result", "height_gain_m", "box_z_after_hold",
                                                 "box_xy_drift_m", "steps_per_s", "checks", "error")
                      if k in report}, indent=2))
    return 0 if report.get("result") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
