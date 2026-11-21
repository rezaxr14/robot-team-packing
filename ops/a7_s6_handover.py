#!/usr/bin/env python3
"""A7 rung S6: two-arm handover and push_aside (one seeded Newton trial per invocation).

FIXTURE script (code prepared, NOT run by the preparing agent; the Newton/GPU environment is
reserved for other jobs).  Invoked later with the merged venv::

    source ops/env.sh && $ISAAC_PY ops/a7_s6_handover.py --seed 0 --mass 0.3 --variant handover
    source ops/env.sh && $ISAAC_PY ops/a7_s6_handover.py --seed 0 --variant push_aside

It builds two FR3-with-hand arms from the derived MJCF (D-76) at x = +-0.35 (D-65), reuses the
S2 grasp/lift recipe (arm-base-frame IK, wrist rolled so the closing axis is world x, 12 cm
slip-compensated lift), and writes ``reports/a7_s6.json``.  ``evidence_class: SIM`` is written
only after a real solver step returns; any build/IK/transition failure writes ``result: FAIL``
with a ``reason``.  These are single trials, not the >= 90 % / >= 200 trial certification.

Pass bounds (docs/AGENT_RUNBOOK.md A7 S6, verbatim): object transferred without ground contact;
blocker moved >= 5 cm without toppling.  Operationalisations (measurement definitions, not
loosened bounds) are documented in the report.  No bound is redefined here.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL = ROOT / "assets" / "derived" / "fr3v2_hand.xml"
OUT = ROOT / "reports" / "a7_s6.json"

# S2 geometry / IK lessons (unchanged)
BASE_Z = 0.90
BOX = 0.05
BOX_BELOW_FLANGE = 0.095
GRASP_Z = 0.12
CLOSED_CTRL = 0.035
LIFT_DELTA = 0.12
DT = 0.001
R_TARGET = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])

ARM_X = 0.35                       # arm bases on the island centre line (D-65)
R_HOME = (0.0, 0.785, 0.0, -2.356, 0.0, 1.571, -0.785)   # mirrored home (a6_scene)

# Pass bounds from the runbook (do not redefine).
MIN_BLOCKER_DISPLACEMENT_M = 0.05  # S6 push_aside: >= 5 cm
MAX_TOPPLE_DEG = 45.0              # operationalisation of "without toppling" (body +z tilt)


class HandoverStateMachine:
    """Pure bookkeeping of the L -> R handover; no physics, no GPU.

    Valid path: L_HOLD --R_GRASPED--> BOTH --L_RELEASED--> R_HOLD --R_LIFTED--> DONE.
    Any other event in a state raises ``ValueError`` (invalid transition).
    """
    TRANSITIONS = {
        ("L_HOLD", "R_GRASPED"): "BOTH",
        ("BOTH", "L_RELEASED"): "R_HOLD",
        ("R_HOLD", "R_LIFTED"): "DONE",
    }

    def __init__(self) -> None:
        self.state = "L_HOLD"
        self.ground_contact = False
        self.history: list[str] = ["L_HOLD"]

    def step(self, event: str) -> str:
        key = (self.state, event)
        if key not in self.TRANSITIONS:
            raise ValueError(f"invalid handover transition {key!r} from history {self.history}")
        self.state = self.TRANSITIONS[key]
        self.history.append(self.state)
        return self.state

    def note_ground_contact(self) -> None:
        self.ground_contact = True

    def transferred_without_ground_contact(self) -> bool:
        return self.state == "DONE" and not self.ground_contact


def tilt_angle_deg(quat_wxyz) -> float:
    """Tilt of a body's local +z axis from world +z, in degrees (pure numpy)."""
    w, x, y, z = (float(v) for v in np.asarray(quat_wxyz, dtype=float).reshape(4))
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return float("nan")
    x, y = x / n, y / n
    uz = 1.0 - 2.0 * (x * x + y * y)            # R(q) @ [0,0,1] dot [0,0,1]
    return float(math.degrees(math.acos(float(np.clip(uz, -1.0, 1.0)))))


def box_below_clearance(box_center, table_top: float, box_half: float,
                        clearance: float = 0.005) -> bool:
    """Ground-contact proxy: box bottom within ``clearance`` of the table top."""
    c = np.asarray(box_center, dtype=float)
    return bool((c[2] - box_half) <= (table_top + clearance))


def _arm_offsets(model):
    """Return (arm_q_start[2], left_finger_q_start[2], right_finger_q_start[2])."""
    labels = list(model.joint_label)
    starts = model.joint_q_start.numpy()

    def starts_for(key):
        return [int(starts[j]) for j, lbl in enumerate(labels) if key in lbl]

    arms = starts_for("fr3v2_joint1")
    left = starts_for("hand_finger_left_joint")
    right = starts_for("hand_finger_right_joint")
    if not (len(arms) == len(left) == len(right) == 2):
        raise RuntimeError(f"expected two arms and two fingers each, got arms={arms} left={left} right={right}")
    return arms, left, right


def build(seed: int, mass: float):
    """Build the two-arm scene and return model + scripted waypoints + body ids."""
    import os
    import newton                       # lazy: CPU tests import this module without newton
    import warp as wp
    from rtp.control import fr3_kinematics as fk

    rng = np.random.default_rng(seed)
    pL, _ = fk.fk_numpy(np.asarray(fk.Q_HOME, float))
    pR, _ = fk.fk_numpy(np.asarray(R_HOME, float))
    gx, gy = float(pL[0]), float(pL[1])
    x = gx + rng.uniform(-0.005, 0.005)
    y = gy + rng.uniform(-0.005, 0.005)
    box_start_world = np.array([-ARM_X + x, y, BASE_Z + GRASP_Z - BOX_BELOW_FLANGE])
    hy = float((pL[1] + pR[1]) / 2.0)
    handover_box_world = np.array([0.0, hy, BASE_Z + GRASP_Z + LIFT_DELTA - BOX_BELOW_FLANGE])
    handover_flange_world = handover_box_world + np.array([0.0, 0.0, BOX_BELOW_FLANGE])
    blocker_y = y - 0.10                       # keep the blocker clear of the transfer object
    blocker_world = np.array([-ARM_X + x, blocker_y, BASE_Z + BOX / 2])

    def to_local(world, arm_index):
        base_x = -ARM_X if arm_index == 0 else ARM_X
        return (float(world[0] - base_x), float(world[1]), float(world[2] - BASE_Z))

    cwd = Path.cwd()
    os.chdir(MODEL.parent)
    try:
        b = newton.ModelBuilder()
        for bx in (-ARM_X, ARM_X):
            b.add_mjcf(MODEL.name,
                       xform=wp.transform(wp.vec3(bx, 0.0, BASE_Z), wp.quat_identity()))
        tbl = b.add_link(label="s6_table")
        b.add_joint_fixed(-1, tbl)
        b.add_shape_box(body=tbl,
                        xform=wp.transform(wp.vec3(0.0, 0.0, BASE_Z - 0.02), wp.quat_identity()),
                        hx=0.9, hy=0.6, hz=0.02, color=(0.24, 0.26, 0.30), label="s6_table_top")
        obj = b.add_body(xform=wp.transform(wp.vec3(*box_start_world), wp.quat_identity()),
                         mass=mass, label="s6_object")
        b.add_shape_box(body=obj, hx=BOX / 2, hy=BOX / 2, hz=BOX / 2,
                        color=(0.70, 0.45, 0.25), label="s6_object_box")
        blk = b.add_body(xform=wp.transform(wp.vec3(*blocker_world), wp.quat_identity()),
                         mass=0.3, label="s6_blocker")
        b.add_shape_box(body=blk, hx=BOX / 2, hy=BOX / 2, hz=BOX / 2,
                        color=(0.30, 0.45, 0.60), label="s6_blocker_box")
        model = b.finalize()
    finally:
        os.chdir(cwd)

    obj_id = int(np.flatnonzero([lbl.split("/")[-1] == "s6_object" for lbl in model.body_label])[0])
    blk_id = int(np.flatnonzero([lbl.split("/")[-1] == "s6_blocker" for lbl in model.body_label])[0])
    arms, left, right = _arm_offsets(model)
    return (model, arms, left, right, obj_id, blk_id, box_start_world,
            blocker_world, handover_flange_world, to_local)


def _make_ik(fk, solve_ik, R_TARGET_local):
    def ik(q0, target_local):
        r = solve_ik(q0, tuple(float(v) for v in target_local), R_TARGET_local,
                     max_iters=250, pos_tol=5e-3, rot_tol=1e-2)
        if not r.converged:
            raise RuntimeError(f"IK failed for {target_local}: pos {r.pos_error_m:.4f}, "
                               f"rot {r.rot_error_rad:.4f}")
        return r.path
    return ik


def main() -> int:
    import newton
    import warp as wp
    from rtp.control import fr3_kinematics as fk
    from rtp.control.scripted_ik import solve_ik

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mass", type=float, default=0.3)
    ap.add_argument("--variant", choices=("handover", "push_aside"), default="handover")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()

    report = {
        "step": f"A7 S6 ({a.variant})",
        "evidence_class": "NOT_RUN",   # flipped to SIM only after a real solver.step returns
        "scope": "one seeded two-arm Newton trial with the derived simplified parallel-jaw hands; "
                 "NOT a certification and NOT a safety result",
        "seed": a.seed, "mass_kg": a.mass, "variant": a.variant,
        "hand_model": str(MODEL.relative_to(ROOT)),
        "pass_bounds": {"min_blocker_displacement_m": MIN_BLOCKER_DISPLACEMENT_M,
                        "max_topple_deg": MAX_TOPPLE_DEG,
                        "source": "docs/AGENT_RUNBOOK.md A7 S6 (unchanged; topple angle is an "
                                  "operationalisation, not a config bound)"},
        "versions": {"python": platform.python_version(), "newton": metadata.version("newton"),
                     "warp": wp.__version__},
        "result": "FAIL", "reason": "not run",
    }
    stepped = [False]

    try:
        (model, arms, left, right, obj_id, blk_id, box_start_world, blocker_world,
         handover_flange_world, to_local) = build(a.seed, a.mass)
        state, control = model.state(), model.control()
        q = np.asarray(state.joint_q.numpy()).astype(float).copy()
        for ai in range(2):
            q[arms[ai]:arms[ai] + 7] = np.asarray(fk.Q_HOME if ai == 0 else R_HOME, float)
            q[left[ai]] = q[right[ai]] = 0.0
        state.joint_q = wp.array(q.astype(np.float32), dtype=state.joint_q.dtype, device=model.device)
        control.joint_target_q = wp.array(q.astype(np.float32), dtype=control.joint_target_q.dtype,
                                          device=model.device)
        solver = newton.solvers.SolverMuJoCo(model, njmax=512, nconmax=512)
        cp = newton.CollisionPipeline(model)
        contacts = cp.contacts()
        ik = _make_ik(fk, solve_ik, R_TARGET)

        def set_cmd(ai, arm_q, finger_ctrl):
            c = np.asarray(control.joint_target_q.numpy()).astype(float).copy()
            c[arms[ai]:arms[ai] + 7] = np.asarray(arm_q, float)
            c[left[ai]] = c[right[ai]] = float(finger_ctrl)
            control.joint_target_q = wp.array(c.astype(np.float32), dtype=control.joint_target_q.dtype,
                                              device=model.device)

        def step(n, track=None):
            for _ in range(n):
                state.clear_forces()
                cp.collide(state, contacts)
                solver.step(state, state, control, contacts, DT)
                if not stepped[0]:
                    stepped[0] = True
                    report["evidence_class"] = "SIM"
                if track is not None:
                    bq = np.asarray(state.body_q.numpy())
                    track.append(bq[obj_id, :3].copy())

        def ramp(ai, path, finger_ctrl, n, track=None):
            for k in range(n):
                u = min(1.0, (k + 1) / n)
                idx = u * (len(path) - 1)
                i = min(int(idx), len(path) - 2)
                arm = path[i] + (idx - i) * (path[i + 1] - path[i])
                set_cmd(ai, arm, finger_ctrl)
                step(1, track)

        t0 = time.perf_counter()
        # The grasp target is expressed in the LEFT arm's local (base) frame for both variants.
        gx, gy = (float(v) for v in fk.fk_numpy(np.asarray(fk.Q_HOME, float))[0][:2])
        if a.variant == "handover":
            sm = HandoverStateMachine()
            track = []
            gx, gy = float(fk.fk_numpy(np.asarray(fk.Q_HOME, float))[0][0]), \
                     float(fk.fk_numpy(np.asarray(fk.Q_HOME, float))[0][1])
            # --- L: approach, grasp, lift (S2 recipe) ---------------------------------------
            pre = ik(np.asarray(fk.Q_HOME, float), (gx, gy, GRASP_Z + 0.10))
            grasp = ik(pre[-1], (gx, gy, GRASP_Z))
            lift = ik(grasp[-1], (gx, gy, GRASP_Z + LIFT_DELTA))
            step(200)
            ramp(0, pre, 0.0, 800, track)
            ramp(0, grasp, 0.0, 500, track)
            ramp(0, grasp, CLOSED_CTRL, 400, track)
            step(300, track)                                    # L grasped
            ramp(0, lift, CLOSED_CTRL, 700, track)
            step(200, track)
            # --- L: carry the box to the handover point --------------------------------------
            carry = ik(lift[-1], to_local(handover_flange_world, 0))
            ramp(0, carry, CLOSED_CTRL, 800, track)
            step(200, track)
            # --- R: approach the same point from above, open ----------------------------------
            r_home = np.asarray(R_HOME, float)
            r_above = np.array(to_local(handover_flange_world + np.array([0.0, 0.0, 0.25]), 1))
            r_hover = np.array(to_local(handover_flange_world + np.array([0.0, 0.0, 0.01]), 1))
            r_pre = ik(r_home, tuple(r_above))
            r_down = ik(r_pre[-1], tuple(r_hover))
            phase_box = {}
            phase_box["after_L_carry"] = np.asarray(state.body_q.numpy())[obj_id, :3].copy()
            ramp(1, r_pre, 0.0, 800, track)
            phase_box["after_R_pre"] = np.asarray(state.body_q.numpy())[obj_id, :3].copy()
            ramp(1, r_down, 0.0, 400, track)
            phase_box["after_R_down"] = np.asarray(state.body_q.numpy())[obj_id, :3].copy()
            ramp(1, r_down, CLOSED_CTRL, 300, track)
            phase_box["after_R_close"] = np.asarray(state.body_q.numpy())[obj_id, :3].copy()            # R closes on the box
            step(200, track)
            # Did R acquire the box? geometric proxy: box centre near R flange - 0.095 z.
            bq = np.asarray(state.body_q.numpy())
            r_flange_world = handover_flange_world.copy()        # commanded, local->world known
            box_near_r = bool(np.linalg.norm(bq[obj_id, :3] - (r_flange_world - np.array([0, 0, BOX_BELOW_FLANGE])))
                              <= 0.04)
            report["phase_box_positions"] = {k: [round(float(v), 5) for v in val]
                                             for k, val in phase_box.items()}
            report["box_z_trace_every_200"] = [round(float(v[2]), 4) for v in track[::200]]
            report["r_acquire_proxy"] = {
                "box_world": [round(float(v), 5) for v in bq[obj_id, :3]],
                "expected_world": [round(float(v), 5) for v in (r_flange_world - np.array([0, 0, BOX_BELOW_FLANGE]))],
                "distance_m": round(float(np.linalg.norm(bq[obj_id, :3] -
                                      (r_flange_world - np.array([0, 0, BOX_BELOW_FLANGE])))), 5),
            }
            if not box_near_r:
                raise RuntimeError("R did not acquire the box (geometry proxy failed); see r_acquire_proxy")
            sm.step("R_GRASPED")
            # --- L: open and retreat ----------------------------------------------------------
            ramp(0, carry, 0.0, 300, track)
            sm.step("L_RELEASED")
            ramp(0, lift, 0.0, 500, track)                       # L lifts away
            # --- R: lift and retreat ----------------------------------------------------------
            r_lift = ik(r_down[-1], tuple(np.array(to_local(handover_flange_world, 1))
                                          + np.array([0.0, 0.0, 0.10])))
            ramp(1, r_lift, CLOSED_CTRL, 600, track)
            step(300, track)
            sm.step("R_LIFTED")
            for c in track:
                if box_below_clearance(c, BASE_Z, BOX / 2):
                    sm.note_ground_contact()
                    break
            wall = time.perf_counter() - t0
            nsteps = (200 + 800 + 500 + 400 + 300 + 700 + 200 + 800 + 200 + 800 + 400 + 300 + 200
                      + 300 + 500 + 600 + 300)
            report.update({
                "handover_states": sm.history,
                "final_state": sm.state,
                "ground_contact_observed": sm.ground_contact,
                "min_box_z_after_lift_m": round(float(np.min(np.asarray(track)[:, 2])), 5),
                "box_start_world": [round(float(v), 5) for v in box_start_world],
                "box_final_world": [round(float(v), 5) for v in bq[obj_id, :3]],
                "ground_contact_definition": "box bottom within 5 mm of table top after lift",
                "wall_s": round(wall, 2), "steps": nsteps, "steps_per_s": round(nsteps / wall, 1),
                "checks": {
                    "finite": bool(np.isfinite(np.asarray(state.body_q.numpy())).all()),
                    "state_machine_done": sm.state == "DONE",
                    "transferred_without_ground_contact": sm.transferred_without_ground_contact(),
                },
            })
        else:  # push_aside
            track = []
            # Fingers closed -> solid pusher face. Start just left of the blocker, push +x.
            blocker_y = float(blocker_world[1])
            start_local = (gx - 0.035, blocker_y, GRASP_Z)
            end_local = (gx + 0.065, blocker_y, GRASP_Z)
            pre = ik(np.asarray(fk.Q_HOME, float), (start_local[0], start_local[1], GRASP_Z + 0.10))
            approach = ik(pre[-1], start_local)
            push = ik(approach[-1], end_local)
            step(200)
            ramp(0, pre, CLOSED_CTRL, 800)
            ramp(0, approach, CLOSED_CTRL, 500)
            bq0 = np.asarray(state.body_q.numpy())
            blocker_start = bq0[blk_id, :3].copy()
            ramp(0, push, CLOSED_CTRL, 900, track)
            step(300, track)
            bq1 = np.asarray(state.body_q.numpy())
            blocker_end = bq1[blk_id, :3].copy()
            displacement = float(np.linalg.norm(blocker_end[:2] - blocker_start[:2]))
            quat = bq1[blk_id, 3:7]                      # Newton body_q layout: pos(3) + quat(4)
            tilt = tilt_angle_deg(quat)
            toppled = tilt > MAX_TOPPLE_DEG
            wall = time.perf_counter() - t0
            nsteps = 200 + 800 + 500 + 900 + 300
            report.update({
                "blocker_start_world": [round(float(v), 5) for v in blocker_start],
                "blocker_end_world": [round(float(v), 5) for v in blocker_end],
                "blocker_displacement_m": round(displacement, 5),
                "blocker_tilt_deg": round(tilt, 3),
                "toppled": bool(toppled),
                "topple_definition": f"body local +z tilt from world +z > {MAX_TOPPLE_DEG} deg",
                "wall_s": round(wall, 2), "steps": nsteps, "steps_per_s": round(nsteps / wall, 1),
                "checks": {
                    "finite": bool(np.isfinite(bq1).all() and np.isfinite(tilt)),
                    "blocker_moved_ge_5cm": displacement >= MIN_BLOCKER_DISPLACEMENT_M,
                    "not_toppled": not toppled,
                },
            })
        report["result"] = "PASS" if all(report["checks"].values()) else "FAIL"
        if report["result"] == "FAIL":
            report["reason"] = "one or more S6 checks false / geometry proxy failed (see checks); one seed only"
        else:
            report["reason"] = "one seeded trial passed; >= 90% / >= 200 trials still required"
    except Exception as exc:  # noqa: BLE001
        report["result"] = "FAIL"
        report["reason"] = f"{type(exc).__name__}: {str(exc)[:500]}"

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report.get(k) for k in ("result", "evidence_class", "variant", "final_state",
                                                 "ground_contact_observed", "blocker_displacement_m",
                                                 "blocker_tilt_deg", "checks", "reason")
                      if k in report}, indent=2))
    return 0 if report.get("result") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
