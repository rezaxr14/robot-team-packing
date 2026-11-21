#!/usr/bin/env python3
"""A7 rung S4: place a grasped box into a bag compartment, release, retreat.

This is a FIXTURE script (code prepared, NOT run by the preparing agent: the Newton/GPU
environment is reserved for other jobs).  When invoked later with the merged venv it runs
*one* seeded Newton trial and writes its own report to ``reports/a7_s4.json``::

    source ops/env.sh && $ISAAC_PY ops/a7_s4_place.py --seed 0 --mass 0.3

`evidence_class` is set to ``SIM`` only after the first real ``solver.step`` returns; if the
scene cannot be built or any stage fails, the script still writes an explicit ``result: FAIL``
with a ``reason`` instead of inventing numbers.  A single trial is NOT the >= 90 % / >= 200
trial certification (``configs/arm_cert.yaml``); no pass bound is loosened here.

The grasp/lift reuses the working S2 recipe unchanged (D-78/D-80): derived simplified
parallel-jaw hand, IK targets in the arm-base frame, wrist rolled so the finger closing axis is
world x, and a 12 cm commanded lift so >= 8 cm survives in-grip slip.  The bag is added as a
static floor + two dividers + outer walls; its compartment volumes are axis-aligned boxes.

Pass bounds (docs/AGENT_RUNBOOK.md A7 S4, verbatim): object centre inside one compartment
volume, hand clear, <= 10 mm penetration.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL = ROOT / "assets" / "derived" / "fr3v2_hand.xml"
OUT = ROOT / "reports" / "a7_s4.json"

# --- S2 geometry / IK lessons (unchanged) ---------------------------------------------------
BASE_Z = 0.90                 # arm base sits on the table plane
BOX = 0.05                    # cube edge (m)
BOX_BELOW_FLANGE = 0.095      # palm 0.055 + finger 0.040 (S1/S2)
GRASP_Z = 0.12                # flange height in the ARM-BASE frame at the grasp pose
CLOSED_CTRL = 0.035           # finger slide command that closes the pads on a 5 cm box
LIFT_DELTA = 0.12             # 12 cm command -> >= 8 cm measured gain after slip (D-80)
DT = 0.001
# Wrist rolled so the finger closing axis is world x (D-78); flange local +z points down.
R_TARGET = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])

# --- Bag construction geometry (scene parameters, NOT acceptance thresholds) ----------------
BAG_OFFSET_Y = 0.08           # bag sits +y of the grasp point so the free box starts clear of it
BAG_HALF = (0.12, 0.04)       # bag half-extents in x, y (m); interior fits a 5 cm box
BAG_FLOOR_T = 0.01
BAG_WALL_T = 0.006
BAG_WALL_H = 0.06
N_COMPARTMENTS = 3
TARGET_COMPARTMENT = 1        # middle pocket
PLACE_CLEARANCE = 0.005       # lower the box centre 5 mm above the bag floor before release

# --- Pass bound from the runbook (do not redefine) ------------------------------------------
MAX_PENETRATION_M = 0.010     # S4: <= 10 mm penetration


@dataclass(frozen=True)
class Compartment:
    """Axis-aligned interior volume of one bag pocket (empty space above the floor)."""
    name: str
    center: np.ndarray        # (3,)
    half: np.ndarray          # (3,)


@dataclass(frozen=True)
class Wall:
    """Static box (floor / divider / outer wall) as centre + half-extents."""
    center: np.ndarray
    half: np.ndarray


def make_bag(bag_center, bag_half, n: int = N_COMPARTMENTS, *, wall_t: float = BAG_WALL_T,
             wall_h: float = BAG_WALL_H, floor_t: float = BAG_FLOOR_T):
    """Return ``(walls, compartments)`` for a floor + ``n-1`` dividers + outer walls.

    ``bag_center[2]`` is the *top* of the floor (the interior base).  Compartment volumes
    account for the wall thickness so a centre test cannot be fooled by a point inside a wall.
    Pure geometry: no newton, no torch, no GPU.
    """
    cx, cy, cz = (float(v) for v in bag_center)
    hx, hy = (float(v) for v in bag_half)
    walls = [Wall(np.array([cx, cy, cz - floor_t / 2.0]), np.array([hx, hy, floor_t / 2.0]))]
    cw = 2.0 * hx / n
    comps = []
    for i in range(n):
        x0 = cx - hx + i * cw
        comps.append(Compartment(
            f"compartment_{i}",
            np.array([x0 + cw / 2.0, cy, cz + wall_h / 2.0]),
            np.array([cw / 2.0 - wall_t / 2.0, hy - wall_t, wall_h / 2.0]),
        ))
    for i in range(1, n):
        xd = cx - hx + i * cw
        walls.append(Wall(np.array([xd, cy, cz + wall_h / 2.0]),
                          np.array([wall_t / 2.0, hy, wall_h / 2.0])))
    walls.append(Wall(np.array([cx, cy - hy, cz + wall_h / 2.0]),
                      np.array([hx, wall_t / 2.0, wall_h / 2.0])))
    walls.append(Wall(np.array([cx, cy + hy, cz + wall_h / 2.0]),
                      np.array([hx, wall_t / 2.0, wall_h / 2.0])))
    walls.append(Wall(np.array([cx - hx, cy, cz + wall_h / 2.0]),
                      np.array([wall_t / 2.0, hy, wall_h / 2.0])))
    walls.append(Wall(np.array([cx + hx, cy, cz + wall_h / 2.0]),
                      np.array([wall_t / 2.0, hy, wall_h / 2.0])))
    return walls, comps


def compartment_contains_center(point, compartment: Compartment, margin: float = 0.0) -> bool:
    """True if ``point`` is inside the compartment volume (shrunk by ``margin``)."""
    p = np.asarray(point, dtype=float)
    return bool(np.all(np.abs(p - compartment.center) <= compartment.half - margin))


def box_bag_penetration(box_center, box_half, walls) -> float:
    """Maximum AABB overlap depth (m) between the box and any static bag wall (>= 0)."""
    bc = np.asarray(box_center, dtype=float)
    bh = np.asarray(box_half, dtype=float)
    pen = 0.0
    for w in walls:
        d = np.abs(bc - w.center) - (bh + np.asarray(w.half, dtype=float))
        if np.all(d < 0.0):
            pen = max(pen, float(np.min(-d)))
    return pen


def build(seed: int, mass: float):
    """Build the S4 scene and return the model plus the scripted IK waypoints."""
    import os
    import newton                       # lazy: CPU tests import this module without newton
    import warp as wp
    from rtp.control import fr3_kinematics as fk
    from rtp.control.scripted_ik import solve_ik

    rng = np.random.default_rng(seed)
    phome, _ = fk.fk_numpy(np.asarray(fk.Q_HOME, dtype=float))
    gx, gy = float(phome[0]), float(phome[1])
    x = gx + rng.uniform(-0.005, 0.005)
    y = gy + rng.uniform(-0.005, 0.005)
    zc = BASE_Z + GRASP_Z - BOX_BELOW_FLANGE
    bag_center = (gx, gy + BAG_OFFSET_Y, BASE_Z)
    walls, comps = make_bag(bag_center, BAG_HALF)
    comp = comps[TARGET_COMPARTMENT]

    cwd = Path.cwd()
    os.chdir(MODEL.parent)
    try:
        b = newton.ModelBuilder()
        b.add_mjcf(MODEL.name, xform=wp.transform(wp.vec3(0.0, 0.0, BASE_Z), wp.quat_identity()))
        tbl = b.add_link(label="s4_table")
        b.add_joint_fixed(-1, tbl)
        b.add_shape_box(body=tbl,
                        xform=wp.transform(wp.vec3(0.0, 0.0, BASE_Z - 0.02), wp.quat_identity()),
                        hx=0.9, hy=0.6, hz=0.02, color=(0.24, 0.26, 0.30), label="s4_table_top")
        for k, w in enumerate(walls):
            wl = b.add_link(label=f"s4_bag_wall_{k}")
            b.add_joint_fixed(-1, wl)
            b.add_shape_box(body=wl,
                            xform=wp.transform(wp.vec3(*w.center), wp.quat_identity()),
                            hx=float(w.half[0]), hy=float(w.half[1]), hz=float(w.half[2]),
                            color=(0.35, 0.20, 0.12), label=f"s4_bag_wall_shape_{k}")
        item = b.add_body(xform=wp.transform(wp.vec3(x, y, zc), wp.quat_identity()),
                          mass=mass, label="s4_item")
        b.add_shape_box(body=item, hx=BOX / 2, hy=BOX / 2, hz=BOX / 2,
                        color=(0.70, 0.45, 0.25), label="s4_item_box")
        model = b.finalize()
    finally:
        os.chdir(cwd)

    def ik(tx, ty, tz, q_start):
        r = solve_ik(q_start, (float(tx), float(ty), float(tz)), R_TARGET,
                     max_iters=250, pos_tol=1e-3, rot_tol=5e-3)
        if not r.converged:
            raise RuntimeError(f"IK failed at ({tx:.3f},{ty:.3f},{tz:.3f}): "
                               f"pos {r.pos_error_m:.4f}, rot {r.rot_error_rad:.4f}")
        return r.path

    # Arm-base-frame flange height that puts the hanging box centre just above the bag floor.
    flange_z_place = BOX_BELOW_FLANGE + BOX / 2 + PLACE_CLEARANCE
    q_pre = ik(x, y, GRASP_Z + 0.10, np.asarray(fk.Q_HOME, float))
    q_grasp = ik(x, y, GRASP_Z, q_pre[-1])
    q_lift = ik(x, y, GRASP_Z + LIFT_DELTA, q_grasp[-1])
    q_transport = ik(comp.center[0], comp.center[1], GRASP_Z + LIFT_DELTA, q_lift[-1])
    q_place = ik(comp.center[0], comp.center[1], flange_z_place, q_transport[-1])
    q_retreat = ik(comp.center[0], comp.center[1], GRASP_Z + LIFT_DELTA + 0.06, q_place[-1])
    return (model, (x, y, zc), q_pre, q_grasp, q_lift, q_transport, q_place, q_retreat,
            walls, comps, comp)


def main() -> int:
    import newton
    import warp as wp
    from rtp.control import fr3_kinematics as fk

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mass", type=float, default=0.3)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()

    report = {
        "step": "A7 S4 (place into bag compartment, release, retreat)",
        "evidence_class": "NOT_RUN",   # flipped to SIM only after a real solver.step returns
        "scope": "one seeded Newton place attempt with the derived simplified parallel-jaw hand; "
                 "NOT a certification and NOT a safety result",
        "seed": a.seed, "mass_kg": a.mass,
        "hand_model": str(MODEL.relative_to(ROOT)),
        "pass_bounds": {"max_penetration_m": MAX_PENETRATION_M,
                        "source": "docs/AGENT_RUNBOOK.md A7 S4 (unchanged)"},
        "bag": {"offset_y_m": BAG_OFFSET_Y, "half_xy_m": list(BAG_HALF),
                "floor_t_m": BAG_FLOOR_T, "wall_t_m": BAG_WALL_T, "wall_h_m": BAG_WALL_H,
                "n_compartments": N_COMPARTMENTS, "target_compartment": TARGET_COMPARTMENT},
        "versions": {"python": platform.python_version(), "newton": metadata.version("newton"),
                     "warp": wp.__version__},
        "result": "FAIL", "reason": "not run",
    }
    stepped = [False]

    try:
        (model, box0, q_pre, q_grasp, q_lift, q_transport, q_place, q_retreat,
         walls, comps, comp) = build(a.seed, a.mass)
        state, control = model.state(), model.control()
        labels = [lbl.split("/")[-1] for lbl in model.joint_label]
        offsets = [int(model.joint_q_start.numpy()[i]) for i, lbl in enumerate(labels)
                   if lbl in ("hand_finger_left_joint", "hand_finger_right_joint")]
        assert len(offsets) == 2, offsets
        fl, fr = offsets
        q = np.asarray(state.joint_q.numpy()).astype(float).copy()
        q[:7] = np.asarray(fk.Q_HOME, float)
        q[fl] = q[fr] = 0.0                        # fingers open
        state.joint_q = wp.array(q.astype(np.float32), dtype=state.joint_q.dtype, device=model.device)
        control.joint_target_q = wp.array(q.astype(np.float32), dtype=control.joint_target_q.dtype,
                                          device=model.device)
        solver = newton.solvers.SolverMuJoCo(model, njmax=512, nconmax=512)
        cp = newton.CollisionPipeline(model)
        contacts = cp.contacts()
        item_body = int(np.flatnonzero(["s4_item" in lbl for lbl in model.body_label])[0])

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
                if not stepped[0]:
                    stepped[0] = True
                    report["evidence_class"] = "SIM"

        def ramp(path, finger_ctrl, n):
            for k in range(n):
                u = min(1.0, (k + 1) / n)
                idx = u * (len(path) - 1)
                i = min(int(idx), len(path) - 2)
                arm = path[i] + (idx - i) * (path[i + 1] - path[i])
                set_cmd(arm, finger_ctrl)
                step(1)

        t0 = time.perf_counter()
        step(200)                                   # settle
        ramp(q_pre, 0.0, 1000)                      # approach
        ramp(q_grasp, 0.0, 600)                     # descend
        ramp(q_grasp, CLOSED_CTRL, 500)             # close
        step(500)                                   # establish grasp
        ramp(q_lift, CLOSED_CTRL, 800)              # slip-compensated lift (D-80)
        step(300)
        ramp(q_transport, CLOSED_CTRL, 800)         # carry over the compartment
        step(200)
        ramp(q_place, CLOSED_CTRL, 600)             # lower into the pocket
        step(200)
        ramp(q_place, 0.0, 300)                     # open fingers -> release
        step(300)                                   # let the box settle
        ramp(q_retreat, 0.0, 600)                   # retreat
        step(300)
        wall = time.perf_counter() - t0
        bq = np.asarray(state.body_q.numpy())
        box_final = bq[item_body, :3]
        box_half = np.array([BOX / 2, BOX / 2, BOX / 2])
        contained_in = [c.name for c in comps if compartment_contains_center(box_final, c)]
        contained = bool(contained_in)
        pen = box_bag_penetration(box_final, box_half, walls)
        qq = np.asarray(state.joint_q.numpy())
        inner_gap = 0.085 - float(qq[fl]) - float(qq[fr])   # S1 pad-inner-face gap
        flange_z_local = float(fk.fk_numpy(qq[:7])[0][2])
        compartment_top_local = float(BAG_WALL_H)
        hand_clear = bool(inner_gap >= BOX and flange_z_local > compartment_top_local and contained)
        steps = 200 + 1000 + 600 + 500 + 500 + 800 + 300 + 800 + 200 + 600 + 200 + 300 + 300 + 600 + 300
        report.update({
            "box_center_start": [round(float(v), 5) for v in box0],
            "box_center_final": [round(float(v), 5) for v in box_final],
            "contained_in_compartments": contained_in,
            "penetration_m": round(pen, 6),
            "finger_inner_gap_m": round(inner_gap, 5),
            "flange_z_local_final_m": round(flange_z_local, 5),
            "hand_clear_definition": ("fingers open (inner gap >= box edge), flange above compartment "
                                      "rim, and object centre inside a compartment"),
            "hand_clear": hand_clear,
            "wall_s": round(wall, 2), "steps": steps, "steps_per_s": round(steps / wall, 1),
            "checks": {
                "finite": bool(np.isfinite(bq).all()),
                "object_inside_a_compartment": contained,
                "hand_clear": hand_clear,
                "penetration_le_10mm": pen <= MAX_PENETRATION_M,
            },
        })
        report["result"] = "PASS" if all(report["checks"].values()) else "FAIL"
        if report["result"] == "FAIL":
            report["reason"] = "one or more S4 checks false (see checks); one seed only"
        else:
            report["reason"] = "one seeded trial passed; >= 90% / >= 200 trials still required"
    except Exception as exc:  # noqa: BLE001
        report["result"] = "FAIL"
        report["reason"] = f"{type(exc).__name__}: {str(exc)[:500]}"

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report.get(k) for k in ("result", "evidence_class", "contained_in_compartments",
                                                 "penetration_m", "hand_clear", "checks", "reason")
                      if k in report}, indent=2))
    return 0 if report.get("result") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
