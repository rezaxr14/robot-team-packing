#!/usr/bin/env python3
"""U065 E1/E2: elbow-push direction diagnosis on the Newton FR3 (SIM, diagnostic).

Run with the Newton venv (never the training venv)::

    source ops/env.sh && $ISAAC_PY ops/a5_elbow_directions.py --out reports/a5_elbow_directions.json

Purpose
-------
The A5 ``elbow_push`` / ``payload`` / ``latency_error`` rows push a *link*
(``fr3v2_link4`` = elbow, ``fr3v2_link6`` = forearm) with a world-frame force
and hand the supervisor the dynamically consistent flange wrench
``F_task = Jbar^T J_b^T F``.  This report measures every link x direction
(+-x, +-y, +-z) x force (10/20/30 N) cell so the row failure can be attributed
to a physical direction instead of a single 6-row aggregate.

E1: the told (injected) wrench path, plus an independent MODEL cross-check with
MuJoCo (Menagerie FR3 at Q_HOME) of ``|F_task|`` for five cells, plus the
``fr3v2_link4 +y`` cells with ``residual_comp_gain=0``.
E2: the same cells with ``f_ext`` from the inverse-dynamics estimator
(``Harness.tick(use_estimate=True)``), which is what hardware has.

Per cell: told ``F_task`` vector and norm, expected ``|F_task|/K_lin``,
measured max EE error during the push, max link displacement, post-recovery EE
error (final sample and the last 200 ms mean), and the state / clamp counters.
No controller, tolerance, clamp or scoring change: this is a diagnostic.

Evidence class: **SIM** (Newton stepping) plus **MODEL** (MuJoCo kinematics /
mass matrix, closed-form mapping).  NOT a safety certification (D-22).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ops"))

from rtp.control.compliance import PROFILES                      # noqa: E402
from rtp.control.newton_arm import ELBOW_BODY, FOREARM_BODY, Q_HOME  # noqa: E402

OUT = ROOT / "reports" / "a5_elbow_directions.json"
LINKS = (("fr3_link4", ELBOW_BODY), ("fr3_link6", FOREARM_BODY))
AXES = (0, 1, 2)
FORCES = (10.0, 20.0, 30.0)

#: Independent MODEL targets quoted in U065 (MuJoCo Menagerie FR3 at Q_HOME,
#: 10 N push): (link, "+x"|..., N).
MODEL_TARGETS = (
    ("fr3_link4", "+x", 9.09),
    ("fr3_link4", "+y", 0.39),
    ("fr3_link6", "+x", 10.0),
    ("fr3_link6", "+y", 8.23),
    ("fr3_link6", "+z", 10.0),
)


def dir_name(axis: int, sign: int) -> str:
    return f"{'+' if sign > 0 else '-'}{'xyz'[axis]}"


# --------------------------------------------------------------------------- #
# Newton cells
# --------------------------------------------------------------------------- #
def elbow_cell(h, body_id: int, body_name: str, axis: int, sign: int, force: float,
               *, use_estimate: bool = False, noise_n: float = 0.0,
               push_s: float = 1.0, recover_s: float = 3.0) -> dict:
    """One push cell; mirrors ``a5_newton_push.elbow_push`` but keeps the direction."""
    h.profile = "grasp"
    h.arm.cfg.profile = "grasp"
    h.reset()
    f_body = np.zeros(6)
    f_body[axis] = sign * force
    f_task = h.task_wrench_from_body_dynamic(f_body, body_id)
    k_lin = float(PROFILES["grasp"][0])
    n_push = int(round(push_s / h.dt))
    n_rec = int(round(recover_s / h.dt))
    link0 = np.asarray(h.state0.body_q.numpy())[body_id, :3].copy()

    ee_err, link_disp, states = [], [], []
    clamp_ticks = rate_ticks = 0
    fhat = []
    t0 = time.time()
    for k in range(n_push + n_rec):
        f_ext = f_task if k < n_push else np.zeros(6)
        body_f = f_body if k < n_push else np.zeros(6)
        rec = h.tick(f_ext, None, body_id, body_wrench=body_f,
                     use_estimate=use_estimate, noise_n=noise_n)
        ee_err.append(float(np.linalg.norm(rec["pos"] - h.p0)))
        link_disp.append(float(np.linalg.norm(rec["body_pos"][body_id] - link0)))
        states.append(rec["state"].value)
        info = rec["info"]
        clamp_ticks += int(bool(info.get("torque_clamped")))
        rate_ticks += int(bool(info.get("rate_limited")))
        if k >= n_push and rec.get("wrench_hat") is not None:
            fhat.append(rec["wrench_hat"][:3].copy())
    wall = time.time() - t0
    n_tail = int(round(0.2 / h.dt))
    f_task_n = float(np.linalg.norm(f_task[:3]))
    return {
        "body": body_name, "body_id": body_id,
        "direction": dir_name(axis, sign), "force_N": force,
        "profile": "grasp", "k_lin_N_per_m": k_lin,
        "mapping": "dynamic", "use_estimate": bool(use_estimate),
        "noise_N": noise_n if use_estimate else 0.0,
        "f_task_N": [round(float(x), 6) for x in f_task],
        "f_task_norm_N": round(f_task_n, 6),
        "expected_deflection_m": round(f_task_n / k_lin, 6),
        "max_ee_error_during_push_m": round(float(np.max(ee_err[:n_push])), 6),
        "max_link_displacement_m": round(float(np.max(link_disp[:n_push])), 6),
        "post_recovery_ee_error_m": round(float(ee_err[-1]), 6),
        "post_recovery_ee_error_tail_mean_m": round(float(np.mean(ee_err[-n_tail:])), 6),
        "ee_error_over_expected": round(float(np.max(ee_err[:n_push])) / (f_task_n / k_lin), 4),
        "states_seen": sorted(set(states)),
        "torque_clamped_ticks": clamp_ticks,
        "rate_limited_ticks": rate_ticks,
        "estimator_force_tail_mean_N": ([round(float(x), 4) for x in np.mean(fhat, axis=0)]
                                        if fhat else None),
        "wall_s": round(wall, 1),
    }


# --------------------------------------------------------------------------- #
# Independent MODEL cross-check (MuJoCo, closed-form mapping, no Newton)
# --------------------------------------------------------------------------- #
def mujoco_mapping() -> dict:
    """``F_task = Jbar^T J_b^T F`` at Q_HOME, with MuJoCo's own M and Jacobians.

    Two Jacobian conventions are reported because they differ by the body-frame
    origin vs the body centre of mass.  ``body_origin`` is the convention the
    U065 target list reproduces; Newton's ``eval_jacobian`` is COM-referenced
    (verified numerically below, see ``jacobian_convention``).
    """
    import os

    import mujoco

    xml = ROOT / "assets" / "cache" / "menagerie" / "franka_fr3_v2" / "fr3v2.xml"
    cwd = Path.cwd()
    os.chdir(xml.parent)
    try:
        m = mujoco.MjModel.from_xml_path(xml.name)
    finally:
        os.chdir(cwd)
    d = mujoco.MjData(m)
    d.qpos[:] = np.asarray(Q_HOME, dtype=float)
    mujoco.mj_forward(m, d)

    def jac(bid: int, com: bool) -> np.ndarray:
        J = np.zeros((6, m.nv))
        (mujoco.mj_jacBodyCom if com else mujoco.mj_jacBody)(m, d, J[:3], J[3:], bid)
        return J

    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, M)
    Minv = np.linalg.inv(0.5 * (M + M.T))
    flange = m.body("fr3v2_link8").id
    out = {}
    for conv, com in (("body_origin", False), ("com", True)):
        Jf = jac(flange, com)
        A = Jf @ Minv @ Jf.T
        jbar_t = np.linalg.pinv(A, rcond=1e-6) @ Jf @ Minv
        cells = {}
        for link, body_id in (("fr3_link4", m.body("fr3v2_link4").id),
                              ("fr3_link6", m.body("fr3v2_link6").id)):
            Jb = jac(body_id, com)
            for axis in AXES:
                for sign in (-1, 1):
                    F = np.zeros(6)
                    F[axis] = 10.0 * sign
                    task = jbar_t @ (Jb.T @ F)
                    cells[f"{link}/{dir_name(axis, sign)}"] = {
                        "norm_N": round(float(np.linalg.norm(task[:3])), 4),
                        "vector_N": [round(float(x), 4) for x in task[:3]],
                    }
        out[conv] = cells
    return {
        "evidence_class": "MODEL",
        "model": str(xml.relative_to(ROOT)),
        "q": list(Q_HOME), "pinv_rcond": 1e-6,
        "note": "closed-form dynamic task wrench; no Newton, no contact, no controller",
        "cells": out,
    }


def newton_mapping(h) -> dict:
    """Newton's own mapping at Q_HOME (no stepping), for the same cells."""
    cells = {}
    for link, body_id in LINKS:
        for axis in AXES:
            for sign in (-1, 1):
                F = np.zeros(6)
                F[axis] = 10.0 * sign
                task = h.task_wrench_from_body_dynamic(F, body_id)
                cells[f"{link}/{dir_name(axis, sign)}"] = {
                    "norm_N": round(float(np.linalg.norm(task[:3])), 4),
                    "vector_N": [round(float(x), 4) for x in task[:3]],
                }
    return cells


def jacobian_convention() -> dict:
    """Compare Newton's ``eval_jacobian`` body rows with MuJoCo origin/COM rows."""
    import os

    import mujoco
    import newton
    import warp as wp

    xml = ROOT / "assets" / "cache" / "menagerie" / "franka_fr3_v2" / "fr3v2.xml"
    b = newton.ModelBuilder()
    cwd = Path.cwd()
    os.chdir(xml.parent)
    try:
        b.add_mjcf(xml.name)
    finally:
        os.chdir(cwd)
    model = b.finalize()
    s0 = model.state()
    s0.joint_q = wp.array(np.asarray(Q_HOME, dtype=np.float32), dtype=s0.joint_q.dtype)
    s0.joint_qd = wp.zeros_like(s0.joint_qd)
    newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
    Jall = np.asarray(newton.eval_jacobian(model, s0).numpy())[0]

    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    d.qpos[:] = np.asarray(Q_HOME, dtype=float)
    mujoco.mj_forward(m, d)

    out = {}
    for link, nid in (("fr3_link4", ELBOW_BODY), ("fr3_link6", FOREARM_BODY)):
        mid = m.body("fr3v2_" + link.replace("fr3_", "")).id
        Jn = Jall[nid * 6:(nid + 1) * 6, :7]
        res = {}
        for conv, com in (("body_origin", False), ("com", True)):
            J = np.zeros((6, m.nv))
            (mujoco.mj_jacBodyCom if com else mujoco.mj_jacBody)(m, d, J[:3], J[3:], mid)
            res[conv] = round(float(np.max(np.abs(J - Jn))), 9)
        out[link] = {"max_abs_diff": res,
                     "matches": min(res, key=res.get)}
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--mode", choices=("told", "estimate", "both"), default="both")
    ap.add_argument("--noise", type=float, default=0.5,
                    help="estimator noise sigma N (the *_estimated steady row uses 0.5)")
    ap.add_argument("--forces", type=float, nargs="*", default=list(FORCES))
    ap.add_argument("--links", nargs="*", default=[n for n, _ in LINKS])
    ap.add_argument("--dirs", nargs="*", default=None,
                    help="subset like '+x -y'; default all six")
    ap.add_argument("--gain0", action="store_true",
                    help="run fr3_link4 +y cells with residual_comp_gain=0")
    ap.add_argument("--push-s", type=float, default=1.0)
    ap.add_argument("--recover-s", type=float, default=3.0)
    ap.add_argument("--skip-model", action="store_true")
    args = ap.parse_args()

    from a5_newton_push import Harness, mass_eigen_range_all, versions  # noqa: E402

    dirs = []
    for axis in AXES:
        for sign in (-1, 1):
            name = dir_name(axis, sign)
            if args.dirs is None or name in args.dirs:
                dirs.append((axis, sign, name))
    links = [(n, i) for n, i in LINKS if n in args.links]
    if not dirs or not links:
        print("nothing to run", file=sys.stderr)
        return 2

    report = {
        "step": "U065-E1/E2",
        "title": "elbow push direction diagnosis (diagnostic only)",
        "evidence_class": "SIM",
        "diagnostic_only": True,
        "controller_change": "none for E1/E2; residual_comp_gain=0 only in the --gain0 cells",
        "seed_plant": 20261120,
        "versions": versions(),
        "params": {"profile": "grasp", "k_lin_N_per_m": PROFILES["grasp"][0],
                   "mapping": "dynamic", "push_s": args.push_s,
                   "recover_s": args.recover_s, "mode": args.mode,
                   "estimator_noise_N": args.noise,
                   "residual_comp_gain": 4.0,
                   "determinism": "outcome-level; each cell run once, no re-rolls; not bit-exact"},
        "cells": {"told": [], "estimate": [], "gain0": []},
        "model_crosscheck": None,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        report["mass_matrix_eigenvalues"] = mass_eigen_range_all()
        out.write_text(json.dumps(report, indent=1) + "\n")

    # Cheap sections first: the mapping cross-check and Newton's own mapping.
    if not args.skip_model:
        print("MODEL cross-check (MuJoCo)...", flush=True)
        report["model_crosscheck"] = {
            "mujoco": mujoco_mapping(),
            "newton_mapping_10N": None,
            "jacobian_convention": jacobian_convention(),
        }

    modes = ["told", "estimate"] if args.mode == "both" else [args.mode]
    t_all = time.time()
    if report["model_crosscheck"] is not None:
        hm = Harness()
        report["model_crosscheck"]["newton_mapping_10N"] = newton_mapping(hm)
        del hm
    save()

    for mode in modes:
        h = Harness(residual_comp_gain=4.0)
        for link, body_id in links:
            for force in args.forces:
                for axis, sign, name in dirs:
                    t0 = time.time()
                    cell = elbow_cell(h, body_id, link, axis, sign, force,
                                     use_estimate=(mode == "estimate"),
                                     noise_n=args.noise if mode == "estimate" else 0.0,
                                     push_s=args.push_s, recover_s=args.recover_s)
                    report["cells"][mode].append(cell)
                    save()
                    print(f"[{mode}] {link} {name} {force:4.0f} N  "
                          f"|F_task|={cell['f_task_norm_N']:8.4f} N  "
                          f"exp={cell['expected_deflection_m']*1e3:7.3f} mm  "
                          f"ee={cell['max_ee_error_during_push_m']*1e3:7.3f} mm  "
                          f"link={cell['max_link_displacement_m']*1e3:7.3f} mm  "
                          f"post={cell['post_recovery_ee_error_m']*1e3:6.3f} mm  "
                          f"({time.time()-t0:.0f} s)", flush=True)
        del h

    if args.gain0:
        h0 = Harness(residual_comp_gain=0.0)
        body_id = ELBOW_BODY
        for force in args.forces:
            t0 = time.time()
            cell = elbow_cell(h0, body_id, "fr3_link4", 1, 1, force,
                              use_estimate=False, push_s=args.push_s,
                              recover_s=args.recover_s)
            cell["residual_comp_gain"] = 0.0
            cell["note"] = ("physical response without the residual-wrench equilibrium "
                            "forcing (told wrench still drives the supervisor)")
            report["cells"]["gain0"].append(cell)
            save()
            print(f"[gain0] fr3_link4 +y {force:4.0f} N  "
                  f"|F_task|={cell['f_task_norm_N']:8.4f} N  "
                  f"expected={cell['expected_deflection_m']*1e3:7.3f} mm  "
                  f"ee={cell['max_ee_error_during_push_m']*1e3:7.3f} mm  "
                  f"link={cell['max_link_displacement_m']*1e3:7.3f} mm  "
                  f"post={cell['post_recovery_ee_error_m']*1e3:6.3f} mm  "
                  f"({time.time()-t0:.0f} s)", flush=True)
        del h0

    # ---- target verdicts -------------------------------------------------- #
    if report["model_crosscheck"] is not None:
        mc = report["model_crosscheck"]
        verdict = []
        for link, dname, target in MODEL_TARGETS:
            key = f"{link}/{dname}"
            org = mc["mujoco"]["cells"]["body_origin"][key]["norm_N"]
            com = mc["mujoco"]["cells"]["com"][key]["norm_N"]
            nwt = mc["newton_mapping_10N"][key]["norm_N"]
            verdict.append({
                "cell": key, "model_target_N": target,
                "mujoco_body_origin_N": org, "mujoco_com_N": com,
                "newton_N": nwt,
                "target_vs_mujoco_origin_pct": round(100.0 * abs(org - target) / target, 2),
                "newton_vs_target_pct": round(100.0 * abs(nwt - target) / target, 2),
                "mujoco_origin_within_5pct": bool(abs(org - target) / target <= 0.05),
                "newton_within_5pct": bool(abs(nwt - target) / target <= 0.05),
            })
        mc["targets"] = verdict
        mc["all_mujoco_origin_within_5pct"] = bool(
            all(v["mujoco_origin_within_5pct"] for v in verdict))
        mc["all_newton_within_5pct"] = bool(all(v["newton_within_5pct"] for v in verdict))

    report["wall_s"] = round(time.time() - t_all, 1)
    save()
    print(f"wrote {out}  ({report['wall_s']} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
