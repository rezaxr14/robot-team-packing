#!/usr/bin/env python3
"""A4 / gate G-7: external wrench injection with Newton solver substeps.

Run with the Newton venv::

    source ops/env.sh && $ISAAC_PY ops/g7_wrench.py

Two paths are measured and kept separate:

1. **Newton direct (SIM)**: a free-floating box, gravity off, force applied to
   ``state.body_f`` on *every* solver substep; check ``|dp - F t| / (F t) <= 2 %``
   for substeps = 1 and 4.  This proves the direct API delivers the full impulse.
2. **Isaac Lab composer path (SIM, wrapper contract)**: the
   ``permanent_wrench_composer`` on the verified A2 USD articulation, using
   ``write_data_to_sim()`` once per frame.  The Isaac Lab checkout contains the
   backport `5608601` ("Apply Newton external wrenches on every solver substep",
   PR #8269); this path checks that a single command still delivers the full
   impulse with 4 substeps.

Report: ``reports/g7_wrench.json``. No safety claim; no real robot.
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
OUT = ROOT / "reports" / "g7_wrench.json"


def versions() -> dict:
    out = {"python": platform.python_version()}
    for name in ("newton", "warp-lang", "isaaclab", "isaaclab-newton", "torch", "numpy"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "MISSING"
    return out


def newton_box_impulse(force_n: float, substeps: int, duration_s: float,
                       dt_frame: float, force_every_substep: bool) -> dict:
    import newton
    import warp as wp

    b = newton.ModelBuilder()
    b.gravity = (0.0, 0.0, 0.0)
    body = b.add_body()
    b.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1)
    model = b.finalize()
    s0, s1 = model.state(), model.state()
    control = model.control()
    solver = newton.solvers.SolverMuJoCo(model)
    cp = newton.CollisionPipeline(model)
    contacts = cp.contacts()

    dt = dt_frame / substeps
    nframes = int(round(duration_s / dt_frame))
    mass = float(np.asarray(model.body_mass.numpy())[0])
    f = np.zeros(6, dtype=np.float32)
    f[0] = force_n  # verified layout: body_f = [force xyz, torque xyz]
    for _ in range(nframes):
        for _ in range(substeps):
            s0.clear_forces()
            if force_every_substep or (s0.body_f.numpy()[0, 0] == 0.0):
                s0.body_f = wp.array(f.reshape(1, 6), dtype=s0.body_f.dtype, device=model.device)
            cp.collide(s0, contacts)
            solver.step(s0, s1, control, contacts, dt)
            s0, s1 = s1, s0
    v = np.asarray(s0.body_qd.numpy())[0, :3]  # [linear xyz, angular xyz]
    p = mass * v
    expected = force_n * duration_s
    return {"force_n": force_n, "substeps": substeps, "duration_s": duration_s,
            "dt_frame_s": dt_frame, "force_every_substep": force_every_substep,
            "mass_kg": mass, "final_velocity_m_s": [round(float(x), 6) for x in v],
            "momentum_change": [round(float(x), 6) for x in p],
            "expected_impulse": expected,
            "rel_error": round(float(abs(p[0] - expected) / expected), 6)}


def isaaclab_composer_impulse(force_n: float, substeps: int, duration_s: float,
                              dt_frame: float) -> dict:
    """Isaac Lab composer on a USD articulation (A2 verified route)."""
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg
    from isaaclab.scene import InteractiveSceneCfg, InteractiveScene
    from isaaclab.sim import SimulationCfg, build_simulation_context
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

    root = Path(os.environ["ISAACLAB_DIR"])
    usd = root / "source/isaaclab_newton/test/assets/data/articulation_ordering_branching.usda"
    if not usd.exists():
        return {"error": f"missing {usd}"}
    sim_cfg = SimulationCfg(device="cuda:0", dt=dt_frame,
                            physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(), num_substeps=substeps,
                                              use_cuda_graph=False))
    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=3.0)
    scene_cfg.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot", spawn=sim_utils.UsdFileCfg(usd_path=str(usd)),
        actuators={"torque": sim_utils.IdealPDActuatorCfg(
            joint_names_expr=[".*"], stiffness=0.0, damping=0.0)},
    ) if hasattr(sim_utils, "IdealPDActuatorCfg") else None
    if scene_cfg.robot is None:
        from isaaclab.actuators import IdealPDActuatorCfg
        scene_cfg.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot", spawn=sim_utils.UsdFileCfg(usd_path=str(usd)),
            actuators={"torque": IdealPDActuatorCfg(joint_names_expr=[".*"],
                                                     stiffness=0.0, damping=0.0)},
        )
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(scene_cfg)
        robot = scene["robot"]
        sim.reset()
        w = robot.permanent_wrench_composer
        w.reset()
        w.add_forces_and_torques_index(
            forces=torch.tensor([[0.0, force_n, 0.0]], device=sim.device),
            torques=torch.zeros((1, 3), device=sim.device),
            body_ids=torch.tensor([2], device=sim.device),
        )
        body_ids = robot.data.body_id.torch if hasattr(robot.data, "body_id") else None
        nframes = int(round(duration_s / dt_frame))
        for _ in range(nframes):
            robot.write_data_to_sim()
            sim.step(render=False)
            scene.update(sim.get_physics_dt())
        vel = robot.data.body_link_vel_w.torch[0, 2, :3].cpu().numpy()
        mass_body = None
        if hasattr(robot.data, "body_mass"):
            mass_body = float(robot.data.body_mass.torch[0, 2].item())
    expected = force_n * duration_s
    measured = float(vel[1])  # force was applied along +Y
    return {"force_n": force_n, "substeps": substeps, "duration_s": duration_s,
            "dt_frame_s": dt_frame, "body_id": 2,
            "velocity_m_s": [round(float(x), 6) for x in vel],
            "velocity_along_force": round(measured, 6),
            "body_mass_kg": mass_body,
            "expected_velocity_if_free_m_s": round(expected / mass_body, 6) if mass_body else None,
            "note": "articulated body: momentum is shared with the chain, so the free-mass velocity is a reference, not a pass criterion. The direct free-body rows are authoritative for |dp - F t| <= 2%."}


def main() -> int:
    rows = []
    for substeps in (1, 4):
        for force in (5.0, 10.0, 20.0, 30.0):
            r = newton_box_impulse(force, substeps=substeps, duration_s=0.5,
                                   dt_frame=0.01, force_every_substep=True)
            rows.append(r)
    # The historical first-substep-only bug, reproduced deliberately: one force
    # assignment and no re-application on later substeps.
    bug = newton_box_impulse(10.0, substeps=4, duration_s=0.5, dt_frame=0.01,
                             force_every_substep=False)

    composer = None
    composer_one = None
    try:
        composer = isaaclab_composer_impulse(10.0, substeps=4, duration_s=0.5, dt_frame=0.01)
        composer_one = isaaclab_composer_impulse(10.0, substeps=1, duration_s=0.5, dt_frame=0.01)
        if "error" not in composer and "error" not in composer_one:
            v4 = composer["velocity_along_force"]
            v1 = composer_one["velocity_along_force"]
            composer["substeps_4_vs_1_velocity_ratio"] = round(v4 / v1, 4) if v1 else None
            composer["substeps_1_row"] = composer_one
    except Exception as exc:  # noqa: BLE001
        composer = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}

    checks = {
        "newton_direct_all_rows_le_2pct": all(r["rel_error"] <= 0.02 for r in rows),
        "substeps_4_full_impulse": all(r["rel_error"] <= 0.02 for r in rows if r["substeps"] == 4),
        "substeps_1_full_impulse": all(r["rel_error"] <= 0.02 for r in rows if r["substeps"] == 1),
        "composer_path_ran": bool(composer and "error" not in composer),
        # With the old first-substep-only bug, 4 substeps delivered 1/4 of the
        # impulse and this ratio would be ~0.25; with the backport it is ~1.
        "composer_substeps4_matches_substeps1": bool(
            composer and "error" not in composer
            and composer.get("substeps_4_vs_1_velocity_ratio") is not None
            and abs(composer["substeps_4_vs_1_velocity_ratio"] - 1.0) <= 0.05
        ),
    }
    report = {
        "gate": "G-7",
        "step": "A4",
        "evidence_class": "SIM",
        "scope": "external wrench on a free body and on an Isaac Lab USD articulation; no FR3, no push test, no safety claim",
        "versions": versions(),
        "backport": "Isaac Lab commit 5608601d4 (PR #8269) re-applies wrenches on every solver substep; confirmed present by A2",
        "direct_rows": rows,
        "first_substep_only_control": bug,
        "composer_row": composer,
        "checks": checks,
        "not_done": ["FR3 end-effector/elbow deflection under push (A5)", "human-hand proxy contact", "hardware"],
    }
    report["result"] = "PASS" if all(checks.values()) else "FAIL"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"result": report["result"], "checks": checks,
                      "max_rel_error": max(r["rel_error"] for r in rows),
                      "composer": composer}, indent=2))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
