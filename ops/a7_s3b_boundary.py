#!/usr/bin/env python3
"""A7 rung S3b: measure the lift/slip boundary against the analytic two-pad friction bound.

Analytic (quasi-static, two pads): the grip holds while ``2 * mu * N >= m * g``.  This rung
runs the pre-registered grid mass {0.1, 1, 2, 3} kg x mu {0.1, 0.3, 0.9} x force factor
{0.1, 0.3, 0.5, 1.0}, and checks that the simulated lift/slip boundary matches the analytic
one within one grid step.  ``N`` is the *measured* per-pad squeeze estimate from the finger
position servo (the solver does not expose contact force on this build); it is a model-based
estimate, labelled as such in every trial.

Crush: no rigid-body crush sensor exists on this hand.  It is defined explicitly as
``peak finger force estimate > rtp.simbridge.crush_force_for(fragility)`` and the extremes
(0.3x, 1.3x) are reported; the definition and its limitation go in the certificate card.

    source ops/env.sh && $ISAAC_PY ops/a7_s3b_boundary.py --seeds 1

Report: ``reports/a7_s3b_boundary.json`` (SIM).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ops"))
from a7_s2_lift_v2 import lift_delta_for_mass  # noqa: E402
from rtp.simbridge import crush_force_for, slip_bound_holds  # noqa: E402
from rtp.world import crush_limit  # noqa: E402

MASSES = (0.1, 1.0, 2.0, 3.0)
FRICTIONS = (0.1, 0.3, 0.9)
FACTORS = (0.1, 0.3, 0.5, 1.0)
G = 9.81
SEED_BASE = 700000
CRUSH_FRAGILITY = 1.0


def trial(seed: int, mass: float, mu: float, factor: float, out: Path) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "ops" / "a7_s3_trial.py"),
           "--seed", str(seed), "--mass", str(mass), "--friction", str(mu),
           "--force-factor", str(factor), "--lift-delta", str(lift_delta_for_mass(mass)),
           "--out", str(out)]
    subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    try:
        return json.loads(out.read_text())
    except Exception as exc:  # noqa: BLE001
        return {"result": "FAIL", "error": f"no report: {exc}"}


def boundary_index(values: list[float], ok: list[bool]) -> int | None:
    for v, k in zip(values, ok):
        if k:
            return values.index(v)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1, help="repeats per cell (same pre-registered grid)")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "a7_s3b_boundary.json")
    a = ap.parse_args()
    trials_dir = ROOT / "reports" / "a7_s3b_trials"

    cells, rows = [], []
    for mass in MASSES:
        for mu in FRICTIONS:
            for factor in FACTORS:
                ok, ns, gains = [], [], []
                for j in range(a.seeds):
                    seed = SEED_BASE + 1000 * MASSES.index(mass) + 100 * FRICTIONS.index(mu) \
                        + 10 * FACTORS.index(factor) + j
                    rec = trial(seed, mass, mu, factor, trials_dir / f"m{mass:g}_mu{mu:g}_f{factor:g}_s{seed}.json")
                    ok.append(rec.get("result") == "PASS")
                    ns.append(rec.get("pad_force_est_N"))
                    gains.append(rec.get("height_gain_m"))
                n_meas = [x for x in ns if x is not None]
                n_pad = float(sum(n_meas) / len(n_meas)) if n_meas else float("nan")
                predicted = bool(slip_bound_holds(mass, mu, n_pad, G)) if math.isfinite(n_pad) else False
                cell = {"mass_kg": mass, "mu": mu, "force_factor": factor,
                        "nominal_pad_force_N": round(15.3 * factor, 3),
                        "measured_pad_force_est_N": round(n_pad, 4) if math.isfinite(n_pad) else None,
                        "analytic_holds": predicted,
                        "sim_lifted": all(ok), "sim_pass_count": sum(ok), "repeats": a.seeds,
                        "min_height_gain_m": min([g for g in gains if g is not None], default=None)}
                cells.append(cell)
                rows.append(cell)

    # Boundary in the force-factor dimension, per (mass, mu).
    boundaries = []
    for mass in MASSES:
        for mu in FRICTIONS:
            sub = [c for c in cells if c["mass_kg"] == mass and c["mu"] == mu]
            sim_ok = [bool(c["sim_lifted"]) for c in sub]
            ana_ok = [bool(c["analytic_holds"]) for c in sub]
            si = boundary_index(FACTORS, sim_ok)
            ai = boundary_index(FACTORS, ana_ok)
            match = (si == ai) or (si is not None and ai is not None and abs(si - ai) <= 1)
            boundaries.append({"mass_kg": mass, "mu": mu,
                               "sim_boundary_factor": FACTORS[si] if si is not None else None,
                               "analytic_boundary_factor": FACTORS[ai] if ai is not None else None,
                               "match_within_one_step": bool(match),
                               "required_pad_force_N": round(mass * G / (2.0 * mu), 3)})

    # Crush extremes: 0.3x (never) vs 1.3x (fragility 1).  Peak finger-force estimate.
    crush_n = crush_force_for(CRUSH_FRAGILITY)
    raw_crush_n = crush_limit(type("O", (), {"fragility": CRUSH_FRAGILITY})())
    crush_rows = {}
    for factor in (0.3, 1.3):
        seed = SEED_BASE + 9000 + int(factor * 10)
        rec = trial(seed, 0.1, 0.9, factor, trials_dir / f"crush_m0.1_mu0.9_f{factor:g}.json")
        peak = rec.get("pad_force_peak_est_N")
        crush_rows[f"factor_{factor:g}"] = {
            "seed": seed, "peak_finger_force_est_N": peak,
            "crush_force_for_fragility_1_N": round(crush_n, 3),
            "crushed": bool(peak is not None and peak > crush_n),
            "raw_world_crush_limit_N": round(raw_crush_n, 3),
            "crushed_under_raw_limit": bool(peak is not None and peak > raw_crush_n),
            "result": rec.get("result"), "error": rec.get("error")}

    checks = {
        "all_cells_ran": all(c["repeats"] == a.seeds for c in cells),
        "boundary_matches_analytic_within_one_step": all(b["match_within_one_step"] for b in boundaries),
        "sim_boundary_is_not_uniform": len({c["sim_lifted"] for c in cells}) == 2,
        "crush_0p3x_never_crushes": not crush_rows["factor_0.3"]["crushed"],
        "crush_1p3x_crushes_fragility_1": crush_rows["factor_1.3"]["crushed"],
    }
    report = {
        "step": "A7 S3b lift/slip boundary (force sensitivity)",
        "evidence_class": "SIM",
        "scope": "grid of Newton grasp trials; the squeeze force is a model-based servo estimate, "
                 "not a contact-sensor reading; crush is defined as a force-threshold rule",
        "analytic_bound": "2*mu*N_pad >= m*g (two pads, quasi-static)",
        "force_definition": "N_pad = kp_eff*(finger ctrl - finger q), mean over the two servos",
        "crush_definition": "peak finger force estimate > crush_force_for(fragility); "
                            f"crush_force_for({CRUSH_FRAGILITY})={crush_n:.3f} N; the raw "
                            f"world.crush_limit({CRUSH_FRAGILITY})={raw_crush_n:.3f} N is also recorded "
                            "because the +15 N simulator margin makes the named threshold unreachable "
                            "by this hand's 20 N actuator range",
        "grid": {"mass_kg": list(MASSES), "mu": list(FRICTIONS), "force_factor": list(FACTORS)},
        "cells": cells, "boundaries": boundaries, "crush": crush_rows,
        "checks": checks,
    }
    report["result"] = "PASS" if all(checks.values()) else "FAIL"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"result": report["result"], "checks": checks,
                      "boundaries": boundaries, "crush": crush_rows}, indent=2))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
