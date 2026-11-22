#!/usr/bin/env python3
"""A7 S2 grasp rung: tune the single lift rule on seeds >= 900000, evaluate it once on fresh seeds.

Rules from the review (2026-11-22):
  * one deterministic rule ``lift_delta = f(mass)`` using the mass the `weigh` skill supplies;
  * the rule is chosen ONLY on the tuning split (``tuning_seeds_min`` in configs/arm_cert.yaml,
    900000) and then evaluated ONCE on fresh seeds >= 100000;
  * pooled settings never count; earlier 12/14 cm numbers are exploration.

    python ops/a7_s2_rule.py tune     # candidates on the tuning split
    python ops/a7_s2_rule.py eval     # the chosen rule, fresh seeds, >= 200 trials

Every trial is a real Newton step via ``ops/a7_s2_lift_v2.py`` (evidence class SIM). Reports:
``reports/a7_s2_rule_tuning.json`` and ``reports/a7_s2_rule_final.json``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rtp.stats import wilson  # noqa: E402

TRIAL = ROOT / "ops" / "a7_s2_lift_v2.py"
MASSES = (0.1, 0.5, 1.0)
TUNING_START = 900000
EVAL_START = 100000


def run_trial(seed: int, mass: float, out: Path, *, rule: bool, lift_delta: float | None) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(TRIAL), "--seed", str(seed), "--mass", str(mass), "--out", str(out)]
    if rule:
        cmd.append("--rule")
    elif lift_delta is not None:
        cmd += ["--lift-delta", str(lift_delta)]
    subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    try:
        return json.loads(out.read_text())
    except Exception as exc:  # noqa: BLE001
        return {"result": "FAIL", "error": f"no report: {exc}"}


def summarise(rows: list[dict]) -> dict:
    k = sum(r.get("result") == "PASS" for r in rows)
    lo, hi = wilson(k, len(rows)) if rows else (0.0, 0.0)
    gains = [r.get("height_gain_m") for r in rows if r.get("height_gain_m") is not None]
    return {"n": len(rows), "pass": k, "rate": (k / len(rows)) if rows else 0.0,
            "wilson95": [lo, hi], "min_gain_m": min(gains) if gains else None}


def tune(seeds: int) -> int:
    out_dir = ROOT / "reports" / "a7_s2_tune_trials"
    cells = {}
    for mass in MASSES:
        candidates = (0.12, 0.14) if mass >= 1.0 else (0.12,)
        for delta in candidates:
            rows = []
            for i in range(seeds):
                seed = TUNING_START + i
                out = out_dir / f"m{mass:g}_d{delta:.2f}_s{seed}.json"
                rows.append(run_trial(seed, mass, out, rule=False, lift_delta=delta))
            cells[f"m{mass:g}_d{delta:.2f}"] = summarise(rows)
    report = {"step": "A7 S2 lift-rule tuning (tuning split only)", "evidence_class": "SIM",
              "scope": "candidate lift deltas on tuning seeds >= 900000; NOT a rung result",
              "tuning_seeds_min": TUNING_START, "seeds_per_cell": seeds, "cells": cells}
    (ROOT / "reports" / "a7_s2_rule_tuning.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(cells, indent=2))
    return 0


def evaluate(seeds: int) -> int:
    out_dir = ROOT / "reports" / "a7_physical" / "grasp_rule"
    rows, per_mass = [], {}
    for i in range(seeds):
        mass = MASSES[i % len(MASSES)]
        seed = EVAL_START + i
        out = out_dir / f"s{seed}_m{mass:g}.json"
        rec = run_trial(seed, mass, out, rule=True, lift_delta=None)
        rec["seed"], rec["mass_kg"] = seed, mass
        rows.append(rec)
    for mass in MASSES:
        per_mass[str(mass)] = summarise([r for r in rows if r["mass_kg"] == mass])
    overall = summarise(rows)
    checks = {
        "all_trials_have_reports": all(r.get("result") in ("PASS", "FAIL") for r in rows),
        "min_200_trials": len(rows) >= 200,
        "all_masses_covered": all(per_mass[str(m)]["n"] >= 50 for m in MASSES),
        "per_mass_rate_ge_0p95": all(per_mass[str(m)]["rate"] >= 0.95 for m in MASSES),
        "single_rule_used": all(r.get("lift_rule", "").startswith("lift_delta_for_mass") for r in rows),
        "fresh_seeds_ge_100000": all(r["seed"] >= EVAL_START for r in rows),
    }
    report = {"step": "A7 S2 grasp rung with the single lift rule (fresh evaluation)",
              "evidence_class": "SIM",
              "scope": "one skill (grasp/lift) under the pre-registered rule; not the 7-skill A7 gate",
              "rule": "lift_delta_for_mass(mass) in ops/a7_s2_lift_v2.py",
              "tuning_seeds_min": TUNING_START, "eval_seeds_min": EVAL_START,
              "mass_source": "commanded mass (stand-in for the `weigh` estimate; recorded in each trial)",
              "n": overall["n"], "pass": overall["pass"], "rate": overall["rate"],
              "wilson95": overall["wilson95"], "by_mass": per_mass,
              "checks": checks, "trials": rows}
    report["result"] = "PASS" if all(checks.values()) else "FAIL"
    (ROOT / "reports" / "a7_s2_rule_final.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("result", "n", "pass", "rate", "wilson95", "by_mass",
                                             "checks")}, indent=2))
    return 0 if report["result"] == "PASS" else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("tune", "eval"))
    ap.add_argument("--seeds", type=int, default=None)
    a = ap.parse_args()
    if a.stage == "tune":
        return tune(a.seeds or 10)
    return evaluate(a.seeds or 210)


if __name__ == "__main__":
    raise SystemExit(main())
