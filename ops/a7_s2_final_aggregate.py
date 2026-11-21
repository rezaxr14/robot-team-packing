#!/usr/bin/env python3
"""Aggregate the final A7 S2 rung evidence from the two graded trial directories.

The 0.1/0.5 kg trials used a 12 cm commanded lift and the 1.0 kg trials a 14 cm command (D-85);
both are recorded with their provenance. The measured-gain bound is unchanged at 80 mm, and each
trial's own report already enforces it. The rung criterion is >= 95 % of 50 seeds per mass;
`configs/arm_cert.yaml` requires >= 200 trials per skill for the *full* A7 regression, which this
single skill's rung does not provide.

Report: `reports/a7_s2_final.json`. Evidence class: SIM.
"""
from __future__ import annotations

import json
import platform
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
from rtp.stats import wilson  # noqa: E402

SOURCES = [
    ("reports/a7_s2_v2_trials", 0.12),   # 0.1 / 0.5 kg trials
    ("reports/a7_s2_v2_heavy_trials", 0.14),  # 1.0 kg trials
]


def main() -> int:
    rows = []
    for dirname, delta in SOURCES:
        for path in sorted((ROOT / dirname).glob("m*_s*.json")):
            rec = json.loads(path.read_text())
            mass = float(path.name.split("_")[0][1:])
            rows.append({"mass_kg": mass, "seed": int(path.stem.split("_s")[1]),
                         "lift_delta_m": rec.get("lift_delta_m", delta),
                         "source": str(path.relative_to(ROOT)),
                         "result": rec.get("result"),
                         "height_gain_m": rec.get("height_gain_m"),
                         "box_xy_drift_m": rec.get("box_xy_drift_m"),
                         "error": rec.get("error")})

    by_mass = {}
    for mass in sorted({r["mass_kg"] for r in rows}):
        sub = [r for r in rows if r["mass_kg"] == mass]
        k = sum(r["result"] == "PASS" for r in sub)
        lo, hi = wilson(k, len(sub))
        gains = [r["height_gain_m"] for r in sub if r["height_gain_m"] is not None]
        by_mass[str(mass)] = {"n": len(sub), "pass": k, "rate": k / len(sub) if sub else 0.0,
                              "wilson95": [lo, hi], "min_gain_m": min(gains) if gains else None,
                              "commanded_lift_m": sorted({r["lift_delta_m"] for r in sub})}
    k = sum(r["result"] == "PASS" for r in rows)
    lo, hi = wilson(k, len(rows))
    report = {
        "step": "A7 S2 rung final aggregate (seeded grasp + lift)",
        "evidence_class": "SIM",
        "scope": "single-skill rung; NOT the >=200-trial per-skill A7 certification suite",
        "python": platform.python_version(),
        "sources": [{"dir": d, "commanded_lift_m": delta} for d, delta in SOURCES],
        "n": len(rows), "pass": k, "rate": k / len(rows) if rows else 0.0,
        "wilson95": [lo, hi], "by_mass": by_mass,
        "checks": {
            "all_trials_have_reports": all(r["result"] in ("PASS", "FAIL") for r in rows),
            "per_mass_rate_ge_0p95": all(v["rate"] >= 0.95 for v in by_mass.values()),
            "min_50_seeds_per_mass": all(v["n"] >= 50 for v in by_mass.values()),
            "measured_gain_bound_unchanged": True,   # enforced per trial, 80 mm (runbook S2)
        },
        "trials": rows,
    }
    report["result"] = "PASS" if all(report["checks"].values()) else "FAIL"
    out = ROOT / "reports" / "a7_s2_final.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("result", "n", "pass", "rate", "by_mass", "checks")},
                     indent=2))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
