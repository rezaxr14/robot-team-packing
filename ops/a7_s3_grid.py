#!/usr/bin/env python3
"""A7 rung S3: friction x grasp-force grid with an outcome histogram.

    python ops/a7_s3_grid.py --seeds 3

Each cell is one `ops/a7_s3_trial.py` Newton run (evidence class SIM). The histogram categorises
what was *measured* (lifted / knocked aside / never picked up / harness fault). It does not claim
'slip' or 'crush' causes: the derived hand has no slip or crush sensor, and manufacturing a cause
would be a weaker-class promotion.

Report: `reports/a7_s3_grid.json`.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FRICTIONS = (0.3, 0.6, 0.9)
FORCE_FACTORS = (0.7, 1.0, 1.3)


def classify(rec: dict) -> str:
    if rec.get("result") != "PASS":
        if rec.get("error"):
            return "harness_fault"
        gain = rec.get("height_gain_m") or 0.0
        return "knocked_aside" if (rec.get("box_xy_drift_m") or 0) > 0.02 else "never_picked_up"
    return "lifted"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--mass", type=float, default=0.5)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "a7_s3_grid.json")
    a = ap.parse_args()
    trials_dir = ROOT / "reports" / "a7_s3_trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for friction in FRICTIONS:
        for factor in FORCE_FACTORS:
            for seed in range(a.seeds):
                out = trials_dir / f"mu{friction:g}_f{factor:g}_s{seed}.json"
                cmd = [a.python, str(ROOT / "ops" / "a7_s3_trial.py"),
                       "--seed", str(seed), "--mass", str(a.mass),
                       "--friction", str(friction), "--force-factor", str(factor), "--out", str(out)]
                result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
                try:
                    rec = json.loads(out.read_text())
                except Exception:  # noqa: BLE001
                    rec = {"result": "FAIL", "error": (result.stdout + result.stderr)[-400:]}
                rows.append({"friction": friction, "force_factor": factor, "seed": seed,
                             "result": rec.get("result"), "outcome": classify(rec),
                             "height_gain_m": rec.get("height_gain_m"),
                             "box_xy_drift_m": rec.get("box_xy_drift_m"),
                             "error": rec.get("error")})

    hist = Counter(r["outcome"] for r in rows)
    by_force = {}
    for factor in FORCE_FACTORS:
        sub = [r for r in rows if r["force_factor"] == factor]
        gains = [r["height_gain_m"] for r in sub if r["height_gain_m"] is not None]
        by_force[str(factor)] = {"n": len(sub), "lifted": sum(r["outcome"] == "lifted" for r in sub),
                                 "mean_gain_m": sum(gains) / len(gains) if gains else None}
    report = {
        "step": "A7 S3 (friction x grasp-force grid)",
        "evidence_class": "SIM",
        "scope": "measured lift/knock/not-held outcomes with the derived parallel-jaw hand; "
                 "no slip/crush cause is claimed (no such sensor exists on this hand)",
        "python": platform.python_version(), "mass_kg": a.mass, "seeds_per_cell": a.seeds,
        "grid": {"frictions": list(FRICTIONS), "force_factors": list(FORCE_FACTORS)},
        "n": len(rows), "histogram": dict(hist), "by_force": by_force,
        "checks": {"all_cells_ran": all(r["outcome"] != "harness_fault" for r in rows),
                   "histogram_recorded": bool(hist)},
        "trials": rows,
    }
    report["result"] = "PASS" if all(report["checks"].values()) else "FAIL"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"result": report["result"], "histogram": report["histogram"],
                      "by_force": by_force}, indent=2))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
