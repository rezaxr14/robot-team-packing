#!/usr/bin/env python3
"""A7 rung S2 suite: seeded free-box grasp/lift trials aggregated from `ops/a7_s2_lift.py`.

    python ops/a7_s2_suite.py --seeds 50 --masses 0.1,0.5,1.0

Every trial is a separate Newton run whose per-trial JSON is kept under `reports/a7_s2_trials/`.
The aggregate uses `rtp.stats.wilson`. Nothing is counted unless the trial report says PASS.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtp.stats import wilson  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--masses", default="0.1,0.5,1.0")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--resume", action="store_true", help="reuse existing PASS trial reports")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "a7_s2_suite.json")
    a = ap.parse_args()
    masses = [float(x) for x in a.masses.split(",") if x.strip()]
    trials_dir = ROOT / "reports" / "a7_s2_trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for mass in masses:
        for seed in range(a.seeds):
            out = trials_dir / f"m{mass:g}_s{seed}.json"
            if a.resume and out.exists():
                try:
                    old = json.loads(out.read_text())
                    if old.get("result") == "PASS":
                        rows.append({"seed": seed, "mass_kg": mass, "result": "PASS",
                                     "height_gain_m": old.get("height_gain_m"),
                                     "box_xy_drift_m": old.get("box_xy_drift_m"), "error": None})
                        continue
                except Exception:  # noqa: BLE001
                    pass
            cmd = [a.python, str(ROOT / "ops" / "a7_s2_lift.py"),
                   "--seed", str(seed), "--mass", str(mass), "--out", str(out)]
            result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            try:
                rec = json.loads(out.read_text())
            except Exception:  # noqa: BLE001
                rec = {"result": "FAIL", "error": (result.stdout + result.stderr)[-400:]}
            rows.append({"seed": seed, "mass_kg": mass, "result": rec.get("result"),
                         "height_gain_m": rec.get("height_gain_m"),
                         "box_xy_drift_m": rec.get("box_xy_drift_m"),
                         "error": rec.get("error")})

    k = sum(r["result"] == "PASS" for r in rows)
    n = len(rows)
    lo, hi = wilson(k, n)
    by_mass = {}
    for mass in masses:
        sub = [r for r in rows if r["mass_kg"] == mass]
        kk = sum(r["result"] == "PASS" for r in sub)
        ll, hh = wilson(kk, len(sub))
        gains = [r["height_gain_m"] for r in sub if r["height_gain_m"] is not None]
        by_mass[str(mass)] = {"pass": kk, "total": len(sub), "rate": kk / len(sub) if sub else None,
                              "wilson95": [ll, hh],
                              "min_height_gain_m": min(gains) if gains else None}
    report = {
        "step": "A7 S2 suite (free-box grasp and lift, 50 seeds per mass)",
        "evidence_class": "SIM",
        "scope": "seeded Newton grasp attempts with the derived simplified parallel-jaw hand; "
                 "not a certification and not a safety result",
        "python": platform.python_version(),
        "seeds_per_mass": a.seeds, "masses_kg": masses,
        "pass": k, "total": n, "rate": k / n if n else None, "wilson95": [lo, hi],
        "by_mass": by_mass,
        "checks": {
            "rate_ge_0p95": k / n >= 0.95 if n else False,
            "all_trials_have_reports": all(r["result"] in ("PASS", "FAIL") for r in rows),
            "min_gain_ge_8cm": all((r["height_gain_m"] or 0) >= 0.08 for r in rows),
        },
        "trials": rows,
    }
    report["result"] = "PASS" if all(report["checks"].values()) else "FAIL"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("result", "pass", "total", "rate",
                                                   "wilson95", "by_mass", "checks")}, indent=2))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
