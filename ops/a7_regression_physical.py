#!/usr/bin/env python3
"""A7 physical regression driver / aggregator (no stand-in, SIM evidence only).

Each skill needs >= 200 seeded Newton trials (`configs/arm_cert.yaml`). `run` invokes the
skill's graded trial script once per seed; `aggregate` merges the per-skill trial reports into
`reports/skill_regression_scripted.json`, the file the mechanical step gate reads. Missing or
ungraded skills are recorded with an explicit reason, never as PASS.

    python ops/a7_regression_physical.py run --skill grasp --seeds 200
    python ops/a7_regression_physical.py aggregate

Evidence class: SIM (each trial is a real Newton step). No result is invented here.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rtp.stats import wilson  # noqa: E402

SKILLS = ("grasp", "place", "release", "retreat", "weigh", "handover", "push_aside")
DRIVERS = {
    "grasp": ("ops/a7_s2_lift_v2.py", ()),
    "place": ("ops/a7_s4_place.py", ()),
    "weigh": ("ops/a7_s5_weigh.py", ()),
    "handover": ("ops/a7_s6_handover.py", ("--variant", "handover")),
    "push_aside": ("ops/a7_s6_handover.py", ("--variant", "push_aside")),
}
MIN_TRIALS = 200
MIN_RATE = 0.90


def trials_dir(skill: str) -> Path:
    return ROOT / "reports" / "a7_physical" / skill


def run(skill: str, seeds: int, start: int, mass: float) -> int:
    if skill not in DRIVERS:
        print(f"NO_DRIVER for {skill}; no trials run", file=sys.stderr)
        return 1
    script, extra = DRIVERS[skill]
    out_dir = trials_dir(skill)
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in range(start, start + seeds):
        out = out_dir / f"s{seed}.json"
        if out.exists() and json.loads(out.read_text()).get("result") == "PASS":
            continue
        cmd = [sys.executable, str(ROOT / script), "--seed", str(seed), "--mass", str(mass),
               "--out", str(out), *extra]
        subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    return 0


def aggregate() -> int:
    per_skill, cases = {}, {}
    for skill in SKILLS:
        files = sorted(trials_dir(skill).glob("s*.json"))
        recs = []
        for f in files:
            try:
                recs.append(json.loads(f.read_text()))
            except Exception:  # noqa: BLE001
                pass
        n = len(recs)
        k = sum(r.get("result") == "PASS" for r in recs)
        lo, hi = wilson(k, n) if n else (0.0, 0.0)
        per_skill[skill] = {"n": n, "passed": k, "rate": k / n if n else 0.0,
                            "wilson95": [lo, hi],
                            "graded": n >= MIN_TRIALS and k / n >= MIN_RATE if n else False}
        cases[skill] = {"rate": per_skill[skill]["rate"], "passed": per_skill[skill]["graded"],
                        "tags": [skill]}
    total_n = sum(v["n"] for v in per_skill.values())
    total_k = sum(v["passed"] for v in per_skill.values())
    report = {
        "step": "A7 physical skill regression",
        "evidence_class": "SIM",
        "kind": "newton_physical",
        "scope": "graded per-skill Newton trials; missing/ungraded skills are not counted as PASS",
        # Gate contract: `trials` = minimum trials for ANY skill, `passed` = number of
        # skills that meet their own >=200 trials and >=90% rate, `total` = 7 skills.
        # Reporting raw trial totals here made 196 passes / 7 skills read as a false DONE.
        "trials": min(v["n"] for v in per_skill.values()),
        "passed": sum(v["graded"] for v in per_skill.values()), "total": len(SKILLS),
        "trial_totals": {"all": total_n, "pass": total_k},
        "per_skill": per_skill, "cases": cases,
        "checks": {"all_skills_graded": all(v["graded"] for v in per_skill.values()),
                   "min_trials_per_skill": all(v["n"] >= MIN_TRIALS for v in per_skill.values())},
    }
    report["result"] = "PASS" if all(report["checks"].values()) else "FAIL"
    out = ROOT / "reports" / "skill_regression_scripted.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"result": report["result"], "trials": report["trials"],
                      "passed": report["passed"], "total": report["total"],
                      "trial_totals": report["trial_totals"], "per_skill": {k: (v["n"], v["passed"]) for k, v in per_skill.items()}},
                     indent=2))
    return 0 if report["result"] == "PASS" else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--skill", required=True, choices=SKILLS)
    r.add_argument("--seeds", type=int, default=200)
    r.add_argument("--start", type=int, default=0)
    r.add_argument("--mass", type=float, default=0.5)
    sub.add_parser("aggregate")
    a = ap.parse_args()
    return run(a.skill, a.seeds, a.start, a.mass) if a.cmd == "run" else aggregate()


if __name__ == "__main__":
    raise SystemExit(main())
