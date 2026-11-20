#!/usr/bin/env python3
"""A7 plumbing: run the executor-agnostic skill regression on the Newton scene backend.

HONEST LIMIT: the current backend's `run_skill` is the A6 SIM-POSE state-machine stand-in, so
this measures interface/case coverage, NOT physical grasp/place outcomes. The report says so.
When A7 scripted IK skills replace the stand-in, this same script produces the real SIM evidence.

    source ops/env.sh && $ISAAC_PY isaac/regression_newton.py --trials 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rtp.simbridge import ObjState
from rtp.skills import run_regression


class NewtonSkillExecutor:
    def __init__(self, backend):
        self.backend = backend

    def reset(self, objects, seed, lock_to):
        self.backend.reset(int(seed))
        items = self.backend._items
        if len(objects) > len(items):
            raise ValueError(f"case has {len(objects)} objects, scene has {len(items)}")
        states = []
        for i, spec in enumerate(objects):
            name, x, y, hx, hy, hz = items[i]
            states.append(ObjState(
                name=spec.get("name", name), cls=spec.get("cls", "object"),
                pos=(x, y, 0.90 + hz), mass=float(spec.get("mass", 0.5)),
                fragility=float(spec.get("fragility", 0.3)),
                size=float(spec.get("size", max(hx, hy, hz) * 2)),
                friction=float(spec.get("friction", 0.6)),
                init_region=spec.get("region", "table_center"),
                is_distractor=bool(spec.get("is_distractor", False)),
            ))
        self.backend._states = states
        self.backend.lock_holder = lock_to
        self.backend.step(2)

    def run(self, arm, skill, obj, params):
        return self.backend.run_skill(arm, skill, obj, params).category

    def state(self):
        return {
            "holding": {a: next((s.name for s in self.backend._states if s.held_by == a), None)
                        for a in ("L", "R")},
            "bag": {"main": [s.name for s in self.backend._states
                             if abs(s.pos[0] - 0.0) < 0.22 and abs(s.pos[1] - 0.25) < 0.14]},
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "skill_regression_scripted.json")
    a = ap.parse_args()
    from isaac.newton_packing_sim import NewtonPackingSim

    rep = run_regression(NewtonSkillExecutor(NewtonPackingSim()),
                         n_trials=a.trials, pass_rate=0.9, seed0=100000)
    payload = {"step": "A7 (plumbing only)", "evidence_class": "SIM-POSE",
               "kind": "sim_pose_standin",
               "scope": "regression suite over the A6 state-machine stand-in; NOT physical skill outcomes",
               "trials": a.trials, **rep}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"passed": rep["passed"], "total": rep["total"],
                      "by_tag": rep["by_tag"]}, indent=2))
    return 0 if rep["passed"] == rep["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
