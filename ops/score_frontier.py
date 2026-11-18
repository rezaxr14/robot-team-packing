#!/usr/bin/env python3
"""Replay Kaggle-hosted frontier CoT plans on the fixed held-out test set.

The hosted task produced verbatim transport records (reasoning + Plan JSON). Here we take the
Plan JSON, replay it with ``ReplayBrain`` on the exact same seeds/worlds/eyes as our fixture
scores, and write records in ``rtp.benchmark``'s strict prediction format. No completion is
invented and no score is assigned without a replay.

    python ops/score_frontier.py --traces data/local_4050/cot_frontier_traces.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="data/local_4050/cot_frontier_traces.jsonl")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default="data/predictions/frontier_zero_shot.jsonl")
    ap.add_argument("--report", default="reports/frontier_replay.json")
    a = ap.parse_args(argv)

    from rtp.benchmark import assert_no_train_overlap, make_cases
    from rtp.brain import ReplayBrain if False else None  # placeholder removed below
    from rtp.frontier_cot import read_traces, split_completion
    from rtp.harness import Harness
    from rtp.hindsight import ReplayBrain
    from rtp.loop import run_episode
    from rtp.plan import parse_plan

    traces = read_traces(ROOT / a.traces)
    if not traces:
        print("no traces")
        return 1
    model = traces[0].get("model") or traces[0].get("source", "frontier")
    name = f"Frontier model as zero-shot Brain ({model})"
    cases = {s: (w, spec, eyes, lc) for s, w, spec, eyes, lc in make_cases(range(a.n))}
    harness = Harness(ROOT / "harness_files")
    rows, failed = [], []
    for t in traces:
        seed = int(t["seed"])
        assert_no_train_overlap([seed])
        if seed not in cases:
            failed.append({"seed": seed, "error": f"seed outside 0..{a.n-1}"})
            continue
        _reasoning, plan_text = split_completion(t["completion"])
        world, spec, eyes, lc = cases[seed]
        try:
            plan = parse_plan(plan_text)
            brain = ReplayBrain(plan)
            res = run_episode(world, eyes, brain, spec, harness, lc)
        except Exception as exc:                          # noqa: BLE001
            failed.append({"seed": seed, "id": t.get("id"), "error": f"{type(exc).__name__}: {exc}"})
            continue
        rows.append({"model": name, "seed": seed, "success": bool(res.success),
                     "plan_pass": bool(res.plan_pass), "score": float(res.score),
                     "plan": plan.to_dict() if hasattr(plan, "to_dict") else plan_text,
                     "reasoning": _reasoning, "msg_tokens": int(res.msg_tokens),
                     "source": t.get("source"), "trace_model": model,
                     "trace_id": t.get("id"), "kind": "frontier-zero-shot"})
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = {"traces": len(traces), "replayed": len(rows), "failed": failed,
              "model": model, "trace_file": str(a.traces), "predictions": str(out),
              "success": sum(r["success"] for r in rows), "plan_pass": sum(r["plan_pass"] for r in rows)}
    rp = ROOT / a.report
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
