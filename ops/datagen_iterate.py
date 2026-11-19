#!/usr/bin/env python3
"""Local driver for the iterative multi-turn data-generation loop (docs/DATAGEN_KAGGLE.md).

    build   write the hosted task for one shard, embedding the current success/failure memory notes
    ingest  verify a downloaded shard (replay), keep accepted records, append the per-model ledger,
            and rebuild memory for the NEXT iteration from all accepted datagen records
    status  print the ledger summary (which model, what succeeded / failed, cost)

State lives under ``runs/datagen/`` (accepted.jsonl, quarantine.jsonl, ledger.jsonl, memory_it<N>.json).
The Kaggle commands themselves (push / run / download) are printed, never run here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ops.kaggle.multiturn.build_task import build                                   # noqa: E402
from rtp.datagen_ingest import append_ledger, ingest, ledger_entry, next_memory, read_records   # noqa: E402

STATE = ROOT / "runs" / "datagen"
HARNESS = ROOT / "harness_files"


def _key(r):
    return (r["model"], r["seed"], r["condition"], r["mode"], r["iteration"])


def cmd_build(a) -> int:
    mem = STATE / f"memory_it{a.iteration}.json"
    notes = json.loads(mem.read_text()) if mem.exists() else []
    spec = build(seeds=a.seeds, condition=a.condition, mode=a.mode, iteration=a.iteration, shard=a.shard,
                 memory_notes=notes, max_turns=a.max_turns, repo_url=a.repo_url)
    print(f"task written for shard {a.shard}: {len(notes)} memory notes, spec_hash {spec['spec_hash']}")
    print("push the repo FIRST (the worker clones the default branch), then:")
    for model in a.models:
        print(f"  kaggle benchmarks tasks push {spec['task_name']} -f ops/kaggle/multiturn/task.py")
        print(f"  kaggle benchmarks tasks run  {spec['task_name']} -m {model}")
        print(f"  kaggle benchmarks tasks download {spec['task_name']} -o runs/datagen/downloads -m {model} -s -f")
    return 0


def cmd_ingest(a) -> int:
    got = ingest(a.episodes, HARNESS, check_identity={"iteration": a.iteration})
    STATE.mkdir(parents=True, exist_ok=True)
    have = {_key(r) for r in read_records(STATE / "accepted.jsonl")}
    fresh = [r for r in got["accepted"] if _key(r) not in have]
    with (STATE / "accepted.jsonl").open("a", encoding="utf-8") as f:
        for r in fresh:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (STATE / "quarantine.jsonl").open("a", encoding="utf-8") as f:
        for q in got["quarantined"]:
            f.write(json.dumps({**q, "shard": a.shard, "iteration": a.iteration}, ensure_ascii=False) + "\n")
    entry = ledger_entry(fresh, got["quarantined"], shard=a.shard, iteration=a.iteration)
    append_ledger(STATE / "ledger.jsonl", entry)
    notes = next_memory(read_records(STATE / "accepted.jsonl"))
    (STATE / f"memory_it{a.iteration + 1}.json").write_text(json.dumps(notes, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(entry, indent=2))
    print(f"memory for iteration {a.iteration + 1}: {len(notes)} notes -> runs/datagen/memory_it{a.iteration + 1}.json")
    return 0


def cmd_status(_a) -> int:
    for e in read_records(STATE / "ledger.jsonl"):
        for m, v in e["models"].items():
            print(f"it{e['iteration']} {e['shard']:24s} {m:36s} n={v['episodes']:4d} success={v['success_rate']} cost={v['cost_usd']} causes={v['causes']}")
        if e["quarantined"]:
            print(f"   quarantined {e['quarantined']}: {e['quarantine_reasons']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--iteration", type=int, required=True)
    b.add_argument("--shard", required=True)
    b.add_argument("--seeds", required=True)
    b.add_argument("--condition", choices=("harness", "zero_shot"), default="harness")
    b.add_argument("--mode", choices=("datagen", "eval"), default="datagen")
    b.add_argument("--max-turns", type=int, default=8)
    b.add_argument("--models", nargs="+", default=["claude-haiku-5-5"])
    b.add_argument("--repo-url", default="https://github.com/rezaxr14/robot-team-packing.git")
    i = sub.add_parser("ingest")
    i.add_argument("--shard", required=True)
    i.add_argument("--iteration", type=int, required=True)
    i.add_argument("--episodes", required=True)
    sub.add_parser("status")
    a = ap.parse_args()
    return {"build": cmd_build, "ingest": cmd_ingest, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
