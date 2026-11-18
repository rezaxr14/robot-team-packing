#!/usr/bin/env python3
"""Build and query success/failure memory RAG on the asset-backed pose-level environment.

    python ops/asset_memory.py --build --n 40 --seed0 300000
    python ops/asset_memory.py --query "pack the notebook and bottle" --out reports/asset_memory_query.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="local_4050")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--query", default="")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=300000)
    ap.add_argument("--out", default="reports/asset_memory.json")
    a = ap.parse_args(argv)

    from rtp.asset_memory import augment_prompt, build_memory, read_memory, retrieve_memory, write_memory
    from rtp.harness import Harness
    from rtp.profiles import get_profile
    from rtp.scene_assets import resolve_assets

    records = resolve_assets()
    profile = get_profile(a.profile, root=ROOT)
    profile.ensure_dirs()
    path = profile.paths.data / "asset_memory.jsonl"
    harness = Harness(ROOT / "harness_files")
    report = {}
    if a.build:
        rows, stats = build_memory(records, a.n, a.seed0, harness)
        write_memory(rows, path)
        report = {"build": stats, "path": str(path)}
    if a.query:
        rows = read_memory(path)
        succ = retrieve_memory(rows, a.query, k=3, outcome="success")
        fail = retrieve_memory(rows, a.query, k=3, outcome="failure")
        report["query"] = a.query
        report["success_hits"] = [{"seed": r["seed"], "score": s} for r, s in succ]
        report["failure_hits"] = [{"seed": r["seed"], "score": s} for r, s in fail]
        report["augmented_prompt_chars"] = len(
            augment_prompt(records, rows, a.seed0, harness, k=3))
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
