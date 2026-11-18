#!/usr/bin/env python3
"""Generate training data from the asset-backed pose-level environment.

    python ops/datagen_assets.py --profile local_4050 --n 64 --seed0 200000

The assets must already be fetched by ops/fetch_simready.py and selected in assets/selection.json.
Everything written keeps the SIM-POSE evidence class: these are real SimReady packages in our
pose-level world, not a claim of a Newton physics result.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="local_4050")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed0", type=int, default=200000)
    ap.add_argument("--out", default="")
    ap.add_argument("--manifest", default="")
    a = ap.parse_args(argv)

    from rtp.asset_datagen import gen_asset_sft
    from rtp.datagen import write_jsonl
    from rtp.harness import Harness
    from rtp.profiles import get_profile
    from rtp.scene_assets import resolve_assets, to_world_objects, write_manifest

    try:
        records = resolve_assets()
    except (FileNotFoundError, ValueError) as exc:
        print(f"assets not ready: {exc}")
        return 2
    profile = get_profile(a.profile, root=ROOT)
    profile.ensure_dirs()
    manifest = Path(a.manifest or ROOT / "reports" / "simready_scene.json")
    objs = to_world_objects(records)
    write_manifest(records, objs, manifest)
    print(f"scene manifest: {manifest} ({len(records)} assets, {len(objs)} objects)")

    rows, funnel = gen_asset_sft(records, a.n, a.seed0, Harness(ROOT / "harness_files"),
                                 manifest=str(manifest))
    out = Path(a.out or profile.paths.data / "asset_sft.jsonl")
    write_jsonl(rows, out)
    funnel.update({"rows": len(rows), "path": str(out), "model": "scripted teacher"})
    report = ROOT / "reports" / "asset_datagen.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(funnel, indent=2) + "\n")
    print(json.dumps(funnel, indent=2))
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
